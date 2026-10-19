"""Strict MATLAB cross-validation of the saffron phase/manifold/binning fixes.

The pre-existing ``test_saffron_matlab_validation.py`` compares spectra with a
cosine similarity and loose thresholds. For pulse-EPR time traces that metric
is dominated by the DC offset: a HYSCORE trace whose modulation is ~100% wrong
still scores cosine > 0.999. Four real defects survived in that blind spot:

1. the second ``Q_`` factor in the predefined 2p/3p/4pESEEM and HYSCORE
   dispatches was not conjugated (MATLAB's ``'`` is a conjugate transpose),
2. the S=1/2 fast path assigned ``Ea`` to the lower instead of the upper
   electron manifold, which only shows up once a nuclear quadrupole makes the
   two manifolds' eigenvectors non-trivial,
3. the 2D frequency-bin accumulation indexed the buffer transposed,
4. ``Opt.OriThreshold`` was applied relative to the excitation-weighted
   maximum instead of the unweighted one.

Every assertion here therefore checks the *mean-centered* relative L2 error --
the modulation -- as well as the raw error, at tolerances near roundoff.
References come from ``tests/data/generate_saffron_phase_refs.m``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

try:
    from scipy.io import loadmat
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

from torchspin import SpinSystem
from torchspin.saffron import saffron, PulseExperiment, SaffronOptions


DATA = Path(__file__).parent.parent.parent / "tests" / "data" / "saffron_phase_refs.mat"

# Agreement is at double-precision roundoff; these leave ~3 orders of headroom
# for BLAS/eigensolver ordering differences across platforms.
RAW_TOL = 1e-11
CENTERED_TOL = 1e-9

SYS_14N = dict(S=[0.5], g=[[2.0100, 2.0050, 1.9980]], Nucs='14N',
               A=[[2.5, 3.1, 5.4]], Q=[[0.8, 0.5]])
SYS_13C = dict(S=[0.5], g=[[2.00432, 2.0036, 2.00217]], Nucs='13C',
               A=[[-13.5, -9, 40]])
SYS_S32 = dict(S=[1.5], g=[[2.01, 2.005, 1.998]], D=[[50, 0]], Nucs='14N',
               A=[[2.5, 3.1, 5.4]], Q=[[-0.3, -0.1, 0.4]])


@pytest.fixture(scope="module")
def refs():
    if not SCIPY_OK:
        pytest.skip("scipy not installed")
    if not DATA.exists():
        pytest.skip(f"reference file not found: {DATA}")
    return loadmat(str(DATA), simplify_cells=True)["data"]


def _errors(y, y_ref) -> tuple[float, float]:
    """Raw and mean-centered relative L2 errors."""
    a = np.asarray(y).ravel()
    b = np.asarray(y_ref).ravel()
    assert a.shape == b.shape, f"shape {a.shape} != reference {b.shape}"
    raw = np.linalg.norm(a - b) / np.linalg.norm(b)
    ac, bc = a - a.mean(), b - b.mean()
    # A reference with no modulation at all cannot constrain the centered error.
    if np.linalg.norm(bc) <= 1e-12 * max(abs(b.mean()), 1e-30):
        return float(raw), 0.0
    centered = np.linalg.norm(ac - bc) / np.linalg.norm(bc)
    return float(raw), float(centered)


def _check(y, y_ref, label: str) -> None:
    raw, centered = _errors(y, y_ref)
    assert raw < RAW_TOL, f"{label}: raw relative error {raw:.3e} >= {RAW_TOL:g}"
    assert centered < CENTERED_TOL, (
        f"{label}: mean-centered (modulation) relative error {centered:.3e} "
        f">= {CENTERED_TOL:g} -- the DC offset can agree while the modulation "
        f"is wrong, so this is the assertion that matters"
    )


def _run(sys_kw, exp_kw, opt_kw):
    _, y, _ = saffron(SpinSystem(**sys_kw), PulseExperiment(**exp_kw),
                      SaffronOptions(**opt_kw))
    return y


# ---------------------------------------------------------------------------
# 1D predefined sequences, direct evolution and frequency binning
# ---------------------------------------------------------------------------

E_2P = dict(Sequence='2pESEEM', Field=340, dt=0.012, nPoints=200, tau=0.096)
E_3P = dict(Sequence='3pESEEM', Field=340, dt=0.012, nPoints=200, tau=0.096, T=0.05)
E_4P = dict(Sequence='4pESEEM', Field=340, dt=0.012, nPoints=200, tau=0.096, T=0.05)


@pytest.mark.parametrize("key,exp_kw", [
    ("y_2p_td", E_2P),
    ("y_3p_td", E_3P),
    ("y_4p_td", E_4P),
])
def test_eseem_direct(refs, key, exp_kw):
    """2p/3p/4pESEEM, 14N + quadrupole, nonzero tau — direct time evolution."""
    y = _run(SYS_14N, exp_kw, dict(GridSize=13, TimeDomain=True))
    _check(y, refs[key], key)


@pytest.mark.parametrize("key,exp_kw", [
    ("y_2p_bin", E_2P),
    ("y_3p_bin", E_3P),
    ("y_4p_bin", E_4P),
])
def test_eseem_binned(refs, key, exp_kw):
    """Same 1D sequences through the frequency-bin accumulation path."""
    y = _run(SYS_14N, exp_kw, dict(GridSize=13, TimeDomain=False, Expand=4))
    _check(y, refs[key], key)


# ---------------------------------------------------------------------------
# HYSCORE — the route the phase defect was reported against
# ---------------------------------------------------------------------------

E_HYS = dict(Sequence='HYSCORE', Field=340, dt=0.012, nPoints=[64, 64],
             tau=0.096, t1=0.02, t2=0.03)
# Unequal axes and time steps: a transposed 2D buffer cannot hide here.
E_HYS_RECT = dict(Sequence='HYSCORE', Field=340, dt=[0.012, 0.008],
                  nPoints=[32, 64], tau=0.096, t1=0.02, t2=0.03)


def test_hyscore_direct(refs):
    y = _run(SYS_14N, E_HYS, dict(GridSize=13, TimeDomain=True))
    _check(y, refs["y_hys_td"], "HYSCORE direct")


def test_hyscore_binned(refs):
    y = _run(SYS_14N, E_HYS, dict(GridSize=13, TimeDomain=False, Expand=2))
    _check(y, refs["y_hys_bin"], "HYSCORE binned")


def test_hyscore_rectangular_direct(refs):
    y = _run(SYS_14N, E_HYS_RECT, dict(GridSize=13, TimeDomain=True))
    _check(y, refs["y_hys_rect_td"], "HYSCORE rectangular direct")


def test_hyscore_rectangular_binned(refs):
    """Non-square nPoints with unequal dt — guards the 2D bin index order."""
    y = _run(SYS_14N, E_HYS_RECT, dict(GridSize=13, TimeDomain=False, Expand=2))
    _check(y, refs["y_hys_rect_bin"], "HYSCORE rectangular binned")


def test_hyscore_direct_matches_binned(refs):
    """The two accumulation modes must agree with each other, not just MATLAB.

    This is the check that would have caught the transposed 2D binning buffer
    without any MATLAB reference at all.
    """
    y_td = np.asarray(_run(SYS_14N, E_HYS_RECT, dict(GridSize=13, TimeDomain=True)))
    y_bin = np.asarray(_run(SYS_14N, E_HYS_RECT,
                            dict(GridSize=13, TimeDomain=False, Expand=6)))
    num = np.linalg.norm(y_td - y_bin)
    den = np.linalg.norm(y_td)
    # Binning discretizes frequencies, so this is a coarse but directional check.
    assert num / den < 0.05, f"direct vs binned disagree by {num / den:.3e}"


# ---------------------------------------------------------------------------
# Orientation selection, quadrupole, S>1/2, and explicit single crystals
# ---------------------------------------------------------------------------

def test_hyscore_orientation_selection(refs):
    """HStrain + ExciteWidth orientation selection (13C, no quadrupole)."""
    sys_kw = dict(SYS_13C, HStrain=[35, 35, 100])
    exp_kw = dict(mwFreq=34.7286, Sequence='HYSCORE', tau=0.112, dt=0.008,
                  nPoints=64, ExciteWidth=80, Field=1238)
    y = _run(sys_kw, exp_kw, dict(GridSize=15, TimeDomain=True, Expand=2))
    _check(y, refs["y_orisel"], "HYSCORE orientation-selective")


def test_s32_hyscore_narrow_excitation(refs):
    """S=3/2 + 14N quadrupole with narrow ExciteWidth.

    Exercises the ``Opt.OriThreshold`` reference level: EasySpin thresholds
    against the unweighted |<i|SyLab|j>| maximum, so a threshold taken after
    excitation weighting keeps transitions EasySpin discards.
    """
    exp_kw = dict(Sequence='HYSCORE', Field=330, dt=0.012, nPoints=[48, 48],
                  tau=0.096, t1=0.02, t2=0.03, mwFreq=9.5, ExciteWidth=100)
    y = _run(SYS_S32, exp_kw, dict(GridSize=11, TimeDomain=True))
    _check(y, refs["y_s32_hys"], "S=3/2 HYSCORE (narrow excitation)")


def test_s32_3peseem_narrow_excitation(refs):
    exp_kw = dict(Sequence='3pESEEM', Field=330, dt=0.012, nPoints=150,
                  tau=0.096, T=0.05, mwFreq=9.5, ExciteWidth=100)
    y = _run(SYS_S32, exp_kw, dict(GridSize=11, TimeDomain=True))
    _check(y, refs["y_s32_3p"], "S=3/2 3pESEEM (narrow excitation)")


def test_p1_crystal_general_orientation(refs):
    """Explicitly declared P1 single crystal at a general orientation.

    ``CrystalSymmetry`` is set explicitly: in EasySpin ``SampleFrame`` alone
    does not select a crystal, so omitting it would compare a crystal against
    a powder.
    """
    exp_kw = dict(E_HYS, SampleFrame=[0.3, 0.7, 0.2], CrystalSymmetry='P1')
    y = _run(SYS_14N, exp_kw, dict(TimeDomain=True))
    _check(y, refs["y_p1_tilt"], "P1 crystal, general orientation")


# ---------------------------------------------------------------------------
# Option-compatibility regression
# ---------------------------------------------------------------------------

def test_expand_default_is_dimension_dependent():
    """EasySpin's Opt.Expand default is 4 for 1D and 2 for 2D sequences."""
    y_default = np.asarray(_run(SYS_14N, E_HYS, dict(GridSize=7, TimeDomain=False)))
    y_explicit = np.asarray(
        _run(SYS_14N, E_HYS, dict(GridSize=7, TimeDomain=False, Expand=2)))
    assert np.allclose(y_default, y_explicit), (
        "2D default Expand should equal Expand=2 (EasySpin's 2D default)"
    )


# ---------------------------------------------------------------------------
# Mims ENDOR — Opt.EndorMethod default
# ---------------------------------------------------------------------------

SYS_ENDOR_1 = dict(S=[0.5], g=[[2.01, 2.005, 1.998]], Nucs='1H',
                   A=[[3, 4, 9]], lwEndor=0.1)
SYS_ENDOR_2 = dict(S=[0.5], g=[[2.01, 2.005, 1.998]], Nucs='1H,1H',
                   A=[[3, 4, 9], [1.5, 2, 4]], lwEndor=0.1)
E_ENDOR = dict(Sequence='MimsENDOR', Field=340, tau=0.1, nPoints=200,
               Range=[8, 20])


def test_mims_endor_single_nucleus(refs):
    """One SHF nucleus: EasySpin's Opt.EndorMethod default is 0, not 1.

    Hardcoding method 1 selects a different RF-pulse model and changes both the
    ENDOR amplitude (~40%) and the lineshape, while leaving peak positions and
    the cosine similarity (0.996) looking healthy.
    """
    y = _run(SYS_ENDOR_1, E_ENDOR, dict(GridSize=13))
    _check(y, refs["y_endor_1nuc"], "MimsENDOR, 1 nucleus")


def test_mims_endor_two_nuclei(refs):
    """Two SHF nuclei: the default switches to EndorMethod 1."""
    y = _run(SYS_ENDOR_2, E_ENDOR, dict(GridSize=13))
    _check(y, refs["y_endor_2nuc"], "MimsENDOR, 2 nuclei")


# ---------------------------------------------------------------------------
# Custom sequences, ProductRule, multi-component
# ---------------------------------------------------------------------------

E_CUSTOM = dict(Field=330.0, Sequence='custom', Flip=[1.0, 1.0, 2.0, 1.0],
                Inc=[0, 1, 2, 0], t=[0.080, 0.0, 0.0, 0.080],
                dt=[0.1, 0.1], nPoints=[32, 32])


@pytest.mark.parametrize("key,opt_kw", [
    ("y_custom_td", dict(TimeDomain=True)),
    ("y_custom_bin", dict(TimeDomain=False, Expand=2)),
])
def test_custom_sequence(refs, key, opt_kw):
    """User-defined HYSCORE-like sequence, 14N + quadrupole.

    EasySpin divides a custom sequence's signal by the largest surviving
    pathway prefactor (saffron.m:1669). Omitting that left the output a factor
    of 8 too small here, with the shape -- and so the cosine -- untouched.
    """
    _, _, info = saffron(SpinSystem(**SYS_14N), PulseExperiment(**E_CUSTOM),
                         SaffronOptions(GridSize=13, **opt_kw))
    _check(info['td'], refs[key], key)


SYS_2N = dict(S=[0.5], g=[[2.0100, 2.0050, 1.9980]], Nucs='14N,1H',
              A=[[2.5, 3.1, 5.4], [1.2, 1.8, 4.0]], Q=[[0.8, 0.5], [0, 0]])
E_HYS_2N = dict(Sequence='HYSCORE', Field=340, dt=0.012, nPoints=[48, 48],
                tau=0.096, t1=0.02, t2=0.03)


@pytest.mark.parametrize("key,pr,td", [
    ("y_pr1_td1", True, True),
    ("y_pr1_td0", True, False),
    ("y_pr0_td1", False, True),
    ("y_pr0_td0", False, False),
])
def test_productrule_two_nuclei(refs, key, pr, td):
    """ProductRule x TimeDomain for two nuclei (14N + quadrupole, and 1H)."""
    opt_kw = dict(GridSize=13, ProductRule=pr, TimeDomain=td)
    if not td:
        opt_kw['Expand'] = 2
    _, y, _ = saffron(SpinSystem(**SYS_2N), PulseExperiment(**E_HYS_2N),
                      SaffronOptions(**opt_kw))
    _check(y, refs[key], key)


def test_two_components_weighted(refs):
    """Two spin systems combined by Sys.weight."""
    s1 = SpinSystem(S=[0.5], g=[[2.01, 2.005, 1.998]], Nucs='14N',
                    A=[[2.5, 3.1, 5.4]], Q=[[0.8, 0.5]])
    s2 = SpinSystem(S=[0.5], g=[[2.02, 2.00, 1.99]], Nucs='1H',
                    A=[[1.2, 1.8, 4.0]], weight=0.3)
    _, y, _ = saffron([s1, s2], PulseExperiment(**E_3P),
                      SaffronOptions(GridSize=13, TimeDomain=True))
    _check(y, refs["y_twocomp"], "two components")
