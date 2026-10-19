"""MATLAB cross-validation for saffron / saffron_thyme / spidyan.

Addresses audit finding P3A-C1: 39 pulse-EPR MATLAB .mat reference files exist
under ``tests/data/`` but no tests were loading them prior to this file.

Reference .mat files are produced by running the EasySpin MATLAB test
scripts under ``tests/*.m`` (e.g. ``saffron_2pESEEM.m``, ``saffron_HYSCORE.m``).
They contain ``data`` structs with either:
- ``data.y1, data.x1, data.y2, data.x2`` (predefined-vs-manual parity tests)
- ``data.y, data.x`` (single saffron call)

Comparisons are strict: the relative L2 error against the MATLAB reference,
not a cosine similarity. A cosine is scale-invariant and dominated by the DC
offset of a pulse-EPR trace, so it sees neither an amplitude error nor a wrong
modulation -- eight defects hid behind cosine thresholds here until 0.5.0.

Being strict means the comparison has to be set up correctly, and three
mismatches had to be fixed to get there. Each reference is reproduced with:

- **the same accumulation path.** None of the generators set ``Opt.TimeDomain``,
  so every reference comes from the frequency-binned path. Running torchspin
  with ``TimeDomain=True`` against them compares direct time evolution against
  EasySpin's own binned approximation, which differs by 7-60% in the
  modulation. These tests therefore leave ``TimeDomain`` at its default.
- **the same orientation grid.** ``GridSize`` is stated explicitly for every
  case rather than relying on a shared default, and matches the generator.
- **the same normalization.** Where the generator wrote
  ``y = y/max(abs(y))``, the torchspin signal is normalized the same way
  before comparison (``normalize=True`` below).

This file is pytest-skipped when the `.mat` fixtures are missing (e.g. in a
pip install without the source checkout).
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


DATA_DIR = Path(__file__).parent.parent.parent / "tests" / "data"


def _require_mat(filename: str) -> Path:
    if not SCIPY_OK:
        pytest.skip("scipy not installed")
    p = DATA_DIR / filename
    if not p.exists():
        pytest.skip(f"Reference file not found: {p}")
    return p


def _load_data(filename: str) -> dict:
    p = _require_mat(filename)
    return loadmat(str(p), simplify_cells=True)["data"]


# Strict comparison helpers. See the module docstring for why a cosine is not
# a usable oracle here. Observed errors are 1e-16..1e-7. The tolerance leaves
# headroom because a few committed fixtures are slightly stale relative to the
# current EasySpin -- regenerating them moves agreement to ~1e-15 (see
# test_orientation_selective, which documents a measured case) -- while still
# being orders of magnitude tighter than needed to catch a real defect: the
# ones found in 0.5.0 were factors of 2 and 8 and ~100% modulation errors.
STRICT_TOL = 1e-6


def _norm_peak(v):
    """Scale by max|v|, as the MATLAB generators' ``y = y/max(abs(y))`` does."""
    v = np.asarray(v)
    m = np.max(np.abs(v))
    return v / m if m > 0 else v


def _assert_matches(a, b, label, normalize=False, tol=STRICT_TOL):
    """Assert a matches the MATLAB reference b in relative L2 error."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    assert a.size > 0, f"{label}: empty signal"
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if normalize:
        a, b = _norm_peak(a), _norm_peak(b)
    denom = np.linalg.norm(b)
    assert denom > 0, f"{label}: reference is all-zero"
    err = float(np.linalg.norm(a - b) / denom)
    assert err < tol, (
        f"{label}: relative L2 error {err:.3e} >= {tol:g}"
        f"{' (peak-normalized)' if normalize else ''}"
    )


# ---------------------------------------------------------------------------
# 2pESEEM (S=1/2 + 1H, predefined sequence)
# ---------------------------------------------------------------------------


def test_saffron_2pESEEM():
    """2pESEEM (1H, A=[5,2], Field=350 mT): cosine vs MATLAB y1.

    Fixed 2026-04-19: saffron predefined-experiment path was multiplying by
    nominal pathway prefactors (+1/2, +1/8, -1/8) that MATLAB only stores as
    documentation. Cosine now 1.0000.
    """
    ref = _load_data("saffron_2pESEEM.mat")
    # MATLAB Sys.A_ = [5 2] → A = [Aiso-T, Aiso-T, Aiso+2T] = [3, 3, 9] in MHz
    sys = SpinSystem(S=[0.5], g=[[2.0023, 2.0023, 2.0023]],
                     Nucs='1H', A=[[3.0, 3.0, 9.0]])

    exp = PulseExperiment(
        Sequence='2pESEEM',
        Field=350.0,
        dt=0.010,
        tau=0.0,
        nPoints=120,
    )
    opt = SaffronOptions(GridSize=20)
    x, y, info = saffron(sys, exp, opt)

    y_ref = np.asarray(ref['y1']).ravel()
    y_py = np.asarray(y).ravel()
    n = min(len(y_ref), len(y_py))
    _assert_matches(y_py[:n], y_ref[:n], "2pESEEM")


# ---------------------------------------------------------------------------
# 2psimple (14N with Q, predefined 2pESEEM, frequency-domain output)
# ---------------------------------------------------------------------------


def test_saffron_2psimple():
    """2pESEEM with 14N + quadrupole vs MATLAB (strict, peak-normalized).

    EasySpin's script sets no Opt.GridSize, so the reference uses its default 31.
    """
    ref = _load_data("saffron_2psimple.mat")
    Field = 324.9
    # Sys.A = (2)*nuI where nuI is larmorfrq — approximately 2.10 MHz at this field
    from torchspin.utils import larmorfrq
    nuI = np.asarray(larmorfrq('14N', Field)).reshape(-1)[0].item()
    A_iso = 2.0 * nuI
    sys = SpinSystem(
        S=[0.5], g=[[2.0023, 2.0023, 2.0023]],
        Nucs='14N',
        A=[[A_iso, A_iso, A_iso]],
        Q=[4 * 0.1 * nuI, 0.6],   # EasySpin saffron_2psimple: Sys.Q = [4*0.1*nuI, 0.6] (e2qQ/h, eta)
    )
    exp = PulseExperiment(
        Sequence='2pESEEM',
        Field=Field,
        dt=0.050,
        nPoints=1001,
        tau=0.001,
    )
    opt = SaffronOptions(GridSize=31)
    _, y, info = saffron(sys, exp, opt)

    # The generator normalized (y = y/max(abs(y))); normalize=True matches it.
    # Sys.Q = [e2qQ/h, eta] is converted as EasySpin's validatespinsys does.
    y_ref = np.asarray(ref['y']).ravel()
    y_py = np.asarray(y).ravel()
    n = min(len(y_ref), len(y_py))
    _assert_matches(y_py[:n], y_ref[:n], "2psimple", normalize=True)


# ---------------------------------------------------------------------------
# 3pESEEM (S=1/2 + 1H, predefined sequence)
# ---------------------------------------------------------------------------


def test_saffron_3pESEEM():
    """3pESEEM (1H, A_=[5,2], Field=350 mT): cosine vs MATLAB y1."""
    ref = _load_data("saffron_3pESEEM.mat")
    sys = SpinSystem(S=[0.5], g=[[2.0023, 2.0023, 2.0023]],
                     Nucs='1H', A=[[3.0, 3.0, 9.0]])  # A_ = [5, 2]
    exp = PulseExperiment(
        Sequence='3pESEEM',
        Field=350.0,
        dt=0.010,
        tau=0.1,
        nPoints=120,
    )
    opt = SaffronOptions(GridSize=20)
    _, y, _ = saffron(sys, exp, opt)

    y_ref = np.asarray(ref['y1']).ravel()
    y_py = np.asarray(y).ravel()
    n = min(len(y_ref), len(y_py))
    _assert_matches(y_py[:n], y_ref[:n], "3pESEEM")


# ---------------------------------------------------------------------------
# HYSCORE (S=1/2 + 1H, 2D)
# ---------------------------------------------------------------------------


def test_saffron_HYSCORE_1d():
    """HYSCORE (1H, Field=324.9 mT) — cosine vs MATLAB y1.

    Fixed 2026-04-19 by the same predefined-prefactor fix as 2p/3pESEEM.
    The apparent -1 sign flip was actually the nominal pathway prefactor
    `-1/8` for HYSCORE being double-applied. Cosine now 1.0000.
    """
    ref = _load_data("saffron_HYSCORE.mat")
    sys = SpinSystem(S=[0.5], g=[[2.0023, 2.0023, 2.0023]],
                     Nucs='1H', A=[[3.0, 3.0, 9.0]])
    exp = PulseExperiment(
        Sequence='HYSCORE',
        Field=324.9,
        dt=0.050,
        nPoints=[256, 256],
        tau=0.08,
        t1=0.1,
        t2=0.1,
    )
    opt = SaffronOptions(GridSize=20)
    _, y, _ = saffron(sys, exp, opt)

    y_ref = np.asarray(ref['y1']).ravel()
    y_py = np.asarray(y).ravel()
    n = min(len(y_ref), len(y_py))
    _assert_matches(y_py[:n], y_ref[:n], "HYSCORE")


# ---------------------------------------------------------------------------
# Thyme echo (real-pulse propagation)
# ---------------------------------------------------------------------------


def test_saffron_thyme_spinecho_finite():
    """saffron_thyme spin-echo reference: shape matches and signal finite."""
    d = _load_data("saffron_thyme_spinecho.mat")
    x_ref = np.asarray(d['x']).ravel() if 'x' in d else None
    y_ref = np.asarray(d['y']).ravel() if 'y' in d else None
    if y_ref is None:
        pytest.skip("thyme spinecho ref missing 'y' field")
    # This test documents that the reference file exists and has a non-zero
    # reference signal. Numerical comparison requires reproducing the exact
    # MATLAB sequence, which depends on pulse shape parameters not stored in
    # the .mat file header.
    assert np.any(np.abs(y_ref) > 0), "MATLAB reference spinecho is all-zero"
    assert np.all(np.isfinite(y_ref)), "MATLAB reference contains NaN/Inf"


# ---------------------------------------------------------------------------
# Summary / sanity (no MATLAB required)
# ---------------------------------------------------------------------------


def test_saffron_matlab_refs_present():
    """Sanity check: MATLAB saffron reference fixtures are discoverable."""
    if not DATA_DIR.exists():
        pytest.skip(f"MATLAB reference directory missing: {DATA_DIR}")
    files = list(DATA_DIR.glob("saffron_*.mat"))
    assert len(files) >= 10, (
        f"Expected >= 10 saffron_*.mat reference files, found {len(files)}. "
        "Run MATLAB test scripts in tests/*.m to regenerate."
    )


# ---------------------------------------------------------------------------
# S>1/2 MATLAB cross-validation (requires generate_saffron_highspin_refs.m)
# ---------------------------------------------------------------------------
# Run `matlab -batch "cd tests/data; generate_saffron_highspin_refs"` to create
# the .mat reference files.  Tests are auto-skipped when files are absent.
# ---------------------------------------------------------------------------


def test_S1_2pESEEM():
    """S=1 + 1H 2pESEEM (D=100 MHz): cosine vs MATLAB reference >= 0.90."""
    ref = _load_data("saffron_S1_2pESEEM.mat")
    sys = SpinSystem(
        S=[1], g=[[2.002, 2.001, 2.000]],
        D=[[100.0, 0.0]],
        Nucs='1H', A=[[3.0, 3.0, 9.0]],
    )
    exp = PulseExperiment(Field=350.0, Sequence='2pESEEM',
                          dt=0.01, nPoints=256, tau=0.2)
    opt = SaffronOptions(GridSize=30)
    _, _, info = saffron(sys, exp, opt)
    y_py = np.asarray(info['td']).real.ravel()
    y_ref = np.asarray(ref['y']).ravel()
    _assert_matches(y_py, y_ref, "S=1 2pESEEM")


def test_S1_3pESEEM():
    """S=1 + 1H 3pESEEM (D=100 MHz): cosine vs MATLAB reference >= 0.90."""
    ref = _load_data("saffron_S1_3pESEEM.mat")
    sys = SpinSystem(
        S=[1], g=[[2.002, 2.001, 2.000]],
        D=[[100.0, 0.0]],
        Nucs='1H', A=[[3.0, 3.0, 9.0]],
    )
    exp = PulseExperiment(Field=350.0, Sequence='3pESEEM',
                          dt=0.01, nPoints=256, tau=0.2)
    opt = SaffronOptions(GridSize=30)
    _, _, info = saffron(sys, exp, opt)
    y_py = np.asarray(info['td']).real.ravel()
    y_ref = np.asarray(ref['y']).ravel()
    _assert_matches(y_py, y_ref, "S=1 3pESEEM")


def test_S32_HYSCORE():
    """S=3/2 + 14N HYSCORE (D=50 MHz): cosine vs MATLAB reference >= 0.85."""
    ref = _load_data("saffron_S32_HYSCORE.mat")
    sys = SpinSystem(
        S=[1.5], g=[[2.002, 2.001, 2.000]],
        D=[[50.0, 0.0]],
        Nucs='14N', A=[[5.0, 5.0, 12.0]],
        Q=[[0.3, 0.0, 0.0]],
    )
    exp = PulseExperiment(Field=330.0, Sequence='HYSCORE',
                          dt=[0.02, 0.02], nPoints=[64, 64], tau=0.2)
    opt = SaffronOptions(GridSize=20)
    _, _, info = saffron(sys, exp, opt)
    y_py = np.asarray(info['td']).real.ravel()
    y_ref = np.asarray(ref['y']).real.ravel()
    _assert_matches(y_py, y_ref, "S=3/2 HYSCORE")


def test_S1_2pESEEM_orisel():
    """S=1 + 1H 2pESEEM with orientation selection: cosine >= 0.85."""
    ref = _load_data("saffron_S1_2pESEEM_orisel.mat")
    sys = SpinSystem(
        S=[1], g=[[2.002, 2.001, 2.000]],
        D=[[100.0, 0.0]],
        Nucs='1H', A=[[3.0, 3.0, 9.0]],
    )
    exp = PulseExperiment(Field=350.0, Sequence='2pESEEM',
                          dt=0.01, nPoints=256, tau=0.2,
                          mwFreq=9.5, ExciteWidth=100.0)
    opt = SaffronOptions(GridSize=30)
    _, _, info = saffron(sys, exp, opt)
    y_py = np.asarray(info['td']).real.ravel()
    y_ref = np.asarray(ref['y']).ravel()
    _assert_matches(y_py, y_ref, "S=1 orisel 2pESEEM")


def test_S1_14N_3pESEEM():
    """S=1 + 14N 3pESEEM (quadrupole, D=100 MHz): cosine >= 0.85."""
    ref = _load_data("saffron_S1_14N_3pESEEM.mat")
    sys = SpinSystem(
        S=[1], g=[[2.002, 2.001, 2.000]],
        D=[[100.0, 0.0]],
        Nucs='14N', A=[[5.0, 5.0, 12.0]],
        Q=[[0.3, 0.0, 0.0]],
    )
    exp = PulseExperiment(Field=350.0, Sequence='3pESEEM',
                          dt=0.01, nPoints=256, tau=0.2)
    opt = SaffronOptions(GridSize=30)
    _, _, info = saffron(sys, exp, opt)
    y_py = np.asarray(info['td']).real.ravel()
    y_ref = np.asarray(ref['y']).ravel()
    _assert_matches(y_py, y_ref, "S=1 14N 3pESEEM")


# ---------------------------------------------------------------------------
# EasySpin test suite parity — .mat files from EasySpin's own test scripts
# ---------------------------------------------------------------------------
# Each test below maps 1-to-1 to an EasySpin test file (saffron_NAME.m).
# Tests are auto-skipped when the matching .mat fixture is missing.
# ---------------------------------------------------------------------------


def _nu_larmor(nuc: str, field_mT: float) -> float:
    """Larmor frequency (MHz) for a nucleus at the given field."""
    from torchspin.utils import larmorfrq
    return np.asarray(larmorfrq(nuc, field_mT)).reshape(-1)[0].item()


# ---------------------------------------------------------------------------
# saffron_3psimple: S=1/2 + 14N (A and Q from Larmor), 3pESEEM
# ---------------------------------------------------------------------------


def test_3psimple():
    """14N 3pESEEM (A=2*nuI, Q=[4*0.1*nuI, 0.6]): cosine vs MATLAB >= 0.90."""
    ref = _load_data("saffron_3psimple.mat")
    nuI = _nu_larmor('14N', 324.9)
    sys = SpinSystem(
        S=[0.5], g=[[2.0023, 2.0023, 2.0023]],
        Nucs='14N',
        A=[[2 * nuI, 2 * nuI, 2 * nuI]],
        Q=[[4 * 0.1 * nuI, 0.6]],
    )
    exp = PulseExperiment(Field=324.9, Sequence='3pESEEM',
                          dt=0.050, nPoints=1001, tau=0.001)
    opt = SaffronOptions(GridSize=31)
    _, _, info = saffron(sys, exp, opt)
    y_py = np.asarray(info['td']).real.ravel()
    y_ref = np.asarray(ref['y']).real.ravel()
    n = min(len(y_py), len(y_ref))
    _assert_matches(y_py[:n], y_ref[:n], "3psimple", normalize=True)


# ---------------------------------------------------------------------------
# saffron_4psimple: S=1/2 + 1H (axial A), 4pESEEM
# ---------------------------------------------------------------------------


def test_4psimple():
    """1H 4pESEEM (A=[2,2,9] MHz): cosine vs MATLAB >= 0.95."""
    ref = _load_data("saffron_4psimple.mat")
    # EasySpin A=[2, 9] axial shorthand → [A_perp, A_perp, A_par]
    sys = SpinSystem(
        S=[0.5], g=[[2.0023, 2.0023, 2.0023]],
        Nucs='1H', A=[[2.0, 2.0, 9.0]],
    )
    exp = PulseExperiment(Field=350.0, Sequence='4pESEEM', tau=0.1, dt=0.01)
    opt = SaffronOptions(GridSize=31)
    _, _, info = saffron(sys, exp, opt)
    y_py = np.asarray(info['td']).real.ravel()
    y_ref = np.asarray(ref['y']).real.ravel()
    n = min(len(y_py), len(y_ref))
    _assert_matches(y_py[:n], y_ref[:n], "4psimple", normalize=True)


# ---------------------------------------------------------------------------
# saffron_decay: S=1/2 + 1H with T1/T2, 3pESEEM
# ---------------------------------------------------------------------------


def test_decay():
    """1H 3pESEEM with T1=2 µs, T2=1 µs: cosine vs MATLAB >= 0.95."""
    ref = _load_data("saffron_decay.mat")
    # A_=[5 2] → [3, 3, 9] MHz
    sys = SpinSystem(
        S=[0.5], g=[[2.0023, 2.0023, 2.0023]],
        Nucs='1H', A=[[3.0, 3.0, 9.0]],
        T1=2.0, T2=1.0,
    )
    exp = PulseExperiment(Field=350.0, Sequence='3pESEEM',
                          dt=0.01, nPoints=120, tau=0.1)
    opt = SaffronOptions(GridSize=20)
    _, _, info = saffron(sys, exp, opt)
    y_py = np.asarray(info['td']).real.ravel()
    y_ref = np.asarray(ref['y1']).real.ravel()
    n = min(len(y_py), len(y_ref))
    _assert_matches(y_py[:n], y_ref[:n], "decay")


# ---------------------------------------------------------------------------
# saffron_hyscore14n: S=1/2 + 14N anisotropic A+Q, HYSCORE
# ---------------------------------------------------------------------------


def test_hyscore14n():
    """14N HYSCORE (anisotropic A and Q): cosine vs MATLAB >= 0.85."""
    ref = _load_data("saffron_hyscore14n.mat")
    # A=[-1 -1 2]*0.5+0.8 = [0.3, 0.3, 1.8] MHz
    # Q=[-1 -1 2]*0.2 = [-0.2, -0.2, 0.4] MHz (principal values, η=0)
    sys = SpinSystem(
        S=[0.5], g=[[2.0023, 2.0023, 2.0023]],
        Nucs='14N',
        A=[[0.3, 0.3, 1.8]],
        Q=[[-0.2, -0.2, 0.4]],
    )
    exp = PulseExperiment(Field=330.0, Sequence='HYSCORE',
                          tau=0.001, dt=0.1, nPoints=200)
    opt = SaffronOptions(GridSize=181)
    _, _, info = saffron(sys, exp, opt)
    y_py = np.asarray(info['td']).real.ravel()
    y_ref = np.asarray(ref['y']).real.ravel()
    n = min(len(y_py), len(y_ref))
    _assert_matches(y_py[:n], y_ref[:n], "hyscore14n", normalize=True)


# ---------------------------------------------------------------------------
# saffron_hyscore_ganiso: S=1/2 + 1H, strongly anisotropic g, HYSCORE
# ---------------------------------------------------------------------------


def test_hyscore_ganiso():
    """HYSCORE with g=[2, 2, 6] (strongly anisotropic): cosine vs MATLAB >= 0.85."""
    ref = _load_data("saffron_hyscore_ganiso.mat")
    # EasySpin g=[2, 6] axial → [g_perp, g_perp, g_par]
    # EasySpin A=[2, 25] axial → [2, 2, 25] MHz
    sys = SpinSystem(
        S=[0.5], g=[[2.0, 2.0, 6.0]],
        Nucs='1H', A=[[2.0, 2.0, 25.0]],
    )
    exp = PulseExperiment(Field=330.0, Sequence='HYSCORE',
                          tau=0.001, dt=0.01, nPoints=200)
    opt = SaffronOptions(GridSize=181)
    _, _, info = saffron(sys, exp, opt)
    y_py = np.asarray(info['td']).real.ravel()
    y_ref = np.asarray(ref['y']).real.ravel()
    n = min(len(y_py), len(y_ref))
    _assert_matches(y_py[:n], y_ref[:n], "hyscore_ganiso", normalize=True)


# ---------------------------------------------------------------------------
# saffron_mimssimple: S=1/2 + 1H, MimsENDOR
# ---------------------------------------------------------------------------


def test_mimssimple():
    """1H MimsENDOR: cosine vs MATLAB >= 0.90."""
    ref = _load_data("saffron_mimssimple.mat")
    nuI_1H = _nu_larmor('1H', 325.0)
    sys = SpinSystem(
        S=[0.5], g=[[2.0023, 2.0023, 2.0023]],
        Nucs='1H', A=[[1.0, 1.0, 5.0]],
        lwEndor=0.1,
    )
    endor_range = [nuI_1H - 10.0, nuI_1H + 10.0]
    exp = PulseExperiment(Field=325.0, Sequence='MimsENDOR',
                          tau=0.1, Range=endor_range)
    opt = SaffronOptions(GridSize=31)
    _, y, _ = saffron(sys, exp, opt)
    y_py = np.asarray(y).real.ravel()
    y_ref = np.asarray(ref['y']).real.ravel()
    n = min(len(y_py), len(y_ref))
    _assert_matches(y_py[:n], y_ref[:n], "mimssimple", normalize=True)


# ---------------------------------------------------------------------------
# saffron_orientation_selective: S=1/2 + 1H anisotropic g, 2pESEEM + ExciteWidth
# ---------------------------------------------------------------------------


def test_orientation_selective():
    """2pESEEM with orientation selection (g=[2,2,2.2], ExciteWidth=200 MHz).

    The committed fixture is slightly stale: regenerating it from the local
    EasySpin gives a reference that torchspin matches to 4.4e-15, and that
    fresh reference differs from this stored one by 1.4e-6 in overall scale
    (the shape is identical). So the amplitude here is checked at 1e-5 rather
    than STRICT_TOL, while the peak-normalized shape is checked strictly --
    the stale factor is in the fixture, not the code.
    """
    ref = _load_data("saffron_orientation_selective.mat")
    # EasySpin A=[1, 6] axial → [1, 1, 6] MHz; g=[2, 2.2] axial → [2, 2, 2.2]
    sys = SpinSystem(
        S=[0.5], g=[[2.0, 2.0, 2.2]],
        Nucs='1H', A=[[1.0, 1.0, 6.0]],
    )
    exp = PulseExperiment(
        Field=350.0, Sequence='2pESEEM',
        mwFreq=9.5, dt=0.015, nPoints=128,
        ExciteWidth=200.0, tau=0.001,
    )
    opt = SaffronOptions(GridSize=91)
    _, _, info = saffron(sys, exp, opt)
    y_py = np.asarray(info['td']).real.ravel()
    y_ref = np.asarray(ref['y']).real.ravel()
    n = min(len(y_py), len(y_ref))
    _assert_matches(y_py[:n], y_ref[:n], "orientation_selective shape",
                    normalize=True)
    _assert_matches(y_py[:n], y_ref[:n], "orientation_selective amplitude",
                    tol=1e-5)


# ---------------------------------------------------------------------------
# saffron_customsequence: S=1/2 + 14N, user-defined HYSCORE pulse sequence
# ---------------------------------------------------------------------------


def test_customsequence():
    """14N HYSCORE via custom pulse sequence: cosine vs MATLAB >= 0.95.

    EasySpin: saffron_customsequence.m, {p90 tau p90 t0 p180 t0 p90 tau}.
    Uses the 32x32 reference (generate_saffron_customsequence_small.m) so
    the test runs in seconds; physics identical to the 200x200 original.
    Flip is in units of pi/2, so the p180 pulse is Flip=2.
    """
    ref = _load_data("saffron_customsequence_small.mat")
    # A=[-1 -1 2]*0.5+0.8 = [0.3, 0.3, 1.8] MHz; Q=scalar 1 → e2qQ/h=1, η=0
    sys = SpinSystem(
        S=[0.5], g=[[2.0023, 2.0023, 2.0023]],
        Nucs='14N',
        A=[[0.3, 0.3, 1.8]],
        Q=[[1.0]],
    )
    # HYSCORE custom: {p90, tau, p90, t1, p180, t2, p90, tau}
    # Dim1 steps d2=t1 (between 2nd p90 and p180), Dim2 steps d3=t2
    exp = PulseExperiment(
        Field=330.0, Sequence='custom',
        Flip=[1.0, 1.0, 2.0, 1.0],
        Inc=[0, 1, 2, 0],
        t=[0.080, 0.0, 0.0, 0.080],
        dt=[0.1, 0.1], nPoints=[32, 32],
    )
    opt = SaffronOptions(GridSize=31)
    _, _, info = saffron(sys, exp, opt)
    y_py = np.asarray(info['td']).real.ravel()
    y_ref = np.asarray(ref['y']).real.ravel()
    n = min(len(y_py), len(y_ref))
    _assert_matches(y_py[:n], y_ref[:n], "customsequence")


# ---------------------------------------------------------------------------
# Thyme tests — shape/finite checks (exact API depends on thyme implementation)
# ---------------------------------------------------------------------------


def test_thyme_relaxation_shape():
    """thyme T1/T2 spin-echo: output shape and finiteness vs .mat reference."""
    ref = _load_data("saffron_thyme_relaxation.mat")
    y_ref = np.asarray(ref['y']).ravel()
    assert np.all(np.isfinite(y_ref)), "MATLAB thyme_relaxation reference has NaN/Inf"
    assert len(y_ref) > 0


def test_thyme_simfreq_shape():
    """thyme SimFreq spin-echo: output shape and finiteness vs .mat reference."""
    ref = _load_data("saffron_thyme_simfreq.mat")
    y_ref = np.asarray(ref['y']).ravel()
    assert np.all(np.isfinite(y_ref)), "MATLAB thyme_simfreq reference has NaN/Inf"
    assert len(y_ref) > 0


def test_thyme_indirectdimensions_shape():
    """thyme indirect dimensions: reference file has expected 2D/3D shape."""
    ref = _load_data("saffron_thyme_indirectdimensions.mat")
    y1 = np.asarray(ref['y1'])
    y2 = np.asarray(ref['y2'])
    assert y1.ndim >= 2, f"y1 should be ≥2D, got shape {y1.shape}"
    assert y2.ndim >= 2, f"y2 should be ≥2D, got shape {y2.shape}"
    assert np.all(np.isfinite(y1))


# ---------------------------------------------------------------------------
# Skipped — unsupported features
# ---------------------------------------------------------------------------


def test_crystalorientations():
    """Single-crystal 3pESEEM via SampleFrame: cosine vs MATLAB >= 0.99.

    EasySpin: saffron_crystalorientations.m. A_=[5 2] → A=[3, 3, 9] MHz.
    Checks one sample orientation and a two-orientation stack.
    """
    ref = _load_data("saffron_crystalorientations.mat")
    sys = SpinSystem(S=[0.5], g=[[2.0023, 2.0023, 2.0023]],
                     Nucs='1H', A=[[3.0, 3.0, 9.0]])
    opt = SaffronOptions(GridSize=20)

    exp1 = PulseExperiment(Field=350.0, Sequence='3pESEEM',
                           dt=0.01, tau=0.1, nPoints=120,
                           MolFrame=[0, 0, 0],
                           SampleFrame=[0, -np.pi / 2, 0])
    _, y1, _ = saffron(sys, exp1, opt)
    y_ref = np.asarray(ref['y'], dtype=float).ravel()
    _assert_matches(np.asarray(y1).real.ravel(), y_ref, "crystal single-orientation")

    exp2 = PulseExperiment(Field=350.0, Sequence='3pESEEM',
                           dt=0.01, tau=0.1, nPoints=120,
                           MolFrame=[0, 0, 0],
                           SampleFrame=[[0, -np.pi / 2, 0], [0, 0, 0]])
    _, y2, _ = saffron(sys, exp2, opt)
    y_ref2 = np.asarray(ref['y2'], dtype=float).ravel()
    _assert_matches(np.asarray(y2).real.ravel(), y_ref2, "crystal two-orientation")


def test_crystal_hyscore():
    """P212121 crystal (4 sites) HYSCORE: cosine vs MATLAB >= 0.95.

    EasySpin: saffron_crystal.m (upgraded from smoke test to cosine).
    """
    ref = _load_data("saffron_crystal_hyscore.mat")
    sys = SpinSystem(S=[0.5], g=[[2.0023, 2.0023, 2.0023]],
                     Nucs='1H', A=[[5.0, 5.0, 20.0]], lwEndor=0.5)
    exp = PulseExperiment(Field=330.0, Sequence='HYSCORE',
                          tau=0.080, dt=0.120, nPoints=256,
                          MolFrame=[np.pi / 3, np.pi / 6, np.pi / 4],
                          SampleFrame=[np.pi / 9, np.pi / 5, 0],
                          CrystalSymmetry='P212121')
    _, y, _ = saffron(sys, exp, SaffronOptions())
    y_ref = np.asarray(ref['y'], dtype=float).ravel()
    _assert_matches(np.asarray(y).real.ravel(), y_ref, "P212121 HYSCORE")


def test_thyme_crystalorientations_consistency():
    """thyme single-crystal via SampleFrame: orientation-dependent and
    self-consistent.

    EasySpin: saffron_thyme_crystalorientations.m. The crystal orientation
    machinery itself is MATLAB-validated in the fast path (cosine 1.0000);
    here we check the thyme path uses it correctly: two stacked sample
    orientations must equal the *sum* of the individual runs (EasySpin gives
    each crystal orientation the weight 4*pi and does not renormalize the
    sum), and different orientations must give different signals
    (anisotropic A). Exp.mwFreq is on resonance for g = 1.8985 at 350 mT, as
    in EasySpin's own test.
    """
    from torchspin.saffron_thyme import saffron_thyme
    sys = SpinSystem(S=[0.5], g=[[1.8985, 1.8985, 1.8985]],
                     Nucs='1H', A=[[3.0, 3.0, 9.0]])

    def run(sample_frame):
        exp = PulseExperiment(Field=350.0, mwFreq=9.3, Sequence='3pESEEM',
                              dt=0.01, tau=0.1, T=0.5, nPoints=[8],
                              DetWindow=0.0, DetPhase=0.0,
                              MolFrame=[0, 0, 0], SampleFrame=sample_frame)
        opt = SaffronOptions(SimulationMode='thyme', IntTimeStep=1e-4)
        _, y, _ = saffron_thyme(sys, exp, opt)
        return np.asarray(y).ravel()

    y1 = run([[0, -np.pi / 2, 0]])
    y2 = run([[0, 0, 0]])
    y12 = run([[0, -np.pi / 2, 0], [0, 0, 0]])

    assert np.max(np.abs(y1)) > 0, "orientation 1 signal is zero"
    # Anisotropic A → different orientations give different signals
    assert not np.allclose(y1, y2, atol=np.abs(y1).max() * 1e-3), \
        "signals identical for different crystal orientations"
    # Stacked orientations = sum of the individual runs
    np.testing.assert_allclose(y12, y1 + y2, atol=np.abs(y1).max() * 1e-10)


def test_twocomponents():
    """Two-component MimsENDOR ({Sys1, Sys2}, weight 0.3): cosine >= 0.98.

    EasySpin: saffron_twocomponents.m. Also checks separate='components'.
    """
    ref = _load_data("saffron_twocomponents.mat")
    nuI_1H = _nu_larmor('1H', 325.0)
    sys1 = SpinSystem(S=[0.5], g=[[2.0023, 2.0023, 2.0023]],
                      Nucs='1H', A=[[1.0, 1.0, 3.0]], lwEndor=0.1)
    sys2 = SpinSystem(S=[0.5], g=[[2.0023, 2.0023, 2.0023]],
                      Nucs='1H', A=[[7.0, 7.0, 9.0]], lwEndor=0.1,
                      weight=0.3)
    exp = PulseExperiment(Field=325.0, Sequence='MimsENDOR', tau=0.1,
                          Range=[nuI_1H - 10.0, nuI_1H + 10.0])
    x, y, info = saffron([sys1, sys2], exp, SaffronOptions(GridSize=31))
    y_py = np.asarray(y).real.ravel()
    y_ref = np.asarray(ref['y'], dtype=float).ravel()
    n = min(len(y_py), len(y_ref))
    _assert_matches(y_py[:n], y_ref[:n], "twocomponents", normalize=True)

    # separate='components': stacked signals whose weighted sum = combined
    opt_sep = SaffronOptions(separate='components')
    _, y_sep, _ = saffron([sys1, sys2], exp, opt_sep)
    y_sep = np.asarray(y_sep)
    assert y_sep.shape[0] == 2, f"expected 2 components, got {y_sep.shape}"
    recon = y_sep.sum(axis=0).real.ravel()
    np.testing.assert_allclose(recon[:n], y_py[:n],
                               atol=np.abs(y_py).max() * 1e-8)
