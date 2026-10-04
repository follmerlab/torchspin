"""
Extended pepper MATLAB validation (PORT_SPEC_2 Phase 1).

Reference: tests/data/ref_pepper_ext.mat from tests/data/generate_pepper_ext_refs.m.
Cases are grouped by the Phase-1 item that introduced them; thresholds are
cosine similarity on the full spectrum.

1.1 symmetry frames: axial / rhombic tensors whose symmetry frame is not the
molecular frame (tilted gFrame/DFrame, x- or y-axial tensors, permuted
principal axes). Before the fix pepper interpolated the rotated grid in the
molecular frame, which broke every non-identity Dinfh frame (cosine 0.34,
or an all-zero spectrum for an x-axial g).
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io import loadmat

from torchspin.spinsystem import SpinSystem
from torchspin.experiment import Experiment, Options
from torchspin.pepper import pepper

REF_FILE = Path(__file__).parent.parent.parent / "tests" / "data" / "ref_pepper_ext.mat"


@pytest.fixture(scope="module")
def cases():
    if not REF_FILE.exists():
        pytest.skip(f"Reference data not found: {REF_FILE}")
    refs = loadmat(str(REF_FILE), squeeze_me=True, struct_as_record=False)
    return {str(c.name): c for c in np.atleast_1d(refs['cases'])}


def _get(o, name, default=None):
    return getattr(o, name) if name in getattr(o, '_fieldnames', []) else default


def sys_from_mat(S) -> SpinSystem:
    Sv = _get(S, 'S', 0.5)
    Sv = np.atleast_1d(np.asarray(Sv, dtype=float)).tolist()
    n_e = len(Sv)
    kw = {'S': Sv}
    g = _get(S, 'g')
    if g is not None:
        g = np.asarray(g, dtype=float)
        if g.ndim == 1 and g.size == 2:        # EasySpin axial shorthand [gperp gpar]
            g = np.array([g[0], g[0], g[1]])
        kw['g'] = g.tolist() if g.ndim else float(g)
    for name in ('gFrame', 'DFrame', 'AFrame', 'A'):
        v = _get(S, name)
        if v is not None:
            kw[name] = np.asarray(v, dtype=float).tolist()
    D = _get(S, 'D')
    if D is not None:
        D = np.asarray(D, dtype=float)
        if D.ndim == 0:
            kw['D'] = float(D)                 # scalar D -> [D, E=0] (EasySpin semantics)
        elif D.ndim == 1 and D.size == 2 and n_e == 1:
            kw['D'] = [D.tolist()]
        else:
            kw['D'] = D.tolist()
    nucs = _get(S, 'Nucs')
    if nucs is not None:
        kw['Nucs'] = str(nucs)
    lw = _get(S, 'lw')
    if lw is not None:
        lw = np.atleast_1d(np.asarray(lw, dtype=float)).tolist()
        kw['lw'] = lw if len(lw) == 2 else [lw[0], 0.0]
    return SpinSystem(**kw)


def exp_from_mat(E) -> Experiment:
    kw = {'mwFreq': float(E.mwFreq)}
    if 'Range' in getattr(E, '_fieldnames', []):
        kw['Range'] = np.asarray(E.Range, dtype=float).tolist()
    else:
        kw['CenterSweep'] = np.asarray(E.CenterSweep, dtype=float).tolist()
    for name in ('nPoints', 'Harmonic'):
        v = _get(E, name)
        if v is not None:
            kw[name] = int(v)
    if 'Harmonic' not in kw:
        kw['Harmonic'] = 1
    return Experiment(**kw)


def opt_from_mat(O) -> Options:
    kw = {'Verbosity': 0}
    gs = _get(O, 'GridSize')
    if gs is not None:
        gs = np.atleast_1d(np.asarray(gs, dtype=float)).astype(int).tolist()
        # EasySpin: a scalar GridSize N means [N 4] (interpolation factor 4)
        kw['GridSize'] = gs if len(gs) == 2 else [int(gs[0]), 4]
    sym = _get(O, 'GridSymmetry')
    if sym is not None:
        kw['GridSymmetry'] = str(sym)
    return Options(**kw)


def cosine(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# Observed after the 1.1 fix (2026-09-01): the four formerly broken cases are
# 0.99996 / 1.00000. The lower thresholds below are pre-existing accumulation
# path differences (Ci scattered interpolation, permuted D2h frames) that
# PORT_SPEC_2 items 1.2/1.4 address; raise them when those land.
FRAME_CASES = {
    'frame_tilted_axial_g': 0.9999,
    'frame_xaxial_g': 0.9999,
    'frame_yaxial_g': 0.9999,
    'frame_tilted_rhombic_g': 0.9999,
    'frame_permuted_rhombic_g': 0.997,      # 0.99743 — item 1.4 (D2h permuted frame)
    'frame_axial_g_perp_A': 0.998,
    'frame_triplet_axial_D': 0.9999,
    'frame_triplet_axial_D_tilted': 0.9999,
    'frame_triplet_rhombic_D_tilted': 0.9999,
    'axialinvariance_Dinfh': 0.9999,
    'axialinvariance_Ci': 0.999,
}


@pytest.mark.parametrize("name", list(FRAME_CASES))
def test_symmetry_frames_vs_matlab(cases, name):
    c = cases[name]
    sys = sys_from_mat(c.Sys)
    exp = exp_from_mat(c.Exp)
    opt = opt_from_mat(c.Opt)
    B, spc = pepper(sys, exp, opt)
    spc = spc.numpy()
    assert np.all(np.isfinite(spc)) and np.abs(spc).max() > 0
    ref = np.asarray(c.spc, dtype=float)
    cs = cosine(spc, ref)
    assert cs > FRAME_CASES[name], f"{name}: cosine {cs:.5f}"
    # Absolute intensity (item 1.2): no empirical factors, EasySpin's chain
    # (dBdE · TransitionRate · 2π · 4π-weights / ΔB). Ci scattered interpolation
    # is the one path still ~5 % low (item 1.4).
    ratio = np.abs(spc).max() / np.abs(ref).max()
    tol = 0.02
    assert abs(ratio - 1) < tol, f"{name}: amplitude ratio {ratio:.4f}"


# ---------------------------------------------------------------------------
# Absolute intensity from first principles (EasySpin pepper_intensity_isopowder,
# pepper_intensity_isopowder_pt, pepper_doubleintegral)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ['matrix', 'perturb2'])
def test_isotropic_powder_integral_formula(method):
    """∫spec dB = 8π² · (μB g / 2h)² · (h/μB)/g  for an isotropic S=1/2 (EasySpin test)."""
    from torchspin.constants import BMAGN, PLANCK
    sys = SpinSystem(S=[0.5], g=2.0, lwpp=[0.5, 0.0])
    exp = Experiment(mwFreq=9.6, Range=[341, 345], Harmonic=0)
    x, y = pepper(sys, exp, Options(Method=method))
    integral = float(y.sum() * (x[1] - x[0]))
    rate = (BMAGN / PLANCK / 1e9 / 2) ** 2 * 2.0 ** 2
    dBdE = (PLANCK / BMAGN) * 1e9 / 2.0
    expected = rate * dBdE * 8 * np.pi ** 2
    assert abs(integral - expected) < 1e-4 * expected, (integral, expected)


def test_double_integral_independent_of_nuclei():
    """pepper_doubleintegral: the double integral does not change when nuclei are added."""
    exp = Experiment(mwFreq=9.5, Harmonic=1, CenterSweep=[339, 10])
    systems = [
        SpinSystem(S=[0.5], lw=[0.3, 0.0]),
        SpinSystem(S=[0.5], Nucs='14N', A=50, lw=[0.3, 0.0]),
        SpinSystem(S=[0.5], Nucs='14N,1H', A=[50, 20], lw=[0.3, 0.0]),
    ]
    def dint(f):
        _trapezoid = getattr(np, "trapezoid", None) or np.trapz
        return float(_trapezoid(np.cumsum(f)))
    d = [dint(pepper(s, exp)[1].numpy()) for s in systems]
    for v in d[1:]:
        assert abs(v / d[0] - 1) < 1e-6, [x / d[0] for x in d]


# ---------------------------------------------------------------------------
# EasySpin pepper_axialinvariance / pepper_rhombicinvariance (consistency)
# ---------------------------------------------------------------------------
def test_axial_invariance_across_grid_symmetries():
    """pepper_axialinvariance: an axial system gives the same powder spectrum
    whichever grid symmetry is used (max deviation < 0.3 % of the maximum)."""
    sys = SpinSystem(S=[0.5], g=[1.9, 1.9, 2.3], lw=[1.0, 0.0])
    exp = Experiment(mwFreq=9.5, Range=[285, 365], Harmonic=0)
    ref = None
    for sym in ['Dinfh', 'D2h', 'C2h', 'Ci', 'C1']:
        _, y = pepper(sys, exp, Options(GridSize=[19, 5], GridSymmetry=sym))
        y = y.numpy()
        if ref is None:
            ref = y / y.max()
            continue
        dev = np.abs(y / y.max() - ref).max()
        assert dev < 0.003, f"{sym}: max deviation {dev:.4f}"


def test_rhombic_invariance_across_grid_symmetries():
    """pepper_rhombicinvariance: D2h, Ci, C2h and C1 grids agree to 1e-3 (of max)
    on a common intensity scale as in EasySpin (C1: 2.5e-3 observed, full-sphere
    rectified interpolation)."""
    sys = SpinSystem(S=[0.5], g=[1.9, 2.0, 2.3], lw=[1.0, 0.0])
    exp = Experiment(mwFreq=9.5, Range=[285, 365], nPoints=1054, Harmonic=0)
    spectra = {}
    for sym in ['D2h', 'Ci', 'C1', 'C2h']:
        _, y = pepper(sys, exp, Options(GridSize=[20, 4], GridSymmetry=sym))
        spectra[sym] = y.numpy()
    scale = max(np.abs(v).max() for v in spectra.values())
    for sym, tol in [('Ci', 1e-3), ('C2h', 1e-3), ('C1', 2.5e-3)]:
        dev = np.abs(spectra[sym] - spectra['D2h']).max() / scale
        assert dev < tol, f"{sym} vs D2h: {dev:.4f}"


def test_tilted_axial_matches_full_sphere():
    """Regression for the Dinfh frame bug: auto symmetry (Dinfh, rotated frame)
    must agree with a dense Ci grid for tilted-axial and x-axial g."""
    exp = Experiment(mwFreq=9.5, Range=[300, 360], nPoints=1024, Harmonic=1)
    for kw in (dict(g=[2.0, 2.0, 2.2], gFrame=[0.3, 0.7, 0.2]), dict(g=[2.2, 2.0, 2.0])):
        sys = SpinSystem(S=[0.5], lw=[1.0, 0.0], **kw)
        _, ya = pepper(sys, exp, Options(GridSize=[19, 4]))
        _, yc = pepper(sys, exp, Options(GridSize=91, GridSymmetry='Ci'))
        assert torch.isfinite(ya).all() and ya.abs().max() > 0
        assert cosine(ya.numpy(), yc.numpy()) > 0.9995
