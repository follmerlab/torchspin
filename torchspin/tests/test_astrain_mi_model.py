"""AStrain effective nuclear mI (B68, 0.9.25).

EasySpin (``resfields.m`` 766-788) gives every level one effective mI of the
first nucleus, computed once for all orientations: the eigenvectors of
``H0 - mean(Exp.Range)*muz`` (field along the *molecular* z axis), levels
sorted by energy, ``mI = <Iz>``; a transition uses the mean over its two levels
at every orientation (``resfields.m`` 1297-1306).  Before 0.9.25 torchspin took
``<Iz>`` at each orientation's own resonance field, which vanishes for the
field perpendicular to a large A: perpendicular lines got no AStrain width
(14N, A = [20 20 90] MHz, AStrain 10 MHz, theta = 90 deg: 0.000 mT against
EasySpin's 0.357 mT; an explicit ensemble with A sampled gives 0.32-0.38 mT).

References: tests/data/ref_astrain_mi.mat (generate_astrain_mi_refs.m), per
case EasySpin pepper spectra (Harmonic 0 and 1) and resfields positions,
widths, intensities and level pairs at phi = 0.3, theta = 0:10:90 deg.

Also checked: a facet with zero mean strain width gives finite spectra and
gradients in pepper_autograd (it gave NaN gradients), and gradients in
AStrain, A and gStrain agree with converged central differences.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io import loadmat

from torchspin.experiment import Experiment, Options
from torchspin.ham import ham
from torchspin.pepper import pepper
from torchspin.pepper_autograd import pepper_autograd
from torchspin.resfields_batch import resfields_batch
from torchspin.spinsystem import SpinSystem
from torchspin.strainwidth import (astrain_level_mI, compute_strain_widths,
                                   compute_strain_widths_batch)

F64 = torch.float64
REF_FILE = Path(__file__).parent.parent.parent / "tests" / "data" / "ref_astrain_mi.mat"
CASES = ['N14', 'H1', 'Cu63', 'N14_gS', 'N14_gSm', 'Cu63_AFrame']
# EasySpin interpolates the eigenvectors between field knots (resfields.m ~1147);
# torchspin diagonalizes at the resonance.  Observed: <= 1.1e-5 relative.
WIDTH_TOL = 1e-4


@pytest.fixture(scope="module")
def refs():
    if not REF_FILE.exists():
        pytest.skip(f"Reference data not found: {REF_FILE}")
    R = loadmat(str(REF_FILE), squeeze_me=True, struct_as_record=False)
    return {str(c.name): c for c in np.atleast_1d(R['cases'])}


def _get(o, name):
    return getattr(o, name) if name in o._fieldnames else None


def sys_from_mat(S, **override) -> SpinSystem:
    kw = dict(S=0.5, g=np.atleast_1d(S.g).astype(float).tolist(), Nucs=str(S.Nucs),
              A=np.atleast_1d(S.A).astype(float).tolist(), lw=float(S.lw),
              AStrain=np.broadcast_to(np.atleast_1d(S.AStrain).astype(float), (3,)).tolist())
    for f in ('gStrain', 'AFrame'):
        v = _get(S, f)
        if v is not None:
            kw[f] = np.atleast_1d(v).astype(float).tolist()
    if _get(S, 'gAStrainCorr') is not None:
        kw['gAStrainCorr'] = float(S.gAStrainCorr)
    kw.update(override)
    return SpinSystem(**kw)


def _easyspin_lines(c):
    """(orientation k, level pair (u, v) 0-based, position, width) of every allowed EasySpin line."""
    Pos = np.atleast_2d(c.Pos); Wid = np.atleast_2d(c.Wid)
    Int = np.atleast_2d(c.Int); Tr = np.atleast_2d(c.Tr).astype(int) - 1
    for k in range(Pos.shape[1]):
        for t in range(Pos.shape[0]):
            if np.isfinite(Pos[t, k]) and Int[t, k] >= 1e-3 * np.nanmax(Int[:, k]):
                yield k, (int(Tr[t, 0]), int(Tr[t, 1])), float(Pos[t, k]), float(Wid[t, k])


def _cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------------------------------------------------------------------------
# Per-orientation widths (resfields_batch, as pepper calls it)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", CASES)
def test_widths_vs_easyspin(refs, name):
    c = refs[name]
    sys = sys_from_mat(c.Sys)
    H0, mux, muy, muz = ham(sys)
    ori = np.asarray(c.ori, dtype=float)
    B, _, W, P = resfields_batch(H0, mux, muy, muz, torch.tensor(ori[:, 0]), torch.tensor(ori[:, 1]),
                                 Experiment(mwFreq=9.5, Range=np.asarray(c.Range, float).tolist()),
                                 Options(), sys=sys, return_pairs=True)
    n = 0
    for k, (u, v), pos, wid in _easyspin_lines(c):
        sel = torch.nonzero((P[k][:, 0] == u) & (P[k][:, 1] == v)).reshape(-1)
        assert sel.numel() > 0, f"{name}: line {u+1}-{v+1} at {pos:.3f} mT missing (theta {ori[k, 1]:.3f})"
        j = sel[torch.argmin((B[k][sel] - pos).abs())]
        assert abs(float(B[k][j]) - pos) < 0.05
        assert abs(float(W[k][j]) - wid) <= WIDTH_TOL * max(0.1, abs(wid)), (
            f"{name} theta={np.degrees(ori[k, 1]):.0f} pair {u+1}-{v+1}: torchspin "
            f"{float(W[k][j]):.6f} vs EasySpin {wid:.6f} mT")
        n += 1
    assert n >= 30


def test_perpendicular_lines_have_astrain_width(refs):
    """14N, theta = 90 deg: the outer lines have AStrain width (B68: it was 0)."""
    c = refs['N14']
    lines = [(pos, wid) for k, _, pos, wid in _easyspin_lines(c) if k == 9]
    assert sorted(round(w, 3) for _, w in lines) == [0.0, 0.357, 0.357]
    sys = sys_from_mat(c.Sys)
    H0, mux, muy, muz = ham(sys)
    B, _, W = resfields_batch(H0, mux, muy, muz, torch.tensor([0.3], dtype=F64),
                              torch.tensor([np.pi / 2], dtype=F64),
                              Experiment(mwFreq=9.5, Range=[315.0, 345.0]), Options(), sys=sys)
    w = W[0][torch.argsort(B[0])]
    torch.testing.assert_close(w, torch.tensor([0.357, 0.0, 0.357], dtype=F64), rtol=0, atol=1e-3)


# ---------------------------------------------------------------------------
# Single-orientation compute_strain_widths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ['N14', 'Cu63', 'N14_gSm', 'Cu63_AFrame'])
def test_single_orientation_widths_vs_easyspin(refs, name):
    c = refs[name]
    sys = sys_from_mat(c.Sys)
    H0, mux, muy, muz = ham(sys)
    ori = np.asarray(c.ori, dtype=float)
    rng = np.asarray(c.Range, float).tolist()
    n = 0
    for k, (u, v), pos, wid in _easyspin_lines(c):
        w = compute_strain_widths(sys, H0, mux, muy, muz, float(ori[k, 0]), float(ori[k, 1]),
                                  torch.tensor([[u, v]]), 9.5, pos, Range=rng)
        assert abs(float(w[0]) - wid) <= WIDTH_TOL * max(0.1, abs(wid)), (
            f"{name} theta={np.degrees(ori[k, 1]):.0f} pair {u+1}-{v+1}: {float(w[0]):.6f} vs {wid:.6f}")
        n += 1
    assert n >= 30


def test_single_orientation_default_labels_levels_at_B0():
    """Without Range (or mI_levels) the levels are labeled at the given resonance field B0."""
    sys = SpinSystem(S=0.5, g=[2, 2, 2.1], Nucs='14N', A=[20, 20, 90], AStrain=[10, 10, 10])
    H0, mux, muy, muz = ham(sys)
    tr = torch.tensor([[0, 5], [1, 4], [2, 3]])
    w = compute_strain_widths(sys, H0, mux, muy, muz, 0.3, 1.2, tr, 9.5, 332.0)
    w_rng = compute_strain_widths(sys, H0, mux, muy, muz, 0.3, 1.2, tr, 9.5, 332.0, Range=[330.0, 334.0])
    mI = astrain_level_mI(sys, H0, muz, 332.0)
    w_mI = compute_strain_widths(sys, H0, mux, muy, muz, 0.3, 1.2, tr, 9.5, 332.0, mI_levels=mI)
    torch.testing.assert_close(w, w_rng, rtol=1e-14, atol=0.0)
    torch.testing.assert_close(w, w_mI, rtol=1e-14, atol=0.0)


def test_batch_needs_mI_levels_for_astrain():
    sys = SpinSystem(S=0.5, g=[2, 2, 2.1], Nucs='14N', A=[20, 20, 90], AStrain=[10, 10, 10])
    H0, mux, muy, muz = ham(sys)
    one = torch.ones(1, dtype=F64)
    with pytest.raises(ValueError, match="mI_levels"):
        compute_strain_widths_batch(sys, H0, muz.unsqueeze(0), 0 * one, 0 * one, 330 * one,
                                    [torch.tensor([[0, 5]])], 9.5)
    # no AStrain: no mI needed, astrain_level_mI returns None
    s2 = SpinSystem(S=0.5, g=[2, 2, 2.1], Nucs='14N', A=[20, 20, 90], HStrain=[10, 10, 10])
    H0, mux, muy, muz = ham(s2)
    assert astrain_level_mI(s2, H0, muz, 330.0) is None
    w = compute_strain_widths_batch(s2, H0, muz.unsqueeze(0), 0 * one, 0 * one, 330 * one,
                                    [torch.tensor([[0, 5]])], 9.5)
    assert w[0].shape == (1,)


# ---------------------------------------------------------------------------
# Spectra: pepper (NumPy path) and pepper_autograd, Harmonic 0 and 1
# ---------------------------------------------------------------------------

def _spectrum(fn, c, h, **override):
    exp = Experiment(mwFreq=9.5, Range=np.asarray(c.Range, float).tolist(), nPoints=int(c.nPoints), Harmonic=h)
    x, y = fn(sys_from_mat(c.Sys, **override), exp)
    return (np.asarray(x.detach() if torch.is_tensor(x) else x, float),
            np.asarray(y.detach() if torch.is_tensor(y) else y, float))


_CU63_DEVIATION = pytest.mark.xfail(strict=True, reason=(
    "Deliberate deviation since 0.9.28 (B88): the 63Cu range [290 350] mT cuts transitions; "
    "torchspin searches them again and interpolates per slot, EasySpin interpolates every "
    "transition linearly. Cosine vs EasySpin h0 0.997448, h1 0.946325; vs the converged "
    "reference (test_cu63_vs_converged_reference) torchspin is closer than EasySpin."))


def _spectra_vs_easyspin_params():
    out = []
    for name in ['N14', 'H1', 'Cu63', 'N14_gS', 'N14_gSm']:
        marks = [_CU63_DEVIATION] if name == 'Cu63' else []
        out.append(pytest.param(name, marks=marks, id=name))
    return out


@pytest.mark.parametrize("fn", [pepper, pepper_autograd], ids=['pepper', 'pepper_autograd'])
@pytest.mark.parametrize("h", [0, 1])
@pytest.mark.parametrize("name", _spectra_vs_easyspin_params())
def test_spectra_vs_easyspin(refs, name, h, fn):
    """Observed (both paths): cosine >= 0.999996, amplitude within 0.12 %."""
    c = refs[name]
    x, y = _spectrum(fn, c, h)
    ref = np.asarray(getattr(c, f'spc{h}'), float)
    np.testing.assert_allclose(x, np.asarray(c.B, float), rtol=0, atol=1e-9)
    assert np.all(np.isfinite(y))
    cs = _cosine(y, ref)
    amp = np.abs(y).max() / np.abs(ref).max()
    assert cs > 0.99999, f"{name} h{h}: cosine {cs:.6f}"
    assert abs(amp - 1) < 0.002, f"{name} h{h}: amplitude {amp:.5f}"


# 1 - cosine of the default-grid spectrum vs the converged reference (measured,
# both paths): h0 4.971e-4, h1 1.348e-2; EasySpin's: 1.867e-3, 3.748e-2.
_CU63_CONV = {0: (5.2e-4, 1.867e-3), 1: (1.42e-2, 3.748e-2)}


@pytest.mark.parametrize("fn", [pepper, pepper_autograd], ids=['pepper', 'pepper_autograd'])
@pytest.mark.parametrize("h", [0, 1])
def test_cu63_vs_converged_reference(refs, h, fn):
    """The 63Cu case cuts transitions at 290 mT (B88). Reference: the same system
    without interpolation on a 1-D Dinfh grid [721 1] (axial, untilted strain),
    swept 80 mT wider on both sides and cropped (agrees with Ci [121 1] to
    1 - cos 7e-7 / 1.7e-5). torchspin is closer to it than EasySpin."""
    c = refs['Cu63']
    _, y = _spectrum(fn, c, h)
    r0, r1 = (float(v) for v in c.Range)
    n = int(c.nPoints)
    dx = (r1 - r0) / (n - 1)
    k = int(round(80 / dx))
    wide = Experiment(mwFreq=9.5, Range=[r0 - k * dx, r1 + k * dx], nPoints=n + 2 * k, Harmonic=h)
    _, ref = pepper(sys_from_mat(c.Sys), wide, Options(GridSymmetry='Dinfh', GridSize=[721, 1]))
    ref = np.asarray(ref, float)[k:k + n]
    bound, easyspin = _CU63_CONV[h]
    d = 1 - _cosine(y, ref)
    assert d < bound, f"Cu63 h{h}: 1 - cos {d:.3e}"
    assert 1 - _cosine(np.asarray(getattr(c, f'spc{h}'), float), ref) == pytest.approx(easyspin, rel=1e-2)


@pytest.mark.xfail(strict=True, reason=(
    "Separate defect, not the AStrain mI model: 63Cu with a tilted AFrame [0 0.4 0] differs "
    "from EasySpin even without any strain (pepper h1 cosine 0.873, pepper_autograd 0.962, "
    "and the two torchspin paths disagree); the per-orientation widths agree (above)."))
@pytest.mark.parametrize("fn", [pepper, pepper_autograd], ids=['pepper', 'pepper_autograd'])
def test_spectra_vs_easyspin_tilted_aframe(refs, fn):
    c = refs['Cu63_AFrame']
    _, y = _spectrum(fn, c, 1)
    assert _cosine(y, np.asarray(c.spc1, float)) > 0.999


# ---------------------------------------------------------------------------
# Zero-width facets (pepper_autograd)
# ---------------------------------------------------------------------------

def _fd4(f, x0, h):
    return (-f(x0 + 2 * h) + 8 * f(x0 + h) - 8 * f(x0 - h) + f(x0 - 2 * h)) / (12 * h)


def test_zero_width_facets_finite_value_and_gradient():
    """14N with A = 0: the mI = 0 lines have exactly zero AStrain width on every facet.

    pepper_autograd gave a NaN spectrum or NaN gradients there (gam = 1/0 and
    0 * inf for Lambda = 0, and d sqrt(lw2)/d lw2 at lw2 = 0); pepper sets gam = 0.
    Explicit D2h: AStrain [10 10 10] with axial g is Dinfh (strains are symmetry
    inputs since 0.9.24), the perturbed AStrain [10+h 10 10] D2h, and on Dinfh
    d/dA_x is counted on the x-z meridian only (-0.016368 vs -0.008442 on D2h,
    where AD and the central differences agree to 2e-10).
    """
    exp = Experiment(mwFreq=9.5, Range=[315.0, 345.0], nPoints=512)
    wts = torch.linspace(-1, 1, 512, dtype=F64) ** 3 + 0.3

    def loss(a_strain, g):
        s = SpinSystem(S=0.5, g=g, Nucs='14N', A=[0.0, 0.0, 0.0], lw=0.3, AStrain=a_strain)
        return pepper_autograd(s, exp, Options(GridSymmetry='D2h'))[1]

    a = torch.tensor([10.0, 10.0, 10.0], dtype=F64, requires_grad=True)
    g = torch.tensor([2.0, 2.0, 2.1], dtype=F64, requires_grad=True)
    y = loss(a, g)
    assert torch.all(torch.isfinite(y))
    _, y_np = pepper(SpinSystem(S=0.5, g=[2.0, 2.0, 2.1], Nucs='14N', A=[0.0, 0.0, 0.0], lw=0.3,
                                AStrain=[10.0, 10.0, 10.0]), exp, Options(GridSymmetry='D2h'))
    np.testing.assert_allclose(y.detach().numpy(), np.asarray(y_np, float), rtol=1e-9,
                               atol=1e-9 * float(np.abs(np.asarray(y_np)).max()))
    (y * wts).sum().backward()
    assert torch.all(torch.isfinite(a.grad)) and torch.all(torch.isfinite(g.grad))
    with torch.no_grad():
        fd = _fd4(lambda t: float((loss(torch.stack([t, a[1], a[2]]), g) * wts).sum()),
                  torch.tensor(10.0, dtype=F64), 1e-2)
    assert abs(float(a.grad[0]) - fd) <= 1e-6 * abs(fd)


def test_zero_hstrain_component_gradient_finite():
    """HStrain = [h h 0]: zero width along z (lw2 = 0); d spc/d h was NaN."""
    exp = Experiment(mwFreq=9.5, Range=[315.0, 345.0], nPoints=512)

    def loss(hx):
        hs = torch.stack([hx, hx, torch.zeros((), dtype=F64)])
        return (pepper_autograd(SpinSystem(S=0.5, g=[2, 2, 2.1], lw=0.3, HStrain=hs), exp)[1] ** 2).sum()

    x = torch.tensor(5.0, dtype=F64, requires_grad=True)
    loss(x).backward()
    with torch.no_grad():
        fd = _fd4(loss, torch.tensor(5.0, dtype=F64), 1e-2)
    assert torch.isfinite(x.grad)
    assert abs(float(x.grad) - fd) <= 1e-6 * abs(fd)


# ---------------------------------------------------------------------------
# Gradients vs converged central differences
# ---------------------------------------------------------------------------

P0 = {'AStrain': [10.0, 8.0, 12.0], 'A': [20.0, 25.0, 90.0], 'gStrain': [0.004, 0.005, 0.01]}
STEP = {'AStrain': 1e-2, 'A': 1e-2, 'gStrain': 1e-5}


def _sys(p):
    return SpinSystem(S=0.5, g=[2.0, 2.0, 2.1], Nucs='14N', A=p['A'], lw=0.3,
                      AStrain=p['AStrain'], gStrain=p['gStrain'])


def _check_grad(f, key, i):
    P = {k: torch.tensor(v, dtype=F64, requires_grad=(k == key)) for k, v in P0.items()}
    f(P).backward()
    ad = float(P[key].grad[i])
    h = STEP[key]

    def g(t):
        q = {k: torch.tensor(v, dtype=F64) for k, v in P0.items()}
        q[key][i] = t
        with torch.no_grad():
            return float(f(q))
    x0 = torch.tensor(P0[key][i], dtype=F64)
    fd1, fd2 = _fd4(g, x0, h), _fd4(g, x0, h / 2)
    # round-off floor of the h/2 difference: 10 eps |f| / (h/2)
    floor = 10 * torch.finfo(F64).eps * abs(g(x0)) / (h / 2)
    assert abs(fd1 - fd2) <= 1e-7 * abs(fd2) + floor, f"{key}[{i}]: FD not converged ({fd1} vs {fd2})"
    assert abs(ad - fd2) <= 1e-6 * abs(fd2) + floor, f"{key}[{i}]: autograd {ad:+.10e} vs FD {fd2:+.10e}"


@pytest.mark.parametrize("key,i", [(k, i) for k in P0 for i in range(3)])
def test_width_gradients_match_finite_differences(key, i):
    """Widths from resfields_batch (mI labeling included) at three orientations."""
    ph = torch.tensor([0.3, 0.3, 1.0], dtype=F64)
    th = torch.tensor([0.4, np.pi / 2, 1.1], dtype=F64)

    def f(P):
        s = _sys(P)
        H0, mux, muy, muz = ham(s)
        B, _, W = resfields_batch(H0, mux, muy, muz, ph, th, Experiment(mwFreq=9.5, Range=[315.0, 345.0]),
                                  Options(), sys=s)
        w = torch.cat([W[k][torch.argsort(B[k].detach())] for k in range(3)])
        return (w * torch.cos(1.3 * torch.arange(w.numel(), dtype=F64))).sum()
    _check_grad(f, key, i)


@pytest.mark.parametrize("key,i", [('AStrain', 0), ('AStrain', 2), ('A', 0), ('A', 2), ('gStrain', 2)])
def test_spectrum_gradients_match_finite_differences(key, i):
    exp = Experiment(mwFreq=9.5, Range=[315.0, 345.0], nPoints=512)
    wts = torch.linspace(-1, 1, 512, dtype=F64) ** 3 + 0.3

    def f(P):
        return (pepper_autograd(_sys(P), exp, Options(GridSize=[19, 4]))[1] * wts).sum()
    _check_grad(f, key, i)
