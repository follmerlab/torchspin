"""φ range of the SOPHE grids (B36) and the accepted ``Options.GridSymmetry``
values (B51).

Up to 0.9.2 ``sphgrid`` covered φ = 0..π/2 for D6h, D4h, Oh and D3d, 0..π/2
for C6h and 0..π for S6, while EasySpin (private/gridparam.m) and torchspin's
own ``gridparam`` use π/6, π/4, π/4, π/3, π/3 and 2π/3. The D3d, C6h and S6
grids then covered a non-integer number of symmetry wedges, so the sphere
average of a function with that symmetry was wrong (D3d: 6-9 %).
"""
import math

import numpy as np
import pytest
import torch

from torchspin.experiment import Options
from torchspin.sphgrid import _GRIDPARAM, gridparam, grid_triangulation, sphgrid

ALL_GROUPS = ('O3', 'Dinfh', 'D6h', 'D4h', 'Oh', 'D3d', 'Th', 'D2h',
              'C4h', 'C6h', 'C2h', 'S6', 'Ci', 'C1')

# EasySpin private/gridparam.m:31-36, written out
EASYSPIN_MAXPHI = {
    'O3': 0.0, 'Dinfh': 0.0,
    'D6h': math.pi / 6, 'D4h': math.pi / 4, 'Oh': math.pi / 4, 'D3d': math.pi / 3,
    'Th': math.pi / 2, 'D2h': math.pi / 2, 'C4h': math.pi / 2, 'C6h': math.pi / 3,
    'C2h': math.pi, 'S6': 2 * math.pi / 3, 'Ci': 2 * math.pi, 'C1': 2 * math.pi,
}


def test_gridparam_has_easyspin_groups_and_maxphi():
    assert set(_GRIDPARAM) == set(ALL_GROUPS)
    for sym in ALL_GROUPS:
        assert gridparam(sym)[0] == pytest.approx(EASYSPIN_MAXPHI[sym], abs=1e-15), sym


@pytest.mark.parametrize('sym', ALL_GROUPS)
@pytest.mark.parametrize('N', [2, 5, 19])
def test_sphgrid_phi_range_matches_gridparam(sym, N):
    """The largest φ of the grid is gridparam's maxPhi (closed φ) or one equator
    step below it (open φ: the φ = maxPhi border is dropped)."""
    maxPhi, closedPhi, nOct = gridparam(sym)
    phi, theta, w, _ = sphgrid(sym, N)
    assert float(w.sum()) == pytest.approx(4 * math.pi, rel=1e-12)
    if nOct <= 0:
        assert float(phi.abs().max()) == 0.0
        return
    n_eq = (4 if nOct == 8 else nOct) * (N - 1) + 1          # equator knots incl. φ = maxPhi
    expected = maxPhi if closedPhi else maxPhi * (n_eq - 2) / (n_eq - 1)
    assert float(phi.max()) == pytest.approx(expected, abs=1e-14), (sym, float(phi.max()), expected)
    # the equator spans the wedge with the knot spacing of the grid
    eq = np.sort(phi[torch.isclose(theta, torch.tensor(math.pi / 2, dtype=theta.dtype))].numpy())
    assert np.allclose(np.diff(eq), maxPhi / (n_eq - 1), rtol=0, atol=1e-14)


# Functions with the symmetry of each Laue class in the frame of its grid
# (mirror planes at φ = 0 and φ = maxPhi for the closed-φ groups). Each has
# only φ harmonics m that the trapezoid rule of both the symmetric grid and
# the C1 grid integrate exactly, so the two sphere averages agree to rounding.
_INVARIANT = {
    'D3d': lambda x, y, z: 1 + z ** 2 + 0.8 * z * (x ** 3 - 3 * x * y ** 2),
    'S6':  lambda x, y, z: 1 + z ** 2 + 0.8 * z * (x ** 3 - 3 * x * y ** 2) + 0.5 * z * (3 * x ** 2 * y - y ** 3),
    'C6h': lambda x, y, z: 1 + z ** 2 + 0.8 * ((x + 1j * y) ** 6).real + 0.5 * ((x + 1j * y) ** 6).imag,
    'D6h': lambda x, y, z: 1 + z ** 2 + 0.8 * ((x + 1j * y) ** 6).real,
    'D4h': lambda x, y, z: 1 + z ** 2 + 0.8 * (x ** 4 + y ** 4),
    'Oh':  lambda x, y, z: 1 + x ** 4 + y ** 4 + z ** 4,
}


def _sphere_average(sym, f, N):
    _, _, w, v = sphgrid(sym, N)
    w = w.numpy(); v = v.numpy()
    return float((w * f(*v)).sum() / w.sum())


@pytest.mark.parametrize('sym', ['D3d', 'S6', 'C6h', 'D6h', 'D4h', 'Oh'])
def test_sphere_average_of_invariant_function_equals_c1(sym):
    """Before the fix the D3d / S6 / C6h grids were off by -2.5e-2 / +1.7e-2 /
    +4.0e-2 relative for these functions (D6h, D4h, Oh: the old φ = 0..π/2
    wedge holds whole symmetry wedges, so they were right)."""
    # N = 91: the m = 6 harmonic aliases on the first θ ring of the C6h grid (6
    # knots per turn), weight ~ sin^6(dθ) dθ^2 < 1e-14. D4h/Oh: the m = 4
    # harmonic aliases on the first ring of the C1 grid (4 knots), ~ sin^4(dθ) dθ^2
    # = 7e-12; the old grids were wrong by 3e-2 or more, so 1e-10 still separates.
    N = 91
    f = _INVARIANT[sym]
    ref = _sphere_average('C1', f, N)
    est = _sphere_average(sym, f, N)
    rel = 1e-10 if sym in ('D4h', 'Oh') else 1e-12
    assert est == pytest.approx(ref, rel=rel, abs=0), (sym, est, ref)


@pytest.mark.parametrize('sym', ['D6h', 'D4h', 'Oh', 'D3d', 'C6h', 'S6', 'D2h', 'C2h', 'Ci', 'C1'])
def test_triangulation_covers_the_grid_wedge(sym):
    """grid_triangulation's areas belong to the sphgrid knots: they sum to the
    solid angle of the wedge, maxPhi per hemisphere (2·maxPhi for C1)."""
    maxPhi, closedPhi, nOct = gridparam(sym)
    N = 9
    tri, areas = grid_triangulation(sym, N)
    nknots = sphgrid(sym, N)[0].numel()
    assert tri.min() >= 0 and tri.max() < nknots
    expected = 2 * maxPhi if nOct == 8 else maxPhi
    assert float(np.sum(areas)) == pytest.approx(expected, rel=1e-12), (sym, float(np.sum(areas)), expected)


@pytest.mark.parametrize('sym', ALL_GROUPS + ('auto', ''))
def test_options_accepts_all_easyspin_grid_symmetries(sym):
    """B51: Options accepted only Dinfh, D2h, C2h, Ci, C1, O3 and 'auto'.
    '' means 'auto' (EasySpin's empty Opt.GridSymmetry; pepper treats both alike)."""
    assert Options(GridSymmetry=sym).GridSymmetry == sym


@pytest.mark.parametrize('sym', ['D3', 'C3v', 'Auto', 'd4h', None])
def test_options_rejects_unknown_grid_symmetry(sym):
    with pytest.raises(ValueError, match='GridSymmetry'):
        Options(GridSymmetry=sym)


def test_pepper_auto_and_empty_grid_symmetry_agree():
    from torchspin import Experiment, SpinSystem, pepper
    sys = SpinSystem(S=[2.5], g=2.0, D=[[300.0, 0.0]], B=[None, [[30.0, 0, 0, 0, 10, 0, 0, 0, 0]]], lw=[1.0, 0.0])
    exp = Experiment(mwFreq=9.5, Range=[290, 390], Harmonic=1, nPoints=512)
    a = pepper(sys, exp, Options(GridSize=[10, 4], GridSymmetry='auto', Verbosity=0))[1]
    b = pepper(sys, exp, Options(GridSize=[10, 4], GridSymmetry='', Verbosity=0))[1]
    c = pepper(sys, exp, Options(GridSize=[10, 4], GridSymmetry='D4h', Verbosity=0))[1]
    assert torch.equal(a, b) and torch.equal(a, c)
