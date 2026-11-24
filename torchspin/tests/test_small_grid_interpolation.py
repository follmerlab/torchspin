"""B137: grid interpolation from a coarse grid of two or three knots.

``Options(GridSize=[2, Ninterp])`` on an axial (Dinfh) grid has two orientations.
``pepper._interp_slot`` dropped every transition with fewer than three knots, so
pepper returned an all-zero spectrum without an error or a warning (matrix and
perturbation methods, strains, frequency sweeps); C4h and C6h grids did the same.
On the open-φ grids (C2h, S6, Ci, C1, and C4h/C6h at three knots) the rectified
array has fewer than four rows or columns and the bicubic spline raised scipy's
``mx must be > kx`` (pepper, pepper_autograd and salt).

Both now interpolate: with two or three knots along an axis the spline is the
straight line or the parabola through them, as in EasySpin's ``esspline1d``
(EasySpin's pepper itself rejects ``GridSize(1) < 10``).
"""
import warnings

import numpy as np
import pytest
import torch

from torchspin import Experiment, Options, SpinSystem, pepper, salt
from torchspin.pepper import _interp_grid, _interp_slot
from torchspin.pepper_autograd import pepper_autograd
from torchspin.sphgrid import sphgrid

SYMMETRIES = ['Dinfh', 'D2h', 'D4h', 'D6h', 'D3d', 'Th', 'Oh', 'C2h', 'C4h', 'C6h', 'S6', 'Ci', 'C1']
OPEN_PHI = ['C2h', 'C4h', 'C6h', 'S6', 'Ci', 'C1']

# axial systems (pepper picks the Dinfh grid): SpinSystem kwargs, Experiment kwargs
AXIAL = {
    'g_axial': (dict(S=0.5, g=[2.0, 2.0, 2.3], lwpp=1.0), dict(mwFreq=9.5, Range=[260, 360])),
    'cu': (dict(S=0.5, g=[2.05, 2.05, 2.25], Nucs='63Cu', A=[50, 50, 500], lwpp=1.0),
           dict(mwFreq=9.5, Range=[260, 360])),
    'cu_n': (dict(S=0.5, g=[2.05, 2.05, 2.25], Nucs='63Cu,14N', A=[[50, 50, 500], [40, 40, 40]], lwpp=1.0),
             dict(mwFreq=9.5, Range=[260, 360])),
    'triplet': (dict(S=1, g=2.0, D=[[900, 0]], lwpp=1.5), dict(mwFreq=9.5, Range=[250, 430])),
}


def _spec(sk, ek, grid, nPoints=1024, **ok):
    with warnings.catch_warnings():
        warnings.simplefilter('error')          # no warning either: a plain, valid run
        y = pepper(SpinSystem(**sk), Experiment(nPoints=nPoints, Harmonic=0, **ek),
                   Options(Verbosity=0, GridSize=grid, **ok))[1]
    return np.asarray(y, dtype=float)


@pytest.mark.parametrize('name', sorted(AXIAL))
@pytest.mark.parametrize('n_interp', [2, 4])
def test_two_knot_axial_grid_gives_a_spectrum(name, n_interp):
    """The B137 case: an absorption spectrum with the integral of a converged one."""
    sk, ek = AXIAL[name]
    y = _spec(sk, ek, [2, n_interp])
    ref = _spec(sk, ek, [19, 4])
    assert np.all(np.isfinite(y))
    assert y.max() > 0.0, 'all-zero spectrum (B137)'
    assert y.min() >= -1e-9 * y.max()
    assert y.sum() == pytest.approx(ref.sum(), rel=0.03)


@pytest.mark.parametrize('name', sorted(AXIAL))
def test_two_knot_axial_grid_matches_pepper_autograd(name):
    """pepper_autograd never dropped the two-knot slots; pepper now gives the same spectrum."""
    sk, ek = AXIAL[name]
    y = _spec(sk, ek, [2, 4])
    ya = pepper_autograd(SpinSystem(**sk), Experiment(nPoints=1024, Harmonic=0, **ek),
                         Options(Verbosity=0, GridSize=[2, 4]))[1].detach().numpy()
    assert np.abs(y - ya).max() <= 1e-9 * np.abs(ya).max()


@pytest.mark.parametrize('variant', ['perturb', 'hstrain', 'freq_sweep', 'harmonic1'])
def test_two_knot_axial_grid_other_routes(variant):
    sk = dict(S=0.5, g=[2.05, 2.05, 2.25], Nucs='63Cu', A=[50, 50, 500], lwpp=1.0)
    ek = dict(mwFreq=9.5, Range=[260, 360], Harmonic=0)
    ok = {}
    if variant == 'perturb':
        ok = dict(Method='perturb')
    elif variant == 'hstrain':
        sk = dict(S=0.5, g=[2.05, 2.05, 2.25], Nucs='63Cu', A=[50, 50, 500], HStrain=[50, 50, 200])
    elif variant == 'freq_sweep':
        sk = dict(sk, lwpp=30.0)
        ek = dict(Field=330, mwRange=[9.0, 11.0], Harmonic=0)
    elif variant == 'harmonic1':
        ek = dict(mwFreq=9.5, Range=[260, 360], Harmonic=1)

    def run(grid):
        return np.asarray(pepper(SpinSystem(**sk), Experiment(nPoints=1024, **ek),
                                 Options(Verbosity=0, GridSize=grid, **ok))[1], dtype=float)
    y, ref = run([2, 4]), run([19, 4])
    assert np.all(np.isfinite(y))
    assert np.abs(y).max() > 0.0, 'all-zero spectrum (B137)'
    if variant == 'harmonic1':
        y, ref = np.cumsum(y), np.cumsum(ref)
    assert y.sum() == pytest.approx(ref.sum(), rel=0.03)


@pytest.mark.parametrize('sym', SYMMETRIES)
@pytest.mark.parametrize('grid', [[2, 2], [2, 4], [3, 4], [4, 3]])
def test_small_grids_on_every_symmetry(sym, grid):
    """An axial system on every grid symmetry: no error, no zeros, same integral."""
    sk, ek = AXIAL['g_axial']
    y = _spec(sk, ek, grid, GridSymmetry=sym)
    ref = _spec(sk, ek, [19, 4])
    assert np.all(np.isfinite(y)) and y.max() > 0.0
    assert y.sum() == pytest.approx(ref.sum(), rel=0.05)


@pytest.mark.parametrize('sym', ['Dinfh', 'D2h'] + OPEN_PHI)
@pytest.mark.parametrize('grid', [[2, 4], [3, 4]])
def test_small_grids_pepper_autograd_equals_pepper(sym, grid):
    sk = dict(S=0.5, g=[2.0, 2.1, 2.3], lwpp=1.0)
    exp = Experiment(mwFreq=9.5, Range=[260, 360], nPoints=1024, Harmonic=0)
    opt = Options(Verbosity=0, GridSize=grid, GridSymmetry=sym)
    y = np.asarray(pepper(SpinSystem(**sk), exp, opt)[1], dtype=float)
    g = torch.tensor([2.0, 2.1, 2.3], dtype=torch.float64, requires_grad=True)
    ya = pepper_autograd(SpinSystem(S=0.5, g=g, lwpp=1.0), exp, opt)[1]
    assert np.abs(y - ya.detach().numpy()).max() <= 1e-8 * np.abs(y).max()
    (grad,) = torch.autograd.grad(ya.sum(), g)
    assert torch.all(torch.isfinite(grad))


@pytest.mark.parametrize('sym', [None, 'C2h', 'Ci', 'C1'])
@pytest.mark.parametrize('grid', [[2, 4], [3, 4]])
def test_small_grids_salt(sym, grid):
    """salt shares the interpolation: the ENDOR integral does not depend on the grid."""
    sys = SpinSystem(S=[0.5], g=[[2.0, 2.01, 2.02]], Nucs='1H', A=[[-2.0, -2.0, 4.0]])
    exp = Experiment(Field=326.5, mwFreq=9.7)
    kw = {} if sym is None else dict(GridSymmetry=sym)

    def run(g):
        return np.asarray(salt(sys, exp, Options(Verbosity=0, GridSize=g, **kw),
                               freq_range=(10, 20), n_points=512, lw_mhz=0.1)[1].detach(), dtype=float)
    y, ref = run(grid), run([20, 5])
    assert np.all(np.isfinite(y)) and y.max() > 0.0
    assert y.sum() == pytest.approx(ref.sum(), rel=1e-3)


# ---------------------------------------------------------------------------
# the interpolation itself
# ---------------------------------------------------------------------------

def _vecs(sym, n):
    v = sphgrid(sym, n)[3]
    return v if v.shape[0] == 3 else v.T


def test_two_knot_axial_positions_are_the_zero_slope_hermite_cubic():
    """EasySpin esspline1d(y, 1, x) with two knots (esspline1d.m:55-60)."""
    out = _interp_slot(np.array([300.0, 340.0]), 'Dinfh', 2, 5, _vecs('Dinfh', 2), _vecs('Dinfh', 5))
    t = np.linspace(0.0, 1.0, 5)
    np.testing.assert_allclose(out, 300.0 + 40.0 * (3 * t ** 2 - 2 * t ** 3), rtol=0, atol=1e-11)
    val = _interp_slot(np.array([1.0, 3.0]), 'Dinfh', 2, 5, _vecs('Dinfh', 2), _vecs('Dinfh', 5), mode='val')
    np.testing.assert_allclose(val, 1.0 + 2.0 * (3 * t ** 2 - 2 * t ** 3), rtol=0, atol=1e-12)


@pytest.mark.parametrize('sym', OPEN_PHI)
@pytest.mark.parametrize('n_c', [2, 3])
def test_small_open_phi_grids_interpolate_through_the_knots(sym, n_c):
    """Global cubic (G3) on 2 or 3 knots: exact at the coarse knots, exact for a
    constant, and bounded by the data on two-knot grids (a straight line in θ)."""
    n_f = (n_c - 1) * 4 + 1
    phi_c, th_c, _, _ = sphgrid(sym, n_c)
    phi_f, th_f, _, _ = sphgrid(sym, n_f)
    vals = (300.0 + 30.0 * torch.cos(th_c) ** 2).numpy()          # axial: no φ dependence
    out = _interp_grid(vals, sym, n_c, n_f)
    assert out.shape == (phi_f.numel(),) and np.all(np.isfinite(out))
    # the coarse knots are a subset of the fine ones
    d = (th_f[:, None] - th_c[None, :]).abs() + (phi_f[:, None] - phi_c[None, :]).abs()
    j = d.argmin(dim=0).numpy()
    assert float(d.min(dim=0).values.max()) < 1e-12
    np.testing.assert_allclose(out[j], vals, rtol=0, atol=1e-9)
    assert out.min() >= vals.min() - (1e-9 if n_c == 2 else 5.0)
    assert out.max() <= vals.max() + (1e-9 if n_c == 2 else 5.0)
    np.testing.assert_allclose(_interp_grid(np.full(vals.shape, 7.0), sym, n_c, n_f), 7.0, rtol=0, atol=1e-12)


def test_slots_with_too_few_knots_are_still_dropped():
    """Unchanged: a transition found at fewer than three knots of a larger grid is
    dropped; on a two-knot grid one missing knot leaves nothing to interpolate."""
    vc, vf = _vecs('Dinfh', 5), _vecs('Dinfh', 17)
    assert _interp_slot(np.array([300.0, 310.0, np.nan, np.nan, np.nan]), 'Dinfh', 5, 17, vc, vf) is None
    assert _interp_slot(np.array([300.0, np.nan]), 'Dinfh', 2, 5, _vecs('Dinfh', 2), _vecs('Dinfh', 5)) is None


@pytest.mark.parametrize('bad', [[1, 4], 1, [0, 4], [2, 0]])
def test_grid_size_below_two_knots_still_raises(bad):
    with pytest.raises(ValueError, match='at least'):
        Options(GridSize=bad)
