"""Grid interpolation with knots without a resonance against EasySpin (B48, B50;
tests/data/ref_interp_nan_edges.mat, generate_interp_nan_edges_refs.m).

When some coarse knots have no resonance (NaN), EasySpin interpolates linearly
(gridinterp 'L1', private/gridinterp.m:129, ``interp2(z,iphi,ithe,'linear')``).
Up to 0.9.9 torchspin's bilinear interpolation (pepper._interp_grid) computed
``0 * NaN = NaN`` for corners with zero weight, so a missing coarse knot also
removed the fine knots on the neighboring coarse row or column; MATLAB's
interp2 ignores zero-weight corners (``interp2([1 NaN; 2 3], 1, 1.5) = 1.5``).
In addition ``sphgrid`` built phi as ``k * dPhi``, while EasySpin uses
``linspace(0, maxPhi, nPhi)`` (sphgrid.m:298,307); the ulp-level differences
decided whether a fine knot lay exactly on a coarse column. Before the fix:

* B48 (S = 1/2, g = [2 2.05 2.1], [325 337] mT cuts the g = 2.1 edge,
  GridSize [31 4]): stick spectrum max deviation 5.6 % of the maximum, with
  lw 1.5 mT 2.4 % (Harmonic 0) and 7.1 % (Harmonic 1);
* B50 (S = 5/2, D = 300 MHz, tetragonal B4, D2h grid [19 4], [290 390] mT):
  0.54 % (Harmonic 1), 0.96 % (stick).

Also: EasySpin's ``sphgrid`` phi/theta bitwise for 12 groups (GridSize 7 and
73), gridinterp 'L1' on synthetic data with missing knots, and pepper_autograd
gradients vs central differences on B48 and B50.

Since 0.9.28 (B88, B75) the knots of these spectra are no longer missing: the
level pairs that leave Exp.Range are searched again over a padded window and
projection clips their out-of-range vertices, while EasySpin keeps the NaNs
(and drops the triangles that cross the range ends, also without
interpolation).  The EasySpin comparisons of the seven spectra are therefore
strict xfails with the measured numbers, and the spectra are checked against
converged references instead (no interpolation, GridSize [181 1], wide sweep
cropped; tests/data/ref_missing_knots.npz).  Every case is closer to them
than the EasySpin-identical 0.9.27 spectra (1 - cosine / max|dev|, interior):

* b48_stick 1.59e-2 / 0.270 -> 7.3e-8 / 7.9e-4; b48_stick_noint (B75)
  8.71e-3 / 0.270 -> 1.88e-5 / 0.0223; b48_h0 3.0e-7 / 1.96e-3 ->
  2.4e-10 / 6.1e-5; b48_h1 2.12e-5 / 0.0222 -> 2.8e-8 / 9.2e-4;
* b50_h1 3.91e-2 / 0.238 -> 7.09e-5 / 0.0144; b50_stick 1.94e-2 / 0.327 ->
  2.70e-5 / 0.0284; b50_stick_wide 4.83e-3 / 0.261 -> 3.28e-5 / 0.0200.

The interpolation itself (zero-weight NaN corners, gridinterp 'L1') is still
checked directly by test_zero_weight_nan_corner and
test_gridinterp_l1_vs_easyspin.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import sys as _sys
import torch
from scipy.io import loadmat

import torchspin.pepper  # noqa: F401  (module, not the re-exported function)
from torchspin.experiment import Experiment, Options
from torchspin.pepper import pepper
from torchspin.pepper_autograd import pepper_autograd
from torchspin.spinsystem import SpinSystem
from torchspin.sphgrid import sphgrid

_pep = _sys.modules['torchspin.pepper']

DATA = Path(__file__).resolve().parents[2] / 'tests' / 'data'
REF = DATA / 'ref_interp_nan_edges.mat'
CONVERGED = DATA / 'ref_missing_knots.npz'

_spec = importlib.util.spec_from_file_location('_gen_missing_knots', DATA / 'generate_missing_knots_refs.py')
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

# Max |torchspin - EasySpin| / max|EasySpin| per case, measured with the fix
# (pepper and pepper_autograd identical to 3 digits) -> tolerance; before the
# fix (0.9.3) in brackets.
#   b48_stick        5.3e-7 [5.6e-2]     b48_stick_noint 6.6e-7 [6.6e-7]
#   b48_h0           2.3e-8 [2.4e-2]     b48_h1          1.1e-7 [7.1e-2]
#   b48_stick_noint has no interpolation and checks the comparison itself.
#   b50_h1           4.4e-5 [5.4e-3]     b50_stick       4.2e-5 [9.6e-3]
#   b50_stick_wide   3.7e-5 [1.2e-4]
# The B50 residual (~4e-5) is present without interpolation too (GridSize
# [19 1]: 4.1e-5) and does not come from the grid interpolation.
MAXD = {
    'b48_stick': 2e-6, 'b48_stick_noint': 2e-6, 'b48_h0': 2e-6, 'b48_h1': 2e-6,
    'b50_h1': 1e-4, 'b50_stick': 1e-4, 'b50_stick_wide': 1e-4,
}
# Since 0.9.28 (B88): measured vs EasySpin (max|dev|/max, 1 - cosine), pepper
# and pepper_autograd alike
DEVIATION = {
    'b48_stick': (0.269, 1.59e-2), 'b48_stick_noint': (0.270, 8.68e-3), 'b48_h0': (0.160, 2.99e-3),
    'b48_h1': (0.463, 3.53e-2), 'b50_h1': (0.230, 4.39e-2), 'b50_stick': (0.278, 1.91e-2),
    'b50_stick_wide': (0.247, 4.59e-3),
}
# converged reference and bounds on (1 - cosine, max|dev|) against it
# (interior; measured x 1.05, see the module docstring)
CONV = {
    'b48_stick': ('b48_stick', 7.7e-8, 8.3e-4), 'b48_stick_noint': ('b48_stick', 1.98e-5, 2.35e-2),
    'b48_h0': ('b48_h0', 2.51e-10, 6.45e-5), 'b48_h1': ('b48_h1', 2.95e-8, 9.66e-4),
    'b50_h1': ('b50_h1', 7.45e-5, 1.51e-2), 'b50_stick': ('b50_stick', 2.84e-5, 2.99e-2),
    'b50_stick_wide': ('b50_stick_wide', 3.44e-5, 2.10e-2),
}


def _load():
    return loadmat(REF, squeeze_me=True, struct_as_record=False)


@pytest.fixture(scope='module')
def ref():
    if not REF.exists():
        pytest.skip(f'reference file missing: {REF}')
    return _load()


def _case_names():
    return [str(c.name) for c in np.atleast_1d(_load()['cases'])] if REF.exists() else []


def _inputs(c):
    S = c.Sys
    f = S._fieldnames
    kw = dict(S=[float(S.S)], g=[[float(v) for v in np.atleast_1d(S.g)]])
    if 'lw' in f and np.any(np.atleast_1d(S.lw)):
        kw['lw'] = [float(v) for v in np.atleast_1d(S.lw)]
    if 'D' in f:
        kw['D'] = [[float(S.D), 0.0]]
    if 'B4' in f:
        kw['B'] = [None, [[float(v) for v in S.B4]]]
    E = c.Exp
    exp = Experiment(mwFreq=float(E.mwFreq), Range=[float(v) for v in E.Range],
                     Harmonic=int(E.Harmonic), nPoints=int(E.nPoints))
    okw = dict(GridSize=[int(v) for v in c.Opt.GridSize], Verbosity=0)
    if 'GridSymmetry' in c.Opt._fieldnames:
        okw['GridSymmetry'] = str(c.Opt.GridSymmetry)
    return SpinSystem(**kw), exp, Options(**okw)


def _xfail_cases():
    out = []
    for n in _case_names():
        d, c = DEVIATION[n]
        reason = (f'B88/B75 (0.9.28): the range cuts transitions; torchspin searches them again over a '
                  f'padded window and clips them in the projection, EasySpin leaves the knots NaN. '
                  f'Measured max dev {d:.3g}, 1 - cosine {c:.3g}; torchspin is closer to the converged '
                  f'reference (test_spectrum_vs_converged)')
        out.append(pytest.param(n, marks=pytest.mark.xfail(strict=True, reason=reason), id=n))
    return out


@pytest.mark.parametrize('func', ['pepper', 'pepper_autograd'])
@pytest.mark.parametrize('name', _xfail_cases())
def test_spectrum_vs_easyspin(ref, name, func):
    c = {str(c.name): c for c in np.atleast_1d(ref['cases'])}[name]
    sys, exp, opt = _inputs(c)
    x, spc = (pepper if func == 'pepper' else pepper_autograd)(sys, exp, opt)
    x = x.detach().numpy(); spc = spc.detach().numpy()
    y = np.asarray(c.spc, dtype=float)
    assert np.allclose(x, np.asarray(c.x, dtype=float), rtol=1e-12, atol=0)
    maxd = float(np.abs(spc - y).max() / np.abs(y).max())
    cs = float(np.dot(spc, y) / (np.linalg.norm(spc) * np.linalg.norm(y)))
    print(f'{name} {func}: max deviation {maxd:.2e}, cosine {cs:.9f}')
    assert maxd < MAXD[name], (name, func, maxd)
    # zero exactly where EasySpin is zero (edge bins of the stick spectra)
    if int(c.Exp.Harmonic) == 0 and 'lw' not in c.Sys._fieldnames:
        nz_ts = np.nonzero(np.abs(spc) > 1e-12 * np.abs(y).max())[0]
        nz_es = np.nonzero(np.abs(y) > 1e-12 * np.abs(y).max())[0]
        assert (nz_ts.min(), nz_ts.max()) == (nz_es.min(), nz_es.max()), name


@pytest.mark.parametrize('func', ['pepper', 'pepper_autograd'])
@pytest.mark.parametrize('name', list(CONV))
def test_spectrum_vs_converged(ref, name, func):
    """The seven EasySpin spectra against converged references."""
    c = {str(c.name): c for c in np.atleast_1d(ref['cases'])}[name]
    sys, exp, opt = _inputs(c)
    rn, cmax, dmax = CONV[name]
    x, spc = (pepper if func == 'pepper' else pepper_autograd)(sys, exp, opt)
    x = x.detach().numpy(); spc = spc.detach().numpy()
    r = np.load(CONVERGED)
    assert np.allclose(x, r[rn + '_x'], rtol=0, atol=1e-9)
    cs, d = gen.metrics(spc, r[rn], gen.interior(x, sys))
    print(f'{name} {func}: 1 - cosine {cs:.3e}, max dev {d:.3e}')
    assert cs <= cmax and d <= dmax, (name, func, cs, d)


def _grid_ids():
    return [f'{g.sym}-{int(g.N)}' for g in np.atleast_1d(_load()['grids'])] if REF.exists() else []


@pytest.mark.parametrize('k', range(len(_grid_ids())), ids=_grid_ids())
def test_sphgrid_phi_bitwise(ref, k):
    """sphgrid phi equals EasySpin's linspace(0, maxPhi, nPhi) bitwise; theta too."""
    g = np.atleast_1d(ref['grids'])[k]
    phi, theta, _, _ = sphgrid(str(g.sym), int(g.N))
    assert np.array_equal(phi.numpy(), np.atleast_1d(g.phi).astype(float)), (str(g.sym), int(g.N))
    assert np.array_equal(theta.numpy(), np.atleast_1d(g.theta).astype(float)), (str(g.sym), int(g.N))


def _interp_ids():
    return [f'{c.sym}-{int(c.Nc)}-{int(c.Nf)}' for c in np.atleast_1d(_load()['interp'])] if REF.exists() else []


# Open-phi grids (periodic rows): EasySpin's 'L1' rectify samples the wrapped
# row at linspace(1, len, nc) (gridinterp.m:188, 217), i.e. it stretches the
# row over the len knots and leaves out the wrap segment; torchspin's _row
# samples linspace(1, len + 1, nc). Not changed here (B58). C1 also differs
# in the lower hemisphere, deliberately: since 0.9.21 torchspin reads those
# rows in phi order, EasySpin mirrors them (B74, test_c1_lower_rows.py).
_OPEN_PHI = {'Ci', 'C1', 'C2h', 'C4h', 'C6h', 'S6'}


@pytest.mark.parametrize('k', range(len(_interp_ids())), ids=_interp_ids())
def test_gridinterp_l1_vs_easyspin(ref, k):
    """pepper._interp_grid(linear=True) vs EasySpin gridinterp(..., 'L1') on
    synthetic data with every fifth coarse knot missing: same NaN pattern,
    values to 1e-13."""
    c = np.atleast_1d(ref['interp'])[k]
    if str(c.sym) in _OPEN_PHI:
        pytest.xfail('B58: EasySpin L1 rectify of periodic rows (gridinterp.m:188)')
    y = np.atleast_1d(c.y).astype(float)
    ye = np.atleast_1d(c.yi).astype(float)
    yi = _pep._interp_grid(y, str(c.sym), int(c.Nc), int(c.Nf), linear=True)
    nan = np.isnan(ye)
    assert np.array_equal(np.isnan(yi), nan), (str(c.sym), int((np.isnan(yi) != nan).sum()))
    assert np.allclose(yi[~nan], ye[~nan], rtol=0, atol=1e-13)


def test_zero_weight_nan_corner():
    """interp2([1 NaN; 2 3], 1, 1.5) = 1.5 in MATLAB: a NaN corner with zero
    weight does not propagate. Here: a fine knot on a coarse row next to a
    missing knot keeps its value."""
    N_c, N_f = 3, 5                           # D2h: coarse knots 1 + 2 + 3
    y = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    yi = _pep._interp_grid(y, 'D2h', N_c, N_f, linear=True)
    # the north pole and the equator do not touch the missing knot
    assert yi[0] == 1.0
    assert np.array_equal(yi[-5:], np.array([4.0, 4.5, 5.0, 5.5, 6.0]))


def _b48(g):
    # PaddingMultiplier=0: this checks the gradient of the B48 interpolation.  With
    # the range padding (0.9.6) the end bins, which since 0.9.28 carry clipped
    # triangles, are copied 225 times; their kinks (vertices crossing bins at the
    # range ends) then limit central differences to ~1e-6 (h 1e-6: 1.0e-6, 1e-7:
    # 1.6e-6, 3e-8: 4.5e-6), while autograd is the exact local slope (each end bin
    # agrees with central differences to their 1e-5 jitter).  Without padding the
    # central differences converge to 1e-8.
    exp = Experiment(mwFreq=9.5, Range=[325, 337], nPoints=601, Harmonic=1)
    return SpinSystem(S=[0.5], g=g, lw=[1.5]), exp, Options(GridSize=[31, 4], Verbosity=0, PaddingMultiplier=0)


def _b50(D):
    exp = Experiment(mwFreq=9.5, Range=[290, 390], nPoints=1024, Harmonic=1)
    sys = SpinSystem(S=[2.5], g=[2.0], D=D, B=[None, [[30.0, 0, 0, 0, 10.0, 0, 0, 0, 0]]], lw=[1.0, 0.0])
    return sys, exp, Options(GridSize=[19, 4], GridSymmetry='D2h', Verbosity=0)


@pytest.mark.parametrize('case', ['b48_g', 'b50_D'])
def test_autograd_vs_central_differences(case):
    """pepper_autograd gradients of a weighted sum of the spectrum vs central
    differences of pepper, on the B48 (d/dg) and B50 (d/dD, d/dE) systems whose
    ranges cut transitions. Measured (relative): B48 g 1.9e-8, 1.6e-8, 7.6e-8
    (h = 1e-7), B50 D 3.3e-6 (h = 1e-2 MHz, truncation; 1e-3: 2.4e-7).

    d/dE at E = 0: the system is axial and the D2h grid is symmetric under
    x <-> y, so the loss is even in E and the derivative is 0.  Measured
    AD -1.8e-14, CD -8.0e-11 (h = 1e-2; OMP 8: -8.9e-15 / -4.0e-9), i.e.
    <= 3.6e-14 / 1.6e-8 of d/dD.  Up to 0.9.27
    both were -1.07e-3 (agreeing to 6e-7): the all-linear interpolation of a
    sweep with missing knots was not x <-> y symmetric; since 0.9.28 (B88) no
    knot is missing here and the symmetry holds."""
    if case == 'b48_g':
        p0, h, build = [2.0, 2.05, 2.1], 1e-7, (lambda p: _b48(p.reshape(1, 3)))
        rel = [1e-6, 1e-6, 1e-6]
    else:
        p0, h, build = [300.0, 0.0], 1e-2, (lambda p: _b50(p.reshape(1, 2)))
        rel = [1e-5, 1e-5]
    p = torch.tensor(p0, dtype=torch.float64, requires_grad=True)
    _, spc = pepper_autograd(*build(p))
    w = torch.linspace(0.0, 1.0, spc.numel(), dtype=torch.float64) ** 2
    grad = torch.autograd.grad((spc * w).sum(), p)[0]
    assert not bool(torch.isnan(grad).any())

    def L(q):
        return float((pepper(*build(q))[1] * w).sum())

    for j in range(len(p0)):
        if case == 'b50_D' and j == 1:
            continue
        pp = torch.tensor(p0, dtype=torch.float64); pp[j] += h
        pm = torch.tensor(p0, dtype=torch.float64); pm[j] -= h
        fd = (L(pp) - L(pm)) / (2 * h)
        print(f'{case}[{j}]: autograd {float(grad[j]):.10g}, central difference {fd:.10g}')
        assert float(grad[j]) == pytest.approx(fd, rel=rel[j]), (case, j, float(grad[j]), fd)
    if case == 'b50_D':
        fdE = (L(torch.tensor([300.0, h], dtype=torch.float64)) - L(torch.tensor([300.0, -h], dtype=torch.float64))) / (2 * h)
        print(f'{case}[1]: autograd {float(grad[1]):.4g}, central difference {fdE:.4g} (0 by symmetry)')
        scale = abs(float(grad[0]))
        # AD exactly 0 up to roundoff; the central difference is the roundoff of
        # the loss difference over 2h (OMP 1: 3.2e-10 of d/dD, OMP 8: 1.6e-8)
        assert abs(float(grad[1])) <= 1e-9 * scale and abs(fdE) <= 1e-7 * scale, (float(grad[1]), fdE)
