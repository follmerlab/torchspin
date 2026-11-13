"""Position interpolation on axial (Dinfh) grids against EasySpin (B30;
tests/data/ref_pepper_dinfh_interp.mat, generate_pepper_dinfh_interp_refs.m).

EasySpin's G3 mode on an axial grid is ``esspline1d(y, 1, iphi)``
(gridinterp.m:77-78): a cubic spline over the θ knot index with zero first
derivatives at both ends (esspline1d.m ``EndCon = 1``).  Up to 0.9.6
torchspin used a not-a-knot spline in cos θ, which bends differently next to
both poles.  Measured on main (pepper and pepper_autograd alike):
cosine 0.99863 (g axial, GridSize [10 10]), 0.99380 (Cu(II) axial, default
grid), 0.93891 (Cu(II), [10 10]) and amplitude up to 1.045.  With the fix all
six cases have cosine >= 1 - 1e-10 and amplitude within 3e-6.

Also: the interpolation matrix against esspline1d written out from
esspline1d.m, and pepper_autograd vs central differences on an axial system.
"""
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io import loadmat

import torchspin  # noqa: F401
from torchspin import SpinSystem, Experiment, Options, pepper
from torchspin.pepper_autograd import pepper_autograd, _interp_matrix
from torchspin.sphgrid import sphgrid
from torchspin.tests.test_pepper_matlab_validation_ext2 import (
    sys_from_mat, exp_from_mat, opt_from_mat, cosine)
from torchspin.tests.test_interp_matrix import _esspline1d_endslopes0

import sys as _sys
pep = _sys.modules['torchspin.pepper']

REF = Path(__file__).resolve().parents[2] / 'tests' / 'data' / 'ref_pepper_dinfh_interp.mat'

COS_MIN = 1 - 1e-8
AMP_TOL = 2e-5


def _load():
    m = loadmat(REF, squeeze_me=True, struct_as_record=False)
    return {str(c.name): c for c in np.atleast_1d(m['cases'])}


@pytest.fixture(scope='module')
def cases():
    if not REF.exists():
        pytest.skip(f'reference file missing: {REF}')
    return _load()


def _case_names():
    return list(_load().keys()) if REF.exists() else []


@pytest.mark.parametrize('func', ['pepper', 'pepper_autograd'])
@pytest.mark.parametrize('name', _case_names())
def test_dinfh_interp_vs_matlab(cases, name, func):
    c = cases[name]
    sys = sys_from_mat(c.Sys)
    exp = exp_from_mat(c.Exp, c.Sys, x_axis=np.asarray(c.x, dtype=float))
    opt = opt_from_mat(c.Opt)
    fn = pepper if func == 'pepper' else pepper_autograd
    x, spc = fn(sys, exp, opt)
    x = x.detach().cpu().numpy().ravel(); spc = spc.detach().cpu().numpy().ravel()
    ref = np.asarray(c.spc, dtype=float).ravel(); ref_x = np.asarray(c.x, dtype=float).ravel()
    assert np.allclose(x, ref_x, rtol=1e-9), 'sweep axis differs'
    cs = cosine(spc, ref)
    amp = abs(spc).max() / abs(ref).max()
    print(f'{name} {func}: cosine {cs:.11f}, amplitude ratio {amp:.7f}')
    assert cs > COS_MIN, (name, cs)
    assert amp == pytest.approx(1.0, abs=AMP_TOL), (name, amp)


@pytest.mark.parametrize('N_c,factor', [(2, 4), (3, 4), (4, 3), (10, 10), (19, 4)])
def test_dinfh_matrix_is_esspline1d_endslopes0(N_c, factor):
    """Dinfh G3 = esspline1d(y, 1, iphi), iphi = 1:1/Factor:n (gridinterp.m:77-78)."""
    N_f = (N_c - 1) * factor + 1
    _, _, _, vc = sphgrid('Dinfh', N_c)
    _, _, _, vf = sphgrid('Dinfh', N_f)
    M = _interp_matrix('Dinfh', N_c, N_f, 'pos', False)
    rng = np.random.default_rng(N_c)
    for _ in range(3):
        v = 300 + 20 * rng.standard_normal(N_c)
        ref = _esspline1d_endslopes0(v, N_f)
        assert np.abs(pep._interp_sph(vc, vf, v) - ref).max() < 1e-10
        assert np.abs(M @ v - ref).max() < 1e-10


def test_dinfh_autograd_vs_central_differences():
    """d spc / d (g_perp, g_par, A_par) through the Dinfh G3 interpolation
    (Cu(II)-like axial system, interpolation factor 10)."""
    exp = Experiment(mwFreq=9.5, Range=[250, 360], nPoints=1024, Harmonic=1)
    opt = Options(GridSize=[10, 10], Verbosity=0)
    p0 = [2.05, 2.25, 500.0]

    def run(gp, gz, az):
        def t(v):
            return torch.as_tensor(v, dtype=torch.float64)
        g = torch.stack([t(gp), t(gp), t(gz)])
        A = torch.stack([t(50.0), t(50.0), t(az)])
        sys = SpinSystem(S=0.5, g=g, Nucs='63Cu', A=A, lw=[0.5, 0.0])
        return pepper_autograd(sys, exp, opt)[1]

    p = torch.tensor(p0, dtype=torch.float64, requires_grad=True)
    spc = run(p[0], p[1], p[2])
    w = torch.linspace(-1.0, 1.0, spc.numel(), dtype=torch.float64)
    (grad,) = torch.autograd.grad((spc * w).sum(), p)
    for k, h in enumerate([1e-6, 1e-6, 1e-3]):
        e = list(p0); e[k] += h; fp = (run(*e).detach() * w).sum()
        e = list(p0); e[k] -= h; fm = (run(*e).detach() * w).sum()
        fd = float((fp - fm) / (2 * h))
        assert abs(fd) > 0
        assert float(grad[k]) == pytest.approx(fd, rel=1e-5), (k, float(grad[k]), fd)
