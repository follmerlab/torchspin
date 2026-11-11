"""Closed-phi grid interpolation against EasySpin (B25, 0.8.9;
tests/data/ref_closedphi_interp.mat, generate_closedphi_interp_refs.m).

EasySpin's G3 interpolation on the triangular D2h-type grid first resamples
every theta row along phi with ``esspline1d(y, 1, nc)`` (zero end slopes,
gridinterp.m:184), then applies the not-a-knot bicubic spline.  torchspin used
a not-a-knot spline for the rows; on the short rows next to the pole (2-5
knots) that changes the interpolated resonance fields near g_z, so Harmonic 1
peak amplitudes were 1.0048-1.0058 of EasySpin's for g = [2 2.05 2.1] at
GridSize [19 4] (relative residual 4e-3 to 7e-3).  With the EasySpin rule the
interpolated spectra agree to a relative residual below 1e-6 (field sweeps;
the EasySpin field-sweep floor without interpolation is the same) and to 1e-12
(frequency sweep, salt).

Checked for pepper, pepper_autograd (field sweeps) and fixed-field salt
(``Method='perturb1'``, the path torchspin salt takes for one electron without
a quadrupole).  Also: the gradient d spc / d g_z of pepper_autograd through the
interpolated grid against central differences of torchspin and of EasySpin.
"""
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io import loadmat

from torchspin.tests.test_pepper_matlab_validation_ext2 import (
    sys_from_mat, exp_from_mat, opt_from_mat)
from torchspin.experiment import Experiment, Options
from torchspin.pepper import pepper
from torchspin.pepper_autograd import pepper_autograd
from torchspin.salt import salt
from torchspin.spinsystem import SpinSystem

REF = Path(__file__).resolve().parents[2] / 'tests' / 'data' / 'ref_closedphi_interp.mat'

# Relative residual ||y - y_ref|| / ||y_ref|| after the fix (measured):
# field sweeps 2e-8 to 6e-7, HStrain 8.0e-5, frequency sweep 6e-13.
# Before (0.8.8 and earlier): 3.1e-4 (Harmonic 0) to 1.5e-2 (14N), HStrain 1.5e-4,
# frequency sweep 4.4e-3.  HStrain's widths are interpolated linearly (EasySpin
# 'L1', unchanged) and its floor is set by that, not by the row spline.
RESID_TOL = {'hstrain_h1_19': 1e-4, 'freq_lorentz_h1_19': 1e-9}
RESID_TOL_DEFAULT = 5e-6
AMP_TOL = 1e-4          # measured |amp - 1| <= 4e-6; before: up to 5.8e-3
COS_MIN = 1 - 1e-7      # measured 1 - 1e-9 or better


def _load():
    m = loadmat(REF, squeeze_me=True, struct_as_record=False)
    return m


@pytest.fixture(scope='module')
def ref():
    if not REF.exists():
        pytest.skip(f'reference file missing: {REF}')
    return _load()


def _cases(key):
    if not REF.exists():
        return []
    return [str(c.name) for c in np.atleast_1d(_load()[key])]


def _case(ref, key, name):
    return {str(c.name): c for c in np.atleast_1d(ref[key])}[name]


def _metrics(y, r):
    resid = float(np.linalg.norm(y - r) / np.linalg.norm(r))
    amp = float(np.abs(y).max() / np.abs(r).max())
    cos = float(y @ r / np.linalg.norm(y) / np.linalg.norm(r))
    return resid, amp, cos


def _check(name, y, r):
    resid, amp, cos = _metrics(y, r)
    print(f'{name}: residual {resid:.2e}, amplitude {amp:.7f}, cosine {cos:.10f}')
    assert resid < RESID_TOL.get(name, RESID_TOL_DEFAULT), (name, resid)
    assert amp == pytest.approx(1.0, abs=AMP_TOL), (name, amp)
    assert cos > COS_MIN, (name, cos)


def _run_pepper(c, fn):
    exp = exp_from_mat(c.Exp, c.Sys, x_axis=np.asarray(c.x, dtype=float))
    x, y = fn(sys_from_mat(c.Sys), exp, opt_from_mat(c.Opt))
    x = x.detach().cpu().numpy()
    ref_x = np.asarray(c.x, dtype=float)
    assert np.allclose(x, ref_x, rtol=1e-9, atol=1e-9 * abs(ref_x).max()), 'sweep axis differs'
    return y.detach().cpu().numpy()


@pytest.mark.parametrize('name', _cases('cases'))
def test_pepper_closedphi_interp_vs_easyspin(ref, name):
    c = _case(ref, 'cases', name)
    _check(name, _run_pepper(c, pepper), np.asarray(c.spc, dtype=float))


@pytest.mark.parametrize('name', [n for n in _cases('cases') if not n.startswith('freq')])
def test_pepper_autograd_closedphi_interp_vs_easyspin(ref, name):
    c = _case(ref, 'cases', name)
    _check(name, _run_pepper(c, pepper_autograd), np.asarray(c.spc, dtype=float))


@pytest.mark.parametrize('name', _cases('salts'))
def test_salt_closedphi_interp_vs_easyspin(ref, name):
    """Before: residual 7.6e-4 ([19 4]) and 1.3e-4 ([31 4]); now ~1e-12."""
    c = _case(ref, 'salts', name)
    S, E = c.Sys, c.Exp
    x, y = salt(SpinSystem(S=[0.5], g=[np.asarray(S.g, float).tolist()], Nucs=str(S.Nucs),
                           A=[np.asarray(S.A, float).tolist()]),
                Experiment(Field=float(E.Field), mwFreq=float(E.mwFreq)),
                Options(GridSize=[int(v) for v in np.ravel(c.Opt.GridSize)]),
                freq_range=tuple(np.asarray(E.Range, float)), n_points=int(E.nPoints),
                lw_mhz=float(S.lwEndor))
    x, y = x.numpy(), y.numpy()
    np.testing.assert_allclose(x, np.asarray(c.x, float), rtol=0, atol=1e-12)
    # torchspin returns density * dx / (4 pi), EasySpin density * 2 pi
    # (see test_salt_first_order_matlab_validation)
    r = np.asarray(c.spc, float) * (x[1] - x[0]) / (8 * math.pi ** 2)
    resid, amp, cos = _metrics(y, r)
    print(f'{name}: residual {resid:.2e}, amplitude {amp:.9f}, cosine {cos:.12f}')
    assert resid < 1e-9, (name, resid)


def test_autograd_dgz_through_interpolated_grid():
    """d spc / d g_z of the interpolated rhombic powder spectrum (GridSize [19 4],
    Harmonic 1): reverse mode of pepper_autograd against central differences of
    pepper (h = 1e-6; h = 1e-4 is a secant 3 % off for random weights),
    contracted with three weight vectors."""
    exp = Experiment(mwFreq=9.5, Range=[320, 345], nPoints=1024, Harmonic=1)
    opt = Options(Verbosity=0, GridSize=[19, 4])
    x = np.linspace(320, 345, 1024)
    g = torch.tensor([2.0, 2.05, 2.1], dtype=torch.float64, requires_grad=True)
    _, y = pepper_autograd(SpinSystem(S=[0.5], g=[g], lw=[1.0, 0.0]), exp, opt)

    def spec(gz):
        return pepper(SpinSystem(S=[0.5], g=[2.0, 2.05, gz], lw=[1.0, 0.0]), exp, opt)[1].numpy()
    h = 1e-6
    fd = (spec(2.1 + h) - spec(2.1 - h)) / (2 * h)
    rng = np.random.default_rng(0)
    edge = (x > 320.5) & (x < 326.0)          # g_z = 2.1 resonates at 323.2 mT
    for k, wv in enumerate([rng.standard_normal(x.size), rng.standard_normal(x.size) * edge, fd / np.abs(fd).max()]):
        (grad,) = torch.autograd.grad((y * torch.as_tensor(wv)).sum(), g, retain_graph=True)
        ad, num = float(grad[2]), float(fd @ wv)
        print(f'weights {k}: autograd {ad:.10e}, central difference {num:.10e}')
        assert ad == pytest.approx(num, rel=1e-6), (k, ad, num)


def test_dgz_finite_difference_vs_easyspin(ref):
    """The same derivative as a secant (h = 1e-4) from torchspin and EasySpin:
    relative norm 4.2e-5 now, 1.9e-2 before (largest at the g_z edge,
    323.6 mT).  Smaller h is limited by the 5e-7 spectrum floor divided by 2h."""
    d = ref['dgz']
    h = float(d.h)
    exp = Experiment(mwFreq=9.5, Range=[320, 345], nPoints=1024, Harmonic=1)
    opt = Options(Verbosity=0, GridSize=[19, 4])

    def spec(gz):
        return pepper(SpinSystem(S=[0.5], g=[2.0, 2.05, gz], lw=[1.0, 0.0]), exp, opt)[1].numpy()
    fd = (spec(2.1 + h) - spec(2.1 - h)) / (2 * h)
    fd_es = np.asarray(d.dspc, float)
    rel = float(np.linalg.norm(fd - fd_es) / np.linalg.norm(fd_es))
    print(f'd/dg_z secant vs EasySpin: relative norm {rel:.2e}')
    assert rel < 2e-4, rel
