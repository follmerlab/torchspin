"""Powder spectra on the D3d/D4h/D6h/Oh/C6h/S6 grids against EasySpin (B36;
tests/data/ref_sphgrid_maxphi.mat, generate_sphgrid_maxphi_refs.m).

EasySpin's grids cover φ = 0..maxPhi with maxPhi from private/gridparam.m
(D6h π/6, D4h/Oh π/4, D3d π/3, C6h π/3, S6 2π/3). Up to 0.9.2 torchspin's
``sphgrid`` used π/2 (π for S6) while its interpolation and triangulation used
gridparam, so

* with orientations without resonances (D4h [200 480] mT, Oh [250 430] mT)
  the knots were interpolated on the wrong grid layout: cosine 0.67 / 0.49;
* even with resonances at all knots the D3d, S6 and C6h grids held a
  non-integer number of symmetry wedges: cosine 0.964 (D3d), 0.901 (D3d system
  on an S6 grid), 0.735 (D6h system on a C6h grid), 0.949 (D6h);
* with missing knots pepper interpolated transitions resonant at all knots
  bicubically, EasySpin linearly (L1): the D4h system on a D2h grid 0.959;
* ``Options`` rejected an explicit ``GridSymmetry`` of these groups (B51).

Also: EasySpin's ``sphgrid`` knots and weights for all 14 groups, and
pepper_autograd gradients vs central differences on a D4h grid.
"""
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io import loadmat

from torchspin.experiment import Experiment, Options
from torchspin.pepper import pepper
from torchspin.pepper_autograd import pepper_autograd
from torchspin.spinsystem import SpinSystem
from torchspin.sphgrid import sphgrid

REF = Path(__file__).resolve().parents[2] / 'tests' / 'data' / 'ref_sphgrid_maxphi.mat'

# Measured with the fix (pepper and pepper_autograd, all 9 cases): cosine
# >= 0.999935 (Oh; D4h and D2h grid 1 - 3e-8, D3d 0.9999976, D6h 0.9999916, C6h grid
# 0.9999801, S6 grid 0.9999975) and amplitude ratio within 4.6e-4 of 1 (C6h
# grid; Oh 0.99975). Before the fix the best case was cosine 0.964.
COS_MIN = 0.9999
AMP_TOL = 1e-3


def _load():
    return loadmat(REF, squeeze_me=True, struct_as_record=False)


@pytest.fixture(scope='module')
def ref():
    if not REF.exists():
        pytest.skip(f'reference file missing: {REF}')
    return _load()


def _case_names():
    return [str(c.name) for c in np.atleast_1d(_load()['cases'])] if REF.exists() else []


def _system(S):
    """SpinSystem of a reference case (S, isotropic g = 2, D, B4/B6, lw [1 0])."""
    f = S._fieldnames
    kw = dict(S=[float(S.S)], g=float(S.g), lw=[float(v) for v in np.atleast_1d(S.lw)])
    if 'D' in f:
        kw['D'] = [[float(S.D), 0.0]]
    B = [None, None, None]                       # B2, B4, B6
    if 'B4' in f:
        B[1] = [[float(v) for v in S.B4]]
    if 'B6' in f:
        B[2] = [[float(v) for v in S.B6]]
    while B[-1] is None:
        B.pop()
    kw['B'] = B
    return SpinSystem(**kw)


def _inputs(c):
    E = c.Exp
    exp = Experiment(mwFreq=float(E.mwFreq), Range=[float(v) for v in E.Range],
                     Harmonic=int(E.Harmonic), nPoints=int(E.nPoints))
    kw = dict(GridSize=[int(v) for v in c.Opt.GridSize], Verbosity=0)
    if 'GridSymmetry' in c.Opt._fieldnames:
        kw['GridSymmetry'] = str(c.Opt.GridSymmetry)
    return _system(c.Sys), exp, Options(**kw)


@pytest.mark.parametrize('func', ['pepper', 'pepper_autograd'])
@pytest.mark.parametrize('name', _case_names())
def test_spectrum_vs_easyspin(ref, name, func):
    c = {str(c.name): c for c in np.atleast_1d(ref['cases'])}[name]
    sys, exp, opt = _inputs(c)
    from torchspin.hamsymm import hamsymm
    assert hamsymm(sys)[0] == str(c.hamsymm)
    x, spc = (pepper if func == 'pepper' else pepper_autograd)(sys, exp, opt)
    x = x.detach().numpy(); spc = spc.detach().numpy()
    y = np.asarray(c.spc, dtype=float)
    assert np.allclose(x, np.asarray(c.x, dtype=float), rtol=1e-12, atol=0)
    cs = float(np.dot(spc, y) / (np.linalg.norm(spc) * np.linalg.norm(y)))
    amp = float(np.abs(spc).max() / np.abs(y).max())
    print(f'{name} {func}: cosine {cs:.7f}, amplitude ratio {amp:.5f}')
    assert cs > COS_MIN, (name, func, cs)
    assert amp == pytest.approx(1.0, abs=AMP_TOL), (name, func, amp)


@pytest.mark.parametrize('k', range(14))
def test_sphgrid_knots_vs_easyspin(ref, k):
    """EasySpin sphgrid(sym, 7): same knots in the same order, same weights."""
    g = np.atleast_1d(ref['grids'])[k]
    phi, theta, w, _ = sphgrid(str(g.sym), 7)
    assert np.allclose(phi.numpy(), np.atleast_1d(g.phi), rtol=0, atol=1e-13), str(g.sym)
    assert np.allclose(theta.numpy(), np.atleast_1d(g.theta), rtol=0, atol=1e-13), str(g.sym)
    assert np.allclose(w.numpy() / w.sum().item(), np.atleast_1d(g.weights) / np.sum(g.weights),
                       rtol=1e-12, atol=0), str(g.sym)


@pytest.mark.parametrize('sym', ['D4h', 'auto'])
def test_d4h_autograd_vs_central_differences(sym):
    """pepper_autograd on the D4h grid with orientations without resonances
    ([200 480] mT): d/dD and d/dg of a weighted sum of the spectrum vs central
    differences of pepper. Measured: 6e-8 (D, h = 1e-2), 2e-9 (g, h = 1e-5)."""
    B4 = [[30.0, 0, 0, 0, 10, 0, 0, 0, 0]]
    exp = Experiment(mwFreq=9.5, Range=[200, 480], Harmonic=1, nPoints=1024)
    opt = Options(GridSize=[19, 4], GridSymmetry=sym, Verbosity=0)

    def sysof(g, D):
        D = torch.stack([D, torch.zeros_like(D)]).unsqueeze(0) if torch.is_tensor(D) else [[D, 0.0]]
        return SpinSystem(S=[2.5], g=g, D=D, B=[None, B4], lw=[1.0, 0.0])

    g = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    D = torch.tensor(300.0, dtype=torch.float64, requires_grad=True)
    _, spc = pepper_autograd(sysof(g, D), exp, opt)
    w = torch.linspace(-1.0, 1.0, spc.numel(), dtype=torch.float64)
    dg, dD = torch.autograd.grad((spc * w).sum(), [g, D])

    def L(gv, Dv):
        return float((pepper(sysof(gv, Dv), exp, opt)[1] * w).sum())

    h = 1e-2
    fdD = (L(2.0, 300.0 + h) - L(2.0, 300.0 - h)) / (2 * h)
    h = 1e-5
    fdg = (L(2.0 + h, 300.0) - L(2.0 - h, 300.0)) / (2 * h)
    assert float(dD) == pytest.approx(fdD, rel=1e-5), (float(dD), fdD)
    assert float(dg) == pytest.approx(fdg, rel=1e-5), (float(dg), fdg)
