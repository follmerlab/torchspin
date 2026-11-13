"""pepper_autograd on the O3 grid (isotropic systems, B66).

An isotropic system (auto symmetry) or ``Options.GridSymmetry='O3'`` has a
single orientation and no facets to project.  pepper draws every line with an
exact template at its position (EasySpin ~anisotropicSpectrum).  pepper_autograd
sent the one orientation to the triangle projection, whose O3 triangulation is
empty, and returned an all-zero spectrum with zero gradient; ``pepper`` with an
isotropic parameter that requires grad routes there and returned zeros too.

* pepper_autograd == pepper for a matrix of isotropic cases (nuclei, line
  shapes, harmonics, explicit and automatic O3, S = 1, Temperature, strain,
  modulation, Opt.Transitions, perturbation theory).
* Both against EasySpin (tests/data/ref_pepper_isotropic.mat,
  generate_pepper_isotropic_refs.m).
* ``pepper`` with an isotropic g/A that requires grad gives the spectrum without
  grad; gradients w.r.t. g, A and lw match central differences of pepper.
* A grid with orientations but no usable facets raises instead of returning 0.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io import loadmat

import importlib
from torchspin import SpinSystem, Experiment, Options, pepper
from torchspin.hamsymm import hamsymm
from torchspin.pepper_autograd import pepper_autograd
from torchspin.tests.test_pepper_matlab_validation_ext2 import (
    sys_from_mat, exp_from_mat, opt_from_mat, cosine)

_pep_mod = importlib.import_module('torchspin.pepper')   # the module, not the function

REF = Path(__file__).resolve().parents[2] / 'tests' / 'data' / 'ref_pepper_isotropic.mat'

E0 = dict(mwFreq=9.5, Range=[320, 360], nPoints=1024)
E_CU = dict(mwFreq=9.5, Range=[250, 400], nPoints=1024)
NUCS = {
    'none': (dict(), E0),
    '1H': (dict(Nucs='1H', A=[30.0]), E0),
    '14N': (dict(Nucs='14N', A=[45.0]), E0),
    '63Cu': (dict(Nucs='63Cu', A=[300.0]), E_CU),
}
LW = {'gauss': [0.5, 0.0], 'lorentz': [0.0, 0.5], 'voigt': [0.3, 0.3]}


def _rel(a, b):
    a = a.detach(); b = b.detach()
    return float((a - b).abs().max() / a.abs().max())


def _both(sys, exp, opt=None):
    opt = opt or Options(Verbosity=0)
    _, yp = pepper(sys, exp, opt)
    x, ya = pepper_autograd(sys, exp, opt)
    assert x.shape == (exp.nPoints,) and torch.all(torch.isfinite(ya))
    assert float(yp.abs().max()) > 0
    return yp, ya


@pytest.mark.parametrize('harmonic', [0, 1, 2])
@pytest.mark.parametrize('shape', list(LW))
@pytest.mark.parametrize('nuc', list(NUCS))
def test_o3_autograd_equals_pepper(nuc, shape, harmonic):
    sd, ek = NUCS[nuc]
    sys = SpinSystem(S=0.5, g=2.0, lw=LW[shape], **sd)
    assert hamsymm(sys)[0] == 'O3'
    yp, ya = _both(sys, Experiment(Harmonic=harmonic, **ek))
    assert _rel(yp, ya) < 1e-9, f'{nuc} {shape} H={harmonic}: {_rel(yp, ya):.2e}'


OTHER = {
    'explicit_O3_1H': (dict(S=0.5, g=2.0, Nucs='1H', A=[30.0], lw=[0.5, 0]), E0, dict(GridSymmetry='O3')),
    'explicit_O3_grid_interp': (dict(S=0.5, g=2.0, lw=[1.0, 0]), E0, dict(GridSymmetry='O3', GridSize=[7, 4])),
    'auto_grid_interp': (dict(S=0.5, g=2.0, lw=[1.0, 0]), E0, dict(GridSize=[31, 4])),
    'no_lw_sticks': (dict(S=0.5, g=2.0, Nucs='1H', A=[30.0]), dict(E0, Harmonic=1), {}),
    'lwpp': (dict(S=0.5, g=2.0, lwpp=[1.0, 0.0]), dict(E0, Harmonic=1), {}),
    '1H_14N': (dict(S=0.5, g=2.0, Nucs='1H,14N', A=[30.0, 45.0], lw=[0.5, 0]), dict(E0, Harmonic=1), {}),
    'Cu_natural': (dict(S=0.5, g=2.1, Nucs='Cu', A=[300.0], lw=[1.0, 0]), dict(E_CU, Harmonic=1), {}),
    'perturb_63Cu': (dict(S=0.5, g=2.0, Nucs='63Cu', A=[300.0], lw=[0.5, 0]), dict(E_CU, Harmonic=1), dict(Method='perturb')),
    'S1': (dict(S=1.0, g=2.0, lw=[1.0, 0]), dict(E0, Harmonic=1), {}),
    'S1_T5': (dict(S=1.0, g=2.0, lw=[1.0, 0]), dict(E0, Harmonic=1, Temperature=5.0), {}),
    'S12_T2': (dict(S=0.5, g=2.0, lw=[1.0, 0]), dict(E0, Harmonic=1, Temperature=2.0), {}),
    'S32_lorentz': (dict(S=1.5, g=2.0, lw=[0, 0.8]), dict(E0, Harmonic=1), {}),
    'S1_transitions_12': (dict(S=1.0, g=2.0, lw=[1.0, 0]), dict(E0, Harmonic=1), dict(Transitions=[[1, 2]])),
    'S1_transitions_all': (dict(S=1.0, g=2.0, lw=[1.0, 0]), dict(E0, Harmonic=1), dict(Transitions='all')),
    'HStrain_iso': (dict(S=0.5, g=2.0, HStrain=[20.0, 20.0, 20.0]), dict(E0, Harmonic=1), {}),
    'HStrain_iso_lorentz_lw': (dict(S=0.5, g=2.0, HStrain=[20.0, 20.0, 20.0], lw=[0, 0.3]), dict(E0, Harmonic=1), {}),
    'HStrainL_iso': (dict(S=0.5, g=2.0, HStrainL=[20.0, 20.0, 20.0]), dict(E0, Harmonic=1), {}),
    'ModAmp': (dict(S=0.5, g=2.0, lw=[1.0, 0]), dict(E0, Harmonic=1, ModAmp=0.5), {}),
    # explicit O3 on an anisotropic system, with field-quadratic Sys.Ham terms
    'explicit_O3_QuadraticZeeman': (dict(S=1.0, g=2.0, D=[300.0, 30.0], lw=[1.0, 0],
                                         Ham={'220': [[2e-4]], '222': [[0, 0, 2e-4, 0, 0]]}),
                                    dict(mwFreq=9.5, Range=[250, 430], nPoints=1024, Harmonic=1),
                                    dict(GridSymmetry='O3', QuadraticZeeman=True)),
}


@pytest.mark.parametrize('name', list(OTHER))
def test_o3_autograd_equals_pepper_options(name):
    sd, ek, ok = OTHER[name]
    yp, ya = _both(SpinSystem(**sd), Experiment(**ek), Options(Verbosity=0, **ok))
    assert _rel(yp, ya) < 1e-9, f'{name}: {_rel(yp, ya):.2e}'


def test_o3_options_change_the_spectrum():
    """The options above are not no-ops on O3 (so equality is not trivial)."""
    exp = Experiment(Harmonic=1, **E0)
    base = pepper_autograd(SpinSystem(S=1.0, g=2.0, lw=[1.0, 0]), exp, Options(Verbosity=0))[1]
    t12 = pepper_autograd(SpinSystem(S=1.0, g=2.0, lw=[1.0, 0]), exp, Options(Verbosity=0, Transitions=[[1, 2]]))[1]
    assert _rel(base, t12) > 0.4


# ---------------------------------------------------------------- EasySpin

def _load():
    m = loadmat(REF, squeeze_me=True, struct_as_record=False)
    return {str(c.name): c for c in np.atleast_1d(m['cases'])}


def _case_names():
    return list(_load().keys()) if REF.exists() else []


@pytest.fixture(scope='module')
def refs():
    if not REF.exists():
        pytest.skip(f'reference file missing: {REF}')
    return _load()


@pytest.mark.parametrize('path', ['pepper', 'pepper_autograd'])
@pytest.mark.parametrize('name', _case_names())
def test_o3_vs_easyspin(refs, name, path):
    c = refs[name]
    sys = sys_from_mat(c.Sys)
    exp = exp_from_mat(c.Exp, c.Sys, x_axis=np.asarray(c.x, dtype=float))
    opt = opt_from_mat(c.Opt)
    fn = pepper if path == 'pepper' else pepper_autograd
    x, spc = fn(sys, exp, opt)
    x = x.detach().cpu().numpy(); spc = spc.detach().cpu().numpy()
    ref = np.asarray(c.spc, dtype=float); ref_x = np.asarray(c.x, dtype=float)
    assert np.allclose(x, ref_x, rtol=1e-9, atol=1e-9 * abs(ref_x).max())
    cs = cosine(spc, ref)
    amp = abs(spc).max() / abs(ref).max()
    print(f'{name} {path}: cosine {cs:.9f}, amplitude ratio {amp:.6f}')
    assert cs > 0.99999, (name, path, cs)
    assert amp == pytest.approx(1.0, abs=1e-3), (name, path, amp)


# ---------------------------------------------------------------- pepper on the graph

def test_pepper_with_requires_grad_g_is_not_zero():
    """B66 report: pepper with an isotropic g tensor that requires grad."""
    exp = Experiment(mwFreq=9.5, Range=[330, 350], nPoints=512, Harmonic=1)
    ref = pepper(SpinSystem(S=0.5, g=2.0, lw=[1.0, 0.0]), exp)[1]
    g = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    y = pepper(SpinSystem(S=0.5, g=g, lw=[1.0, 0.0]), exp)[1]
    assert y.requires_grad
    assert float(ref.abs().max()) > 700
    assert _rel(ref, y) < 1e-9
    (gr,) = torch.autograd.grad((y * y).sum(), g)
    assert float(gr.abs()) > 0


def test_pepper_with_requires_grad_A_and_lw_equals_plain_pepper():
    exp = Experiment(Harmonic=1, **E_CU)
    plain = pepper(SpinSystem(S=0.5, g=2.1, Nucs='63Cu', A=[300.0], lw=[1.0, 0.4]), exp)[1]
    A = torch.tensor([300.0], dtype=torch.float64, requires_grad=True)
    lw = torch.tensor([1.0, 0.4], dtype=torch.float64, requires_grad=True)
    y = pepper(SpinSystem(S=0.5, g=2.1, Nucs='63Cu', A=A, lw=lw), exp)[1]
    assert y.requires_grad and _rel(plain, y) < 1e-9


# ---------------------------------------------------------------- gradients

def _grad_vs_fd(build, p0, steps, exp, opt=None, tol=1e-6):
    """Gradient of a fixed projection of the spectrum w.r.t. parameter vector
    p0 vs central differences of pepper (the reference model), per entry; the
    smallest error over the steps counts."""
    opt = opt or Options(Verbosity=0)
    p = torch.tensor(p0, dtype=torch.float64, requires_grad=True)
    y = pepper_autograd(build(p), exp, opt)[1]
    w = torch.linspace(-1.0, 1.0, y.numel(), dtype=torch.float64) ** 3 + 0.3
    (gr,) = torch.autograd.grad((y * w).sum(), p)
    for k in range(len(p0)):
        errs = []
        for h in steps[k]:
            q = list(p0); q[k] += h
            fp = (pepper(build(torch.tensor(q, dtype=torch.float64)), exp, opt)[1] * w).sum()
            q = list(p0); q[k] -= h
            fm = (pepper(build(torch.tensor(q, dtype=torch.float64)), exp, opt)[1] * w).sum()
            fd = float((fp - fm) / (2 * h))
            errs.append(abs(float(gr[k]) - fd) / max(abs(fd), 1e-12))
        assert min(errs) < tol, (k, float(gr[k]), errs)
        assert abs(float(gr[k])) > 0


@pytest.mark.parametrize('harmonic', [0, 1])
def test_o3_gradients_g_A_lw_gauss(harmonic):
    exp = Experiment(Harmonic=harmonic, **E_CU)

    def build(p):
        return SpinSystem(S=0.5, g=p[0:1], Nucs='63Cu', A=p[1:2], lw=p[2:4])
    # Voigt widths: Gaussian template, Lorentzian convolved (lw[1] = 0 would be
    # the kink where the Lorentzian part switches on)
    _grad_vs_fd(build, [2.1, 300.0, 1.0, 0.4], [[1e-6, 1e-7], [1e-2, 1e-3], [1e-4, 1e-5], [1e-4, 1e-5]], exp)


def test_o3_gradients_lorentzian_template():
    exp = Experiment(Harmonic=1, **E0)

    def build(p):
        return SpinSystem(S=0.5, g=p[0:1], Nucs='14N', A=p[1:2], lw=torch.stack([torch.zeros((), dtype=torch.float64), p[2]]))
    _grad_vs_fd(build, [2.005, 45.0, 0.5], [[1e-6, 1e-7], [1e-2, 1e-3], [1e-4, 1e-5]], exp)


def test_o3_gradients_hstrain():
    exp = Experiment(Harmonic=1, **E0)

    def build(p):
        return SpinSystem(S=0.5, g=p[0:1], Nucs='1H', A=p[1:2], HStrain=p[2].expand(3))
    _grad_vs_fd(build, [2.0, 30.0, 20.0], [[1e-6, 1e-7], [1e-2, 1e-3], [1e-3, 1e-4]], exp)


# ---------------------------------------------------------------- degenerate grids

def test_empty_triangulation_raises_instead_of_zero(monkeypatch):
    """A projectable grid whose facets come back empty must raise."""
    real = _pep_mod._facets

    def empty(sym, N, theta, ordering=None):
        tri, w = real(sym, N, theta, ordering=ordering)
        return (None, np.zeros(0)) if tri is None else (np.zeros((0, 3), dtype=int), np.zeros(0))
    monkeypatch.setattr(_pep_mod, '_facets', empty)
    sys = SpinSystem(S=0.5, g=[2.0, 2.05, 2.1], lw=[1.0, 0])
    exp = Experiment(mwFreq=9.5, Range=[310, 350], nPoints=256)
    for sym in ('D2h', 'Dinfh', 'Ci'):
        with pytest.raises(RuntimeError, match='no usable facets'):
            pepper_autograd(sys, exp, Options(Verbosity=0, GridSymmetry=sym, GridSize=7))


@pytest.mark.parametrize('sym', ['Dinfh', 'D2h', 'C2h', 'Ci', 'C1'])
def test_smallest_grids_are_not_degenerate(sym):
    """GridSize 2 (the smallest Options accepts) still has facets on every
    projectable grid that Options.GridSymmetry accepts, and pepper_autograd
    matches pepper there."""
    sys = SpinSystem(S=0.5, g=[2.0, 2.05, 2.1], lw=[1.0, 0])
    exp = Experiment(mwFreq=9.5, Range=[310, 350], nPoints=256)
    opt = Options(Verbosity=0, GridSymmetry=sym, GridSize=2)
    yp, ya = _both(sys, exp, opt)
    assert _rel(yp, ya) < 1e-9
