"""salt fixed-field mode for isotropic systems (O3 grid, one orientation; B86, 0.9.36).

Before 0.9.36 a fixed-field system on the O3 grid (a scalar-leaf A keeps O3) went
through the stick paths: ``B_res_all = [B_fixed] * n_orient``, ``.item()`` per line and
nearest-point rounding, then the lw convolution.  The spectrum was detached
(``RuntimeError: element 0 of tensors does not require grad``), and the stick deposit
put each line on the nearest point (forward error 2.6 / 7.8 / 15 % of the maximum at
lw 0.3 / 0.1 / 0.05 MHz on 512 points over 10 MHz).  Now the transitions at the single
orientation are computed on the graph (``_fixed_field_transitions``) and every line is a
bin-integrated Gaussian of width lw at its exact position (pepper's O3 template path);
lw = 0 splits each line linearly between the two nearest points (centered bins, B85).
"""
import numpy as np
import pytest
import torch

from torchspin import Experiment, Options, SpinSystem
from torchspin.salt import salt

_F64 = torch.float64
EXP = Experiment(Field=326.5, mwFreq=9.7)
FR = (10.0, 20.0)


def _sys(A, Q=None, nuc='1H'):
    kw = dict(S=[0.5], g=[2.0], Nucs=nuc, A=A)
    if Q is not None:
        kw['Q'] = Q
    return SpinSystem(**kw)


def _w(x):
    return torch.exp(-((x - 15.2) / 0.8) ** 2)


def _obj(A, n=512, lw=0.3, opt=None, Q=None, nuc='1H', fr=FR):
    x, y = salt(_sys(A, Q, nuc), EXP, opt or Options(), freq_range=fr, n_points=n, lw_mhz=lw)
    return (_w(x) * y).sum()


@pytest.mark.parametrize('lw', [0.3, 0.05])
def test_o3_gradient_vs_converged_central_differences(lw):
    """Scalar leaf a, A = a [1 1 1] (grid O3).  0.9.35: RuntimeError (detached
    spectrum).  Measured: autograd vs central differences 3e-11 (h 1e-5)."""
    a = torch.tensor(3.0, dtype=_F64, requires_grad=True)
    (g,) = torch.autograd.grad(_obj(a * torch.ones(1, 3, dtype=_F64), lw=lw), a)

    def f(v):
        return float(_obj(torch.full((1, 3), v, dtype=_F64), lw=lw))
    cds = [(f(3 + h) - f(3 - h)) / (2 * h) for h in (1e-4, 1e-5)]
    print(f'lw {lw}: autograd {float(g):.10e}, CD {cds}')
    assert abs(cds[0] - cds[1]) < 1e-6 * abs(cds[1])           # converged
    assert abs(float(g) - cds[1]) < 1e-8 * abs(cds[1])


def test_o3_matrix_path_gradient():
    """Options.Enhancement (matrix path) on O3: gradient w.r.t. an isotropic A leaf vs
    central differences.  (0.9.35 passed this too: its matrix stick path kept the
    intensities on the graph and the rounded positions made the spectrum piecewise
    constant in them; now the positions are differentiated as well.)"""
    a = torch.tensor(3.0, dtype=_F64, requires_grad=True)
    opt = Options(Enhancement=True)
    (g,) = torch.autograd.grad(_obj(a * torch.ones(1, 3, dtype=_F64), opt=opt), a)

    def f(v):
        return float(_obj(torch.full((1, 3), v, dtype=_F64), opt=opt))
    cds = [(f(3 + h) - f(3 - h)) / (2 * h) for h in (1e-4, 1e-5)]
    print(f'Enhancement: autograd {float(g):.10e}, CD {cds}')
    assert abs(cds[0] - cds[1]) < 1e-5 * abs(cds[1])
    assert abs(float(g) - cds[1]) < 1e-6 * abs(cds[1])


@pytest.mark.parametrize('n,bound', [(512, 5e-2), (2048, 1e-2)])
def test_three_leaf_d2h_gradient_matches_o3(n, bound):
    """Three A leaves (auto grid D2h, the projection path) against the scalar leaf (O3):
    the sum of the three gradients equals the O3 gradient up to the projection's
    discretization (a point mass split linearly: chord slope, first order in dx).
    Measured 4.1e-2 (512 points, dx/sigma = 0.15) and 4.2e-3 (2048 points).
    0.9.34: D2h ~1e-23 (truncated sub-bin facets, B85) and O3 raised; 0.9.35: O3 raised."""
    leaves = [torch.tensor(3.0, dtype=_F64, requires_grad=True) for _ in range(3)]
    gs = torch.autograd.grad(_obj(torch.stack(leaves).reshape(1, 3), n=n), leaves)
    a = torch.tensor(3.0, dtype=_F64, requires_grad=True)
    (go,) = torch.autograd.grad(_obj(a * torch.ones(1, 3, dtype=_F64), n=n), a)
    tot, go = float(sum(gs)), float(go)
    print(f'n={n}: three leaves (D2h) {tot:.10e}, O3 {go:.10e}, rel {abs(tot / go - 1):.2e}')
    assert abs(tot / go - 1) < bound


@pytest.mark.parametrize('lw,bound', [(0.3, 3e-3), (0.1, 1.5e-2), (0.05, 5e-2)])
def test_o3_forward_vs_converged(lw, bound):
    """512 points over 10 MHz against the same spectrum on a 32x finer axis (sampled
    every 32nd point, per-bin scale x32).  Measured 9.8e-4 / 8.8e-3 / 3.4e-2 of the
    maximum (lw 0.3 / 0.1 / 0.05 MHz); 0.9.35 (stick deposit + convolution): 2.6e-2 /
    7.8e-2 / 1.6e-1."""
    A = [[3.0, 3.0, 3.0]]
    x, y = salt(_sys(A), EXP, Options(), freq_range=FR, n_points=512, lw_mhz=lw)
    xf, yf = salt(_sys(A), EXP, Options(), freq_range=FR, n_points=511 * 32 + 1, lw_mhz=lw)
    ref = yf.numpy()[::32] * 32
    err = np.abs(y.numpy() - ref).max() / np.abs(ref).max()
    print(f'lw {lw}: max error {err:.3e}')
    assert err < bound


def test_o3_zero_lw_linear_split():
    """lw = 0: each line split linearly between its two nearest points; the
    spectrum's centroid is the intensity-weighted mean line position."""
    a = torch.tensor(3.0, dtype=_F64, requires_grad=True)
    x, y = salt(_sys(a * torch.ones(1, 3, dtype=_F64)), EXP, Options(), freq_range=FR, n_points=512, lw_mhz=0.0)
    yd = y.detach().numpy()
    nz = np.nonzero(yd)[0]
    assert nz.size == 4                                        # two lines, two points each
    xs = x.numpy()
    x1 = (xs[nz[:2]] * yd[nz[:2]]).sum() / yd[nz[:2]].sum()
    x2 = (xs[nz[2:]] * yd[nz[2:]]).sum() / yd[nz[2:]].sum()
    from torchspin.constants import NMAGN, PLANCK
    from torchspin.nucdata import nucgval
    nu = float(nucgval('1H')) * NMAGN * 326.5e-3 / PLANCK * 1e-6
    assert abs(x1 - (nu - 1.5)) < 1e-9 and abs(x2 - (nu + 1.5)) < 1e-9
    (g,) = torch.autograd.grad((torch.arange(512, dtype=_F64) * y).sum(), a)
    assert abs(float(g)) > 0


def test_o3_lw_gradient():
    lw = torch.tensor(0.3, dtype=_F64, requires_grad=True)
    (g,) = torch.autograd.grad(_obj([[3.0, 3.0, 3.0]], lw=lw), lw)
    cd = (float(_obj([[3.0, 3.0, 3.0]], lw=0.3 + 1e-6)) - float(_obj([[3.0, 3.0, 3.0]], lw=0.3 - 1e-6))) / 2e-6
    assert abs(float(g) - cd) < 1e-6 * abs(cd)
