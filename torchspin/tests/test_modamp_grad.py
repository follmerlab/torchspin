"""Exp.ModAmp gradients on every torch version (B92, 0.9.27).

The pseudo-modulation kernel (dataproc.fieldmod_t) is a Bessel function of
ModAmp.  ``torch.special.bessel_j0``/``bessel_j1`` have no derivative before
torch 2.15: their outputs silently leave the graph.  So a grad ModAmp raised
"does not require grad" at Harmonic 1 on torch <= 2.14, and at Harmonic 2 the
upward recurrence J_{k+1} = 2k/x J_k - J_{k-1} kept only its 2k/x path: a
silently wrong gradient (S = 1/2 rhombic g, lw 0.5, ModAmp 0.5, sum(y^2):
autograd -90.59 vs central differences 10.004 on torch 2.10 and 2.14.1; torch
2.15 correct).  dataproc._bessel_j01 now supplies the backward itself
(J0' = -J1, J1' = J0 - J1/x, 1/2 at x = 0), identically on every version.
"""
import math

import numpy as np
import pytest
import torch

from torchspin.dataproc import _bessel_j01, _besselj_t, fieldmod_t
from torchspin.experiment import Experiment, Options
from torchspin.pepper import pepper
from torchspin.pepper_autograd import pepper_autograd
from torchspin.spinsystem import SpinSystem

F64 = torch.float64


# ---------------------------------------------------------------------------
# Bessel kernel
# ---------------------------------------------------------------------------
def test_bessel_j01_is_on_the_graph():
    x = torch.tensor([0.3, 2.0], dtype=F64, requires_grad=True)
    j0, j1 = _bessel_j01(x)
    assert j0.requires_grad and j1.requires_grad
    # forward values are torch's own
    assert torch.equal(j0.detach(), torch.special.bessel_j0(x.detach()))
    assert torch.equal(j1.detach(), torch.special.bessel_j1(x.detach()))


@pytest.mark.parametrize('n', [0, 1, 2, 3])
def test_besselj_derivative_identity(n):
    """d/dx J_n = (J_{n-1} - J_{n+1}) / 2 (J_{-1} = -J_1), evaluated with the same
    torch J_n: equal up to roundoff of the recurrence (measured <= 3e-14), away
    from the small-argument series branch (|x| < 1e-3)."""
    x = torch.tensor([0.01, 0.2, 0.9, 1.7, 3.3, 5.0, 8.6, 13.0], dtype=F64, requires_grad=True)
    (g,) = torch.autograd.grad(_besselj_t(n, x).sum(), x)
    xd = x.detach()
    jm = -_besselj_t(1, xd) if n == 0 else _besselj_t(n - 1, xd)
    ref = (jm - _besselj_t(n + 1, xd)) / 2
    assert float((g - ref).abs().max()) < 1e-12


@pytest.mark.parametrize('n', [0, 1, 2, 3])
def test_besselj_derivative_vs_scipy(n):
    """Against scipy.special.jvp: bounded by the accuracy of torch's own
    bessel_j0/j1 forward values (about 4e-7 absolute on torch 2.10-2.15)."""
    special = pytest.importorskip('scipy.special')
    xs = np.array([0.0, 0.01, 0.2, 0.9, 1.7, 3.3, 5.0, 8.6, 13.0])
    x = torch.tensor(xs, dtype=F64, requires_grad=True)
    (g,) = torch.autograd.grad(_besselj_t(n, x).sum(), x)
    assert np.abs(g.numpy() - special.jvp(n, xs)).max() < 2e-6


def test_besselj_derivative_at_zero():
    x = torch.zeros(1, dtype=F64, requires_grad=True)
    (g0,) = torch.autograd.grad(_bessel_j01(x)[0].sum(), x)
    (g1,) = torch.autograd.grad(_bessel_j01(x)[1].sum(), x)
    assert float(g0) == 0.0 and float(g1) == 0.5


# Harmonic 1-2 (pepper_autograd's range).  Not 3: _besselj_t's upward recurrence
# loses precision at small x for n >= 3 (a 1.4 % jump at its 1e-3 series switch,
# absolute 3e-13), so central differences of the H3 kernel do not converge.
@pytest.mark.parametrize('H', [1, 2])
def test_fieldmod_t_gradient_vs_central_differences(H):
    B = torch.linspace(330.0, 350.0, 401, dtype=F64)
    spc = torch.exp(-((B - 339.0) / 0.6) ** 2) + 0.5 * torch.exp(-((B - 343.0) / 0.4) ** 2)
    w = torch.cos(B / 3.0)

    def f(m):
        return (w * fieldmod_t(B, spc, m, H)).sum()

    m = torch.tensor(0.7, dtype=F64, requires_grad=True)
    (g,) = torch.autograd.grad(f(m), m)
    h = 1e-3
    with torch.no_grad():
        cd = (8 * (f(m + h) - f(m - h)) - (f(m + 2 * h) - f(m - 2 * h))) / (12 * h)
    assert abs(float(g) - float(cd)) <= 1e-8 * abs(float(cd))


# ---------------------------------------------------------------------------
# pepper_autograd
# ---------------------------------------------------------------------------
_SYS = dict(S=0.5, g=[2.0, 2.05, 2.1], lw=0.5)


def _loss(m, H, **exp_kw):
    e = Experiment(mwFreq=9.5, Range=[300.0, 360.0], nPoints=256, Harmonic=H, ModAmp=m, **exp_kw)
    return (pepper_autograd(SpinSystem(**_SYS), e, Options(GridSize=[19, 4]))[1] ** 2).sum()


@pytest.mark.parametrize('H,m0,exp_kw', [
    (1, 0.5, {}),
    (2, 0.5, {}),                       # was silently wrong on torch <= 2.14
    (1, 0.05, {}),                      # below the bin width (0.235 mT)
    (1, 5.0, {}),                       # overmodulated
    (1, 0.5, dict(Orientations=[[0.0, 0.0], [0.5, 0.7], [1.0, 1.2]])),
    (2, 0.5, dict(Orientations=[[0.0, 0.0], [0.5, 0.7], [1.0, 1.2]])),
], ids=['H1', 'H2', 'H1-small', 'H1-overmod', 'H1-orientations', 'H2-orientations'])
def test_pepper_autograd_modamp_gradient(H, m0, exp_kw):
    m = torch.tensor(m0, dtype=F64, requires_grad=True)
    (g,) = torch.autograd.grad(_loss(m, H, **exp_kw), m)
    h = 1e-4 * m0
    with torch.no_grad():
        f = lambda d: float(_loss(torch.tensor(m0 + d, dtype=F64), H, **exp_kw))
        cd = (8 * (f(h) - f(-h)) - (f(2 * h) - f(-2 * h))) / (12 * h)
    assert abs(float(g) - cd) <= 1e-6 * abs(cd), (float(g), cd)


def test_pepper_with_grad_modamp_matches_plain_pepper():
    e0 = Experiment(mwFreq=9.5, Range=[300.0, 360.0], nPoints=256, Harmonic=2, ModAmp=0.5)
    _, y0 = pepper(SpinSystem(**_SYS), e0, Options(GridSize=[19, 4]))
    m = torch.tensor(0.5, dtype=F64, requires_grad=True)
    e = Experiment(mwFreq=9.5, Range=[300.0, 360.0], nPoints=256, Harmonic=2, ModAmp=m)
    _, y = pepper(SpinSystem(**_SYS), e, Options(GridSize=[19, 4]))
    assert y.requires_grad
    assert float((y.detach() - y0).abs().max() / y0.abs().max()) < 1e-11


def test_grad_modamp_zero_raises():
    m = torch.tensor(0.0, dtype=F64, requires_grad=True)
    with pytest.raises(ValueError, match='ModAmp requires grad'):
        _loss(m, 1)
