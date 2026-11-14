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
    from the small-argument series branch (|x| < 1e-3).

    Since 0.9.37 (B93) J_n for n >= 3 uses its power series for |x| < n and the
    recurrence above, so J_2, J_3 and J_4 can come from different formulas
    whose values differ by torch's J_0/J_1 accuracy (1.6e-11 at x = 3.3).  The
    reference is therefore built with the formula J_n itself uses: for n = 3 on
    the series branch (x < 3) J_2 and J_4 by the power series (in-test, exact
    to roundoff); on the recurrence branch J_4 = 6/x J_3 - J_2 from the same
    torch values.  The bound stays 1e-12 for every n."""
    x = torch.tensor([0.01, 0.2, 0.9, 1.7, 3.3, 5.0, 8.6, 13.0], dtype=F64, requires_grad=True)
    (g,) = torch.autograd.grad(_besselj_t(n, x).sum(), x)
    xd = x.detach()
    jm = -_besselj_t(1, xd) if n == 0 else _besselj_t(n - 1, xd)
    if n < 3:
        ref = (jm - _besselj_t(n + 1, xd)) / 2
    else:
        def series(k):
            return sum((-1) ** m * (xd / 2) ** (2 * m + k) / float(math.factorial(m) * math.factorial(m + k))
                       for m in range(40))
        on_series = xd.abs() < n
        ref_series = (series(n - 1) - series(n + 1)) / 2
        ref_rec = (jm - (2 * n / xd * _besselj_t(n, xd) - jm)) / 2
        ref = torch.where(on_series, ref_series, ref_rec)
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


# ---------------------------------------------------------------------------
# J_n for n >= 3 (B93, 0.9.37)
#
# Before 0.9.37, _besselj_t used the upward recurrence J_{k+1} = 2k/x J_k - J_{k-1}
# from torch's J_0/J_1 for every |x| >= 1e-3.  It is unstable for |x| < n: J_3
# jumped 6.4 % at the 1e-3 series switch (2.2157e-11 vs 2.0833e-11), J_4(1e-3)
# was 7.9e-9 (true 2.6e-15), and J_6 was off by 0.22 absolute on this grid.
# Now: power series for |x| < n, recurrence only for |x| >= n.
#
# Error floor: torch.special.bessel_j0/j1 themselves are accurate to only about
# 4e-7 absolute (x in 5-10, torch 2.10 and 2.15), which the recurrence branch
# inherits.  Measured on the grid below (both torch versions): max |J_n - jv|
# 3.6e-7 (n = 3) and |J_n' - jvp| 2.4e-7 (n = 5), all in the recurrence branch; on
# the series branch the relative error is <= 1.8e-14 and the derivative error
# <= 1.2e-16 absolute.  Main: relative error 2.2e-2 (n = 3) to 4e21 (n = 6).
# ---------------------------------------------------------------------------
_XGRID = np.logspace(-6.0, np.log10(20.0), 400)        # spans 1e-3 and x = n


@pytest.mark.parametrize('n', [3, 4, 5, 6])
def test_besselj_high_order_vs_scipy(n):
    special = pytest.importorskip('scipy.special')
    v = _besselj_t(n, torch.tensor(_XGRID, dtype=F64)).numpy()
    ref = special.jv(n, _XGRID)
    assert np.abs(v - ref).max() < 1e-6                 # J_0/J_1 floor (3.6e-7)
    lo = _XGRID < n                                     # series branch
    assert (np.abs(v - ref)[lo] / np.abs(ref[lo])).max() < 1e-12


@pytest.mark.parametrize('n', [3, 4, 5, 6])
def test_besselj_high_order_derivative_vs_scipy(n):
    special = pytest.importorskip('scipy.special')
    x = torch.tensor(_XGRID, dtype=F64, requires_grad=True)
    (g,) = torch.autograd.grad(_besselj_t(n, x).sum(), x)
    err = np.abs(g.numpy() - special.jvp(n, _XGRID))
    assert err.max() < 1e-6                             # J_0/J_1 floor (2.4e-7)
    assert err[_XGRID < n].max() < 1e-13                # series branch (1.2e-16)


@pytest.mark.parametrize('n', [3, 4, 5, 6, 8])
def test_besselj_high_order_continuous_at_switches(n):
    """At the series/recurrence switch |x| = n the two branches differ by the
    J_0/J_1 floor only (measured <= 2.7e-7, n = 5; value and derivative), and the
    old 1e-3 switch is gone (main: J_3 jumped 6.4 % there)."""
    special = pytest.importorskip('scipy.special')
    x = torch.tensor([n * (1 - 1e-13), n * (1 + 1e-13)], dtype=F64, requires_grad=True)
    y = _besselj_t(n, x)
    (g,) = torch.autograd.grad(y.sum(), x)
    assert abs(float((y[1] - y[0]).detach())) < 1e-6
    assert abs(float(g[1] - g[0])) < 1e-6
    assert abs(float(y[0].detach()) - special.jv(n, n)) < 1e-12      # series side: 8e-14
    y = _besselj_t(n, torch.tensor([1e-3 * (1 - 1e-12), 1e-3 * (1 + 1e-12)], dtype=F64))
    assert abs(float(y[1] / y[0]) - 1.0) < 1e-10


@pytest.mark.parametrize('n', [3, 4, 5, 6])
def test_besselj_high_order_parity_and_zero(n):
    xs = np.concatenate([[0.0], _XGRID, [25.0, 40.0]])
    x = torch.tensor(xs, dtype=F64, requires_grad=True)
    (g,) = torch.autograd.grad(_besselj_t(n, x).sum(), x)
    y = _besselj_t(n, torch.tensor(xs, dtype=F64))
    ym = _besselj_t(n, torch.tensor(-xs, dtype=F64))
    assert torch.equal(ym, (-1) ** n * y)
    assert float(y[0]) == 0.0 and float(g[0]) == 0.0
    assert bool(torch.isfinite(g).all())


def test_besselj_order_limit():
    special = pytest.importorskip('scipy.special')
    xs = np.linspace(0.0, 60.0, 601)
    v = _besselj_t(30, torch.tensor(xs, dtype=F64)).numpy()
    assert np.abs(v - special.jv(30, xs)).max() < 1e-10     # measured 1.3e-11
    with pytest.raises(ValueError, match='not supported'):
        _besselj_t(31, torch.tensor([1.0], dtype=F64))
    with pytest.raises(ValueError, match='not supported'):
        fieldmod_t(torch.linspace(0.0, 1.0, 11, dtype=F64), torch.ones(11, dtype=F64), 0.2, 31)


def test_fieldmod_t_harmonic3_gradient_vs_converged_central_differences():
    """Harmonic 3 (reachable only by calling fieldmod_t directly; pepper caps
    Harmonic at 2).  The kernel argument runs 0-22, across x = 3.  Measured
    (torch 2.10 and 2.15), five-point CD of fieldmod_t at h = 1e-3, 3e-4, 1e-4:
    -3.2705542042e-4 for all three (spread 1e-10 relative); autograd
    -3.2705543431e-4, 4.2e-8 relative from them (CD differentiates torch's own
    J_0/J_1 forward values, the autograd backward the exact identities; against
    CD of the scipy-based dataproc.fieldmod autograd is 1.9e-9 off).  Main: the
    CD did not converge (-3.27055385e-4, -3.27055403e-4, -3.27055674e-4,
    -3.27060250e-4 at h = 1e-5)."""
    B = torch.linspace(330.0, 350.0, 401, dtype=F64)
    spc = torch.exp(-((B - 339.0) / 0.6) ** 2) + 0.5 * torch.exp(-((B - 343.0) / 0.4) ** 2)
    w = torch.cos(B / 3.0)

    def f(m):
        return (w * fieldmod_t(B, spc, m, 3)).sum()

    m = torch.tensor(0.7, dtype=F64, requires_grad=True)
    (g,) = torch.autograd.grad(f(m), m)
    cds = []
    with torch.no_grad():
        for h in (1e-3, 3e-4, 1e-4):
            cds.append(float((8 * (f(m + h) - f(m - h)) - (f(m + 2 * h) - f(m - 2 * h))) / (12 * h)))
    assert max(cds) - min(cds) <= 1e-8 * abs(cds[-1]), cds   # converged
    assert abs(float(g) - cds[-1]) <= 2e-7 * abs(cds[-1]), (float(g), cds)


# Harmonic 1-2 (pepper_autograd's range); Harmonic 3 above.
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
