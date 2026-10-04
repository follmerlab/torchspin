"""garlic on the autograd graph: torch port of the isotropic/fast-motion path.

Forward values are covered by the MATLAB validation suites (unchanged to 1e-9);
here: gradients against central finite differences of garlic itself, and the
accumulation rules for differentiable spectra.
"""
import pytest
import torch

from torchspin import SpinSystem, Experiment, Options
from torchspin.garlic import garlic

torch.set_default_dtype(torch.float64)


def _jac(y, params, step):
    rows = []
    for k in range(0, y.numel(), step):
        e = torch.zeros(y.numel()); e[k] = 1.0
        rows.append(torch.cat([g.reshape(-1) for g in torch.autograd.grad(y, params, grad_outputs=e, retain_graph=True)]))
    return torch.stack(rows)


def _rel(a, b):
    return float((a - b).norm() / b.norm())


def test_fast_motion_gradients():
    """Nitroxide, Kivelson–Freed widths (explicit accumulation): d spec / d(g, A, tcorr, lwG)."""
    exp = Experiment(mwFreq=9.5, Range=[336, 342], nPoints=512, Harmonic=1)

    def run(gv, Av, tc, lwg):
        return garlic(SpinSystem(S=[0.5], g=gv, Nucs='14N', A=Av, tcorr=tc, lw=[lwg, 0.0]), exp)[1]
    g = torch.tensor([[2.008, 2.006, 2.003]], requires_grad=True)
    A = torch.tensor([[16.0, 16.0, 95.0]], requires_grad=True)
    tc = torch.tensor(1e-10, requires_grad=True)
    lwg = torch.tensor(0.1, requires_grad=True)
    y = run(g, A, tc, lwg)
    assert y.requires_grad
    J = _jac(y, [g, A, tc, lwg], 4)
    base_g, base_A = [2.008, 2.006, 2.003], [16.0, 16.0, 95.0]
    # tcorr: the fine accumulation grid is re-discretised with the smallest width,
    # so the finite difference needs a step that stays inside one discretisation
    cases = (('g', 0, 0, 1e-7, 1e-6), ('g', 2, 2, 1e-7, 1e-6), ('A', 2, 5, 1e-4, 1e-6),
             ('tc', 0, 6, 1e-15, 1e-4), ('lw', 0, 7, 1e-5, 1e-6))
    for name, i, col, h, tol in cases:
        def f(d):
            gv, Av, t, l = list(base_g), list(base_A), 1e-10, 0.1
            if name == 'g': gv[i] += d
            if name == 'A': Av[i] += d
            if name == 'tc': t += d
            if name == 'lw': l += d
            return run([gv], [Av], t, l)[::4]
        fd = (f(h) - f(-h)) / (2 * h)
        assert _rel(J[:, col], fd) < tol, f'd/d{name}[{i}]: {_rel(J[:, col], fd):.2e}'


def test_isotropic_linear_accumulation_gradients():
    """Two protons, Gaussian line, differentiable (two-bin) accumulation: d spec / d(g, A, lw)."""
    exp = Experiment(mwFreq=9.5, Range=[336, 342], nPoints=512, Harmonic=1)
    opt = Options(AccumMethod='linear')

    def run(gv, Av, lwg):
        return garlic(SpinSystem(S=[0.5], g=gv, Nucs='1H,1H', A=Av, lw=[lwg, 0.0]), exp, opt)[1]
    g = torch.tensor([[2.0023] * 3], requires_grad=True)
    A = torch.tensor([[30.0] * 3, [10.0] * 3], requires_grad=True)
    lwg = torch.tensor(0.3, requires_grad=True)
    J = _jac(run(g, A, lwg), [g, A, lwg], 4)
    for name, i, col, h in (('g', 0, 0, 1e-7), ('A', 0, 3, 1e-4), ('lw', 0, 9, 1e-5)):
        def f(d):
            gv, Av, l = [2.0023] * 3, [[30.0] * 3, [10.0] * 3], 0.3
            if name == 'g': gv[i] += d
            if name == 'A': Av[0][i] += d
            if name == 'lw': l += d
            return run([gv], Av, l)[::4]
        fd = (f(h) - f(-h)) / (2 * h)
        assert _rel(J[:, col], fd) < 1e-6, f'd/d{name}[{i}]: {_rel(J[:, col], fd):.2e}'


def test_grad_tensors_switch_binning_to_linear():
    """With grad tensors the default nearest-bin sticks are replaced by the linear
    split (same physics; differs by the sub-bin quantisation of the reference)."""
    exp = Experiment(mwFreq=9.5, Range=[336, 342], nPoints=1024, Harmonic=0)
    A = torch.tensor([[30.0] * 3, [10.0] * 3], requires_grad=True)
    _, y_grad = garlic(SpinSystem(S=[0.5], g=[2.0023] * 3, Nucs='1H,1H', A=A, lw=[0.3, 0.0]), exp)
    _, y_lin = garlic(SpinSystem(S=[0.5], g=[2.0023] * 3, Nucs='1H,1H', A=[[30.0] * 3, [10.0] * 3], lw=[0.3, 0.0]), exp,
                      Options(AccumMethod='linear'))
    _, y_bin = garlic(SpinSystem(S=[0.5], g=[2.0023] * 3, Nucs='1H,1H', A=[[30.0] * 3, [10.0] * 3], lw=[0.3, 0.0]), exp)
    assert y_grad.requires_grad
    assert torch.allclose(y_grad.detach(), y_lin)
    cos = float(y_lin @ y_bin / (y_lin.norm() * y_bin.norm()))
    assert cos > 0.9999


def test_modamp_gradient_flows():
    """Field modulation (Exp.ModAmp) is a torch FFT with a Bessel kernel: on the graph."""
    A = torch.tensor([[50.0] * 3], requires_grad=True)
    sys = SpinSystem(S=[0.5], g=[2.0023] * 3, Nucs='1H', A=A, lw=[0.2, 0.2])
    _, y = garlic(sys, Experiment(mwFreq=9.5, Range=[336, 342], nPoints=256, Harmonic=1, ModAmp=0.1))
    assert y.requires_grad
    (gA,) = torch.autograd.grad((y ** 2).sum(), A)
    assert torch.all(torch.isfinite(gA)) and float(gA.abs().sum()) > 0
