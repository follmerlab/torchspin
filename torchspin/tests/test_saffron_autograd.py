"""saffron on the autograd graph.

With grad tensors in the spin system, saffron returns tensors (signal, info['td'],
info['fd']) and uses time-domain accumulation (exact evolution instead of
frequency binning). Gradients are checked against central finite differences of
saffron itself at a rhombic point with a fixed grid symmetry (at symmetric points
the automatic grid changes under the finite-difference step).
"""
import numpy as np
import pytest
import torch

from torchspin import SpinSystem
from torchspin.saffron import saffron, PulseExperiment, SaffronOptions

torch.set_default_dtype(torch.float64)


def _jac(y, params, step):
    rows = []
    for k in range(0, y.numel(), step):
        e = torch.zeros(y.numel()); e[k] = 1.0
        rows.append(torch.cat([g.reshape(-1) for g in torch.autograd.grad(y, params, grad_outputs=e, retain_graph=True)]))
    return torch.stack(rows)


def _check(ag, fd, tol, label):
    rel = float((ag - fd).norm() / fd.norm())
    assert rel < tol, f'{label}: relative norm error {rel:.2e}'


def test_outputs_numpy_without_grad_and_tensor_with_grad():
    exp = PulseExperiment(Sequence='2pESEEM', Field=350, dt=0.01, nPoints=64, tau=0.1)
    opt = SaffronOptions(GridSize=7)
    _, y, info = saffron(SpinSystem(S=[0.5], g=[[2.0023] * 3], Nucs='1H', A=[[2, 2, 8]]), exp, opt)
    assert isinstance(y, np.ndarray) and isinstance(info['td'], np.ndarray)
    A = torch.tensor([[2.0, 2.0, 8.0]], requires_grad=True)
    _, y, info = saffron(SpinSystem(S=[0.5], g=[[2.0023] * 3], Nucs='1H', A=A), exp, opt)
    assert torch.is_tensor(y) and y.requires_grad and torch.is_tensor(info['fd'])


def test_2pESEEM_gradients_14N():
    exp = PulseExperiment(Sequence='2pESEEM', Field=350, dt=0.01, nPoints=128, tau=0.1)
    opt = SaffronOptions(GridSize=11, GridSymmetry='D2h', TimeDomain=True)   # the differentiable accumulation
    A0 = [[3.2, 2.7, 6.1]]; Q0 = [[-0.6, -0.4, 1.0]]; g0 = [[2.01, 2.03, 2.06]]

    def run(Av, Qv, gv):
        return saffron(SpinSystem(S=[0.5], g=gv, Nucs='14N', A=Av, Q=Qv), exp, opt)[1]
    A = torch.tensor(A0, requires_grad=True); Q = torch.tensor(Q0, requires_grad=True); g = torch.tensor(g0, requires_grad=True)
    yr = run(A, Q, g).real
    J = _jac(yr, [A, Q, g], 2)
    for name, i, col, h in (('A', 0, 0, 1e-4), ('A', 2, 2, 1e-4), ('Q', 0, 3, 1e-4), ('Q', 2, 5, 1e-4), ('g', 0, 6, 1e-6), ('g', 2, 8, 1e-6)):
        def f(d):
            Av = [list(A0[0])]; Qv = [list(Q0[0])]; gv = [list(g0[0])]
            {'A': Av, 'Q': Qv, 'g': gv}[name][0][i] += d
            return torch.as_tensor(np.real(run(Av, Qv, gv)))[::2]
        _check(J[:, col], (f(h) - f(-h)) / (2 * h), 1e-5, f'd/d{name}[{i}]')


def test_hyscore_gradient_A():
    exp = PulseExperiment(Sequence='HYSCORE', Field=350, dt=0.016, nPoints=24, tau=0.1)
    opt = SaffronOptions(GridSize=7, GridSymmetry='D2h', TimeDomain=True)
    A0 = [[2.3, 1.8, 8.4]]

    def run(Av):
        return saffron(SpinSystem(S=[0.5], g=[[2.01, 2.03, 2.06]], Nucs='1H', A=Av), exp, opt)[1]
    A = torch.tensor(A0, requires_grad=True)
    yr = run(A).real.reshape(-1)
    J = _jac(yr, [A], 3)
    h = 1e-4
    for i in (0, 2):
        def f(d):
            Av = [list(A0[0])]; Av[0][i] += d
            return torch.as_tensor(np.real(run(Av))).reshape(-1)[::3]
        _check(J[:, i], (f(h) - f(-h)) / (2 * h), 1e-5, f'HYSCORE d/dA[{i}]')


def test_endor_with_grad_raises():
    A = torch.tensor([[2.0, 2.0, 8.0]], requires_grad=True)
    sys = SpinSystem(S=[0.5], g=[[2.0023] * 3], Nucs='1H', A=A, lwEndor=0.2)
    exp = PulseExperiment(Sequence='MimsENDOR', Field=350, tau=0.1, Range=[5, 25], nPoints=64)
    with pytest.raises(NotImplementedError):
        saffron(sys, exp, SaffronOptions(GridSize=7))
