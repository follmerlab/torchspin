"""curry is pure torch: gradients of the magnetic susceptibility / effective moment
with respect to the exchange coupling and g agree with finite differences."""
import torch

from torchspin import SpinSystem
from torchspin.curry import curry

torch.set_default_dtype(torch.float64)


def _run(ee, g):
    sys = SpinSystem(S=[3.5, 0.5], g=[g, g], ee=ee)
    _, chi = curry(sys, [100.0], [20.0, 50.0, 100.0, 200.0], delta_B=0.01)
    return torch.as_tensor(chi).reshape(-1)


def test_curry_gradients_vs_finite_differences():
    ee = torch.tensor([-2 * 5 * 30e3], requires_grad=True)
    g = torch.tensor([2.0023] * 3, requires_grad=True)
    chi = _run(ee, g)
    assert chi.requires_grad
    J = torch.stack([torch.cat([v.reshape(-1) for v in torch.autograd.grad(chi, [ee, g], grad_outputs=torch.eye(chi.numel())[i], retain_graph=True)])
                     for i in range(chi.numel())])
    h = 10.0
    fd = (_run([-2 * 5 * 30e3 + h], [2.0023] * 3) - _run([-2 * 5 * 30e3 - h], [2.0023] * 3)) / (2 * h)
    assert float((J[:, 0] - fd).norm() / fd.norm()) < 1e-5
    h = 1e-3   # chi comes from a field finite difference; smaller steps amplify its round-off
    fd = (_run([-2 * 5 * 30e3], [2.0023 + h, 2.0023, 2.0023]) - _run([-2 * 5 * 30e3], [2.0023 - h, 2.0023, 2.0023])) / (2 * h)
    assert float((J[:, 1] - fd).norm() / fd.norm()) < 1e-6
