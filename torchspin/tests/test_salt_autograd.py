"""salt (powder ENDOR) on the autograd graph: the fixed-field path uses the torch
interpolation / projection / convolution shared with pepper_autograd, so spectra are
differentiable in the spin-system tensors. Forward values are covered by
test_salt_matlab_validation.py; here: gradients vs central finite differences.
"""
import pytest
import torch

from torchspin import SpinSystem, Experiment, Options
from torchspin.salt import salt

torch.set_default_dtype(torch.float64)


def _jac(y, params, step):
    rows = []
    for k in range(0, y.numel(), step):
        e = torch.zeros(y.numel()); e[k] = 1.0
        rows.append(torch.cat([g.reshape(-1) for g in torch.autograd.grad(y, params, grad_outputs=e, retain_graph=True)]))
    return torch.stack(rows)


def _check(ag, fd, tol):
    rel = float((ag - fd).norm() / fd.norm())
    assert rel < tol, f'relative norm error {rel:.2e} (tol {tol})'


def test_perturbative_path_gradients():
    """1H, rhombic g, GridSize [20,5]: d spec / d(g, A, lw)."""
    exp = Experiment(Field=326.5, mwFreq=9.7); opt = Options(GridSize=[20, 5])

    def run(gv, Av, lw):
        sys = SpinSystem(S=[0.5], g=gv if isinstance(gv, torch.Tensor) else [gv], Nucs='1H',
                         A=Av if isinstance(Av, torch.Tensor) else [Av])
        return salt(sys, exp, opt, freq_range=(10, 20), n_points=512, lw_mhz=lw)[1]
    g = torch.tensor([[2.0, 2.01, 2.02]], requires_grad=True)
    A = torch.tensor([[-2.0, -2.0, 4.0]], requires_grad=True)
    lw = torch.tensor(0.1, requires_grad=True)
    y = run(g, A, lw)
    assert y.requires_grad
    J = _jac(y, [g, A, lw], 4)
    base_g, base_A = [2.0, 2.01, 2.02], [-2.0, -2.0, 4.0]
    for name, i, col, h, tol in (('g', 2, 2, 1e-7, 1e-3), ('A', 0, 3, 1e-5, 1e-5), ('A', 2, 5, 1e-5, 1e-5), ('lw', 0, 6, 1e-5, 1e-6)):
        def f(d):
            gv, Av, lwv = list(base_g), list(base_A), 0.1
            if name == 'g': gv[i] += d
            if name == 'A': Av[i] += d
            if name == 'lw': lwv += d
            return run(gv, Av, lwv)[::4]
        _check(J[:, col], (f(h) - f(-h)) / (2 * h), tol)


def test_matrix_path_gradients():
    """14N with quadrupole (matrix diagonalization), fixed D2h grid: d spec / d(A, Q)."""
    exp = Experiment(Field=3394, mwFreq=95); opt = Options(GridSize=19, GridSymmetry='D2h')

    def run(Av, Qv):
        sys = SpinSystem(S=[0.5], g=[[2.2, 2.2, 2.0]], Nucs='14N', A=Av, Q=Qv)
        return salt(sys, exp, opt, freq_range=(0, 35), n_points=512, lw_mhz=0.1)[1]
    A = torch.tensor([[4.0, 4.0, 5.0]], requires_grad=True)
    Q = torch.tensor([[0.84, 0.84, -1.68]], requires_grad=True)
    y = run(A, Q)
    assert y.requires_grad
    J = _jac(y, [A, Q], 4)
    for name, i, col in (('A', 2, 2), ('Q', 0, 3), ('Q', 2, 5)):
        def f(d):
            Av, Qv = [[4.0, 4.0, 5.0]], [[0.84, 0.84, -1.68]]
            (Av if name == 'A' else Qv)[0][i] += d
            return run(Av, Qv)[::4]
        h = 1e-5
        _check(J[:, col], (f(h) - f(-h)) / (2 * h), 1e-4)


def test_grad_tensor_in_field_swept_mode_raises():
    """The field-swept (multi-B) ENDOR mode still bins sticks; a grad tensor must not be silently detached."""
    A = torch.tensor([[-2.0, -2.0, 4.0]], requires_grad=True)
    sys = SpinSystem(S=[0.5], g=[[2.0, 2.01, 2.02]], Nucs='1H', A=A)
    exp = Experiment(mwFreq=9.7, Range=[320, 340])
    with pytest.raises((RuntimeError, NotImplementedError)):
        salt(sys, exp, Options(GridSize=7), freq_range=(10, 20), n_points=128)
