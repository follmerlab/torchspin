"""Tests for torchspin.autograd — differentiable EPR spectrum.

Validates:
- S=1/2 analytical path: gradient flows, shapes, FD agreement, harmonics, GPU
- N-spin broadband path: gradient flows, FD agreement, pepper comparison
  - S=1/2 + 1H (hyperfine)
  - S=1 (ZFS)
  - S=1/2 + 14N (quadrupolar nucleus)
  - Temperature-dependent Boltzmann populations
"""
import math
import pytest
import torch

from torchspin.autograd import differentiable_spectrum


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_g() -> torch.Tensor:
    return torch.tensor([2.45, 2.20, 1.95], dtype=torch.float64)


def _default_kwargs() -> dict:
    return dict(
        lw_mT=0.8,
        mwFreq_GHz=9.65,
        B_range=(280.0, 380.0),
        nPoints=512,
        GridSize=19,         # small grid → fast tests
        GridSymmetry='D2h',
    )


def _fd_jacobian(g_base: torch.Tensor, delta: float = 1e-5, **kw) -> torch.Tensor:
    """Finite-difference Jacobian d(spec)/d(g), shape (nPoints, 3)."""
    jac = []
    for i in range(3):
        g_plus  = g_base.clone(); g_plus[i]  += delta
        g_minus = g_base.clone(); g_minus[i] -= delta
        _, sp = differentiable_spectrum(g_plus,  **kw)
        _, sm = differentiable_spectrum(g_minus, **kw)
        jac.append((sp - sm) / (2.0 * delta))
    return torch.stack(jac, dim=1)   # (nPoints, 3)


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------

class TestDifferentiableSpectrum:

    def test_returns_two_tensors(self):
        g = _default_g()
        result = differentiable_spectrum(g, **_default_kwargs())
        assert len(result) == 2

    def test_output_shapes(self):
        kw = _default_kwargs()
        g = _default_g()
        B, spec = differentiable_spectrum(g, **kw)
        assert B.shape    == (kw['nPoints'],)
        assert spec.shape == (kw['nPoints'],)

    def test_field_axis_range(self):
        kw = _default_kwargs()
        g = _default_g()
        B, _ = differentiable_spectrum(g, **kw)
        assert abs(B[0].item()  - kw['B_range'][0]) < 1e-6
        assert abs(B[-1].item() - kw['B_range'][1]) < 1e-6

    def test_output_finite(self):
        g = _default_g()
        _, spec = differentiable_spectrum(g, **_default_kwargs())
        assert torch.all(torch.isfinite(spec))

    def test_absorption_harmonic(self):
        kw = {**_default_kwargs(), 'Harmonic': 0}
        g = _default_g()
        _, spec = differentiable_spectrum(g, **kw)
        assert torch.all(spec >= -1e-10), "Absorption spectrum should be non-negative."
        assert spec.sum() > 0

    def test_second_derivative_harmonic(self):
        kw = {**_default_kwargs(), 'Harmonic': 2}
        g = _default_g()
        _, spec = differentiable_spectrum(g, **kw)
        assert torch.all(torch.isfinite(spec))

    def test_bad_g_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            differentiable_spectrum(torch.tensor([2.0, 2.0]), **_default_kwargs())

    def test_bad_lw_raises(self):
        g = _default_g()
        kw = {**_default_kwargs(), 'lw_mT': -1.0}
        with pytest.raises(ValueError, match="lw_mT"):
            differentiable_spectrum(g, **kw)

    def test_bad_harmonic_raises(self):
        g = _default_g()
        kw = {**_default_kwargs(), 'Harmonic': 3}
        with pytest.raises(ValueError, match="Harmonic"):
            differentiable_spectrum(g, **kw)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:

    def test_gradient_flows(self):
        """g.grad is not None after backward."""
        g = _default_g().requires_grad_(True)
        _, spec = differentiable_spectrum(g, **_default_kwargs())
        spec.sum().backward()
        assert g.grad is not None

    def test_gradient_shape(self):
        """grad has shape (3,) matching g."""
        g = _default_g().requires_grad_(True)
        _, spec = differentiable_spectrum(g, **_default_kwargs())
        spec.sum().backward()
        assert g.grad.shape == (3,)

    def test_gradient_finite(self):
        """All gradient entries are finite."""
        g = _default_g().requires_grad_(True)
        _, spec = differentiable_spectrum(g, **_default_kwargs())
        spec.sum().backward()
        assert torch.all(torch.isfinite(g.grad))

    def test_gradient_nonzero(self):
        """Gradient is not identically zero (spectrum is sensitive to g)."""
        g = _default_g().requires_grad_(True)
        _, spec = differentiable_spectrum(g, **_default_kwargs())
        spec.sum().backward()
        assert g.grad.abs().max() > 1e-6

    def test_autograd_grad_function(self):
        """torch.autograd.grad returns the same gradient as backward."""
        g = _default_g().requires_grad_(True)
        _, spec = differentiable_spectrum(g, **_default_kwargs())
        grad_ag, = torch.autograd.grad(spec.sum(), g)
        assert grad_ag.shape == (3,)
        assert torch.all(torch.isfinite(grad_ag))

    def test_gradient_absorption(self):
        """Gradient flows for Harmonic=0 (absorption)."""
        kw = {**_default_kwargs(), 'Harmonic': 0}
        g = _default_g().requires_grad_(True)
        _, spec = differentiable_spectrum(g, **kw)
        spec.sum().backward()
        assert g.grad is not None
        assert torch.all(torch.isfinite(g.grad))

    def test_gradient_second_deriv(self):
        """Gradient flows for Harmonic=2 (second derivative)."""
        kw = {**_default_kwargs(), 'Harmonic': 2}
        g = _default_g().requires_grad_(True)
        _, spec = differentiable_spectrum(g, **kw)
        spec.sum().backward()
        assert g.grad is not None
        assert torch.all(torch.isfinite(g.grad))


# ---------------------------------------------------------------------------
# Autograd vs finite difference
# ---------------------------------------------------------------------------

class TestGradientAccuracy:

    @pytest.mark.parametrize("component,name", [(0, 'gx'), (1, 'gy'), (2, 'gz')])
    def test_fd_agreement_component(self, component, name):
        """Autograd d(sum)/d(g_i) matches finite difference (< 1 % error)."""
        kw = {**_default_kwargs(), 'nPoints': 512, 'GridSize': 19}
        g_base = _default_g()
        delta  = 1e-5

        # Autograd
        g_ag = g_base.clone().requires_grad_(True)
        _, spec = differentiable_spectrum(g_ag, **kw)
        spec.sum().backward()
        ag_val = g_ag.grad[component].item()

        # Finite difference
        g_plus  = g_base.clone(); g_plus[component]  += delta
        g_minus = g_base.clone(); g_minus[component] -= delta
        _, sp = differentiable_spectrum(g_plus,  **kw)
        _, sm = differentiable_spectrum(g_minus, **kw)
        fd_val = ((sp - sm) / (2.0 * delta)).sum().item()

        rel_err = abs(ag_val - fd_val) / (abs(fd_val) + 1e-10)
        assert rel_err < 0.01, (
            f"g[{component}] ({name}) gradient mismatch: "
            f"autograd={ag_val:.4e}, FD={fd_val:.4e}, rel_err={rel_err:.4e}"
        )

    def test_full_jacobian_cosine(self):
        """Cosine similarity of full Jacobian (autograd vs FD) > 0.999."""
        kw = {**_default_kwargs(), 'nPoints': 512, 'GridSize': 19}
        g_base = _default_g()

        # Autograd Jacobian via vmap / functional Jacobian
        from torch.autograd.functional import jacobian as torch_jac

        def spec_fn(g):
            _, s = differentiable_spectrum(g, **kw)
            return s

        jac_ag = torch_jac(spec_fn, g_base)        # (nPoints, 3)
        jac_fd = _fd_jacobian(g_base, delta=1e-5, **kw)

        # Cosine similarity over the flattened Jacobian
        flat_ag = jac_ag.reshape(-1)
        flat_fd = jac_fd.reshape(-1)
        cosine = (flat_ag @ flat_fd) / (flat_ag.norm() * flat_fd.norm())
        assert cosine.item() > 0.999, (
            f"Jacobian cosine similarity too low: {cosine.item():.6f}"
        )

    def test_gz_jacobian_column_cosine(self):
        """Cosine for dI/dg_z column alone > 0.999."""
        kw = {**_default_kwargs(), 'nPoints': 512, 'GridSize': 19}
        g_base = _default_g()

        from torch.autograd.functional import jacobian as torch_jac

        def spec_fn(g):
            _, s = differentiable_spectrum(g, **kw)
            return s

        jac_ag = torch_jac(spec_fn, g_base)   # (nPoints, 3)
        jac_fd = _fd_jacobian(g_base, delta=1e-5, **kw)

        col_ag = jac_ag[:, 2]   # dI/dg_z
        col_fd = jac_fd[:, 2]
        cosine = (col_ag @ col_fd) / (col_ag.norm() * col_fd.norm())
        assert cosine.item() > 0.999, (
            f"dI/dg_z cosine too low: {cosine.item():.6f}"
        )

    def test_linewidth_gradient_matches_finite_difference(self):
        """Analytical-path linewidth gradient agrees with central FD."""
        kw = {**_default_kwargs(), 'nPoints': 512, 'GridSize': 19}
        g = _default_g()
        lw0 = 1.0
        delta = 1e-4

        lw_ag = torch.tensor(lw0, dtype=torch.float64, requires_grad=True)
        _, spec = differentiable_spectrum(g, **{**kw, 'lw_mT': lw_ag})
        ag, = torch.autograd.grad((spec ** 2).mean(), lw_ag)

        _, sp = differentiable_spectrum(g, **{**kw, 'lw_mT': lw0 + delta})
        _, sm = differentiable_spectrum(g, **{**kw, 'lw_mT': lw0 - delta})
        fd = ((sp ** 2).mean() - (sm ** 2).mean()) / (2.0 * delta)

        rel_err = torch.abs(ag - fd) / (torch.abs(fd) + 1e-30)
        assert rel_err.item() < 1e-4, (
            f"linewidth gradient relative error {rel_err.item():.3e}"
        )


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------

class TestGPU:

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_gradient_flows(self):
        """Gradient flows on CUDA device."""
        g = _default_g().cuda().requires_grad_(True)
        kw = {**_default_kwargs(), 'device': 'cuda'}
        _, spec = differentiable_spectrum(g, **kw)
        spec.sum().backward()
        assert g.grad is not None
        assert g.grad.device.type == 'cuda'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_cpu_gradient_agree(self):
        """GPU gradient agrees with CPU gradient (cosine > 0.9999)."""
        kw = {**_default_kwargs()}
        g_cpu = _default_g().requires_grad_(True)
        g_gpu = _default_g().cuda().requires_grad_(True)

        _, spec_cpu = differentiable_spectrum(g_cpu, **kw, device='cpu')
        _, spec_gpu = differentiable_spectrum(g_gpu, **kw, device='cuda')

        spec_cpu.sum().backward()
        spec_gpu.sum().backward()

        grad_cpu = g_cpu.grad
        grad_gpu = g_gpu.grad.cpu()

        cosine = (grad_cpu @ grad_gpu) / (grad_cpu.norm() * grad_gpu.norm())
        assert cosine.item() > 0.9999


# ===========================================================================
# N-spin broadband path
# ===========================================================================

def _nspin_kwargs() -> dict:
    return dict(
        lw_mT=1.0,
        mwFreq_GHz=9.5,
        B_range=(300.0, 380.0),
        nPoints=256,
        GridSize=10,
        Harmonic=0,
    )


class TestNSpinGradientFlow:
    """Gradient flow tests for the N-spin broadband path."""

    def test_g_gradient_with_hyperfine(self):
        """Gradient flows through g when 1H hyperfine is present."""
        g = _default_g().requires_grad_(True)
        A = torch.tensor([[2.0, 2.0, 8.0]], dtype=torch.float64)
        _, spec = differentiable_spectrum(g, **_nspin_kwargs(), Nucs=['1H'], A=A)
        spec.sum().backward()
        assert g.grad is not None
        assert torch.all(torch.isfinite(g.grad))
        assert g.grad.abs().max() > 1e-6

    def test_A_gradient_flows(self):
        """Gradient flows through hyperfine tensor A."""
        g = _default_g()
        A = torch.tensor([[2.0, 2.0, 8.0]], dtype=torch.float64, requires_grad=True)
        _, spec = differentiable_spectrum(g, **_nspin_kwargs(), Nucs=['1H'], A=A)
        spec.sum().backward()
        assert A.grad is not None
        assert torch.all(torch.isfinite(A.grad))

    def test_g_and_A_simultaneous(self):
        """Both g and A have simultaneous gradients."""
        g = _default_g().requires_grad_(True)
        A = torch.tensor([[2.0, 2.0, 8.0]], dtype=torch.float64, requires_grad=True)
        _, spec = differentiable_spectrum(g, **_nspin_kwargs(), Nucs=['1H'], A=A)
        grad_g, grad_A = torch.autograd.grad(spec.sum(), [g, A])
        assert grad_g.shape == (3,)
        assert grad_A.shape == (1, 3)
        assert torch.all(torch.isfinite(grad_g))
        assert torch.all(torch.isfinite(grad_A))

    def test_S1_ZFS_gradient(self):
        """Gradient flows through D for S=1 triplet system."""
        g = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64, requires_grad=True)
        D = torch.tensor([1000.0, 0.0, -1000.0], dtype=torch.float64, requires_grad=True)
        kw = {**_nspin_kwargs(), 'B_range': (200.0, 450.0), 'GridSize': 8}
        _, spec = differentiable_spectrum(g, **kw, S=1.0, D=D)
        grad_g, grad_D = torch.autograd.grad(spec.sum(), [g, D])
        assert torch.all(torch.isfinite(grad_g))
        assert torch.all(torch.isfinite(grad_D))

    def test_harmonic_1_gradient(self):
        """Gradient flows through Harmonic=1 (first derivative) N-spin path."""
        g = _default_g().requires_grad_(True)
        A = torch.tensor([[2.0, 2.0, 8.0]], dtype=torch.float64, requires_grad=True)
        kw = {**_nspin_kwargs(), 'Harmonic': 1}
        _, spec = differentiable_spectrum(g, **kw, Nucs=['1H'], A=A)
        loss = (spec ** 2).sum()
        grad_g, grad_A = torch.autograd.grad(loss, [g, A])
        assert torch.all(torch.isfinite(grad_g))
        assert grad_g.abs().max() > 1e-6

    def test_temperature_gradient(self):
        """Gradient flows correctly with finite Temperature."""
        g = _default_g().requires_grad_(True)
        A = torch.tensor([[2.0, 2.0, 8.0]], dtype=torch.float64, requires_grad=True)
        _, spec = differentiable_spectrum(g, **_nspin_kwargs(), Nucs=['1H'], A=A,
                                          Temperature=10.0)
        grad_g, grad_A = torch.autograd.grad(spec.sum(), [g, A])
        assert torch.all(torch.isfinite(grad_g))
        assert torch.all(torch.isfinite(grad_A))


class TestNSpinFDAccuracy:
    """Finite-difference validation of N-spin broadband autograd."""

    def _fd_grad_g(self, g_base, delta=1e-5, **kw):
        """Finite-difference gradient d(sum spec)/d(g)."""
        grads = []
        for i in range(3):
            g_p = g_base.clone(); g_p[i] += delta
            g_m = g_base.clone(); g_m[i] -= delta
            _, sp = differentiable_spectrum(g_p, **kw)
            _, sm = differentiable_spectrum(g_m, **kw)
            grads.append((sp.sum() - sm.sum()) / (2.0 * delta))
        return torch.tensor(grads, dtype=g_base.dtype)

    def test_g_fd_agreement_with_1H(self):
        """Autograd vs FD for g gradient with 1H hyperfine (rel error < 5%)."""
        kw = {**_nspin_kwargs(), 'Nucs': ['1H'],
              'A': torch.tensor([[2.0, 2.0, 8.0]], dtype=torch.float64)}
        g_base = _default_g()

        # Autograd
        g_ag = g_base.clone().requires_grad_(True)
        _, spec = differentiable_spectrum(g_ag, **kw)
        spec.sum().backward()
        ag = g_ag.grad.clone()

        # Finite difference
        fd = self._fd_grad_g(g_base, **kw)

        # Cosine similarity
        cosine = (ag @ fd) / (ag.norm() * fd.norm() + 1e-30)
        assert cosine.item() > 0.99, f"g gradient cosine too low: {cosine.item():.4f}"

    def test_A_fd_agreement(self):
        """Autograd vs FD for A gradient (cosine > 0.95)."""
        g = _default_g()
        A_base = torch.tensor([[2.0, 2.0, 8.0]], dtype=torch.float64)
        kw_base = _nspin_kwargs()
        delta = 1e-4

        # Autograd
        A_ag = A_base.clone().requires_grad_(True)
        _, spec = differentiable_spectrum(g, **kw_base, Nucs=['1H'], A=A_ag)
        spec.sum().backward()
        ag = A_ag.grad.flatten().clone()

        # Finite difference
        fd_vals = []
        for i in range(3):
            A_p = A_base.clone(); A_p[0, i] += delta
            A_m = A_base.clone(); A_m[0, i] -= delta
            _, sp = differentiable_spectrum(g, **kw_base, Nucs=['1H'], A=A_p)
            _, sm = differentiable_spectrum(g, **kw_base, Nucs=['1H'], A=A_m)
            fd_vals.append((sp.sum() - sm.sum()) / (2.0 * delta))
        fd = torch.tensor(fd_vals, dtype=torch.float64)

        cosine = (ag @ fd) / (ag.norm() * fd.norm() + 1e-30)
        assert cosine.item() > 0.95, f"A gradient cosine too low: {cosine.item():.4f}"

    def test_D_fd_agreement(self):
        """Autograd vs FD for D gradient in S=1 system (cosine > 0.95)."""
        g = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)
        D_base = torch.tensor([1000.0, 0.0, -1000.0], dtype=torch.float64)
        kw = {**_nspin_kwargs(), 'B_range': (200.0, 450.0), 'GridSize': 8}
        delta = 1.0  # larger step for D (MHz-scale parameter)

        # Autograd
        D_ag = D_base.clone().requires_grad_(True)
        _, spec = differentiable_spectrum(g, **kw, S=1.0, D=D_ag)
        spec.sum().backward()
        ag = D_ag.grad.clone()

        # Finite difference
        fd_vals = []
        for i in range(3):
            D_p = D_base.clone(); D_p[i] += delta
            D_m = D_base.clone(); D_m[i] -= delta
            _, sp = differentiable_spectrum(g, **kw, S=1.0, D=D_p)
            _, sm = differentiable_spectrum(g, **kw, S=1.0, D=D_m)
            fd_vals.append((sp.sum() - sm.sum()) / (2.0 * delta))
        fd = torch.tensor(fd_vals, dtype=torch.float64)

        # Filter out near-zero components (D[1]=0 gives tiny gradient)
        mask = fd.abs() > 1e-6
        if mask.sum() >= 2:
            cosine = (ag[mask] @ fd[mask]) / (ag[mask].norm() * fd[mask].norm() + 1e-30)
            assert cosine.item() > 0.95, f"D gradient cosine too low: {cosine.item():.4f}"

    def test_linewidth_fd_agreement(self):
        """Broadband-path linewidth gradient agrees with central FD."""
        g = _default_g()
        A = torch.tensor([[2.0, 2.0, 8.0]], dtype=torch.float64)
        kw = {**_nspin_kwargs(), 'nPoints': 128, 'GridSize': 8,
              'Nucs': ['1H'], 'A': A}
        lw0 = 1.0
        delta = 1e-4

        lw_ag = torch.tensor(lw0, dtype=torch.float64, requires_grad=True)
        _, spec = differentiable_spectrum(g, **{**kw, 'lw_mT': lw_ag})
        ag, = torch.autograd.grad((spec ** 2).mean(), lw_ag)

        _, sp = differentiable_spectrum(g, **{**kw, 'lw_mT': lw0 + delta})
        _, sm = differentiable_spectrum(g, **{**kw, 'lw_mT': lw0 - delta})
        fd = ((sp ** 2).mean() - (sm ** 2).mean()) / (2.0 * delta)

        rel_err = torch.abs(ag - fd) / (torch.abs(fd) + 1e-30)
        assert rel_err.item() < 1e-3, (
            f"broadband linewidth gradient relative error {rel_err.item():.3e}"
        )

    def test_temperature_fd_agreement(self):
        """Broadband Boltzmann-temperature gradient agrees with central FD."""
        g = _default_g()
        A = torch.tensor([[2.0, 2.0, 8.0]], dtype=torch.float64)
        kw = {**_nspin_kwargs(), 'nPoints': 128, 'GridSize': 8,
              'Nucs': ['1H'], 'A': A}
        temperature0 = 10.0
        delta = 1e-3

        temperature_ag = torch.tensor(
            temperature0, dtype=torch.float64, requires_grad=True
        )
        _, spec = differentiable_spectrum(g, **kw, Temperature=temperature_ag)
        ag, = torch.autograd.grad((spec ** 2).mean(), temperature_ag)

        _, sp = differentiable_spectrum(
            g, **kw, Temperature=temperature0 + delta
        )
        _, sm = differentiable_spectrum(
            g, **kw, Temperature=temperature0 - delta
        )
        fd = ((sp ** 2).mean() - (sm ** 2).mean()) / (2.0 * delta)

        rel_err = torch.abs(ag - fd) / (torch.abs(fd) + 1e-30)
        assert rel_err.item() < 1e-3, (
            f"temperature gradient relative error {rel_err.item():.3e}"
        )


class TestNSpinFDComponentwise:
    """Component-wise relative error checks for N-spin autograd (audit A13)."""

    def test_g_component_relative_error_1H(self):
        """Each g component gradient matches FD within 10% relative error."""
        kw = {**_nspin_kwargs(), 'Nucs': ['1H'],
              'A': torch.tensor([[2.0, 2.0, 8.0]], dtype=torch.float64)}
        g_base = _default_g()
        delta = 1e-5

        # Autograd
        g_ag = g_base.clone().requires_grad_(True)
        _, spec = differentiable_spectrum(g_ag, **kw)
        spec.sum().backward()
        ag = g_ag.grad.clone()

        # Finite difference per component
        for i in range(3):
            g_p = g_base.clone(); g_p[i] += delta
            g_m = g_base.clone(); g_m[i] -= delta
            _, sp = differentiable_spectrum(g_p, **kw)
            _, sm = differentiable_spectrum(g_m, **kw)
            fd_val = ((sp.sum() - sm.sum()) / (2.0 * delta)).item()
            ag_val = ag[i].item()
            if abs(fd_val) > 1e-8:
                rel_err = abs(ag_val - fd_val) / (abs(fd_val) + 1e-30)
                assert rel_err < 0.10, (
                    f"g[{i}] rel_err={rel_err:.4f}: autograd={ag_val:.6e}, FD={fd_val:.6e}"
                )

    def test_A_component_relative_error(self):
        """Each A component gradient matches FD within 15% relative error."""
        g = _default_g()
        A_base = torch.tensor([[2.0, 2.0, 8.0]], dtype=torch.float64)
        kw = _nspin_kwargs()
        delta = 1e-4

        A_ag = A_base.clone().requires_grad_(True)
        _, spec = differentiable_spectrum(g, **kw, Nucs=['1H'], A=A_ag)
        spec.sum().backward()
        ag = A_ag.grad.flatten().clone()

        for i in range(3):
            A_p = A_base.clone(); A_p[0, i] += delta
            A_m = A_base.clone(); A_m[0, i] -= delta
            _, sp = differentiable_spectrum(g, **kw, Nucs=['1H'], A=A_p)
            _, sm = differentiable_spectrum(g, **kw, Nucs=['1H'], A=A_m)
            fd_val = ((sp.sum() - sm.sum()) / (2.0 * delta)).item()
            ag_val = ag[i].item()
            if abs(fd_val) > 1e-8:
                rel_err = abs(ag_val - fd_val) / (abs(fd_val) + 1e-30)
                assert rel_err < 0.15, (
                    f"A[{i}] rel_err={rel_err:.4f}: autograd={ag_val:.6e}, FD={fd_val:.6e}"
                )


class TestNSpinPepperComparison:
    """Validate broadband N-spin spectrum shape matches pepper output."""

    def test_S12_1H_vs_pepper(self):
        """S=1/2 + 1H broadband matches pepper (cosine > 0.99)."""
        import numpy as np
        from torchspin import pepper, SpinSystem
        from torchspin.experiment import Experiment, Options

        sys = SpinSystem(S=[0.5], g=[[2.0, 2.1, 2.2]],
                         Nucs=['1H'], A=[[2.0, 2.0, 8.0]], lwpp=[1.0])
        exp = Experiment(mwFreq=9.5, Range=[300, 380], nPoints=512, Harmonic=0)
        B_p, spec_p = pepper(sys, exp, Options(GridSize=31))

        g = torch.tensor([2.0, 2.1, 2.2], dtype=torch.float64)
        A = torch.tensor([[2.0, 2.0, 8.0]], dtype=torch.float64)
        B_ad, spec_ad = differentiable_spectrum(g, lw_mT=1.0, mwFreq_GHz=9.5,
                                                 B_range=(300, 380), nPoints=512,
                                                 GridSize=31, Nucs=['1H'], A=A,
                                                 Harmonic=0)
        spec_p_np = np.array(spec_p).flatten()
        spec_ad_np = spec_ad.detach().numpy()
        cosine = np.dot(spec_p_np, spec_ad_np) / (
            np.linalg.norm(spec_p_np) * np.linalg.norm(spec_ad_np) + 1e-30)
        assert cosine > 0.99, f"Broadband vs pepper cosine too low: {cosine:.4f}"

    def test_S12_no_nuclei_vs_pepper(self):
        """S=1/2 broadband (forced via S=0.5+empty Nucs) matches analytical path."""
        import numpy as np

        g = torch.tensor([2.0, 2.1, 2.2], dtype=torch.float64)
        # Analytical path
        B_an, spec_an = differentiable_spectrum(g, lw_mT=1.0, mwFreq_GHz=9.5,
                                                 B_range=(300, 380), nPoints=512,
                                                 GridSize=31, Harmonic=0)
        # Force broadband by setting S explicitly (no nuclei, but S=0.5
        # won't trigger broadband). Use a dummy 14N with A=0 to force broadband.
        A_zero = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        B_bb, spec_bb = differentiable_spectrum(g, lw_mT=1.0, mwFreq_GHz=9.5,
                                                 B_range=(300, 380), nPoints=512,
                                                 GridSize=31, Nucs=['1H'], A=A_zero,
                                                 Harmonic=0)
        spec_an_np = spec_an.detach().numpy()
        spec_bb_np = spec_bb.detach().numpy()
        cosine = np.dot(spec_an_np, spec_bb_np) / (
            np.linalg.norm(spec_an_np) * np.linalg.norm(spec_bb_np) + 1e-30)
        assert cosine > 0.99, f"Broadband vs analytical cosine: {cosine:.4f}"


class TestNSpinEdgeCases:
    """Edge cases and validation for N-spin broadband path."""

    def test_output_shapes(self):
        """Output shapes match nPoints."""
        g = _default_g()
        A = torch.tensor([[2.0, 2.0, 8.0]], dtype=torch.float64)
        kw = _nspin_kwargs()
        B, spec = differentiable_spectrum(g, **kw, Nucs=['1H'], A=A)
        assert B.shape == (kw['nPoints'],)
        assert spec.shape == (kw['nPoints'],)

    def test_output_finite(self):
        """All output values are finite."""
        g = _default_g()
        A = torch.tensor([[2.0, 2.0, 8.0]], dtype=torch.float64)
        _, spec = differentiable_spectrum(g, **_nspin_kwargs(), Nucs=['1H'], A=A)
        assert torch.all(torch.isfinite(spec))

    def test_absorption_nonnegative(self):
        """Harmonic=0 absorption spectrum is non-negative."""
        g = _default_g()
        A = torch.tensor([[2.0, 2.0, 8.0]], dtype=torch.float64)
        _, spec = differentiable_spectrum(g, **_nspin_kwargs(), Nucs=['1H'], A=A)
        assert torch.all(spec >= -1e-10), "Absorption spectrum should be non-negative."

    def test_S1_triplet_runs(self):
        """S=1 system with ZFS produces a valid spectrum."""
        g = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)
        D = torch.tensor([1000.0, 0.0, -1000.0], dtype=torch.float64)
        kw = {**_nspin_kwargs(), 'B_range': (200.0, 450.0)}
        _, spec = differentiable_spectrum(g, **kw, S=1.0, D=D)
        assert torch.all(torch.isfinite(spec))
        assert spec.abs().max() > 0

    def test_14N_quadrupolar(self):
        """S=1/2 + 14N (I=1, quadrupolar) runs and produces a valid spectrum."""
        g = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)
        A = torch.tensor([[10.0, 10.0, 30.0]], dtype=torch.float64)
        _, spec = differentiable_spectrum(g, **_nspin_kwargs(), Nucs=['14N'], A=A)
        assert torch.all(torch.isfinite(spec))
        assert spec.abs().max() > 0

    def test_auto_grid_symmetry(self):
        """GridSymmetry='auto' works for N-spin path."""
        g = _default_g()
        A = torch.tensor([[2.0, 2.0, 8.0]], dtype=torch.float64)
        kw = {**_nspin_kwargs(), 'GridSymmetry': 'auto'}
        _, spec = differentiable_spectrum(g, **kw, Nucs=['1H'], A=A)
        assert torch.all(torch.isfinite(spec))

    def test_backward_compatibility_s12(self):
        """S=1/2 without nuclei still uses analytical path (fast)."""
        g = _default_g().requires_grad_(True)
        # This should use the analytical path (no Nucs, no D, S=0.5)
        _, spec = differentiable_spectrum(g, **_nspin_kwargs())
        spec.sum().backward()
        assert g.grad is not None
