"""Tests for torchspin/levels.py — energy level diagram."""
import math

import pytest
import torch

from torchspin.spinsystem import SpinSystem
from torchspin.levels import levels, _parse_field


# ---------------------------------------------------------------------------
# _parse_field helper
# ---------------------------------------------------------------------------

class TestParseField:
    def test_scalar_list(self):
        B = _parse_field([340.0])
        assert B.shape == (1,)
        assert B.dtype == torch.float64

    def test_array_passed_through(self):
        B_in = torch.linspace(300, 400, 50)
        B_out = _parse_field(B_in)
        assert B_out.shape == (50,)
        assert torch.allclose(B_in.double(), B_out)

    def test_two_element_range_expands_to_101(self):
        B = _parse_field([300.0, 400.0])
        assert B.shape == (101,)
        assert B[0].item() == pytest.approx(300.0)
        assert B[-1].item() == pytest.approx(400.0)

    def test_single_value_tensor(self):
        B = _parse_field(torch.tensor(340.0))
        assert B.shape == (1,)


# ---------------------------------------------------------------------------
# Basic output shape and type
# ---------------------------------------------------------------------------

class TestLevelsShape:
    def setup_method(self):
        self.sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])

    def test_shape_single_orientation_array_B(self):
        B = torch.linspace(0, 500, 50)
        B_ax, E = levels(self.sys, 0.0, math.pi / 2, B)
        assert B_ax.shape == (50,)
        assert E.shape == (2, 50)  # nStates=2, nB=50

    def test_shape_two_element_range(self):
        B_ax, E = levels(self.sys, 0.0, math.pi / 2, [300.0, 400.0])
        assert B_ax.shape == (101,)
        assert E.shape == (2, 101)

    def test_output_dtype_float64(self):
        B = torch.linspace(0, 500, 10)
        B_ax, E = levels(self.sys, 0.0, 0.0, B)
        assert E.dtype == torch.float64
        assert B_ax.dtype == torch.float64

    def test_return_vectors_shape(self):
        B = torch.linspace(0, 500, 10)
        B_ax, E, V = levels(self.sys, 0.0, 0.0, B, return_vectors=True)
        assert E.shape == (2, 10)
        assert V.shape == (2, 2, 10)   # (nStates, nStates, nB)
        assert V.dtype == torch.complex128

    def test_S1_shape(self):
        sys_s1 = SpinSystem(S=[1], g=[[2.0, 2.0, 2.0]])
        B = torch.linspace(0, 600, 20)
        B_ax, E = levels(sys_s1, 0.0, 0.0, B)
        assert E.shape == (3, 20)


# ---------------------------------------------------------------------------
# Eigenvalue properties
# ---------------------------------------------------------------------------

class TestEigenvalueProperties:
    def test_eigenvalues_real(self):
        """Output is real-valued (imag part = 0 by dtype)."""
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        B = torch.linspace(0, 500, 30)
        _, E = levels(sys, 0.0, 0.0, B)
        # Should be finite
        assert torch.isfinite(E).all()

    def test_eigenvalues_sorted_ascending(self):
        """Eigenvalues must be monotonically non-decreasing in state index."""
        sys = SpinSystem(S=[1], g=[[2.0, 2.0, 2.0]], D=[[0, 0, 1000.0]])
        B = torch.linspace(0, 600, 20)
        _, E = levels(sys, 0.0, 0.0, B)
        # E[k, :] >= E[k-1, :] for all k
        for k in range(1, E.shape[0]):
            assert (E[k, :] >= E[k - 1, :] - 1e-9).all(), \
                f"Level {k} not >= level {k-1}"

    def test_eigenvalues_continuous(self):
        """Eigenvalues should not jump discontinuously between adjacent fields.

        Use a fine grid (dB ≈ 0.6 mT) so the maximum expected Zeeman jump
        per step is ~28 MHz/mT × 0.6 mT ≈ 17 MHz, well below the threshold.
        """
        sys = SpinSystem(S=[1], g=[[2.0, 2.0, 2.0]], D=[[0, 0, 1000.0]])
        B = torch.linspace(0, 600, 1000)
        _, E = levels(sys, 0.0, 0.0, B)
        # Max jump between consecutive B points (should be small)
        diffs = (E[:, 1:] - E[:, :-1]).abs()
        assert diffs.max().item() < 25.0, "Unexpectedly large eigenvalue jump"

    def test_b_axis_matches_input(self):
        """B_axis must equal the input B array."""
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        B = torch.linspace(300.0, 400.0, 50)
        B_ax, _ = levels(sys, 0.0, 0.0, B)
        assert torch.allclose(B_ax, B.double())


# ---------------------------------------------------------------------------
# Physical correctness — Zeeman splitting
# ---------------------------------------------------------------------------

class TestPhysics:
    def test_zeeman_splitting_isotropic(self):
        """
        For an isotropic S=1/2 system, the gap between the two levels at
        field B is:  ΔE = g * BMAGN / PLANCK * B  (converted to MHz/mT).

        With g=2.0 and BMAGN/PLANCK = 13.9962... MHz/mT → slope = 27.9925 MHz/mT.
        """
        from torchspin.constants import BMAGN, PLANCK
        g = 2.0
        slope_expected = g * BMAGN / PLANCK * 1e-9  # MHz/mT

        sys = SpinSystem(S=[0.5], g=[[g, g, g]])
        B = torch.tensor([100.0, 200.0, 300.0, 400.0, 500.0])
        _, E = levels(sys, 0.0, 0.0, B)

        gaps = E[1, :] - E[0, :]  # MHz

        # Check each gap is approximately slope * B
        for i, Bval in enumerate(B):
            expected = slope_expected * Bval.item()
            assert abs(gaps[i].item() - expected) < 1.0, (
                f"B={Bval:.0f}mT: gap={gaps[i]:.2f} MHz, expected {expected:.2f} MHz"
            )

    def test_zero_field_degeneracy_isotropic(self):
        """At B=0 with no ZFS, all states are degenerate (energy = 0)."""
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        B = torch.tensor([0.0])
        _, E = levels(sys, 0.0, 0.0, B)
        # Both levels at zero energy (or at least degenerate)
        gap = abs((E[1, 0] - E[0, 0]).item())
        assert gap < 1e-8, f"Non-degenerate at B=0: gap={gap} MHz"

    def test_d_splitting_at_zero_field(self):
        """
        For S=1 with axial ZFS D=1000 MHz (Dzz = 2*D/3 in [Dxx,Dyy,Dzz] form),
        the zero-field splitting is D between the m=0 and m=±1 levels.
        """
        D_val = 1000.0  # MHz
        # Dxx = -D/3, Dyy = -D/3, Dzz = 2*D/3
        Dzz = 2 * D_val / 3
        Dxx = -D_val / 3
        sys = SpinSystem(
            S=[1],
            g=[[2.0, 2.0, 2.0]],
            D=[[Dxx, Dxx, Dzz]],
        )
        B = torch.tensor([0.0])
        _, E = levels(sys, 0.0, 0.0, B)
        # Three levels; gap between middle and top/bottom ≈ D
        E_sorted = E[:, 0].sort().values
        gap_lower = (E_sorted[1] - E_sorted[0]).abs().item()
        gap_upper = (E_sorted[2] - E_sorted[1]).abs().item()
        assert gap_lower < 1.0 or gap_upper < 1.0, (
            "Expected a degenerate pair at zero field"
        )
        max_gap = max(gap_lower, gap_upper)
        assert abs(max_gap - D_val) < 5.0, (
            f"ZFS gap {max_gap:.1f} MHz differs from expected D={D_val} MHz"
        )

    def test_orientation_independence_isotropic(self):
        """
        Isotropic g-tensor: energy levels independent of orientation.
        Two different (phi, theta) pairs should give identical spectra.
        """
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        B = torch.linspace(0, 500, 50)
        _, E1 = levels(sys, 0.0, 0.0, B)
        _, E2 = levels(sys, 1.2, 0.7, B)
        assert torch.allclose(E1, E2, atol=1e-6), "Isotropic levels depend on orientation"

    def test_resonance_at_correct_field(self):
        """
        For S=1/2, g=2.0, mwFreq=9.5 GHz, resonance is at B_res = hν/(g*μ_B).
        levels() crossing at mwFreq*1e3=9500 MHz confirms the resonance field.
        """
        from torchspin.constants import BMAGN, PLANCK
        g = 2.0
        mwFreq_GHz = 9.5
        # Expected: B_res = hν / (g * μ_B) in mT
        B_res_expected = (PLANCK * mwFreq_GHz * 1e9) / (g * BMAGN) * 1e3  # mT

        sys = SpinSystem(S=[0.5], g=[[g, g, g]])
        B = torch.linspace(300.0, 400.0, 1001)
        _, E = levels(sys, 0.0, 0.0, B)

        # Find where E[1]-E[0] = mwFreq*1e3 MHz
        gap = E[1, :] - E[0, :]
        mwFreq_MHz = mwFreq_GHz * 1e3
        # Find closest crossing
        diff = (gap - mwFreq_MHz).abs()
        idx = diff.argmin().item()
        B_res_found = B[idx].item()
        assert abs(B_res_found - B_res_expected) < 1.0, (
            f"Resonance at {B_res_found:.2f} mT, expected {B_res_expected:.2f} mT"
        )


# ---------------------------------------------------------------------------
# Multiple orientations
# ---------------------------------------------------------------------------

class TestMultipleOrientations:
    def test_two_orientations_shape(self):
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        B = torch.linspace(0, 500, 30)
        phi = torch.tensor([0.0, math.pi / 2])
        theta = torch.tensor([0.0, math.pi / 2])
        B_ax, E = levels(sys, phi, theta, B)
        assert E.shape == (2, 2, 30)  # (nOri=2, nStates=2, nB=30)

    def test_anisotropic_orientation_dependence(self):
        """
        For an anisotropic g-tensor, levels at (phi=0, theta=0) vs (phi=0, theta=pi/2)
        should differ.
        """
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.05, 2.15]])
        B = torch.linspace(0, 500, 50)
        phi = torch.tensor([0.0, 0.0])
        theta = torch.tensor([0.0, math.pi / 2])
        _, E = levels(sys, phi, theta, B)
        # Levels at different orientations should differ at high field
        assert not torch.allclose(E[0, :, -1], E[1, :, -1], atol=1.0), \
            "Anisotropic g-tensor should give orientation-dependent levels"

    def test_phi_mismatch_raises(self):
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        B = torch.linspace(0, 500, 10)
        phi = torch.tensor([0.0, 1.0, 2.0])  # 3 values
        theta = torch.tensor([0.0, 1.0])      # 2 values
        with pytest.raises(ValueError, match="same length"):
            levels(sys, phi, theta, B)
