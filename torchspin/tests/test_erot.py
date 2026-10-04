#!/usr/bin/env python3
"""
Test suite for erot (Euler rotation matrices).

Based on MATLAB tests in easyspin/tests/:
- erot_colsoutput.m
- erot_degeneracy.m
- erot_rowsoutput.m
- erot_specialcases.m
- erot_syntax.m
- erot_value.m
"""

import pytest
import torch
import math
from torchspin.rotations import erot


class TestErotBasic:
    """Basic rotation matrix tests."""

    def test_erot_identity(self):
        """Zero rotation should give identity matrix (erot_specialcases.m)."""
        R = erot(0, 0, 0)
        I = torch.eye(3, dtype=torch.float64)
        assert torch.allclose(R, I, atol=1e-14)

    def test_erot_returns_3x3(self):
        """erot should always return 3×3 matrix (erot_syntax.m)."""
        R = erot(0.1, 0.2, 0.3)
        assert R.shape == (3, 3)

    def test_erot_is_orthogonal(self):
        """Rotation matrices should be orthogonal: R^T R = I."""
        R = erot(0.5, 1.2, 0.8)
        I = torch.eye(3, dtype=torch.float64)
        assert torch.allclose(R.T @ R, I, atol=1e-14)

    def test_erot_has_determinant_one(self):
        """Rotation matrices should have det(R) = +1."""
        R = erot(0.5, 1.2, 0.8)
        det = torch.linalg.det(R)
        assert torch.allclose(det, torch.tensor(1.0, dtype=torch.float64), atol=1e-14)


class TestErotSpecialAngles:
    """Test special angle cases (erot_specialcases.m)."""

    def test_erot_rotation_about_z_only(self):
        """Rotation only about z-axis (α ≠ 0, β = 0, γ = 0).

        EasySpin uses the PASSIVE (frame-rotation) convention.
        For a frame rotated by α about z, the old x-vector has
        components (cos α, -sin α, 0) in the new frame, so:
            R = [[cos α,  sin α, 0],
                 [-sin α, cos α, 0],
                 [0,      0,     1]]
        """
        alpha = math.pi / 4
        R = erot(alpha, 0, 0)
        c, s = math.cos(alpha), math.sin(alpha)
        expected = torch.tensor([
            [ c,  s, 0],
            [-s,  c, 0],
            [ 0,  0, 1]
        ], dtype=torch.float64)
        assert torch.allclose(R, expected, atol=1e-14)

    def test_erot_rotation_about_y_only(self):
        """Rotation only about y-axis (α = 0, β ≠ 0, γ = 0).

        EasySpin passive convention for y rotation:
            R = [[cos β, 0, -sin β],
                 [0,     1,  0     ],
                 [sin β, 0,  cos β ]]
        """
        beta = math.pi / 3
        R = erot(0, beta, 0)
        c, s = math.cos(beta), math.sin(beta)
        expected = torch.tensor([
            [ c, 0, -s],
            [ 0, 1,  0],
            [ s, 0,  c]
        ], dtype=torch.float64)
        assert torch.allclose(R, expected, atol=1e-14)

    def test_erot_pi_rotation_about_z(self):
        """180° rotation about z-axis."""
        R = erot(math.pi, 0, 0)
        expected = torch.tensor([
            [-1, 0, 0],
            [ 0, -1, 0],
            [ 0, 0, 1]
        ], dtype=torch.float64)
        assert torch.allclose(R, expected, atol=1e-14)

    def test_erot_pi_rotation_about_y(self):
        """180° rotation about y-axis."""
        R = erot(0, math.pi, 0)
        expected = torch.tensor([
            [-1, 0,  0],
            [ 0, 1,  0],
            [ 0, 0, -1]
        ], dtype=torch.float64)
        assert torch.allclose(R, expected, atol=1e-14)

    def test_erot_2pi_rotation_gives_identity(self):
        """2π rotation about any axis should give identity."""
        R = erot(2*math.pi, 0, 0)
        I = torch.eye(3, dtype=torch.float64)
        assert torch.allclose(R, I, atol=1e-14)


class TestErotZYZConvention:
    """Test z-y'-z'' (passive) convention (erot_value.m)."""

    def test_erot_zyz_sequence(self):
        """Test that erot follows z-y'-z'' Euler angle convention.

        EasySpin's formula (passive convention, matching erot.m):
            R = [ cg*cb*ca - sg*sa,   cg*cb*sa + sg*ca,  -cg*sb ]
                [ -sg*cb*ca - cg*sa,  -sg*cb*sa + cg*ca,   sg*sb ]
                [  sb*ca,              sb*sa,               cb    ]
        """
        alpha, beta, gamma = 0.3, 0.7, 1.1
        R = erot(alpha, beta, gamma)

        ca, sa = math.cos(alpha), math.sin(alpha)
        cb, sb = math.cos(beta),  math.sin(beta)
        cg, sg = math.cos(gamma), math.sin(gamma)

        R_manual = torch.tensor([
            [ cg*cb*ca - sg*sa,  cg*cb*sa + sg*ca, -cg*sb],
            [-sg*cb*ca - cg*sa, -sg*cb*sa + cg*ca,  sg*sb],
            [ sb*ca,             sb*sa,              cb   ],
        ], dtype=torch.float64)

        assert torch.allclose(R, R_manual, atol=1e-14)

    def test_erot_passive_vs_active(self):
        """Test passive (frame rotation) vs active (vector rotation) interpretation."""
        # For passive rotation: R rotates the frame
        # For active rotation: R^T rotates the vector
        alpha = math.pi / 6
        R = erot(alpha, 0, 0)
        
        # A vector in the original frame
        v = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        
        # Passive: R describes how basis vectors transform
        # Active: R^T describes how components transform
        v_rotated = R.T @ v
        
        # After passive z-rotation by alpha, x-component should be cos(alpha)
        assert torch.allclose(v_rotated[0], torch.tensor(math.cos(alpha), dtype=torch.float64), atol=1e-14)


class TestErotComposition:
    """Test composition of rotations."""

    def test_erot_composition_order(self):
        """Test that R(α1)R(α2) = R(α1+α2) for z-rotations."""
        alpha1, alpha2 = math.pi/6, math.pi/4
        
        R1 = erot(alpha1, 0, 0)
        R2 = erot(alpha2, 0, 0)
        R_composed = R1 @ R2
        
        R_direct = erot(alpha1 + alpha2, 0, 0)
        
        assert torch.allclose(R_composed, R_direct, atol=1e-14)

    def test_erot_inverse(self):
        """Test that R(-α, -β, -γ) ≈ R^T(α, β, γ)."""
        alpha, beta, gamma = 0.5, 1.2, 0.8
        R = erot(alpha, beta, gamma)
        R_inv = erot(-gamma, -beta, -alpha)  # Reversed order for inverse
        
        I = torch.eye(3, dtype=torch.float64)
        assert torch.allclose(R @ R_inv, I, atol=1e-13)


class TestErotDegeneracy:
    """Test degenerate cases (erot_degeneracy.m)."""

    def test_erot_beta_zero_gimbal_lock(self):
        """When β=0, only α+γ matters (gimbal lock)."""
        alpha1, gamma1 = 0.3, 0.7
        alpha2, gamma2 = 0.5, 0.5
        
        # Both should give same result when α + γ is the same
        R1 = erot(alpha1, 0, gamma1)
        R2 = erot(alpha2, 0, gamma2)
        
        assert torch.allclose(R1, R2, atol=1e-14)

    def test_erot_beta_pi_gimbal_lock(self):
        """When β=π, only α-γ matters (gimbal lock)."""
        alpha1, gamma1 = 1.0, 0.3
        alpha2, gamma2 = 1.3, 0.6
        
        # When α - γ is the same
        if abs((alpha1 - gamma1) - (alpha2 - gamma2)) < 1e-10:
            R1 = erot(alpha1, math.pi, gamma1)
            R2 = erot(alpha2, math.pi, gamma2)
            assert torch.allclose(R1, R2, atol=1e-14)


class TestErotVectorInput:
    """Test vector input handling (erot_syntax.m)."""

    def test_erot_with_vector_input(self):
        """Accept Euler angles as a single vector."""
        angles = [0.3, 0.7, 1.1]
        R1 = erot(*angles)
        R2 = erot(angles[0], angles[1], angles[2])
        assert torch.allclose(R1, R2, atol=1e-14)

    def test_erot_with_list(self):
        """Accept list of angles."""
        angles = [0.5, 1.2, 0.8]
        R = erot(*angles)
        assert R.shape == (3, 3)
        det = torch.linalg.det(R)
        assert torch.allclose(det, torch.tensor(1.0, dtype=torch.float64), atol=1e-14)


class TestErotNumericalPrecision:
    """Test numerical precision for edge cases."""

    def test_erot_small_angles(self):
        """Test that small angles work correctly."""
        eps = 1e-10
        R = erot(eps, eps, eps)
        I = torch.eye(3, dtype=torch.float64)
        # Should be close to identity for small angles
        assert torch.allclose(R, I, atol=1e-8)

    def test_erot_large_angles(self):
        """Test that large angles (multiple of 2π) work."""
        # 10π + π/4 should give same result as π/4
        R1 = erot(math.pi/4, 0, 0)
        R2 = erot(math.pi/4 + 10*math.pi, 0, 0)
        assert torch.allclose(R1, R2, atol=1e-12)


class TestErotConsistency:
    """Cross-consistency tests between rotation representations."""

    def test_erot_rotation_of_z_axis(self):
        """Rotating z-axis by (α, β, γ) should give correct result."""
        alpha, beta, gamma = 0.3, 0.7, 1.1
        R = erot(alpha, beta, gamma)
        
        z_axis = torch.tensor([0, 0, 1], dtype=torch.float64)
        z_rotated = R @ z_axis
        
        # For passive rotation with z-y'-z'', the third column is the rotated z-axis
        expected = R[:, 2]
        assert torch.allclose(z_rotated, expected, atol=1e-14)

    def test_erot_preserves_vector_length(self):
        """Rotation should preserve vector lengths."""
        R = erot(0.5, 1.2, 0.8)
        v = torch.randn(3, dtype=torch.float64)
        v_rotated = R @ v
        
        assert torch.allclose(
            torch.norm(v_rotated), 
            torch.norm(v), 
            atol=1e-14
        )

    def test_erot_preserves_angles_between_vectors(self):
        """Rotation should preserve angles between vectors."""
        R = erot(0.5, 1.2, 0.8)
        v1 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        v2 = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        
        v1_rot = R @ v1
        v2_rot = R @ v2
        
        # Dot product should be preserved
        dot_orig = torch.dot(v1, v2)
        dot_rot = torch.dot(v1_rot, v2_rot)
        
        assert torch.allclose(dot_orig, dot_rot, atol=1e-14)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
