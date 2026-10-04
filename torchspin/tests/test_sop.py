#!/usr/bin/env python3
"""
Test suite for spin operators (sop module).

Based on MATLAB tests in easyspin/tests/:
- sop_alphabeta.m
- sop_comparesyntaxes.m
- sop_multipleops.m
- sop_numericsyntax.m
- sop_onespinselective.m
- sop_sparsefull.m
- sop_spinone.m
- sop_spinonehalf.m
- sop_spinsys.m
- sop_spinthreehalves.m
- sop_threespins.m
- sop_twospinsonehalve.m
- sop_twospinsselective.m
- sop_zerospin.m
"""

import pytest
import torch
import numpy as np
from torchspin.spinops import sop


class TestSopBasic:
    """Basic spin operator tests."""

    def test_sop_spin_half_Sx(self):
        """Test S_x for spin-1/2 (sop_spinonehalf.m)."""
        Sx = sop(1/2, 'x')
        expected = 0.5 * torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
        assert torch.allclose(Sx, expected)

    def test_sop_spin_half_Sy(self):
        """Test S_y for spin-1/2 (sop_spinonehalf.m)."""
        Sy = sop(1/2, 'y')
        expected = 0.5 * torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
        assert torch.allclose(Sy, expected)

    def test_sop_spin_half_Sz(self):
        """Test S_z for spin-1/2 (sop_spinonehalf.m)."""
        Sz = sop(1/2, 'z')
        expected = 0.5 * torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
        assert torch.allclose(Sz, expected)

    def test_sop_spin_half_Sp(self):
        """Test S_+ for spin-1/2 (sop_spinonehalf.m)."""
        Sp = sop(1/2, '+')
        expected = torch.tensor([[0, 1], [0, 0]], dtype=torch.complex128)
        assert torch.allclose(Sp, expected)

    def test_sop_spin_half_Sm(self):
        """Test S_- for spin-1/2 (sop_spinonehalf.m)."""
        Sm = sop(1/2, '-')
        expected = torch.tensor([[0, 0], [1, 0]], dtype=torch.complex128)
        assert torch.allclose(Sm, expected)

    def test_sop_spin_half_identity(self):
        """Test identity for spin-1/2 (sop_spinonehalf.m)."""
        Se = sop(1/2, 'e')
        expected = torch.eye(2, dtype=torch.complex128)
        assert torch.allclose(Se, expected)


class TestSopSpinOne:
    """Tests for spin-1 operators (sop_spinone.m)."""

    def test_sop_spin_one_dimension(self):
        """Spin-1 operators should be 3×3."""
        Sx = sop(1, 'x')
        assert Sx.shape == (3, 3)

    def test_sop_spin_one_Sz(self):
        """Test S_z for spin-1."""
        Sz = sop(1, 'z')
        expected = torch.tensor([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=torch.complex128)
        assert torch.allclose(Sz, expected)

    def test_sop_spin_one_Sx(self):
        """Test S_x for spin-1."""
        Sx = sop(1, 'x')
        sqrt2 = torch.sqrt(torch.tensor(2.0))
        expected = (1/sqrt2) * torch.tensor(
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.complex128
        )
        assert torch.allclose(Sx, expected, atol=1e-14)

    def test_sop_spin_one_Sy(self):
        """Test S_y for spin-1."""
        Sy = sop(1, 'y')
        sqrt2 = torch.sqrt(torch.tensor(2.0))
        expected = (1j/sqrt2) * torch.tensor(
            [[0, -1, 0], [1, 0, -1], [0, 1, 0]], dtype=torch.complex128
        )
        assert torch.allclose(Sy, expected, atol=1e-14)


class TestSopSpinThreeHalves:
    """Tests for spin-3/2 operators (sop_spinthreehalves.m)."""

    def test_sop_spin_threehalf_dimension(self):
        """Spin-3/2 operators should be 4×4."""
        Sz = sop(3/2, 'z')
        assert Sz.shape == (4, 4)

    def test_sop_spin_threehalf_Sz_diagonal(self):
        """S_z for spin-3/2 should be diagonal with values [3/2, 1/2, -1/2, -3/2]."""
        Sz = sop(3/2, 'z')
        diagonal = torch.diag(Sz).real
        expected = torch.tensor([3/2, 1/2, -1/2, -3/2], dtype=torch.float64)
        assert torch.allclose(diagonal, expected)


class TestSopNumericSyntax:
    """Test numeric specification syntax (sop_numericsyntax.m)."""

    def test_sop_numeric_index_1_is_x(self):
        """Index 1 should give S_x."""
        S1 = sop(1/2, 1)
        Sx = sop(1/2, 'x')
        assert torch.allclose(S1, Sx)

    def test_sop_numeric_index_2_is_y(self):
        """Index 2 should give S_y."""
        S2 = sop(1/2, 2)
        Sy = sop(1/2, 'y')
        assert torch.allclose(S2, Sy)

    def test_sop_numeric_index_3_is_z(self):
        """Index 3 should give S_z."""
        S3 = sop(1/2, 3)
        Sz = sop(1/2, 'z')
        assert torch.allclose(S3, Sz)

    def test_sop_numeric_index_4_is_plus(self):
        """Index 4 should give S_+."""
        S4 = sop(1/2, 4)
        Sp = sop(1/2, '+')
        assert torch.allclose(S4, Sp)

    def test_sop_numeric_index_5_is_minus(self):
        """Index 5 should give S_-."""
        S5 = sop(1/2, 5)
        Sm = sop(1/2, '-')
        assert torch.allclose(S5, Sm)

    def test_sop_numeric_index_0_is_identity(self):
        """Index 0 should give identity."""
        S0 = sop(1/2, 0)
        Se = sop(1/2, 'e')
        assert torch.allclose(S0, Se)


class TestSopMultiSpin:
    """Multi-spin system tests (sop_twospinsonehalve.m, sop_threespins.m)."""

    def test_sop_two_spins_dimension(self):
        """Two spin-1/2 system should have 4×4 operators."""
        S1x = sop([1/2, 1/2], [1, 1])
        assert S1x.shape == (4, 4)

    def test_sop_two_spins_first_spin_x(self):
        """Operator acting on first spin only."""
        S1x = sop([1/2, 1/2], [1, 1])
        Sx = sop(1/2, 'x')
        I = torch.eye(2, dtype=torch.complex128)
        expected = torch.kron(Sx, I)
        assert torch.allclose(S1x, expected)

    def test_sop_two_spins_second_spin_x(self):
        """Operator acting on second spin only."""
        S2x = sop([1/2, 1/2], [2, 1])
        Sx = sop(1/2, 'x')
        I = torch.eye(2, dtype=torch.complex128)
        expected = torch.kron(I, Sx)
        assert torch.allclose(S2x, expected)

    def test_sop_three_spins_dimension(self):
        """Three spin-1/2 system should have 8×8 operators."""
        S1z = sop([1/2, 1/2, 1/2], [1, 3])
        assert S1z.shape == (8, 8)

    def test_sop_two_spins_product_operator(self):
        """Test product operator S1x*S2y."""
        S1x_S2y = sop([1/2, 1/2], [[1, 1], [2, 2]])
        Sx = sop(1/2, 'x')
        Sy = sop(1/2, 'y')
        expected = torch.kron(Sx, Sy)
        assert torch.allclose(S1x_S2y, expected, atol=1e-14)


class TestSopAlphaBeta:
    """Alpha and beta projection operators (sop_alphabeta.m)."""

    def test_sop_alpha_projection(self):
        """Alpha projection: |α⟩⟨α| = (1/2)(I + 2*Sz) = (1/2)(I + σ_z)."""
        Sz = sop(1/2, 'z')
        I = sop(1/2, 'e')
        P_alpha = 0.5 * (I + 2 * Sz)
        expected = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex128)
        assert torch.allclose(P_alpha, expected)

    def test_sop_beta_projection(self):
        """Beta projection: |β⟩⟨β| = (1/2)(I - 2*Sz) = (1/2)(I - σ_z)."""
        Sz = sop(1/2, 'z')
        I = sop(1/2, 'e')
        P_beta = 0.5 * (I - 2 * Sz)
        expected = torch.tensor([[0, 0], [0, 1]], dtype=torch.complex128)
        assert torch.allclose(P_beta, expected)


class TestSopCommutationRelations:
    """Test spin commutation relations."""

    def test_sop_angular_momentum_commutation(self):
        """Test [S_x, S_y] = i*S_z."""
        Sx = sop(1/2, 'x')
        Sy = sop(1/2, 'y')
        Sz = sop(1/2, 'z')
        commutator = Sx @ Sy - Sy @ Sx
        expected = 1j * Sz
        assert torch.allclose(commutator, expected, atol=1e-14)

    def test_sop_cyclic_commutation(self):
        """Test [S_y, S_z] = i*S_x."""
        Sy = sop(1/2, 'y')
        Sz = sop(1/2, 'z')
        Sx = sop(1/2, 'x')
        commutator = Sy @ Sz - Sz @ Sy
        expected = 1j * Sx
        assert torch.allclose(commutator, expected, atol=1e-14)

    def test_sop_ladder_commutation(self):
        """Test [S_+, S_-] = 2*S_z."""
        Sp = sop(1/2, '+')
        Sm = sop(1/2, '-')
        Sz = sop(1/2, 'z')
        commutator = Sp @ Sm - Sm @ Sp
        expected = 2.0 * Sz
        assert torch.allclose(commutator, expected, atol=1e-14)


class TestSopProperties:
    """Test mathematical properties of spin operators."""

    def test_sop_hermitian(self):
        """S_x, S_y, S_z should be Hermitian."""
        for op in ['x', 'y', 'z']:
            S = sop(1/2, op)
            assert torch.allclose(S, S.conj().T, atol=1e-14), f"S_{op} not Hermitian"

    def test_sop_traceless(self):
        """S_x, S_y, S_z should be traceless."""
        zero_c128 = torch.tensor(0.0 + 0j, dtype=torch.complex128)
        for op in ['x', 'y', 'z']:
            S = sop(1/2, op)
            assert torch.allclose(torch.trace(S), zero_c128, atol=1e-14)

    def test_sop_Sp_adjoint_Sm(self):
        """S_+ should be adjoint of S_-."""
        Sp = sop(1/2, '+')
        Sm = sop(1/2, '-')
        assert torch.allclose(Sp, Sm.conj().T)

    def test_sop_S_squared_eigenvalues(self):
        """S² should have eigenvalue S(S+1) for spin-1/2."""
        Sx = sop(1/2, 'x')
        Sy = sop(1/2, 'y')
        Sz = sop(1/2, 'z')
        S2 = Sx @ Sx + Sy @ Sy + Sz @ Sz
        S = 1/2
        expected_eigenvalue = S * (S + 1)
        eigvals = torch.linalg.eigvalsh(S2)
        expected = torch.full_like(eigvals, expected_eigenvalue, dtype=torch.float64)
        assert torch.allclose(eigvals, expected, atol=1e-14)


class TestSopSparse:
    """Test sparse vs full format (sop_sparsefull.m)."""

    def test_sop_returns_dense_by_default(self):
        """sop should return dense tensors by default."""
        Sx = sop(1/2, 'x')
        assert not Sx.is_sparse

    def test_sop_large_hilbert_space(self):
        """Test that sop works for larger Hilbert spaces."""
        # Three spin-1 particles → 27-dimensional Hilbert space
        S1z = sop([1, 1, 1], [1, 3])
        assert S1z.shape == (27, 27)
        # Should be Hermitian
        assert torch.allclose(S1z, S1z.conj().T, atol=1e-14)


class TestSopMultipleOperators:
    """Test returning multiple operators at once (sop_multipleops.m)."""

    def test_sop_multiple_components(self):
        """Request multiple components at once."""
        Sx, Sy, Sz = sop(1/2, ['x', 'y', 'z'])
        
        # Check dimensions
        assert Sx.shape == (2, 2)
        assert Sy.shape == (2, 2)
        assert Sz.shape == (2, 2)
        
        # Check commutation
        commutator = Sx @ Sy - Sy @ Sx
        assert torch.allclose(commutator, 1j * Sz, atol=1e-14)


class TestSopEdgeCases:
    """Edge cases and error handling."""

    def test_sop_spin_zero_returns_identity(self):
        """Spin-0 should return 1×1 identity (sop_zerospin.m)."""
        Se = sop(0, 'e')
        assert Se.shape == (1, 1)
        assert torch.allclose(Se, torch.tensor([[1.0+0j]], dtype=torch.complex128))

    def test_sop_invalid_spin_raises_error(self):
        """Invalid spin quantum number should raise error."""
        with pytest.raises((ValueError, AssertionError)):
            sop(-1, 'x')  # Negative spin

    def test_sop_invalid_component_raises_error(self):
        """Invalid component specification should raise error."""
        with pytest.raises((ValueError, KeyError, IndexError)):
            sop(1/2, 'invalid')


class TestSopMultiSpinSelective:
    """Selective spin operator tests (sop_onespinselective.m)."""

    def test_sop_selective_middle_spin(self):
        """Apply operator to middle spin in three-spin system."""
        S2y = sop([1/2, 1/2, 1/2], [2, 2])
        Sy = sop(1/2, 'y')
        I = torch.eye(2, dtype=torch.complex128)
        expected = torch.kron(torch.kron(I, Sy), I)
        assert torch.allclose(S2y, expected, atol=1e-14)

    def test_sop_selective_last_spin(self):
        """Apply operator to last spin in three-spin system."""
        S3z = sop([1/2, 1/2, 1/2], [3, 3])
        Sz = sop(1/2, 'z')
        I = torch.eye(2, dtype=torch.complex128)
        expected = torch.kron(torch.kron(I, I), Sz)
        assert torch.allclose(S3z, expected, atol=1e-14)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
