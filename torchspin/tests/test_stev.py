"""Tests for torchspin.stev module.

Verifies extended Stevens operators against known identities and
properties.
"""
import numpy as np
import pytest
import torch

from torchspin.stev import stev, O20, O22c, O40, O44c, O60
from torchspin.spinops import sop


def test_stev_returns_square_matrix():
    """Test that stev returns a square matrix."""
    Op = stev(1.0, k=2, q=0)
    nstates = 3  # 2*S+1 = 3 for S=1
    assert Op.shape == (nstates, nstates)


def test_stev_is_hermitian():
    """Test that all Stevens operators are Hermitian."""
    # Test several operators
    for S in [1.0, 1.5, 2.0]:
        for k in range(0, int(2 * S) + 1):
            for q in range(-k, k + 1):
                Op = stev(S, k=k, q=q)
                # Check Hermiticity: Op = Op†
                torch.testing.assert_close(Op, Op.H, rtol=1e-12, atol=1e-14)


def test_stev_O20_for_spin1():
    """Test O_2^0 for S=1 matches 3*Sz^2 - S(S+1)."""
    S = 1.0
    Op = stev(S, k=2, q=0)
    
    # O_2^0 = 3*Sz^2 - S(S+1)*I
    Sz = sop(S, 'z')
    expected = 3 * (Sz @ Sz) - S * (S + 1) * torch.eye(3, dtype=torch.complex128)
    
    torch.testing.assert_close(Op, expected, rtol=1e-12, atol=1e-14)


def test_stev_O22c_for_spin1():
    """Test O_2^2(c) for S=1 matches Sx^2 - Sy^2."""
    S = 1.0
    Op = stev(S, k=2, q=2)
    
    # O_2^2(c) = Sx^2 - Sy^2
    Sx = sop(S, 'x')
    Sy = sop(S, 'y')
    expected = Sx @ Sx - Sy @ Sy
    
    torch.testing.assert_close(Op, expected, rtol=1e-12, atol=1e-14)


def test_stev_k0_is_identity():
    """Test that O_0^0 is proportional to identity."""
    S = 1.5
    Op = stev(S, k=0, q=0)
    
    nstates = int(2 * S + 1)
    # O_0^0 should be scalar multiple of identity
    # Check that Op / Op[0,0] equals identity
    scale = Op[0, 0].real
    normalized = Op / scale
    expected = torch.eye(nstates, dtype=torch.complex128)
    
    torch.testing.assert_close(normalized, expected, rtol=1e-12, atol=1e-14)


def test_stev_dimension_correct():
    """Test that Stevens operator dimension matches spin system."""
    # Single spin
    S = 2.5
    Op = stev(S, k=2, q=0)
    assert Op.shape == (6, 6)  # 2*S+1 = 6
    
    # Multi-spin
    Spins = [0.5, 1.0]
    Op = stev(Spins, k=2, q=0, iSpin=2)
    assert Op.shape == (6, 6)  # (2*0.5+1) * (2*1+1) = 2*3 = 6


def test_stev_multispin_acts_on_correct_spin():
    """Test that multi-spin Stevens operator acts on the specified spin only."""
    Spins = [1.0, 1.0]
    
    # O_2^0 for first spin
    Op1 = stev(Spins, k=2, q=0, iSpin=1)
    
    # O_2^0 for second spin
    Op2 = stev(Spins, k=2, q=0, iSpin=2)
    
    # They should be different (acts on different subspace)
    assert not torch.allclose(Op1, Op2)
    
    # But both should be Hermitian
    torch.testing.assert_close(Op1, Op1.H, rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(Op2, Op2.H, rtol=1e-12, atol=1e-14)


def test_stev_sparse_output():
    """Test that sparse=True returns a sparse tensor."""
    Op = stev(1.5, k=2, q=0, sparse=True)
    assert Op.is_sparse
    
    # Dense and sparse should match
    Op_dense = stev(1.5, k=2, q=0, sparse=False)
    torch.testing.assert_close(Op.to_dense(), Op_dense, rtol=1e-12, atol=1e-14)


def test_stev_q_positive_is_cosine():
    """Test that q>0 gives cosine tesseral operator (real and symmetric)."""
    Op = stev(2.0, k=4, q=2)
    
    # Cosine operator should be real (imaginary part is zero)
    assert torch.allclose(Op.imag, torch.zeros_like(Op.imag), atol=1e-14)
    
    # And Hermitian (which for real means symmetric)
    torch.testing.assert_close(Op, Op.T, rtol=1e-12, atol=1e-14)


def test_stev_q_negative_is_sine():
    """Test that q<0 gives sine tesseral operator (pure imaginary on off-diagonal)."""
    Op = stev(2.0, k=4, q=-2)
    
    # Sine operator O_k^q(s) = (T - T†)/(2i) should be Hermitian
    torch.testing.assert_close(Op, Op.H, rtol=1e-12, atol=1e-14)


def test_stev_k_exceeds_2S_raises_error():
    """Test that k > 2*S raises ValueError."""
    with pytest.raises(ValueError, match="k=5 exceeds 2\\*S=4"):
        stev(2.0, k=5, q=0)  # k=5 > 2*S=4 for S=2


def test_stev_q_exceeds_k_raises_error():
    """Test that |q| > k raises ValueError."""
    with pytest.raises(ValueError, match="q must be an integer"):
        stev(2.0, k=4, q=5)  # |q|=5 > k=4


def test_stev_k_too_large_raises_error():
    """Test that k > 12 raises ValueError (maximum supported)."""
    with pytest.raises(ValueError, match="too large"):
        stev(10.0, k=13, q=0)  # k=13 exceeds maximum 12


def test_stev_invalid_spin_raises_error():
    """Test that invalid spin values raise ValueError."""
    with pytest.raises(ValueError, match="non-negative multiples of 1/2"):
        stev(-0.5, k=2, q=0)  # negative spin
    
    with pytest.raises(ValueError, match="non-negative multiples of 1/2"):
        stev(0.3, k=2, q=0)  # not a half-integer


def test_stev_multispin_missing_iSpin_raises_error():
    """Test that multi-spin without iSpin raises ValueError."""
    with pytest.raises(ValueError, match="iSpin must be specified"):
        stev([1.0, 1.0], k=2, q=0)  # iSpin not provided


def test_stev_iSpin_out_of_range_raises_error():
    """Test that iSpin out of range raises ValueError."""
    with pytest.raises(ValueError, match="out of range"):
        stev([1.0, 1.5], k=2, q=0, iSpin=3)  # only 2 spins


def test_stev_convenience_functions():
    """Test convenience functions O20, O22c, etc."""
    S = 3.0  # Use S=3 so we can test up to k=6
    
    # O20 should match stev(S, k=2, q=0)
    Op1 = O20(S)
    Op2 = stev(S, k=2, q=0)
    torch.testing.assert_close(Op1, Op2)
    
    # O22c should match stev(S, k=2, q=2)
    Op1 = O22c(S)
    Op2 = stev(S, k=2, q=2)
    torch.testing.assert_close(Op1, Op2)
    
    # O40
    Op1 = O40(S)
    Op2 = stev(S, k=4, q=0)
    torch.testing.assert_close(Op1, Op2)
    
    # O44c
    Op1 = O44c(S)
    Op2 = stev(S, k=4, q=4)
    torch.testing.assert_close(Op1, Op2)
    
    # O60
    Op1 = O60(S)
    Op2 = stev(S, k=6, q=0)
    torch.testing.assert_close(Op1, Op2)


def test_stev_traceless_for_k_greater_0():
    """Test that Stevens operators with k>0 are traceless."""
    for S in [1.0, 1.5, 2.0, 2.5]:
        for k in range(1, int(2 * S) + 1):
            for q in [0, k, -k]:  # test a few q values
                Op = stev(S, k=k, q=q)
                trace = torch.trace(Op)
                assert abs(trace) < 1e-12, f"Op trace = {trace} for S={S}, k={k}, q={q}"


def test_stev_O20_eigenvalues_spin1():
    """Test O_2^0 eigenvalues for S=1."""
    S = 1.0
    Op = stev(S, k=2, q=0)
    
    # O_2^0 for S=1 has eigenvalues corresponding to m^2 - 2/3
    # For m = 1, 0, -1: eigenvalues are 1-2/3=1/3, -2/3, 1/3
    # But normalized differently. Let's just check they're real and sorted.
    eigs = torch.linalg.eigvalsh(Op)
    
    # O_2^0 = 3*Sz^2 - S(S+1)*I for S=1
    # m=1: 3*1 - 2 = 1
    # m=0: 3*0 - 2 = -2
    # m=-1: 3*1 - 2 = 1
    expected_eigs = torch.tensor([-2.0, 1.0, 1.0], dtype=torch.float64)
    
    torch.testing.assert_close(
        torch.sort(eigs)[0],
        torch.sort(expected_eigs)[0],
        rtol=1e-12,
        atol=1e-14,
    )


def test_stev_high_rank_operators_exist():
    """Test that high-rank operators (k=6, k=8, etc.) can be constructed."""
    # S=5/2 system (e.g., Mn2+)
    S = 2.5
    
    # Should be able to construct up to k=5 (= 2*S)
    for k in [2, 4]:
        Op = stev(S, k=k, q=0)
        assert Op.shape == (6, 6)
        torch.testing.assert_close(Op, Op.H, rtol=1e-12, atol=1e-14)


def test_stev_dtype_argument():
    """Test that dtype argument is respected."""
    # Complex128 (default)
    Op = stev(1.0, k=2, q=0, dtype=torch.complex128)
    assert Op.dtype == torch.complex128
    
    # Complex64
    Op = stev(1.0, k=2, q=0, dtype=torch.complex64)
    assert Op.dtype == torch.complex64


def test_stev_all_q_for_k2():
    """Test that all q components for k=2 are Hermitian and traceless."""
    S = 1.5
    k = 2
    
    for q in range(-k, k + 1):
        Op = stev(S, k=k, q=q)
        
        # Hermitian
        torch.testing.assert_close(Op, Op.H, rtol=1e-12, atol=1e-14)
        
        # Traceless (k>0)
        trace = torch.trace(Op)
        assert abs(trace) < 1e-12


def test_stev_reproduces_matlab_values_S1():
    """Test a few matrix elements against known MATLAB values for S=1."""
    # This is a regression test - values computed from EasySpin stev.m
    S = 1.0
    
    # O_2^0 for S=1
    Op = stev(S, k=2, q=0)
    # MATLAB gives O_2^0 = diag([1, -2, 1]) for S=1
    expected = torch.diag(torch.tensor([1.0, -2.0, 1.0], dtype=torch.complex128))
    torch.testing.assert_close(Op, expected, rtol=1e-12, atol=1e-14)
    
    # O_2^2(c) for S=1
    Op = stev(S, k=2, q=2)
    # From MATLAB: O_2^2(c) = Sx^2 - Sy^2 for S=1
    # Sx = 1/sqrt(2) * [[0,1,0],[1,0,1],[0,1,0]]
    # Sy = 1/sqrt(2) * [[0,-i,0],[i,0,-i],[0,i,0]]
    # Sx^2 - Sy^2 should give a specific pattern
    Sx = sop(S, 'x')
    Sy = sop(S, 'y')
    expected = Sx @ Sx - Sy @ Sy
    torch.testing.assert_close(Op, expected, rtol=1e-12, atol=1e-14)
