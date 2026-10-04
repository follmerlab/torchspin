"""Tests for torchspin.ham_zf."""
import torch

from torchspin import SpinSystem
from torchspin.ham_zf import ham_zf


def test_zfs_zero_for_half_spin():
    """S=1/2 has no ZFS (quadratic term vanishes for S<1)."""
    sys = SpinSystem(S=[0.5], D=[[100.0, 100.0, -200.0]])
    H = ham_zf(sys)
    # For S=1/2 the D·S·S term is proportional to identity and cancels (trace zero)
    # More precisely the D operator has zero off-diagonal elements and traceless diag
    assert H.shape == (2, 2)
    # Eigenvalues should be equal (degenerate) because S=1/2 has no ZFS
    evals = torch.linalg.eigvalsh(H).real
    assert abs((evals[1] - evals[0]).item()) < 1e-12


def test_axial_zfs_eigenvalues():
    """S=1, axial D tensor: eigenvalues are -D/3, -D/3, 2D/3.

    For H = D*(Sz^2 - S(S+1)/3), the diagonal elements in the |ms> basis are:
        ms=+1: D*(1 - 2/3) = D/3
        ms= 0: D*(0 - 2/3) = -2D/3
        ms=-1: D*(1 - 2/3) = D/3

    But EasySpin's convention stores D as [Dxx, Dyy, Dzz].
    For axial D (D=axial parameter), the standard convention is:
        Dxx = Dyy = -D/3,  Dzz = 2D/3
    giving H_ZF = Dxx*Sx^2 + Dyy*Sy^2 + Dzz*Sz^2
    """
    D = 100.0  # MHz, axial ZFS parameter
    # Standard axial: D tensor principal values
    Dxx = -D / 3.0
    Dyy = -D / 3.0
    Dzz = 2.0 * D / 3.0

    sys = SpinSystem(S=[1.0], D=[[Dxx, Dyy, Dzz]])
    H = ham_zf(sys)

    evals = sorted(torch.linalg.eigvalsh(H).real.tolist())

    # Expected: ms=0 -> lowest, ms=±1 -> doubly degenerate at higher energy
    # H eigenvalues: -2D/3 (ms=0), D/3 (ms=±1)
    expected = sorted([-2.0 * D / 3.0, D / 3.0, D / 3.0])

    for got, exp in zip(evals, expected):
        assert abs(got - exp) < 1e-10, f"ZFS eigenvalue: expected {exp:.4f}, got {got:.4f}"


def test_zfs_matrix_is_hermitian():
    """ZFS Hamiltonian must be Hermitian."""
    sys = SpinSystem(S=[1.0], D=[[50.0, -20.0, -30.0]])
    H = ham_zf(sys)
    diff = (H - H.conj().T).abs().max().item()
    assert diff < 1e-14, f"ZFS Hamiltonian not Hermitian: {diff}"


def test_zfs_zero_if_D_none():
    """With D=None, ham_zf returns the zero matrix."""
    sys = SpinSystem(S=[1.0])  # D defaults to None
    H = ham_zf(sys)
    assert H.abs().max().item() == 0.0


def test_zfs_isotropic_gives_identity():
    """Isotropic D=a*I shifts all levels equally — eigenvalues are all equal."""
    a = 50.0  # MHz
    sys = SpinSystem(S=[1.0], D=[[a, a, a]])
    H = ham_zf(sys)
    evals = torch.linalg.eigvalsh(H).real
    # All eigenvalues should be equal (H proportional to identity for isotropic D)
    assert (evals.max() - evals.min()).item() < 1e-10

def test_stevens_rank2_B20():
    """Stevens B2^0 for S=1 should give same result as corresponding D tensor.
    
    The relationship is: B2^0 = D_zz / 3  where D_zz = (2/3)*D parameter.
    For axial D, Dzz = 2D/3, so B2^0 = 2D/9.
    
    But from Stevens operator O2^0 = 3*Sz^2 - S(S+1)*I, and
    H_ZF = B2^0 * O2^0 should match H = D_zz * Sz^2 - ...
    """
    # Define using Stevens B coefficient
    B20 = 100.0  # MHz, just B2^0 coefficient
    # B coefficients are ordered [B_k^k, B_k^{k-1}, ..., B_k^{-k}]
    # For k=2: [B_2^2, B_2^1, B_2^0, B_2^{-1}, B_2^{-2}]
    # So B2^0 is at index 2
    sys = SpinSystem(S=[1.0], B=[torch.tensor([[0, 0, B20, 0, 0]])])
    H = ham_zf(sys)
    
    # Compute eigenvalues
    evals = sorted(torch.linalg.eigvalsh(H).real.tolist())
    
    # Expected from O2^0 = 3*Sz^2 - 2*I eigenvalues: (1, -2, 1)
    # H = B20 * O2^0 eigenvalues: (B20, -2*B20, B20)
    expected = sorted([B20, -2*B20, B20])
    
    for got, exp in zip(evals, expected):
        assert abs(got - exp) < 1e-10, f"Stevens B2^0: expected {exp:.4f}, got {got:.4f}"


def test_stevens_rank4_S32():
    """Test rank-4 Stevens operator for S=5/2 (e.g., Mn(II) or Fe(III))."""
    # For S=5/2, the states are ms = 5/2, 3/2, 1/2, -1/2, -3/2, -5/2
    # k=4 is valid since k ≤ 2*S = 5
    B40 = 50.0  # MHz
    
    # B[0] = rank-2 (None), B[1] = rank-4
    # For k=4: [B_4^4, B_4^3, B_4^2, B_4^1, B_4^0, B_4^{-1}, B_4^{-2}, B_4^{-3}, B_4^{-4}]
    # B4^0 is at index 4
    sys = SpinSystem(S=[2.5], B=[None, torch.tensor([[0, 0, 0, 0, B40, 0, 0, 0, 0]])])
    H = ham_zf(sys)
    
    # Check that H is Hermitian
    diff = (H - H.conj().T).abs().max().item()
    assert diff < 1e-14, f"Stevens rank-4 Hamiltonian not Hermitian: {diff}"
    
    # Eigenvalues should match B40 * eigenvalues(O4^0)
    # For S=5/2, O4^0 eigenvalues can be computed from the operator
    from torchspin.stev import stev
    O = stev([2.5], k=4, q=0, iSpin=1)
    O_evals = sorted(torch.linalg.eigvalsh(O).real.tolist())
    H_evals = sorted(torch.linalg.eigvalsh(H).real.tolist())
    
    for got, exp in zip(H_evals, [B40 * ev for ev in O_evals]):
        assert abs(got - exp) < 1e-10, f"Stevens B4^0: expected {exp:.4f}, got {got:.4f}"


def test_stevens_skip_k2_if_D_present():
    """If D is present, rank-2 Stevens should be skipped."""
    D_val = 100.0
    B20 = 50.0  # This should be ignored
    
    # System with both D and B2^0
    sys = SpinSystem(
        S=[1.0],
        D=[[-D_val/3, -D_val/3, 2*D_val/3]],  # axial D
        B=[torch.tensor([[0, 0, B20, 0, 0]])]  # rank-2 B2^0: should be skipped
    )
    
    # Should only use D, not B
    H_with_B = ham_zf(sys)
    
    # Reference without B
    sys_ref = SpinSystem(S=[1.0], D=[[-D_val/3, -D_val/3, 2*D_val/3]])
    H_ref = ham_zf(sys_ref)
    
    diff = (H_with_B - H_ref).abs().max().item()
    assert diff < 1e-12, f"Rank-2 Stevens not skipped when D present: {diff}"


def test_stevens_multiple_ranks():
    """Test system with both rank-2 and rank-4 Stevens coefficients."""
    B20 = 100.0
    B40 = 20.0
    
    # Use S=5/2 so both k=2 and k=4 are valid (k ≤ 2*S = 5)
    sys = SpinSystem(
        S=[2.5],
        B=[
            torch.tensor([[0, 0, B20, 0, 0]]),  # rank-2: B2^0
            torch.tensor([[0, 0, 0, 0, B40, 0, 0, 0, 0]])  # rank-4: B4^0
        ]
    )
    
    H = ham_zf(sys)
    
    # Check Hermiticity
    diff = (H - H.conj().T).abs().max().item()
    assert diff < 1e-14, f"Multi-rank Stevens Hamiltonian not Hermitian: {diff}"
    
    # Check that H = B20*O2^0 + B40*O4^0
    from torchspin.stev import stev
    O2 = stev([2.5], k=2, q=0, iSpin=1)
    O4 = stev([2.5], k=4, q=0, iSpin=1)
    H_expected = B20 * O2 + B40 * O4
    
    diff = (H - H_expected).abs().max().item()
    assert diff < 1e-10, f"Multi-rank Stevens H mismatch: {diff}"

class TestBFrameRotation:
    """Sys.BFrame tilts high-order Stevens terms via rank-k Wigner D.

    Verified against EasySpin ham_zf (S=2, B4 with B4Frame) to 5e-13.
    """

    def test_bframe_rotation_invariance_trace(self):
        import numpy as np
        from torchspin.ham_zf import ham_zf
        B4 = [80, 0, 40, 0, 10, 0, 5, 0, 2]
        sys0 = SpinSystem(S=[2.0], g=[[2.0, 2.0, 2.0]], B=[None, [B4]])
        sys1 = SpinSystem(S=[2.0], g=[[2.0, 2.0, 2.0]], B=[None, [B4]],
                          BFrame=[None, [[np.pi / 5, np.pi / 3, np.pi / 7]]])
        H0 = ham_zf(sys0).detach().cpu().numpy()
        H1 = ham_zf(sys1).detach().cpu().numpy()
        # Rotation changes the matrix but preserves the eigenvalues
        assert not np.allclose(H0, H1, atol=1e-6 * np.abs(H0).max())
        e0 = np.sort(np.linalg.eigvalsh(H0))
        e1 = np.sort(np.linalg.eigvalsh(H1))
        np.testing.assert_allclose(e0, e1, atol=1e-8 * np.abs(e0).max())

    def test_bframe_zero_angles_noop(self):
        import numpy as np
        from torchspin.ham_zf import ham_zf
        B4 = [80, 0, 40, 0, 10, 0, 5, 0, 2]
        sys0 = SpinSystem(S=[2.0], g=[[2.0, 2.0, 2.0]], B=[None, [B4]])
        sys1 = SpinSystem(S=[2.0], g=[[2.0, 2.0, 2.0]], B=[None, [B4]],
                          BFrame=[None, [[0.0, 0.0, 0.0]]])
        H0 = ham_zf(sys0).detach().cpu().numpy()
        H1 = ham_zf(sys1).detach().cpu().numpy()
        np.testing.assert_allclose(H0, H1, atol=1e-12)
