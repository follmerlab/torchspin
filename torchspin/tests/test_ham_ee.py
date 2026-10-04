"""Tests for torchspin.ham_ee."""
import torch

from torchspin import SpinSystem
from torchspin.ham_ee import ham_ee


def test_ee_zero_for_single_electron():
    """Single electron → zero Hamiltonian."""
    sys = SpinSystem(S=[0.5])
    H = ham_ee(sys)
    assert H.abs().max().item() == 0.0
    assert H.shape == (2, 2)


def test_heisenberg_eigenvalues():
    """Two S=1/2 with isotropic J: eigenvalues from exact solution.

    H = J * (S1·S2) = J/2 * (S_total^2 - S1^2 - S2^2)
    S_total=1 (triplet): J/2*(2 - 3/4 - 3/4) = J/4   (3-fold degenerate)
    S_total=0 (singlet): J/2*(0 - 3/4 - 3/4) = -3J/4 (1-fold)
    """
    J = 50.0  # MHz
    sys = SpinSystem(S=[0.5, 0.5], ee=[[J, J, J]])
    H = ham_ee(sys)
    evals = sorted(torch.linalg.eigvalsh(H).real.tolist())

    e_singlet = -3.0 * J / 4.0
    e_triplet = J / 4.0
    expected = sorted([e_singlet, e_triplet, e_triplet, e_triplet])

    for got, exp in zip(evals, expected):
        assert abs(got - exp) < 1e-10, \
            f"EE eigenvalue mismatch: expected {exp:.4f}, got {got:.4f}"


def test_ee_matrix_is_hermitian():
    """Electron-electron Hamiltonian must be Hermitian."""
    sys = SpinSystem(S=[0.5, 0.5], ee=[[30.0, 40.0, 50.0]])
    H = ham_ee(sys)
    diff = (H - H.conj().T).abs().max().item()
    assert diff < 1e-14, f"EE Hamiltonian not Hermitian: {diff}"


def test_ee_zero_coupling_gives_zero():
    """Zero coupling → zero Hamiltonian."""
    sys = SpinSystem(S=[0.5, 0.5], ee=[[0.0, 0.0, 0.0]])
    H = ham_ee(sys)
    assert H.abs().max().item() < 1e-14


def test_ee_dimension_two_spin1():
    """Two S=1 electrons → Hilbert space 3×3 = 9."""
    sys = SpinSystem(S=[1.0, 1.0], ee=[[10.0, 10.0, 10.0]])
    H = ham_ee(sys)
    assert H.shape == (9, 9)
