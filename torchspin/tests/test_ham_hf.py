"""Tests for torchspin.ham_hf."""
import torch

from torchspin import SpinSystem
from torchspin.ham_hf import ham_hf


def test_hf_zero_without_nuclei():
    """No nuclei → zero Hamiltonian."""
    sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
    H = ham_hf(sys)
    assert H.abs().max().item() == 0.0
    assert H.shape == (2, 2)


def test_hf_dimension_with_14N():
    """S=1/2 + 14N (I=1): Hilbert space is 2*3 = 6."""
    sys = SpinSystem(S=[0.5], Nucs=['14N'], A=[[10.0, 10.0, 50.0]])
    H = ham_hf(sys)
    assert H.shape == (6, 6)


def test_hf_matrix_is_hermitian():
    """Hyperfine Hamiltonian must be Hermitian."""
    sys = SpinSystem(
        S=[0.5],
        Nucs=['14N'],
        A=[[10.0, 10.0, 50.0]],
    )
    H = ham_hf(sys)
    diff = (H - H.conj().T).abs().max().item()
    assert diff < 1e-14, f"HF Hamiltonian not Hermitian: {diff}"


def test_hf_isotropic_1H_eigenvalues():
    """S=1/2, I=1/2, isotropic A: eigenvalues from exact 2-spin solution.

    H = A * (Sx*Ix + Sy*Iy + Sz*Iz) = A/2 * (F^2 - S^2 - I^2)
    where F = S + I.  Eigenvalues:
        F=1 (triplet): A/2 * (2 - 3/4 - 3/4) = A/4   (3-fold degenerate)
        F=0 (singlet): A/2 * (0 - 3/4 - 3/4) = -3A/4 (1-fold)
    """
    A = 20.0  # MHz isotropic
    sys = SpinSystem(S=[0.5], Nucs=['1H'], A=[[A, A, A]])
    H = ham_hf(sys)
    evals = sorted(torch.linalg.eigvalsh(H).real.tolist())

    # Expected: -3A/4 (singlet) and A/4 (triplet ×3)
    e_singlet = -3.0 * A / 4.0
    e_triplet = A / 4.0
    expected = sorted([e_singlet, e_triplet, e_triplet, e_triplet])

    for got, exp in zip(evals, expected):
        assert abs(got - exp) < 1e-10, \
            f"HF eigenvalue mismatch: expected {exp:.4f}, got {got:.4f}"


def test_hf_zero_A_gives_zero_matrix():
    """Zero hyperfine tensor → zero Hamiltonian."""
    sys = SpinSystem(S=[0.5], Nucs=['14N'], A=[[0.0, 0.0, 0.0]])
    H = ham_hf(sys)
    assert H.abs().max().item() < 1e-14
