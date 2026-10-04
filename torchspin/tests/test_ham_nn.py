"""Tests for torchspin.ham_nn."""
import torch

from torchspin import SpinSystem
from torchspin.ham_nn import ham_nn


def test_nn_zero_for_single_nucleus():
    """Single nucleus → zero Hamiltonian (need at least 2 for coupling)."""
    sys = SpinSystem(S=[0.5], Nucs=['1H'], A=[[0.0, 0.0, 0.0]])
    H = ham_nn(sys)
    assert H.abs().max().item() == 0.0
    assert H.shape == (4, 4)


def test_nn_zero_for_no_nuclei():
    """No nuclei → zero Hamiltonian."""
    sys = SpinSystem(S=[0.5])
    H = ham_nn(sys)
    assert H.abs().max().item() == 0.0
    assert H.shape == (2, 2)


def test_nn_matrix_is_hermitian():
    """Nuclear-nuclear Hamiltonian must be Hermitian."""
    sys = SpinSystem(
        S=[0.5],
        Nucs=['1H', '1H'],
        A=[[10.0, 10.0, 10.0], [5.0, 5.0, 5.0]],
        nn=[[3.0, 3.0, 5.0]],
    )
    H = ham_nn(sys)
    diff = (H - H.conj().T).abs().max().item()
    assert diff < 1e-13, f"NN Hamiltonian not Hermitian: {diff}"


def test_nn_zero_coupling_gives_zero():
    """Zero J tensor → zero Hamiltonian."""
    sys = SpinSystem(
        S=[0.5],
        Nucs=['1H', '1H'],
        A=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        nn=[[0.0, 0.0, 0.0]],
    )
    H = ham_nn(sys)
    assert H.abs().max().item() < 1e-14


def test_nn_isotropic_two_1H_eigenvalues():
    """Two 1H (I=1/2) coupled isotropically: triplet/singlet eigenvalue structure.

    H = J * (I1·I2) = J/2 * (F^2 - I1^2 - I2^2)
    F=1 (triplet): J/2*(2 - 3/4 - 3/4) = J/4   (3-fold degenerate)
    F=0 (singlet): J/2*(0 - 3/4 - 3/4) = -3J/4 (1-fold)

    With an S=1/2 electron, the Hilbert space is 2×2×2=8. The electron
    and nuclear degrees of freedom decouple (no HF), so each electron
    manifold has the same nuclear structure.
    """
    J = 10.0  # MHz, isotropic
    sys = SpinSystem(
        S=[0.5],
        Nucs=['1H', '1H'],
        A=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        nn=[[J, J, J]],
    )
    H = ham_nn(sys)
    evals = sorted(torch.linalg.eigvalsh(H).real.tolist())

    e_singlet = -3.0 * J / 4.0
    e_triplet = J / 4.0
    # Each electron manifold contributes: 1×singlet + 3×triplet → total 8 values
    expected = sorted([e_singlet] * 2 + [e_triplet] * 6)

    for got, exp in zip(evals, expected):
        assert abs(got - exp) < 1e-10, f"NN eigenvalue mismatch: got {got:.4f}, expected {exp:.4f}"


def test_nn_dimension():
    """S=1/2 + two 1H: Hilbert space 2×2×2 = 8."""
    sys = SpinSystem(
        S=[0.5],
        Nucs=['1H', '1H'],
        A=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        nn=[[5.0, 5.0, 5.0]],
    )
    H = ham_nn(sys)
    assert H.shape == (8, 8)


def test_nn_three_nuclei_six_pairs():
    """Three nuclei → 3 pairs; dimension and Hermiticity check."""
    sys = SpinSystem(
        S=[0.5],
        Nucs=['1H', '1H', '1H'],
        A=[[0.0, 0.0, 0.0]] * 3,
        nn=[[2.0, 2.0, 2.0], [1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
    )
    H = ham_nn(sys)
    assert H.shape == (16, 16)
    diff = (H - H.conj().T).abs().max().item()
    assert diff < 1e-13
