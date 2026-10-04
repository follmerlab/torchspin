"""Tests for torchspin.ham_nq."""
import math

import torch

from torchspin import SpinSystem
from torchspin.ham_nq import ham_nq


def test_nq_zero_without_nuclei():
    """No nuclei → zero Hamiltonian."""
    sys = SpinSystem(S=[0.5])
    H = ham_nq(sys)
    assert H.abs().max().item() == 0.0
    assert H.shape == (2, 2)


def test_nq_zero_for_spin_half_nucleus():
    """1H (I=1/2) → quadrupole is zero (only I≥1 nuclei have Q)."""
    sys = SpinSystem(S=[0.5], Nucs=['1H'], A=[[0.0, 0.0, 0.0]],
                     Q=[[1.0, 1.0, -2.0]])
    H = ham_nq(sys)
    assert H.abs().max().item() < 1e-14


def test_nq_matrix_is_hermitian():
    """NQ Hamiltonian must be Hermitian."""
    sys = SpinSystem(
        S=[0.5],
        Nucs=['14N'],
        A=[[10.0, 10.0, 50.0]],
        Q=[[-0.5, -0.5, 1.0]],   # MHz, axial
    )
    H = ham_nq(sys)
    diff = (H - H.conj().T).abs().max().item()
    assert diff < 1e-13, f"NQ Hamiltonian not Hermitian: {diff}"


def test_nq_zero_tensor_gives_zero():
    """Q = 0 → zero Hamiltonian."""
    sys = SpinSystem(S=[0.5], Nucs=['14N'], A=[[0.0, 0.0, 0.0]],
                     Q=[[0.0, 0.0, 0.0]])
    H = ham_nq(sys)
    assert H.abs().max().item() < 1e-14


def test_nq_no_Q_gives_zero():
    """If sys.Q is None, NQ Hamiltonian is zero."""
    sys = SpinSystem(S=[0.5], Nucs=['14N'], A=[[10.0, 10.0, 50.0]])
    H = ham_nq(sys)
    assert H.abs().max().item() < 1e-14


def test_nq_14N_axial_eigenvalues():
    """14N (I=1), axial Q=[Qxx,Qyy,Qzz] with Qzz=-Qxx-Qyy.

    For an axial Q-tensor in the molecular z frame with principal values
    (-eq, -eq, 2eq) where eq = e²qQ/(4I(2I-1)*h):
    The eigenvalues of I·Q·I are:
        -2*eq*Iz^2 + eq*(Ix^2 + Iy^2)  (using standard I·Q·I form)
    For I=1, Iz eigenvalues are {-1, 0, +1}.
    Eigenvalues of H_NQ:
      mI=0:  eq*(Ix^2+Iy^2)=eq*I(I+1)/3... Let's compute numerically.

    For Q = diag(-e, -e, 2e) and I·Q·I:
    H_NQ(mI,mI) = mI**2 * Qzz + ...
    For I=1:
      Ix^2 + Iy^2 = I(I+1) - Iz^2 = 2 - mI^2
    So H_NQ ≈ Qzz*mI^2 + Qxx*(2 - mI^2)/2 + Qyy*(2-mI^2)/2  (diagonal approx)
    For Qxx=Qyy=-e, Qzz=2e:
      mI = ±1: e_1 = 2e - e*(2-1) = 2e - e = e
      mI = 0:  e_0 = 0 - e*(2)    = -2e
    """
    e = 1.5  # MHz
    sys = SpinSystem(
        S=[0.5],
        Nucs=['14N'],
        A=[[0.0, 0.0, 0.0]],
        Q=[[-e, -e, 2 * e]],  # axial: Qxx=Qyy=-e, Qzz=2e
    )
    H = ham_nq(sys)
    evals = sorted(torch.linalg.eigvalsh(H).real.tolist())

    # Each electron manifold (mS = ±1/2) has nuclear eigenvalues
    # For I=1 with axial Q on z: mI=±1 → +e, mI=0 → -2e
    # Total 6 eigenvalues: -2e, -2e, e, e, e, e  (degenerate with electron)
    expected = sorted([-2 * e, -2 * e, e, e, e, e])
    for got, exp in zip(evals, expected):
        assert abs(got - exp) < 1e-8, f"NQ eigenvalue mismatch: got {got:.4f}, expected {exp:.4f}"


def test_nq_dimension():
    """S=1/2 + 14N (I=1): Hilbert space 2×3 = 6."""
    sys = SpinSystem(S=[0.5], Nucs=['14N'], A=[[0.0, 0.0, 0.0]],
                     Q=[[-1.0, -1.0, 2.0]])
    H = ham_nq(sys)
    assert H.shape == (6, 6)
