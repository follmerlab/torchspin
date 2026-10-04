"""Integration tests for torchspin.ham (top-level Hamiltonian assembler)."""
import torch

from torchspin import SpinSystem, ham
from torchspin.constants import BMAGN, GFREE, PLANCK


def test_ham_returns_hermitian_matrix():
    """Full Hamiltonian is Hermitian for a nitroxide-like system."""
    sys = SpinSystem(
        S=[0.5],
        g=[[2.009, 2.006, 2.002]],
        Nucs=['14N'],
        A=[[10.0, 10.0, 95.0]],
    )
    B0 = [0.0, 0.0, 340.0]  # mT
    H = ham(sys, B0=B0)
    diff = (H - H.conj().T).abs().max().item()
    assert diff < 1e-12, f"Hamiltonian not Hermitian: max diff = {diff}"


def test_ham_dimension_nitroxide():
    """S=1/2 + 14N (I=1): Hilbert space 2*3 = 6."""
    sys = SpinSystem(
        S=[0.5],
        g=[[2.0, 2.0, 2.0]],
        Nucs=['14N'],
        A=[[10.0, 10.0, 50.0]],
    )
    B0 = [0.0, 0.0, 340.0]
    H = ham(sys, B0=B0)
    assert H.shape == (6, 6)


def test_ham_no_field_returns_tuple():
    """With B0=None, ham returns (H0, mux, muy, muz)."""
    sys = SpinSystem(S=[0.5])
    result = ham(sys, B0=None)
    assert isinstance(result, tuple) and len(result) == 4


def test_ham_field_independent_part_is_zero_for_free_radical():
    """Free radical (g only, no D/hf/ee): H0 is zero."""
    sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
    H0, mux, muy, muz = ham(sys, B0=None)
    assert H0.abs().max().item() < 1e-14


def test_ham_eigenvalues_simple_zeeman():
    """Simple Zeeman: eigenvalues match analytical free-electron splitting."""
    sys = SpinSystem(S=[0.5], g=[[GFREE, GFREE, GFREE]])
    B_mT = 350.0
    H = ham(sys, B0=[0.0, 0.0, B_mT])
    evals = torch.linalg.eigvalsh(H).real

    B_T = B_mT * 1e-3
    delta = GFREE * BMAGN / PLANCK * B_T * 1e-6  # MHz
    assert abs((evals[1] - evals[0]).item() - delta) < 1e-6


def test_ham_two_electron_system():
    """Two-electron system with Heisenberg coupling: correct Hilbert space."""
    sys = SpinSystem(
        S=[0.5, 0.5],
        g=[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
        ee=[[10.0, 10.0, 10.0]],
    )
    H = ham(sys, B0=[0.0, 0.0, 300.0])
    assert H.shape == (4, 4)
    diff = (H - H.conj().T).abs().max().item()
    assert diff < 1e-12
