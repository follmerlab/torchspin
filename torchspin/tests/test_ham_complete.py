"""Integration tests for the complete spin Hamiltonian (ham).

Checks that all interaction terms combine correctly in the full ham() assembler.
"""
import torch

from torchspin import SpinSystem, ham


def test_ham_nitroxide_hermitian():
    """S=1/2 + 14N nitroxide-like system: full H is Hermitian at finite B0."""
    sys = SpinSystem(
        S=[0.5],
        g=[[2.009, 2.006, 2.002]],
        Nucs=['14N'],
        A=[[10.0, 10.0, 95.0]],
        Q=[[-0.3, -0.3, 0.6]],
    )
    B0 = [0.0, 0.0, 340.0]  # mT
    H = ham(sys, B0)
    diff = (H - H.conj().T).abs().max().item()
    assert diff < 1e-12, f"Full Hamiltonian not Hermitian: {diff}"


def test_ham_nitroxide_dimension():
    """S=1/2 + 14N: Hilbert space 2×3 = 6."""
    sys = SpinSystem(
        S=[0.5],
        Nucs=['14N'],
        A=[[10.0, 10.0, 95.0]],
    )
    B0 = [0.0, 0.0, 340.0]
    H = ham(sys, B0)
    assert H.shape == (6, 6)


def test_ham_returns_tuple_without_B0():
    """ham(sys) without B0 returns (H0, mux, muy, muz)."""
    sys = SpinSystem(S=[0.5], Nucs=['1H'], A=[[20.0, 20.0, 20.0]])
    result = ham(sys)
    assert isinstance(result, tuple)
    assert len(result) == 4
    H0, mux, muy, muz = result
    assert H0.shape == (4, 4)
    assert mux.shape == (4, 4)


def test_ham_eigenvalues_finite():
    """All eigenvalues of the full Hamiltonian are finite real numbers."""
    sys = SpinSystem(
        S=[0.5],
        g=[[2.008, 2.006, 2.002]],
        Nucs=['14N'],
        A=[[10.0, 10.0, 95.0]],
        Q=[[-0.3, -0.3, 0.6]],
    )
    B0 = [0.0, 0.0, 340.0]
    H = ham(sys, B0)
    evals = torch.linalg.eigvalsh(H)
    assert torch.all(torch.isfinite(evals))


def test_ham_two_electrons_two_nuclei():
    """Two-electron, two-nucleus system: Hermitian and correct dimension."""
    sys = SpinSystem(
        S=[0.5, 0.5],
        Nucs=['1H', '14N'],
        A=[[20.0, 20.0, 20.0, 5.0, 5.0, 5.0],
           [3.0, 3.0, 3.0, 10.0, 10.0, 50.0]],
        ee=[[5.0, 5.0, 5.0]],
    )
    B0 = [0.0, 0.0, 340.0]
    H = ham(sys, B0)
    # States: 2 * 2 * 2 * 3 = 24
    assert H.shape == (24, 24)
    diff = (H - H.conj().T).abs().max().item()
    assert diff < 1e-12


def test_ham_nuclear_zeeman_shifts_eigenvalues():
    """Adding nuclear Zeeman (via full ham) shifts eigenvalues vs. no-NZ case.

    At 340 mT, the 1H nuclear Zeeman is ~0.5 MHz, which should produce a
    measurable shift in at least some eigenvalues.
    """
    sys = SpinSystem(S=[0.5], Nucs=['1H'], A=[[0.0, 0.0, 0.0]])
    B0 = [0.0, 0.0, 340.0]
    H_full = ham(sys, B0)
    evals_full = torch.linalg.eigvalsh(H_full).real

    # Compare against electron-only Zeeman (no nuclear contributions)
    from torchspin.ham_ez import ham_ez
    H_ez_only = ham_ez(sys, B0)
    evals_ez = torch.linalg.eigvalsh(H_ez_only).real

    # The eigenvalues from the full H should differ from electron-only
    max_diff = (evals_full - evals_ez).abs().max().item()
    assert max_diff > 1e-4, "Nuclear Zeeman contribution is unexpectedly zero"


def test_ham_nn_included():
    """Nuclear-nuclear coupling is included in H0."""
    J = 5.0  # MHz
    sys = SpinSystem(
        S=[0.5],
        Nucs=['1H', '1H'],
        A=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        nn=[[J, J, J]],
    )
    H0, mux, muy, muz = ham(sys)

    # Reference: same system without NN
    sys_no_nn = SpinSystem(
        S=[0.5],
        Nucs=['1H', '1H'],
        A=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    )
    H0_no_nn, _, _, _ = ham(sys_no_nn)

    diff = (H0 - H0_no_nn).abs().max().item()
    assert diff > 1e-6, "NN coupling not found in H0"
