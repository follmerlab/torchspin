"""Tests for torchspin.ham_ez."""
import math

import torch

from torchspin import SpinSystem
from torchspin.constants import BMAGN, GFREE, PLANCK
from torchspin.ham_ez import ham_ez


def test_zeeman_eigenvalues_free_electron():
    """Free-electron S=1/2 Zeeman splitting matches analytical value.

    At field B = 350 mT along z, eigenvalues are:
        E± = ±(gfree * bmagn / planck) * B/2
    in Hz, divided by 1e6 to get MHz.
    The splitting is gfree * bmagn/planck * B [Hz/T] * B[T] / 1e6 [->MHz]
    with B = 350e-3 T.
    """
    sys = SpinSystem(S=[0.5], g=[[GFREE, GFREE, GFREE]])
    B0 = [0.0, 0.0, 350.0]  # mT

    H = ham_ez(sys, B0=B0)
    evals = torch.linalg.eigvalsh(H).real  # MHz, ascending order

    # Expected splitting  Δ = gfree*bmagn/planck * 0.350 [T] * 1e-6 [Hz->MHz]
    B_T = 350.0e-3  # T
    delta = GFREE * BMAGN / PLANCK * B_T * 1e-6  # MHz

    computed_delta = (evals[1] - evals[0]).item()
    assert abs(computed_delta - delta) < 1e-6, \
        f"Zeeman splitting: expected {delta:.6f} MHz, got {computed_delta:.6f} MHz"


def test_zeeman_matrix_is_hermitian():
    sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
    H = ham_ez(sys, B0=[100.0, 50.0, 200.0])
    diff = (H - H.conj().T).abs().max().item()
    assert diff < 1e-14, f"Hamiltonian is not Hermitian: max diff = {diff}"


def test_moment_operators_traceless():
    """Magnetic-moment operators must be traceless."""
    sys = SpinSystem(S=[0.5], g=[[2.006, 2.006, 2.002]])
    mux, muy, muz = ham_ez(sys, B0=None)
    assert mux.trace().abs().item() < 1e-14
    assert muy.trace().abs().item() < 1e-14
    assert muz.trace().abs().item() < 1e-14


def test_zeeman_g_anisotropy():
    """With anisotropic g, the Hamiltonian still has correct eigenvalues."""
    gx, gy, gz = 2.005, 2.006, 2.002
    sys = SpinSystem(S=[0.5], g=[[gx, gy, gz]])
    # B along z — only gz contributes
    B = 300.0  # mT
    H = ham_ez(sys, B0=[0.0, 0.0, B])
    evals = torch.linalg.eigvalsh(H).real

    B_T = B * 1e-3
    delta = gz * BMAGN / PLANCK * B_T * 1e-6  # MHz
    computed_delta = (evals[1] - evals[0]).item()
    assert abs(computed_delta - delta) < 1e-6


def test_two_electron_zeeman_dimension():
    """Two-electron system has correct Hilbert-space dimension (4×4)."""
    sys = SpinSystem(S=[0.5, 0.5], g=[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
    H = ham_ez(sys, B0=[0.0, 0.0, 300.0])
    assert H.shape == (4, 4)
