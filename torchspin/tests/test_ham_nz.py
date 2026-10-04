"""Tests for torchspin.ham_nz."""
import torch

from torchspin import SpinSystem
from torchspin.constants import NMAGN, PLANCK
from torchspin.ham_nz import ham_nz


def test_nz_zero_without_nuclei():
    """No nuclei → zero Hamiltonian (and zero moment operators)."""
    sys = SpinSystem(S=[0.5])
    H = ham_nz(sys, B0=[0.0, 0.0, 300.0])
    assert H.abs().max().item() == 0.0
    assert H.shape == (2, 2)


def test_nz_moment_operators_shape():
    """Moment operators have correct shape."""
    sys = SpinSystem(S=[0.5], Nucs=['1H'], A=[[0.0, 0.0, 0.0]])
    mux, muy, muz = ham_nz(sys, B0=None)
    assert mux.shape == (4, 4)
    assert muy.shape == (4, 4)
    assert muz.shape == (4, 4)


def test_nz_1H_eigenvalue_gap():
    """S=1/2 + 1H: nuclear Zeeman gap at B=[0,0,B_z] mT.

    The 1H nuclear resonance frequency at B_z mT is:
        nu_H = NMAGN/PLANCK * gn_1H * B_z [T] * 1e-9  [MHz/mT] * B_z
    Since the nucleus is I=1/2, the gap between mI=+1/2 and mI=-1/2 is nu_H.
    We compute ham_nz alone (no hyperfine) and check that the eigenvalue
    spread of the nuclear-Zeeman-only contribution matches.
    """
    gn_1H = 5.585694702
    B_z = 300.0  # mT
    expected_gap = NMAGN / PLANCK * gn_1H * (B_z * 1e-3) * 1e-6  # MHz

    # Use zero hyperfine so the spectrum is purely from nuclear Zeeman
    sys = SpinSystem(S=[0.5], Nucs=['1H'], A=[[0.0, 0.0, 0.0]])
    H = ham_nz(sys, B0=[0.0, 0.0, B_z])
    evals = torch.linalg.eigvalsh(H).real.tolist()
    evals_sorted = sorted(evals)

    # The 4 eigenvalues are: two pairs separated by the NZ gap
    # (electron manifold doesn't matter, NZ acts on nuclear subspace only)
    unique_gaps = set()
    for i in range(len(evals_sorted)):
        for j in range(i + 1, len(evals_sorted)):
            g = abs(evals_sorted[j] - evals_sorted[i])
            if g > 1e-10:
                unique_gaps.add(round(g, 8))

    assert any(abs(g - expected_gap) < 1e-6 for g in unique_gaps), (
        f"Expected nuclear Zeeman gap {expected_gap:.6f} MHz in {unique_gaps}"
    )


def test_nz_matrix_is_hermitian():
    """Nuclear Zeeman Hamiltonian must be Hermitian."""
    sys = SpinSystem(S=[0.5], Nucs=['14N'], A=[[10.0, 10.0, 50.0]])
    H = ham_nz(sys, B0=[10.0, 20.0, 300.0])
    diff = (H - H.conj().T).abs().max().item()
    assert diff < 1e-13, f"NZ Hamiltonian not Hermitian: {diff}"


def test_nz_scales_linearly_with_field():
    """Nuclear Zeeman energy scales linearly with B0."""
    sys = SpinSystem(S=[0.5], Nucs=['1H'], A=[[0.0, 0.0, 0.0]])
    H1 = ham_nz(sys, B0=[0.0, 0.0, 100.0])
    H2 = ham_nz(sys, B0=[0.0, 0.0, 200.0])
    ratio = (H2 / H1).real
    # All non-zero elements should have ratio ≈ 2.0
    mask = H1.abs() > 1e-10
    assert (ratio[mask] - 2.0).abs().max().item() < 1e-10


def test_nz_zero_field_gives_zero():
    """Zero B0 → zero Hamiltonian."""
    sys = SpinSystem(S=[0.5], Nucs=['1H'], A=[[20.0, 20.0, 20.0]])
    H = ham_nz(sys, B0=[0.0, 0.0, 0.0])
    assert H.abs().max().item() < 1e-14
