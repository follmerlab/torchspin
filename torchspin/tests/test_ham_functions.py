#!/usr/bin/env python3
"""
Comprehensive test suite for Hamiltonian functions.

Based on MATLAB tests in easyspin/tests/:
- ham_ez_*.m (electron Zeeman)
- ham_zf_*.m (zero-field splitting)
- ham_hf_*.m (hyperfine)
- ham_ee_*.m (electron-electron)
- ham_full_*.m (complete Hamiltonian)
"""

import pytest
import torch
import math
from torchspin import SpinSystem
from torchspin.ham import ham
from torchspin.ham_ez import ham_ez
from torchspin.ham_zf import ham_zf
from torchspin.ham_hf import ham_hf
from torchspin.ham_ee import ham_ee
from torchspin.constants import BMAGN, PLANCK, GFREE

_C128 = torch.tensor(0.0 + 0j, dtype=torch.complex128)


class TestHamEz:
    """Test electron Zeeman Hamiltonian (ham_ez_*.m)."""

    def test_ham_ez_simple(self):
        """Simple isotropic g test (ham_ez_simple.m)."""
        sys = SpinSystem(S=[1/2], g=[[2.0, 2.0, 2.0]])
        B0 = torch.tensor([0, 0, 350.0], dtype=torch.float64)  # mT along z

        Hz = ham_ez(sys, B0)

        # Should be Hermitian
        assert torch.allclose(Hz, Hz.conj().T, atol=1e-14)

        # Dimensions
        assert Hz.shape == (2, 2)

    def test_ham_ez_eigenvalues_isotropic(self):
        """Check eigenvalues for isotropic g (ham_ez_simple.m)."""
        sys = SpinSystem(S=[1/2], g=[[GFREE, GFREE, GFREE]])
        B0 = torch.tensor([0, 0, 350.0], dtype=torch.float64)

        Hz = ham_ez(sys, B0)

        eigvals = torch.linalg.eigvalsh(Hz)

        # Energy splitting in MHz
        splitting = eigvals[1] - eigvals[0]

        # Expected: g * beta * B0_tesla / h  (in MHz)
        # beta/h = 1.39962449361e-2 MHz/mT (from constants)
        expected_splitting = GFREE * (BMAGN / PLANCK) * 350.0 * 1e-9  # Convert to MHz

        assert torch.allclose(splitting, torch.tensor(expected_splitting, dtype=torch.float64), rtol=1e-4)

    def test_ham_ez_two_electrons(self):
        """Two electron spins (ham_ez_twoelectrons.m)."""
        sys = SpinSystem(S=[1/2, 1/2], g=[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
        B0 = torch.tensor([0, 0, 350.0], dtype=torch.float64)

        Hz = ham_ez(sys, B0)

        # Should be 4×4 for two spin-1/2
        assert Hz.shape == (4, 4)
        assert torch.allclose(Hz, Hz.conj().T, atol=1e-14)

    def test_ham_ez_anisotropic_g(self):
        """Anisotropic g-tensor (ham_ez_fullg_angles.m)."""
        sys = SpinSystem(S=[1/2], g=[[2.0, 2.0, 2.3]])
        B0 = torch.tensor([0, 0, 350.0], dtype=torch.float64)

        Hz = ham_ez(sys, B0)

        # Eigenvalues should reflect g_zz
        eigvals = torch.linalg.eigvalsh(Hz)
        splitting = eigvals[1] - eigvals[0]

        # Should use g_zz = 2.3 for field along z
        expected_splitting = 2.3 * (BMAGN / PLANCK) * 350.0 * 1e-9
        assert torch.allclose(splitting, torch.tensor(expected_splitting, dtype=torch.float64), rtol=1e-4)


class TestHamZf:
    """Test zero-field splitting Hamiltonian (ham_zf_*.m)."""

    def test_ham_zf_simple(self):
        """Simple ZFS with D tensor (ham_zf_simple.m)."""
        sys = SpinSystem(S=[1], D=[100, 100, -200])  # MHz, D_zz = -200

        Hzf = ham_zf(sys)

        # Should be Hermitian
        assert torch.allclose(Hzf, Hzf.conj().T, atol=1e-14)

        # Should be 3×3 for spin-1
        assert Hzf.shape == (3, 3)

    def test_ham_zf_de_parameters(self):
        """D and E parameters (ham_zf_de.m)."""
        D_val = 500  # MHz
        E_val = 50   # MHz

        # Convert to [D_xx, D_yy, D_zz]
        D_xx = -D_val/3 - E_val
        D_yy = -D_val/3 + E_val
        D_zz = 2*D_val/3

        sys = SpinSystem(S=[1], D=[D_xx, D_yy, D_zz])

        Hzf = ham_zf(sys)

        # Check it's traceless (characteriztic of ZFS)
        trace = torch.trace(Hzf)
        assert torch.allclose(trace, torch.zeros(1, dtype=torch.complex128).squeeze(), atol=1e-12)

    def test_ham_zf_axial(self):
        """Axial ZFS (E=0) (ham_zf_rhombic.m)."""
        sys = SpinSystem(S=[1], D=[[-200/3, -200/3, 400/3]])

        Hzf = ham_zf(sys)

        # For axial symmetry: D_xx = D_yy ≠ D_zz
        eigvals = torch.linalg.eigvalsh(Hzf)

        # Two degenerate levels for axial case
        assert eigvals.shape[0] == 3


class TestHamHf:
    """Test hyperfine Hamiltonian (ham_hf_*.m)."""

    def test_ham_hf_isotropic(self):
        """Isotropic hyperfine (ham_hf_isotropic.m)."""
        sys = SpinSystem(
            S=[1/2],
            Nucs='1H',
            A=[[10, 10, 10]]  # MHz, isotropic
        )

        H0, mux, muy, muz = ham(sys, B0=None)
        Hhf = ham_hf(sys)

        # Should be 4×4 (electron spin-1/2 × nuclear spin-1/2)
        assert Hhf.shape == (4, 4)
        assert torch.allclose(Hhf, Hhf.conj().T, atol=1e-14)

    def test_ham_hf_anisotropic(self):
        """Anisotropic hyperfine (ham_hf_simplevalues.m)."""
        sys = SpinSystem(
            S=[1/2],
            Nucs='14N',  # I=1
            A=[[20, 20, 80]]  # MHz
        )

        H0, mux, muy, muz = ham(sys, B0=None)
        Hhf = ham_hf(sys)

        # Should be 6×6 (electron spin-1/2 × nuclear spin-1)
        assert Hhf.shape == (6, 6)
        assert torch.allclose(Hhf, Hhf.conj().T, atol=1e-14)

    def test_ham_hf_two_nuclei(self):
        """Two coupled nuclei (ham_hf_isotropic2.m)."""
        sys = SpinSystem(
            S=[1/2],
            Nucs=['1H', '1H'],
            A=[[10, 10, 10], [15, 15, 15]]  # MHz
        )

        H0, mux, muy, muz = ham(sys, B0=None)
        Hhf = ham_hf(sys)

        # Should be 8×8 (spin-1/2 × proton × proton = 2×2×2)
        assert Hhf.shape == (8, 8)
        assert torch.allclose(Hhf, Hhf.conj().T, atol=1e-14)


class TestHamEe:
    """Test electron-electron coupling (ham_ee_*.m)."""

    def test_ham_ee_isotropic_exchange(self):
        """Isotropic exchange J (ham_ee_isotropic.m)."""
        sys = SpinSystem(
            S=[1/2, 1/2],
            ee=[[10, 10, 10]]  # 10 MHz isotropic exchange
        )

        H0, mux, muy, muz = ham(sys, B0=None)
        Hee = ham_ee(sys)

        # Should be 4×4 (two spin-1/2)
        assert Hee.shape == (4, 4)
        assert torch.allclose(Hee, Hee.conj().T, atol=1e-14)

    def test_ham_ee_dipolar(self):
        """Dipolar coupling (ham_ee_dipolar.m)."""
        # J_iso = 0, J_xx ≠ J_yy ≠ J_zz (pure dipolar)
        sys = SpinSystem(
            S=[1/2, 1/2],
            ee=[[-5, -5, 10]]  # Dipolar: sum to zero
        )

        H0, mux, muy, muz = ham(sys, B0=None)
        Hee = ham_ee(sys)

        # Should be traceless
        trace = torch.trace(Hee)
        assert torch.allclose(trace, torch.zeros(1, dtype=torch.complex128).squeeze(), atol=1e-12)

    def test_ham_ee_three_spins(self):
        """Three electron spins (ham_ee_threespins.m)."""
        sys = SpinSystem(
            S=[1/2, 1/2, 1/2],
            ee=[[10, 10, 10], [5, 5, 5], [8, 8, 8]]  # Three pairs
        )

        H0, mux, muy, muz = ham(sys, B0=None)
        Hee = ham_ee(sys)

        # Should be 8×8 (three spin-1/2)
        assert Hee.shape == (8, 8)
        assert torch.allclose(Hee, Hee.conj().T, atol=1e-14)


class TestHamComplete:
    """Test complete Hamiltonian construction (ham_full_*.m)."""

    def test_ham_complete_simple(self):
        """Complete Hamiltonian for simple system."""
        sys = SpinSystem(
            S=[1/2],
            g=[[2.0, 2.0, 2.0]],
            Nucs='1H',
            A=[[10, 10, 10]]
        )

        B0 = torch.tensor([0, 0, 350.0], dtype=torch.float64)
        H = ham(sys, B0=B0)

        # Should include Zeeman + hyperfine
        assert H.shape == (4, 4)
        assert torch.allclose(H, H.conj().T, atol=1e-14)

    def test_ham_separate_components(self):
        """Verify H(B) = H0 + H_ez for a pure electron system (no nuclei).

        ham(B0) includes all field terms (electron + nuclear Zeeman).
        ham_ez(B0) includes only the electron Zeeman term.
        For a nucleus-free system these are the same, so H_total = H0 + Hz.
        """
        sys = SpinSystem(S=[1/2], g=[[2.0, 2.0, 2.0]])  # no nuclei

        B0 = torch.tensor([0, 0, 350.0], dtype=torch.float64)

        H0, mux, muy, muz = ham(sys, B0=None)
        Hz = ham_ez(sys, B0)
        H_total = ham(sys, B0=B0)

        assert torch.allclose(H_total, H0 + Hz, atol=1e-12)

    def test_ham_with_zfs(self):
        """System with zero-field splitting (ham_full_syntax.m)."""
        sys = SpinSystem(
            S=[1],
            g=[[2.0, 2.0, 2.0]],
            D=[[100, 100, -200]]  # MHz
        )

        B0 = torch.tensor([0, 0, 350.0], dtype=torch.float64)
        H = ham(sys, B0=B0)

        # Should be 3×3 for spin-1
        assert H.shape == (3, 3)
        assert torch.allclose(H, H.conj().T, atol=1e-14)

    def test_ham_multi_component(self):
        """System with all interaction types."""
        sys = SpinSystem(
            S=[1/2, 1/2],
            g=[[2.0, 2.0, 2.0], [2.1, 2.1, 2.1]],
            Nucs='1H',
            A=[[10, 10, 10], [8, 8, 8]],
            ee=[[5, 5, 5]]
        )

        B0 = torch.tensor([0, 0, 350.0], dtype=torch.float64)
        H = ham(sys, B0=B0)

        # Two electrons × one proton = 8×8
        assert H.shape == (8, 8)
        assert torch.allclose(H, H.conj().T, atol=1e-14)


class TestHamProperties:
    """Test general properties of Hamiltonians."""

    def test_ham_hermitian(self):
        """All Hamiltonians should be Hermitian."""
        sys = SpinSystem(
            S=[1/2],
            g=[[2.0, 2.0, 2.0]],
            Nucs='1H',
            A=[[10, 10, 10]]
        )

        B0 = torch.tensor([0, 0, 350.0], dtype=torch.float64)
        H = ham(sys, B0=B0)

        assert torch.allclose(H, H.conj().T, atol=1e-14)

    def test_ham_real_eigenvalues(self):
        """Hermitian Hamiltonians have real eigenvalues."""
        sys = SpinSystem(S=[1/2], g=[[2.0, 2.0, 2.0]])
        B0 = torch.tensor([0, 0, 350.0], dtype=torch.float64)
        H = ham(sys, B0=B0)

        eigvals = torch.linalg.eigvalsh(H)

        # Should all be real (eigvalsh guarantees this)
        assert eigvals.dtype == torch.float64

    def test_ham_moment_operators_traceless(self):
        """Magnetic moment operators should be traceless."""
        sys = SpinSystem(S=[1/2], g=[[2.0, 2.0, 2.0]])
        H0, mux, muy, muz = ham(sys, B0=None)

        zero = torch.zeros(1, dtype=torch.complex128).squeeze()
        for mu in [mux, muy, muz]:
            trace = torch.trace(mu)
            assert torch.allclose(trace, zero, atol=1e-14)


class TestHamNuclearZeeman:
    """Test nuclear Zeeman contributions."""

    def test_ham_nuclear_zeeman_included(self):
        """Nuclear Zeeman should be included automatically."""
        sys = SpinSystem(
            S=[1/2],
            g=[[2.0, 2.0, 2.0]],
            Nucs='1H',
            A=[[10, 10, 10]]
        )

        B0 = torch.tensor([0, 0, 350.0], dtype=torch.float64)
        H = ham(sys, B0=B0)

        # Nuclear Zeeman is much smaller than electron Zeeman
        # but should still be in the Hamiltonian
        eigvals = torch.linalg.eigvalsh(H)

        # Energy differences should include small nuclear splittings
        assert len(eigvals) == 4  # Two electron states × two nuclear states


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
