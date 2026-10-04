"""Tests for Tier 3a: OAM Hamiltonians (ham_cf, ham_oz, ham_so) + SpinSystem OAM extension.

Tests cover:
- SpinSystem with orbital angular momenta (L, gL, soc, CF fields)
- Crystal-field Hamiltonian (ham_cf)
- Orbital Zeeman Hamiltonian (ham_oz)
- Spin-orbit coupling Hamiltonian (ham_so)
- Integration with top-level ham() assembler
"""
import math

import numpy as np
import pytest
import torch

from torchspin import SpinSystem, ham, sop
from torchspin.ham_cf import ham_cf
from torchspin.ham_oz import ham_oz
from torchspin.ham_so import ham_so
from torchspin.stev import stev
from torchspin.constants import BMAGN, PLANCK


# ============================================================================
# SpinSystem OAM Extension
# ============================================================================

class TestSpinSystemOAM:
    """SpinSystem with orbital angular momenta."""

    def test_spins_with_oam(self):
        """Spins list includes OAMs after electrons and nuclei."""
        sys = SpinSystem(S=[0.5], L=[1])
        assert sys.Spins == [0.5, 1.0]
        assert sys.nL == 1
        assert sys.nElectrons == 1
        assert sys.nStates == 2 * 3  # (2*0.5+1) * (2*1+1) = 6

    def test_spins_with_oam_and_nuclei(self):
        """OAMs come after nuclei in Spins list."""
        sys = SpinSystem(S=[0.5], Nucs=['1H'], L=[2])
        assert sys.Spins == [0.5, 0.5, 2.0]  # electron, 1H (I=1/2), L=2
        assert sys.nL == 1
        assert sys.nNuclei == 1
        assert sys.nStates == 2 * 2 * 5  # 20

    def test_no_oam_unchanged(self):
        """Without L, Spins and nStates are unchanged."""
        sys = SpinSystem(S=[0.5], Nucs=['14N'])
        assert sys.Spins == [0.5, 1.0]
        assert sys.nL == 0
        assert sys.nStates == 2 * 3

    def test_default_gL(self):
        """gL defaults to 1.0 per OAM."""
        sys = SpinSystem(S=[0.5], L=[1, 2])
        assert sys.gL == [1.0, 1.0]

    def test_custom_gL(self):
        """Custom gL values are preserved."""
        sys = SpinSystem(S=[0.5], L=[1], gL=[0.5])
        assert sys.gL == [0.5]

    def test_soc_normalization(self):
        """soc is normalized to 2D tensor."""
        sys = SpinSystem(S=[0.5], L=[1], soc=[100.0])
        assert sys.soc.shape == (1, 1)
        assert sys.soc[0, 0].item() == 100.0

    def test_scalar_L(self):
        """Scalar L is wrapped in a list."""
        sys = SpinSystem(S=[0.5], L=1)
        assert sys.L == [1.0]
        assert sys.nL == 1

    def test_multiple_oams(self):
        """Multiple OAMs expand Spins correctly."""
        sys = SpinSystem(S=[0.5, 1.0], L=[1, 2])
        assert sys.Spins == [0.5, 1.0, 1.0, 2.0]
        assert sys.nL == 2
        assert sys.nStates == 2 * 3 * 3 * 5  # 90


# ============================================================================
# ham_cf — Crystal-Field Hamiltonian
# ============================================================================

class TestHamCF:
    """Crystal-field Hamiltonian tests."""

    def test_no_oam_returns_zero(self):
        """ham_cf returns zero matrix when no OAMs are present."""
        sys = SpinSystem(S=[0.5])
        H = ham_cf(sys)
        assert torch.allclose(H, torch.zeros_like(H))

    def test_single_cf2_term(self):
        """Single CF2 term matches direct Stevens operator call."""
        # S=0.5, L=2: CF2 with only O_2^0 non-zero
        # CF2 shape: (nL, 5), columns for q = 2, 1, 0, -1, -2
        coeff = 100.0
        sys = SpinSystem(S=[0.5], L=[2], CF2=[[0, 0, coeff, 0, 0]])
        H = ham_cf(sys)

        # The OAM is spin index 2 (1-based) in Spins = [0.5, 2.0]
        O20 = stev(sys.Spins, k=2, q=0, iSpin=2)
        H_expected = coeff * O20
        H_expected = (H_expected + H_expected.conj().T) / 2
        assert torch.allclose(H, H_expected, atol=1e-10)

    def test_hermitian(self):
        """Crystal-field Hamiltonian is Hermitian."""
        sys = SpinSystem(S=[0.5], L=[3],
                         CF2=[[10, 20, 30, 40, 50]],
                         CF4=[[1, 2, 3, 4, 5, 6, 7, 8, 9]])
        H = ham_cf(sys)
        assert torch.allclose(H, H.conj().T, atol=1e-10)

    def test_L_less_than_1_returns_zero(self):
        """ham_cf skips OAMs with L < 1."""
        sys = SpinSystem(S=[0.5], L=[0.5], CF2=[[10, 20, 30, 40, 50]])
        H = ham_cf(sys)
        assert torch.allclose(H, torch.zeros_like(H))

    def test_cf4_term(self):
        """CF4 rank-4 Stevens operator term."""
        coeff = 50.0
        # CF4 shape: (nL, 9), columns for q = 4, 3, 2, 1, 0, -1, -2, -3, -4
        cf4 = [[0, 0, 0, 0, coeff, 0, 0, 0, 0]]  # Only O_4^0
        sys = SpinSystem(S=[0.5], L=[3], CF4=cf4)
        H = ham_cf(sys)

        O40 = stev(sys.Spins, k=4, q=0, iSpin=2)
        H_expected = coeff * O40
        H_expected = (H_expected + H_expected.conj().T) / 2
        assert torch.allclose(H, H_expected, atol=1e-10)

    def test_idxL_selection(self):
        """Selecting specific OAMs via idxL."""
        sys = SpinSystem(S=[0.5], L=[2, 3],
                         CF2=[[100, 0, 0, 0, 0], [200, 0, 0, 0, 0]])
        H_all = ham_cf(sys)
        H_first = ham_cf(sys, idxL=[1])
        H_second = ham_cf(sys, idxL=[2])
        # Should add up
        assert torch.allclose(H_all, H_first + H_second, atol=1e-10)

    def test_invalid_idxL_raises(self):
        """Invalid OAM index raises ValueError."""
        sys = SpinSystem(S=[0.5], L=[2])
        with pytest.raises(ValueError, match="out of range"):
            ham_cf(sys, idxL=[3])


# ============================================================================
# ham_oz — Orbital Zeeman Hamiltonian
# ============================================================================

class TestHamOZ:
    """Orbital Zeeman Hamiltonian tests."""

    def test_no_oam_returns_zero(self):
        """ham_oz returns zero operators when no OAMs present."""
        sys = SpinSystem(S=[0.5])
        mux, muy, muz = ham_oz(sys)
        assert torch.allclose(mux, torch.zeros_like(mux))
        assert torch.allclose(muy, torch.zeros_like(muy))
        assert torch.allclose(muz, torch.zeros_like(muz))

    def test_muz_proportional_to_Lz(self):
        """muz should be proportional to Lz operator."""
        sys = SpinSystem(S=[0.5], L=[1], gL=[1.0])
        mux, muy, muz = ham_oz(sys)

        # Expected: pre = -gL * BMAGN / PLANCK * 1e-9
        pre = -1.0 * BMAGN / PLANCK * 1e-9

        # Lz is sop for spin 2 (1-based), component z(3)
        Lz = sop(sys.Spins, [[2, 3]])
        expected_muz = pre * Lz
        assert torch.allclose(muz, expected_muz, atol=1e-10)

    def test_returns_H_with_B0(self):
        """ham_oz returns full Hamiltonian when B0 is given."""
        sys = SpinSystem(S=[0.5], L=[1], gL=[1.0])
        B0 = [0.0, 0.0, 340.0]
        H = ham_oz(sys, B0=B0)
        assert H.shape == (sys.nStates, sys.nStates)
        # H should be Hermitian
        assert torch.allclose(H, H.conj().T, atol=1e-10)

    def test_custom_gL(self):
        """Custom gL scales the moment operators."""
        sys1 = SpinSystem(S=[0.5], L=[1], gL=[1.0])
        sys2 = SpinSystem(S=[0.5], L=[1], gL=[2.0])
        _, _, muz1 = ham_oz(sys1)
        _, _, muz2 = ham_oz(sys2)
        assert torch.allclose(muz2, 2.0 * muz1, atol=1e-10)

    def test_L_zero_skipped(self):
        """OAM with L=0 produces zero contribution."""
        sys = SpinSystem(S=[0.5], L=[0])
        mux, muy, muz = ham_oz(sys)
        assert torch.allclose(mux, torch.zeros_like(mux))

    def test_oam_selection(self):
        """Selecting specific OAMs via oam parameter."""
        sys = SpinSystem(S=[0.5], L=[1, 2], gL=[1.0, 1.0])
        _, _, muz_all = ham_oz(sys)
        _, _, muz_1 = ham_oz(sys, oam=[1])
        _, _, muz_2 = ham_oz(sys, oam=[2])
        assert torch.allclose(muz_all, muz_1 + muz_2, atol=1e-10)


# ============================================================================
# ham_so — Spin-Orbit Coupling Hamiltonian
# ============================================================================

class TestHamSO:
    """Spin-orbit coupling Hamiltonian tests."""

    def test_no_oam_returns_zero(self):
        """ham_so returns zero when no OAMs present."""
        sys = SpinSystem(S=[0.5])
        H = ham_so(sys)
        assert torch.allclose(H, torch.zeros_like(H))

    def test_no_soc_returns_zero(self):
        """ham_so returns zero when soc is None."""
        sys = SpinSystem(S=[0.5], L=[1])
        H = ham_so(sys)
        assert torch.allclose(H, torch.zeros_like(H))

    def test_linear_soc_hermitian(self):
        """Linear spin-orbit coupling produces Hermitian H."""
        sys = SpinSystem(S=[1.0], L=[1], soc=[[200.0]])
        H = ham_so(sys)
        assert torch.allclose(H, H.conj().T, atol=1e-10)

    def test_linear_soc_is_SdotL(self):
        """Linear soc * S·L: verify by comparing eigenvalues."""
        lam = 100.0
        sys = SpinSystem(S=[1.0], L=[1], soc=[[lam]])
        H = ham_so(sys)

        # Build S·L = Sx*Lx + Sy*Ly + Sz*Lz manually
        spins = sys.Spins  # [1.0, 1.0]
        SL = torch.zeros_like(H)
        for c in range(1, 4):
            SL = SL + sop(spins, [[1, c], [2, c]])

        H_expected = lam * SL
        assert torch.allclose(H, H_expected, atol=1e-10)

    def test_quadratic_soc(self):
        """Quadratic soc: H = lam1*(S·L) + lam2*(S·L)^2."""
        lam1, lam2 = 100.0, 10.0
        sys = SpinSystem(S=[1.0], L=[1], soc=[[lam1, lam2]])
        H = ham_so(sys)

        spins = sys.Spins
        SL = torch.zeros(sys.nStates, sys.nStates, dtype=torch.complex128)
        for c in range(1, 4):
            SL = SL + sop(spins, [[1, c], [2, c]])

        H_expected = lam1 * SL + lam2 * (SL @ SL)
        assert torch.allclose(H, H_expected, atol=1e-10)

    def test_zero_soc_returns_zero(self):
        """All-zero soc returns zero matrix."""
        sys = SpinSystem(S=[0.5], L=[1], soc=[[0.0, 0.0]])
        H = ham_so(sys)
        assert torch.allclose(H, torch.zeros_like(H))

    def test_eSpins_selection(self):
        """eSpins parameter selects which electrons to include."""
        sys = SpinSystem(S=[0.5, 1.0], L=[1, 1], soc=[[100.0], [200.0]])
        H_all = ham_so(sys)
        H_1 = ham_so(sys, eSpins=[1])
        H_2 = ham_so(sys, eSpins=[2])
        assert torch.allclose(H_all, H_1 + H_2, atol=1e-10)

    def test_invalid_eSpin_raises(self):
        """Invalid electron spin index raises ValueError."""
        sys = SpinSystem(S=[0.5], L=[1], soc=[[100.0]])
        with pytest.raises(ValueError, match="out of range"):
            ham_so(sys, eSpins=[3])


# ============================================================================
# ham.py Integration
# ============================================================================

class TestHamIntegration:
    """Integration of OAM Hamiltonians with top-level ham()."""

    def test_ham_with_oam_runs(self):
        """ham() works with a spin system that has OAMs."""
        sys = SpinSystem(S=[0.5], L=[1], gL=[1.0],
                         soc=[[100.0]],
                         CF2=[[0, 0, 50.0, 0, 0]])
        H0, mux, muy, muz = ham(sys)
        assert H0.shape == (sys.nStates, sys.nStates)
        assert mux.shape == (sys.nStates, sys.nStates)

    def test_ham_with_B0_and_oam(self):
        """ham() returns full Hamiltonian with B0 and OAMs."""
        sys = SpinSystem(S=[0.5], L=[1], gL=[1.0],
                         soc=[[100.0]],
                         CF2=[[0, 0, 50.0, 0, 0]])
        H = ham(sys, B0=[0.0, 0.0, 340.0])
        assert H.shape == (sys.nStates, sys.nStates)
        assert torch.allclose(H, H.conj().T, atol=1e-10)

    def test_ham_oam_contributes_to_H0(self):
        """CF and SO terms appear in H0."""
        sys_no_oam = SpinSystem(S=[0.5])
        sys_with_oam = SpinSystem(S=[0.5], L=[1],
                                   soc=[[100.0]],
                                   CF2=[[0, 0, 50.0, 0, 0]])
        H0_no, _, _, _ = ham(sys_no_oam)
        H0_with, _, _, _ = ham(sys_with_oam)
        # H0_with should be larger (different dimensions)
        assert H0_no.shape[0] < H0_with.shape[0]
        # H0_with should be non-zero (CF + SO contribute)
        assert not torch.allclose(H0_with, torch.zeros_like(H0_with))

    def test_ham_oz_contributes_to_moment_operators(self):
        """Orbital Zeeman contributes to moment operators."""
        sys = SpinSystem(S=[0.5], L=[1], gL=[1.0])
        _, _, _, muz = ham(sys)
        # muz should have contributions from both electron and orbital Zeeman
        # For S=0.5, electron muz has 6 entries; orbital adds Lz terms
        assert not torch.allclose(muz, torch.zeros_like(muz))

    def test_ham_no_oam_unchanged(self):
        """ham() output is unchanged for systems without OAMs."""
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        H0, mux, muy, muz = ham(sys)
        # Should be standard electron Zeeman only
        assert H0.shape == (2, 2)
        # H0 should be zero (no ZFS, no hyperfine, etc.)
        assert torch.allclose(H0, torch.zeros_like(H0))
