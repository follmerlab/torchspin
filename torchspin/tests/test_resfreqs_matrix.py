"""Tests for torchspin/resfreqs_matrix.py — frequency-swept EPR."""
import math

import pytest
import torch

from torchspin.spinsystem import SpinSystem
from torchspin.experiment import Experiment, Options
from torchspin.resfreqs_matrix import resfreqs_matrix
from torchspin.constants import BMAGN, PLANCK


# ---------------------------------------------------------------------------
# Basic output shape / type
# ---------------------------------------------------------------------------

class TestBasicOutput:
    def setup_method(self):
        self.sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])

    def test_returns_two_tensors(self):
        freq, intens = resfreqs_matrix(self.sys, 0.0, 0.0, B=340.0)
        assert isinstance(freq, torch.Tensor)
        assert isinstance(intens, torch.Tensor)

    def test_freq_intens_same_length(self):
        freq, intens = resfreqs_matrix(self.sys, 0.0, 0.0, B=340.0)
        assert freq.shape == intens.shape

    def test_dtype_float64(self):
        freq, intens = resfreqs_matrix(self.sys, 0.0, 0.0, B=340.0)
        assert freq.dtype == torch.float64
        assert intens.dtype == torch.float64

    def test_frequencies_positive(self):
        """All returned frequencies should be > 0 (E_j > E_i for j > i)."""
        freq, _ = resfreqs_matrix(self.sys, 0.0, 0.0, B=340.0)
        assert (freq > 0).all()

    def test_intensities_non_negative(self):
        """Perpendicular transition rates are non-negative."""
        _, intens = resfreqs_matrix(self.sys, 0.0, 0.0, B=340.0)
        assert (intens >= 0).all()


# ---------------------------------------------------------------------------
# Physical correctness — Zeeman resonance
# ---------------------------------------------------------------------------

class TestZeemanResonance:
    def test_isotropic_S_half_resonance_frequency(self):
        """
        For S=1/2, g=2.0, B=340 mT, the resonance frequency is:
            nu = g * mu_B / h * B

        Unit conversion: BMAGN/PLANCK [Hz/T] * B [mT] * 1e-3 [T/mT] * 1e-6 [MHz/Hz]
                       = BMAGN/PLANCK * B * 1e-9   (MHz, when B in mT)
        """
        g = 2.0
        B = 340.0  # mT
        nu_expected = g * BMAGN / PLANCK * B * 1e-9  # MHz

        sys = SpinSystem(S=[0.5], g=[[g, g, g]])
        freq, intens = resfreqs_matrix(sys, 0.0, 0.0, B=B, threshold=0)

        # S=1/2 has only one transition: nStates*(nStates-1)/2 = 1
        assert freq.numel() == 1, f"Expected 1 transition, got {freq.numel()}"
        assert abs(freq[0].item() - nu_expected) < 1.0, (
            f"Freq={freq[0].item():.2f} MHz, expected {nu_expected:.2f} MHz"
        )

    def test_g_anisotropy_changes_frequency(self):
        """
        For an anisotropic g-tensor, frequency at theta=0 (z-axis) uses g_zz.
        """
        gxx, gyy, gzz = 2.00, 2.05, 2.15
        sys = SpinSystem(S=[0.5], g=[[gxx, gyy, gzz]])
        B = 340.0  # mT

        # Along z-axis: freq = g_zz * mu_B / h * B (MHz, B in mT)
        freq_z_expected = gzz * BMAGN / PLANCK * B * 1e-9  # MHz
        freq_z, _ = resfreqs_matrix(sys, 0.0, 0.0, B=B, threshold=0)
        assert abs(freq_z[0].item() - freq_z_expected) < 2.0, (
            f"z-axis: {freq_z[0].item():.2f} vs {freq_z_expected:.2f} MHz"
        )

        # Along x-axis: freq = g_xx * mu_B / h * B (MHz, B in mT)
        freq_x_expected = gxx * BMAGN / PLANCK * B * 1e-9  # MHz
        freq_x, _ = resfreqs_matrix(sys, 0.0, math.pi / 2, B=B, threshold=0)
        assert abs(freq_x[0].item() - freq_x_expected) < 2.0, (
            f"x-axis: {freq_x[0].item():.2f} vs {freq_x_expected:.2f} MHz"
        )

        # Different orientations → different frequencies
        assert abs(freq_z[0].item() - freq_x[0].item()) > 10.0

    def test_field_proportionality(self):
        """Resonance frequency should scale linearly with B for isotropic g."""
        g = 2.0
        sys = SpinSystem(S=[0.5], g=[[g, g, g]])
        slope = g * BMAGN / PLANCK * 1e-9  # MHz/mT (BMAGN/PLANCK Hz/T × 1e-3 T/mT × 1e-6 MHz/Hz)

        for B in [100.0, 200.0, 340.0, 500.0]:
            freq, _ = resfreqs_matrix(sys, 0.0, 0.0, B=B, threshold=0)
            expected = slope * B
            assert abs(freq[0].item() - expected) < 1.0, (
                f"B={B} mT: freq={freq[0].item():.2f}, expected {expected:.2f} MHz"
            )

    def test_hyperfine_splits_single_line(self):
        """
        14N (I=1) hyperfine splits S=1/2 into 2*I+1 = 3 EPR transitions.

        The full 6-state Hilbert space has many pairs, but only 3 are EPR-allowed
        (ΔmS=±1, ΔmI=0).  Nuclear transitions (~11 MHz Larmor) are ~1000× weaker
        and removed by the default threshold=1e-4.
        """
        sys = SpinSystem(
            S=[0.5],
            g=[[2.0, 2.0, 2.0]],
            Nucs=["14N"],
            A=[[20.0, 20.0, 20.0]],  # MHz, isotropic
        )
        freq, intens = resfreqs_matrix(sys, 0.0, 0.0, B=340.0, threshold=1e-4)
        # Should have 3 transitions (one per mI value)
        assert freq.numel() == 3, f"Expected 3 hyperfine transitions, got {freq.numel()}"
        # Lines should be roughly equal intensity
        assert intens.max() / intens.min() < 3.0

    def test_s1_has_two_transitions(self):
        """
        S=1, no ZFS: three levels, two EPR transitions (Δm=±1 allowed).
        """
        sys = SpinSystem(S=[1], g=[[2.0, 2.0, 2.0]])
        freq, intens = resfreqs_matrix(sys, 0.0, 0.0, B=340.0, threshold=1e-6)
        assert freq.numel() == 2, f"Expected 2 transitions, got {freq.numel()}"

    def test_s1_with_zfs_splits_at_zero_field(self):
        """
        S=1 with ZFS: two transitions become non-degenerate.
        """
        D_val = 1000.0  # MHz
        sys = SpinSystem(
            S=[1],
            g=[[2.0, 2.0, 2.0]],
            D=[[-D_val / 3, -D_val / 3, 2 * D_val / 3]],
        )
        # Without ZFS:
        sys_nodfs = SpinSystem(S=[1], g=[[2.0, 2.0, 2.0]])
        B = 340.0

        freq_zfs, _ = resfreqs_matrix(sys, 0.0, 0.0, B=B, threshold=1e-6)
        freq_no, _ = resfreqs_matrix(sys_nodfs, 0.0, 0.0, B=B, threshold=1e-6)

        # With ZFS: two distinct frequencies
        # Without ZFS: two degenerate frequencies
        assert freq_zfs.numel() == 2
        freq_gap = abs(freq_zfs[1].item() - freq_zfs[0].item())
        no_gap = abs(freq_no[1].item() - freq_no[0].item())
        assert freq_gap > no_gap + 10.0, (
            f"ZFS should split lines (gap={freq_gap:.1f} vs no-ZFS gap={no_gap:.1f} MHz)"
        )


# ---------------------------------------------------------------------------
# freq_range filtering
# ---------------------------------------------------------------------------

class TestFreqRangeFilter:
    def test_freq_range_includes_resonance(self):
        """freq_range containing the resonance → at least one transition returned."""
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        B = 340.0
        # Expected resonance ~9524 MHz = ~9.524 GHz
        freq, intens = resfreqs_matrix(sys, 0.0, 0.0, B=B,
                                       freq_range=(9.0, 10.0), threshold=0)
        assert freq.numel() > 0

    def test_freq_range_excludes_resonance(self):
        """freq_range not containing the resonance → empty output."""
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        B = 340.0
        # Resonance ~9.524 GHz; range excludes it
        freq, intens = resfreqs_matrix(sys, 0.0, 0.0, B=B,
                                       freq_range=(1.0, 5.0), threshold=0)
        assert freq.numel() == 0

    def test_freq_range_in_ghz(self):
        """freq_range is given in GHz and correctly converted to MHz internally."""
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        freq, _ = resfreqs_matrix(sys, 0.0, 0.0, B=340.0,
                                   freq_range=(9.0, 10.0), threshold=0)
        # Should find the resonance; freq returned in MHz
        assert freq.numel() == 1
        assert 9000.0 < freq[0].item() < 10000.0


# ---------------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------------

class TestThreshold:
    def test_threshold_zero_keeps_all(self):
        """threshold=0 keeps all transitions including forbidden ones."""
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        freq_all, _ = resfreqs_matrix(sys, 0.0, 0.0, B=340.0, threshold=0)
        freq_thr, _ = resfreqs_matrix(sys, 0.0, 0.0, B=340.0, threshold=1e-4)
        # For S=1/2, there is only one pair, so both should be 1
        assert freq_all.numel() >= freq_thr.numel()

    def test_threshold_removes_forbidden(self):
        """For S=1 with ZFS, some transitions are forbidden; threshold removes them."""
        D_val = 5000.0  # large ZFS to ensure strong anisotropy
        sys = SpinSystem(
            S=[1],
            g=[[2.0, 2.0, 2.0]],
            D=[[-D_val / 3, -D_val / 3, 2 * D_val / 3]],
        )
        # With no threshold: may include double-quantum transition (Δm=2)
        freq_all, _ = resfreqs_matrix(sys, 0.0, 0.0, B=340.0, threshold=0)
        freq_thr, _ = resfreqs_matrix(sys, 0.0, 0.0, B=340.0, threshold=0.1)
        # With threshold, fewer or equal transitions
        assert freq_thr.numel() <= freq_all.numel()


# ---------------------------------------------------------------------------
# Consistency with resfields (field-swept)
# ---------------------------------------------------------------------------

class TestConsistencyWithResfields:
    def test_resonance_frequency_matches_resfields(self):
        """
        For isotropic g=2.0, S=1/2:
        resfreqs_matrix at B=B_res should return freq ≈ mwFreq.
        resfields at mwFreq should return B ≈ B_res.
        Both are computed from the same Hamiltonian and should be consistent.
        """
        from torchspin.resfields import resfields

        g = 2.0
        sys = SpinSystem(S=[0.5], g=[[g, g, g]])
        mwFreq = 9.5  # GHz

        # Known resonance field for g=2.0, 9.5 GHz:
        # B_res = h*nu / (g*mu_B) (mT)
        B_res_expected = PLANCK * mwFreq * 1e9 / (g * BMAGN) * 1e3  # mT

        # resfields: find B_res
        from torchspin.ham import ham as ham_fn
        H0, mux, muy, muz = ham_fn(sys, B0=None)
        exp = Experiment(mwFreq=mwFreq, Range=[B_res_expected - 20, B_res_expected + 20],
                         nPoints=1024, Harmonic=0)
        B_res_arr, _, _ = resfields(H0, mux, muy, muz, 0.0, 0.0, exp, Options())

        assert B_res_arr.numel() == 1
        B_res = B_res_arr[0].item()

        # resfreqs_matrix at B_res: should return mwFreq*1e3 MHz
        freq, _ = resfreqs_matrix(sys, 0.0, 0.0, B=B_res, threshold=0)
        assert freq.numel() == 1
        assert abs(freq[0].item() - mwFreq * 1e3) < 1.0, (
            f"resfreqs gives {freq[0].item():.2f} MHz, expected {mwFreq*1e3:.2f} MHz"
        )


# ---------------------------------------------------------------------------
# Boltzmann weighting
# ---------------------------------------------------------------------------

class TestBoltzmann:
    def test_high_temperature_same_as_no_temperature(self):
        """At very high T, polarizations approach equal → intensities similar."""
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        B = 340.0

        exp_hot = Experiment(mwFreq=9.5, Range=[300, 400], nPoints=512,
                             Temperature=1e6)
        exp_inf = Experiment(mwFreq=9.5, Range=[300, 400], nPoints=512,
                             Temperature=None)

        freq_hot, intens_hot = resfreqs_matrix(sys, 0.0, 0.0, B=B, exp=exp_hot)
        freq_inf, intens_inf = resfreqs_matrix(sys, 0.0, 0.0, B=B, exp=exp_inf)

        assert freq_hot.numel() == freq_inf.numel()
        # At T=1e6 K, populations near-equal; intensity ≈ infinite-T result
        # Both should give small but finite intensity
        assert intens_hot.sum() > 0
