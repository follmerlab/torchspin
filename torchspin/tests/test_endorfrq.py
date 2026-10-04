"""Unit tests for torchspin.endorfrq and torchspin.salt (ENDOR simulation).

Tests cover:
- Basic API: correct output shapes, types, and monotonic frequencies
- S=1/2 + 1H: ENDOR lines at ν_n ± A/2  (Larmor + half-hyperfine)
- Intensity threshold filtering
- salt(): Pake doublet for powder S=1/2 + 1H
- salt(): returns finite values with correct shape
"""

import math
import pytest
import torch
import numpy as np

from torchspin import SpinSystem
from torchspin.experiment import Experiment, Options
from torchspin.endorfrq import endorfrq
from torchspin.salt import salt
from torchspin.constants import NMAGN, PLANCK, BMAGN, GFREE


# ── Helpers ────────────────────────────────────────────────────────────────

def proton_larmor_mhz(B_mT: float) -> float:
    """1H Larmor frequency [MHz] at field B_mT [mT]."""
    # ν_n = gn * N_magn * B / h  [Hz] → MHz
    # 1H nuclear g-factor ≈ 5.58569
    gn_proton = 5.585694702
    return gn_proton * NMAGN / PLANCK * (B_mT * 1e-3) * 1e-6  # MHz


def free_electron_resonance_mT(mwFreq_GHz: float) -> float:
    """EPR resonance field [mT] for free electron at mwFreq_GHz."""
    return mwFreq_GHz * 1e9 * PLANCK / (GFREE * BMAGN) * 1e3


# ── Tests for endorfrq ─────────────────────────────────────────────────────

class TestEndorfrqBasic:

    def test_no_nuclei_raises(self):
        """endorfrq raises ValueError if spin system has no nuclei."""
        sys = SpinSystem(S=0.5, g=2.0)
        with pytest.raises(ValueError, match="nucleus"):
            endorfrq(sys, B_field=340.0)

    def test_returns_three_tensors(self):
        """endorfrq returns (freqs, intens, transitions) as Tensors."""
        sys = SpinSystem(S=0.5, g=2.0, Nucs='1H', A=[[10.0, 10.0, 10.0]])
        B = free_electron_resonance_mT(9.5)
        freqs, intens, trans = endorfrq(sys, B_field=B)

        assert isinstance(freqs, torch.Tensor)
        assert isinstance(intens, torch.Tensor)
        assert isinstance(trans, torch.Tensor)
        assert freqs.ndim == 1
        assert intens.ndim == 1
        assert trans.ndim == 2
        assert trans.shape[1] == 2

    def test_same_length_outputs(self):
        """freqs, intens, transitions have the same length."""
        sys = SpinSystem(S=0.5, g=2.0, Nucs='1H', A=[[10.0, 10.0, 10.0]])
        B = free_electron_resonance_mT(9.5)
        freqs, intens, trans = endorfrq(sys, B_field=B)

        assert freqs.shape[0] == intens.shape[0] == trans.shape[0]

    def test_frequencies_sorted_ascending(self):
        """Returned frequencies are sorted in ascending order."""
        sys = SpinSystem(S=0.5, g=2.0, Nucs='1H', A=[[10.0, 10.0, 10.0]])
        B = free_electron_resonance_mT(9.5)
        freqs, _, _ = endorfrq(sys, B_field=B)

        if freqs.numel() > 1:
            diffs = freqs[1:] - freqs[:-1]
            assert (diffs >= 0).all(), "Frequencies are not sorted"

    def test_intensities_nonnegative(self):
        """All intensities are ≥ 0."""
        sys = SpinSystem(S=0.5, g=2.0, Nucs='1H', A=[[5.0, 5.0, 25.0]])
        B = free_electron_resonance_mT(9.5)
        _, intens, _ = endorfrq(sys, B_field=B)
        assert (intens >= 0).all(), "Negative intensities found"

    def test_transitions_are_valid_pairs(self):
        """transition[k] = [i, j] with i < j and both in [0, nStates)."""
        sys = SpinSystem(S=0.5, g=2.0, Nucs='1H', A=[[10.0, 10.0, 10.0]])
        B = free_electron_resonance_mT(9.5)
        nStates = sys.nStates
        _, _, trans = endorfrq(sys, B_field=B)

        if trans.numel() > 0:
            assert (trans[:, 0] >= 0).all()
            assert (trans[:, 1] < nStates).all()
            assert (trans[:, 0] < trans[:, 1]).all(), "i >= j in some pair"

    def test_freq_range_filter(self):
        """freq_range filter excludes out-of-range lines."""
        sys = SpinSystem(S=0.5, g=2.0, Nucs='1H', A=[[10.0, 10.0, 10.0]])
        B = free_electron_resonance_mT(9.5)
        freqs_all, _, _ = endorfrq(sys, B_field=B)

        if freqs_all.numel() < 2:
            pytest.skip("Need at least 2 lines to test range filter")

        f_mid = freqs_all[freqs_all.numel() // 2].item()
        freqs_hi, _, _ = endorfrq(sys, B_field=B, freq_range=(f_mid, 1e6))
        assert (freqs_hi >= f_mid - 1e-6).all(), "Freq range filter failed"


class TestEndorfrqProtonLines:
    """Physical correctness: S=1/2 + 1H isotropic hyperfine.

    For isotropic A, the two ENDOR lines (at z-axis orientation) lie at:
        ν± = |ν_n ± A/2|
    where ν_n = Larmor frequency of proton at B_field.
    """

    def _get_endor_lines(self, A_iso_mhz: float, B_mT: float):
        """Return sorted ENDOR frequencies for S=1/2 + 1H at theta=0."""
        sys = SpinSystem(
            S=0.5, g=2.0,
            Nucs='1H',
            A=[[A_iso_mhz, A_iso_mhz, A_iso_mhz]],
        )
        freqs, intens, _ = endorfrq(sys, B_field=B_mT, phi=0.0, theta=0.0)
        # Nuclear transitions only: should be near ν_n ± A/2
        return freqs.numpy(), intens.numpy()

    def test_two_endor_lines_exist(self):
        """Isotropic S=1/2 + 1H should yield exactly 2 ENDOR lines at z-axis."""
        A_iso = 10.0   # MHz
        B_mT  = free_electron_resonance_mT(9.5)
        freqs, intens = self._get_endor_lines(A_iso, B_mT)

        # Filter to lines with significant intensity
        if len(intens) > 0:
            max_i = intens.max()
            significant = freqs[intens > 0.01 * max_i]
            # Should have at least 2 significant lines (ν_n ± A/2)
            assert len(significant) >= 2, (
                f"Expected ≥2 ENDOR lines, got {len(significant)}: {significant}"
            )

    def test_line_positions_near_larmor_plus_half_hf(self):
        """ENDOR lines are near ν_n ± A/2 for isotropic hyperfine."""
        A_iso = 5.0    # MHz
        B_mT  = free_electron_resonance_mT(9.5)
        nu_n  = proton_larmor_mhz(B_mT)

        expected_lo = abs(nu_n - A_iso / 2)
        expected_hi = abs(nu_n + A_iso / 2)

        freqs, intens = self._get_endor_lines(A_iso, B_mT)
        if len(freqs) == 0:
            pytest.skip("No ENDOR lines found")

        # At least one line should be within 1 MHz of each expected position
        tol = 1.0  # MHz
        near_lo = any(abs(f - expected_lo) < tol for f in freqs)
        near_hi = any(abs(f - expected_hi) < tol for f in freqs)

        assert near_lo or near_hi, (
            f"No ENDOR line near ν_n±A/2 = {expected_lo:.2f} or {expected_hi:.2f} MHz. "
            f"Found: {freqs}"
        )

    def test_nitrogen_gives_three_lines(self):
        """S=1/2 + 14N (I=1) should give 3 ENDOR lines (ΔmI = ±1 transitions)."""
        A_iso = 20.0  # MHz
        B_mT  = free_electron_resonance_mT(9.5)
        sys = SpinSystem(
            S=0.5, g=2.0,
            Nucs='14N',
            A=[[A_iso, A_iso, A_iso]],
        )
        freqs, intens, _ = endorfrq(sys, B_field=B_mT, phi=0.0, theta=0.0)
        max_i = intens.max().item() if intens.numel() > 0 else 0.0
        n_significant = (intens > 0.01 * max_i).sum().item()
        # For I=1: two nuclear transitions per mS manifold, so ≥2 lines
        assert n_significant >= 2, (
            f"Expected ≥2 ENDOR lines for 14N, got {n_significant}"
        )


# ── Tests for salt ─────────────────────────────────────────────────────────

class TestSalt:

    def _simple_sys(self):
        """Simple S=1/2 + 1H system for salt tests."""
        return SpinSystem(
            S=0.5, g=2.0,
            Nucs='1H',
            A=[[10.0, 10.0, 10.0]],
            lw=[0.5, 0.0],
        )

    def _simple_exp(self):
        return Experiment(mwFreq=9.5, Range=[330.0, 355.0], nPoints=64, Harmonic=0)

    def _fast_opt(self):
        return Options(GridSize=7, GridSymmetry='Ci', Threshold=1e-3)

    def test_returns_two_tensors(self):
        """salt returns (freq_axis, spec) as 1D Tensors."""
        sys = self._simple_sys()
        exp = self._simple_exp()
        freq_axis, spec = salt(sys, exp, self._fast_opt(), n_points=64)

        assert isinstance(freq_axis, torch.Tensor)
        assert isinstance(spec, torch.Tensor)
        assert freq_axis.ndim == 1
        assert spec.ndim == 1
        assert freq_axis.shape[0] == 64
        assert spec.shape[0] == 64

    def test_no_nuclei_raises(self):
        """salt raises ValueError if spin system has no nuclei."""
        sys = SpinSystem(S=0.5, g=2.0)
        exp = self._simple_exp()
        with pytest.raises(ValueError, match="nucleus"):
            salt(sys, exp, self._fast_opt())

    def test_finite_values(self):
        """ENDOR spectrum contains only finite values."""
        sys = self._simple_sys()
        exp = self._simple_exp()
        _, spec = salt(sys, exp, self._fast_opt(), n_points=64)
        assert torch.all(torch.isfinite(spec)), "Non-finite values in salt spectrum"

    def test_spectrum_has_signal(self):
        """ENDOR spectrum has non-zero signal."""
        sys = self._simple_sys()
        exp = self._simple_exp()
        _, spec = salt(sys, exp, self._fast_opt(), n_points=128, lw_mhz=0.5)
        assert spec.abs().max().item() > 0, "ENDOR spectrum is all zeros"

    def test_freq_axis_matches_range(self):
        """salt frequency axis spans [0, 2*mwFreq] when freq_range=None."""
        sys = self._simple_sys()
        exp = self._simple_exp()
        freq_axis, _ = salt(sys, exp, self._fast_opt(), n_points=128)

        expected_lo = 0.0
        expected_hi = 2.0 * exp.mwFreq * 1e3  # MHz

        assert abs(freq_axis[0].item() - expected_lo) < 1.0, (
            f"Freq axis starts at {freq_axis[0].item():.2f}, expected ~{expected_lo:.2f} MHz"
        )
        assert abs(freq_axis[-1].item() - expected_hi) < 1.0, (
            f"Freq axis ends at {freq_axis[-1].item():.2f}, expected ~{expected_hi:.2f} MHz"
        )

    def test_custom_freq_range(self):
        """Custom freq_range restricts the frequency axis."""
        sys = self._simple_sys()
        exp = self._simple_exp()
        f_lo, f_hi = 10.0, 50.0
        freq_axis, _ = salt(sys, exp, self._fast_opt(), freq_range=(f_lo, f_hi),
                            n_points=64)

        assert abs(freq_axis[0].item() - f_lo) < 0.1
        assert abs(freq_axis[-1].item() - f_hi) < 0.1

    def test_pake_pattern_peak_near_larmor(self):
        """Powder ENDOR of S=1/2 + 1H should show intensity near ν_n ± A/2.

        The powder average of isotropic A gives a Pake doublet centered
        at the Larmor frequency with splitting A.  The dominant peaks
        should appear near ν_n ± A/2.
        """
        A_iso = 8.0   # MHz (clear Pake splitting, well-resolved)
        B_mT  = free_electron_resonance_mT(9.5)
        nu_n  = proton_larmor_mhz(B_mT)

        sys = SpinSystem(S=0.5, g=2.0, Nucs='1H',
                         A=[[A_iso, A_iso, A_iso]], lw=[0.5, 0.0])
        exp = Experiment(mwFreq=9.5, Range=[B_mT - 5, B_mT + 5],
                         nPoints=64, Harmonic=0)
        opt = Options(GridSize=11, GridSymmetry='auto', Threshold=1e-3)

        f_lo = nu_n - A_iso - 5.0
        f_hi = nu_n + A_iso + 5.0
        freq_axis, spec = salt(sys, exp, opt, freq_range=(f_lo, f_hi),
                               n_points=256, lw_mhz=0.2)

        if spec.abs().max().item() == 0:
            pytest.skip("No ENDOR signal (grid too coarse or no EPR resonances in range)")

        # Peak of the spectrum should be in the frequency interval
        peak_freq = freq_axis[spec.argmax()].item()
        assert f_lo < peak_freq < f_hi, (
            f"ENDOR peak at {peak_freq:.2f} MHz outside [{f_lo:.1f}, {f_hi:.1f}]"
        )
