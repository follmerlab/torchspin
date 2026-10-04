"""
Tests for torchspin.autoguess — automatic parameter estimation from EPR spectra.
"""

import math

import numpy as np
import pytest

from torchspin.autoguess import (
    estimate_parameters,
    field_from_g,
    g_from_field,
    _MHZ_PER_MT,
)
from torchspin.constants import BMAGN, PLANCK


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_derivative(B: np.ndarray, g: float, lw_mT: float, mwFreq: float) -> np.ndarray:
    """Derivative of a Gaussian absorption peak at the resonance field of g."""
    B_res = field_from_g(g, mwFreq)
    sigma = lw_mT / (2 * math.sqrt(2 * math.log(2)))
    x = (B - B_res) / sigma
    return -x / sigma * np.exp(-0.5 * x**2)


# ---------------------------------------------------------------------------
# Unit tests: field_from_g and g_from_field
# ---------------------------------------------------------------------------

class TestFieldGConversion:
    """field_from_g / g_from_field must be mutually inverse."""

    def test_field_from_g_known(self):
        # At 9.5 GHz, g=2.0 → B ≈ 339.0 mT (exact value from resonance condition)
        B = field_from_g(2.0, 9.5)
        expected = 9.5e3 / (2.0 * _MHZ_PER_MT)  # mwFreq_MHz / (g * MHz_per_mT)
        assert abs(B - expected) < 1e-10

    def test_field_from_g_numerical(self):
        # B = h*mwFreq / (g * BMAGN) * 1e3  [mT]
        mwFreq = 9.5  # GHz
        g = 2.0
        B_expected = PLANCK * mwFreq * 1e9 / (g * BMAGN) * 1e3
        assert abs(field_from_g(g, mwFreq) - B_expected) < 1e-6  # <1 µT

    def test_g_from_field_roundtrip(self):
        for g in [1.5, 2.0, 2.5, 3.0]:
            B = field_from_g(g, 9.5)
            g_back = g_from_field(B, 9.5)
            assert abs(g_back - g) < 1e-12

    def test_g_from_field_known(self):
        # g=2.0 at 9.5 GHz: B ≈ 339 mT
        B = field_from_g(2.0, 9.5)
        assert abs(g_from_field(B, 9.5) - 2.0) < 1e-12

    def test_field_scales_with_frequency(self):
        # Higher mwFreq → higher resonance field (proportional)
        B1 = field_from_g(2.0, 9.5)
        B2 = field_from_g(2.0, 34.0)
        assert abs(B2 / B1 - 34.0 / 9.5) < 1e-10

    def test_field_scales_inversely_with_g(self):
        # Higher g → lower resonance field
        B_small_g = field_from_g(2.5, 9.5)
        B_large_g = field_from_g(1.8, 9.5)
        assert B_small_g < B_large_g


# ---------------------------------------------------------------------------
# Unit tests: estimate_parameters — isotropic spectrum
# ---------------------------------------------------------------------------

class TestEstimateParametersIsotropic:
    """Single isotropic g-value, one derivative line."""

    def setup_method(self):
        self.mwFreq = 9.5   # GHz
        self.g_true = 2.003
        self.lw_mT = 1.5
        self.B_res = field_from_g(self.g_true, self.mwFreq)
        self.B = np.linspace(self.B_res - 20, self.B_res + 20, 1001)
        self.spc = _synthetic_derivative(self.B, self.g_true, self.lw_mT, self.mwFreq)

    def test_returns_dict_with_required_keys(self):
        result = estimate_parameters(self.B, self.spc, self.mwFreq)
        for key in ('g', 'lwpp', 'Range', 'CenterSweep', 'giso', 'peaks'):
            assert key in result

    def test_g_tensor_length_3(self):
        result = estimate_parameters(self.B, self.spc, self.mwFreq)
        assert len(result['g']) == 3

    def test_g_values_in_physical_range(self):
        result = estimate_parameters(self.B, self.spc, self.mwFreq)
        for g in result['g']:
            assert 1.5 <= g <= 3.0

    def test_g_value_close_to_truth(self):
        result = estimate_parameters(self.B, self.spc, self.mwFreq)
        # giso should be within 0.05 of true g (derivative peak ≠ perfect zero-crossing)
        assert abs(result['giso'] - self.g_true) < 0.05

    def test_peaks_found(self):
        result = estimate_parameters(self.B, self.spc, self.mwFreq)
        assert len(result['peaks']['fields']) > 0

    def test_linewidth_reasonable(self):
        result = estimate_parameters(self.B, self.spc, self.mwFreq)
        lw_est = result['lwpp'][0]
        # Estimated lw should be within factor of 3 of the true linewidth
        assert 0.3 * self.lw_mT < lw_est < 3.0 * self.lw_mT

    def test_range_covers_spectrum(self):
        result = estimate_parameters(self.B, self.spc, self.mwFreq)
        assert result['Range'][0] <= self.B_res
        assert result['Range'][1] >= self.B_res


# ---------------------------------------------------------------------------
# Unit tests: estimate_parameters — rhombic spectrum (3 lines)
# ---------------------------------------------------------------------------

class TestEstimateParametersRhombic:
    """Three distinct g-values — rhombic EPR spectrum."""

    def setup_method(self):
        self.mwFreq = 9.5
        self.g_xyz = [2.45, 2.20, 1.92]   # rhombic, sorted high to low
        self.lw_mT = 1.0
        B_vals = [field_from_g(g, self.mwFreq) for g in self.g_xyz]
        B_min = min(B_vals) - 20
        B_max = max(B_vals) + 20
        self.B = np.linspace(B_min, B_max, 2001)
        self.spc = sum(
            _synthetic_derivative(self.B, g, self.lw_mT, self.mwFreq)
            for g in self.g_xyz
        )

    def test_three_g_values_returned(self):
        result = estimate_parameters(self.B, self.spc, self.mwFreq)
        assert len(result['g']) == 3

    def test_g_sorted_descending(self):
        result = estimate_parameters(self.B, self.spc, self.mwFreq)
        g = result['g']
        assert g[0] >= g[1] >= g[2]

    def test_peaks_detected(self):
        result = estimate_parameters(self.B, self.spc, self.mwFreq)
        # Should find at least 2 distinct spectral features (positive + negative per line)
        assert len(result['peaks']['fields']) >= 2


# ---------------------------------------------------------------------------
# Unit tests: estimate_parameters — fallback (no valid peaks)
# ---------------------------------------------------------------------------

class TestEstimateParametersFallback:
    """When no valid peaks are found, function should return sensible defaults."""

    def test_empty_spectrum_returns_fallback(self):
        B = np.linspace(300, 400, 512)
        spc = np.zeros(512)  # totally flat
        result = estimate_parameters(B, spc, 9.5, prominence=0.5)
        # Should still return a dict with all required keys
        for key in ('g', 'lwpp', 'Range', 'CenterSweep', 'giso', 'peaks'):
            assert key in result
        # g should be length 3
        assert len(result['g']) == 3

    def test_out_of_range_field_falls_back(self):
        # Spectrum at 100 mT → g ≈ 6.8 which is outside (1.5, 3.0) filter
        B = np.linspace(90, 110, 512)
        B_res = 100.0
        sigma = 1.0
        spc = -(B - B_res) / sigma**2 * np.exp(-0.5 * ((B - B_res) / sigma)**2)
        result = estimate_parameters(B, spc, 9.5)
        # No valid peaks after filter → giso comes from B mean
        g_from_mean = g_from_field(np.mean(B), 9.5)
        assert abs(result['giso'] - g_from_mean) < 0.5


# ---------------------------------------------------------------------------
# Constant sanity check
# ---------------------------------------------------------------------------

def test_mhz_per_mt_constant():
    """_MHZ_PER_MT should equal BMAGN/PLANCK*1e-9 ≈ 13.996 MHz/mT."""
    expected = BMAGN / PLANCK * 1e-9
    assert abs(_MHZ_PER_MT - expected) < 1e-6
    # Should be roughly 14, not 28
    assert 13.5 < _MHZ_PER_MT < 14.5
