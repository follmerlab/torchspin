"""Tests for torchspin.dataproc — basecorr, fieldmod, rescaledata."""

import math
import numpy as np
import pytest

from torchspin.dataproc import basecorr, fieldmod, rescaledata
from torchspin.lineshape import lorentzian


# ===========================================================================
# basecorr
# ===========================================================================

class TestBasecorr:
    def _linear_signal(self, n=200):
        x = np.linspace(0, 1, n)
        return x + 0.5 * x  # increasing baseline

    def test_returns_two_arrays(self):
        y = np.ones(100)
        yc, bl = basecorr(y, 1, 0)
        assert isinstance(yc, np.ndarray)
        assert isinstance(bl, np.ndarray)
        assert yc.shape == y.shape
        assert bl.shape == y.shape

    def test_zero_order_removes_constant(self):
        """Degree-0 fit removes the mean."""
        y = np.ones(100) * 5.0
        yc, bl = basecorr(y, 1, 0)
        assert np.allclose(yc, 0.0, atol=1e-10)
        assert np.allclose(bl, 5.0, atol=1e-10)

    def test_first_order_removes_linear_baseline(self):
        """Degree-1 fit removes a linear trend."""
        x = np.linspace(0, 1, 200)
        baseline = 3.0 * x + 1.0
        signal = np.sin(2 * math.pi * 3 * x) * 0.0  # zero signal + baseline
        y = baseline
        yc, bl = basecorr(y, 1, 1)
        assert np.allclose(yc, 0.0, atol=1e-8)
        assert np.allclose(bl, baseline, atol=1e-8)

    def test_second_order_removes_quadratic(self):
        """Degree-2 fit removes a quadratic baseline."""
        x = np.linspace(-1, 1, 300)
        baseline = 2 * x ** 2 - x + 0.5
        yc, bl = basecorr(baseline, 1, 2)
        assert np.allclose(yc, 0.0, atol=1e-8)

    def test_signal_preserved_after_correction(self):
        """Corrected signal + baseline ≈ original data."""
        x = np.linspace(0, 1, 200)
        y = np.sin(6 * math.pi * x) + 2.0 * x
        yc, bl = basecorr(y, 1, 1)
        assert np.allclose(yc + bl, y, atol=1e-12)

    def test_region_mask(self):
        """Using a region mask should only fit to masked points."""
        x = np.linspace(0, 1, 200)
        baseline = 3.0 * x
        peak_region = (x > 0.4) & (x < 0.6)
        # Put a large spike in the peak region
        y = baseline.copy()
        y[peak_region] += 100.0
        # Exclude peak from fit
        region = ~peak_region
        yc, bl = basecorr(y, 1, 1, region=region)
        # Outside peak, correction should remove baseline well
        outside = ~peak_region
        assert np.allclose(yc[outside], 0.0, atol=1e-6)

    def test_dim2_fits_along_rows(self):
        """dim=2 fits along rows (axis 1) of a 2D array."""
        data = np.ones((10, 50))  # constant rows
        for i in range(10):
            data[i] += i  # each row has a different constant offset
        yc, bl = basecorr(data, 2, 0)
        # Each row should be approximately zero after removing its constant
        assert np.allclose(yc, 0.0, atol=1e-8)

    def test_2d_fit(self):
        """2D polynomial fit: flat plane is removed exactly."""
        r, c = 20, 30
        xi = np.linspace(-1, 1, r)
        yi = np.linspace(-1, 1, c)
        xg, yg = np.meshgrid(xi, yi, indexing='ij')
        plane = 2.0 * xg + 3.0 * yg + 1.0
        yc, bl = basecorr(plane, None, [1, 1])
        assert np.allclose(yc, 0.0, atol=1e-8)

    def test_bad_order_raises(self):
        with pytest.raises(ValueError):
            basecorr(np.ones(100), 1, 7)  # order > 6

    def test_output_same_shape(self):
        data = np.random.default_rng(0).standard_normal((5, 50))
        yc, bl = basecorr(data, 1, 2)
        assert yc.shape == data.shape


# ===========================================================================
# fieldmod
# ===========================================================================

class TestFieldmod:
    def _lorentzian_spectrum(self, n=512, center=342.0, fwhm=4.0,
                              b_range=(300.0, 400.0)):
        B = np.linspace(b_range[0], b_range[1], n)
        spc, _ = lorentzian(B, center, fwhm)
        return B, spc

    def test_output_shape(self):
        B, spc = self._lorentzian_spectrum()
        out = fieldmod(B, spc, 2.0)
        assert out.shape == spc.shape

    def test_output_real(self):
        B, spc = self._lorentzian_spectrum()
        out = fieldmod(B, spc, 2.0)
        assert np.isrealobj(out)

    def test_harmonic0_preserves_integral(self):
        """Harmonic 0 modulation should preserve the integral (no differentiation)."""
        B, spc = self._lorentzian_spectrum()
        out = fieldmod(B, spc, 0.5, harmonic=0)
        # With small ModAmp, harmonic=0 output should resemble original
        corr = np.corrcoef(spc, out)[0, 1]
        assert corr > 0.99

    def test_harmonic1_derivative_shape(self):
        """Harmonic 1 output should have derivative-like shape (zero crossings)."""
        B, spc = self._lorentzian_spectrum()
        out = fieldmod(B, spc, 1.0, harmonic=1)
        # Should have positive and negative values (derivative, not absorption)
        assert out.max() > 0
        assert out.min() < 0

    def test_small_modamp_resembles_derivative(self):
        """Small mod amplitude → output ≈ scaled first derivative."""
        B, spc = self._lorentzian_spectrum(n=2001)
        out = fieldmod(B, spc, 0.05)  # very small modulation
        dx = B[1] - B[0]
        d_spc = np.gradient(spc, dx)
        # Should be correlated with derivative (up to a scale factor)
        corr = np.corrcoef(out[10:-10], d_spc[10:-10])[0, 1]
        assert abs(corr) > 0.99

    def test_zero_modamp_raises(self):
        B, spc = self._lorentzian_spectrum()
        with pytest.raises(ValueError, match="mod_amp"):
            fieldmod(B, spc, 0.0)

    def test_negative_harmonic_raises(self):
        B, spc = self._lorentzian_spectrum()
        with pytest.raises(ValueError, match="harmonic"):
            fieldmod(B, spc, 1.0, harmonic=-1)

    def test_length_mismatch_raises(self):
        B, spc = self._lorentzian_spectrum()
        with pytest.raises(ValueError):
            fieldmod(B, spc[:-1], 1.0)

    def test_harmonic2_output(self):
        """Harmonic 2 should give a valid output array."""
        B, spc = self._lorentzian_spectrum()
        out = fieldmod(B, spc, 2.0, harmonic=2)
        assert out.shape == spc.shape
        assert np.isfinite(out).all()

    def test_zero_signal_gives_zero(self):
        """Zero input spectrum → zero output."""
        B = np.linspace(300, 400, 256)
        spc = np.zeros(256)
        out = fieldmod(B, spc, 2.0)
        assert np.allclose(out, 0.0, atol=1e-12)


# ===========================================================================
# rescaledata
# ===========================================================================

class TestRescaledata:
    def test_returns_two_values(self):
        y = np.array([1.0, 2.0, 3.0])
        yscaled, scale = rescaledata(y, mode='maxabs')
        assert isinstance(yscaled, np.ndarray)
        assert isinstance(scale, float)

    def test_maxabs_no_ref(self):
        """maxabs without ref: max(|y|) → 1."""
        y = np.array([1.0, -3.0, 2.0])
        yscaled, scale = rescaledata(y, mode='maxabs')
        assert abs(np.max(np.abs(yscaled)) - 1.0) < 1e-10

    def test_maxabs_with_ref(self):
        """maxabs with ref: max(|yscaled|) = max(|yref|)."""
        y = np.array([1.0, 2.0, -3.0])
        yref = np.array([6.0, 0.0, -6.0])
        yscaled, scale = rescaledata(y, yref=yref, mode='maxabs')
        assert abs(np.max(np.abs(yscaled)) - 6.0) < 1e-10

    def test_lsq_matches_ref(self):
        """lsq: yscaled ≈ yref."""
        rng = np.random.default_rng(0)
        yref = rng.standard_normal(100)
        y = yref * 3.7 + rng.standard_normal(100) * 0.01  # y ≈ 3.7 * yref
        yscaled, scale = rescaledata(y, yref=yref, mode='lsq')
        corr = np.corrcoef(yscaled, yref)[0, 1]
        assert corr > 0.999
        # yscaled should be close to yref in scale
        ratio = np.std(yscaled) / np.std(yref)
        assert abs(ratio - 1.0) < 0.01

    def test_int_normalizes_sum(self):
        """int mode: sum(yscaled) = 1."""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        yscaled, scale = rescaledata(y, mode='int')
        assert abs(yscaled.sum() - 1.0) < 1e-10

    def test_dint_normalizes_double_integral(self):
        """dint mode: sum(cumsum(yscaled)) = 1."""
        y = np.array([1.0, 2.0, 1.0, 0.5])
        yscaled, scale = rescaledata(y, mode='dint')
        assert abs(np.sum(np.cumsum(yscaled)) - 1.0) < 1e-10

    def test_none_returns_copy(self):
        y = np.array([1.0, 2.0, 3.0])
        yscaled, scale = rescaledata(y, mode='none')
        assert np.allclose(yscaled, y)
        assert scale == 1.0

    def test_positive_scale_enforced(self):
        """If y is all negative, scaling factor is made positive (no inversion)."""
        y = np.array([-3.0, -1.0, -2.0])
        yscaled, scale = rescaledata(y, mode='maxabs')
        assert scale > 0

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown scaling mode"):
            rescaledata(np.ones(10), mode='xyz')

    def test_obsolete_mode_raises(self):
        with pytest.raises(ValueError, match="obsolete"):
            rescaledata(np.ones(10), mode='lsq0')

    def test_lsq_needs_ref_raises(self):
        with pytest.raises(ValueError, match="reference"):
            rescaledata(np.ones(10), mode='lsq')

    def test_lsq_different_length_ref(self):
        """lsq with different-length yref resamples yref."""
        rng = np.random.default_rng(1)
        y = rng.standard_normal(100)
        yref = y.copy() * 2.0
        yref_short = yref[::2]  # 50 points
        # Should not raise; lsq is approximate
        yscaled, scale = rescaledata(y, yref=yref_short, mode='lsq')
        assert yscaled.shape == y.shape

    def test_region_mask_maxabs(self):
        """Region mask limits which points are used for scaling."""
        y = np.array([1.0, 2.0, 100.0, 1.0])
        region = np.array([True, True, False, True])
        yscaled, scale = rescaledata(y, mode='maxabs', region=region)
        # Scaling should be based on max=2 (not 100)
        assert abs(scale - 0.5) < 1e-10

    def test_2d_raises(self):
        with pytest.raises(ValueError):
            rescaledata(np.ones((10, 10)), mode='maxabs')
