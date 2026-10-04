"""Tests for torchspin.lineshape — EPR lineshape functions."""

import math
import numpy as np
import pytest
from scipy.special import dawsn

from torchspin.lineshape import (
    gaussian,
    lorentzian,
    voigtian,
    apowin,
    addnoise,
    deriv,
    lshape,
)


_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")


# Common test abscissa
X = np.linspace(-5, 5, 2001)
DX = X[1] - X[0]


# ===========================================================================
# gaussian
# ===========================================================================

class TestGaussian:
    def test_returns_tuple(self):
        ya, yd = gaussian(X, 0.0, 1.0)
        assert isinstance(ya, np.ndarray)
        assert isinstance(yd, np.ndarray)
        assert ya.shape == X.shape

    def test_area_normalized(self):
        """∫ gaussian dx ≈ 1 (area-normalized)."""
        ya, _ = gaussian(X, 0.0, 1.0)
        assert abs(_trapezoid(ya, X) - 1.0) < 1e-5

    def test_peak_at_center(self):
        ya, _ = gaussian(X, 0.5, 1.0)
        idx = np.argmax(ya)
        assert abs(X[idx] - 0.5) < DX * 2

    def test_fwhm_correct(self):
        """Peak height of zero-order Gaussian equals 1/(sig*sqrt(2π))."""
        fwhm = 1.5
        ya, _ = gaussian(X, 0.0, fwhm)
        sig = fwhm / math.sqrt(8 * math.log(2))
        expected_peak = 1.0 / (sig * math.sqrt(2 * math.pi))
        assert abs(ya.max() - expected_peak) < 1e-6

    def test_first_derivative_zero_integral(self):
        """First derivative integrates to ≈0 (odd function)."""
        ya, _ = gaussian(X, 0.0, 1.0, diff=1)
        assert abs(_trapezoid(ya, X)) < 1e-8

    def test_first_derivative_antisymmetric(self):
        """First derivative is antisymmetric about center."""
        ya, _ = gaussian(X, 0.0, 1.0, diff=1)
        assert np.allclose(ya, -ya[::-1], atol=1e-10)

    def test_second_derivative_shape(self):
        """Second derivative has a central negative lobe."""
        ya, _ = gaussian(X, 0.0, 1.0, diff=2)
        assert ya[len(ya) // 2] < 0

    def test_integral_diff_minus1(self):
        """diff=-1 gives cumulative integral: approaches 1 at +∞."""
        ya, _ = gaussian(X, 0.0, 1.0, diff=-1)
        assert abs(ya[-1] - 1.0) < 1e-4
        assert abs(ya[0]) < 1e-4

    def test_dispersion_antisymmetric(self):
        """Dispersion lineshape is antisymmetric."""
        _, yd = gaussian(X, 0.0, 1.0)
        assert np.allclose(yd, -yd[::-1], atol=1e-10)

    def test_phase_pi2_swaps(self):
        """phase=π/2 puts dispersion into ya."""
        ya0, yd0 = gaussian(X, 0.0, 1.0, phase=0)
        ya_ph, _ = gaussian(X, 0.0, 1.0, phase=math.pi / 2)
        # ya at phase=π/2 should equal the dispersion yd at phase=0
        assert np.allclose(ya_ph, yd0, atol=1e-10)

    def test_negative_fwhm_raises(self):
        with pytest.raises(ValueError):
            gaussian(X, 0.0, -1.0)

    def test_bad_diff_raises(self):
        with pytest.raises(ValueError):
            gaussian(X, 0.0, 1.0, diff=0.5)


# ===========================================================================
# lorentzian
# ===========================================================================

class TestLorentzian:
    def test_returns_tuple(self):
        ya, yd = lorentzian(X, 0.0, 1.0)
        assert ya.shape == X.shape

    def test_area_normalized(self):
        """∫ lorentzian dx ≈ 1 (Lorentzian has heavy tails; use wide range)."""
        xw = np.linspace(-500, 500, 50001)
        ya, _ = lorentzian(xw, 0.0, 0.5)
        assert abs(_trapezoid(ya, xw) - 1.0) < 1e-3

    def test_peak_at_center(self):
        ya, _ = lorentzian(X, -0.5, 1.0)
        idx = np.argmax(ya)
        assert abs(X[idx] + 0.5) < DX * 2

    def test_first_derivative_antisymmetric(self):
        ya, _ = lorentzian(X, 0.0, 1.0, diff=1)
        assert np.allclose(ya, -ya[::-1], atol=1e-10)

    def test_second_derivative_central_negative(self):
        ya, _ = lorentzian(X, 0.0, 1.0, diff=2)
        assert ya[len(ya) // 2] < 0

    def test_integral_diff_minus1(self):
        """diff=-1 approaches 1 at +∞ (use wide range for Lorentzian tails)."""
        xw = np.linspace(-500, 500, 5001)
        ya, _ = lorentzian(xw, 0.0, 0.5, diff=-1)
        assert abs(ya[-1] - 1.0) < 5e-3
        assert abs(ya[0]) < 5e-3

    def test_dispersion_antisymmetric(self):
        _, yd = lorentzian(X, 0.0, 1.0)
        assert np.allclose(yd, -yd[::-1], atol=1e-10)

    def test_phase_pi2_swaps(self):
        ya0, yd0 = lorentzian(X, 0.0, 1.0, phase=0)
        ya_ph, _ = lorentzian(X, 0.0, 1.0, phase=math.pi / 2)
        assert np.allclose(ya_ph, yd0, atol=1e-10)

    def test_bad_diff_raises(self):
        with pytest.raises(ValueError):
            lorentzian(X, 0.0, 1.0, diff=3)

    def test_negative_fwhm_raises(self):
        with pytest.raises(ValueError):
            lorentzian(X, 0.0, -0.5)

    def test_fwhm_correct(self):
        """Peak of zero-order Lorentzian matches analytical formula."""
        fwhm = 1.0
        gamma = fwhm / math.sqrt(3)
        pre = 2.0 / (math.pi * math.sqrt(3))
        expected_peak = pre / gamma  # at z=0: 1/(1+0)
        ya, _ = lorentzian(X, 0.0, fwhm)
        assert abs(ya.max() - expected_peak) < 1e-6


# ===========================================================================
# voigtian
# ===========================================================================

class TestVoigtian:
    def test_pure_gaussian(self):
        """voigtian with fwhmL=0 should equal gaussian."""
        fwhm = 1.0
        ya_v, _ = voigtian(X, 0.0, [fwhm, 0.0])
        ya_g, _ = gaussian(X, 0.0, fwhm)
        assert np.allclose(ya_v, ya_g, atol=1e-10)

    def test_pure_lorentzian(self):
        """voigtian with fwhmG=0 should equal lorentzian."""
        fwhm = 1.0
        ya_v, _ = voigtian(X, 0.0, [0.0, fwhm])
        ya_l, _ = lorentzian(X, 0.0, fwhm)
        assert np.allclose(ya_v, ya_l, atol=1e-10)

    def test_area_normalized(self):
        """∫ voigtian dx ≈ 1 (Lorentzian tails; use wide range)."""
        xw = np.linspace(-500, 500, 50001)
        ya, _ = voigtian(xw, 0.0, [0.8, 0.6])
        assert abs(_trapezoid(ya, xw) - 1.0) < 1e-2

    def test_peak_at_center(self):
        ya, _ = voigtian(X, 0.0, [0.8, 0.6])
        idx = np.argmax(ya)
        assert abs(X[idx]) < DX * 3

    def test_symmetric(self):
        """Voigt absorption is symmetric."""
        ya, _ = voigtian(X, 0.0, [0.8, 0.6])
        assert np.allclose(ya, ya[::-1], atol=1e-4)

    def test_both_zero_raises(self):
        with pytest.raises(ValueError):
            voigtian(X, 0.0, [0.0, 0.0])

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            voigtian(X, 0.0, [-0.5, 1.0])

    def test_returns_tuple(self):
        ya, yd = voigtian(X, 0.0, [1.0, 0.5])
        assert ya.shape == X.shape
        assert yd.shape == X.shape


# ===========================================================================
# apowin
# ===========================================================================

class TestApowin:
    def test_hamming_max_one(self):
        w = apowin('ham', 100)
        assert abs(w.max() - 1.0) < 1e-10

    def test_hann_max_one(self):
        w = apowin('han', 100)
        assert abs(w.max() - 1.0) < 1e-10

    def test_blackman_max_one(self):
        w = apowin('bla', 100)
        assert abs(w.max() - 1.0) < 1e-10

    def test_output_length(self):
        for n in [50, 100, 200]:
            w = apowin('ham', n)
            assert len(w) == n

    def test_symmetric_windows(self):
        """Full windows are symmetric."""
        for key in ['ham', 'han', 'bla', 'bar', 'con', 'cos', 'wel']:
            w = apowin(key, 101)
            assert np.allclose(w, w[::-1], atol=1e-10), f"not symmetric: {key}"

    def test_right_half_monotone(self):
        """Right-half window ('+') is non-increasing (peaks at start)."""
        w = apowin('ham+', 100)
        assert np.all(np.diff(w) <= 1e-12)

    def test_left_half_monotone(self):
        """Left-half window ('-') is non-decreasing (peaks at end)."""
        w = apowin('ham-', 100)
        assert np.all(np.diff(w) >= -1e-12)

    def test_kaiser_with_alpha(self):
        w = apowin('kai', 100, alpha=5.0)
        assert abs(w.max() - 1.0) < 1e-10
        assert len(w) == 100

    def test_gaussian_with_alpha(self):
        w = apowin('gau', 100, alpha=1.0)
        assert abs(w.max() - 1.0) < 1e-10

    def test_exp_with_alpha(self):
        w = apowin('exp', 100, alpha=3.0)
        assert abs(w.max() - 1.0) < 1e-10

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            apowin('xyz', 100)

    def test_needs_alpha_raises(self):
        with pytest.raises(ValueError):
            apowin('kai', 100)  # alpha missing


# ===========================================================================
# addnoise
# ===========================================================================

class TestAddnoise:
    def _make_signal(self):
        x = np.linspace(-1, 1, 1001)
        y, _ = lorentzian(x, 0.0, 0.3)
        return y

    def test_shape_preserved(self):
        y = self._make_signal()
        yn = addnoise(y, 20, 'n', rng=np.random.default_rng(0))
        assert yn.shape == y.shape

    def test_snr_gaussian(self):
        """Gaussian noise: noise std ≈ signal_amplitude / SNR."""
        y = self._make_signal()
        snr = 20
        yn = addnoise(y, snr, 'n', rng=np.random.default_rng(42))
        noise = yn - y
        signal_amp = y.max() - y.min()
        expected_std = signal_amp / snr
        assert abs(noise.std() - expected_std) / expected_std < 0.1

    def test_uniform_noise(self):
        y = self._make_signal()
        yn = addnoise(y, 10, 'u', rng=np.random.default_rng(1))
        assert yn.shape == y.shape

    def test_1f_noise(self):
        y = self._make_signal()
        yn = addnoise(y, 10, 'f', rng=np.random.default_rng(2))
        assert yn.shape == y.shape

    def test_bad_snr_raises(self):
        y = np.ones(100)
        with pytest.raises(ValueError):
            addnoise(y, -1, 'n')

    def test_bad_model_raises(self):
        y = np.ones(100)
        with pytest.raises(ValueError):
            addnoise(y, 10, 'z')

    def test_reproducible_with_rng(self):
        y = self._make_signal()
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        yn1 = addnoise(y, 10, 'n', rng=rng1)
        yn2 = addnoise(y, 10, 'n', rng=rng2)
        assert np.allclose(yn1, yn2)


# ===========================================================================
# deriv
# ===========================================================================

class TestDeriv:
    def test_constant_zero(self):
        """Derivative of constant signal is zero."""
        y = np.ones(100)
        d = deriv(y)
        assert np.allclose(d, 0.0, atol=1e-12)

    def test_linear_slope_one(self):
        """Derivative of y=x is 1 everywhere."""
        x = np.linspace(0, 10, 201)
        y = x.copy()
        d = deriv(x, y)
        assert np.allclose(d, 1.0, atol=1e-10)

    def test_linear_slope_general(self):
        """Derivative of y = a*x + b is a."""
        x = np.linspace(-5, 5, 501)
        a = 3.7
        y = a * x + 2.1
        d = deriv(x, y)
        assert np.allclose(d, a, atol=1e-10)

    def test_sine_deriv(self):
        """d/dx sin(x) ≈ cos(x)."""
        x = np.linspace(0, 2 * math.pi, 2001)
        y = np.sin(x)
        d = deriv(x, y)
        # Interior only (endpoints have one-sided approximation)
        assert np.allclose(d[10:-10], np.cos(x[10:-10]), atol=1e-5)

    def test_unit_spacing(self):
        """Without x, uses unit spacing."""
        y = np.arange(10, dtype=float)
        d = deriv(y)
        assert np.allclose(d, 1.0, atol=1e-12)

    def test_output_same_length(self):
        y = np.random.default_rng(0).standard_normal(137)
        d = deriv(y)
        assert len(d) == len(y)

    def test_gaussian_deriv_antisymmetric(self):
        """Derivative of symmetric Gaussian is antisymmetric."""
        ya, _ = gaussian(X, 0.0, 1.0)
        d = deriv(X, ya)
        assert np.allclose(d, -d[::-1], atol=1e-8)


# ---------------------------------------------------------------------------
# lshape
# ---------------------------------------------------------------------------

class TestLshape:
    X = np.linspace(-10, 10, 2000)

    def test_gaussian_integral(self):
        """Pure Gaussian lshape integrates to ~1."""
        y = lshape(self.X, 0, 1.0, alpha=1.0)
        assert abs(_trapezoid(y, self.X) - 1.0) < 1e-3

    def test_lorentzian_integral(self):
        """Pure Lorentzian lshape integrates to ~1 (needs wide range)."""
        X = np.linspace(-50, 50, 10000)
        y = lshape(X, 0, 1.0, alpha=0.0)
        assert abs(_trapezoid(y, X) - 1.0) < 0.01

    def test_mixed_alpha(self):
        """alpha=0.5 gives mix of G and L."""
        y = lshape(self.X, 0, 1.0, alpha=0.5)
        # Should be positive, centred at 0
        assert y[len(y)//2] > 0
        assert y.max() == y[len(y)//2]

    def test_centre_offset(self):
        """x0 shifts the peak."""
        y0 = lshape(self.X, 0, 1.0)
        y1 = lshape(self.X, 2.0, 1.0)
        assert np.argmax(y1) > np.argmax(y0)

    def test_diff_0_positive(self):
        """Absorption (diff=0) is positive everywhere."""
        y = lshape(self.X, 0, 1.0)
        assert np.all(y >= 0)

    def test_diff_1_antisymmetric(self):
        """First derivative is antisymmetric about x0."""
        y = lshape(self.X, 0, 1.0, diff=1)
        assert np.allclose(y, -y[::-1], atol=1e-6)

    def test_diff_1_zero_crossing(self):
        """First derivative crosses zero at the peak."""
        y = lshape(self.X, 0, 1.0, diff=1)
        # Zero crossing near the centre
        centre = len(y) // 2
        assert y[centre - 5] > 0 and y[centre + 5] < 0  # positive left, negative right

    def test_diff_2_symmetric(self):
        """Second derivative is symmetric about x0."""
        y = lshape(self.X, 0, 1.0, diff=2)
        assert np.allclose(y, y[::-1], atol=1e-4)

    def test_diff_minus1_monotone(self):
        """Cumulative integral (diff=-1) is monotonically non-decreasing."""
        y = lshape(self.X, 0, 1.0, diff=-1)
        assert np.all(np.diff(y) >= -1e-12)

    def test_two_widths(self):
        """fwhm=[G_width, L_width] uses separate widths."""
        y = lshape(self.X, 0, [1.0, 2.0])
        assert y.max() > 0

    def test_dispersion_phase_pi2(self):
        """phase=pi/2 gives pure dispersion (antisymmetric)."""
        y = lshape(self.X, 0, 1.0, phase=np.pi/2)
        assert np.allclose(y, -y[::-1], atol=1e-6)

    def test_invalid_diff(self):
        """diff != -1,0,1,2 raises ValueError."""
        with pytest.raises(ValueError, match="diff"):
            lshape(self.X, 0, 1.0, diff=3)
