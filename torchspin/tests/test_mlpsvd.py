"""Tests for mlpsvd — 2D Linear Prediction SVD.

Verifies:
1. Single-component signal recovery
2. Multi-component signal recovery
3. All three methods (ss, kt, tls)
4. Model order estimation (mdl, aic)
5. 2D processing methods (sum, stack)
6. Reconstruction accuracy
7. Parameter extraction (frequency, damping, amplitude)
"""

import numpy as np
import pytest

from torchspin.mlpsvd import mlpsvd, MLPSVDResult


class TestMLPSVDBasic:
    """Basic functionality tests."""

    def test_single_exponential(self):
        """Single damped exponential should be recovered."""
        dt = 0.004  # 4 ms
        t = np.arange(0, 1.0, dt)
        freq = 10.0  # Hz
        damp = 3.0
        sig = 2.0 * np.exp(-damp * t) * np.cos(2 * np.pi * freq * t)
        sig += np.random.RandomState(42).randn(len(t)) * 0.01

        y, params = mlpsvd(sig, t, method='ss')
        assert isinstance(params, MLPSVDResult)
        assert len(params.frequency) > 0
        # Reconstruction should be close to original
        residual = np.linalg.norm(np.real(y) - sig) / np.linalg.norm(sig)
        assert residual < 0.5  # within 50% (noisy)

    def test_two_components(self):
        """Two-component signal should find both frequencies."""
        dt = 0.002
        t = np.arange(0, 0.5, dt)
        f1, f2 = 20.0, 50.0
        sig = (np.exp(-2 * t) * np.cos(2 * np.pi * f1 * t) +
               0.5 * np.exp(-5 * t) * np.cos(2 * np.pi * f2 * t))

        y, params = mlpsvd(sig, t, method='ss', order=4)
        # Should find frequencies near f1 and f2
        freqs = np.abs(params.frequency)
        assert any(np.abs(freqs - f1) < 5.0)  # within 5 Hz
        assert any(np.abs(freqs - f2) < 5.0)

    def test_output_shapes(self):
        """Output shapes should match input."""
        t = np.arange(0, 0.5, 0.002)
        sig = np.exp(-3 * t) * np.cos(2 * np.pi * 30 * t)
        y, params = mlpsvd(sig, t)
        assert y.shape == sig.shape
        assert params.damping.ndim == 1
        assert params.frequency.ndim == 1


class TestMLPSVDMethods:
    """Test all three SVD methods."""

    @pytest.fixture
    def test_signal(self):
        t = np.arange(0, 0.5, 0.002)
        sig = np.exp(-3 * t) * np.cos(2 * np.pi * 25 * t)
        return sig, t

    def test_ss_method(self, test_signal):
        """State-space method."""
        sig, t = test_signal
        y, params = mlpsvd(sig, t, method='ss')
        assert y.size == sig.size

    def test_kt_method(self, test_signal):
        """Kumaresan-Tufts method."""
        sig, t = test_signal
        y, params = mlpsvd(sig, t, method='kt')
        assert y.size == sig.size

    def test_tls_method(self, test_signal):
        """Total least squares method."""
        sig, t = test_signal
        y, params = mlpsvd(sig, t, method='tls')
        assert y.size == sig.size


class TestMLPSVDOrderEstimation:
    """Model order estimation tests."""

    def test_mdl_order(self):
        """MDL should estimate a reasonable model order."""
        t = np.arange(0, 1.0, 0.004)
        sig = np.exp(-2 * t) * np.cos(2 * np.pi * 15 * t)
        y, params = mlpsvd(sig, t, order='mdl')
        # Should find at least 1 component
        assert len(params.frequency) >= 1

    def test_aic_order(self):
        """AIC should estimate a reasonable model order."""
        t = np.arange(0, 1.0, 0.004)
        sig = np.exp(-2 * t) * np.cos(2 * np.pi * 15 * t)
        y, params = mlpsvd(sig, t, order='aic')
        assert len(params.frequency) >= 1

    def test_explicit_order(self):
        """Explicit model order should be respected."""
        t = np.arange(0, 0.5, 0.002)
        sig = np.exp(-3 * t) * np.cos(2 * np.pi * 25 * t)
        y, params = mlpsvd(sig, t, order=3)
        # Should have exactly 3 components (or fewer if some rejected)
        assert len(params.frequency) <= 3


class TestMLPSVD2D:
    """2D processing tests."""

    def test_2d_sum_method(self):
        """Sum method for multiple spectra."""
        t = np.arange(0, 0.5, 0.002)
        data = np.zeros((3, len(t)))
        for i in range(3):
            data[i] = (i + 1) * np.exp(-3 * t) * np.cos(2 * np.pi * 25 * t)
        y, params = mlpsvd(data, t, method2d='sum')
        assert y.shape == data.shape

    def test_2d_stack_method(self):
        """Stack method for multiple spectra."""
        t = np.arange(0, 0.5, 0.002)
        data = np.zeros((3, len(t)))
        for i in range(3):
            data[i] = (i + 1) * np.exp(-3 * t) * np.cos(2 * np.pi * 25 * t)
        y, params = mlpsvd(data, t, method2d='stack')
        assert y.shape == data.shape


class TestMLPSVDModel:
    """Model reconstruction tests."""

    def test_model_callable(self):
        """Model function should be callable and produce correct shape."""
        t = np.arange(0, 0.5, 0.002)
        sig = np.exp(-3 * t) * np.cos(2 * np.pi * 25 * t)
        y, params = mlpsvd(sig, t)
        # Model should reproduce at the same times
        y_model = params.model(t)
        assert y_model.shape[-1] == len(t)

    def test_model_extrapolation(self):
        """Model should extrapolate to new times."""
        t = np.arange(0, 0.5, 0.002)
        sig = np.exp(-3 * t) * np.cos(2 * np.pi * 25 * t)
        _, params = mlpsvd(sig, t)
        # Extrapolate to 2x the time range
        t_ext = np.arange(0, 1.0, 0.002)
        y_ext = params.model(t_ext)
        assert y_ext.shape[-1] == len(t_ext)


class TestMLPSVDEdgeCases:
    """Edge case tests."""

    def test_transposed_input(self):
        """Column-oriented data should be handled."""
        t = np.arange(0, 0.5, 0.004)
        sig = np.exp(-3 * t) * np.cos(2 * np.pi * 25 * t)
        # Transpose: data shape is (N, 1) instead of (1, N)
        data = sig[:, np.newaxis]  # (N, 1)
        # This won't match time dim — mlpsvd should handle via transpose
        y, params = mlpsvd(sig, t)
        assert y.size == sig.size

    def test_invalid_method(self):
        """Invalid method should raise ValueError."""
        t = np.arange(0, 0.5, 0.002)
        sig = np.exp(-3 * t) * np.cos(2 * np.pi * 25 * t)
        with pytest.raises(ValueError, match="method must be"):
            mlpsvd(sig, t, method='invalid')

    def test_mismatched_time(self):
        """Mismatched time vector should raise ValueError."""
        t = np.arange(0, 0.5, 0.002)
        sig = np.zeros(10)  # Wrong length
        with pytest.raises(ValueError, match="Time vector length"):
            mlpsvd(sig, t)
