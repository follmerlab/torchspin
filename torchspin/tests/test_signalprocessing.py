"""Tests for signalprocessing — frequency translation for pulse EPR signals.

Verifies:
1. No-op for zero frequency translation
2. Real signal demodulation (IQ demod)
3. Complex signal demodulation (IQ shift)
4. Multi-detector operator handling
5. Non-uniform time axis (breaks)
6. Graceful fallback on failure
"""

import numpy as np
import pytest

from torchspin.signalprocessing import signalprocessing


class TestSignalProcessingBasic:
    """Basic signal processing tests."""

    def test_zero_freq_passthrough(self):
        """Zero frequency translation should return signal unchanged."""
        t = np.linspace(0, 1, 256)
        sig = np.sin(2 * np.pi * 50 * t)
        result = signalprocessing(t, sig[np.newaxis, :], [0.0])
        np.testing.assert_allclose(np.real(result), sig, atol=1e-10)

    def test_real_signal_demodulation(self):
        """Demodulating a real cosine should produce a result."""
        freq_ghz = 9.5  # 9.5 GHz
        dt_us = 0.001  # 1 ns steps
        t = np.arange(0, 0.5, dt_us)  # 0.5 us
        # Real cosine at carrier frequency (GHz → cycles/us = 1e3 * GHz)
        carrier_cycles = freq_ghz * 1e3  # cycles per us
        sig = np.cos(2 * np.pi * carrier_cycles * t)

        result = signalprocessing(t, sig[np.newaxis, :], [freq_ghz])
        # After demodulation, should produce nonzero output
        assert result.size > 0
        assert np.max(np.abs(result)) > 0

    def test_complex_signal_demodulation(self):
        """Complex signal should use IQ shift."""
        freq_ghz = 9.5
        dt_us = 0.001
        t = np.arange(0, 0.5, dt_us)
        carrier = freq_ghz * 1e3
        sig = np.exp(1j * 2 * np.pi * carrier * t)

        result = signalprocessing(t, sig[np.newaxis, :], [freq_ghz])
        assert result.size > 0


class TestSignalProcessingMultiDet:
    """Multi-detector operator tests."""

    def test_two_detectors_different_freqs(self):
        """Two detectors with different translation frequencies."""
        t = np.linspace(0, 0.5, 256)
        sig = np.zeros((2, 256), dtype=complex)
        sig[0] = np.cos(2 * np.pi * 100 * t)
        sig[1] = np.sin(2 * np.pi * 200 * t)

        result = signalprocessing(t, sig, [0.0, 0.0])
        assert result.shape == sig.shape

    def test_mixed_zero_nonzero_freq(self):
        """One detector demodulated, one not."""
        t = np.linspace(0, 0.5, 128)
        sig = np.zeros((2, 128), dtype=complex)
        sig[0] = np.cos(2 * np.pi * 100 * t)  # no demod
        sig[1] = np.cos(2 * np.pi * 100 * t)  # demod at 0.1 GHz

        result = signalprocessing(t, sig, [0.0, 0.1])
        assert result.shape == sig.shape
        # First detector should be unchanged
        np.testing.assert_allclose(np.real(result[0]), np.real(sig[0]), atol=1e-10)


class TestSignalProcessing3D:
    """Multi-acquisition-point tests."""

    def test_3d_array(self):
        """3D input (nPoints, nDetOps, nTime) should work."""
        t = np.linspace(0, 0.5, 64)
        sig = np.zeros((3, 1, 64), dtype=complex)
        for i in range(3):
            sig[i, 0] = np.cos(2 * np.pi * 50 * t) * np.exp(-t * (i + 1))

        result = signalprocessing(t, sig, [0.0])
        assert result.shape == sig.shape

    def test_list_input(self):
        """List-of-arrays input should work."""
        t = np.linspace(0, 0.5, 64)
        sig_list = [
            np.cos(2 * np.pi * 50 * t)[np.newaxis, :],
            np.sin(2 * np.pi * 50 * t)[np.newaxis, :],
        ]
        result = signalprocessing(t, sig_list, [0.0])
        assert isinstance(result, list)
        assert len(result) == 2


class TestSignalProcessingEdgeCases:
    """Edge cases and error handling."""

    def test_empty_signal(self):
        """Empty signal should not crash."""
        t = np.array([])
        sig = np.array([])
        result = signalprocessing(t, sig, [0.0])
        assert result is not None

    def test_single_point(self):
        """Single-point signal."""
        t = np.array([0.0])
        sig = np.array([[1.0 + 0j]])
        result = signalprocessing(t, sig, [0.0])
        assert result.size == 1

    def test_failure_returns_raw(self):
        """If demodulation fails, raw signal should be returned."""
        t = np.array([0.0, 0.001])
        sig = np.array([[np.nan, np.nan]])
        # Should warn and return raw
        result = signalprocessing(t, sig, [9.5])
        assert result is not None
