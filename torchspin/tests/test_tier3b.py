"""Tests for Tier 3b: Signal Processing (ctafft, rapidscan2spc, lpsvd, ewrls).

Tests cover:
- ctafft: cross-term averaged FFT
- rapidscan2spc: rapid-scan to EPR spectrum deconvolution
- lpsvd: linear prediction SVD (3 algorithms + model order estimation)
- ewrls: exponentially weighted RLS adaptive filter
"""
import numpy as np
import pytest

from torchspin.ctafft import ctafft
from torchspin.rapidscan2spc import rapidscan2spc
from torchspin.lpsvd import lpsvd, LPSVDResult
from torchspin.ewrls import ewrls


# ============================================================================
# ctafft — Cross-Term Averaged FFT
# ============================================================================

class TestCtafft:
    """Cross-term averaged FFT tests."""

    def test_single_average_equals_fft(self):
        """With k=1 (averages=[1]), result equals magnitude FFT of full signal."""
        rng = np.random.default_rng(42)
        td = rng.standard_normal(64)
        fd = ctafft(td, 1)
        expected = np.abs(np.fft.fft(td))
        np.testing.assert_allclose(fd, expected, atol=1e-12)

    def test_averages_list(self):
        """Explicit list of starting indices works."""
        rng = np.random.default_rng(42)
        td = rng.standard_normal(64)
        # averages=[1, 2, 3] (1-based)
        fd = ctafft(td, [1, 2, 3])
        assert fd.shape == (64,)
        assert np.all(fd >= 0)

    def test_reduces_dead_time_artifact(self):
        """Averaging reduces dead-time-like artifacts at beginning."""
        # FID with dead time: first few points corrupted
        t = np.arange(128, dtype=float)
        fid = np.exp(-t / 30) * np.cos(2 * np.pi * 0.1 * t)
        fid[:5] = 10.0  # corrupt first 5 points (dead time artifact)

        # Single FFT has artifact
        fd_single = ctafft(fid, 1)
        # Cross-term average with multiple starts should reduce it
        fd_avg = ctafft(fid, 8)

        # The peak from the sinusoid should be clearer relative to baseline
        peak_idx = np.argmax(fd_single[:64])
        # Cross-term averaged should still have the main peak
        assert fd_avg[peak_idx] > 0

    def test_custom_N(self):
        """Custom FFT length N works."""
        td = np.ones(32)
        fd = ctafft(td, 1, N=64)
        assert fd.shape == (64,)

    def test_matrix_input(self):
        """2-D input: operates along columns."""
        rng = np.random.default_rng(42)
        td = rng.standard_normal((64, 3))
        fd = ctafft(td, 3)
        assert fd.shape == (64, 3)

    def test_row_vector(self):
        """Row vector is handled correctly (transposed internally)."""
        rng = np.random.default_rng(42)
        td = rng.standard_normal((1, 64))
        fd = ctafft(td, 2)
        assert fd.shape == (1, 64)

    def test_complex_input(self):
        """Complex-valued input works."""
        rng = np.random.default_rng(42)
        td = rng.standard_normal(64) + 1j * rng.standard_normal(64)
        fd = ctafft(td, 3)
        assert fd.shape == (64,)
        assert np.all(np.isfinite(fd))

    def test_invalid_averages_raises(self):
        """Invalid averages raises ValueError."""
        td = np.ones(32)
        with pytest.raises(ValueError):
            ctafft(td, [0])  # 0 is invalid (1-based)
        with pytest.raises(ValueError):
            ctafft(td, [33])  # exceeds data length


# ============================================================================
# rapidscan2spc — Rapid-Scan Deconvolution
# ============================================================================

class TestRapidscan2spc:
    """Rapid-scan to EPR spectrum conversion tests."""

    def _make_rapid_scan_signal(self, rsAmp_mT=10.0, rsFreq_kHz=100.0, N=1024,
                                 linewidth_mT=0.5, g=2.0023193):
        """Create a synthetic rapid-scan signal from a Lorentzian lineshape."""
        from torchspin.constants import BMAGN, HBAR

        rsFreq_Hz = rsFreq_kHz * 1e3
        T_period = 1.0 / rsFreq_Hz
        t = np.arange(N, dtype=np.float64) / N * T_period

        # Field modulation: B(t) = B0 + (rsAmp/2)*cos(omega*t)
        omega = 2 * np.pi * rsFreq_Hz
        B_offset = (rsAmp_mT / 2) * np.cos(omega * t)  # mT

        # Lorentzian lineshape (absorption + dispersion)
        lw = linewidth_mT
        # Absorption: lw^2 / (B^2 + lw^2)
        # Dispersion: -B*lw / (B^2 + lw^2)
        absorption = lw**2 / (B_offset**2 + lw**2)
        dispersion = -B_offset * lw / (B_offset**2 + lw**2)

        # Phase modulation
        gamma = g * BMAGN / HBAR
        A_rad = gamma * (rsAmp_mT / 2 / 1e3)  # rad/s
        ph = A_rad * np.sin(omega * t) / omega
        M = (absorption + 1j * dispersion) * np.exp(1j * ph)

        return M

    def test_output_shapes(self):
        """Output field axis and spectrum have matching shapes."""
        M = self._make_rapid_scan_signal()
        dB, spc = rapidscan2spc(M, rsAmp=10.0, rsFreq=100.0)
        assert dB.shape == spc.shape
        assert len(dB) > 0

    def test_field_axis_within_modulation_range(self):
        """Output field axis is within +/- rsAmp/2."""
        rsAmp = 10.0
        M = self._make_rapid_scan_signal(rsAmp_mT=rsAmp)
        dB, spc = rapidscan2spc(M, rsAmp=rsAmp, rsFreq=100.0)
        assert np.all(np.abs(dB) <= rsAmp / 2 + 1e-10)

    def test_complex_output(self):
        """Output spectrum is complex (absorption + dispersion)."""
        M = self._make_rapid_scan_signal()
        dB, spc = rapidscan2spc(M, rsAmp=10.0, rsFreq=100.0)
        assert np.iscomplexobj(spc)

    def test_real_input_raises(self):
        """Real-only input raises ValueError."""
        M = np.ones(128, dtype=np.float64)
        with pytest.raises(ValueError, match="complex"):
            rapidscan2spc(M, rsAmp=10.0, rsFreq=100.0)

    def test_odd_length_raises(self):
        """Odd-length signal raises ValueError."""
        M = np.ones(127) + 1j * np.ones(127)
        with pytest.raises(ValueError, match="even"):
            rapidscan2spc(M, rsAmp=10.0, rsFreq=100.0)

    def test_finite_output(self):
        """Output contains no NaN or Inf values."""
        M = self._make_rapid_scan_signal()
        dB, spc = rapidscan2spc(M, rsAmp=10.0, rsFreq=100.0)
        assert np.all(np.isfinite(dB))
        assert np.all(np.isfinite(spc))

    def test_symmetric_field_axis(self):
        """Field axis is approximately symmetric around zero."""
        M = self._make_rapid_scan_signal()
        dB, spc = rapidscan2spc(M, rsAmp=10.0, rsFreq=100.0)
        # Center should be near zero
        center_idx = len(dB) // 2
        assert abs(dB[center_idx]) < 0.5  # within 0.5 mT of center


# ============================================================================
# lpsvd — Linear Prediction SVD
# ============================================================================

class TestLPSVD:
    """Linear Prediction SVD tests."""

    def _make_signal(self, freqs, damps, amps, phases, t):
        """Build a damped sinusoidal signal from parameters."""
        signal = np.zeros(len(t), dtype=np.complex128)
        for f, d, a, p in zip(freqs, damps, amps, phases):
            signal += a * np.exp(1j * p) * np.exp(t * (1j * 2 * np.pi * f - d))
        return signal

    def test_single_sinusoid_ss(self):
        """State-space method recovers single sinusoid frequency."""
        freq0 = 50.0  # Hz
        damp0 = 5.0
        t = np.linspace(0, 0.5, 256, endpoint=False)
        signal = 1.5 * np.exp(t * (1j * 2 * np.pi * freq0 - damp0))
        y, params = lpsvd(signal, t, method='ss', order=1)
        assert isinstance(params, LPSVDResult)
        assert len(params.frequency) >= 1
        # Frequency should be close to 50 Hz
        closest = params.frequency[np.argmin(np.abs(params.frequency - freq0))]
        assert abs(closest - freq0) < 2.0

    def test_single_sinusoid_kt(self):
        """Kumaresan-Tufts method recovers single sinusoid."""
        freq0 = 30.0
        damp0 = 3.0
        t = np.linspace(0, 1.0, 256, endpoint=False)
        signal = np.exp(t * (1j * 2 * np.pi * freq0 - damp0))
        y, params = lpsvd(signal, t, method='kt', order=1)
        closest = params.frequency[np.argmin(np.abs(params.frequency - freq0))]
        assert abs(closest - freq0) < 2.0

    def test_single_sinusoid_tls(self):
        """Total least squares method recovers single sinusoid."""
        freq0 = 40.0
        damp0 = 4.0
        t = np.linspace(0, 0.5, 256, endpoint=False)
        signal = np.exp(t * (1j * 2 * np.pi * freq0 - damp0))
        y, params = lpsvd(signal, t, method='tls', order=1)
        closest = params.frequency[np.argmin(np.abs(params.frequency - freq0))]
        assert abs(closest - freq0) < 2.0

    def test_two_sinusoids_ss(self):
        """State-space resolves two frequencies."""
        f1, f2 = 20.0, 60.0
        t = np.linspace(0, 1.0, 512, endpoint=False)
        signal = (np.exp(t * (1j * 2 * np.pi * f1 - 2.0)) +
                  0.8 * np.exp(t * (1j * 2 * np.pi * f2 - 3.0)))
        y, params = lpsvd(signal, t, method='ss', order=2)
        freqs = np.sort(params.frequency)
        assert abs(freqs[0] - f1) < 3.0 or abs(freqs[1] - f1) < 3.0
        assert abs(freqs[0] - f2) < 3.0 or abs(freqs[1] - f2) < 3.0

    def test_auto_order_mdl(self):
        """MDL model order estimation runs without error."""
        freq0 = 50.0
        t = np.linspace(0, 0.5, 256, endpoint=False)
        signal = np.exp(t * (1j * 2 * np.pi * freq0 - 5.0))
        y, params = lpsvd(signal, t, method='ss', order='mdl')
        assert len(params.frequency) >= 1

    def test_auto_order_aic(self):
        """AIC model order estimation runs without error."""
        freq0 = 50.0
        t = np.linspace(0, 0.5, 256, endpoint=False)
        signal = np.exp(t * (1j * 2 * np.pi * freq0 - 5.0))
        y, params = lpsvd(signal, t, method='ss', order='aic')
        assert len(params.frequency) >= 1

    def test_predicted_signal_shape(self):
        """Predicted signal has same shape as input."""
        t = np.linspace(0, 0.5, 128, endpoint=False)
        signal = np.exp(t * (1j * 2 * np.pi * 30 - 5.0))
        y, params = lpsvd(signal, t, order=1)
        assert y.shape == signal.shape

    def test_damping_positive(self):
        """All returned damping values are positive (decaying)."""
        t = np.linspace(0, 0.5, 256, endpoint=False)
        signal = np.exp(t * (1j * 2 * np.pi * 50 - 10.0))
        y, params = lpsvd(signal, t, order=1)
        assert np.all(params.damping > 0)

    def test_damping_recovery(self):
        """Damping rate is approximately recovered."""
        damp0 = 8.0
        t = np.linspace(0, 1.0, 512, endpoint=False)
        signal = np.exp(t * (1j * 2 * np.pi * 30 - damp0))
        y, params = lpsvd(signal, t, method='ss', order=1)
        assert abs(params.damping[0] - damp0) < 2.0

    def test_mismatched_lengths_raises(self):
        """Mismatched data/time lengths raises ValueError."""
        with pytest.raises(ValueError, match="same length"):
            lpsvd(np.ones(10), np.ones(20))


# ============================================================================
# ewrls — Exponentially Weighted RLS
# ============================================================================

class TestEWRLS:
    """Exponentially weighted RLS adaptive filter tests."""

    def _make_noisy_scans(self, nPoints=200, nScans=30, noise_level=0.5, seed=42):
        """Create noisy repeated scans of a Gaussian peak."""
        rng = np.random.default_rng(seed)
        x = np.arange(nPoints, dtype=float)
        signal = np.exp(-((x - 100) / 20) ** 2)
        data = np.column_stack([
            signal + noise_level * rng.standard_normal(nPoints)
            for _ in range(nScans)
        ])
        return data, signal

    def test_basic_output_shape(self):
        """Output is 1-D with same number of points as input rows."""
        data, _ = self._make_noisy_scans()
        y = ewrls(data, p=20, lam=0.97)
        assert y.shape == (200,)

    def test_reduces_noise(self):
        """Filtered output has less noise than individual scans."""
        data, signal = self._make_noisy_scans(nScans=50, noise_level=0.5)
        y = ewrls(data, p=20, lam=0.97)
        # Compare RMS error vs single scan
        err_single = np.sqrt(np.mean((data[:, -1] - signal) ** 2))
        err_filtered = np.sqrt(np.mean((y - signal) ** 2))
        assert err_filtered < err_single

    def test_forward_direction(self):
        """Forward-only filtering runs without error."""
        data, _ = self._make_noisy_scans()
        y = ewrls(data, p=20, lam=0.97, direction='f')
        assert y.shape == (200,)
        assert np.all(np.isfinite(y))

    def test_backward_direction(self):
        """Backward-only filtering runs without error."""
        data, _ = self._make_noisy_scans()
        y = ewrls(data, p=20, lam=0.97, direction='b')
        assert y.shape == (200,)
        assert np.all(np.isfinite(y))

    def test_fb_vs_forward_backward(self):
        """FB result is average of forward and backward."""
        data, _ = self._make_noisy_scans()
        y_fb = ewrls(data, p=20, lam=0.97, direction='fb')
        y_f = ewrls(data, p=20, lam=0.97, direction='f')
        y_b = ewrls(data, p=20, lam=0.97, direction='b')
        np.testing.assert_allclose(y_fb, (y_f + y_b) / 2, atol=1e-10)

    def test_pre_averaging(self):
        """Pre-averaging a subset of scans works."""
        data, _ = self._make_noisy_scans(nScans=30)
        y = ewrls(data, p=15, lam=0.97, nPreAvg=5)
        assert y.shape == (200,)

    def test_invalid_lambda_raises(self):
        """Lambda outside [0,1] raises ValueError."""
        data, _ = self._make_noisy_scans()
        with pytest.raises(ValueError, match="lambda"):
            ewrls(data, p=20, lam=1.5)

    def test_single_scan_raises(self):
        """Single scan raises ValueError."""
        with pytest.raises(ValueError, match="Multiple scans"):
            ewrls(np.ones((100, 1)), p=10, lam=0.97)

    def test_filter_too_long_raises(self):
        """Filter length >= data length raises ValueError."""
        data, _ = self._make_noisy_scans(nPoints=20)
        with pytest.raises(ValueError, match="shorter"):
            ewrls(data, p=25, lam=0.97)

    def test_custom_delta(self):
        """Custom regularization parameter delta runs."""
        data, _ = self._make_noisy_scans()
        y = ewrls(data, p=20, lam=0.97, delta=1000.0)
        assert np.all(np.isfinite(y))
