"""Tests for Tier 3d-i: Pulse EPR primitives.

Tests cover:
- resonatorprofile: RLC resonator transfer/reflection profiles
- transmitter: nonlinear amplitude compression/compensation
- rfmixer: digital up/downconversion (DSB, SSB, IQ)
- propint: piecewise-constant propagator integration
"""
import numpy as np
import pytest

from torchspin.resonatorprofile import resonatorprofile
from torchspin.transmitter import transmitter
from torchspin.rfmixer import rfmixer
from torchspin.propint import propint


# ============================================================================
# resonatorprofile
# ============================================================================

class TestResonatorProfile:
    """Resonator profile tests."""

    def test_transfer_peak_at_nu0(self):
        """Transfer function peaks at resonance frequency."""
        nu0 = 9.5
        nu = np.linspace(9.0, 10.0, 1001)
        H = resonatorprofile(nu, nu0=nu0, Qu=1000, beta=1,
                             mode='transferfunction')
        peak_idx = np.argmax(np.abs(H))
        assert abs(nu[peak_idx] - nu0) < 0.01

    def test_transfer_at_resonance_is_one(self):
        """Transfer function at resonance with no coupling has magnitude ~1."""
        nu = np.array([9.5])
        H = resonatorprofile(nu, nu0=9.5, Qu=1000,
                             mode='transferfunction')
        assert abs(abs(H[0]) - 1.0) < 0.01

    def test_voltage_reflection_at_critical_coupling(self):
        """At critical coupling (beta=1), reflection is zero at resonance."""
        nu = np.array([9.5])
        Gamma = resonatorprofile(nu, nu0=9.5, Qu=1000, beta=1,
                                 mode='voltagereflection')
        assert abs(Gamma[0]) < 0.01

    def test_power_reflection_off_resonance(self):
        """Power reflection is ~1 far from resonance."""
        nu = np.array([5.0])  # very far from nu0=9.5
        P = resonatorprofile(nu, nu0=9.5, Qu=1000, beta=1,
                             mode='powerreflection')
        assert P[0] > 0.9

    def test_Q_from_bandwidth(self):
        """Q-factor estimated from -3dB bandwidth matches input."""
        nu0 = 9.5
        Qu = 500
        nu = np.linspace(9.0, 10.0, 10001)
        H = resonatorprofile(nu, nu0=nu0, Qu=Qu,
                             mode='transferfunction')
        H_mag = np.abs(H)
        # -3dB points
        threshold = np.max(H_mag) / np.sqrt(2)
        above = nu[H_mag >= threshold]
        bw = above[-1] - above[0]
        Q_est = nu0 / bw
        assert abs(Q_est - Qu) / Qu < 0.05

    def test_auto_frequency_range(self):
        """None frequency generates automatic range."""
        nu, H = resonatorprofile(None, nu0=9.5, Qu=1000, beta=1,
                                 mode='transferfunction')
        assert len(nu) == 1001
        assert nu[0] < 9.5 < nu[-1]

    def test_overcoupled_reflection(self):
        """Overcoupled resonator has higher reflection at resonance than critical."""
        nu = np.array([9.5])
        Gamma_crit = abs(resonatorprofile(
            nu, nu0=9.5, Qu=1000, beta=1, mode='voltagereflection')[0])
        Gamma_over = abs(resonatorprofile(
            nu, nu0=9.5, Qu=1000, beta=5, mode='voltagereflection')[0])
        # Overcoupled should have non-zero reflection (sign flip)
        assert Gamma_over > Gamma_crit

    def test_unknown_mode_raises(self):
        """Unknown mode raises ValueError."""
        nu = np.linspace(9, 10, 100)
        with pytest.raises(ValueError, match="Unknown mode"):
            resonatorprofile(nu, nu0=9.5, Qu=1000, mode='invalid')


# ============================================================================
# transmitter
# ============================================================================

class TestTransmitter:
    """Transmitter nonlinearity tests."""

    def _linear_curve(self, n=50):
        Ain = np.linspace(0, 1, n)
        Aout = Ain.copy()
        return Ain, Aout

    def _compressed_curve(self, n=50):
        """Compression curve where Aout < Ain (saturation behaviour)."""
        Ain = np.linspace(0, 1, n)
        Aout = Ain / (1 + 0.5 * Ain)  # always < Ain for Ain > 0
        return Ain, Aout

    def test_linear_no_change(self):
        """Linear transmitter doesn't change the signal."""
        Ain, Aout = self._linear_curve()
        signal = np.sin(np.linspace(0, 2 * np.pi, 200))
        result = transmitter(signal, Ain, Aout, 'simulate')
        np.testing.assert_allclose(result, signal, atol=0.02)

    def test_compression_reduces_peaks(self):
        """Compressed transmitter reduces peak amplitudes."""
        Ain, Aout = self._compressed_curve()
        # Use amplitude well within the curve range to avoid polynomial overshoot
        signal = 0.8 * np.sin(np.linspace(0, 2 * np.pi, 200))
        result = transmitter(signal, Ain, Aout, 'simulate')
        assert np.max(np.abs(result)) < np.max(np.abs(signal))

    def test_compensate_increases_peaks(self):
        """Compensated signal has higher peaks to counteract compression."""
        Ain, Aout = self._compressed_curve()
        # Use moderate amplitude where compression is clearly active
        signal = 0.6 * np.sin(np.linspace(0, 2 * np.pi, 200))
        result = transmitter(signal, Ain, Aout, 'compensate')
        # Compensated signal should have larger amplitude
        assert np.max(np.abs(result)) > np.max(np.abs(signal))

    def test_simulate_compensate_roundtrip(self):
        """simulate(compensate(signal)) ≈ signal for moderate amplitudes."""
        Ain, Aout = self._compressed_curve()
        signal = 0.3 * np.sin(np.linspace(0, 2 * np.pi, 200))
        compensated = transmitter(signal, Ain, Aout, 'compensate')
        roundtrip = transmitter(compensated, Ain, Aout, 'simulate')
        np.testing.assert_allclose(roundtrip, signal, atol=0.05)

    def test_complex_signal(self):
        """Complex input works (I and Q processed separately)."""
        Ain, Aout = self._compressed_curve()
        signal = (0.5 * np.sin(np.linspace(0, 2 * np.pi, 100))
                  + 0.3j * np.cos(np.linspace(0, 2 * np.pi, 100)))
        result = transmitter(signal, Ain, Aout, 'simulate')
        assert np.iscomplexobj(result)
        assert np.all(np.isfinite(result))

    def test_unknown_option_raises(self):
        """Unknown option raises ValueError."""
        Ain, Aout = self._linear_curve()
        with pytest.raises(ValueError, match="Unknown option"):
            transmitter(np.ones(10), Ain, Aout, 'invalid')


# ============================================================================
# rfmixer
# ============================================================================

class TestRFMixer:
    """RF mixer tests."""

    def _baseband_signal(self, f_mhz=50, n=10000, t_max=1.0):
        """Create a baseband cosine signal."""
        t = np.linspace(0, t_max, n, endpoint=False)  # µs
        signal = np.cos(2 * np.pi * f_mhz * t)  # f_mhz in MHz, t in µs
        return t, signal

    def test_dsb_output_real(self):
        """DSB output is real."""
        t, signal = self._baseband_signal(f_mhz=10, n=4000)
        t_out, s_out = rfmixer(t, signal, mwFreq=0.01, mode='DSB')
        assert np.isrealobj(s_out)

    def test_dsb_has_carrier(self):
        """DSB output contains carrier frequency component."""
        t, signal = self._baseband_signal(f_mhz=10, n=10000)
        mwFreq = 0.1  # GHz = 100 MHz
        t_out, s_out = rfmixer(t, signal, mwFreq=mwFreq, mode='DSB')
        # Check FFT for peaks near 100±10 MHz
        f = np.fft.fftfreq(len(s_out), d=t_out[1] - t_out[0])
        ft = np.abs(np.fft.fft(s_out))
        # Should have energy around 90 and 110 MHz
        peak_freqs = np.abs(f[np.argsort(ft)[-4:]])
        assert any(abs(pf - 90) < 15 or abs(pf - 110) < 15 for pf in peak_freqs)

    def test_iqmod_output_real(self):
        """IQmod output is real."""
        t = np.linspace(0, 1, 5000, endpoint=False)
        signal = np.cos(2 * np.pi * 10 * t) + 1j * np.sin(2 * np.pi * 10 * t)
        t_out, s_out = rfmixer(t, signal, mwFreq=0.05, mode='IQmod')
        assert np.isrealobj(s_out)

    def test_iqdemod_output_complex(self):
        """IQdemod output is complex."""
        t = np.linspace(0, 1, 10000, endpoint=False)
        signal = np.cos(2 * np.pi * 100 * t)  # 100 MHz carrier
        t_out, s_out = rfmixer(t, signal, mwFreq=0.1, mode='IQdemod')
        assert np.iscomplexobj(s_out)

    def test_complex_input_for_dsb_raises(self):
        """Complex input with DSB mode raises ValueError."""
        t = np.linspace(0, 1, 1000)
        signal = np.ones(1000) + 1j * np.ones(1000)
        with pytest.raises(ValueError, match="real"):
            rfmixer(t, signal, mwFreq=0.01, mode='DSB')

    def test_unknown_mode_raises(self):
        """Unknown mode raises ValueError."""
        t = np.linspace(0, 1, 100)
        signal = np.ones(100)
        with pytest.raises(ValueError, match="Unknown mixer"):
            rfmixer(t, signal, mwFreq=0.01, mode='XYZ')

    def test_output_length_with_resampling(self):
        """Output length is reasonable after resampling."""
        t = np.linspace(0, 1, 1000, endpoint=False)
        signal = np.cos(2 * np.pi * 10 * t)
        t_out, s_out = rfmixer(t, signal, mwFreq=0.01, mode='DSB')
        assert len(t_out) == len(s_out)
        assert len(t_out) > 0

    def test_iqshift_preserves_complex(self):
        """IQshift output is complex."""
        t = np.linspace(0, 1, 5000, endpoint=False)
        signal = np.exp(2j * np.pi * 10 * t)
        t_out, s_out = rfmixer(t, signal, mwFreq=0.01, mode='IQshift')
        assert np.iscomplexobj(s_out)


# ============================================================================
# propint
# ============================================================================

class TestPropint:
    """Propagator integration tests."""

    def test_identity_for_zero_H(self):
        """Zero Hamiltonian gives identity propagator."""
        H0 = np.zeros((2, 2))
        H1 = np.zeros((2, 2))
        U = propint(H0, H1, 1.0, 10.0)
        np.testing.assert_allclose(U, np.eye(2), atol=1e-10)

    def test_no_time_dep_matches_expm(self):
        """Without time-dependent part, matches simple matrix exponential."""
        from scipy.linalg import expm
        H0 = np.array([[1.0, 0.5], [0.5, -1.0]], dtype=complex)
        H1 = np.zeros((2, 2), dtype=complex)
        t = 0.5
        freq = 10.0
        U = propint(H0, H1, t, freq)
        U_expected = expm(-2j * np.pi * t * H0)
        np.testing.assert_allclose(U, U_expected, atol=1e-10)

    def test_unitary(self):
        """Propagator is unitary."""
        Sz = np.diag([0.5, -0.5]).astype(complex)
        Sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
        mwFreq = 100.0  # MHz
        tp = 0.01  # µs
        U = propint(mwFreq * Sz, 0.5 / tp * Sx, tp, mwFreq)
        I = U @ U.conj().T
        np.testing.assert_allclose(I, np.eye(2), atol=1e-6)

    def test_resonant_pi_pulse_inversion(self):
        """Resonant pi pulse inverts population for S=1/2."""
        Sz = np.diag([0.5, -0.5]).astype(complex)
        Sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
        mwFreq = 1000.0  # MHz
        # For a pi pulse: H1 = 1/tp (cos modulation → RWA gives H1/2 nutation)
        tp = 0.01  # µs
        B1 = 1.0 / tp
        U = propint(mwFreq * Sz, B1 * Sy, tp, mwFreq, n=512)
        # Start in |↑⟩ = [1, 0]
        psi_0 = np.array([1, 0], dtype=complex)
        psi_f = U @ psi_0
        # After pi pulse, should be mostly in |↓⟩ = [0, 1]
        pop_down = abs(psi_f[1]) ** 2
        assert pop_down > 0.9

    def test_time_interval(self):
        """[t1, t2] interval works."""
        Sz = np.diag([0.5, -0.5]).astype(complex)
        Sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
        U = propint(100 * Sz, 10 * Sx, [0.001, 0.002], 100.0)
        assert U.shape == (2, 2)
        # Should be unitary
        I = U @ U.conj().T
        np.testing.assert_allclose(I, np.eye(2), atol=1e-6)

    def test_S1_system(self):
        """Works for S=1 (3x3 matrices)."""
        Sz = np.diag([1.0, 0.0, -1.0]).astype(complex)
        Sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                      dtype=complex) / np.sqrt(2)
        U = propint(100 * Sz, 10 * Sx, 0.01, 100.0)
        assert U.shape == (3, 3)
        I = U @ U.conj().T
        np.testing.assert_allclose(I, np.eye(3), atol=1e-5)

    def test_phase_offset(self):
        """Phase offset changes the propagator."""
        Sz = np.diag([0.5, -0.5]).astype(complex)
        Sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
        U1 = propint(100 * Sz, 10 * Sx, 0.01, 100.0, phase=0)
        U2 = propint(100 * Sz, 10 * Sx, 0.01, 100.0, phase=np.pi / 2)
        assert not np.allclose(U1, U2, atol=1e-6)

    def test_negative_freq_raises(self):
        """Negative frequency raises ValueError."""
        H0 = np.eye(2, dtype=complex)
        H1 = np.zeros((2, 2), dtype=complex)
        with pytest.raises(ValueError, match="positive"):
            propint(H0, H1, 0.01, -100.0)
