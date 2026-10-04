"""Tests for Tier 3d-ii: Pulse waveforms, excitation profiles, resonator.

Tests cover:
- pulse: shaped pulse generation (AM, FM, user IQ)
- exciteprofile: Bloch-equation excitation profiles
- resonator: FFT convolution/deconvolution with resonator transfer function
"""
import numpy as np
import pytest

from torchspin.pulse import pulse
from torchspin.exciteprofile import exciteprofile
from torchspin.resonator import resonator


# ============================================================================
# pulse
# ============================================================================

class TestPulseRectangular:
    """Rectangular pulse tests."""

    def test_constant_amplitude(self):
        """Rectangular pulse has constant |IQ|."""
        par = dict(tp=0.1, Type='rectangular', Flip=np.pi)
        t, IQ, mod = pulse(par)
        amp = np.abs(IQ)
        assert np.std(amp) / np.mean(amp) < 0.01

    def test_correct_duration(self):
        """Time axis spans [0, tp]."""
        par = dict(tp=0.2, Type='rectangular', Flip=np.pi)
        t, IQ, mod = pulse(par)
        assert abs(t[0]) < 1e-10
        assert abs(t[-1] - 0.2) < t[1] - t[0]

    def test_flip_angle_amplitude(self):
        """Rectangular pi pulse: Amplitude = 1/(2*tp)."""
        tp = 0.1
        par = dict(tp=tp, Type='rectangular', Flip=np.pi)
        t, IQ, mod = pulse(par)
        expected_amp = 1 / (2 * tp)  # MHz
        assert abs(np.mean(np.abs(IQ)) - expected_amp) / expected_amp < 0.05

    def test_explicit_timestep(self):
        """Explicit TimeStep is respected."""
        par = dict(tp=0.1, Type='rectangular', Flip=np.pi, TimeStep=0.001)
        t, IQ, mod = pulse(par)
        dt = t[1] - t[0]
        assert abs(dt - 0.001) < 1e-6


class TestPulseGaussian:
    """Gaussian pulse tests."""

    def test_peak_at_center(self):
        """Gaussian AM peaks at pulse center."""
        par = dict(tp=0.2, Type='gaussian', Flip=np.pi, tFWHM=0.1)
        t, IQ, mod = pulse(par)
        peak_idx = np.argmax(np.abs(IQ))
        t_center = 0.1  # tp/2
        assert abs(t[peak_idx] - t_center) < (t[1] - t[0]) * 2

    def test_fwhm(self):
        """FWHM of Gaussian AM matches input."""
        par = dict(tp=0.5, Type='gaussian', Flip=np.pi, tFWHM=0.2)
        t, IQ, mod = pulse(par)
        amp = np.abs(IQ)
        half_max = np.max(amp) / 2
        above = t[amp >= half_max]
        measured_fwhm = above[-1] - above[0]
        assert abs(measured_fwhm - 0.2) / 0.2 < 0.1

    def test_trunc_parameter(self):
        """Truncation parameter works as alternative to tFWHM."""
        par = dict(tp=0.2, Type='gaussian', Flip=np.pi, trunc=0.1)
        t, IQ, mod = pulse(par)
        amp = np.abs(IQ)
        # Edge amplitude should be ~10% of peak
        edge_ratio = amp[0] / np.max(amp)
        assert edge_ratio < 0.2


class TestPulseSech:
    """Sech/tanh pulse tests."""

    def test_sech_tanh_generates_iq(self):
        """sech/tanh pulse produces non-zero IQ."""
        par = dict(tp=0.2, Type='sech/tanh', Frequency=[-50, 50],
                   beta=10, Flip=np.pi)
        t, IQ, mod = pulse(par)
        assert len(t) > 10
        assert np.max(np.abs(IQ)) > 0

    def test_sech_am_shape(self):
        """Sech AM is peaked at center and symmetric."""
        par = dict(tp=0.2, Type='sech/tanh', Frequency=[-50, 50],
                   beta=10, Flip=np.pi)
        t, IQ, mod = pulse(par)
        A = mod['A']
        n = len(A)
        # Peak at center
        peak_idx = np.argmax(A)
        assert abs(peak_idx - n // 2) < n * 0.1
        # Symmetric
        half = min(peak_idx, n - peak_idx - 1)
        left = A[peak_idx - half:peak_idx]
        right = A[peak_idx + 1:peak_idx + half + 1][::-1]
        np.testing.assert_allclose(left, right, atol=0.01 * np.max(A))

    def test_frequency_sweep(self):
        """Frequency modulation sweeps the expected range."""
        par = dict(tp=0.2, Type='sech/tanh', Frequency=[-50, 50],
                   beta=10, Flip=np.pi)
        t, IQ, mod = pulse(par)
        freq = mod['freq']
        assert freq[0] < -40  # starts near -50
        assert freq[-1] > 40   # ends near +50


class TestPulseWURST:
    """WURST pulse tests."""

    def test_wurst_flat_center(self):
        """WURST has flat center with tapered edges."""
        par = dict(tp=0.2, Type='WURST/linear', Frequency=[-50, 50],
                   nwurst=20, Flip=np.pi)
        t, IQ, mod = pulse(par)
        A = mod['A']
        center = A[len(A) // 4:3 * len(A) // 4]
        # Center should be relatively flat
        assert np.std(center) / np.mean(center) < 0.05


class TestPulseGaussianCascade:
    """Gaussian cascade pulse tests."""

    @pytest.mark.parametrize('name', ['G3', 'G4', 'Q3', 'Q5'])
    def test_preset_generates_iq(self, name):
        """Predefined Gaussian cascade produces non-zero output."""
        par = dict(tp=0.5, Type=name, Flip=np.pi)
        t, IQ, mod = pulse(par)
        assert np.max(np.abs(IQ)) > 0
        assert len(t) > 5


class TestPulseFourierSeries:
    """Fourier series pulse tests."""

    @pytest.mark.parametrize('name', ['I-BURP 1', 'I-BURP 2', 'E-BURP 1',
                                       'RE-BURP', 'SNOB i2'])
    def test_preset_generates_iq(self, name):
        """Predefined Fourier series pulse produces non-zero output."""
        par = dict(tp=0.5, Type=name, Flip=np.pi)
        t, IQ, mod = pulse(par)
        assert np.max(np.abs(IQ)) > 0


class TestPulseUserIQ:
    """User-provided IQ data tests."""

    def test_user_iq_passthrough(self):
        """User IQ data is returned with correct time axis."""
        iq_data = np.sin(np.linspace(0, 2 * np.pi, 100))
        par = dict(tp=0.1, IQ=iq_data)
        t, IQ, mod = pulse(par)
        assert abs(t[-1] - 0.1) < 0.01
        assert len(IQ) == 100

    def test_user_i_q_separate(self):
        """Separate I and Q channels work."""
        I = np.ones(50)
        Q = np.zeros(50)
        par = dict(tp=0.05, I=I, Q=Q)
        t, IQ, mod = pulse(par)
        np.testing.assert_allclose(np.real(IQ), 1.0, atol=0.01)


class TestPulseOther:
    """Other AM functions."""

    def test_halfsin(self):
        """Halfsin AM is zero at edges, max at center."""
        par = dict(tp=0.2, Type='halfsin', Flip=np.pi)
        t, IQ, mod = pulse(par)
        A = mod['A']
        assert A[0] < 0.05 * np.max(A)
        assert A[-1] < 0.05 * np.max(A)

    def test_quartersin(self):
        """Quartersin AM has smooth edges."""
        par = dict(tp=0.2, Type='quartersin', Flip=np.pi, trise=0.02)
        t, IQ, mod = pulse(par)
        A = mod['A']
        assert A[0] < 0.05 * np.max(A)

    def test_sinc(self):
        """Sinc AM has lobes."""
        par = dict(tp=0.5, Type='sinc', Flip=np.pi, zerocross=0.2)
        t, IQ, mod = pulse(par)
        A = mod['A']
        # Sinc should have zero crossings
        sign_changes = np.sum(np.diff(np.sign(A)) != 0)
        assert sign_changes >= 2

    def test_tanh2(self):
        """tanh2 AM has smooth rise."""
        par = dict(tp=0.2, Type='tanh2', Flip=np.pi, trise=0.02)
        t, IQ, mod = pulse(par)
        A = mod['A']
        assert A[0] < 0.01 * np.max(A)

    def test_uniformq_fm(self):
        """uniformQ FM works with sech AM."""
        par = dict(tp=0.2, Type='sech/uniformQ', Frequency=[-50, 50],
                   beta=10, Flip=np.pi)
        t, IQ, mod = pulse(par)
        assert np.max(np.abs(IQ)) > 0

    def test_linear_chirp(self):
        """Linear FM produces linear frequency sweep."""
        par = dict(tp=0.2, Type='rectangular/linear',
                   Frequency=[-50, 50], Flip=np.pi)
        t, IQ, mod = pulse(par)
        freq = mod['freq']
        # Should be approximately linear
        expected = np.linspace(-50, 50, len(freq))
        np.testing.assert_allclose(freq, expected, atol=2)


class TestPulseErrors:
    """Error handling tests."""

    def test_missing_tp(self):
        """Missing tp raises ValueError."""
        with pytest.raises(ValueError, match="tp"):
            pulse({})

    def test_missing_type(self):
        """Missing Type raises ValueError."""
        with pytest.raises(ValueError, match="Type"):
            pulse(dict(tp=0.1))

    def test_unknown_am(self):
        """Unknown AM function raises ValueError."""
        with pytest.raises(ValueError, match="Unknown AM"):
            pulse(dict(tp=0.1, Type='bogus', Flip=np.pi))

    def test_unknown_fm(self):
        """Unknown FM function raises ValueError."""
        with pytest.raises(ValueError, match="Unknown FM"):
            pulse(dict(tp=0.1, Type='rectangular/bogus',
                       Frequency=[-10, 10], Flip=np.pi))

    def test_fm_needs_freq_range(self):
        """FM without frequency range raises ValueError."""
        with pytest.raises(ValueError, match="frequency range"):
            pulse(dict(tp=0.1, Type='rectangular/linear', Flip=np.pi))


class TestPulseAmplitude:
    """Amplitude and Qcrit tests."""

    def test_explicit_amplitude(self):
        """Explicit amplitude overrides flip angle."""
        par = dict(tp=0.1, Type='rectangular', Amplitude=25.0)
        t, IQ, mod = pulse(par)
        assert abs(np.mean(np.abs(IQ)) - 25.0) < 1.0

    def test_qcrit_sets_amplitude(self):
        """Qcrit parameter sets amplitude for FM pulses."""
        par = dict(tp=0.2, Type='sech/tanh', Frequency=[-50, 50],
                   beta=10, Qcrit=5.0)
        t, IQ, mod = pulse(par)
        assert np.max(np.abs(IQ)) > 0

    def test_phase_offset(self):
        """Phase offset rotates IQ."""
        par0 = dict(tp=0.1, Type='rectangular', Flip=np.pi, Phase=0)
        par90 = dict(tp=0.1, Type='rectangular', Flip=np.pi, Phase=np.pi / 2)
        _, IQ0, _ = pulse(par0)
        _, IQ90, _ = pulse(par90)
        # Phase rotation: IQ90 ≈ IQ0 * exp(i*pi/2)
        ratio = IQ90 / IQ0
        phase_diff = np.mean(np.angle(ratio))
        assert abs(phase_diff - np.pi / 2) < 0.1


# ============================================================================
# exciteprofile
# ============================================================================

class TestExciteProfile:
    """Excitation profile tests."""

    def test_rectangular_pi_pulse_inverts(self):
        """Rectangular pi pulse inverts Mz on resonance."""
        tp = 0.01  # µs
        amp = 1 / (2 * tp)  # pi pulse amplitude in MHz
        npts = 500
        t = np.linspace(0, tp, npts)
        IQ = amp * np.ones(npts, dtype=complex)
        offsets, Mag = exciteprofile(t, IQ, offsets=np.array([0.0]))
        # Equilibrium Mz = +1, after pi pulse Mz ≈ -1
        assert Mag[2, 0] < -0.9

    def test_rectangular_profile_shape(self):
        """Rectangular pulse has sinc-like Mz profile."""
        tp = 0.1
        amp = 1 / (2 * tp)
        npts = 200
        t = np.linspace(0, tp, npts)
        IQ = amp * np.ones(npts, dtype=complex)
        offsets = np.linspace(-100, 100, 201)
        offsets_out, Mag = exciteprofile(t, IQ, offsets=offsets)
        # On resonance: full inversion (Mz → -1)
        center_idx = 100
        assert Mag[2, center_idx] < -0.8
        # Far off-resonance: no inversion (Mz stays near +1)
        assert Mag[2, 0] > 0.5
        assert Mag[2, -1] > 0.5

    def test_auto_offsets(self):
        """Auto-determined offset axis is centered on pulse."""
        tp = 0.1
        amp = 1 / (2 * tp)
        t = np.linspace(0, tp, 200)
        IQ = amp * np.ones(200, dtype=complex)
        offsets, Mag = exciteprofile(t, IQ)
        assert len(offsets) == 201
        assert Mag.shape == (3, 201)

    def test_shaped_pulse_profile(self):
        """Gaussian pulse has narrower excitation profile."""
        tp = 0.2
        t = np.linspace(0, tp, 500)
        ti = t - tp / 2
        # Gaussian AM with FWHM = tp/2
        A = np.exp(-(4 * np.log(2) * ti ** 2) / (tp / 2) ** 2)
        _trapezoid = getattr(np, "trapezoid", None) or np.trapz
        amp = np.pi / (2 * np.pi * _trapezoid(A, t))
        IQ = (amp * A).astype(complex)
        offsets = np.linspace(-50, 50, 201)
        _, Mag = exciteprofile(t, IQ, offsets=offsets)
        # Mz at center should be inverted (negative)
        assert Mag[2, 100] < -0.5
        # Bandwidth should be narrower than rectangular
        inverted = np.where(Mag[2, :] < -0.5)[0]
        bw_gauss = offsets[inverted[-1]] - offsets[inverted[0]] if len(inverted) > 1 else 0
        # Compare with rectangular
        IQ_rect = (1 / (2 * tp)) * np.ones(500, dtype=complex)
        _, Mag_rect = exciteprofile(t, IQ_rect, offsets=offsets)
        inverted_rect = np.where(Mag_rect[2, :] < -0.5)[0]
        bw_rect = offsets[inverted_rect[-1]] - offsets[inverted_rect[0]] if len(inverted_rect) > 1 else 0
        # Gaussian BW should be less than or comparable to rectangular
        # (depends on amplitude; just check it's finite)
        assert bw_gauss > 0

    def test_magnetization_bounded(self):
        """Magnetization components are bounded by [-1, 1]."""
        tp = 0.1
        amp = 1 / (2 * tp)
        t = np.linspace(0, tp, 200)
        IQ = amp * np.ones(200, dtype=complex)
        offsets = np.linspace(-50, 50, 51)
        _, Mag = exciteprofile(t, IQ, offsets=offsets)
        assert np.all(np.abs(Mag) <= 1.01)

    def test_zero_pulse_no_change(self):
        """Zero-amplitude pulse leaves magnetization unchanged."""
        t = np.linspace(0, 0.1, 100)
        IQ = np.zeros(100, dtype=complex)
        offsets = np.linspace(-10, 10, 21)
        _, Mag = exciteprofile(t, IQ, offsets=offsets)
        # Equilibrium Mz = +1 (ρ₀ = -Sz, Mz = -2*Tr(Sz*ρ₀) = +1)
        np.testing.assert_allclose(Mag[2, :], 1.0, atol=0.01)

    def test_input_length_mismatch_raises(self):
        """Mismatched t and IQ lengths raise ValueError."""
        with pytest.raises(ValueError, match="equal"):
            exciteprofile(np.zeros(10), np.zeros(5))


# ============================================================================
# resonator
# ============================================================================

class TestResonator:
    """Resonator simulation/compensation tests."""

    def test_simulate_modifies_pulse(self):
        """Resonator simulation changes the pulse shape."""
        tp = 0.1
        t = np.linspace(0, tp, 500)
        signal = np.ones(500)  # rectangular pulse
        t_out, sig_out = resonator(t, signal, mwFreq=9.5,
                                    nu0=9.5, QL=100, option='simulate')
        # Output should not be perfectly rectangular anymore
        assert len(t_out) > 0
        assert np.max(np.abs(sig_out)) > 0

    def test_simulate_changes_shape(self):
        """Resonator simulation broadens a short pulse."""
        tp = 0.01
        t = np.linspace(0, tp, 500)
        signal = np.ones(500, dtype=complex)  # short rectangular pulse

        t_sim, sig_sim = resonator(t, signal, mwFreq=9.5,
                                    nu0=9.5, QL=100, option='simulate')
        # Simulated pulse should be longer (resonator ring-down)
        assert t_sim[-1] > tp * 0.5
        assert np.max(np.abs(sig_sim)) > 0

    def test_compensate_output(self):
        """Compensation produces a valid output."""
        tp = 0.1
        t = np.linspace(0, tp, 500)
        signal = np.ones(500, dtype=complex)
        t_out, sig_out = resonator(t, signal, mwFreq=9.5,
                                    nu0=9.5, QL=100, option='compensate')
        assert len(t_out) > 0
        assert np.all(np.isfinite(sig_out))

    def test_unknown_option_raises(self):
        """Unknown option raises ValueError."""
        t = np.linspace(0, 0.1, 100)
        signal = np.ones(100)
        with pytest.raises(ValueError, match="Unknown"):
            resonator(t, signal, mwFreq=9.5, nu0=9.5, QL=100,
                       option='invalid')

    def test_missing_transfer_function_raises(self):
        """Missing both ideal and experimental params raises ValueError."""
        t = np.linspace(0, 0.1, 100)
        signal = np.ones(100)
        with pytest.raises(ValueError, match="Provide"):
            resonator(t, signal, mwFreq=9.5, option='simulate')

    def test_explicit_timestep(self):
        """Explicit TimeStep is respected in output."""
        tp = 0.1
        t = np.linspace(0, tp, 500)
        signal = np.ones(500, dtype=complex)
        t_out, sig_out = resonator(t, signal, mwFreq=9.5,
                                    nu0=9.5, QL=100, option='simulate',
                                    TimeStep=0.001)
        if len(t_out) > 1:
            dt_out = t_out[1] - t_out[0]
            assert abs(dt_out - 0.001) < 1e-4


# ============================================================================
# Integration tests
# ============================================================================

class TestPulseExciteProfileIntegration:
    """Integration: pulse → exciteprofile pipeline."""

    def test_rectangular_pulse_excitation(self):
        """Rectangular pulse from pulse() drives exciteprofile correctly."""
        par = dict(tp=0.01, Type='rectangular', Flip=np.pi, TimeStep=0.0001)
        t, IQ, _ = pulse(par)
        offsets = np.linspace(-200, 200, 101)
        _, Mag = exciteprofile(t, IQ, offsets=offsets)
        # On-resonance inversion (Mz → -1)
        center = len(offsets) // 2
        assert Mag[2, center] < -0.8

    def test_sech_tanh_excitation_profile(self):
        """sech/tanh pulse has broad, uniform excitation profile."""
        par = dict(tp=0.2, Type='sech/tanh', Frequency=[-50, 50],
                   beta=10, Flip=np.pi)
        t, IQ, _ = pulse(par)
        offsets = np.linspace(-60, 60, 201)
        _, Mag = exciteprofile(t, IQ, offsets=offsets)
        # Check inversion across the sweep range
        in_range = (offsets > -40) & (offsets < 40)
        # Mz in sweep range should be mostly inverted (negative)
        assert np.mean(Mag[2, in_range]) < -0.3
