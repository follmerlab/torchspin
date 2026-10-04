"""Tests for Tier 2 modules: dipkernel, dipbackground, exponfit, nucfrq2d, evolve."""
from __future__ import annotations

import math
import numpy as np
import pytest


# ============================================================================
# dipkernel
# ============================================================================

class TestDipkernel:
    """Tests for torchspin.dipkernel.dipkernel."""

    def test_shape(self):
        """Kernel should have shape (len(t), len(r))."""
        from torchspin.dipkernel import dipkernel
        t = np.linspace(0, 3, 201)
        r = np.linspace(1.5, 6, 100)
        K = dipkernel(t, r)
        assert K.shape == (201, 100)

    def test_t_zero_unity(self):
        """At t=0, kernel should be 1 (before dr scaling)."""
        from torchspin.dipkernel import dipkernel
        t = np.array([0.0, 0.5, 1.0])
        r = np.array([3.0])  # single distance, no dr scaling
        K = dipkernel(t, r)
        assert abs(K[0, 0] - 1.0) < 1e-10

    def test_decay_with_time(self):
        """Kernel should decay with increasing |t|."""
        from torchspin.dipkernel import dipkernel
        t = np.linspace(0, 5, 501)
        r = np.linspace(2, 5, 50)
        K = dipkernel(t, r)
        # Column sums should decrease from t=0
        col_sums = np.sum(np.abs(K), axis=1)
        assert col_sums[0] >= col_sums[-1]

    def test_closer_distance_faster_decay(self):
        """Shorter distances should cause faster oscillation/decay."""
        from torchspin.dipkernel import dipkernel
        t = np.linspace(0, 2, 201)
        r_close = np.array([2.0])
        r_far = np.array([5.0])
        K_close = dipkernel(t, r_close)
        K_far = dipkernel(t, r_far)
        # Close distance: more oscillation → lower last value
        assert np.abs(K_close[-1, 0]) <= np.abs(K_far[-1, 0]) + 0.1

    def test_negative_time_symmetry(self):
        """K(-t, r) should equal K(t, r) (uses |t|)."""
        from torchspin.dipkernel import dipkernel
        t = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        r = np.array([3.0])
        K = dipkernel(t, r)
        assert abs(K[0, 0] - K[4, 0]) < 1e-14
        assert abs(K[1, 0] - K[3, 0]) < 1e-14

    def test_nonuniform_r_raises(self):
        """Non-equally-spaced r should raise ValueError."""
        from torchspin.dipkernel import dipkernel
        t = np.array([0.0, 1.0])
        r = np.array([1.0, 2.0, 4.0])  # not equally spaced
        with pytest.raises(ValueError):
            dipkernel(t, r)

    def test_custom_g_factors(self):
        """Custom g-factors should change the coupling strength."""
        from torchspin.dipkernel import dipkernel
        t = np.linspace(0, 2, 101)
        r = np.linspace(2, 5, 50)
        K_default = dipkernel(t, r)
        K_custom = dipkernel(t, r, g=[2.1, 2.1])
        # Different g → different kernel
        assert not np.allclose(K_default, K_custom)


# ============================================================================
# dipbackground
# ============================================================================

class TestDipbackground:
    """Tests for torchspin.dipbackground.dipbackground."""

    def test_t_zero_unity(self):
        """V(0) should be 1."""
        from torchspin.dipbackground import dipbackground
        t = np.array([0.0, 1.0, 2.0])
        V = dipbackground(t, conc=100.0, lam=0.3)
        assert abs(V[0] - 1.0) < 1e-14

    def test_monotonic_decay(self):
        """Background should decay monotonically for t > 0."""
        from torchspin.dipbackground import dipbackground
        t = np.linspace(0, 10, 100)
        V = dipbackground(t, conc=100.0, lam=0.5)
        assert np.all(np.diff(V) <= 0)

    def test_higher_conc_faster_decay(self):
        """Higher concentration should cause faster decay."""
        from torchspin.dipbackground import dipbackground
        t = np.linspace(0, 5, 100)
        V_low = dipbackground(t, conc=10.0, lam=0.3)
        V_high = dipbackground(t, conc=100.0, lam=0.3)
        assert V_high[-1] < V_low[-1]

    def test_higher_lambda_faster_decay(self):
        """Higher modulation depth should cause faster decay."""
        from torchspin.dipbackground import dipbackground
        t = np.linspace(0, 5, 100)
        V_low = dipbackground(t, conc=50.0, lam=0.1)
        V_high = dipbackground(t, conc=50.0, lam=0.5)
        assert V_high[-1] < V_low[-1]

    def test_zero_lambda_no_decay(self):
        """lambda=0 should give V=1 everywhere."""
        from torchspin.dipbackground import dipbackground
        t = np.linspace(0, 10, 100)
        V = dipbackground(t, conc=100.0, lam=0.0)
        np.testing.assert_allclose(V, 1.0, atol=1e-14)

    def test_negative_time_symmetry(self):
        """V(-t) = V(t) (uses |t|)."""
        from torchspin.dipbackground import dipbackground
        t = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        V = dipbackground(t, conc=50.0, lam=0.3)
        assert abs(V[0] - V[4]) < 1e-14
        assert abs(V[1] - V[3]) < 1e-14


# ============================================================================
# exponfit
# ============================================================================

class TestExponfit:
    """Tests for torchspin.exponfit.exponfit."""

    def test_mono_decay(self):
        """Recover known mono-exponential decay rate."""
        from torchspin.exponfit import exponfit
        T = 2.5  # true time constant
        t = np.linspace(0, 15, 300)
        y = 0.8 * np.exp(-t / T) + 0.1
        k, c, yfit = exponfit(t, y, n_exp=1)
        # k should be -1/T
        recovered_T = -1.0 / k[0]
        assert abs(recovered_T - T) / T < 0.01  # 1% tolerance

    def test_mono_recovery(self):
        """Recover mono-exponential recovery (increasing signal)."""
        from torchspin.exponfit import exponfit
        T = 3.0
        t = np.linspace(0, 20, 300)
        y = 1.0 - 0.9 * np.exp(-t / T)
        k, c, yfit = exponfit(t, y, n_exp=1)
        recovered_T = -1.0 / k[0]
        assert abs(recovered_T - T) / T < 0.02

    def test_no_offset(self):
        """Fit without offset."""
        from torchspin.exponfit import exponfit
        T = 1.5
        t = np.linspace(0, 10, 200)
        y = 2.0 * np.exp(-t / T)
        k, c, yfit = exponfit(t, y, n_exp=1, include_offset=False)
        recovered_T = -1.0 / k[0]
        assert abs(recovered_T - T) / T < 0.01

    def test_biexponential(self):
        """Recover two decay rates."""
        from torchspin.exponfit import exponfit
        T1, T2 = 1.0, 5.0
        t = np.linspace(0, 20, 400)
        y = 0.6 * np.exp(-t / T1) + 0.3 * np.exp(-t / T2) + 0.1
        k, c, yfit = exponfit(t, y, n_exp=2)
        # Sort recovered rates by magnitude
        T_recovered = np.sort(-1.0 / k)
        T_expected = np.sort([T1, T2])
        np.testing.assert_allclose(T_recovered, T_expected, rtol=0.05)

    def test_fit_quality(self):
        """Fitted curve should closely match input."""
        from torchspin.exponfit import exponfit
        t = np.linspace(0, 10, 200)
        y = 0.8 * np.exp(-t / 2.0) + 0.2
        k, c, yfit = exponfit(t, y)
        residual = np.sqrt(np.mean((y - yfit) ** 2))
        assert residual < 1e-6

    def test_multiple_columns(self):
        """Fit multiple series simultaneously."""
        from torchspin.exponfit import exponfit
        t = np.linspace(0, 10, 200)
        y1 = np.exp(-t / 1.0)
        y2 = np.exp(-t / 3.0)
        y = np.column_stack([y1, y2])
        k, c, yfit = exponfit(t, y, n_exp=1, include_offset=False)
        assert k.shape == (1, 2)
        T_recovered = -1.0 / k[0, :]
        np.testing.assert_allclose(T_recovered, [1.0, 3.0], rtol=0.02)


# ============================================================================
# nucfrq2d
# ============================================================================

class TestNucfrq2d:
    """Tests for torchspin.nucfrq2d.nucfrq2d."""

    def test_basic_output(self):
        """Should return dict with expected keys."""
        from torchspin import SpinSystem
        from torchspin.nucfrq2d import nucfrq2d
        sys = SpinSystem(
            S=[0.5], g=[[2, 2, 2]],
            Nucs=['1H'], A=[[3, 3, 8]],
        )
        data = nucfrq2d(sys, 350.0)
        assert 'FreqsAlpha' in data
        assert 'FreqsBeta' in data
        assert 'LarmorFreqs' in data
        assert 'maxFreq' in data

    def test_frequency_dimensions(self):
        """FreqsAlpha and FreqsBeta should have shape (nOri, nfreq)."""
        from torchspin import SpinSystem
        from torchspin.nucfrq2d import nucfrq2d
        sys = SpinSystem(
            S=[0.5], g=[[2, 2, 2]],
            Nucs=['1H'], A=[[3, 3, 8]],
        )
        data = nucfrq2d(sys, 350.0, GridSize=11)
        # S=1/2 + I=1/2 → 4 states, 2 per manifold → 1 transition each
        assert data['FreqsAlpha'].shape[1] == 1
        assert data['FreqsBeta'].shape[1] == 1

    def test_larmor_frequency(self):
        """Larmor frequency of 1H at 350 mT should be ~14.9 MHz."""
        from torchspin import SpinSystem
        from torchspin.nucfrq2d import nucfrq2d
        sys = SpinSystem(
            S=[0.5], g=[[2, 2, 2]],
            Nucs=['1H'], A=[[3, 3, 8]],
        )
        data = nucfrq2d(sys, 350.0)
        # 1H Larmor at 350 mT ≈ 14.9 MHz
        assert abs(data['LarmorFreqs'][0] - 14.9) < 0.5

    def test_blind_spot_modulation(self):
        """Blind spot pattern should be computed when tau given."""
        from torchspin import SpinSystem
        from torchspin.nucfrq2d import nucfrq2d
        sys = SpinSystem(
            S=[0.5], g=[[2, 2, 2]],
            Nucs=['1H'], A=[[3, 3, 8]],
        )
        data = nucfrq2d(sys, 350.0, tau=0.120)
        assert data['Modulation'] is not None
        assert data['Modulation'].shape[0] == data['Modulation'].shape[1]

    def test_s_gt_half_raises(self):
        """S > 1/2 should raise ValueError."""
        from torchspin import SpinSystem
        from torchspin.nucfrq2d import nucfrq2d
        sys = SpinSystem(
            S=[1], g=[[2, 2, 2]],
            Nucs=['1H'], A=[[3, 3, 8]],
            D=[[-100, -100, 200]],
        )
        with pytest.raises(ValueError, match="S=1/2"):
            nucfrq2d(sys, 350.0)

    def test_positive_frequencies(self):
        """Maximum frequency should be positive."""
        from torchspin import SpinSystem
        from torchspin.nucfrq2d import nucfrq2d
        sys = SpinSystem(
            S=[0.5], g=[[2, 2, 2]],
            Nucs=['1H'], A=[[3, 3, 8]],
        )
        data = nucfrq2d(sys, 350.0)
        assert data['maxFreq'] > 0


# ============================================================================
# evolve
# ============================================================================

class TestEvolve:
    """Tests for torchspin.evolve.evolve."""

    def test_scheme_1_fid(self):
        """Scheme [1]: FID oscillates at transition frequency."""
        from torchspin.evolve import evolve
        freq = 50.0  # MHz
        H = np.diag([0.0, freq])
        # Off-diagonal density → coherence at freq
        Sig = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        Det = np.array([[0, 1], [0, 0]], dtype=complex)
        dt = 0.001  # us
        n = 1000
        signal = evolve(Sig, Det, H, n, dt)
        assert signal.shape == (n,)
        # FFT should peak near freq
        freqs = np.fft.fftfreq(n, dt)
        power = np.abs(np.fft.fft(signal)) ** 2
        peak_freq = abs(freqs[np.argmax(power)])
        assert abs(peak_freq - freq) < 2.0  # within 2 MHz

    def test_scheme_1_trace_preservation(self):
        """Signal at t=0 should equal Tr(Det * Sig)."""
        from torchspin.evolve import evolve
        H = np.diag([0.0, 100.0])
        Sig = np.array([[0.6, 0.3], [0.3, 0.4]], dtype=complex)
        Det = np.array([[1, 0], [0, -1]], dtype=complex)
        signal = evolve(Sig, Det, H, 10, 0.01)
        expected = np.trace(Det @ Sig)
        assert abs(signal[0] - expected) < 1e-12

    def test_scheme_1_1(self):
        """Scheme [1, 1]: 2p-ESEEM with mixing."""
        from torchspin.evolve import evolve
        H = np.diag([0.0, 50.0, 100.0])
        N = 3
        Sig = np.eye(N, dtype=complex) / N
        Det = np.eye(N, dtype=complex)
        # Simple pi/2 mixing (identity for trivial test)
        Mix = np.eye(N, dtype=complex)
        signal = evolve(Sig, Det, H, 64, 0.01, [1, 1], [Mix])
        assert signal.shape == (64,)
        assert np.isfinite(signal).all()

    def test_scheme_1_m1(self):
        """Scheme [1, -1]: refocused echo."""
        from torchspin.evolve import evolve
        H = np.diag([0.0, 30.0])
        Sig = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        Det = np.array([[0, 1], [0, 0]], dtype=complex)
        Mix = np.array([[0, 1], [1, 0]], dtype=complex)  # pi pulse
        signal = evolve(Sig, Det, H, 64, 0.01, [1, -1], [Mix])
        assert signal.shape == (64,)
        assert np.isfinite(signal).all()

    def test_scheme_1_2_hyscore(self):
        """Scheme [1, 2]: 2D HYSCORE-like."""
        from torchspin.evolve import evolve
        H = np.diag([0.0, 30.0, 60.0])
        N = 3
        Sig = np.ones((N, N), dtype=complex) / N
        Det = np.eye(N, dtype=complex)
        Mix = np.eye(N, dtype=complex)
        signal = evolve(Sig, Det, H, [32, 32], [0.01, 0.01], [1, 2], [Mix])
        assert signal.shape == (32, 32)
        assert np.isfinite(signal).all()

    def test_diagonal_hamiltonian(self):
        """Already-diagonal Hamiltonian should work without diagonalization."""
        from torchspin.evolve import evolve
        H = np.diag([0.0, 100.0])
        Sig = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        Det = np.array([[0, 1], [0, 0]], dtype=complex)
        signal = evolve(Sig, Det, H, 100, 0.005)
        assert len(signal) == 100

    def test_unsupported_scheme_raises(self):
        """Unsupported IncScheme should raise NotImplementedError."""
        from torchspin.evolve import evolve
        H = np.diag([0.0, 100.0])
        Sig = Det = np.eye(2, dtype=complex)
        Mix = [np.eye(2, dtype=complex)] * 3
        with pytest.raises(NotImplementedError):
            evolve(Sig, Det, H, 10, 0.01, [1, -1, -1, 1], Mix)

    def test_mixing_count_mismatch_raises(self):
        """Wrong number of mixing propagators should raise."""
        from torchspin.evolve import evolve
        H = np.diag([0.0, 100.0])
        Sig = Det = np.eye(2, dtype=complex)
        with pytest.raises(ValueError):
            evolve(Sig, Det, H, 10, 0.01, [1, 1], [])  # need 1 mixer
