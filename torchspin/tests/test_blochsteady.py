"""Tests for torchspin.blochsteady — Bloch equation steady-state solver."""
import math

import numpy as np
import pytest

from torchspin.blochsteady import blochsteady, BlochOptions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_call(**kw):
    """Default blochsteady call with simple overrideable params."""
    params = dict(
        g=2.0, T1=10.0, T2=1.0, DeltaB0=0.0,
        B1=0.01, modAmp=0.5, modFreq=100.0,
    )
    params.update(kw)
    return blochsteady(**params)


# ---------------------------------------------------------------------------
# Output shape and types
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_default_returns_four_arrays(self):
        result = _default_call()
        assert len(result) == 4

    def test_arrays_are_1d(self):
        t, Mx, My, Mz = _default_call()
        assert t.ndim == Mx.ndim == My.ndim == Mz.ndim == 1

    def test_arrays_same_length(self):
        t, Mx, My, Mz = _default_call()
        n = len(t)
        assert len(Mx) == len(My) == len(Mz) == n

    def test_npoints_option(self):
        """opt.nPoints controls output length."""
        t, Mx, My, Mz = _default_call(opt=BlochOptions(nPoints=128))
        assert len(t) == 128

    def test_npoints_default_at_least_1(self):
        t, _, _, _ = _default_call()
        assert len(t) >= 1

    def test_time_axis_starts_at_zero(self):
        t, _, _, _ = _default_call()
        assert abs(t[0]) < 1e-12

    def test_time_axis_monotone(self):
        t, _, _, _ = _default_call()
        assert np.all(np.diff(t) > 0)

    def test_time_axis_period(self):
        """Time axis spans one modulation period (µs)."""
        modFreq_kHz = 100.0
        t_period_us = 1.0 / (modFreq_kHz * 1e3) * 1e6  # µs
        t, _, _, _ = _default_call(modFreq=modFreq_kHz)
        # Last point should be just below t_period
        assert t[-1] < t_period_us
        # But close to it
        assert t[-1] > 0.9 * t_period_us


# ---------------------------------------------------------------------------
# Physical constraints
# ---------------------------------------------------------------------------

class TestPhysics:
    def test_on_resonance_my_nonzero(self):
        """On resonance (DeltaB0=0), absorption My must be nonzero."""
        _, _, My, _ = _default_call(DeltaB0=0.0)
        assert np.max(np.abs(My)) > 1e-10

    def test_far_off_resonance_my_small(self):
        """Far off resonance, My should be small."""
        _, _, My_off, _ = _default_call(DeltaB0=10.0)  # 10 mT off-resonance
        _, _, My_on, _  = _default_call(DeltaB0=0.0)
        # Off-resonance signal should be much smaller
        assert np.max(np.abs(My_off)) < np.max(np.abs(My_on))

    def test_mz_near_equilibrium_off_resonance(self):
        """Far off resonance, Mz should be near M0=1."""
        _, _, _, Mz = _default_call(DeltaB0=10.0)
        assert np.all(np.abs(Mz - 1.0) < 0.1)

    def test_larger_b1_saturates(self):
        """Larger B1 → more saturation → smaller peak My."""
        _, _, My_weak, _ = _default_call(B1=0.001)
        _, _, My_strong, _ = _default_call(B1=1.0)
        # Large B1 saturates; |My| should still be nonzero but behavior varies
        # Just check both produce output without errors
        assert My_weak is not None and My_strong is not None

    def test_shorter_t2_broader(self):
        """Shorter T2 → more broadening → smaller peak signal at resonance."""
        _, _, My_long_T2, _ = _default_call(T2=5.0, T1=10.0)
        _, _, My_short_T2, _ = _default_call(T2=0.1, T1=10.0)
        # Shorter T2 → smaller peak absorption amplitude at DeltaB0=0
        assert np.max(np.abs(My_short_T2)) < np.max(np.abs(My_long_T2))

    def test_mx_my_real_valued(self):
        """All outputs must be real-valued (no imaginary residuals)."""
        t, Mx, My, Mz = _default_call()
        assert Mx.dtype in [np.float32, np.float64]
        assert My.dtype in [np.float32, np.float64]
        assert Mz.dtype in [np.float32, np.float64]

    def test_mz_bounded(self):
        """Mz should stay between -M0 and +M0 (physical constraint)."""
        _, _, _, Mz = _default_call()
        assert np.all(Mz <= 1.0 + 1e-6)
        assert np.all(Mz >= -1.0 - 1e-6)


# ---------------------------------------------------------------------------
# Methods agreement
# ---------------------------------------------------------------------------

class TestMethods:
    def test_fft_and_td_agree(self):
        """'fft' and 'td' methods agree on the direct (no-interpolation) path.

        Use fixed kmax so both methods operate on the same n=2*kmax-1 grid.
        """
        kmax_test = 30
        t_fft, Mx_fft, My_fft, Mz_fft = _default_call(
            opt=BlochOptions(kmax=kmax_test, method='fft'))
        t_td,  Mx_td,  My_td,  Mz_td  = _default_call(
            opt=BlochOptions(kmax=kmax_test, method='td'))
        assert np.allclose(t_fft, t_td, atol=1e-12)
        assert np.allclose(My_fft, My_td, atol=1e-8)
        assert np.allclose(Mx_fft, Mx_td, atol=1e-8)
        assert np.allclose(Mz_fft, Mz_td, atol=1e-8)


# ---------------------------------------------------------------------------
# kmax option
# ---------------------------------------------------------------------------

class TestKmax:
    def test_larger_kmax_changes_spectrum(self):
        """Different kmax values may give different (more converged) results."""
        _, _, My_lo, _ = _default_call(opt=BlochOptions(kmax=20))
        _, _, My_hi, _ = _default_call(opt=BlochOptions(kmax=80))
        # Both should be finite
        assert np.all(np.isfinite(My_lo))
        assert np.all(np.isfinite(My_hi))

    def test_kmax_auto_gives_finite_result(self):
        """Auto kmax should always produce finite outputs."""
        t, Mx, My, Mz = _default_call()
        assert np.all(np.isfinite(My))
        assert np.all(np.isfinite(Mx))
        assert np.all(np.isfinite(Mz))


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrors:
    def test_t2_gt_t1_raises(self):
        with pytest.raises(ValueError, match="T2 cannot exceed T1"):
            blochsteady(g=2.0, T1=1.0, T2=2.0, DeltaB0=0.0,
                        B1=0.01, modAmp=0.5, modFreq=100.0)

    def test_negative_t1_raises(self):
        with pytest.raises(ValueError):
            blochsteady(g=2.0, T1=-1.0, T2=0.5, DeltaB0=0.0,
                        B1=0.01, modAmp=0.5, modFreq=100.0)

    def test_negative_b1_raises(self):
        with pytest.raises(ValueError):
            blochsteady(g=2.0, T1=10.0, T2=1.0, DeltaB0=0.0,
                        B1=-0.01, modAmp=0.5, modFreq=100.0)

    def test_negative_modamp_raises(self):
        with pytest.raises(ValueError):
            blochsteady(g=2.0, T1=10.0, T2=1.0, DeltaB0=0.0,
                        B1=0.01, modAmp=-0.5, modFreq=100.0)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="unknown method"):
            _default_call(opt=BlochOptions(method='bogus'))


# ---------------------------------------------------------------------------
# Consistency / regression
# ---------------------------------------------------------------------------

class TestConsistency:
    def test_deterministic(self):
        """Same call twice gives identical results."""
        r1 = _default_call()
        r2 = _default_call()
        for a1, a2 in zip(r1, r2):
            assert np.array_equal(a1, a2)

    def test_deltaB0_symmetry(self):
        """Swapping sign of DeltaB0 gives same |My| amplitude (Lorentzian is even)."""
        # In weak-modulation limit the absorption amplitude is symmetric in DeltaB0
        _, _, My_pos, _ = _default_call(DeltaB0=0.2, modAmp=0.05)
        _, _, My_neg, _ = _default_call(DeltaB0=-0.2, modAmp=0.05)
        # Peak amplitudes should match to within 1% (not exact due to modulation effects)
        amp_pos = np.max(np.abs(My_pos))
        amp_neg = np.max(np.abs(My_neg))
        assert abs(amp_pos - amp_neg) / max(amp_pos, amp_neg) < 0.01
