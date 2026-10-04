"""Tests for garlic fast-motion (tcorr > 0) path and fastmotion.py."""
import math

import numpy as np
import pytest
import torch

from torchspin.spinsystem import SpinSystem
from torchspin.experiment import Experiment, Options
from torchspin.garlic import garlic
from torchspin.fastmotion import fastmotion, _all_mI
from torchspin.constants import BMAGN, PLANCK


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nitroxide_sys():
    """di-tert-butyl nitroxide-like system: anisotropic g and A."""
    return SpinSystem(
        S=[0.5],
        g=[[2.0088, 2.0064, 2.0027]],
        Nucs='14N',
        A=[[20.0, 20.0, 85.0]],   # MHz: axial A_perp=20, A_par=85
        lw=[0.0, 0.0],
    )


def _nitroxide_exp():
    return Experiment(mwFreq=9.5, Range=[330.0, 355.0], nPoints=512, Harmonic=1)


# ---------------------------------------------------------------------------
# Tests for fastmotion.py
# ---------------------------------------------------------------------------

class TestFastmotionBasic:
    def test_returns_two_arrays(self):
        sys = _nitroxide_sys()
        lw, mI_all = fastmotion(sys, 340.0, 1e-10)
        assert isinstance(lw, np.ndarray)
        assert isinstance(mI_all, np.ndarray)

    def test_nlines_14N(self):
        """14N (I=1) → 3 EPR lines."""
        sys = _nitroxide_sys()
        lw, mI_all = fastmotion(sys, 340.0, 1e-10)
        assert lw.shape == (3,)
        assert mI_all.shape == (3, 1)

    def test_mI_ordering_14N(self):
        """mI should be ordered -1, 0, +1 (same as resfields_perturb)."""
        sys = _nitroxide_sys()
        _, mI_all = fastmotion(sys, 340.0, 1e-10)
        np.testing.assert_array_equal(mI_all[:, 0], [-1.0, 0.0, 1.0])

    def test_positive_widths(self):
        """All linewidths must be strictly positive."""
        sys = _nitroxide_sys()
        lw, _ = fastmotion(sys, 340.0, 1e-10)
        assert np.all(lw > 0), f"Non-positive linewidths: {lw}"

    def test_domain_field_vs_freq(self):
        """Field and frequency widths should be related by g*muB/h."""
        sys = _nitroxide_sys()
        g0 = np.mean([2.0088, 2.0064, 2.0027])
        lw_mT, _ = fastmotion(sys, 340.0, 1e-10, domain='field')
        lw_MHz, _ = fastmotion(sys, 340.0, 1e-10, domain='freq')
        # MHz → mT: 1e6 * h / (g0 * muB) * 1e3
        conv = 1e6 * PLANCK / (g0 * BMAGN) * 1e3
        ratio = lw_mT / (lw_MHz * conv)
        np.testing.assert_allclose(ratio, 1.0, rtol=1e-6)

    def test_invalid_domain(self):
        sys = _nitroxide_sys()
        with pytest.raises(ValueError, match="domain"):
            fastmotion(sys, 340.0, 1e-10, domain='energy')

    def test_isotropic_system_raises(self):
        """Fully isotropic g and A → ValueError (no anisotropy)."""
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        with pytest.raises(ValueError, match="anisotrop"):
            fastmotion(sys, 340.0, 1e-10)

    def test_width_increases_with_tcorr(self):
        """In the fast-motion regime, larger tcorr → broader lines."""
        sys = _nitroxide_sys()
        lw_fast, _ = fastmotion(sys, 340.0, 1e-11)
        lw_slow, _ = fastmotion(sys, 340.0, 1e-10)
        assert np.mean(lw_slow) > np.mean(lw_fast), (
            "Larger tcorr should give broader lines (at fast-motion limit)"
        )

    def test_B_coefficient_nonzero(self):
        """B coefficient (mI-linear term) must be nonzero for anisotropic A."""
        sys = _nitroxide_sys()
        lw, mI_all = fastmotion(sys, 340.0, 1e-10)
        # For mI=-1 and mI=+1, B term makes widths different
        # lw(-1) = A - B, lw(+1) = A + B  (or reversed sign)
        outer_diff = abs(lw[2] - lw[0])  # mI=+1 vs mI=-1
        assert outer_diff > 1e-6, (
            "Outer lines should have different widths (B coefficient ≠ 0)"
        )

    def test_no_nuclei_system(self):
        """g-anisotropy only (no nuclei) → one line."""
        sys = SpinSystem(S=[0.5], g=[[2.0088, 2.0064, 2.0027]])
        lw, mI_all = fastmotion(sys, 340.0, 1e-10)
        assert lw.shape == (1,)
        assert lw[0] > 0

    def test_two_nuclei(self):
        """Two nuclei → nLines = (2*I1+1) * (2*I2+1)."""
        sys = SpinSystem(
            S=[0.5],
            g=[[2.0088, 2.0064, 2.0027]],
            Nucs='14N,1H',
            A=[[20.0, 20.0, 80.0], [5.0, 5.0, 5.0]],  # (nNuclei, 3)
        )
        lw, mI_all = fastmotion(sys, 340.0, 1e-10)
        # 14N: 2*1+1=3, 1H: 2*0.5+1=2 → 6 lines
        assert lw.shape == (6,)
        assert mI_all.shape == (6, 2)


# ---------------------------------------------------------------------------
# Tests for _all_mI helper
# ---------------------------------------------------------------------------

class TestAllMI:
    def test_single_nucleus_I1(self):
        mI = _all_mI(np.array([1.0]))
        np.testing.assert_array_equal(mI[:, 0], [-1.0, 0.0, 1.0])

    def test_single_nucleus_I_half(self):
        mI = _all_mI(np.array([0.5]))
        np.testing.assert_array_equal(mI[:, 0], [-0.5, 0.5])

    def test_two_nuclei(self):
        mI = _all_mI(np.array([1.0, 0.5]))
        assert mI.shape == (6, 2)  # 3 × 2 = 6
        # First column: 14N mI
        np.testing.assert_array_equal(mI[:, 0], [-1, -1, 0, 0, 1, 1])
        # Second column: 1H mI
        np.testing.assert_array_equal(mI[:, 1], [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5])

    def test_no_nuclei(self):
        mI = _all_mI(np.array([]))
        assert mI.shape == (1, 0)


# ---------------------------------------------------------------------------
# Tests for garlic fast-motion path
# ---------------------------------------------------------------------------

class TestGarlicFastMotion:
    def test_returns_two_tensors(self):
        sys = _nitroxide_sys()
        exp = _nitroxide_exp()
        B, spec = garlic(sys, exp)
        assert isinstance(B, torch.Tensor)
        assert isinstance(spec, torch.Tensor)

    def test_output_shape(self):
        sys = _nitroxide_sys()
        exp = _nitroxide_exp()
        B, spec = garlic(sys, exp)
        assert B.shape == (512,)
        assert spec.shape == (512,)

    def test_spectrum_is_finite(self):
        sys = _nitroxide_sys()
        exp = _nitroxide_exp()
        _, spec = garlic(sys, exp)
        assert torch.isfinite(spec).all()

    def test_spectrum_nonzero(self):
        sys = _nitroxide_sys()
        exp = _nitroxide_exp()
        sys.tcorr = 1e-10
        _, spec = garlic(sys, exp)
        assert spec.abs().max() > 0, "Fast-motion spectrum is empty"

    def test_fast_motion_vs_isotropic(self):
        """
        Fast-motion (tcorr>0) and isotropic (tcorr=0) should both produce
        spectra with the same number of lines (3 for 14N) but possibly
        different linewidths.  Both should be non-zero within the range.
        """
        sys_iso = _nitroxide_sys()
        sys_iso.tcorr = 0.0
        sys_fast = _nitroxide_sys()
        sys_fast.tcorr = 1e-10

        exp = _nitroxide_exp()
        opt = Options(Verbosity=0)

        _, spec_iso = garlic(sys_iso, exp, opt)
        _, spec_fast = garlic(sys_fast, exp, opt)

        # Both have content
        assert spec_iso.abs().max() > 0
        assert spec_fast.abs().max() > 0

    def test_tcorr_broadens_lines(self):
        """
        Larger tcorr → broader lines → smaller peak amplitude (derivative).
        """
        sys1 = _nitroxide_sys()
        sys1.tcorr = 1e-11  # short tcorr → narrow lines

        sys2 = _nitroxide_sys()
        sys2.tcorr = 5e-10  # longer tcorr → broader lines

        exp = _nitroxide_exp()
        opt = Options(Verbosity=0)

        _, spec1 = garlic(sys1, exp, opt)
        _, spec2 = garlic(sys2, exp, opt)

        # Broader lines → smaller peak height for derivative spectrum
        assert spec1.abs().max() > spec2.abs().max(), (
            "Shorter tcorr should give taller (narrower) derivative peaks"
        )

    def test_gaussian_broadening_with_fastmotion(self):
        """Adding Gaussian broadening to fast-motion should further broaden the spectrum."""
        sys_no_g = _nitroxide_sys()
        sys_no_g.tcorr = 1e-10

        sys_with_g = _nitroxide_sys()
        sys_with_g.tcorr = 1e-10
        sys_with_g.lw = torch.tensor([0.5, 0.0], dtype=torch.float64)

        exp = _nitroxide_exp()
        opt = Options(Verbosity=0)

        _, spec_no_g = garlic(sys_no_g, exp, opt)
        _, spec_g = garlic(sys_with_g, exp, opt)

        # Gaussian broadening reduces peak height
        assert spec_no_g.abs().max() > spec_g.abs().max(), (
            "Gaussian broadening should reduce peak height"
        )

    def test_logtcorr_attribute(self):
        """sys.logtcorr should be handled equivalently to sys.tcorr."""
        sys1 = _nitroxide_sys()
        sys1.tcorr = 1e-10

        sys2 = _nitroxide_sys()
        sys2.logtcorr = -10.0  # log10(1e-10) = -10

        exp = _nitroxide_exp()
        opt = Options(Verbosity=0)

        _, spec1 = garlic(sys1, exp, opt)
        _, spec2 = garlic(sys2, exp, opt)

        # Should give identical spectra
        assert torch.allclose(spec1, spec2, atol=1e-10), (
            "tcorr=1e-10 and logtcorr=-10 should give identical spectra"
        )


# ---------------------------------------------------------------------------
# Physics: fast-motion linewidth pattern for nitroxide
# ---------------------------------------------------------------------------

class TestFastMotionPhysics:
    def test_linewidth_pattern_nitroxide(self):
        """
        For a nitroxide with isotropic A_perp < A_par, the B coefficient is
        significant. The linewidth pattern should show asymmetry:
        lw(mI=-1) ≠ lw(mI=+1) when B ≠ 0.

        Specifically, for positive A anisotropy (A_par > A_perp),
        the B coefficient contribution means the outer lines have different widths.
        """
        sys = _nitroxide_sys()
        lw, mI_all = fastmotion(sys, 340.0, 1e-10)

        # mI order: -1, 0, +1
        lw_m1 = lw[0]   # mI = -1
        lw_0  = lw[1]   # mI =  0
        lw_p1 = lw[2]   # mI = +1

        # Center line (mI=0) has no B contribution → typically narrowest
        # (C contributes equally to mI=±1, and B contributes antisymmetrically)
        # All should be positive
        assert lw_m1 > 0 and lw_0 > 0 and lw_p1 > 0

        # The two outer lines should NOT be equal (B term breaks symmetry)
        assert abs(lw_m1 - lw_p1) > 1e-5 * max(lw_m1, lw_p1), (
            f"Outer linewidths should differ: lw(-1)={lw_m1:.4f}, lw(+1)={lw_p1:.4f}"
        )

    def test_only_g_anisotropy(self):
        """
        With only g anisotropy (no nuclei), there is just one EPR line and
        A=gg*(2/15*j0+1/10*j1) > 0.
        """
        sys = SpinSystem(S=[0.5], g=[[2.009, 2.006, 2.002]])
        lw, mI_all = fastmotion(sys, 340.0, 1e-10)
        assert lw.shape == (1,)
        assert lw[0] > 0

    def test_curie_law_consistency(self):
        """
        Linewidths should scale approximately linearly with j0 = tcorr in
        the extreme narrowing limit (ω₀ τ ≪ 1).

        At X-band (9.5 GHz), ω₀ ≈ 6×10¹⁰ rad/s, so extreme narrowing
        requires τ ≪ 1/ω₀ ≈ 17 ps.  We use τ₁ = 1e-12 and τ₂ = 1e-11
        (both in extreme narrowing) → ratio should be close to 10.
        """
        sys = _nitroxide_sys()
        tcorr1 = 1e-12   # ω₀τ ≈ 0.06 → well in extreme narrowing
        tcorr2 = 1e-11   # ω₀τ ≈ 0.6 → marginally in extreme narrowing
        lw1, _ = fastmotion(sys, 340.0, tcorr1)
        lw2, _ = fastmotion(sys, 340.0, tcorr2)
        ratio = lw2 / lw1
        # Allow wider range [3, 20] to accommodate j1 ≠ j0 at ω₀τ ≈ 0.6
        assert np.all(ratio > 3.0) and np.all(ratio < 20.0), (
            f"Expected lw ratio ~10 for tcorr ratio 10, got {ratio}"
        )
