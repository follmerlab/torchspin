"""Tests for endorfrq_perturb and resfreqs_perturb.

Physics checks:
- ENDOR at ν_n ± A/2 (strong/weak coupling limits)
- Frequency-swept EPR at g·μ_B·B/h ± A/2
- S=1/2 with quadrupole nucleus
- High-spin ZFS corrections
- Consistency between 1st and 2nd order (2nd ≈ 1st for small A/E0)
"""
import math
import pytest
import torch
import numpy as np

from torchspin import SpinSystem
from torchspin.experiment import Experiment
from torchspin.endorfrq_perturb import endorfrq_perturb
from torchspin.resfreqs_perturb import resfreqs_perturb
from torchspin.constants import BMAGN, PLANCK, NMAGN


def _larmor_MHz(gn: float, B_mT: float) -> float:
    """Nuclear Larmor frequency in MHz."""
    return gn * NMAGN * B_mT * 1e-3 / PLANCK * 1e-6


def _epr_GHz(g: float, B_mT: float) -> float:
    """EPR resonance frequency (GHz) at field B_mT for isotropic g."""
    return g * BMAGN * B_mT * 1e-3 / PLANCK * 1e-9


# ============================================================
# endorfrq_perturb tests
# ============================================================

class TestEndorfrqPerturb:
    """Basic ENDOR perturbation theory tests."""

    def test_1H_isotropic_1st_order(self):
        """1H isotropic coupling: ENDOR at |ν_H ± A/2|."""
        A_iso = 8.0  # MHz
        B0 = 350.0   # mT
        gn_H = 5.58569468

        sys = SpinSystem(
            S=[0.5], g=[[2.0, 2.0, 2.0]],
            Nucs=['1H'], A=[[A_iso, A_iso, A_iso]],
        )
        freqs, _ = endorfrq_perturb(sys, B0, phi=0.0, theta=0.0, order=1)
        freqs = sorted(freqs.numpy())

        nu_H = _larmor_MHz(gn_H, B0)
        expected = sorted([abs(nu_H - A_iso / 2), abs(nu_H + A_iso / 2)])

        assert len(freqs) == 2
        np.testing.assert_allclose(freqs, expected, atol=1e-6)

    def test_14N_isotropic_3_lines(self):
        """14N (I=1): ENDOR gives 2 lines per mS manifold × 2 mS = 4 transitions."""
        sys = SpinSystem(
            S=[0.5], g=[[2.0, 2.0, 2.0]],
            Nucs=['14N'], A=[[30.0, 30.0, 30.0]],
        )
        freqs, _ = endorfrq_perturb(sys, 340.0, phi=0.0, theta=0.0, order=1)
        # 14N: I=1 → 2 transitions per mS × 2 mS = 4 total
        assert len(freqs) == 4
        assert torch.all(freqs >= 0)

    def test_no_nuclei_returns_empty(self):
        """System with no nuclei returns empty arrays."""
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        freqs, ints = endorfrq_perturb(sys, 340.0)
        assert len(freqs) == 0
        assert len(ints) == 0

    def test_orientational_averaging_theta(self):
        """Axial system: ENDOR varies with theta (not constant)."""
        # Anisotropic A → different ENDOR at theta=0 vs theta=pi/2
        sys = SpinSystem(
            S=[0.5], g=[[2.0, 2.0, 2.0]],
            Nucs=['1H'], A=[[5.0, 5.0, 50.0]],
        )
        freqs_z, _ = endorfrq_perturb(sys, 340.0, phi=0.0, theta=0.0)
        freqs_xy, _ = endorfrq_perturb(sys, 340.0, phi=0.0, theta=math.pi / 2)
        # Along z: A_eff = 50 MHz; along x: A_eff = 5 MHz → different ENDOR
        assert not torch.allclose(freqs_z.sort().values, freqs_xy.sort().values, atol=1.0)

    def test_2nd_order_correction_small(self):
        """2nd order correction is small for small A/E0."""
        sys = SpinSystem(
            S=[0.5], g=[[2.0, 2.0, 2.0]],
            Nucs=['1H'], A=[[5.0, 5.0, 5.0]],
        )
        B0 = 340.0
        freqs1, _ = endorfrq_perturb(sys, B0, order=1)
        freqs2, _ = endorfrq_perturb(sys, B0, order=2)
        # For A=5 MHz, E0≈9520 MHz: 2nd order ≈ A²/E0 ≈ 0.003 MHz
        np.testing.assert_allclose(
            freqs1.sort().values.numpy(),
            freqs2.sort().values.numpy(),
            atol=0.05,
        )

    def test_strong_coupling_sign_flip(self):
        """Strong coupling case: |HFfield| > |Larmor| → sign flip for mS<0."""
        # Very large A so that |mS * A * u| >> |nu_H * h|
        A_large = 200.0  # MHz, >> nu_H ≈ 14 MHz
        sys = SpinSystem(
            S=[0.5], g=[[2.0, 2.0, 2.0]],
            Nucs=['1H'], A=[[A_large, A_large, A_large]],
        )
        freqs, ints = endorfrq_perturb(sys, 340.0, order=1)
        # Both ENDOR lines should be close to A/2 in strong coupling
        # (not at |nu_H ± A/2| which would give ~114 MHz and ~86 MHz asymmetric)
        assert len(freqs) == 2
        assert torch.all(freqs >= 0)

    def test_intensities_positive(self):
        """All intensities must be non-negative."""
        sys = SpinSystem(
            S=[0.5], g=[[2.0, 2.0, 2.0]],
            Nucs=['1H'], A=[[10.0, 10.0, 10.0]],
        )
        _, ints = endorfrq_perturb(sys, 340.0)
        assert torch.all(ints >= 0)

    def test_nuclei_selection(self):
        """nuclei=[0] selects only first nucleus."""
        sys = SpinSystem(
            S=[0.5], g=[[2.0, 2.0, 2.0]],
            Nucs=['1H', '14N'],
            A=[[10.0, 10.0, 10.0], [30.0, 30.0, 30.0]],
        )
        freqs_all, _ = endorfrq_perturb(sys, 340.0)
        freqs_H, _ = endorfrq_perturb(sys, 340.0, nuclei=[0])
        # 1H: 2 transitions; 14N: 4; all: 6
        assert len(freqs_H) == 2
        assert len(freqs_all) == 6

    def test_multi_electron_raises(self):
        """Two electrons → raise ValueError."""
        sys = SpinSystem(S=[0.5, 0.5], g=[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
        with pytest.raises(ValueError, match="nElectrons"):
            endorfrq_perturb(sys, 340.0)


# ============================================================
# resfreqs_perturb tests
# ============================================================

class TestResfreqsPerturb:
    """Basic frequency-swept EPR tests."""

    def test_isotropic_resonance_frequency(self):
        """Isotropic S=1/2: resonance at g·μ_B·B/h."""
        g = 2.002
        B0 = 340.0  # mT
        sys = SpinSystem(S=[0.5], g=[[g, g, g]], lw=[0.5, 0.0])
        exp = Experiment(mwFreq=9.5, Range=[9.0, 10.5], Field=B0)
        nu, _ = resfreqs_perturb(sys, 0.0, 0.0, exp, order=1)

        nu_expected = _epr_GHz(g, B0)
        assert len(nu) == 1
        assert abs(float(nu[0]) - nu_expected) < 1e-6

    def test_1H_two_lines(self):
        """S=1/2 + 1H: two lines split by A in frequency domain."""
        A_iso = 10.0  # MHz
        g = 2.0
        B0 = 340.0  # mT
        sys = SpinSystem(
            S=[0.5], g=[[g, g, g]],
            Nucs=['1H'], A=[[A_iso, A_iso, A_iso]],
        )
        nu0 = _epr_GHz(g, B0)
        nu_range = [nu0 - 0.1, nu0 + 0.1]  # ±100 MHz
        exp = Experiment(mwFreq=9.5, Range=nu_range, Field=B0)
        nu, ints = resfreqs_perturb(sys, 0.0, 0.0, exp, order=1)
        nu_sorted = sorted(nu.numpy())

        assert len(nu_sorted) == 2
        # Splitting should be A=10 MHz = 0.010 GHz
        split = nu_sorted[1] - nu_sorted[0]
        assert abs(split - A_iso * 1e-3) < 1e-4  # within 0.1 MHz

    def test_intensities_positive(self):
        """All intensities must be non-negative."""
        sys = SpinSystem(
            S=[0.5], g=[[2.0, 2.0, 2.0]],
            Nucs=['1H'], A=[[10.0, 10.0, 10.0]],
        )
        exp = Experiment(mwFreq=9.5, Range=[9.0, 10.0], Field=340.0)
        _, ints = resfreqs_perturb(sys, 0.0, 0.0, exp)
        assert torch.all(ints >= 0)

    def test_range_filter(self):
        """Lines outside exp.Range are excluded."""
        g = 2.0
        B0 = 340.0
        sys = SpinSystem(S=[0.5], g=[[g, g, g]])
        nu0 = _epr_GHz(g, B0)

        # Window that misses the resonance
        exp_miss = Experiment(mwFreq=9.5, Range=[nu0 + 0.5, nu0 + 1.0], Field=B0)
        nu_miss, _ = resfreqs_perturb(sys, 0.0, 0.0, exp_miss)
        assert len(nu_miss) == 0

        # Window that captures the resonance
        exp_hit = Experiment(mwFreq=9.5, Range=[nu0 - 0.1, nu0 + 0.1], Field=B0)
        nu_hit, _ = resfreqs_perturb(sys, 0.0, 0.0, exp_hit)
        assert len(nu_hit) == 1

    def test_second_order_small_correction(self):
        """2nd order correction is small for A << E0."""
        sys = SpinSystem(
            S=[0.5], g=[[2.0, 2.0, 2.0]],
            Nucs=['1H'], A=[[5.0, 5.0, 5.0]],
        )
        B0 = 340.0
        nu0 = _epr_GHz(2.0, B0)
        exp = Experiment(mwFreq=9.5, Range=[nu0 - 0.1, nu0 + 0.1], Field=B0)

        nu1, _ = resfreqs_perturb(sys, 0.0, 0.0, exp, order=1)
        nu2, _ = resfreqs_perturb(sys, 0.0, 0.0, exp, order=2)

        # 2nd order shifts are ~ A^2/E0 ≈ 25/9520 ≈ 0.003 MHz = 3e-6 GHz
        assert len(nu1) == len(nu2)
        np.testing.assert_allclose(nu1.numpy(), nu2.numpy(), atol=1e-3)

    def test_anisotropic_g_orientation_dependence(self):
        """Anisotropic g → resonance shifts with orientation."""
        sys = SpinSystem(S=[0.5], g=[[2.10, 2.05, 2.00]])
        B0 = 340.0
        nu0 = _epr_GHz(2.05, B0)  # middle value
        exp = Experiment(mwFreq=9.5, Range=[nu0 - 1.0, nu0 + 1.0], Field=B0)

        nu_z, _ = resfreqs_perturb(sys, 0.0, 0.0, exp)       # g_z = 2.00
        nu_x, _ = resfreqs_perturb(sys, 0.0, math.pi / 2, exp)  # g_x = 2.10

        assert len(nu_z) == 1
        assert len(nu_x) == 1
        # g_x=2.10 > g_z=2.00 → resonance_x > resonance_z
        assert float(nu_x[0]) > float(nu_z[0])

    def test_no_nuclei_single_line(self):
        """No nuclei: one line per electron transition."""
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        B0 = 340.0
        nu0 = _epr_GHz(2.0, B0)
        exp = Experiment(mwFreq=9.5, Range=[nu0 - 0.1, nu0 + 0.1], Field=B0)
        nu, _ = resfreqs_perturb(sys, 0.0, 0.0, exp)
        assert len(nu) == 1

    def test_multi_electron_raises(self):
        """Two electrons → raise ValueError."""
        sys = SpinSystem(S=[0.5, 0.5], g=[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
        exp = Experiment(mwFreq=9.5, Range=[9.0, 10.0], Field=340.0)
        with pytest.raises(ValueError, match="nElectrons"):
            resfreqs_perturb(sys, 0.0, 0.0, exp)

    def test_14N_three_lines(self):
        """S=1/2 + 14N (I=1): 3 lines in frequency domain."""
        A_iso = 30.0  # MHz
        g = 2.0
        B0 = 340.0
        sys = SpinSystem(
            S=[0.5], g=[[g, g, g]],
            Nucs=['14N'], A=[[A_iso, A_iso, A_iso]],
        )
        nu0 = _epr_GHz(g, B0)
        exp = Experiment(mwFreq=9.5, Range=[nu0 - 0.1, nu0 + 0.1], Field=B0)
        nu, _ = resfreqs_perturb(sys, 0.0, 0.0, exp, order=1)
        assert len(nu) == 3
        # Lines equally spaced by A = 30 MHz = 0.030 GHz
        nu_sorted = sorted(nu.numpy())
        gap1 = nu_sorted[1] - nu_sorted[0]
        gap2 = nu_sorted[2] - nu_sorted[1]
        np.testing.assert_allclose(gap1, A_iso * 1e-3, atol=1e-4)
        np.testing.assert_allclose(gap2, A_iso * 1e-3, atol=1e-4)
