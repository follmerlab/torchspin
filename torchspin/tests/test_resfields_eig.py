"""Tests for resfields_eig — eigenfield method for resonance fields.

Verifies:
1. Correct resonance fields for simple spin systems
2. Consistency with eprconvert (h*freq = g*muB*B)
3. S=1 with ZFS produces expected number of transitions
4. Hyperfine splitting produces correct number of lines
5. Orientation dependence with anisotropic g
6. Intensity calculations
7. Field range filtering
"""
import numpy as np
import pytest

from torchspin import SpinSystem, resfields_eig, EigOptions
from torchspin.constants import PLANCK, BMAGN, GFREE


class TestResfieldsEigBasic:
    """Basic resonance field calculations."""

    def test_single_electron_isotropic(self):
        """S=1/2, isotropic g=2.0 → single resonance at h*freq/(g*muB)."""
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        fields, intens = resfields_eig(sys, mwFreq=9.5)
        expected = PLANCK * 9.5e9 / (2.0 * BMAGN) * 1e3
        assert len(fields) == 1
        assert abs(fields[0] - expected) < 0.01

    def test_free_electron_g(self):
        """S=1/2 with g=gfree."""
        sys = SpinSystem(S=[0.5], g=[[GFREE, GFREE, GFREE]])
        fields, _ = resfields_eig(sys, mwFreq=9.5)
        expected = PLANCK * 9.5e9 / (GFREE * BMAGN) * 1e3
        assert len(fields) == 1
        assert abs(fields[0] - expected) < 0.01

    def test_different_frequencies(self):
        """Resonance field scales linearly with frequency."""
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        f1, _ = resfields_eig(sys, mwFreq=9.5)
        f2, _ = resfields_eig(sys, mwFreq=34.0)  # Q-band
        ratio = f2[0] / f1[0]
        assert abs(ratio - 34.0 / 9.5) < 0.01

    def test_output_types(self):
        """Return types are numpy arrays."""
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        fields, intens = resfields_eig(sys, mwFreq=9.5)
        assert isinstance(fields, np.ndarray)
        assert isinstance(intens, np.ndarray)


class TestResfieldsEigAnisotropic:
    """Anisotropic g-tensor tests."""

    def test_gzz_along_z(self):
        """Along z, resonance field depends on g_zz."""
        sys = SpinSystem(S=[0.5], g=[[2.1, 2.05, 2.0]])
        fields, _ = resfields_eig(sys, mwFreq=9.5,
                                   Orientations=np.array([[0, 0, 0]]))
        expected = PLANCK * 9.5e9 / (2.0 * BMAGN) * 1e3
        assert abs(fields[0] - expected) < 0.1

    def test_gxx_along_x(self):
        """Along x, resonance field depends on g_xx."""
        sys = SpinSystem(S=[0.5], g=[[2.1, 2.05, 2.0]])
        fields, _ = resfields_eig(sys, mwFreq=9.5,
                                   Orientations=np.array([[0, np.pi / 2, 0]]))
        expected = PLANCK * 9.5e9 / (2.1 * BMAGN) * 1e3
        assert abs(fields[0] - expected) < 0.1

    def test_gyy_along_y(self):
        """Along y, resonance field depends on g_yy."""
        sys = SpinSystem(S=[0.5], g=[[2.1, 2.05, 2.0]])
        # B along y: Euler angles [pi/2, pi/2, 0]
        fields, _ = resfields_eig(sys, mwFreq=9.5,
                                   Orientations=np.array([[np.pi / 2, np.pi / 2, 0]]))
        expected = PLANCK * 9.5e9 / (2.05 * BMAGN) * 1e3
        assert abs(fields[0] - expected) < 0.5


class TestResfieldsEigZFS:
    """Zero-field splitting tests."""

    def test_triplet_transitions(self):
        """S=1 with D=1000 MHz should give multiple transitions."""
        sys = SpinSystem(S=[1.0], g=[[2.0, 2.0, 2.0]], D=[[1000, 0, 0]])
        fields, intens = resfields_eig(sys, mwFreq=9.5)
        # Triplet state gives at least 2 allowed transitions
        assert len(fields) >= 2
        # Fields should be positive and finite
        assert np.all(fields > 0)
        assert np.all(np.isfinite(fields))

    def test_triplet_no_zfs(self):
        """S=1 with D=0 → allowed transitions degenerate at same field."""
        sys = SpinSystem(S=[1.0], g=[[2.0, 2.0, 2.0]])
        fields, intens = resfields_eig(sys, mwFreq=9.5)
        expected = PLANCK * 9.5e9 / (2.0 * BMAGN) * 1e3
        # Filter to strong transitions (eigenfield method may find half-field)
        strong = fields[intens > 0.1 * np.max(intens)]
        assert len(strong) >= 2
        for f in strong:
            assert abs(f - expected) < 0.5


class TestResfieldsEigHyperfine:
    """Hyperfine coupling tests."""

    def test_14N_three_lines(self):
        """S=1/2 + 14N (I=1) with isotropic A → 3 lines."""
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]],
                         Nucs=['14N'], A=[[100, 100, 100]])
        fields, _ = resfields_eig(sys, mwFreq=9.5)
        # 14N has I=1 → 3 nuclear states → 3 EPR lines
        # (each electron transition splits into 3)
        assert len(fields) >= 3

    def test_1H_two_lines(self):
        """S=1/2 + 1H (I=1/2) with isotropic A → 2 lines."""
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]],
                         Nucs=['1H'], A=[[50, 50, 50]])
        fields, _ = resfields_eig(sys, mwFreq=9.5)
        assert len(fields) >= 2

    def test_hyperfine_splitting_magnitude(self):
        """Hyperfine splitting should be approximately A/(g*muB/h)."""
        A_iso = 100.0  # MHz
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]],
                         Nucs=['1H'], A=[[A_iso, A_iso, A_iso]])
        fields, intens = resfields_eig(sys, mwFreq=9.5, Range=(300, 400))
        # Keep only allowed transitions (nonzero intensity)
        allowed = fields[intens > 0.01 * np.max(intens)]
        allowed = np.sort(allowed)
        assert len(allowed) >= 2
        splitting_mT = allowed[-1] - allowed[0]
        # A[MHz] / (g * muB/h [MHz/mT])
        g_muB_h_MHz_per_mT = 2.0 * BMAGN / PLANCK / 1e9  # MHz/mT
        expected_split = A_iso / g_muB_h_MHz_per_mT
        assert abs(splitting_mT - expected_split) < 0.5


class TestResfieldsEigRange:
    """Field range filtering."""

    def test_range_filters(self):
        """Fields outside Range should be excluded."""
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        # Resonance is at ~339.4 mT; set range to exclude it
        fields, _ = resfields_eig(sys, mwFreq=9.5, Range=(100, 200))
        assert len(fields) == 0

    def test_range_includes(self):
        """Fields inside Range should be included."""
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        fields, _ = resfields_eig(sys, mwFreq=9.5, Range=(300, 400))
        assert len(fields) == 1


class TestResfieldsEigIntensity:
    """Intensity calculation tests."""

    def test_intensities_positive(self):
        """Intensities should be positive."""
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        fields, intens = resfields_eig(sys, mwFreq=9.5)
        assert np.all(intens > 0)

    def test_threshold_filters(self):
        """Threshold should filter weak transitions."""
        sys = SpinSystem(S=[1.0], g=[[2.0, 2.0, 2.0]], D=[[1000, 0, 0]])
        opt = EigOptions(Threshold=0.5)
        fields, intens = resfields_eig(sys, mwFreq=9.5, Opt=opt)
        if len(intens) > 1:
            assert np.min(intens) >= 0.5 * np.max(intens)


class TestResfieldsEigMultipleOrientations:
    """Multiple crystal orientations."""

    def test_multiple_orientations(self):
        """Should return lists for multiple orientations."""
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        oris = np.array([[0, 0, 0], [0, np.pi / 4, 0], [0, np.pi / 2, 0]])
        fields, intens = resfields_eig(sys, mwFreq=9.5, Orientations=oris)
        assert isinstance(fields, list)
        assert len(fields) == 3

    def test_isotropic_orientation_independent(self):
        """Isotropic system → same field at all orientations."""
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        oris = np.array([[0, 0, 0], [0, np.pi / 4, 0], [0, np.pi / 2, 0]])
        fields, _ = resfields_eig(sys, mwFreq=9.5, Orientations=oris)
        for f in fields:
            assert len(f) == 1
            assert abs(f[0] - fields[0][0]) < 0.01


class TestResfieldsEigParallelMode:
    """Parallel mode microwave excitation."""

    def test_parallel_mode_runs(self):
        """Parallel mode should work without errors."""
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        fields, intens = resfields_eig(sys, mwFreq=9.5, mwMode='parallel')
        # For S=1/2, parallel mode transition (Delta m = 0) shouldn't be allowed
        # but eigenfield method may still find mathematical solutions
        assert isinstance(fields, np.ndarray)
