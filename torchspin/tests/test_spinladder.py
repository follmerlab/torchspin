"""Tests for torchspin.spinladder — exchange-coupled spin manifolds."""
import math
import pytest

from torchspin.spinladder import spinladder, SpinManifold


class TestSpinladderBasic:
    def test_half_half_count(self):
        """S1=S2=1/2: two manifolds (singlet + triplet)."""
        manifolds = spinladder(0.5, 0.5, 100.0)
        assert len(manifolds) == 2

    def test_half_half_S_values(self):
        """S_total = 0 and 1 for two S=1/2 spins."""
        manifolds = spinladder(0.5, 0.5, 100.0)
        S_vals = [m.S_total for m in manifolds]
        assert 0.0 in S_vals
        assert 1.0 in S_vals

    def test_half_half_multiplicities(self):
        """Singlet mult=1, triplet mult=3."""
        manifolds = spinladder(0.5, 0.5, 100.0)
        mults = {m.S_total: m.multiplicity for m in manifolds}
        assert mults[0.0] == 1
        assert mults[1.0] == 3

    def test_antiferro_singlet_lowest(self):
        """J>0 (antiferro): singlet has lowest energy."""
        manifolds = spinladder(0.5, 0.5, 100.0)
        assert manifolds[0].S_total == 0.0
        assert manifolds[0].energy_MHz == pytest.approx(0.0, abs=1e-6)

    def test_antiferro_triplet_energy(self):
        """J=100 MHz: triplet energy above singlet = J."""
        manifolds = spinladder(0.5, 0.5, 100.0)
        d = {m.S_total: m.energy_MHz for m in manifolds}
        # E(S=1) - E(S=0) = J/2 * [1*2 - 0] = J (since offset is constant)
        assert abs(d[1.0] - 100.0) < 1e-6

    def test_ferro_triplet_lowest(self):
        """J<0 (ferromagnetic): highest spin has lowest energy."""
        manifolds = spinladder(0.5, 0.5, -100.0)
        assert manifolds[0].S_total == 1.0

    def test_one_one_count(self):
        """S1=S2=1: three manifolds (S_tot = 0, 1, 2)."""
        manifolds = spinladder(1.0, 1.0, 50.0)
        assert len(manifolds) == 3
        S_vals = sorted([m.S_total for m in manifolds])
        assert S_vals == pytest.approx([0.0, 1.0, 2.0])

    def test_one_half_count(self):
        """S1=1, S2=0.5: two manifolds (S_tot = 0.5, 1.5)."""
        manifolds = spinladder(1.0, 0.5, 100.0)
        assert len(manifolds) == 2
        S_vals = sorted([m.S_total for m in manifolds])
        assert S_vals == pytest.approx([0.5, 1.5])

    def test_sorted_by_energy(self):
        """Manifolds are sorted by ascending energy."""
        manifolds = spinladder(1.0, 1.0, 50.0)
        energies = [m.energy_MHz for m in manifolds]
        assert energies == sorted(energies)

    def test_energy_offset(self):
        """Minimum energy is always 0."""
        manifolds = spinladder(0.5, 0.5, 100.0)
        assert manifolds[0].energy_MHz == pytest.approx(0.0, abs=1e-9)

    def test_g_eff(self):
        """Effective g = (g1 + g2)/2."""
        manifolds = spinladder(0.5, 0.5, 100.0, g1=2.1, g2=2.0)
        for m in manifolds:
            assert m.g_eff == pytest.approx(2.05)

    def test_returns_spinmanifold(self):
        """Elements are SpinManifold named tuples."""
        manifolds = spinladder(0.5, 0.5, 100.0)
        assert isinstance(manifolds[0], SpinManifold)


class TestSpinladderBoltzmann:
    def test_weights_sum_to_one(self):
        """Boltzmann weights sum to 1.0."""
        manifolds = spinladder(0.5, 0.5, 100.0, temperature=100.0)
        assert sum(m.weight for m in manifolds) == pytest.approx(1.0, rel=1e-8)

    def test_low_T_singlet_dominant(self):
        """At low T (AF coupling): singlet dominates.
        Note: 1 MHz << kT at 1K, so need large J (1e10 MHz) to see polarisation.
        """
        # kT at 1K ≈ 20.8 GHz*h → need J >> 20800 MHz for singlet domination
        manifolds = spinladder(0.5, 0.5, 1e9, temperature=1.0)
        d = {m.S_total: m.weight for m in manifolds}
        assert d[0.0] > d[1.0]

    def test_high_T_equal_weights(self):
        """At very high T: weights proportional to multiplicity."""
        manifolds = spinladder(0.5, 0.5, 100.0, temperature=1e10)
        # singlet (mult=1) vs triplet (mult=3): weight ratio ~1:3
        d = {m.S_total: m.weight for m in manifolds}
        assert abs(d[1.0] / d[0.0] - 3.0) < 0.01

    def test_none_temperature_uniform_weights(self):
        """temperature=None → all weights = 1.0 (unnormalised)."""
        manifolds = spinladder(0.5, 0.5, 100.0, temperature=None)
        for m in manifolds:
            assert m.weight == pytest.approx(1.0)

    def test_ferromagnetic_triplet_dominant_at_low_T(self):
        """Ferromagnetic J<0 at low T: triplet dominant."""
        manifolds = spinladder(0.5, 0.5, -1e9, temperature=1.0)
        d = {m.S_total: m.weight for m in manifolds}
        assert d[1.0] > d[0.0]
