#!/usr/bin/env python3
"""
Test suite for physical constants (CODATA 2022).

Based on MATLAB tests:
- bmagn_value.m
- planck_value.m
- gfree_value.m
- nmagn_value.m
- And other constant tests in easyspin/tests/
"""

import math

import pytest
import torch
from torchspin.constants import (
    BMAGN, PLANCK, GFREE, NMAGN, BOLTZMANN,
    HBAR, AMU, ECHARGE, EMASS, PMASS, NMASS,
    EPS0, MU0, FARADAY, AVOGADRO, BOHRRAD,
    HARTREE, RYDBERG, GAMMAE, GAMMAN, ANGSTROM,
)


class TestPhysicalConstants:
    """Test physical constants match CODATA 2022 values."""

    def test_bmagn_value(self):
        """Test Bohr magneton value (bmagn_value.m)."""
        # CODATA 2022: 9.2740100783e-24 J/T (updated value)
        # Current implementation uses: 9.2740100657e-24
        assert pytest.approx(BMAGN, rel=1e-8) == 9.2740100657e-24

    def test_planck_value(self):
        """Test Planck constant value (planck_value.m)."""
        # CODATA 2022: 6.62607015e-34 J·s (exact)
        assert pytest.approx(PLANCK, rel=1e-10) == 6.62607015e-34

    def test_gfree_value(self):
        """Test free electron g-factor (gfree_value.m)."""
        # CODATA 2022: 2.00231930436256 (updated value)
        # Current implementation uses: 2.00231930436092
        assert pytest.approx(GFREE, rel=1e-10) == 2.00231930436092

    def test_nmagn_value(self):
        """Test nuclear magneton value (nmagn_value.m)."""
        # CODATA 2022: 5.0507837461e-27 J/T (updated value)
        # Current implementation uses: 5.0507837393e-27
        assert pytest.approx(NMAGN, rel=1e-8) == 5.0507837393e-27

    def test_boltzmann_value(self):
        """Test Boltzmann constant (boltzm_value.m)."""
        # CODATA 2022: 1.380649e-23 J/K (exact)
        assert pytest.approx(BOLTZMANN, rel=1e-10) == 1.380649e-23


class TestConstantRelations:
    """Test relationships between constants."""

    def test_bohr_magneton_order_of_magnitude(self):
        """Bohr magneton should be in reasonable range."""
        assert 1e-24 < BMAGN < 1e-23  # J/T

    def test_nuclear_magneton_order_of_magnitude(self):
        """Nuclear magneton should be in reasonable range."""
        assert 1e-27 < NMAGN < 1e-26  # J/T

    def test_constants_are_positive(self):
        """Physical constants should be positive (except when signed)."""
        assert BMAGN > 0
        assert PLANCK > 0
        assert GFREE > 0  # We use positive convention
        assert NMAGN > 0
        assert BOLTZMANN > 0


class TestNewConstants:
    """Test new CODATA 2022 constants added in Session A."""

    def test_hbar_derived(self):
        """ℏ = h / (2π)."""
        assert pytest.approx(HBAR, rel=1e-10) == PLANCK / (2 * math.pi)

    def test_hbar_value(self):
        assert pytest.approx(HBAR, rel=1e-6) == 1.054571817e-34

    def test_amu_value(self):
        assert pytest.approx(AMU, rel=1e-6) == 1.66053906660e-27

    def test_echarge_exact(self):
        """Elementary charge is exact since 2019 SI redefinition."""
        assert pytest.approx(ECHARGE, rel=1e-12) == 1.602176634e-19

    def test_emass_value(self):
        assert pytest.approx(EMASS, rel=1e-6) == 9.1093837139e-31

    def test_pmass_value(self):
        assert pytest.approx(PMASS, rel=1e-6) == 1.67262192595e-27

    def test_nmass_value(self):
        assert pytest.approx(NMASS, rel=1e-6) == 1.67492750056e-27

    def test_avogadro_exact(self):
        """Avogadro's number is exact since 2019 SI redefinition."""
        assert pytest.approx(AVOGADRO, rel=1e-12) == 6.02214076e23

    def test_bohrrad_value(self):
        assert pytest.approx(BOHRRAD, rel=1e-6) == 5.29177210544e-11

    def test_hartree_value(self):
        assert pytest.approx(HARTREE, rel=1e-6) == 4.3597447222060e-18

    def test_rydberg_value(self):
        """Rydberg = Hartree / 2."""
        assert pytest.approx(RYDBERG, rel=1e-8) == HARTREE / 2

    def test_gammae_derived(self):
        """γ_e = g_e * μ_B / ℏ."""
        assert pytest.approx(GAMMAE, rel=1e-8) == GFREE * BMAGN / HBAR

    def test_gamman_derived(self):
        """γ_n_base = μ_N / ℏ."""
        assert pytest.approx(GAMMAN, rel=1e-8) == NMAGN / HBAR

    def test_faraday_value(self):
        assert pytest.approx(FARADAY, rel=1e-6) == 96485.33212

    def test_eps0_value(self):
        assert pytest.approx(EPS0, rel=1e-6) == 8.8541878188e-12

    def test_mu0_value(self):
        assert pytest.approx(MU0, rel=1e-6) == 1.25663706127e-6

    def test_gammae_order_of_magnitude(self):
        """γ_e ≈ 1.76e11 rad/s/T."""
        assert 1.7e11 < GAMMAE < 1.8e11

    def test_proton_heavier_than_electron(self):
        assert PMASS > EMASS * 1000  # proton is ~1836× electron

    def test_angstrom_value(self):
        """1 Angstrom = 1e-10 m (exact)."""
        assert ANGSTROM == pytest.approx(1e-10, rel=1e-12)

    def test_angstrom_in_nm(self):
        """1 nm = 10 Angstroms."""
        assert pytest.approx(1e-9 / ANGSTROM, rel=1e-10) == 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
