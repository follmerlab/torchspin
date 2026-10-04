"""Tests for Tier 4 quick wins: MOLGAS, eprconvert, SpinSystem.validate()."""
import math
import pytest
import numpy as np

from torchspin import MOLGAS, BOLTZMANN, AVOGADRO, BMAGN, PLANCK, GFREE
from torchspin import eprconvert
from torchspin import SpinSystem


# ---------------------------------------------------------------------------
# MOLGAS constant
# ---------------------------------------------------------------------------

class TestMOLGAS:
    def test_value(self):
        """R = N_A * k_B ≈ 8.314 J/(mol·K)."""
        assert abs(MOLGAS - 8.314462618) < 1e-6

    def test_definition(self):
        assert MOLGAS == AVOGADRO * BOLTZMANN


# ---------------------------------------------------------------------------
# eprconvert
# ---------------------------------------------------------------------------

class TestEprconvert:
    def test_compute_field(self):
        """Compute field from freq and g."""
        result = eprconvert(freq=9.5, g=GFREE)
        # B = h*freq/(g*mu_B) in mT
        expected = PLANCK * 9.5e9 / (GFREE * BMAGN) * 1e3
        assert abs(result['field'] - expected) < 1e-6
        assert result['freq'] == 9.5
        assert result['g'] == GFREE

    def test_compute_freq(self):
        """Compute frequency from field and g."""
        result = eprconvert(field=340.0, g=2.0)
        expected = 2.0 * BMAGN * 340e-3 / PLANCK / 1e9
        assert abs(result['freq'] - expected) < 1e-6

    def test_compute_g(self):
        """Compute g from field and freq."""
        result = eprconvert(field=340.0, freq=9.5)
        expected = PLANCK * 9.5e9 / (BMAGN * 340e-3)
        assert abs(result['g'] - expected) < 1e-6

    def test_round_trip(self):
        """Compute field, then recover g."""
        r1 = eprconvert(freq=9.5, g=2.0023)
        r2 = eprconvert(field=r1['field'], freq=9.5)
        assert abs(r2['g'] - 2.0023) < 1e-10

    def test_error_zero_args(self):
        with pytest.raises(ValueError, match="exactly two"):
            eprconvert()

    def test_error_one_arg(self):
        with pytest.raises(ValueError, match="exactly two"):
            eprconvert(freq=9.5)

    def test_error_three_args(self):
        with pytest.raises(ValueError, match="exactly two"):
            eprconvert(field=340.0, freq=9.5, g=2.0)


# ---------------------------------------------------------------------------
# SpinSystem.validate()
# ---------------------------------------------------------------------------

class TestValidate:
    def test_valid_system_passes(self):
        """A well-formed spin system should pass validation."""
        sys = SpinSystem(S=[0.5], g=[[2.009, 2.006, 2.002]],
                         Nucs=['14N'], A=[[10, 10, 95]])
        sys.validate()  # Should not raise

    def test_invalid_spin(self):
        sys = SpinSystem(S=[0.3])
        with pytest.raises(ValueError, match="multiple of 1/2"):
            sys.validate()

    def test_zero_spin(self):
        sys = SpinSystem(S=[0.0])
        with pytest.raises(ValueError, match="positive"):
            sys.validate()

    def test_negative_spin(self):
        sys = SpinSystem(S=[-0.5])
        with pytest.raises(ValueError, match="positive"):
            sys.validate()

    def test_g_wrong_rows(self):
        """g tensor with wrong number of rows."""
        sys = SpinSystem(S=[0.5, 0.5])
        # Force bad g shape after construction
        import torch
        sys.g = torch.zeros(1, 3)  # 1 row for 2 electrons
        with pytest.raises(ValueError, match="g tensor"):
            sys.validate()

    def test_D_nonzero_for_S_half(self):
        """D tensor nonzero for S=1/2 should warn."""
        sys = SpinSystem(S=[0.5], D=[[100, -50, -50]])
        with pytest.raises(ValueError, match="ZFS requires S >= 1"):
            sys.validate()

    def test_D_valid_for_S1(self):
        """D tensor for S=1 is valid."""
        sys = SpinSystem(S=[1.0], D=[[100, -50, -50]])
        sys.validate()

    def test_Q_nonzero_for_spin_half_nucleus(self):
        """Q for I=1/2 (1H) should raise."""
        sys = SpinSystem(S=[0.5], Nucs=['1H'], A=[[10, 10, 10]],
                         Q=[[1.0, -0.5, -0.5]])
        with pytest.raises(ValueError, match="quadrupole"):
            sys.validate()

    def test_unknown_nucleus(self):
        """Unknown nucleus string should raise."""
        sys = SpinSystem(S=[0.5], Nucs=['99Zz'], A=[[10, 10, 10]])
        with pytest.raises(ValueError, match="Unknown nucleus"):
            sys.validate()

    def test_nuclei_without_A(self):
        """Nuclei specified but no A tensor."""
        sys = SpinSystem(S=[0.5], Nucs=['14N'], A=None)
        # A gets set to zero in __post_init__, so manually remove it
        sys.A = None
        with pytest.raises(ValueError, match="hyperfine"):
            sys.validate()

    def test_negative_tcorr(self):
        sys = SpinSystem(S=[0.5], tcorr=-1e-9)
        with pytest.raises(ValueError, match="tcorr"):
            sys.validate()

    def test_negative_T1(self):
        sys = SpinSystem(S=[0.5], T1=-1.0)
        with pytest.raises(ValueError, match="T1"):
            sys.validate()

    def test_negative_T2(self):
        sys = SpinSystem(S=[0.5], T2=-1.0)
        with pytest.raises(ValueError, match="T2"):
            sys.validate()

    def test_two_electrons_passes(self):
        """Multi-electron system with ee coupling validates."""
        sys = SpinSystem(S=[0.5, 0.5], ee=[[100, 100, 100]])
        sys.validate()

    def test_oam_validates(self):
        """System with OAM should validate."""
        sys = SpinSystem(S=[0.5], L=[2], gL=[1.0])
        sys.validate()
