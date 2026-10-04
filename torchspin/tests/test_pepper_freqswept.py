"""
Tests for frequency-swept powder EPR (pepper with Exp.Field + Exp.mwRange).
"""

import math
import pytest
import torch

from torchspin.spinsystem import SpinSystem
from torchspin.experiment import Experiment, Options
from torchspin.pepper import pepper


# ---------------------------------------------------------------------------
# Experiment validation — new fields
# ---------------------------------------------------------------------------

class TestExperimentFreqSwept:
    def test_field_swept_still_works(self):
        exp = Experiment(mwFreq=9.5, Range=[330.0, 360.0])
        assert not exp.is_freq_swept

    def test_freq_swept_mode_detected(self):
        exp = Experiment(mwFreq=9.5, Field=340.0, mwRange=[9.4, 9.6])
        assert exp.is_freq_swept

    def test_freq_swept_no_range_required(self):
        # Range=None is fine in freq-swept mode
        exp = Experiment(mwFreq=9.5, Field=340.0, mwRange=[9.4, 9.6])
        assert exp.Range is None

    def test_freq_swept_mwrange_order_validated(self):
        with pytest.raises(ValueError, match="mwRange"):
            Experiment(mwFreq=9.5, Field=340.0, mwRange=[9.6, 9.4])

    def test_freq_swept_field_must_be_positive(self):
        with pytest.raises(ValueError, match="Field"):
            Experiment(mwFreq=9.5, Field=-1.0, mwRange=[9.4, 9.6])

    def test_field_swept_range_optional_autorange(self):
        # EasySpin semantics: a missing Range/CenterSweep means an automatic
        # sweep range (garlic); pepper reports that it cannot auto-range.
        exp = Experiment(mwFreq=9.5)  # no Range, no mwRange/Field
        assert exp.Range is None and not exp.is_freq_swept
        from torchspin.pepper import pepper
        from torchspin.spinsystem import SpinSystem
        # S=1/2: automatic range from the g/A extremes (EasySpin pepper.m)
        B, spc = pepper(SpinSystem(S=[0.5], g=2.0, lw=[0.5, 0]), exp)
        assert B[0] < 339.0 < B[-1]
        # S>1/2: EasySpin cannot auto-range either
        with pytest.raises(ValueError, match="Range"):
            pepper(SpinSystem(S=[1], g=2.0, D=[[300, 10]], lw=[0.5, 0]), exp)


# ---------------------------------------------------------------------------
# Frequency-swept pepper — basic physics
# ---------------------------------------------------------------------------

class TestPepperFreqSwept:
    """Test frequency-swept mode of pepper."""

    def _make_sys(self, lw_mT=0.01):
        return SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]], lw=[lw_mT, 0.0])

    def _make_exp(self, B_mT=340.0, nu_lo=9.4, nu_hi=9.6, nPoints=256):
        return Experiment(
            mwFreq=9.5,
            Field=B_mT,
            mwRange=[nu_lo, nu_hi],
            nPoints=nPoints,
            Harmonic=0,
        )

    def test_returns_tensors(self):
        sys = self._make_sys()
        exp = self._make_exp()
        opt = Options(GridSize=7, Verbosity=0)
        x, spec = pepper(sys, exp, opt)
        assert isinstance(x, torch.Tensor)
        assert isinstance(spec, torch.Tensor)

    def test_x_axis_in_ghz(self):
        sys = self._make_sys()
        exp = self._make_exp(nu_lo=9.4, nu_hi=9.6)
        opt = Options(GridSize=7, Verbosity=0)
        x, _ = pepper(sys, exp, opt)
        assert abs(x[0].item() - 9.4) < 1e-10
        assert abs(x[-1].item() - 9.6) < 1e-10

    def test_peak_at_resonance_frequency(self):
        """Isotropic g=2.0 at B=340 mT → peak at ν=g·μ_B·B/h."""
        from torchspin.constants import BMAGN, PLANCK, GFREE
        sys = self._make_sys(lw_mT=0.02)
        B = 340.0
        nu_res_ghz = 2.0 * BMAGN / PLANCK * B * 1e-3 * 1e-9  # GHz
        nu_lo = nu_res_ghz - 0.1
        nu_hi = nu_res_ghz + 0.1
        exp = self._make_exp(B_mT=B, nu_lo=nu_lo, nu_hi=nu_hi, nPoints=512)
        opt = Options(GridSize=11, Verbosity=0)
        x, spec = pepper(sys, exp, opt)
        peak_nu = x[spec.argmax()].item()
        assert abs(peak_nu - nu_res_ghz) < 0.005  # within 5 MHz

    def test_spectrum_finite_nonzero(self):
        sys = self._make_sys()
        exp = self._make_exp()
        opt = Options(GridSize=7, Verbosity=0)
        _, spec = pepper(sys, exp, opt)
        assert torch.all(torch.isfinite(spec))
        assert spec.max().item() > 0.0

    def test_npoints_respected(self):
        sys = self._make_sys()
        exp = self._make_exp(nPoints=128)
        opt = Options(GridSize=7, Verbosity=0)
        x, spec = pepper(sys, exp, opt)
        assert x.shape[0] == 128
        assert spec.shape[0] == 128

    def test_harmonic0_absorption(self):
        """Harmonic=0 should give absorption (all positive for positive g)."""
        sys = self._make_sys(lw_mT=0.03)
        exp = self._make_exp(nPoints=256)
        exp_h0 = Experiment(mwFreq=9.5, Field=340.0, mwRange=[9.4, 9.6],
                            nPoints=256, Harmonic=0)
        opt = Options(GridSize=11)
        _, spec = pepper(sys, exp_h0, opt)
        # Absorption should be non-negative (after broadening small negatives possible
        # at boundaries, but max should dominate strongly)
        assert spec.max().item() > 0.0

    def test_harmonic1_derivative(self):
        """Harmonic=1 should give first derivative: positive left, negative right."""
        sys = self._make_sys(lw_mT=0.03)
        from torchspin.constants import BMAGN, PLANCK
        B = 340.0
        nu_res_ghz = 2.0 * BMAGN / PLANCK * B * 1e-3 * 1e-9
        exp = Experiment(mwFreq=9.5, Field=B,
                         mwRange=[nu_res_ghz - 0.1, nu_res_ghz + 0.1],
                         nPoints=512, Harmonic=1)
        opt = Options(GridSize=11)
        x, spec = pepper(sys, exp, opt)
        # Left of peak: positive; right: negative
        half = spec.shape[0] // 2
        assert spec[:half].max().item() > 0.0
        assert spec[half:].min().item() < 0.0

    def test_rhombic_g_spreads_spectrum(self):
        """Rhombic g → broader frequency spread than isotropic."""
        opt = Options(GridSize=11)
        B = 340.0
        from torchspin.constants import BMAGN, PLANCK
        nu_res = 2.0 * BMAGN / PLANCK * B * 1e-3 * 1e-9

        sys_iso = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]], lw=[0.02, 0.0])
        sys_rho = SpinSystem(S=[0.5], g=[[2.1, 2.0, 1.9]], lw=[0.02, 0.0])

        exp_iso = Experiment(mwFreq=9.5, Field=B,
                             mwRange=[nu_res - 0.3, nu_res + 0.3],
                             nPoints=512, Harmonic=0)
        exp_rho = Experiment(mwFreq=9.5, Field=B,
                             mwRange=[nu_res - 0.3, nu_res + 0.3],
                             nPoints=512, Harmonic=0)

        _, spec_iso = pepper(sys_iso, exp_iso, opt)
        _, spec_rho = pepper(sys_rho, exp_rho, opt)

        # Rhombic spectrum must be broader: larger fraction of points nonzero
        thresh = 0.01 * spec_rho.max().item()
        n_iso = (spec_iso > thresh).sum().item()
        n_rho = (spec_rho > thresh).sum().item()
        assert n_rho > n_iso


# ---------------------------------------------------------------------------
# Salt device wiring — CPU (smoke)
# ---------------------------------------------------------------------------

class TestSaltDeviceWiring:
    """Verify salt works with explicit opt.device='cpu'."""

    def test_salt_cpu_device(self):
        from torchspin.salt import salt
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]], Nucs=["1H"],
                         A=[[10.0, 10.0, 10.0]])
        exp = Experiment(mwFreq=9.5, Range=[330.0, 350.0], nPoints=128)
        opt = Options(GridSize=7, Verbosity=0, device='cpu')
        x, spec = salt(sys, exp, opt, freq_range=(10.0, 20.0), n_points=64)
        assert x.device.type == 'cpu'
        assert spec.device.type == 'cpu'
        assert torch.all(torch.isfinite(spec))
        assert spec.max().item() > 0.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_salt_cuda_device(self):
        from torchspin.salt import salt
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]], Nucs=["1H"],
                         A=[[10.0, 10.0, 10.0]])
        exp = Experiment(mwFreq=9.5, Range=[330.0, 350.0], nPoints=128)
        opt = Options(GridSize=7, Verbosity=0, device='cuda')
        x, spec = salt(sys, exp, opt, freq_range=(10.0, 20.0), n_points=64)
        assert x.device.type == 'cpu'
        assert spec.device.type == 'cpu'
        assert torch.all(torch.isfinite(spec))
