"""
Test basic garlic functionality (isotropic mode, no fast-motion).

Tests garlic with tcorr=0 (isotropic/rigid-limit spectra).
"""

import pytest
import torch
import numpy as np

from torchspin.spinsystem import SpinSystem
from torchspin.experiment import Experiment, Options
from torchspin.garlic import garlic


def test_garlic_validation_S_half_only():
    """garlic should only accept S=1/2 systems."""
    
    # Valid: S=1/2
    sys = SpinSystem(S=[0.5], g=2.0, lw=[0.1, 0.0])
    exp = Experiment(mwFreq=9.5, Range=[330, 350])
    # Should not raise
    B, spec = garlic(sys, exp)
    
    # Invalid: S=1
    with pytest.raises(ValueError, match="S=1/2"):
        sys_S1 = SpinSystem(S=[1.0], g=2.0, lw=[0.1, 0.0])
        garlic(sys_S1, exp)
    
    # Invalid: Two electrons
    with pytest.raises(ValueError, match="one electron"):
        sys_2e = SpinSystem(S=[0.5, 0.5], g=[2.0, 2.0], lw=[0.1, 0.0])
        garlic(sys_2e, exp)


def test_garlic_isotropic_g_no_nuclei():
    """Isotropic g, no hyperfine: single line."""
    
    sys = SpinSystem(
        S=[0.5],
        g=2.006,
        lw=[0.5, 0.0]  # Gaussian broadening
    )
    
    exp = Experiment(
        mwFreq=9.5,  # GHz
        Range=[330, 350],  # mT
        nPoints=512,
        Harmonic=0  # Absorption
    )
    
    B, spec = garlic(sys, exp)
    
    # Check output shapes
    assert B.shape == (512,)
    assert spec.shape == (512,)
    
    # Check field range
    assert B[0].item() == pytest.approx(330.0, abs=1e-6)
    assert B[-1].item() == pytest.approx(350.0, abs=1e-6)
    
    # Absorption should have single peak
    peak_idx = spec.argmax().item()
    peak_field = B[peak_idx].item()
    
    # Expected resonance field: B = h*nu/(g*mu_B) ≈ 339.0 mT for g=2.006, nu=9.5 GHz
    # Tolerance: 0.5 mT
    assert 338.0 < peak_field < 340.0
    
    # Check spectrum is non-negative (absorption)
    assert spec.min().item() >= -1e-6  # Allow tiny numerical noise


def test_garlic_isotropic_g_single_nucleus():
    """Isotropic g and A: hyperfine triplet (14N)."""
    
    sys = SpinSystem(
        S=[0.5],
        g=2.006,
        Nucs='14N',  # I=1
        A=16.0,  # MHz, isotropic
        lw=[0.3, 0.0]
    )
    
    exp = Experiment(
        mwFreq=9.5,
        Range=[333, 345],
        nPoints=512,
        Harmonic=1  # First derivative
    )
    
    B, spec = garlic(sys, exp)
    
    assert B.shape == (512,)
    assert spec.shape == (512,)
    
    # 14N with I=1 gives 2I+1 = 3 lines (triplet)
    # Check for 3 extrema in derivative spectrum
    # (positive peak, negative peak, positive peak)
    
    # Find zero crossings (peaks in absorption = zero crossings in derivative)
    spec_np = spec.numpy()
    zero_crossings = np.where(np.diff(np.sign(spec_np)))[0]
    
    # Should have at least 3 zero crossings (3 lines)
    assert len(zero_crossings) >= 3, f"Expected 3 hyperfine lines, found {len(zero_crossings)} zero crossings"


def test_garlic_anisotropic_g_averaged():
    """Anisotropic g should be averaged to isotropic value."""
    
    sys = SpinSystem(
        S=[0.5],
        g=[2.002, 2.006, 2.010],  # Anisotropic g
        lw=[0.5, 0.0]
    )
    
    exp = Experiment(
        mwFreq=9.5,
        Range=[330, 350],
        nPoints=256,
        Harmonic=0
    )
    
    B, spec = garlic(sys, exp)
    
    # Expected g_iso = (2.002 + 2.006 + 2.010) / 3 = 2.006
    # Peak should be at same position as test_garlic_isotropic_g_no_nuclei
    peak_field = B[spec.argmax()].item()
    assert 338.0 < peak_field < 340.0


def test_garlic_tcorr_validation():
    """tcorr validation: must be positive, not too small."""
    
    sys_base = SpinSystem(S=[0.5], g=2.006, lw=[0.1, 0.0])
    exp = Experiment(mwFreq=9.5, Range=[330, 350])
    
    # Valid: tcorr = 1e-10 s (100 ps)
    sys_valid = SpinSystem(S=[0.5], g=[2.002, 2.006, 2.010], lw=[0.0, 0.0])
    sys_valid.tcorr = 1e-10
    B, spec = garlic(sys_valid, exp)  # Should not raise
    
    # Invalid: tcorr too small
    with pytest.raises(ValueError, match="too small"):
        sys_tiny = SpinSystem(S=[0.5], g=[2.002, 2.006, 2.010], lw=[0.0, 0.0])
        sys_tiny.tcorr = 1e-14
        garlic(sys_tiny, exp)
    
    # Invalid: tcorr too large (not fast-motion regime)
    with pytest.raises(ValueError, match="too large"):
        sys_large = SpinSystem(S=[0.5], g=[2.002, 2.006, 2.010], lw=[0.0, 0.0])
        sys_large.tcorr = 1e-2
        garlic(sys_large, exp)


def test_garlic_logtcorr():
    """logtcorr should override tcorr."""
    
    exp = Experiment(mwFreq=9.5, Range=[330, 350])
    
    # Use logtcorr instead of tcorr
    sys = SpinSystem(
        S=[0.5],
        g=[2.002, 2.006, 2.010],
        lw=[0.0, 0.0]
    )
    sys.logtcorr = -10  # 1e-10 s
    
    B, spec = garlic(sys, exp)
    # Should compute with tcorr = 1e-10 s
    assert B.shape == (exp.nPoints,)


def test_garlic_unsupported_features():
    """garlic should reject unsupported features."""
    
    exp = Experiment(mwFreq=9.5, Range=[330, 350])
    
    # Unsupported: electron-electron coupling (two electrons)
    # Note: garlic first checks nElectrons==1, so this error comes before ee check
    with pytest.raises(ValueError, match="one electron"):
        sys_ee = SpinSystem(S=[0.5, 0.5], g=[2.0, 2.0], ee=[10, 0, 0], lw=[0.1, 0.0])
        garlic(sys_ee, exp)
    
    # Unsupported: nucleus-nucleus coupling
    # Note: This requires 2+ nuclei
    with pytest.raises(ValueError, match="nucleus-nucleus"):
        sys_nn = SpinSystem(
            S=[0.5],
            g=2.0,
            Nucs='1H,1H',
            A=[10.0, 10.0],
            nn=[1.0],
            lw=[0.1, 0.0]
        )
        garlic(sys_nn, exp)


def test_garlic_output_normalization():
    """Check that spectrum has reasonable amplitude."""
    
    sys = SpinSystem(S=[0.5], g=2.006, lw=[0.5, 0.0])
    exp = Experiment(mwFreq=9.5, Range=[330, 350], Harmonic=0)
    
    B, spec = garlic(sys, exp)
    
    # Absorption spectrum should integrate to ~1 (after normalization)
    dx = (exp.Range[1] - exp.Range[0]) / (exp.nPoints - 1)
    integral = torch.sum(spec) * dx
    
    # Rough check: integral should be positive and not crazy large/small
    assert integral > 0
    assert integral < 1000  # Arbitrary sanity check


def test_garlic_harmonic_absorption_vs_derivative():
    """Harmonic=0 (absorption) vs Harmonic=1 (derivative)."""
    
    sys = SpinSystem(S=[0.5], g=2.006, lw=[0.5, 0.0])
    exp_abs = Experiment(mwFreq=9.5, Range=[330, 350], Harmonic=0)
    exp_der = Experiment(mwFreq=9.5, Range=[330, 350], Harmonic=1)
    
    B, spec_abs = garlic(sys, exp_abs)
    B, spec_der = garlic(sys, exp_der)
    
    # Absorption: should be all positive (single peak)
    assert spec_abs.min() >= -1e-6
    
    # Derivative: should have positive and negative parts
    assert spec_der.max() > 0
    assert spec_der.min() < 0
    
    # Derivative should integrate to ~0
    integral_der = torch.sum(spec_der)
    assert abs(integral_der.item()) < 1e-3  # Close to zero


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
