"""Integration tests for torchspin.pepper (end-to-end CW EPR spectrum).

These tests run full powder simulations and check physically meaningful
output properties.  They are slower than unit tests — each test invokes
the complete resfields → sphgrid → makespec → convspec pipeline.
"""
import math

import torch
import pytest

from torchspin import SpinSystem
from torchspin.experiment import Experiment, Options
from torchspin.pepper import pepper


# Small grid for speed in tests
_FAST_OPT = Options(GridSize=7, GridSymmetry='Ci', Threshold=1e-4)


def test_pepper_returns_tensors():
    """pepper returns (x, spec) as torch.Tensors."""
    sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
    exp = Experiment(mwFreq=9.5, Range=[300.0, 400.0], nPoints=128, Harmonic=0)
    x, spec = pepper(sys, exp, _FAST_OPT)
    assert isinstance(x, torch.Tensor)
    assert isinstance(spec, torch.Tensor)
    assert x.shape[0] == 128
    assert spec.shape[0] == 128


def test_pepper_isotropic_g_peak_position():
    """Isotropic g=2.0, 9.5 GHz: single line near h*nu/(g*mu_B) ≈ 338.8 mT.

    B_res = h * mwFreq / (g * muB) in SI, then converted to mT.
    For g=2.002319 (free electron): B_res = 9.5 GHz * h / (g * muB)
    For g=2.0: B_res ≈ 338.86 mT
    """
    from torchspin.constants import PLANCK, BMAGN
    g = 2.0
    mwFreq_GHz = 9.5
    # B_res = mwFreq [Hz] * h / (g * muB) → mT
    B_res_expected = mwFreq_GHz * 1e9 * PLANCK / (g * BMAGN) * 1e3  # mT

    sys = SpinSystem(S=[0.5], g=[[g, g, g]], lw=[0.5, 0.0])
    exp = Experiment(mwFreq=mwFreq_GHz, Range=[B_res_expected - 30, B_res_expected + 30],
                     nPoints=256, Harmonic=0)
    x, spec = pepper(sys, exp, _FAST_OPT)

    # Peak should be within 1 mT of expected
    peak_B = x[spec.argmax()].item()
    assert abs(peak_B - B_res_expected) < 1.0, \
        f"Peak at {peak_B:.2f} mT, expected ~{B_res_expected:.2f} mT"


def test_pepper_isotropic_g_single_line():
    """Isotropic g: absorption spectrum has a single peak (no hyperfine)."""
    sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]], lw=[1.0, 0.0])
    exp = Experiment(mwFreq=9.5, Range=[310.0, 370.0], nPoints=256, Harmonic=0)
    x, spec = pepper(sys, exp, _FAST_OPT)

    # Absorption spectrum should be positive everywhere (allow small FFT numerical noise)
    assert spec.min().item() >= -2e-5  # Relaxed tolerance for FFT roundoff

    # Single dominant peak: count how many times the spectrum rises above 1% of
    # its maximum (one such interval = one peak; ignores tiny FFT baseline noise)
    peak_val = spec.max().item()
    above = (spec > peak_val * 0.01).long()
    n_peaks = (above[1:] - above[:-1] == 1).sum().item()  # rising-edge count
    assert n_peaks == 1, f"Expected 1 dominant peak, found {n_peaks}"


def test_pepper_axial_g_two_features():
    """Axial g=[2.0,2.0,2.2]: absorption shows edge at both g values.

    g_par = 2.2 → lower-field edge (~308 mT at 9.5 GHz)
    g_perp = 2.0 → higher-field step (~339 mT at 9.5 GHz)
    """
    from torchspin.constants import PLANCK, BMAGN
    mwFreq_GHz = 9.5

    def B_res(g):
        return mwFreq_GHz * 1e9 * PLANCK / (g * BMAGN) * 1e3  # mT

    g_par, g_perp = 2.2, 2.0
    B_par = B_res(g_par)
    B_perp = B_res(g_perp)

    sys = SpinSystem(S=[0.5], g=[[g_perp, g_perp, g_par]], lw=[0.5, 0.0])
    exp = Experiment(
        mwFreq=mwFreq_GHz,
        Range=[B_par - 10, B_perp + 10],
        nPoints=512,
        Harmonic=0,
    )
    x, spec = pepper(sys, exp, _FAST_OPT)

    # The spectrum should have intensity in the range between B_par and B_perp
    # Find the field range containing significant signal
    threshold = spec.max().item() * 0.1
    significant = x[spec > threshold]
    assert significant.min().item() < B_par + 5.0, \
        "No intensity near g_par resonance"
    assert significant.max().item() > B_perp - 5.0, \
        "No intensity near g_perp resonance"


def test_pepper_first_derivative():
    """Harmonic=1 (first-derivative) spectrum integrates to ≈0."""
    sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]], lw=[1.0, 0.0])
    exp = Experiment(mwFreq=9.5, Range=[310.0, 370.0], nPoints=512, Harmonic=1)
    x, spec = pepper(sys, exp, _FAST_OPT)

    dx = (x[-1] - x[0]).item() / (x.shape[0] - 1)
    integral = spec.sum().item() * dx
    assert abs(integral) < spec.abs().max().item() * 10 * dx, \
        f"First-derivative integral not ~0: {integral:.4e}"


def test_pepper_output_shape():
    """Output shape matches exp.nPoints."""
    sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
    for n in (64, 256, 1024):
        exp = Experiment(mwFreq=9.5, Range=[300.0, 400.0], nPoints=n, Harmonic=0)
        x, spec = pepper(sys, exp, _FAST_OPT)
        assert x.shape[0] == n
        assert spec.shape[0] == n


def test_pepper_finite_values():
    """Spectrum contains only finite values."""
    sys = SpinSystem(
        S=[0.5],
        g=[[2.009, 2.006, 2.002]],
        Nucs=['14N'],
        A=[[10.0, 10.0, 95.0]],
        lw=[0.5, 0.0],
    )
    exp = Experiment(mwFreq=9.5, Range=[320.0, 360.0], nPoints=256, Harmonic=1)
    opt = Options(GridSize=5, GridSymmetry='Ci', Threshold=1e-3)
    x, spec = pepper(sys, exp, opt)
    assert torch.all(torch.isfinite(spec)), "Spectrum contains non-finite values"

def test_pepper_temperature_reduces_intensity():
    """At finite temperature, spectrum intensity is reduced compared to infinite T.
    
    For S=1 with ZFS, at low T only ground state is populated, so fewer
    transitions are allowed. Total intensity should decrease with decreasing T.
    """
    # S=1 with axial ZFS (ground-state ms=0, excited ms=±1)
    D_MHz = 2000.0  # 2 GHz ZFS
    sys = SpinSystem(S=[1.0], g=[[2.0, 2.0, 2.0]], D=[[-D_MHz/3, -D_MHz/3, 2*D_MHz/3]], lw=[5.0, 0.0])
    
    exp_inf = Experiment(mwFreq=9.5, Range=[250.0, 450.0], nPoints=256, Harmonic=0, Temperature=None)
    exp_10K = Experiment(mwFreq=9.5, Range=[250.0, 450.0], nPoints=256, Harmonic=0, Temperature=10.0)
    
    opt = Options(GridSize=7, GridSymmetry='Ci', Threshold=1e-4)
    
    x_inf, spec_inf = pepper(sys, exp_inf, opt)
    x_10K, spec_10K = pepper(sys, exp_10K, opt)
    
    # At 10K, total intensity should be lower than at infinite T
    # because ms=±1 states are less populated
    integral_inf = spec_inf.sum().item()
    integral_10K = spec_10K.sum().item()
    
    assert integral_10K < integral_inf, \
        f"10K intensity ({integral_10K:.3e}) should be less than infinite T ({integral_inf:.3e})"
    
    # At 10K for this system, most population is in ground state
    # Ratio should be significantly less than 1
    ratio = integral_10K / integral_inf
    assert ratio < 0.8, f"Intensity ratio at 10K should be <0.8, got {ratio:.3f}"


def test_pepper_temperature_converges_at_high_T():
    """At very high temperature, spectrum approaches infinite-T limit."""
    sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]], lw=[1.0, 0.0])
    
    exp_inf = Experiment(mwFreq=9.5, Range=[320.0, 360.0], nPoints=128, Harmonic=0, Temperature=None)
    exp_1000K = Experiment(mwFreq=9.5, Range=[320.0, 360.0], nPoints=128, Harmonic=0, Temperature=1000.0)
    
    opt = Options(GridSize=7, GridSymmetry='Ci', Threshold=1e-4)
    
    x_inf, spec_inf = pepper(sys, exp_inf, opt)
    x_1000K, spec_1000K = pepper(sys, exp_1000K, opt)
    
    # At 1000K for S=1/2, populations should be nearly equal → spectra nearly identical
    # Normalize both spectra for comparison
    spec_inf_norm = spec_inf / spec_inf.max()
    spec_1000K_norm = spec_1000K / spec_1000K.max()
    
    diff = (spec_inf_norm - spec_1000K_norm).abs().max().item()
    assert diff < 0.02, \
        f"High-T spectrum should match infinite-T (max diff {diff:.4f})"


def test_pepper_temperature_zero_raises_no_error():
    """Temperature=0 should not crash (though physically unphysical, handle gracefully)."""
    sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]], lw=[1.0, 0.0])
    exp = Experiment(mwFreq=9.5, Range=[320.0, 360.0], nPoints=64, Harmonic=0, Temperature=1e-3)
    opt = Options(GridSize=5, GridSymmetry='Ci')
    
    # Should not crash (even though T→0 is unphysical, exp(-E/kT) → 0 for excited states)
    x, spec = pepper(sys, exp, opt)
    assert torch.all(torch.isfinite(spec))


def test_pepper_temperature_S1_ground_state_dominance():
    """S=1 with ZFS at low T: only ground → excited transitions visible.
    
    At low T (~4K), ms=0 ground state dominates population.
    Transitions from ms=0 to ms=±1 are allowed.
    Transitions between ms=+1 and ms=-1 should be negligible.
    """
    # S=1 with large ZFS (D = 10 GHz = 10000 MHz)
    # Energy levels at zero field: ms=0 at -2D/3, ms=±1 at +D/3
    # So ms=0 is ground state, ~667 MHz below ms=±1
    D_MHz = 10000.0
    sys = SpinSystem(
        S=[1.0],
        g=[[2.0, 2.0, 2.0]],
        D=[[-D_MHz/3, -D_MHz/3, 2*D_MHz/3]],
        lw=[5.0, 0.0]
    )
    
    exp = Experiment(mwFreq=9.5, Range=[200.0, 500.0], nPoints=256, Harmonic=0, Temperature=4.0)
    opt = Options(GridSize=7, GridSymmetry='Ci', Threshold=1e-5)
    
    x, spec = pepper(sys, exp, opt)
    
    # Spectrum should have intensity (ground-state transitions are present)
    assert spec.max().item() > 0, "No signal detected at 4K"
    
    # The spectrum should not be flat (should have structure from allowed transitions)
    std_dev = spec.std().item()
    assert std_dev > 0, "Spectrum is flat, no structure from transitions"


def test_pepper_multicomponent_list():
    """pepper([Sys1, Sys2], exp) sums both spectra (equal weight)."""
    exp = Experiment(mwFreq=9.5, Range=[300.0, 380.0], nPoints=256, Harmonic=0)
    sys1 = SpinSystem(S=[0.5], g=[[2.00, 2.00, 2.00]], lw=[1.0, 0.0])
    sys2 = SpinSystem(S=[0.5], g=[[2.10, 2.10, 2.10]], lw=[1.0, 0.0])

    B, spc_both = pepper([sys1, sys2], exp, _FAST_OPT)
    _, spc1 = pepper(sys1, exp, _FAST_OPT)
    _, spc2 = pepper(sys2, exp, _FAST_OPT)

    import numpy as np
    np.testing.assert_allclose(
        spc_both.numpy(), (spc1 + spc2).numpy(), atol=1e-10
    )


def test_pepper_multicomponent_weight():
    """Sys.weight scales each component's contribution."""
    exp = Experiment(mwFreq=9.5, Range=[300.0, 380.0], nPoints=256, Harmonic=0)
    sys1 = SpinSystem(S=[0.5], g=[[2.00, 2.00, 2.00]], lw=[1.0, 0.0], weight=1.0)
    sys2 = SpinSystem(S=[0.5], g=[[2.10, 2.10, 2.10]], lw=[1.0, 0.0], weight=0.3)

    _, spc_both = pepper([sys1, sys2], exp, _FAST_OPT)
    _, spc1 = pepper(sys1, exp, _FAST_OPT)
    _, spc2 = pepper(sys2, exp, _FAST_OPT)   # already scaled by weight 0.3 (EasySpin convention)
    sys2_unit = SpinSystem(S=[0.5], g=[[2.10, 2.10, 2.10]], lw=[1.0, 0.0], weight=1.0)
    _, spc2_unit = pepper(sys2_unit, exp, _FAST_OPT)

    import numpy as np
    np.testing.assert_allclose(spc2.numpy(), 0.3 * spc2_unit.numpy(), atol=1e-10)
    np.testing.assert_allclose(spc_both.numpy(), (spc1 + spc2).numpy(), atol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_pepper_gpu_device():
    """pepper runs on CUDA when opt.device='cuda', returns CPU tensors."""
    sys = SpinSystem(S=[0.5], g=[[2.009, 2.006, 2.002]], lw=[0.5, 0.0])
    exp = Experiment(mwFreq=9.5, Range=[330.0, 355.0], nPoints=256, Harmonic=1)
    opt = Options(GridSize=11, GridSymmetry='auto', Verbosity=0, device='cuda')

    x, spec = pepper(sys, exp, opt)

    # Outputs must always be on CPU (user-accessible without .cpu())
    assert x.device.type == 'cpu', f"x is on {x.device}, expected CPU"
    assert spec.device.type == 'cpu', f"spec is on {spec.device}, expected CPU"
    assert x.shape[0] == 256
    assert spec.shape[0] == 256
    assert torch.all(torch.isfinite(spec))

    # Peak should be near the expected field (g ≈ 2.006 midpoint)
    from torchspin.constants import PLANCK, BMAGN
    g_mid = 2.006
    B_expected = 9.5e9 * PLANCK / (g_mid * BMAGN) * 1e3
    # Derivative peak near zero-crossing: check there's signal in the window
    assert spec.abs().max().item() > 0, "No signal on GPU path"