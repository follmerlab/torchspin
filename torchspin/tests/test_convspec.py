"""Tests for torchspin.convspec."""
import math

import torch

from torchspin.convspec import convspec


def _gaussian_fwhm(spec: torch.Tensor, dx: float) -> float:
    """Estimate FWHM of a Gaussian peak in a 1-D spectrum."""
    peak = spec.max().item()
    half = peak / 2.0
    n = spec.shape[0]
    # Find left and right half-max crossings by linear interpolation
    left = right = None
    for k in range(n - 1):
        if spec[k].item() <= half <= spec[k + 1].item():
            t = (half - spec[k].item()) / (spec[k + 1] - spec[k]).item()
            left = (k + t) * dx
        if spec[k].item() >= half >= spec[k + 1].item():
            t = (spec[k].item() - half) / (spec[k] - spec[k + 1]).item()
            right = (k + t) * dx
    if left is None or right is None:
        return float('nan')
    return right - left


def test_convspec_passthrough_no_broadening():
    """Zero linewidths with no derivative returns spec unchanged."""
    spec = torch.zeros(256, dtype=torch.float64)
    spec[128] = 1.0
    out = convspec(spec, dx=1.0, fwhm_g=0.0, fwhm_l=0.0, deriv=0)
    # Should be essentially unchanged (up to numerical noise)
    assert (out - spec).abs().max().item() < 1e-10


def test_convspec_gaussian_fwhm():
    """Delta input convolved with Gaussian gives correct FWHM."""
    N = 1024
    dx = 0.1  # mT
    fwhm_target = 2.0  # mT

    spec = torch.zeros(N, dtype=torch.float64)
    spec[N // 2] = 1.0

    out = convspec(spec, dx=dx, fwhm_g=fwhm_target, fwhm_l=0.0, deriv=0)

    fwhm_meas = _gaussian_fwhm(out, dx)
    assert not math.isnan(fwhm_meas), "Could not measure FWHM — peak not found."
    rel_err = abs(fwhm_meas - fwhm_target) / fwhm_target
    assert rel_err < 0.05, f"Gaussian FWHM: expected {fwhm_target}, got {fwhm_meas:.3f}"


def test_convspec_integral_preserved_absorption():
    """Gaussian convolution (deriv=0) preserves spectral integral."""
    N = 512
    dx = 0.5  # mT
    spec = torch.zeros(N, dtype=torch.float64)
    spec[N // 3] = 2.0
    spec[2 * N // 3] = 3.0

    integral_before = spec.sum().item() * dx
    out = convspec(spec, dx=dx, fwhm_g=3.0, fwhm_l=0.0, deriv=0)
    integral_after = out.sum().item() * dx

    rel_err = abs(integral_after - integral_before) / abs(integral_before)
    assert rel_err < 0.02, f"Integral not preserved: {integral_before:.4f} → {integral_after:.4f}"


def test_convspec_first_derivative_integrates_to_zero():
    """First-derivative convolution of a positive peak integrates to ≈0."""
    N = 1024
    dx = 0.1
    spec = torch.zeros(N, dtype=torch.float64)
    spec[N // 2] = 1.0

    out = convspec(spec, dx=dx, fwhm_g=3.0, fwhm_l=0.0, deriv=1)
    # First derivative of Gaussian integrates to 0
    integral = out.sum().item() * dx
    assert abs(integral) < 1e-4, f"First-derivative integral = {integral:.2e} (expected ~0)"


def test_convspec_first_derivative_antisymmetric():
    """First derivative of a symmetric Gaussian peak is qualitatively antisymmetric.

    The FFT zero-padding + trimming approach introduces modest edge asymmetry
    (typically ~5-10% relative), so we only verify the qualitative shape:
    the first derivative is negative before the center and positive after.
    """
    N = 1024
    dx = 0.1
    spec = torch.zeros(N, dtype=torch.float64)
    spec[N // 2] = 1.0

    out = convspec(spec, dx=dx, fwhm_g=5.0, fwhm_l=0.0, deriv=1)

    # The first derivative of a positive absorption peak goes positive-to-negative:
    # - Rising slope (left of peak): d/dx > 0 → positive lobe
    # - Falling slope (right of peak): d/dx < 0 → negative lobe
    center = N // 2
    sigma_pts = int(5.0 / (2.355 * dx))  # ~21 points
    left_region = out[center - 2 * sigma_pts: center - sigma_pts]
    right_region = out[center + sigma_pts: center + 2 * sigma_pts]
    assert left_region.min().item() > 0, "Left wing of first-derivative should be positive"
    assert right_region.max().item() < 0, "Right wing of first-derivative should be negative"
    # Value at center (zero-crossing) should be near zero
    assert abs(out[center].item()) < out.abs().max().item() * 0.1


def test_convspec_lorentzian():
    """Lorentzian broadening produces a positive peak."""
    N = 1024
    dx = 0.1
    spec = torch.zeros(N, dtype=torch.float64)
    spec[N // 2] = 1.0

    out = convspec(spec, dx=dx, fwhm_g=0.0, fwhm_l=2.0, deriv=0)
    assert out.max().item() > 0.0
    assert out.min().item() >= -1e-10  # absorption lineshape is positive


def test_convspec_output_shape():
    """Output has same shape as input."""
    spec = torch.zeros(512, dtype=torch.float64)
    spec[256] = 1.0
    out = convspec(spec, dx=0.5, fwhm_g=2.0)
    assert out.shape == spec.shape


def test_convspec_matlab_normalization_convention():
    """Regression guard for the MATLAB-parity normalization convention.

    Phase 3F audit (2026-04-18) initially claimed Python's convspec output
    was `dx`-times smaller than MATLAB's for the derivative path, suggesting
    it was the root cause of pepper's four empirical correction factors
    (8.48 / 6.29 / 4.27 / 3.806). A direct numerical test (2026-04-19)
    disproved this: both Python and MATLAB use the unit-sum-in-index-space
    convention (peak of output for a unit-amplitude delta input is
    `physical_unit_area_peak × dx`). This test locks that behavior in.
    """
    N = 1001
    dx = 0.1  # mT per bin
    fwhm = 1.0  # mT
    spec = torch.zeros(N, dtype=torch.float64)
    spec[N // 2] = 1.0  # unit-amplitude delta (NOT unit-area)

    # Absorption path
    out = convspec(spec, dx, fwhm_g=fwhm, fwhm_l=0.0, deriv=0)
    sigma = fwhm / math.sqrt(8 * math.log(2))
    expected_physical_peak = 1.0 / (sigma * math.sqrt(2 * math.pi))
    # MATLAB-parity: output peak == physical_peak × dx
    assert abs(out.max().item() - expected_physical_peak * dx) < 1e-6, (
        f"convspec absorption normalization changed: peak={out.max().item():.6f}, "
        f"expected {expected_physical_peak * dx:.6f} = (physical peak) × dx"
    )

    # First derivative path
    out1 = convspec(spec, dx, fwhm_g=fwhm, fwhm_l=0.0, deriv=1)
    expected_deriv_peak = math.exp(-0.5) / (sigma ** 2 * math.sqrt(2 * math.pi))
    assert abs(out1.abs().max().item() - expected_deriv_peak * dx) < 1e-3, (
        f"convspec derivative normalization changed: |peak|={out1.abs().max().item():.6f}, "
        f"expected {expected_deriv_peak * dx:.6f} = (physical derivative peak) × dx"
    )
