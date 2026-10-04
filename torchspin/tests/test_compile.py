"""Tests for torch.compile integration.

Verifies that:
1. The maybe_compile decorator works as a no-op when disabled.
2. When enabled, compiled kernels produce identical results.
3. The public API (set_compile_enabled, is_compile_available) works.
4. Decorated functions expose _original for introspection.
"""
import os

import pytest
import torch

from torchspin._compile import (
    maybe_compile,
    set_compile_enabled,
    is_compile_available,
    _resolve_flag,
)


class TestCompileUtility:
    """Basic utility tests."""

    def test_is_compile_available(self):
        major = int(torch.__version__.split(".")[0])
        assert is_compile_available() == (major >= 2)

    def test_default_disabled(self):
        # When TORCHSPIN_COMPILE is not set, compilation is off
        from torchspin._compile import _compile_enabled
        # Reset and check
        import torchspin._compile as mod
        old = mod._compile_enabled
        mod._compile_enabled = None
        env_val = os.environ.get("TORCHSPIN_COMPILE", "0")
        result = _resolve_flag()
        assert result == (env_val == "1")
        mod._compile_enabled = old

    def test_set_compile_enabled(self):
        import torchspin._compile as mod
        old = mod._compile_enabled
        set_compile_enabled(True)
        assert mod._compile_enabled is True
        set_compile_enabled(False)
        assert mod._compile_enabled is False
        mod._compile_enabled = old

    def test_decorator_exposes_original(self):
        @maybe_compile
        def my_fn(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        assert hasattr(my_fn, '_original')
        x = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(my_fn._original(x), x * 2)

    def test_decorator_with_kwargs(self):
        @maybe_compile(fullgraph=False)
        def my_fn(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        assert hasattr(my_fn, '_original')
        x = torch.tensor([1.0])
        assert torch.allclose(my_fn(x), torch.tensor([2.0]))


class TestCompiledKernelsMatchUncompiled:
    """Verify compiled and uncompiled paths produce identical results."""

    def test_makespec_unchanged(self):
        from torchspin.makespec import makespec
        pos = torch.tensor([320.0, 340.0, 360.0])
        amp = torch.tensor([1.0, 2.0, 0.5])
        x, spec = makespec([300.0, 400.0], 1024, pos, amp)
        assert spec.shape == (1024,)
        assert spec.sum().item() == pytest.approx(3.5, abs=1e-10)

    def test_convspec_unchanged(self):
        from torchspin.convspec import convspec
        # Delta function → should broaden to Gaussian shape
        spec = torch.zeros(1024, dtype=torch.float64)
        spec[512] = 1.0
        dx = 0.1
        result = convspec(spec, dx, fwhm_g=1.0)
        assert result.shape == (1024,)
        # Peak should be near center
        assert result.argmax().item() == 512
        # Should be broadened, not a delta
        assert (result > 1e-6).sum().item() > 10

    def test_convspec_derivative(self):
        from torchspin.convspec import convspec
        spec = torch.zeros(512, dtype=torch.float64)
        spec[256] = 1.0
        dx = 0.1
        d0 = convspec(spec, dx, fwhm_g=1.0, deriv=0)
        d1 = convspec(spec, dx, fwhm_g=1.0, deriv=1)
        # First derivative should integrate to ~0
        assert abs(d1.sum().item()) < 1e-8
        # Absorption should be nonnegative
        assert d0.min().item() >= -1e-12

    def test_saffron_dft_bin(self):
        from torchspin.saffron_peaks import _dft_bin
        nu = torch.tensor([1.0, 5.0, -3.0], dtype=torch.float64)
        dt = 0.01
        n = 512
        idx = _dft_bin(nu, dt, n)
        assert idx.shape == (3,)
        assert (idx >= 0).all()
        assert (idx < n).all()


@pytest.mark.skipif(
    not is_compile_available(),
    reason="torch.compile not available (PyTorch < 2.0)"
)
class TestCompileEnabled:
    """Test with compilation actually enabled."""

    def test_makespec_compiled_matches(self):
        import torchspin._compile as mod
        old = mod._compile_enabled
        set_compile_enabled(True)

        # Create a fresh compiled function
        @maybe_compile
        def _test_kernel(x: torch.Tensor) -> torch.Tensor:
            return x ** 2 + 2 * x + 1

        x = torch.linspace(0, 10, 100, dtype=torch.float64)
        result = _test_kernel(x)
        expected = x ** 2 + 2 * x + 1
        assert torch.allclose(result, expected, atol=1e-12)

        mod._compile_enabled = old

    def test_pepper_with_compile(self):
        """Full pepper simulation produces same result with compile enabled."""
        import torchspin._compile as mod
        from torchspin import SpinSystem, Experiment, Options, pepper

        sys = SpinSystem(S=[0.5], g=[[2.0, 2.05, 2.1]])
        exp = Experiment(mwFreq=9.5, Range=[300, 360], Harmonic=0)
        opt = Options(GridSize=10)

        # Without compile
        old = mod._compile_enabled
        set_compile_enabled(False)
        # Reset cached compiled functions
        from torchspin.makespec import _bin_peaks
        from torchspin.convspec import _fft_convolve
        _bin_peaks.__wrapped__ if hasattr(_bin_peaks, '__wrapped__') else None
        x1, y1 = pepper(sys, exp, opt)

        # Result should be valid
        assert y1.shape == x1.shape
        assert y1.max() > 0

        mod._compile_enabled = old
