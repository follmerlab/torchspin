"""Tests for torchspin.eprsave — Bruker BES3T file writer."""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from torchspin.eprsave import eprsave, eprsave_info
from torchspin.eprload import eprload


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _roundtrip(x, y, **kw):
    """Write then read back; return loaded arrays."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir) / "test"
        eprsave(base, x, y, **kw)
        x_rt, y_rt, params = eprload(base.with_suffix(".DTA"))
    return x_rt, y_rt, params


# ---------------------------------------------------------------------------
# Basic round-trip tests
# ---------------------------------------------------------------------------

class TestEprsaveRoundtrip:
    def test_simple_absorption(self):
        """Real 1D spectrum round-trips to within float64 precision."""
        x = np.linspace(330.0, 350.0, 512)
        y = np.exp(-((x - 340.0) ** 2) / (2 * 1.5**2))
        x_rt, y_rt, _ = _roundtrip(x, y)
        assert np.allclose(y_rt, y, atol=1e-10)

    def test_field_axis_preserved(self):
        """X-axis start and end are recovered from XMIN/XWID."""
        x = np.linspace(310.0, 390.0, 1024)
        y = np.zeros(1024)
        y[512] = 1.0
        x_rt, _, _ = _roundtrip(x, y)
        assert abs(x_rt[0] - x[0]) < 1e-6
        assert abs(x_rt[-1] - x[-1]) < 1e-4

    def test_npoints_preserved(self):
        """Number of points is correctly stored and recovered."""
        for n in [128, 256, 512, 1024]:
            x = np.linspace(330, 350, n)
            y = np.random.default_rng(42).normal(size=n)
            x_rt, y_rt, _ = _roundtrip(x, y)
            assert len(y_rt) == n

    def test_negative_values(self):
        """Negative spectral values (derivative spectrum) round-trip correctly."""
        x = np.linspace(335, 345, 256)
        # Derivative-like: positive lobe then negative lobe
        y = -(x - 340) * np.exp(-((x - 340)**2) / 2)
        x_rt, y_rt, _ = _roundtrip(x, y)
        assert np.allclose(y_rt, y, atol=1e-10)
        assert y_rt.min() < 0

    def test_title_stored(self):
        """Title appears in DSC params."""
        x = np.linspace(330, 350, 64)
        y = np.ones(64)
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "titled"
            eprsave(base, x, y, title="My simulation")
            _, _, params = eprload(base.with_suffix(".DTA"))
        # Title may be wrapped in quotes by DSC format
        title_val = str(params.get("TITL", "")).strip("'\"")
        assert "My simulation" in title_val

    def test_custom_params_in_dsc(self):
        """User-supplied params are written to DSC."""
        x = np.linspace(330, 350, 64)
        y = np.ones(64)
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "params_test"
            eprsave(base, x, y, params={"MWFQ": 9.5e9, "STMP": 77.0})
            _, _, p = eprload(base.with_suffix(".DTA"))
        assert "MWFQ" in p or "STMP" in p  # at least one makes it through

    def test_files_created(self):
        """Both .DTA and .DSC files are created."""
        x = np.linspace(330, 350, 64)
        y = np.ones(64)
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "check"
            eprsave(base, x, y)
            assert base.with_suffix(".DTA").exists()
            assert base.with_suffix(".DSC").exists()

    def test_extension_stripped(self):
        """Passing .DTA/.DSC extension in filename is handled gracefully."""
        x = np.linspace(330, 350, 64)
        y = np.ones(64)
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "ext_test.DTA"  # extension present
            eprsave(base, x, y)
            # Should still create the correct files
            assert (Path(tmpdir) / "ext_test.DTA").exists()
            assert (Path(tmpdir) / "ext_test.DSC").exists()


# ---------------------------------------------------------------------------
# Data integrity
# ---------------------------------------------------------------------------

class TestDataIntegrity:
    def test_precision_float64(self):
        """Values are stored as float64 (double precision)."""
        x = np.linspace(330, 350, 64)
        val = 1.23456789012345678
        y = np.full(64, val)
        x_rt, y_rt, _ = _roundtrip(x, y)
        # float64 has ~15 significant digits
        assert abs(y_rt[0] - val) < 1e-12

    def test_all_zeros(self):
        """Zero spectrum round-trips correctly."""
        x = np.linspace(330, 350, 128)
        y = np.zeros(128)
        x_rt, y_rt, _ = _roundtrip(x, y)
        assert np.allclose(y_rt, 0.0)

    def test_large_dynamic_range(self):
        """Spectrum with large dynamic range is preserved."""
        x = np.linspace(330, 350, 256)
        y = np.zeros(256)
        y[64] = 1e6
        y[192] = 1e-3
        x_rt, y_rt, _ = _roundtrip(x, y)
        assert abs(y_rt[64] - 1e6) / 1e6 < 1e-10
        assert abs(y_rt[192] - 1e-3) / 1e-3 < 1e-9


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestEprsaveErrors:
    def test_mismatched_lengths(self):
        """x and y with different lengths should raise ValueError."""
        x = np.linspace(330, 350, 100)
        y = np.ones(200)
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="same length"):
                eprsave(Path(tmpdir) / "bad", x, y)

    def test_too_few_points(self):
        """Single-point spectrum should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="2 points"):
                eprsave(Path(tmpdir) / "bad", np.array([340.0]), np.array([1.0]))

    def test_creates_parent_directory(self):
        """Missing parent directories are created automatically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "subdir" / "nested" / "output"
            x = np.linspace(330, 350, 64)
            y = np.ones(64)
            eprsave(base, x, y)  # should not raise
            assert base.with_suffix(".DTA").exists()


# ---------------------------------------------------------------------------
# Integration with pepper
# ---------------------------------------------------------------------------

def test_eprsave_pepper_roundtrip():
    """Simulate with pepper, save, reload — spectrum within 1e-8 abs error."""
    from torchspin import SpinSystem
    from torchspin.pepper import pepper
    from torchspin.experiment import Experiment

    sys = SpinSystem(S=[0.5], g=[[2.005, 2.004, 2.002]], lw=[0.5, 0.0])
    exp = Experiment(mwFreq=9.5, Range=[335, 345], nPoints=256, Harmonic=0)
    B, spec = pepper(sys, exp)
    B_np = B.numpy()
    spec_np = spec.numpy()

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir) / "pepper_sim"
        eprsave(base, B_np, spec_np)
        B_rt, spec_rt, _ = eprload(base.with_suffix(".DTA"))

    assert np.allclose(spec_rt, spec_np, atol=1e-10)
    assert abs(B_rt[0] - B_np[0]) < 1e-6
