"""
Validate pepper CW EPR simulations against MATLAB EasySpin reference data.

Reference data in tests/data/:
  - pepper_rhombiclw.mat   : S=1/2, g=[2, 2.1, 2.2], lw=1 mT (Gaussian)
  - pepper_axiallw.mat     : S=1/2, g=[2, 2, 2.2],  lw=1 mT (Gaussian)
  - pepper_gstrain.mat     : S=1/2, g=[2, 2.1, 2.2], gStrain=[0.01, 0.02, 0.03]
  - pepper_temperature.mat : S=1, ZFS, 7 temperatures [20, 10, 5, 2, 1, 0.5, 0.2] K

The MATLAB scripts that generated these files do not store the field axis x,
so we reconstruct it from the known Exp.Range (1024-point default).

Cosine similarity threshold: ≥ 0.995 (GridSize=50 captures >99.5% of the
spectral shape; the remaining deviation is powder-grid discretisation noise).
"""

from pathlib import Path

import numpy as np
import pytest
import torch

try:
    from scipy.io import loadmat
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from torchspin.spinsystem import SpinSystem
from torchspin.experiment import Experiment, Options
from torchspin.pepper import pepper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent.parent / "tests" / "data"

# GridSize=50 gives cosine > 0.997 for these cases and runs in ~4 s total.
_OPT = Options(GridSize=50, GridSymmetry="Ci", Verbosity=0)
# Axial systems need a higher-density grid to resolve the sharp perpendicular
# turning point; GridSize=100 is still fast for S=1/2.
_OPT_AXIAL = Options(GridSize=100, GridSymmetry="Ci", Verbosity=0)

_COS_THRESHOLD = 0.995


def _require_mat(filename: str):
    if not SCIPY_AVAILABLE:
        pytest.skip("scipy not installed")
    p = DATA_DIR / filename
    if not p.exists():
        pytest.skip(f"Reference file not found: {p}")
    return p


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    an = a / np.abs(a).max()
    bn = b / np.abs(b).max()
    return float(np.dot(an, bn) / (np.linalg.norm(an) * np.linalg.norm(bn)))


# ---------------------------------------------------------------------------
# Test: pepper_rhombiclw
# ---------------------------------------------------------------------------
# From tests/pepper_rhombiclw.m:
#   Sys.S=1/2, Sys.g=[2 2.1 2.2], Sys.lw=1  (Gaussian FWHM 1 mT)
#   Exp.mwFreq=9.5, Exp.Range=[300 350], nPoints=1024 (default), Harmonic=1 (default)

def _find_significant_zero_crossings(x, y, min_prominence=0.05):
    """Find field positions of significant zero-crossings in first-derivative spectrum.

    Filters out edge artifacts and noise crossings by requiring that the
    signal amplitude near the crossing exceeds min_prominence * max(|y|).
    Excludes crossings in the first/last 5% of the field range.
    """
    amp_max = np.max(np.abs(y))
    if amp_max < 1e-30:
        return np.array([])
    threshold = min_prominence * amp_max
    signs = np.sign(y)
    crossings = np.where(np.diff(signs))[0]
    margin = int(0.05 * len(x))  # exclude edge 5%
    positions = []
    for idx in crossings:
        if idx < margin or idx >= len(x) - margin:
            continue  # skip edge crossings
        # Check that signal near crossing is above threshold
        local_amp = max(abs(y[idx]), abs(y[min(idx + 1, len(y) - 1)]))
        if local_amp < threshold:
            continue
        # Linear interpolation for sub-bin accuracy
        x0, x1 = x[idx], x[idx + 1]
        y0, y1 = y[idx], y[idx + 1]
        if y1 != y0:
            positions.append(x0 - y0 * (x1 - x0) / (y1 - y0))
    return np.array(positions)


class TestPepperRhombicLw:
    """Rhombic g-tensor with 1 mT Gaussian linewidth."""

    @pytest.fixture(scope="class")
    def ref(self):
        p = _require_mat("pepper_rhombiclw.mat")
        d = loadmat(str(p), squeeze_me=True, struct_as_record=False)["data"]
        return np.linspace(300.0, 350.0, 1024), d.y1

    @pytest.fixture(scope="class")
    def result(self):
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.1, 2.2]], lw=[1.0, 0.0])
        exp = Experiment(mwFreq=9.5, Range=[300.0, 350.0], nPoints=1024, Harmonic=1)
        return pepper(sys, exp, _OPT)

    def test_cosine_similarity(self, ref, result):
        _, y_ref = ref
        _, y_py = result
        cs = _cosine_similarity(y_py.numpy(), y_ref)
        assert cs >= _COS_THRESHOLD, f"rhombiclw cosine {cs:.4f} < {_COS_THRESHOLD}"

    def test_field_axis(self, ref, result):
        x_ref, _ = ref
        x_py, _ = result
        np.testing.assert_allclose(x_py.numpy(), x_ref, atol=1e-4)

    def test_finite(self, result):
        _, y = result
        assert torch.isfinite(y).all()

    def test_peak_positions_match_matlab(self, ref, result):
        """Zero-crossings (peak positions) should match MATLAB within 0.5 mT."""
        x = np.linspace(300.0, 350.0, 1024)
        _, y_ref = ref
        _, y_py = result
        zc_ref = _find_significant_zero_crossings(x, y_ref)
        zc_py = _find_significant_zero_crossings(x, y_py.numpy())
        # Each MATLAB zero-crossing should have a matching Python one
        for b_ref in zc_ref:
            if len(zc_py) > 0:
                closest = np.min(np.abs(zc_py - b_ref))
                assert closest < 0.5, (
                    f"Peak at {b_ref:.1f} mT: nearest Python peak is {closest:.2f} mT away"
                )

    def test_amplitude_ratio_match_matlab(self, ref, result):
        """Peak-to-peak amplitude ratio should match MATLAB within 15%."""
        _, y_ref = ref
        _, y_py = result
        ratio_ref = float(y_ref.max() / (-y_ref.min())) if y_ref.min() < 0 else 1.0
        y_np = y_py.numpy()
        ratio_py = float(y_np.max() / (-y_np.min())) if y_np.min() < 0 else 1.0
        rel_diff = abs(ratio_py - ratio_ref) / (abs(ratio_ref) + 1e-10)
        assert rel_diff < 0.15, (
            f"Amplitude ratio mismatch: MATLAB={ratio_ref:.3f}, Python={ratio_py:.3f}"
        )


# ---------------------------------------------------------------------------
# Test: pepper_axiallw
# ---------------------------------------------------------------------------
# From tests/pepper_axiallw.m:
#   Sys.S=1/2, Sys.g=[2 2 2.2], Sys.lw=1
#   Exp.mwFreq=9.5, Exp.Range=[300 350]

class TestPepperAxialLw:
    """Axial g-tensor with 1 mT Gaussian linewidth."""

    @pytest.fixture(scope="class")
    def ref(self):
        p = _require_mat("pepper_axiallw.mat")
        d = loadmat(str(p), squeeze_me=True, struct_as_record=False)["data"]
        return np.linspace(300.0, 350.0, 1024), d.spc

    @pytest.fixture(scope="class")
    def result(self):
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.2]], lw=[1.0, 0.0])
        exp = Experiment(mwFreq=9.5, Range=[300.0, 350.0], nPoints=1024, Harmonic=1)
        return pepper(sys, exp, _OPT_AXIAL)

    def test_cosine_similarity(self, ref, result):
        _, y_ref = ref
        _, y_py = result
        cs = _cosine_similarity(y_py.numpy(), y_ref)
        assert cs >= _COS_THRESHOLD, f"axiallw cosine {cs:.4f} < {_COS_THRESHOLD}"

    def test_perpendicular_peak_position(self, ref, result):
        """g_perp=2.0 edge should appear near 339 mT at 9.5 GHz."""
        x_ref, y_ref = ref
        _, y_py = result
        x_py = np.linspace(300.0, 350.0, 1024)
        # Perpendicular turning point: h*nu / (g_perp * muB) ≈ 339 mT
        from torchspin.constants import PLANCK, BMAGN
        B_perp = 9.5e9 * PLANCK / (2.0 * BMAGN) * 1e3
        # Find sign change near B_perp in first-derivative spectrum
        idx = np.argmin(np.abs(x_py - B_perp))
        # Spectrum should have a feature within ±3 mT of B_perp
        window = slice(max(0, idx-30), min(1023, idx+30))
        assert np.abs(y_py.numpy()[window]).max() > 0.05 * np.abs(y_py.numpy()).max()

    def test_finite(self, result):
        _, y = result
        assert torch.isfinite(y).all()


# ---------------------------------------------------------------------------
# Test: pepper_gstrain
# ---------------------------------------------------------------------------
# From tests/pepper_gstrain.m:
#   Sys.S=1/2, Sys.g=[2 2.1 2.2], Sys.gStrain=[1 2 3]*0.01
#   Exp.mwFreq=9.5, Exp.Range=[290 350]  (Harmonic=1 default)

class TestPepperGstrain:
    """Rhombic g-tensor with anisotropic g-strain broadening."""

    @pytest.fixture(scope="class")
    def ref(self):
        p = _require_mat("pepper_gstrain.mat")
        d = loadmat(str(p), squeeze_me=True, struct_as_record=False)["data"]
        return np.linspace(290.0, 350.0, 1024), d.y

    @pytest.fixture(scope="class")
    def result(self):
        sys = SpinSystem(
            S=[0.5],
            g=[[2.0, 2.1, 2.2]],
            gStrain=[0.01, 0.02, 0.03],
            lw=[0.0, 0.0],
        )
        exp = Experiment(mwFreq=9.5, Range=[290.0, 350.0], nPoints=1024, Harmonic=1)
        return pepper(sys, exp, _OPT)

    def test_cosine_similarity(self, ref, result):
        _, y_ref = ref
        _, y_py = result
        cs = _cosine_similarity(y_py.numpy(), y_ref)
        assert cs >= _COS_THRESHOLD, f"gstrain cosine {cs:.4f} < {_COS_THRESHOLD}"

    def test_finite(self, result):
        _, y = result
        assert torch.isfinite(y).all()

    def test_peak_positions_match_matlab(self, ref, result):
        """Zero-crossings should match MATLAB within 0.5 mT."""
        x = np.linspace(290.0, 350.0, 1024)
        _, y_ref = ref
        _, y_py = result
        zc_ref = _find_significant_zero_crossings(x, y_ref)
        zc_py = _find_significant_zero_crossings(x, y_py.numpy())
        for b_ref in zc_ref:
            if len(zc_py) > 0:
                closest = np.min(np.abs(zc_py - b_ref))
                assert closest < 0.5, (
                    f"gstrain peak at {b_ref:.1f} mT: nearest Python peak is {closest:.2f} mT away"
                )


# ---------------------------------------------------------------------------
# Test: pepper_gstrain_sband — validates B4 fix (gStrain at low frequency)
# ---------------------------------------------------------------------------
# Sys.g = [2.0104 2.0074 2.0026], gStrain = [0.001 0.0008 0.0005]
# Exp.mwFreq = 3.0 GHz (S-band), Range = [102 112], Harmonic = 0
# Generated by /tmp/generate_gstrain_multifreq.m

class TestPepperGstrainSband:
    """gStrain at S-band (3 GHz) — regression test for B4 low-frequency issue."""

    @pytest.fixture(scope="class")
    def ref(self):
        p = _require_mat("pepper_gstrain_sband.mat")
        d = loadmat(str(p), squeeze_me=True)
        return d["B"], d["spc"]

    @pytest.fixture(scope="class")
    def result(self):
        sys = SpinSystem(
            S=[0.5],
            g=[[2.0104, 2.0074, 2.0026]],
            gStrain=[[0.001, 0.0008, 0.0005]],
            lw=[0.0, 0.0],
        )
        exp = Experiment(mwFreq=3.0, Range=[102.0, 112.0], nPoints=1024, Harmonic=0)
        opt = Options(GridSize=31, Verbosity=0)
        return pepper(sys, exp, opt)

    def test_cosine_similarity(self, ref, result):
        B_ref, y_ref = ref
        _, y_py = result
        cs = _cosine_similarity(y_py.numpy(), y_ref)
        assert cs >= 0.999, f"S-band gstrain cosine {cs:.4f} < 0.999"

    def test_finite(self, result):
        _, y = result
        assert torch.isfinite(y).all()

    def test_snr(self, result):
        """Verify spectrum is well above numerical noise."""
        _, y = result
        spc = y.numpy()
        sorted_abs = np.sort(np.abs(spc))
        noise_floor = np.percentile(sorted_abs, 25)
        if noise_floor > 0:
            snr = spc.max() / noise_floor
            assert snr > 10, f"S-band gStrain SNR too low: {snr:.1f}"


# ---------------------------------------------------------------------------
# Test: pepper_temperature
# ---------------------------------------------------------------------------
# From tests/pepper_temperature.m:
#   Sys.S=1, Sys.g=[2 2 2], Sys.D=200*[1 1 -2] MHz, Sys.lw=1
#   Exp.Range=[300 380], mwFreq=9.5, Harmonic=0
#   Temps = [20 10 5 2 1 0.5 0.2] K   (7 rows in data.y)

_TEMPS = [20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2]


@pytest.fixture(scope="module")
def _temperature_ref():
    p = _require_mat("pepper_temperature.mat")
    d = loadmat(str(p), squeeze_me=True, struct_as_record=False)["data"]
    return d.y  # shape (7, 1024)


def _temperature_result(T):
    sys = SpinSystem(
        S=[1.0],
        g=[[2.0, 2.0, 2.0]],
        D=[200.0, 200.0, -400.0],  # traceless; matches MATLAB's D=200*[1 1 -2]
        lw=[1.0, 0.0],
    )
    exp = Experiment(
        mwFreq=9.5, Range=[300.0, 380.0], nPoints=1024, Harmonic=0, Temperature=T
    )
    return pepper(sys, exp, _OPT)


@pytest.mark.parametrize("i,T", list(enumerate(_TEMPS)))
def test_pepper_temperature_cosine(i, T, _temperature_ref):
    """Spectrum at each temperature must match MATLAB within cosine ≥ 0.995."""
    y_ref = _temperature_ref[i]
    _, y_py = _temperature_result(T)
    cs = _cosine_similarity(y_py.numpy(), y_ref)
    assert cs >= _COS_THRESHOLD, (
        f"Temperature {T} K: cosine {cs:.4f} < {_COS_THRESHOLD}"
    )


def test_pepper_temperature_ordering(_temperature_ref):
    """Lower T → stronger EPR signal for this system (ms=±1 ground state).

    With D=[200, 200, -400] MHz the ms=±1 doublet lies lower in energy.
    As T decreases, more population concentrates in the ground state, so
    the population difference for the ground-state EPR transitions grows
    and the total EPR signal increases.
    """
    _, y_20K = _temperature_result(20.0)
    _, y_02K = _temperature_result(0.2)
    # Cold (0.2 K) signal > warm (20 K) signal for this ground-state doublet system
    assert y_02K.abs().max() > y_20K.abs().max(), (
        "EPR signal at 0.2K should exceed 20K for S=1 with ms=±1 ground state"
    )


def test_pepper_temperature_monotonic(_temperature_ref):
    """Signal amplitude should increase monotonically as T decreases."""
    amplitudes = []
    for T in _TEMPS:
        _, y = _temperature_result(T)
        amplitudes.append(float(y.abs().max()))
    # _TEMPS = [20, 10, 5, 2, 1, 0.5, 0.2] (decreasing T → increasing signal)
    for i in range(len(amplitudes) - 1):
        assert amplitudes[i] <= amplitudes[i + 1] * 1.01, (
            f"Non-monotonic: T={_TEMPS[i]}K amp={amplitudes[i]:.4e} > "
            f"T={_TEMPS[i+1]}K amp={amplitudes[i+1]:.4e}"
        )


def test_pepper_temperature_finite():
    _, y = _temperature_result(5.0)
    assert torch.isfinite(y).all()
