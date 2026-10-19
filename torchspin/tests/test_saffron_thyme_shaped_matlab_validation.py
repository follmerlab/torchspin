"""Strict MATLAB cross-validation of saffron's real-pulse route with shaped pulses.

The companion suite ``test_saffron_thyme_matlab_validation.py`` validates the
detection stage and the propagation with rectangular pulses. The ten cases
here are transcribed from EasySpin's own ``tests/saffron_thyme_*.m``, four of
which use ``quartersin/linear`` chirps, and cover what rectangular pulses
cannot:

- ``Pulse.Type``, ``Pulse.trise`` and a swept ``Pulse.Frequency``
  (``saffron_thyme_spinecho``),
- relaxation during a chirp (``saffron_thyme_relaxation``),
- an explicit simulation frame far from the automatic one
  (``saffron_thyme_simfreq``, ``Opt.SimFreq = 15`` against
  ``floor(mwFreq-2) = 32``),
- a delay sweep, and a delay sweep crossed with a *pulse parameter* sweep that
  regenerates the chirp waveform at every point
  (``saffron_thyme_indirectdimensions``, ``Exp.Dim2 = {'p1.Flip' pi/3}``),
- ``Sys.gFrame``/``Sys.AFrame`` tensor tilts against an equivalent sample tilt
  (``saffron_thyme_framerotations``),
- a predefined sequence detected over a transient window, single crystal
  (``saffron_thyme_crystalorientations``).

No new physics was needed for these: ``torchspin/pulse.py`` already builds
shaped pulses for ``spidyan``, and a pulse dict in a list-valued
``Exp.Sequence`` passes through to it unchanged. What had kept EasySpin's own
thyme cases from running was the propagation and detection, fixed in the
rectangular round -- the missing pulse carrier, the mixed molecular/lab frame,
the renormalized orientation sum and the absent detection stage.

Every assertion checks the raw *and* the mean-centered relative L2 error,
never a cosine. Every option that affects the result is stated explicitly on
both sides. ``Sys.g`` uses :data:`torchspin.constants.GFREE` where EasySpin's
test uses ``gfree``, never a rounded 2.00232.

References come from ``tests/data/generate_saffron_thyme_shaped_refs.m``. The
CI matrix deselects ``-k "not matlab_validation"``, so these run only in the
full-suite job; ``test_shaped_references_present`` fails rather than skips if
the reference file is missing, so a silent skip cannot look like a pass.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

try:
    from scipy.io import loadmat
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

from torchspin import SpinSystem
from torchspin.constants import GFREE
from torchspin.saffron import saffron, PulseExperiment, SaffronOptions

DATA = (Path(__file__).parent.parent.parent / "tests" / "data"
        / "saffron_thyme_shaped_refs.mat")

# Observed agreement is 3e-13..8e-12 on both the raw and the mean-centered
# signal; these leave ~2 orders of headroom for eigensolver/BLAS ordering
# differences across platforms.
RAW_TOL = 1e-9
CENTERED_TOL = 1e-8

# Integration time steps, as stated in the generator. The chirps sweep
# +-300 MHz about 34.78 GHz (Nyquist time 1.6e-4 us); the two cases without
# chirps run at 9.3 GHz (Nyquist time 2.2e-4 us).
ITS = 2e-5
ITS_RECT = 1e-4

CHIRP90 = {'Type': 'quartersin/linear', 'trise': 0.030, 'tp': 0.200,
           'Flip': np.pi / 2, 'Frequency': [-300, 300]}   # MHz
CHIRP180 = {'Type': 'quartersin/linear', 'trise': 0.030, 'tp': 0.100,
            'Flip': np.pi, 'Frequency': [-300, 300]}
# EasySpin's test centers the window one pi-pulse length after the sequence,
# where a chirp echo refocuses.
Q_WINDOW = [-0.1 + CHIRP180['tp'], 0.1 + CHIRP180['tp']]
ANGLES = [np.pi / 3, np.pi / 5, 0.0]


@pytest.fixture(scope="module")
def refs():
    if not SCIPY_OK:
        pytest.skip("scipy not installed")
    if not DATA.exists():
        pytest.skip(f"reference file not found: {DATA}")
    return loadmat(str(DATA), simplify_cells=True)["data"]


def _errors(y, y_ref) -> tuple[float, float]:
    a = np.atleast_1d(np.asarray(y))
    b = np.atleast_1d(np.asarray(y_ref))
    assert a.shape == b.shape, f"shape {a.shape} != reference {b.shape}"
    a, b = a.ravel(), b.ravel()
    raw = np.linalg.norm(a - b) / np.linalg.norm(b)
    ac, bc = a - a.mean(), b - b.mean()
    if np.linalg.norm(bc) <= 1e-12 * max(abs(b.mean()), 1e-30):
        return float(raw), 0.0
    return float(raw), float(np.linalg.norm(ac - bc) / np.linalg.norm(bc))


def _check(y, ref_case, label: str) -> None:
    raw, centered = _errors(y, ref_case['y'])
    assert raw < RAW_TOL, f"{label}: raw relative error {raw:.3e} >= {RAW_TOL:g}"
    assert centered < CENTERED_TOL, (
        f"{label}: mean-centered (modulation) relative error "
        f"{centered:.3e} >= {CENTERED_TOL:g}")


def _check_axis(x, ref_case, label: str) -> None:
    xr = ref_case['x']
    if isinstance(xr, (list, tuple)) or (isinstance(xr, np.ndarray)
                                         and xr.dtype == object):
        assert isinstance(x, list), f"{label}: expected a list of axes"
        assert len(x) == len(xr), f"{label}: {len(x)} axes, expected {len(xr)}"
        for i, (xi, xri) in enumerate(zip(x, xr)):
            np.testing.assert_allclose(
                np.asarray(xi, dtype=float).ravel(),
                np.asarray(xri, dtype=float).ravel(),
                rtol=0, atol=1e-12, err_msg=f"{label}: axis {i}")
    else:
        np.testing.assert_allclose(
            np.asarray(x, dtype=float).ravel(),
            np.atleast_1d(np.asarray(xr, dtype=float)).ravel(),
            rtol=0, atol=1e-12, err_msg=f"{label}: x axis")


# ---------------------------------------------------------------------------
# Q-band chirp spin echo and its variants
# ---------------------------------------------------------------------------


def sys_q(**kw) -> SpinSystem:
    """EasySpin's Q-band 14N system. Sys.lwpp is ignored by the thyme route."""
    return SpinSystem(S=[0.5], g=[[2.00906, 2.0062, 2.0023]], Nucs=['14N'],
                      A=[[11.5, 11.5, 95.0]], lwpp=[10.0, 0.0], **kw)


def exp_q(**kw) -> PulseExperiment:
    base = dict(Field=1240, mwFreq=34.78,
                Sequence=[CHIRP90, 0.25, CHIRP180, 0.25],
                DetWindow=Q_WINDOW, DetPhase=0.0)
    base.update(kw)
    return PulseExperiment(**base)


def opt_q(**kw) -> SaffronOptions:
    return SaffronOptions(**{'GridSize': 7, 'SimulationMode': 'thyme',
                             'IntTimeStep': ITS, **kw})


def test_shaped_references_present():
    """Fail, not skip, if the reference file is missing."""
    assert DATA.exists(), (
        f"{DATA.name} is missing. Regenerate it with\n"
        "  matlab -nodisplay -batch "
        "\"run('tests/data/generate_saffron_thyme_shaped_refs.m')\"")


def test_shaped_chirp_spin_echo(refs):
    """Chirp pi/2 - tau - pi - tau echo (EasySpin saffron_thyme_spinecho).

    This is the case that needs ``Pulse.Type``, ``Pulse.trise`` and a swept
    ``Pulse.Frequency`` at once: a quartersin-amplitude linear chirp over
    600 MHz, which is nothing like the rectangular pulses the companion suite
    uses.
    """
    x, y, info = saffron(sys_q(), exp_q(), opt_q())
    assert info['SinglePointDetection'] is False
    _check_axis(x, refs['shaped_spinecho'], "shaped_spinecho")
    _check(y, refs['shaped_spinecho'], "shaped_spinecho")
    # The chirp echo refocuses in the middle of the window, not at an edge.
    assert abs(int(np.argmax(np.abs(y))) - (len(y) - 1) / 2) < 0.3 * len(y)


def test_shaped_chirp_spin_echo_finer_time_step(refs):
    """The same echo at a four times finer Opt.IntTimeStep.

    The chirp waveform is resampled at the new step and the propagation takes
    four times as many sub-steps, so this checks the agreement comes from
    doing the same calculation rather than from a coarse step hiding a
    difference.
    """
    _, y, _ = saffron(sys_q(), exp_q(), opt_q(IntTimeStep=5e-6))
    _check(y, refs['shaped_spinecho_fine'], "shaped_spinecho_fine")


def test_shaped_relaxation(refs):
    """T1/T2 relaxation during the chirps (EasySpin saffron_thyme_relaxation).

    Relaxation routes the propagation through the Liouvillian, including
    inside the shaped pulses, where the propagator lookup table is built per
    waveform point.
    """
    _, y, _ = saffron(sys_q(T1=2.0, T2=0.5), exp_q(), opt_q())
    _check(y, refs['shaped_relaxation'], "shaped_relaxation")
    # Guard the guard: a conjugated density matrix would be invisible on a
    # trace without a substantial imaginary part.
    ref_y = np.asarray(refs['shaped_relaxation']['y']).ravel()
    assert np.linalg.norm(ref_y.imag) > 0.1 * np.linalg.norm(ref_y.real)


def test_shaped_explicit_simulation_frame(refs):
    """Opt.SimFreq = 15 GHz (EasySpin saffron_thyme_simfreq).

    The automatic frame for this sequence is floor(34.78-2) = 32 GHz, so
    naming 15 GHz moves the pulse carriers, the g shift and the detection
    translation together. Getting any one of them wrong shows up here.
    """
    window = [-0.05 + CHIRP180['tp'], 0.05 + CHIRP180['tp']]
    _, y, _ = saffron(sys_q(), exp_q(DetWindow=window), opt_q(SimFreq=15))
    _check(y, refs['shaped_simfreq'], "shaped_simfreq")


def test_shaped_delay_sweep(refs):
    """One indirect dimension over the two chirp delays."""
    exp = exp_q(nPoints=2, Dim1=['d1,d2', 0.5])
    x, y, _ = saffron(sys_q(), exp, opt_q())
    assert isinstance(x, list) and len(x) == 2
    assert y.shape == (2, len(x[1]))
    _check_axis(x, refs['shaped_dim1'], "shaped_dim1")
    _check(y, refs['shaped_dim1'], "shaped_dim1")


def test_shaped_pulse_parameter_sweep(refs):
    """A delay sweep crossed with a pulse-parameter sweep.

    EasySpin saffron_thyme_indirectdimensions: ``Exp.Dim2 = {'p1.Flip' pi/3}``
    rebuilds the first chirp's waveform at each point of the second dimension,
    so this exercises the shaped-pulse generator inside the acquisition loop
    rather than once up front. nPoints = [3 2] is deliberately not square, so
    a transposed buffer cannot hide.
    """
    exp = exp_q(nPoints=[3, 2], Dim1=['d1,d2', 0.5],
                Dim2=['p1.Flip', np.pi / 3])
    x, y, _ = saffron(sys_q(), exp, opt_q())
    assert isinstance(x, list) and len(x) == 3
    assert y.shape == (3, 2, len(x[2]))
    _check_axis(x, refs['shaped_dim2'], "shaped_dim2")
    _check(y, refs['shaped_dim2'], "shaped_dim2")


# ---------------------------------------------------------------------------
# Tensor frames vs sample frame
# ---------------------------------------------------------------------------


def _sys_frames(tilted: bool) -> SpinSystem:
    """EasySpin saffron_thyme_framerotations: Sys.g = gfree + [0 .005 .01]."""
    g = [GFREE, GFREE + 0.005, GFREE + 0.01]
    kw = dict(S=[0.5], g=[g], Nucs=['1H'], A=[[5.0, 2.0, 1.0]])
    if tilted:
        kw.update(gFrame=[ANGLES], AFrame=[ANGLES])
    return SpinSystem(**kw)


def _exp_frames(sample_frame) -> PulseExperiment:
    return PulseExperiment(Field=332, mwFreq=9.3,
                           Sequence=[{'tp': 0.01, 'Flip': np.pi}],
                           DetWindow=[0.0, 0.3], DetPhase=0.0,
                           MolFrame=[0, 0, 0], SampleFrame=[sample_frame])


def opt_rect(**kw) -> SaffronOptions:
    return SaffronOptions(**{'GridSize': 7, 'SimulationMode': 'thyme',
                             'IntTimeStep': ITS_RECT, **kw})


def test_frame_tilts_match_an_equivalent_sample_tilt(refs):
    """Tensor tilts and an equal sample tilt give the same signal.

    EasySpin saffron_thyme_framerotations asserts this equality to 1e-12 and
    nothing else; here both sides are also pinned against MATLAB, so a shared
    error in the frame handling cannot pass. It is the only real-pulse case
    that exercises ``Sys.gFrame``/``Sys.AFrame``, and the sequence ends in a
    *pulse*, which is the ``Exp.DetWindow`` branch that appends a detection
    event instead of lengthening a delay.
    """
    _, y_tensor, _ = saffron(_sys_frames(True), _exp_frames([0, 0, 0]),
                             opt_rect())
    _, y_sample, _ = saffron(_sys_frames(False), _exp_frames(ANGLES),
                             opt_rect())
    _check(y_tensor, refs['frames_tensor_tilt'], "frames_tensor_tilt")
    _check(y_sample, refs['frames_sample_tilt'], "frames_sample_tilt")
    np.testing.assert_allclose(y_tensor, y_sample, rtol=0,
                               atol=1e-12 * np.max(np.abs(y_tensor)))


# ---------------------------------------------------------------------------
# Predefined sequence with a transient detection window
# ---------------------------------------------------------------------------


def _sys_crystal() -> SpinSystem:
    """EasySpin's Sys.A_ = [5 2] axial shorthand, used in both codes."""
    return SpinSystem(S=[0.5], g=[[1.898468237763169] * 3], Nucs=['1H'],
                      A_=[[5.0, 2.0]])


def _exp_crystal(sample_frame) -> PulseExperiment:
    return PulseExperiment(Sequence='3pESEEM', Field=350, mwFreq=9.3,
                           dt=0.01, tau=0.1, T=0.5, nPoints=10,
                           DetWindow=[-0.05, 0.05], DetPhase=0.0,
                           MolFrame=[0, 0, 0], SampleFrame=sample_frame)


def test_predefined_3peseem_transient_crystal(refs):
    """Predefined 3pESEEM over a transient window, one and two crystal sites.

    EasySpin saffron_thyme_crystalorientations. The companion suite only
    detects predefined sequences at a single point, so this is the one case
    where ``Exp.DetWindow`` shortens the last delay of a template-built
    sequence -- and it keeps both the increment axis and the transient axis.
    The two-site result must be the *sum* of the single-site runs, since
    EasySpin gives each site the weight 4*pi and does not renormalize.
    """
    x1, y1, _ = saffron(_sys_crystal(),
                        _exp_crystal([[0, -np.pi / 2, 0]]), opt_rect())
    x2, y2, _ = saffron(_sys_crystal(),
                        _exp_crystal([[0, -np.pi / 2, 0], [0, 0, 0]]),
                        opt_rect())
    assert y1.shape == (10, len(x1[1]))
    _check_axis(x1, refs['crystal_3peseem_one'], "crystal_3peseem_one")
    _check(y1, refs['crystal_3peseem_one'], "crystal_3peseem_one")
    _check(y2, refs['crystal_3peseem_two'], "crystal_3peseem_two")

    # The second site alone, which MATLAB pins only through the stack.
    _, y_b, _ = saffron(_sys_crystal(), _exp_crystal([[0, 0, 0]]), opt_rect())
    np.testing.assert_allclose(y2, y1 + y_b, rtol=1e-9)
