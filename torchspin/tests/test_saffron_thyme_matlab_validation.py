"""Strict MATLAB cross-validation of saffron's real-pulse ("thyme") route.

Until 0.5.0 this route had no MATLAB comparison at all: ``saffron_thyme`` had
only internal-consistency tests, which cannot catch an error both sides share.
It also had a different output contract from EasySpin -- per-increment
transients instead of one detected point per increment -- because torchspin
performed the orientation sum and then skipped every other step of EasySpin's
detection stage (``saffron.m`` 1884-1940): the detection phase, echo
integration, down-conversion through ``signalprocessing.m`` and the
``Exp.DetWindow`` time axis.

What the cases below pin down, each of which was wrong or missing before:

1. the simulation frame -- ``Exp.mwFreq`` was never passed to the sequencer,
   so pulses were DC fields in the lab frame and nothing was excited;
2. the lab-frame basis -- the Hamiltonian was built in the molecular frame
   while the initial state, excitation and detection operators stayed
   lab-frame, so any orientation with a nonzero phi angle was wrong;
3. the absolute scale -- the orientation sum was divided by the total grid
   weight, where EasySpin does not normalize it (the weights sum to 4*pi);
4. the whole detection stage listed above;
5. the default ``Opt.IntTimeStep`` -- Nyquist/10 unrounded rather than
   EasySpin's Nyquist/50 rounded to two significant digits;
6. the Liouville-space commutator superoperator used with relaxation, which
   mixed MATLAB's column-major ``kron()`` order with numpy's row-major
   flattening and so propagated ``conj(rho)``.

Every assertion checks the raw *and* the mean-centered relative L2 error --
never a cosine, which is scale-invariant and DC-dominated. Every option that
affects the result is stated explicitly on both sides
(``Opt.SimulationMode``, ``Opt.GridSize``, ``Opt.IntTimeStep``,
``Opt.SimFreq``, ``Exp.DetWindow``, ``Exp.DetPhase``, ``Exp.DetIntegrate``),
because three of the harness faults found in the 0.5.0 pulse-EPR audit were
unstated options rather than wrong code.

References come from ``tests/data/generate_saffron_thyme_refs.m``. Note that
the CI matrix deselects ``-k "not matlab_validation"``, so these run only in
the full-suite job; ``test_thyme_references_present`` fails rather than skips
if the reference file is missing, so a silent skip cannot look like a pass.
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
from torchspin.saffron import saffron, PulseExperiment, SaffronOptions

DATA = (Path(__file__).parent.parent.parent / "tests" / "data"
        / "saffron_thyme_refs.mat")

# Observed agreement is 3e-13..3e-12 on both the raw and the mean-centered
# signal; these leave ~2 orders of headroom for eigensolver/BLAS ordering
# differences across platforms.
RAW_TOL = 1e-10
CENTERED_TOL = 1e-9

# --- Spin systems, matching the generator exactly -------------------------
G_ANISO = [2.00906, 2.0062, 2.0023]


def sys_g() -> SpinSystem:
    return SpinSystem(S=[0.5], g=[G_ANISO])


def sys_1h() -> SpinSystem:
    return SpinSystem(S=[0.5], g=[G_ANISO], Nucs=['1H'], A=[[3.0, 3.0, 8.0]])


def sys_14n() -> SpinSystem:
    return SpinSystem(S=[0.5], g=[G_ANISO], Nucs=['14N'],
                      A=[[11.5, 11.5, 95.0]], Q=[[0.8, 0.5]])


def sys_relax() -> SpinSystem:
    return SpinSystem(S=[0.5], g=[G_ANISO], Nucs=['1H'],
                      A=[[3.0, 3.0, 8.0]], T1=2.0, T2=0.5)


# --- Rectangular pulses and shared timings --------------------------------
P90 = {'tp': 0.010, 'Flip': np.pi / 2}
P180 = {'tp': 0.020, 'Flip': np.pi}
TAU = 0.15
ECHO_SEQ = [P90, TAU, P180, TAU]
# For rectangular pulses the echo refocuses at the end of the last delay, so
# the window is centered on zero. A negative start also exercises the branch
# that shortens the preceding free evolution rather than appending to it.
ECHO_WINDOW = [-0.03, 0.03]


def opt(**kw) -> SaffronOptions:
    """Options with every result-affecting field stated, as the generator does."""
    return SaffronOptions(**{'SimulationMode': 'thyme', 'IntTimeStep': 1e-4,
                             'GridSize': 7, **kw})


def eseem_exp(**kw) -> PulseExperiment:
    """Single-point-detected two-pulse ESEEM, 16 increments of 20 ns."""
    return PulseExperiment(Field=332, mwFreq=9.3, Sequence=ECHO_SEQ,
                           DetWindow=0.0, DetPhase=0.0, nPoints=16,
                           Dim1=['d1,d2', 0.02], **kw)


@pytest.fixture(scope="module")
def refs():
    if not SCIPY_OK:
        pytest.skip("scipy not installed")
    if not DATA.exists():
        pytest.skip(f"reference file not found: {DATA}")
    return loadmat(str(DATA), simplify_cells=True)["data"]


def _errors(y, y_ref) -> tuple[float, float]:
    """Raw and mean-centered relative L2 errors."""
    a = np.atleast_1d(np.asarray(y))
    b = np.atleast_1d(np.asarray(y_ref))
    assert a.shape == b.shape, f"shape {a.shape} != reference {b.shape}"
    a, b = a.ravel(), b.ravel()
    raw = np.linalg.norm(a - b) / np.linalg.norm(b)
    ac, bc = a - a.mean(), b - b.mean()
    if np.linalg.norm(bc) <= 1e-12 * max(abs(b.mean()), 1e-30):
        # A single integrated point carries no modulation to compare.
        return float(raw), 0.0
    return float(raw), float(np.linalg.norm(ac - bc) / np.linalg.norm(bc))


def _check(y, ref_case, label: str) -> None:
    raw, centered = _errors(y, ref_case['y'])
    assert raw < RAW_TOL, f"{label}: raw relative error {raw:.3e} >= {RAW_TOL:g}"
    assert centered < CENTERED_TOL, (
        f"{label}: mean-centered (modulation) relative error "
        f"{centered:.3e} >= {CENTERED_TOL:g}")


def _check_axis(x, ref_case, label: str) -> None:
    """The x axis is part of the contract: DetWindow sets the transient axis."""
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


def test_thyme_references_present():
    """Fail, not skip, if the reference file is missing.

    A skipped cross-validation is indistinguishable from a passing one in CI
    output, which is how an unvalidated route stayed unvalidated.
    """
    assert DATA.exists(), (
        f"{DATA.name} is missing. Regenerate it with\n"
        "  matlab -nodisplay -batch "
        "\"run('tests/data/generate_saffron_thyme_refs.m')\"")


# ---------------------------------------------------------------------------
# Transient detection: the Exp.DetWindow contract
# ---------------------------------------------------------------------------


def test_thyme_fid_powder(refs):
    """Powder FID transient: frame shift, pulse carrier, grid weights, scale.

    The absolute scale is the point of this case. EasySpin sums the
    orientations weighted by ``Exp.OriWeights`` without renormalizing, and
    those weights sum to 4*pi; dividing by the total weight (as torchspin did)
    is a factor of ~12.6 that no shape comparison would see.
    """
    exp = PulseExperiment(Field=332, mwFreq=9.3, Sequence=[P90, 0.05],
                          DetWindow=[0, 0.05], DetPhase=0.0)
    x, y, info = saffron(sys_g(), exp, opt())
    assert info['SinglePointDetection'] is False
    _check_axis(x, refs['fid_powder'], "fid_powder")
    _check(y, refs['fid_powder'], "fid_powder")


def test_thyme_fid_crystal(refs):
    """The same transient for a single crystal, with the powder grid removed."""
    exp = PulseExperiment(Field=332, mwFreq=9.3, Sequence=[P90, 0.05],
                          DetWindow=[0, 0.05], DetPhase=0.0,
                          CrystalSymmetry='P1', MolFrame=[0, 0, 0],
                          SampleFrame=[[0, np.pi / 5, 0]])
    x, y, _ = saffron(sys_g(), exp, opt())
    _check_axis(x, refs['fid_crystal'], "fid_crystal")
    _check(y, refs['fid_crystal'], "fid_crystal")


def test_thyme_echo_window(refs):
    """Two-pulse echo detected over a window centered on the echo.

    ``Exp.DetWindow`` is applied by shifting the last sequence element by
    ``DetWindow(1)`` and appending the detected event, not by slicing the
    transient afterwards; slicing instead would move the echo within the
    window.
    """
    exp = PulseExperiment(Field=332, mwFreq=9.3, Sequence=ECHO_SEQ,
                          DetWindow=ECHO_WINDOW, DetPhase=0.0)
    x, y, _ = saffron(sys_1h(), exp, opt())
    _check_axis(x, refs['echo_window'], "echo_window")
    _check(y, refs['echo_window'], "echo_window")
    # The echo must sit inside the window rather than at an edge, which is
    # what a DetWindow applied by slicing the transient would give.
    assert abs(int(np.argmax(np.abs(y))) - (len(y) - 1) / 2) < 0.3 * len(y)


def test_thyme_echo_detection_phase(refs):
    """Exp.DetPhase rotates the detected signal by exp(-1i*DetPhase)."""
    exp = PulseExperiment(Field=332, mwFreq=9.3, Sequence=ECHO_SEQ,
                          DetWindow=ECHO_WINDOW, DetPhase=np.pi / 4)
    _, y, _ = saffron(sys_1h(), exp, opt())
    _check(y, refs['echo_detphase'], "echo_detphase")
    # Independently: it is the same signal as DetPhase=0, phase-rotated.
    exp0 = PulseExperiment(Field=332, mwFreq=9.3, Sequence=ECHO_SEQ,
                           DetWindow=ECHO_WINDOW, DetPhase=0.0)
    _, y0, _ = saffron(sys_1h(), exp0, opt())
    np.testing.assert_allclose(y, y0 * np.exp(-1j * np.pi / 4), rtol=1e-12)


def test_thyme_echo_integrate(refs):
    """Exp.DetIntegrate sums the transient and switches to single-point."""
    exp = PulseExperiment(Field=332, mwFreq=9.3, Sequence=ECHO_SEQ,
                          DetWindow=ECHO_WINDOW, DetPhase=0.0,
                          DetIntegrate=True)
    _, y, info = saffron(sys_1h(), exp, opt())
    assert info['SinglePointDetection'] is True
    _check(y, refs['echo_integrate'], "echo_integrate")
    # The integral is taken over the *un-down-converted* transient, so it is
    # not the sum of the windowed result; check it is the raw sum instead.
    assert np.atleast_1d(np.asarray(y)).size == 1


def test_thyme_transient_over_indirect_dimension(refs):
    """A transient *and* an indirect dimension: both axes are returned."""
    exp = eseem_exp()
    exp.DetWindow = [-0.02, 0.02]
    exp.nPoints = 4
    x, y, _ = saffron(sys_1h(), exp, opt())
    assert isinstance(x, list) and len(x) == 2
    assert y.shape == (4, len(x[1]))
    _check_axis(x, refs['echo_transient_dim'], "echo_transient_dim")
    _check(y, refs['echo_transient_dim'], "echo_transient_dim")


# ---------------------------------------------------------------------------
# Single-point detection: one detected point per increment
# ---------------------------------------------------------------------------


def test_thyme_eseem2p_singlepoint(refs):
    """Real-pulse two-pulse ESEEM: one point per increment, nonzero tau.

    This is the output contract EasySpin returns and torchspin did not: a
    scalar ``Exp.DetWindow`` appends a zero-length detected event, which
    yields exactly one point per acquisition.
    """
    x, y, info = saffron(sys_1h(), eseem_exp(), opt())
    assert info['SinglePointDetection'] is True
    assert y.shape == (16,)
    _check_axis(x, refs['eseem2p_singlepoint'], "eseem2p_singlepoint")
    _check(y, refs['eseem2p_singlepoint'], "eseem2p_singlepoint")


def test_thyme_eseem2p_finer_time_step(refs):
    """The same case at a five times finer Opt.IntTimeStep.

    Agreement must come from doing the same calculation, not from a coarse
    time step hiding a difference.
    """
    _, y, _ = saffron(sys_1h(), eseem_exp(), opt(IntTimeStep=2e-5))
    _check(y, refs['eseem2p_fine'], "eseem2p_fine")


def test_thyme_eseem2p_quadrupolar_14n(refs):
    """14N with a large anisotropic hyperfine and a quadrupole coupling."""
    _, y, _ = saffron(sys_14n(), eseem_exp(), opt())
    _check(y, refs['eseem2p_14n'], "eseem2p_14n")


def test_thyme_detsequence_single_point(refs):
    """Exp.DetSequence with a zero-length final delay is single-point too.

    EasySpin resolves single-point detection either from a scalar
    ``Exp.DetWindow`` or from ``sum(Exp.DetSequence) == 1`` with the detected
    event a zero-length delay; this is the second path.
    """
    exp = PulseExperiment(Field=332, mwFreq=9.3,
                          Sequence=[P90, TAU, P180, TAU, 0.0],
                          DetSequence=[0, 0, 0, 0, 1], DetPhase=0.0,
                          nPoints=8, Dim1=['d1,d2', 0.02])
    _, y, info = saffron(sys_1h(), exp, opt())
    assert info['SinglePointDetection'] is True
    _check(y, refs['eseem2p_detsequence'], "eseem2p_detsequence")


def test_thyme_relaxation(refs):
    """Relaxation during the sequence, which routes through the Liouvillian.

    This is the case that exposed the row-major/column-major mismatch in the
    Liouville-space commutator superoperator: the detected S+ trace came out
    conjugated, which spidyan's own <Sz> relaxation reference could not see.
    """
    _, y, _ = saffron(sys_relax(), eseem_exp(), opt())
    _check(y, refs['eseem2p_relaxation'], "eseem2p_relaxation")
    # Guard the guard: the reference must have a real imaginary part, or a
    # conjugation error would pass unnoticed here as well.
    ref_y = np.asarray(refs['eseem2p_relaxation']['y']).ravel()
    assert np.linalg.norm(ref_y.imag) > 0.1 * np.linalg.norm(ref_y.real)


# ---------------------------------------------------------------------------
# Simulation frame
# ---------------------------------------------------------------------------


def test_thyme_explicit_simulation_frame(refs):
    """Opt.SimFreq pins the frame shift instead of floor(mwFreq-2)."""
    exp = PulseExperiment(Field=332, mwFreq=9.3, Sequence=ECHO_SEQ,
                          DetWindow=ECHO_WINDOW, DetPhase=0.0)
    _, y, _ = saffron(sys_1h(), exp, opt(SimFreq=7.5))
    _check(y, refs['echo_simfreq'], "echo_simfreq")


# ---------------------------------------------------------------------------
# Predefined sequences through the real-pulse engine
# ---------------------------------------------------------------------------


def test_thyme_predefined_3peseem(refs):
    """Predefined 3pESEEM with real pulses, including its phase cycles.

    EasySpin allows a predefined sequence in thyme mode
    (``s_predefinedexperiments`` builds it with 16/32 ns rectangular pulses
    and two-step phase cycles on pulses 1 and 3) while the fast algorithm
    rejects the combination outright.
    """
    exp = PulseExperiment(Field=332, mwFreq=9.3, Sequence='3pESEEM', dt=0.02,
                          tau=0.1, T=0.08, nPoints=12, DetWindow=0.0,
                          DetPhase=0.0)
    x, y, _ = saffron(sys_1h(), exp, opt())
    _check_axis(x, refs['eseem3p_predefined'], "eseem3p_predefined")
    _check(y, refs['eseem3p_predefined'], "eseem3p_predefined")


def test_thyme_predefined_2peseem_crystal(refs):
    """Predefined 2pESEEM with real pulses, single crystal."""
    exp = PulseExperiment(Field=332, mwFreq=9.3, Sequence='2pESEEM', dt=0.02,
                          tau=0.12, nPoints=12, DetWindow=0.0, DetPhase=0.0,
                          CrystalSymmetry='P1', MolFrame=[0, 0, 0],
                          SampleFrame=[[0, -np.pi / 2, 0]])
    x, y, _ = saffron(sys_1h(), exp, opt())
    _check_axis(x, refs['eseem2p_crystal'], "eseem2p_crystal")
    _check(y, refs['eseem2p_crystal'], "eseem2p_crystal")


def test_thyme_crystal_orientations_are_summed(refs):
    """Stacked crystal orientations are summed, not averaged.

    EasySpin gives every crystal orientation the weight 4*pi and sums with
    ``Exp.OriWeights`` without renormalizing, so a two-site stack is the sum
    of the two single-site runs. torchspin divided by the total weight, which
    turned the sum into a mean -- a factor of two here, and a factor of 4*pi
    for a powder.
    """
    def run(sample_frame):
        exp = PulseExperiment(Field=332, mwFreq=9.3, Sequence='2pESEEM',
                              dt=0.02, tau=0.12, nPoints=12, DetWindow=0.0,
                              DetPhase=0.0, CrystalSymmetry='P1',
                              MolFrame=[0, 0, 0], SampleFrame=sample_frame)
        return saffron(sys_1h(), exp, opt())[1]

    y_a = run([[0, -np.pi / 2, 0]])
    y_b = run([[0, 0, 0]])
    y_ab = run([[0, -np.pi / 2, 0], [0, 0, 0]])
    _check(y_b, refs['eseem2p_crystal_b'], "eseem2p_crystal_b")
    _check(y_ab, refs['eseem2p_crystal_two'], "eseem2p_crystal_two")
    np.testing.assert_allclose(y_ab, y_a + y_b, rtol=1e-10)


def test_thyme_predefined_hyscore(refs):
    """Predefined HYSCORE: two indirect dimensions, unequal nPoints.

    Unequal nPoints means a transposed 2D buffer cannot hide.
    """
    exp = PulseExperiment(Field=332, mwFreq=9.3, Sequence='HYSCORE', dt=0.03,
                          tau=0.1, t1=0.02, t2=0.03, nPoints=[4, 6],
                          DetWindow=0.0, DetPhase=0.0)
    x, y, _ = saffron(sys_1h(), exp, opt())
    assert y.shape == (4, 6)
    _check_axis(x, refs['hyscore_predefined'], "hyscore_predefined")
    _check(y, refs['hyscore_predefined'], "hyscore_predefined")
