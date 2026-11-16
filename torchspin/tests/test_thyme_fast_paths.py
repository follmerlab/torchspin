"""B11 (real-pulse engine): the eigenbasis closed form for detected free evolutions equals
the step-by-step U σ U† loop to roundoff, and the event-order / delay-propagator caches leave
results unchanged (bit for bit) across repeated calls and orientations."""
import sys

import numpy as np
import pytest

from torchspin.saffron import PulseExperiment, SaffronOptions, saffron
from torchspin.spinsystem import SpinSystem

P90 = {'tp': 0.016, 'Flip': np.pi / 2}
P180 = {'tp': 0.032, 'Flip': np.pi}
H1 = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]], Nucs=['1H'], A=[[-2.0, -2.0, 4.0]])
NX = SpinSystem(S=[0.5], g=[[2.009, 2.006, 2.002]], Nucs=['14N'], A=[[16.0, 16.0, 95.0]])
CASES = [
    (H1, dict(Sequence=[P90, 0.15, P180, 0.15], DetWindow=[-0.02, 0.02], nPoints=6, Dim1=['d1,d2', 0.03])),
    (NX, dict(Sequence=[P90, 0.12, P180, 0.12], DetWindow=[-0.016, 0.016], nPoints=4, Dim1=['d1,d2', 0.04],
              DetOperator=['+1', 'z1'])),
]


@pytest.mark.parametrize('sys_, kw', CASES, ids=['1H', 'nitroxide_two_detectors'])
def test_closed_form_detection_equals_step_loop(sys_, kw, monkeypatch):
    sp = sys.modules['torchspin.spidyan']
    e = PulseExperiment(Field=340.0, mwFreq=9.5, DetPhase=0.0, **kw)
    o = SaffronOptions(GridSize=5)
    new = np.asarray(saffron(sys_, e, o)[1])
    monkeypatch.setattr(sp, '_CLOSED_FORM_MIN_STEPS', 10 ** 9)
    ref = np.asarray(saffron(sys_, e, o)[1])
    assert np.abs(ref).max() > 0
    assert np.abs(new - ref).max() <= 1e-10 * np.abs(ref).max()


def test_caches_do_not_change_repeated_results():
    sp = sys.modules['torchspin.spidyan']
    e = PulseExperiment(Field=340.0, mwFreq=9.5, Sequence=[P90, 0.2, P90, 0.05, P90, 0.2], DetWindow=0.0,
                        DetPhase=0.0, nPoints=[6, 5], Dim1=['d1', 0.02], Dim2=['d2', 0.02])
    sp._REORDER_CACHE.clear()
    a = np.asarray(saffron(NX, e, SaffronOptions(GridSize=5))[1])          # cold cache
    b = np.asarray(saffron(NX, e, SaffronOptions(GridSize=5))[1])          # warm cache
    np.testing.assert_array_equal(a, b)
