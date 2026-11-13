"""B83: hamsymm's grid lives in the symmetry frame; consumers must rotate it.

saffron, saffron_thyme, the broadband model of torchspin.autograd and nucfrq2d
dropped hamsymm's frame (grid used as if in the molecular frame) and replaced a
hamsymm error by a silent 'C1'.  Check: a powder result is invariant under a
rigid rotation of all tensors, so a system tilted by an Euler rotation must give
the result of the untilted system (same group, grid rotated with it).
"""
import numpy as np
import pytest
import torch

from torchspin import SpinSystem, PulseExperiment, SaffronOptions, saffron
from torchspin.hamsymm import hamsymm

F64 = torch.float64


def _hyperfine(tilted):
    kw = dict(AFrame=[[0.0, 0.7, 0.0]]) if tilted else {}
    return SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]], Nucs=['1H'], A=[[2.0, 2.0, 12.0]], **kw)


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max() / np.abs(b).max())


def test_tilted_frame_symmetry():
    pg, R = hamsymm(_hyperfine(True))
    assert pg == 'Dinfh'
    assert not torch.allclose(R, torch.eye(3, dtype=F64), atol=1e-3)


def test_saffron_rotated_grid():
    """3pESEEM: tilted == untilted (was 1.9 % apart)."""
    exp = PulseExperiment(Sequence='3pESEEM', Field=340.0, mwFreq=9.5, dt=0.02, tau=0.12,
                          T=0.1, nPoints=[64])
    y_t = saffron(_hyperfine(True), exp, SaffronOptions(GridSize=31))[1]
    y_u = saffron(_hyperfine(False), exp, SaffronOptions(GridSize=31))[1]
    assert _rel(y_t, y_u) < 1e-10


def test_saffron_thyme_rotated_grid():
    """2pESEEM with real pulses: tilted == untilted (was 111 % apart)."""
    from torchspin.saffron_thyme import saffron_thyme
    exp = PulseExperiment(Sequence='2pESEEM', Field=340.0, mwFreq=9.5, dt=0.02, tau=0.2,
                          nPoints=[16], DetWindow=0.0, DetPhase=0.0)
    y_t = saffron_thyme(_hyperfine(True), exp, SaffronOptions(GridSize=7))[1]
    y_u = saffron_thyme(_hyperfine(False), exp, SaffronOptions(GridSize=7))[1]
    assert _rel(y_t, y_u) < 1e-8


def test_nucfrq2d_rotated_grid():
    """HYSCORE frequencies: the tilted system sweeps the untilted one's orientations."""
    from torchspin.nucfrq2d import nucfrq2d
    d_t = nucfrq2d(_hyperfine(True), 340.0, GridSize=31)
    d_u = nucfrq2d(_hyperfine(False), 340.0, GridSize=31)
    for key in ('FreqsAlpha', 'FreqsBeta'):
        np.testing.assert_allclose(d_t[key], d_u[key], rtol=0, atol=1e-9)


def test_broadband_autograd_rotated_grid():
    """Deprecated broadband model: axial g along x == axial g along z (was 78 % apart)."""
    from torchspin.autograd import _broadband_spectrum

    def run(gv):
        s = SpinSystem(S=[0.5], g=torch.tensor([gv], dtype=F64))
        return _broadband_spectrum(s, 9.5, (300., 350.), 512, 1, 0.5, 31, 'auto', None, 'cpu', F64)[1]
    assert _rel(run([2.2, 2.0, 2.0]), run([2.0, 2.0, 2.2])) < 1e-10


def test_hamsymm_errors_propagate(monkeypatch):
    """A hamsymm failure is an error, not a silent C1 grid."""
    import sys
    saffron_mod = sys.modules['torchspin.saffron']     # the module (the package exports the function)

    def boom(sys, debug=False):
        raise RuntimeError('hamsymm failed')
    monkeypatch.setattr(saffron_mod, 'hamsymm', boom)
    exp = PulseExperiment(Sequence='3pESEEM', Field=340.0, mwFreq=9.5, dt=0.02, tau=0.12,
                          T=0.1, nPoints=[8])
    with pytest.raises(RuntimeError, match='hamsymm failed'):
        saffron(_hyperfine(True), exp, SaffronOptions(GridSize=5))
