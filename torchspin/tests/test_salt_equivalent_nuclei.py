"""salt rejects sets of equivalent nuclei (Sys.n > 1), as EasySpin does (B21).

EasySpin salt.m:135-136 raises 'salt does not support sets of equivalent
nuclei (Sys.n>1).' Before 0.8.10 torchspin's salt ignored ``n`` and returned
the spectrum of a single nucleus for n = 2.
"""

import pytest
import torch

from torchspin.experiment import Experiment, Options
from torchspin.salt import salt
from torchspin.spinsystem import SpinSystem

MSG = r"salt does not support sets of equivalent nuclei \(Sys\.n>1\)\."
OPT = Options(GridSize=7, GridSymmetry='Ci')
FIXED = Experiment(Field=326.5, mwFreq=9.7)


def _sys_1h(**kw):
    return SpinSystem(S=0.5, g=[2.0, 2.01, 2.02], Nucs='1H',
                      A=[[-2.0, -2.0, 4.0]], **kw)


def _run(sys, exp=FIXED, opt=OPT, fr=(10, 20)):
    return salt(sys, exp, opt, freq_range=fr, n_points=128, lw_mhz=0.2)


def test_n2_raises_first_order_path():
    with pytest.raises(ValueError, match=MSG):
        _run(_sys_1h(n=[2]))


def test_n2_raises_matrix_path():
    """14N with Q goes through the matrix path."""
    sys = SpinSystem(S=0.5, g=[2.0, 2.0, 2.0], Nucs='1H,14N',
                     A=[[3.0, 3.0, 5.0], [2.0, 2.0, 3.0]],
                     Q=[[0.0, 0.0, 0.0], [-0.3, -0.3, 0.6]], n=[1, 2])
    with pytest.raises(ValueError, match=MSG):
        _run(sys, fr=(0.1, 20))


def test_n2_raises_two_electrons():
    sys = SpinSystem(S=[0.5, 0.5], g=[[2.0, 2.0, 2.0], [2.01, 2.01, 2.01]],
                     ee=100.0, Nucs='1H', A=[[2.0, 2.0, 4.0, 0.0, 0.0, 0.0]],
                     n=[3])
    with pytest.raises(ValueError, match=MSG):
        _run(sys)


def test_n2_raises_field_swept_mode():
    exp = Experiment(mwFreq=9.7, Range=[340.0, 350.0], nPoints=64)
    with pytest.raises(ValueError, match=MSG):
        _run(_sys_1h(n=[2]), exp=exp)


def test_n2_raises_zero_field():
    sys = SpinSystem(S=1, D=[1000.0, 100.0], Nucs='1H', A=[2.0, 2.0, 4.0], n=[2])
    with pytest.raises(ValueError, match=MSG):
        _run(sys, exp=Experiment(Field=0), opt=Options(ZeroField=True, GridSize=7))


def test_n_set_after_construction_raises():
    """Attribute assignment is normalized (0.8.7) and still checked."""
    sys = _sys_1h()
    sys.n = 2
    with pytest.raises(ValueError, match=MSG):
        _run(sys)


def test_n1_runs_and_equals_default():
    x0, y0 = _run(_sys_1h())
    x1, y1 = _run(_sys_1h(n=[1]))
    assert torch.equal(x0, x1)
    assert torch.equal(y0, y1)
    assert float(y1.abs().max()) > 0.0
