"""Perturbation theory and Sys.Ham (B46, B49).

The perturbation-theory solvers build the electron Zeeman term from Sys.g only.
With g given as Ham110/Ham112, SpinSystem sets g = 0, so before 0.9.8
endorfrq_perturb and resfreqs_perturb returned NaN frequencies,
resfields_perturb found no resonances and pepper/pepper_autograd with
Options.Method='perturb*' returned an all-zero spectrum, all without an error.
They now raise. EasySpin's resfreqs_perturb rejects every Sys.Ham*
(resfreqs_perturb.m:75-77); its pepper perturbation path and resfields_perturb
silently give zero, as torchspin did. The matrix method includes Ham and
reproduces g.
"""

import numpy as np
import pytest
import torch

from torchspin.endorfrq_perturb import endorfrq_perturb
from torchspin.experiment import Experiment, Options
from torchspin.pepper import pepper
from torchspin.pepper_autograd import pepper_autograd
from torchspin.resfields_perturb import resfields_perturb, resfields_perturb_batch
from torchspin.resfreqs_perturb import resfreqs_perturb
from torchspin.spinsystem import SpinSystem
from torchspin.tests.test_saffron_ham import G, HAM_LIN

MSG = r"does not support Sys\.Ham\* parameters"
HAMS = [HAM_LIN, {'112': [[0.0, 0.0, 0.01, 0.0, 0.0]]}]
EXP_FIELD = Experiment(mwFreq=9.5, Range=[330.0, 350.0], nPoints=512)
EXP_FREQ = Experiment(Field=340.0, mwRange=[9.0, 10.0], nPoints=100)
PHI, THETA = 0.3, 0.7


def _sys(Ham=None, **kw):
    d = dict(S=[0.5], Nucs='1H', A=[[3.0, 5.0, 12.0]], lw=[0.5, 0.0])
    d.update(kw)
    return SpinSystem(Ham=Ham, **d) if Ham is not None else SpinSystem(g=[G], **d)


def _sys_g_ham(Ham):
    return SpinSystem(S=[0.5], g=[G], Nucs='1H', A=[[3.0, 5.0, 12.0]], lw=[0.5, 0.0], Ham=Ham)


def _batch(sys):
    return resfields_perturb_batch(sys, torch.tensor([PHI], dtype=torch.float64),
                                   torch.tensor([THETA], dtype=torch.float64), EXP_FIELD)


SOLVERS = {
    'endorfrq_perturb': lambda s: endorfrq_perturb(s, 340.0, PHI, THETA)[0],
    'resfreqs_perturb': lambda s: resfreqs_perturb(s, PHI, THETA, EXP_FREQ)[0],
    'resfields_perturb': lambda s: resfields_perturb(s, PHI, THETA, EXP_FIELD)[0],
    'resfields_perturb_batch': lambda s: _batch(s)[0][0],
}


@pytest.mark.parametrize("Ham", HAMS, ids=['Ham110+112', 'Ham112'])
@pytest.mark.parametrize("name", list(SOLVERS))
def test_perturb_solvers_raise_for_ham(name, Ham):
    with pytest.raises(ValueError, match=MSG):
        SOLVERS[name](_sys(Ham))


@pytest.mark.parametrize("name", list(SOLVERS))
def test_perturb_solvers_with_g_unchanged(name):
    out = SOLVERS[name](_sys())
    assert out.numel() == 2 and bool(torch.all(torch.isfinite(out)))


@pytest.mark.parametrize("method", ['perturb', 'perturb1', 'perturb2'])
def test_pepper_perturb_raises_for_ham(method):
    with pytest.raises(ValueError, match=r"pepper with Options\.Method='%s'" % method):
        pepper(_sys(HAM_LIN), EXP_FIELD, Options(Method=method))
    # frequency sweeps (EasySpin: resfreqs_perturb raises) and components
    with pytest.raises(ValueError, match=MSG):
        pepper(_sys(HAM_LIN), Experiment(Field=340.0, mwRange=[9.4, 9.6], nPoints=256),
               Options(Method=method))
    with pytest.raises(ValueError, match=MSG):
        pepper([_sys(), _sys(HAM_LIN)], EXP_FIELD, Options(Method=method))


@pytest.mark.parametrize("method", ['perturb', 'perturb1', 'perturb2'])
def test_pepper_perturb_with_g_unchanged(method):
    _, spec = pepper(_sys(), EXP_FIELD, Options(Method=method))
    assert float(np.abs(np.asarray(spec)).max()) > 0


def test_pepper_autograd_perturb_raises_for_ham():
    s = _sys(HAM_LIN)
    s.A = s.A.clone().requires_grad_(True)
    with pytest.raises(ValueError, match=r"pepper_autograd with Options\.Method='perturb'"):
        pepper_autograd(s, EXP_FIELD, Options(Method='perturb'))
    # pepper redirects inputs that require grad to pepper_autograd
    with pytest.raises(ValueError, match=MSG):
        pepper(s, EXP_FIELD, Options(Method='perturb'))


def test_pepper_autograd_perturb_with_g_unchanged():
    s = _sys()
    s.A = s.A.clone().requires_grad_(True)
    _, spec = pepper_autograd(s, EXP_FIELD, Options(Method='perturb'))
    assert float(spec.detach().abs().max()) > 0
    spec.abs().sum().backward()
    assert s.A.grad is not None and bool(torch.all(torch.isfinite(s.A.grad)))


def test_matrix_method_keeps_ham():
    """The default (matrix) method includes Ham: Ham-given g reproduces g."""
    y_g = np.asarray(pepper(_sys(), EXP_FIELD, Options())[1])
    y_h = np.asarray(pepper(_sys(HAM_LIN), EXP_FIELD, Options())[1])
    assert np.max(np.abs(y_h - y_g)) < 1e-9 * np.max(np.abs(y_g))


def test_field_quadratic_ham_still_dropped():
    """Terms of order lB >= 2 are dropped by every solver unless
    Options.QuadraticZeeman is on (then the perturbation paths raise
    NotImplementedError, test_quadratic_zeeman.py), so they do not raise here."""
    s = _sys_g_ham({'202': [[1e-4, 0.0, 2e-4, 0.0, 0.0]]})
    _, spec = pepper(s, EXP_FIELD, Options(Method='perturb'))
    _, spec_g = pepper(_sys(), EXP_FIELD, Options(Method='perturb'))
    assert np.array_equal(np.asarray(spec), np.asarray(spec_g))
    assert resfields_perturb(s, PHI, THETA, EXP_FIELD)[0].numel() == 2
