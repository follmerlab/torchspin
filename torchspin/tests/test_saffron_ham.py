"""saffron and Sys.Ham (B23).

saffron's S = 1/2 fast path builds the electron Zeeman term from Sys.g only.
Before 0.8.13 a linear Zeeman term given as Ham110/Ham112 (g = 0) was dropped
there and the signal was exactly zero. EasySpin's saffron rejects every Sys.Ham*
(saffron.m:130-131); torchspin now raises on that path and keeps Ham on the
S > 1/2 fast path and in the real-pulse engine ('thyme'), where the full
Hamiltonian is built by ham(). There is no EasySpin reference for those.
"""

import numpy as np
import pytest
import torch

from torchspin.ham import ham
from torchspin.saffron import PulseExperiment, SaffronOptions, saffron
from torchspin.spinsystem import SpinSystem

G = [2.01, 2.005, 1.998]
# Ham110/Ham112 coefficients that reproduce G exactly (least squares on ham()'s
# magnetic moments, residual 2e-15; the same values for S = 1/2 and S = 3/2).
HAM_LIN = {'110': [[-48.58946418805701]],
           '112': [[0.04948419845999526, 0.0, -0.10856491814573295, 0.0, 0.0]]}
MSG = r"saffron does not support Sys\.Ham\* parameters for S = 1/2"

ESEEM3P = dict(Sequence='3pESEEM', Field=330, dt=0.012, nPoints=150,
               tau=0.096, T=0.05, mwFreq=9.5, ExciteWidth=100)


def _fast(sys):
    return np.asarray(saffron(sys, PulseExperiment(**ESEEM3P),
                              SaffronOptions(GridSize=11, TimeDomain=True))[1]).ravel()


def test_ham_and_g_are_the_same_hamiltonian():
    s_g = SpinSystem(S=[0.5], g=[G])
    s_h = SpinSystem(S=[0.5], Ham=HAM_LIN)
    mu_g = torch.stack(ham(s_g, B0=None)[1:])
    mu_h = torch.stack(ham(s_h, B0=None)[1:])
    assert torch.allclose(mu_g, mu_h, atol=1e-12)


@pytest.mark.parametrize("Ham", [HAM_LIN, {'112': [[0.0, 0.0, 0.01, 0.0, 0.0]]}],
                         ids=['linear-zeeman', 'Ham112-only'])
def test_s_half_fast_path_raises(Ham):
    sys = SpinSystem(S=[0.5], Nucs='1H', A=[[3.0, 5.0, 12.0]], Ham=Ham)
    with pytest.raises(ValueError, match=MSG):
        _fast(sys)


def test_s_half_fast_path_raises_for_ideal_pulse_hyscore():
    sys = SpinSystem(S=[0.5], Nucs='14N', A=[[2.5, 3.1, 5.4]], Q=[[-0.3, -0.1, 0.4]], Ham=HAM_LIN)
    exp = PulseExperiment(Sequence='HYSCORE', Field=330, dt=0.05, nPoints=16, tau=0.1)
    with pytest.raises(ValueError, match=MSG):
        saffron(sys, exp, SaffronOptions(GridSize=5))


def test_s_three_halves_fast_path_keeps_ham():
    common = dict(S=[1.5], D=[[50, 0]], Nucs='14N', A=[[2.5, 3.1, 5.4]], Q=[[-0.3, -0.1, 0.4]])
    y_g = _fast(SpinSystem(g=[G], **common))
    y_h = _fast(SpinSystem(Ham=HAM_LIN, **common))
    assert np.linalg.norm(y_g) > 0
    assert np.linalg.norm(y_h - y_g) / np.linalg.norm(y_g) < 1e-10


def test_s_half_thyme_keeps_ham():
    """The real-pulse engine builds its Hamiltonian with ham(), which has Ham."""
    p90 = {'tp': 0.016, 'Flip': np.pi / 2}
    p180 = {'tp': 0.032, 'Flip': np.pi}
    mw = float(np.mean(G)) * 9.2740100783e-24 * 0.330 / 6.62607015e-34 / 1e9
    exp = PulseExperiment(Field=330.0, mwFreq=mw, Sequence=[p90, 0.2, p180, 0.2],
                          DetWindow=0.0, DetPhase=0.0, nPoints=24, Dim1=['d1,d2', 0.02])
    opt = SaffronOptions(GridSize=5, IntTimeStep=1e-4, SimulationMode='thyme')
    common = dict(S=[0.5], Nucs='1H', A=[[3.0, 5.0, 12.0]])
    y_g = np.asarray(saffron(SpinSystem(g=[G], **common), exp, opt)[1]).ravel()
    y_h = np.asarray(saffron(SpinSystem(Ham=HAM_LIN, **common), exp, opt)[1]).ravel()
    assert np.linalg.norm(y_g) > 0
    assert np.linalg.norm(y_h - y_g) / np.linalg.norm(y_g) < 1e-10
