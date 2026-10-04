"""CardamomPar.seed makes the stochastic trajectory ensemble reproducible."""
import numpy as np

from torchspin import SpinSystem, Experiment, cardamom, CardamomPar, CardamomOptions


def _run(seed):
    sys = SpinSystem(S=[0.5], g=[[2.008, 2.006, 2.003]], Nucs='14N', A=[[20, 20, 85]], tcorr=1e-9)
    exp = Experiment(mwFreq=9.5, Range=[332, 352], nPoints=128, Harmonic=0)
    par = CardamomPar(Model='diffusion', nTraj=10, nSteps=100, dtSpin=1e-10, dtSpatial=1e-10, seed=seed)
    return np.asarray(cardamom(sys, exp, par, CardamomOptions(Method='fast', Verbosity=0))[1])


def test_seed_reproducible_and_distinct():
    assert np.array_equal(_run(3), _run(3))
    assert not np.array_equal(_run(3), _run(4))
