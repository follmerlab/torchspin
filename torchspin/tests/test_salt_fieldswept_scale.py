"""Field-swept salt has the same scale on both EPR solvers (B44).

In the field-swept mode (``Experiment.Range``; EasySpin's salt has no such
mode) the EPR resonances come from ``resfields_perturb_batch`` for one electron
and from ``resfields_batch`` otherwise.  At infinite temperature EasySpin shares
the polarization among the nuclear sublevels, 1/prod(2I+1), in both solvers
(resfields.m:1265-1266, resfields_perturb.m:486-487).  Before 0.9.9 salt called
``resfields_batch`` without the spin system, which skips that division, so the
matrix path was prod(2I+1) times too large (6x for 1H + 14N).

The matrix path is reached with a second electron that is a pure spectator: no
hyperfine coupling, no exchange, and its resonance (g = 4, ~170 mT) is outside
the sweep.  Its two states double every EPR line of the first electron (the
spectator's mS = +-1/2) and every nuclear transition summed by the ENDOR step,
so the spectrum is exactly 4 times that of the one-electron system.  14N carries
a negligible Q so that the one-electron system also takes the matrix ENDOR step
and the two runs differ only in the EPR solver; Threshold = 0 keeps every line.
"""

import numpy as np
import pytest

from torchspin.experiment import Experiment, Options
from torchspin.salt import salt
from torchspin.spinsystem import SpinSystem

G = [2.01, 2.005, 1.998]
A = [[3.0, 5.0, 12.0], [1.0, 1.5, 2.0]]
Q = [[0.0, 0.0, 0.0], [1e-9, 1e-9, -2e-9]]
OPT = Options(GridSize=7, Threshold=0.0)


def _spectra(exp):
    one = SpinSystem(S=[0.5], g=[G], Nucs='1H,14N', A=A, Q=Q)
    two = SpinSystem(S=[0.5, 0.5], g=[G, [4.0, 4.0, 4.0]], Nucs='1H,14N',
                     A=[a + [0.0, 0.0, 0.0] for a in A], ee=[0.0, 0.0, 0.0], Q=Q)
    kw = dict(freq_range=(0, 30), n_points=601, lw_mhz=0.3)
    y1 = salt(one, exp, OPT, **kw)[1].numpy()
    y2 = salt(two, exp, OPT, **kw)[1].numpy()
    return y1, y2


def test_matrix_and_perturbation_solvers_same_scale():
    """main: ratio 24 (= 4 x prod(2I+1) = 4 x 6); fixed: 4."""
    y1, y2 = _spectra(Experiment(Range=[330, 345], mwFreq=9.5))
    assert y1.sum() > 0
    np.testing.assert_allclose(y2.sum() / y1.sum(), 4.0, rtol=1e-6)
    # Shapes differ only by the perturbative resonance fields.
    cos = y1 @ y2 / np.linalg.norm(y1) / np.linalg.norm(y2)
    assert cos > 0.9999


def test_finite_temperature_unchanged():
    """At finite temperature the Boltzmann populations of the matrix path are
    normalized over the full space, which already shares them among the
    nuclear sublevels (no extra 1/prod(2I+1)); they are also shared with the
    spectator electron, hence 2 rather than 4.  Same on main."""
    y1, y2 = _spectra(Experiment(Range=[330, 345], mwFreq=9.5, Temperature=10.0))
    assert y1.sum() > 0
    np.testing.assert_allclose(y2.sum() / y1.sum(), 2.0, rtol=1e-6)
