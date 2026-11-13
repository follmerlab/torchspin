"""salt with Sys.Ham (B35) and Sys.L (B45).

B35. A linear Zeeman term given as Ham110/Ham112 sets g = 0. salt's first-order
ENDOR path (fixed field) and its field-swept perturbation resonance search build
the Zeeman term from Sys.g only, so before 0.9.5 such a system gave a fixed-field
spectrum 12.8x too large with Larmor-only lines (8.9x with orientation selection)
and an all-zero field-swept spectrum. EasySpin's salt rejects every Sys.Ham*
(salt.m:132); torchspin deliberately supports them instead: any system with a
Sys.Ham entry goes to the matrix paths, which use ham(). An explicit perturbation
method (Options.Method='perturb*') with Sys.Ham raises.

There is no EasySpin reference (EasySpin raises). The checks are internal: the
Ham110/Ham112 coefficients below reproduce G exactly, so the Ham spectrum must
equal that of g = G on the same (matrix) path. ``ZERO_HAM`` is an all-zero Ham
entry that sends a g system down that path; against the default first-order
spectrum of g = G the comparison is by shape and amplitude.

B45. EasySpin salt.m:133 raises 'salt does not support Sys.L.'; before 0.9.5
torchspin's first-order path ignored Sys.L (S = 1/2, L = 1, soc = 100 MHz gave
the spectrum without L) and the matrix path was never checked for it.
"""

import numpy as np
import pytest
import torch

from torchspin.experiment import Experiment, Options
from torchspin.salt import salt
from torchspin.spinsystem import SpinSystem

G = [2.01, 2.005, 1.998]
# Ham110/Ham112 coefficients that reproduce G exactly (as in test_saffron_ham.py)
HAM_LIN = {'110': [[-48.58946418805701]],
           '112': [[0.04948419845999526, 0.0, -0.10856491814573295, 0.0, 0.0]]}
ZERO_HAM = {'011': [[0.0, 0.0, 0.0]]}
A_1H = [[3.0, 5.0, 12.0]]

FIXED = Experiment(Field=338, mwFreq=9.5)
ORISEL = Experiment(Field=338, mwFreq=9.5, ExciteWidth=30)
SWEPT = Experiment(Range=[330, 345], mwFreq=9.5)
OPT = Options(GridSize=7)


def _sys(**kw):
    return SpinSystem(S=[0.5], Nucs='1H', A=kw.pop('A', A_1H), **kw)


def _run(sys, exp, opt=OPT):
    return salt(sys, exp, opt, freq_range=(0, 30), n_points=601, lw_mhz=0.3)[1]


def _rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


@pytest.mark.parametrize('exp', [FIXED, ORISEL], ids=['fixed', 'fixed_orisel'])
def test_ham_equals_g_fixed_field(exp):
    """Default fixed-field path, with and without orientation selection.

    Before the fix: amplitude 12.77x (8.90x with ExciteWidth), cosine 0.00.
    """
    y_ham = _run(_sys(Ham=HAM_LIN), exp).numpy()
    y_gm = _run(_sys(g=[G], Ham=ZERO_HAM), exp).numpy()
    y_g = _run(_sys(g=[G]), exp).numpy()
    assert y_ham.max() > 0
    # same matrix path: Ham110/112 == g
    assert _rel(y_ham, y_gm) < 1e-9
    # vs the default first-order spectrum of g (matrix vs first order: 0.6 %)
    assert _cos(y_ham, y_g) > 0.9999
    assert abs(y_ham.max() / y_g.max() - 1.0) < 0.02


def test_ham_equals_g_field_swept():
    """Field-swept mode: resfields_batch (Ham) vs resfields_perturb_batch (g).

    Before the fix the Ham spectrum was all zero. Since 0.9.9 (B44) the two
    resonance-field searches share the 1/prod(2I+1) nuclear-sublevel
    polarization; the amplitude vs the first-order g spectrum is 0.994 (it was
    1.99 before 0.9.9).
    """
    y_ham = _run(_sys(Ham=HAM_LIN), SWEPT).numpy()
    y_gm = _run(_sys(g=[G], Ham=ZERO_HAM), SWEPT).numpy()
    y_g = _run(_sys(g=[G]), SWEPT).numpy()
    assert y_ham.max() > 0
    assert _rel(y_ham, y_gm) < 1e-9
    assert _cos(y_ham, y_g) > 0.999
    assert abs(y_ham.max() / y_g.max() - 1.0) < 0.02


def test_field_quadratic_ham_dropped_by_default():
    """lB >= 2 terms are dropped (QuadraticZeeman=False), as in pepper; on raises."""
    ham2 = dict(HAM_LIN, **{'212': [[0.0, 0.0, 0.05, 0.0, 0.0]]})
    y2 = _run(_sys(Ham=ham2), FIXED)
    y1 = _run(_sys(Ham=HAM_LIN), FIXED)
    assert torch.equal(y1, y2)
    with pytest.raises(NotImplementedError, match='QuadraticZeeman'):
        _run(_sys(Ham=ham2), FIXED, Options(GridSize=7, QuadraticZeeman=True))


@pytest.mark.parametrize('method', ['perturb', 'perturb1', 'perturb2'])
@pytest.mark.parametrize('exp', [FIXED, SWEPT], ids=['fixed', 'swept'])
def test_explicit_perturbation_method_with_ham_raises(method, exp):
    with pytest.raises(ValueError, match=r"cannot treat Sys\.Ham"):
        _run(_sys(Ham=HAM_LIN), exp, Options(GridSize=7, Method=method))


def test_explicit_perturbation_method_without_ham_unchanged():
    """salt otherwise ignores Options.Method (unchanged)."""
    y0 = _run(_sys(g=[G]), FIXED)
    y1 = _run(_sys(g=[G]), FIXED, Options(GridSize=7, Method='perturb1'))
    assert torch.equal(y0, y1)


def test_gradient_on_ham_routed_path():
    """d spec / dA on the matrix path chosen for Ham, vs central differences.

    (On this version Sys.Ham coefficients themselves are constants: ham_ezho
    detaches them. B53 makes them differentiable.)
    """
    exp = ORISEL

    def run(Av):
        return _run(_sys(Ham=HAM_LIN, A=Av), exp)
    A = torch.tensor(A_1H, dtype=torch.float64, requires_grad=True)
    y = run(A)
    assert y.requires_grad
    w = torch.linspace(0.0, 1.0, y.numel(), dtype=torch.float64)
    (gA,) = torch.autograd.grad((y * w).sum(), A)
    # FD round-off dominates below h ~ 1e-4 (relative 1e-4 at h = 1e-5)
    h = 1e-3
    for i in range(3):
        Ap, Am = [list(A_1H[0])], [list(A_1H[0])]
        Ap[0][i] += h
        Am[0][i] -= h
        fd = float(((run(Ap) - run(Am)) * w).sum()) / (2 * h)
        assert abs(float(gA[0, i]) - fd) <= 1e-5 * abs(fd), (i, float(gA[0, i]), fd)


def test_field_swept_rejects_ham_requiring_grad():
    """The field-swept mode is not differentiable; a Ham coefficient that
    requires grad raises like g/A/D/Q/ee, instead of a silently partial gradient
    (only through the stick intensities). The fixed-field mode still works."""
    h110 = torch.tensor(HAM_LIN['110'], dtype=torch.float64, requires_grad=True)
    ham = dict(HAM_LIN, **{'110': h110})
    with pytest.raises(NotImplementedError, match='not differentiable'):
        _run(_sys(Ham=ham), SWEPT)
    y = _run(_sys(Ham=ham), FIXED)
    assert torch.isfinite(y).all() and float(y.max()) > 0


L_MSG = r"salt does not support Sys\.L\."


@pytest.mark.parametrize('exp', [FIXED, SWEPT], ids=['fixed', 'swept'])
def test_orbital_angular_momentum_raises(exp):
    with pytest.raises(ValueError, match=L_MSG):
        _run(_sys(g=[[2.0, 2.0, 2.0]], L=[1], soc=[[100.0]]), exp)


def test_orbital_angular_momentum_raises_matrix_path():
    sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]], Nucs='1H,14N',
                     A=[[3.0, 5.0, 12.0], [1.0, 1.5, 2.0]],
                     Q=[[0.0, 0.0, 0.0], [-0.3, -0.1, 0.4]], L=[1], soc=[[100.0]])
    with pytest.raises(ValueError, match=L_MSG):
        _run(sys, FIXED)
