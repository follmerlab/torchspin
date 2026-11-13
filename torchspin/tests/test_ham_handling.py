"""Sys.Ham handling (0.9.13): B53, B54, B55, B56.

B53  Sys.Ham coefficients are differentiable. ham_ezho detached them before
     0.9.13, so ham(), pepper_autograd and pepper gave no gradient (and pepper
     did not route a system whose only grad input was a Ham coefficient to its
     differentiable path). Values are unchanged.
B54  chili, cardamom and fastmotion build the Zeeman term from Sys.g only and
     silently ignored Sys.Ham; they now raise, as EasySpin's chili does
     (chili.m:160).
B55  pepper's automatic field range divided by g = 0 for a Ham110/Ham112
     system (ZeroDivisionError); EasySpin also fails there (index error at
     pepper.m:324). It now raises a ValueError that says why.
B56  gStrain with a Ham110/Ham112 Zeeman term divided by g = 0 and gave an
     all-zero spectrum (EasySpin's pepper does the same). gStrain is now
     relative to the g-equivalent of the Ham terms. References:
     tests/data/ref_ham_gstrain.mat (generate_ham_gstrain_refs.m), EasySpin
     spectra of the same systems with that g given as Sys.g.
"""

import importlib
import os

import numpy as np
import pytest
import scipy.io as sio
import torch

from torchspin.experiment import Experiment, Options
from torchspin.ham import ham
from torchspin.ham_ezho import ham_ezho
from torchspin.levels import levels
from torchspin.pepper import pepper
from torchspin.pepper_autograd import pepper_autograd
from torchspin.spinsystem import SpinSystem
from torchspin.tests.test_saffron_ham import G, HAM_LIN

hez = importlib.import_module('torchspin.ham_ezho')   # the package exports a function of that name

F64 = torch.float64
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'data')
EXP = Experiment(mwFreq=9.5, Range=[330.0, 350.0], nPoints=512)
EXP_ORI = Experiment(mwFreq=9.5, Range=[330.0, 350.0], nPoints=512,
                     Orientations=[[0.3, 0.7], [1.1, 1.3], [0.0, 0.2]])
KW = dict(S=[0.5], Nucs='1H', A=[[3.0, 5.0, 12.0]], lw=[0.5, 0.0])
W = torch.linspace(-1.0, 1.0, 512, dtype=F64) ** 2
C0 = [HAM_LIN['110'][0][0]] + list(HAM_LIN['112'][0])   # Ham110, Ham112 (q = 2 … -2)
# the same g with principal axes gFrame = [0.3 0.4 0.5] (least squares, residual 4e-16)
GFRAME = [0.3, 0.4, 0.5]
C0_ROT = [-48.589464188057015, -0.013029584283266012, -0.06880297648212826,
          -0.08035836887783973, -0.03825653265796862, 0.03755499926915795]


def _ham(c):
    return {'110': c[:1].reshape(1, 1), '112': c[1:].reshape(1, 5)}


def _c(vals=C0, grad=False):
    return torch.tensor(vals, dtype=F64, requires_grad=grad)


def _central(f, x0, steps):
    return [(f(x0 + h) - f(x0 - h)) / (2 * h) for h in steps]


# ---------------------------------------------------------------------------
# B53: gradients with respect to Sys.Ham coefficients
# ---------------------------------------------------------------------------

def test_ham_values_unchanged_by_requires_grad():
    """Only the graph changes: H and the decomposition are bit-identical with
    and without requires_grad on the coefficients."""
    ham2 = [[1e-4, 0.0, 2e-4, 0.0, 3e-5]]
    s0 = SpinSystem(S=[1.5], Ham={**_ham(_c()), '222': ham2})
    s1 = SpinSystem(S=[1.5], Ham={**_ham(_c(grad=True)), '222': torch.tensor(ham2, dtype=F64, requires_grad=True)})
    for B in ([0.0, 0.0, 340.0], [100.0, -50.0, 300.0]):
        assert torch.equal(ham_ezho(s0, B0=B), ham_ezho(s1, B0=B).detach())
        assert torch.equal(ham(s0, B0=B), ham(s1, B0=B).detach())
    for k, (a, b) in enumerate(zip(ham(s0), ham(s1))):
        assert torch.equal(a, b.detach())
        assert b.requires_grad == (k > 0)      # no lB = 0 term: H0 is a constant


@pytest.mark.parametrize("k", range(6), ids=['110', '112_q2', '112_q1', '112_q0', '112_q-1', '112_q-2'])
def test_ham_gradient_vs_central_differences(k):
    """d/dc of ham(Sys) (H0, mu) and ham(Sys, B0) for each Ham110/Ham112
    coefficient, including those that are zero (q = ±1, -2)."""
    B0 = [120.0, -80.0, 310.0]
    Wm = torch.linspace(0.3, 1.7, 16, dtype=F64).reshape(4, 4)

    def obs(c):
        s = SpinSystem(S=[1.5], Ham=_ham(c))
        _, mx, my, mz = ham(s)
        H = ham(s, B0=B0)
        return sum((Wm * M.real).sum() + (Wm.T * M.imag).sum() for M in (mx, my, mz, H))

    c = _c(grad=True)
    obs(c).backward()
    g = float(c.grad[k])
    e = torch.zeros(6, dtype=F64)
    e[k] = 1.0
    fd = _central(lambda x: float(obs(_c() + (x - C0[k]) * e)), C0[k], (1e-2, 1e-3))
    assert abs(g) > 1e-3
    assert min(abs(g - d) for d in fd) < 1e-9 * abs(g), (g, fd)


def test_ham_quadratic_gradient_vs_central_differences():
    """An lB = 2 coefficient (Ham222) in ham(Sys, B0)."""
    B0 = [120.0, -80.0, 310.0]

    def obs(x):
        s = SpinSystem(S=[1.0], g=[G], Ham={'222': torch.stack([x * 0 + 1e-4, x * 0, x, x * 0, x * 0]).reshape(1, 5)})
        return torch.linalg.eigvalsh(ham(s, B0=B0)) @ torch.tensor([1.0, -2.0, 3.0], dtype=F64)

    x = torch.tensor(2e-4, dtype=F64, requires_grad=True)
    obs(x).backward()
    fd = _central(lambda v: float(obs(torch.tensor(v, dtype=F64))), 2e-4, (1e-5, 1e-6))
    assert abs(float(x.grad)) > 1.0
    assert min(abs(float(x.grad) - d) for d in fd) < 1e-7 * abs(float(x.grad)), (float(x.grad), fd)


def test_levels_ham_gradient():
    Bv = torch.tensor([300.0, 320.0, 340.0], dtype=F64)

    def obs(c):
        _, E = levels(SpinSystem(Ham=_ham(c), **KW), 0.3, 0.7, Bv)
        return (E * torch.arange(1, E.numel() + 1, dtype=F64).reshape(E.shape)).sum()

    c = _c(grad=True)
    obs(c).backward()
    fd = _central(lambda x: float(obs(_c([x] + C0[1:]))), C0[0], (1e-2, 1e-3))
    g = float(c.grad[0])
    assert abs(g) > 1.0
    assert min(abs(g - d) for d in fd) < 1e-8 * abs(g), (g, fd)


def test_pepper_autograd_ham_gradient_vs_central_differences():
    """Crystal orientations (smooth in the parameters): autograd w.r.t. every
    Ham110/Ham112 coefficient against central differences."""
    def L(c):
        return (pepper_autograd(SpinSystem(Ham=_ham(c), **KW), EXP_ORI, Options())[1] * W).sum()

    c = _c(grad=True)
    L(c).backward()
    for k in range(6):
        e = torch.zeros(6, dtype=F64)
        e[k] = 1.0
        steps = (1e-2, 3e-3) if k == 0 else (1e-4, 3e-5)
        fd = _central(lambda x: float(L(_c() + (x - C0[k]) * e)), C0[k], steps)
        g = float(c.grad[k])
        assert abs(g) > 1.0, k
        assert min(abs(g - d) for d in fd) < 1e-8 * abs(g), (k, g, fd)


def test_pepper_autograd_ham_gradient_equals_g_gradient():
    """Powder: d/dc through Ham equals the chain rule through the g-equivalent
    matrix (dL/dg from the same spectrum with g given as Sys.g)."""
    gm = torch.diag(torch.tensor(G, dtype=F64)).requires_grad_(True)
    Lg = (pepper_autograd(SpinSystem(g=gm, **KW), EXP, Options())[1] * W).sum()
    Lg.backward()
    c = _c(grad=True)
    Lh = (pepper_autograd(SpinSystem(Ham=_ham(c), **KW), EXP, Options())[1] * W).sum()
    Lh.backward()
    assert abs(float(Lh.detach()) - float(Lg.detach())) < 1e-11 * abs(float(Lg.detach()))
    J = torch.autograd.functional.jacobian(lambda cc: hez.ham_linear_g(SpinSystem(S=[0.5], Ham=_ham(cc)), 0), _c())
    chain = torch.einsum('ij,ijk->k', gm.grad, J)
    assert torch.allclose(c.grad, chain, rtol=1e-9, atol=1e-9 * float(chain.abs().max()))


def test_pepper_autograd_quadratic_zeeman_gradient():
    """An lB = 2 coefficient (Ham222, S = 1) with Options.QuadraticZeeman."""
    kw1 = dict(S=[1.0], g=[[2.0, 2.0, 2.0]], D=[[300.0, 30.0]], lw=[2.0, 0.0])
    exp1 = Experiment(mwFreq=9.5, Range=[250.0, 430.0], nPoints=512)
    opt = Options(QuadraticZeeman=True)

    def L(x):
        z = torch.zeros((), dtype=F64)
        H = {'222': torch.stack([z + 2e-4, z, x, z, z]).reshape(1, 5)}
        return (pepper_autograd(SpinSystem(Ham=H, **kw1), exp1, opt)[1] * W).sum()

    x = torch.tensor(3e-4, dtype=F64, requires_grad=True)
    L(x).backward()
    g = float(x.grad)
    fd = _central(lambda v: float(L(torch.tensor(v, dtype=F64))), 3e-4, (1e-6, 3e-7))
    assert abs(g) > 1.0
    assert min(abs(g - d) for d in fd) < 1e-4 * abs(g), (g, fd)


def test_pepper_routes_ham_grad_to_autograd():
    """A system whose only grad input is a Ham coefficient goes to the
    differentiable path (before 0.9.13 pepper returned a detached spectrum)."""
    c = _c(grad=True)
    s = SpinSystem(Ham=_ham(c), **KW)
    _, y = pepper(s, EXP, Options())
    assert torch.is_tensor(y) and y.requires_grad
    (y * W).sum().backward()
    assert c.grad is not None and bool(torch.all(torch.isfinite(c.grad)))
    y0 = np.asarray(pepper(SpinSystem(Ham=_ham(_c()), **KW), EXP, Options())[1])
    assert np.max(np.abs(y.detach().numpy() - y0)) < 1e-12 * np.max(np.abs(y0))


# ---------------------------------------------------------------------------
# B54: chili, cardamom, fastmotion reject Sys.Ham
# ---------------------------------------------------------------------------

HAMS = [HAM_LIN, {'222': [[1e-4, 0.0, 2e-4, 0.0, 0.0]]}]


def _slow(Ham, **kw):
    d = dict(S=[0.5], Nucs='14N', A=[[20.0, 20.0, 90.0]], tcorr=1e-9, lw=[0.3, 0.0])
    d.update(kw)
    if 'g' not in d and not any(k in ('110', '112') for k in Ham):
        d['g'] = [[2.008, 2.006, 2.003]]
    return SpinSystem(Ham=Ham, **d)


@pytest.mark.parametrize("Ham", HAMS, ids=['Ham110+112', 'Ham222'])
def test_slow_motion_raise_for_ham(Ham):
    from torchspin.cardamom import CardamomPar, cardamom
    from torchspin.chili import chili
    from torchspin.chili_sle import chili_sle
    from torchspin.fastmotion import fastmotion, fastmotion_t
    exp = Experiment(mwFreq=9.5, Range=[325.0, 350.0], nPoints=256)
    sys = _slow(Ham) if '222' not in Ham else _slow(Ham, S=[1.0], D=[[300.0, 0.0]])
    with pytest.raises(ValueError, match=r"chili does not support Sys\.Ham\* parameters"):
        chili(sys, exp)
    with pytest.raises(ValueError, match=r"chili does not support Sys\.Ham\* parameters"):
        chili_sle(sys, exp)
    with pytest.raises(ValueError, match=r"cardamom does not support Sys\.Ham\* parameters"):
        cardamom(sys, exp, CardamomPar(nTraj=2, nSteps=10))
    with pytest.raises(ValueError, match=r"fastmotion does not support Sys\.Ham\* parameters"):
        fastmotion(sys, 340.0, 1e-10)
    with pytest.raises(ValueError, match=r"fastmotion does not support Sys\.Ham\* parameters"):
        fastmotion_t(sys, 340.0, 1e-10)


def test_slow_motion_without_ham_unchanged():
    from torchspin.chili import chili
    from torchspin.fastmotion import fastmotion
    sys = SpinSystem(S=[0.5], g=[[2.008, 2.006, 2.003]], Nucs='14N', A=[[20.0, 20.0, 90.0]], tcorr=1e-9, lw=[0.3, 0.0])
    _, y = chili(sys, Experiment(mwFreq=9.5, Range=[325.0, 350.0], nPoints=256))
    assert float(np.max(np.abs(np.asarray(y)))) > 0
    lw, _ = fastmotion(sys, 340.0, 1e-10)
    assert np.all(np.isfinite(lw)) and np.all(lw > 0)


# ---------------------------------------------------------------------------
# B55: pepper's automatic field range with a Ham-only Zeeman term
# ---------------------------------------------------------------------------

def test_pepper_auto_range_raises_for_ham_zeeman():
    exp = Experiment(mwFreq=9.5, nPoints=512)
    with pytest.raises(ValueError, match=r"Cannot automatically determine field range.*Sys\.Ham\* \(110, 112\)"):
        pepper(SpinSystem(S=[0.5], Ham=HAM_LIN, lw=0.5), exp)
    with pytest.raises(ValueError, match=r"Cannot automatically determine field range.*Sys\.Ham\*"):
        pepper(SpinSystem(Ham=HAM_LIN, **KW), exp)
    with pytest.raises(ValueError, match=r"a principal g value is zero"):
        pepper(SpinSystem(S=[0.5], g=[2.0, 2.0, 0.0], lw=0.5), exp)


def test_pepper_auto_range_with_g_unchanged():
    """EasySpin's auto range for g = [2.01 2.005 1.998], lw = 0.5:
    [336.497281, 340.907478] mT (pepper.m:241-272)."""
    x, y = pepper(SpinSystem(S=[0.5], g=G, lw=0.5), Experiment(mwFreq=9.5, nPoints=512))
    assert abs(float(x[0]) - 336.497281) < 1e-6 and abs(float(x[-1]) - 340.907478) < 1e-6
    assert float(np.max(np.abs(np.asarray(y)))) > 0


# ---------------------------------------------------------------------------
# B56: gStrain with a Ham-only Zeeman term
# ---------------------------------------------------------------------------

def test_ham_linear_g():
    gm = hez.ham_linear_g(SpinSystem(S=[0.5], Ham=HAM_LIN), 0)
    assert torch.allclose(gm, torch.diag(torch.tensor(G, dtype=F64)), atol=1e-14)
    assert torch.allclose(hez.ham_linear_g(SpinSystem(S=[1.5], Ham=HAM_LIN), 0), gm, atol=1e-14)
    from torchspin.rotations import erot
    R = erot(GFRAME).T.to(F64)
    gm = hez.ham_linear_g(SpinSystem(S=[0.5], Ham=_ham(_c(C0_ROT))), 0)
    assert torch.allclose(gm, R @ torch.diag(torch.tensor(G, dtype=F64)) @ R.T, atol=1e-14)
    assert hez.ham_linear_g(SpinSystem(S=[0.5], g=G), 0) is None


def _ref_cases():
    d = sio.loadmat(os.path.join(DATA_DIR, 'ref_ham_gstrain.mat'), squeeze_me=True, struct_as_record=False)
    return {c.name: c for c in np.atleast_1d(d['cases'])}


def _gstrain_case(name):
    c = _ref_cases()[name]
    H = c.SysH
    kw = dict(S=[0.5], lw=float(H.lw), gStrain=list(np.atleast_1d(H.gStrain).astype(float)))
    if hasattr(H, 'gFrame'):
        kw['gFrame'] = list(np.atleast_1d(H.gFrame).astype(float))
    if hasattr(H, 'Nucs'):
        kw['Nucs'] = str(H.Nucs)
        kw['A'] = [list(np.atleast_1d(H.A).astype(float))]
    Ham = {'110': [[float(H.Ham110)]], '112': [list(np.atleast_1d(H.Ham112).astype(float))]}
    exp = Experiment(mwFreq=float(c.Exp.mwFreq), Range=list(np.atleast_1d(c.Exp.Range).astype(float)),
                     nPoints=int(c.Exp.nPoints))
    return c, SpinSystem(Ham=Ham, **kw), SpinSystem(g=G, **kw), exp


def _cos(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


@pytest.mark.parametrize("name", ['diag', 'gframe', 'diag_1H'])
def test_pepper_gstrain_ham_matches_g(name):
    """Ham110/Ham112 + gStrain equals the same system with g given as Sys.g.
    With a gFrame the automatic grid (B82, 0.9.24) is Ci for both, but in
    different frames (Ham vs g 1.0e-3 of the maximum), so that case compares
    them on an explicit Ci grid (7.8e-14)."""
    c, s_h, s_g, exp = _gstrain_case(name)
    opt = Options(GridSymmetry='Ci') if name == 'gframe' else Options()
    y_h = np.asarray(pepper(s_h, exp, opt)[1])
    y_g = np.asarray(pepper(s_g, exp, opt)[1])
    assert np.max(np.abs(y_h - y_g)) < 1e-11 * np.max(np.abs(y_g))
    # EasySpin's Ham spectrum without strain equals its g spectrum (the Ham
    # coefficients are right); with gStrain it is not the g spectrum.
    ref = np.asarray(c.spc, dtype=float)
    assert np.max(np.abs(np.asarray(c.spcHam) - ref)) > 0.1 * np.max(np.abs(ref))


# B82 (0.9.24): with gStrain and a gFrame EasySpin's automatic grid ignores the
# strain and is under-symmetric; torchspin picks Ci.  Against EasySpin's g
# spectrum: cosine 0.997209, max deviation 0.125 of the maximum (it was
# > 0.9999999 when both used EasySpin's grid).  test_pepper_gstrain_ham_gframe_
# converged_reference checks the default against a converged grid.
_B82_XFAIL = pytest.mark.xfail(
    strict=True, reason='B82: EasySpin\'s automatic grid ignores gStrain with a gFrame '
                        '(KNOWN_LIMITATIONS, automatic grid symmetry and strains)')


@pytest.mark.parametrize("name", ['diag', pytest.param('gframe', marks=_B82_XFAIL), 'diag_1H'])
def test_pepper_gstrain_ham_vs_easyspin(name):
    """Ham110/Ham112 + gStrain against EasySpin's spectrum of the g system.
    EasySpin itself gives zero for the Ham system (diag cases) or a wrong
    spectrum (gFrame case)."""
    c, s_h, _, exp = _gstrain_case(name)
    y_h = np.asarray(pepper(s_h, exp)[1])
    ref = np.asarray(c.spc, dtype=float)
    cos = _cos(y_h, ref)
    assert cos > 0.9999999, cos
    assert np.max(np.abs(y_h - ref)) < 3e-4 * np.max(np.abs(ref))


def test_pepper_gstrain_ham_gframe_converged_reference():
    """gFrame case on the default grid against a no-interpolation C1 [91 1] grid
    (which agrees with C1 [181 1] to 6.7e-9 in 1 - cos): 1 - cos 1.070e-7 (Ham)
    and 1.058e-7 (g); EasySpin's spectrum 2.777e-3."""
    c, s_h, s_g, exp = _gstrain_case('gframe')
    y_ref = np.asarray(pepper(s_g, exp, Options(GridSymmetry='C1', GridSize=[91, 1]))[1])
    for s in (s_h, s_g):
        assert 1 - _cos(np.asarray(pepper(s, exp)[1]), y_ref) < 1.2e-7
    assert 1 - _cos(np.asarray(c.spc, dtype=float), y_ref) > 1e-3     # EasySpin's grid


def test_pepper_autograd_gstrain_ham_matches_g():
    kw = dict(S=[0.5], lw=0.5, gStrain=[0.01, 0.005, 0.002])
    c = _c(grad=True)
    y_h = pepper_autograd(SpinSystem(Ham=_ham(c), **kw), EXP, Options())[1]
    y_g = pepper_autograd(SpinSystem(g=G, **kw), EXP, Options())[1]
    assert float((y_h - y_g).abs().max()) < 1e-11 * float(y_g.abs().max())
    (y_h * W).sum().backward()
    assert bool(torch.all(torch.isfinite(c.grad))) and float(c.grad.abs().max()) > 0


def test_resfields_gstrain_widths_ham_match_g():
    """Single-orientation resfields strain widths (strainwidth.compute_strain_widths)."""
    from torchspin.resfields import resfields
    kw = dict(S=[0.5], lw=0.5, gStrain=[0.01, 0.005, 0.002], gFrame=GFRAME)
    s_h = SpinSystem(Ham=_ham(_c(C0_ROT)), **kw)
    s_g = SpinSystem(g=G, **kw)
    out_h = resfields(*ham(s_h), 0.3, 0.7, EXP, Options(), sys=s_h)
    out_g = resfields(*ham(s_g), 0.3, 0.7, EXP, Options(), sys=s_g)
    assert out_g[2] is not None and float(out_g[2].min()) > 0
    assert torch.allclose(out_h[0], out_g[0], rtol=1e-12)
    assert torch.allclose(out_h[2], out_g[2], rtol=1e-10)


def test_gstrain_ham_not_diagonal_in_gframe_raises():
    """As gStrain with a full g matrix: the g-equivalent of the Ham terms must
    be diagonal in Sys.gFrame."""
    with pytest.raises(ValueError, match=r"gStrain with a linear Zeeman term given as Sys\.Ham110/Ham112"):
        pepper(SpinSystem(S=[0.5], Ham=_ham(_c(C0_ROT)), lw=0.5, gStrain=[0.01, 0.005, 0.002]), EXP)
    with pytest.raises(ValueError, match=r"gStrain with a linear Zeeman term given as Sys\.Ham110/Ham112"):
        pepper(SpinSystem(S=[0.5], Ham=HAM_LIN, lw=0.5, gStrain=[0.01, 0.005, 0.002], gFrame=GFRAME), EXP)
