"""pepper_autograd with Exp.Orientations honors the resonance-search options that
pepper honors (B64, 0.9.16): Options.QuadraticZeeman (field-quadratic Sys.Ham2xx
terms) and Options.Transitions.  Before 0.9.16 the orientation branch of
pepper_autograd called resfields_batch without ``zeeman_ho`` and ``pairs``, so its
spectrum equaled the one without the option.

Checks
------
* forward: pepper_autograd equals pepper (relative max 1e-9, the agreement of
  test_pepper_orientations.py) and differs from the spectrum without the option;
* Method='perturb' with either option raises on the orientation path too;
* EasySpin: tests/data/ref_pepper_orientations_qz.mat (chi-averaged crystal sums,
  generate_pepper_orientations_qz_refs.m);
* gradients w.r.t. D, g and the orientation weights vs central differences.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from torchspin import Experiment, Options, SpinSystem, pepper
from torchspin.pepper_autograd import pepper_autograd

REF = Path(__file__).resolve().parents[2] / 'tests' / 'data' / 'ref_pepper_orientations_qz.mat'

HAM = {'220': [[2e-4]], '222': [[2e-4, 0.0, 3e-4, 0.0, 0.0]]}
S1 = dict(S=[1.0], g=[2.0, 2.02, 2.04], D=[[300.0, 30.0]])
ORI1 = np.array([[0.7, 1.1]])
ORI5 = np.array([[0.3, 0.7], [1.9, 2.2], [4.0, 1.1], [5.5, 0.4], [2.7, 1.6]])
W5 = np.array([0.1, 0.3, 0.2, 0.25, 0.15])


def _relmax(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return float(np.abs(a - b).max() / np.abs(b).max())


def _exp(ori, harmonic=1, **kw):
    return Experiment(mwFreq=9.5, Range=[150, 500], nPoints=600, Harmonic=harmonic, Orientations=ori, **kw)


# ---------------------------------------------------------------------------
# forward: pepper_autograd == pepper
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('ori', [ORI1, ORI5], ids=['ori1', 'ori5'])
@pytest.mark.parametrize('sysk, harmonic', [(dict(lw=[1.0, 0.0]), 1), (dict(lw=[0.0, 1.2]), 0),
                                            (dict(HStrain=[40, 50, 60]), 0)], ids=['gauss', 'lorentz', 'hstrain'])
def test_quadratic_zeeman_autograd_equals_pepper(ori, sysk, harmonic):
    sys = SpinSystem(**S1, Ham=HAM, **sysk)
    e = _exp(ori, harmonic, OrientationWeights=W5 if len(ori) == 5 else None)
    on = Options(QuadraticZeeman=True)
    _, p = pepper(sys, e, on)
    _, a = pepper_autograd(sys, e, on)
    assert _relmax(a.detach(), p) < 1e-9
    _, p_off = pepper(sys, e, Options())
    assert _relmax(p_off, p) > 0.05          # the option matters here


@pytest.mark.parametrize('ori', [ORI1, ORI5], ids=['ori1', 'ori5'])
@pytest.mark.parametrize('opt', [Options(Transitions=[[1, 2]]), Options(Transitions=[[2, 3]], Threshold=0.3),
                                 Options(Transitions='all', Threshold=0.3)], ids=['12', '23', 'all'])
def test_transitions_autograd_equals_pepper(ori, opt):
    sys = SpinSystem(**S1, lw=[1.0, 0.0])
    e = _exp(ori)
    _, p = pepper(sys, e, opt)
    _, a = pepper_autograd(sys, e, opt)
    assert _relmax(a.detach(), p) < 1e-9


def test_transitions_select_on_orientation_path():
    """One level pair is a strict subset of the default spectrum."""
    sys = SpinSystem(**S1, lw=[1.0, 0.0])
    e = _exp(ORI5)
    _, a = pepper_autograd(sys, e, Options(Transitions=[[1, 2]]))
    _, a_all = pepper_autograd(sys, e, Options())
    assert _relmax(a.detach(), a_all.detach()) > 0.1


@pytest.mark.parametrize('opt, exc', [(Options(QuadraticZeeman=True, Method='perturb'), NotImplementedError),
                                      (Options(Transitions=[[1, 2]], Method='perturb'), ValueError)])
def test_perturb_with_option_raises_on_orientation_path(opt, exc):
    sys = SpinSystem(**S1, Ham=HAM, lw=[1.0, 0.0])
    with pytest.raises(exc):
        pepper(sys, _exp(ORI5), opt)
    with pytest.raises(exc):
        pepper_autograd(sys, _exp(ORI5), opt)


# ---------------------------------------------------------------------------
# EasySpin
# ---------------------------------------------------------------------------

def _load_cases():
    from scipy.io import loadmat
    m = loadmat(REF, squeeze_me=True, struct_as_record=False)
    return {str(c.name): c for c in np.atleast_1d(m['cases'])}


def _case_names():
    return list(_load_cases()) if REF.exists() else []


@pytest.mark.parametrize('fn', [pepper, pepper_autograd], ids=['pepper', 'pepper_autograd'])
@pytest.mark.parametrize('name', _case_names())
def test_orientations_vs_easyspin(name, fn):
    """EasySpin's crystal sum over two B1 directions per orientation (an exact chi
    average). With Ham2xx terms EasySpin's dB/dE uses the linear moment only
    (torchspin: the full slope), which changes amplitudes by up to 0.25 % here
    (see test_quadratic_zeeman.py); line positions agree."""
    c = _load_cases()[name]
    S = c.Sys; f = S._fieldnames
    kw = dict(S=[float(S.S)], g=np.atleast_1d(S.g).astype(float).tolist(),
              D=[np.atleast_1d(S.D).astype(float).tolist()])
    ham_keys = [k for k in f if k.startswith('Ham')]
    if ham_keys:
        kw['Ham'] = {k[3:]: np.atleast_1d(getattr(S, k)).astype(float).reshape(1, -1).tolist() for k in ham_keys}
    if 'lwpp' in f:
        kw['lwpp'] = [float(S.lwpp), 0.0]
    if 'HStrain' in f:
        kw['HStrain'] = np.atleast_1d(S.HStrain).astype(float).tolist()
    E = c.Exp
    exp = Experiment(mwFreq=float(E.mwFreq), Range=np.atleast_1d(E.Range).astype(float).tolist(),
                     nPoints=int(E.nPoints), Harmonic=int(E.Harmonic),
                     Orientations=np.atleast_2d(np.asarray(c.ori, dtype=float)))
    okw = dict(QuadraticZeeman=bool(ham_keys))
    if 'Transitions' in c.Opt._fieldnames:
        okw['Transitions'] = np.atleast_2d(c.Opt.Transitions).astype(int).tolist()
    x, y = fn(SpinSystem(**kw), exp, Options(**okw))
    ref = np.asarray(c.spc, dtype=float)
    assert np.abs(x.numpy() - np.asarray(c.x, dtype=float)).max() < 1e-9
    tol = 3e-3 if ham_keys else 5e-4         # measured 2.5e-3 (dB/dE, above) and 1.9e-4
    assert _relmax(y.detach(), ref) < tol


# ---------------------------------------------------------------------------
# gradients
# ---------------------------------------------------------------------------

def test_quadratic_zeeman_gradient_vs_central_differences():
    exp = Experiment(mwFreq=9.5, Range=[150, 500], nPoints=512, Harmonic=1, Orientations=ORI5)
    opt = Options(QuadraticZeeman=True)
    D = torch.tensor([[300.0, 30.0]], dtype=torch.float64, requires_grad=True)
    g = torch.tensor([2.0, 2.02, 2.04], dtype=torch.float64, requires_grad=True)
    w = torch.tensor(W5, dtype=torch.float64, requires_grad=True)
    _, y = pepper_autograd(SpinSystem(S=[1.0], g=g, D=D, Ham=HAM, lw=[1.5, 0.0]),
                           replace(exp, OrientationWeights=w), opt)
    wv = torch.tensor(np.random.default_rng(0).normal(size=y.numel()))
    gD, gg, gw = torch.autograd.grad((y * wv).sum(), [D, g, w])

    D0 = [[300.0, 30.0]]; g0 = [2.0, 2.02, 2.04]

    def loss(Dv, gv, wts=W5):
        ss = SpinSystem(S=[1.0], g=gv, D=Dv, Ham=HAM, lw=[1.5, 0.0])
        return float((pepper(ss, replace(exp, OrientationWeights=wts), opt)[1] * wv).sum())
    for idx, an, (hs, mk) in (
        (0, float(gD[0, 0]), ((1e-2, 3e-3, 1e-3), lambda h: ([[300.0 + h, 30.0]], g0))),
        (1, float(gD[0, 1]), ((1e-2, 3e-3, 1e-3), lambda h: ([[300.0, 30.0 + h]], g0))),
        (2, float(gg[2]), ((1e-6, 3e-7, 1e-7), lambda h: (D0, [2.0, 2.02, 2.04 + h]))),
    ):
        errs = []
        for h in hs:
            fd = (loss(*mk(h)) - loss(*mk(-h))) / (2 * h)
            errs.append(abs(fd - an) / max(abs(fd), 1e-30))
        assert min(errs) < 1e-6, (idx, an, errs)
    for i in range(5):
        h = 1e-4
        wp, wm = W5.copy(), W5.copy(); wp[i] += h; wm[i] -= h
        fd = (loss(D0, g0, wp) - loss(D0, g0, wm)) / (2 * h)
        assert abs(fd - float(gw[i])) <= 1e-7 * max(abs(fd), 1.0), (i, fd, float(gw[i]))
