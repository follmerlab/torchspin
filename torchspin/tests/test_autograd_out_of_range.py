"""B77: pepper_autograd and salt with no resonance in range stay on the autograd graph.

Before 0.9.22 pepper_autograd returned a detached zero spectrum when no line of a
system fell inside Exp.Range (powder early return, isotropic/O3 and Exp.Orientations
paths), and a component without lines in range was not used in the graph of a
multi-component sum. torch.autograd.grad then raised ("does not require grad" or
"appears to not have been used in the graph"), so a fit whose lines drifted out of
the window crashed. salt's fixed-field mode did the same with no ENDOR line in
freq_range. Every differentiable input now gets an exact zero gradient there, the
spectrum values are unchanged bit for bit, and inputs the forward path detaches
(gnscale, Exp.Range, Exp.Field, ...) still raise instead of getting a silent zero.
The spin-system Euler frames (on the graph since 0.9.17, B3) and Exp.ModAmp
(since 0.9.27, B92) are connected like the other differentiable inputs.
"""
import importlib

import pytest
import torch

from torchspin import SpinSystem, Experiment, Options, pepper
from torchspin.salt import salt
import torchspin.autograd as ts_autograd

pa = importlib.import_module('torchspin.pepper_autograd')

F64 = torch.float64
OUT = dict(mwFreq=9.5, Range=[100.0, 120.0], nPoints=256, Harmonic=1)   # g ~ 2 lines at ~340 mT
IN = dict(mwFreq=9.5, Range=[300.0, 360.0], nPoints=256, Harmonic=1)
GRID = Options(GridSize=[19, 4])


def T(v):
    return torch.tensor(v, dtype=F64, requires_grad=True)


def _assert_zero_grads(y, params):
    assert y.requires_grad
    assert float(y.detach().abs().max()) == 0.0
    grads = torch.autograd.grad((y ** 2).sum(), params)
    for p, g in zip(params, grads):
        assert g is not None and g.shape == p.shape
        assert torch.all(g == 0.0), g


# ---------------------------------------------------------------------------
# pepper_autograd / pepper, no line in Exp.Range
# ---------------------------------------------------------------------------

def _powder_rhombic():
    g = T([2.0, 2.1, 2.2])
    return SpinSystem(S=[0.5], g=[g], lw=[1.0, 0.0]), Experiment(**OUT), GRID, [g]


def _powder_rhombic_lw():
    g, lw = T([2.0, 2.1, 2.2]), T(1.0)
    return SpinSystem(S=[0.5], g=[g], lw=[lw, 0.0]), Experiment(**OUT), GRID, [g, lw]


def _isotropic():
    g = T([2.0, 2.0, 2.0])
    return SpinSystem(S=[0.5], g=[g], lw=[1.0, 0.0]), Experiment(**OUT), GRID, [g]


def _isotropic_voigt_lw():
    g, lwg, lwl = T([2.0, 2.0, 2.0]), T(1.0), T(0.5)
    return SpinSystem(S=[0.5], g=[g], lw=[lwg, lwl]), Experiment(**OUT), GRID, [g, lwg, lwl]


def _lwpp():
    g, lwpp = T([2.0, 2.1, 2.2]), T([1.0, 0.0])
    return SpinSystem(S=[0.5], g=[g], lwpp=lwpp), Experiment(**OUT), GRID, [g, lwpp]


def _hstrain():
    g, H = T([2.0, 2.1, 2.2]), T([50.0, 60.0, 70.0])
    return SpinSystem(S=[0.5], g=[g], HStrain=H), Experiment(**OUT), GRID, [g, H]


def _gstrain():
    g, gs = T([2.0, 2.1, 2.2]), T([0.01, 0.02, 0.01])
    return SpinSystem(S=[0.5], g=[g], gStrain=gs, lw=[0.5, 0.0]), Experiment(**OUT), GRID, [g, gs]


def _hyperfine_astrain():
    g, A, As = T([2.0, 2.1, 2.2]), T([[10.0, 20.0, 30.0]]), T([5.0, 5.0, 5.0])
    s = SpinSystem(S=[0.5], g=[g], Nucs='14N', A=A, AStrain=As, gStrain=[0.01, 0.01, 0.01], lw=[0.5, 0.0])
    return s, Experiment(**OUT), GRID, [g, A, As]


def _triplet_d():
    D = T([300.0, 30.0])
    s = SpinSystem(S=[1.0], g=[2.0, 2.0, 2.0], D=D, lw=[1.0, 0.0])
    return s, Experiment(**OUT), GRID, [D]


def _orientations():
    g, w = T([2.0, 2.1, 2.2]), T([0.5, 0.5])
    e = Experiment(**OUT, Orientations=[[0.0, 0.0], [0.5, 0.7]], OrientationWeights=w)
    return SpinSystem(S=[0.5], g=[g], lw=[1.0, 0.0]), e, Options(), [g, w]


def _perturb():
    g, A = T([2.0, 2.1, 2.2]), T([[10.0, 20.0, 30.0]])
    s = SpinSystem(S=[0.5], g=[g], Nucs='1H', A=A, lw=[1.0, 0.0])
    return s, Experiment(**OUT), Options(GridSize=[19, 4], Method='perturb'), [g, A]


def _temperature_mwfreq():
    g, Temp, mw = T([2.0, 2.1, 2.2]), T(10.0), T(9.5)
    e = Experiment(mwFreq=mw, Range=[100.0, 120.0], nPoints=256, Harmonic=1, Temperature=Temp)
    return SpinSystem(S=[0.5], g=[g], lw=[1.0, 0.0]), e, GRID, [g, Temp, mw]


def _two_components_weight():
    g1, g2, w = T([2.0, 2.1, 2.2]), T([2.0, 2.0, 2.0]), T(0.7)
    comps = [SpinSystem(S=[0.5], g=[g1], lw=[1.0, 0.0], weight=w), SpinSystem(S=[0.5], g=[g2], lw=[1.0, 0.0])]
    return comps, Experiment(**OUT), GRID, [g1, g2, w]


def _exchange_ee():
    ee = T([100.0, 110.0, 120.0])
    s = SpinSystem(S=[0.5, 0.5], g=[[2.0, 2.02, 2.04], [2.1, 2.12, 2.14]], ee=ee, lw=[1.0, 0.0])
    return s, Experiment(**OUT), GRID, [ee]


def _frames():
    g, gF, AF = T([2.0, 2.1, 2.2]), T([0.3, 0.4, 0.5]), T([0.2, -0.3, 0.1])
    s = SpinSystem(S=[0.5], g=[g], gFrame=gF, Nucs='1H', A=[10.0, 20.0, 30.0], AFrame=AF, lw=[1.0, 0.0])
    return s, Experiment(**OUT), GRID, [g, gF, AF]


def _modamp():
    g, ma = T([2.0, 2.1, 2.2]), T(0.5)
    return SpinSystem(S=[0.5], g=[g], lw=[1.0, 0.0]), Experiment(**OUT, ModAmp=ma), GRID, [g, ma]


def _natural_abundance():
    g, A = T([2.05, 2.05, 2.25]), T([[30.0, 30.0, 600.0]])
    return SpinSystem(S=[0.5], g=[g], Nucs='Cu', A=A, lw=[1.0, 0.0]), Experiment(**OUT), GRID, [g, A]


OUT_CASES = {
    'powder_rhombic': _powder_rhombic, 'powder_rhombic_lw': _powder_rhombic_lw,
    'isotropic': _isotropic, 'isotropic_voigt_lw': _isotropic_voigt_lw, 'lwpp': _lwpp,
    'hstrain': _hstrain, 'gstrain': _gstrain, 'hyperfine_astrain': _hyperfine_astrain,
    'triplet_d': _triplet_d, 'orientations': _orientations, 'perturb': _perturb,
    'temperature_mwfreq': _temperature_mwfreq, 'two_components_weight': _two_components_weight,
    'exchange_ee': _exchange_ee, 'natural_abundance': _natural_abundance,
    'frames': _frames, 'modamp': _modamp,
}


@pytest.mark.parametrize('name', list(OUT_CASES))
def test_pepper_autograd_out_of_range_zero_gradients(name):
    sys, exp, opt, params = OUT_CASES[name]()
    x, y = pa.pepper_autograd(sys, exp, opt)
    assert y.shape == (256,)
    _assert_zero_grads(y, params)


@pytest.mark.parametrize('name', ['powder_rhombic', 'isotropic', 'orientations', 'two_components_weight'])
def test_pepper_routes_out_of_range_zero_gradients(name):
    sys, exp, opt, params = OUT_CASES[name]()
    _, y = pepper(sys, exp, opt)
    _assert_zero_grads(torch.as_tensor(y), params)


def test_frequency_sweep_still_raises():
    g = T([2.0, 2.1, 2.2])
    sys = SpinSystem(S=[0.5], g=[g], lw=[1.0, 0.0])
    exp = Experiment(Field=350.0, mwRange=[1.0, 2.0], nPoints=256, Harmonic=1)
    with pytest.raises(NotImplementedError):
        pa.pepper_autograd(sys, exp, GRID)
    with pytest.raises(NotImplementedError):
        pepper(sys, exp, GRID)


# ---------------------------------------------------------------------------
# One component in range, one out: the in-range gradients are unchanged
# ---------------------------------------------------------------------------

def _mixed(g2_grad):
    g1, lw1 = T([2.0, 2.1, 2.2]), T(1.0)
    g2 = T([4.3, 4.3, 4.3]) if g2_grad else torch.tensor([4.3, 4.3, 4.3], dtype=F64)   # g2 lines ~158 mT
    comps = [SpinSystem(S=[0.5], g=[g1], lw=[lw1, 0.0]), SpinSystem(S=[0.5], g=[g2], lw=[1.0, 0.0])]
    return comps, (g1, lw1, g2)


def test_one_component_out_of_range():
    exp = Experiment(**IN)
    comps, (g1, lw1, g2) = _mixed(True)
    y = pa.pepper_autograd(comps, exp, GRID)[1]
    dg1, dlw1, dg2 = torch.autograd.grad((y ** 2).sum(), [g1, lw1, g2])
    assert torch.all(dg2 == 0.0) and dg2.shape == g2.shape
    assert float(dg1.abs().max()) > 0 and float(dlw1.abs()) > 0

    # same gradients as without the out-of-range grad input (the path of 0.9.21)
    comps0, (h1, hlw1, _) = _mixed(False)
    y0 = pa.pepper_autograd(comps0, exp, GRID)[1]
    eg1, elw1 = torch.autograd.grad((y0 ** 2).sum(), [h1, hlw1])
    assert torch.equal(y.detach(), y0.detach())
    assert torch.equal(dg1, eg1) and torch.equal(dlw1, elw1)

    # and they still match central differences of pepper (no grad inputs)
    def loss(lw):
        c = [SpinSystem(S=[0.5], g=[2.0, 2.1, 2.2], lw=[lw, 0.0]), SpinSystem(S=[0.5], g=[4.3, 4.3, 4.3], lw=[1.0, 0.0])]
        return float((torch.as_tensor(pepper(c, exp, GRID)[1]) ** 2).sum())
    h = 1e-5
    fd = (loss(1.0 + h) - loss(1.0 - h)) / (2 * h)
    assert abs(float(dlw1) - fd) / abs(fd) < 1e-6


# ---------------------------------------------------------------------------
# Values: the connecting term changes no bit
# ---------------------------------------------------------------------------

def _identity(spec, *args, **kw):
    return spec


@pytest.mark.parametrize('rng', [IN, OUT])
@pytest.mark.parametrize('with_grad', [True, False])
def test_pepper_autograd_values_bit_identical(monkeypatch, rng, with_grad):
    def build():
        mk = T if with_grad else (lambda v: torch.tensor(v, dtype=F64))
        g, lw, A = mk([2.0, 2.1, 2.2]), mk(1.0), mk([[10.0, 20.0, 30.0]])
        comps = [SpinSystem(S=[0.5], g=[g], Nucs='1H', A=A, lw=[lw, 0.0], weight=mk(0.6)),
                 SpinSystem(S=[0.5], g=[4.3, 4.3, 4.3], lw=[1.0, 0.0])]
        return comps
    exp = Experiment(**rng)
    y = pa.pepper_autograd(build(), exp, GRID)[1].detach()
    monkeypatch.setattr(ts_autograd, 'connect_inputs', _identity)
    y_ref = pa.pepper_autograd(build(), exp, GRID)[1].detach()
    assert torch.equal(y, y_ref)
    assert torch.equal(torch.signbit(y), torch.signbit(y_ref))


def test_pepper_values_without_grad_bit_identical(monkeypatch):
    sys = SpinSystem(S=[0.5], g=[2.0, 2.1, 2.2], Nucs='1H', A=[10.0, 20.0, 30.0], lw=[1.0, 0.0])
    exp = Experiment(**IN)
    y = torch.as_tensor(pepper(sys, exp, GRID)[1])
    monkeypatch.setattr(ts_autograd, 'connect_inputs', _identity)
    assert torch.equal(y, torch.as_tensor(pepper(sys, exp, GRID)[1]))


def test_connect_inputs_exact_zero_term():
    y = torch.tensor([-0.0, 0.0, 1.5, -2.25e-300], dtype=F64)
    t = T([1.0, float('inf'), float('nan')])
    s = SpinSystem(S=[0.5], g=[2.0, 2.0, 2.0], lw=[T(1.0), 0.0])
    z = ts_autograd.connect_inputs(y, [s], ('g', 'lw'), t)
    assert torch.equal(z.detach(), y) and torch.equal(torch.signbit(z), torch.signbit(y))
    gt, glw = torch.autograd.grad(z.sum(), [t, s.lw[0]])
    assert torch.all(gt == 0.0) and torch.all(glw == 0.0)
    # nothing to connect, or no grad mode: the same tensor comes back
    assert ts_autograd.connect_inputs(y, [s], ('g',)) is y
    with torch.no_grad():
        assert ts_autograd.connect_inputs(y, [s], ('lw',), t) is y


# ---------------------------------------------------------------------------
# Inputs the forward path detaches still fail loudly (no silent zero gradient)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('where,field,val', [
    ('sys', 'gnscale', [1.0]),
    ('exp', 'Range', [300.0, 360.0]),
])
def test_detached_inputs_still_raise(where, field, val):
    t = T(val)
    s = dict(S=[0.5], g=[2.0, 2.1, 2.2], lw=[1.0, 0.0])
    if field == 'gnscale':
        s.update(Nucs='1H', A=[10.0, 20.0, 30.0])
    e = dict(IN)
    (s if where == 'sys' else e)[field] = t
    y = pa.pepper_autograd(SpinSystem(**s), Experiment(**e), GRID)[1]
    with pytest.raises(RuntimeError):
        torch.autograd.grad((y ** 2).sum(), [t])


@pytest.mark.parametrize('where,field,val', [
    ('sys', 'gFrame', [0.3, 0.4, 0.5]),
    ('exp', 'ModAmp', 0.5),
])
def test_frame_and_modamp_on_graph_in_range(where, field, val):
    """gFrame (0.9.17, B3) and ModAmp (0.9.27, B92) are no longer detached: in
    range they get a finite, nonzero gradient (values vs central differences:
    test_frame_gradients.py, test_modamp_grad.py); out of range an exact zero
    (OUT_CASES 'frames', 'modamp')."""
    t = T(val)
    s = dict(S=[0.5], g=[2.0, 2.1, 2.2], lw=[1.0, 0.0])
    e = dict(IN)
    (s if where == 'sys' else e)[field] = t
    y = pa.pepper_autograd(SpinSystem(**s), Experiment(**e), GRID)[1]
    (g,) = torch.autograd.grad((y ** 2).sum(), [t])
    assert g.shape == t.shape and bool(torch.all(torch.isfinite(g)))
    assert float(g.abs().max()) > 0


# ---------------------------------------------------------------------------
# salt (fixed-field ENDOR), no line in freq_range
# ---------------------------------------------------------------------------

def _salt_sys(A, Q):
    return SpinSystem(S=[0.5], g=[[2.2, 2.2, 2.0]], Nucs='14N', A=A, Q=Q)


SALT_EXP = dict(Field=3394.0, mwFreq=95.0)


def test_salt_out_of_range_zero_gradients():
    A, Q, lw = T([[4.0, 4.0, 5.0]]), T([[0.84, 0.84, -1.68]]), T(0.2)
    _, y = salt(_salt_sys(A, Q), Experiment(**SALT_EXP), Options(GridSize=19, GridSymmetry='D2h'),
                freq_range=(100, 120), n_points=256, lw_mhz=lw)
    assert y.shape == (256,)
    _assert_zero_grads(y, [A, Q, lw])


def test_salt_out_of_range_frame_zero_gradients():
    A, AF, Q = T([[4.0, 4.0, 5.0]]), T([[0.3, 0.4, 0.5]]), [[0.84, 0.84, -1.68]]
    s = SpinSystem(S=[0.5], g=[[2.2, 2.2, 2.0]], Nucs='14N', A=A, AFrame=AF, Q=Q)
    _, y = salt(s, Experiment(**SALT_EXP), Options(GridSize=19, GridSymmetry='D2h'),
                freq_range=(100, 120), n_points=256, lw_mhz=0.2)
    _assert_zero_grads(y, [A, AF])


def test_salt_hstrain_without_orientation_selection_zero_gradient():
    H, A = T([50.0, 50.0, 50.0]), T([[4.0, 4.0, 5.0]])
    s = SpinSystem(S=[0.5], g=[[2.2, 2.2, 2.0]], Nucs='14N', A=A, HStrain=H)
    _, y = salt(s, Experiment(**SALT_EXP), Options(GridSize=19, GridSymmetry='D2h'),
                freq_range=(0, 35), n_points=256, lw_mhz=0.2)
    dH, dA = torch.autograd.grad((y ** 2).sum(), [H, A])
    assert torch.all(dH == 0.0) and float(dA.abs().max()) > 0


def test_salt_in_range_values_and_gradients_unchanged(monkeypatch):
    def run():
        A, Q = T([[4.0, 4.0, 5.0]]), T([[0.84, 0.84, -1.68]])
        y = salt(_salt_sys(A, Q), Experiment(**SALT_EXP), Options(GridSize=19, GridSymmetry='D2h'),
                 freq_range=(0, 35), n_points=256, lw_mhz=0.2)[1]
        return y.detach(), torch.autograd.grad((y ** 2).sum(), [A, Q])
    y, (dA, dQ) = run()
    monkeypatch.setattr(ts_autograd, 'connect_inputs', _identity)
    y0, (eA, eQ) = run()
    assert torch.equal(y, y0) and torch.equal(torch.signbit(y), torch.signbit(y0))
    assert torch.equal(dA, eA) and torch.equal(dQ, eQ)


def test_salt_detached_field_still_raises():
    A, Fd = torch.tensor([[4.0, 4.0, 5.0]], dtype=F64), T(3394.0)
    _, y = salt(_salt_sys(A, [[0.84, 0.84, -1.68]]), Experiment(Field=Fd, mwFreq=95.0),
                Options(GridSize=19, GridSymmetry='D2h'), freq_range=(0, 35), n_points=256, lw_mhz=0.2)
    with pytest.raises(RuntimeError):
        torch.autograd.grad((y ** 2).sum(), [Fd])
