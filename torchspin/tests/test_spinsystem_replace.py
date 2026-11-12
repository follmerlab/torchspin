"""Internal rebuilds of a SpinSystem built with D_ or A_ (B22).

``dataclasses.replace`` passes every stored field, so for a system built with
``D_`` (``A_``) it also passes the ``D`` (``A``) derived from it, and the
constructor raises "given in both D and D_". Before 0.8.12 this broke the
isotopologue expansion (pepper with a natural-abundance nucleus and D_), the
Lorentzian-strain path (HStrainL with D_ or A_) and spin2sys's U term.
``spinsystem._replace`` rebuilds from ``_init_kwargs`` instead.
"""

import dataclasses

import pytest
import torch

from torchspin.experiment import Experiment, Options
from torchspin.isotopologues import isotopologues
from torchspin.pepper import pepper
from torchspin.pepper_autograd import pepper_autograd
from torchspin.spinsystem import SpinSystem, _replace
from torchspin.strainwidth import lorentzian_strain_system

F64 = torch.float64
D_IN = [300.0, 0.1]                       # D = 300 MHz, E/D = 0.1
D_PRINCIPAL = [-70.0, -130.0, 200.0]      # same tensor as principal values


def _rel(a, b):
    return float((a - b).abs().max() / b.abs().max())


# ---------------------------------------------------------------------------
# The helper
# ---------------------------------------------------------------------------

def test_replace_keeps_D_and_changes_the_field():
    s = SpinSystem(S=1, D_=D_IN, Nucs='1H', A_=[[5.0, 1.0]])
    r = _replace(s, weight=0.5)
    assert r.weight == 0.5
    assert torch.equal(r.D, s.D) and torch.equal(r.D_, s.D_)
    assert torch.equal(r.A, s.A) and torch.equal(r.A_, s.A_)


def test_replace_of_one_alternative_clears_the_other():
    s = SpinSystem(S=1, D_=D_IN, Nucs='1H', A_=[[5.0, 1.0]])
    r = _replace(s, D=[100.0, 10.0], A=[[1.0, 2.0, 3.0]])
    assert r.D_ is None and r.A_ is None
    assert torch.allclose(r.D, torch.tensor([[-100 / 3 + 10, -100 / 3 - 10, 200 / 3]], dtype=F64))
    assert torch.equal(r.A, torch.tensor([[1.0, 2.0, 3.0]], dtype=F64))


def test_dataclasses_replace_still_raises():
    """The constructor is unchanged (documented in KNOWN_LIMITATIONS)."""
    with pytest.raises(ValueError, match="both D and D_"):
        dataclasses.replace(SpinSystem(S=1, D_=D_IN), weight=0.5)


# ---------------------------------------------------------------------------
# Isotopologue expansion
# ---------------------------------------------------------------------------

def test_isotopologues_with_D_():
    s = SpinSystem(S=1, D_=D_IN, Nucs='Cu', A=[[50.0, 50.0, 400.0]])
    out = isotopologues(s)
    assert [o.Nucs for o in out] == [['63Cu'], ['65Cu']]
    for o in out:
        assert torch.equal(o.D, s.D)
    assert abs(sum(o.weight for o in out) - 1.0) < 1e-12


def test_isotopologues_with_A_and_D_():
    s = SpinSystem(S=1, D_=D_IN, Nucs='Cu', A_=[[100.0, 20.0]])
    ref = SpinSystem(S=1, D=D_PRINCIPAL, Nucs='Cu', A=s.A)
    out, out_ref = isotopologues(s), isotopologues(ref)
    assert len(out) == len(out_ref) == 2
    for o, r in zip(out, out_ref):
        assert o.Nucs == r.Nucs and o.weight == r.weight
        assert torch.allclose(o.D, r.D, rtol=0, atol=1e-12)
        assert torch.equal(o.A, r.A)


def test_pepper_natural_copper_with_D_equals_D():
    exp = Experiment(mwFreq=9.5, Range=[200, 450], nPoints=256)
    opt = Options(GridSize=7)
    kw = dict(S=1, Nucs='Cu', A=[[50.0, 50.0, 400.0]], lw=2.0)
    _, y = pepper(SpinSystem(D_=D_IN, **kw), exp, opt)
    _, y_ref = pepper(SpinSystem(D=D_PRINCIPAL, **kw), exp, opt)
    assert _rel(y, y_ref) < 1e-10


# ---------------------------------------------------------------------------
# Lorentzian strain (HStrainL)
# ---------------------------------------------------------------------------

def test_lorentzian_strain_system_with_D_and_A_():
    s = SpinSystem(S=1, D_=D_IN, Nucs='1H', A_=[[10.0, 2.0]], HStrainL=[10.0, 20.0, 30.0])
    r, used = lorentzian_strain_system(s)
    assert used
    assert r.HStrainL is None and torch.equal(r.HStrain, s.HStrainL)
    assert torch.equal(r.D, s.D) and torch.equal(r.A, s.A)


def test_pepper_hstrainl_with_D_equals_D():
    exp = Experiment(mwFreq=9.5, Range=[250, 420], nPoints=256)
    opt = Options(GridSize=[13, 4])
    _, y = pepper(SpinSystem(S=1, D_=D_IN, HStrainL=[50.0, 50.0, 80.0]), exp, opt)
    _, y_ref = pepper(SpinSystem(S=1, D=D_PRINCIPAL, HStrainL=[50.0, 50.0, 80.0]), exp, opt)
    assert _rel(y, y_ref) < 1e-10


def test_pepper_autograd_D_gradient_through_rebuilt_system():
    """D_ requiring grad stays on the graph through lorentzian_strain_system."""
    exp = Experiment(mwFreq=9.5, Range=[250, 420], nPoints=341, Harmonic=1)
    opt = Options(GridSize=[13, 4], GridSymmetry='D2h')
    r = torch.randn(341, dtype=F64, generator=torch.Generator().manual_seed(0))

    def f(Dv):
        return pepper_autograd(SpinSystem(S=1, D_=Dv, HStrainL=[50.0, 50.0, 80.0]), exp, opt)[1]

    D0 = torch.tensor(D_IN, dtype=F64)
    Dv = D0.clone().requires_grad_()
    (f(Dv) @ r).backward()
    assert Dv.grad is not None
    fd = []
    for i, h in ((0, 1e-3), (1, 1e-6)):
        e = torch.zeros(2, dtype=F64)
        e[i] = h
        fd.append(float((f(D0 + e) - f(D0 - e)) @ r) / (2 * h))
    fd = torch.tensor(fd, dtype=F64)
    assert float(fd.abs().min()) > 0
    assert float((Dv.grad - fd).abs().max() / fd.abs().max()) < 1e-6


# ---------------------------------------------------------------------------
# spin2sys U term
# ---------------------------------------------------------------------------

def test_spin2sys_u_term_with_D_():
    from torchspin.lft.telser2torchspin import spin2sys
    kw = dict(s=1.5, g=(2.0, 2.0, 2.0), I=2.5, g_n=1.0, A=(100.0, 100.0, 100.0), U=5.0)
    sy = spin2sys(**kw, D_=[3000.0, 0.1])
    assert sy.H0Extra is not None and len(sy.H0Extra) == 1
    assert torch.equal(sy.D_, torch.tensor([[3000.0, 0.1]], dtype=F64))
    ref = spin2sys(**kw)
    assert torch.equal(sy.A, ref.A)
    assert torch.equal(sy.H0Extra[0][1], ref.H0Extra[0][1])
