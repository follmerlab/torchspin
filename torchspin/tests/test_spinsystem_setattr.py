"""Assigning a SpinSystem field after construction normalizes it (B12).

torchspin <= 0.8.6 normalized fields only in ``__post_init__``, so
``sys.HStrain = [1, 2, 3]`` left a raw list and pepper later failed with
``TypeError: '>' not supported between instances of 'list' and 'int'``.
``SpinSystem.__setattr__`` now rebuilds the system with the new value.
"""
import copy
import dataclasses
import pickle

import numpy as np
import pytest
import torch

from torchspin import Experiment, Options, SpinSystem, garlic, ham, pepper

F64 = torch.float64


def _same(a, b):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
                and a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _assert_same_system(s1, s2):
    for f in dataclasses.fields(SpinSystem):
        assert _same(getattr(s1, f.name), getattr(s2, f.name)), f.name


def _exp():
    return Experiment(mwFreq=9.5, Range=[320, 360], nPoints=1024)


def _opt():
    return Options(GridSize=[19, 4], Verbosity=0)


def test_repro_hstrain_assignment_pepper():
    """The reported failure: on 0.8.6 pepper raised TypeError."""
    s = SpinSystem(S=0.5, g=[2.0, 2.01, 2.02])
    s.HStrain = [1, 2, 3]
    assert isinstance(s.HStrain, torch.Tensor) and s.HStrain.dtype == F64
    ref = SpinSystem(S=0.5, g=[2.0, 2.01, 2.02], HStrain=[1, 2, 3])
    _assert_same_system(s, ref)
    B, spc = pepper(s, _exp(), _opt())
    B_ref, spc_ref = pepper(ref, _exp(), _opt())
    assert torch.equal(B, B_ref) and torch.equal(spc, spc_ref)


@pytest.mark.parametrize('name, value', [
    ('g', [2.0, 2.0, 2.1]), ('g', 2.0), ('gStrain', 0.01), ('D', [300.0, 30.0]),
    ('D_', [300.0, 0.1]), ('A', [10.0, 10.0, 30.0]), ('A_', [[10.0, 2.0]]),
    ('Q', [[2.0, 0.1]]), ('lw', 0.5), ('lwpp', [0.3, 0.1]), ('DStrain', [20.0, 5.0]),
    ('DStrainCorr', 0.3), ('ee', None), ('AStrain', 3.0), ('Nucs', '14N'), ('n', [2]),
    ('Ham', {'132': [0, 0, 0.01, 0, 0]}), ('HStrainL', 2.0), ('sigma', 0.99),
])
def test_assignment_equals_construction(name, value):
    base = dict(S=1.0, g=[2.0, 2.0, 2.0], Nucs='14N', A=[1.0, 1.0, 2.0])
    s = SpinSystem(**base)
    s.__setattr__(name, value)
    kw = dict(base)
    if name == 'A_':
        kw.pop('A')
    kw[name] = value
    _assert_same_system(s, SpinSystem(**kw))


def test_alternative_pairs_clear_each_other():
    s = SpinSystem(S=1, D_=[300.0, 0.1], Nucs='1H', A_=[[3.0, 1.0]], lwpp=0.5,
                   tcorr=1e-9, Diff=1e8)
    s.D = [200.0, 20.0]
    assert s.D_ is None and torch.allclose(s.D, SpinSystem(S=1, D=[200.0, 20.0]).D)
    s.D_ = [100.0, 0.2]
    assert torch.allclose(s.D, SpinSystem(S=1, D_=[100.0, 0.2]).D)
    s.A = [5.0, 5.0, 5.0]
    assert s.A_ is None and torch.equal(s.A, torch.full((1, 3), 5.0, dtype=F64))
    s.A_ = [[4.0, 1.0]]
    assert torch.equal(s.A, torch.tensor([[3.0, 3.0, 6.0]], dtype=F64))
    s.lw = 1.0
    assert s.lwpp is None and s.lw == [1.0, 0.0]
    s.lwpp = 0.4
    assert s.lw is None and s.lwpp == [0.4, 0.0]
    s.logtcorr = -10.0
    assert s.tcorr is None and s.logtcorr == -10.0
    s.tcorr = 2e-9
    assert s.logtcorr is None and s.tcorr == 2e-9
    s.logDiff = 9.0
    assert s.Diff is None
    s.Diff = 1e7
    assert s.logDiff is None


def test_logtcorr_assignment_gives_same_garlic_result():
    """Consumers read logtcorr first, so assigning it over a set tcorr gives the
    result it gave before (the tcorr it now clears was ignored anyway)."""
    exp = Experiment(mwFreq=9.5, Range=[334, 344], nPoints=512)
    s1 = SpinSystem(S=0.5, g=[2.0088, 2.0061, 2.0027], Nucs='14N', A=[16, 16, 86], lw=0.05)
    s1.tcorr = 3e-9
    s1.logtcorr = -10.0
    s2 = SpinSystem(S=0.5, g=[2.0088, 2.0061, 2.0027], Nucs='14N', A=[16, 16, 86], lw=0.05,
                    tcorr=3e-9, logtcorr=-10.0)
    _, y1 = garlic(s1, exp)
    _, y2 = garlic(s2, exp)
    assert np.allclose(np.asarray(y1), np.asarray(y2), rtol=0, atol=0)


@pytest.mark.parametrize('name, value, match', [
    ('HStrain', [1.0, 2.0], 'HStrain'),
    ('HStrainL', [-1.0, 1.0, 1.0], 'HStrainL'),
    ('DStrain', [[1.0, 2.0, 3.0, 4.0]], 'DStrain'),
    ('n', [1, 2], 'SpinSystem.n'),
    ('D_', [1.0, 2.0, 3.0, 4.0], 'D_'),
    ('lw', [1.0, 0.0], None),     # valid: control case
])
def test_invalid_assignment_raises_and_leaves_system_unchanged(name, value, match):
    s = SpinSystem(S=1, D=[300.0, 30.0], Nucs='1H', A=[1.0, 1.0, 1.0], lwpp=0.5)
    before = copy.deepcopy(s)
    if match is None:
        s.lwpp = None
        s.__setattr__(name, value)
        return
    with pytest.raises(ValueError, match=match):
        s.__setattr__(name, value)
    _assert_same_system(s, before)


def test_g_with_linear_ham_raises_on_assignment():
    s = SpinSystem(S=0.5, Ham={'112': [[1.0, 0, 0, 0, 0.5]]})
    with pytest.raises(ValueError, match='Ham112'):
        s.g = [2.0, 2.0, 2.0]


def test_A_none_assignment_is_the_constructor_default():
    """Assigning None gives what the constructor gives for None (zero hyperfine);
    tests that need a field left at None use object.__setattr__ instead."""
    s = SpinSystem(S=0.5, Nucs='14N', A=[10.0, 10.0, 30.0])
    s.A = None
    assert torch.equal(s.A, torch.zeros(1, 3, dtype=F64))


def test_non_field_attributes_are_plain():
    s = SpinSystem(S=0.5)
    s.ZeemanFreq = [9.5]
    assert s.ZeemanFreq == [9.5]
    s.HStrain = 4.0              # a field assignment keeps the extra attribute
    assert s.ZeemanFreq == [9.5]
    assert '_normalized' not in repr(s)
    _assert_same_system(s, SpinSystem(S=0.5, HStrain=4.0))


def test_copies_and_pickle():
    s = SpinSystem(S=0.5, g=[2.0, 2.0, 2.1], Nucs='1H', A=[1, 1, 1])
    for c in (copy.copy(s), copy.deepcopy(s), pickle.loads(pickle.dumps(s))):
        c.HStrain = [5, 5, 5]
        assert isinstance(c.HStrain, torch.Tensor)
        assert s.HStrain is None
    d = copy.deepcopy(s)
    d.g = [2.1, 2.1, 2.1]
    assert torch.equal(s.g, torch.tensor([[2.0, 2.0, 2.1]], dtype=F64))


def test_assigned_tensor_keeps_the_graph():
    """An assigned requires-grad tensor is not copied or detached: the energy
    levels differentiate through it, matching central differences."""
    gz = torch.tensor(2.05, dtype=F64, requires_grad=True)
    s = SpinSystem(S=1, D=[300.0, 30.0], Nucs='1H', A=[5.0, 5.0, 10.0])
    s.g = torch.stack([gz.new_tensor(2.0), gz.new_tensor(2.01), gz])
    assert s.g.requires_grad

    def energy_sum_sq(g3):
        t = SpinSystem(S=1, D=[300.0, 30.0], Nucs='1H', A=[5.0, 5.0, 10.0])
        t.g = torch.stack([g3.new_tensor(2.0), g3.new_tensor(2.01), g3])
        E = torch.linalg.eigvalsh(ham(t, B0=[100.0, 50.0, 330.0]))
        return (E ** 2).sum()

    val = energy_sum_sq(gz)
    (grad,) = torch.autograd.grad(val, gz)
    h = 1e-6
    fd = (energy_sum_sq(torch.tensor(2.05 + h, dtype=F64))
          - energy_sum_sq(torch.tensor(2.05 - h, dtype=F64))) / (2 * h)
    assert abs(float(grad) - float(fd)) < 1e-6 * abs(float(fd))


def test_assigned_hstrain_gradient_through_pepper():
    """pepper (reverse mode) through an HStrain tensor assigned after construction."""
    hs = torch.tensor([10.0, 12.0, 15.0], dtype=F64, requires_grad=True)

    def peak(h):
        s = SpinSystem(S=0.5, g=[2.0, 2.01, 2.02])
        s.HStrain = h
        return pepper(s, _exp(), _opt())[1].abs().max()

    (grad,) = torch.autograd.grad(peak(hs), hs)
    eps = 1e-4
    for k in range(3):
        e = torch.zeros(3, dtype=F64)
        e[k] = eps
        fd = (peak(hs.detach() + e) - peak(hs.detach() - e)) / (2 * eps)
        assert abs(float(grad[k]) - float(fd)) < 1e-4 * max(1.0, abs(float(fd)))
