"""SpinSystem input shapes follow EasySpin's validatespinsys.m (B62, B63).

Where EasySpin accepts a short form and expands it, SpinSystem expands it the
same way; where EasySpin errors, SpinSystem raises ``ValueError`` at
construction.  Before 0.9.15 many of these shapes were kept unnormalized and
failed later in pepper with an IndexError or RuntimeError (B62), and a 1-D
``gStrain``, ``HStrain`` or ``AStrain`` of one or two values raised although
EasySpin expands ``[x]`` to ``[x x x]`` and ``[x y]`` to ``[x x y]`` (B63).

A 1-D Python list is read as a MATLAB row vector (1 x k).  torchspin
shorthands that EasySpin does not have (a scalar ``g``, ``A`` or ``gStrain``
for every spin, ``[x, y, z]`` for every electron or nucleus) are unchanged.

EasySpin references: tests/data/ref_spinsystem_shapes.mat
(generate_spinsystem_shapes_refs.m), each case run in the short form.
"""
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io import loadmat

from torchspin.experiment import Experiment, Options
from torchspin.pepper import pepper
from torchspin.pepper_autograd import pepper_autograd
from torchspin.spinsystem import SpinSystem
from torchspin.tests.test_pepper_matlab_validation_ext2 import cosine

REF = Path(__file__).resolve().parents[2] / 'tests' / 'data' / 'ref_spinsystem_shapes.mat'

E2 = dict(S=[0.5, 0.5], ee=0.0)
EXP = Experiment(mwFreq=9.5, Range=[300, 350], nPoints=512)
OPT = Options(Verbosity=0, GridSize=10)


def _t(x):
    return torch.tensor(x, dtype=torch.float64)


# --- accepted short forms: same normalized field as the explicit form --------

# (field, short kwargs, explicit kwargs)
EXPANSIONS = [
    # g: [gperp, gpar] for one electron is the (nElectrons, 2) axial form (141-143)
    ('g', dict(S=0.5, g=[2.0, 2.2]), dict(S=0.5, g=[[2.0, 2.0, 2.2]])),
    ('g', dict(S=0.5, g=[[2.0, 2.2]]), dict(S=0.5, g=[[2.0, 2.0, 2.2]])),
    ('g', dict(g=[[2.0, 2.1], [2.0, 2.2]], **E2), dict(g=[[2.0, 2.0, 2.1], [2.0, 2.0, 2.2]], **E2)),
    # numel(g) == nElectrons in any shape (138-140)
    ('g', dict(g=[[2.0, 2.1]], **E2), dict(g=[[2.0] * 3, [2.1] * 3], **E2)),
    # empty Euler angles are zeros (189-191, 240-242)
    ('gFrame', dict(S=0.5, gFrame=[]), dict(S=0.5)),
    ('DFrame', dict(S=1, D=300.0, DFrame=[]), dict(S=1, D=300.0)),
    # A, one nucleus, one electron: (1, 1) isotropic, [Aperp, Apar] axial (594-611)
    ('A', dict(S=0.5, Nucs='1H', A=[[10.0]]), dict(S=0.5, Nucs='1H', A=[[10.0, 10.0, 10.0]])),
    ('A', dict(S=0.5, Nucs='1H', A=[10.0, 20.0]), dict(S=0.5, Nucs='1H', A=[[10.0, 10.0, 20.0]])),
    ('A', dict(S=0.5, Nucs='1H', A=[[10.0, 20.0]]), dict(S=0.5, Nucs='1H', A=[[10.0, 10.0, 20.0]])),
    # A, two nuclei: (nNuclei, 1) isotropic, (nNuclei, 2) axial
    ('A', dict(S=0.5, Nucs='1H,1H', A=[[10.0], [20.0]]),
     dict(S=0.5, Nucs='1H,1H', A=[[10.0] * 3, [20.0] * 3])),
    ('A', dict(S=0.5, Nucs='1H,1H', A=[[10.0, 20.0], [5.0, 6.0]]),
     dict(S=0.5, Nucs='1H,1H', A=[[10.0, 10.0, 20.0], [5.0, 5.0, 6.0]])),
    # A, one nucleus, two electrons: [A1 A2] isotropic per electron, [A1p A1z A2p A2z]
    # axial, a 6-vector principal values (602-613)
    ('A', dict(Nucs='1H', A=[10.0, 20.0], **E2), dict(Nucs='1H', A=[[10.0] * 3 + [20.0] * 3], **E2)),
    ('A', dict(Nucs='1H', A=[[10.0, 20.0]], **E2), dict(Nucs='1H', A=[[10.0] * 3 + [20.0] * 3], **E2)),
    ('A', dict(Nucs='1H', A=[1.0, 2.0, 3.0, 4.0], **E2), dict(Nucs='1H', A=[[1.0, 1.0, 2.0, 3.0, 3.0, 4.0]], **E2)),
    ('A', dict(Nucs='1H', A=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], **E2),
     dict(Nucs='1H', A=[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], **E2)),
    # A_ scalar: aiso of the one nucleus (540-543)
    ('A', dict(S=0.5, Nucs='1H', A_=10.0), dict(S=0.5, Nucs='1H', A_=[10.0])),
    # AFrame of one nucleus, two electrons as a 1x6 row (627)
    ('AFrame', dict(Nucs='1H', A=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], AFrame=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], **E2),
     dict(Nucs='1H', A=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], AFrame=[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], **E2)),
    ('AFrame', dict(S=0.5, Nucs='1H', A=10.0, AFrame=[]), dict(S=0.5, Nucs='1H', A=10.0)),
    # ee: numel == nPairs in any shape is isotropic (329-331)
    ('ee', dict(S=[0.5, 0.5], ee=[[10.0]]), dict(S=[0.5, 0.5], ee=[[10.0, 10.0, 10.0]])),
    ('ee', dict(S=[0.5, 0.5, 0.5], ee=[[1.0], [2.0], [3.0]]),
     dict(S=[0.5, 0.5, 0.5], ee=[[1.0] * 3, [2.0] * 3, [3.0] * 3])),
    # lw: a 1x2 row (776)
    ('lw', dict(S=0.5, lw=[[1.0, 0.5]]), dict(S=0.5, lw=[1.0, 0.5])),
    ('lwpp', dict(S=0.5, lwpp=[[1.0, 0.5]]), dict(S=0.5, lwpp=[1.0, 0.5])),
    # gStrain: one row per electron, 1 or 2 columns expanded (822-823)
    ('gStrain', dict(S=0.5, gStrain=[0.01]), dict(S=0.5, gStrain=[[0.01, 0.01, 0.01]])),
    ('gStrain', dict(S=0.5, gStrain=[0.01, 0.02]), dict(S=0.5, gStrain=[[0.01, 0.01, 0.02]])),
    ('gStrain', dict(S=0.5, gStrain=[[0.01, 0.02]]), dict(S=0.5, gStrain=[[0.01, 0.01, 0.02]])),
    ('gStrain', dict(S=0.5, gStrain=[[0.01]]), dict(S=0.5, gStrain=[[0.01, 0.01, 0.01]])),
    ('gStrain', dict(gStrain=[[0.01, 0.02], [0.01, 0.03]], **E2),
     dict(gStrain=[[0.01, 0.01, 0.02], [0.01, 0.01, 0.03]], **E2)),
    ('gStrain', dict(gStrain=[[0.01], [0.02]], **E2), dict(gStrain=[[0.01] * 3, [0.02] * 3], **E2)),
    ('gStrain', dict(S=0.5, gStrain=[]), dict(S=0.5)),
    # HStrain / AStrain: 1 -> [x x x], 2 -> [x x y], 3 in any shape (887-891)
    ('HStrain', dict(S=0.5, HStrain=[10.0]), dict(S=0.5, HStrain=[10.0, 10.0, 10.0])),
    ('HStrain', dict(S=0.5, HStrain=[10.0, 20.0]), dict(S=0.5, HStrain=[10.0, 10.0, 20.0])),
    ('HStrain', dict(S=0.5, HStrain=[[10.0, 10.0, 20.0]]), dict(S=0.5, HStrain=[10.0, 10.0, 20.0])),
    ('HStrain', dict(S=0.5, HStrain=[[10.0], [10.0], [20.0]]), dict(S=0.5, HStrain=[10.0, 10.0, 20.0])),
    ('HStrain', dict(S=0.5, HStrain=[]), dict(S=0.5)),
    ('AStrain', dict(S=0.5, Nucs='1H', A=10.0, AStrain=[5.0]),
     dict(S=0.5, Nucs='1H', A=10.0, AStrain=[5.0, 5.0, 5.0])),
    ('AStrain', dict(S=0.5, Nucs='1H', A=10.0, AStrain=[5.0, 8.0]),
     dict(S=0.5, Nucs='1H', A=10.0, AStrain=[5.0, 5.0, 8.0])),
    ('AStrain', dict(S=0.5, Nucs='1H', A=10.0, AStrain=[[5.0, 5.0, 8.0]]),
     dict(S=0.5, Nucs='1H', A=10.0, AStrain=[5.0, 5.0, 8.0])),
    # DStrain: empty is no strain (837-839)
    ('DStrain', dict(S=1, D=300.0, DStrain=[]), dict(S=1, D=300.0)),
]


def _same(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return torch.equal(torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64))
    return list(a) == list(b)


@pytest.mark.parametrize('field, short, full', EXPANSIONS,
                         ids=[f'{f}-{i}' for i, (f, _, _) in enumerate(EXPANSIONS)])
def test_short_form_normalizes_like_explicit_form(field, short, full):
    a, b = SpinSystem(**short), SpinSystem(**full)
    assert _same(getattr(a, field), getattr(b, field)), (getattr(a, field), getattr(b, field))


# A spectrum check for one representative of each field (pepper, torch.equal).
SPECTRA = [
    (dict(S=0.5, g=[2.0, 2.2], lw=1.0), dict(S=0.5, g=[[2.0, 2.0, 2.2]], lw=1.0)),
    (dict(S=0.5, g=[2.0, 2.05, 2.1], gFrame=[], lw=1.0), dict(S=0.5, g=[2.0, 2.05, 2.1], lw=1.0)),
    (dict(S=0.5, Nucs='1H', A=[[10.0]], lw=0.2), dict(S=0.5, Nucs='1H', A=10.0, lw=0.2)),
    (dict(S=0.5, Nucs='14N', A=[20.0, 90.0], lw=0.5), dict(S=0.5, Nucs='14N', A=[[20.0, 20.0, 90.0]], lw=0.5)),
    (dict(Nucs='1H', A=[30.0, 60.0], g=[2.0, 2.1], ee=200.0, S=[0.5, 0.5], lw=0.5),
     dict(Nucs='1H', A=[[30.0] * 3 + [60.0] * 3], g=[2.0, 2.1], ee=200.0, S=[0.5, 0.5], lw=0.5)),
    (dict(S=[0.5, 0.5], g=[2.0, 2.1], ee=[[200.0]], lw=0.5), dict(S=[0.5, 0.5], g=[2.0, 2.1], ee=200.0, lw=0.5)),
    (dict(S=0.5, g=[2.0, 2.05, 2.1], lw=[[0.5, 0.3]]), dict(S=0.5, g=[2.0, 2.05, 2.1], lw=[0.5, 0.3])),
    (dict(S=0.5, g=[2.0, 2.05, 2.1], gStrain=[0.004, 0.008]),
     dict(S=0.5, g=[2.0, 2.05, 2.1], gStrain=[[0.004, 0.004, 0.008]])),
    (dict(S=0.5, g=[2.0, 2.05, 2.1], gStrain=[0.004]), dict(S=0.5, g=[2.0, 2.05, 2.1], gStrain=0.004)),
    (dict(S=0.5, g=[2.0, 2.05, 2.1], HStrain=[30.0, 60.0]),
     dict(S=0.5, g=[2.0, 2.05, 2.1], HStrain=[30.0, 30.0, 60.0])),
    (dict(S=0.5, Nucs='14N', A=[20.0, 20.0, 90.0], AStrain=[10.0], lw=0.3),
     dict(S=0.5, Nucs='14N', A=[20.0, 20.0, 90.0], AStrain=10.0, lw=0.3)),
    (dict(S=0.5, g=[2.0, 2.05, 2.1], gStrain=[], lw=1.0), dict(S=0.5, g=[2.0, 2.05, 2.1], lw=1.0)),
]


@pytest.mark.parametrize('short, full', SPECTRA, ids=[str(i) for i in range(len(SPECTRA))])
def test_short_form_spectrum_equals_explicit_form(short, full):
    _, ya = pepper(SpinSystem(**short), EXP, OPT)
    _, yb = pepper(SpinSystem(**full), EXP, OPT)
    assert ya.abs().max() > 0
    assert torch.equal(ya, yb)


def test_short_forms_by_assignment():
    """Assigning a field normalizes it as the constructor does (B63 by assignment)."""
    s = SpinSystem(S=0.5, g=[2.0, 2.05, 2.1])
    s.gStrain = [0.01, 0.02]
    assert torch.equal(s.gStrain, _t([[0.01, 0.01, 0.02]]))
    s.HStrain = [5.0]
    assert torch.equal(s.HStrain, _t([5.0, 5.0, 5.0]))


# --- rejected shapes: ValueError at construction ----------------------------

# (short id, kwargs, message fragment); EasySpin errors for every one of these.
REJECTED = [
    ('g [g] 2e', dict(g=[2.0], **E2), 'g has wrong size'),
    ('g [[g]] 2e', dict(g=[[2.0]], **E2), 'g has wrong size'),
    ('g 1x3 2e', dict(g=[[2.0, 2.1, 2.2]], **E2), 'g has wrong size'),
    ('g 6-vector 2e', dict(g=[2.0] * 6, **E2), 'g has wrong size'),
    ('g 3x3 2e', dict(g=np.eye(3).tolist(), **E2), 'g has wrong size'),
    ('g 4-vector', dict(S=0.5, g=[2.0] * 4), 'g has wrong size'),
    ('g 2x3 1e', dict(S=0.5, g=[[2.0, 2.0, 2.0]] * 2), 'g has wrong size'),
    ('gFrame scalar', dict(S=0.5, gFrame=0.1), 'gFrame has wrong size'),
    ('gFrame [a]', dict(S=0.5, gFrame=[0.1]), 'gFrame has wrong size'),
    ('gFrame [a b]', dict(S=0.5, gFrame=[0.1, 0.2]), 'gFrame has wrong size'),
    ('gFrame 1x3 2e', dict(gFrame=[0.1, 0.2, 0.3], **E2), 'gFrame has wrong size'),
    ('DFrame [a]', dict(S=1, D=300.0, DFrame=[0.1]), 'DFrame has wrong size'),
    ('DFrame 1x4', dict(S=1, D=300.0, DFrame=[[0.1, 0.2, 0.3, 0.4]]), 'DFrame has wrong size'),
    ('D_ scalar', dict(S=1, D_=300.0), 'D_ has wrong size'),
    ('D_ [D]', dict(S=1, D_=[300.0]), 'D_ has wrong size'),
    ('D_ 3 values', dict(S=1, D_=[300.0, 0.1, 0.0]), 'D_ has wrong size'),
    ('A [a] 2 nuclei', dict(S=0.5, Nucs='1H,1H', A=[10.0]), 'Size of A'),
    ('A [[a]] 2 nuclei', dict(S=0.5, Nucs='1H,1H', A=[[10.0]]), 'Size of A'),
    ('A 1x3 2 nuclei', dict(S=0.5, Nucs='1H,1H', A=[[10.0, 20.0, 30.0]]), 'Size of A'),
    ('A 4-vector', dict(S=0.5, Nucs='1H', A=[1.0, 2.0, 3.0, 4.0]), 'Size of A'),
    ('A [a] 2e', dict(Nucs='1H', A=[10.0], **E2), 'Size of A'),
    ('A [ax ay az] 2e', dict(Nucs='1H', A=[1.0, 2.0, 3.0], **E2), 'Size of A'),
    ('A without Nucs', dict(S=0.5, A=[10.0]), 'no nucleus is specified'),
    ('AFrame without Nucs', dict(S=0.5, AFrame=[0.1, 0.2, 0.3]), 'no nucleus is specified'),
    ('Q without Nucs', dict(S=0.5, Q=[1.0]), 'no nucleus is specified'),
    ('QFrame without Nucs', dict(S=0.5, QFrame=[0.1, 0.2, 0.3]), 'no nucleus is specified'),
    ('AFrame scalar', dict(S=0.5, Nucs='1H', A=10.0, AFrame=0.1), 'AFrame has wrong size'),
    ('AFrame [a]', dict(S=0.5, Nucs='1H', A=10.0, AFrame=[0.1]), 'AFrame has wrong size'),
    ('AFrame [[a b]]', dict(S=0.5, Nucs='1H', A=10.0, AFrame=[[0.1, 0.2]]), 'AFrame has wrong size'),
    ('AFrame 1x3 2e', dict(Nucs='1H', A=10.0, AFrame=[0.1, 0.2, 0.3], **E2), 'AFrame has wrong size'),
    ('Q 4-vector', dict(S=0.5, Nucs='14N', A=10.0, Q=[1.0, 2.0, 3.0, -4.0]), 'Q has wrong size'),
    ('Q scalar 2 nuclei', dict(S=0.5, Nucs='14N,14N', A=10.0, Q=1.0), 'Q has wrong size'),
    ('Q [q] 2 nuclei', dict(S=0.5, Nucs='14N,14N', A=10.0, Q=[1.0]), 'Q has wrong size'),
    ('Q 1x3 2 nuclei', dict(S=0.5, Nucs='14N,14N', A=10.0, Q=[[1.0, 2.0, -3.0]]), 'Q has wrong size'),
    ('QFrame [a]', dict(S=0.5, Nucs='14N', A=10.0, Q=1.0, QFrame=[0.1]), 'QFrame has wrong size'),
    ('QFrame [[a b]]', dict(S=0.5, Nucs='14N', A=10.0, Q=1.0, QFrame=[[0.1, 0.2]]), 'QFrame has wrong size'),
    ('ee [a b] 1 pair', dict(S=[0.5, 0.5], ee=[1.0, 2.0]), 'ee has wrong size'),
    ('ee [[a b]] 1 pair', dict(S=[0.5, 0.5], ee=[[1.0, 2.0]]), 'ee has wrong size'),
    ('ee 9-vector 3 pairs', dict(S=[0.5, 0.5, 0.5], ee=[1.0] * 9), 'ee has wrong size'),
    ('ee [a b] 3 pairs', dict(S=[0.5, 0.5, 0.5], ee=[1.0, 2.0]), 'ee has wrong size'),
    ('eeFrame [a]', dict(S=[0.5, 0.5], ee=1.0, eeFrame=[0.1]), 'eeFrame has wrong size'),
    ('eeFrame [[a b]]', dict(S=[0.5, 0.5], ee=1.0, eeFrame=[[0.1, 0.2]]), 'eeFrame has wrong size'),
    ('lw 3 values', dict(S=0.5, lw=[1.0, 0.5, 0.2]), 'lw has wrong size'),
    ('lw empty', dict(S=0.5, lw=[]), 'lw has wrong size'),
    ('lwpp 3 values', dict(S=0.5, lwpp=[1.0, 0.5, 0.2]), 'lwpp has wrong size'),
    ('gStrain 4 columns', dict(S=0.5, gStrain=[0.01] * 4), 'gStrain must have 1, 2, or 3 columns'),
    ('gStrain 2 rows 1e', dict(S=0.5, gStrain=[[0.01] * 3] * 2), 'gStrain must have 1 rows'),
    ('gStrain [x] 2e', dict(gStrain=[0.01], **E2), 'gStrain must have 2 rows'),
    ('gStrain [x y] 2e', dict(gStrain=[0.01, 0.02], **E2), 'gStrain must have 2 rows'),
    ('gStrain 1x3 2e', dict(gStrain=[[0.01, 0.01, 0.02]], **E2), 'gStrain must have 2 rows'),
    ('HStrain 4 values', dict(S=0.5, HStrain=[1.0, 2.0, 3.0, 4.0]), 'HStrain must have 3 elements'),
    ('AStrain 4 values', dict(S=0.5, Nucs='1H', A=10.0, AStrain=[1.0, 2.0, 3.0, 4.0]),
     'AStrain must have 3 elements'),
]


@pytest.mark.parametrize('kw, match', [(k, m) for _, k, m in REJECTED], ids=[i for i, _, _ in REJECTED])
def test_wrong_shape_raises_at_construction(kw, match):
    with pytest.raises(ValueError, match=match):
        SpinSystem(**kw)


# --- valid inputs that were already accepted are unchanged -------------------

def test_existing_forms_unchanged():
    # torchspin shorthands EasySpin does not have
    assert torch.equal(SpinSystem(g=[2.0, 2.1, 2.2], **E2).g, _t([[2.0, 2.1, 2.2]] * 2))
    assert torch.equal(SpinSystem(S=0.5, Nucs='1H,1H', A=[1.0, 2.0, 3.0]).A, _t([[1.0, 2.0, 3.0]] * 2))
    assert torch.equal(SpinSystem(Nucs='1H', A=10.0, **E2).A, _t([[10.0] * 6]))
    assert torch.equal(SpinSystem(gStrain=[0.01, 0.02, 0.03], **E2).gStrain, _t([[0.01, 0.02, 0.03]] * 2))
    assert torch.equal(SpinSystem(S=[1, 1], D=300.0, ee=0.0, DStrain=[10.0]).DStrain, _t([[10.0, 0.0]] * 2))
    # one value per nucleus (one electron), per electron (g), per pair (ee)
    assert torch.equal(SpinSystem(S=0.5, Nucs='1H,1H', A=[10.0, 20.0]).A, _t([[10.0] * 3, [20.0] * 3]))
    assert torch.equal(SpinSystem(g=[2.0, 2.1], **E2).g, _t([[2.0] * 3, [2.1] * 3]))
    assert torch.equal(SpinSystem(S=0.5, g=[2.0]).g, _t([[2.0] * 3]))
    assert torch.equal(SpinSystem(S=0.5, Nucs='1H', A=[10.0]).A, _t([[10.0] * 3]))
    # a stored HStrain / gStrain tensor is the given tensor (no copy)
    hs = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
    assert SpinSystem(S=0.5, HStrain=hs).HStrain is hs
    # a rebuild without nuclei passes the default zero QFrame row on: not "given"
    s = SpinSystem(S=0.5)
    s.lw = 1.0
    assert s.nNuclei == 0 and torch.equal(s.QFrame, torch.zeros(1, 3, dtype=torch.float64))


# --- gradients through expanded forms ----------------------------------------

def test_gradient_flows_through_expanded_forms():
    """d spectrum / d(short-form leaves) equals the gradient through the explicit forms."""
    exp = Experiment(mwFreq=9.5, Range=[320, 350], nPoints=256, Harmonic=1)
    opt = Options(Verbosity=0, GridSize=[19, 4])

    def run(short):
        gS = torch.tensor([0.004, 0.008], dtype=torch.float64, requires_grad=True)
        A = torch.tensor([20.0, 90.0], dtype=torch.float64, requires_grad=True)
        HS = torch.tensor([10.0], dtype=torch.float64, requires_grad=True)
        if short:
            kw = dict(gStrain=gS, A=A, HStrain=HS)
        else:
            kw = dict(gStrain=gS[[0, 0, 1]].unsqueeze(0), A=A[[0, 0, 1]].unsqueeze(0), HStrain=HS[[0, 0, 0]])
        s = SpinSystem(S=0.5, g=[2.0, 2.0, 2.1], Nucs='14N', lw=0.3, **kw)
        _, y = pepper_autograd(s, exp, opt)
        (y * torch.linspace(-1, 1, y.numel(), dtype=y.dtype)).sum().backward()
        return y.detach(), gS.grad, A.grad, HS.grad

    ys, *gs = run(True)
    yf, *gf = run(False)
    assert torch.equal(ys, yf)
    for a, b in zip(gs, gf):
        assert a is not None and bool(torch.all(a != 0))
        assert torch.equal(a, b)


# --- EasySpin ---------------------------------------------------------------

# torchspin systems in the same short form as the MATLAB cases
_ES_CASES = {
    'g_axial': dict(S=0.5, g=[2.0, 2.2], lw=[1.0, 0.0]),
    'gStrain_xy': dict(S=0.5, g=[2.0, 2.05, 2.1], gStrain=[0.004, 0.008]),
    'HStrain_xy': dict(S=0.5, g=[2.0, 2.05, 2.1], HStrain=[30.0, 60.0]),
    'A_axial_14N': dict(S=0.5, g=[2.0, 2.0, 2.1], Nucs='14N', A=[20.0, 90.0], lw=[0.5, 0.0]),
    'A_iso_per_e_2e': dict(S=[0.5, 0.5], g=[[2.0] * 3, [2.1] * 3], ee=[200.0], Nucs='1H', A=[30.0, 60.0],
                           lw=[0.5, 0.0]),
}

# Measured (0.9.15, default Options): cosine 0.99999996 (g_axial), 0.9999999965
# (gStrain_xy, HStrain_xy), 0.99999909 (A_axial_14N), 0.99999999 (A_iso_per_e_2e);
# amplitude ratio 0.99986, 0.99999, 0.99999, 1.00086, 0.99996.  The explicit
# forms give bit-identical spectra (test_short_form_spectrum_equals_explicit_form),
# so these are pepper's own deviations from EasySpin at the default grid.
COS_MIN = 1 - 2e-6
AMP_TOL = 1.5e-3


@pytest.mark.parametrize('name', list(_ES_CASES))
def test_short_forms_vs_easyspin(name):
    if not REF.exists():
        pytest.skip(f'reference file missing: {REF}')
    m = loadmat(REF, squeeze_me=True, struct_as_record=False)
    c = {str(k.name): k for k in np.atleast_1d(m['cases'])}[name]
    ref_x = np.asarray(c.x, dtype=float)
    ref = np.asarray(c.spc, dtype=float)
    exp = Experiment(mwFreq=9.5, Range=[300, 350], nPoints=1024)
    x, spc = pepper(SpinSystem(**_ES_CASES[name]), exp, Options(Verbosity=0))
    x = x.detach().numpy(); spc = spc.detach().numpy()
    assert np.allclose(x, ref_x, rtol=1e-12, atol=1e-9)
    cs = cosine(spc, ref)
    amp = np.abs(spc).max() / np.abs(ref).max()
    print(f'{name}: cosine {cs:.10f}, amplitude ratio {amp:.7f}')
    assert cs > COS_MIN, (name, cs)
    assert amp == pytest.approx(1.0, abs=AMP_TOL), (name, amp)
