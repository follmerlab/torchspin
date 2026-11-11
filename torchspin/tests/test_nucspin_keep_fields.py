"""nucspinrmv / nucspinkeep / nucspinadd keep every field of the spin system (B14).

torchspin <= 0.8.1 rebuilt the new SpinSystem from a short list of fields, so
``ee`` became zero, ``Ham`` None, and ``ee2``, ``eeFrame``, ``sigma``, ``nn``,
``L``/``soc``/``CF*``, ``initState``, ``tdm``, ``D_``, ``Abund``, ``T1``/``T2``,
``tcorr`` ... were silently dropped.  pepper ``Method='hybrid'`` and saffron's
fast path build their exact core with ``nucspinrmv``, so an exchange-coupled
dimer or a system with ``Sys.Ham`` used the wrong core Hamiltonian.

EasySpin's ``nucspinrmv.m`` starts from ``NewSys = Sys`` and filters only the
per-nucleus fields; that is what is checked here, together with the
round trip of the rebuild helper ``_init_kwargs`` that the fix relies on.
"""
import dataclasses
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io import loadmat

from torchspin import Experiment, Options, SpinSystem, ham, pepper
from torchspin.spinsystem import _init_kwargs, nucspinadd, nucspinkeep, nucspinrmv

F64 = torch.float64


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _same(a, b):
    """Field values equal, tensors compared exactly (shape, dtype, values)."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
                and a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _assert_same_system(s1, s2, skip=()):
    for f in dataclasses.fields(SpinSystem):
        if f.name in skip:
            continue
        assert _same(getattr(s1, f.name), getattr(s2, f.name)), f.name


def _broad_systems():
    """Systems that exercise every conversion in __post_init__."""
    H0 = torch.diag(torch.tensor([1.0, -1.0], dtype=F64)).to(torch.complex128)
    return {
        'radical': SpinSystem(S=0.5, g=[2.0, 2.0, 2.1]),
        'dimer_ham': SpinSystem(S=[0.5, 0.5], g=[[2, 2, 2], [2, 2, 2]], ee=[1000.0], ee2=[3.0],
                                eeFrame=[0.1, 0.2, 0.3], Nucs='1H,14N',
                                A=[[10, 10, 10, 0, 0, 0], [0, 0, 0, 5, 5, 5]]),
        'triplet_D_': SpinSystem(S=1, D_=[300.0, 0.1], DFrame=[0.1, 0.2, 0.3],
                                 initState=([0.2, 0.3, 0.5], 'xyz'), tdm='x',
                                 DStrain=[10.0, 2.0], DStrainCorr=0.5, lwpp=[0.5, 0.1]),
        'A_spherical': SpinSystem(S=0.5, Nucs='1H,13C', A_=[[3.0, 1.0], [5.0, 2.0]],
                                  sigma=[1.0, 0.99], Q=None),
        'quad_scalar': SpinSystem(S=0.5, Nucs='14N,2H', A=[[10, 10, 30], [1, 1, 2]],
                                  Q=[[2.0, 0.1], [0.2, 0.0]], QFrame=[[0, 0.3, 0], [0, 0, 0]],
                                  gnscale=[1.0, 0.9], n=[1, 2], AStrain=[1, 2, 3]),
        'full_A': SpinSystem(S=0.5, Nucs='1H', A=[[1, 0.1, 0], [0.1, 2, 0], [0, 0, 3]]),
        'ham_linear': SpinSystem(S=1.5, Nucs='1H', A=[10] * 3, Ham={'112': [[1.0, 0, 0.2, 0, 0.5]]}),
        'ham134': SpinSystem(S=1.5, g=[2] * 3, Nucs='1H', A=[10] * 3,
                             Ham={'134': [[0.1, 0, 0, 0, 0.2, 0, 0, 0, 0]]}),
        'oam': SpinSystem(S=0.5, L=1, soc=[[100.0]], CF2=[[1, 0, 2, 0, 0]]),
        'stevens_B': SpinSystem(S=2, B=[None, [[0, 0, 0, 0, 0.01, 0, 0, 0, 0]]]),
        'dynamics': SpinSystem(S=0.5, Nucs='14N', A=[15, 15, 90], tcorr=1e-9, logtcorr=-9.5,
                               Diff=1e8, T1=10.0, T2=1.0, lwEndor=0.3, weight=0.4,
                               HStrain=[5, 5, 10], gStrain=[0.01, 0.01, 0.02], gAStrainCorr=-1),
        'mixture_abund': SpinSystem(S=0.5, Nucs='(63,65)Cu,1H', Abund=[[0.6, 0.4], None],
                                    A=[[50, 50, 500], [3, 3, 3]], nn=[[0.0, 0.0, 0.0]]),
        'h0extra': SpinSystem(S=0.5, H0Extra=[(0.5, H0)]),
        'lorentz_strain': SpinSystem(S=0.5, HStrainL=4.0, lw=1.0),
    }


# ---------------------------------------------------------------------------
# the reported defect
# ---------------------------------------------------------------------------

def test_repro_ee_and_ham_preserved():
    """Repro from the bug report: on main ee became [[0, 0, 0]] and Ham None."""
    s = SpinSystem(S=[0.5, 0.5], g=[[2, 2, 2], [2, 2, 2]], ee=[1000.0], Nucs='1H,14N',
                   A=[[10, 10, 10, 0, 0, 0], [0, 0, 0, 5, 5, 5]])
    assert torch.equal(nucspinrmv(s, [1]).ee, torch.full((1, 3), 1000.0, dtype=F64))
    s2 = SpinSystem(S=1.5, g=[2] * 3, Nucs='1H', A=[10] * 3,
                    Ham={'134': [[0.1, 0, 0, 0, 0.2, 0, 0, 0, 0]]})
    r = nucspinrmv(s2, [0])
    assert r.Ham is not None and torch.equal(r.Ham['134'], s2.Ham['134'])


@pytest.mark.parametrize('name', ['dimer_ham', 'triplet_D_', 'quad_scalar', 'ham134', 'dynamics'])
def test_every_non_nuclear_field_kept(name):
    """Removing nuclei leaves every field that is not per-nucleus unchanged."""
    s = _broad_systems()[name]
    per_nuc = {'Nucs', 'n', 'A', 'A_', 'AFrame', 'gnscale', 'sigma', 'sigmaFrame', 'Q',
               'QFrame', 'nn', 'nnFrame', 'Abund', 'AStrain'}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        r = nucspinrmv(s, list(range(s.nNuclei))) if s.nNuclei else nucspinrmv(s, [])
    _assert_same_system(s, r, skip=per_nuc)


@pytest.mark.parametrize('remove', [[0], [1], [0, 1]])
def test_core_hamiltonian_equals_system_built_without_the_nuclei(remove):
    """ham(nucspinrmv(sys)) is the Hamiltonian of the system written without
    those nuclei, including exchange and a higher-order Zeeman term."""
    common = dict(S=[1.5, 1.5], g=[[2.0, 2.01, 2.02], [2.1, 2.1, 2.0]], ee=[[900, 950, 1100]],
                  ee2=[2.0], eeFrame=[0.2, 0.4, 0.1], D=[[0, 0, 0], [-100, -60, 160]],
                  Ham={'132': [[0, 0, 0.05, 0, 0], [0, 0, 0.02, 0, 0.01]]})
    nucs = ['1H', '14N']
    A = [[10, 11, 12, 1, 1, 1], [3, 3, 4, 20, 20, 30]]
    Q = [[0.0, 0.0, 0.0], [-0.4, -0.6, 1.0]]
    s = SpinSystem(Nucs=nucs, A=A, Q=Q, gnscale=[1.0, 0.95], **common)
    keep = [i for i in range(2) if i not in remove]
    ref = SpinSystem(Nucs=[nucs[i] for i in keep], A=[A[i] for i in keep] or None,
                     Q=[Q[i] for i in keep] or None,
                     gnscale=[[1.0, 0.95][i] for i in keep] or None, **common)
    r = nucspinrmv(s, remove)
    for M1, M2 in zip(ham(r, B0=None), ham(ref, B0=None)):
        assert torch.allclose(M1, M2, atol=1e-12)
    H1 = ham(r, B0=[100.0, 200.0, 300.0])
    H2 = ham(ref, B0=[100.0, 200.0, 300.0])
    assert torch.allclose(H1, H2, atol=1e-12)


# ---------------------------------------------------------------------------
# _init_kwargs round trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('name', list(_broad_systems()))
def test_init_kwargs_round_trip(name):
    s = _broad_systems()[name]
    _assert_same_system(s, SpinSystem(**_init_kwargs(s)))


def test_init_kwargs_does_not_alias_containers():
    s = _broad_systems()['stevens_B']
    kw = _init_kwargs(s)
    assert kw['B'] is not s.B and kw['S'] is not s.S
    s2 = SpinSystem(**kw)
    s2.B.append(None)
    assert len(s.B) == 2


def test_init_kwargs_keeps_the_graph():
    """Rebuilding keeps tensors that require grad on the autograd graph."""
    g = torch.tensor([2.0, 2.0, 2.1], dtype=F64, requires_grad=True)
    D_ = torch.tensor([300.0, 0.1], dtype=F64, requires_grad=True)
    A_ = torch.tensor([[3.0, 1.0]], dtype=F64, requires_grad=True)
    J = torch.tensor(1000.0, dtype=F64, requires_grad=True)
    s = SpinSystem(S=[1, 0.5], g=[g, g], D_=[D_, torch.tensor([0.0, 0.0], dtype=F64)],
                   ee=J, Nucs='1H', A_=torch.cat([A_, A_], dim=1))
    s2 = SpinSystem(**_init_kwargs(s))
    loss = s2.g.sum() + s2.D.pow(2).sum() + s2.ee.sum() + s2.A.pow(2).sum()
    loss.backward()
    for t in (g, D_, A_, J):
        assert t.grad is not None and torch.all(torch.isfinite(t.grad))
    ref = (s.g.sum() + s.D.pow(2).sum() + s.ee.sum() + s.A.pow(2).sum())
    gs = torch.autograd.grad(ref, (g, D_, A_, J))
    for t, gr in zip((g, D_, A_, J), gs):
        assert torch.allclose(t.grad, gr)


def test_nucspinrmv_keeps_the_graph():
    """A requires-grad electron parameter and a kept hyperfine row stay on the graph."""
    J = torch.tensor(800.0, dtype=F64, requires_grad=True)
    a = torch.tensor(12.0, dtype=F64, requires_grad=True)
    A = torch.stack([torch.stack([a, a, a, a * 0, a * 0, a * 0]),
                     torch.tensor([1.0, 1, 1, 0, 0, 0], dtype=F64)])
    s = SpinSystem(S=[0.5, 0.5], ee=J, Nucs='1H,13C', A=A)
    r = nucspinrmv(s, 1)
    (r.ee.sum() + r.A.sum()).backward()
    assert float(J.grad) == 3.0 and float(a.grad) == 3.0


# ---------------------------------------------------------------------------
# per-nucleus fields
# ---------------------------------------------------------------------------

def test_principal_per_nucleus_fields_filtered():
    s = SpinSystem(S=0.5, Nucs='1H,14N,2H', n=[2, 1, 3],
                   A=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                   AFrame=[[0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0]],
                   gnscale=[1.0, 0.9, 0.8], sigma=[[1, 1, 1], [0.9, 0.9, 0.9], [0.8, 0.8, 0.8]],
                   sigmaFrame=[[0, 0.1, 0], [0, 0.2, 0], [0, 0.3, 0]],
                   Q=[[0, 0, 0], [-1, -1, 2], [-0.1, -0.1, 0.2]],
                   QFrame=[[0, 0, 0.1], [0, 0, 0.2], [0, 0, 0.3]])
    r = nucspinrmv(s, 1)
    assert r.Nucs == ['1H', '2H'] and r.n == [2, 3]
    for name in ('A', 'AFrame', 'sigma', 'sigmaFrame', 'Q', 'QFrame'):
        assert torch.equal(getattr(r, name), getattr(s, name)[[0, 2]]), name
    assert torch.equal(r.gnscale, s.gnscale[[0, 2]])
    k = nucspinkeep(s, [1])
    assert k.Nucs == ['14N'] and k.n == [1]
    assert torch.equal(k.A, s.A[[1]]) and torch.equal(k.Q, s.Q[[1]])


def test_full_tensor_fields_filtered_by_block():
    """Full 3x3 A, sigma, Q: three rows per nucleus."""
    rng = np.random.default_rng(0)
    def sym(n):
        M = rng.normal(size=(3 * n, 3))
        return [M[3 * i:3 * i + 3] + M[3 * i:3 * i + 3].T for i in range(n)]
    A = np.vstack(sym(3)); sig = np.vstack(sym(3)); Q = np.vstack(sym(3))
    s = SpinSystem(S=0.5, Nucs='1H,14N,2H', A=A, sigma=sig, Q=Q)
    assert s.fullA and s.fullsigma and s.fullQ
    r = nucspinrmv(s, [0])
    rows = list(range(3, 9))
    assert r.fullA and r.fullsigma and r.fullQ
    assert torch.equal(r.A, s.A[rows]) and torch.equal(r.sigma, s.sigma[rows])
    assert torch.equal(r.Q, s.Q[rows])


def test_A_spherical_form_filtered():
    s = SpinSystem(S=0.5, Nucs='1H,13C,14N', A_=[[3.0, 1.0], [5.0, 2.0], [7.0, 0.5]])
    r = nucspinrmv(s, 1)
    assert torch.equal(r.A_, s.A_[[0, 2]]) and torch.equal(r.A, s.A[[0, 2]])
    e = nucspinrmv(s, [0, 1, 2])
    assert e.A_ is None and e.A is None and e.nNuclei == 0


def test_abund_filtered():
    s = SpinSystem(S=0.5, Nucs='1H,(63,65)Cu', Abund=[None, [0.6, 0.4]], A=[[1] * 3, [50, 50, 500]])
    assert nucspinrmv(s, 0).Abund == [[0.6, 0.4]]
    assert nucspinrmv(s, 1).Abund is None
    s1 = SpinSystem(S=0.5, Nucs='(63,65)Cu', Abund=[0.6, 0.4], A=[50, 50, 500])
    assert nucspinadd(s1, '1H', A=1.0).Abund == [[0.6, 0.4], None]


def test_remove_all_nuclei_gives_defaults():
    s = _broad_systems()['quad_scalar']
    with pytest.warns(UserWarning, match='AStrain'):
        r = nucspinrmv(s, [0, 1])
    d = SpinSystem(S=0.5)
    for name in ('Nucs', 'n', 'A', 'AFrame', 'gnscale', 'sigma', 'sigmaFrame', 'Q', 'QFrame',
                 'nn', 'nnFrame', 'AStrain'):
        assert _same(getattr(r, name), getattr(d, name)), name


# ---------------------------------------------------------------------------
# AStrain, nn, initState
# ---------------------------------------------------------------------------

def test_astrain_kept_when_first_nucleus_stays():
    s = SpinSystem(S=0.5, Nucs='63Cu,1H', A=[[50, 50, 500], [3, 3, 3]], AStrain=[10, 10, 30])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert torch.equal(nucspinrmv(s, 1).AStrain, s.AStrain)
        assert torch.equal(nucspinadd(s, '14N', A=2.0).AStrain, s.AStrain)


def test_astrain_dropped_with_warning_when_first_nucleus_removed():
    s = SpinSystem(S=0.5, Nucs='63Cu,1H', A=[[50, 50, 500], [3, 3, 3]], AStrain=[10, 10, 30])
    with pytest.warns(UserWarning, match='AStrain applies to the first nucleus'):
        r = nucspinrmv(s, 0)
    assert r.AStrain is None
    with pytest.warns(UserWarning, match='AStrain'):
        assert nucspinkeep(s, 1).AStrain is None


def test_nonzero_nn_raises_like_easyspin():
    s = SpinSystem(S=0.5, Nucs='1H,1H,1H', A=[[1] * 3] * 3, nn=[[0.1, 0.1, 0.1]] * 3)
    with pytest.raises(ValueError, match='nn'):
        nucspinrmv(s, 0)
    with pytest.raises(ValueError, match='nn'):
        nucspinkeep(s, [0, 1])
    with pytest.raises(ValueError, match='nn'):
        nucspinadd(s, '1H', A=1.0)
    # no pair left for nn to couple: removed exactly
    assert nucspinkeep(s, 0).nn is None
    # all-zero nn is dropped
    z = SpinSystem(S=0.5, Nucs='1H,1H,1H', A=[[1] * 3] * 3, nn=[[0.0, 0.0, 0.0]] * 3)
    assert nucspinrmv(z, 0).nn is None


@pytest.mark.parametrize('state', ['singlet', ([1, 0, 0, 0], 'eigen'),
                                   ([0.25, 0.25, 0.25, 0.25], 'uncoupled'),
                                   (np.diag([1.0, 0, 0, 0]), 'coupled'),
                                   np.diag([0.5, 0.5, 0.0, 0.0])])
def test_electron_space_init_state_kept(state):
    s = SpinSystem(S=[0.5, 0.5], ee=100.0, Nucs='1H,14N', A=[[1, 1, 1, 0, 0, 0]] * 2,
                   initState=state)
    r = nucspinrmv(s, 1)
    assert _same(r.initState, s.initState)
    assert _same(nucspinadd(s, '13C', A=1.0).initState, s.initState)


def test_full_space_init_state_raises():
    s = SpinSystem(S=1, D=[300, 30], Nucs='1H', A=[5, 5, 5],
                   initState=(np.ones(6) / 6, 'zerofield'))
    with pytest.raises(ValueError, match='initState'):
        nucspinrmv(s, 0)
    with pytest.raises(ValueError, match='initState'):
        nucspinadd(s, '1H', A=1.0)
    t = SpinSystem(S=1, D=[300, 30], initState=('T0'))
    assert nucspinadd(t, '1H', A=1.0).initState == 'T0'
    x = SpinSystem(S=1, D=[300, 30], Nucs='1H', A=[5] * 3, initState=([0.1, 0.3, 0.6], 'xyz'))
    assert _same(nucspinrmv(x, 0).initState, x.initState)
    # removing a spin-0 nucleus leaves the dimension unchanged
    z = SpinSystem(S=1, D=[300, 30], Nucs='1H,12C', A=[[5] * 3, [0] * 3],
                   initState=(np.ones(6) / 6, 'zerofield'))
    assert _same(nucspinrmv(z, 1).initState, z.initState)


# ---------------------------------------------------------------------------
# nucspinadd
# ---------------------------------------------------------------------------

def test_nucspinadd_keeps_fields_and_appends():
    s = _broad_systems()['dimer_ham']
    r = nucspinadd(s, '13C', A=[1.0, 2.0, 3.0], AFrame=[0.1, 0.2, 0.3], gnscale=0.9)
    _assert_same_system(s, r, skip={'Nucs', 'n', 'A', 'AFrame', 'gnscale', 'sigma',
                                    'sigmaFrame', 'QFrame', 'nnFrame'})
    assert r.Nucs == ['1H', '14N', '13C'] and r.n == [1, 1, 1]
    assert torch.equal(r.A[:2], s.A) and torch.equal(r.A[2], torch.tensor([1., 2, 3, 1, 2, 3], dtype=F64))
    assert torch.equal(r.AFrame[2], torch.tensor([0.1, 0.2, 0.3] * 2, dtype=F64))
    assert torch.allclose(r.gnscale, torch.tensor([1.0, 1.0, 0.9], dtype=F64))
    assert r.sigma.shape == (3, 3) and r.sigmaFrame.shape == (3, 3) and r.QFrame.shape == (3, 3)


def test_nucspinadd_round_trips_with_nucspinrmv():
    s = _broad_systems()['quad_scalar']
    r = nucspinrmv(nucspinadd(s, '13C', A=[1.0, 2.0, 3.0]), 2)
    _assert_same_system(s, r)


def test_nucspinadd_quadrupole_shorthand_and_full_mixing():
    s = SpinSystem(S=0.5, Nucs='14N', A=[10, 10, 30], AFrame=[0, 0.5, 0], Q=[1.0, 0.2])
    r = nucspinadd(s, '2H', A=[1, 1, 2], Q=[0.2, 0.1])
    I = 1.0
    q = 0.2 / (4 * I * (2 * I - 1)) * np.array([-0.9, -1.1, 2])
    assert np.allclose(r.Q[1].numpy(), q)
    # a full new A turns the existing principal row (with its frame) into a full block
    Af = [[1.0, 0.2, 0.0], [0.2, 2.0, 0.0], [0.0, 0.0, 3.0]]
    f = nucspinadd(s, '1H', A=Af)
    assert f.fullA and f.A.shape == (6, 3)
    assert torch.allclose(f.A[3:], torch.tensor(Af, dtype=F64))
    assert torch.all(f.AFrame == 0)
    # the converted 14N block gives the same Hamiltonian as the principal form
    p = nucspinadd(s, '1H', A=[1, 2, 3])
    f2 = nucspinadd(s, '1H', A=np.diag([1.0, 2.0, 3.0]))
    assert f2.fullA and not p.fullA
    assert torch.allclose(ham(p, B0=[0, 0, 350.0]), ham(f2, B0=[0, 0, 350.0]), atol=1e-10)


def test_nucspinadd_keeps_graph_of_new_coupling():
    a = torch.tensor(5.0, dtype=F64, requires_grad=True)
    s = SpinSystem(S=0.5, Nucs='14N', A=[10, 10, 30])
    r = nucspinadd(s, '1H', A=a)
    r.A.sum().backward()
    assert float(a.grad) == 3.0


# ---------------------------------------------------------------------------
# consumers: pepper hybrid (EasySpin) and saffron's fast path
# ---------------------------------------------------------------------------

REF_FILE = Path(__file__).parent.parent.parent / 'tests' / 'data' / 'ref_nucspin_core.mat'

_HYBRID_CASES = {
    'dimer_aniso_cu_core': (dict(S=[0.5, 0.5], g=[[2, 2, 2.05], [2.1, 2.1, 2.0]],
                                 ee=[[2900, 2900, 3200]], Nucs='63Cu,1H',
                                 A=[[50, 50, 500, 0, 0, 0], [3, 3, 3, 0, 0, 0]], lw=[1, 0]), [1]),
    'dimer_iso': (dict(S=[0.5, 0.5], g=[2.0, 2.1], ee=1000.0, Nucs='1H',
                       A=[[4, 4, 4, 2, 2, 2]], lw=[1, 0]), []),
    'S32_ham132': (dict(S=1.5, g=[2, 2, 2], D=[300, 30], Ham={'132': [[0, 0, 0.05, 0, 0]]},
                        Nucs='1H', A=[6, 6, 6], lw=[1, 0]), []),
}


def _refs():
    if not REF_FILE.exists():
        pytest.skip(f'Reference data not found: {REF_FILE}')
    return loadmat(str(REF_FILE), squeeze_me=True, struct_as_record=False)


def _cos(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


@pytest.mark.parametrize('name', list(_HYBRID_CASES))
def test_pepper_hybrid_core_matches_easyspin(name):
    """Hybrid on a dimer / a Sys.Ham system against EasySpin's hybrid (whose core
    comes from nucspinrmv keeping every field).  On torchspin 0.8.1 the core
    lost ee / Ham: cosine to EasySpin 0.00 (dimer_iso), -0.02 (dimer_aniso_cu_core),
    0.945 (S32_ham132)."""
    ref = _refs()[name]
    kw, core = _HYBRID_CASES[name]
    exp = Experiment(mwFreq=9.5, Range=[100, 600], nPoints=2000)
    opt = Options(Method='hybrid', HybridCoreNuclei=core, GridSize=[31, 4], Verbosity=0)
    _, spc = pepper(SpinSystem(**kw), exp, opt)
    spc = spc.numpy()
    assert _cos(spc, ref.hybrid) > 0.9999
    assert abs(np.abs(spc).max() / np.abs(ref.hybrid).max() - 1) < 0.02


def test_pepper_hybrid_dimer_agrees_with_matrix():
    """Isotropic exchange: the decoupled 1H shifts are exact to first order and
    the hybrid spectrum is the matrix spectrum (on 0.8.1: cosine 0.0)."""
    kw, core = _HYBRID_CASES['dimer_iso']
    s = SpinSystem(**kw)
    exp = Experiment(mwFreq=9.5, Range=[100, 600], nPoints=2000)
    m = pepper(s, exp, Options(GridSize=[31, 4], Verbosity=0))[1].numpy()
    h = pepper(s, exp, Options(Method='hybrid', GridSize=[31, 4], Verbosity=0))[1].numpy()
    assert _cos(h, m) > 0.9999


def test_saffron_fast_core_keeps_ham():
    """saffron's fast path for S > 1/2 builds its electronic Hamiltonian from
    nucspinrmv(sys, all nuclei).  The Zeeman term written as Ham110/Ham112
    (coefficients fitted to reproduce g exactly) must give the same 3pESEEM as
    g itself; on 0.8.1 the core lost Sys.Ham, had no Zeeman term, and the
    signal was zero (relative difference 1.0).  EasySpin's saffron rejects
    Sys.Ham*, so there is no EasySpin reference."""
    from torchspin.saffron import PulseExperiment, SaffronOptions, saffron
    g = [2.01, 2.005, 1.998]
    common = dict(S=[1.5], D=[[50, 0]], Nucs='14N', A=[[2.5, 3.1, 5.4]], Q=[[-0.3, -0.1, 0.4]])
    ham_lin = {'110': [[-48.58946418805701]],
               '112': [[0.04948419845999526, 0.0, -0.10856491814573295, 0.0, 0.0]]}
    s_g = SpinSystem(g=[g], **common)
    s_h = SpinSystem(Ham=ham_lin, **common)
    mu_g = torch.stack(ham(s_g, B0=None)[1:])
    mu_h = torch.stack(ham(s_h, B0=None)[1:])
    assert torch.allclose(mu_g, mu_h, atol=1e-12)     # the two forms are the same Hamiltonian
    exp = PulseExperiment(Sequence='3pESEEM', Field=330, dt=0.012, nPoints=150,
                          tau=0.096, T=0.05, mwFreq=9.5, ExciteWidth=100)
    opt = SaffronOptions(GridSize=11, TimeDomain=True)
    y_g = np.asarray(saffron(s_g, exp, opt)[1]).ravel()
    y_h = np.asarray(saffron(s_h, exp, opt)[1]).ravel()
    yc_g, yc_h = y_g - y_g.mean(), y_h - y_h.mean()
    assert np.linalg.norm(y_h - y_g) / np.linalg.norm(y_g) < 1e-10
    assert np.linalg.norm(yc_h - yc_g) / np.linalg.norm(yc_g) < 1e-9


def test_pepper_hybrid_keeps_electron_init_state():
    """The hybrid core also carries Sys.initState: on 0.8.1 it was dropped and the
    hybrid spectrum was the thermal one (cosine 0.06 to the matrix result)."""
    s = SpinSystem(S=1, D=[1000, 100], Nucs='1H', A=[5, 5, 5], lw=[1, 0],
                   initState=([0.1, 0.3, 0.6], 'xyz'))
    exp = Experiment(mwFreq=9.5, Range=[200, 480], nPoints=2000)
    m = pepper(s, exp, Options(GridSize=[31, 4], Verbosity=0))[1].numpy()
    h = pepper(s, exp, Options(Method='hybrid', GridSize=[31, 4], Verbosity=0))[1].numpy()
    assert _cos(h, m) > 0.9999
    full = SpinSystem(S=1, D=[1000, 100], Nucs='1H', A=[5, 5, 5], lw=[1, 0],
                      initState=(np.ones(6) / 6, 'zerofield'))
    with pytest.raises(ValueError, match='initState'):
        pepper(full, exp, Options(Method='hybrid', GridSize=[31, 4], Verbosity=0))
