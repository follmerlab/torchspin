"""Sets of equivalent nuclei as total-spin blocks in pepper's matrix method (B139, B135).

n equivalent nuclei enter the Hamiltonian through their total spin K only, so
the full space is a direct sum of blocks with one nucleus of spin K.  The full
space has exactly degenerate levels (the copies of a K); there the eigenvectors
of each resonance are an arbitrary basis of the degenerate set, and summed
intensities came out wrong (three equivalent 1H on a single crystal: 7 % lost)
and dependent on ``Options.BatchSize`` (Cu + 3 14N: 5 %).
"""
import importlib

import numpy as np
import pytest
import torch

from torchspin import Experiment, Options, SpinSystem, pepper
from torchspin.equivnuclei import equivalent_sets, reduce_equivalent_nuclei
from torchspin.ham import ham

SYS = {
    'cu2n': dict(S=[0.5], g=[2.04, 2.06, 2.25], Nucs='63Cu,14N,14N',
                 A=[[30, 40, 500], [38, 40, 45], [38, 40, 45]],
                 AFrame=[[0, 0, 0], [0.3, 0.5, 0.1], [0.3, 0.5, 0.1]], lwpp=[0.5, 0]),
    'ch3': dict(S=0.5, g=[2.002, 2.003, 2.004], Nucs='1H,1H,1H', A=[[60, 65, 70]] * 3, lwpp=0.2),
    'quartet_2h': dict(S=1.5, g=2.0, D=[600, 100], Nucs='1H,1H', A=[[8, 8, 14], [8, 8, 14]], lwpp=0.4),
    'trip_2n': dict(S=1, g=2.0, D=[900, 150], Nucs='14N,14N', A=[[10, 12, 40]] * 2, lwpp=0.5),
    'n_and_2h_2h': dict(S=0.5, g=[2.003, 2.004, 2.008], Nucs='14N,1H,1H,1H,1H',
                        A=[[15, 15, 90], [10, 11, 12], [10, 11, 12], [4, 5, 6], [4, 5, 6]], lwpp=0.2),
}
RANGE = {'cu2n': [260, 360], 'ch3': [325, 345], 'quartet_2h': [200, 480], 'trip_2n': [280, 400],
         'n_and_2h_2h': [328, 346]}


def _crystal(name, **kw):
    return Experiment(nPoints=2048, Harmonic=0, mwFreq=9.5, Range=RANGE[name], MolFrame=[0.4, 0.9, 0.2],
                      SampleFrame=[0.1, 0.2, 0.3], **kw)


def _powder(name, **kw):
    return Experiment(nPoints=1024, Harmonic=0, mwFreq=9.5, Range=RANGE[name], **kw)


def _spec(sys, exp, **opt):
    return pepper(sys, exp, Options(Verbosity=0, **opt))[1].numpy()


def _rel(a, b):
    return np.abs(a - b).max() / np.abs(b).max()


# ── the reduction ────────────────────────────────────────────────────────────

class TestBlocks:
    def test_three_14N(self):
        s = SpinSystem(S=0.5, g=2.0, Nucs='63Cu,14N,14N,14N', A=[[30, 40, 500], [40] * 3, [40] * 3, [40] * 3])
        assert equivalent_sets(s) == [([1, 2, 3], 3)]
        blocks = reduce_equivalent_nuclei(s)
        assert [(b.Nucs, b.I) for _, b in blocks] == [(['63Cu', '14N'], [1.5, 3.0]), (['63Cu', '14N'], [1.5, 2.0]),
                                                      (['63Cu', '14N'], [1.5, 1.0]), (['63Cu'], [1.5])]
        np.testing.assert_allclose([w for w, _ in blocks], [7 / 27, 10 / 27, 9 / 27, 1 / 27], rtol=1e-14)

    @pytest.mark.parametrize('name', list(SYS))
    def test_weights_sum_to_one_and_count_the_states(self, name):
        s = SpinSystem(**SYS[name])
        blocks = reduce_equivalent_nuclei(s)
        assert len(blocks) > 1
        assert sum(w for w, _ in blocks) == pytest.approx(1.0, abs=1e-14)
        dim = ham(s)[0].shape[0]
        for w, b in blocks:
            mult = w * dim / ham(b)[0].shape[0]
            assert mult == pytest.approx(round(mult), abs=1e-9)

    def test_two_sets(self):
        blocks = reduce_equivalent_nuclei(SpinSystem(**SYS['n_and_2h_2h']))
        assert sorted(b.I for _, b in blocks) == sorted([[1.0, 1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0]])

    def test_n_counts(self):
        s = SpinSystem(S=0.5, g=2.0, Nucs='1H', n=3, A=[[60, 65, 70]])
        blocks = reduce_equivalent_nuclei(s)
        assert [(w, b.I, b.n) for w, b in blocks] == [(0.5, [1.5], [1]), (0.5, [0.5], [1])]

    def test_block_keeps_the_other_fields(self):
        s = SpinSystem(S=0.5, g=[2.0, 2.1, 2.2], gFrame=[0.1, 0.2, 0.3], Nucs='14N,14N', A=[[10, 12, 40]] * 2,
                       AFrame=[[0.3, 0.2, 0.1]] * 2, gnscale=[1.1, 1.1], lwpp=0.5, HStrain=[5, 6, 7], weight=0.3)
        for _, b in reduce_equivalent_nuclei(s):
            assert b.weight == 0.3 and torch.equal(b.g, s.g) and torch.equal(b.HStrain, s.HStrain)
            if b.Nucs:
                assert torch.equal(b.A, s.A[:1]) and torch.equal(b.AFrame, s.AFrame[:1])
                assert torch.equal(torch.as_tensor(b.gnscale), torch.as_tensor(s.gnscale)[:1])

    @pytest.mark.parametrize('change', [
        dict(A=[[10, 12, 40], [10, 12, 40.001]]),
        dict(AFrame=[[0, 0, 0], [0, 0.1, 0]]),
        dict(gnscale=[1.0, 1.01]),
        dict(Nucs='14N,15N'),
        dict(Q=[[0.5, 0.1], [0.5, 0.1]]),
        dict(nn=[[0.01, 0.01, -0.02]]),
        dict(AStrain=[1, 1, 3]),
    ])
    def test_not_reduced(self, change):
        kw = dict(S=0.5, g=2.0, Nucs='14N,14N', A=[[10, 12, 40]] * 2, lwpp=0.5)
        kw.update(change)
        s = SpinSystem(**kw)
        assert equivalent_sets(s) == []
        blocks = reduce_equivalent_nuclei(s)
        assert len(blocks) == 1 and blocks[0][0] == 1.0 and blocks[0][1] is s

    def test_astrain_nucleus_stays_out_of_the_set(self):
        s = SpinSystem(S=0.5, g=2.0, Nucs='14N,14N,14N', A=[[10, 12, 40]] * 3, AStrain=[1, 1, 3], lwpp=0.5)
        assert equivalent_sets(s) == [([1, 2], 2)]
        for _, b in reduce_equivalent_nuclei(s):
            assert b.Nucs[0] == '14N' and b.I[0] == 1.0 and b.AStrain is not None

    def test_h0extra_and_full_space_init_state(self):
        op = torch.eye(18, dtype=torch.complex128)
        assert equivalent_sets(SpinSystem(S=0.5, g=2.0, Nucs='14N,14N', A=[[10, 12, 40]] * 2, H0Extra=[(1.0, op)])) == []
        s = SpinSystem(S=1, g=2.0, D=[900, 150], Nucs='1H,1H', A=[[5, 5, 9]] * 2, initState='T0')
        assert equivalent_sets(s) == [([0, 1], 2)]

    def test_quadrupole_free_spin_half_is_reduced_with_Q_rows_of_zero(self):
        s = SpinSystem(S=0.5, g=2.0, Nucs='14N,1H,1H', A=[[10, 12, 40], [5, 5, 9], [5, 5, 9]],
                       Q=[[0.5, 0.1], [0, 0], [0, 0]])
        assert equivalent_sets(s) == [([1, 2], 2)]


@pytest.mark.parametrize('name', list(SYS))
def test_levels_are_the_union_of_the_block_levels(name):
    s = SpinSystem(**SYS[name])
    n = np.array([0.3, 0.5, 0.81]); n /= np.linalg.norm(n)

    def levels(q):
        H0, mx, my, mz = (t.numpy() for t in ham(q))
        return np.linalg.eigvalsh(H0 - 337.0 * (n[0] * mx + n[1] * my + n[2] * mz))

    full = np.sort(levels(s))
    parts = []
    for w, b in reduce_equivalent_nuclei(s):
        e = levels(b)
        parts += list(np.repeat(e, int(round(w * full.size / e.size))))
    assert len(parts) == full.size
    np.testing.assert_allclose(np.sort(parts), full, atol=1e-8)


# ── spectra ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('name', ['cu2n', 'quartet_2h', 'trip_2n'])
@pytest.mark.parametrize('temperature', [None, 1.0])
def test_single_crystal_equals_the_full_space(name, temperature):
    """One orientation and every transition: both routes are exact for these systems."""
    s, e = SpinSystem(**SYS[name]), _crystal(name, Temperature=temperature)
    full = _spec(s, e, Threshold=0, EquivalentNuclei=False)
    blocks = _spec(s, e, Threshold=0)
    assert full.max() > 0
    assert _rel(blocks, full) < 5e-6


@pytest.mark.parametrize('name', ['ch3', 'n_and_2h_2h'])
def test_single_crystal_intensity_sum_rule(name):
    """S = 1/2: nuclei redistribute the intensity of the electron transition but
    keep its sum.  The full space lost 7 % for three equivalent 1H."""
    kw = dict(SYS[name]); e = _crystal(name)
    bare = {k: v for k, v in kw.items() if k not in ('Nucs', 'A', 'AFrame')}
    ref = _spec(SpinSystem(**bare), e, Threshold=0).sum()
    assert _spec(SpinSystem(**kw), e, Threshold=0).sum() == pytest.approx(ref, rel=1e-5)


@pytest.mark.parametrize('name', ['quartet_2h', 'trip_2n'])
def test_powder_equals_the_full_space(name):
    s, e = SpinSystem(**SYS[name]), _powder(name)
    full = _spec(s, e, Threshold=0, GridSize=[8, 2], EquivalentNuclei=False)
    blocks = _spec(s, e, Threshold=0, GridSize=[8, 2])
    assert _rel(blocks, full) < 1e-6


@pytest.mark.parametrize('name', ['cu2n', 'ch3'])
def test_powder_integral_equals_the_bare_electron(name):
    kw = dict(SYS[name]); e = _powder(name)
    bare = {k: v for k, v in kw.items() if k not in ('Nucs', 'A', 'AFrame')}
    ref = _spec(SpinSystem(**bare), e, Threshold=0, GridSize=[8, 2]).sum()
    assert _spec(SpinSystem(**kw), e, Threshold=0, GridSize=[8, 2]).sum() == pytest.approx(ref, rel=2e-3)


@pytest.mark.parametrize('name', ['cu2n', 'ch3', 'n_and_2h_2h'])
def test_spectrum_does_not_depend_on_the_batch(name):
    s, e = SpinSystem(**SYS[name]), _powder(name)
    one = _spec(s, e, GridSize=[6, 2], BatchSize=1)
    for bs in (3, 1000):
        assert _rel(_spec(s, e, GridSize=[6, 2], BatchSize=bs), one) < 1e-6


def test_cu_3n_does_not_depend_on_the_batch():
    """B139: 5.3 % between BatchSize 1 and 10 in the full space (dimension 216)."""
    s = SpinSystem(S=[0.5], g=[2.04, 2.06, 2.25], Nucs='63Cu,14N,14N,14N',
                   A=[[30, 40, 500], [40] * 3, [40] * 3, [40] * 3], lwpp=[0.6, 0])
    e = Experiment(nPoints=1024, Harmonic=0, mwFreq=9.5, Range=[260, 360])
    one = _spec(s, e, GridSize=[4, 4], BatchSize=1)
    assert _rel(_spec(s, e, GridSize=[4, 4], BatchSize=10), one) < 1e-5
    bare = _spec(SpinSystem(S=[0.5], g=[2.04, 2.06, 2.25], lwpp=[0.6, 0]), e, GridSize=[4, 4])
    assert one.sum() == pytest.approx(bare.sum(), rel=5e-3)        # the full space kept 97.6 to 98.5 %


# ── SpinSystem.n and the options ─────────────────────────────────────────────

@pytest.mark.parametrize('name, short', [
    ('ch3', dict(Nucs='1H', n=3, A=[[60, 65, 70]])),
    ('trip_2n', dict(Nucs='14N', n=2, A=[[10, 12, 40]])),
    ('cu2n', dict(Nucs='63Cu,14N', n=[1, 2], A=[[30, 40, 500], [38, 40, 45]], AFrame=[[0, 0, 0], [0.3, 0.5, 0.1]])),
])
def test_n_equals_listing_the_nuclei(name, short):
    kw = dict(SYS[name]); e = _powder(name)
    listed = _spec(SpinSystem(**kw), e, GridSize=[6, 2])
    kw.update(short)
    assert _rel(_spec(SpinSystem(**kw), e, GridSize=[6, 2]), listed) < 1e-9


@pytest.mark.parametrize('sys_kw, opt', [
    (dict(Q=[[0.5, 0.1]]), {}),
    ({}, dict(EquivalentNuclei=False)),
    ({}, dict(separate='transitions')),
    ({}, dict(Transitions=[(1, 4)])),
])
def test_n_that_cannot_be_reduced_raises(sys_kw, opt):
    s = SpinSystem(S=0.5, g=2.0, Nucs='14N', n=2, A=[[10, 12, 40]], lwpp=0.5, **sys_kw)
    with pytest.raises(ValueError, match='only as total-spin blocks'):
        pepper(s, _powder('trip_2n'), Options(Verbosity=0, GridSize=[6, 2], **opt))


@pytest.fixture
def dims(monkeypatch):
    """Hilbert-space dimensions seen by the batched resonance search."""
    pp = importlib.import_module('torchspin.pepper')
    seen = []
    orig = pp.resfields_batch

    def spy(H0, *a, **k):
        seen.append(H0.shape[-1])
        return orig(H0, *a, **k)

    monkeypatch.setattr(pp, 'resfields_batch', spy)
    return seen


def test_default_uses_the_blocks(dims):
    _spec(SpinSystem(**SYS['trip_2n']), _powder('trip_2n'), GridSize=[6, 2])
    assert sorted(set(dims)) == [3, 9, 15]


@pytest.mark.parametrize('opt', [dict(EquivalentNuclei=False), dict(separate='transitions'),
                                 dict(separate='orientations'), dict(Transitions=[(1, 10), (2, 11)])])
def test_full_space_when_asked_or_needed(dims, opt):
    _spec(SpinSystem(**SYS['trip_2n']), _powder('trip_2n'), GridSize=[6, 2], **opt)
    assert set(dims) == {27}


def test_quadrupole_keeps_the_full_space(dims):
    kw = dict(SYS['trip_2n'], Q=[[0.5, 0.1], [0.5, 0.1]])
    _spec(SpinSystem(**kw), _powder('trip_2n'), GridSize=[6, 2])
    assert set(dims) == {27}


def test_separate_components_keeps_one_row_per_component():
    a = SpinSystem(**SYS['trip_2n'])
    b = SpinSystem(S=0.5, g=2.0, lwpp=0.5, weight=0.5)
    e = _powder('trip_2n')
    rows = _spec([a, b], e, GridSize=[6, 2], separate='components')
    assert rows.shape == (2, 1024)
    np.testing.assert_allclose(rows[0], _spec(a, e, GridSize=[6, 2]), rtol=0, atol=1e-12 * rows[0].max())
    np.testing.assert_allclose(rows.sum(0), _spec([a, b], e, GridSize=[6, 2]), rtol=0, atol=1e-12 * rows.max())


@pytest.mark.parametrize('method', ['perturb', 'hybrid'])
def test_other_methods_are_unchanged_by_the_option(method):
    s, e = SpinSystem(**SYS['cu2n']), _powder('cu2n')
    opt = dict(GridSize=[6, 2], Method=method)
    if method == 'hybrid':
        opt['HybridCoreNuclei'] = [1]
    assert np.array_equal(_spec(s, e, **opt), _spec(s, e, EquivalentNuclei=False, **opt))
