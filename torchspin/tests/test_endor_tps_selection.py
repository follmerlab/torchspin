"""ENDOR transition selection as EasySpin endorfrq (0.9.14; bugs B57/B41, B40, B59).

EasySpin endorfrq.m:258-341 selects the level pairs once, on a coarse grid
``sphgrid(Opt.TPSGridSymm, Opt.TPSGridSize)`` (default ``'D2h'``, 4): nuclear-moment
rates summed with the grid weights, pairs whose frequency never enters the range on
that grid dropped, electronic pairs always removed, rate ``> Threshold * max`` kept.
torchspin's ``endorfrq`` and every ``salt`` matrix path now use the same rule
(``torchspin.endorfrq._endor_tps_select``).  Before, ``salt`` kept a pair if it was
strong at any orientation of the simulation grid (so the selection depended on
``GridSize``), ``endorfrq`` thresholded per orientation, electronic pairs were kept
without ``Enhancement``, and the field-swept path thresholded per memory batch.

References: tests/data/endor_tps_selection_refs.mat from
tests/data/generate_endor_tps_selection_refs.m.
"""
import math
import os
import sys

import numpy as np
import pytest
import torch

from torchspin import Experiment, Options, SpinSystem
from torchspin.endorfrq import _endor_tps_select, endorfrq
from torchspin.ham import ham
from torchspin.ham_nz import ham_nz
from torchspin.salt import _fixed_field_transitions, salt
import torchspin.salt  # noqa: F401  (the package attribute 'salt' is the function)

_salt_mod = sys.modules['torchspin.salt']
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "data")
DT = torch.float64


def _refs():
    try:
        from scipy.io import loadmat
    except ImportError:  # pragma: no cover
        pytest.skip("scipy not installed")
    path = os.path.join(DATA_DIR, "endor_tps_selection_refs.mat")
    if not os.path.exists(path):
        pytest.skip(f"Reference file not found: {path}")
    return loadmat(path, simplify_cells=True)["ref"]


def _sys_from(S, A=None):
    kw = dict(S=[0.5], g=[np.atleast_1d(S['g']).astype(float).tolist()], Nucs=S['Nucs'],
              A=np.atleast_2d(S['A']).astype(float).tolist() if A is None else A)
    if 'Q' in S:
        kw['Q'] = np.atleast_2d(S['Q']).astype(float).tolist()
    return SpinSystem(**kw)


def _ops(s):
    return tuple(ham(s, None)) + tuple(ham_nz(s, None))


def _pairs(tr):
    return {tuple(int(v) for v in row) for row in np.atleast_2d(tr).tolist()}


def _tps_symm(r):
    v = r['TPSGridSymm']
    return v if isinstance(v, str) and v else None


def _opt(r, enh):
    kw = dict(Enhancement=enh)
    if _tps_symm(r):
        kw.update(TPSGridSymm=str(r['TPSGridSymm']), TPSGridSize=int(r['TPSGridSize']))
    return Options(**kw)


def _range(r):
    rg = np.atleast_1d(r['Range']).astype(float)
    return tuple(rg.tolist()) if rg.size == 2 else None


# ---------------------------------------------------------------------------
# endorfrq vs EasySpin endorfrq (B41, B40)
# ---------------------------------------------------------------------------

def test_endorfrq_selection_vs_easyspin():
    """Level pairs equal EasySpin's Transitions at EasySpin's threshold, range and TPS grid, for every
    orientation; positions and intensities (Enhancement off and on) per pair.  On the 0.9.2 base the
    14N+1H default-threshold set had 14 pairs where EasySpin has 16, 63Cu+14N 34-97 vs 81, and the
    1H set at threshold 0 had 6 pairs vs 2."""
    refs = _refs()["endorfrq"]
    assert len(refs) == 18
    for r in refs:
        s = _sys_from(r['Sys'])
        tr = np.atleast_2d(r['Tr']).astype(int)
        for enh, P_ref, I_ref in ((False, r['Poff'], r['Ioff']), (True, r['Pon'], r['Ion'])):
            f, I, t = endorfrq(s, float(r['Field']), float(r['phi']), float(r['theta']),
                               freq_range=_range(r), threshold=float(r['Threshold']), opt=_opt(r, enh))
            got = {tuple(x): (fi, ii) for x, fi, ii in zip(t.tolist(), f.tolist(), I.tolist())}
            assert set(got) == _pairs(tr), (r['Sys']['Nucs'], r['Threshold'], sorted(got), sorted(_pairs(tr)))
            ours = np.array([got[(u, v)] for u, v in tr.tolist()])
            np.testing.assert_allclose(ours[:, 0], np.atleast_1d(P_ref), rtol=0, atol=1e-9)
            I_ref = np.atleast_1d(I_ref)
            assert np.abs(ours[:, 1] - I_ref).max() < 1e-10 * I_ref.max()


def test_endorfrq_default_threshold_counts():
    """The EasySpin counts named in the bug reports (endorfrq, Threshold 1e-4, no range)."""
    counts = {}
    for r in _refs()["endorfrq"]:
        if float(r['Threshold']) == 1e-4 and _range(r) is None and not _tps_symm(r):
            counts.setdefault(r['Sys']['Nucs'], set()).add(len(np.atleast_2d(r['Tr'])))
            f, _, t = endorfrq(_sys_from(r['Sys']), float(r['Field']), float(r['phi']), float(r['theta']))
            assert len(t) == len(np.atleast_2d(r['Tr']))
    assert counts == {'14N,1H': {16}, '63Cu,14N': {81}}


def test_endorfrq_removes_electronic_pairs_at_threshold_zero():
    """B40: S = 1/2 + 1H at threshold 0 gives EasySpin's 2 nuclear pairs (6 on the 0.9.2 base)."""
    s = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]], Nucs='1H', A=[[3.0, 7.0, 12.0]])
    for opt in (None, Options(), Options(Enhancement=True)):
        _, _, t = endorfrq(s, 350.0, 0.3, 0.7, threshold=0.0, opt=opt)
        assert {tuple(x) for x in t.tolist()} == {(1, 2), (3, 4)}


def test_endorfrq_selection_does_not_depend_on_orientation():
    s = _sys_from(_refs()["endorfrq"][3]['Sys'])
    sets = [{tuple(x) for x in endorfrq(s, 340.0, ph, th)[2].tolist()} for ph, th in
            [(0.0, 0.0), (0.3, 0.7), (1.1, 1.3), (2.0, 2.9)]]
    assert all(x == sets[0] for x in sets) and len(sets[0]) == 16


def test_endorfrq_user_transitions_skip_selection():
    """Options.Transitions gives the level pairs directly (EasySpin Opt.Transitions): no threshold, no
    electronic-pair removal, no range filter."""
    s = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]], Nucs='1H', A=[[3.0, 7.0, 12.0]])
    f, I, t = endorfrq(s, 350.0, 0.3, 0.7, opt=Options(Transitions='all'))
    assert {tuple(x) for x in t.tolist()} == {(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)}
    f1, I1, t1 = endorfrq(s, 350.0, 0.3, 0.7, freq_range=(0, 1), threshold=0.5,
                          opt=Options(Transitions=[(4, 3)]))
    assert t1.tolist() == [[3, 4]]
    k = [tuple(x) for x in t.tolist()].index((3, 4))
    assert float(f1[0]) == float(f[k]) and float(I1[0]) == float(I[k])


# ---------------------------------------------------------------------------
# salt vs EasySpin salt (B57, B40)
# ---------------------------------------------------------------------------

def _salt_ref_case(r, enh, **opt_kw):
    S, E = r['Sys'], r['Exp']
    gs = np.atleast_1d(r['Opt']['GridSize']).astype(int).tolist()
    _, y = salt(_sys_from(S), Experiment(Field=float(E['Field']), mwFreq=float(E['mwFreq'])),
                Options(GridSize=gs, Threshold=float(r['Opt']['Threshold']), Enhancement=enh, **opt_kw),
                freq_range=tuple(np.atleast_1d(E['Range']).astype(float).tolist()), n_points=int(E['nPoints']),
                lw_mhz=float(S['lwEndor']))
    return y.numpy()


@pytest.mark.parametrize("k", [0, 1])
def test_salt_vs_easyspin(k):
    """Fixed-field salt 14N+1H, Enhancement off and on, against EasySpin at GridSize [31 6] and [13 4].
    Residual of the scaled spectrum relative to the maximum: [31 6] on 4.6e-3 on the 0.9.2 base
    (zero-quantum pairs near 0 MHz kept), now ~2e-12."""
    r = _refs()["salt"][k]
    scales = {}
    for enh, yref in ((False, r['yoff']), (True, r['yon'])):
        y, yref = _salt_ref_case(r, enh), np.asarray(yref, float)
        sc = (y @ yref) / (y @ y)
        scales[enh] = sc
        assert np.abs(sc * y - yref).max() / np.abs(yref).max() < 1e-9, (k, enh)
    assert abs(scales[True] / scales[False] - 1) < 1e-9


def test_salt_selected_pairs_are_easyspin_info_transitions():
    """B57: salt's level pairs are EasySpin's info.Transitions (16 for 14N+1H at Threshold 1e-4, on
    the 0.9.2 base 23 with Enhancement: zero-quantum pairs (8,9), (10,11) near 0 MHz) and do not
    depend on GridSize."""
    for r in _refs()["salt"][:2]:
        assert _pairs(r['Troff']) == _pairs(r['Tron'])
        ref = _pairs(r['Tron'])
        assert len(ref) == 16
        S, E = r['Sys'], r['Exp']
        s = _sys_from(S)
        sel = _endor_tps_select(_ops(s), float(E['Field']), freq_range=tuple(np.atleast_1d(E['Range']).tolist()),
                                threshold=1e-4, n_electron_states=2)
        assert {(int(u) + 1, int(v) + 1) for u, v in zip(*torch.nonzero(sel, as_tuple=True))} == ref
    # rows of the salt matrix path at several grid sizes: all 16 pairs enter the 0-70 MHz window
    s = _sys_from(_refs()["salt"][0]['Sys'])
    from torchspin.sphgrid import sphgrid
    for N in (5, 13, 31):
        ph, th, _, _ = sphgrid('D2h', N)
        for enh in (False, True):
            P, _ = _fixed_field_transitions(s, _ops(s), ph, th, 340.0, torch.ones(ph.numel(), dtype=DT),
                                            0.0, 70.0, 1e-4, False, enhancement=enh)
            assert P.shape[0] == 16, (N, enh, P.shape)


def test_salt_no_zero_frequency_spike_with_enhancement():
    """B57 symptom: the 14N+1H Enhancement spectrum had a spike in the first bins (0 MHz)."""
    r = _refs()["salt"][0]
    y, yref = _salt_ref_case(r, True), np.asarray(r['yon'], float)
    sc = (y @ yref) / (y @ y)
    assert np.abs(sc * y[:5] - yref[:5]).max() < 1e-9 * yref.max()


def test_salt_cu_no_electronic_lines():
    """B40: no intensity at the electronic (EPR) pairs near 8.9 GHz on the default rf axis
    (3e-4 of the maximum, 0.05 % of the integral, on the 0.9.2 base); EasySpin has none, and the
    level pairs are EasySpin's info.Transitions.  (The full spectrum is not compared: below
    ~600 MHz this case differs from EasySpin by up to 1e-2 of the maximum with the same level
    pairs, on the base as well; not a selection effect.)"""
    r = _refs()["salt"][2]
    x = np.asarray(r['x'], float)
    hi = (x > 5000) & (x < 15000)
    for enh, yref in ((False, r['yoff']), (True, r['yon'])):
        yref = np.asarray(yref, float)
        assert np.abs(yref[hi]).max() <= 1e-12 * np.abs(yref).max()
        y = _salt_ref_case(r, enh)
        assert np.abs(y[hi]).max() <= 1e-12 * np.abs(y).max()
    s = _sys_from(r['Sys'])
    sel = _endor_tps_select(_ops(s), 320.0, freq_range=(0.0, 19400.0), threshold=1e-4, n_electron_states=2)
    got = {(int(u) + 1, int(v) + 1) for u, v in zip(*torch.nonzero(sel, as_tuple=True))}
    assert got == _pairs(r['Troff']) == _pairs(r['Tron'])


def test_salt_user_transitions():
    """Options.Transitions in salt (EasySpin passes Opt.Transitions to endorfrq): EasySpin's 16 pairs
    give the default spectrum; with the first-order path's system it routes to the matrix path."""
    r = _refs()["salt"][0]
    y0 = _salt_ref_case(r, True)
    y1 = _salt_ref_case(r, True, Transitions=sorted(_pairs(r['Tron'])))
    np.testing.assert_allclose(y1, y0, rtol=0, atol=1e-13 * np.abs(y0).max())
    s = SpinSystem(S=[0.5], g=[[2.0, 2.01, 2.02]], Nucs='1H', A=[[6.0, 9.0, 16.0]])
    exp = Experiment(Field=326.5, mwFreq=9.7)
    kw = dict(freq_range=(2, 25), n_points=400, lw_mhz=0.3)
    _, ya = salt(s, exp, Options(GridSize=[11, 3], Transitions=[(1, 2), (3, 4)]), **kw)
    _, yb = salt(s, exp, Options(GridSize=[11, 3], Enhancement=False, Transitions='all'), **kw)
    assert torch.allclose(ya, yb, rtol=0, atol=1e-12 * float(ya.abs().max()))   # electronic pairs out of range
    with pytest.raises(NotImplementedError, match="Transitions"):
        salt(s, Experiment(Field=0.0), Options(ZeroField=True, GridSize=10, Transitions='all'),
             freq_range=(1, 20))


# ---------------------------------------------------------------------------
# Field-swept matrix path (B59)
# ---------------------------------------------------------------------------

def test_field_swept_threshold_independent_of_endor_batching(monkeypatch):
    """B59: the field-swept selection does not depend on the batch size of the ENDOR eigh (before,
    the threshold was relative to the maximum within each batch of at most 512 entries)."""
    s = SpinSystem(S=[0.5], g=[[2.0, 2.01, 2.02]], Nucs='1H,14N',
                   A=[[-2, -2, 10], [20, 20, 40]], Q=[[0, 0, 0], [-0.6, -1.0, 1.6]])
    e = Experiment(Range=[336, 350], mwFreq=9.7)
    for enh in (False, True):
        ys = []
        for cap in (1, 16, 10 ** 9):
            monkeypatch.setattr(_salt_mod, '_ENDOR_BATCH_CAP', cap)
            ys.append(salt(s, e, Options(GridSize=19, Threshold=1e-2, Enhancement=enh),
                           freq_range=(0, 40), n_points=801, lw_mhz=0.2)[1].numpy())
        for y in ys[:-1]:
            np.testing.assert_allclose(y, ys[-1], rtol=0, atol=1e-12 * ys[-1].max())


def test_field_swept_no_electronic_lines():
    """B40 in the field-swept matrix path: no lines at the EPR pairs (~9 GHz)."""
    s = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.2]], Nucs='63Cu', A=[[60.0, 60.0, 600.0]], Q=[[-5.0, -5.0, 10.0]])
    x, y = salt(s, Experiment(Range=[300, 340], mwFreq=9.7), Options(GridSize=10), n_points=4001, lw_mhz=0)
    x, y = x.numpy(), y.numpy()
    assert y.sum() > 0 and y[x > 2000].sum() == 0.0


# ---------------------------------------------------------------------------
# Gradients: selection detached, positions and intensities differentiable
# ---------------------------------------------------------------------------

def test_salt_enhanced_gradients_two_nuclei():
    """14N+1H fixed-field matrix path with Enhancement: d spec / dA vs central differences."""
    exp = Experiment(Field=340.0, mwFreq=9.55)
    opt = Options(GridSize=[7, 3], Enhancement=True)
    S = _refs()["salt"][0]['Sys']
    A0 = np.atleast_2d(S['A']).astype(float)

    def run(A):
        return salt(_sys_from(S, A=A), exp, opt, freq_range=(0, 70), n_points=300, lw_mhz=0.5)[1]
    A = torch.tensor(A0, dtype=DT, requires_grad=True)
    y = run(A)
    w = torch.linspace(0.0, 1.0, y.numel(), dtype=DT) ** 2
    (g,) = torch.autograd.grad((w * y).sum(), A)
    for (i, c) in ((0, 2), (1, 0)):
        best = math.inf
        for h in (1e-4, 1e-5, 1e-6):
            Ap, Am = A0.copy(), A0.copy()
            Ap[i, c] += h
            Am[i, c] -= h
            fd = float(((w * run(Ap.tolist())).sum() - (w * run(Am.tolist())).sum()) / (2 * h))
            best = min(best, abs(float(g[i, c]) - fd) / abs(fd))
        assert best < 1e-5, (i, c, best)
