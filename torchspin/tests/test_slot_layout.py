"""Vectorized slot bookkeeping (perf train P1): ``pepper._slot_layout``,
``pepper._transition_slots``, ``pepper._pair_counts`` and ``pepper_autograd._slots_t``
must reproduce the per-orientation loops they replaced exactly: same slots, same
order, same values, same slot keys, on random line lists with looping transitions
(repeated level pairs), empty orientations, orientations without level pairs, and
width arrays shorter than the line list.  The reference loops below are the 0.9.38
implementations."""
import sys

import numpy as np
import pytest
import torch

import torchspin.pepper  # noqa: F401  (module, not the re-exported function)
from torchspin.pepper_autograd import _slots_t

_pep = sys.modules['torchspin.pepper']


# ----------------------------------------------------------------------------- 0.9.38 reference loops
def _ref_transition_slots(all_B, all_I, all_W, all_pairs, n_orient):
    have_widths = any(w is not None and w.numel() > 0 for w in all_W) if all_W is not None else False
    if all_pairs is None or all(p is None for p in all_pairs):
        n_slots = max((b.numel() if b is not None else 0) for b in all_B) if all_B else 0
        B_arrs = [np.full(n_orient, np.nan) for _ in range(n_slots)]
        I_arrs = [np.full(n_orient, np.nan) for _ in range(n_slots)]
        W_arrs = [np.full(n_orient, np.nan) for _ in range(n_slots)] if have_widths else None
        for k in range(n_orient):
            B = all_B[k]
            if B is None:
                continue
            Bn = B.numpy(); In = all_I[k].numpy()
            Wn = all_W[k].numpy() if (have_widths and all_W[k] is not None) else None
            for i in range(Bn.shape[0]):
                B_arrs[i][k] = Bn[i]; I_arrs[i][k] = In[i]
                if Wn is not None and i < Wn.shape[0]:
                    W_arrs[i][k] = Wn[i]
        return B_arrs, I_arrs, W_arrs, None
    slots: dict = {}
    B_arrs, I_arrs, W_arrs = [], [], []
    for k in range(n_orient):
        B = all_B[k]; pairs = all_pairs[k]
        if B is None or pairs is None or B.numel() == 0:
            continue
        Bn = B.numpy(); In = all_I[k].numpy()
        Wn = all_W[k].numpy() if (have_widths and all_W[k] is not None) else None
        pn = pairs.numpy()
        order = sorted(range(Bn.shape[0]), key=lambda i: (int(pn[i, 0]), int(pn[i, 1]), float(Bn[i])))
        occ: dict = {}
        for i in order:
            uv = (int(pn[i, 0]), int(pn[i, 1]))
            c = occ.get(uv, 0); occ[uv] = c + 1
            key = (uv[0], uv[1], c)
            if key not in slots:
                slots[key] = len(B_arrs)
                B_arrs.append(np.full(n_orient, np.nan)); I_arrs.append(np.full(n_orient, np.nan))
                W_arrs.append(np.full(n_orient, np.nan))
            j = slots[key]
            B_arrs[j][k] = Bn[i]; I_arrs[j][k] = In[i]
            if Wn is not None and i < Wn.shape[0]:
                W_arrs[j][k] = Wn[i]
    keys = sorted(slots, key=lambda kk: slots[kk])
    return B_arrs, I_arrs, (W_arrs if have_widths else None), keys


def _ref_pair_counts(all_B, all_pairs):
    counts: dict = {}
    if all_pairs is None or all(p is None for p in all_pairs):
        return counts
    n_orient = len(all_B)
    for k in range(n_orient):
        B, P = all_B[k], all_pairs[k]
        if B is None or P is None or B.numel() == 0:
            continue
        uv, c = np.unique(P.numpy().astype(np.int64), axis=0, return_counts=True)
        for (u, v), n in zip(uv, c):
            counts.setdefault((int(u), int(v)), np.zeros(n_orient, dtype=np.int64))[k] = n
    return counts


def _ref_slots_keys_t(all_B, all_pairs, n_orient):
    slots: dict = {}
    s_idx, o_idx = [], []
    for k in range(n_orient):
        B = all_B[k]
        if B is None or B.numel() == 0:
            continue
        Bn = B.numpy()
        pn = all_pairs[k].numpy() if all_pairs is not None and all_pairs[k] is not None else None
        if pn is None:
            for i in range(Bn.shape[0]):
                slots.setdefault(('idx', i), len(slots)); s_idx.append(slots[('idx', i)]); o_idx.append(k)
            continue
        order = sorted(range(Bn.shape[0]), key=lambda i: (int(pn[i, 0]), int(pn[i, 1]), float(Bn[i])))
        occ: dict = {}
        for i in order:
            uv = (int(pn[i, 0]), int(pn[i, 1]))
            c = occ.get(uv, 0); occ[uv] = c + 1
            slots.setdefault((uv[0], uv[1], c), len(slots)); s_idx.append(slots[(uv[0], uv[1], c)]); o_idx.append(k)
    return sorted(slots, key=lambda kk: slots[kk])


# ----------------------------------------------------------------------------- random line lists
def _lists(seed, n_orient=40, pairs='all', widths=True):
    rng = np.random.default_rng(seed)
    all_B, all_I, all_W, all_P = [], [], [], []
    for k in range(n_orient):
        m = int(rng.integers(0, 9))
        if rng.random() < 0.1:
            all_B.append(None if rng.random() < 0.5 else torch.zeros(0, dtype=torch.float64))
            all_I.append(None if all_B[-1] is None else torch.zeros(0, dtype=torch.float64))
            all_W.append(None); all_P.append(None)
            continue
        # few distinct pairs, so pairs repeat (looping transitions); some equal fields
        P = torch.as_tensor(rng.integers(0, 3, (m, 2)) + np.array([0, 3]), dtype=torch.long)
        B = torch.as_tensor(np.round(rng.uniform(300, 320, m), 1), dtype=torch.float64)
        all_B.append(B); all_I.append(torch.as_tensor(rng.random(m)))
        w = None
        if widths and rng.random() < 0.8:
            w = torch.as_tensor(rng.random(int(rng.integers(max(m - 2, 0), m + 1))))
        all_W.append(w)
        if pairs == 'none' or (pairs == 'mixed' and rng.random() < 0.3):
            all_P.append(None)
        else:
            all_P.append(P)
    return all_B, all_I, all_W, all_P


def _eq(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert np.array_equal(x, y, equal_nan=True)


@pytest.mark.parametrize('seed', range(12))
@pytest.mark.parametrize('pairs', ['all', 'none', 'mixed'])
@pytest.mark.parametrize('widths', [True, False])
def test_transition_slots_match_loop(seed, pairs, widths):
    B, I, W, P = _lists(seed, pairs=pairs, widths=widths)
    n = len(B)
    rB, rI, rW, rkeys = _ref_transition_slots(B, I, W, P, n)
    nB, nI, nW = _pep._transition_slots(B, I, W, P, n)
    _eq(rB, nB); _eq(rI, nI)
    assert (rW is None) == (nW is None)
    if rW is not None:
        _eq(rW, nW)
    assert _pep._slot_state.keys == rkeys


@pytest.mark.parametrize('seed', range(12))
@pytest.mark.parametrize('pairs', ['all', 'none', 'mixed'])
def test_pair_counts_match_loop(seed, pairs):
    B, _, _, P = _lists(seed, pairs=pairs)
    ref, new = _ref_pair_counts(B, P), _pep._pair_counts(B, P)
    assert list(ref) == list(new)
    for uv in ref:
        assert np.array_equal(ref[uv], new[uv]) and new[uv].dtype == np.int64


@pytest.mark.parametrize('seed', range(12))
@pytest.mark.parametrize('pairs', ['all', 'none', 'mixed'])
def test_slots_t_keys_and_values(seed, pairs):
    B, I, W, P = _lists(seed, pairs=pairs, widths=False)
    n = len(B)
    keys = _ref_slots_keys_t(B, P, n)
    assert _pep._slot_layout(B, P, n, unpaired='index')[3] == keys
    Bs, Is = _slots_t(B, I, P, n, threshold=0.0)
    assert Bs.shape == (len(keys), n)
    # every resonance sits in its slot at its orientation, everything else is NaN
    filled = (~torch.isnan(Bs)).sum().item()
    assert filled == sum(b.numel() for b in B if b is not None)
    for k in range(n):
        if B[k] is None or B[k].numel() == 0:
            assert torch.isnan(Bs[:, k]).all()
            continue
        col = Bs[:, k][~torch.isnan(Bs[:, k])]
        assert sorted(col.tolist()) == sorted(B[k].tolist())
