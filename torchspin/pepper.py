"""Solid-state CW EPR simulator (port of EasySpin's ``pepper``).

The public entry point is :func:`pepper`; see its docstring for the
supported inputs.  Outline of one component simulation::

    ham(sys)                     -> H0, mux, muy, muz (MHz, MHz/mT)
    sphgrid(GridSymmetry, N)     -> powder orientations and weights
                                    (or crystal / Exp.Orientations directions)
    resfields_batch / resfreqs   -> resonance positions, rates, strain widths
                                    per orientation and transition
    grid interpolation           -> fine grid ([N_coarse, N_interp])
    projection / line templates  -> spectral density on the sweep axis
    convolution with Sys.lw      -> line shape, harmonic, field modulation

Components and isotopologues are simulated separately and summed with
``Sys.weight`` times abundance.  :func:`pepper` switches to the
differentiable :func:`torchspin.pepper_autograd` when an
input requires grad.
"""
from __future__ import annotations

import functools
import math
import threading
import warnings
from typing import Optional

import torch

import numpy as np

from torchspin.convspec import convspec
from torchspin import _projection as _proj
from torchspin.isotopologues import expand_components
from torchspin.experiment import (auto_batch_size, Experiment, Options,
                                  PEPPER_PERTURB_ORDER)
from torchspin.ham import ham
from torchspin.hamsymm import grid_symmetry
from torchspin.rotations import erot
from torchspin.makespec import makespec
from torchspin.resfields import resfields
from torchspin.resfields_batch import resfields_batch
from torchspin.resfields_perturb import resfields_perturb_batch
from torchspin.resfreqs_matrix import resfreqs_matrix, resfreqs_batch
from torchspin.sphgrid import sphgrid, _d2h_triangulation, _triangle_areas, grid_triangulation, gridparam
from torchspin.spinsystem import SpinSystem
from torchspin.utils import hsdim

__all__ = ['pepper']

# Grid symmetries that use a closedPhi, nOct=1 structure (triangulatable by _d2h_triangulation)
_CLOSEDPHI_1OCT = frozenset({'D2h', 'D4h', 'D6h', 'D3d', 'Th', 'Oh'})


_INTERP_MATRIX_MAX = 8_000_000     # n_coarse × n_fine above which the scipy splines are used
# Facets per slot from which the slot projections run in parallel workers; below,
# the per-op Python overhead dominates and the GIL serializes a thread pool.
_POOL_MIN_FACETS = 4000


@functools.lru_cache(maxsize=16)
def _d2h_interp_matrix(N_c: int, N_f: int) -> np.ndarray:
    """Dense ``(n_fine, n_coarse)`` matrix of EasySpin's G3 interpolation on the
    triangular D2h SOPHE grid (row-wise cubic resampling along φ to an
    N_c × N_c rectangle with zero end slopes, as EasySpin gridinterp.m:184
    ``esspline1d(...,1,nc)``; then the tensor-product not-a-knot bicubic
    spline — identical to ``RectBivariateSpline(kx=ky=3, s=0)`` and EasySpin
    ``esspline2d`` — evaluated at the fine knots).  Both steps are linear in
    the values, so one matrix product per transition slot replaces ~N_c spline
    constructions.
    """
    from scipy.interpolate import CubicSpline
    n_c = N_c * (N_c + 1) // 2
    phi_rect = np.linspace(0.0, np.pi / 2, N_c)
    theta_1d = np.linspace(0.0, np.pi / 2, N_c)
    R = np.zeros((N_c * N_c, n_c))
    idx = 0
    for r in range(1, N_c + 1):
        rows = slice((r - 1) * N_c, r * N_c)
        if r == 1:
            R[rows, idx] = 1.0
        elif r == N_c:
            R[rows, idx:idx + r] = np.eye(r)
        else:
            # zero end slopes (EasySpin esspline1d EndCon=1; for r = 2 the cubic
            # Hermite polynomial with zero slopes, esspline1d.m:55-60)
            R[rows, idx:idx + r] = CubicSpline(np.linspace(0.0, np.pi / 2, r), np.eye(r), axis=0,
                                               bc_type='clamped')(phi_rect)
        idx += r
    rows_f = np.arange(1, N_f + 1)
    theta_all = np.repeat((rows_f - 1) / (N_f - 1) * np.pi / 2 if N_f > 1 else np.zeros(1), rows_f)
    phi_all = np.concatenate([np.array([0.0])] + [np.linspace(0.0, np.pi / 2, r) for r in range(2, N_f + 1)])
    S_th = CubicSpline(theta_1d, np.eye(N_c), axis=0)(theta_all)      # (n_f, N_c)
    S_ph = CubicSpline(phi_rect, np.eye(N_c), axis=0)(phi_all)        # (n_f, N_c)
    # value_k = Σ_ij S_th[k,i] Z[i,j] S_ph[k,j] with Z = (R v).reshape(N_c, N_c)
    W = (S_th[:, :, None] * S_ph[:, None, :]).reshape(S_th.shape[0], N_c * N_c)
    return W @ R


def _interp_dinfh_g3(th_c: np.ndarray, th_f: np.ndarray, values: np.ndarray) -> np.ndarray:
    """EasySpin ``gridinterp`` G3 on an axial (Dinfh) grid: ``esspline1d(y, 1, iphi)``
    (gridinterp.m:77-78), a cubic spline over the knot index ``1..n`` with zero
    first derivatives at both ends (esspline1d.m ``EndCon = 1``; for two knots the
    zero-slope cubic Hermite polynomial), evaluated at the fine knots'
    fractional indices.  The Dinfh knots are equally spaced in θ, so the index
    is θ/Δθ.  Up to 0.9.6 torchspin used a not-a-knot spline in cos θ here.
    Linear in ``values``, so pepper_autograd's cached matrix stays exact."""
    from scipy.interpolate import CubicSpline
    order = np.argsort(th_c)
    th_s = th_c[order]
    n = th_s.size
    if n < 2:
        return np.full(th_f.shape, float(values[0]))
    step = (th_s[-1] - th_s[0]) / (n - 1)
    if not np.allclose(np.diff(th_s), step, rtol=1e-9, atol=1e-12):
        raise ValueError('Dinfh interpolation needs knots equally spaced in theta.')
    xi = np.clip((th_f - th_s[0]) / step, 0.0, n - 1.0)
    return CubicSpline(np.arange(n, dtype=float), values[order], bc_type='clamped')(xi)


def _interp_sph(
    coarse_vecs: torch.Tensor,
    fine_vecs: torch.Tensor,
    values: np.ndarray,
) -> np.ndarray:
    """Interpolate per-orientation scalar data from a coarse to a fine SOPHE grid.

    For D2h-type triangular SOPHE grids, uses MATLAB's G3 structured interpolation:
    first a row-by-row cubic spline along phi (converting triangular→rectangular),
    then a 2-D bicubic spline in (theta, phi) space.  This matches EasySpin's
    ``gridinterp(..., 'G3')`` and eliminates the quantization artifacts that arise
    from unstructured 2-D scattered interpolation.

    For Dinfh grids (1-D meridional), uses :func:`_interp_dinfh_g3` (EasySpin's
    spline over the θ knot index with zero end slopes).

    For full-sphere (C1) grids, falls back to CloughTocher 2-D scattered interpolation.

    Parameters
    ----------
    coarse_vecs : torch.Tensor, shape (3, N_coarse)
        Unit vectors for the coarse grid (rows: x, y, z).
    fine_vecs : torch.Tensor, shape (3, N_fine)
        Unit vectors for the fine grid.
    values : np.ndarray, shape (N_coarse,)
        Scalar quantity to interpolate (resonance field, intensity, or width).

    Returns
    -------
    np.ndarray, shape (N_fine,)
    """
    x_c = coarse_vecs[0].numpy()
    y_c = coarse_vecs[1].numpy()
    z_c = coarse_vecs[2].numpy()
    z_f = fine_vecs[2].numpy()

    # ---- 1-D meridional grids (Dinfh: all phi = 0, y ≈ 0) ----
    if np.std(y_c) < 1e-8:
        x_f, y_f = fine_vecs[0].numpy(), fine_vecs[1].numpy()
        th_c = np.arctan2(np.hypot(x_c, y_c), z_c)
        th_f = np.arctan2(np.hypot(x_f, y_f), z_f)
        return _interp_dinfh_g3(th_c, th_f, np.asarray(values, dtype=float))

    n_c = len(values)

    # ---- Detect D2h triangular SOPHE grid structure ----
    # A triangular SOPHE grid with N knots along the meridian has N*(N+1)//2 points.
    disc_c = 1 + 8 * n_c
    sqrt_c = math.isqrt(disc_c)
    is_d2h_structured = (sqrt_c * sqrt_c == disc_c) and ((sqrt_c - 1) % 2 == 0)

    n_f = fine_vecs.shape[1]
    disc_f = 1 + 8 * n_f
    sqrt_f = math.isqrt(disc_f)
    is_d2h_fine = (sqrt_f * sqrt_f == disc_f) and ((sqrt_f - 1) % 2 == 0)

    if is_d2h_structured and is_d2h_fine:
        from scipy.interpolate import CubicSpline, RectBivariateSpline
        N_c = (sqrt_c - 1) // 2
        N_f = (sqrt_f - 1) // 2
        if n_c * n_f <= _INTERP_MATRIX_MAX:
            # the G3 interpolation is linear in the values: cached matrix
            return _d2h_interp_matrix(N_c, N_f) @ np.asarray(values, dtype=float)

        # Step 1: Convert triangular coarse grid → rectangular array z_rect[N_c × N_c]
        # by interpolating each theta row along phi via a cubic spline with zero
        # end slopes (EasySpin gridinterp.m:184, esspline1d(...,1,nc)).
        theta_1d = np.linspace(0.0, np.pi / 2, N_c)
        phi_rect = np.linspace(0.0, np.pi / 2, N_c)  # target phi grid (N_c columns)

        z_rect = np.zeros((N_c, N_c))
        idx = 0
        for r in range(1, N_c + 1):
            row_vals = values[idx: idx + r]
            idx += r
            if r == 1:
                z_rect[0, :] = row_vals[0]  # north pole: constant in phi
            elif r == N_c:
                z_rect[N_c - 1, :] = row_vals   # equator: already has N_c points
            else:
                phi_row = np.linspace(0.0, np.pi / 2, r)
                cs = CubicSpline(phi_row, row_vals, bc_type='clamped')
                z_rect[r - 1, :] = cs(phi_rect)

        # Step 2: 2-D bicubic spline on the rectangular coarse grid.
        spline2d = RectBivariateSpline(theta_1d, phi_rect, z_rect, kx=3, ky=3, s=0)

        # Step 3: Evaluate at the fine triangular grid positions (one call).
        rows = np.arange(1, N_f + 1)
        theta_all = np.repeat((rows - 1) / (N_f - 1) * np.pi / 2 if N_f > 1 else np.zeros(1), rows)
        phi_all = np.concatenate([np.array([0.0])] + [np.linspace(0.0, np.pi / 2, r) for r in range(2, N_f + 1)])
        return spline2d.ev(theta_all, phi_all)

    # ---- Fallback: 2-D scattered interpolation (Ci / C1 / other) ----
    from scipy.interpolate import CloughTocher2DInterpolator, NearestNDInterpolator

    x_f = fine_vecs[0].numpy()
    y_f = fine_vecs[1].numpy()

    def _cubic2d(xc, yc, vals, xf, yf):
        pts_c = np.column_stack([xc, yc])
        pts_f = np.column_stack([xf, yf])
        res = CloughTocher2DInterpolator(pts_c, vals)(pts_f)
        nan_mask = np.isnan(res)
        if nan_mask.any():
            res[nan_mask] = NearestNDInterpolator(pts_c, vals)(pts_f[nan_mask])
        return res

    is_full_sphere = z_c.min() < -0.01
    if not is_full_sphere:
        return _cubic2d(x_c, y_c, values, x_f, y_f)

    # Full sphere (C1): interpolate upper and lower hemispheres separately
    result = np.empty(n_f)
    y_c = coarse_vecs[1].numpy()
    for c_mask, f_mask in [(z_c >= 0, z_f >= 0), (z_c <= 0, z_f < 0)]:
        if c_mask.sum() < 3 or f_mask.sum() == 0:
            continue
        result[f_mask] = _cubic2d(
            x_c[c_mask], y_c[c_mask], values[c_mask],
            x_f[f_mask], y_f[f_mask],
        )
    return result


# The (u, v, occurrence) key of every slot from the most recent _transition_slots
# call, kept in sync by _filter_slots and read by the separate='transitions' path.
# It is per thread: it used to be a function attribute shared by the whole process,
# so concurrent pepper calls (esfit workers='threads') could read each other's
# keys, which raised IndexError or mislabeled transitions.
_slot_state = threading.local()


def _slot_layout(all_B, all_pairs, n_orient, unpaired='skip'):
    """Slot bookkeeping of a per-orientation line list, vectorized.

    A slot is a level pair ``(u, v)`` plus its occurrence index (looping
    transitions can resonate more than once): at every orientation the
    resonances are ordered by ``(u, v, B)`` (stable) and numbered per pair.
    Orientations without level pairs (perturbation path) are index-matched:
    resonance ``i`` goes to slot ``('idx', i)``.  Slots are numbered in order of
    first appearance, orientation by orientation and, within one, in that order.

    ``unpaired='skip'`` drops orientations without pairs when any orientation has
    them (``pepper``); ``'index'`` index-matches them (``pepper_autograd``).

    Returns ``(k, pos, slot, keys, B_flat)`` for the kept resonances: orientation,
    position in ``B_flat`` (the concatenation of every non-empty ``all_B[k]`` in
    orientation order, as a NumPy array), slot number, and the slot keys
    (``(u, v, occurrence)`` or ``('idx', i)``) in slot order.  One host copy per
    call instead of one per orientation.
    """
    lens = [int(b.numel()) if b is not None else 0 for b in all_B]
    ks = [k for k in range(n_orient) if lens[k] > 0]
    empty = np.zeros(0, dtype=np.int64)
    if not ks:
        return empty, empty, empty, [], np.zeros(0)
    B_flat = torch.cat([all_B[k].detach().reshape(-1) for k in ks]).cpu().numpy()
    no_pairs = all_pairs is None or all(p is None for p in all_pairs)
    paired = [(not no_pairs) and all_pairs[k] is not None for k in ks]
    keep_o = [pk or no_pairs or unpaired == 'index' for pk in paired]
    lens_k = np.asarray([lens[k] for k in ks], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lens_k)[:-1]])
    k_all = np.repeat(np.asarray(ks, dtype=np.int64), lens_k)
    i_all = np.arange(lens_k.sum(), dtype=np.int64) - np.repeat(offsets, lens_k)
    if no_pairs:
        # index-matched (perturbation path): resonance i of every orientation is slot i
        return k_all, np.arange(i_all.size, dtype=np.int64), i_all, [('idx', i) for i in range(int(lens_k.max()))], B_flat
    typ = np.repeat(np.asarray([0 if pk else 1 for pk in paired], dtype=np.int64), lens_k)
    u = i_all.copy(); v = np.zeros_like(i_all)
    if any(paired):
        P = torch.cat([all_pairs[k].detach().reshape(-1, 2) for k, pk in zip(ks, paired) if pk]).cpu().numpy()
        sel = typ == 0
        u[sel] = P[:, 0]; v[sel] = P[:, 1]
    Bkey = np.where(typ == 0, B_flat, 0.0)
    keep = np.repeat(np.asarray(keep_o), lens_k)
    pos = np.nonzero(keep)[0]
    # per orientation: (u, v, B) for paired, index order otherwise; stable
    order = pos[np.lexsort((i_all[pos], Bkey[pos], v[pos], u[pos], k_all[pos]))]
    ko, to, uo, vo = k_all[order], typ[order], u[order], v[order]
    n = order.size
    new_grp = np.ones(n, dtype=bool)
    new_grp[1:] = (ko[1:] != ko[:-1]) | (uo[1:] != uo[:-1]) | (vo[1:] != vo[:-1]) | (to[1:] != to[:-1])
    grp_start = np.maximum.accumulate(np.where(new_grp, np.arange(n), 0))
    occ = np.where(to == 0, np.arange(n) - grp_start, 0)
    # one int64 per (type, u, v, occurrence): 1-D unique is much faster than axis=0
    nu, nv, nc = int(uo.max()) + 1, int(vo.max()) + 1, int(occ.max()) + 1
    code = ((to * nu + uo) * nv + vo) * nc + occ
    uc, first, inv = np.unique(code, return_index=True, return_inverse=True)
    rank = np.empty(uc.shape[0], dtype=np.int64)
    rank[np.argsort(first, kind='stable')] = np.arange(uc.shape[0])
    slot = rank[inv.reshape(-1)]
    keys = [None] * uc.shape[0]
    for r, f in zip(rank.tolist(), first.tolist()):
        keys[r] = (int(uo[f]), int(vo[f]), int(occ[f])) if to[f] == 0 else ('idx', int(uo[f]))
    return ko, order, slot, keys, B_flat


def _transition_slots(all_B, all_I, all_W, all_pairs, n_orient):
    """Group per-orientation resonances into transition slots across the grid.

    Mirrors EasySpin's ``Pdat``/``Idat``/``Wdat`` (nTransitions × nOrientations)
    bookkeeping: a slot is a level pair ``(u, v)`` plus its occurrence index
    (looping transitions can resonate more than once), and every slot holds one
    value per orientation with NaN where that transition has no resonance.
    Interpolation and projection then act on physically the same transition at
    every grid point. Without level pairs (perturbation path, whose slot
    structure is fixed by construction) the arrays are matched by index.
    See :func:`_slot_layout`.

    Returns (B_arrs, I_arrs, W_arrs): lists of (n_orient,) float arrays; W_arrs
    is None when no widths were supplied.
    """
    have_widths = any(w is not None and w.numel() > 0 for w in all_W) if all_W is not None else False
    no_pairs = all_pairs is None or all(p is None for p in all_pairs)
    k, pos, slot, keys, B_flat = _slot_layout(all_B, all_pairs, n_orient, unpaired='skip')
    n_slots = len(keys)
    B_mat = np.full((n_slots, n_orient), np.nan)
    I_mat = np.full((n_slots, n_orient), np.nan)
    W_mat = np.full((n_slots, n_orient), np.nan) if have_widths else None
    if n_slots:
        ks = [kk for kk in range(n_orient) if all_B[kk] is not None and all_B[kk].numel() > 0]
        B_mat[slot, k] = B_flat[pos]
        I_flat = torch.cat([all_I[kk].detach().reshape(-1) for kk in ks]).cpu().numpy()
        I_mat[slot, k] = I_flat[pos]
        if have_widths:
            parts = []
            for kk in ks:
                m = int(all_B[kk].numel())
                w = all_W[kk]
                wn = np.full(m, np.nan)
                if w is not None:
                    w = w.detach().reshape(-1).cpu().numpy()[:m]
                    wn[:w.shape[0]] = w
                parts.append(wn)
            W_mat[slot, k] = np.concatenate(parts)[pos]
    _slot_state.keys = None if no_pairs else keys   # (u, v, occurrence) per slot; None: index-matched
    B_arrs = [row.copy() for row in B_mat]
    I_arrs = [row.copy() for row in I_mat]
    W_arrs = [row.copy() for row in W_mat] if have_widths else None
    return B_arrs, I_arrs, W_arrs


# Missing knots (B88, B75): the matrix resonance search covers Exp.Range only, so
# a transition that leaves the sweep window at some orientations has NaN knots
# there.  Those level pairs are searched again over a window padded by these
# fractions of the sweep width (lower edge clamped at 0), widening while partly
# missing pairs remain.
_RESEARCH_PADS = (0.5, 1.0, 2.0, 4.0)


def _pair_counts(all_B, all_pairs):
    """``{(u, v): counts}``: number of resonances of each level pair at every
    orientation (int array).  Empty without level pairs (perturbation path).
    Pairs appear in order of first occurrence (orientation, then ``(u, v)``)."""
    counts: dict = {}
    if all_pairs is None or all(p is None for p in all_pairs):
        return counts
    n_orient = len(all_B)
    ks = [k for k in range(n_orient)
          if all_B[k] is not None and all_pairs[k] is not None and all_B[k].numel() > 0]
    if not ks:
        return counts
    P = [all_pairs[k].detach().reshape(-1, 2) for k in ks]
    lens = np.asarray([int(p.shape[0]) for p in P], dtype=np.int64)
    P = torch.cat(P).cpu().numpy().astype(np.int64)
    K = np.repeat(np.asarray(ks, dtype=np.int64), lens)
    nv = int(P[:, 1].max()) + 1
    code = P[:, 0] * nv + P[:, 1]
    uc, inv = np.unique(code, return_inverse=True)
    inv = inv.reshape(-1)
    C = np.bincount(inv * n_orient + K, minlength=uc.shape[0] * n_orient).reshape(uc.shape[0], n_orient)
    first = np.full(uc.shape[0], n_orient, dtype=np.int64)
    np.minimum.at(first, inv, K)
    for i in np.lexsort((uc, first)):
        counts[(int(uc[i] // nv), int(uc[i] % nv))] = C[i].astype(np.int64)
    return counts


def _missing_resonances(all_B, all_pairs):
    """``{(u, v): n}`` for the level pairs whose resonance count differs between
    orientations; ``n`` is the number of resonances missing relative to the
    orientation with the most (summed over orientations).

    A slot is a level pair plus its occurrence index, so such a pair leaves NaN
    knots in at least one of its slots (``_transition_slots``).  Index-matched
    results without level pairs (perturbation path) give an empty dict.
    """
    return {uv: int((c.max() - c).sum()) for uv, c in sorted(_pair_counts(all_B, all_pairs).items())
            if c.min() < c.max()}


def _replace_pairs(lists, new, bad):
    """Per orientation, replace the resonances of the level pairs ``bad`` in
    ``lists = (B, I, W, pairs)`` by those of ``new`` (same layout; its other
    pairs are ignored).  Slot assembly sorts by (pair, field) itself, so the
    order does not matter.  Tensors stay on the autograd graph (masking and
    concatenation only)."""
    def _in_bad(P):
        badt = torch.as_tensor(bad, dtype=P.dtype, device=P.device)
        return (P[:, None, :] == badt[None]).all(-1).any(-1)

    out = ([], [], [], [])
    for k in range(len(lists[0])):
        B1, I1, W1, P1 = (l[k] for l in lists)
        B2, I2, W2, P2 = (l[k] for l in new)
        keep = take = None
        if P1 is not None and B1 is not None and B1.numel():
            keep = ~_in_bad(P1)
        if P2 is not None and B2 is not None and B2.numel():
            take = _in_bad(P2)

        def _cat(x1, x2):
            parts = []
            if x1 is not None and x1.numel() and keep is not None:
                parts.append(x1[keep])
            if x2 is not None and x2.numel() and take is not None:
                parts.append(x2[take])
            if parts:
                return torch.cat(parts) if len(parts) > 1 else parts[0]
            for x in (x1, x2):
                if x is not None:
                    return x[:0]
            return None
        Bk = _cat(B1, B2)
        out[0].append(Bk)
        out[1].append(_cat(I1, I2))
        out[2].append(None if (W1 is None and W2 is None) else _cat(W1, W2))
        out[3].append(None if (Bk is None or Bk.numel() == 0) else _cat(P1, P2))
    return out


def _merge_line_lists(a, b, width):
    """Per orientation, the union of two ``(B, I, W, pairs)`` line lists; a
    resonance of the same level pair within ``1e-6 * width`` of one already
    present (found again at the joint of two adjacent search windows) is dropped.
    Tensors stay on the autograd graph (concatenation and indexing only)."""
    out = ([], [], [], [])
    for k in range(len(a[0])):
        parts = [(x[0][k], x[1][k], x[2][k], x[3][k]) for x in (a, b)
                 if x[0][k] is not None and x[0][k].numel() and x[3][k] is not None]
        if not parts:
            for i in range(4):
                out[i].append(a[i][k])
            continue
        if len(parts) == 1:
            for i in range(4):
                out[i].append(parts[0][i])
            continue
        B = torch.cat([p[0] for p in parts]); I = torch.cat([p[1] for p in parts])
        P = torch.cat([p[3] for p in parts])
        W = None if any(p[2] is None for p in parts) else torch.cat([p[2] for p in parts])
        Pd = P.detach()
        code = Pd[:, 0] * (int(Pd[:, 1].max()) + 1) + Pd[:, 1]
        o = torch.argsort(B.detach(), stable=True)
        o = o[torch.argsort(code[o], stable=True)]                  # by pair, then field
        B, I, P, code = B[o], I[o], P[o], code[o]
        W = None if W is None else W[o]
        dup = torch.zeros(B.shape[0], dtype=torch.bool, device=B.device)
        dup[1:] = (code[1:] == code[:-1]) & ((B.detach()[1:] - B.detach()[:-1]).abs() < 1e-6 * width)
        keep = ~dup
        out[0].append(B[keep]); out[1].append(I[keep])
        out[2].append(None if W is None else W[keep]); out[3].append(P[keep])
    return out


def _research_missing_knots(lists, rng, search, verbose=False):
    """Search again the level pairs that miss resonances at some orientations
    (B88, B75).

    ``lists = (B, I, W, pairs)`` per orientation from the search over
    ``rng = Exp.Range``; ``search(window, pairs)`` repeats that search over
    all orientations for the given level pairs and field window and returns
    the same layout.  The partly missing pairs are searched in the bands that
    extend the range to Range ± f·W (W = sweep width, f in ``_RESEARCH_PADS``,
    lower edge clamped at 0): at each f only the two bands added since the
    previous f, ``[lo - f·W, previous lo]`` and ``[previous hi, hi + f·W]``, are
    searched (the range itself and the earlier bands have been searched), and a
    pair's resonances become those of the main search plus every band so far,
    all orientations at once, so that the occurrence index of a looping
    transition means the same at every knot.  A pair's new resonances are kept
    only if they leave fewer of its resonances missing and none of its
    orientations loses one; pairs that improved but are still incomplete go on
    to the next, wider band, the others keep what they have (what is still
    missing then does not resonate at those orientations at any field nearby,
    e.g. a looping transition, and very wide windows make the detection of its
    near-tangent resonances roundoff-sensitive).  Out-of-range positions are
    kept: projection and summation clip them to the sweep window, and the
    interpolation sees the transition at every knot.  Returns ``lists`` itself
    (same objects) when no pair is partly missing or none improves, so such
    spectra are unchanged bit for bit.

    Up to 0.9.38 each round searched the whole window Range ± f·W again (for
    the S = 5/2 Mn(II) benchmark 44.6k diagonalizations against 23.1k in the
    main search); the band search finds the same resonances (spectra agree to
    ~1e-11) at about half the cost.
    """
    missing = _missing_resonances(lists[0], lists[3])
    if not missing:
        return lists
    lo0, hi0 = float(rng[0]), float(rng[1])
    W = hi0 - lo0
    active = list(missing)
    acc = lists                       # main search plus every band searched so far
    lo_prev, hi_prev = lo0, hi0
    for f in _RESEARCH_PADS:
        lo_n, hi_n = max(0.0, lo0 - f * W), hi0 + f * W
        found = None
        for band in ([lo_n, lo_prev], [hi_prev, hi_n]):
            if band[1] - band[0] <= 0:
                continue
            if verbose:
                print(f"pepper: {len(active)} level pair(s) miss resonances at some knots; "
                      f"searching them again over [{band[0]:.6g}, {band[1]:.6g}] mT")
            r = search(band, active)
            found = r if found is None else _merge_line_lists(found, r, W)
        lo_prev, hi_prev = lo_n, hi_n
        if found is None:
            break
        acc = _merge_line_lists(acc, found, W)
        old = _pair_counts(lists[0], lists[3])
        trial = _pair_counts(acc[0], acc[3])
        # the bands only add resonances: a pair improves if it has no fewer
        # resonances at any orientation and fewer missing
        better = [uv for uv in active if uv in trial and (trial[uv] >= old[uv]).all()
                  and int((trial[uv].max() - trial[uv]).sum()) < missing[uv]]
        if better:
            lists = _replace_pairs(lists, acc, better)
            missing = _missing_resonances(lists[0], lists[3])
        active = [uv for uv in better if uv in missing]
        if not active:
            break
    return lists


def _filter_slots(B_arrs, I_arrs, W_arrs, threshold):
    """EasySpin transition pre-selection: keep a transition (slot) when its
    maximum |intensity| over the grid reaches ``threshold`` × the global maximum.
    Resonances are never dropped per orientation, so slots stay contiguous for
    interpolation."""
    if threshold <= 0 or not I_arrs:
        return B_arrs, I_arrs, W_arrs
    slot_max = np.array([np.nanmax(np.abs(I)) if np.isfinite(I).any() else 0.0 for I in I_arrs])
    gmax = slot_max.max() if slot_max.size else 0.0
    keep = [k for k in range(len(I_arrs)) if slot_max[k] >= threshold * gmax]
    keys = getattr(_slot_state, 'keys', None)
    if keys is not None:
        _slot_state.keys = [keys[k] for k in keep]
    return ([B_arrs[k] for k in keep], [I_arrs[k] for k in keep],
            None if W_arrs is None else [W_arrs[k] for k in keep])


def _gaussian_bins(x: torch.Tensor, pos: np.ndarray, fwhm: np.ndarray, amp: np.ndarray,
                   min_fwhm: float, chunk: int = 256) -> torch.Tensor:
    """Sum of bin-integrated Gaussians (EasySpin lisum1i with a Gaussian template).

    Each line contributes ``amp`` × [Φ(upper bin edge) − Φ(lower bin edge)], so
    the sum over points equals ``amp`` (a spectral density after the final
    1/ΔB). Widths below ``min_fwhm`` (ΔB/100, lisum1i's clip) are clipped.

    Every line is evaluated only inside a window of ±7σ√2 around its center
    (erfc(7) ≈ 4e-23, i.e. the omitted bins are zero to double precision);
    lines are sorted by width and processed in chunks of 256 (cache-sized:
    larger chunks are 3-4× slower per line) with a tight common window; on the
    CPU the chunks run concurrently in a thread pool
    (``_linalg.pool_map``), on CUDA sequentially on the device.
    """
    n = x.shape[0]
    dev = x.device
    if pos.size == 0:
        return torch.zeros(n, dtype=torch.float64, device=dev)
    dx = float(x[1] - x[0])
    x0 = float(x[0])
    sig = np.maximum(np.asarray(fwhm, dtype=float), min_fwhm) / math.sqrt(8.0 * math.log(2.0))
    order = np.argsort(sig, kind='stable')
    pos_t = torch.from_numpy(np.asarray(pos, dtype=float)[order]).to(dev)
    sg_all = torch.from_numpy(sig[order] * math.sqrt(2.0)).to(dev)
    amp_t = torch.from_numpy(np.asarray(amp, dtype=float)[order]).to(dev)
    half_all = 7.0 * sg_all                                      # window half-width (mT)
    nw_np = ((2.0 * half_all / dx).ceil().to(torch.long) + 3).cpu().numpy()   # bins per window
    ar = torch.arange(n, dtype=torch.long, device=dev)
    xr = x.to(dev)

    def _chunk(k0: int) -> torch.Tensor:
        sl = slice(k0, k0 + chunk)
        p = pos_t[sl, None]
        sg = sg_all[sl, None]
        a = amp_t[sl, None]
        nw = int(nw_np[sl].max())
        if nw >= n:
            hi = torch.erf((xr[None, :] + 0.5 * dx - p) / sg)
            lo = torch.erf((xr[None, :] - 0.5 * dx - p) / sg)
            return (0.5 * a * (hi - lo)).sum(dim=0)
        start = ((p - half_all[sl, None] - x0) / dx).floor().to(torch.long) - 1
        idx = start + ar[None, :nw]
        idxc = idx.clamp(0, n - 1)
        z = (xr[idxc] - p) / sg                                   # bin centers
        d = (0.5 * dx) / sg                                       # half bin in σ√2 units
        vals = (0.5 * a) * (torch.erf(z + d) - torch.erf(z - d))
        vals = torch.where((idx >= 0) & (idx < n), vals, torch.zeros((), dtype=torch.float64, device=dev))
        return torch.bincount(idxc.reshape(-1), weights=vals.reshape(-1), minlength=n)

    starts = list(range(0, pos_t.shape[0], chunk))
    if dev.type == 'cpu':
        from torchspin._linalg import pool_map
        parts = pool_map(_chunk, starts)
    else:
        parts = [_chunk(k0) for k0 in starts]
    return torch.stack(parts).sum(dim=0) if len(parts) > 1 else parts[0]


def _interp_grid(values: np.ndarray, symmetry: str, N_c: int, N_f: int, linear: bool = False) -> np.ndarray:
    """EasySpin ``gridinterp`` for open-φ grids (Ci, C2h, C1, C4h, C6h, S6).

    The coarse per-knot values are rectified into a (θ-row × φ-column) array
    (each θ slice resampled along φ with a cubic spline, periodic in φ where
    the grid wraps and with zero end slopes where it does not), then
    interpolated with a global bicubic spline at the fine grid's (θ, φ) —
    EasySpin mode ``G3``. When values are missing (NaN,
    e.g. a transition without a resonance at some knots) EasySpin switches to
    linear interpolation (``L1``) so the NaNs propagate to the affected fine
    knots only; the same is done here.
    """
    from scipy.interpolate import CubicSpline, RectBivariateSpline
    maxPhi, closedPhi, nOct = gridparam(symmetry)
    periodic = not closedPhi
    full = nOct == 8
    y = np.asarray(values, dtype=float)
    cubic = (not linear) and not np.isnan(y).any()
    dlen = 4 if full else nOct
    nr = 2 * N_c - 1 if full else N_c
    nc = dlen * (N_c - 1) + 1
    z = np.full((nr, nc), np.nan)

    def _row(vals, is_periodic):
        if is_periodic:
            vals = np.concatenate([vals, vals[:1]])
        n = vals.size
        xx = np.linspace(1.0, float(n), nc)
        if cubic:
            if is_periodic:
                return CubicSpline(np.arange(1, n + 1), vals, bc_type='periodic')(xx)
            # closed φ: zero end slopes (gridinterp.m:184, esspline1d(...,1,nc))
            return CubicSpline(np.arange(1, n + 1), vals, bc_type='clamped')(xx)
        k = np.minimum(np.maximum(np.floor(xx - 1).astype(int), 0), n - 2)
        return vals[k] + (xx - (k + 1)) * (vals[k + 1] - vals[k])

    z[0, :] = y[0]
    idx = 1
    ln = dlen + 1 - int(periodic)
    for ir in range(1, N_c - 1):
        z[ir, :] = _row(y[idx:idx + ln], periodic)
        idx += ln
        ln += dlen
    z[N_c - 1, :] = np.concatenate([y[idx:idx + ln], y[idx:idx + 1]]) if periodic else y[idx:idx + ln]
    if full:
        # C1 lower hemisphere: sphgrid stores each of these rows as a reversed
        # upper row (sphgrid.m:320-325), i.e. with phi decreasing; put it back
        # in phi-increasing order before resampling it on the common phi
        # columns (B74). EasySpin's gridinterp.m:211-219 reads the rows as
        # increasing: a lower knot at phi gets the value from -phi - dphi.
        idx += ln
        ln -= dlen
        for ir in range(N_c, 2 * N_c - 2):
            z[ir, :] = _row(y[idx:idx + ln][::-1], periodic)
            idx += ln
            ln += -dlen
        z[2 * N_c - 2, :] = y[idx]

    phi_f, theta_f, _, _ = sphgrid(symmetry, N_f)
    phi_f = phi_f.numpy(); theta_f = theta_f.numpy()
    # 1-based as in gridinterp.m:121-125, then shifted (exactly) to 0-based:
    # the ``1 +`` rounds away ulp-level excess at integer knots, which decides
    # whether a NaN neighbor gets a zero weight.
    if full:
        iphi = (1 + 4 * (N_c - 1) * (phi_f / maxPhi)) - 1
        ithe = (1 + 2 * (N_c - 1) * (theta_f / math.pi)) - 1
    else:
        iphi = (1 + nOct * (N_c - 1) * (phi_f / maxPhi)) - 1
        ithe = (1 + (N_c - 1) * (theta_f / (math.pi / 2))) - 1
    iphi = np.clip(iphi, 0, nc - 1); ithe = np.clip(ithe, 0, nr - 1)
    if cubic:
        spl = RectBivariateSpline(np.arange(nr), np.arange(nc), z, kx=3, ky=3, s=0)
        return spl.ev(ithe, iphi)
    # bilinear with NaN propagation (EasySpin interp2 'linear')
    r0 = np.minimum(np.floor(ithe).astype(int), nr - 2); c0 = np.minimum(np.floor(iphi).astype(int), nc - 2)
    fr = ithe - r0; fc = iphi - c0
    # MATLAB interp2 'linear' (gridinterp.m:129) ignores corners with zero
    # weight, so a NaN neighbor does not propagate to fine knots lying on a
    # coarse row or column (0*NaN would be NaN here).
    def _term(w, zz):
        return np.where(w == 0.0, 0.0, w * zz)
    return (_term((1 - fr) * (1 - fc), z[r0, c0]) + _term((1 - fr) * fc, z[r0, c0 + 1])
            + _term(fr * (1 - fc), z[r0 + 1, c0]) + _term(fr * fc, z[r0 + 1, c0 + 1]))


def _interp_l3(y: np.ndarray, N_f: int) -> np.ndarray:
    """EasySpin gridinterp 'L3': local cubic Hermite interpolation in index space
    with Fritsch-Carlson monotone tangents (1-D, axial grids). NaNs propagate."""
    y = np.asarray(y, dtype=float)
    n = y.size
    factor = int(round((N_f - 1) / (n - 1)))
    x = np.arange(factor) / factor
    X = np.stack([x ** 3, x ** 2, x, np.ones_like(x)], axis=1)              # (factor, 4)
    H = np.array([[2, -2, 1, 1], [-3, 3, -2, -1], [0, 0, 1, 0], [1, 0, 0, 0]], dtype=float)
    d = np.diff(y)
    T = np.zeros(n)
    with np.errstate(invalid='ignore'):
        k = np.where(np.sign(d[:n - 2]) * np.sign(d[1:n - 1]) > 0)[0]
        dmax = np.maximum(np.abs(d[k]), np.abs(d[k + 1]))
        dmin = np.minimum(np.abs(d[k]), np.abs(d[k + 1]))
        T[k + 1] = 2 * dmin * dmax / (d[k] + d[k + 1])
    C = np.stack([y[:-1], y[1:], T[:-1], T[1:]], axis=0)                     # (4, n-1)
    yii = X @ H @ C                                                          # (factor, n-1)
    return np.concatenate([yii.reshape(-1, order='F'), y[-1:]])


def _interp_slot(vals: np.ndarray, symmetry: str, N_c: int, N_f: int,
                 vecs_c: torch.Tensor, vecs_f: torch.Tensor, mode: str = 'pos'):
    """Interpolate one transition slot from the coarse to the fine grid.

    Closed-φ one-octant grids and Dinfh use :func:`_interp_sph` (structured
    G3 / meridional spline); open-φ grids use :func:`_interp_grid`. Returns
    ``None`` when fewer than three knots carry a value.

    EasySpin pepper.m interpolation modes: positions ``G3`` (global cubic),
    intensities and widths ``L3`` (local cubic, axial grids) or ``L1``
    (linear, all other grids).  Pass ``mode='val'`` for intensities/widths.

    Positions of a slot that still has knots without a resonance (after the
    re-search of :func:`_research_missing_knots`, e.g. a looping transition)
    are interpolated with ``L3`` (axial) / ``L1``, so the NaNs propagate and
    the facets touching those knots are dropped instead of being filled from
    neighboring knots.  This is decided per slot (B88): EasySpin switches the
    positions of every transition to linear as soon as any one has a missing
    knot (pepper.m:701, 763-772), which reproduces the coarse-grid facets for
    all the others.
    """
    vals = np.asarray(vals, dtype=float)
    valid = ~np.isnan(vals)
    if valid.sum() < 3:
        return None
    if mode == 'val':
        if symmetry == 'Dinfh':
            return _interp_l3(vals, N_f)
        return _interp_grid(vals, symmetry, N_c, N_f, linear=True)
    if not valid.all():
        if symmetry == 'Dinfh':
            return _interp_l3(vals, N_f)          # EasySpin 'L3' for axial grids with NaNs
        return _interp_grid(vals, symmetry, N_c, N_f, linear=True)
    if symmetry in _CLOSEDPHI_1OCT or symmetry == 'Dinfh':
        return _interp_sph(vecs_c, vecs_f, vals)
    return _interp_grid(vals, symmetry, N_c, N_f)


def _facets(symmetry: str, N: int, theta_sym: torch.Tensor, ordering=None):
    """(tri_idx or None, facet weights summing to 4π) for projection/summation.

    ``ordering=(f, R_L2S)`` weights every facet by the orientational
    distribution evaluated at the facet center (EasySpin pepper.m
    ``orderingWeights`` for partially ordered samples).
    """
    if symmetry == 'Dinfh':
        if ordering is not None:
            raise ValueError('Cannot use axial grid for partially ordered samples.')
        th = theta_sym.cpu().numpy()
        return None, (np.cos(th[:-1]) - np.cos(th[1:])) * 4.0 * math.pi
    tri, areas = grid_triangulation(symmetry, N)
    if ordering is not None:
        from torchspin.ordering import orifun_M2L
        fun, R_L2S = ordering
        phi_g, theta_g, _, _ = sphgrid(symmetry, N)
        c_phi = phi_g.detach().cpu().numpy()[tri].mean(axis=1)
        c_theta = theta_g.detach().cpu().numpy()[tri].mean(axis=1)
        ow = orifun_M2L(fun, R_L2S, c_phi, c_theta)
        if np.any(ow < 0):
            raise ValueError('User-supplied orientation distribution gives negative values.')
        if np.all(ow == 0):
            raise ValueError('User-supplied orientation distribution is all-zero.')
        areas = areas * ow / ow.sum()
    return tri, areas * (4.0 * math.pi / areas.sum())


def _central_difference(spec: torch.Tensor, dx: float) -> torch.Tensor:
    """EasySpin DerivHarmonic: mean of forward and backward differences."""
    spec_der = torch.zeros_like(spec)
    spec_der[1:-1] = (spec[2:] - spec[:-2]) / (2.0 * dx)
    spec_der[0] = (spec[1] - spec[0]) / dx
    spec_der[-1] = (spec[-1] - spec[-2]) / dx
    return spec_der


def _lorentzian_bins(x: torch.Tensor, pos: np.ndarray, fwhm: np.ndarray, amp: np.ndarray,
                     min_fwhm: float, chunk: int = 256, phase: float = 0.0) -> torch.Tensor:
    """Sum of bin-integrated Lorentzians (EasySpin lisum1i with a Lorentzian template).

    ``phase`` mixes in the dispersion line shape (EasySpin ``Exp.mwPhase``; the
    isotropic template ``lorentzian(xT,x0T,wT,-1,Exp.mwPhase)``)."""
    n = x.shape[0]
    out = torch.zeros(n, dtype=torch.float64)
    if pos.size == 0:
        return out
    dx = float(x[1] - x[0])
    hw = np.maximum(np.asarray(fwhm, dtype=float), min_fwhm) / 2.0
    if phase:
        from torchspin.lineshape import lorentzian as _lor
        xn = x.detach().cpu().numpy()
        for k in range(pos.size):
            F_hi, _ = _lor(xn + 0.5 * dx, float(pos[k]), 2.0 * float(hw[k]), -1, float(phase))
            F_lo, _ = _lor(xn - 0.5 * dx, float(pos[k]), 2.0 * float(hw[k]), -1, float(phase))
            out += float(amp[k]) * torch.from_numpy(np.asarray(F_hi - F_lo, dtype=float))
        return out
    pos_t = torch.from_numpy(np.asarray(pos, dtype=float)); hw_t = torch.from_numpy(hw)
    amp_t = torch.from_numpy(np.asarray(amp, dtype=float))
    for k0 in range(0, pos_t.shape[0], chunk):
        p = pos_t[k0:k0 + chunk, None]; g = hw_t[k0:k0 + chunk, None]; a = amp_t[k0:k0 + chunk, None]
        hi = torch.atan((x[None, :] + 0.5 * dx - p) / g)
        lo = torch.atan((x[None, :] - 0.5 * dx - p) / g)
        out += (a * (hi - lo) / math.pi).sum(dim=0)
    return out


# Relative tolerance of the range-padding decision (see _edge_pad_count).
_EDGE_PAD_RTOL = 1e-9


def _edge_pad_count(spec: torch.Tensor, dx: float, fwhm_g, fwhm_l,
                    multiplier: float = 3.0) -> int:
    """EasySpin's range padding before the final convolution (pepper.m:1171-1193,
    salt.m:858-882): when the spectrum is nonzero at the first or last point of
    the range, both ends are extended by
    ``nPad = round(max(fwhmG, fwhmL)/deltaX * Opt.PaddingMultiplier)`` copies of
    the edge values, so the convolution does not see a step to zero there.

    Returns ``nPad`` (0: no padding).  As EasySpin's ``any(spec(:,1)~=0)``, one
    decision covers all rows of a multi-row ``spec``.  An end counts as nonzero
    when ``|edge| > 1e-9 * max|row|`` for some row, not at exact ``~= 0``:
    torchspin's projections leave a ~1e-14 relative round-off floor in empty
    bins that EasySpin's leave exactly zero, and an exact test would pad
    spectra EasySpin does not pad (KNOWN_LIMITATIONS.md).  ``round`` is
    MATLAB's (half away from zero).  The decision is discrete and reads
    detached values only.
    """
    mult = float(multiplier)
    if mult <= 0 or spec.shape[-1] < 2:
        return 0
    g_val = float(fwhm_g.detach()) if torch.is_tensor(fwhm_g) else float(fwhm_g)
    l_val = float(fwhm_l.detach()) if torch.is_tensor(fwhm_l) else float(fwhm_l)
    rd = spec.detach().reshape(-1, spec.shape[-1]).abs()
    tol = _EDGE_PAD_RTOL * rd.max(dim=-1).values
    if not bool(((rd[:, 0] > tol) | (rd[:, -1] > tol)).any()):
        return 0
    return int(math.floor(max(g_val, l_val) / dx * mult + 0.5))


def _lw_convolve(row: torch.Tensor, dx: float, fwhm_g, fwhm_l, harmonic: int,
                 phase: float = 0.0, conv=None, pad_multiplier: float = 3.0,
                 n_pad: Optional[int] = None) -> torch.Tensor:
    """EasySpin pepper.m's final ``Sys.lw`` convolution (pepper.m:1150-1223).

    * The harmonic and ``Exp.mwPhase`` go to the Lorentzian when
      ``Sys.lw(2) > 0``, otherwise to the Gaussian (pepper.m:1152-1163); a
      Voigt width is applied Lorentzian first, then Gaussian.
    * A component whose FWHM is at most two increments and whose harmonic is
      0 is not convolved: it is a delta-like line (pepper.m:1195-1223).
      EasySpin raises for such a component with a harmonic > 0; torchspin
      convolves it anyway (KNOWN_LIMITATIONS.md).
    * Range padding (pepper.m:1171-1193): a spectrum that is nonzero at an end
      of the range is extended by ``n_pad`` edge copies on both sides before
      the convolutions and cropped afterwards (:func:`_edge_pad_count`, from
      ``pad_multiplier`` = ``Options.PaddingMultiplier``; 0 disables it).
      ``n_pad`` overrides the count, for callers that decide once for several
      rows.  The copies are torch slices, so gradients flow through them.

    ``conv(spec, dx, fwhm_g, fwhm_l, deriv, phase)`` is the convolution
    (default :func:`convspec`; :mod:`torchspin.pepper_autograd` passes its
    torch one).  Widths may be tensors; only the skip decision reads their
    values (a discrete selection), the convolution keeps them in the graph.
    """
    if conv is None:
        def conv(s, d, g, l, deriv, ph):
            return convspec(s, d, fwhm_g=g, fwhm_l=l, deriv=deriv, phase=ph)
    g_val = float(fwhm_g.detach()) if torch.is_tensor(fwhm_g) else float(fwhm_g)
    l_val = float(fwhm_l.detach()) if torch.is_tensor(fwhm_l) else float(fwhm_l)
    if l_val > 0:
        harm_l, phase_l, harm_g, phase_g = harmonic, phase, 0, 0.0
    else:
        harm_l, phase_l, harm_g, phase_g = 0, 0.0, harmonic, phase
    use_l = l_val > 0 and not (l_val <= 2 * dx and harm_l == 0)
    use_g = g_val > 0 and not (g_val <= 2 * dx and harm_g == 0)
    # A skipped width that requires grad stays in the graph with zero gradient
    # (the spectrum does not depend on it), so autograd.grad does not fail.
    for w_, used in ((fwhm_g, use_g), (fwhm_l, use_l)):
        if not used and torch.is_tensor(w_) and w_.requires_grad:
            row = row + 0.0 * w_.to(row.dtype)
    if not (use_l or use_g):
        return row          # padding followed by cropping alone is the identity
    n = row.shape[-1]
    if n_pad is None:
        n_pad = _edge_pad_count(row, dx, fwhm_g, fwhm_l, pad_multiplier)
    if n_pad > 0:
        lead = row.shape[:-1]
        row = torch.cat([row[..., :1].expand(*lead, n_pad), row,
                         row[..., -1:].expand(*lead, n_pad)], dim=-1)
    if use_l:
        row = conv(row, dx, 0.0, fwhm_l, harm_l, phase_l)
    if use_g:
        row = conv(row, dx, fwhm_g, 0.0, harm_g, phase_g)
    if n_pad > 0:
        row = row[..., n_pad:n_pad + n]
    return row


def _finish_field_sweep(spec: torch.Tensor, x: torch.Tensor, dx: float, exp,
                        fwhm_g: float, fwhm_l: float, fd_deriv: bool,
                        phase: float = 0.0, pad_multiplier: float = 3.0) -> torch.Tensor:
    """EasySpin's final broadening/harmonic bookkeeping for field sweeps.

    With field modulation (``Exp.ModAmp``) the absorption spectrum is convolved
    (ConvHarmonic 0) and the harmonic comes from the pseudo-modulation
    (``fieldmod``); otherwise the harmonic is produced by the convolution
    (ConvHarmonic) or, without residual line width, by differentiation
    (DerivHarmonic: finite differences for template/strain spectra, spectral
    derivative for stick spectra).  ``phase`` (``Exp.mwPhase``) enters the
    convolution kernel.  Multi-row spectra are processed row-wise, with one
    range-padding decision for all rows (``pad_multiplier`` =
    ``Options.PaddingMultiplier``), as EasySpin pads the whole ``spec`` matrix.
    """
    mod_amp = float(getattr(exp, 'ModAmp', 0.0) or 0.0)
    if mod_amp > 0 and exp.Harmonic < 1:
        raise ValueError("With field modulation (Exp.ModAmp), Exp.Harmonic=0 does not work.")
    conv_harm = 0 if mod_amp > 0 else int(exp.Harmonic)
    rows = [spec] if spec.ndim == 1 else list(spec)
    n_pad = _edge_pad_count(spec, dx, fwhm_g, fwhm_l, pad_multiplier) if (fwhm_g > 0 or fwhm_l > 0) else 0
    out = []
    for row in rows:
        if fwhm_g > 0 or fwhm_l > 0:
            row = _lw_convolve(row, dx, fwhm_g, fwhm_l, conv_harm, phase, n_pad=n_pad)
        elif conv_harm > 0:
            if fd_deriv:
                for _h in range(conv_harm):
                    row = _central_difference(row, dx)
            else:
                row = convspec(row, dx, fwhm_g=0.0, fwhm_l=0.0, deriv=conv_harm)
        if mod_amp > 0:
            from torchspin.dataproc import fieldmod as _fieldmod
            row = torch.tensor(_fieldmod(x.detach().cpu().numpy(), row.detach().cpu().numpy(),
                                         mod_amp, int(exp.Harmonic)), dtype=torch.float64)
        out.append(row)
    return out[0] if spec.ndim == 1 else torch.stack(out)


def _preselect_pairs(H0, mux, muy, muz, sys, exp, opt, B_center, freq_sweep=False, zeeman_ho=None,
                     zero_field=False):
    """EasySpin resfields transition pre-selection: transition rates of all level
    pairs at the center field on a small D2h grid (TPSGridSize 4); keep the pairs
    whose maximum rate reaches Opt.Threshold × the largest; pure nuclear
    transitions are dropped when the hyperfine interaction is weak
    (HFIStrength = max|A|·(I+1/2)/ν < 0.5).  Returns a list of (u, v) pairs.
    With ``zeeman_ho`` (Options.QuadraticZeeman) the eigenvectors are those of the
    full H(B_center) including B² n·G2·n + B³ G3[n,n,n].

    ``zero_field`` (B_center = 0, Options.ZeroField): the rates are summed over
    degenerate level clusters before the maximum and the threshold, and the
    pairs returned are the cluster labels of :func:`resfreqs_batch`
    (first level of each cluster)."""
    from torchspin.excitation import excitation_geometry, exp_mw_mode
    thr = float(opt.Threshold)
    n = H0.shape[0]
    if thr <= 0:
        return None
    if zero_field:
        return _preselect_pairs_zero_field(H0, mux, muy, muz, sys, exp, thr)
    phi_t, theta_t, _, _ = sphgrid('D2h', 4)
    geom = excitation_geometry(exp_mw_mode(exp))
    Ex, Ey, Ez = -mux, -muy, -muz
    dev = H0.device
    max_rate = torch.zeros(n, n, dtype=torch.float64, device=dev)
    for ph, th in zip(phi_t.tolist(), theta_t.tolist()):
        st, ct, sp, cp = math.sin(th), math.cos(th), math.sin(ph), math.cos(ph)
        muzL = st * (cp * mux + sp * muy) + ct * muz
        H_c = H0 - float(B_center) * muzL
        if zeeman_ho is not None:
            from torchspin.ham_ezho import project_field_nonlinear_zeeman
            Q2, Q3 = project_field_nonlinear_zeeman(
                zeeman_ho, torch.tensor([[st * cp, st * sp, ct]], dtype=torch.float64))
            if Q2 is not None:
                H_c = H_c + float(B_center) ** 2 * Q2[0].to(device=dev, dtype=H_c.dtype)
            if Q3 is not None:
                H_c = H_c + float(B_center) ** 3 * Q3[0].to(device=dev, dtype=H_c.dtype)
        _, V = torch.linalg.eigh(H_c)
        Exy = cp * Ex + sp * Ey
        if geom.parallel:
            EzL = st * Exy + ct * Ez
            rate = (V.conj().T @ EzL @ V).abs() ** 2
        else:
            EyL = -sp * Ex + cp * Ey
            ExL = ct * Exy - st * Ez
            rate = ((V.conj().T @ ExL @ V).abs() ** 2 + (V.conj().T @ EyL @ V).abs() ** 2) / 2
        max_rate = torch.maximum(max_rate, rate.real.to(torch.float64))
    keep = torch.triu(torch.ones(n, n, dtype=torch.bool, device=dev), diagonal=1)
    if sys.nNuclei > 0:
        nu = float(exp.mwFreq) * 1e3 if not freq_sweep else 1e9   # MHz (frequency sweeps: no exclusion)
        A = sys.A.detach().cpu().numpy()
        Amax = np.abs(A).reshape(sys.nNuclei, -1).max(axis=1)
        hfi = Amax * (0.5 + np.asarray(sys.I, dtype=float)) / nu
        if hfi.max() < 0.5:
            n_el = int(np.prod([2 * float(S_) + 1 for S_ in sys.S]))
            n_nuc = n // n_el
            el_idx = torch.arange(n, device=dev) // n_nuc
            keep &= el_idx.unsqueeze(1) != el_idx.unsqueeze(0)   # drop pure nuclear transitions
    rates = max_rate[keep]
    cutoff = thr * float(rates.detach().max()) if rates.numel() else 0.0
    sel = keep & (max_rate > cutoff)
    ii, jj = torch.nonzero(sel, as_tuple=True)
    return [(int(a), int(b)) for a, b in zip(ii.tolist(), jj.tolist())]


def _user_transitions(transitions, n_states):
    """Resolve ``Options.Transitions`` (EasySpin ``Opt.Transitions``).

    ``None`` → ``(None, False)`` (automatic pre-selection).  ``'all'`` → every
    level pair ``u < v`` and ``True`` (no intensity post-selection, as EasySpin
    sets its post-selection threshold to zero).  A list of 1-based level pairs
    ``(u, v)`` (levels ordered by energy at the resonance field) is validated
    (``1 <= u, v <= n_states``, ``u != v``), normalized to ``u < v``, stripped of
    duplicates (first occurrence kept, order otherwise unchanged) and returned
    0-based with ``False`` (Opt.Threshold post-selection still applies)."""
    if transitions is None:
        return None, False
    if isinstance(transitions, str):
        if transitions.strip().lower() != 'all':
            raise ValueError("Options.Transitions must be None, 'all' or a list of 1-based "
                             f"level pairs (u, v); got {transitions!r}.")
        return [(i, j) for i in range(n_states) for j in range(i + 1, n_states)], True
    if isinstance(transitions, torch.Tensor):
        transitions = transitions.detach().cpu().numpy()
    if isinstance(transitions, np.ndarray):
        transitions = transitions.tolist()
    try:
        rows = list(transitions)
    except TypeError:
        raise ValueError("Options.Transitions must be None, 'all' or a list of 1-based "
                         f"level pairs (u, v); got {transitions!r}.") from None
    if len(rows) == 0:
        raise ValueError('Options.Transitions is empty: no transitions selected.')
    out, seen = [], set()
    for row in rows:
        try:
            uv = list(row)
        except TypeError:
            raise ValueError(f'Options.Transitions entries must be level pairs (u, v); got {row!r}.') from None
        if len(uv) != 2:
            raise ValueError(f'Options.Transitions entries must be level pairs (u, v); got {row!r}.')
        idx = []
        for k in uv:
            if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer, float, np.floating)) \
                    or not math.isfinite(float(k)) or float(k) != int(k):
                raise ValueError(f'Options.Transitions level indices must be integers; got {row!r}.')
            idx.append(int(k))
        u, v = idx
        if not (1 <= u <= n_states and 1 <= v <= n_states):
            raise ValueError(f'Options.Transitions level pair {tuple(row)} is out of range: '
                             f'levels are numbered 1..{n_states}.')
        if u == v:
            raise ValueError(f'Options.Transitions level pair {tuple(row)} has u == v.')
        p = (min(u, v) - 1, max(u, v) - 1)
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out, False


def _zf_const_pos(B_f, B_c):
    """Zero field: a transition has the same position at every orientation.
    Give every interpolated vertex the coarse position exactly, so every facet
    of the line has zero width and the line is one point mass, split linearly
    between the two nearest axis points (all of it on x_k when it sits on x_k,
    as in EasySpin). Interpolation round-off scatters the fine-grid positions
    by a few ulp; before 0.9.35 (bin k covering [x_k, x_k + dx)) a line on x_k
    then straddled the bin edge and part of it went into the lower bin (5-39 %
    per line for S = 1, D = [-700 -1300 2000] MHz on a 10 MHz axis, GridSize
    [19 4]; test_pepper_zerofield_edge.py). With the centered hat bins (B85)
    such ulp scatter changes the split only by ulps, so this is now exact
    bookkeeping rather than a necessity."""
    fin = B_c[np.isfinite(B_c)]
    if fin.size == 0:
        return B_f
    return np.where(np.isnan(B_f), np.nan, fin[0])


def _preselect_pairs_zero_field(H0, mux, muy, muz, sys, exp, thr):
    """Zero-field variant of :func:`_preselect_pairs`: H = H0 at every
    orientation, rates summed over pairs of degenerate level clusters
    (basis invariant), maximum over the D2h N=4 grid, threshold on the cluster
    pairs; returns their (first level, first level) labels."""
    from torchspin.excitation import excitation_geometry, exp_mw_mode
    from torchspin._zerofield import level_clusters
    phi_t, theta_t, _, _ = sphgrid('D2h', 4)
    geom = excitation_geometry(exp_mw_mode(exp))
    Ex, Ey, Ez = -mux, -muy, -muz
    dev = H0.device
    E0, V = torch.linalg.eigh(H0)
    labels, first, onehot = level_clusters(E0)
    oh = onehot.to(dev)
    K = first.numel()
    max_rate = torch.zeros(K, K, dtype=torch.float64, device=dev)
    for ph, th in zip(phi_t.tolist(), theta_t.tolist()):
        st, ct, sp, cp = math.sin(th), math.cos(th), math.sin(ph), math.cos(ph)
        Exy = cp * Ex + sp * Ey
        if geom.parallel:
            EzL = st * Exy + ct * Ez
            rate = (V.conj().T @ EzL @ V).abs() ** 2
        else:
            EyL = -sp * Ex + cp * Ey
            ExL = ct * Exy - st * Ez
            rate = ((V.conj().T @ ExL @ V).abs() ** 2 + (V.conj().T @ EyL @ V).abs() ** 2) / 2
        rate_c = oh.T @ rate.real.to(torch.float64).detach() @ oh            # summed over cluster pairs
        max_rate = torch.maximum(max_rate, rate_c)
    keep = torch.triu(torch.ones(K, K, dtype=torch.bool, device=dev), diagonal=1)
    # No exclusion of 'pure nuclear' transitions: at zero field the levels are
    # not ordered by electron manifold (e.g. the F = 0 <-> F = 1 hyperfine line
    # of S = 1/2, I = 1/2 would be dropped by the high-field heuristic).
    rates = max_rate[keep]
    cutoff = thr * float(rates.max()) if rates.numel() else 0.0
    sel = keep & (max_rate > cutoff)
    ii, jj = torch.nonzero(sel, as_tuple=True)
    return [(int(first[a]), int(first[b])) for a, b in zip(ii.tolist(), jj.tolist())]


_GL2 = (0.5 - 0.5 / math.sqrt(3.0), 0.5 + 0.5 / math.sqrt(3.0))   # 2-point Gauss-Legendre on [0, 1]


def _hat_piece_loop(spec: np.ndarray, a: float, b: float, va: float, vb: float) -> None:
    """Reference for one linear density piece (``va`` at ``a``, ``vb`` at ``b``, axis
    coordinates, ``a < b``): ``spec[k] += ∫ L(u) Λ(u − k) du`` by two-point Gauss-Legendre
    on every interval between the breakpoints ``a``, ``b`` and the integers (the integrand
    is quadratic there, so the rule is exact)."""
    n = spec.shape[0]
    cuts = [a] + [float(c) for c in range(int(math.floor(a)) + 1, int(math.ceil(b)))] + [b]
    for p, q in zip(cuts[:-1], cuts[1:]):
        if q <= p:
            continue
        k_lo = int(math.floor(0.5 * (p + q)))       # the interval lies in [k_lo, k_lo + 1]
        for t in _GL2:
            u = p + t * (q - p)
            dens = va + (vb - va) * (u - a) / (b - a)
            wgt = 0.5 * (q - p) * dens
            f = u - k_lo                              # weight of k_lo + 1; 1 − f of k_lo
            if 0 <= k_lo < n:
                spec[k_lo] += wgt * (1.0 - f)
            if 0 <= k_lo + 1 < n:
                spec[k_lo + 1] += wgt * f


def _hat_point_loop(spec: np.ndarray, u: float, mass: float) -> None:
    """Reference point mass: linear split between the two nearest axis points."""
    n = spec.shape[0]
    k = int(math.floor(u))
    t = u - k
    if 0 <= k < n:
        spec[k] += mass * (1.0 - t)
    if 0 <= k + 1 < n:
        spec[k + 1] += mass * t


def _projectzones_loop(
    pos: np.ndarray,
    amp: np.ndarray,
    seg_weights: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Segment-by-segment reference for :func:`_projectzones` (axial-grid projection,
    EasySpin ``projectzones``, with the centered hat bins of :mod:`torchspin._projection`).

    Each segment between consecutive orientations (pos[i], pos[i+1]) is a uniform
    density of integrated intensity ``mean(amp)·seg_weights[i]``; ``spec[k]`` is its
    integral against the hat of half-width ``delta`` centered on ``x[k]``, divided by
    ``delta``.  A zero-width segment is split linearly between the two nearest points.

    Parameters
    ----------
    pos : (N,) resonance field positions in mT
    amp : (N,) or (1,) intensities; per-segment mean of adjacent values is used
    seg_weights : (N-1,) solid-angle weights (should sum to 4π)
    x : (nPoints,) field axis (mT, uniform spacing)

    Returns
    -------
    spec : (nPoints,) spectral density (same units as projecttriangles output)
    """
    nPoints = int(len(x))
    delta = float(x[1] - x[0])
    x0 = float(x[0])
    spec = np.zeros(nPoints)
    anisoAmp = (len(amp) > 1)
    for iSeg in range(len(seg_weights)):
        left, right = float(pos[iSeg]), float(pos[iSeg + 1])
        if np.isnan(left) or np.isnan(right):
            continue
        left, right = min(left, right), max(left, right)
        meanAmp = (float(amp[iSeg]) + float(amp[iSeg + 1])) / 2.0 if anisoAmp else float(amp[0])
        mass = meanAmp * float(seg_weights[iSeg]) / delta
        a, b = (left - x0) / delta, (right - x0) / delta
        if b > a:
            _hat_piece_loop(spec, a, b, mass / (b - a), mass / (b - a))
        else:
            _hat_point_loop(spec, a, mass)
    return spec


def _projecttriangles_loop(
    tri_idx: np.ndarray,
    areas: np.ndarray,
    fun: np.ndarray,
    amp: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Triangle-by-triangle reference for :func:`_projecttriangles` (EasySpin
    ``projecttriangles``, with the centered hat bins of :mod:`torchspin._projection`).

    For each triangle the three resonance-field positions define a tent function of
    integrated intensity ``mean(amp)·area``: linearly rising from the leftmost to the
    middle vertex, then falling to the rightmost vertex.  ``spec[k]`` is its integral
    against the hat of half-width ``delta`` centered on ``x[k]``, divided by ``delta``;
    a zero-width triangle is split linearly between the two points nearest to it.

    Parameters
    ----------
    tri_idx : (nTri, 3) 0-based vertex indices into *fun* / *amp*
    areas : (nTri,) normalized solid-angle areas (must sum to 4π)
    fun : (nPts,) resonance field positions (mT)
    amp : (nPts,) transition intensities
    x : (nPoints,) field axis (mT, uniform spacing)

    Returns
    -------
    spec : (nPoints,) spectral density
    """
    nPoints = int(len(x))
    delta = float(x[1] - x[0])
    x0 = float(x[0])
    spec = np.zeros(nPoints)
    for iTri in range(len(tri_idx)):
        i1, i2, i3 = int(tri_idx[iTri, 0]), int(tri_idx[iTri, 1]), int(tri_idx[iTri, 2])
        p = [float(fun[i1]), float(fun[i2]), float(fun[i3])]
        if any(np.isnan(v) for v in p):
            continue
        mass = (float(amp[i1]) + float(amp[i2]) + float(amp[i3])) / 3.0 * float(areas[iTri]) / delta
        left, middle, right = sorted((v - x0) / delta for v in p)
        width = right - left
        if width == 0.0:
            _hat_point_loop(spec, (sum(p) / 3.0 - x0) / delta, mass)
            continue
        h = 2.0 * mass / width
        if middle > left:
            _hat_piece_loop(spec, left, middle, 0.0, h)
        if right > middle:
            _hat_piece_loop(spec, middle, right, h, 0.0)
    return spec


def _projectzones(
    pos: np.ndarray,
    amp: np.ndarray,
    seg_weights: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Axial-grid projection (EasySpin ``projectzones``) onto the axis ``x`` with centered
    hat bins: numpy front end of :func:`torchspin._projection.projectzones_t` (same
    arithmetic as ``pepper_autograd``); :func:`_projectzones_loop` is the reference."""
    x = np.asarray(x, dtype=float)
    pos_t = torch.from_numpy(np.asarray(pos, dtype=float).reshape(1, -1))
    amp_t = torch.from_numpy(np.asarray(amp, dtype=float).reshape(1, -1))
    sw = torch.from_numpy(np.asarray(seg_weights, dtype=float).reshape(-1))
    return _proj.projectzones_t(pos_t, amp_t, sw, float(x[0]), float(x[1] - x[0]), int(len(x))).numpy()


def _projecttriangles(
    tri_idx: np.ndarray,
    areas: np.ndarray,
    fun: np.ndarray,
    amp: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Triangle projection (EasySpin ``projecttriangles``) onto the axis ``x`` with
    centered hat bins: numpy front end of :func:`torchspin._projection.projecttriangles_t`
    (same arithmetic as ``pepper_autograd``); :func:`_projecttriangles_loop` is the
    reference.  See :mod:`torchspin._projection` for the convention."""
    x = np.asarray(x, dtype=float)
    n = int(len(x))
    tri = np.asarray(tri_idx, dtype=np.int64).reshape(-1, 3)
    if tri.shape[0] == 0:
        return np.zeros(n)
    fun_t = torch.from_numpy(np.asarray(fun, dtype=float).reshape(1, -1))
    amp_t = torch.from_numpy(np.asarray(amp, dtype=float).reshape(1, -1))
    return _proj.projecttriangles_t(torch.from_numpy(tri), torch.from_numpy(np.asarray(areas, dtype=float)),
                                    fun_t, amp_t, float(x[0]), float(x[1] - x[0]), n).numpy()


def _user_orientations(exp):
    """Exp.Orientations / Exp.OrientationWeights as float64 tensors
    ``(phi, theta, w)`` (molecular frame, radians; weights default 1/N), or
    ``None`` when no orientation list is given.  Tensors keep their graph."""
    ori = getattr(exp, 'Orientations', None)
    if ori is None:
        return None
    ori = ori if torch.is_tensor(ori) else torch.as_tensor(np.asarray(ori, dtype=float))
    ori = ori.to(torch.float64).reshape(-1, 2)
    w = getattr(exp, 'OrientationWeights', None)
    if w is None:
        w = torch.full((ori.shape[0],), 1.0 / ori.shape[0], dtype=torch.float64)
    else:
        w = (w if torch.is_tensor(w) else torch.as_tensor(np.asarray(w, dtype=float))).to(torch.float64).reshape(-1)
    return ori[:, 0], ori[:, 1], w


def _check_user_orientations(exp, opt) -> None:
    """Exp.Orientations cannot be combined with sample-orientation or ordering
    fields, nor with the per-transition/per-site outputs."""
    for name in ('Ordering', 'SampleFrame', 'SampleRotation', 'MolFrame', 'CrystalSymmetry'):
        v = getattr(exp, name, None)
        if v is not None and not (isinstance(v, str) and v == ''):
            raise ValueError(f'Exp.Orientations cannot be combined with Exp.{name}: the orientations '
                             f'are field directions in the molecular frame of a disordered sample.')
    if getattr(opt, 'Sites', None) is not None:
        raise ValueError('Exp.Orientations cannot be combined with Opt.Sites (crystal sites).')
    if str(getattr(opt, 'separate', '') or '') in ('transitions', 'sites'):
        raise ValueError(f"Exp.Orientations does not support Opt.separate='{opt.separate}'; "
                         f"use '' or 'orientations'.")


def pepper(sys_or_list, exp: Experiment, opt: Optional[Options] = None):
    """Solid-state CW EPR spectrum (EasySpin ``pepper``).

    Simulates the field-swept or frequency-swept CW EPR spectrum of a powder
    (SOPHE grid with interpolation and projection), a single crystal, a
    partially ordered sample, or a user-given set of orientations, by
    finding resonance fields (or frequencies) and transition rates for each
    orientation and accumulating broadened lines.

    A list of spin systems is simulated component by component and summed
    with weights ``SpinSystem.weight``.  Every component is first expanded
    into isotopologues (``Nucs`` given as elements such as ``'Cu'`` or
    mixtures such as ``'(63,65)Cu'``), each weighted by its abundance and
    dropped below ``Options.IsoCutoff``.

    Parameters
    ----------
    sys_or_list : SpinSystem or list of SpinSystem
        Spin system(s).  pepper builds the full spin Hamiltonian with
        :func:`torchspin.ham`, so every interaction field is used (``g``,
        ``D``, ``B``, ``A``, ``Q``, ``ee``, ``nn``, ``sigma``, ``gnscale``,
        frames, OAM and crystal-field terms, ``Ham``, ``H0Extra``).  Line
        widths: ``lw``/``lwpp`` ``[GaussianFWHM, LorentzianFWHM]`` in mT for
        field sweeps and in MHz for frequency sweeps, convolved after the
        accumulation; ``HStrain``, ``gStrain``, ``AStrain``, ``DStrain``
        (with ``gAStrainCorr``, ``DStrainCorr``) give orientation-dependent
        Gaussian widths per line, ``HStrainL`` Lorentzian ones.  Also read:
        ``weight``, ``Abund``, ``n`` (only ``Method='perturb*'``/``'hybrid'``;
        the matrix method raises for ``n > 1``), ``initState`` (not with
        perturbation theory), ``tdm`` (with ``Experiment.lightBeam``).
        Dynamics, relaxation and pulse fields (``tcorr``, ``Diff``, ``T1``,
        ``T2``, ``lwEndor``, ``Potential``, ...) are ignored.
    exp : Experiment
        Sweep and sample.  Field sweep: ``mwFreq`` (GHz) and ``Range`` or
        ``CenterSweep`` (mT; ``None`` = automatic range for S = 1/2 systems).
        Frequency sweep: ``Field`` (mT) with ``mwRange``/``mwCenterSweep``
        (GHz; ``None`` = automatic range); ``Field`` alone is a frequency
        sweep with automatic range.  Also read: ``nPoints``, ``Harmonic``,
        ``ModAmp`` (field sweeps), ``mwPhase``, ``Temperature``,
        ``Mode``/``mwMode``; single crystals: ``MolFrame`` and/or
        ``CrystalSymmetry`` (either makes the sample a crystal),
        ``SampleFrame``, ``SampleRotation``; ordered samples: ``Ordering``;
        photoselection: ``lightBeam``, ``lightScatter``; explicit
        orientations: ``Orientations``, ``OrientationWeights``.  ``SampleFrame``
        or ``SampleRotation`` alone leave a powder unchanged.  ``ExciteWidth``
        is ignored.
    opt : Options, optional
        Default ``None`` = ``Options()``.  Read: ``Method`` (``'matrix'``,
        ``'exact'``, ``'perturb'``/``'perturb1'``/``'perturb2'``,
        ``'hybrid'`` with ``HybridCoreNuclei``, ``HybridIntThreshold``),
        ``GridSize``, ``GridSymmetry``, ``Threshold``, ``Transitions``,
        ``separate``, ``Sites``, ``IsoCutoff``, ``ModellingAccuracy``,
        ``QuadraticZeeman``, ``ZeroField``, ``PaddingMultiplier``,
        ``BatchSize``, ``device``, ``Verbosity``.  garlic- and ENDOR-only
        options are ignored.

    Returns
    -------
    x : torch.Tensor, shape (nPoints,)
        Sweep axis, ``float64``: magnetic field in mT for a field sweep,
        microwave frequency in GHz for a frequency sweep; equally spaced from
        the first to the last point of the (possibly automatic) range.
    spec : torch.Tensor, shape (nPoints,) or (nSpectra, nPoints)
        Spectrum, ``float64``, on EasySpin's pepper intensity scale
        (arbitrary units, comparable between simulations), at the detection
        harmonic ``Experiment.Harmonic``.  With ``Options.separate`` set,
        one row per component/isotopologue, transition, crystal orientation
        or site, stacked along axis 0 (for a list of components the rows of
        all components are concatenated).  When an input requires grad, the
        result of :func:`~torchspin.pepper_autograd` on the
        autograd graph.

    Raises
    ------
    ValueError
        For an unsupported combination of inputs, among others:
        ``Method='perturb3'`` to ``'perturb5'``; ``n > 1`` with the matrix
        method; ``Range`` missing for a system that is not a single S = 1/2
        (or has a zero g value); ``ModAmp`` with a frequency sweep or with
        ``Harmonic = 0``; ``Method='hybrid'`` with a frequency sweep, ``nn``
        couplings, ``H0Extra`` or a core nucleus with ``n > 1``;
        ``Transitions`` or ``initState`` with perturbation theory; an
        invalid ``Transitions`` list; ``Field = 0`` without
        ``ZeroField=True``, or ``ZeroField`` with ``initState``; ``Ordering``
        with ``MolFrame``/``CrystalSymmetry``, several sample orientations or
        a non-Ci grid; ``Orientations`` with crystal, ordering, ``Sites`` or
        ``separate='transitions'``/``'sites'``; ``separate='transitions'``
        for a crystal; ``lightBeam`` without ``tdm`` or with an unknown
        string; ``HStrainL`` with the Gaussian strains; multi-electron
        ``AStrain`` (or ``gStrain`` in a frequency sweep).
    NotImplementedError
        ``QuadraticZeeman=True`` with field-quadratic ``Ham`` terms and a
        perturbation or hybrid method; a configuration the differentiable
        path does not support when an input requires grad.

    See Also
    --------
    torchspin.pepper_autograd.pepper_autograd : The differentiable path.
    torchspin.garlic.garlic : Isotropic and fast-motion spectra.
    torchspin.chili.chili : Slow-motion spectra.
    torchspin.gridcheck.grid_convergence : Check ``Options.GridSize``.
    SpinSystem, Experiment, Options

    Notes
    -----
    * Methods: ``'matrix'`` is exact.  Second-order perturbation theory is
      inaccurate when a hyperfine coupling is a sizable fraction of the
      microwave frequency: for isotropic 91Zr with A = 780 MHz at 9.65 GHz
      (A/nu = 0.08), ``'perturb'`` differs from ``'matrix'`` by 0.18 relative
      L2.  See KNOWN_LIMITATIONS.md, "Pepper — choosing a method, and what
      each costs (2026-10-07)".
    * If any ``SpinSystem``/``Experiment`` tensor requires grad, pepper runs
      :func:`~torchspin.pepper_autograd` instead and returns
      a spectrum on the autograd graph (KNOWN_LIMITATIONS.md, "Autograd —
      differentiable pepper").
    * With neither line width nor strain the spectrum is a stick/projection
      absorption spectrum and ``Harmonic`` falls back to 0 (EasySpin raises
      when a harmonic is requested explicitly).
    * Line widths (``lw``) are applied by FFT convolution after accumulation,
      with edge padding (``Options.PaddingMultiplier``); strain widths are
      added per line.  See KNOWN_LIMITATIONS.md, "Pepper — final line-width
      convolution (0.8.5)".
    * Powder averaging uses the SOPHE grid (Wang & Hanson, 1995) with
      EasySpin's grid interpolation and triangle/zone projection.
    * Validated against EasySpin (MATLAB) reference spectra
      (``torchspin/tests/test_pepper*_matlab_validation*.py``).

    Examples
    --------
    A nitroxide at X band (first-derivative field sweep):

    >>> from torchspin import SpinSystem, Experiment, Options, pepper
    >>> sys = SpinSystem(S=0.5, g=[2.0088, 2.0061, 2.0027], Nucs='14N',
    ...                  A=[20.0, 20.0, 95.0], lw=0.5)
    >>> exp = Experiment(mwFreq=9.5, Range=[330.0, 345.0], nPoints=512)
    >>> B, spc = pepper(sys, exp)
    >>> B.shape, spc.shape, float(B[0]), float(B[-1])
    (torch.Size([512]), torch.Size([512]), 330.0, 345.0)

    Two components weighted 70:30, returned separately:

    >>> sys_a = SpinSystem(S=0.5, g=[2.0, 2.1, 2.2], lw=1.0, weight=0.7)
    >>> sys_b = SpinSystem(S=0.5, g=2.0037, lw=0.8, weight=0.3)
    >>> exp = Experiment(mwFreq=9.5, Range=[300.0, 350.0], nPoints=256)
    >>> B, spc = pepper([sys_a, sys_b], exp, Options(separate='components'))
    >>> spc.shape
    torch.Size([2, 256])

    A frequency sweep at fixed field (``lw`` in MHz; absorption by default):

    >>> exp = Experiment(Field=340.0, mwRange=[9.4, 9.7], nPoints=300)
    >>> nu, spc = pepper(SpinSystem(S=0.5, g=[2.0, 2.0, 2.05], lw=10.0), exp)
    >>> round(float(nu[0]), 2), round(float(nu[-1]), 2), exp.Harmonic
    (9.4, 9.7, 0)
    """
    opt_ = opt if opt is not None else Options()
    from torchspin.experiment import require_positive_field
    require_positive_field(exp, opt_)    # Field = 0 only with Options.ZeroField
    # Components × isotopologues (EasySpin compisoloop): every component is
    # expanded into its isotopologues (natural abundance or Sys.Abund), each
    # simulated separately and summed with weight Sys.weight·abundance.  With
    # Opt.separate='components' the isotopologue spectra are returned as rows.
    components = expand_components(sys_or_list, getattr(opt_, 'IsoCutoff', 1e-4))
    _method = str(getattr(opt_, 'Method', 'matrix'))
    # Perturbation order: resfields_perturb implements first and second order, so
    # the higher names that garlic accepts are rejected here rather than silently
    # run as second order.
    if _method.startswith('perturb') and _method not in PEPPER_PERTURB_ORDER:
        raise ValueError(
            f"pepper does not implement Options.Method='{_method}'. Its perturbation "
            f"solver is second order: use 'perturb' or 'perturb2' for second order, "
            f"'perturb1' for first order, or 'hybrid'/'matrix'."
        )
    # Perturbation theory takes the electron Zeeman term from Sys.g only: with
    # g given as Ham110/Ham112 (g = 0) the spectrum was all zero, and other
    # Sys.Ham terms were dropped.  EasySpin's pepper does the same silently;
    # its resfreqs_perturb raises (resfreqs_perturb.m:75-77).
    if _method.startswith('perturb'):
        from torchspin.spinsystem import _reject_ham
        for _s in components:
            _reject_ham(_s, f"pepper with Options.Method='{_method}'",
                        "Options(Method='matrix')")
    # Inputs on the autograd graph: run the differentiable forward path, as
    # garlic does, instead of returning a silently detached spectrum.
    from torchspin.autograd import inputs_require_grad
    if inputs_require_grad(sys_or_list, exp):
        from torchspin.pepper_autograd import pepper_autograd
        try:
            return pepper_autograd(sys_or_list, exp, opt_)
        except NotImplementedError as e:
            raise NotImplementedError(
                f"pepper: some SpinSystem/Experiment inputs require grad, so pepper runs "
                f"its differentiable path (torchspin.pepper_autograd.pepper_autograd), "
                f"which does not support this configuration: {e} Detach the inputs "
                f"(tensor.detach()) for a non-differentiable spectrum.") from e
    # Sets of equivalent nuclei multiply the Hilbert space, so the exact solver
    # cannot collapse them (EasySpin pepper rejects Sys.n > 1 outright).  The
    # perturbation and hybrid solvers treat nuclei one at a time and combine
    # them combinatorially, so there the multiplicity is a cost saving.
    if _method in ('matrix', 'exact'):
        for _s in components:
            if any(v > 1 for v in (_s.n or [])):
                raise ValueError(
                    "pepper's matrix method does not support sets of equivalent nuclei "
                    "(SpinSystem.n > 1). List each nucleus separately in Nucs, or use "
                    "Options(Method='hybrid') or Method='perturb', which do support n."
                )
    rows, B_out = [], None
    for sys in components:
        w = float(getattr(sys, 'weight', 1.0))
        B_out, spec = _pepper_single(sys, exp, opt_)
        rows.append(w * spec)
    if str(getattr(opt_, 'separate', '')) in ('components', 'transitions', 'orientations', 'sites'):
        # EasySpin compisoloop: any separate output concatenates the rows of all
        # components / isotopologues along the first dimension
        return B_out, torch.cat([r if r.ndim == 2 else r.unsqueeze(0) for r in rows], dim=0)
    return B_out, sum(rows)



def _pepper_single(
    sys: SpinSystem,
    exp: Experiment,
    opt: Optional[Options] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate one spin system (one component, one isotopologue).

    Internal worker of :func:`pepper`, which expands components and
    isotopologues, validates ``Options.Method`` and dispatches to the
    differentiable path; call :func:`pepper` instead.  Arguments, returns
    and units are those of :func:`pepper` for a single ``SpinSystem``
    (``opt=None`` means ``Options()``); the returned spectrum is not yet
    multiplied by ``sys.weight``.
    """
    if opt is None:
        opt = Options()
    # Sys.HStrainL: Lorentzian strain = HStrain widths with Lorentzian lines.
    # Default (None) leaves sys untouched and strain_lorentz False.
    from torchspin.strainwidth import lorentzian_strain_system
    sys, strain_lorentz = lorentzian_strain_system(sys)

    # Dispatch to frequency-swept path when Exp.Field and Exp.mwRange are set
    # Frequency sweeps (Exp.Field + Exp.mwRange) run through the same grid /
    # interpolation / projection / summation machinery as field sweeps
    # (EasySpin pepper.m): the x axis is the frequency range in GHz and the
    # resonances come from resfreqs_batch instead of resfields_batch.
    freq_sweep = bool(exp.is_freq_swept) or (exp.Field is not None and exp.mwFreq is None)
    from torchspin.experiment import require_positive_field
    # Exp.Field = 0 (Options.ZeroField): degenerate levels are summed in clusters
    zero_field = require_positive_field(exp, opt) and freq_sweep
    from dataclasses import replace as _dc_replace_f
    if not freq_sweep and exp.Range is None:
        # EasySpin pepper.m: automatic field range for S=1/2 from the g and A extremes
        if sys.nElectrons != 1 or float(sys.S[0]) != 0.5:
            raise ValueError('Cannot automatically determine field range. Please provide Exp.CenterSweep or Exp.Range.')
        from torchspin.constants import PLANCK, BMAGN
        if sys.nNuclei > 0:
            A = sys.A.detach().cpu().numpy()
            if sys.fullA:
                Amax = np.abs(A).reshape(sys.nNuclei, 3, -1).max(axis=(1, 2))
            else:
                Amax = np.abs(A).max(axis=1)
            hf = float(np.sum(np.asarray(sys.I, dtype=float) * Amax)) * 1e6   # Hz
        else:
            hf = 0.0
        g = sys.g.detach().cpu().numpy()
        gvals = np.linalg.eigvalsh(g) if sys.fullg else g.reshape(-1)
        gmax, gmin = float(np.max(gvals)), float(np.min(gvals))
        if gmin == 0.0 or gmax == 0.0:
            # B55: g = 0 when the linear Zeeman term is given as Ham110/Ham112.
            # EasySpin's pepper.m (241-272) takes the range from Sys.g alone
            # too and then fails with an index error (pepper.m:324).
            from torchspin.spinsystem import _ham_linear_orders
            _keys = _ham_linear_orders(sys)
            why = (f"the linear Zeeman term is given as Sys.Ham* ({', '.join(_keys)}) "
                   f"and Sys.g is zero" if _keys else 'a principal g value is zero')
            raise ValueError(
                f'Cannot automatically determine field range: it is computed from '
                f'Sys.g, and {why}. Please provide Exp.CenterSweep or Exp.Range.')
        minB =PLANCK * (float(exp.mwFreq) * 1e9 - hf) / BMAGN / gmax / 1e-3
        maxB = PLANCK * (float(exp.mwFreq) * 1e9 + hf) / BMAGN / gmin / 1e-3
        lw_max = float(max(sys.get_lw()))
        center = (maxB + minB) / 2
        sweep = maxB - minB + 3 * lw_max
        if sweep == 0:
            sweep = 5 * lw_max
        if sweep == 0:
            sweep = 10.0
        sweep *= 1.25
        exp = _dc_replace_f(exp, Range=[center - sweep / 2, center + sweep / 2], CenterSweep=None)
    freq_auto_range = freq_sweep and exp.mwRange is None
    if freq_sweep:
        if float(getattr(exp, 'ModAmp', 0.0) or 0.0) > 0:
            raise ValueError('Exp.ModAmp cannot be used with frequency sweeps.')
        rng0 = [float(exp.mwRange[0]), float(exp.mwRange[1])] if not freq_auto_range else [0.0, 1.0]
        exp = _dc_replace_f(exp, Range=rng0, mwRange=rng0, CenterSweep=None, mwCenterSweep=None)

    # -----------------------------------------------------------------------
    # Step 0: Determine grid symmetry (automatic or user-specified)
    # -----------------------------------------------------------------------
    grid_sym = opt.GridSymmetry
    R_sym2mol = torch.eye(3, dtype=torch.float64)  # symmetry → molecular frame rotation
    if grid_sym == 'auto' or grid_sym == '':
        # Automatic symmetry detection
        grid_sym, R_sym2mol = grid_symmetry(sys)
        if opt.Verbosity >= 1:
            print(f"pepper: automatic symmetry detection → '{grid_sym}'")
    else:
        if opt.Verbosity >= 1:
            print(f"pepper: user-specified symmetry '{grid_sym}'")

    # EasySpin p_sampletype/p_gridsetup: partially ordered samples (Exp.Ordering)
    # use a Ci grid in the molecular frame; a non-equilibrium state reduces an
    # axial (Dinfh) grid to D2h.
    from torchspin.ordering import ordering_function
    ordering_fun = ordering_function(getattr(exp, 'Ordering', None))
    if ordering_fun is not None:
        if getattr(exp, 'MolFrame', None) is not None or getattr(exp, 'CrystalSymmetry', None) is not None:
            raise ValueError('Exp.MolFrame/Exp.CrystalSymmetry cannot be used for partially ordered samples (Exp.Ordering given).')
        if opt.GridSymmetry in ('auto', ''):
            grid_sym, R_sym2mol = 'Ci', torch.eye(3, dtype=torch.float64)
        elif grid_sym != 'Ci':
            raise ValueError('For partially ordered samples, Ci grid symmetry is required.')
    if getattr(sys, 'initState', None) is not None and grid_sym == 'Dinfh' and opt.GridSymmetry in ('auto', ''):
        grid_sym = 'D2h'
    # Exp.Orientations: user-supplied field directions (molecular frame) replace
    # the powder grid; GridSize/GridSymmetry are not used.
    user_ori = _user_orientations(exp)
    if user_ori is not None:
        _check_user_orientations(exp, opt)
    if zero_field:
        if getattr(sys, 'initState', None) is not None:
            raise ValueError('Options.ZeroField cannot be combined with non-equilibrium populations (Sys.initState).')
        if grid_sym == 'O3':
            # the one-orientation O3 shortcut is not used at zero field: the
            # B1 average is taken over a real powder grid
            grid_sym, R_sym2mol = 'D2h', torch.eye(3, dtype=torch.float64)
            if opt.Verbosity >= 1:
                print("pepper: zero field, O3 grid replaced by D2h")

    # -----------------------------------------------------------------------
    # Step 1: Build field-independent Hamiltonian and moment operators
    # -----------------------------------------------------------------------
    # EasySpin's pepper runs second order for both 'perturb' and 'perturb2'.
    pt_order = PEPPER_PERTURB_ORDER.get(str(opt.Method), 2)

    # Hybrid (EasySpin Opt.Method='hybrid') splits the system in two: the
    # operators below, and everything else that scales with the Hilbert space,
    # are built from the core — all electron spins plus Opt.HybridCoreNuclei —
    # while the remaining nuclei only shift and split the core lines, inside
    # resfields_batch.  Grid symmetry, linewidths and strains stay those of the
    # system as given, as in EasySpin.
    #
    # A component without nuclei has nothing to treat perturbationally, so it is
    # simulated exactly whatever Method says (EasySpin resfields.m enters the
    # hybrid branch only `if CoreSys.nNuclei>=1`).  That is what makes
    # Method='hybrid' usable on a list of components where only some carry
    # nuclei, such as a metal centre plus a radical impurity.
    use_hybrid = str(opt.Method) == 'hybrid' and sys.nNuclei >= 1
    hybrid_info = None
    ham_sys = sys
    if use_hybrid:
        if freq_sweep:
            raise ValueError("Options.Method='hybrid' is not available for frequency sweeps.")
        if sys.nn is not None and torch.as_tensor(sys.nn).any():
            raise ValueError("Options.Method='hybrid' cannot treat nucleus-nucleus "
                             "couplings (Sys.nn); use Method='matrix'.")
        from torchspin.resfields_hybrid import core_nuclei_split, core_spin_operators
        from torchspin.spinsystem import nucspinrmv
        _core_idx, _perturb_idx = core_nuclei_split(sys, opt.HybridCoreNuclei)
        _n_sys = list(sys.n or [1] * sys.nNuclei)
        # A core nucleus is diagonalized exactly, so its copies each need their
        # own place in the Hilbert space and cannot be collapsed into a
        # multiplicity.  Only the perturbational nuclei benefit from Sys.n.
        _bad = [i + 1 for i in _core_idx if _n_sys[i] > 1]
        if _bad:
            raise ValueError(
                f"Options.HybridCoreNuclei names nuclei {_bad}, which have SpinSystem.n > 1. "
                f"A nucleus in the exactly diagonalized core cannot be a set of equivalent "
                f"nuclei: list its copies separately in Nucs, or leave it perturbational."
            )
        if _perturb_idx:
            if getattr(sys, 'H0Extra', None) is not None:
                raise ValueError("Options.Method='hybrid' removes the perturbational nuclei "
                                 "from the Hamiltonian, so it cannot include SpinSystem.H0Extra "
                                 "(an operator on the full spin space); use Method='matrix'.")
            ham_sys = nucspinrmv(sys, _perturb_idx)
            hybrid_info = {
                'full_sys': sys,
                'perturb_idx': _perturb_idx,
                'n_equiv': list(sys.n or [1] * sys.nNuclei),
                'int_threshold': float(opt.HybridIntThreshold),
                'S_core': core_spin_operators(ham_sys),
            }
        else:
            # Every nucleus in the core: this is the matrix method.
            use_hybrid = False

    dim = hsdim(ham_sys)          # Hilbert space dimension, for batch sizing
    device = torch.device(opt.device)
    _ops: list = []

    def _ham_ops():
        """(H0, mux, muy, muz) on the target device, built and cached on demand.

        The perturbation solver works from the spin system directly and never
        touches these, while building them is a full Hilbert-space Kronecker
        product — 648x648 for Cu + 4x14N, which was about 80 % of the run time
        of a perturbation simulation, once per isotopologue.
        """
        if not _ops:
            _ops.extend(t.to(device) for t in ham(ham_sys, B0=None))
        return _ops

    # Options.QuadraticZeeman: field-quadratic/cubic Sys.Ham terms (lB = 2, 3) in
    # the matrix resonance search.  Perturbation theory and the hybrid method work
    # from H0 - B·μ and cannot include them.
    _zho = None
    if bool(getattr(opt, 'QuadraticZeeman', False)):
        from torchspin.ham_ezho import field_nonlinear_zeeman_ops, has_field_nonlinear_zeeman
        if has_field_nonlinear_zeeman(sys):
            if str(opt.Method).startswith('perturb') or hybrid_info is not None:
                raise NotImplementedError(
                    f"pepper: Options.Method='{opt.Method}' does not support Options.QuadraticZeeman=True "
                    f"with field-quadratic/cubic Sys.Ham terms (lB >= 2); use Method='matrix'.")
            _zho = tuple(None if g is None else g.to(device) for g in field_nonlinear_zeeman_ops(ham_sys))

    # Non-equilibrium populations (Sys.initState) → density matrix in the
    # Hamiltonian basis; not available with perturbation theory (EasySpin).
    from torchspin.initstate import init_state_density
    init_state = init_state_density(ham_sys, _ham_ops()[0]) if getattr(sys, 'initState', None) is not None else None
    if init_state is not None and str(opt.Method).startswith('perturb'):
        raise ValueError('Perturbation theory not available for systems with non-equilibrium populations.')
    # Photoselection (Exp.lightBeam, Sys.tdm): per-orientation weights
    light_beam = getattr(exp, 'lightBeam', None)
    light_scatter = float(getattr(exp, 'lightScatter', 0.0) or 0.0)
    use_photo = light_beam not in (None, '') and light_scatter < 1
    if use_photo:
        if getattr(sys, 'tdm', None) is None:
            raise ValueError('To include photoselection weights, Sys.tdm must be given.')
        if isinstance(light_beam, str):
            photo_k = [0.0, 1.0, 0.0]                 # beam along yL
            photo_alpha = {'perpendicular': -math.pi / 2, 'parallel': math.pi,
                           'unpolarized': float('nan')}.get(light_beam)
            if photo_alpha is None:
                raise ValueError("Unknown string in Exp.lightBeam. Use '', 'perpendicular', 'parallel' or 'unpolarized'.")
        else:
            photo_k, photo_alpha = light_beam[0], float(light_beam[1])

        def _photo_w(ori):
            from torchspin.photoselect import photoselect
            w = photoselect(sys.tdm, np.asarray(ori, dtype=float), photo_k, photo_alpha)
            return torch.tensor((1.0 - light_scatter) * w + light_scatter, dtype=torch.float64)
    # perturbation shortcuts are only taken for equilibrium, non-photoselected spectra
    _pt_ok = init_state is None and not freq_sweep
    from dataclasses import replace as _dc_replace_o
    _opt_rf = _dc_replace_o(opt, Threshold=0.0)   # slot-based paths pre-select transitions globally (EasySpin)
    # Opt.Transitions: user-chosen level pairs replace the automatic pre-selection
    # in every matrix path; 'all' also switches the intensity post-selection off.
    _user_tr = getattr(opt, 'Transitions', None)
    if _user_tr is not None and str(opt.Method).startswith('perturb'):
        raise ValueError("Options.Transitions cannot be used with perturbation theory "
                         f"(Options.Method='{opt.Method}'); use Method='matrix' or 'hybrid'.")
    _user_pairs, _trans_all = _user_transitions(_user_tr, _ham_ops()[0].shape[0]) if _user_tr is not None else (None, False)
    _post_thr = 0.0 if _trans_all else float(opt.Threshold)
    if _user_pairs is not None:
        _pairs_sel = _user_pairs
    elif zero_field:
        _pairs_sel = _preselect_pairs(*_ham_ops(), ham_sys, exp, opt, 0.0, freq_sweep=True,
                                      zero_field=True) if not str(opt.Method).startswith('perturb') else None
    else:
        _pairs_sel = _preselect_pairs(*_ham_ops(), ham_sys, exp, opt,
                                      float(exp.Field) if freq_sweep else 0.5 * (exp.Range[0] + exp.Range[1]),
                                      freq_sweep=freq_sweep, zeeman_ho=_zho) if not str(opt.Method).startswith('perturb') else None

    # Opt.separate='transitions': one spectrum row per transition slot (EasySpin
    # nSpectra = nTransitions).  Rows are collected per slot by _acc and, for the
    # matrix path, ordered by descending maximum intensity like EasySpin's
    # transition pre-selection.
    sep_trans = str(getattr(opt, 'separate', '')) == 'transitions'
    _rows: dict = {}
    _row_max: dict = {}

    def _rid(i_t):
        keys = getattr(_slot_state, 'keys', None)
        return (keys[i_t][0], keys[i_t][1]) if keys is not None and i_t < len(keys) else i_t

    def _row(i_t):
        rid = _rid(i_t)
        if rid not in _rows:
            _rows[rid] = torch.zeros(exp.nPoints, dtype=torch.float64)
            _row_max[rid] = 0.0
        return _rows[rid]

    def _acc(i_t, contrib, imax=None):
        nonlocal spec
        if sep_trans:
            rid = _rid(i_t)
            _rows[rid] = _row(i_t) + contrib
            if imax is not None:
                _row_max[rid] = max(_row_max[rid], float(imax))
        else:
            spec = spec + contrib

    def _orientation_batch_size(n_orient: int) -> int:
        """Orientations per call to the exact solver.

        Hybrid must see every orientation in one call: the nuclear sub-lines are
        pruned by their amplitude summed over the orientations present, so
        splitting the grid would let different batches keep different sub-line
        sets, and a given slot would then mean different lines for different
        orientations.  The hybrid core is small by construction, so one batch is
        what the memory budget allows anyway; if it does not, say so rather than
        silently interpolating across inconsistent slots.
        """
        size = auto_batch_size(opt.BatchSize, dim, n_orient)
        if hybrid_info is None or size >= n_orient:
            return size
        raise ValueError(
            f"Options.Method='hybrid' needs all {n_orient} orientations in one batch, but "
            f"the {dim}-dimensional core only allows {size}. Move nuclei out of "
            f"Options.HybridCoreNuclei to shrink the core, or reduce Options.GridSize."
        )

    def _resonances(phi_b, theta_b, exp_, opt_, sys_, **kw):
        """Exact resonance search; builds the Hamiltonian operators on first call.

        With Opt.Method='hybrid' the solver sees the core system and the
        perturbational-nucleus description, and returns the expanded line list.
        """
        opt_ = kw.pop('opt_override', opt_)
        if _user_pairs is not None:
            # Opt.Transitions: the crystal, O3, per-orientation and auto-range
            # calls search exactly the user's level pairs too
            if kw.get('pairs') is None:
                kw['pairs'] = _user_pairs
            if _trans_all and opt_.Threshold != 0:
                opt_ = _dc_replace_o(opt_, Threshold=0.0)
        if _zho is not None:
            kw['zeeman_ho'] = _zho
        if freq_sweep:
            if zero_field:
                kw['zero_field'] = True
            return resfreqs_batch(*_ham_ops(), phi_b, theta_b, exp_, opt_, ham_sys, **kw)
        if hybrid_info is not None:
            kw['hybrid'] = hybrid_info
        return resfields_batch(*_ham_ops(), phi_b, theta_b, exp_, opt_, ham_sys, **kw)

    def _fill_missing_knots(lists):
        """Field sweeps, matrix search: search again the level pairs that miss
        resonances at some orientations over a padded window (B88, B75; see
        :func:`_research_missing_knots`).  ``lists = (B, I, W, pairs)`` for the
        coarse grid; returned unchanged when nothing is missing.  Frequency
        sweeps need nothing (resfreqs_batch returns every level pair at every
        orientation), and the hybrid line list is pruned per call, so it is
        left as it is."""
        if freq_sweep or hybrid_info is not None:
            return lists

        def _search(window, pairs_):
            exp_w = _dc_replace(exp, Range=list(window), CenterSweep=None)
            out = ([None] * n_orient, [None] * n_orient, [None] * n_orient, [None] * n_orient)
            bs = _orientation_batch_size(n_orient)
            for s0 in range(0, n_orient, bs):
                e0 = min(s0 + bs, n_orient)
                res = _resonances(phi_arr[s0:e0], theta_arr[s0:e0], exp_w, opt, sys, return_pairs=True,
                                  init_state=init_state, photo_weights=_pw(s0, e0), opt_override=_opt_rf,
                                  pairs=pairs_)
                for col, vals in zip(out, res):
                    col[s0:e0] = list(vals)
            return out
        return _research_missing_knots(lists, exp.Range, _search, verbose=opt.Verbosity >= 1)

    # -----------------------------------------------------------------------
    # Step 2: Spherical powder grid
    # -----------------------------------------------------------------------
    # Parse GridSize: int N → compute + accumulate at N knots (no interpolation)
    #                 [N1, Ni] → compute at coarse N1-knot grid, interpolate to
    #                            fine nfKnots = (N1-1)*Ni + 1 knot grid
    N_coarse = opt.grid_size_coarse
    N_interp = opt.grid_size_interp
    nfKnots = (N_coarse - 1) * N_interp + 1 if N_interp > 1 else N_coarse

    phi_arr, theta_arr, weights_arr, vecs_orig = sphgrid(grid_sym, N_coarse)
    n_orient = phi_arr.shape[0]
    # Symmetry-frame polar angles: grid interpolation and zone/triangle weights
    # live in the symmetry frame; only the Hamiltonian is evaluated along the
    # rotated (molecular-frame) field directions below.
    theta_sym = theta_arr.clone()
    vecs_sym_c = vecs_orig if vecs_orig.shape[0] == 3 else vecs_orig.T  # (3, n_orient)

    # Rotate grid orientations from the symmetry frame to the molecular frame.
    # erot(gFrame) uses the passive convention: v_g = erot(gFrame) @ v_mol (mol→g).
    # hamsymm's _tensor_symmetry stores erot(euler_angles) directly as R, so
    # R maps mol→symmetry.  To convert sphgrid unit vectors (in symmetry frame)
    # to the molecular frame: v_mol = R.T @ v_sym.
    if not torch.allclose(R_sym2mol, torch.eye(3, dtype=torch.float64), atol=1e-10):
        sin_t = torch.sin(theta_arr)
        v_sym = torch.stack(
            [sin_t * torch.cos(phi_arr), sin_t * torch.sin(phi_arr), torch.cos(theta_arr)],
            dim=1,
        )  # (N, 3)
        v_mol = (R_sym2mol.T @ v_sym.T).T  # (N, 3)
        theta_arr = torch.acos(v_mol[:, 2].clamp(-1.0, 1.0))
        phi_arr = torch.atan2(v_mol[:, 1], v_mol[:, 0])

    # Move grid arrays to target device
    if user_ori is not None:
        # Exp.Orientations: weights are fractions of the sphere; the grid
        # weights of the powder sum to 4π, so 4π·w keeps pepper's amplitude.
        phi_arr = user_ori[0].detach().clone()
        theta_arr = user_ori[1].detach().clone()
        weights_arr = 4.0 * math.pi * user_ori[2].detach().clone()
        n_orient = phi_arr.shape[0]

    phi_arr     = phi_arr.to(device)
    theta_arr   = theta_arr.to(device)
    weights_arr = weights_arr.to(device)

    if opt.Verbosity >= 1:
        print(f"pepper: {n_orient} orientations, grid '{grid_sym}', N={opt.GridSize}")

    # -----------------------------------------------------------------------
    # Step 3: Accumulate spectrum over all orientations (with optional batching)
    # -----------------------------------------------------------------------
    # x and spec stay on CPU; only Hamiltonians/grid move to device for fast eigh
    if freq_auto_range:
        # EasySpin pepper.m: automatic frequency range from the resonance
        # frequencies of the coarse grid (± spread/5, ≥ 5×strain width, ≥ 5×Σlw)
        pw0 = _photo_w(torch.stack([phi_arr, theta_arr], dim=1).detach().cpu().numpy()) if use_photo else None
        P_l, _, W_l = _resonances(phi_arr, theta_arr, exp, opt, sys,
                                  init_state=init_state, photo_weights=pw0)
        pos = torch.cat([p_ for p_ in P_l if p_ is not None and p_.numel() > 0])
        f_min, f_max = float(pos.min()), float(pos.max())
        padding = (f_max - f_min) / 5.0
        if padding == 0:
            padding = 0.1
        w_all = [w_ for w_ in W_l if w_ is not None and w_.numel() > 0]
        if w_all:
            padding = max(padding, 5.0 * float(torch.cat(w_all).max()))
        padding = max(padding, 5.0 * float(sum(sys.get_lw())) / 1e3)   # Sys.lw in MHz
        rng_auto = [max(0.0, f_min - padding), f_max + padding]
        exp = _dc_replace_f(exp, Range=rng_auto, mwRange=rng_auto)
    x = torch.linspace(exp.Range[0], exp.Range[1], exp.nPoints, dtype=torch.float64)
    spec = torch.zeros(exp.nPoints, dtype=torch.float64)
    dx = (exp.Range[1] - exp.Range[0]) / (exp.nPoints - 1)  # mT per point
    # EasySpin pepper.m: the perturbation solver (resfields_perturb) searches
    # Exp.SearchRange = Range ± 20 % of the sweep width (clamped at 0) so that
    # lines just outside the window contribute their in-range part; the
    # matrix-diagonalization solver (resfields) searches the sweep range only.
    from dataclasses import replace as _dc_replace
    _w = exp.Range[1] - exp.Range[0]
    exp_search = _dc_replace(exp, Range=[max(0.0, exp.Range[0] - 0.2 * _w), exp.Range[1] + 0.2 * _w],
                             CenterSweep=None)

    total_weight = weights_arr.sum().item()  # should be ~4π

    # Get effective linewidth (handles both lw and lwpp)
    lw_effective = sys.get_lw()
    mw_phase = float(getattr(exp, 'mwPhase', 0.0) or 0.0)   # EasySpin Exp.mwPhase (dispersion)
    if freq_sweep:
        lw_effective = [float(v) / 1e3 for v in lw_effective]   # Sys.lw in MHz → GHz axis
        mw_phase = -mw_phase                                      # EasySpin: mwPhase negated for frequency sweeps

    def _strain_bins(x_, pos, fwhm, amp, min_fwhm):
        """Per-line strain template: Gaussian (EasySpin strains) or, with
        Sys.HStrainL, a Lorentzian that carries Exp.mwPhase itself."""
        if strain_lorentz:
            return _lorentzian_bins(x_, pos, fwhm, amp, min_fwhm, phase=mw_phase)
        return _gaussian_bins(x_, pos, fwhm, amp, min_fwhm)
    # With Lorentzian strain the dispersion admixture is already in every line,
    # so the remaining Sys.lw is convolved with an absorption kernel.
    _conv_phase = 0.0 if strain_lorentz else mw_phase
    # EasySpin auto-harmonic: with no broadening at all (no lw, no strain) the
    # spectrum is a stick/projection absorption spectrum; a derivative of it
    # is not meaningful, so Harmonic falls back to 0 (EasySpin errors when the
    # harmonic was requested explicitly; torchspin cannot tell that apart).

    # Check if we need per-line broadening (strain present)
    has_strain = (
        (sys.HStrain is not None and torch.any(sys.HStrain > 0))
        or (sys.gStrain is not None and torch.any(sys.gStrain > 0))
        or (sys.AStrain is not None and torch.any(sys.AStrain > 0))
        or (sys.DStrain is not None and torch.any(sys.DStrain > 0))
    )
    if not has_strain and lw_effective[0] == 0 and lw_effective[1] == 0 and exp.Harmonic > 0:
        if opt.Verbosity >= 1:
            print(f"pepper: no broadening given; using Harmonic=0 instead of {exp.Harmonic}")
        from dataclasses import replace as _dc_replace0
        exp = _dc_replace0(exp, Harmonic=0, CenterSweep=None)

    # photoselection weights for the (χ-averaged) powder orientations, molecular frame
    photo_w_all = _photo_w(torch.stack([phi_arr, theta_arr], dim=1).detach().cpu().numpy()) if use_photo else None
    _pw = (lambda a, b: None) if photo_w_all is None else (lambda a, b: photo_w_all[a:b])
    # ordering: sample orientation R_L2S from Exp.SampleFrame / Exp.SampleRotation (single)
    _ordering = None
    if ordering_fun is not None:
        from torchspin.rotutils import rotaxi2mat as _rotaxi2mat
        sf_ = getattr(exp, 'SampleFrame', None)
        sf_ = np.zeros((1, 3)) if sf_ is None else np.atleast_2d(np.asarray(sf_, dtype=float))
        if sf_.shape[0] != 1:
            raise ValueError('For partially ordered samples, only a single sample orientation (Exp.SampleFrame) can be used.')
        R_L2S = erot(sf_[0].tolist()).numpy()
        rot_ = getattr(exp, 'SampleRotation', None)
        if rot_ is not None:
            axis_, rho_ = rot_
            if isinstance(axis_, str):
                axis_ = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}[axis_.lower()]
            rho_ = np.atleast_1d(np.asarray(rho_, dtype=float))
            if rho_.size != 1:
                raise ValueError('For partially ordered samples, only a single sample rotation can be used.')
            R_L2S = R_L2S @ _rotaxi2mat(np.asarray(axis_, dtype=float), float(rho_[0]))
        _ordering = (ordering_fun, R_L2S)

    if user_ori is not None:
        # ---------------------------------------------------------------
        # Exp.Orientations: sum of stick spectra over the given molecular-
        # frame field directions, χ-averaged rates as for a powder (no R_batch),
        # amplitude I·4π·w; every line is accumulated with an exact template at
        # its exact position (as on the O3 and crystal paths) — Gaussian
        # (strain width, or Sys.lw(1)) or Lorentzian (Sys.lw(2) alone) — and the
        # rest of Sys.lw is convolved afterwards.  No interpolation/projection.
        # ---------------------------------------------------------------
        _min_fwhm0 = dx / 100.0
        use_pt = (sys.nElectrons == 1 and not has_strain
                  and not (sys.S[0] > 0.5 and sys.D is not None)
                  and _pt_ok and str(opt.Method).startswith('perturb'))
        lines = []          # (B, I, W) per orientation, numpy
        if use_pt:
            Bl, Il = resfields_perturb_batch(sys, phi_arr, theta_arr, exp_search, opt, order=pt_order,
                                             return_full=True, photo_weights=_pw(0, n_orient))
            for k in range(n_orient):
                B0 = Bl[k].detach().cpu().numpy(); I0 = Il[k].detach().cpu().numpy()
                keep = np.isfinite(B0) & (I0 != 0)
                lines.append((B0[keep], I0[keep], np.zeros(int(keep.sum()))))
        else:
            batch_size = _orientation_batch_size(n_orient)
            for s0 in range(0, n_orient, batch_size):
                e0 = min(s0 + batch_size, n_orient)
                Bl, Il, Wl = _resonances(phi_arr[s0:e0], theta_arr[s0:e0], exp, opt, sys,
                                         init_state=init_state, photo_weights=_pw(s0, e0))
                for k in range(e0 - s0):
                    B0 = Bl[k].detach().cpu().numpy() if Bl[k] is not None else np.zeros(0)
                    I0 = Il[k].detach().cpu().numpy() if Il[k] is not None else np.zeros(0)
                    W0 = (Wl[k].detach().cpu().numpy() if (Wl[k] is not None and Wl[k].numel() == B0.size)
                          else np.zeros_like(B0))
                    lines.append((B0, I0, W0))
        w_np = weights_arr.detach().cpu().numpy()
        fwhm_g0, fwhm_l0 = lw_effective[0], lw_effective[1]

        def _ori_template(B0, W0, amp0):
            if has_strain:
                return _strain_bins(x, B0, W0, amp0, _min_fwhm0)
            if fwhm_g0 > 0:
                return _gaussian_bins(x, B0, np.full(B0.size, fwhm_g0), amp0, _min_fwhm0)
            if fwhm_l0 > 0:
                return _lorentzian_bins(x, B0, np.full(B0.size, fwhm_l0), amp0, _min_fwhm0, phase=mw_phase)
            return _gaussian_bins(x, B0, np.zeros(B0.size), amp0, _min_fwhm0)   # delta-like sticks
        if str(opt.separate or '') == 'orientations':
            spec = torch.stack([_ori_template(B0, W0, I0 * w_np[k]) for k, (B0, I0, W0) in enumerate(lines)])
        else:
            B_all = np.concatenate([ln[0] for ln in lines])
            I_all = np.concatenate([ln[1] * w_np[k] for k, ln in enumerate(lines)])
            W_all = np.concatenate([ln[2] for ln in lines])
            spec = _ori_template(B_all, W_all, I_all)
        spec = spec / dx * (2.0 * math.pi)
        if has_strain:
            g_rem, l_rem, deriv_fd = fwhm_g0, fwhm_l0, (fwhm_g0 == 0 and fwhm_l0 == 0)
        elif fwhm_g0 > 0:
            g_rem, l_rem, deriv_fd = 0.0, fwhm_l0, (fwhm_l0 == 0)
        else:
            g_rem, l_rem, deriv_fd = 0.0, 0.0, True
        spec = _finish_field_sweep(spec, x, dx, exp, g_rem, l_rem, deriv_fd,
                                   phase=_conv_phase if l_rem > 0 else 0.0,
                                   pad_multiplier=opt.PaddingMultiplier)
        return x, spec

    # EasySpin p_sampletype: a crystal needs Exp.MolFrame or Exp.CrystalSymmetry;
    # Exp.SampleFrame/SampleRotation alone describe a (rotated) disordered sample,
    # which is orientation-independent and therefore simulated as a plain powder.
    crystal_sample = (getattr(exp, 'CrystalSymmetry', None) is not None
                      or getattr(exp, 'MolFrame', None) is not None)
    if crystal_sample:
        if sep_trans:
            raise ValueError("Cannot return separate transitions for crystal spectra (Opt.separate='transitions').")
        # ---------------------------------------------------------------
        # Single crystal (EasySpin pepper.m crystalSample branch): explicit
        # molecule→lab orientations for every sample orientation × site,
        # intensities with B1 along lab x (no χ average), each line
        # accumulated with an exact template, weight 2π·4π/(nSites·nOri).
        # ---------------------------------------------------------------
        from torchspin.sitetransforms import crystal_orientations, sitetransforms
        from torchspin.rotutils import eulang, rotaxi2mat
        sf = getattr(exp, 'SampleFrame', None)
        sample_frames = np.zeros((1, 3)) if sf is None else np.atleast_2d(np.asarray(sf, dtype=float))
        rot = getattr(exp, 'SampleRotation', None)
        if rot is not None:
            axis, rho = rot
            if isinstance(axis, str):
                axis = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}[axis.lower()]
            rhos = np.atleast_1d(np.asarray(rho, dtype=float))
            frames = []
            for sfrow in sample_frames:
                R_L2S0 = erot(sfrow.tolist()).numpy()
                for r_ in rhos:
                    frames.append(eulang(R_L2S0 @ rotaxi2mat(np.asarray(axis, dtype=float), float(r_))))
            sample_frames = np.array(frames).reshape(-1, 3)
        csym = getattr(exp, 'CrystalSymmetry', None)
        n_sites = len(sitetransforms(csym if csym not in (None, '') else 'P1'))
        angles = crystal_orientations(sample_frames, csym, getattr(exp, 'MolFrame', None))  # (nSamples*nSites, 3)
        n_samples = sample_frames.shape[0]
        site_of = np.tile(np.arange(n_sites), n_samples)
        sample_of = np.repeat(np.arange(n_samples), n_sites)
        if opt.Sites is not None:
            keep_s = np.isin(site_of + 1, np.atleast_1d(opt.Sites))
            angles, site_of, sample_of = angles[keep_s], site_of[keep_s], sample_of[keep_s]
            n_sites_eff = int(np.unique(site_of).size)
        else:
            n_sites_eff = n_sites
        photo_w_c = _photo_w(angles) if use_photo else None        # crystal: no χ average
        Rs = np.stack([erot(a.tolist()).numpy() for a in angles])       # (N, 3, 3); rows = xLab_M, yLab_M, zLab_M
        zlab = Rs[:, 2, :]; xlab = Rs[:, 0, :]
        theta_c = torch.tensor(np.arccos(np.clip(zlab[:, 2], -1, 1)), dtype=torch.float64)
        phi_c = torch.tensor(np.arctan2(zlab[:, 1], zlab[:, 0]), dtype=torch.float64)
        B_l, I_l, W_l = _resonances(phi_c.to(device), theta_c.to(device), exp, opt, sys,
                                        R_batch=torch.tensor(Rs, dtype=torch.float64), init_state=init_state, photo_weights=photo_w_c)
        _min_fwhm0 = dx / 100.0
        fwhm_g0, fwhm_l0 = lw_effective[0], lw_effective[1]
        n_ori_tot = angles.shape[0]
        sep = str(opt.separate or '')
        n_spec = n_samples if sep == 'orientations' else (n_sites_eff if sep == 'sites' else 1)
        specs = torch.zeros(n_spec, exp.nPoints, dtype=torch.float64)
        for k in range(n_ori_tot):
            B0 = B_l[k].detach().cpu().numpy(); I0 = I_l[k].detach().cpu().numpy()
            if B0.size == 0:
                continue
            W0 = W_l[k].detach().cpu().numpy() if (W_l[k] is not None and W_l[k].numel() == B0.size) else np.zeros_like(B0)
            amp0 = I0 * (4.0 * math.pi) / n_sites_eff / n_samples
            if has_strain:
                line = _strain_bins(x, B0, W0, amp0, _min_fwhm0)
            elif fwhm_g0 > 0:
                line = _gaussian_bins(x, B0, np.full(B0.size, fwhm_g0), amp0, _min_fwhm0)
            elif fwhm_l0 > 0:
                line = _lorentzian_bins(x, B0, np.full(B0.size, fwhm_l0), amp0, _min_fwhm0, phase=mw_phase)
            else:
                line = _gaussian_bins(x, B0, np.zeros(B0.size), amp0, _min_fwhm0)
            idx_s = sample_of[k] if sep == 'orientations' else (int(np.searchsorted(np.unique(site_of), site_of[k])) if sep == 'sites' else 0)
            specs[idx_s] += line
        specs = specs / dx * (2.0 * math.pi)
        if has_strain:
            g_rem, l_rem, deriv_fd = fwhm_g0, fwhm_l0, (fwhm_g0 == 0 and fwhm_l0 == 0)
        elif fwhm_g0 > 0:
            g_rem, l_rem, deriv_fd = 0.0, fwhm_l0, (fwhm_l0 == 0)
        else:
            g_rem, l_rem, deriv_fd = 0.0, 0.0, True
        spec = torch.stack(list(specs)) if n_spec > 1 else specs[0]
        # phase enters the Lorentzian template (above) or the Lorentzian convolution
        spec = _finish_field_sweep(spec, x, dx, exp, g_rem, l_rem, deriv_fd,
                                   phase=_conv_phase if l_rem > 0 else 0.0,
                                   pad_multiplier=opt.PaddingMultiplier)
        return x, spec

    if grid_sym == 'O3':
        # Isotropic system, one orientation (EasySpin ~anisotropicSpectrum):
        # every line is accumulated with an exact template at its exact
        # position — Gaussian (strain width, or Sys.lw(1)) or Lorentzian
        # (Sys.lw(2) when there is no Gaussian part) — instead of stick binning.
        _min_fwhm0 = dx / 100.0
        use_pt = (sys.nElectrons == 1 and not has_strain
                  and not (sys.S[0] > 0.5 and sys.D is not None)
                  and _pt_ok and str(opt.Method).startswith('perturb'))
        if use_pt:
            Bl, Il = resfields_perturb_batch(sys, phi_arr, theta_arr, exp_search, opt, order=pt_order, return_full=True, photo_weights=_pw(0, n_orient))
            B0 = Bl[0].detach().cpu().numpy(); I0 = Il[0].detach().cpu().numpy()
            keep = np.isfinite(B0) & (I0 != 0)
            B0, I0 = B0[keep], I0[keep]; W0 = np.zeros_like(B0)
        else:
            Bl, Il, Wl = _resonances(phi_arr, theta_arr, exp, opt, sys, init_state=init_state, photo_weights=_pw(0, n_orient))
            B0 = Bl[0].detach().cpu().numpy(); I0 = Il[0].detach().cpu().numpy()
            W0 = Wl[0].detach().cpu().numpy() if (Wl[0] is not None and Wl[0].numel() == B0.size) else np.zeros_like(B0)
        amp0 = I0 * float(weights_arr[0])            # 4π
        fwhm_g0, fwhm_l0 = lw_effective[0], lw_effective[1]
        def _o3_template(sel):
            if has_strain:
                return _strain_bins(x, B0[sel], W0[sel], amp0[sel], _min_fwhm0)
            if fwhm_g0 > 0:
                return _gaussian_bins(x, B0[sel], np.full(B0[sel].size, fwhm_g0), amp0[sel], _min_fwhm0)
            if fwhm_l0 > 0:
                # EasySpin: Lorentzian template built with Exp.mwPhase
                return _lorentzian_bins(x, B0[sel], np.full(B0[sel].size, fwhm_l0), amp0[sel], _min_fwhm0, phase=mw_phase)
            return _gaussian_bins(x, B0[sel], np.zeros(B0[sel].size), amp0[sel], _min_fwhm0)   # delta-like sticks
        if has_strain:
            g_rem, l_rem, deriv_fd = fwhm_g0, fwhm_l0, (fwhm_g0 == 0 and fwhm_l0 == 0)
        elif fwhm_g0 > 0:
            g_rem, l_rem, deriv_fd = 0.0, fwhm_l0, (fwhm_l0 == 0)
        else:
            g_rem, l_rem, deriv_fd = 0.0, 0.0, True
        if sep_trans:
            # one row per line: matrix path by descending intensity (EasySpin
            # transition pre-selection), perturbation path in line order
            order = np.arange(B0.size) if use_pt else np.argsort(-np.abs(I0), kind='stable')
            spec = torch.stack([_o3_template(slice(int(k_), int(k_) + 1)) for k_ in order]) if B0.size else spec.unsqueeze(0)
        else:
            spec = _o3_template(slice(None))
        spec = spec / dx * (2.0 * math.pi)
        # EasySpin disregards mwPhase for a Gaussian-only template spectrum
        spec = _finish_field_sweep(spec, x, dx, exp, g_rem, l_rem, deriv_fd,
                                   phase=_conv_phase if l_rem > 0 else 0.0,
                                   pad_multiplier=opt.PaddingMultiplier)
        return x, spec

    # Strain on a projectable grid (D2h-type or Dinfh): EasySpin pepper.m's
    # "doSummation" — per transition slot the positions, intensities and strain
    # widths are (optionally) interpolated to the fine grid, then every facet
    # (triangle, or meridian segment for Dinfh) contributes one Gaussian with
    # the facet's mean width, inflated by the Lambda smoothing when the line is
    # narrower than the facet's spread on the field axis. The global linewidth
    # (Sys.lw) is applied afterwards by convolution, as in EasySpin.
    # O3 (isotropic, one orientation) has no facets: it uses the stick / per-line path.
    use_summation = has_strain and grid_sym != 'O3'

    _inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
    _sqrt_8log2 = math.sqrt(8.0 * math.log(2.0))
    _min_fwhm = dx / 100.0  # matches MATLAB lisum1i.c lower-bound clipping

    # True when the SOPHE projection produced a spectral density already
    # (no 1/ΔB conversion needed at the end).
    used_projection = False

    if use_summation:
        all_B_res = [None] * n_orient
        all_intens = [None] * n_orient
        all_widths = [None] * n_orient
        all_pairs = [None] * n_orient
        batch_size = _orientation_batch_size(n_orient)
        for batch_idx in range((n_orient + batch_size - 1) // batch_size):
            s = batch_idx * batch_size
            e = min(s + batch_size, n_orient)
            B_list, I_list, W_list, P_list = _resonances(phi_arr[s:e], theta_arr[s:e], exp, opt, sys,
                return_pairs=True, init_state=init_state, photo_weights=_pw(s, e), opt_override=_opt_rf, pairs=_pairs_sel
            )
            for j, (B_res, intens, widths, prs) in enumerate(zip(B_list, I_list, W_list, P_list)):
                all_B_res[s + j] = B_res
                all_intens[s + j] = intens
                all_widths[s + j] = widths
                all_pairs[s + j] = prs
        all_B_res, all_intens, all_widths, all_pairs = _fill_missing_knots(
            (all_B_res, all_intens, all_widths, all_pairs))
        slot_B, slot_I, slot_W = _transition_slots(all_B_res, all_intens, all_widths, all_pairs, n_orient)
        slot_B, slot_I, slot_W = _filter_slots(slot_B, slot_I, slot_W, _post_thr)
        if slot_W is None:
            slot_W = [np.zeros(n_orient) for _ in slot_B]

        do_interp_s = (N_interp > 1) and (nfKnots != N_coarse) and (n_orient > 1)
        if do_interp_s:
            _, theta_fine_s, _, vecs_fine_s = sphgrid(grid_sym, nfKnots)
            vecs_sym_f_s = vecs_fine_s if vecs_fine_s.shape[0] == 3 else vecs_fine_s.T
            theta_final = theta_fine_s
            vecs_final = vecs_fine_s
            N_final = nfKnots
        else:
            theta_final = theta_sym
            vecs_final = vecs_orig
            N_final = N_coarse
        tri_idx, facet_w = _facets(grid_sym, N_final, theta_final, ordering=_ordering)
        c1, c2 = (1.57246, 18.6348) if tri_idx is None else (2.8269, 42.6843)
        smoothing = 2.0  # EasySpin Opt.Smoothing default

        for i_t in range(len(slot_B)):
            B_c, I_c, W_c = slot_B[i_t], slot_I[i_t], slot_W[i_t]
            valid = ~np.isnan(B_c)
            if do_interp_s:
                B_f = _interp_slot(B_c, grid_sym, N_coarse, nfKnots, vecs_sym_c, vecs_sym_f_s)
                if B_f is None:
                    continue
                if zero_field:
                    # one position for all orientations: one point mass (exact zero-width facets)
                    B_f = _zf_const_pos(B_f, B_c)
                I_f = _interp_slot(I_c, grid_sym, N_coarse, nfKnots, vecs_sym_c, vecs_sym_f_s, mode='val')
                W_f = _interp_slot(W_c, grid_sym, N_coarse, nfKnots, vecs_sym_c, vecs_sym_f_s, mode='val')
            else:
                B_f, I_f, W_f = B_c, I_c, W_c
            if tri_idx is None:
                fPosC = 0.5 * (B_f[:-1] + B_f[1:])
                fSpread = np.abs(B_f[1:] - B_f[:-1])
                fIntC = facet_w * 0.5 * (I_f[:-1] + I_f[1:])
                fWidM = 0.5 * (W_f[:-1] + W_f[1:])
            else:
                Bv = B_f[tri_idx]                       # (nTri, 3)
                fPosC = Bv.mean(axis=1)
                fSpread = Bv.max(axis=1) - Bv.min(axis=1)
                fIntC = facet_w * I_f[tri_idx].mean(axis=1)
                fWidM = W_f[tri_idx].mean(axis=1)
            ok = np.isfinite(fPosC) & np.isfinite(fIntC) & np.isfinite(fWidM) & (fIntC > 0)
            fPosC, fSpread, fIntC, fWidM = fPosC[ok], fSpread[ok], fIntC[ok], fWidM[ok]
            with np.errstate(divide='ignore', invalid='ignore'):
                Lambda = np.where(fSpread > 0, fWidM / np.where(fSpread > 0, fSpread, 1.0), np.inf)
                gam = 1.0 / np.sqrt(c1 * Lambda ** 2 + c2 * Lambda ** 4)
            gam[~np.isfinite(gam)] = 0.0
            fWidC = fWidM * (1.0 + smoothing * gam)
            _acc(i_t, _strain_bins(x, fPosC, fWidC, fIntC, _min_fwhm), np.nanmax(np.abs(I_c)) if np.isfinite(I_c).any() else 0.0)
        if opt.Verbosity >= 1:
            print(f"pepper: strain summation over {len(facet_w)} facets, {len(slot_B)} transition slots"
                  f"{' (interpolated)' if do_interp_s else ''}")

    else:
        # ---------------------------------------------------------------
        # Standard path: per-orientation accumulation, with optional
        # spherical interpolation from coarse to fine grid (N_interp > 1).
        # ---------------------------------------------------------------
        batch_size = _orientation_batch_size(n_orient)
        n_batches = (n_orient + batch_size - 1) // batch_size

        # Isotropic systems (O3, a single orientation) have nothing to interpolate
        do_interp = (N_interp > 1) and (nfKnots != N_coarse) and (n_orient > 1)

        if do_interp:
            # -----------------------------------------------------------
            # Interpolation path: collect coarse data, interpolate to
            # fine grid (matching MATLAB's GridSize = [N_coarse, N_interp]).
            # -----------------------------------------------------------
            all_B_res  = [None] * n_orient
            all_intens = [None] * n_orient
            all_widths = [None] * n_orient
            all_pairs  = [None] * n_orient

            # Use perturbation theory for single-electron systems without strain.
            # Falls back to matrix diagonalization for high-spin + ZFS:
            # perturbation theory cannot reproduce ZFS-split energy levels regardless
            # of temperature (kT comparison is not the only issue — the ZFS mixes
            # states and shifts resonance fields in ways perturbation theory misses).
            # S=1/2 + temperature is fine: Zeeman-only Boltzmann is exact for S=1/2.
            _needs_exact_boltz = (sys.S[0] > 0.5 and sys.D is not None)
            _use_perturb = (sys.nElectrons == 1
                            and not has_strain
                            and not _needs_exact_boltz
                            and _pt_ok and str(opt.Method).startswith('perturb'))
            if _use_perturb:
                B_list, I_list = resfields_perturb_batch(sys, phi_arr, theta_arr, exp_search, opt, order=pt_order, return_full=True, photo_weights=_pw(0, n_orient)
                )
                all_B_res  = B_list
                all_intens = I_list
                all_widths = [None] * n_orient
                if opt.Verbosity >= 1:
                    print(f"pepper: coarse grid {n_orient} orientations (perturb)")
            else:
                for batch_idx in range(n_batches):
                    s = batch_idx * batch_size
                    e = min(s + batch_size, n_orient)
                    B_list, I_list, W_list, P_list = _resonances(phi_arr[s:e], theta_arr[s:e], exp, opt, sys,
                        return_pairs=True, init_state=init_state, photo_weights=_pw(s, e), opt_override=_opt_rf, pairs=_pairs_sel
                    )
                    for j, (B_res, intens, widths, prs) in enumerate(zip(B_list, I_list, W_list, P_list)):
                        all_B_res[s + j] = B_res
                        all_intens[s + j] = intens
                        all_widths[s + j] = widths
                        all_pairs[s + j] = prs
                    if opt.Verbosity >= 1:
                        print(f"pepper: coarse grid {e}/{n_orient} orientations computed")
                all_B_res, all_intens, all_widths, all_pairs = _fill_missing_knots(
                    (all_B_res, all_intens, all_widths, all_pairs))

            # Build the fine grid
            phi_fine, theta_fine, weights_fine, vecs_fine_orig = sphgrid(grid_sym, nfKnots)
            n_orient_fine = phi_fine.shape[0]
            total_weight  = weights_fine.sum().item()
            theta_fine_sym = theta_fine.clone()
            vecs_sym_f = vecs_fine_orig if vecs_fine_orig.shape[0] == 3 else vecs_fine_orig.T

            # Apply frame rotation to fine grid if needed
            if not torch.allclose(R_sym2mol, torch.eye(3, dtype=torch.float64), atol=1e-10):
                sin_t_f = torch.sin(theta_fine)
                v_sym_f = torch.stack([
                    sin_t_f * torch.cos(phi_fine),
                    sin_t_f * torch.sin(phi_fine),
                    torch.cos(theta_fine),
                ], dim=1)
                v_mol_f = (R_sym2mol.T @ v_sym_f.T).T
                theta_fine = torch.acos(v_mol_f[:, 2].clamp(-1.0, 1.0))
                phi_fine   = torch.atan2(v_mol_f[:, 1], v_mol_f[:, 0])

            if opt.Verbosity >= 1:
                print(f"pepper: interpolating to fine grid ({n_orient_fine} orientations)")

            # Coarse unit vectors (already rotated to mol frame from phi_arr/theta_arr)
            sin_tc = torch.sin(theta_arr)
            vecs_c = torch.stack([
                sin_tc * torch.cos(phi_arr),
                sin_tc * torch.sin(phi_arr),
                torch.cos(theta_arr),
            ], dim=0)  # (3, n_orient)

            # Fine unit vectors
            sin_tf = torch.sin(theta_fine)
            vecs_f = torch.stack([
                sin_tf * torch.cos(phi_fine),
                sin_tf * torch.sin(phi_fine),
                torch.cos(theta_fine),
            ], dim=0)  # (3, n_orient_fine)

            # Determine consistent transition count
            n_trans_list = [b.numel() if b is not None else 0 for b in all_B_res]
            n_trans = max(n_trans_list) if n_trans_list else 0

            if n_trans > 0:
                # SOPHE projection: for D2h/Dinfh grids without strain, use
                # projecttriangles / projectzones instead of per-orientation
                # stick binning.  This eliminates quantization artifacts at
                # small linewidths (the root cause of the oscillation problem).
                x_np = x.numpy()
                # SOPHE projection for every grid symmetry (EasySpin: zones for
                # Dinfh, triangles otherwise — including the Delaunay-triangulated
                # open-φ grids Ci/C2h/C1).
                use_proj = (not has_strain) and grid_sym != 'O3'
                if use_proj:
                    _tri_f, _areas_f = _facets(grid_sym, nfKnots, theta_fine_sym, ordering=_ordering)
                    _seg_wts = _areas_f

                slot_B, slot_I, slot_W = _transition_slots(
                    all_B_res, all_intens, all_widths, all_pairs, n_orient
                )
                slot_B, slot_I, slot_W = _filter_slots(slot_B, slot_I, slot_W, _post_thr)
                if use_proj:
                    # SOPHE triangle/zone projection: each facet contributes a
                    # tent/rect density → smooth spectrum without quantization noise.
                    # The slots are independent: interpolate and project them in
                    # single-threaded workers (hundreds of mid-sized tensor ops lose
                    # time with intra-op threads, B128) and accumulate in slot order,
                    # so the result equals the 1-thread result at any thread count.
                    def _proj_slot(i_t):
                        B_c = slot_B[i_t]
                        B_f = _interp_slot(B_c, grid_sym, N_coarse, nfKnots, vecs_sym_c, vecs_sym_f)
                        if B_f is None:
                            return None
                        if zero_field:
                            # one position for all orientations: one point mass (exact zero-width facets)
                            B_f = _zf_const_pos(B_f, B_c)
                        I_f = _interp_slot(slot_I[i_t], grid_sym, N_coarse, nfKnots, vecs_sym_c, vecs_sym_f, mode='val')
                        if grid_sym == 'Dinfh':
                            return _projectzones(B_f, I_f, _seg_wts, x_np)
                        return _projecttriangles(_tri_f, _areas_f, B_f, I_f, x_np)
                    from torchspin._linalg import pool_map as _pool_map, single_threaded as _single_threaded
                    _n_fac = len(_seg_wts) if grid_sym == 'Dinfh' else int(np.asarray(_tri_f).shape[0])
                    if _n_fac >= _POOL_MIN_FACETS:
                        _contribs = _pool_map(_proj_slot, list(range(len(slot_B))))
                    else:
                        # small slots: Python overhead dominates and the GIL would
                        # serialize a pool anyway; one thread, no wake-ups
                        with _single_threaded():
                            _contribs = [_proj_slot(i_t) for i_t in range(len(slot_B))]
                    for i_t, contrib in enumerate(_contribs):
                        if contrib is None:
                            continue
                        I_c = slot_I[i_t]
                        _acc(i_t, torch.from_numpy(contrib * (2.0 * math.pi)), np.nanmax(np.abs(I_c)) if np.isfinite(I_c).any() else 0.0)
                        used_projection = True
                for i_t in range(0 if use_proj else len(slot_B)):
                    B_c = slot_B[i_t]
                    I_c = slot_I[i_t]
                    valid = ~np.isnan(B_c)
                    B_f = _interp_slot(B_c, grid_sym, N_coarse, nfKnots, vecs_sym_c, vecs_sym_f)
                    if B_f is None:
                        continue
                    if zero_field:
                        # one position for all orientations: one point mass (exact zero-width facets)
                        B_f = _zf_const_pos(B_f, B_c)
                    I_f = _interp_slot(I_c, grid_sym, N_coarse, nfKnots, vecs_sym_c, vecs_sym_f, mode='val')

                    # Fallback: per-orientation accumulation.
                    # Used for has_strain (spatial Gaussians) or grids without a
                    # standard triangulation (e.g. Ci symmetry → stick binning).
                    if has_strain and slot_W is not None:
                        W_c = slot_W[i_t]
                        W_f = _interp_slot(W_c, grid_sym, N_coarse, nfKnots, vecs_sym_c, vecs_sym_f, mode='val')
                    else:
                        W_f = np.zeros(n_orient_fine)

                    if not has_strain:
                        # Vectorized stick binning for non-projectable no-strain grids
                        # (e.g. Ci symmetry). Replaces O(n_fine) Python loop.
                        _vld = (~np.isnan(B_f)) & (~np.isnan(I_f)) & (I_f > 0.0)
                        if _vld.any():
                            _Bv = torch.from_numpy(B_f[_vld])
                            _Iv = torch.from_numpy(I_f[_vld]).to(spec.dtype)
                            _wv = weights_fine[_vld].to(spec.dtype)
                            _frac = (_Bv - exp.Range[0]) / dx
                            _ilo  = _frac.floor().long()
                            _ihi  = _ilo + 1
                            _a    = (_frac - _ilo.to(_frac.dtype)).to(spec.dtype)
                            _amp  = _wv * _Iv
                            _mlo  = (_ilo >= 0) & (_ilo < exp.nPoints)
                            _mhi  = (_ihi >= 0) & (_ihi < exp.nPoints)
                            if _mlo.any():
                                (_row(i_t) if sep_trans else spec).scatter_add_(0, _ilo[_mlo],
                                                  _amp[_mlo] * (1.0 - _a[_mlo]))
                            if _mhi.any():
                                (_row(i_t) if sep_trans else spec).scatter_add_(0, _ihi[_mhi],
                                                  _amp[_mhi] * _a[_mhi])
                    else:
                        for j in range(n_orient_fine):
                            Bj = B_f[j]
                            Ij = I_f[j]
                            Wj = W_f[j]
                            wj = weights_fine[j].item()

                            if np.isnan(Bj) or np.isnan(Ij) or Ij <= 0.0:
                                continue

                            fwhm_strain  = max(float(Wj), _min_fwhm)
                            if strain_lorentz:
                                _acc(i_t, wj * _lorentzian_bins(x, np.array([float(Bj)]), np.array([fwhm_strain]),
                                                                np.array([float(Ij)]), _min_fwhm, phase=mw_phase))
                                continue
                            fwhm_g_total = fwhm_strain  # global lw is convolved afterwards (EasySpin)
                            fwhm_l       = 0.0
                            if fwhm_l == 0.0:
                                if fwhm_g_total > 0.0:
                                    sigma    = fwhm_g_total / _sqrt_8log2
                                    peak_amp = float(Ij) * dx * _inv_sqrt_2pi / sigma
                                    spec     = spec + wj * peak_amp * torch.exp(
                                        -0.5 * ((x - Bj) / sigma) ** 2
                                    )
                                else:
                                    frac = (Bj - exp.Range[0]) / dx
                                    ilo = math.floor(frac)
                                    ihi = ilo + 1
                                    a = frac - ilo
                                    if 0 <= ilo < exp.nPoints:
                                        spec[ilo] = spec[ilo] + wj * float(Ij) * (1.0 - a)
                                    if 0 <= ihi < exp.nPoints:
                                        spec[ihi] = spec[ihi] + wj * float(Ij) * a
                            else:
                                Bt = torch.tensor([Bj], dtype=torch.float64)
                                It = torch.tensor([float(Ij)], dtype=torch.float64)
                                _, line_i = makespec(exp.Range, exp.nPoints, Bt, It)
                                line_i = convspec(line_i, dx,
                                                  fwhm_g=fwhm_g_total, fwhm_l=fwhm_l, deriv=0)
                                _acc(i_t, wj * line_i)

        else:
            # -----------------------------------------------------------
            # Direct path: no interpolation — accumulate at coarse grid.
            # For D2h/Dinfh without strain: two-phase collect → project.
            # For other grids or has_strain: existing per-orientation path.
            # -----------------------------------------------------------
            use_proj_d = (not has_strain) and grid_sym != 'O3'

            if use_proj_d:
                # Phase 1: collect B_res and intensities for every orientation
                all_B_d = [None] * n_orient
                all_I_d = [None] * n_orient
                all_P_d = [None] * n_orient
                _needs_exact_boltz_d = (sys.S[0] > 0.5 and sys.D is not None)
                if (sys.nElectrons == 1
                        and not has_strain
                        and not _needs_exact_boltz_d
                        and _pt_ok and str(opt.Method).startswith('perturb')):
                    all_B_d, all_I_d = resfields_perturb_batch(sys, phi_arr, theta_arr, exp_search, opt, order=pt_order, return_full=True, photo_weights=_pw(0, n_orient)
                    )
                else:
                    for batch_idx in range(n_batches):
                        s = batch_idx * batch_size
                        e = min(s + batch_size, n_orient)
                        B_list, I_list, W_list, P_list = _resonances(phi_arr[s:e], theta_arr[s:e], exp, opt, sys,
                            return_pairs=True, init_state=init_state, photo_weights=_pw(s, e), opt_override=_opt_rf, pairs=_pairs_sel
                        )
                        for j, (B_res, intens, _, prs) in enumerate(zip(B_list, I_list, W_list, P_list)):
                            all_B_d[s + j] = B_res
                            all_I_d[s + j] = intens
                            all_P_d[s + j] = prs
                    all_B_d, all_I_d, _, all_P_d = _fill_missing_knots(
                        (all_B_d, all_I_d, [None] * n_orient, all_P_d))

                # Phase 2: project per-transition onto the spectrum
                x_np_d = x.numpy()
                n_td_list = [b.numel() if b is not None else 0 for b in all_B_d]
                n_td = max(n_td_list) if n_td_list else 0

                _tri_c, _areas_c = _facets(grid_sym, N_coarse, theta_sym, ordering=_ordering)
                _seg_wts_d = _areas_c

                slot_B_d, slot_I_d, _ = _transition_slots(all_B_d, all_I_d, None, all_P_d, n_orient)
                slot_B_d, slot_I_d, _ = _filter_slots(slot_B_d, slot_I_d, None, _post_thr)
                for i_t in range(len(slot_B_d)):
                    B_t = slot_B_d[i_t]
                    I_t = slot_I_d[i_t]
                    if grid_sym == 'Dinfh':
                        contrib = _projectzones(B_t, I_t, _seg_wts_d, x_np_d)
                    else:
                        contrib = _projecttriangles(_tri_c, _areas_c, B_t, I_t, x_np_d)
                    _acc(i_t, torch.from_numpy(contrib * (2.0 * math.pi)), np.nanmax(np.abs(I_t)) if np.isfinite(I_t).any() else 0.0)

                used_projection = True

                if opt.Verbosity >= 1:
                    print(f"pepper: direct projection over {n_orient} orientations, {n_td} transitions")

            else:
                # Per-orientation accumulation (Ci/C2h/C1 grids, or has_strain).
                # Pre-collect (B_res, intens, widths, w_k) for all orientations.
                # For single-electron no-strain systems, use perturbation theory
                # (one vectorized call) instead of batched matrix eigh.
                _needs_exact_boltz_p = (sys.S[0] > 0.5 and sys.D is not None)
                _use_perturb_plain = (sys.nElectrons == 1
                                      and not has_strain
                                      and not _needs_exact_boltz_p
                                      and _pt_ok and str(opt.Method).startswith('perturb'))
                if _use_perturb_plain:
                    # Vectorized accumulation: all orientations × transitions in one pass.
                    # resfields_perturb_batch returns (M, n_slots) tensors (return_full=True)
                    # with NaN for out-of-range slots. We scatter all valid sticks at once.
                    _B_full, _I_full = resfields_perturb_batch(sys, phi_arr, theta_arr, exp_search, opt, order=pt_order, return_full=True, photo_weights=_pw(0, n_orient)
                    )  # each is list of (n_slots,) tensors, length M
                    _B_mat = torch.stack(_B_full, dim=0)   # (M, n_slots)
                    _I_mat = torch.stack(_I_full, dim=0)   # (M, n_slots)
                    _W_col = weights_arr.unsqueeze(1)      # (M, 1)
                    _WI    = (_W_col * _I_mat).reshape(-1) # (M * n_slots,)
                    if sep_trans:
                        for _k in range(_I_mat.shape[1]):
                            _Bk = _B_mat[:, _k]; _WIk = (_W_col[:, 0] * _I_mat[:, _k])
                            _vk = ~torch.isnan(_Bk) & (_WIk != 0)
                            if _vk.any():
                                # centered bins: linear split between the two nearest points (B85)
                                _r = _row(_k)
                                _r += _proj.hat_points_t(exp.nPoints, ((_Bk[_vk] - exp.Range[0]) / dx).cpu().to(torch.float64),
                                                         _WIk[_vk].cpu().to(torch.float64))
                    else:
                        _B_flat = _B_mat.reshape(-1)            # (M * n_slots,)
                        # Bin valid (non-NaN, in-range) transitions
                        _valid  = ~torch.isnan(_B_flat) & (_WI != 0)
                        if _valid.any():
                            _B_v = _B_flat[_valid]
                            _W_v = _WI[_valid]
                            # centered bins: linear split between the two nearest points (B85);
                            # out-of-window weights are dropped
                            spec = spec + _proj.hat_points_t(exp.nPoints, ((_B_v - exp.Range[0]) / dx).cpu().to(torch.float64),
                                                             _W_v.cpu().to(torch.float64))
                    used_projection = False  # will still apply normalization below
                else:
                    _all_orient = []
                    for batch_idx in range(n_batches):
                        start_idx = batch_idx * batch_size
                        end_idx   = min(start_idx + batch_size, n_orient)
                        phi_b     = phi_arr[start_idx:end_idx]
                        theta_b   = theta_arr[start_idx:end_idx]
                        weights_b = weights_arr[start_idx:end_idx]
                        B_res_list, intens_list, widths_list = _resonances(phi_b, theta_b, exp, opt, sys, init_state=init_state, photo_weights=_pw(start_idx, end_idx)
                        )
                        for k_in_batch, (B_res, intens, widths) in enumerate(
                            zip(B_res_list, intens_list, widths_list)
                        ):
                            _all_orient.append(
                                (B_res, intens, widths, weights_b[k_in_batch].item())
                            )

                _orient_seq = _all_orient if not _use_perturb_plain else []
                if sep_trans and _orient_seq:
                    raise NotImplementedError("Opt.separate='transitions' is not available on the per-orientation accumulation path.")

                if not has_strain and _orient_seq:
                    # Vectorized scatter_add: collect all (B_res, w*intens) pairs across
                    # all orientations in one pass, then bin in a single scatter operation.
                    # Replaces O(n_orient) per-orientation makespec calls.
                    _B_parts: list = []
                    _WI_parts: list = []
                    for B_res, intens, widths, w_k in _orient_seq:
                        if B_res is not None and B_res.numel() > 0:
                            _B_parts.append(B_res)
                            _WI_parts.append((w_k * intens).to(spec.dtype))
                    if _B_parts:
                        _B_cat  = torch.cat(_B_parts)
                        _WI_cat = torch.cat(_WI_parts)
                        _valid  = ~torch.isnan(_B_cat)
                        if _valid.any():
                            # centered bins: linear split between the two nearest points (B85)
                            spec = spec + _proj.hat_points_t(exp.nPoints, ((_B_cat[_valid] - exp.Range[0]) / dx).cpu().to(torch.float64),
                                                             _WI_cat[_valid].cpu().to(torch.float64))
                else:
                    for k_orient, (B_res, intens, widths, w_k) in enumerate(_orient_seq):
                        if B_res.numel() == 0:
                            continue

                        # has_strain: per-line spatial-domain Gaussian broadening
                        for i in range(B_res.numel()):
                            Bi = B_res[i].item()
                            Ai = intens[i].item()
                            fwhm_strain  = max(widths[i].item(), _min_fwhm)
                            if strain_lorentz:
                                spec = spec + w_k * _lorentzian_bins(x, np.array([Bi]), np.array([fwhm_strain]),
                                                                     np.array([Ai]), _min_fwhm, phase=mw_phase)
                                continue
                            fwhm_g_total = fwhm_strain  # global lw is convolved afterwards (EasySpin)
                            fwhm_l       = 0.0

                            if fwhm_l == 0.0:
                                if fwhm_g_total > 0.0:
                                    sigma    = fwhm_g_total / _sqrt_8log2
                                    peak_amp = Ai * dx * _inv_sqrt_2pi / sigma
                                    spec     = spec + w_k * peak_amp * torch.exp(
                                        -0.5 * ((x - Bi) / sigma) ** 2
                                    )
                                else:
                                    frac = (Bi - exp.Range[0]) / dx
                                    ilo = math.floor(frac)
                                    ihi = ilo + 1
                                    a = frac - ilo
                                    if 0 <= ilo < exp.nPoints:
                                        spec[ilo] = spec[ilo] + w_k * Ai * (1.0 - a)
                                    if 0 <= ihi < exp.nPoints:
                                        spec[ihi] = spec[ihi] + w_k * Ai * a
                            else:
                                _, line_i = makespec(
                                    exp.Range, exp.nPoints, B_res[i:i+1], intens[i:i+1],
                                )
                                line_i = convspec(
                                    line_i, dx, fwhm_g=fwhm_g_total, fwhm_l=fwhm_l, deriv=0
                                )
                                spec = spec + w_k * line_i

                        if opt.Verbosity >= 2:
                            print(f"  orient {k_orient+1}/{n_orient}: {B_res.numel()} transitions")

    if sep_trans:
        if not _rows:
            spec = spec.unsqueeze(0)
        else:
            keys = sorted(_rows.keys(), key=lambda kk: (kk if isinstance(kk, tuple) else (-1, kk)))
            if any(v > 0 for v in _row_max.values()):
                keys = sorted(keys, key=lambda k_: -_row_max[k_])   # EasySpin: transitions by descending max rate
            spec = torch.stack([_rows[k_] for k_ in keys])
    if not used_projection:
        # Non-projection paths accumulate ∑ w·I per bin (stick), or per-line
        # shapes whose sum over points equals w·I. EasySpin pepper.m converts
        # these to a spectral density (spec/deltaX) and multiplies by 2π for
        # the χ integral; the orientation weights already sum to 4π, so no
        # further normalization is applied. Together with the dBdE factor in
        # resfields this reproduces EasySpin's absolute intensity
        # (pepper_intensity_isopowder: ∫spec dB = 8π²·TransitionRate·dBdE).
        spec = spec / dx
        spec = spec * (2.0 * math.pi)

    # -----------------------------------------------------------------------
    # Step 5: Global broadening / derivative
    # -----------------------------------------------------------------------
    fwhm_g = lw_effective[0]  # Gaussian FWHM (mT) from sys.lw
    fwhm_l = lw_effective[1]  # Lorentzian FWHM (mT) from sys.lw
    # EasySpin harmonic bookkeeping: with field modulation (Exp.ModAmp) the
    # absorption spectrum is convolved and the harmonic comes from the pseudo-
    # modulation; otherwise the harmonic is produced by the convolution
    # (ConvHarmonic) or, without lw, by finite differences (DerivHarmonic).
    # Exp.mwPhase (dispersion admixture) enters through the line-shape kernel.
    spec = _finish_field_sweep(spec, x, dx, exp, fwhm_g, fwhm_l, has_strain, phase=_conv_phase,
                              pad_multiplier=opt.PaddingMultiplier)
    return x, spec
