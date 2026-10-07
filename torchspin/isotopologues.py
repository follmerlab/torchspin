"""Isotopologue expansion (port of EasySpin ``isotopologues.m`` / ``compisoloop.m``).

A nucleus in ``SpinSystem.Nucs`` may be given as

* an isotope (``'63Cu'``): a single, fully specified nucleus;
* an element (``'Cu'``): the natural-abundance mixture of all its isotopes;
* a custom mixture (``'(63,65)Cu'``): the listed isotopes with abundances
  given in ``SpinSystem.Abund`` (one array per nucleus, or a flat array for
  a single nucleus).

:func:`isotopologues` expands such a system into the list of spin systems
with definite isotopes, each carrying ``weight = Sys.weight * abundance``.
Following EasySpin,

* hyperfine tensors are scaled by the ratio of nuclear g-factors relative to
  the most abundant magnetic isotope of the element (``A_k = A/gn_ref*gn_k``),
  quadrupole tensors by the ratio of quadrupole moments (``Q_k = Q/qm_ref*qm_k``);
* non-magnetic isotopes (I = 0) are consolidated into one entry and dropped
  from ``Nucs`` of the resulting isotopologue;
* groups of ``n`` equivalent nuclei are expanded into multisets with the
  appropriate combinatorial multiplicity (``multisetlist``);
* isotopologues with an abundance below ``rel_threshold`` times the largest
  abundance are discarded (EasySpin ``Opt.IsoCutoff``, default 1e-4).
"""
from __future__ import annotations

import dataclasses
import math
import re
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from .nucdata import _DB, nucdata

if TYPE_CHECKING:  # pragma: no cover
    from .spinsystem import SpinSystem

__all__ = ['isotopologues', 'is_isotope_mixture', 'multisetlist',
           'expand_components', 'DEFAULT_ISO_CUTOFF']

DEFAULT_ISO_CUTOFF = 1e-4

_ISOTOPE_RE = re.compile(r'^(\d+)([A-Za-z]+)$')
_CUSTOM_RE = re.compile(r'^\(([\d,\s]+)\)([A-Za-z]+)$')


def is_isotope_mixture(nuc: str) -> bool:
    """True if *nuc* denotes an isotope mixture ('Cu', '(63,65)Cu') rather than
    a single isotope ('63Cu')."""
    nuc = nuc.strip()
    return not _ISOTOPE_RE.match(nuc)


def multisetlist(n: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Multisets of *n* equivalent positions filled with *k* isotopes.

    Returns ``(kvec, multiplicity)``: ``kvec`` has one row per group of
    spectrally indistinguishable isotopologues (how many positions carry each
    isotope), ``multiplicity`` the number of permutations giving that group
    (multinomial coefficient).  Row order follows EasySpin's ``multisetlist``.
    """
    def compositions(total, parts):
        if parts == 1:
            yield (total,)
            return
        for first in range(total, -1, -1):
            for rest in compositions(total - first, parts - 1):
                yield (first,) + rest

    kvec = np.array(list(compositions(n, k)), dtype=int).reshape(-1, k)
    mult = np.array([math.factorial(n) // math.prod(math.factorial(int(c)) for c in row)
                     for row in kvec], dtype=float)
    return kvec, mult


def _element_isotopes(element: str) -> tuple[list[int], list[float]]:
    """Mass numbers and natural abundances (fraction) of all isotopes of *element*
    with nonzero natural abundance, in database order."""
    _DB.load()
    mass, abund = [], []
    for el, nuc, ab in zip(_DB.elements, _DB.nucleons, _DB.abundances):
        if el == element and ab > 0:
            mass.append(int(nuc))
            abund.append(float(ab) / 100.0)
    if not mass:
        raise ValueError(f"Could not find element '{element}'")
    return mass, abund


def _parse_nucleus(nuc: str, abund_given) -> tuple[str, list[int], list[float]]:
    """(element, mass numbers, abundances) for one Nucs entry."""
    nuc = nuc.strip()
    m = _CUSTOM_RE.match(nuc)
    if m:
        element = m.group(2)
        mass = [int(x) for x in m.group(1).replace(' ', '').split(',') if x]
        if abund_given is None:
            raise ValueError('For custom isotope mixtures, abundances must be given (Sys.Abund).')
        abund = [float(a) for a in np.asarray(abund_given, dtype=float).reshape(-1)]
        if len(abund) != len(mass):
            raise ValueError(f"Nucleus '{nuc}': {len(mass)} isotopes but {len(abund)} abundances given.")
        return element, mass, abund
    m = _ISOTOPE_RE.match(nuc)
    if m:
        return m.group(2), [int(m.group(1))], [1.0]
    # element name → natural abundance mixture
    if abund_given is not None and np.size(abund_given) > 0:
        raise ValueError(f"Nucleus '{nuc}' is a natural-abundance mixture; provide None in Abund for it.")
    mass, abund = _element_isotopes(nuc)
    return nuc, mass, abund


def _abundance_tree(n_iso: list[int], abund: list[np.ndarray], abs_threshold: float):
    """Enumerate isotope-index combinations with product abundance above the
    threshold (EasySpin ``abundancetreetraversal`` order)."""
    n_nucs = len(abund)
    i_iso = [0] * n_nucs
    cum = [0.0] * n_nucs
    idx_list, ab_list = [], []
    i_nuc = 0
    while i_nuc >= 0:
        if i_nuc > n_nucs - 1:
            if cum[n_nucs - 1] > abs_threshold:
                idx_list.append([i - 1 for i in i_iso])
                ab_list.append(cum[n_nucs - 1])
            i_nuc -= 1
        else:
            if i_iso[i_nuc] < n_iso[i_nuc]:
                i_iso[i_nuc] += 1
                prev = cum[i_nuc - 1] if i_nuc > 0 else 1.0
                cum[i_nuc] = prev * abund[i_nuc][i_iso[i_nuc] - 1]
                i_nuc += 1
            else:
                i_iso[i_nuc] = 0
                i_nuc -= 1
    return idx_list, ab_list


def _cat(blocks: list[torch.Tensor], like: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if like is None:
        return None
    if not blocks:
        return torch.zeros((0,) + tuple(like.shape[1:]), dtype=like.dtype)
    return torch.cat(blocks, dim=0)


def isotopologues(sys, n=None, abund=None, rel_threshold: float = DEFAULT_ISO_CUTOFF) -> list['SpinSystem']:
    """Expand *sys* into its isotopologues (list of SpinSystem with weights).

    *sys* may also be a nucleus string (``'Cu,Cu'``, ``'(63,65)Cu,1H'``) with
    optional equivalent-nucleus counts *n* and custom abundances *abund*
    (one array per custom mixture, ``None`` for natural ones), as in
    ``isotopologues(NucList, n, Abundances, threshold)``; the result is a list
    of minimal S=1/2 spin systems carrying ``Nucs``, ``n`` and ``weight``.

    A system without isotope mixtures is returned unchanged in a one-element
    list (weight untouched).  Abundances multiply ``sys.weight``.
    """
    if isinstance(sys, str):
        from .spinsystem import SpinSystem
        nuc_list = [x.strip() for x in re.split(r',(?![^()]*\))', sys) if x.strip()]
        n_list = None if (n is None or not nuc_list) else [int(v) for v in np.atleast_1d(np.asarray(n)).reshape(-1)]
        if abund is not None and len(nuc_list) == 1 and not (isinstance(abund, (list, tuple)) and len(abund) > 0
                                                            and isinstance(abund[0], (list, tuple, np.ndarray))):
            abund = [abund]
        sys = SpinSystem(S=[0.5], Nucs=nuc_list, n=n_list, Abund=None if abund is None else list(abund))
    if rel_threshold is None:
        rel_threshold = DEFAULT_ISO_CUTOFF
    nucs = list(sys.Nucs)
    n_n = len(nucs)
    if n_n == 0 or not any(is_isotope_mixture(nc) for nc in nucs):
        return [sys]
    if rel_threshold is None:
        rel_threshold = DEFAULT_ISO_CUTOFF

    # abundances per nucleus (custom mixtures)
    abund_in = getattr(sys, 'Abund', None)
    if abund_in is None:
        abund_list = [None] * n_n
    elif n_n == 1 and not (isinstance(abund_in, (list, tuple)) and len(abund_in) > 0
                           and isinstance(abund_in[0], (list, tuple, np.ndarray, torch.Tensor))):
        abund_list = [abund_in]
    else:
        if len(abund_in) != n_n:
            raise ValueError('Abund must have one entry (array or None) per nucleus.')
        abund_list = list(abund_in)

    n_equiv = list(sys.n) if getattr(sys, 'n', None) is not None else [1] * n_n
    n_e = sys.nElectrons
    A, AFrame, Q, QFrame = sys.A, sys.AFrame, sys.Q, sys.QFrame
    gnscale = sys.gnscale
    fullA = A is not None and A.shape[0] == 3 * n_n and (n_n != 1 or A.shape[1] == 3 * n_e and A.ndim == 2 and A.shape[0] == 3 and n_n == 1 and _is_full_single(A, n_e))
    fullQ = Q is not None and Q.shape[0] == 3 * n_n and (n_n != 1 or Q.shape == (3, 3) and getattr(sys, 'fullQ', False))

    def block(T, i, full):
        if T is None:
            return None
        return T[3 * i:3 * i + 3] if full else T[i:i + 1]

    # --- per-nucleus isotope groups ---------------------------------------
    groups = []
    for i_nuc, nc in enumerate(nucs):
        element, mass, ab = _parse_nucleus(nc, abund_list[i_nuc])
        isotopes = [f'{m}{element}' for m in mass]
        I, gn, qm, _ = nucdata(isotopes)
        I = np.atleast_1d(np.asarray(I, dtype=float))
        gn = np.atleast_1d(np.asarray(gn, dtype=float))
        qm = np.atleast_1d(np.asarray(qm, dtype=float))
        ab = np.asarray(ab, dtype=float)
        # consolidate non-magnetic isotopes into one entry
        zero = np.where(I == 0)[0]
        if zero.size > 0:
            ab[zero[0]] = ab[zero].sum()
            keep = np.array([k for k in range(len(I)) if k not in set(zero[1:].tolist())])
            isotopes = [isotopes[k] for k in keep]
            I, gn, qm, ab = I[keep], gn[keep], qm[keep], ab[keep]
        # reference isotopes for A and Q scaling
        ab_ = ab.copy(); ab_[I < 0.5] = -1
        k = int(np.argmax(ab_)); gn_ref = gn[k] if ab_[k] > 0 else 1.0
        if not gn_ref:
            gn_ref = 1.0
        ab_ = ab.copy(); ab_[I < 1] = -1
        k = int(np.argmax(ab_)); qm_ref = qm[k] if ab_[k] > 0 else 1.0
        if not qm_ref or np.isnan(qm_ref):
            qm_ref = 1.0
        qm = np.where(np.isnan(qm), 0.0, qm)
        A_blk = block(A, i_nuc, fullA)
        Q_blk = block(Q, i_nuc, fullQ)
        A_k = [None if A_blk is None else A_blk / gn_ref * float(g) for g in gn]
        Q_k = [None if Q_blk is None else Q_blk / qm_ref * float(q) for q in qm]
        # multisets for equivalent nuclei
        n = int(n_equiv[i_nuc])
        kvec, mult = multisetlist(n, len(isotopes))
        g_abund = np.array([np.prod(ab ** row) for row in kvec]) * mult
        if np.any(g_abund == 0) and n > 1 and np.all(ab > 0):
            raise ValueError(f'Abundance underflow: cannot handle n={n} for nucleus #{i_nuc + 1}.')
        groups.append(dict(isotopes=isotopes, I=I, A=A_k, Q=Q_k, kvec=kvec, abund=g_abund,
                           AFrame=block(AFrame, i_nuc, False), QFrame=block(QFrame, i_nuc, False),
                           gnscale=None if gnscale is None else gnscale[i_nuc:i_nuc + 1]))

    max_abund = float(np.prod([g['abund'].max() for g in groups]))
    abs_threshold = rel_threshold * max_abund
    idx_list, ab_list = _abundance_tree([len(g['abund']) for g in groups],
                                        [g['abund'] for g in groups], abs_threshold)

    # --- assemble isotopologue spin systems --------------------------------
    result = []
    base_weight = float(getattr(sys, 'weight', 1.0))
    for idx, ab_tot in zip(idx_list, ab_list):
        nucs_, n_, A_, AF_, Q_, QF_, gs_ = [], [], [], [], [], [], []
        for g, im in zip(groups, idx):
            kv = g['kvec'][im]
            for i_iso in np.where(kv != 0)[0]:
                if g['I'][i_iso] == 0:
                    continue
                nucs_.append(g['isotopes'][i_iso])
                n_.append(int(kv[i_iso]))
                if g['A'][i_iso] is not None:
                    A_.append(g['A'][i_iso])
                if g['Q'][i_iso] is not None:
                    Q_.append(g['Q'][i_iso])
                if g['AFrame'] is not None:
                    AF_.append(g['AFrame'])
                if g['QFrame'] is not None:
                    QF_.append(g['QFrame'])
                if g['gnscale'] is not None:
                    gs_.append(g['gnscale'])
        kw = dict(Nucs=nucs_, n=n_ if nucs_ else None, Abund=None, A_=None,   # A_ already folded into A
                  weight=base_weight * float(ab_tot))
        if nucs_:
            kw['A'] = _cat(A_, A)
            kw['AFrame'] = _cat(AF_, AFrame)
            kw['Q'] = _cat(Q_, Q)
            kw['QFrame'] = _cat(QF_, QFrame)
            kw['gnscale'] = _cat(gs_, gnscale)
        else:
            kw.update(A=None, AFrame=None, Q=None, QFrame=None, gnscale=None)
        result.append(dataclasses.replace(sys, **kw))
    return result


def _is_full_single(A: torch.Tensor, n_e: int) -> bool:
    # A of a single nucleus stored as (3, 3*n_e) is a full tensor; (1, 3*n_e) principal values
    return A.ndim == 2 and A.shape[0] == 3


def expand_components(sys_or_list, rel_threshold: float = DEFAULT_ISO_CUTOFF) -> list['SpinSystem']:
    """Flatten a spin system or list of components into the list of all
    isotopologues of all components (EasySpin ``compisoloop`` order)."""
    comps = sys_or_list if isinstance(sys_or_list, (list, tuple)) else [sys_or_list]
    out = []
    for c in comps:
        # Keyword, not positional: the second parameter of isotopologues() is n.
        out.extend(isotopologues(c, rel_threshold=rel_threshold))
    return out
