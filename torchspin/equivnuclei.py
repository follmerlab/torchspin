"""Sets of equivalent nuclei in exact (matrix) simulations.

n equivalent nuclei of spin I (same isotope, same hyperfine tensor and frame,
same chemical shift, no quadrupole and no nucleus-nucleus coupling) enter the
spin Hamiltonian only through their total spin K = I_1 + ... + I_n, so the
Hamiltonian is block diagonal in K.  The spectrum is the sum of the spectra of
systems in which the set is replaced by one nucleus of spin K, weighted with
the multiplicity of K (:func:`torchspin.utils.equivcouple`) times
(2K + 1)/(2I + 1)^n, the fraction of the nuclear states in the block.

Three 14N are blocks K = 3, 2 (twice), 1 (three times) and 0: Hilbert-space
dimensions 7, 5, 3 and 1 times that of the rest, in place of 27.  Besides the
cost, the full space has exactly degenerate levels (the copies of a K), in
which eigenvectors and therefore single-transition intensities are arbitrary;
the blocks do not (B139, B135).
"""
from __future__ import annotations

import itertools

import torch

from torchspin.spinsystem import SpinSystem, _filter_nuclei, _nonzero, _nuc_rows, _replace
from torchspin.utils import equivcouple

__all__ = ['equivalent_sets', 'reduce_equivalent_nuclei']

_PER_NUCLEUS = ('A', 'AFrame', 'gnscale', 'sigma', 'sigmaFrame')


def _row(sys: SpinSystem, name: str, i: int):
    v = getattr(sys, name)
    if v is None:
        return None
    return _nuc_rows(torch.as_tensor(v), [i], sys.nNuclei, name)


def _same(sys: SpinSystem, i: int, j: int) -> bool:
    if sys.Nucs[i] != sys.Nucs[j] or sys.I[i] != sys.I[j]:
        return False
    for name in _PER_NUCLEUS:
        a, b = _row(sys, name, i), _row(sys, name, j)
        if (a is None) != (b is None) or (a is not None and not torch.equal(a, b)):
            return False
    return True


def _groupable(sys: SpinSystem, i: int) -> bool:
    """A nucleus that can be part of a total-spin set: no quadrupole coupling
    (it does not commute with K^2), and not the nucleus that carries AStrain."""
    if _nonzero(_row(sys, 'Q', i)):
        return False
    if i == 0 and _nonzero(sys.AStrain):
        return False
    return True


def equivalent_sets(sys: SpinSystem) -> list[tuple[list[int], int]]:
    """Sets of equivalent nuclei of *sys*: ``(indices into Nucs, number of nuclei)``.

    The number counts ``SpinSystem.n``; only sets of two or more nuclei are
    returned.  Empty when the system cannot be reduced (``nn``, ``H0Extra`` or
    a non-equilibrium ``initState`` given as states of the full spin space).
    """
    n_n = sys.nNuclei
    if n_n == 0 or _nonzero(sys.nn) or sys.H0Extra:
        return []
    if sys.initState is not None and not isinstance(sys.initState, str):
        return []
    sets, used = [], set()
    for i in range(n_n):
        if i in used or not _groupable(sys, i):
            continue
        idx = [i] + [j for j in range(i + 1, n_n) if j not in used and _groupable(sys, j) and _same(sys, i, j)]
        used.update(idx)
        count = sum(int(sys.n[k]) for k in idx)
        if count > 1:
            sets.append((idx, count))
    return sets


def reduce_equivalent_nuclei(sys: SpinSystem) -> list[tuple[float, SpinSystem]]:
    """Total-spin blocks of *sys*: ``(weight, system)`` pairs whose weighted
    spectra sum to the spectrum of *sys*.  ``[(1.0, sys)]`` when *sys* has no
    set of equivalent nuclei.  Every block keeps ``sys.weight``."""
    sets = equivalent_sets(sys)
    if not sets:
        return [(1.0, sys)]
    n_n = sys.nNuclei
    per_set = []
    for idx, count in sets:
        I = sys.I[idx[0]]
        K, mult = equivcouple(I, count)
        per_set.append([(float(k), int(m) * (2 * float(k) + 1) / (2 * I + 1) ** count) for k, m in zip(K, mult)])
    out = []
    for combo in itertools.product(*per_set):
        weight, drop = 1.0, set()
        Ieff = list(sys._Ieff) if sys._Ieff is not None else [None] * n_n
        n = list(sys.n)
        for (idx, _), (k, w) in zip(sets, combo):
            weight *= w
            drop.update(idx[1:] if k > 0 else idx)
            Ieff[idx[0]] = k
            n[idx[0]] = 1
        keep = [i for i in range(n_n) if i not in drop]
        marked = _replace(sys, _Ieff=Ieff, n=n)
        out.append((weight, _filter_nuclei(marked, keep, 'reduce_equivalent_nuclei')))
    return out
