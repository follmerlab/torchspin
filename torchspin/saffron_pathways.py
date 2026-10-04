"""saffron_pathways — Coherence transfer pathway enumeration for pulse EPR.

Port of EasySpin's ``sf_pathways.m``. Determines which coherence transfer
pathways (CTPs) lead to refocusing echoes for a given pulse sequence and
incrementation scheme.

Pathway encoding:
    1 = alpha (electron ms manifold, coherence order 0)
    2 = beta  (electron ms manifold, coherence order 0)
    3 = +1 electron coherence
    4 = -1 electron coherence
"""

import numpy as np
from itertools import product as iterproduct


def find_refocusing_pathways(t: list[float], inc: list[int]) -> np.ndarray:
    """Determine refocusing coherence transfer pathways.

    Parameters
    ----------
    t : list of float
        Initial inter-pulse delays (microseconds). Length = number of
        free-evolution intervals.
    inc : list of int
        Incrementation scheme. Same length as *t*.
        0 = constant delay, +1/+2/... = incremented along dimension 1/2/...,
        -1/-2/... = decremented along dimension 1/2/...

    Returns
    -------
    np.ndarray
        Array of shape ``(nPathways, nIntervals)`` with values 1--4.
        Each row is a refocusing CTP.
    """
    t = np.asarray(t, dtype=np.float64)
    inc = np.asarray(inc, dtype=np.int64)
    n_intervals = len(t)

    if len(inc) != n_intervals:
        raise ValueError("t and inc must have the same number of elements.")

    # --- Single pulse (FID): no echo, just -1 coherence ---
    if n_intervals <= 1:
        return np.array([[4]], dtype=np.int64)

    # --- Enumerate all CTPs ---
    # First N-1 intervals: each can be 1,2,3,4; last interval forced to 4 (-1)
    N = n_intervals - 1
    grids = [np.arange(1, 5)] * N
    combos = np.array(list(iterproduct(*grids)), dtype=np.int64)  # (4^N, N)
    pathways = np.empty((combos.shape[0], n_intervals), dtype=np.int64)
    pathways[:, :N] = combos
    pathways[:, N] = 4  # last interval is always -1 coherence (detection)

    # --- Electron coherence order per interval ---
    # alpha=0, beta=0, +=+1, -=-1
    co_map = np.array([0, 0, 1, -1], dtype=np.float64)  # index 0..3 → order
    co = co_map[pathways - 1]  # (nPathways, nIntervals)

    # --- Check for at least one coherence flip (|delta_p| = 2) ---
    # Remove zero-order intervals for flip counting
    has_flip = np.zeros(len(pathways), dtype=bool)
    for p_idx in range(len(pathways)):
        co_nonzero = co[p_idx][co[p_idx] != 0]
        if len(co_nonzero) >= 2:
            has_flip[p_idx] = np.any(np.abs(np.diff(co_nonzero)) == 2)
        elif len(co_nonzero) == 1:
            # Single nonzero: can't have a flip
            has_flip[p_idx] = False

    # --- Echo refocusing condition ---
    # t0 = coherence_order . t  (total phase from initial delays)
    t0 = co @ t  # (nPathways,)

    max_dim = int(np.max(np.abs(inc))) if np.any(inc != 0) else 0

    if max_dim > 0:
        # For each sweep dimension, compute walk speed
        incdec = np.zeros((len(pathways), max_dim), dtype=np.float64)
        for d in range(1, max_dim + 1):
            mask = np.abs(inc) == d
            incdec[:, d - 1] = co[:, mask].sum(axis=1)

        # Refocusing: flip exists, t0==0, and all walk speeds == 0
        refocusing = has_flip & (t0 == 0) & np.all(incdec == 0, axis=1)
    else:
        # No incremented delays: refocusing just means flip + t0==0
        refocusing = has_flip & (t0 == 0)

    return pathways[refocusing]
