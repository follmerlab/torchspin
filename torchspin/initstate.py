"""Non-equilibrium initial states (port of EasySpin ``Sys.initState`` handling).

``SpinSystem.initState`` may be

* ``(pops, basis)`` with a population vector and a basis name
  ``'zerofield'``, ``'eigen'``, ``'xyz'``, ``'coupled'`` or ``'uncoupled'``;
* ``(rho, basis)`` with a density matrix in that basis;
* a density matrix (uncoupled product basis, electron part or full);
* ``'T0'`` (triplet, ``[0 1 0]`` in the eigenbasis) or ``'singlet'`` (two
  identical electron spins).

:func:`parse_init_state` reproduces EasySpin's ``validatespinsys`` conversion
(coupled → uncoupled via ``cgmatrix``, xyz → zero-field triplet states rotated
by ``DFrame``), :func:`init_state_density` the ``resfields``/``resfreqs``
expansion to the full state space (nuclear sublevels equally populated) and the
zero-field-basis transformation with the eigenvectors of the zero-field
Hamiltonian.  The result is used as a polarization ``<u|ρ|u> − <v|ρ|v>`` for
each transition (``'eigen'``: diagonal populations by level index).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .angmom import cgmatrix, wignerd

__all__ = ['parse_init_state', 'init_state_density']

_BASES = ('uncoupled', 'coupled', 'eigen', 'zerofield', 'xyz')


def _as_array(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def parse_init_state(sys) -> Optional[tuple[np.ndarray, str]]:
    """Normalise ``sys.initState`` to ``(matrix_or_vector, basis)`` with basis
    ``'uncoupled'``, ``'eigen'`` or ``'zerofield'`` (EasySpin validatespinsys)."""
    st = getattr(sys, 'initState', None)
    if st is None:
        return None
    S = [float(s) for s in sys.S]
    n_e = len(S)
    if isinstance(st, str):
        if st == 'singlet':
            if n_e != 2 or S[0] != S[1]:
                raise ValueError("initState='singlet' is only available for systems of two identical electron spins.")
            U2C, _, _ = cgmatrix(S[0], S[1], 0)
            Svec = np.asarray(U2C).conj().T          # column: singlet in uncoupled basis
            return (Svec @ Svec.conj().T, 'uncoupled')
        if st == 'T0':
            if n_e != 1 or S[0] != 1:
                raise ValueError("initState='T0' is only available for triplet states (S = 1).")
            return (np.diag([0.0, 1.0, 0.0]), 'eigen')
        raise ValueError("String input for initial state not supported: %r" % st)
    if isinstance(st, (tuple, list)) and len(st) == 2 and isinstance(st[1], str):
        state = _as_array(st[0]).astype(complex)
        basis = st[1]
        if basis not in _BASES:
            raise ValueError("initState basis must be 'zerofield', 'xyz', 'eigen', 'coupled' or 'uncoupled'.")
        is_vec = state.ndim == 1 or 1 in state.shape
        if not is_vec and state.shape[0] != state.shape[1]:
            raise ValueError('initState must contain a density matrix or a population vector.')
        if basis == 'coupled':
            if n_e != 2:
                raise ValueError('initState in the coupled basis is only available for two electron spins.')
            U2C, _, _ = cgmatrix(S[0], S[1])
            C2U = np.asarray(U2C).conj().T             # coupled → uncoupled
            n_es = C2U.shape[0]
            if max(state.shape) != n_es:
                n_states = int(np.prod([2 * s + 1 for s in S] + [2 * I + 1 for I in sys.I]))
                C2U = np.kron(C2U, np.eye(n_states // n_es))
            rho = np.diag(state.reshape(-1)) if is_vec else state
            return (C2U @ rho @ C2U.conj().T, 'uncoupled')
        if basis == 'xyz':
            if n_e != 1 or S[0] != 1:
                raise ValueError("initState with 'xyz' basis is only allowed for triplet states (S = 1).")
            if state.size != 3:
                raise ValueError("initState with 'xyz' basis requires three populations [px py pz].")
            Tx = np.array([1, 0, -1]) / np.sqrt(2)
            Ty = np.array([1, 0, 1]) / np.sqrt(2)
            Tz = np.array([0, 1, 0])
            ZF = np.stack([Tx, Ty, Tz], axis=1).astype(complex)
            DF = getattr(sys, 'DFrame', None)
            if DF is not None:
                df = _as_array(DF).reshape(-1)[:3]
                if np.any(df):
                    D = np.asarray(wignerd(1, float(df[0]), float(df[1]), float(df[2])))
                    ZF = D @ ZF
            return (ZF @ np.diag(state.reshape(-1)) @ ZF.conj().T, 'uncoupled')
        if basis == 'eigen':
            return (np.diag(state.reshape(-1)) if is_vec else state, 'eigen')
        if basis == 'zerofield':
            return (state.reshape(-1) if is_vec else state, 'zerofield')
        return (state.reshape(-1) if is_vec else state, 'uncoupled')
    state = _as_array(st).astype(complex)
    if state.ndim != 2 or state.shape[0] != state.shape[1]:
        raise ValueError("initState given as a population vector requires a basis: (popvec, 'basis').")
    return (state, 'uncoupled')


def init_state_density(sys, H0: Optional[torch.Tensor] = None) -> Optional[tuple[torch.Tensor, str]]:
    """Density matrix in the full uncoupled state space (EasySpin resfields.m).

    Returns ``(rho, basis)`` with ``rho`` an (n, n) complex tensor and basis
    ``'uncoupled'`` (use ``<u|rho|u>``) or ``'eigen'`` (use ``rho[u,u]``).
    ``H0`` (zero-field Hamiltonian, MHz) is required for the ``'zerofield'`` basis.
    """
    parsed = parse_init_state(sys)
    if parsed is None:
        return None
    state, basis = parsed
    n_es = int(np.prod([2 * float(s) + 1 for s in sys.S]))
    n_core = int(n_es * np.prod([2 * float(I) + 1 for I in sys.I])) if sys.nNuclei > 0 else n_es
    if state.ndim == 2 and state.shape[0] == state.shape[1] and state.ndim == 2 and min(state.shape) > 1 or (state.ndim == 2 and state.shape == (1, 1)):
        if state.shape[0] not in (n_es, n_core):
            raise ValueError(f'The density matrix in initState must be {n_es}x{n_es} or {n_core}x{n_core}.')
        if state.shape[0] == n_es and n_core > n_es:
            r = n_core // n_es
            state = np.kron(state, np.eye(r)) / r
        rho = state
    else:
        vec = state.reshape(-1)
        if vec.size not in (n_es, n_core):
            raise ValueError(f'The population vector in initState must have {n_es} or {n_core} elements.')
        if vec.size == n_es and n_core > n_es:
            r = n_core // n_es
            vec = np.kron(vec, np.ones(r)) / r
        rho = np.diag(vec)
    if basis == 'zerofield':
        if H0 is None:
            raise ValueError("init_state_density: H0 is required for the 'zerofield' basis.")
        H = H0.detach().cpu().numpy() if isinstance(H0, torch.Tensor) else np.asarray(H0)
        if H.shape[0] != rho.shape[0]:
            raise ValueError('initState/H0 dimension mismatch.')
        E, V = np.linalg.eigh(H)
        idx = np.argsort(E.real)
        E, V = E[idx].real, V[:, idx]
        # Degenerate zero-field levels make the assignment of populations ambiguous
        # (EasySpin errors on exactly equal eigenvalues).  Degenerate groups with
        # equal populations (e.g. Kramers doublets from equally populated nuclear
        # sublevels) are unambiguous and accepted.
        scale = max(1.0, float(np.abs(E).max()))
        groups = np.cumsum(np.concatenate([[True], np.diff(E) > 1e-9 * scale]))
        pops_diag = np.real(np.diag(rho))
        for g in np.unique(groups):
            sel = groups == g
            if sel.sum() > 1 and (np.ptp(pops_diag[sel]) > 1e-10 or
                                  np.abs(rho[np.ix_(sel, sel)] - np.diag(np.diag(rho[np.ix_(sel, sel)]))).max() > 1e-10):
                raise ValueError('Degenerate energy levels detected at zero field; provide the non-equilibrium '
                                 'state as a full density matrix instead of zero-field populations.')
        rho = V @ rho @ V.conj().T
        basis = 'uncoupled'
    return (torch.tensor(np.asarray(rho, dtype=complex), dtype=torch.complex128), basis)
