"""Microwave excitation geometry (port of EasySpin ``p_excitationgeometry``).

``Experiment.mwMode`` is ``'perpendicular'`` (B1 ⊥ B0, along xL),
``'parallel'`` (B1 ∥ B0), or a pair ``(k, mode2)`` with the propagation
direction ``k`` (letter, 3-vector, ``[phi, theta]`` or a single polar angle)
and ``mode2`` either the polarisation angle alpha (linear polarisation) or one
of ``'unpolarized'``, ``'circular+'``, ``'circular-'``.  The returned geometry
holds the lab-frame unit vectors ``nB1`` and ``nk``, their projections
``xi1 = nB1·nB0`` and ``xik = nk·nB0`` on the static field, and the mode kind.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .rotations import erot

__all__ = ['ExcitationGeometry', 'excitation_geometry']


@dataclass
class ExcitationGeometry:
    kind: str            # 'linear', 'unpolarized', 'circular'
    sense: float         # ±1 for circular polarisation, nan otherwise
    parallel: bool
    nB1: np.ndarray      # lab-frame B1 direction (linear polarisation)
    nk: np.ndarray       # lab-frame propagation direction
    xi1: float
    xik: float

    @property
    def is_default_perpendicular(self) -> bool:
        return self.kind == 'linear' and abs(self.xi1) < 1e-12


_LETTERS = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}


def _vec2ang(v):
    v = np.asarray(v, dtype=float).reshape(-1)
    v = v / np.linalg.norm(v)
    return float(np.arctan2(v[1], v[0])), float(np.arccos(np.clip(v[2], -1, 1)))


def _k_angles(k):
    if isinstance(k, str):
        s = k.strip()
        sign = -1.0 if s.startswith('-') else 1.0
        s = s.lstrip('+-')
        if s not in _LETTERS:
            raise ValueError(f"Unknown direction letter '{k}' in Exp.mwMode.")
        return _vec2ang(sign * np.asarray(_LETTERS[s], dtype=float))
    arr = np.asarray(k, dtype=float).reshape(-1)
    if arr.size == 3:
        return _vec2ang(arr)
    if arr.size == 2:
        return float(arr[0]), float(arr[1])
    if arr.size == 1:
        return 0.0, float(arr[0])
    raise ValueError('k (first element of Exp.mwMode) must be a letter, a 3-vector, two angles or one polar angle.')


def excitation_geometry(mw_mode) -> ExcitationGeometry:
    """Parse ``Experiment.mwMode`` (EasySpin p_excitationgeometry)."""
    if mw_mode is None or (isinstance(mw_mode, str) and mw_mode == ''):
        mw_mode = 'perpendicular'
    kind, sense, parallel = 'linear', float('nan'), False
    if isinstance(mw_mode, str):
        if mw_mode == 'perpendicular':
            k, alpha = 'y', -np.pi / 2       # B1 along xL
        elif mw_mode == 'parallel':
            k, alpha = 'y', np.pi            # B1 along zL
            parallel = True
        else:
            raise ValueError(f"Unrecognized Exp.mwMode: '{mw_mode}'.")
    elif isinstance(mw_mode, (list, tuple)) and len(mw_mode) == 2:
        k, mode2 = mw_mode
        if isinstance(mode2, str):
            if mode2 == 'unpolarized':
                kind = 'unpolarized'
            elif mode2 == 'circular+':
                kind, sense = 'circular', 1.0
            elif mode2 == 'circular-':
                kind, sense = 'circular', -1.0
            else:
                raise ValueError(f"Unrecognized 2nd element in Exp.mwMode: '{mode2}'")
            alpha = 0.0
        else:
            alpha = float(mode2)
    else:
        raise ValueError("Exp.mwMode must be 'perpendicular' or 'parallel' or a 2-element (k, mode) pair.")
    phi_k, theta_k = _k_angles(k)
    R = erot([phi_k, theta_k, float(alpha)])
    R = R.detach().cpu().numpy() if hasattr(R, 'detach') else np.asarray(R)
    nB1, nk = R[0, :].copy(), R[2, :].copy()
    return ExcitationGeometry(kind=kind, sense=sense, parallel=parallel, nB1=nB1, nk=nk,
                              xi1=float(nB1[2]), xik=float(nk[2]))


def exp_mw_mode(exp):
    """The excitation mode of an Experiment: ``mwMode`` if set, else the legacy ``Mode``."""
    mm = getattr(exp, 'mwMode', None)
    if mm is None or (isinstance(mm, str) and mm == ''):
        mm = getattr(exp, 'Mode', 'perpendicular') or 'perpendicular'
    return mm
