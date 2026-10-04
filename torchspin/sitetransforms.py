"""Site transformations for crystal simulations.

Port of EasySpin's ``sitetransforms.m`` and ``p_crystalorientations.m``.

Provides the rotation matrices of the proper rotation subgroup of the Laue
class a space group belongs to.  Used to enumerate the magnetically distinct
sites of a crystal and to build the molecule→lab orientation list for
single-crystal simulations.

References
----------
J.A. Weil, T. Buch, J.E. Clapp, Adv. Magn. Reson. 6, 183–257 (1973), Table II.
"""
from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Optional, Sequence

import numpy as np

_SQ3 = math.sqrt(3.0)

# Common rotation matrices (active rotations)
_E    = np.eye(3)
_C2X  = np.diag([+1.0, -1.0, -1.0])
_C2Y  = np.diag([-1.0, +1.0, -1.0])
_C2Z  = np.diag([-1.0, -1.0, +1.0])
_C4ZP = np.array([[0.0, -1, 0], [+1, 0, 0], [0, 0, +1]])
_C4ZM = np.array([[0.0, +1, 0], [-1, 0, 0], [0, 0, +1]])
_C2XY1 = np.array([[0.0, +1, 0], [+1, 0, 0], [0, 0, -1]])
_C2XY2 = np.array([[0.0, -1, 0], [-1, 0, 0], [0, 0, -1]])
_C3ZP = np.array([[-0.5, -_SQ3 / 2, 0], [+_SQ3 / 2, -0.5, 0], [0, 0, +1]])
_C3ZM = np.array([[-0.5, +_SQ3 / 2, 0], [-_SQ3 / 2, -0.5, 0], [0, 0, +1]])
_C6ZP = np.array([[+0.5, -_SQ3 / 2, 0], [+_SQ3 / 2, +0.5, 0], [0, 0, +1]])
_C6ZM = np.array([[+0.5, +_SQ3 / 2, 0], [-_SQ3 / 2, +0.5, 0], [0, 0, +1]])

_T_ROTS = [
    np.array([[0.0, 0, +1], [+1, 0, 0], [0, +1, 0]]),   # C3+ (+1,+1,+1)
    np.array([[0.0, +1, 0], [0, 0, +1], [+1, 0, 0]]),   # C3- (+1,+1,+1)
    np.array([[0.0, 0, -1], [-1, 0, 0], [0, +1, 0]]),   # C3+ (+1,-1,-1)
    np.array([[0.0, -1, 0], [0, 0, +1], [-1, 0, 0]]),   # C3- (+1,-1,-1)
    np.array([[0.0, 0, +1], [-1, 0, 0], [0, -1, 0]]),   # C3+ (-1,+1,-1)
    np.array([[0.0, -1, 0], [0, 0, -1], [+1, 0, 0]]),   # C3- (-1,+1,-1)
    np.array([[0.0, 0, -1], [+1, 0, 0], [0, -1, 0]]),   # C3+ (-1,-1,+1)
    np.array([[0.0, +1, 0], [0, 0, -1], [-1, 0, 0]]),   # C3- (-1,-1,+1)
]

_POINT_GROUPS_SCHOENFLIES = [
    'C1', 'Ci', 'C2', 'Cs', 'C2h', 'D2', 'C2v', 'D2h',
    'C4', 'S4', 'C4h', 'D4', 'C4v', 'D2d', 'D4h', 'C3', 'C3i', 'D3', 'C3v',
    'D3d', 'C6', 'C3h', 'C6h', 'D6', 'C6v', 'D3h', 'D6h', 'T', 'Th', 'O',
    'Td', 'Oh',
]
_POINT_GROUPS_HM = [
    '1', '-1', '2', 'm', '2/m', '222', 'mm2', 'mmm',
    '4', '-4', '4/m', '422', '4mm', '-42m', '4/mmm', '3', '-3', '32', '3m',
    '-3m', '6', '-6', '6/m', '622', '6mm', '-6m2', '6/mmm', '23', 'm-3',
    '432', '-43m', 'm-3m',
]
_POINT_GROUP_LAUE = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 7, 7,
                     7, 8, 8, 8, 9, 9, 9, 9, 10, 10, 11, 11, 11]


@lru_cache(maxsize=1)
def _load_spacegroups() -> tuple[list[int], list[str], list[str]]:
    path = os.path.join(os.path.dirname(__file__), 'data', 'spacegroups.txt')
    numbers, names, axes = [], [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            parts = line.split()
            numbers.append(int(parts[0]))
            names.append(parts[1])
            axes.append(parts[2] if len(parts) > 2 else '-')
    return numbers, names, axes


def _groupnumber_to_laueclass(n: int) -> int:
    if n <= 2:
        return 1
    if n <= 15:
        return 2
    if n <= 74:
        return 3
    if n <= 88:
        return 4
    if n <= 142:
        return 5
    if n <= 148:
        return 6
    if n <= 167:
        return 7
    if n <= 176:
        return 8
    if n <= 194:
        return 9
    if n <= 206:
        return 10
    return 11


def sitetransforms(ID) -> list[np.ndarray]:
    """Active rotation matrices between equivalent crystal sites.

    Parameters
    ----------
    ID:
        Space group number (1–230), Hermann–Mauguin space group symbol
        (e.g. ``'P212121'``), or crystallographic point group symbol
        (Schoenflies like ``'D2h'`` or Hermann–Mauguin like ``'mmm'``).

    Returns
    -------
    List of 3×3 active rotation matrices, one per magnetically distinct site.
    """
    axis_convention = 'default'
    if isinstance(ID, str):
        numbers, names, axes = _load_spacegroups()
        if ID in names:
            i = names.index(ID)
            laue = _groupnumber_to_laueclass(numbers[i])
            axis_convention = axes[i]
        elif ID in _POINT_GROUPS_SCHOENFLIES:
            laue = _POINT_GROUP_LAUE[_POINT_GROUPS_SCHOENFLIES.index(ID)]
        elif ID in _POINT_GROUPS_HM:
            laue = _POINT_GROUP_LAUE[_POINT_GROUPS_HM.index(ID)]
        else:
            raise ValueError(
                f"Point or space group symmetry symbol '{ID}' is unknown.")
    else:
        n = int(ID)
        if n != ID or n < 1 or n > 230:
            raise ValueError(
                f"Invalid space group number {ID}. Must be between 1 and 230.")
        laue = _groupnumber_to_laueclass(n)

    if axis_convention in ('default', '-'):
        axis_convention = 'b' if laue == 2 else 'z'

    if laue == 1:      # triclinic C1
        R = [_E]
    elif laue == 2:    # monoclinic C2
        if axis_convention in ('b', '-b'):
            R2 = _C2Y
        elif axis_convention in ('a', '-a'):
            R2 = _C2X
        elif axis_convention in ('c', '-c', 'z'):
            R2 = _C2Z
        else:
            # Orthorhombic-style cell-setting codes (e.g. 'abc') should not
            # appear for monoclinic groups; treat unique axis b as default.
            R2 = _C2Y
        R = [_E, R2]
    elif laue == 3:    # orthorhombic D2
        R = [_E, _C2Z, _C2X, _C2Y]
    elif laue == 4:    # tetragonal C4
        R = [_E, _C2Z, _C4ZP, _C4ZM]
    elif laue == 5:    # tetragonal D4
        R = [_E, _C2Z, _C4ZP, _C4ZM, _C2X, _C2Y, _C2XY1, _C2XY2]
    elif laue == 6:    # trigonal C3
        R = [_E, _C3ZP, _C3ZM]
    elif laue == 7:    # trigonal D3
        R = [_E, _C3ZP, _C3ZM, _C2X,
             np.array([[-0.5, +_SQ3 / 2, 0], [+_SQ3 / 2, +0.5, 0], [0, 0, -1.0]]),
             np.array([[-0.5, -_SQ3 / 2, 0], [-_SQ3 / 2, +0.5, 0], [0, 0, -1.0]])]
    elif laue == 8:    # hexagonal C6
        R = [_E, _C2Z, _C3ZP, _C3ZM, _C6ZP, _C6ZM]
    elif laue == 9:    # hexagonal D6
        R = [_E, _C2Z, _C3ZP, _C3ZM, _C6ZP, _C6ZM, _C2X, _C2Y,
             np.array([[-0.5, +_SQ3 / 2, 0], [+_SQ3 / 2, +0.5, 0], [0, 0, -1.0]]),
             np.array([[-0.5, -_SQ3 / 2, 0], [-_SQ3 / 2, +0.5, 0], [0, 0, -1.0]]),
             np.array([[+0.5, +_SQ3 / 2, 0], [+_SQ3 / 2, -0.5, 0], [0, 0, -1.0]]),
             np.array([[+0.5, -_SQ3 / 2, 0], [-_SQ3 / 2, -0.5, 0], [0, 0, -1.0]])]
    elif laue == 10:   # cubic T
        R = [_E, _C2Z, _C2X, _C2Y] + _T_ROTS
    else:              # laue == 11, cubic O
        R = [_E, _C2Z, _C4ZP, _C4ZM, _C2X, _C2Y, _C2XY1, _C2XY2] + _T_ROTS + [
            np.array([[+1.0, 0, 0], [0, 0, -1], [0, +1, 0]]),   # C4+ x
            np.array([[+1.0, 0, 0], [0, 0, +1], [0, -1, 0]]),   # C4- x
            np.array([[0.0, 0, +1], [0, +1, 0], [-1, 0, 0]]),   # C4+ y
            np.array([[0.0, 0, -1], [0, +1, 0], [+1, 0, 0]]),   # C4- y
            np.array([[0.0, 0, +1], [0, -1, 0], [+1, 0, 0]]),   # C2 (1,0,+1)
            np.array([[0.0, 0, -1], [0, -1, 0], [-1, 0, 0]]),   # C2 (1,0,-1)
            np.array([[-1.0, 0, 0], [0, 0, +1], [0, +1, 0]]),   # C2 (0,1,+1)
            np.array([[-1.0, 0, 0], [0, 0, -1], [0, -1, 0]]),   # C2 (0,1,-1)
        ]
    return [r.copy() for r in R]


def crystal_orientations(
    sample_frame,
    crystal_symmetry,
    mol_frame,
) -> np.ndarray:
    """Molecule→lab Euler angle list for a single-crystal simulation.

    Port of EasySpin's ``p_crystalorientations.m`` (crystal branch, no
    ``SampleRotation``).

    Parameters
    ----------
    sample_frame:
        ``None``, a 3-vector, or an (N, 3) array of Euler angles (rad)
        defining the lab→sample transformation(s).
    crystal_symmetry:
        Space group number/symbol or point group symbol.  ``None``/'' → P1.
    mol_frame:
        ``None`` or 3-vector of Euler angles (rad) defining the
        sample→molecule transformation.

    Returns
    -------
    (nSampleOrientations*nSites, 3) array of ``[phi, theta, chi]`` Euler
    angles for the molecule→lab transformation of each site.
    """
    from torchspin.rotations import erot
    from torchspin.rotutils import eulang

    if crystal_symmetry is None or crystal_symmetry == '':
        crystal_symmetry = 'P1'
    Rsite_C = sitetransforms(crystal_symmetry)
    n_sites = len(Rsite_C)

    if mol_frame is not None and len(np.ravel(mol_frame)) == 3:
        R_S2M = np.asarray(erot(list(np.ravel(mol_frame))), dtype=float)
    else:
        R_S2M = np.eye(3)
    R_M2S = R_S2M.T

    if sample_frame is None:
        sample_frames = np.zeros((1, 3))
    else:
        sample_frames = np.atleast_2d(np.asarray(sample_frame, dtype=float))
        if sample_frames.shape[1] != 3:
            raise ValueError(
                "SampleFrame requires three Euler angles per row.")

    # Molecular axes represented in the sample/crystal frame, per site
    xyzM_S = [Rs @ R_M2S for Rs in Rsite_C]

    angles_M2L = np.zeros((sample_frames.shape[0] * n_sites, 3))
    idx = 0
    for sf in sample_frames:
        R_L2S = np.asarray(erot(list(sf)), dtype=float)
        R_S2L = R_L2S.T
        for xyz in xyzM_S:
            R_M2L = R_S2L @ xyz
            angles_M2L[idx] = eulang(R_M2L)
            idx += 1
    return angles_M2L
