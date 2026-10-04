"""Photo-selection weight for photo-excited spin systems.

Port of EasySpin's ``photoselect.m``.

Calculates an orientation-dependent weight (0 to 1) describing the probability
of photo-excitation based on the alignment of the electric transition dipole
moment with the light polarization.

Example
-------
>>> import numpy as np
>>> from torchspin.photoselect import photoselect
>>> # TDM along molecular z, B0 along z, light propagation along y,
>>> # E-field polarized along z (alpha=0)
>>> w = photoselect([0, 0, 1], [[0.0, 0.0, 0.0]], [0, 1, 0], 0.0)
>>> w  # strong excitation for this geometry
"""
from __future__ import annotations

import math
import numpy as np

from torchspin.rotations import erot
from torchspin.rotutils import vec2ang, ang2vec


def photoselect(
    tdm,
    ori: np.ndarray,
    k,
    alpha: float,
) -> np.ndarray:
    """Compute photo-selection weight.

    Parameters
    ----------
    tdm:
        Transition dipole moment orientation in the molecular frame.
        Can be:

        * A string like ``'x'``, ``'z'``, ``'xz'``, ``'-y'``
        * A 3-element vector ``[mx, my, mz]`` (normalized automatically)
        * A 2-element array ``[phi, theta]`` (spherical angles in radians)

    ori:
        Orientation(s) as an ``(N, 2)`` or ``(N, 3)`` array of Euler angles
        ``[phi, theta]`` or ``[phi, theta, chi]`` (radians).
        If ``chi`` is omitted, the integral over chi from 0 to 2*pi is computed.

    k:
        Light propagation direction in the lab frame.  Same input formats
        as ``tdm``.

    alpha:
        Polarization angle (radians).  Defines the E-field orientation in
        the plane perpendicular to ``k``.  Use ``float('nan')`` for
        unpolarized/depolarized light.

    Returns
    -------
    weight:
        Photo-selection weight(s), shape ``(N,)``, values in [0, 1].
    """
    # --- Parse tdm ---
    tdm_mol = _parse_direction(tdm, "tdm")

    # --- Parse ori ---
    ori = np.atleast_2d(np.asarray(ori, dtype=np.float64))
    if ori.ndim != 2 or ori.shape[1] not in (2, 3):
        raise ValueError("ori must be (N,2) or (N,3) array of Euler angles")
    integrate_chi = ori.shape[1] == 2
    n_ori = ori.shape[0]

    # --- Parse k ---
    k_vec = _parse_direction(k, "k")
    k_angles = vec2ang(k_vec)  # (phi, theta)

    # --- Parse alpha ---
    unpolarized = math.isnan(alpha) if isinstance(alpha, float) else np.isnan(alpha)
    alpha_val = 0.0 if unpolarized else float(alpha)

    # --- Light frame vectors ---
    # erot returns R as 3x3 rotation matrix; rows are [p, q, k] in lab frame
    light_euler = [k_angles[0], k_angles[1], alpha_val]
    R_light = erot(light_euler)
    # R_light is a torch tensor; convert to numpy
    R_light_np = R_light.detach().cpu().numpy() if hasattr(R_light, 'numpy') else np.asarray(R_light)
    p_lab = R_light_np[0, :]  # E-field direction (first row)
    k_lab = R_light_np[2, :]  # propagation direction (third row)

    # --- Loop over orientations ---
    weight = np.zeros(n_ori)

    for i in range(n_ori):
        if integrate_chi:
            # Append chi=0 for rotation
            euler = [ori[i, 0], ori[i, 1], 0.0]
        else:
            euler = ori[i, :].tolist()

        R_M2L = erot(euler)
        R_np = R_M2L.detach().cpu().numpy() if hasattr(R_M2L, 'numpy') else np.asarray(R_M2L)

        # Transform tdm to lab frame
        tdm_lab = R_np @ tdm_mol

        if integrate_chi:
            # Analytical chi-integrated expressions (Mathematica-derived)
            tdm_xy2 = tdm_lab[0] ** 2 + tdm_lab[1] ** 2
            tdm_z2 = tdm_lab[2] ** 2

            if unpolarized:
                k_xy2 = k_lab[0] ** 2 + k_lab[1] ** 2
                weight_k = tdm_xy2 * k_xy2 / 2 + tdm_z2 * k_lab[2] ** 2
                weight[i] = (1.0 - weight_k) / 2.0
            else:
                p_xy2 = p_lab[0] ** 2 + p_lab[1] ** 2
                weight[i] = tdm_xy2 * p_xy2 / 2 + tdm_z2 * p_lab[2] ** 2
        else:
            if unpolarized:
                weight_k = abs(np.dot(tdm_lab, k_lab)) ** 2
                weight[i] = (1.0 - weight_k) / 2.0
            else:
                weight[i] = abs(np.dot(tdm_lab, p_lab)) ** 2

    return weight


def _parse_direction(d, name: str) -> np.ndarray:
    """Parse a direction input to a unit 3-vector in the molecular frame."""
    if isinstance(d, str):
        return _letter2vec(d)

    d = np.asarray(d, dtype=np.float64).ravel()
    if d.size == 3:
        norm = np.linalg.norm(d)
        if norm < 1e-15:
            raise ValueError(f"{name} vector has zero norm")
        return d / norm
    elif d.size == 2:
        return np.array(ang2vec(d[0], d[1]), dtype=np.float64).ravel()
    else:
        raise ValueError(f"{name} must be a string, 3-vector, or 2-element angle array")


def _letter2vec(s: str) -> np.ndarray:
    """Convert letter direction string to unit vector, e.g. 'x' -> [1,0,0]."""
    v = np.zeros(3)
    s = s.strip()
    i = 0
    while i < len(s):
        sign = 1.0
        if s[i] == '-':
            sign = -1.0
            i += 1
        elif s[i] == '+':
            i += 1
        if i >= len(s):
            raise ValueError(f"Invalid direction string: '{s}'")
        c = s[i].lower()
        if c == 'x':
            v[0] += sign
        elif c == 'y':
            v[1] += sign
        elif c == 'z':
            v[2] += sign
        else:
            raise ValueError(f"Unknown direction character '{c}' in '{s}'")
        i += 1

    norm = np.linalg.norm(v)
    if norm < 1e-15:
        raise ValueError(f"Direction string '{s}' results in zero vector")
    return v / norm
