"""Associated Legendre polynomials.

Port of EasySpin's ``plegendre.m``.

Computes the (associated) Legendre polynomial P_L^M(z) using the standard
three-term recurrence relation (Numerical Recipes convention).

Example
-------
>>> import numpy as np
>>> from torchspin.plegendre import plegendre
>>> z = np.linspace(-1, 1, 101)
>>> P2 = plegendre(2, z)        # P_2(z) = (3z^2 - 1)/2
>>> P21 = plegendre(2, 1, z)    # P_2^1(z)
"""
from __future__ import annotations

import math
import numpy as np


def plegendre(
    L: int,
    M_or_z=None,
    z_or_CSphase=None,
    CSphase_arg=None,
) -> np.ndarray:
    r"""Compute associated Legendre polynomial :math:`P_L^M(z)`.

    Parameters
    ----------
    L:
        Degree (non-negative integer, ``L >= 0``).
    M_or_z:
        If an integer and ``z_or_CSphase`` is also provided, this is the
        order ``M`` (``|M| <= L``).  Otherwise this is the evaluation
        points ``z`` (with ``M=0``).
    z_or_CSphase:
        Evaluation points ``z`` (when ``M`` is given), or ``CSphase``
        flag (when ``M`` is omitted).  ``z`` must satisfy ``-1 <= z <= 1``.
    CSphase_arg:
        If ``True`` (default), include the Condon-Shortley phase
        :math:`(-1)^M`.

    Returns
    -------
    y:
        Associated Legendre polynomial values, same shape as ``z``.
    """
    # Parse flexible arguments: plegendre(L, z), plegendre(L, M, z), plegendre(L, M, z, CS)
    if M_or_z is None:
        raise ValueError("At least two arguments required: plegendre(L, z) or plegendre(L, M, z)")

    if z_or_CSphase is None:
        # plegendre(L, z) — M defaults to 0
        M = 0
        z = np.asarray(M_or_z, dtype=np.float64)
        CSphase = True
    elif isinstance(M_or_z, (int, np.integer)):
        # plegendre(L, M, z) or plegendre(L, M, z, CS)
        M = int(M_or_z)
        z = np.asarray(z_or_CSphase, dtype=np.float64)
        CSphase = CSphase_arg if CSphase_arg is not None else True
    else:
        # plegendre(L, z, CSphase)
        M = 0
        z = np.asarray(M_or_z, dtype=np.float64)
        CSphase = bool(z_or_CSphase)

    # Validate
    L = int(L)
    if L < 0:
        raise ValueError(f"L must be non-negative, got {L}")
    if abs(M) > L:
        raise ValueError(f"|M| must be <= L, got M={M}, L={L}")
    if np.any(z < -1) or np.any(z > 1):
        raise ValueError("z values must be in [-1, 1]")

    # Handle negative M
    neg_M = M < 0
    M_abs = abs(M)

    # Compute P_L^|M|(z) via three-term recurrence
    y = _plegendre_recurrence(L, M_abs, z)

    # Apply negative-M relation: P_L^{-M} = (-1)^M * (L-M)!/(L+M)! * P_L^M
    if neg_M:
        factor = (-1.0) ** M_abs * math.factorial(L - M_abs) / math.factorial(L + M_abs)
        y = factor * y

    # Apply Condon-Shortley phase
    if CSphase and M_abs > 0:
        y = (-1.0) ** M_abs * y

    return y


def _plegendre_recurrence(L: int, M: int, z: np.ndarray) -> np.ndarray:
    """Three-term recurrence for P_L^M(z) with M >= 0."""
    shape = z.shape
    z = z.ravel()

    # Base case: P_M^M(z) = (2M-1)!! * (1 - z^2)^(M/2)
    if M == 0:
        pmm = np.ones_like(z)
    else:
        # double factorial (2M-1)!!
        double_fact = 1.0
        for k in range(1, 2 * M, 2):
            double_fact *= k
        pmm = double_fact * (1.0 - z * z) ** (M / 2.0)

    if L == M:
        return pmm.reshape(shape)

    # P_{M+1}^M(z) = z * (2M+1) * P_M^M(z)
    pmm1 = z * (2 * M + 1) * pmm

    if L == M + 1:
        return pmm1.reshape(shape)

    # Recurrence for L >= M+2:
    # P_l^M = [(2l-1)*z*P_{l-1}^M - (l+M-1)*P_{l-2}^M] / (l-M)
    p_prev2 = pmm
    p_prev1 = pmm1
    for ell in range(M + 2, L + 1):
        p_curr = ((2 * ell - 1) * z * p_prev1 - (ell + M - 1) * p_prev2) / (ell - M)
        p_prev2 = p_prev1
        p_prev1 = p_curr

    return p_prev1.reshape(shape)
