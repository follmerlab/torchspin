"""DEER dipolar kernel matrix.

Port of EasySpin's ``dipkernel.m``.

Constructs the kernel matrix K(t, r) for DEER (Double Electron-Electron
Resonance) spectroscopy, relating the time-domain signal to the distance
distribution.

Example
-------
>>> import numpy as np
>>> from torchspin.dipkernel import dipkernel
>>> t = np.linspace(0, 3, 201)   # microseconds
>>> r = np.linspace(1.5, 6, 100)  # nanometers
>>> K = dipkernel(t, r)
>>> K.shape  # (201, 100)
"""
from __future__ import annotations

import numpy as np
from scipy.special import fresnel as _scipy_fresnel

from torchspin.constants import MU0, BMAGN, PLANCK, GFREE


def dipkernel(
    t: np.ndarray,
    r: np.ndarray,
    g: tuple[float, float] | None = None,
) -> np.ndarray:
    """Compute the DEER dipolar kernel matrix.

    Parameters
    ----------
    t:
        Time values in microseconds.
    r:
        Distance values in nanometers.  Must be equally spaced.
    g:
        g-factors of the two spins ``[gA, gB]``.
        Default: ``[gfree, gfree]``.

    Returns
    -------
    K:
        Kernel matrix, shape ``(len(t), len(r))``.
        ``K[i, j]`` is the dipolar signal at time ``t[i]`` for distance ``r[j]``.
    """
    t = np.asarray(t, dtype=np.float64).ravel()
    r = np.asarray(r, dtype=np.float64).ravel()

    if g is None:
        gA = gB = GFREE
    else:
        gA, gB = float(g[0]), float(g[1])

    # Validate equally spaced r
    if len(r) > 1:
        dr = np.diff(r)
        if np.max(np.abs(dr - dr[0])) / dr[0] > 1e-6:
            raise ValueError("Distance values r must be equally spaced")
        dr_val = dr[0]
    else:
        dr_val = None

    # Dipolar coupling constant D in MHz*nm^3
    # D = (mu0/4pi) * gA*gB*bmagn^2 / (planck * 1e-6 * 1e-27)
    # Careful with units: t in us, r in nm
    # omega_dd = 2*pi*D/r^3 [rad/us]
    # D in MHz*nm^3:
    D = (MU0 / (4 * np.pi)) * gA * gB * BMAGN ** 2 / PLANCK  # Hz*m^3
    D = D * 1e-6  # Hz*m^3 -> MHz*m^3
    D = D * 1e27   # MHz*m^3 -> MHz*nm^3

    # Build kernel using Fresnel integral formulation
    # For each (t, r): omega_dd = 2*pi*D/r^3, phase = omega_dd * |t|
    # kappa = sqrt(6*phase/pi)
    # K = cos(phase)*C(kappa)/kappa + sin(phase)*S(kappa)/kappa

    t_abs = np.abs(t)

    # Outer product: phase[i,j] = 2*pi*D/r[j]^3 * |t[i]|
    omega_dd = 2 * np.pi * D / r ** 3  # shape (nr,)
    phase = np.outer(t_abs, omega_dd)   # shape (nt, nr)

    kappa = np.sqrt(6 * phase / np.pi)

    # Fresnel integrals: scipy.special.fresnel returns (S, C)
    S_vals, C_vals = _scipy_fresnel(kappa)

    # K = cos(phase)*C(kappa)/kappa + sin(phase)*S(kappa)/kappa
    with np.errstate(divide='ignore', invalid='ignore'):
        K = np.cos(phase) * C_vals / kappa + np.sin(phase) * S_vals / kappa

    # At t=0 (phase=0, kappa=0): K should be 1
    K[np.isnan(K)] = 1.0

    # Multiply by dr for numerical integration (trapezoidal-like)
    if dr_val is not None and len(r) > 1:
        K = K * dr_val

    return K
