"""Intermolecular dipolar background decay for DEER.

Port of EasySpin's ``dipbackground.m``.

Computes the background decay function arising from a homogeneous 3D
distribution of spins, used in DEER/PELDOR analysis.

Example
-------
>>> import numpy as np
>>> from torchspin.dipbackground import dipbackground
>>> t = np.linspace(0, 5, 501)  # microseconds
>>> V = dipbackground(t, conc=50.0, lam=0.3)  # 50 uM, lambda=0.3
"""
from __future__ import annotations

import numpy as np

from torchspin.constants import MU0, BMAGN, PLANCK, HBAR, GFREE, AVOGADRO


def dipbackground(
    t: np.ndarray,
    conc: float,
    lam: float,
    g: tuple[float, float] | None = None,
) -> np.ndarray:
    """Compute intermolecular dipolar background decay.

    Parameters
    ----------
    t:
        Time values in microseconds.
    conc:
        Spin concentration in micromolar (uM).
    lam:
        Modulation depth, between 0 and 1.
    g:
        g-factors of the two spins ``[gA, gB]``.
        Default: ``[gfree, gfree]``.

    Returns
    -------
    V:
        Background decay function, same shape as ``t``.
    """
    t = np.asarray(t, dtype=np.float64)

    if g is None:
        gA = gB = GFREE
    else:
        gA, gB = float(g[0]), float(g[1])

    # Dipolar coupling constant D in rad/s * m^3
    D = (MU0 / (4 * np.pi)) * gA * gB * BMAGN ** 2 / HBAR  # rad/s * m^3

    # Convert concentration: uM -> mol/L -> spins/m^3
    # 1 uM = 1e-6 mol/L = 1e-6 * 1e3 mol/m^3 = 1e-3 mol/m^3
    conc_m3 = conc * 1e-3 * AVOGADRO  # spins/m^3

    # Background decay rate constant
    # k = (8*pi^2)/(9*sqrt(3)) * D * conc * lambda
    k = (8 * np.pi ** 2) / (9 * np.sqrt(3)) * D * conc_m3 * lam

    # Convert k from 1/s to 1/us
    k = k * 1e-6

    # V(t) = exp(-k * |t|)
    return np.exp(-k * np.abs(t))
