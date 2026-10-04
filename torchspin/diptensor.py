"""Dipolar coupling tensor calculation for torchspin.

Port of EasySpin's ``diptensor.m``.

Computes the point-dipole coupling tensor between two magnetic moments
(electron-electron, electron-nuclear, or nuclear-nuclear) given the
inter-spin distance vector.
"""
from __future__ import annotations

import math
import numpy as np

from torchspin.constants import MU0, HBAR, BMAGN, NMAGN, GFREE, PLANCK

__all__ = ['diptensor']


def _effective_mug(spin_spec):
    """Return the effective magnetic-moment tensor (or scalar) for a spin.

    Follows MATLAB EasySpin's ``diptensor.m`` convention:

    * Electron (g-factor or tensor): ``+μ_B * g`` (3x3 matrix in J/T)
    * Nucleus (isotope string): ``-μ_N * gn`` (scalar in J/T, NOTE negative sign)

    The nuclear minus sign is combined with the overall minus in the T
    formula to produce physically correct signs for all e-e, e-n, n-n
    coupling combinations.

    Returns
    -------
    mug : float or (3, 3) ndarray
        For electrons, a 3x3 matrix (anisotropic g) or scalar-times-I effective
        matrix. For nuclei, a scalar (isotropic).
    """
    if isinstance(spin_spec, str):
        from torchspin.nucdata import nucgval
        gn = nucgval(spin_spec)
        return -NMAGN * gn   # NEGATIVE sign per MATLAB convention
    else:
        g = np.asarray(spin_spec, dtype=float)
        if g.ndim == 0:
            g_mat = float(g) * np.eye(3)
        elif g.ndim == 1 and g.shape[0] == 3:
            g_mat = np.diag(g)
        elif g.ndim == 2 and g.shape == (3, 3):
            g_mat = g
        else:
            raise ValueError(f"g must be scalar, (3,) vector, or (3,3) matrix; got shape {g.shape}")
        return BMAGN * g_mat   # POSITIVE sign


def diptensor(spin1, spin2, rvec_nm) -> np.ndarray:
    """Compute the point-dipole coupling tensor between two spins.

    Port of EasySpin's ``diptensor.m``.

    Parameters
    ----------
    spin1 : float, array_like, or str
        Spin 1 specification:

        * float — g-factor of electron spin
        * (3,) array — principal g-values of electron spin
        * (3, 3) array — full g-tensor of electron spin
        * str — isotope string, e.g. ``'1H'``, ``'14N'`` (nuclear spin)

    spin2 : float, array_like, or str
        Spin 2 specification (same format as *spin1*).

    rvec_nm : array_like, shape (3,)
        Distance vector from spin 1 to spin 2, in **nanometres** [nm].

    Returns
    -------
    T : ndarray, shape (3, 3)
        Dipolar coupling tensor in **MHz**.  The full secular Hamiltonian is::

            H_dip = S1 · T · S2

        where T_ij = (μ₀/4π) * (γ₁ γ₂ ℏ) / r³ * (3 r̂ᵢ r̂ⱼ − δᵢⱼ) / h

    Notes
    -----
    Formula::

        T_ij = (μ₀ / 4π) * (ℏ γ₁ γ₂) / (r³) * (3 r̂_i r̂_j − δ_ij) / (2π × 10⁶)

    The prefactor ``μ₀/(4π) * ℏ γ₁ γ₂ / r³`` is in rad/s (angular frequency).
    Dividing by ``2π × 10⁶`` converts to MHz (linear frequency).

    Examples
    --------
    >>> T = diptensor(2.0, 2.0, [0, 0, 1.0])   # e-e along z, r=1 nm
    >>> print(T.diagonal())                      # [Txx, Tyy, Tzz] in MHz
    >>> T = diptensor('1H', '1H', [0, 0, 0.3])  # H-H at 0.3 nm (≈3 Å)
    """
    r = np.asarray(rvec_nm, dtype=float)
    if r.shape != (3,):
        raise ValueError(f"rvec_nm must be shape (3,); got {r.shape}")

    # Convert nm → m
    r_m = r * 1e-9
    r_abs = np.linalg.norm(r_m)
    if r_abs == 0:
        raise ValueError("rvec_nm must be non-zero.")

    r_hat = r_m / r_abs
    r3 = r_abs ** 3

    # Effective magnetic-moment matrices (MATLAB convention).
    # For electrons mug is a 3x3 (anisotropic g possible); for nuclei mug is
    # a scalar (isotropic). Cast to 3x3 form for unified matrix algebra below.
    mug1 = _effective_mug(spin1)
    mug2 = _effective_mug(spin2)
    if np.isscalar(mug1):
        mug1 = mug1 * np.eye(3)
    if np.isscalar(mug2):
        mug2 = mug2 * np.eye(3)
    mug1 = np.asarray(mug1, dtype=float)
    mug2 = np.asarray(mug2, dtype=float)

    # Geometry tensor d = 3 r̂ r̂ᵀ − I
    d = 3.0 * np.outer(r_hat, r_hat) - np.eye(3)

    # MATLAB diptensor.m line 109:
    #   T = -mu0/(4*pi) * r^-3 * mug{1}.' * d * mug{2}     [J]
    # Note the LEADING MINUS SIGN — this combines with the nuclear sign in
    # `_effective_mug` so that:
    #   e-e gives T_zz < 0 (along ẑ),  e-n gives T_zz > 0,  n-n gives T_zz < 0
    # which is the physically correct sign convention from -μ₁·μ₂ structure.
    T_J = -(MU0 / (4.0 * math.pi)) / r3 * (mug1.T @ d @ mug2)

    # Convert J → MHz: divide by Planck constant × 1e6
    T_MHz = T_J / PLANCK / 1e6

    return T_MHz
