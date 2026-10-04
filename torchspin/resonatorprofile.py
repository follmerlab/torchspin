"""Resonator profile calculation.

Port of EasySpin's ``resonatorprofile.m``.

Computes the transfer function, voltage reflection coefficient, or power
reflection coefficient for an ideal RLC series resonator.

Example
-------
>>> import numpy as np
>>> from torchspin.resonatorprofile import resonatorprofile
>>> nu = np.linspace(9.2, 9.8, 1001)
>>> H = resonatorprofile(nu, nu0=9.5, Qu=1000, beta=1, mode='transferfunction')
>>> H.shape
(1001,)
"""
from __future__ import annotations

import numpy as np


def resonatorprofile(
    nu: np.ndarray | None,
    nu0: float,
    Qu: float,
    beta: float | None = None,
    mode: str = 'transferfunction',
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute resonator profile.

    Parameters
    ----------
    nu:
        Frequency array in GHz.  If ``None``, an appropriate range is
        chosen automatically and returned as the first output.
    nu0:
        Resonator frequency in GHz.
    Qu:
        Unloaded Q-factor.
    beta:
        Coupling coefficient.  ``beta < 1``: undercoupled, ``beta = 1``:
        critically coupled, ``beta > 1``: overcoupled.  Required for
        reflection modes.
    mode:
        ``'transferfunction'``, ``'voltagereflection'``, or
        ``'powerreflection'``.

    Returns
    -------
    out : np.ndarray
        Profile values.  If *nu* was ``None``, returns ``(nu, out)``.
    """
    return_nu = (nu is None)

    if nu is None:
        dnu = nu0 / Qu * 10
        nu = np.linspace(max(nu0 - dnu, 0), nu0 + dnu, 1001)
    else:
        nu = np.asarray(nu, dtype=np.float64)

    if mode in ('voltagereflection', 'powerreflection') and beta is None:
        raise ValueError(
            "Coupling coefficient beta is required for reflection modes.")

    # Transfer function for an ideal RLC series circuit
    def H_func(nu_arr, nu0_val, Q_val):
        return 1.0 / (1 + 1j * Q_val * (nu_arr / nu0_val - nu0_val / nu_arr))

    if mode == 'transferfunction':
        Q = Qu / (1 + beta) if beta is not None else Qu
        out = H_func(nu, nu0, Q)
    elif mode == 'voltagereflection':
        H = H_func(nu, nu0, Qu)
        out = (1 - beta * H) / (1 + beta * H)
    elif mode == 'powerreflection':
        H = H_func(nu, nu0, Qu)
        out = np.abs((1 - beta * H) / (1 + beta * H)) ** 2
    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Use 'transferfunction', "
            "'voltagereflection', or 'powerreflection'.")

    if return_nu:
        return nu, out
    return out
