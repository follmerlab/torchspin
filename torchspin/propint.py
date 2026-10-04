"""Propagator integration for time-dependent Hamiltonians.

Port of EasySpin's ``propint.m``.

Computes the propagator for a Hamiltonian with a time-dependent cosine
modulation::

    H(t) = H0 + H1 * cos(2*pi*freq*t + phase)

by piecewise-constant integration over one period and exponentiation of
full periods.

Example
-------
>>> import numpy as np
>>> from scipy.linalg import expm
>>> from torchspin.propint import propint
>>> Sz = np.diag([0.5, -0.5])
>>> Sy = np.array([[0, -0.5j], [0.5j, 0]])
>>> mwFreq = 10e3  # 10 GHz in MHz
>>> tp = 0.010     # 10 ns in µs
>>> U = propint(mwFreq * Sz, 0.5/tp * Sy, tp, mwFreq)
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm


def propint(
    H0: np.ndarray,
    H1: np.ndarray,
    t: float | list | np.ndarray,
    freq: float,
    phase: float = 0.0,
    n: int = 256,
) -> np.ndarray:
    """Compute propagator by integrating over a cosine-modulated Hamiltonian.

    Parameters
    ----------
    H0:
        Time-independent Hamiltonian (frequency units, e.g. MHz).
    H1:
        Amplitude of the cosine-modulated part (same units as *H0*).
    t:
        Pulse duration or time interval ``[t1, t2]``.  Scalar ``t`` is
        equivalent to ``[0, t]``.  Units complementary to *freq*
        (e.g. µs if *freq* is in MHz).
    freq:
        Modulation frequency (must be positive).
    phase:
        Modulation phase in radians (default 0).
    n:
        Number of integration intervals per period (default 256).

    Returns
    -------
    U:
        Propagator matrix, shape ``(dim, dim)``.
    """
    H0 = np.asarray(H0, dtype=np.complex128)
    H1 = np.asarray(H1, dtype=np.complex128)

    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    if len(t) == 1:
        tlim = np.array([0.0, t[0]])
    else:
        tlim = t[:2].copy()

    if freq <= 0:
        raise ValueError("Frequency must be positive.")

    # Short-cut: no time dependence
    if np.all(H1 == 0):
        return expm(-2j * np.pi * (tlim[1] - tlim[0]) * H0)

    dim = H0.shape[0]
    nIntervals = int(n)

    # Time grid within one period
    tPeriod = 1.0 / freq
    dt = tPeriod / nIntervals
    t_centers = (np.arange(nIntervals) + 0.5) * dt  # centre of each interval

    # Cosine modulation values at interval centres
    ct = np.cos(2 * np.pi * freq * t_centers + phase)

    # Precompute constant factors
    cH0 = -2j * np.pi * dt * H0
    cH1 = -2j * np.pi * dt * H1

    # Integrate propagator over one full period
    U_period = np.eye(dim, dtype=np.complex128)

    # Determine which intervals correspond to start/end of pulse
    j_start = int(np.fix(nIntervals * np.mod(tlim[0] / tPeriod, 1)))
    j_end = int(np.fix(nIntervals * np.mod(tlim[1] / tPeriod, 1)))

    U1 = None
    U2 = None

    for k in range(nIntervals):
        if k == j_start:
            U1 = U_period.copy()
        if k == j_end:
            U2 = U_period.copy()

        U_step = expm(cH0 + cH1 * ct[k])
        U_period = U_step @ U_period

    # Handle edge cases where j falls at boundary
    if U1 is None:
        U1 = np.eye(dim, dtype=np.complex128)
    if U2 is None:
        U2 = U_period.copy()

    # Number of full periods
    n_periods_start = int(np.fix(tlim[0] / tPeriod))
    n_periods_end = int(np.fix(tlim[1] / tPeriod))
    n_full_periods = n_periods_end - n_periods_start

    # Total propagator: U = U2 * U_period^n_full_periods * U1^†
    U_power = np.linalg.matrix_power(U_period, n_full_periods)
    U = U2 @ U_power @ U1.conj().T

    return U
