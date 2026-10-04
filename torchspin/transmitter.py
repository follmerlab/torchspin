"""Transmitter nonlinearity simulation and compensation.

Port of EasySpin's ``transmitter.m``.

Models the effect of a nonlinear power amplifier on a pulse signal.
The transmitter characteristic (input amplitude → output amplitude) is
described by a polynomial fit constrained through the origin.

Example
-------
>>> import numpy as np
>>> from torchspin.transmitter import transmitter
>>> Ain = np.linspace(0, 1, 50)
>>> Aout = np.tanh(2 * Ain) / np.tanh(2)  # compression curve
>>> signal = np.sin(np.linspace(0, 2 * np.pi, 200))
>>> compressed = transmitter(signal, Ain, Aout, 'simulate')
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


def transmitter(
    signal: np.ndarray,
    Ain: np.ndarray,
    Aout: np.ndarray,
    option: str,
    poly_order: int = 4,
) -> np.ndarray:
    """Simulate or compensate transmitter nonlinearity.

    Parameters
    ----------
    signal:
        Input signal vector.  For ``'simulate'``, amplitudes refer to the
        *Ain* scale.  For ``'compensate'``, amplitudes refer to the *Aout*
        scale.
    Ain:
        Input amplitude calibration points.
    Aout:
        Output amplitude calibration points (same length as *Ain*).
    option:
        ``'simulate'`` or ``'compensate'``.
    poly_order:
        Order of the polynomial fit (default 4).

    Returns
    -------
    signal_out:
        Signal with simulated compression or compensated nonlinearity.
    """
    signal = np.asarray(signal)
    Ain = np.asarray(Ain, dtype=np.float64).ravel()
    Aout = np.asarray(Aout, dtype=np.float64).ravel()

    if len(Ain) != len(Aout):
        raise ValueError("Ain and Aout must have the same length.")

    # Polynomial fit constrained through origin
    # Build Vandermonde matrix without constant term: A^N, A^{N-1}, ..., A^1
    N = poly_order
    V = np.column_stack([Ain ** (N - j) for j in range(N)])
    coeffs_no_const = np.linalg.lstsq(V, Aout, rcond=None)[0]
    # Full coefficients including zero constant: [c_N, c_{N-1}, ..., c_1, 0]
    coeffs = np.append(coeffs_no_const, 0.0)
    # Smooth the calibration data
    Aout_smooth = np.polyval(coeffs, Ain)

    if option == 'simulate':
        # Forward: Ain → Aout
        F = interp1d(Ain, Aout_smooth, kind='cubic',
                     fill_value='extrapolate')
    elif option == 'compensate':
        # Inverse: Aout → Ain (need unique Aout values)
        unique_idx = np.unique(Aout_smooth, return_index=True)[1]
        unique_idx.sort()
        F = interp1d(Aout_smooth[unique_idx], Ain[unique_idx], kind='cubic',
                     fill_value='extrapolate')
    else:
        raise ValueError(
            f"Unknown option '{option}'. Use 'simulate' or 'compensate'.")

    if np.isrealobj(signal):
        return np.sign(signal) * F(np.abs(signal))
    else:
        real_part = np.sign(np.real(signal)) * F(np.abs(np.real(signal)))
        imag_part = np.sign(np.imag(signal)) * F(np.abs(np.imag(signal)))
        return real_part + 1j * imag_part
