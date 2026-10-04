"""Signal translation and cleanup for spidyan pulse EPR simulations.

Port of EasySpin's ``signalprocessing.m``.  Performs frequency down-conversion
(demodulation) on raw pulse EPR signals using :func:`rfmixer`.

The function handles:
- Multi-detection-operator signals
- Variable-length time axes across acquisition points
- Automatic detection of real, imaginary, or complex signals
- Non-uniform time axes (breaks from undetected events)

References
----------
EasySpin signalprocessing.m — spidyan signal post-processing.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from torchspin.rfmixer import rfmixer


__all__ = ['signalprocessing']


def signalprocessing(
    time_axis: np.ndarray | list,
    raw_signal: np.ndarray | list,
    freq_translations: list[float] | np.ndarray,
) -> np.ndarray | list:
    """Translate (down-convert) pulse EPR signals to baseband.

    Parameters
    ----------
    time_axis : ndarray or list
        Time axis in microseconds.  For a single acquisition point, a 1D
        array.  For multiple points, a 2D array ``(nPoints, nTime)`` or a
        list of arrays (one per acquisition point).
    raw_signal : ndarray or list
        Raw signal from spidyan/saffron.  Shape conventions:

        - 2D ``(nDetOps, nTime)`` — single acquisition point
        - 3D ``(nPoints, nDetOps, nTime)`` — multiple acquisition points
        - list of 2D arrays — cell-array equivalent
    freq_translations : array_like, shape (nDetOps,)
        Frequency (GHz) for each detection operator.  Set to 0 for operators
        that should not be demodulated (e.g. Sz).

    Returns
    -------
    processed : ndarray or list
        Down-converted signal in the same layout as *raw_signal*.

    Notes
    -----
    If down-conversion fails (e.g. wrong frequency, non-oscillating signal),
    the raw signal is returned with a warning.
    """
    freq_translations = np.atleast_1d(np.asarray(freq_translations, dtype=float))
    n_det = len(freq_translations)

    try:
        if isinstance(raw_signal, list):
            return _process_cell(time_axis, raw_signal, freq_translations, n_det)
        else:
            return _process_array(time_axis, np.asarray(raw_signal, dtype=complex),
                                  freq_translations, n_det)
    except Exception as e:
        warnings.warn(
            f"Down-conversion failed; returning raw signal. Error: {e}",
            stacklevel=2,
        )
        return raw_signal


def _process_cell(time_axis, raw_signal, freq_translations, n_det):
    """Process list-of-arrays (cell array) signals."""
    processed = []
    for i, sig in enumerate(raw_signal):
        if sig is None or (isinstance(sig, np.ndarray) and sig.size == 0):
            processed.append(sig)
            continue

        sig = np.asarray(sig, dtype=complex)

        # Get time axis for this point
        if isinstance(time_axis, list):
            t = np.asarray(time_axis[i], dtype=float).ravel()
        else:
            t_arr = np.asarray(time_axis, dtype=float)
            if t_arr.ndim == 2 and t_arr.shape[0] > 1:
                t = t_arr[i]
            else:
                t = t_arr.ravel()

        # Ensure shape is (nDetOps, nTime)
        if sig.ndim == 1:
            sig = sig[np.newaxis, :]
        if sig.shape[0] != n_det and sig.shape[1] == n_det:
            sig = sig.T

        out = np.zeros_like(sig)
        for iTrace in range(min(n_det, sig.shape[0])):
            out[iTrace] = _demod_trace(t, sig[iTrace], freq_translations[iTrace])

        processed.append(out.T)  # Transpose back to match EasySpin convention
    return processed


def _process_array(time_axis, raw_signal, freq_translations, n_det):
    """Process ndarray signals."""
    time_axis = np.asarray(time_axis, dtype=float)
    original_shape = raw_signal.shape

    # Normalize to 3D: (nPoints, nDetOps, nTime)
    if raw_signal.ndim == 2:
        n_points = 1
        if original_shape[-1] == n_det:
            raw_signal = raw_signal.reshape(1, n_det, -1)
        else:
            raw_signal = raw_signal.reshape(1, *original_shape)
    elif raw_signal.ndim >= 3:
        n_points = int(np.prod(original_shape[:-2]))
        raw_signal = raw_signal.reshape(n_points, original_shape[-2], original_shape[-1])
    else:
        raw_signal = raw_signal.reshape(1, 1, -1)
        n_points = 1

    # Normalize time axis
    if time_axis.ndim > 2:
        time_axis = time_axis.reshape(n_points, -1)

    processed = np.zeros_like(raw_signal)

    for iPoint in range(n_points):
        # Get time axis for this point
        if time_axis.ndim == 2 and time_axis.shape[0] > 1:
            t = time_axis[iPoint]
        else:
            t = time_axis.ravel()

        for iTrace in range(min(n_det, raw_signal.shape[1])):
            rf_signal = raw_signal[iPoint, iTrace, :]
            processed[iPoint, iTrace, :] = _demod_trace(t, rf_signal, freq_translations[iTrace])

    # Restore original shape
    processed = processed.reshape(original_shape)

    # Squeeze singleton leading dimension
    if processed.ndim == 3 and processed.shape[0] == 1:
        processed = processed[0]
    if processed.ndim == 2 and processed.shape[0] == 1 and n_det == 1:
        processed = processed[0]

    return processed


def _demod_trace(t: np.ndarray, signal: np.ndarray, freq_ghz: float) -> np.ndarray:
    """Demodulate a single trace at a given frequency.

    Parameters
    ----------
    t : ndarray, shape (N,)
        Time axis in microseconds.
    signal : ndarray, shape (N,)
        Complex signal trace.
    freq_ghz : float
        Down-conversion frequency in GHz.  0 means no demodulation.

    Returns
    -------
    demod : ndarray, shape (N,)
        Demodulated signal.
    """
    if freq_ghz == 0.0:
        return signal.copy()

    # Detect signal type: purely real, purely imaginary, or complex
    max_abs = np.max(np.abs(signal))
    if max_abs < 1e-30:
        return signal.copy()

    max_imag = np.max(np.abs(np.imag(signal)))
    max_real = np.max(np.abs(np.real(signal)))

    purely_imag = False
    if max_imag > 0 and max_abs / max_imag > 1e8:
        # Purely real
        rf_signal = np.real(signal)
    elif max_real > 0 and max_abs / max_real > 1e8:
        # Purely imaginary
        rf_signal = np.imag(signal)
        purely_imag = True
    else:
        # Complex (S+ or S-)  — normalize by factor of 2
        rf_signal = 2 * signal

    # Detect non-uniform time steps (breaks from undetected events)
    dt_arr = np.round(np.diff(t), decimals=10)
    dt = np.min(dt_arr) if len(dt_arr) > 0 else 1.0
    break_indices = [0] + list(np.where(dt_arr != dt)[0] + 1) + [len(t)]

    demod = np.zeros(len(signal), dtype=complex)

    for j in range(len(break_indices) - 1):
        idx = slice(break_indices[j], break_indices[j + 1])
        t_seg = t[idx]
        sig_seg = rf_signal[idx]

        if np.isrealobj(sig_seg):
            # IQ demodulation for real or imaginary signals
            _, dc = rfmixer(t_seg, sig_seg, freq_ghz, 'IQdemod', dt=dt)
            if purely_imag:
                demod[idx] = np.real(dc) * 1j
            else:
                demod[idx] = np.real(dc)
        else:
            # IQ shift for complex signals
            _, dc = rfmixer(t_seg, sig_seg, freq_ghz, 'IQshift', dt=dt)
            demod[idx] = dc

    return demod
