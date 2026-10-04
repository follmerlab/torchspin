"""Digital up- or downconversion (RF mixer).

Port of EasySpin's ``rfmixer.m``.

Mixes an input signal with a local oscillator (LO) frequency to perform
digital up- or downconversion.  Supports double-sideband (DSB),
single-sideband (SSB), and IQ modulation/demodulation.

Example
-------
>>> import numpy as np
>>> from torchspin.rfmixer import rfmixer
>>> t = np.linspace(0, 1, 10000, endpoint=False)  # µs
>>> signal = np.cos(2 * np.pi * 50e-3 * 1e3 * t)  # 50 MHz baseband
>>> t_out, s_out = rfmixer(t, signal, mwFreq=9.5, mode='DSB')
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from torchspin.fdaxis import fdaxis
from torchspin.utils import hilberttrans


def rfmixer(
    t: np.ndarray,
    signal: np.ndarray,
    mwFreq: float,
    mode: str,
    dt: float | None = None,
    oversample_factor: float = 1.25,
    bandwidth_threshold: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Digital up- or downconversion.

    Parameters
    ----------
    t:
        Time axis in microseconds.
    signal:
        Input signal.  Real for DSB/USB/LSB/IQdemod; complex (I+jQ) for
        IQmod/IQshift.
    mwFreq:
        LO frequency in GHz.  For IQ frequency shifts, the sign determines
        shift direction.
    mode:
        Mixer type: ``'DSB'``, ``'USB'``, ``'LSB'``, ``'IQmod'``,
        ``'IQdemod'``, ``'IQshift'``.
    dt:
        Output time step in microseconds.  If ``None``, determined
        automatically from Nyquist criterion.
    oversample_factor:
        Factor for oversampling in automatic time-step computation
        (default 1.25).
    bandwidth_threshold:
        Threshold for input bandwidth determination (default 0.1).

    Returns
    -------
    t_out:
        Output time axis in microseconds.
    signal_out:
        Output signal.  Real for DSB/USB/LSB/IQmod; complex for
        IQdemod/IQshift.
    """
    t = np.asarray(t, dtype=np.float64).ravel()
    signal = np.asarray(signal).ravel()

    if len(t) != len(signal):
        raise ValueError("Signal and time axis must have equal lengths.")

    mode_upper = mode.upper()

    # Validate signal type for selected mode
    if mode_upper in ('DSB', 'USB', 'LSB', 'IQDEMOD'):
        if np.iscomplexobj(signal):
            raise ValueError(
                "A real input signal is required for the selected mixer type.")
    elif mode_upper in ('IQMOD', 'IQSHIFT'):
        if not np.iscomplexobj(signal):
            signal = signal.astype(np.complex128)
    else:
        raise ValueError(
            f"Unknown mixer type '{mode}'. Use 'DSB', 'USB', 'LSB', "
            "'IQmod', 'IQdemod', or 'IQshift'.")

    real_input = np.isrealobj(signal)
    dt_in = t[1] - t[0]

    # Determine max frequency for resampling
    f = fdaxis(t)  # MHz (since t is in µs)
    signal_ft = np.fft.fftshift(np.fft.fft(signal))
    input_band = f[np.abs(signal_ft) > bandwidth_threshold * np.max(np.abs(signal_ft))]
    max_freq_in = np.max(np.abs(input_band)) if len(input_band) > 0 else 0

    mw_freq_mhz = abs(mwFreq) * 1e3  # GHz → MHz
    if mw_freq_mhz > 2 * max_freq_in:
        max_freq_out = mw_freq_mhz + max_freq_in
    else:
        max_freq_out = max(mw_freq_mhz, max_freq_in)

    nyq_dt = 1.0 / (2 * max_freq_out) if max_freq_out > 0 else dt_in

    if dt is None:
        if dt_in > nyq_dt:
            dt = 1.0 / (2 * oversample_factor * max_freq_out)
        else:
            dt = dt_in

    # Construct quadrature for real SSB/IQ demod signals via Hilbert transform
    if real_input and mode_upper in ('USB', 'LSB', 'IQDEMOD'):
        signal = hilberttrans(signal)

    # Set LO sign based on mode
    if mode_upper == 'USB':
        LO = abs(mwFreq)
    elif mode_upper in ('LSB', 'IQDEMOD'):
        LO = -abs(mwFreq)
    else:
        LO = mwFreq

    # Mix: multiply by exp(j*2π*LO*t)
    signal_mixed = signal * np.exp(2j * np.pi * LO * 1e3 * t)

    # Resample to output time grid
    t_out = np.arange(t[0], t[-1] + dt * 0.5, dt)
    # Clip to not exceed original range
    t_out = t_out[t_out <= t[-1]]

    if len(t_out) == len(t) and np.allclose(t_out, t):
        signal_out = signal_mixed
    else:
        interp_re = interp1d(t, np.real(signal_mixed), kind='cubic',
                             fill_value='extrapolate')
        if np.iscomplexobj(signal_mixed):
            interp_im = interp1d(t, np.imag(signal_mixed), kind='cubic',
                                 fill_value='extrapolate')
            signal_out = interp_re(t_out) + 1j * interp_im(t_out)
        else:
            signal_out = interp_re(t_out)

    # DSB/USB/LSB/IQmod → real output
    if mode_upper in ('DSB', 'USB', 'LSB', 'IQMOD'):
        signal_out = np.real(signal_out)

    return t_out, signal_out
