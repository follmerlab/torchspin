"""Resonator effect simulation and compensation for pulses.

Port of EasySpin's ``resonator.m``.

Simulates or compensates for the effect of a microwave resonator on a
pulse waveform via FFT convolution/deconvolution with the resonator
transfer function.

Example
-------
>>> import numpy as np
>>> from torchspin.resonator import resonator
>>> t = np.linspace(0, 0.1, 500)
>>> IQ = 25 * np.ones_like(t)  # rectangular pulse
>>> t_out, signal_out = resonator(t, IQ, mwFreq=9.5, nu0=9.5, QL=100,
...                                option='simulate')
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from torchspin.fdaxis import fdaxis
from torchspin.resonatorprofile import resonatorprofile
from torchspin.lineshape import apowin
from torchspin.utils import hilberttrans


def resonator(
    t0: np.ndarray,
    signal0: np.ndarray,
    mwFreq: float,
    nu0: float | np.ndarray | None = None,
    QL: float | np.ndarray | None = None,
    option: str = 'simulate',
    nu: np.ndarray | None = None,
    TransferFunction: np.ndarray | None = None,
    CutoffFactor: float = 1e-3,
    OverSampleFactor: float = 10.0,
    TimeStep: float | None = None,
    N: int = 20,
    Window: str = 'gau',
    alpha: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate or compensate for resonator effects on a pulse.

    Parameters
    ----------
    t0:
        Input time axis in µs.
    signal0:
        Input signal (real or complex).
    mwFreq:
        Microwave frequency in GHz.
    nu0:
        Resonator center frequency in GHz (for ideal transfer function).
    QL:
        Loaded Q-value (used with *nu0*).
    option:
        ``'simulate'`` or ``'compensate'``.
    nu:
        Frequency axis in GHz for experimental transfer function.
    TransferFunction:
        Experimental resonator transfer function (same length as *nu*).
    CutoffFactor:
        Pulse truncation threshold (default 1e-3).
    OverSampleFactor:
        Oversampling factor for time-step estimation (default 10).
    TimeStep:
        Output time step in µs (auto if ``None``).
    N:
        Frequency window width multiplier (default 20).
    Window:
        Apodization window type (default ``'gau'``).
    alpha:
        Apodization parameter (default 0.6).

    Returns
    -------
    t:
        Output time axis in µs.
    signal:
        Modified signal.
    """
    t0 = np.asarray(t0, dtype=np.float64).ravel()
    signal0 = np.asarray(signal0).ravel()

    if len(t0) != len(signal0):
        raise ValueError("Time axis and signal must have equal lengths.")

    is_real = np.isrealobj(signal0)

    # Get transfer function
    f_tf, H_tf = _get_transfer_function(nu0, QL, nu, TransferFunction)

    estimate_ts = TimeStep is None

    # Interpolate input to finer grid
    dt = (t0[1] - t0[0]) / 4
    t_fine = np.arange(0, t0[-1] + dt * 0.5, dt)
    if len(t_fine) < 2:
        t_fine = np.linspace(t0[0], t0[-1], max(len(t0) * 4, 4))
        dt = t_fine[1] - t_fine[0]

    interp_func = interp1d(t0, signal0, kind='cubic', fill_value='extrapolate')
    signal_fine = interp_func(t_fine)

    # Zero-pad to match transfer function length
    n_pad = max(0, (len(f_tf) - len(signal_fine)) // 2)
    signal_padded = np.concatenate([
        np.zeros(max(n_pad - 1, 0)),
        signal_fine,
        np.zeros(n_pad),
    ])

    # FFT
    FT = np.fft.ifftshift(np.fft.fft(np.fft.fftshift(signal_padded)))
    f_ = fdaxis(dt, len(FT))

    # Determine pulse frequency region
    FT_abs = np.abs(FT)
    intg = np.cumsum(FT_abs)
    idx_max = np.argmin(np.abs(intg - 0.5 * intg[-1]))
    above = np.where(FT_abs[idx_max:] > 0.01 * np.max(FT_abs))[0]
    indbw = above[-1] if len(above) > 0 else 1

    startind = idx_max - N * indbw
    endind = idx_max + N * indbw

    # Clip to valid range
    delta = max(max(endind - len(f_), 0), max(1 - startind, 0), 0)
    startind = max(startind + delta, 0)
    endind = min(endind - delta, len(f_))

    indpulse = np.arange(startind, endind)
    if len(indpulse) == 0:
        indpulse = np.arange(len(f_))

    f_pulse = f_[indpulse]
    FT_pulse = FT[indpulse]

    # Interpolate transfer function onto pulse frequency axis
    # f_tf is in MHz, f_pulse is in MHz, mwFreq is in GHz
    H_interp = interp1d(f_tf - mwFreq * 1e3, H_tf, kind='cubic',
                        fill_value=0, bounds_error=False)
    H_ = H_interp(f_pulse)

    # Apply transfer function
    if option == 'simulate':
        FTc_pulse = FT_pulse * H_
    elif option == 'compensate':
        # Avoid division by zero
        H_safe = np.where(np.abs(H_) > 1e-10, H_, 1e-10)
        FTc_pulse = FT_pulse / H_safe
    else:
        raise ValueError(f"Unknown option '{option}'. Use 'simulate' or 'compensate'.")

    # Windowing
    if Window in ('gau', 'exp', 'kai'):
        win = apowin(Window, len(FTc_pulse), alpha)
    else:
        win = apowin(Window, len(FTc_pulse))
    FTc_pulse = FTc_pulse * win

    # Inverse FFT
    FTc = np.zeros(len(FT), dtype=complex)
    FTc[indpulse] = FTc_pulse
    t_full = (np.arange(len(FTc)) - len(FTc) / 2) * dt
    signal_full = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(FTc)))

    # Extract pulse
    sig_abs = np.abs(signal_full)
    threshold = CutoffFactor * np.max(sig_abs)

    if option == 'simulate':
        above_idx = np.where(sig_abs > threshold)[0]
        if len(above_idx) > 0:
            # Start from input signal start
            start_idx = np.argmin(np.abs(t_full + t0[-1] / 2))
            end_idx = above_idx[-1]
            if start_idx > end_idx:
                start_idx = above_idx[0]
        else:
            start_idx, end_idx = 0, len(signal_full) - 1
    else:  # compensate
        above_idx = np.where(sig_abs > threshold)[0]
        if len(above_idx) > 0:
            start_idx = above_idx[0]
            end_idx = above_idx[-1]
        else:
            start_idx, end_idx = 0, len(signal_full) - 1

    t_extracted = t_full[start_idx:end_idx + 1] - t_full[start_idx]
    signal_extracted = signal_full[start_idx:end_idx + 1]

    # Determine output time step
    if estimate_ts:
        FTc_abs = np.abs(FTc)
        if np.max(FTc_abs) > 0:
            idx_m = np.argmax(FTc_abs)
            above_ftc = np.where(FTc_abs[idx_m:] > 0.05 * np.max(FTc_abs))[0]
            if len(above_ftc) > 0:
                max_freq = 2 * (f_[min(idx_m + above_ftc[-1], len(f_) - 1)] - f_[idx_m])
                if max_freq > 0:
                    nyq_dt = 1 / (2 * max_freq)
                    new_ts = nyq_dt / OverSampleFactor
                    input_ts = t0[1] - t0[0]
                    TimeStep = min(new_ts, input_ts)
                else:
                    TimeStep = t0[1] - t0[0]
            else:
                TimeStep = t0[1] - t0[0]
        else:
            TimeStep = t0[1] - t0[0]

    # Resample to output grid
    if len(t_extracted) > 1:
        t_out = np.arange(0, t_extracted[-1] + TimeStep * 0.5, TimeStep)
        t_out = t_out[t_out <= t_extracted[-1]]
        interp_out = interp1d(t_extracted, signal_extracted, kind='cubic',
                              fill_value='extrapolate')
        signal_out = interp_out(t_out)
    else:
        t_out = np.array([0.0])
        signal_out = signal_extracted[:1]

    if is_real and mwFreq == 0:
        signal_out = 2 * np.real(signal_out)

    return t_out, signal_out


def _get_transfer_function(
    nu0: float | None,
    QL: float | None,
    nu: np.ndarray | None,
    TransferFunction: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build or interpolate the resonator transfer function.

    Returns (f_MHz, H) where f is in MHz.
    """
    if nu0 is not None and QL is not None:
        # Ideal RLC transfer function
        f0 = float(nu0)  # GHz
        fmax = 2 * f0
        N = 2 ** 18
        df = fmax / N
        f = np.arange(N) * df  # GHz
        H = resonatorprofile(f, nu0=f0, Qu=float(QL),
                             mode='transferfunction')
        H[0] = H[1]
        H = H / np.max(np.real(H))
        f_mhz = f * 1e3  # GHz → MHz
        return f_mhz, H

    elif nu is not None and TransferFunction is not None:
        nu = np.asarray(nu, dtype=np.float64).ravel()
        TF = np.asarray(TransferFunction).ravel()

        fmax = 2 * np.ceil(np.max(nu))
        N = 2 ** 14
        df = fmax / N
        f = np.arange(N) * df  # GHz

        # Interpolate experimental data
        mask = (f > nu[0]) & (f < nu[-1])
        ind = np.where(mask)[0]
        if len(ind) == 0:
            raise ValueError("Frequency axis does not overlap with transfer function.")

        interp_func = interp1d(nu, TF, kind='cubic', fill_value='extrapolate')
        H_interp = interp_func(f[ind])

        H_abs = np.abs(H_interp)
        idx_max = np.argmax(H_abs)
        f0 = f[ind[0] + idx_max]

        # Estimate QL from -3dB bandwidth
        v1max = H_abs[idx_max]
        v1_3dB = 0.75 * v1max
        above_3dB = np.where(H_abs > v1_3dB)[0]
        if len(above_3dB) > 1:
            f_3dB = f[ind[above_3dB[-1]]] - f[ind[above_3dB[0]]]
            QL_est = f0 / f_3dB if f_3dB > 0 else 1000
        else:
            QL_est = 1000

        # Build ideal baseline
        H_ideal = resonatorprofile(f, nu0=f0, Qu=QL_est,
                                   mode='transferfunction')
        H_ideal = v1max * H_ideal / np.max(np.abs(H_ideal))

        # Build full transfer function with exponential extrapolation
        H_full = np.abs(H_ideal).copy()
        H_full[ind] = H_abs

        # Phase from Hilbert transform of log magnitude
        log_abs = np.log(np.maximum(np.abs(H_full), 1e-30))
        phase = -np.imag(hilberttrans(log_abs))
        H = H_full * np.exp(1j * phase)

        H = H / np.max(np.real(H))
        f_mhz = f * 1e3
        return f_mhz, H

    else:
        raise ValueError(
            "Provide either (nu0, QL) for ideal transfer function or "
            "(nu, TransferFunction) for experimental data.")
