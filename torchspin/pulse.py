"""Shaped pulse definition for pulsed EPR.

Port of EasySpin's ``pulse.m``.

Generates IQ waveforms for common amplitude- and frequency-modulated
pulse shapes used in pulsed EPR spectroscopy.

Example
-------
>>> from torchspin.pulse import pulse
>>> par = dict(tp=0.200, Type='sech/tanh', Frequency=[-50, 50],
...            beta=10, Flip=3.14159)
>>> t, IQ, mod = pulse(par)
"""
from __future__ import annotations

import os
import numpy as np
from scipy.interpolate import interp1d

from torchspin.fdaxis import fdaxis


# NumPy 2.0 removed np.trapz; use np.trapezoid with fallback for NumPy 1.x.
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")


# ============================================================================
# Predefined pulse shape coefficients
# ============================================================================

def _load_gaussian_cascade_coefficients() -> dict:
    """Load Gaussian cascade coefficients from data file."""
    path = os.path.join(os.path.dirname(__file__), 'data',
                        'GaussianCascadeCoefficients.txt')
    if os.path.exists(path):
        return _parse_gaussian_cascade_file(path)
    # Fallback: hardcoded from EasySpin
    return _GAUSSIAN_CASCADE_BUILTIN


def _parse_gaussian_cascade_file(path: str) -> dict:
    """Parse EasySpin GaussianCascadeCoefficients.txt format."""
    result = {}
    with open(path, encoding='latin-1') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Look for pulse name in comment headers
        if line.startswith('%') and not line.startswith('%=') and not line.startswith('%-'):
            for name in ('G3', 'G4', 'Q3', 'Q5'):
                if name in line and line.index(name) < 10:
                    # Skip comment lines until data
                    i += 1
                    while i < len(lines) and lines[i].strip().startswith('%'):
                        i += 1
                    # Next 3 lines are x0, A0, FWHM
                    if i + 2 < len(lines):
                        x0 = np.array([float(x) for x in lines[i].split()])
                        A0 = np.array([float(x) for x in lines[i + 1].split()])
                        FWHM = np.array([float(x) for x in lines[i + 2].split()])
                        result[name.lower()] = {'x0': x0, 'A0': A0, 'FWHM': FWHM}
                        i += 2
                    break
        i += 1
    return result


# Hardcoded fallback
_GAUSSIAN_CASCADE_BUILTIN = {
    'g3': {
        'x0': np.array([0.287, 0.508, 0.795]),
        'A0': np.array([-1.0, 1.37, 0.49]),
        'FWHM': np.array([0.189, 0.183, 0.243]),
    },
    'g4': {
        'x0': np.array([0.177, 0.492, 0.653, 0.892]),
        'A0': np.array([0.62, 0.72, -0.91, -0.33]),
        'FWHM': np.array([0.172, 0.129, 0.119, 0.139]),
    },
    'q3': {
        'x0': np.array([0.306, 0.545, 0.804]),
        'A0': np.array([-4.39, 4.57, 2.60]),
        'FWHM': np.array([0.180, 0.183, 0.245]),
    },
    'q5': {
        'x0': np.array([0.162, 0.307, 0.497, 0.525, 0.803]),
        'A0': np.array([-1.48, -4.34, 7.33, -2.30, 5.66]),
        'FWHM': np.array([0.186, 0.139, 0.143, 0.290, 0.137]),
    },
}


def _load_fourier_series_coefficients() -> dict:
    """Load Fourier series coefficients from data file."""
    path = os.path.join(os.path.dirname(__file__), 'data',
                        'FourierSeriesCoefficients.txt')
    if os.path.exists(path):
        return _parse_fourier_series_file(path)
    return _FOURIER_SERIES_BUILTIN


def _parse_fourier_series_file(path: str) -> dict:
    """Parse EasySpin FourierSeriesCoefficients.txt format."""
    result = {}
    names = ['i-burp 1', 'i-burp 2', 'e-burp 1', 'e-burp 2',
             'u-burp', 're-burp', 'snob i2', 'snob i3']
    with open(path, encoding='latin-1') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('%') and not line.startswith('%=') and not line.startswith('%-'):
            low = line.lower()
            for name in names:
                if name in low:
                    # Skip to data
                    i += 1
                    while i < len(lines) and lines[i].strip().startswith('%'):
                        i += 1
                    if i + 2 < len(lines):
                        A0 = float(lines[i].strip())
                        An = np.array([float(x) for x in lines[i + 1].split()])
                        Bn = np.array([float(x) for x in lines[i + 2].split()])
                        result[name] = {'A0': A0, 'An': An, 'Bn': Bn}
                        i += 2
                    break
        i += 1
    return result


# Hardcoded fallback
_FOURIER_SERIES_BUILTIN = {
    'i-burp 1': {
        'A0': 0.5,
        'An': np.array([0.70, -0.15, -0.94, 0.11, -0.02, -0.04, 0.01, -0.02, -0.01]),
        'Bn': np.array([-1.54, 1.01, -0.24, -0.04, 0.08, -0.04, -0.01, 0.01, -0.01]),
    },
    'i-burp 2': {
        'A0': 0.5,
        'An': np.array([0.81, 0.07, -1.25, -0.24, 0.07, 0.11, 0.05, -0.02, -0.03, -0.02, 0.00]),
        'Bn': np.array([-0.68, -1.38, 0.20, 0.45, 0.23, 0.05, -0.04, -0.04, 0.00, 0.01, 0.01]),
    },
    'e-burp 1': {
        'A0': 0.23,
        'An': np.array([0.88, -1.04, -0.24, 0.14, 0.03, 0.04, -0.03, 0.00]),
        'Bn': np.array([-0.40, -1.42, 0.77, 0.06, 0.03, -0.04, -0.02, 0.01]),
    },
    'e-burp 2': {
        'A0': 0.26,
        'An': np.array([0.91, 0.29, -1.28, -0.05, 0.04, 0.02, 0.06, 0.00, -0.02, 0.00]),
        'Bn': np.array([-0.16, -1.82, 0.18, 0.42, 0.07, 0.07, -0.01, -0.04, 0.00, 0.00]),
    },
    'u-burp': {
        'A0': 0.27,
        'An': np.array([-1.42, -0.37, -1.84, 4.40, -1.19, 0.00, -0.37, 0.50, -0.31,
                         0.18, -0.21, 0.23, -0.12, 0.07, -0.06, 0.06, -0.04, 0.03, -0.02, 0.02]),
        'Bn': np.zeros(20),
    },
    're-burp': {
        'A0': 0.49,
        'An': np.array([-1.02, 1.11, -1.57, 0.83, -0.42, 0.26, -0.16, 0.10, -0.07,
                         0.04, -0.03, 0.01, -0.02, 0.00, -0.01]),
        'Bn': np.zeros(15),
    },
    'snob i2': {
        'A0': 0.5,
        'An': np.array([-0.2687, -0.2972, 0.0989, -0.0010, -0.0168, 0.0009, -0.0017, -0.0013, -0.0014]),
        'Bn': np.array([-1.1461, 0.4016, 0.0736, -0.0307, 0.0079, 0.0062, 0.0003, -0.0002, 0.0009]),
    },
    'snob i3': {
        'A0': 0.5,
        'An': np.array([0.2801, -0.9995, 0.1928, 0.0967, -0.0480, -0.0148, 0.0088, -0.0002, -0.0030]),
        'Bn': np.array([-1.1990, 0.4893, 0.2439, -0.0816, -0.0409, 0.0234, 0.0036, -0.0042, 0.0001]),
    },
}

# Lazy-loaded caches
_gc_cache: dict | None = None
_fs_cache: dict | None = None


def _get_gc_coeffs() -> dict:
    global _gc_cache
    if _gc_cache is None:
        _gc_cache = _load_gaussian_cascade_coefficients()
    return _gc_cache


def _get_fs_coeffs() -> dict:
    global _fs_cache
    if _fs_cache is None:
        _fs_cache = _load_fourier_series_coefficients()
    return _fs_cache


# ============================================================================
# AM functions
# ============================================================================

def _am_rectangular(t, ti, tp, par):
    return np.ones_like(t)


def _am_gaussian(t, ti, tp, par):
    tFWHM = par.get('tFWHM')
    if tFWHM is None:
        trunc = par.get('trunc')
        if trunc is None:
            raise ValueError("Gaussian AM requires 'tFWHM' or 'trunc'.")
        tFWHM = np.sqrt(-(tp ** 2) / np.log2(trunc)) if trunc > 0 else tp * 0.001
    return np.exp(-(4 * np.log(2) * ti ** 2) / tFWHM ** 2)


def _am_sinc(t, ti, tp, par):
    zc = par.get('zerocross')
    if zc is None:
        raise ValueError("Sinc AM requires 'zerocross'.")
    x = 2 * np.pi * ti / zc
    A = np.where(np.abs(x) < 1e-12, 1.0, np.sin(x) / x)
    return A / np.max(A)


def _am_halfsin(t, ti, tp, par):
    return np.cos(np.pi * ti / tp)


def _am_quartersin(t, ti, tp, par):
    trise = par.get('trise')
    if trise is None:
        raise ValueError("Quartersin AM requires 'trise'.")
    dt = t[1] - t[0]
    A = np.ones_like(t)
    if trise != 0 and 2 * trise < tp:
        tpartial = np.arange(0, trise + dt * 0.5, dt)
        npts = len(tpartial)
        A[:npts] = np.sin(tpartial * (np.pi / (2 * trise)))
        A[-npts:] = A[npts - 1::-1]
    return A


def _am_tanh2(t, ti, tp, par):
    trise = par.get('trise')
    if trise is None:
        raise ValueError("tanh2 AM requires 'trise'.")
    return (1 / np.tanh(tp / 2 / trise)) ** 4 * \
           np.tanh(t / trise) ** 2 * np.tanh((tp - t) / trise) ** 2


def _am_sech(t, ti, tp, par):
    beta = par.get('beta')
    if beta is None:
        raise ValueError("Sech AM requires 'beta'.")
    n_vals = par.get('n', [1])
    if isinstance(n_vals, (int, float)):
        n_vals = [int(n_vals)]
    n_vals = [int(v) for v in n_vals]

    if len(n_vals) == 1:
        n = n_vals[0]
        if n == 1:
            return 1.0 / np.cosh(beta * ti / tp)
        else:
            return 1.0 / np.cosh(beta * 0.5 * (2 * ti / tp) ** n)
    else:
        # Asymmetric sech
        A = np.empty_like(t)
        left = ti < 0
        A[left] = 1.0 / np.cosh(beta * 0.5 * (2 * ti[left] / tp) ** n_vals[0])
        A[~left] = 1.0 / np.cosh(beta * 0.5 * (2 * ti[~left] / tp) ** n_vals[1])
        return A


def _am_wurst(t, ti, tp, par):
    nwurst = par.get('nwurst')
    if nwurst is None:
        raise ValueError("WURST AM requires 'nwurst'.")
    return 1 - np.abs(np.sin(np.pi * ti / tp)) ** nwurst


def _am_gaussian_cascade(t, ti, tp, par, name=None):
    A0 = par.get('A0')
    x0 = par.get('x0')
    FWHM = par.get('FWHM')
    if A0 is None or x0 is None or FWHM is None:
        if name is None:
            raise ValueError("GaussianCascade requires 'A0', 'x0', 'FWHM'.")
        coeffs = _get_gc_coeffs().get(name)
        if coeffs is None:
            raise ValueError(f"Unknown Gaussian cascade preset '{name}'.")
        A0, x0, FWHM = coeffs['A0'], coeffs['x0'], coeffs['FWHM']
    else:
        A0 = np.asarray(A0, dtype=float)
        x0 = np.asarray(x0, dtype=float)
        FWHM = np.asarray(FWHM, dtype=float)

    A = np.zeros_like(t)
    for j in range(len(A0)):
        A += A0[j] * np.exp(-(4 * np.log(2) / (FWHM[j] * tp) ** 2) *
                            (t - x0[j] * tp) ** 2)
    return A / np.max(np.abs(A))


def _am_fourier_series(t, ti, tp, par, name=None):
    A0_val = par.get('A0')
    An = par.get('An')
    Bn = par.get('Bn')
    if A0_val is None or An is None or Bn is None:
        if name is None:
            raise ValueError("FourierSeries requires 'A0', 'An', 'Bn'.")
        coeffs = _get_fs_coeffs().get(name)
        if coeffs is None:
            raise ValueError(f"Unknown Fourier series preset '{name}'.")
        A0_val, An, Bn = coeffs['A0'], coeffs['An'], coeffs['Bn']
    else:
        An = np.asarray(An, dtype=float)
        Bn = np.asarray(Bn, dtype=float)

    A = np.full_like(t, float(A0_val))
    for j in range(len(An)):
        A += An[j] * np.cos((j + 1) * 2 * np.pi * t / tp) + \
             Bn[j] * np.sin((j + 1) * 2 * np.pi * t / tp)
    return A / np.max(np.abs(A))


# Map of AM function name → handler
_GAUSSIAN_CASCADE_NAMES = {'gaussiancascade', 'g3', 'g4', 'q3', 'q5'}
_FOURIER_SERIES_NAMES = {'fourierseries', 'i-burp 1', 'i-burp 2',
                         'e-burp 1', 'e-burp 2', 'u-burp', 're-burp',
                         'snob i2', 'snob i3'}

_AM_FUNCS = {
    'rectangular': _am_rectangular,
    'gaussian': _am_gaussian,
    'sinc': _am_sinc,
    'halfsin': _am_halfsin,
    'quartersin': _am_quartersin,
    'tanh2': _am_tanh2,
    'sech': _am_sech,
    'wurst': _am_wurst,
}


def _compute_am(name: str, t, ti, tp, par):
    """Dispatch to the correct AM function."""
    if name in _AM_FUNCS:
        return _AM_FUNCS[name](t, ti, tp, par)
    elif name in _GAUSSIAN_CASCADE_NAMES:
        preset = name if name != 'gaussiancascade' else None
        return _am_gaussian_cascade(t, ti, tp, par, name=preset)
    elif name in _FOURIER_SERIES_NAMES:
        preset = name if name != 'fourierseries' else None
        return _am_fourier_series(t, ti, tp, par, name=preset)
    else:
        raise ValueError(f"Unknown AM function '{name}'.")


# ============================================================================
# FM functions
# ============================================================================

def _fm_none(t, ti, tp, par):
    n = len(t)
    return np.zeros(n), np.zeros(n)


def _fm_linear(t, ti, tp, par):
    freq = par['Frequency']
    k = (freq[1] - freq[0]) / tp
    freq_mod = k * ti
    phase_mod = 2 * np.pi * ((k / 2) * ti ** 2)
    return freq_mod, phase_mod


def _fm_tanh(t, ti, tp, par):
    freq = par['Frequency']
    beta = par['beta']
    BWinf = (freq[1] - freq[0]) / np.tanh(beta / 2)
    freq_mod = (BWinf / 2) * np.tanh((beta / tp) * ti)
    phase_mod = (BWinf / 2) * (tp / beta) * np.log(np.cosh((beta / tp) * ti))
    phase_mod = 2 * np.pi * phase_mod
    return freq_mod, phase_mod


def _fm_uniformq(t, ti, tp, par, am):
    """UniformQ: FM = integral of AM²."""
    freq = par['Frequency']
    freq_mod = np.cumsum(am ** 2) * (ti[1] - ti[0]) / _trapezoid(am ** 2, ti)
    freq_mod = (freq[1] - freq[0]) * (freq_mod - 0.5)
    phase_mod = 2 * np.pi * np.cumsum(freq_mod) * (ti[1] - ti[0])
    phase_mod = phase_mod + np.abs(np.min(phase_mod))
    return freq_mod, phase_mod


# ============================================================================
# Main pulse function
# ============================================================================

def pulse(
    par: dict,
    opt: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate shaped pulse IQ waveform.

    Parameters
    ----------
    par:
        Pulse parameters dictionary with fields:

        - ``tp``: pulse length in µs (required)
        - ``Type``: pulse shape as ``'AM/FM'`` string (default ``'rectangular'``)
        - ``Flip``: flip angle in radians (default π)
        - ``Amplitude``: pulse amplitude in MHz
        - ``Qcrit``: critical adiabaticity (frequency-swept pulses)
        - ``Frequency``: center frequency or ``[f_start, f_end]`` in MHz
        - ``Phase``: phase offset in radians (default 0)
        - ``TimeStep``: time step in µs (auto if omitted)
        - ``I``, ``Q``, ``IQ``: user-defined waveform data

        Shape-specific parameters (see EasySpin documentation):

        - ``tFWHM``, ``trunc`` (gaussian)
        - ``zerocross`` (sinc)
        - ``trise`` (quartersin, tanh2)
        - ``beta``, ``n`` (sech, tanh FM)
        - ``nwurst`` (WURST)
        - ``A0``, ``x0``, ``FWHM`` (GaussianCascade)
        - ``A0``, ``An``, ``Bn`` (FourierSeries)

    opt:
        Options dictionary:

        - ``OverSampleFactor``: oversampling factor (default 10)

    Returns
    -------
    t:
        Time axis in µs.
    IQ:
        Complex IQ waveform.
    modulation:
        Dictionary with ``'A'`` (amplitude in MHz), ``'freq'`` (frequency in
        MHz), and ``'phase'`` (phase in radians) modulation functions.
    """
    if opt is None:
        opt = {}

    tp = par.get('tp')
    if tp is None:
        raise ValueError("par['tp'] (pulse length, in µs) is required.")

    oversample = opt.get('OverSampleFactor', 10)

    # Check for user-provided IQ data
    user_IQ = par.get('IQ')
    user_I = par.get('I')
    user_Q = par.get('Q')
    has_user_iq = (user_IQ is not None or user_I is not None or
                   user_Q is not None)

    if has_user_iq:
        return _pulse_user_iq(par, tp)

    # Parse pulse type
    pulse_type = par.get('Type')
    if pulse_type is None:
        raise ValueError("par['Type'] (pulse shape) is required. "
                         "Example: par['Type'] = 'sech/tanh'")

    shape_parts = pulse_type.split('/')
    if len(shape_parts) == 1:
        am_names = [s.strip().lower() for s in shape_parts[0].split('*')]
        fm_name = 'none'
    elif len(shape_parts) == 2:
        am_names = [s.strip().lower() for s in shape_parts[0].split('*')]
        fm_name = shape_parts[1].strip().lower() if shape_parts[1].strip() else 'none'
    else:
        raise ValueError("par['Type'] must be 'AM' or 'AM/FM'.")

    # Defaults
    frequency = par.get('Frequency', 0)
    if isinstance(frequency, (int, float)):
        frequency = [frequency]
    frequency = list(frequency)
    phase_offset = par.get('Phase', 0.0)

    # Validate frequency range for FM pulses
    if fm_name != 'none':
        if len(frequency) != 2 or frequency[0] == frequency[1]:
            raise ValueError(
                f"FM function '{fm_name}' requires a frequency range "
                "in par['Frequency'] = [f_start, f_end] (MHz).")

    # Compute time step
    time_step = par.get('TimeStep')
    if time_step is None:
        time_step = _estimate_timestep(par, am_names, fm_name, frequency,
                                       tp, oversample)

    t = np.arange(0, tp + time_step * 0.5, time_step)
    # Ensure last point is exactly tp
    if t[-1] > tp:
        t = t[:-1]
    ti = t - tp / 2
    nPoints = len(t)

    # Compute AM
    am = np.ones(nPoints)
    for name in am_names:
        am = am * _compute_am(name, t, ti, tp, par)

    # Compute FM
    if fm_name == 'none':
        freq_mod, phase_mod = _fm_none(t, ti, tp, par)
    elif fm_name == 'linear':
        freq_mod, phase_mod = _fm_linear(t, ti, tp, par)
    elif fm_name == 'tanh':
        freq_mod, phase_mod = _fm_tanh(t, ti, tp, par)
    elif fm_name == 'uniformq':
        freq_mod, phase_mod = _fm_uniformq(t, ti, tp, par, am)
    else:
        raise ValueError(f"Unknown FM function '{fm_name}'.")

    # Determine amplitude
    amplitude = par.get('Amplitude')
    qcrit = par.get('Qcrit')
    flip = par.get('Flip')

    if amplitude is None:
        if fm_name == 'none':
            # AM-only: amplitude from flip angle
            if flip is None:
                flip = np.pi
            amplitude = flip / (2 * np.pi * _trapezoid(am, t))
        else:
            # FM pulse: from Qcrit or Flip
            if qcrit is None:
                if flip is None:
                    flip = np.pi
                qcrit = (2 / np.pi) * np.log(2 / (1 + np.cos(flip)))
                qcrit = min(qcrit, 5.0)

            # Compute sweep rate
            if fm_name == 'linear':
                sweeprate = abs(frequency[1] - frequency[0]) / tp
            elif fm_name == 'tanh':
                BWinf = (frequency[1] - frequency[0]) / np.tanh(par['beta'] / 2)
                sweeprate = par['beta'] * abs(BWinf) / (2 * tp)
            elif fm_name == 'uniformq':
                idx_center = np.argmin(np.abs(ti))
                dt = t[1] - t[0]
                dnu = np.abs(np.diff(2 * np.pi * freq_mod / dt))
                sweeprate = dnu[idx_center] / (2 * np.pi * am[idx_center] ** 2)
            else:
                sweeprate = abs(frequency[1] - frequency[0]) / tp

            amplitude = np.sqrt(2 * np.pi * qcrit * sweeprate) / (2 * np.pi)

    # Build modulation output
    modulation = {
        'A': amplitude * am,
        'freq': freq_mod.copy(),
        'phase': phase_mod.copy(),
    }

    # Build IQ
    center_freq = np.mean(frequency)
    total_phase = phase_mod + 2 * np.pi * center_freq * t + phase_offset
    IQ = amplitude * am * np.exp(1j * total_phase)

    return t, IQ, modulation


def _pulse_user_iq(par: dict, tp: float):
    """Handle user-provided IQ data."""
    user_IQ = par.get('IQ')
    user_I = par.get('I')
    user_Q = par.get('Q')

    if user_IQ is not None:
        I_data = np.real(np.asarray(user_IQ, dtype=complex))
        Q_data = np.imag(np.asarray(user_IQ, dtype=complex))
    else:
        I_data = np.asarray(user_I, dtype=float) if user_I is not None else None
        Q_data = np.asarray(user_Q, dtype=float) if user_Q is not None else None
        if I_data is None:
            I_data = np.zeros_like(Q_data)
        if Q_data is None:
            Q_data = np.zeros_like(I_data)

    t = np.linspace(0, tp, len(I_data))
    IQ = I_data + 1j * Q_data

    time_step = par.get('TimeStep')
    if time_step is not None:
        t_new = np.arange(0, tp + time_step * 0.5, time_step)
        t_new = t_new[t_new <= tp]
        interp_re = interp1d(t, np.real(IQ), kind='cubic', fill_value='extrapolate')
        interp_im = interp1d(t, np.imag(IQ), kind='cubic', fill_value='extrapolate')
        IQ = interp_re(t_new) + 1j * interp_im(t_new)
        t = t_new

    amplitude = par.get('Amplitude')
    if amplitude is not None:
        IQ = amplitude * IQ

    modulation = {'A': None, 'freq': None, 'phase': None}
    return t, IQ, modulation


def _estimate_timestep(par, am_names, fm_name, frequency, tp, oversample):
    """Estimate appropriate time step from pulse parameters."""
    # FM bandwidth
    if fm_name != 'none' and len(frequency) == 2:
        fm_bw = abs(frequency[1] - frequency[0])
    else:
        fm_bw = 0

    # AM bandwidth from quick Fourier transform
    dt_test = 0.1e-3
    t0 = np.arange(0, tp + dt_test * 0.5, dt_test)
    ti0 = t0 - tp / 2
    A0 = np.ones_like(t0)
    for name in am_names:
        A0 = A0 * _compute_am(name, t0, ti0, tp, par)

    zf = max(1024, 4 * (2 ** int(np.ceil(np.log2(len(t0))))))
    A0ft = np.abs(np.fft.fftshift(np.fft.fft(A0, zf)))
    f = fdaxis(dt_test, zf)
    intg = np.cumsum(A0ft)
    idx_max = np.argmin(np.abs(intg - 0.5 * intg[-1]))
    above = np.where(A0ft[idx_max:] > 0.1 * np.max(A0ft))[0]
    am_bw = 2 * (f[idx_max + above[-1]] - f[idx_max]) if len(above) > 0 else 0

    bw = max(fm_bw, am_bw)
    center = np.mean(frequency) if len(frequency) > 0 else 0
    max_freq = max(abs(center + bw / 2), abs(center - bw / 2))

    if max_freq > 0:
        nyq_dt = 1 / (2 * max_freq)
        time_step = nyq_dt / oversample
    else:
        time_step = 0.002  # µs

    if time_step > tp:
        time_step = tp
    # Round to make last point fall on tp
    time_step = tp / round(tp / time_step)
    return time_step
