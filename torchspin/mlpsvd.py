"""2D Linear Prediction Singular Value Decomposition.

Port of EasySpin's ``mlpsvd.m``.  Fits a damped exponential model to spectral
data using SVD-based methods, predicting signal parameters (amplitude, phase,
frequency, damping).

Model::

    y = amp * exp(i*phase) * exp(time * (i*2*pi*freq - damp))

References
----------
[1] Kumaresan, R.; Tufts, D.W.; IEEE Trans. Acoust. Speech Signal
    ASSP-30, 833 (1982).
[2] Kung, S.Y.; Arun, K.S.; Bhaskar Rao, D.V.; J. Opt. Soc. Am. 73,
    1799 (1983).
[3] Barkhuijsen, H.; De Beer, R.; Van Ormondt, D.; J. Mag. Reson. 73,
    553 (1987).
[4] Van Huffel, S.; Chen, H.; Decanniere, C.; Van Hecke, P.; J. Mag.
    Reson. Series A 110, 228 (1994).
[5] Wax, M.; Kailath, T.; IEEE Trans. Acoust. Speech Signal ASSP-39,
    387 (1985).  — Model order estimation.
[6] Vanhamme, L.; Van Huffel, S.; SPIE 3461, 237 (1998). — 2D methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.linalg import svd, hankel


__all__ = ['mlpsvd', 'MLPSVDResult']


@dataclass
class MLPSVDResult:
    """Result of mlpsvd analysis.

    Attributes
    ----------
    damping : ndarray, shape (M,)
        Damping rates of the M components.
    frequency : ndarray, shape (M,)
        Oscillation frequencies.
    amplitude : ndarray, shape (nSpectra, M)
        Complex amplitudes for each spectrum.
    phase : ndarray, shape (nSpectra, M)
        Phases (radians) for each spectrum.
    model : callable
        ``model(t)`` reconstructs the predicted signal at times *t*.
    """
    damping: np.ndarray
    frequency: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    model: object  # callable


def mlpsvd(
    data: np.ndarray,
    time: np.ndarray,
    method: str = 'ss',
    order: int | str = 'mdl',
    method2d: str = 'sum',
) -> tuple[np.ndarray, MLPSVDResult]:
    """Perform 2D Linear Prediction SVD on spectral data.

    Parameters
    ----------
    data : ndarray, shape (N,) or (nSpectra, N)
        Input data.  *N* must match ``len(time)``.
    time : ndarray, shape (N,)
        Time vector.
    method : {'ss', 'kt', 'tls'}
        LPSVD algorithm:

        - ``'ss'``  — state-space (default) [2, 3]
        - ``'kt'``  — Kumaresan-Tufts [1]
        - ``'tls'`` — Hankel total least squares [4]
    order : int or {'mdl', 'aic'}
        Model order (number of sinusoidal components).
        ``'mdl'`` (default) or ``'aic'`` for automatic estimation [5].
    method2d : {'sum', 'stack'}
        Method for handling multiple spectra:

        - ``'sum'``   — sum of time series for pole determination (default)
        - ``'stack'`` — stacked Hankel matrices

    Returns
    -------
    y : ndarray, same shape as *data*
        Predicted spectrum.
    parameters : MLPSVDResult
        Fitted parameters (damping, frequency, amplitude, phase, model).

    Examples
    --------
    >>> t = np.linspace(0, 1, 256)
    >>> sig = 2*np.exp(-3*t)*np.cos(2*np.pi*50*t) + np.random.randn(256)*0.1
    >>> y, params = mlpsvd(sig, t)
    >>> print(params.frequency)
    """
    data = np.asarray(data, dtype=complex)
    time = np.asarray(time, dtype=float).ravel()
    N = len(time)

    # Orient data so time runs along axis 1
    transposed = False
    if data.ndim == 1:
        data = data[np.newaxis, :]
    elif data.shape[0] == N and data.shape[1] != N:
        data = data.T
        transposed = True

    if data.shape[1] != N:
        raise ValueError(f"Time vector length ({N}) must match data dimension "
                         f"({data.shape[1]})")

    n_spectra = data.shape[0]

    # Hankel matrix size — L > M (number of signal poles)
    L = int(np.floor(0.6 * N))

    # Build Hankel matrix and compute SVD
    if method2d == 'sum':
        dat = np.sum(data, axis=0)
        dat = dat / np.max(np.abs(dat))
        A = hankel(np.conj(dat[1:N - L + 1]), np.conj(dat[N - L:N]))
        U, S, Vt = svd(A, full_matrices=False)
        V = Vt.T.conj()
    elif method2d == 'stack':
        dat_norm = data / np.max(np.abs(data))
        blocks = []
        for i in range(n_spectra):
            row = np.conj(dat_norm[i])
            H = hankel(row[1:N - L + 1], row[N - L:N])
            blocks.append(H)
        A = np.vstack(blocks)
        U, S, Vt = svd(A, full_matrices=False)
        V = Vt.T.conj()
    else:
        raise ValueError(f"method2d must be 'sum' or 'stack', got '{method2d}'")

    # Determine model order
    if isinstance(order, str):
        m = _estimate_order(S, N, order)
    else:
        m = int(order)

    if m < 1:
        m = 1
    m = min(m, len(S))

    # Truncate to model order
    Um = U[:, :m]
    Sm = S[:m]
    Vm = V[:, :m]

    # Extract signal poles
    if method == 'kt':
        dat_sum = np.sum(data, axis=0)
        dat_sum = dat_sum / np.max(np.abs(dat_sum))
        h = -np.conj(dat_sum[:N - L])
        b = Vm @ (np.diag(1.0 / Sm) @ (Um.T.conj() @ h))
        coeffs = np.concatenate([b[::-1], [1.0]])
        s = np.roots(coeffs)
        # Keep only poles inside unit circle
        s = s[np.abs(s) < 1]
    elif method == 'ss':
        Umt = Um[1:, :]
        Umb = Um[:-1, :]
        Zp = np.linalg.solve(Umb.T.conj() @ Umb, Umb.T.conj() @ Umt)
        s = np.linalg.eigvals(Zp)
    elif method == 'tls':
        Umt = Um[1:, :]
        Umb = Um[:-1, :]
        _, _, Vt_aug = svd(np.hstack([Umb, Umt]), full_matrices=False)
        V_aug = Vt_aug.T.conj()
        Zp = -np.linalg.solve(V_aug[m:2 * m, m:2 * m].T,
                               V_aug[:m, m:2 * m].T).T
        s = np.linalg.eigvals(Zp)
    else:
        raise ValueError(f"method must be 'kt', 'ss', or 'tls', got '{method}'")

    if len(s) == 0:
        raise RuntimeError("No prediction coefficients calculated; "
                           "algorithm failed to converge.")

    # Extract damping and frequency from signal poles
    s_log = -np.log(s)
    dt = time[1] - time[0]
    damp = np.real(s_log) / dt
    freq = np.imag(s_log) / (2 * np.pi * dt)

    # Reject negative damping
    mask = damp > 0
    damp = damp[mask]
    freq = freq[mask]

    if len(damp) == 0:
        raise RuntimeError("All components have negative damping; no valid model.")

    # Sort by frequency
    sort_idx = np.argsort(freq)
    freq = freq[sort_idx]
    damp = damp[sort_idx]

    # Generate basis signals
    basis = np.exp((-damp[:, np.newaxis] + 1j * 2 * np.pi * freq[:, np.newaxis])
                   * time[np.newaxis, :])  # (m, N)

    # Calculate amplitude and phase via least squares
    # a = data @ basis' / (basis @ basis')
    YYH = basis @ basis.T.conj()
    a = (data @ basis.T.conj()) @ np.linalg.inv(YYH)  # (nSpectra, m)

    # Predicted signal
    y = a @ basis  # (nSpectra, N)

    # Amplitude and phase
    amp = np.abs(a)
    phase = np.angle(a)

    # Correct for negative amplitude
    neg = phase < 0
    phase[neg] += np.pi
    amp[neg] = -amp[neg]

    # Build model function
    def model_fn(t):
        t = np.asarray(t, dtype=float).ravel()
        return (amp * np.exp(1j * phase)) @ np.exp(
            (-damp[:, np.newaxis] + 1j * 2 * np.pi * freq[:, np.newaxis])
            * t[np.newaxis, :]
        )

    params = MLPSVDResult(
        damping=damp,
        frequency=freq,
        amplitude=amp,
        phase=phase,
        model=model_fn,
    )

    # Restore original orientation
    if transposed:
        y = y.T
    if y.shape[0] == 1 and n_spectra == 1:
        y = y[0]

    return y, params


def _estimate_order(S: np.ndarray, N: int, criterion: str = 'mdl') -> int:
    """Estimate model order using MDL or AIC.

    Parameters
    ----------
    S : ndarray
        Singular values from SVD.
    N : int
        Number of time points.
    criterion : {'mdl', 'aic'}
        Information criterion.

    Returns
    -------
    m : int
        Estimated model order.
    """
    M = len(S)
    values = np.zeros(M)

    for k in range(M):
        tail = S[k:]
        n_tail = len(tail)
        sum_tail = np.sum(tail)
        # Avoid log(0)
        if sum_tail <= 0 or n_tail == 0:
            values[k] = np.inf
            continue
        mean_tail = sum_tail / n_tail
        log_sum = np.sum(np.log(np.maximum(tail, 1e-300)))
        base = n_tail * np.log(mean_tail) - log_sum

        if criterion == 'aic':
            values[k] = 2 * N * base + 2 * k * (2 * M - k)
        elif criterion == 'mdl':
            values[k] = N * base + k * (2 * M - k) * np.log(N) / 2
        else:
            raise ValueError(f"criterion must be 'mdl' or 'aic', got '{criterion}'")

    m = int(np.argmin(values))
    return m
