"""Linear Prediction Singular Value Decomposition.

Port of EasySpin's ``lpsvd.m``.

Estimates parameters of damped exponentials from time-domain data using
the damped exponential model::

    y(t) = sum_k amp_k * exp(1j*phase_k) * exp(t * (1j*2*pi*freq_k - damp_k))

Three algorithms are supported:

- ``'kt'``: Kumaresan-Tufts method
  (Kumaresan & Tufts, IEEE Trans. ASSP-30, 833, 1982)
- ``'ss'``: State-space method (default)
  (Kung et al., J. Opt. Soc. Am. 73, 1799, 1983;
   Barkhuijsen et al., J. Magn. Reson. 73, 553, 1987)
- ``'tls'``: Hankel Total Least Squares method
  (Van Huffel et al., J. Magn. Reson. A 110, 228, 1994)

Model order estimation via MDL or AIC:
  (Wax & Kailath, IEEE Trans. ASSP-39, 387, 1985)

Example
-------
>>> import numpy as np
>>> from torchspin.lpsvd import lpsvd
>>> t = np.linspace(0, 0.5, 256, endpoint=False)
>>> signal = 1.5 * np.exp(1j * np.pi / 4) * np.exp(t * (1j * 2 * np.pi * 50 - 5))
>>> y, params = lpsvd(signal, t)
>>> abs(params.frequency[0] - 50) < 1  # frequency recovered
True
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.linalg import svd, eig, cholesky, solve


@dataclass
class LPSVDResult:
    """Parameters estimated by LPSVD."""
    damping: np.ndarray
    frequency: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray


def lpsvd(
    data: np.ndarray,
    time: np.ndarray,
    method: str = 'ss',
    order: int | str = 'mdl',
) -> tuple[np.ndarray, LPSVDResult]:
    """Linear Prediction SVD.

    Parameters
    ----------
    data:
        Time-domain signal (1-D vector).
    time:
        Corresponding time vector (same length as *data*).
    method:
        Algorithm: ``'kt'``, ``'ss'`` (default), or ``'tls'``.
    order:
        Number of sinusoidal components, or ``'mdl'``/``'aic'`` for
        automatic estimation.

    Returns
    -------
    y:
        Predicted signal (same shape as *data*).
    params:
        :class:`LPSVDResult` with estimated damping, frequency, amplitude,
        and phase arrays.
    """
    data = np.asarray(data, dtype=np.complex128).ravel()
    time = np.asarray(time, dtype=np.float64).ravel()

    if len(data) != len(time):
        raise ValueError("Time vector must be the same length as data.")

    N = len(data)
    original_shape = data.shape

    # Normalize
    scale = np.max(np.abs(data))
    if scale == 0:
        raise ValueError("Data is all zeros.")
    dat = data / scale

    # Estimate L such that L > M (number of sinusoidal signals)
    L = int(np.floor(0.6 * N))

    # Generate the Hankel matrix (L × (N-L))
    # MATLAB: hankel(conj(dat(2:N-L+1)), conj(dat(N-L+1:N)))
    col = np.conj(dat[1:N - L + 1])
    row = np.conj(dat[N - L:N])
    nrows = len(col)
    ncols = len(row)
    A = np.empty((nrows, ncols), dtype=np.complex128)
    for i in range(nrows):
        for j in range(ncols):
            idx = i + j + 1  # 0-based index into dat (MATLAB 2-based → 0-based +1)
            A[i, j] = np.conj(dat[idx])

    U, S_vals, Vh = svd(A, full_matrices=False)
    V = Vh.conj().T
    S_diag = S_vals  # singular values as 1-D array

    # Determine model order
    if isinstance(order, str):
        M_sv = len(S_diag)
        if order == 'aic':
            criterion = np.zeros(M_sv)
            for k in range(M_sv):
                sv_tail = S_diag[k:]
                n_tail = M_sv - k
                geo_mean_log = np.sum(np.log(sv_tail))
                arith_mean_log = n_tail * np.log(np.sum(sv_tail) / n_tail)
                criterion[k] = 2 * N * (arith_mean_log - geo_mean_log) + 2 * k * (2 * M_sv - k)
            m = int(np.argmin(criterion))
        elif order == 'mdl':
            criterion = np.zeros(M_sv)
            for k in range(M_sv):
                sv_tail = S_diag[k:]
                n_tail = M_sv - k
                geo_mean_log = np.sum(np.log(sv_tail))
                arith_mean_log = n_tail * np.log(np.sum(sv_tail) / n_tail)
                criterion[k] = N * (arith_mean_log - geo_mean_log) + k * (2 * M_sv - k) * np.log(N) / 2
            m = int(np.argmin(criterion))
        else:
            raise ValueError(f"Unknown order method '{order}'. Use 'mdl' or 'aic'.")
    else:
        m = int(order)

    if m == 0:
        m = 1  # Need at least one component

    # Truncate to model order
    Um = U[:, :m]
    Sm = S_diag[:m]
    Vm = V[:, :m]

    # Compute signal poles
    if method == 'kt':
        h = -np.conj(dat[:N - L])
        b = Vm @ (np.diag(1.0 / Sm) @ (Um.conj().T @ h))
        coeffs = np.concatenate([b[::-1], [1.0]])
        s = np.roots(coeffs)
        # Keep only poles inside the unit circle
        s = s[np.abs(s) < 1]
    elif method == 'ss':
        Umt = Um[1:, :]
        Umb = Um[:-1, :]
        Zp = np.linalg.solve(Umb.conj().T @ Umb, Umb.conj().T @ Umt)
        s = eig(Zp)[0]
    elif method == 'tls':
        Umt = Um[1:, :]
        Umb = Um[:-1, :]
        _, _, Vh_aug = svd(np.hstack([Umb, Umt]), full_matrices=False)
        Vu = Vh_aug.conj().T
        Zp = -np.linalg.solve(Vu[m:2*m, m:2*m].T, Vu[:m, m:2*m].T).T
        s = eig(Zp)[0]
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'kt', 'ss', or 'tls'.")

    if len(s) == 0:
        raise RuntimeError("No prediction coefficients calculated; algorithm failed to converge.")

    # Extract damping and frequencies
    s = -np.log(s)
    dt = time[1] - time[0]
    damp = np.real(s) / dt
    freq = np.imag(s) / (2 * np.pi * dt)

    # Reject negative damping (growing exponentials)
    mask = damp > 0
    damp = damp[mask]
    freq = freq[mask]

    if len(damp) == 0:
        raise RuntimeError("All components have negative damping; no valid predictions.")

    # Sort by frequency
    sort_idx = np.argsort(freq)
    freq = freq[sort_idx]
    damp = damp[sort_idx]

    # Generate the signal matrix
    y_mat = np.exp(time[:, np.newaxis] * (-damp[np.newaxis, :] + 1j * 2 * np.pi * freq[np.newaxis, :]))

    # Calculate amplitude and phase using Cholesky decomposition
    G = y_mat.conj().T @ y_mat
    try:
        R = cholesky(G)
        a = solve(R, solve(R.conj().T, y_mat.conj().T @ data))
    except np.linalg.LinAlgError:
        # Fall back to least-squares if Cholesky fails
        a = np.linalg.lstsq(y_mat, data, rcond=None)[0]

    y = y_mat @ a
    amp = np.abs(a)
    phase = np.imag(np.log(a / amp))

    # Correct for negative amplitude
    neg_phase = phase < 0
    phase[neg_phase] = phase[neg_phase] + np.pi
    amp[neg_phase] = -amp[neg_phase]

    params = LPSVDResult(
        damping=damp,
        frequency=freq,
        amplitude=amp,
        phase=phase,
    )

    return y, params
