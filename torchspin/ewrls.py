"""Exponentially weighted recursive least squares adaptive filter.

Port of EasySpin's ``ewrls.m``.

Given a series of scans, computes a filtered average using the exponentially
weighted recursive least squares (RLS) adaptive filter.

Example
-------
>>> import numpy as np
>>> from torchspin.ewrls import ewrls
>>> rng = np.random.default_rng(42)
>>> signal = np.exp(-((np.arange(200) - 100) / 20) ** 2)
>>> data = np.column_stack([signal + 0.3 * rng.standard_normal(200) for _ in range(20)])
>>> y = ewrls(data, p=20, lam=0.97)
>>> y.shape
(200,)
"""
from __future__ import annotations

import numpy as np


def ewrls(
    data: np.ndarray,
    p: int,
    lam: float,
    nPreAvg: int = 0,
    delta: float = 100.0,
    direction: str = 'fb',
) -> np.ndarray:
    """Exponentially weighted RLS adaptive filter.

    Parameters
    ----------
    data:
        2-D array where each column is one scan (shape ``(nPoints, nScans)``).
    p:
        Filter length (typically 5-50).
    lam:
        Memory factor (typically 0.96-0.99).  Must be between 0 and 1.
    nPreAvg:
        Number of scans to average for the desired signal.  If 0 (default),
        all scans are averaged.
    delta:
        Regularization parameter (default 100).
    direction:
        Filtering direction: ``'f'`` forward, ``'b'`` backward, ``'fb'``
        average of both (default).

    Returns
    -------
    y:
        Filtered averaged signal (1-D array of length ``nPoints``).
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("Data must be a 2-D array with scans as columns.")

    nPoints, nScans = data.shape

    if nScans <= 1:
        raise ValueError("Multiple scans (columns) are needed.")
    if nPoints <= 1:
        raise ValueError("Multiple points (rows) are needed.")
    if p >= nPoints:
        raise ValueError(f"Filter length ({p}) must be shorter than data length ({nPoints}).")
    if not (0 <= lam <= 1):
        raise ValueError("lambda must be between 0 and 1.")
    if p <= 0:
        raise ValueError("Filter length p must be positive.")

    if direction == 'f':
        do_forward, do_backward = True, False
    elif direction == 'b':
        do_forward, do_backward = False, True
    elif direction in ('fb', 'bf'):
        do_forward, do_backward = True, True
    else:
        raise ValueError(f"Unknown direction '{direction}'. Use 'f', 'b', or 'fb'.")

    # Desired signal: average of scans
    if nPreAvg == 0:
        d = np.mean(data, axis=1)
    else:
        nPreAvg = min(nPreAvg, nScans)
        d = np.mean(data[:, :nPreAvg], axis=1)

    start_scan = max(nPreAvg, 1)  # skip pre-averaged scans (at least start at 1)

    def _rls_filter(data_in, d_in):
        """Run RLS filter on data, return per-scan outputs."""
        y_out = np.zeros((nPoints, nScans - start_scan))
        P = delta * np.eye(p)
        w = np.zeros(p)
        scan_idx = 0
        for iScan in range(start_scan, nScans):
            for k in range(p - 1, nPoints):
                x = data_in[k - p + 1:k + 1, iScan]
                phi = P @ x
                g = phi / (lam + x @ phi)
                y_out[k, scan_idx] = x @ w
                e = d_in[k] - y_out[k, scan_idx]
                w = w + g * e
                P = (P - np.outer(g, phi)) / lam
                P = (P + P.T) / 2  # symmetrize for numerical stability
            scan_idx += 1
        return np.mean(y_out, axis=1)

    if do_forward:
        y_forward = _rls_filter(data, d)

    if do_backward:
        # Reverse data and desired signal
        data_rev = data[::-1, :].copy()
        d_rev = d[::-1].copy()
        y_backward = _rls_filter(data_rev, d_rev)[::-1]

    if do_forward and do_backward:
        return (y_forward + y_backward) / 2
    elif do_forward:
        return y_forward
    else:
        return y_backward
