"""Cross-term averaged FFT.

Port of EasySpin's ``ctafft.m``.

Removes dead-time artifacts from time-domain signals using cross-term
averaging of multiple FFTs starting from different indices.

Example
-------
>>> import numpy as np
>>> from torchspin.ctafft import ctafft
>>> td = np.exp(-np.arange(128) / 20) * np.cos(2 * np.pi * 0.1 * np.arange(128))
>>> fd = ctafft(td, 3)
>>> fd.shape
(128,)
"""
from __future__ import annotations

import numpy as np


def ctafft(
    td: np.ndarray,
    averages: int | list[int] | np.ndarray,
    N: int | None = None,
) -> np.ndarray:
    """Cross-term averaged FFT.

    Parameters
    ----------
    td:
        Time-domain data.  For 2-D arrays, operates along columns.
    averages:
        Starting indices for the FFTs.  If a single integer ``k``, expands
        to ``range(1, k+1)`` (1-based, matching MATLAB convention).
        Otherwise a list/array of 1-based starting indices.
    N:
        FFT length.  Defaults to the number of rows in *td*.

    Returns
    -------
    fd:
        Cross-term averaged magnitude spectrum, same shape as *td*
        (or ``(N, ncols)`` if *N* differs from input length).
    """
    td = np.asarray(td, dtype=np.complex128 if np.iscomplexobj(td) else np.float64)

    # Handle row vector → column vector (MATLAB convention)
    row_vec = (td.ndim == 2 and td.shape[0] == 1 and td.shape[1] > 1)
    squeeze = (td.ndim == 1)
    if squeeze:
        td = td[:, np.newaxis]
    elif row_vec:
        td = td.T

    npts = td.shape[0]

    if N is None:
        N = npts

    # Parse averages
    if isinstance(averages, (int, np.integer)):
        # Single integer k → 1-based indices 1..k
        avg_indices = list(range(1, int(averages) + 1))
    else:
        avg_indices = [int(a) for a in averages]

    for a in avg_indices:
        if a < 1 or not float(a).is_integer():
            raise ValueError("Averages must contain positive integers (1-based).")
        if a > npts:
            raise ValueError(
                f"Index {a} in averages exceeds data length {npts}."
            )

    # Cross-term averaging: accumulate |FFT|^2
    fd = np.zeros((N, td.shape[1]), dtype=np.float64)
    for a in avg_indices:
        # Convert 1-based to 0-based
        new_fd = np.fft.fft(td[a - 1:, :], n=N, axis=0)
        fd += np.abs(new_fd) ** 2

    fd = np.sqrt(fd / len(avg_indices))

    # Restore original shape
    if row_vec:
        fd = fd.T
    elif squeeze:
        fd = fd.ravel()

    return fd
