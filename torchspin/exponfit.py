"""Exponential fitting for relaxation data.

Port of EasySpin's ``exponfit.m``.

Fits mono- or bi-exponential decay/recovery curves, commonly used for
T1 and T2 relaxation time analysis.

Example
-------
>>> import numpy as np
>>> from torchspin.exponfit import exponfit
>>> t = np.linspace(0, 10, 200)
>>> y = 0.8 * np.exp(-t / 2.0) + 0.2  # mono-exponential decay + offset
>>> k, c, yfit = exponfit(t, y)
>>> print(f"T = {-1/k[0]:.2f} us")
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


# ``np.trapezoid`` was added after the minimum supported NumPy version.
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def exponfit(
    x: np.ndarray,
    y: np.ndarray,
    n_exp: int = 1,
    *,
    include_offset: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Fit exponential decay or recovery curves.

    Fits the model:

    .. math::

        y(x) = c_0 + c_1 \exp(k_1 x) + [c_2 \exp(k_2 x)]

    where ``k`` values are negative for decay curves.

    Parameters
    ----------
    x:
        Abscissa (time) values.
    y:
        Ordinate values.  If 2-D with shape ``(N, M)``, each column is
        fitted independently.
    n_exp:
        Number of exponential components (1 or 2).
    include_offset:
        If ``True`` (default), include a constant offset ``c_0`` in the
        model.

    Returns
    -------
    k:
        Decay rate constants (negative for decays).  Shape ``(n_exp,)``
        for single series, ``(n_exp, M)`` for multiple.
    c:
        Linear coefficients ``[c_0, c_1, ...]`` (offset first if included).
        Shape ``(n_exp + offset, )`` or ``(n_exp + offset, M)``.
    yfit:
        Fitted function values, same shape as ``y``.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64)

    if n_exp not in (1, 2):
        raise ValueError("n_exp must be 1 or 2")

    # Handle 1-D input
    squeeze = y.ndim == 1
    if squeeze:
        y = y.reshape(-1, 1)

    n_pts, n_cols = y.shape
    if n_pts != len(x):
        raise ValueError(f"x has {len(x)} points but y has {n_pts} rows")

    # Normalize x for numerical stability
    x_scale = np.max(np.abs(x))
    if x_scale == 0:
        x_scale = 1.0
    x_norm = x / x_scale

    n_coeff = n_exp + (1 if include_offset else 0)
    k_all = np.zeros((n_exp, n_cols))
    c_all = np.zeros((n_coeff, n_cols))
    yfit_all = np.zeros_like(y)

    for col in range(n_cols):
        yi = y[:, col].copy()

        # Detect recovery curves (signal increases)
        is_recovery = yi[0] < yi[-1]
        if is_recovery:
            yi = -yi

        # Initial guess for decay constant
        k_init = _guess_single_decay(x_norm, yi, include_offset)

        if n_exp == 1:
            k_start = np.array([k_init])
        else:
            k_start = np.array([0.8 * k_init, 1.3 * k_init])

        # Separable NLS: optimize k, solve c linearly
        def residuals(k_vec):
            A = _design_matrix(x_norm, k_vec, include_offset)
            c_ls, _, _, _ = np.linalg.lstsq(A, yi, rcond=None)
            return yi - A @ c_ls

        result = least_squares(
            residuals, k_start,
            method='lm',
            max_nfev=10000,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
        k_opt = result.x

        # Final linear solve
        A = _design_matrix(x_norm, k_opt, include_offset)
        c_ls, _, _, _ = np.linalg.lstsq(A, yi, rcond=None)
        yfit_col = A @ c_ls

        # Undo recovery inversion
        if is_recovery:
            c_ls = -c_ls
            yfit_col = -yfit_col

        # Rescale k back to original x units
        k_opt = k_opt / x_scale

        k_all[:, col] = k_opt
        c_all[:, col] = c_ls
        yfit_all[:, col] = yfit_col

    if squeeze:
        k_all = k_all.ravel()
        c_all = c_all.ravel()
        yfit_all = yfit_all.ravel()

    return k_all, c_all, yfit_all


def _design_matrix(x: np.ndarray, k: np.ndarray, include_offset: bool) -> np.ndarray:
    """Build the design matrix for the exponential model."""
    cols = [np.exp(ki * x) for ki in k]
    if include_offset:
        # Offset column first (matches MATLAB convention)
        cols = [np.ones_like(x)] + cols
    return np.column_stack(cols)


def _guess_single_decay(x: np.ndarray, y: np.ndarray, include_offset: bool) -> float:
    """Estimate initial decay constant from data."""
    if include_offset:
        C = y[-1]
    else:
        C = 0.0

    A = y[0] - C
    if abs(A) < 1e-30:
        return -1.0  # fallback

    # k ~ A / integral(y - C)
    integral = _trapezoid(y - C, x)
    if abs(integral) < 1e-30:
        return -1.0

    k = -abs(A / integral)
    return k
