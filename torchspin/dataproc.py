"""Data-processing utilities for EPR spectroscopy.

Port of EasySpin's ``basecorr.m``, ``fieldmod.m``, and ``rescaledata.m``.
"""

import math
import numpy as np
from scipy.special import jv as besselj

__all__ = [
    'basecorr',
    'fieldmod',
    'rescaledata',
]


# ---------------------------------------------------------------------------
# basecorr
# ---------------------------------------------------------------------------

def basecorr(data: np.ndarray, dim, n, region=None):
    """Polynomial baseline correction.

    Parameters
    ----------
    data : array_like
        1-D or 2-D data array.
    dim : int or None
        Dimension along which to fit: 1 (columns), 2 (rows), or ``None`` for
        a 2-D polynomial surface fit.
    n : int or length-2 sequence of int
        Polynomial order (0–6).  For a 2-D fit, provide ``[n_dim1, n_dim2]``.
    region : array_like of bool, optional
        Boolean mask indicating which points to include in the fit.
        ``True`` = include, ``False`` = exclude.  Same shape as *data*
        (1-D fits) or same shape as *data* (2-D fits).

    Returns
    -------
    datacorr : ndarray
        Baseline-corrected data (same shape as *data*).
    baseline : ndarray
        Fitted baseline (same shape as *data*).

    Examples
    --------
    >>> import numpy as np
    >>> y = np.linspace(0, 1, 100) + np.random.default_rng(0).standard_normal(100) * 0.01
    >>> yc, bl = basecorr(y, 1, 1)   # linear baseline along dim 1
    """
    _MAX_ORDER = 6

    data = np.asarray(data, dtype=float)
    if data.ndim > 2 or data.size == 0:
        raise ValueError("data must be a non-empty 1D or 2D array")

    two_dim_fit = (dim is None)

    n_arr = np.atleast_1d(n)
    if np.any(n_arr < 0) or np.any(n_arr != n_arr.astype(int)) or np.any(n_arr > _MAX_ORDER):
        raise ValueError(f"Polynomial order must be integers between 0 and {_MAX_ORDER}")

    if two_dim_fit:
        if len(n_arr) != 2:
            raise ValueError("For 2D fit, n must have 2 elements")
        n0, n1 = int(n_arr[0]), int(n_arr[1])
        if n0 >= data.shape[0] or n1 >= data.shape[1]:
            raise ValueError("Polynomial order must be less than the number of points")
    else:
        if len(n_arr) != 1:
            raise ValueError("For 1D fit, n must be a single integer")
        n_val = int(n_arr[0])
        if dim not in (1, 2):
            raise ValueError("dim must be 1, 2, or None")
        fit_dim = dim - 1  # convert to 0-based

    if region is not None:
        region = np.asarray(region, dtype=bool)

    if two_dim_fit:
        baseline = _basecorr_2d(data, n0, n1, region)
    else:
        # Handle 1D case (data might be 1D or 2D)
        row_vector = False
        if data.ndim == 1:
            data2d = data[:, np.newaxis]
            if fit_dim == 1:  # dim=2 on a 1D array doesn't make sense; treat as dim=1
                pass
            row_vector = False
        elif fit_dim == 0:  # dim=1 → along rows (axis 0)
            if data.shape[0] == 1:
                row_vector = True
                data2d = data.T
            else:
                data2d = data
        else:  # dim=2 → along columns (axis 1) → transpose first
            data2d = data.T

        if region is not None:
            reg1d = region.ravel() if region.ndim > 1 else region
        else:
            reg1d = None

        x = np.linspace(-1, 1, data2d.shape[0])
        D = x[:, np.newaxis] ** np.arange(n_val + 1)  # shape (m, n+1)

        if reg1d is not None:
            p = np.linalg.lstsq(D[reg1d], data2d[reg1d], rcond=None)[0]
        else:
            p = np.linalg.lstsq(D, data2d, rcond=None)[0]
        baseline2d = D @ p

        if data.ndim == 1:
            baseline = baseline2d.ravel()
        elif fit_dim == 1:  # dim=2, undo transpose
            baseline = baseline2d.T
        else:
            baseline = baseline2d
            if row_vector:
                baseline = baseline.T

    datacorr = data - baseline
    return datacorr, baseline


def _basecorr_2d(data: np.ndarray, n0: int, n1: int, region=None):
    """2-D polynomial baseline fit."""
    r, c = data.shape
    xi = np.linspace(-1, 1, r)
    yi = np.linspace(-1, 1, c)
    xg, yg = np.meshgrid(xi, yi, indexing='ij')
    x = xg.ravel()
    y = yg.ravel()

    # Build design matrix: columns = x^i * y^j for j=0..n1, i=0..n0
    cols = []
    for j in range(n1 + 1):
        for i in range(n0 + 1):
            cols.append(x ** i * y ** j)
    D = np.column_stack(cols)  # shape (r*c, (n0+1)*(n1+1))

    z = data.ravel()
    if region is not None:
        mask = region.ravel()
        p = np.linalg.lstsq(D[mask], z[mask], rcond=None)[0]
    else:
        p = np.linalg.lstsq(D, z, rcond=None)[0]

    baseline = (D @ p).reshape(r, c)
    return baseline


# ---------------------------------------------------------------------------
# fieldmod
# ---------------------------------------------------------------------------

def fieldmod(B: np.ndarray, spc: np.ndarray, mod_amp: float, harmonic: int = 1) -> np.ndarray:
    """Simulate the effect of field modulation on an absorption EPR spectrum.

    Parameters
    ----------
    B : array_like
        Magnetic field axis (mT), uniformly spaced.
    spc : array_like
        Absorption spectrum (same length as *B*).
    mod_amp : float
        Peak-to-peak modulation amplitude (mT).  Must be > 0.
    harmonic : int, optional
        Detection harmonic (0, 1, 2, …).  Default 1.

    Returns
    -------
    spc_mod : ndarray
        Pseudo-modulated spectrum (real part, in-phase component).

    Notes
    -----
    Uses FFT-based convolution with the Jacobi–Anger Bessel function kernel
    (Kaelin & Schweiger 2003):

        ``spc_mod = IFFT( FFT(spc) · J_h(2π·Ampl·S) ) · i^h``

    where ``Ampl = mod_amp / (2 · dx)`` and ``S = k / NN`` for k = 0, …, NN-1.
    """
    B = np.asarray(B, dtype=float)
    spc = np.asarray(spc, dtype=float)

    if B.ndim != 1 or spc.ndim != 1:
        raise ValueError("B and spc must be 1D arrays")
    if len(B) != len(spc):
        raise ValueError("B and spc must have the same length")
    if mod_amp <= 0:
        raise ValueError("mod_amp must be positive")
    if harmonic < 0 or harmonic != int(harmonic):
        raise ValueError("harmonic must be a non-negative integer")

    n = len(B)
    dx = float(B[1] - B[0])
    ampl = mod_amp / 2.0 / dx  # base-to-peak amplitude in data-point units

    # FFT with zero-padding to avoid wrap-around
    NN = 2 * n + 1
    ffty = np.fft.fft(spc, n=NN)
    # Zero out negative frequencies (analytic signal)
    ffty[math.ceil(NN / 2) + 1:] = 0.0

    S = np.arange(NN) / NN
    kernel = besselj(harmonic, 2 * math.pi * ampl * S)
    y_mod = np.fft.ifft(ffty * kernel)
    y_mod = y_mod[:n]

    # Phase adjustment: multiply by i^h
    y_mod *= (1j) ** harmonic
    return np.real(y_mod)


def _besselj_t(n: int, x):
    """Bessel function J_n(x) of integer order in torch (orders 0-2 in closed form,
    higher orders by upward recurrence with a small-argument series)."""
    import torch
    x = torch.as_tensor(x)
    if n == 0:
        return torch.special.bessel_j0(x)
    if n == 1:
        return torch.special.bessel_j1(x)
    j0, j1 = torch.special.bessel_j0(x), torch.special.bessel_j1(x)
    small = x.abs() < 1e-3
    xs = torch.where(small, torch.ones_like(x), x)
    jm, jc = j0, j1
    for k in range(1, n):
        jn = 2.0 * k / xs * jc - jm           # J_{k+1} = (2k/x) J_k - J_{k-1}
        jm, jc = jc, jn
    # series for |x| -> 0: J_n(x) ≈ (x/2)^n / n!
    series = (x / 2.0) ** n / math.factorial(n)
    return torch.where(small, series, jc)


def fieldmod_t(B, spc, mod_amp, harmonic: int = 1):
    """:func:`fieldmod` in torch (differentiable in ``spc`` and ``mod_amp``): the
    same Jacobi–Anger Bessel kernel applied by FFT on the 2N+1 grid."""
    import torch
    B = torch.as_tensor(B, dtype=torch.float64)
    spc = torch.as_tensor(spc)
    if B.ndim != 1 or spc.ndim != 1 or B.numel() != spc.numel():
        raise ValueError("B and spc must be 1D arrays of the same length")
    if float(torch.as_tensor(mod_amp).detach()) <= 0:
        raise ValueError("mod_amp must be positive")
    if harmonic < 0 or harmonic != int(harmonic):
        raise ValueError("harmonic must be a non-negative integer")
    n = B.numel()
    dx = float(B[1] - B[0])
    ampl = torch.as_tensor(mod_amp, dtype=torch.float64, device=spc.device) / 2.0 / dx
    NN = 2 * n + 1
    ffty = torch.fft.fft(spc.to(torch.complex128), n=NN)
    mask = torch.ones(NN, dtype=torch.float64, device=spc.device)
    mask[math.ceil(NN / 2) + 1:] = 0.0
    ffty = ffty * mask
    S = torch.arange(NN, dtype=torch.float64, device=spc.device) / NN
    kernel = _besselj_t(int(harmonic), 2 * math.pi * ampl * S)
    y_mod = torch.fft.ifft(ffty * kernel.to(torch.complex128))[:n]
    y_mod = y_mod * (1j) ** int(harmonic)
    return y_mod.real


# ---------------------------------------------------------------------------
# rescaledata
# ---------------------------------------------------------------------------

def rescaledata(y: np.ndarray, yref=None, mode: str = 'maxabs', region=None):
    """Rescale a data vector.

    Parameters
    ----------
    y : array_like
        1-D data to rescale.
    yref : array_like, optional
        Reference vector.  Required for ``mode='lsq'``.  If length differs
        from *y*, it is resampled via linear interpolation.
    mode : str, optional
        Scaling mode:

        * ``'maxabs'`` — scale so that max(|y|) = 1 (or max(|yref|) if given)
        * ``'lsq'``    — least-squares scale: a·y ≈ yref; requires *yref*
        * ``'int'``    — normalize sum to 1
        * ``'dint'``   — normalize double integral (cumsum sum) to 1
        * ``'none'``   — no scaling (returns copy)

        Default ``'maxabs'``.
    region : array_like of bool, optional
        Mask selecting points to include in the scale computation.

    Returns
    -------
    yscaled : ndarray
        Rescaled data.
    scale : float
        Scaling factor applied (``yscaled = y * scale``).
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")

    valid_modes = ('maxabs', 'lsq', 'int', 'dint', 'none')
    if mode not in valid_modes:
        # Check for obsolete modes
        obsolete = {'lsq0', 'lsq1', 'lsq2', 'minmax'}
        if mode in obsolete:
            raise ValueError(f"Mode '{mode}' is obsolete. Use basecorr() instead.")
        raise ValueError(f"Unknown scaling mode '{mode}'. Valid: {valid_modes}")

    if mode == 'lsq' and yref is None:
        raise ValueError("mode='lsq' requires a reference vector yref")

    if yref is not None:
        yref = np.asarray(yref, dtype=float)
        if yref.ndim != 1:
            raise ValueError("yref must be a 1D array")
        # Resample yref to match length of y if needed
        if len(yref) != len(y):
            t_src = np.linspace(0, 1, len(yref))
            t_dst = np.linspace(0, 1, len(y))
            yref = np.interp(t_dst, t_src, yref)

    if region is not None:
        mask = np.asarray(region, dtype=bool).ravel()
    else:
        mask = np.ones(len(y), dtype=bool)

    # Exclude NaN
    mask = mask & ~np.isnan(y)

    if mode == 'maxabs':
        max_val = np.max(np.abs(y[mask]))
        scale = 1.0 / max_val if max_val != 0 else 1.0
        if yref is not None:
            ref_mask = mask & ~np.isnan(yref)
            ref_max = np.max(np.abs(yref[ref_mask]))
            scale *= ref_max
    elif mode == 'lsq':
        ref_mask = mask & ~np.isnan(yref)
        # Fit: yref ≈ a·y → a = (y'y)^{-1} y'yref; scale = 1/a
        a, _, _, _ = np.linalg.lstsq(
            yref[ref_mask, np.newaxis], y[ref_mask, np.newaxis], rcond=None
        )
        a = float(a[0, 0])
        scale = 1.0 / a if a != 0 else 1.0
    elif mode == 'int':
        total = np.sum(y[mask])
        scale = 1.0 / total if total != 0 else 1.0
    elif mode == 'dint':
        double_int = np.sum(np.cumsum(y[mask]))
        scale = 1.0 / double_int if double_int != 0 else 1.0
    else:  # 'none'
        scale = 1.0

    # Enforce positive scaling (never invert)
    if np.real(scale) < 0:
        scale = abs(scale)

    yscaled = y * scale
    return yscaled, float(scale)
