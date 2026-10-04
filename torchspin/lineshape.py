"""Lineshape functions for EPR spectroscopy.

Port of EasySpin's ``gaussian.m``, ``lorentzian.m``, ``voigtian.m``,
``apowin.m``, ``addnoise.m``, and ``deriv.m``.

All functions return area-normalized profiles (unit integral over the abscissa)
unless ``diff=-1`` (which returns the cumulative integral from -∞).
"""

import math

import numpy as np
from scipy.special import dawsn
from scipy.signal import fftconvolve

__all__ = [
    'gaussian',
    'lorentzian',
    'voigtian',
    'apowin',
    'addnoise',
    'deriv',
    'lshape',
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hermite_phys(x: np.ndarray, n: int) -> np.ndarray:
    """Physics Hermite polynomial of order *n* (highest coeff = 2^n).

    Recurrence: H_0=1, H_1=2x, H_n = 2x·H_{n-1} − 2(n-1)·H_{n-2}.
    """
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    h_prev2 = np.ones_like(x)
    h_prev1 = 2 * x
    for k in range(2, n + 1):
        h = 2 * x * h_prev1 - 2 * (k - 1) * h_prev2
        h_prev2, h_prev1 = h_prev1, h
    return h_prev1


def _gpoly(x: np.ndarray, n: int) -> np.ndarray:
    """G_n polynomial (Barakat 1971) for Dawson function derivatives.

    G_{-1}=0, G_0=1, G_1=2x, G_n = 2x·G_{n-1} − 2n·G_{n-2}.
    """
    if n == -1:
        return np.zeros_like(x)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    g_prev2 = np.ones_like(x)
    g_prev1 = 2 * x
    for k in range(2, n + 1):
        g = 2 * x * g_prev1 - 2 * k * g_prev2
        g_prev2, g_prev1 = g_prev1, g
    return g_prev1


def _dawson_deriv(x: np.ndarray, n: int) -> np.ndarray:
    """n-th derivative of the Dawson function F(x).

    d^n F / dx^n = (-1)^n · [H_n(x)·F(x) − G_{n-1}(x)]
    """
    return (-1) ** n * (_hermite_phys(x, n) * dawsn(x) - _gpoly(x, n - 1))


# ---------------------------------------------------------------------------
# gaussian
# ---------------------------------------------------------------------------

def gaussian(
    x: np.ndarray,
    x0: float,
    fwhm: float,
    diff: int = 0,
    phase: float = 0.0,
) -> tuple:
    """Area-normalized Gaussian absorption (and dispersion) lineshape.

    Parameters
    ----------
    x : array_like
        Abscissa values (any units).
    x0 : float
        Centre of the lineshape.
    fwhm : float
        Full width at half maximum (same units as *x*). Must be > 0.
    diff : int, optional
        Derivative order.  0 = absorption, 1 = first derivative, 2 = second,
        etc.; -1 = cumulative integral from -∞.  Default 0.
    phase : float, optional
        Phase rotation in radians; mixes absorption and dispersion.
        ``phase=π/2`` puts the dispersion into the first return value.
        Default 0.

    Returns
    -------
    ya : ndarray
        Absorption lineshape (or phase-rotated combination).
    yd : ndarray
        Dispersion lineshape (same shape as *ya*).

    Notes
    -----
    Dispersion uses the Dawson function F(x) (scipy.special.dawsn).
    Higher-order dispersion derivatives use the Barakat (1971) recurrence.
    """
    x = np.asarray(x, dtype=float)
    if not np.isscalar(x0) or not np.isreal(x0):
        raise ValueError("x0 must be a real scalar")
    if fwhm <= 0:
        raise ValueError("fwhm must be positive")
    if diff < -1 or diff != int(diff):
        raise ValueError("diff must be -1, 0, 1, 2, …")

    # sig = standard deviation; k = normalised coordinate
    sig = fwhm / math.sqrt(8 * math.log(2))   # = fwhm / (2*sqrt(2*ln2))
    k = (x - x0) / (sig * math.sqrt(2))

    if diff == -1:
        yabs = 0.5 * (1.0 + _erf_safe(k))
        ydisp = _gaussian_disp_integral(k, sig)
    else:
        n = int(diff)
        prefactor = math.sqrt(2 / math.pi) / (2 * sig)
        deriv_factor = (-1 / sig) ** n * 2 ** (-n / 2)
        yabs = prefactor * deriv_factor * _hermite_phys(k, n) * np.exp(-k ** 2)

        # dispersion
        pre_d = math.sqrt(2 / math.pi) * 2 / math.sqrt(math.pi) / (2 * sig)
        pre_d *= (1 / (math.sqrt(2) * sig)) ** n
        ydisp = pre_d * _dawson_deriv(k, n)

    if phase != 0.0:
        ya = yabs * math.cos(phase) + ydisp * math.sin(phase)
        yd = -yabs * math.sin(phase) + ydisp * math.cos(phase)
    else:
        ya = yabs
        yd = ydisp

    return ya, yd


def _erf_safe(k: np.ndarray) -> np.ndarray:
    from scipy.special import erf
    return erf(k)


def _gaussian_disp_integral(k: np.ndarray, sig: float) -> np.ndarray:
    """Dispersion integral for diff=-1 (Gaussian)."""
    # Use the relation: disp_integral ∝ integral of dawsn
    # Numerically integrate using the series expansion or just use dawsn
    # Actually for diff=-1 dispersion we need ∫_{-∞}^{k} F(t) dt
    # This is rarely used; implement via numerical cumtrapz
    from scipy.integrate import cumulative_trapezoid
    pre = math.sqrt(2 / math.pi) * 2 / math.sqrt(math.pi) / (2 * sig)
    vals = pre * dawsn(k)
    if k.ndim == 0 or len(k) < 2:
        return np.zeros_like(k)
    dx = k[1] - k[0]
    cumint = cumulative_trapezoid(vals, dx=dx, initial=0.0)
    return cumint


# ---------------------------------------------------------------------------
# lorentzian
# ---------------------------------------------------------------------------

def lorentzian(
    x: np.ndarray,
    x0: float,
    fwhm: float,
    diff: int = 0,
    phase: float = 0.0,
) -> tuple:
    """Area-normalized Lorentzian absorption (and dispersion) lineshape.

    Parameters
    ----------
    x : array_like
        Abscissa values.
    x0 : float
        Centre.
    fwhm : float
        Full width at half maximum. Must be > 0.
    diff : int, optional
        Derivative order: -1, 0, 1, or 2.  Default 0.
    phase : float, optional
        Phase rotation in radians.  Default 0.

    Returns
    -------
    ya : ndarray
        Absorption (or phase-rotated) lineshape.
    yd : ndarray
        Dispersion lineshape.
    """
    x = np.asarray(x, dtype=float)
    if fwhm <= 0:
        raise ValueError("fwhm must be positive")
    if diff not in (-1, 0, 1, 2):
        raise ValueError("diff must be -1, 0, 1, or 2 for Lorentzian")

    # gamma = distance from centre to inflection point
    gamma = fwhm / math.sqrt(3)
    pre = 2.0 / (math.pi * math.sqrt(3))
    z = (x - x0) / gamma

    denom1 = 1.0 + (4.0 / 3.0) * z ** 2
    if diff == -1:
        yabs = np.arctan(2.0 / math.sqrt(3) * z) / math.pi + 0.5
        ydisp = (1.0 / (2.0 * math.pi)) * np.log(3.0 + 4.0 * z ** 2)
    elif diff == 0:
        yabs = pre / gamma / denom1
        ydisp = pre ** 2 * math.pi / gamma * z / denom1
    elif diff == 1:
        yabs = -8.0 / 3.0 * pre / gamma ** 2 * z / denom1 ** 2
        ydisp = pre ** 2 * math.pi / gamma ** 2 * (1.0 - 4.0 / 3.0 * z ** 2) / denom1 ** 2
    else:  # diff == 2
        yabs = 8.0 / 3.0 * pre / gamma ** 3 * (4.0 * z ** 2 - 1.0) / denom1 ** 3
        ydisp = (pre ** 2 * math.pi / gamma ** 3 *
                 2.0 * (4.0 / 3.0) * z * (4.0 / 3.0 * z ** 2 - 3.0) / denom1 ** 3)

    if phase != 0.0:
        ya = yabs * math.cos(phase) + ydisp * math.sin(phase)
        yd = -yabs * math.sin(phase) + ydisp * math.cos(phase)
    else:
        ya = yabs
        yd = ydisp

    return ya, yd


# ---------------------------------------------------------------------------
# voigtian
# ---------------------------------------------------------------------------

def voigtian(
    x: np.ndarray,
    x0: float,
    fwhmGL,
    diff: int = 0,
    phase: float = 0.0,
) -> tuple:
    """Area-normalized Voigt lineshape (convolution of Gaussian and Lorentzian).

    Parameters
    ----------
    x : array_like
        Abscissa values (must be uniformly spaced for the convolution).
    x0 : float
        Centre of the lineshape.
    fwhmGL : array_like, length 2
        ``[fwhm_Gauss, fwhm_Lorentz]``.  Both ≥ 0; at least one > 0.
    diff : int, optional
        Derivative order (0, 1, 2, -1).  Default 0.
    phase : float, optional
        Phase rotation in radians.  Default 0.

    Returns
    -------
    ya : ndarray
        Absorption (or phase-rotated) Voigt profile.
    yd : ndarray
        Dispersion Voigt profile.
    """
    x = np.asarray(x, dtype=float)
    fwhmGL = np.asarray(fwhmGL, dtype=float)
    if fwhmGL.shape != (2,):
        raise ValueError("fwhmGL must have exactly two elements [fwhm_G, fwhm_L]")
    if np.any(fwhmGL < 0):
        raise ValueError("Both elements of fwhmGL must be non-negative")
    if np.all(fwhmGL == 0):
        raise ValueError("At least one element of fwhmGL must be positive")

    fwhmG, fwhmL = float(fwhmGL[0]), float(fwhmGL[1])

    # Centre of x array for the narrower lineshape (to avoid positional shift)
    x0center = x[len(x) // 2]

    # Apply diff/phase to the wider lineshape; convolve with the narrow one
    if fwhmG > fwhmL:
        ya_G, yd_G = gaussian(x, x0, fwhmG, diff, phase)
        if fwhmL > 0:
            ya_L, yd_L = lorentzian(x, x0center, fwhmL, 0, 0.0)
        do_conv = fwhmL > 0
        wider = 'G'
    else:
        ya_L, yd_L = lorentzian(x, x0, fwhmL, diff, phase)
        if fwhmG > 0:
            ya_G, yd_G = gaussian(x, x0center, fwhmG, 0, 0.0)
        do_conv = fwhmG > 0
        wider = 'L'

    if do_conv:
        dx = x[1] - x[0]
        if wider == 'G':
            ya = fftconvolve(ya_G, ya_L, mode='same') * dx
            yd = fftconvolve(yd_G, ya_L, mode='same') * dx
        else:
            ya = fftconvolve(ya_L, ya_G, mode='same') * dx
            yd = fftconvolve(yd_L, ya_G, mode='same') * dx
    else:
        if wider == 'G':
            ya, yd = ya_G, yd_G
        else:
            ya, yd = ya_L, yd_L

    return ya, yd


# ---------------------------------------------------------------------------
# apowin
# ---------------------------------------------------------------------------

def apowin(type_str: str, n: int, alpha: float = None) -> np.ndarray:
    """Apodization window.

    Parameters
    ----------
    type_str : str
        Window type (3 or 4 characters).  3-character types:

        * ``'bla'`` — Blackman
        * ``'bar'`` — Bartlett
        * ``'con'`` — Connes
        * ``'cos'`` — Cosine
        * ``'ham'`` — Hamming
        * ``'han'`` — Hann
        * ``'wel'`` — Welch

        The following require *alpha*:

        * ``'exp'`` — Exponential (alpha 2–6)
        * ``'gau'`` — Gaussian     (alpha 0.6–1.2)
        * ``'kai'`` — Kaiser       (alpha 3–9)

        Append ``'+'`` for right half only (0→1), ``'-'`` for left half
        only (-1→0).

    n : int
        Number of points.
    alpha : float, optional
        Shape parameter for ``'exp'``, ``'gau'``, ``'kai'`` windows.

    Returns
    -------
    w : ndarray, shape (n,)
        Window values normalised so that max(w) = 1.
    """
    if not isinstance(type_str, str) or len(type_str) < 3 or len(type_str) > 4:
        raise ValueError("type_str must be a 3- or 4-character string")

    # Half-window suffix
    xmin, xmax = -1.0, 1.0
    if len(type_str) == 4:
        suffix = type_str[3]
        if suffix == '+':
            xmin, xmax = 0.0, 1.0
        elif suffix == '-':
            xmin, xmax = -1.0, 0.0
        else:
            raise ValueError("4th character of type_str must be '+' or '-'")

    n = int(n)
    if n < 1:
        raise ValueError("n must be a positive integer")

    x = np.linspace(xmin, xmax, n)
    key = type_str[:3].lower()

    _needs_alpha = {'exp', 'gau', 'kai'}
    _no_alpha = {'ham', 'han', 'bla', 'bar', 'con', 'cos', 'wel'}

    if key in _needs_alpha and alpha is None:
        raise ValueError(f"Window type '{key}' requires an alpha parameter")
    if key in _no_alpha and alpha is not None:
        raise ValueError(f"Window type '{key}' does not use an alpha parameter")

    if key == 'ham':
        w = 0.54 + 0.46 * np.cos(math.pi * x)
    elif key == 'han':
        w = 0.5 + 0.5 * np.cos(math.pi * x)
    elif key == 'bla':
        w = 0.42 + 0.5 * np.cos(math.pi * x) + 0.08 * np.cos(2 * math.pi * x)
    elif key == 'bar':
        w = 1.0 - np.abs(x)
    elif key == 'con':
        w = (1.0 - x ** 2) ** 2
    elif key == 'cos':
        w = np.cos(math.pi * x / 2.0)
    elif key == 'wel':
        w = 1.0 - x ** 2
    elif key == 'exp':
        w = np.exp(-alpha * np.abs(x))
    elif key == 'gau':
        w = np.exp(-2.0 * x ** 2 / alpha ** 2)
    elif key == 'kai':
        from scipy.special import iv as besseli
        denom = besseli(0, alpha)
        w = besseli(0, alpha * np.sqrt(1.0 - x ** 2)) / denom
    else:
        raise ValueError(f"Unknown apodization window: '{key}'. "
                         "Valid types: bla, bar, con, cos, exp, gau, ham, han, kai, wel")

    # Symmetrize for full windows
    if xmax - xmin == 2.0:
        w = (w + w[::-1]) / 2.0

    w = w / np.max(w)
    return w.astype(float)


# ---------------------------------------------------------------------------
# addnoise
# ---------------------------------------------------------------------------

def addnoise(
    y: np.ndarray,
    snr: float,
    noise_model: str = 'n',
    rng=None,
) -> np.ndarray:
    """Add noise to a signal at a specified signal-to-noise ratio.

    Parameters
    ----------
    y : array_like
        Input signal.
    snr : float
        Signal-to-noise ratio (signal amplitude / noise standard deviation).
        Must be > 0.
    noise_model : str, optional
        ``'n'`` — Gaussian (normal), ``'u'`` — uniform [-0.5, 0.5],
        ``'f'`` — 1/f noise.  Default ``'n'``.
    rng : numpy.random.Generator, optional
        Random number generator for reproducibility.  If *None*, a new
        default generator is used.

    Returns
    -------
    yn : ndarray
        Noisy signal (same shape as *y*).
    """
    y = np.asarray(y, dtype=complex if np.iscomplexobj(y) else float)
    if snr <= 0:
        raise ValueError("snr must be positive")
    if noise_model not in ('n', 'u', 'f'):
        raise ValueError(f"Unknown noise_model '{noise_model}'. Use 'n', 'u', or 'f'")

    if rng is None:
        rng = np.random.default_rng()

    shape = y.shape

    def _gen_noise():
        if noise_model == 'n':
            return rng.standard_normal(shape)
        elif noise_model == 'u':
            return rng.uniform(-0.5, 0.5, shape)
        else:  # '1/f'
            return _oneoverfnoise(shape, alpha=1.0, rng=rng)

    if np.iscomplexobj(y):
        noise = _gen_noise().astype(complex) + 1j * _gen_noise()
    else:
        noise = _gen_noise()

    # Scale to unit standard deviation
    std = noise.std()
    if std > 0:
        noise /= std

    # Signal amplitude = peak-to-peak
    signal_level = float(np.max(y.real) - np.min(y.real))
    if np.iscomplexobj(y):
        signal_level = max(signal_level,
                           float(np.max(y.imag) - np.min(y.imag)))
    if signal_level <= 0:
        signal_level = 1.0

    noise_level = signal_level / snr
    return y + noise * noise_level


def _oneoverfnoise(shape, alpha: float = 1.0, rng=None) -> np.ndarray:
    """Generate 1/f^alpha noise with random phases."""
    if rng is None:
        rng = np.random.default_rng()
    n_pts = int(np.prod(shape))
    freq = np.arange(n_pts // 2 + 1, dtype=float)
    freq[0] = 1.0   # avoid 0**negative = inf; overwritten below
    amps = freq ** (-alpha / 2.0)
    amps[0] = 1.0   # DC amplitude
    phases = np.exp(2j * math.pi * rng.uniform(0, 1, size=amps.shape))
    fd = amps * phases
    if n_pts % 2 == 1:
        d = np.concatenate([fd, np.conj(fd[-1:0:-1])])
    else:
        d = np.concatenate([fd, np.conj(fd[-2:0:-1])])
    noise = np.real(np.fft.ifft(d))
    return noise.reshape(shape)


# ---------------------------------------------------------------------------
# deriv
# ---------------------------------------------------------------------------

def deriv(y_or_x: np.ndarray, y: np.ndarray = None) -> np.ndarray:
    """Numerical derivative using central differences.

    Parameters
    ----------
    y_or_x : array_like
        Either *y* (if *y* is not provided), or the abscissa *x*.
    y : array_like, optional
        Ordinate values.  If *None*, *y_or_x* is treated as *y* with
        unit spacing.

    Returns
    -------
    dydx : ndarray
        Derivative, same shape as *y*.

    Notes
    -----
    Uses first differences averaged at interior points, with one-sided
    differences at the endpoints (identical to EasySpin's ``deriv.m``).
    """
    if y is None:
        y_arr = np.asarray(y_or_x, dtype=float)
        x_arr = np.arange(1, len(y_arr) + 1, dtype=float)
    else:
        x_arr = np.asarray(y_or_x, dtype=float)
        y_arr = np.asarray(y, dtype=float)

    row_vector = y_arr.ndim == 1 and y_arr.shape[0] == y_arr.size
    if y_arr.ndim == 1:
        y_arr = y_arr[:, np.newaxis]
        x_arr = x_arr.ravel()

    # Finite differences
    dy = np.diff(y_arr, axis=0)
    dx = np.diff(x_arr)[:, np.newaxis]
    slopes = dy / dx

    # Average neighbouring slopes (central difference at interior points)
    # endpoints use one-sided
    dydx = (np.concatenate([slopes[:1], slopes], axis=0) +
            np.concatenate([slopes, slopes[-1:]], axis=0)) / 2.0

    if row_vector:
        dydx = dydx.ravel()

    return dydx


# ---------------------------------------------------------------------------
# lshape — combined Gaussian/Lorentzian lineshape
# ---------------------------------------------------------------------------

def lshape(
    x,
    x0: float = 0.0,
    fwhm=1.0,
    diff: int = 0,
    alpha: float = 1.0,
    phase: float = 0.0,
):
    """Combined Gaussian + Lorentzian absorption lineshape.

    Port of EasySpin's ``lshape.m``.

    Parameters
    ----------
    x : array_like
        Abscissa values (any units).
    x0 : float
        Centre of the lineshape.  Default 0.
    fwhm : float or [float, float]
        Full width at half maximum (same units as *x*).

        * Scalar: the same FWHM is used for both Gaussian and Lorentzian
          components (the blend controlled by *alpha*).
        * Two-element list ``[fwhm_G, fwhm_L]``: Gaussian and Lorentzian
          each have their own width (independent of *alpha*).

    diff : int
        Derivative order.

        *  0 — absorption (default)
        *  1 — first derivative
        *  2 — second derivative
        * -1 — integral (cumulative absorption from -∞)

    alpha : float
        Gaussian fraction (0 ≤ alpha ≤ 1).  ``alpha=1`` → pure Gaussian;
        ``alpha=0`` → pure Lorentzian.  Ignored when *fwhm* has two elements.
    phase : float
        Phase angle (radians) for mixing absorption and dispersion
        components.  0 → pure absorption; π/2 → pure dispersion.

    Returns
    -------
    y : ndarray
        Lineshape values.  Area-normalised when ``diff=0`` and ``phase=0``.

    Notes
    -----
    Dispersion components are computed analytically:

    * Gaussian dispersion: Dawson integral ``F(t) = e^{-t²} ∫₀ᵗ e^{s²} ds``
      via ``scipy.special.dawsn``.
    * Lorentzian dispersion: ``-(x-x0) / (π * ((x-x0)² + (Γ/2)²))``.

    Normalization: the absorption peak integrates to 1 (unit area).

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(-5, 5, 1000)
    >>> y = lshape(x, 0, 1.0)            # Gaussian FWHM=1, pure absorption
    >>> print(np.trapezoid(y, x))        # ≈ 1.0
    >>> y1 = lshape(x, 0, 1.0, diff=1)  # first derivative
    """
    x = np.asarray(x, dtype=float)
    fwhm = np.asarray(fwhm, dtype=float)

    two_widths = fwhm.ndim > 0 and fwhm.size == 2
    if two_widths:
        fwhm_G, fwhm_L = float(fwhm[0]), float(fwhm[1])
        alpha = 1.0  # blend not used; separate widths
    else:
        w = float(fwhm)
        fwhm_G = w
        fwhm_L = w

    # --- Gaussian component ---
    def _g_abs(xx, fw):
        """Area-normalised Gaussian absorption."""
        if fw == 0:
            return np.where(xx == 0, np.inf, 0.0)
        sigma = fw / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        return np.exp(-xx**2 / (2.0 * sigma**2)) / (sigma * math.sqrt(2.0 * math.pi))

    def _g_disp(xx, fw):
        """Normalised Gaussian dispersion (Hilbert transform of _g_abs)."""
        if fw == 0:
            return np.zeros_like(xx)
        sigma = fw / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        t = xx / (sigma * math.sqrt(2.0))
        # H[G](x) = -sqrt(2/π) / sigma * dawsn(t)
        return -math.sqrt(2.0 / math.pi) / sigma * dawsn(t)

    def _l_abs(xx, fw):
        """Area-normalised Lorentzian absorption."""
        if fw == 0:
            return np.where(xx == 0, np.inf, 0.0)
        gamma = fw / 2.0
        return (gamma / math.pi) / (xx**2 + gamma**2)

    def _l_disp(xx, fw):
        """Normalised Lorentzian dispersion."""
        if fw == 0:
            return np.zeros_like(xx)
        gamma = fw / 2.0
        return -xx / (math.pi * (xx**2 + gamma**2))

    xx = x - x0

    if two_widths:
        # Each component uses its own width; alpha is the Gaussian fraction
        # for the AMPLITUDE (not set by the caller when two_widths) so use 0.5/0.5
        g_abs = _g_abs(xx, fwhm_G) if fwhm_G > 0 else np.zeros_like(xx)
        l_abs = _l_abs(xx, fwhm_L) if fwhm_L > 0 else np.zeros_like(xx)
        g_disp = _g_disp(xx, fwhm_G) if fwhm_G > 0 else np.zeros_like(xx)
        l_disp = _l_disp(xx, fwhm_L) if fwhm_L > 0 else np.zeros_like(xx)
        abs_part = 0.5 * g_abs + 0.5 * l_abs
        disp_part = 0.5 * g_disp + 0.5 * l_disp
    else:
        g_abs = _g_abs(xx, fwhm_G)
        l_abs = _l_abs(xx, fwhm_L)
        g_disp = _g_disp(xx, fwhm_G)
        l_disp = _l_disp(xx, fwhm_L)
        abs_part = alpha * g_abs + (1.0 - alpha) * l_abs
        disp_part = alpha * g_disp + (1.0 - alpha) * l_disp

    # Phase mixing: y = cos(phase)*absorption + sin(phase)*dispersion
    cp, sp = math.cos(phase), math.sin(phase)
    y0 = cp * abs_part + sp * disp_part

    if diff == 0:
        return y0
    elif diff == -1:
        # Cumulative integral (Simpson or trapezoidal)
        dx = np.diff(x)
        if dx.size > 0:
            dy = np.concatenate([[0.0], np.cumsum(0.5 * (y0[:-1] + y0[1:]) * dx)])
        else:
            dy = np.zeros_like(y0)
        return dy
    elif diff == 1:
        return deriv(x, y0)
    elif diff == 2:
        return deriv(x, deriv(x, y0))
    else:
        raise ValueError(f"diff must be -1, 0, 1, or 2; got {diff}")
