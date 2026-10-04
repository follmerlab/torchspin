"""Angular momentum functions for torchspin.

Port of EasySpin's:
- ``wigner3j.m``     — Wigner 3-j symbol
- ``wignerd.m``      — Wigner rotation matrix D^J_{m1,m2}
- ``spherharm.m``    — spherical harmonics Y_L^M(theta, phi)
- ``clebschgordan.m``— Clebsch-Gordan coefficient
- ``cgmatrix.m``     — CG transformation matrix (uncoupled→coupled)

All functions follow EasySpin's conventions.

Notation
--------
* Wigner 3-j symbols: ``wigner3j(j1,j2,j3,m1,m2,m3)``
* Wigner D-matrix: ``wignerd(J, alpha, beta, gamma)``
  - Default phase convention '-': D = exp(-i·m1·α) d(β) exp(-i·m2·γ)
  - Basis ordered m = +J, +J-1, ..., -J (same as EasySpin)
* Spherical harmonics: ``spherharm(L, M, theta, phi)``
  - theta = polar angle (from +z), phi = azimuthal (from +x)
  - Condon-Shortley phase included for complex harmonics
* Clebsch-Gordan: ``clebschgordan(j1,j2,j,m1,m2,m)``
  - Related to 3j by: CG = (-1)^(j1-j2+m) * sqrt(2j+1) * W3j(j1,j2,j; m1,m2,-m)
"""
from __future__ import annotations

import cmath
import math
from typing import Optional, Union

import numpy as np
import scipy.special as sps


__all__ = [
    'wigner3j',
    'wigner6j',
    'wignerd',
    'spherharm',
    'clebschgordan',
    'cgmatrix',
    'isto',
    'isto2stev',
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _logfact(n: int) -> float:
    """Natural log of n! (n must be a non-negative integer)."""
    return float(sps.gammaln(n + 1))


def _istriangle(a, b, c) -> bool:
    return (a + b >= c) and (b + c >= a) and (c + a >= b)


def _ishalfint(x) -> bool:
    return (2 * x) == int(2 * x)


# ---------------------------------------------------------------------------
# wigner3j
# ---------------------------------------------------------------------------

def wigner3j(j1, j2, j3, m1, m2, m3) -> float:
    """Wigner 3-j symbol.

    Parameters
    ----------
    j1, j2, j3 : int or half-int
        Angular momentum quantum numbers (non-negative, integer or half-integer).
    m1, m2, m3 : int or half-int
        Magnetic quantum numbers with ``|mi| <= ji``.

    Returns
    -------
    float
        Value of the Wigner 3-j symbol.

    Notes
    -----
    Uses the Racah formula with log-factorials for numerical stability
    (accurate for j up to ~100).  Returns 0 for any selection-rule violation.

    Selection rules (return 0 if violated)
    - m1 + m2 + m3 = 0
    - Triangle inequality for (j1, j2, j3)
    - |mi| <= ji
    - 2*(ji + mi) is integer (half-integer or integer quantum numbers only)
    """
    # Coerce to float
    j1, j2, j3 = float(j1), float(j2), float(j3)
    m1, m2, m3 = float(m1), float(m2), float(m3)

    # Selection rules
    if abs(m1 + m2 + m3) > 1e-10:
        return 0.0
    if not _istriangle(j1, j2, j3):
        return 0.0
    if abs(m1) > j1 + 1e-10 or abs(m2) > j2 + 1e-10 or abs(m3) > j3 + 1e-10:
        return 0.0
    for j, m in ((j1, m1), (j2, m2), (j3, m3)):
        if not _ishalfint(j) or not _ishalfint(m):
            return 0.0
        if not _ishalfint(j + m):
            return 0.0

    # Triangle coefficient (log scale)
    tri_ln = (
        _logfact(int(round(j1 + j2 - j3))) +
        _logfact(int(round(j1 - j2 + j3))) +
        _logfact(int(round(-j1 + j2 + j3))) -
        _logfact(int(round(j1 + j2 + j3 + 1)))
    )

    # Pre-factor (log scale)
    pre_ln = (
        _logfact(int(round(j1 + m1))) + _logfact(int(round(j1 - m1))) +
        _logfact(int(round(j2 + m2))) + _logfact(int(round(j2 - m2))) +
        _logfact(int(round(j3 + m3))) + _logfact(int(round(j3 - m3)))
    )

    prefactor_ln = (tri_ln + pre_ln) / 2.0

    # Racah summation
    # t runs over values making all factorials non-negative
    tmin = max(0,
               int(round(j2 - m1 - j3)),
               int(round(j1 + m2 - j3)))
    tmax = min(int(round(j1 + j2 - j3)),
               int(round(j1 - m1)),
               int(round(j2 + m2)))

    if tmin > tmax:
        return 0.0

    total = 0.0
    for t in range(tmin, tmax + 1):
        sign = (-1) ** t
        term_ln = -(
            _logfact(t) +
            _logfact(int(round(j1 + j2 - j3 - t))) +
            _logfact(int(round(j1 - m1 - t))) +
            _logfact(int(round(j2 + m2 - t))) +
            _logfact(int(round(j3 - j2 + m1 + t))) +
            _logfact(int(round(j3 - j1 - m2 + t)))
        )
        total += sign * math.exp(prefactor_ln + term_ln)

    overall_sign = (-1) ** int(round(j1 - j2 - m3))
    return overall_sign * total


# ---------------------------------------------------------------------------
# wignerd
# ---------------------------------------------------------------------------

def wignerd(J, alpha_or_beta, beta=None, gamma=None,
            phase: str = '-') -> Union[complex, np.ndarray]:
    """Wigner rotation matrix D^J (or reduced d^J).

    Parameters
    ----------
    J : float or sequence [J, m1, m2]
        Angular momentum quantum number.  If a 3-element sequence [J, m1, m2],
        only the single element D^J_{m1,m2} is returned.
    alpha_or_beta : float
        If ``beta`` is ``None``: the polar angle β only (α=γ=0, returns real d).
        Otherwise: the first Euler angle α.
    beta : float, optional
        Middle Euler angle β.
    gamma : float, optional
        Last Euler angle γ.
    phase : {'-', '+'}, optional
        Sign convention.  Default ``'-'``:
        ``D = exp(-i·m1·α) · d(β) · exp(-i·m2·γ)`` (Brink/Satchler, Varshalovich).
        ``'+'``: ``D = exp(+i·m1·α) · d(β) · exp(+i·m2·γ)`` (Edmonds).

    Returns
    -------
    ndarray, shape (2J+1, 2J+1) or complex scalar
        Full matrix (or single element).  Basis ordered m = +J, ..., -J.
        When called with β only (α=γ=0), returns a real ndarray.

    Notes
    -----
    The reduced Wigner d-matrix element is computed via the Wigner sum formula
    using log-factorials (exact for integer and half-integer J).
    """
    # Parse first argument: scalar J or [J, m1, m2]
    J_arg = np.asarray(J, dtype=float).ravel()
    if J_arg.size == 3:
        Jval, m1_req, m2_req = J_arg
        single_element = True
    elif J_arg.size == 1:
        Jval = float(J_arg[0])
        single_element = False
        m1_req = m2_req = None
    else:
        raise ValueError("First argument must be J (scalar) or [J, m1, m2].")

    beta_only = beta is None
    if beta_only:
        alpha_val = 0.0
        beta_val = float(alpha_or_beta)
        gamma_val = 0.0
    else:
        alpha_val = float(alpha_or_beta)
        beta_val = float(beta)
        gamma_val = float(gamma)

    if phase not in ('+', '-'):
        raise ValueError("phase must be '+' or '-'.")
    sign = -1.0 if phase == '-' else +1.0

    # Basis: m = +J, J-1, ..., -J
    nJ = int(round(2 * Jval)) + 1
    ms = np.arange(Jval, -Jval - 0.5, -1)  # +J ... -J

    def _d_element(m1: float, m2: float) -> float:
        """Reduced Wigner d^J_{m1,m2}(beta_val)."""
        cb2 = math.cos(beta_val / 2)
        sb2 = math.sin(beta_val / 2)

        tmin = max(0, int(round(m2 - m1)))
        tmax = min(int(round(Jval + m2)), int(round(Jval - m1)))

        if tmin > tmax:
            return 0.0

        # Pre-factor ln: sqrt[(J+m1)!(J-m1)!(J+m2)!(J-m2)!]
        pre_ln = 0.5 * (
            _logfact(int(round(Jval + m1))) +
            _logfact(int(round(Jval - m1))) +
            _logfact(int(round(Jval + m2))) +
            _logfact(int(round(Jval - m2)))
        )

        total = 0.0
        for t in range(tmin, tmax + 1):
            exp_cos = int(round(2 * Jval - 2 * t + m2 - m1))
            exp_sin = int(round(2 * t + m1 - m2))

            if exp_cos < 0 or exp_sin < 0:
                continue

            # Avoid log(0) for sin/cos at 0 or pi
            if cb2 == 0.0 and exp_cos > 0:
                continue
            if sb2 == 0.0 and exp_sin > 0:
                continue

            # Compute log|term|
            denom_ln = (
                _logfact(int(round(Jval - m1 - t))) +
                _logfact(t) +
                _logfact(int(round(t + m1 - m2))) +
                _logfact(int(round(Jval + m2 - t)))
            )
            ln_abs = pre_ln - denom_ln
            if exp_cos > 0:
                ln_abs += exp_cos * math.log(abs(cb2))
            if exp_sin > 0:
                ln_abs += exp_sin * math.log(abs(sb2))

            # Signs: (-1)^(m2-m1+t) from cos^a * sin^b convention
            sign_t = (-1) ** int(round(m2 - m1 + t))
            # Account for negative values of cb2, sb2 (sin always ≥ 0 for beta in [0,pi])
            if cb2 < 0 and exp_cos % 2 == 1:
                sign_t *= -1
            if sb2 < 0 and exp_sin % 2 == 1:
                sign_t *= -1

            total += sign_t * math.exp(ln_abs)

        return total

    if single_element:
        m1v = float(m1_req); m2v = float(m2_req)
        d_val = _d_element(m1v, m2v)
        if beta_only:
            return d_val
        # Apply Euler angles with phase sign
        return _cexp(sign * 1j * m1v * alpha_val) * d_val * _cexp(sign * 1j * m2v * gamma_val)

    # Build full matrix
    D = np.zeros((nJ, nJ), dtype=complex if not beta_only else float)
    for i, m1v in enumerate(ms):
        for j, m2v in enumerate(ms):
            d_val = _d_element(m1v, m2v)
            if beta_only:
                D[i, j] = d_val
            else:
                D[i, j] = (_cexp(sign * 1j * m1v * alpha_val) *
                            d_val *
                            _cexp(sign * 1j * m2v * gamma_val))

    return D


def _cexp(x: complex) -> complex:
    """Complex exponential exp(x) for pure-imaginary x = i*theta."""
    return cmath.exp(x)


# ---------------------------------------------------------------------------
# spherharm
# ---------------------------------------------------------------------------

def spherharm(L: int, M: int, theta, phi, kind: str = 'c') -> np.ndarray:
    """Spherical harmonic Y_L^M(theta, phi).

    Parameters
    ----------
    L : int
        Degree (non-negative integer).
    M : int
        Order (``-L <= M <= L``).
    theta : float or array_like
        Polar angle (colatitude from +z), radians.
    phi : float or array_like
        Azimuthal angle (longitude from +x), radians.
    kind : {'c', 'r'}, optional
        ``'c'`` (default): complex spherical harmonics with Condon-Shortley phase.
        ``'r'``: real spherical harmonics.

    Returns
    -------
    ndarray or complex
        Spherical harmonic value(s).  Shape matches ``theta`` / ``phi``.

    Notes
    -----
    Uses ``scipy.special.sph_harm`` for complex harmonics.
    Argument order follows EasySpin: ``spherharm(L, M, theta, phi)``
    (theta = polar, phi = azimuthal), which is the reverse of scipy's
    ``sph_harm(M, L, phi, theta)``.

    Real harmonics follow Blanco et al. 1997, Table 1 (non-negative near θ=0, φ=0).
    """
    L, M = int(L), int(M)
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    if theta.shape != phi.shape:
        raise ValueError("theta and phi must have the same shape.")

    if kind == 'c':
        # scipy.special.sph_harm_y(n, m, theta, phi) — EasySpin arg order matches
        return sps.sph_harm_y(L, M, theta, phi)
    elif kind == 'r':
        # Real spherical harmonics (Condon-Shortley-free, non-negative near 0)
        absM = abs(M)
        # Normalization
        p = math.factorial(L - absM) / math.factorial(L + absM)
        N = math.sqrt((2 * L + 1) / (4 * math.pi) * p)
        P = sps.lpmv(absM, L, np.cos(theta))
        if M > 0:
            return math.sqrt(2) * N * P * np.cos(M * phi)
        elif M < 0:
            return math.sqrt(2) * N * P * np.sin(absM * phi)
        else:
            return N * P
    else:
        raise ValueError("kind must be 'c' (complex) or 'r' (real).")


# ---------------------------------------------------------------------------
# clebschgordan
# ---------------------------------------------------------------------------

def clebschgordan(j1, j2, j, m1, m2, m) -> float:
    """Clebsch-Gordan coefficient ``<j1,m1; j2,m2 | j,m>``.

    Parameters
    ----------
    j1, j2 : float
        Angular momenta of the two subsystems.
    j : float
        Total angular momentum.
    m1, m2, m : float
        Magnetic quantum numbers.

    Returns
    -------
    float
        Clebsch-Gordan coefficient.

    Notes
    -----
    Related to the Wigner 3-j symbol by:
    ``CG = (-1)^(j1-j2+m) * sqrt(2j+1) * W3j(j1,j2,j; m1,m2,-m)``
    (Brink & Satchler, Angular Momentum, 3rd ed., eq. 3.3).
    """
    j1, j2, j = float(j1), float(j2), float(j)
    m1, m2, m = float(m1), float(m2), float(m)
    return ((-1) ** int(round(j1 - j2 + m)) *
            math.sqrt(2 * j + 1) *
            wigner3j(j1, j2, j, m1, m2, -m))


# ---------------------------------------------------------------------------
# cgmatrix
# ---------------------------------------------------------------------------

def cgmatrix(S1: float, S2: float, Stot=None) -> tuple:
    """Transformation matrix between uncoupled and coupled spin representations.

    Parameters
    ----------
    S1, S2 : float
        Spin quantum numbers (1/2, 1, 3/2, 2, …).
    Stot : float or None, optional
        If given, return only coupled states with total spin ``Stot``.

    Returns
    -------
    U2C : ndarray, shape (n, n)
        Transformation matrix with elements ``<Stot,mStot|mS1,mS2>``.
        Apply as: ``psi_c = U2C @ psi_u`` (uncoupled→coupled).
    Smtot : ndarray, shape (n, 2)
        ``[Stot, mStot]`` for each coupled basis state.
    m12 : ndarray, shape (n, 2)
        ``[mS1, mS2]`` for each uncoupled basis state.

    Notes
    -----
    Basis ordering matches EasySpin:

    * Uncoupled: descending mS1 (outer), descending mS2 (inner).
    * Coupled:   descending Stot (outer), descending mStot (inner).

    Uses :func:`clebschgordan` to compute each matrix element.
    """
    S1, S2 = float(S1), float(S2)

    if Stot is None:
        Stot_list = list(np.arange(S1 + S2, abs(S1 - S2) - 0.5, -1))
    else:
        Stot = float(Stot)
        if Stot > S1 + S2 or Stot < abs(S1 - S2):
            raise ValueError(f"Stot={Stot} is outside the valid range "
                             f"[{abs(S1-S2)}, {S1+S2}].")
        Stot_list = [Stot]

    nCoupled = int(sum(2 * St + 1 for St in Stot_list))
    nUncoupled = int(round((2 * S1 + 1) * (2 * S2 + 1)))
    U2C = np.zeros((nCoupled, nUncoupled))
    Smtot = np.zeros((nCoupled, 2))
    m12 = np.zeros((nUncoupled, 2))

    # Build uncoupled index → (mS1, mS2) mapping
    iu = 0
    for mS1 in np.arange(S1, -S1 - 0.5, -1):
        for mS2 in np.arange(S2, -S2 - 0.5, -1):
            m12[iu] = [mS1, mS2]
            ic = 0
            for St in Stot_list:
                for mSt in np.arange(St, -St - 0.5, -1):
                    U2C[ic, iu] = clebschgordan(S1, S2, St, mS1, mS2, mSt)
                    if iu == 0:
                        Smtot[ic] = [St, mSt]
                    ic += 1
            iu += 1

    return U2C, Smtot, m12


# ---------------------------------------------------------------------------
# wigner6j
# ---------------------------------------------------------------------------

def wigner6j(j1, j2, j3, J1, J2, J3) -> float:
    """Wigner 6-j symbol.

    Computes::

        / j1  j2  j3 \\
        \\ J1  J2  J3 /

    Parameters
    ----------
    j1, j2, j3, J1, J2, J3 : int or half-int
        Angular momentum quantum numbers.

    Returns
    -------
    float
        Value of the Wigner 6-j symbol (0 if any triangle rule is violated).

    Notes
    -----
    Uses the Tuzun–Burkhardt–Secrest algorithm (CPC 112, 1998, Formula 80,
    Method 2 — recursive binomial expansion).  Accurate for typical EPR
    angular momentum values.

    Selection rules (return 0 if violated)
    - Triangle inequality for (j1,j2,j3), (j1,J2,J3), (J1,j2,J3), (J1,J2,j3)
    - All six arguments must be non-negative integers or half-integers
    """
    j1, j2, j3 = float(j1), float(j2), float(j3)
    J1, J2, J3 = float(J1), float(J2), float(J3)

    # All must be half-integers (2*j is integer)
    for v in (j1, j2, j3, J1, J2, J3):
        if abs(round(2 * v) - 2 * v) > 1e-10 or v < 0:
            return 0.0

    # Four triangle inequalities
    if not (_istriangle(j1, j2, j3) and _istriangle(j1, J2, J3) and
            _istriangle(J1, j2, J3) and _istriangle(J1, J2, j3)):
        return 0.0

    def _fac(n: int) -> float:
        """Factorial as float."""
        return float(math.factorial(int(round(n))))

    def _bino(a: float, b: float) -> float:
        """Binomial coefficient C(a, b) = a! / (b! * (a-b)!)."""
        ai, bi = int(round(a)), int(round(b))
        if bi < 0 or ai < bi:
            return 0.0
        v = 1.0
        for i in range(bi):
            v = v * (ai - i) / (i + 1)
        return v

    # Tuzun 1998 Formula (80) Method 2
    alpha = sorted([j1 + j2 + J1 + J2,
                    j1 + j3 + J1 + J3,
                    j2 + j3 + J2 + J3])
    A1, A2, A3 = alpha

    beta = sorted([j1 + j2 + j3,
                   j1 + J2 + J3,
                   J1 + j2 + J3,
                   J1 + J2 + j3])
    B1, B2, B3, B4 = beta

    # Prefactor (square root form)
    pre = math.sqrt(
        _fac(A1 - B2) * _fac(A1 - B3) * _fac(A1 - B4) / _fac(A1 - B1) / _fac(B1 + 1) *
        _fac(A2 - B1) * _fac(A2 - B3) * _fac(A2 - B4) / _fac(A2 - B2) / _fac(B2 + 1) *
        _fac(A3 - B1) * _fac(A3 - B2) * _fac(A3 - B4) / _fac(A3 - B3) / _fac(B3 + 1) *
        _fac(B4 + 1)
    )
    sumfactor = ((-1) ** int(round(B4)) *
                 _bino(A1 - B1, A1 - B4) *
                 _bino(A2 - B2, A2 - B4) *
                 _bino(A3 - B3, A3 - B4))
    pre = pre * sumfactor

    if B4 >= A1 - 1e-10:
        return pre

    # Recursive sum from n = B4+1 to A1
    t = 1.0
    s = t
    for n in range(int(round(B4)) + 1, int(round(A1)) + 1):
        t = -t * (n + 1) / (n - B4) * (A1 - n + 1) / (n - B1) * \
            (A2 - n + 1) / (n - B2) * (A3 - n + 1) / (n - B3)
        s += t
    return pre * s


# ---------------------------------------------------------------------------
# isto  (Irreducible Spherical Tensor Operators)
# ---------------------------------------------------------------------------

def isto(J, kq, q_arg=None) -> np.ndarray:
    """Irreducible spherical tensor operator T^k_q.

    Port of EasySpin's ``isto.m`` (Ryabov 1999, J. Magn. Reson. 140, 141).

    Parameters
    ----------
    J : float or list of float
        Angular momentum quantum number(s).  A single float for one spin,
        a list for a multi-spin product operator.
    kq : array_like or int
        If ``q_arg`` is ``None``:
          - Shape (nSpins, 2): each row ``[k, q]`` for the corresponding spin.
          - Shape (nTerms, 3): each row ``[k, q, spin_index]`` (1-based).
        If ``q_arg`` is given, ``kq`` is the rank ``k`` (single-spin shorthand).
    q_arg : int, optional
        When provided, ``kq`` is ``k`` and ``q_arg`` is ``q`` (single-spin call).

    Returns
    -------
    ndarray, complex128
        The (dim × dim) irreducible spherical tensor operator matrix, where
        dim = product of (2*Ji + 1) for all spins.

    Examples
    --------
    >>> T = isto(0.5, 1, 0)    # T^1_0 for S=1/2  (= Sz / sqrt(2), unnorm)
    >>> T = isto(1.0, [2, 0])  # T^2_0 for S=1
    """
    # --- Normalise input ---
    if q_arg is not None:
        # isto(J, k, q) shorthand
        k_single = int(round(float(kq)))
        q_single = int(round(float(q_arg)))
        kq_arr = np.array([[k_single, q_single]], dtype=float)
        J_list = [float(J)]
    else:
        kq_arr = np.atleast_2d(np.asarray(kq, dtype=float))
        if np.isscalar(J) or (hasattr(J, '__len__') and len(J) == 1):
            J_list = [float(J) if np.isscalar(J) else float(J[0])]
        else:
            J_list = [float(j) for j in J]

    # If kq has 3 columns, it uses [k, q, spin_index] format
    if kq_arr.shape[1] == 3:
        nSpins = len(J_list)
        kq_full = np.zeros((nSpins, 2))
        for row in kq_arr:
            i = int(round(row[2])) - 1  # 1-based → 0-based
            kq_full[i] = row[:2]
        kq_arr = kq_full
    elif kq_arr.shape[0] != len(J_list) and len(J_list) == 1:
        # Single spin, kq given as 1×2 or (2,)
        pass  # already fine
    elif kq_arr.shape[0] != len(J_list):
        raise ValueError("kq must have one row per spin in J.")

    # --- Build product operator via Kronecker products ---
    T = np.array([[1.0 + 0j]])  # 1×1 identity (will be kron-expanded)
    for idx, Ji in enumerate(J_list):
        k_i = int(round(kq_arr[idx, 0]))
        q_i = int(round(kq_arr[idx, 1]))
        Tj = _isto_single(Ji, k_i, q_i)
        T = np.kron(T, Tj)
    return T


def _isto_single(J: float, k: int, q: int) -> np.ndarray:
    """ISTO T^k_q for a single spin J (Ryabov 1999).

    Parameters
    ----------
    J : float  spin quantum number (0, 1/2, 1, ...)
    k : int    rank (0, 1, 2, ...)
    q : int    projection (−k ≤ q ≤ k)
    """
    if abs(q) > k:
        raise ValueError(f"q={q} must satisfy |q| <= k={k}.")
    if k < 0:
        raise ValueError("k must be a non-negative integer.")

    dim = int(round(2 * J + 1))
    if J == 0:
        return np.ones((1, 1), dtype=complex)

    # Build J+ and J- raising/lowering operators
    # basis: m = +J, J-1, ..., -J  (descending)
    from torchspin.spinops import sop
    Jp = sop(J, '+').numpy().astype(complex)
    Jm = sop(J, '-').numpy().astype(complex)

    # T^k_k = N_kk * (J+)^k   (Ryabov Eq.[2] + [17])
    Nkk = ((-1.0) ** k) / (2.0 ** (k / 2.0))
    T = Nkk * np.linalg.matrix_power(Jp, k)

    # Lower projection from k down to |q| via Racah commutation (Ryabov Eq.[1])
    for q_ in range(k, abs(q), -1):
        denom = math.sqrt(k * (k + 1) - q_ * (q_ - 1))
        T = (Jm @ T - T @ Jm) / denom

    # Negative q via Hermitian conjugate (Ryabov Eq.[3])
    if q < 0:
        T = ((-1.0) ** q) * T.conj().T

    return T


# ---------------------------------------------------------------------------
# isto2stev  (ISTO ↔ Stevens operator transformation matrix)
# ---------------------------------------------------------------------------

def isto2stev(k: int) -> np.ndarray:
    """Transformation matrix from ISTOs to Stevens operators.

    Port of EasySpin's ``isto2stev.m``.

    Computes the (2k+1) × (2k+1) matrix **C** such that::

        O_k = C @ T_k

    where T_k = [T^k_k, T^k_{k-1}, ..., T^k_{-k}] (ISTO column vector)
    and O_k = [O^k_k, O^k_{k-1}, ..., O^k_{-k}] (Stevens operator vector).

    Parameters
    ----------
    k : int
        Rank (non-negative integer).

    Returns
    -------
    C : ndarray, shape (2k+1, 2k+1), complex128
        Transformation matrix.  Inverse is ``numpy.linalg.inv(C)``.

    Notes
    -----
    Only diagonal and anti-diagonal elements are non-zero.
    Uses ``J = k/2`` (smallest spin for which rank-k operators exist).
    The result is independent of the choice of J.

    Examples
    --------
    >>> C = isto2stev(2)   # 5×5 matrix for rank-2 operators
    >>> C.shape
    (5, 5)
    """
    if not isinstance(k, int) or k < 0:
        raise ValueError("k must be a non-negative integer.")

    from torchspin.stev import stev as _stev

    nq = 2 * k + 1
    # Use smallest usable spin: J = k/2
    # (stev and isto require 2J >= k)
    J = k / 2.0

    # Build ISTO and Stevens operator vectors (ordering: q = k, k-1, ..., -k)
    T_list = []
    O_list = []
    for q in range(k, -k - 1, -1):
        T_list.append(_isto_single(J, k, q))
        O_arr = _stev(J, k, q).numpy().astype(complex)
        O_list.append(O_arr)

    C = np.zeros((nq, nq), dtype=complex)
    for t in range(nq):
        traceTT = np.trace(T_list[t].conj().T @ T_list[t])
        if abs(traceTT) < 1e-14:
            continue
        # Only diagonal (o=t) and anti-diagonal (o=nq-1-t) are non-zero
        for o in {t, nq - 1 - t}:
            C[o, t] = np.trace(O_list[o].T @ T_list[t]) / traceTT

    return C
