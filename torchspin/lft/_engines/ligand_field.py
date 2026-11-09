"""NumPy constants of the one-electron d-shell ligand field (copied from telser-tools).

Real-orbital basis in DDN's order (x2-y2, xy, xz, yz, z2), defined geometrically (each a
positive multiple of the named polynomial); with Condon-Shortley |m> = Y_2^m:
dz2 = |0>, dx2-y2 = (|2> + |-2>)/sqrt2, dxz = -(|1> - |-1>)/sqrt2, dxy = -i(|2> - |-2>)/sqrt2,
dyz = i(|1> + |-1>)/sqrt2  (DDNCONVERT1.F ll. 97-100).
"""

from __future__ import annotations

from math import sqrt

import numpy as np

from .dn import U_REAL

__all__ = ["Q", "AOM_LOCAL", "crystal_field", "from_m_basis"]

_R2 = 1.0 / sqrt(2.0)


def _quadratic_forms() -> np.ndarray:
    """Frobenius-orthonormal symmetric traceless Q with d_k(r) proportional to r^T Q_k r (DDN order)."""
    e = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            e[i, j, i, j] = 1.0
    sym = lambda i, j: (e[i, j] + e[j, i]) * _R2  # noqa: E731
    return np.array([(e[0, 0] - e[1, 1]) * _R2, sym(0, 1), sym(0, 2), sym(1, 2),
                     (2 * e[2, 2] - e[0, 0] - e[1, 1]) / sqrt(6.0)])


Q = _quadratic_forms()

#: local-frame real orbital carrying each AOM bonding mode, in the order
#: (sigma, pi_x, pi_y, delta_x2y2, delta_xy)
AOM_LOCAL = (4, 2, 3, 0, 1)


def from_m_basis(v_m: np.ndarray) -> np.ndarray:
    """|m = -2..2> matrix -> real-orbital matrix; raises if the field is not real (time-even)."""
    v = U_REAL.conj().T @ np.asarray(v_m) @ U_REAL
    if np.max(np.abs(v.imag)) > 1e-9 * max(1.0, np.abs(v).max()):
        raise ValueError("matrix is not a real (time-reversal even) ligand field")
    return v.real


def crystal_field(Dq: float = 0.0, Ds: float = 0.0, Dt: float = 0.0, Dr: float = 0.0,
                  geometry: str = "Oh", axis: str = "tetragonal") -> np.ndarray:
    """Ballhausen crystal field (real-orbital matrix), DDN conventions (DDNIO.F ll. 418-449).

    z is the C4 axis ("tetragonal") or the C3 axis ("trigonal").  In the |m> basis:
        <+-2|V|+-2> = b Dq + 2Ds - Dt      <+-1|V|+-1> = -4b Dq - Ds + 4Dt
        <0|V|0>     = 6b Dq - 2Ds - 6Dt    <1|V|-1> = Dr,  <2|V|0> = <0|V|-2> = sqrt(2/3) Dr
        tetragonal: b = 1,    <2|V|-2> = 5Dq
        trigonal:   b = -2/3, <2|V|-1> = -<1|V|-2> = (10 sqrt2 / 3) Dq
    so e(g) lies at +6Dq and t2(g) at -4Dq.  ``geometry="Td"`` multiplies the whole matrix
    by -1, as DDN's SIGNXF does.
    """
    ax = axis.lower()
    if ax.startswith("tetra") or ax == "4":
        b4 = 1.0
    elif ax.startswith("trig") or ax == "3":
        b4 = -2.0 / 3.0
    else:
        raise ValueError("axis must be 'tetragonal' or 'trigonal'")
    geo = geometry.lower()
    if geo not in ("oh", "td", "o", "t"):
        raise ValueError("geometry must be 'Oh' or 'Td'")
    v = np.zeros((5, 5), dtype=complex)
    i = lambda m: m + 2  # noqa: E731
    v[i(2), i(2)] = v[i(-2), i(-2)] = b4 * Dq + 2 * Ds - Dt
    v[i(1), i(1)] = v[i(-1), i(-1)] = -4 * b4 * Dq - Ds + 4 * Dt
    v[i(0), i(0)] = 6 * b4 * Dq - 2 * Ds - 6 * Dt
    upper = {(1, -1): Dr, (2, 0): sqrt(2.0 / 3.0) * Dr, (0, -2): sqrt(2.0 / 3.0) * Dr}
    if b4 == 1.0:
        upper[(2, -2)] = 5 * Dq
    else:
        upper[(2, -1)] = 10.0 / 3.0 * sqrt(2.0) * Dq
        upper[(1, -2)] = -10.0 / 3.0 * sqrt(2.0) * Dq
    for (a, b), val in upper.items():
        v[i(a), i(b)] = val
        v[i(b), i(a)] = np.conj(val)
    sign = -1.0 if geo in ("td", "t") else 1.0
    return sign * from_m_basis(v)
