"""DSOXF single-term constants (NumPy): configuration table, Stevens operators, crystal-field model.

Copied from telser-tools (telser-tools commit 89ba7be) so that torchspin has no runtime
dependency on telser-tools:

* ``telser_tools.spin.operators``: :func:`spin_matrices` and the rank <= 4 Stevens operators;
* ``telser_tools.dsoxf.operators``: rank-6 Stevens operators and the |m_L m_S> product basis;
* ``telser_tools.dsoxf.configurations``: DSOXF's configuration table (L, S, SIGNXF, SIGNLAM,
  B2X/B4X, the f-shell F_kq divisors) and the independent operator-equivalent ratios;
* ``telser_tools.dsoxf.hamiltonian``: :class:`DsoxfModel` (crystal field, spin-orbit sign,
  spin-spin operator, both ``conventions``) and :func:`orbital_vectors`.

The NumPy Solution class, input reader, report and the legacy EPR probabilities are not
copied.  Energies are in cm^-1 by convention of the callers (everything here is linear in the
energy parameters, so nothing carries units).  Parity with the original is checked in
torchspin/tests/test_lft_dsoxf.py when telser-tools is installed.

Basis (``dsoxf.f`` ll. 117-126): index ``NS*(INL-1) + INS`` with m_L = L, L-1, ..., -L outer and
m_S = S, ..., -S inner, i.e. ``np.kron(orbital_op, spin_op)``.

Stevens operators (Abragam & Bleaney Table 16, cosine forms, X = J(J+1)); rank 6 as DSOXF codes
them (``dsoxf.f`` ll. 393-403, 499-509, 537-551, 603-621)::

    O60 = 231 Jz^6 - (315X - 735) Jz^4 + (105X^2 - 525X + 294) Jz^2 - 5X^3 + 40X^2 - 60X
    O62 = 1/4 [(33 Jz^4 - (18X + 123) Jz^2 + X^2 + 10X + 102)(J+^2 + J-^2) + h.c.]
    O63 = 1/4 [(11 Jz^3 - (3X + 59) Jz)(J+^3 + J-^3) + h.c.]
    O64 = 1/4 [(11 Jz^2 - X - 38)(J+^4 + J-^4) + h.c.]
    O66 = (J+^6 + J-^6)/2

Crystal field, d shell (``dsoxf.f`` ll. 175-338), sigma = SIGNXF (reversed for Td), B2X = 3 (D)
or 15 (F), B4X = 12 (D) or 60 (F), b = 1 (tetragonal) or -2/3 (trigonal)::

    sigma [ Ds/B2X O20 + (b Dq + Delta4)/B4X O40 + V2/B2X O22 + V4/B4X O42
            + Dq/B4X * 5 O44            (tetragonal: z = C4)
            + b Dq/B4X * 20 sqrt2 O43 ]  (trigonal:   z = C3)

f shell (ll. 381-669), B4 = B4(in)/600, B6 = B6(in)/5040, B_kq = B_kq(in)/F_kq(L),
(b4, b6) = (1, 1) tetragonal or (-2/3, 16/9) trigonal::

    sigma [ b4 B4 O40 + b6 B6 O60 + B20 O20 + B40 O40 + B60 O60 + B22 O22 + B42 O42 + B62 O62
            + B4 (5 O44) - B6 (21 O64) + B44 O44 + B64 O64                             (tetragonal)
            + b4 B4 20 sqrt2 O43 - b6 B6 35/sqrt8 O63 + B43 O43 + B63 O63
            + b6 B6 77/8 O66 + B66 O66 ]                                              (trigonal)

``conventions="dsoxf"`` reproduces the compiled program; ``"corrected"`` (default) fixes the
low-spin d4/d5 (L~ = 1) spin-orbit and orbital-Zeeman signs, the F-term rank-2 signs and the
low-spin d7 signs (telser-tools findings DSOXF-01 to -03; docs/dsoxf.md there).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from functools import cached_property, lru_cache

import numpy as np

__all__ = ["CONVENTIONS", "Configuration", "D_CODES", "D_ORBITALS", "DsoxfModel", "F_CODES", "F_CUBIC_ORBITALS",
           "F_GENERAL_ORBITALS", "F_SCALE", "LS_TILDE_CODES", "ProductBasis", "SpinMatrices", "configuration",
           "f_scale_divisors", "orbital_vectors", "product_basis", "spin_matrices", "stevens", "stevens_ratios"]


# ============================================================================ spin matrices
# (telser_tools.spin.operators)
@dataclass(frozen=True)
class SpinMatrices:
    s: float
    m: np.ndarray        # (2s+1,) descending
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    plus: np.ndarray
    minus: np.ndarray

    @property
    def vector(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.x, self.y, self.z

    @property
    def identity(self) -> np.ndarray:
        return np.eye(len(self.m), dtype=complex)


def _check_spin(s: float) -> float:
    two_s = round(2 * float(s))
    if two_s < 0 or abs(2 * float(s) - two_s) > 1e-9:
        raise ValueError(f"spin must be a non-negative multiple of 1/2, got {s}")
    return two_s / 2.0


@lru_cache(maxsize=64)
def _spin_matrices_cached(s: float) -> SpinMatrices:
    m = s - np.arange(int(round(2 * s)) + 1)
    n = len(m)
    sp = np.zeros((n, n), dtype=complex)
    for k in range(1, n):                          # <m+1| S+ |m> = sqrt(X - m(m+1))
        sp[k - 1, k] = np.sqrt(s * (s + 1) - m[k] * (m[k] + 1))
    sm = sp.T.copy()
    out = SpinMatrices(s, m, (sp + sm) / 2, (sp - sm) / 2j, np.diag(m).astype(complex), sp, sm)
    for a in (out.m, out.x, out.y, out.z, out.plus, out.minus):
        a.setflags(write=False)
    return out


def spin_matrices(s: float) -> SpinMatrices:
    """Sx, Sy, Sz, S+, S- for spin ``s`` (read-only arrays), |m = s ... -s>."""
    return _spin_matrices_cached(_check_spin(s))


def _ac(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a b + b a) / 4: the symmetrized product used in the Stevens definitions."""
    return (a @ b + b @ a) / 4


def _stevens_k4(s: float, k: int, q: int) -> np.ndarray:
    """Cosine Stevens operator O_k^q (0 <= q <= k <= 4) for spin ``s``."""
    sm = spin_matrices(s)
    x = sm.s * (sm.s + 1)
    one = sm.identity
    z, p, mi = sm.z, sm.plus, sm.minus
    z2 = z @ z
    z3 = z2 @ z
    p1, p2 = p + mi, p @ p + mi @ mi
    p3 = p @ p @ p + mi @ mi @ mi
    p4 = p @ p @ p @ p + mi @ mi @ mi @ mi
    table = {
        (0, 0): lambda: one,
        (1, 0): lambda: z,
        (1, 1): lambda: p1 / 2,
        (2, 0): lambda: 3 * z2 - x * one,
        (2, 1): lambda: _ac(z, p1),
        (2, 2): lambda: p2 / 2,
        (3, 0): lambda: 5 * z3 - (3 * x - 1) * z,
        (3, 1): lambda: _ac(5 * z2 - (x + 0.5) * one, p1),
        (3, 2): lambda: _ac(z, p2),
        (3, 3): lambda: p3 / 2,
        (4, 0): lambda: 35 * z2 @ z2 - (30 * x - 25) * z2 + (3 * x * x - 6 * x) * one,
        (4, 1): lambda: _ac(7 * z3 - (3 * x + 1) * z, p1),
        (4, 2): lambda: _ac(7 * z2 - (x + 5) * one, p2),
        (4, 3): lambda: _ac(z, p3),
        (4, 4): lambda: p4 / 2,
    }
    if (k, q) not in table:
        raise ValueError(f"O_{k}^{q} not available (need 0 <= q <= k <= 4)")
    return table[(k, q)]()


# ============================================================================ Stevens, product basis
# (telser_tools.dsoxf.operators)
@lru_cache(maxsize=256)
def _stevens_cached(j: float, k: int, q: int) -> np.ndarray:
    if k <= 4:
        out = np.real(_stevens_k4(j, k, q)).copy()
    elif k == 6:
        sm = spin_matrices(j)
        x = sm.s * (sm.s + 1)
        one = np.eye(len(sm.m))
        z, p, mi = np.real(sm.z), np.real(sm.plus), np.real(sm.minus)
        z2 = z @ z
        z4 = z2 @ z2

        def pq(n: int) -> np.ndarray:
            return np.linalg.matrix_power(p, n) + np.linalg.matrix_power(mi, n)

        if q == 0:
            out = (231 * z4 @ z2 - (315 * x - 735) * z4 + (105 * x * x - 525 * x + 294) * z2
                   + (-5 * x ** 3 + 40 * x * x - 60 * x) * one)
        elif q == 2:
            out = _ac(33 * z4 - (18 * x + 123) * z2 + (x * x + 10 * x + 102) * one, pq(2))
        elif q == 3:
            out = _ac(11 * z2 @ z - (3 * x + 59) * z, pq(3))
        elif q == 4:
            out = _ac(11 * z2 - (x + 38) * one, pq(4))
        elif q == 6:
            out = pq(6) / 2
        else:
            raise ValueError(f"O_6^{q} not implemented (DSOXF uses q = 0, 2, 3, 4, 6)")
    else:
        raise ValueError(f"rank {k} not available")
    out.setflags(write=False)
    return out


def stevens(j: float, k: int, q: int) -> np.ndarray:
    """Real cosine Stevens operator O_k^q for angular momentum ``j`` (|j m>, m descending)."""
    return _stevens_cached(float(j), int(k), int(q))


@dataclass(frozen=True)
class ProductBasis:
    """Operators on |m_L m_S> (m_L outer).  All arrays are read-only."""

    L: int
    S: float
    ml: np.ndarray
    ms: np.ndarray
    Lv: tuple[np.ndarray, np.ndarray, np.ndarray]   # L_x, L_y, L_z (complex)
    Sv: tuple[np.ndarray, np.ndarray, np.ndarray]   # S_x, S_y, S_z (complex)
    LS: np.ndarray                                   # L.S (real)

    @property
    def nl(self) -> int:
        return 2 * self.L + 1

    @property
    def ns(self) -> int:
        return int(round(2 * self.S + 1))

    @property
    def dim(self) -> int:
        return self.nl * self.ns

    def orbital(self, op: np.ndarray) -> np.ndarray:
        """Lift an orbital operator (2L+1 square) into the product basis."""
        return np.kron(op, np.eye(self.ns))

    def stevens(self, k: int, q: int) -> np.ndarray:
        """O_k^q(L) lifted into the product basis (zero if 2L < k)."""
        if 2 * self.L < k:
            return np.zeros((self.dim, self.dim))
        return self.orbital(stevens(self.L, k, q))


@lru_cache(maxsize=64)
def product_basis(L: int, S: float) -> ProductBasis:
    lm, sm = spin_matrices(L), spin_matrices(S)
    il, is_ = np.eye(len(lm.m)), np.eye(len(sm.m))
    Lv = tuple(np.kron(a, is_) for a in lm.vector)
    Sv = tuple(np.kron(il, a) for a in sm.vector)
    LS = np.real(sum(a @ b for a, b in zip(Lv, Sv)))
    ml = np.repeat(lm.m, len(sm.m))
    ms = np.tile(sm.m, len(lm.m))
    for a in (*Lv, *Sv, LS, ml, ms):
        a.setflags(write=False)
    return ProductBasis(int(L), float(S), ml, ms, Lv, Sv, LS)  # type: ignore[arg-type]


# ============================================================================ configurations
# (telser_tools.dsoxf.configurations; transcribed from INOUT, dsoxfio.f ll. 69-268, 320-500)
@dataclass(frozen=True)
class Configuration:
    """One DSOXF configuration.

    ``sign_cf`` is SIGNXF before the Td reversal and ``sign_lambda`` is SIGNLAM; both are 0 where
    DSOXF leaves them undefined (L = 0: they are never used).  ``b2x``/``b4x`` are DSOXF's divisors
    for the d-shell parameters (Ds/B2X, Dq/B4X); 0 where undefined.
    """

    shell: str          # "d" or "f"
    code: int           # the number typed at the prompt (negative = low spin, d only)
    L: int
    S: float
    sign_cf: float
    sign_lambda: float
    b2x: float = 0.0
    b4x: float = 0.0

    @property
    def n(self) -> int:
        return abs(self.code)

    @property
    def low_spin(self) -> bool:
        return self.code < 0

    @property
    def ls_tilde(self) -> bool:
        """Low-spin d4/d5: DSOXF (since 16-Jan-2025) uses a fictitious L~ = 1 (t2g^4 3T1, t2g^5 2T2)."""
        return self.shell == "d" and self.code in LS_TILDE_CODES

    @property
    def dim(self) -> int:
        return (2 * self.L + 1) * int(round(2 * self.S + 1))

    @property
    def label(self) -> str:
        if self.shell == "f":
            return f"f{self.n}"
        if self.n in (4, 5, 6, 7):
            return f"d{self.n} ({'LS' if self.low_spin else 'HS'})"
        return f"d{self.n}"

    @property
    def term(self) -> str:
        letters = "SPDFGHI"
        mult = int(round(2 * self.S + 1))
        return f"{mult}{letters[self.L]}" + ("~" if self.ls_tilde else "")


LS_TILDE_CODES = (-4, -5)

#                code   L   S    SIGNXF SIGNLAM B2X  B4X        dsoxfio.f
D_CODES = {
    1: Configuration("d", 1, 2, 0.5, 1.0, 1.0, 3.0, 12.0),       # ll. 70-78
    2: Configuration("d", 2, 3, 1.0, -1.0, 1.0, 15.0, 60.0),     # ll. 79-87
    3: Configuration("d", 3, 3, 1.5, 1.0, 1.0, 15.0, 60.0),      # ll. 88-96
    4: Configuration("d", 4, 2, 2.0, -1.0, 1.0, 3.0, 12.0),      # ll. 97-105
    -4: Configuration("d", -4, 1, 1.0, 1.0, -1.0, 3.0, 0.0),     # ll. 106-113 (L~ = 1)
    5: Configuration("d", 5, 0, 2.5, 0.0, 0.0, 3.0, 12.0),       # ll. 114-120
    -5: Configuration("d", -5, 1, 0.5, -1.0, -1.0, 3.0, 0.0),    # ll. 121-128 (L~ = 1)
    6: Configuration("d", 6, 2, 2.0, 1.0, -1.0, 3.0, 12.0),      # ll. 129-137
    -6: Configuration("d", -6, 0, 0.0, 0.0, 0.0, 3.0, 0.0),      # ll. 138-143
    7: Configuration("d", 7, 3, 1.5, -1.0, -1.0, 15.0, 60.0),    # ll. 144-152
    -7: Configuration("d", -7, 2, 0.5, 1.0, 1.0, 3.0, 12.0),     # ll. 153-161
    8: Configuration("d", 8, 3, 1.0, 1.0, -1.0, 15.0, 60.0),     # ll. 162-170
    9: Configuration("d", 9, 2, 0.5, -1.0, -1.0, 3.0, 12.0),     # ll. 171-179
}

#                         code  L   S   SIGNXF SIGNLAM               dsoxfio.f ll. 196-268
F_CODES = {
    1: Configuration("f", 1, 3, 0.5, 1.0, 1.0),
    2: Configuration("f", 2, 5, 1.0, 1.0, 1.0),
    3: Configuration("f", 3, 6, 1.5, 1.0, 1.0),
    4: Configuration("f", 4, 6, 2.0, 1.0, 1.0),
    5: Configuration("f", 5, 5, 2.5, 1.0, 1.0),
    6: Configuration("f", 6, 3, 3.0, 1.0, 1.0),
    7: Configuration("f", 7, 0, 3.5, 0.0, 0.0),
    8: Configuration("f", 8, 3, 3.0, -1.0, -1.0),
    9: Configuration("f", 9, 5, 2.5, -1.0, -1.0),
    10: Configuration("f", 10, 6, 2.0, -1.0, -1.0),
    11: Configuration("f", 11, 6, 1.5, -1.0, -1.0),
    12: Configuration("f", 12, 5, 1.0, 1.0, -1.0),   # SIGNXF = +1 breaks the f8-f13 pattern (finding)
    13: Configuration("f", 13, 3, 0.5, -1.0, -1.0),
}

# f-shell scaling divisors (dsoxfio.f ll. 320-500, "A2B2 Table 17 F values"):
# B_kq(used) = B_kq(entered) / F_kq, by L of the term (3: F, 5: H, 6: I).  Cubic B4, B6 are NOT
# scaled by term: B4 = B4(entered)/600 and B6 = B6(entered)/5040 for every f^n (ll. 349, 354).
F_KEYS = ("B20", "B40", "B60", "B22", "B42", "B62", "B43", "B63", "B44", "B64", "B66")
F_SCALE = {
    3: (3.0, 60.0, 180.0, 1.0, 3.0, 24.0, 3.0, 18.0, 12.0, 60.0, 360.0),
    5: (3.0, 420.0, 2520.0, 1.0, 21.0, 120.0, 3.0, 180.0, 12.0, 60.0, 360.0),
    6: (3.0, 60.0, 7560.0, 1.0, 3.0, 72.0, 3.0, 36.0, 12.0, 180.0, 360.0),
}
F_CUBIC_B4 = 600.0
F_CUBIC_B6 = 5040.0


def f_scale_divisors(L: int) -> dict[str, float]:
    """{"B20": 3, ...} for an f term of orbital momentum L (empty for L = 0, f7)."""
    return dict(zip(F_KEYS, F_SCALE[L])) if L in F_SCALE else {}


def configuration(shell: str, code: int) -> Configuration:
    """DSOXF's configuration for ``shell`` ('d'/'f', any case) and the typed code."""
    s = shell.strip().lower()[:1]
    table = D_CODES if s == "d" else F_CODES if s == "f" else None
    if table is None:
        raise ValueError(f"shell must be 'd' or 'f', got {shell!r}")
    if code not in table:
        raise ValueError(f"no {s}-shell configuration with code {code} (DSOXF accepts {sorted(table)})")
    return table[code]


def _stevens_diag(j: float, m: float, k: int) -> float:
    """<j m|O_k^0|j m> (Abragam & Bleaney Table 16)."""
    x = j * (j + 1)
    if k == 2:
        return 3 * m * m - x
    if k == 4:
        return 35 * m ** 4 - (30 * x - 25) * m * m + 3 * x * x - 6 * x
    if k == 6:
        return (231 * m ** 6 - (315 * x - 735) * m ** 4 + (105 * x * x - 525 * x + 294) * m * m
                - 5 * x ** 3 + 40 * x * x - 60 * x)
    raise ValueError(k)


@lru_cache(maxsize=None)
def stevens_ratios(l: int, n: int) -> dict[int, Fraction]:
    """Operator-equivalent factor of the Hund ground term of l^n relative to one electron, per rank k.

    Within the term, sum_i O_k^q(l_i) = R_k O_k^q(L) (Wigner-Eckart).  R_k follows from the
    |M_L = L, M_S = S> state, which for a Hund term is a single determinant:
    R_k = sum_i <m_i|O_k^0(l)|m_i> / <L L|O_k^0(L)|L L>.  (d1: R = 1; d2 3F: R_2 = 1/5, R_4 = -1/5.)
    """
    ml = list(range(l, -l - 1, -1))
    up = ml[:min(n, 2 * l + 1)]
    down = ml[:max(0, n - (2 * l + 1))]
    occ = up + down
    L = sum(occ)
    out: dict[int, Fraction] = {}
    for k in range(2, 2 * l + 1, 2):
        if 2 * L < k:
            continue
        num = sum(_stevens_diag(l, m, k) for m in occ)
        out[k] = Fraction(num).limit_denominator(10 ** 6) / Fraction(_stevens_diag(L, L, k)).limit_denominator(10 ** 6)
    return out


# ============================================================================ model
# (telser_tools.dsoxf.hamiltonian.DsoxfModel, without the NumPy Zeeman/solution parts)
CONVENTIONS = ("corrected", "dsoxf")


@dataclass(frozen=True)
class DsoxfModel:
    """All DSOXF parameters (cm-1).  ``lam`` is the magnitude; DSOXF chooses the sign.

    d shell: ``dq``, ``ds`` (DELTA2, "Ds"), ``dt4`` (DELTA4 = -Dt), ``v2``, ``v4``.
    f shell: ``b4``, ``b6`` (as entered: "2T1u / 2T2u energy for f1") and ``b20 ... b66`` as
    entered (scaled internally by F_kq).  ``symmetry`` cubic/axial/rhombic and ``axis``
    tetragonal/trigonal decide which parameters are used, exactly as DSOXF reads them.
    ``geometry`` = "Td" reverses all crystal-field parameters.
    """

    shell: str = "d"
    code: int = 1
    lam: float = 0.0
    k: tuple[float, float, float] = (1.0, 1.0, 1.0)
    rho: float = 0.0
    geometry: str = "Oh"
    symmetry: str = "cubic"
    axis: str = "tetragonal"
    dq: float = 0.0
    ds: float = 0.0
    dt4: float = 0.0
    v2: float = 0.0
    v4: float = 0.0
    b4: float = 0.0
    b6: float = 0.0
    b20: float = 0.0
    b40: float = 0.0
    b60: float = 0.0
    b44: float = 0.0
    b64: float = 0.0
    b43: float = 0.0
    b63: float = 0.0
    b66: float = 0.0
    b22: float = 0.0
    b42: float = 0.0
    b62: float = 0.0
    conventions: str = "corrected"

    def __post_init__(self):
        if self.conventions not in CONVENTIONS:
            raise ValueError(f"conventions must be one of {CONVENTIONS}")
        if self.geometry not in ("Oh", "Td"):
            raise ValueError("geometry must be 'Oh' or 'Td'")
        if self.symmetry not in ("cubic", "axial", "rhombic"):
            raise ValueError("symmetry must be 'cubic', 'axial' or 'rhombic'")
        if self.axis not in ("tetragonal", "trigonal"):
            raise ValueError("axis must be 'tetragonal' or 'trigonal'")
        configuration(self.shell, self.code)

    def with_(self, **kw) -> "DsoxfModel":
        return replace(self, **kw)

    @property
    def config(self) -> Configuration:
        return configuration(self.shell, self.code)

    @property
    def basis(self) -> ProductBasis:
        c = self.config
        return product_basis(c.L, c.S)

    @property
    def corrected(self) -> bool:
        return self.conventions == "corrected"

    @property
    def sign_cf(self) -> float:
        """SIGNXF as DSOXF uses it (Td reversal included; for low-spin d DSOXF never asks Oh/Td)."""
        c = self.config
        s = c.sign_cf
        asks_geometry = c.shell == "f" or c.code > 0
        if asks_geometry and self.geometry == "Td":
            s = -s
        return s

    @property
    def d_signs(self) -> tuple[float, float, float]:
        """Signs of the d-shell (cubic Dq; rank-4 distortion Delta4, V4; rank-2 Ds, V2) terms."""
        c = self.config
        s = self.sign_cf
        if not self.corrected:
            return s, s, s
        if c.code == -7:
            return -1.0, 1.0, 1.0
        if c.code > 0 and c.L == 3:
            r = stevens_ratios(2, c.n)
            return s, s, s * float(np.sign(r[2]) * np.sign(r[4]))
        return s, s, s

    @property
    def spin_orbit(self) -> float:
        """The coefficient of L.S actually used (DSOXF: SIGNLAM |lambda|)."""
        c = self.config
        if c.L == 0:
            return 0.0
        if self.corrected and c.ls_tilde:
            return abs(self.lam)
        if self.corrected and c.code == -7:
            return -abs(self.lam)
        return c.sign_lambda * abs(self.lam)

    @property
    def orbital_zeeman_sign(self) -> float:
        return -1.0 if (self.corrected and self.config.ls_tilde) else 1.0

    def f_parameters(self) -> dict[str, float]:
        """The f-shell B_kq used in H (after DSOXF's scaling and the symmetry selection)."""
        c = self.config
        div = f_scale_divisors(c.L)
        raw = {k: getattr(self, k.lower()) for k in div}
        keep = set()
        if self.symmetry != "cubic":
            keep |= {"B20", "B40", "B60"}
            keep |= {"B43", "B63", "B66"} if self.axis == "trigonal" else {"B44", "B64"}
        if self.symmetry == "rhombic":
            keep |= {"B22", "B42", "B62"}
        out = {k: (raw[k] / div[k] if k in keep else 0.0) for k in div}
        out["B4"] = self.b4 / F_CUBIC_B4
        out["B6"] = self.b6 / F_CUBIC_B6
        return out

    @cached_property
    def crystal_field(self) -> np.ndarray:
        """H_CF (real symmetric) in the |m_L m_S> basis."""
        b, c = self.basis, self.config
        h = np.zeros((b.dim, b.dim))
        if c.L == 0:
            return h
        sig = self.sign_cf
        trig = self.symmetry != "cubic" and self.axis == "trigonal"
        if c.shell == "d":
            dist = self.symmetry != "cubic"
            rh = self.symmetry == "rhombic"
            ds = self.ds if dist else 0.0
            dt4 = self.dt4 if dist else 0.0
            v2 = self.v2 if rh else 0.0
            v4 = self.v4 if rh else 0.0
            bt = -2.0 / 3.0 if trig else 1.0
            sc, s4, s2 = self.d_signs
            h += s2 * (ds / c.b2x) * b.stevens(2, 0) + s2 * (v2 / c.b2x) * b.stevens(2, 2)
            if c.L >= 2:
                h += ((sc * bt * self.dq + s4 * dt4) / c.b4x) * b.stevens(4, 0)
                h += s4 * (v4 / c.b4x) * b.stevens(4, 2)
                if trig:
                    h += sc * (bt * self.dq / c.b4x) * 20.0 * np.sqrt(2.0) * b.stevens(4, 3)
                else:
                    h += sc * (self.dq / c.b4x) * 5.0 * b.stevens(4, 4)
            return h
        p = self.f_parameters()
        b4t, b6t = ((-2.0 / 3.0, 16.0 / 9.0) if trig else (1.0, 1.0))
        B4, B6 = p["B4"], p["B6"]
        h += (b4t * B4 + p["B40"]) * b.stevens(4, 0) + (b6t * B6 + p["B60"]) * b.stevens(6, 0)
        h += p["B20"] * b.stevens(2, 0)
        h += p["B22"] * b.stevens(2, 2) + p["B42"] * b.stevens(4, 2) + p["B62"] * b.stevens(6, 2)
        if trig:
            h += (b4t * B4 * 20.0 * np.sqrt(2.0) + p["B43"]) * b.stevens(4, 3)
            h += (-b6t * B6 * 35.0 / np.sqrt(8.0) + p["B63"]) * b.stevens(6, 3)
            h += (b6t * B6 * 77.0 / 8.0 + p["B66"]) * b.stevens(6, 6)
        else:
            h += (5.0 * B4 + p["B44"]) * b.stevens(4, 4) + (-21.0 * B6 + p["B64"]) * b.stevens(6, 4)
        return sig * h

    @cached_property
    def spin_spin(self) -> np.ndarray:
        """-[(L.S)^2 + L.S/2 - L(L+1)S(S+1)/3] (multiply by rho); zero for S < 1 as in DSOXF."""
        b = self.basis
        if b.S < 1.0:
            return np.zeros((b.dim, b.dim))
        ls = b.LS
        return -(ls @ ls + 0.5 * ls - b.L * (b.L + 1) * b.S * (b.S + 1) / 3.0 * np.eye(b.dim))

    @cached_property
    def zero_field(self) -> np.ndarray:
        b = self.basis
        return self.crystal_field + self.spin_orbit * b.LS + self.rho * self.spin_spin


# ============================================================================ orbital bases
D_ORBITALS = ("x2-y2", "xy", "xz", "yz", "z2")
F_GENERAL_ORBITALS = ("x(x2-3y2)", "y(3x2-y2)", "z(x2-y2)", "xyz", "xz2", "yz2", "z3")
F_CUBIC_ORBITALS = ("x3", "y3", "z3", "x(z2-y2)", "y(z2-x2)", "z(x2-y2)", "xyz")


def orbital_vectors(L: int, kind: str = "real") -> tuple[tuple[str, ...], np.ndarray]:
    """(labels, U): column j of U is orbital j in the |m_L = L ... -L> basis (Condon-Shortley).

    d (dsoxfsub.f ll. 287-292): x2-y2 = (|2>+|-2>)/sqrt2, xy = -i(|2>-|-2>)/sqrt2,
    xz = -(|1>-|-1>)/sqrt2, yz = i(|1>+|-1>)/sqrt2, z2 = |0>.
    f general (ll. 423-428): x(x2-3y2) = -(|3>-|-3>)/sqrt2, y(3x2-y2) = i(|3>+|-3>)/sqrt2,
    z(x2-y2) = (|2>+|-2>)/sqrt2, xyz = -i(|2>-|-2>)/sqrt2, xz2 = -(|1>-|-1>)/sqrt2,
    yz2 = i(|1>+|-1>)/sqrt2, z3 = |0>.
    f cubic (ll. 589-596, Cotton): x3 = -[sqrt6 xz2 - sqrt10 x(x2-3y2)]/4,
    y3 = -[sqrt6 yz2 + sqrt10 y(3x2-y2)]/4, x(z2-y2) = [sqrt10 xz2 + sqrt6 x(x2-3y2)]/4,
    y(z2-x2) = [sqrt10 yz2 - sqrt6 y(3x2-y2)]/4; z3, z(x2-y2), xyz unchanged.
    """
    r = 1.0 / np.sqrt(2.0)
    n = 2 * L + 1

    def ket(m: int) -> np.ndarray:
        v = np.zeros(n, dtype=complex)
        v[L - m] = 1.0
        return v

    def cos_(m: int) -> np.ndarray:      # (-1)^m-type combinations, real "cos m phi"
        return (ket(m) + ket(-m)) * r if m % 2 == 0 else -(ket(m) - ket(-m)) * r

    def sin_(m: int) -> np.ndarray:
        return -1j * (ket(m) - ket(-m)) * r if m % 2 == 0 else 1j * (ket(m) + ket(-m)) * r

    if L == 2:
        cols = [cos_(2), sin_(2), cos_(1), sin_(1), ket(0)]
        return D_ORBITALS, np.stack(cols, axis=1)
    if L == 3:
        g = [cos_(3), sin_(3), cos_(2), sin_(2), cos_(1), sin_(1), ket(0)]
        if kind == "general":
            return F_GENERAL_ORBITALS, np.stack(g, axis=1)
        s6, s10 = np.sqrt(6.0) / 4.0, np.sqrt(10.0) / 4.0
        cub = [-(s6 * g[4] - s10 * g[0]), -(s6 * g[5] + s10 * g[1]), g[6],
               s10 * g[4] + s6 * g[0], s10 * g[5] - s6 * g[1], g[2], g[3]]
        return F_CUBIC_ORBITALS, np.stack(cub, axis=1)
    raise ValueError("orbital occupancies are defined for L = 2 (d orbitals) and L = 3 (f orbitals) only")
