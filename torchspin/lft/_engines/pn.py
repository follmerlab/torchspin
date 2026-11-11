"""Parameter-independent p^n operators (Slater determinants), NumPy, built once and cached.

Copied from telser-tools (``telser_tools.dpn.engine``, ``telser_tools.dpn.terms`` and the
real-orbital transform of ``telser_tools.dpn.ligand_field``) so that torchspin has no runtime
dependency on telser-tools.  Parity with the original is checked in
torchspin/tests/test_lft_dpn.py when telser-tools is installed.

* electron repulsion split into its F^0, F^2 parts by the Slater-Condon rules with Gaunt
  coefficients ``c^k(l m, l m')`` (Condon-Shortley phases; :func:`torchspin.lft._engines.dn.gaunt`);
* the 9 spin-free one-electron excitation operators ``E[m, m'] = sum_sigma a+_{m sigma} a_{m' sigma}``,
  so that any one-electron ligand field ``V`` (3x3 Hermitian in the |l=1, m> basis) is
  ``H_LF = sum_{m m'} V[m, m'] E[m, m']``;
* the spin-orbit operator ``sum_i l_i . s_i``; total L, S, L^2, S^2 and orbital occupations;
* the free-ion terms of p^n in Telser's DPN order and their projectors.

Basis: every determinant of n electrons in the 6 spin orbitals ``index = 3*spin + (m + 1)``
(spin 0 = alpha, m = -1, 0, 1); n = 0..6 with no hole formalism (p^3: 20 determinants).
Nothing here carries units (the callers use cm^-1).

Real p orbitals (px, py, pz), positive multiples of x, y, z; with Condon-Shortley |m> = Y_1^m:
pz = |0>, px = (|-1> - |1>)/sqrt2, py = i(|1> + |-1>)/sqrt2.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from math import sqrt

import numpy as np

from .dn import gaunt

__all__ = ["L_LETTERS", "L_SHELL", "M_VALUES", "NSO", "Operators", "REAL_ORBITALS", "Term", "U_REAL", "dpn_terms",
           "one_body", "operators", "spin_orbital_l", "spin_orbital_s", "term_projectors"]

REAL_ORBITALS = ("px", "py", "pz")
_R2 = 1.0 / sqrt(2.0)
# columns: real orbitals (px, py, pz); rows: m = -1, 0, 1
U_REAL = np.array([
    [_R2, 1j * _R2, 0],     # m = -1
    [0, 0, 1],              # m =  0
    [-_R2, 1j * _R2, 0],    # m = +1
], dtype=complex)

L_SHELL = 1
M_VALUES = tuple(range(-L_SHELL, L_SHELL + 1))   # (-1, 0, 1)
NORB = len(M_VALUES)
NSO = 2 * NORB
K_VALUES = tuple(range(0, 2 * L_SHELL + 1, 2))   # (0, 2)


def _sign(det: int, p: int) -> int:
    """(-1)^(number of occupied spin orbitals below p)."""
    return -1 if bin(det & ((1 << p) - 1)).count("1") % 2 else 1


def _l_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """l_x, l_y, l_z for l = L_SHELL in the |m = -l..l> basis (Condon-Shortley phases)."""
    m = np.array(M_VALUES, dtype=float)
    ll = L_SHELL * (L_SHELL + 1)
    lp = np.zeros((NORB, NORB), dtype=complex)  # l+ |m> = sqrt(l(l+1) - m(m+1)) |m+1>
    for i in range(NORB - 1):
        lp[i + 1, i] = sqrt(ll - m[i] * (m[i] + 1.0))
    return (lp + lp.conj().T) / 2, (lp - lp.conj().T) / 2j, np.diag(m).astype(complex)


def _s_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    return sx, sy, sz


def spin_orbital_l() -> list[np.ndarray]:
    """l_x, l_y, l_z as 6x6 spin-orbital matrices (identity in spin)."""
    return [np.kron(np.eye(2), lc) for lc in _l_matrices()]


def spin_orbital_s() -> list[np.ndarray]:
    """s_x, s_y, s_z as 6x6 spin-orbital matrices (identity in orbital)."""
    return [np.kron(sc, np.eye(NORB)) for sc in _s_matrices()]


def _determinants(n: int) -> list[int]:
    return [sum(1 << p for p in occ) for occ in combinations(range(NSO), n)]


def one_body(n: int, o: np.ndarray) -> np.ndarray:
    """Matrix of sum_pq o[p, q] a+_p a_q over the p^n determinants (o is 6x6 spin-orbital)."""
    dets = _determinants(n)
    index = {d: i for i, d in enumerate(dets)}
    nd = len(dets)
    out = np.zeros((nd, nd), dtype=np.result_type(o, float))
    nz = [(p, q, o[p, q]) for p in range(NSO) for q in range(NSO) if o[p, q] != 0]
    for i, det in enumerate(dets):
        for p, q, val in nz:
            if not det >> q & 1:
                continue
            d1 = det ^ (1 << q)
            if d1 >> p & 1:
                continue
            s = _sign(det, q) * _sign(d1, p)
            out[index[d1 | (1 << p)], i] += s * val
    return out


@lru_cache(maxsize=None)
def _antisymmetrized_pairs():
    """For every r<s: list of (p, q, [c^k]) with p<q and <pq||rs> != 0 (F^k coefficients, k in K_VALUES)."""
    orb = [p % NORB for p in range(NSO)]
    spin = [p // NORB for p in range(NSO)]
    mv = [M_VALUES[o] for o in orb]

    def direct(p, q, r, s):  # <pq|rs> = delta(sp,sr) delta(sq,ss) sum_k c^k(m_p, m_r) c^k(m_s, m_q) F^k
        if spin[p] != spin[r] or spin[q] != spin[s] or mv[p] + mv[q] != mv[r] + mv[s]:
            return np.zeros(len(K_VALUES))
        return np.array([gaunt(k, mv[p], mv[r], l=L_SHELL) * gaunt(k, mv[s], mv[q], l=L_SHELL) for k in K_VALUES])

    table = {}
    for r, s in combinations(range(NSO), 2):
        lst = []
        for p, q in combinations(range(NSO), 2):
            c = direct(p, q, r, s) - direct(p, q, s, r)
            if np.any(np.abs(c) > 1e-14):
                lst.append((p, q, c))
        table[(r, s)] = lst
    return table


def _two_body(n: int) -> np.ndarray:
    """(len(K_VALUES), nd, nd): coefficients of F^0, F^2 in sum_{i<j} 1/r_ij."""
    dets = _determinants(n)
    index = {d: i for i, d in enumerate(dets)}
    nd = len(dets)
    out = np.zeros((len(K_VALUES), nd, nd))
    table = _antisymmetrized_pairs()
    for i, det in enumerate(dets):
        occ = [p for p in range(NSO) if det >> p & 1]
        for r, s in combinations(occ, 2):
            d1 = det ^ (1 << r)
            s1 = _sign(det, r)
            d2 = d1 ^ (1 << s)
            s2 = _sign(d1, s)
            for p, q, c in table[(r, s)]:
                if d2 >> q & 1 or d2 >> p & 1:
                    continue
                d3 = d2 | (1 << q)
                s3 = _sign(d2, q)
                s4 = _sign(d3, p)
                out[:, index[d3 | (1 << p)], i] += s1 * s2 * s3 * s4 * c
    return out


@dataclass(frozen=True, eq=False)
class Operators:
    """Parameter-independent operators of p^n in the determinant basis (nd = C(6, n)).

    ``dets``       bit pattern of each determinant (bit p = spin orbital 3*spin + m + 1)
    ``coulomb``    (2, nd, nd) real; coefficients of the Slater integrals F^0 and F^2
    ``excitation`` (3, 3, nd, nd) real; E[a, b] = sum_sigma a+_{m_a sigma} a_{m_b sigma}, a = m + 1
    ``so``         (nd, nd) real; sum_i l_i . s_i
    ``L``, ``S``   (3, nd, nd) complex; total orbital and spin angular momentum (x, y, z)
    ``L2``, ``S2`` total L^2 and S^2
    ``n_orb``      (3, nd) occupation number of orbital m (diagonal in determinants)
    """

    n: int
    dets: tuple[int, ...]
    coulomb: np.ndarray
    excitation: np.ndarray
    so: np.ndarray
    L: np.ndarray
    S: np.ndarray
    L2: np.ndarray
    S2: np.ndarray
    n_orb: np.ndarray

    @property
    def size(self) -> int:
        return len(self.dets)

    @property
    def n_pairs(self) -> int:
        return self.n * (self.n - 1) // 2

    def ligand_field(self, vm: np.ndarray) -> np.ndarray:
        """Many-electron matrix of the one-electron operator vm (3x3, |m = -1, 0, 1> basis)."""
        return np.tensordot(np.asarray(vm), self.excitation, axes=([0, 1], [0, 1]))


@lru_cache(maxsize=None)
def operators(n: int) -> Operators:
    """Build (once per n) every parameter-independent operator of p^n."""
    if int(n) != n or not 0 <= n <= NSO:
        raise ValueError(f"n must be an integer 0..{NSO}")
    n = int(n)
    dets = _determinants(n)
    nd = len(dets)
    exc = np.zeros((NORB, NORB, nd, nd))
    for a in range(NORB):
        for b in range(NORB):
            o = np.zeros((NSO, NSO))
            o[a, b] = o[NORB + a, NORB + b] = 1.0
            exc[a, b] = one_body(n, o)
    lso, sso = spin_orbital_l(), spin_orbital_s()
    so1 = sum(lc @ sc for lc, sc in zip(lso, sso))
    so = one_body(n, so1)
    if np.max(np.abs(so.imag)) > 1e-12:
        raise AssertionError("l.s should be real in the Condon-Shortley |m, ms> basis")
    L = np.array([one_body(n, x) for x in lso])
    S = np.array([one_body(n, x) for x in sso])
    L2 = np.real(sum(x @ x for x in L))
    S2 = np.real(sum(x @ x for x in S))
    n_orb = np.array([[((d >> a) & 1) + ((d >> (NORB + a)) & 1) for d in dets] for a in range(NORB)], dtype=float)
    ops = Operators(n=n, dets=tuple(dets), coulomb=_two_body(n), excitation=exc, so=so.real, L=L, S=S, L2=L2, S2=S2,
                    n_orb=n_orb)
    for arr in (ops.coulomb, ops.excitation, ops.so, ops.L, ops.S, ops.L2, ops.S2, ops.n_orb):
        arr.setflags(write=False)
    return ops


# ------------------------------------------------------------------------------ terms (telser_tools.dpn.terms)
L_LETTERS = "SPDFGHIK"

_DPN_ORDER = {
    0: [(1, "S")],
    1: [(2, "P")],
    2: [(3, "P"), (1, "D"), (1, "S")],
    3: [(4, "S"), (2, "D"), (2, "P")],
}


@dataclass(frozen=True)
class Term:
    """A free-ion term (2S+1)L; ``label`` is e.g. '3P'."""

    multiplicity: int
    L: int

    @property
    def S(self) -> float:
        return (self.multiplicity - 1) / 2

    @property
    def dimension(self) -> int:
        return self.multiplicity * (2 * self.L + 1)

    @property
    def label(self) -> str:
        return f"{self.multiplicity}{L_LETTERS[self.L]}"

    symbol = label


def dpn_terms(n: int) -> list[Term]:
    """Terms of p^n in DPN's matrix order (the list of p^(6-n) for n > 3)."""
    npn = n if n <= 3 else 6 - n
    return [Term(m, L_LETTERS.index(l)) for m, l in _DPN_ORDER[npn]]


@lru_cache(maxsize=None)
def _term_data(n: int):
    ops = operators(n)
    terms = dpn_terms(n)
    w_l, v_l = np.linalg.eigh(ops.L2)
    w_s, v_s = np.linalg.eigh(ops.S2)

    def proj(w, v, val):
        sub = v[:, np.abs(w - val) < 1e-6]
        return sub @ sub.T

    projs = []
    for t in terms:
        p = proj(w_l, v_l, t.L * (t.L + 1)) @ proj(w_s, v_s, t.S * (t.S + 1))
        if abs(np.trace(p) - t.dimension) > 1e-6:
            raise AssertionError(f"term {t.label} of p{n}: dimension {np.trace(p)} != {t.dimension}")
        projs.append(p)
    if not np.allclose(sum(projs), np.eye(ops.size), atol=1e-9):
        raise AssertionError(f"terms of p{n} do not span the determinant space")
    arr = np.array(projs)
    arr.setflags(write=False)
    return terms, arr


def term_projectors(n: int) -> tuple[list[Term], np.ndarray]:
    """(terms, P) with P[t] the real projector onto term t in the determinant basis."""
    return _term_data(n)
