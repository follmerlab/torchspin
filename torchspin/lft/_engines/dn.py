"""Parameter-independent d^n operators (Slater determinants), NumPy, built once and cached.

Copied from telser-tools (telser_tools.ddn.engine, telser_tools.ddn.terms and the
Gaunt coefficients of telser_tools.tanabe_sugano.determinants) so that torchspin has no
runtime dependency on telser-tools.  Parity with the original is checked in
torchspin/tests/test_lft_engines.py when telser-tools is installed.

* electron repulsion split into its F^0, F^2, F^4 parts (Slater-Condon rules, Gaunt
  coefficients c^k; Condon & Shortley, Griffith 1964 eq. 4.8);
* the 25 spin-free excitation operators E[m, m'] = sum_sigma a+_{m sigma} a_{m' sigma}, so that
  any one-electron ligand field V (5x5 Hermitian, |l=2, m> basis) is sum V[m, m'] E[m, m'];
* the spin-orbit operator sum_i l_i . s_i; total L, S, L^2, S^2 and the seniority operator;
* the free-ion terms in Telser's DDN order and their projectors.

Basis: every determinant of n electrons in the 10 spin orbitals index = 5*spin + (m + 2)
(spin 0 = alpha, m = -2..2), Condon-Shortley phases.  Energies are in cm^-1 by convention of
the callers; nothing here carries units.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from math import factorial, sqrt

import numpy as np

__all__ = ["M_VALUES", "Operators", "operators", "Term", "ddn_terms", "term_projectors", "term_basis",
           "U_REAL", "REAL_ORBITALS", "gaunt"]

REAL_ORBITALS = ("x2-y2", "xy", "xz", "yz", "z2")
_R2 = 1.0 / sqrt(2.0)
# columns: real orbitals (x2-y2, xy, xz, yz, z2); rows: m = -2..2
U_REAL = np.array([
    [_R2, 1j * _R2, 0, 0, 0],        # m = -2
    [0, 0, _R2, 1j * _R2, 0],        # m = -1
    [0, 0, 0, 0, 1],                 # m =  0
    [0, 0, -_R2, 1j * _R2, 0],       # m = +1
    [_R2, -1j * _R2, 0, 0, 0],       # m = +2
], dtype=complex)


def _w3j(j1, j2, j3, m1, m2, m3) -> float:
    """Wigner 3j symbol for integer arguments (Racah formula)."""
    if m1 + m2 + m3 != 0 or abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0
    if j3 < abs(j1 - j2) or j3 > j1 + j2:
        return 0.0
    f = factorial
    tri = f(j1 + j2 - j3) * f(j1 - j2 + j3) * f(-j1 + j2 + j3) / f(j1 + j2 + j3 + 1)
    pre = sqrt(tri * f(j1 + m1) * f(j1 - m1) * f(j2 + m2) * f(j2 - m2) * f(j3 + m3) * f(j3 - m3))
    s = 0.0
    for t in range(0, j1 + j2 + j3 + 1):
        d = (t, j3 - j2 + t + m1, j3 - j1 + t - m2, j1 + j2 - j3 - t, j1 - t - m1, j2 - t + m2)
        if min(d) < 0:
            continue
        s += (-1) ** t / np.prod([f(x) for x in d], dtype=float)
    return (-1) ** (j1 - j2 - m3) * pre * s


def gaunt(k: int, m: int, mp: int, l: int = 2) -> float:
    """c^k(l m, l m') = (-1)^m (2l+1) (l k l; 0 0 0)(l k l; -m m-m' m')."""
    return (-1) ** m * (2 * l + 1) * _w3j(l, k, l, 0, 0, 0) * _w3j(l, k, l, -m, m - mp, mp)


M_VALUES = (-2, -1, 0, 1, 2)
NSO = 10


def _sign(det: int, p: int) -> int:
    """(-1)^(number of occupied spin orbitals below p)."""
    return -1 if bin(det & ((1 << p) - 1)).count("1") % 2 else 1


def _l_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """l_x, l_y, l_z for l = 2 in the |m = -2..2> basis (Condon-Shortley phases)."""
    m = np.array(M_VALUES, dtype=float)
    lp = np.zeros((5, 5), dtype=complex)  # l+ |m> = sqrt(6 - m(m+1)) |m+1>
    for i in range(4):
        lp[i + 1, i] = sqrt(6.0 - m[i] * (m[i] + 1.0))
    return (lp + lp.conj().T) / 2, (lp - lp.conj().T) / 2j, np.diag(m).astype(complex)


def _s_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """s_x, s_y, s_z for s = 1/2 in the (alpha, beta) basis."""
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    return sx, sy, sz


def spin_orbital_l() -> list[np.ndarray]:
    """l_x, l_y, l_z as 10x10 spin-orbital matrices (identity in spin)."""
    return [np.kron(np.eye(2), lc) for lc in _l_matrices()]


def spin_orbital_s() -> list[np.ndarray]:
    """s_x, s_y, s_z as 10x10 spin-orbital matrices (identity in orbital)."""
    return [np.kron(sc, np.eye(5)) for sc in _s_matrices()]


def _determinants(n: int) -> list[int]:
    return [sum(1 << p for p in occ) for occ in combinations(range(NSO), n)]


def one_body(n: int, o: np.ndarray) -> np.ndarray:
    """Matrix of sum_pq o[p, q] a+_p a_q over the d^n determinants (o is 10x10 spin-orbital)."""
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
    """For every r<s: list of (p, q, [c0, c2, c4]) with p<q and <pq||rs> != 0 (F^k coefficients)."""
    orb = [p % 5 for p in range(NSO)]
    spin = [p // 5 for p in range(NSO)]
    mv = [M_VALUES[o] for o in orb]
    ks = (0, 2, 4)

    def direct(p, q, r, s):  # <pq|rs> = delta(sp,sr) delta(sq,ss) R^k
        if spin[p] != spin[r] or spin[q] != spin[s] or mv[p] + mv[q] != mv[r] + mv[s]:
            return np.zeros(3)
        return np.array([gaunt(k, mv[p], mv[r]) * gaunt(k, mv[s], mv[q]) for k in ks])

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
    """(3, nd, nd): coefficients of F^0, F^2, F^4 in sum_{i<j} 1/r_ij."""
    dets = _determinants(n)
    index = {d: i for i, d in enumerate(dets)}
    nd = len(dets)
    out = np.zeros((3, nd, nd))
    table = _antisymmetrized_pairs()
    for i, det in enumerate(dets):
        occ = [p for p in range(NSO) if det >> p & 1]
        for r, s in combinations(occ, 2):
            # a_s a_r |det>  (r < s): annihilate r first, then s
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
    """Parameter-independent operators of d^n in the determinant basis.

    Attributes (nd = C(10, n) determinants):

    ``dets``      bit-pattern of each determinant (bit p = spin orbital 5*spin + m + 2)
    ``coulomb``   (3, nd, nd) real; coefficients of F^0, F^2, F^4 (Condon-Shortley F^k)
    ``excitation`` (5, 5, nd, nd) real; E[a, b] = sum_sigma a+_{m_a sigma} a_{m_b sigma},
                  orbital index a = m + 2
    ``so``        (nd, nd) real; sum_i l_i . s_i
    ``L``, ``S``  (3, nd, nd) complex; total orbital and spin angular momentum (x, y, z)
    ``L2``, ``S2`` total L^2 and S^2; ``pair`` = Q+Q (seniority operator)
    ``n_orb``     (5, nd) occupation number of orbital m (diagonal in determinants)
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
    pair: np.ndarray
    n_orb: np.ndarray

    @property
    def size(self) -> int:
        return len(self.dets)

    def ligand_field(self, vm: np.ndarray) -> np.ndarray:
        """Many-electron matrix of the one-electron operator vm (5x5, |m=-2..2> basis)."""
        return np.tensordot(np.asarray(vm), self.excitation, axes=([0, 1], [0, 1]))


@lru_cache(maxsize=None)
def operators(n: int) -> Operators:
    """Build (once per n) every parameter-independent operator of d^n."""
    if int(n) != n or not 0 <= n <= 10:
        raise ValueError("n must be an integer 0..10")
    n = int(n)
    dets = _determinants(n)
    nd = len(dets)
    exc = np.zeros((5, 5, nd, nd))
    for a in range(5):
        for b in range(5):
            o = np.zeros((NSO, NSO))
            o[a, b] = o[5 + a, 5 + b] = 1.0
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
    # pair creation Q+ = sum_m (-1)^m a+_{m alpha} a+_{-m beta}  (L = 0, S = 0 pair)
    pair = np.zeros((nd, nd))
    if n >= 2:
        lower = _determinants(n - 2)
        lindex = {d: i for i, d in enumerate(lower)}
        qdag = np.zeros((nd, len(lower)))
        index = {d: i for i, d in enumerate(dets)}
        for j, det in enumerate(lower):
            for a, m in enumerate(M_VALUES):
                pa, pb = a, 5 + (4 - a)  # m alpha, -m beta
                if det >> pb & 1:
                    continue
                d1 = det | (1 << pb)
                s1 = _sign(det, pb)
                if d1 >> pa & 1:
                    continue
                s2 = _sign(d1, pa)
                qdag[index[d1 | (1 << pa)], j] += (-1) ** m * s1 * s2
        del lindex
        pair = qdag @ qdag.T
    n_orb = np.array([[((d >> a) & 1) + ((d >> (5 + a)) & 1) for d in dets] for a in range(5)], dtype=float)
    ops = Operators(n=n, dets=tuple(dets), coulomb=_two_body(n), excitation=exc, so=so.real,
                    L=L, S=S, L2=L2, S2=S2, pair=pair, n_orb=n_orb)
    for arr in (ops.coulomb, ops.excitation, ops.so, ops.L, ops.S, ops.L2, ops.S2, ops.pair):
        arr.setflags(write=False)
    return ops


L_LETTERS = "SPDFGHIKLM"

# (2S+1, L letter, seniority, DDN label) in DDNIO.F order (ll. 125-313)
_DDN_ORDER = {
    0: [(1, "S", 0, "1S")],
    1: [(2, "D", 1, "2D")],
    2: [(3, "F", 2, "3F"), (3, "P", 2, "3P"), (1, "G", 2, "1G"), (1, "D", 2, "1D"), (1, "S", 0, "1S")],
    3: [(4, "F", 3, "4F"), (4, "P", 3, "4P"), (2, "H", 3, "2H"), (2, "G", 3, "2G"), (2, "F", 3, "2F"),
        (2, "D", 1, "2Da"), (2, "D", 3, "2Db"), (2, "P", 3, "2P")],
    4: [(5, "D", 4, "5D"), (3, "H", 4, "3H"), (3, "G", 4, "3G"), (3, "F", 2, "3Fa"), (3, "F", 4, "3Fb"),
        (3, "D", 4, "3D"), (3, "P", 2, "3Pa"), (3, "P", 4, "3Pb"), (1, "I", 4, "1I"), (1, "G", 2, "1Ga"),
        (1, "G", 4, "1Gb"), (1, "F", 4, "1F"), (1, "D", 2, "1Da"), (1, "D", 4, "1Db"), (1, "S", 0, "1Sa"),
        (1, "S", 4, "1Sb")],
    5: [(6, "S", 5, "6S"), (4, "G", 5, "4G"), (4, "F", 3, "4F"), (4, "D", 5, "4D"), (4, "P", 3, "4P"),
        (2, "I", 5, "2I"), (2, "H", 3, "2H"), (2, "G", 3, "2Ga"), (2, "G", 5, "2Gb"), (2, "F", 3, "2Fa"),
        (2, "F", 5, "2Fb"), (2, "D", 1, "2Da"), (2, "D", 3, "2Db"), (2, "D", 5, "2Dc"), (2, "P", 3, "2P"),
        (2, "S", 5, "2S")],
}


@dataclass(frozen=True)
class Term:
    """A free-ion term (2S+1)L with seniority v; ``label`` is DDN's (e.g. '2Da')."""

    multiplicity: int
    L: int
    seniority: int
    label: str

    @property
    def S(self) -> float:
        return (self.multiplicity - 1) / 2

    @property
    def dimension(self) -> int:
        return self.multiplicity * (2 * self.L + 1)

    @property
    def symbol(self) -> str:
        return f"{self.multiplicity}{L_LETTERS[self.L]}"


def ddn_terms(n: int) -> list[Term]:
    """Terms of d^n in DDN's matrix order (the list of d^(10-n) for n > 5)."""
    nd = n if n <= 5 else 10 - n
    return [Term(m, L_LETTERS.index(l), v, lab) for m, l, v, lab in _DDN_ORDER[nd]]


def _spectral_projector(op: np.ndarray, value: float, tol: float = 1e-6) -> np.ndarray:
    w, v = np.linalg.eigh(op)
    sub = v[:, np.abs(w - value) < tol]
    return sub @ sub.conj().T


@lru_cache(maxsize=None)
def _term_data(n: int):
    ops = operators(n)
    terms = ddn_terms(n)
    w_l, v_l = np.linalg.eigh(ops.L2)
    w_s, v_s = np.linalg.eigh(ops.S2)
    w_p, v_p = np.linalg.eigh(ops.pair)

    def proj(w, v, val):
        sub = v[:, np.abs(w - val) < 1e-6]
        return sub @ sub.T

    projs = []
    for t in terms:
        lam = (n - t.seniority) * (12 - n - t.seniority) / 4
        p = proj(w_l, v_l, t.L * (t.L + 1)) @ proj(w_s, v_s, t.S * (t.S + 1)) @ proj(w_p, v_p, lam)
        if abs(np.trace(p) - t.dimension) > 1e-6:
            raise AssertionError(f"term {t.label} of d{n}: dimension {np.trace(p)} != {t.dimension}")
        projs.append(p)
    total = sum(projs)
    if not np.allclose(total, np.eye(ops.size), atol=1e-9):
        raise AssertionError(f"terms of d{n} do not span the determinant space")
    return terms, np.array(projs)


def term_projectors(n: int) -> tuple[list[Term], np.ndarray]:
    """(terms, P) with P[t] the real projector onto term t in the determinant basis."""
    return _term_data(n)


@lru_cache(maxsize=None)
def term_basis(n: int) -> tuple[list[Term], np.ndarray, list[tuple[int, float, float]]]:
    """Orthonormal |v S L M_S M_L> states in DDN's index order (columns of T).

    Index order is DDN's: terms in DDN order; inside a term M_L runs from +L down, and inside
    each M_L, M_S runs from +S down.  Each term's top state (M_L = L, M_S = S) is the unique
    vector of the term subspace with those quantum numbers, phased so that its largest
    determinant coefficient is positive; the other states follow by Condon-Shortley lowering
    (L-, S-).  Overall term phases are therefore conventional (DDN's may differ by a sign per
    term).  Returns (terms, T, labels) with labels[i] = (term index, M_L, M_S).
    """
    ops = operators(n)
    terms, projs = _term_data(n)
    lz, sz = ops.L[2].real.diagonal(), ops.S[2].real.diagonal()
    lminus = ops.L[0] - 1j * ops.L[1]
    sminus = ops.S[0] - 1j * ops.S[1]
    cols, labels = [], []
    for it, (t, p) in enumerate(zip(terms, projs)):
        sel = np.where((np.abs(lz - t.L) < 1e-9) & (np.abs(sz - t.S) < 1e-9))[0]
        sub = p[np.ix_(sel, sel)]
        w, v = np.linalg.eigh(sub)
        if np.sum(w > 0.5) != 1:
            raise AssertionError(f"top state of {t.label} is not unique")
        top = np.zeros(ops.size, dtype=complex)
        top[sel] = v[:, np.argmax(w)]
        top *= np.sign(top[np.argmax(np.abs(top))].real)
        ml_states = [top]
        for ml in np.arange(t.L, -t.L, -1):  # lower M_L
            nxt = lminus @ ml_states[-1] / np.sqrt(t.L * (t.L + 1) - ml * (ml - 1))
            ml_states.append(nxt)
        for ml, st in zip(np.arange(t.L, -t.L - 1, -1), ml_states):
            cur = st
            for ms in np.arange(t.S, -t.S - 1, -1):
                cols.append(cur)
                labels.append((it, float(ml), float(ms)))
                if ms > -t.S:
                    cur = sminus @ cur / np.sqrt(t.S * (t.S + 1) - ms * (ms - 1))
    tb = np.array(cols).T
    if not np.allclose(tb.conj().T @ tb, np.eye(ops.size), atol=1e-9):
        raise AssertionError("term basis is not orthonormal")
    return terms, tb, labels
