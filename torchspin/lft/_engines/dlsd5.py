"""NumPy DLSD5 solver: low-spin d5 g values <-> ligand field (copied from telser-tools).

Verbatim copy of three telser-tools modules (J. Telser's DLSD5.F v1.52 reimplemented), merged
into one file; :mod:`torchspin.lft.dlsd5` runs these on detached values to make every discrete
choice of the inverse (branches, sign/permutation of g, axis relabeling) and attaches exact
derivatives afterwards.

* ``telser_tools.dlsd5.hamiltonian``: exact one-hole t2g Hamiltonian (``solve_t2g_hole``, ...).
* ``telser_tools.dlsd5.tetragonal``: McGarvey's closed forms and the g -> (a, b, c, k) solver.
  ``from_g`` is renamed :func:`tetragonal_from_g`.
* ``telser_tools.dlsd5.trigonal``: trigonal model.  Renamed to avoid clashes:
  ``from_g`` -> :func:`trigonal_from_g`, ``from_coefficients`` -> :func:`trigonal_from_coefficients`,
  ``g_from_coefficients`` -> :func:`trigonal_g_from_coefficients`, ``delta_from_coefficients`` ->
  :func:`trigonal_delta_from_coefficients`.

Energies are in units of the spin-orbit constant xi (DLSD5's convention).  Only the imports and
the names above differ from the originals; ``torchspin/tests/test_lft_dlsd5.py`` checks that the
copy reproduces telser-tools exactly.

References: B. R. McGarvey, Coord. Chem. Rev. 1998, 170, 75-92; twist angle: Dolder et al.,
Inorg. Chem. 2003, 42, 1343-1349.  Differences from the original Fortran (E0 sign, axis
relabeling, Newton polish, iteration caps) are documented in telser-tools ``docs/dlsd5.md``.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import brentq, root, root_scalar

__all__ = ["T2G_LABELS", "BETA_OCT", "RHOMBIC_LIMIT", "K_PLAUSIBLE", "HoleState", "TetragonalSolution",
           "TrigonalSolution", "solve_t2g_hole", "tetragonal_cf", "trigonal_cf", "abc_to_ABC", "ABC_to_abc",
           "g_from_abc", "ligand_field_from_abc", "to_standard_axes", "from_real_coefficients",
           "from_complex_coefficients", "tetragonal_from_g", "orbital_energies", "plausible",
           "trigonal_g_from_coefficients", "trigonal_delta_from_coefficients", "epi_from_twist",
           "trigonal_from_coefficients", "trigonal_from_g"]

RT2 = math.sqrt(2.0)
RT6 = math.sqrt(6.0)

# ======== hamiltonian
T2G_LABELS = ("xy", "xz", "yz")

_PAULI = (
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
)
_SPIN = tuple(p / 2 for p in _PAULI)


def _t2g_orbital_operators() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lx, Ly, Lz of l = 2 projected onto the real (xy, xz, yz) basis."""
    m = np.arange(2, -3, -1)  # |2>, |1>, |0>, |-1>, |-2>
    lz = np.diag(m).astype(complex)
    lp = np.zeros((5, 5), dtype=complex)
    for i in range(1, 5):
        lp[i - 1, i] = np.sqrt(6 - m[i] * (m[i] + 1))
    lx = (lp + lp.conj().T) / 2
    ly = (lp - lp.conj().T) / 2j

    r = 1 / np.sqrt(2)
    xy = np.array([-1j * r, 0, 0, 0, 1j * r])  # Condon-Shortley phases
    xz = np.array([0, -r, 0, r, 0], dtype=complex)
    yz = np.array([0, 1j * r, 0, 1j * r, 0])
    u = np.stack([xy, xz, yz], axis=1)
    return tuple(u.conj().T @ op @ u for op in (lx, ly, lz))  # type: ignore[return-value]


_L = _t2g_orbital_operators()


def tetragonal_cf(delta: float, v: float) -> np.ndarray:
    """Crystal-field matrix in (xy, xz, yz): xy at Delta, xz at +V/2, yz at -V/2."""
    return np.diag([delta, v / 2.0, -v / 2.0])


def trigonal_cf(delta: float) -> np.ndarray:
    """Trigonal field along [111]: a1 = (xy+xz+yz)/sqrt3 at +2Delta/3, e at -Delta/3.

    E(a1) - E(e) = Delta.  This is the octahedral-beta (beta^2 = 1/3) limit of the
    DLSD5 trigonal model, i.e. no t2g/eg mixing.
    """
    a1 = np.ones(3) / np.sqrt(3)
    return delta * (np.outer(a1, a1) - np.eye(3) / 3.0)


@dataclass(frozen=True)
class HoleState:
    """Result of :func:`solve_t2g_hole`."""

    g_principal: np.ndarray  # |g| principal values, ascending
    g_axes: np.ndarray  # columns: principal axes in the (x, y, z) molecular frame
    g_tensor: np.ndarray  # signed 3x3 g (rows: field direction, cols: effective spin)
    doublet_energies: np.ndarray  # 3 Kramers-doublet energies (one-electron, / xi), ascending
    hole_weights: dict  # orbital populations of the hole doublet {"xy","xz","yz"}

    @property
    def g_sign_product(self) -> float:
        """sign(gx*gy*gz) = sign(det g); the only sign accessible to EPR (via e.g. hyperfine)."""
        return float(np.sign(np.linalg.det(self.g_tensor)))

    @property
    def hole_energy(self) -> float:
        """One-electron energy of the doublet holding the hole (the highest one)."""
        return float(self.doublet_energies[-1])


def solve_t2g_hole(cf: np.ndarray, k: float = 1.0, xi: float = 1.0) -> HoleState:
    """Diagonalize CF + xi L.S in t2g and return the hole (highest) Kramers doublet.

    Parameters
    ----------
    cf : (3, 3) real symmetric crystal-field matrix in the (xy, xz, yz) basis, units of xi.
    k : orbital reduction (Stevens) factor applied to L in the Zeeman operator only.
    xi : spin-orbit constant (leave at 1 to work in units of xi).
    """
    cf = np.asarray(cf, dtype=float)
    i2, i3 = np.eye(2), np.eye(3)
    h = np.kron(cf, i2) + xi * sum(np.kron(_L[i], _SPIN[i]) for i in range(3))
    w, vecs = np.linalg.eigh(h)
    doublet = vecs[:, -2:]

    g = np.zeros((3, 3))
    for i in range(3):
        mu = k * np.kron(_L[i], i2) + 2.0 * np.kron(i3, _SPIN[i])
        m = doublet.conj().T @ mu @ doublet
        for j in range(3):
            g[i, j] = np.real(np.trace(m @ _PAULI[j]))

    gg_vals, gg_vecs = np.linalg.eigh(g @ g.T)
    proj = (doublet @ doublet.conj().T).real.diagonal().reshape(3, 2).sum(axis=1) / 2.0
    return HoleState(
        g_principal=np.sqrt(np.clip(gg_vals, 0.0, None)),
        g_axes=gg_vecs,
        g_tensor=g,
        doublet_energies=w[::2].copy(),
        hole_weights=dict(zip(T2G_LABELS, proj)),
    )

# ======== tetragonal
RHOMBIC_LIMIT = 2.0 / 3.0


# --------------------------------------------------------------------------- basis


def abc_to_ABC(a: float, b: float, c: float) -> tuple[float, float, float]:
    """(d+1, xy, d-1) -> (xy, yz, xz).  McGarvey eq. 10."""
    return b, (a + c) / RT2, (a - c) / RT2


def ABC_to_abc(A: float, B: float, C: float) -> tuple[float, float, float]:
    """(xy, yz, xz) -> (d+1, xy, d-1).  Inverse of McGarvey eq. 10."""
    return (B + C) / RT2, A, (B - C) / RT2


# --------------------------------------------------------------------------- closed forms


def g_from_abc(a: float, b: float, c: float, k: float) -> tuple[float, float, float]:
    """Signed (gx, gy, gz) from McGarvey eq. 4."""
    gx = 2.0 * (-b * b + 2.0 * a * c - RT2 * k * b * (a - c))
    gy = 2.0 * (-b * b - 2.0 * a * c - RT2 * k * b * (a + c))
    gz = 2.0 * (b * b - a * a - c * c + k * (c * c - a * a))
    return gx, gy, gz


def ligand_field_from_abc(a: float, b: float, c: float) -> tuple[float, float, float]:
    """(Delta/xi, V/xi, E0/xi) implied by the hole wavefunction.

    Raises ZeroDivisionError when b = 0 or a^2 = c^2 (Delta undefined / infinite).
    """
    denom = a * a - c * c
    if b == 0.0 or denom == 0.0:
        raise ZeroDivisionError("Delta undefined: b = 0 or a^2 = c^2")
    delta = (b * (1.0 - b * b) + RT2 * a * (1.0 - 2.0 * a * a)) / (2.0 * b * denom)
    v = c * (2.0 * a + RT2 * b) / (-denom)
    e0 = delta + a / (RT2 * b)
    return delta, v, e0


# --------------------------------------------------------------------------- result type


@dataclass(frozen=True)
class TetragonalSolution:
    """One solution of the tetragonal / rhombic LS d5 problem.

    Coefficients are given in both of McGarvey's bases:
    ``a, b, c`` multiply |d+1>, |xy>, |d-1>;  ``A, B, C`` multiply |xy>, |yz>, |xz>.
    Energies are in units of the spin-orbit constant xi.
    """

    a: float
    b: float
    c: float
    k: float
    gx: float
    gy: float
    gz: float
    delta: float | None
    v: float | None
    e0: float | None
    label: str = ""
    converged: bool = True
    iterations: int = 0
    relabeled: str | None = None
    ground_doublet: bool | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def A(self) -> float:  # noqa: N802 - literature notation
        return abc_to_ABC(self.a, self.b, self.c)[0]

    @property
    def B(self) -> float:  # noqa: N802
        return abc_to_ABC(self.a, self.b, self.c)[1]

    @property
    def C(self) -> float:  # noqa: N802
        return abc_to_ABC(self.a, self.b, self.c)[2]

    @property
    def weights(self) -> dict[str, float]:
        """Orbital populations of the hole: {"xy": A^2, "yz": B^2, "xz": C^2}."""
        A, B, C = abc_to_ABC(self.a, self.b, self.c)
        return {"xy": A * A, "yz": B * B, "xz": C * C}

    @property
    def g(self) -> tuple[float, float, float]:
        return self.gx, self.gy, self.gz

    @property
    def v_over_delta(self) -> float | None:
        if self.delta is None or self.v is None or self.delta == 0.0:
            return None
        return self.v / self.delta

    @property
    def is_physical(self) -> bool:
        """Converged, finite |Delta/xi| <= 100 (DLSD5's limit) and the hole ground doublet."""
        return (self.converged and self.delta is not None and abs(self.delta) <= 100.0
                and self.ground_doublet is not False)


def _solution_from_abc(a, b, c, k, **kw) -> TetragonalSolution:
    gx, gy, gz = g_from_abc(a, b, c, k)
    notes = list(kw.pop("notes", ()))
    try:
        delta, v, e0 = ligand_field_from_abc(a, b, c)
    except ZeroDivisionError as exc:
        delta = v = e0 = None
        notes.append(str(exc))
    return TetragonalSolution(a, b, c, k, gx, gy, gz, delta, v, e0, notes=tuple(notes), **kw)


# --------------------------------------------------------------------------- relabeling

# Axis relabelings expressed in the real (A=xy, B=yz, C=xz) basis and on (gx, gy, gz).
_RELABELINGS = {
    "x<->z": (lambda A, B, C: (B, A, C), lambda gx, gy, gz: (gz, gy, gx)),
    "y<->z": (lambda A, B, C: (C, B, A), lambda gx, gy, gz: (gx, gz, gy)),
}


def to_standard_axes(sol: TetragonalSolution) -> TetragonalSolution:
    """Relabel axes so that |V/Delta| <= 2/3 (the unique orbital is called xy).

    g values are permuted, not recomputed differently: the physics is unchanged.
    """
    rho = sol.v_over_delta
    if rho is None or abs(rho) <= RHOMBIC_LIMIT + 1e-12:
        return sol
    best = None
    for name, (coef_map, _g_map) in _RELABELINGS.items():
        a, b, c = ABC_to_abc(*coef_map(sol.A, sol.B, sol.C))
        cand = _solution_from_abc(
            a, b, c, sol.k, label=sol.label, converged=sol.converged,
            iterations=sol.iterations, relabeled=name, notes=sol.notes,
        )
        r = cand.v_over_delta
        if r is not None and (best is None or abs(r) < abs(best.v_over_delta)):
            best = cand
    if best is None:
        return replace(sol, notes=sol.notes + ("could not relabel to |V/Delta| <= 2/3",))
    return replace(best, notes=best.notes + (
        f"axes relabeled ({best.relabeled}) so that |V/Delta| <= 2/3; g values permuted",))


# --------------------------------------------------------------------------- forward


def _normalize_pair(p: float, q: float, names: tuple[str, str, str]) -> tuple[float, float, float, list[str]]:
    """DLSD5's normalization: return (p, q, |r|) with p^2 + q^2 + r^2 = 1."""
    notes: list[str] = []
    if p * p >= 1.0:
        notes.append(f"{names[0]}^2 >= 1: set {names[0]} = 1, {names[1]} = {names[2]} = 0")
        return math.copysign(1.0, p), 0.0, 0.0, notes
    if p * p + q * q >= 1.0:
        notes.append(f"{names[0]}^2 + {names[1]}^2 >= 1: {names[1]} rescaled, {names[2]} = 0")
        return p, math.copysign(math.sqrt(1.0 - p * p), q), 0.0, notes
    return p, q, math.sqrt(1.0 - p * p - q * q), notes


def from_real_coefficients(A: float, B: float, k: float, standardize: bool = True
                           ) -> tuple[TetragonalSolution, TetragonalSolution]:
    """g values and ligand field from the dxy (A) and dyz (B) coefficients.

    C (dxz) = +/- sqrt(1 - A^2 - B^2); both signs are returned (they are physically
    distinct states).  Equivalent to DLSD5 option G / tetragonal / A.
    """
    A, B, Cabs, notes = _normalize_pair(A, B, ("A", "B", "C"))
    out = []
    for sign, lab in ((1.0, "+C"), (-1.0, "-C")):
        a, b, c = ABC_to_abc(A, B, sign * Cabs)
        out.append(_finish(_solution_from_abc(a, b, c, k, label=lab, notes=notes), standardize))
    return out[0], out[1]


def from_complex_coefficients(a: float, b: float, k: float, standardize: bool = True
                              ) -> tuple[TetragonalSolution, TetragonalSolution]:
    """g values and ligand field from the d+1 (a) and dxy (b) coefficients.

    c (d-1) = +/- sqrt(1 - a^2 - b^2).  The two signs differ only by an x<->y
    relabeling (gx <-> gy, V -> -V).  Equivalent to DLSD5 option G / tetragonal / a.
    """
    a, b, cabs, notes = _normalize_pair(a, b, ("a", "b", "c"))
    out = []
    for sign, lab in ((1.0, "+c"), (-1.0, "-c")):
        out.append(_finish(_solution_from_abc(a, b, sign * cabs, k, label=lab, notes=notes), standardize))
    return out[0], out[1]


# --------------------------------------------------------------------------- inverse


def _telser_fixed_point(gx, gy, gz, max_iter):
    """DLSD5.F lines 444-470: damped fixed point on k.  Returns (a, c, k, iters, ok)."""
    gxy, gmxy = (gx + gy) / 2.0, (gx - gy) / 2.0
    k, c = 1.0, 0.0
    a = 0.0
    for it in range(1, max_iter + 1):
        a2 = 0.5 * (2.0 - gz) / (2.0 + k) - (2.0 - k) * c * c / (2.0 + k)
        if a2 <= 0.0:
            return a, c, k, it, False
        a = math.sqrt(a2)
        b2 = 1.0 - a2 - c * c
        if b2 <= 0.0:
            return a, c, k, it, False
        b = math.sqrt(b2)
        k_new = -(2.0 * b2 + gxy) / (2.0 * RT2 * b * a)
        if abs(k_new - k) < 1e-5:
            return a, c, k, it, True
        k = 0.5 * (k + k_new)
        c = gmxy / 2.0 / (2.0 * a + RT2 * k * b)
    return a, c, k, max_iter, False


def _polish(gx, gy, gz, a0, c0, k0):
    target = np.array([gx, gy, gz])

    def resid(p):
        a, c, k = p
        b2 = 1.0 - a * a - c * c
        b = math.sqrt(b2) if b2 > 0 else 0.0
        return np.array(g_from_abc(a, b, c, k)) - target

    # Success is judged on the residual itself, not the solver's step-size flag.
    r = root(resid, [a0, c0, k0], method="hybr", options={"xtol": 1e-13})
    a, c, k = r.x
    b2 = 1.0 - a * a - c * c
    ok = b2 > 0 and bool(np.all(np.isfinite(r.x))) and np.max(np.abs(resid(r.x))) < 1e-10
    return a, (math.sqrt(b2) if b2 > 0 else 0.0), c, k, ok


def _is_hole_doublet(sol: TetragonalSolution) -> bool | None:
    """True if (a,b,c) is the *highest* Kramers doublet of H(Delta, V), i.e. the t2g^5 ground state."""
    if sol.delta is None or sol.e0 is None:
        return None
    st = solve_t2g_hole(tetragonal_cf(sol.delta, sol.v), sol.k)
    return bool(abs(st.hole_energy - sol.e0) < 1e-6 * max(1.0, abs(sol.e0)))


def _finish(sol: TetragonalSolution, standardize: bool) -> TetragonalSolution:
    if sol.delta is not None and abs(sol.delta) > 100.0:
        sol = replace(sol, notes=sol.notes + ("|Delta/xi| > 100: no meaningful solution",))
    sol = to_standard_axes(sol) if standardize else sol
    ground = _is_hole_doublet(sol)
    if ground is False:
        sol = replace(sol, ground_doublet=False, notes=sol.notes + (
            "wavefunction is an EXCITED Kramers doublet of the implied ligand field "
            "(not a valid LS d5 ground state)",))
    else:
        sol = replace(sol, ground_doublet=ground)
    return sol


def _solve_signed(gx: float, gy: float, gz: float, max_iter: int, extra_starts: bool):
    """All distinct (a, b, c, k) reproducing signed (gx, gy, gz) exactly."""
    a0, c0, k0, iters, fp_ok = _telser_fixed_point(gx, gy, gz, max_iter)
    starts = [(a0, c0, k0)] if a0 > 0.0 else []
    if extra_starts or not starts:
        starts += [(a, c, k) for a in (0.15, 0.5, 0.85) for c in (-0.5, -0.1, 0.1, 0.5) for k in (0.7, 1.0)]
    found = []
    for st in starts:
        a, b, c, k, ok = _polish(gx, gy, gz, *st)
        if ok and not any(abs(a - f[0]) + abs(c - f[2]) + abs(k - f[3]) < 1e-7 for f in found):
            found.append((a, b, c, k))
    return found, iters, fp_ok


def tetragonal_from_g(g1: float, g2: float, g3: float, mode: str = "mcgarvey", max_iter: int = 2000,
           standardize: bool = True) -> list[TetragonalSolution]:
    """Hole wavefunction, k, Delta/xi and V/xi from three observed |g| values.

    mode="mcgarvey" (default; alias "dlsd5"): McGarvey's procedure as in DLSD5:
        g1 -> -gx, g2 -> -gy, gz = -g3 (#1) or +g3 (#2), then relabel axes to
        |V/Delta| <= 2/3.  McGarvey (1998, p. 78) shows these two are the only
        assignments needed for case 1a; EPR magnitudes cannot tell them apart.
    mode="all": every assignment of the three |g| to x, y, z and every sign
        pattern (an exhaustive cross-check).  In testing it finds no plausible
        solution beyond McGarvey's two (tests/test_dlsd5_vs_hamiltonian.py);
        the extra roots are excited doublets or have unphysical k.

    Every solution reproduces the input |g| to < 1e-10.  ``ground_doublet`` says
    whether the wavefunction is the highest (hole) Kramers doublet of its own
    ligand field.
    """
    mags = [abs(g1), abs(g2), abs(g3)]
    if mode in ("mcgarvey", "dlsd5"):
        if sorted(mags, reverse=True) != mags:
            raise ValueError("enter g values in order g1 >= g2 >= g3 (largest first)")
        out = []
        for label, gz in (("#1 (gz = -g3)", -mags[2]), ("#2 (gz = +g3)", mags[2])):
            found, iters, fp_ok = _solve_signed(-mags[0], -mags[1], gz, max_iter, extra_starts=False)
            notes = () if fp_ok else ("DLSD5 fixed-point iteration failed; solved by Newton polish",)
            if not found:
                out.append(TetragonalSolution(0.0, 0.0, 0.0, float("nan"), -mags[0], -mags[1], gz,
                                              None, None, None, label=label, converged=False,
                                              iterations=iters, notes=notes + ("no solution",)))
                continue
            a, b, c, k = found[0]
            out.append(_finish(_solution_from_abc(a, b, c, k, label=label, iterations=iters,
                                                  notes=notes), standardize))
        return out
    if mode != "all":
        raise ValueError("mode must be 'mcgarvey' (= 'dlsd5') or 'all'")

    out: list[TetragonalSolution] = []
    for perm in sorted(set(itertools.permutations(mags))):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            gx, gy, gz = (s * m for s, m in zip(signs, perm))
            found, iters, _ = _solve_signed(gx, gy, gz, 200, extra_starts=True)
            for a, b, c, k in found:
                sol = _finish(_solution_from_abc(a, b, c, k, iterations=iters), standardize)
                if sol.delta is None:
                    continue
                if not any(_same_physics(sol, o) for o in out):
                    out.append(sol)
    out.sort(key=lambda s: (not s.ground_doublet, abs(s.k - 1.0)))
    return [replace(s, label=f"solution {i + 1}") for i, s in enumerate(out)]


def orbital_energies(delta: float, v: float) -> list[float]:
    """Sorted one-electron t2g energies {Delta, V/2, -V/2} relative to their mean (axis-free)."""
    e = (delta, v / 2.0, -v / 2.0)
    m = sum(e) / 3.0
    return sorted(x - m for x in e)


K_PLAUSIBLE = (0.0, 1.2)


def plausible(sol: TetragonalSolution, k_range: tuple[float, float] = K_PLAUSIBLE) -> bool:
    """Physically reasonable: :attr:`TetragonalSolution.is_physical` and k_range[0] < k <= k_range[1].

    The orbital reduction factor k is 1 for a pure metal d orbital and is lowered
    by covalency; values slightly above 1 are tolerated because k also absorbs
    model error.  This is a display filter only: :func:`from_g` returns everything.
    """
    return sol.is_physical and k_range[0] < sol.k <= k_range[1]


def _same_physics(s: TetragonalSolution, o: TetragonalSolution, tol: float = 1e-6) -> bool:
    """Same k and same orbital-energy pattern, ignoring axis labels.

    Delta is measured from the xz/yz barycenter, so a relabeling shifts all three
    energies by a constant; compare them relative to their mean.
    """
    e1 = orbital_energies(s.delta, s.v)
    e2 = orbital_energies(o.delta, o.v)
    return abs(s.k - o.k) < tol and all(abs(x - y) < tol * max(1.0, abs(x)) for x, y in zip(e1, e2))

# ======== trigonal
BETA_OCT = 1.0 / math.sqrt(3.0)


def trigonal_g_from_coefficients(at: float, bt: float, k: float, beta: float) -> tuple[float, float]:
    """Signed (g_perp, g_par)."""
    alpha2 = 1.0 - beta * beta
    g_perp = 2.0 * (-bt * bt - RT6 * k * at * bt * beta)
    g_par = 2.0 * (bt * bt - at * at - (2.0 * alpha2 - beta * beta) * k * at * at)
    return g_perp, g_par


def trigonal_delta_from_coefficients(at: float, bt: float, beta: float) -> float | None:
    """Delta/xi = E(a1) - E(e); None if at*bt = 0 (Delta infinite)."""
    if at == 0.0 or bt == 0.0:
        return None
    return 0.5 * (2.0 - 3.0 * beta * beta) + RT6 * beta * (bt * bt - at * at) / (2.0 * at * bt)


def epi_from_twist(delta: float, phi_deg: float) -> float:
    """E_pi(perp - para)/xi for ligand twist angle phi (Dolder et al. 2003)."""
    return delta / (3.0 * math.sin(2.0 * math.radians(phi_deg)))


@dataclass(frozen=True)
class TrigonalSolution:
    at: float
    bt: float
    k: float
    beta: float
    g_perp: float
    g_par: float
    delta: float | None
    label: str = ""
    converged: bool = True
    iterations: int = 0
    epi: float | None = None
    twist_deg: float | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    ground_doublet: bool | None = None  # checked only for octahedral beta

    @property
    def is_physical(self) -> bool:
        return self.converged and self.delta is not None and self.ground_doublet is not False

    @property
    def alpha(self) -> float:
        return math.sqrt(max(0.0, 1.0 - self.beta * self.beta))


def _check_ground(sol: TrigonalSolution) -> TrigonalSolution:
    """For octahedral beta, compare with the exact t2g Hamiltonian (see hamiltonian.trigonal_cf)."""
    if sol.delta is None or abs(sol.beta - BETA_OCT) > 1e-9:
        return sol
    st = solve_t2g_hole(trigonal_cf(sol.delta), sol.k)
    ok = bool(np.allclose(np.sort(st.g_principal),
                          np.sort([abs(sol.g_perp), abs(sol.g_perp), abs(sol.g_par)]), atol=1e-7))
    notes = sol.notes if ok else sol.notes + (
        "wavefunction is an EXCITED Kramers doublet of the implied trigonal field "
        "(not a valid LS d5 ground state)",)
    return replace(sol, ground_doublet=ok, notes=notes)


def trigonal_from_coefficients(at: float, k: float, beta: float = BETA_OCT
                      ) -> tuple[TrigonalSolution, TrigonalSolution]:
    """g values and Delta from at, k and beta; bt = +/- sqrt(1 - at^2), both returned.

    Equivalent to DLSD5 option G / trigonal (octahedral case: beta = 1/sqrt3).
    """
    notes: tuple[str, ...] = ()
    bt2 = 1.0 - at * at
    if bt2 < 0.0:
        notes, bt2 = ("bt^2 < 0: set to 0",), 0.0
    if beta > 1.0:
        raise ValueError("beta must be <= 1")
    out = []
    for sign, lab in ((1.0, "+bt"), (-1.0, "-bt")):
        bt = sign * math.sqrt(bt2)
        gp, gz = trigonal_g_from_coefficients(at, bt, k, beta)
        out.append(_check_ground(TrigonalSolution(at, bt, k, beta, gp, gz,
                                                  trigonal_delta_from_coefficients(at, bt, beta),
                                                  label=lab, notes=notes)))
    return out[0], out[1]


def _at2(g_par_signed: float, k: float, beta: float) -> float:
    return (2.0 - g_par_signed) / 2.0 / (2.0 + k * (2.0 - 3.0 * beta * beta))


def trigonal_from_g(g_par: float, g_perp: float, *, fix: str = "beta", beta: float = BETA_OCT,
           k: float = 1.0, twist_deg: float | None = None, max_iter: int = 100
           ) -> list[TrigonalSolution]:
    """Hole wavefunction and Delta/xi from observed |g_par|, |g_perp|.

    fix="beta": beta fixed (default 1/sqrt3), solve for k.
    fix="k":    k fixed, solve for beta.
    Two solutions: g_par -> -|g_par| (#1) and +|g_par| (#2); g_perp -> -|g_perp|.
    Equivalent to DLSD5 option C / trigonal.
    """
    if fix not in ("beta", "k"):
        raise ValueError("fix must be 'beta' or 'k'")
    gx = -abs(g_perp)
    out = []
    for label, gz in (("#1 (g_par < 0)", -abs(g_par)), ("#2 (g_par > 0)", abs(g_par))):
        kk, bb = k, beta

        def resid(x):
            kx, bx = (x, bb) if fix == "beta" else (kk, x)
            a2 = _at2(gz, kx, bx)
            if not 0.0 < a2 < 1.0:
                return float("nan")
            at, bt = math.sqrt(a2), math.sqrt(1.0 - a2)
            return trigonal_g_from_coefficients(at, bt, kx, bx)[0] - gx

        # Telser's damped fixed point (DLSD5.F 598-635), used as the starting guess.
        x = k if fix == "beta" else beta
        it, notes = 0, []
        for it in range(1, max_iter + 1):
            kx, bx = (x, bb) if fix == "beta" else (kk, x)
            a2 = _at2(gz, kx, bx)
            if not 0.0 < a2 < 1.0:
                break
            at, bt = math.sqrt(a2), math.sqrt(1.0 - a2)
            if fix == "beta":
                xt = (2.0 * (a2 - 1.0) - gx) / (bx * 2.0 * RT6 * bt * at)
                x_new = 0.5 * (x + xt)
            else:
                xt = (2.0 * (a2 - 1.0) - gx) / 2.0 / RT6 / kx / at / bt
                x_new = xt
            if abs(xt - x) < 1e-5:
                x = xt if fix == "k" else x
                break
            x = x_new
        else:
            notes.append(f"fixed-point iteration hit {max_iter} cycles (DLSD5 would report this iterate)")

        sol_x = _solve_1d(resid, x)
        if sol_x is None:
            out.append(TrigonalSolution(0.0, 0.0, kk if fix == "k" else float("nan"),
                                        bb if fix == "beta" else float("nan"), gx, gz, None,
                                        label=label, converged=False, iterations=it,
                                        notes=tuple(notes + ["no solution for these g values"])))
            continue
        kx, bx = (sol_x, bb) if fix == "beta" else (kk, sol_x)
        if fix == "k" and not 0.0 <= bx <= 1.0:
            out.append(TrigonalSolution(0.0, 0.0, kx, bx, gx, gz, None, label=label, converged=False,
                                        iterations=it, notes=tuple(notes + [f"beta = {bx:.5f} outside [0, 1]"])))
            continue
        a2 = _at2(gz, kx, bx)
        at, bt = math.sqrt(a2), math.sqrt(1.0 - a2)
        gp, gzz = trigonal_g_from_coefficients(at, bt, kx, bx)
        delta = trigonal_delta_from_coefficients(at, bt, bx)
        epi = epi_from_twist(delta, twist_deg) if (twist_deg and delta is not None) else None
        out.append(_check_ground(TrigonalSolution(at, bt, kx, bx, gp, gzz, delta, label=label,
                                                  iterations=it, epi=epi, twist_deg=twist_deg or None,
                                                  notes=tuple(notes))))
    return out


def _solve_1d(f, x0: float) -> float | None:
    """Root of f near x0: secant from x0, then a bracketing scan as fallback."""
    try:
        r = root_scalar(f, x0=x0, x1=x0 * 1.001 + 1e-4, method="secant", xtol=1e-14, maxiter=200)
        if math.isfinite(r.root) and abs(f(r.root)) < 1e-10:
            return r.root
    except (ValueError, ZeroDivisionError, OverflowError):
        pass
    grid = [x0 * (1 + s) for s in (-0.9, -0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6, 1.0, 2.0)]
    grid = sorted(set(grid + [i * 0.002 for i in range(1, 1501)]))  # 0.002 .. 3.0
    best = None
    for lo, hi in zip(grid, grid[1:]):
        flo, fhi = f(lo), f(hi)
        if math.isfinite(flo) and math.isfinite(fhi) and flo * fhi < 0:
            xr = brentq(f, lo, hi, xtol=1e-15)
            if best is None or abs(xr - x0) < abs(best - x0):
                best = xr
    return best
