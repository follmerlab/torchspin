"""DLSD5: low-spin d5 (t2g^5) g values <-> ligand field, differentiable (Telser's DLSD5).

Ported from telser-tools (``telser_tools_diff.dlsd5``), with the same bases, axes and sign
conventions as ``telser_tools.dlsd5`` (McGarvey, Coord. Chem. Rev. 1998, 170, 75).  Every
continuous input may carry gradients, with leading batch dimensions where noted.

Units
-----
DLSD5 works in units of the spin-orbit constant xi: Delta/xi, V/xi and E0/xi are
dimensionless ratios, as are g, k, the orbital coefficients and beta.  The exact Hamiltonian
(:func:`t2g_hamiltonian`, :func:`solve_t2g_hole`) takes the crystal-field matrix and ``xi`` in
any one energy unit and returns energies in that unit (``xi=1``, the default, is DLSD5's
convention).  Multiply a ratio by xi in the unit you want, e.g.
``delta * lft.units.convert_energy(xi_cm, 'cm^-1', 'MHz')``.  Angles (the trigonal twist) are
in radians; ``degrees=True`` takes Telser-style input.

Forward, exact Hamiltonian
--------------------------
``t2g_hamiltonian(cf, xi)`` (..., 6, 6): H = CF (x) 1 + xi L.S in (xy, xz, yz) x (alpha, beta).
``solve_t2g_hole(cf, k, xi)`` -> :class:`HoleState`: |g| principal values, g g^T, det g, the
three Kramers-doublet energies and the hole orbital populations.  The hole is the HIGHEST
doublet, a projector computed with :func:`torchspin.lft._linalg.ground_manifold` on -H, so the
derivative is exact at the Kramers degeneracy.  g g^T = 2 tr(P mu_i P mu_j) and
det g = -4i tr(P mu_x P mu_y P mu_z) are gauge invariant.
``tetragonal_cf(delta, v)``, ``trigonal_cf(delta)``.

Forward, McGarvey closed forms
------------------------------
``g_from_abc``, ``ligand_field_from_abc``, ``abc_to_ABC``, ``ABC_to_abc``,
``trigonal_g_from_coefficients``, ``trigonal_delta_from_coefficients``, ``epi_from_twist``,
``from_complex_coefficients``, ``from_real_coefficients`` (both signs of the third coefficient;
DLSD5's normalization and the axis relabeling are decided on detached values),
``trigonal_from_coefficients``.

Inverse (g -> ligand field), every branch
-----------------------------------------
:func:`tetragonal_from_g` and :func:`trigonal_from_g` run the NumPy solver (copied into
:mod:`torchspin.lft._engines.dlsd5`) on DETACHED values; its discrete outcomes (which branches,
which sign/permutation of g, which relabeling) are taken as given.  The derivative is attached to
each converged branch by the implicit-function theorem (:func:`implicit_attach`): with unknowns
x and data theta solving F(x, theta) = 0,

    x = x* - J^-1 (F(x*, theta) - F(x*, theta).detach()),     J = dF/dx at x* (detached),

which has the value x* exactly and the derivative dx/dtheta = -J^-1 dF/dtheta, in reverse and
forward mode (``torch.autograd.forward_ad``; not under ``torch.func`` transforms, since the
solver runs in NumPy).

* tetragonal: x = (a, b, c, k), F = (g_from_abc(a, b, c, k) - s o P g, a^2 + b^2 + c^2 - 1), where
  s o P g is the signed, permuted assignment of the input |g| that the branch reproduces.
* trigonal: x = k (fix="beta") or beta (fix="k"), F = g_perp(at(x), bt(x), ...) + |g_perp|, with
  at^2 from g_par.  theta = (g_par, g_perp, the fixed beta or k).

Derived quantities (Delta/xi, V/xi, E0/xi, weights, epi) follow by plain autograd.  At a branch
where J is singular (a fold of the g map) no derivative exists: the result is inf/nan.

Not ported: the DLSD5 CLI and reports.  ``HoleState.g_axes`` and the signed g matrix of the NumPy
version are gauge dependent; use ``ggT`` and ``g_det``.

    >>> from torchspin.lft import dlsd5
    >>> for b in dlsd5.tetragonal_from_g(2.62, 2.38, 1.60):      # |g| largest first
    ...     print(b.numpy.label, float(b.delta), float(b.v), float(b.k))
    >>> st = dlsd5.solve_t2g_hole(dlsd5.tetragonal_cf(3.0, 1.0), k=0.9)
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import torch

from ._engines import dlsd5 as _np
from ._linalg import eigvalsh_gauge, ground_manifold

__all__ = [
    "BETA_OCT",
    "ABC_to_abc",
    "CoefficientSolution",
    "HoleState",
    "TetragonalBranch",
    "TrigonalBranch",
    "abc_to_ABC",
    "epi_from_twist",
    "from_complex_coefficients",
    "from_real_coefficients",
    "g_from_abc",
    "implicit_attach",
    "ligand_field_from_abc",
    "solve_t2g_hole",
    "t2g_hamiltonian",
    "tetragonal_cf",
    "tetragonal_from_g",
    "trigonal_cf",
    "trigonal_delta_from_coefficients",
    "trigonal_from_coefficients",
    "trigonal_from_g",
    "trigonal_g_from_coefficients",
]

_F64, _C128 = torch.float64, torch.complex128
RT2, RT6 = math.sqrt(2.0), math.sqrt(6.0)
#: Octahedral beta = 1/sqrt3 (no t2g/eg mixing), the trigonal default.
BETA_OCT = _np.BETA_OCT


def _t(x, device=None) -> torch.Tensor:
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=_F64, device=device)


def _angle(x, degrees: bool) -> torch.Tensor:
    x = _t(x)
    return torch.deg2rad(x) if degrees else x


# ------------------------------------------------------------------------------ exact Hamiltonian
def _ops(device=None):
    L = torch.as_tensor(np.array(_np._L), dtype=_C128, device=device)                    # (3, 3, 3)
    S = torch.as_tensor(np.array(_np._SPIN), dtype=_C128, device=device)                 # (3, 2, 2)
    i2 = torch.eye(2, dtype=_C128, device=device)
    i3 = torch.eye(3, dtype=_C128, device=device)
    LS = sum(torch.kron(L[i], S[i]) for i in range(3))
    Lf = torch.stack([torch.kron(L[i], i2) for i in range(3)])
    Sf = torch.stack([torch.kron(i3, S[i]) for i in range(3)])
    E = torch.zeros(3, 3, 6, 6, dtype=_C128, device=device)                             # kron(e_ab, 1_2)
    for a in range(3):
        for b in range(3):
            e = torch.zeros(3, 3, dtype=_C128, device=device)
            e[a, b] = 1.0
            E[a, b] = torch.kron(e, i2)
    return LS, Lf, Sf, E


def tetragonal_cf(delta, v) -> torch.Tensor:
    """diag(Delta, V/2, -V/2) in (xy, xz, yz), shape (..., 3, 3); in the unit of ``delta`` and ``v``."""
    delta, v = torch.broadcast_tensors(_t(delta), _t(v))
    return torch.diag_embed(torch.stack([delta, v / 2.0, -v / 2.0], -1))


def trigonal_cf(delta) -> torch.Tensor:
    """Delta (|a1><a1| - 1/3), a1 = (xy + xz + yz)/sqrt3 (octahedral-beta trigonal field); E(a1) - E(e) = Delta."""
    delta = _t(delta)
    a1 = torch.full((3,), 1.0 / math.sqrt(3.0), dtype=_F64, device=delta.device)
    return delta[..., None, None] * (torch.outer(a1, a1) - torch.eye(3, dtype=_F64, device=delta.device) / 3.0)


def t2g_hamiltonian(cf, xi=1.0) -> torch.Tensor:
    """H = CF (x) 1_2 + xi L.S in (xy, xz, yz) x (alpha, beta), shape (..., 6, 6), in the unit of ``cf`` and ``xi``."""
    cf = _t(cf)
    LS, _, _, E = _ops(cf.device)
    return torch.einsum("...ab,abij->...ij", cf.to(_C128), E) + _t(xi, cf.device)[..., None, None].to(_C128) * LS


class HoleState(NamedTuple):
    """The t2g^5 hole (highest) Kramers doublet of CF + xi L.S.

    ``g_principal`` (..., 3) ascending |g|; ``ggT`` (..., 3, 3); ``g_det`` = det g (its sign is
    ``g_sign_product``, the only g sign EPR can reach); ``doublet_energies`` (..., 3) ascending, in
    the unit of the crystal field; ``hole_weights`` (..., 3) in (xy, xz, yz) order; ``projector``
    (..., 6, 6) onto the hole doublet; ``gap`` from the hole doublet down to the next one (the
    derivative scales as 1/gap).
    """

    g_principal: torch.Tensor
    ggT: torch.Tensor
    g_det: torch.Tensor
    doublet_energies: torch.Tensor
    hole_weights: torch.Tensor
    projector: torch.Tensor
    gap: torch.Tensor

    @property
    def g_sign_product(self) -> torch.Tensor:
        return torch.sign(self.g_det.detach())

    @property
    def hole_energy(self) -> torch.Tensor:
        return self.doublet_energies[..., -1]


def solve_t2g_hole(cf, k=1.0, xi=1.0) -> HoleState:
    """Exact hole (highest) Kramers doublet of CF + xi L.S and its g (mu = k L + 2 S).

    ``cf`` (..., 3, 3) real symmetric in (xy, xz, yz); ``k`` the orbital reduction (Zeeman term
    only); ``xi`` the spin-orbit constant in the unit of ``cf`` (1: DLSD5's units of xi).
    """
    cf = _t(cf)
    H = t2g_hamiltonian(cf, xi)
    _, Lf, Sf, _ = _ops(cf.device)
    k = _t(k, cf.device)
    mu = k[..., None, None, None].to(_C128) * Lf + 2.0 * Sf                              # (..., 3, 6, 6)
    man = ground_manifold(-H, 2)                                                         # highest doublet of H
    P = man.projector
    pm = P.unsqueeze(-3) @ mu
    ggT = 2.0 * torch.einsum("...inm,...jmn->...ij", pm, pm).real
    ggT = 0.5 * (ggT + ggT.transpose(-1, -2))
    det = (-4j * torch.einsum("...ab,...bc,...ca->...", pm[..., 0, :, :], pm[..., 1, :, :], pm[..., 2, :, :])).real
    gp = torch.sqrt(torch.clamp(eigvalsh_gauge(ggT), min=0.0))
    w = eigvalsh_gauge(H)
    weights = torch.diagonal(P, dim1=-2, dim2=-1).real.reshape(*P.shape[:-2], 3, 2).sum(-1) / 2.0
    return HoleState(gp, ggT, det, w[..., ::2], weights, P, man.gap)


# ------------------------------------------------------------------------------ McGarvey closed forms
def abc_to_ABC(a, b, c):
    """(d+1, xy, d-1) -> (xy, yz, xz)  (McGarvey eq. 10)."""
    return b, (a + c) / RT2, (a - c) / RT2


def ABC_to_abc(A, B, C):
    """(xy, yz, xz) -> (d+1, xy, d-1)."""
    return (B + C) / RT2, A, (B - C) / RT2


def g_from_abc(a, b, c, k) -> torch.Tensor:
    """Signed (gx, gy, gz) (..., 3), McGarvey eq. 4."""
    a, b, c, k = (_t(x) for x in (a, b, c, k))
    gx = 2.0 * (-b * b + 2.0 * a * c - RT2 * k * b * (a - c))
    gy = 2.0 * (-b * b - 2.0 * a * c - RT2 * k * b * (a + c))
    gz = 2.0 * (b * b - a * a - c * c + k * (c * c - a * a))
    return torch.stack(torch.broadcast_tensors(gx, gy, gz), -1)


def ligand_field_from_abc(a, b, c) -> torch.Tensor:
    """(Delta/xi, V/xi, E0/xi) (..., 3) implied by the hole wavefunction (inf/nan where b = 0 or a^2 = c^2)."""
    a, b, c = (_t(x) for x in (a, b, c))
    denom = a * a - c * c
    delta = (b * (1.0 - b * b) + RT2 * a * (1.0 - 2.0 * a * a)) / (2.0 * b * denom)
    v = c * (2.0 * a + RT2 * b) / (-denom)
    e0 = delta + a / (RT2 * b)
    return torch.stack(torch.broadcast_tensors(delta, v, e0), -1)


def trigonal_g_from_coefficients(at, bt, k, beta) -> torch.Tensor:
    """Signed (g_perp, g_par) (..., 2)."""
    at, bt, k, beta = (_t(x) for x in (at, bt, k, beta))
    alpha2 = 1.0 - beta * beta
    g_perp = 2.0 * (-bt * bt - RT6 * k * at * bt * beta)
    g_par = 2.0 * (bt * bt - at * at - (2.0 * alpha2 - beta * beta) * k * at * at)
    return torch.stack(torch.broadcast_tensors(g_perp, g_par), -1)


def trigonal_delta_from_coefficients(at, bt, beta) -> torch.Tensor:
    """Delta/xi = E(a1) - E(e)."""
    at, bt, beta = (_t(x) for x in (at, bt, beta))
    return 0.5 * (2.0 - 3.0 * beta * beta) + RT6 * beta * (bt * bt - at * at) / (2.0 * at * bt)


def epi_from_twist(delta, phi, degrees: bool = False) -> torch.Tensor:
    """E_pi/xi = Delta / (3 sin 2 phi) for the ligand twist angle phi (radians; Dolder et al. 2003)."""
    return _t(delta) / (3.0 * torch.sin(2.0 * _angle(phi, degrees)))


# ------------------------------------------------------------------------------ implicit derivative
def implicit_attach(F: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], x_star: torch.Tensor,
                    theta: torch.Tensor) -> torch.Tensor:
    """Attach dx/dtheta = -(dF/dx)^-1 dF/dtheta to a root x* of F(x, theta) = 0 (the value stays x*).

    ``F(x, theta)`` -> (n,) residuals for x (n,); x* is detached.  Works in reverse and forward
    mode (only plain differentiable ops are applied to theta).
    """
    x_star = x_star.detach()
    with torch.enable_grad():
        J = torch.func.jacrev(lambda x: F(x, theta.detach()))(x_star)                    # (n, n)
    r = F(x_star, theta)
    return x_star - torch.linalg.solve(J, (r - r.detach()).unsqueeze(-1)).squeeze(-1)


# ------------------------------------------------------------------------------ tetragonal inverse
class TetragonalBranch(NamedTuple):
    """One solution of :func:`tetragonal_from_g` with tensors attached to the input g.

    ``numpy`` is the NumPy ``TetragonalSolution`` (``label``, ``ground_doublet``, ``notes``, ...);
    ``perm``, ``sign`` map the solution's (gx, gy, gz) to the input: g_sol[i] = sign[i] * |g_in[perm[i]]|.
    ``delta``, ``v``, ``e0`` are in units of xi; ``weights`` (3,) = (xy, yz, xz) populations (A^2, B^2, C^2).
    """

    a: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor
    k: torch.Tensor
    g: torch.Tensor
    delta: torch.Tensor
    v: torch.Tensor
    e0: torch.Tensor
    weights: torch.Tensor
    perm: tuple[int, int, int]
    sign: tuple[float, float, float]
    numpy: object

    @property
    def ABC(self):
        return abc_to_ABC(self.a, self.b, self.c)


def _assignment(sol, mags):
    """(perm, sign) with g_sol[i] = sign[i] * mags[perm[i]], decided on detached values."""
    g = [sol.gx, sol.gy, sol.gz]
    perm, used = [], set()
    for gi in g:
        cands = [j for j in range(3) if j not in used]
        j = min(cands, key=lambda j: abs(abs(gi) - mags[j]))
        used.add(j)
        perm.append(j)
    return tuple(perm), tuple(math.copysign(1.0, gi) for gi in g)


_SWAP = {None: (0, 1, 2), "x<->z": (2, 1, 0), "y<->z": (0, 2, 1)}


def _mcgarvey_assignment(sol):
    """McGarvey mode: (gx, gy, gz) = (-g1, -g2, -/+ g3) before relabeling; a relabeling swaps axes."""
    base_perm = (0, 1, 2)
    base_sign = (-1.0, -1.0, -1.0 if sol.label.startswith("#1") else 1.0)
    sw = _SWAP[sol.relabeled]
    return tuple(base_perm[i] for i in sw), tuple(base_sign[i] for i in sw)


def tetragonal_from_g(g1, g2, g3, mode: str = "mcgarvey", standardize: bool = True, max_iter: int = 2000
                      ) -> list[TetragonalBranch | object]:
    """Every branch of DLSD5's tetragonal/rhombic inverse, differentiable in (g1, g2, g3).

    ``mode="mcgarvey"`` (alias ``"dlsd5"``): McGarvey's two assignments (g1 >= g2 >= g3 required);
    ``mode="all"``: every assignment and sign pattern.  ``standardize`` relabels axes so that
    |V/Delta| <= 2/3.  Converged branches with a defined ligand field come back as
    :class:`TetragonalBranch`; the others (no solution, b = 0, a^2 = c^2) as the bare NumPy
    solution.  Scalar g only (the branch structure is per point).
    """
    gin = torch.stack(torch.broadcast_tensors(_t(g1), _t(g2), _t(g3)), -1)
    if gin.dim() != 1:
        raise ValueError("tetragonal_from_g takes scalar g values (the branch structure is per point)")
    gd = gin.detach().cpu().numpy()
    sols = _np.tetragonal_from_g(float(gd[0]), float(gd[1]), float(gd[2]), mode=mode, max_iter=max_iter,
                                 standardize=standardize)
    mags = np.abs(gd)
    out = []
    for s in sols:
        if not s.converged or s.delta is None:
            out.append(s)
            continue
        perm, sign = _mcgarvey_assignment(s) if mode in ("mcgarvey", "dlsd5") else _assignment(s, mags)
        sgn = torch.tensor(sign, dtype=_F64, device=gin.device)
        target = sgn * gin.abs()[list(perm)]

        def F(x, tgt):
            a, b, c, k = x.unbind(-1)
            return torch.cat([g_from_abc(a, b, c, k) - tgt, (a * a + b * b + c * c - 1.0).unsqueeze(-1)], -1)

        x0 = torch.tensor([s.a, s.b, s.c, s.k], dtype=_F64, device=gin.device)
        x = implicit_attach(F, x0, target)
        a, b, c, k = x.unbind(-1)
        lf = ligand_field_from_abc(a, b, c)
        A, B, C = abc_to_ABC(a, b, c)
        out.append(TetragonalBranch(a, b, c, k, g_from_abc(a, b, c, k), lf[0], lf[1], lf[2],
                                    torch.stack([A * A, B * B, C * C]), perm, sign, s))
    return out


# ------------------------------------------------------------------------------ trigonal inverse
class TrigonalBranch(NamedTuple):
    """One solution of :func:`trigonal_from_g`; ``delta``, ``epi`` in units of xi (``epi`` None without a twist)."""

    at: torch.Tensor
    bt: torch.Tensor
    k: torch.Tensor
    beta: torch.Tensor
    g_perp: torch.Tensor
    g_par: torch.Tensor
    delta: torch.Tensor
    epi: torch.Tensor | None
    numpy: object


def _at2(g_par_signed, k, beta):
    return (2.0 - g_par_signed) / 2.0 / (2.0 + k * (2.0 - 3.0 * beta * beta))


def trigonal_from_g(g_par, g_perp, *, fix: str = "beta", beta=BETA_OCT, k=1.0, twist=None, degrees: bool = False,
                    max_iter: int = 100) -> list[TrigonalBranch | object]:
    """Both branches (g_par < 0, g_par > 0) of DLSD5's trigonal inverse, differentiable in g_par, g_perp,
    the fixed parameter (``beta`` for fix="beta", ``k`` for fix="k") and the twist angle.

    ``twist`` is the ligand twist angle phi (radians; ``degrees=True`` for degrees); with it each
    branch carries E_pi/xi (Dolder et al. 2003).  Branches that do not converge come back as the
    bare NumPy solution.
    """
    gpar, gperp, beta_t, k_t = (_t(x) for x in (g_par, g_perp, beta, k))
    tw = None if twist is None else _angle(twist, degrees)
    sols = _np.trigonal_from_g(float(gpar.detach()), float(gperp.detach()), fix=fix, beta=float(beta_t.detach()),
                               k=float(k_t.detach()), twist_deg=None if tw is None else math.degrees(float(tw.detach())),
                               max_iter=max_iter)
    out = []
    for s in sols:
        if not s.converged or s.delta is None:
            out.append(s)
            continue
        gz_sign = -1.0 if s.label.startswith("#1") else 1.0
        theta = torch.stack(torch.broadcast_tensors(gpar, gperp, beta_t if fix == "beta" else k_t))

        def parts(x, th):
            gz = gz_sign * th[0].abs()
            kx, bx = (x[0], th[2]) if fix == "beta" else (th[2], x[0])
            a2 = _at2(gz, kx, bx)
            at, bt = torch.sqrt(a2), torch.sqrt(1.0 - a2)
            return at, bt, kx, bx

        def F(x, th):
            at, bt, kx, bx = parts(x, th)
            return (trigonal_g_from_coefficients(at, bt, kx, bx)[0] + th[1].abs()).unsqueeze(-1)

        x0 = torch.tensor([s.k if fix == "beta" else s.beta], dtype=_F64)
        x = implicit_attach(F, x0, theta)
        at, bt, kx, bx = parts(x, theta)
        gg = trigonal_g_from_coefficients(at, bt, kx, bx)
        delta = trigonal_delta_from_coefficients(at, bt, bx)
        epi = epi_from_twist(delta, tw) if (tw is not None and float(tw.detach()) != 0.0) else None
        out.append(TrigonalBranch(at, bt, kx, bx, gg[0], gg[1], delta, epi, s))
    return out


# ------------------------------------------------------------------------------ forward from coefficients
class CoefficientSolution(NamedTuple):
    """Forward result for one sign of the third coefficient (see :func:`from_complex_coefficients`).

    ``g`` signed (gx, gy, gz); ``delta``, ``v``, ``e0`` in units of xi; ``numpy`` the NumPy solution.
    """

    a: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor
    k: torch.Tensor
    g: torch.Tensor
    delta: torch.Tensor
    v: torch.Tensor
    e0: torch.Tensor
    numpy: object


def _normalize_pair(p, q):
    """DLSD5's normalization (p, q, |r|), branches decided on detached values."""
    pd, qd = p.detach(), q.detach()
    full = pd * pd >= 1.0
    over = (pd * pd + qd * qd >= 1.0) & ~full
    one = torch.ones_like(p)
    p2 = torch.where(full, torch.sign(pd) * one, p)
    q_res = torch.sign(qd) * torch.sqrt(torch.clamp(1.0 - p * p, min=0.0))
    q2 = torch.where(full, torch.zeros_like(q), torch.where(over, q_res, q))
    rr = 1.0 - p2 * p2 - q2 * q2
    ok = (rr.detach() > 0) & ~full & ~over
    r = torch.where(ok, torch.sqrt(torch.where(ok, rr, one)), torch.zeros_like(rr))
    return p2, q2, r


def _relabel_abc(a, b, c, name):
    A, B, C = abc_to_ABC(a, b, c)
    if name == "x<->z":
        A, B = B, A
    elif name == "y<->z":
        A, C = C, A
    return ABC_to_abc(A, B, C)


def _coefficient_solutions(abc_pairs, k, np_sols):
    out = []
    for (a, b, c), s in zip(abc_pairs, np_sols):
        a, b, c = _relabel_abc(a, b, c, s.relabeled)
        lf = ligand_field_from_abc(a, b, c)
        out.append(CoefficientSolution(a, b, c, k, g_from_abc(a, b, c, k), lf[..., 0], lf[..., 1], lf[..., 2], s))
    return tuple(out)


def from_complex_coefficients(a, b, k, standardize: bool = True) -> tuple[CoefficientSolution, CoefficientSolution]:
    """(+c, -c) solutions from the d+1 (a) and dxy (b) coefficients (scalar inputs; DLSD5 option G/tetragonal/a)."""
    a, b, k = (_t(x) for x in (a, b, k))
    np_sols = _np.from_complex_coefficients(float(a.detach()), float(b.detach()), float(k.detach()), standardize)
    a2, b2, cabs = _normalize_pair(a, b)
    return _coefficient_solutions([(a2, b2, cabs), (a2, b2, -cabs)], k, np_sols)


def from_real_coefficients(A, B, k, standardize: bool = True) -> tuple[CoefficientSolution, CoefficientSolution]:
    """(+C, -C) solutions from the dxy (A) and dyz (B) coefficients (scalar inputs; DLSD5 option G/tetragonal/A)."""
    A, B, k = (_t(x) for x in (A, B, k))
    np_sols = _np.from_real_coefficients(float(A.detach()), float(B.detach()), float(k.detach()), standardize)
    A2, B2, Cabs = _normalize_pair(A, B)
    return _coefficient_solutions([ABC_to_abc(A2, B2, Cabs), ABC_to_abc(A2, B2, -Cabs)], k, np_sols)


def trigonal_from_coefficients(at, k, beta=BETA_OCT) -> torch.Tensor:
    """(g_perp, g_par, Delta/xi) (..., 2, 3) for bt = +sqrt(1 - at^2) and -sqrt(1 - at^2)."""
    at, k, beta = (_t(x) for x in (at, k, beta))
    bt2 = 1.0 - at * at
    ok = bt2.detach() > 0
    bt = torch.where(ok, torch.sqrt(torch.where(ok, bt2, torch.ones_like(bt2))), torch.zeros_like(bt2))
    rows = []
    for sgn in (1.0, -1.0):
        g = trigonal_g_from_coefficients(at, sgn * bt, k, beta)
        rows.append(torch.cat([g, trigonal_delta_from_coefficients(at, sgn * bt, beta).unsqueeze(-1)], -1))
    return torch.stack(rows, -2)
