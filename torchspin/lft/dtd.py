"""DTD: Townes-Dailey analysis of a 14N quadrupole tensor, differentiable (Telser's DTD).

Ported from telser-tools (``telser_tools_diff.dtd``), the default (``mode="corrected"``) path of
``telser_tools.dtd``.  An sp2 nitrogen bound to a metal (pyrrole/imine N of a macrocycle): z along
the M-N bond (the N lone pair), x in the ligand plane, y out of plane; the two N-C sigma bonds lie
in the xz plane at C-N-C angle 2 theta.  With orthonormal sp hybrids,

    lone pair s character = cot^2 theta,     n = (1 - cot^2 theta)/cot^2 theta   (sp^n)
    n_z = c (1 - cot^2 theta) + b cot^2 theta,   n_x = b,   n_y = a
    P_ii / P0 = n_i - (n_j + n_k)/2                                   (Townes-Dailey)

c = lone-pair (M-N sigma) population, b = population of EACH N-C sigma hybrid, a = N 2p_pi (y)
population, N_T = c + a + 2b.  The model is explicit (linear in the populations), so both
directions are closed forms: batched, every input may carry gradients, ``torch.func`` compatible.

    efg_from_populations(p0, two_theta, a, b, c)       -> (P_zz, P_xx, P_yy)       (forward)
    from_angle(p0, two_theta, c, p_zz, p_xx, p_yy)     -> TownesDailey             (DTD option 'D')
    from_alphas(p0, alpha2, alpha3, p_zz, p_xx, p_yy)  -> TownesDailey             (DTD option 'A')

Units
-----
``p0`` (the quadrupole coupling of one p electron along its own axis) and the traceless principal
values ``p_zz, p_xx, p_yy`` are in any one unit (MHz, as Telser) and with the same sign
convention; the populations depend only on their ratios, and :func:`efg_from_populations` returns
the unit of ``p0``.  Angles are in radians; ``degrees=True`` takes and returns degrees (Telser).

Input validation (c <= 2, 90 < 2 theta < 180 degrees, traceless tensor within ``trace_tol``,
alpha1^2 > 0, ...) is done on detached values, for every batch element, and raises ValueError
as telser-tools does.  The asymmetry parameter eta involves a sort by |P| (decided on detached
values); its derivative is that of the selected branch and does not exist where two |P| tie.

Not ported: ``mode="fortran"`` (dtd.f's pi/180 = 1.745329252e-2 and its IEEE division by zero at
2 theta = 90 degrees), the CLI and the report.

Reference: C. H. Townes, B. P. Dailey, J. Chem. Phys. 1949, 17, 782.

    >>> from torchspin.lft import dtd
    >>> r = dtd.from_angle(10.0, 120.0, 1.5, 3.2, -1.8, -1.4, degrees=True)
    >>> r.a, r.b, r.nt, dtd.asymmetry([3.2, -1.8, -1.4])
"""

from __future__ import annotations

from typing import NamedTuple

import torch

__all__ = ["TownesDailey", "angle_from_hybridization", "asymmetry", "efg_from_populations", "from_alphas",
           "from_angle", "hybridization_from_angle"]

_F64 = torch.float64


def _t(x) -> torch.Tensor:
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=_F64)


def _rad(x, degrees: bool) -> torch.Tensor:
    x = _t(x)
    return torch.deg2rad(x) if degrees else x


def _cot2(two_theta_rad) -> torch.Tensor:
    return 1.0 / torch.tan(two_theta_rad / 2.0) ** 2


def hybridization_from_angle(two_theta, degrees: bool = False) -> torch.Tensor:
    """Lone-pair sp^n: n = tan^2 theta - 1 for the C-N-C angle 2 theta."""
    return torch.tan(_rad(two_theta, degrees) / 2.0) ** 2 - 1.0


def angle_from_hybridization(n, degrees: bool = False) -> torch.Tensor:
    """C-N-C angle 2 theta = 2 atan(sqrt(n + 1)) (radians, or degrees with ``degrees=True``)."""
    tt = 2.0 * torch.atan(torch.sqrt(_t(n) + 1.0))
    return torch.rad2deg(tt) if degrees else tt


def efg_from_populations(p0, two_theta, a, b, c, degrees: bool = False) -> torch.Tensor:
    """(P_zz, P_xx, P_yy) (..., 3) in the unit of ``p0`` from the populations and the C-N-C angle."""
    cot2 = _cot2(_rad(two_theta, degrees))
    p0, a, b, c = (_t(x) for x in (p0, a, b, c))
    nz, nx, ny = c * (1 - cot2) + b * cot2, b, a
    return torch.stack(torch.broadcast_tensors(p0 * (nz - (nx + ny) / 2), p0 * (nx - (ny + nz) / 2),
                                               p0 * (ny - (nx + nz) / 2)), -1)


def asymmetry(p) -> torch.Tensor:
    """eta = (V_XX - V_YY)/V_ZZ with |V_ZZ| >= |V_YY| >= |V_XX| for p (..., 3); the order is chosen detached."""
    p = _t(p)
    idx = torch.argsort(p.detach().abs(), dim=-1, stable=True)
    v = torch.gather(p, -1, idx)
    return torch.where(v[..., 2] == 0, torch.zeros_like(v[..., 2]), (v[..., 0] - v[..., 1]) / v[..., 2])


class TownesDailey(NamedTuple):
    """Townes-Dailey solution: ``two_theta`` (radians, or degrees if the call had ``degrees=True``),
    lone-pair sp^``n``, populations ``c`` (lone pair), ``a`` (p_pi), ``b`` (each N-C sigma),
    ``nt`` = c + a + 2b, and ``alpha1`` (metal coefficient; :func:`from_alphas` only)."""

    two_theta: torch.Tensor
    n: torch.Tensor
    c: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    nt: torch.Tensor
    alpha1: torch.Tensor | None = None

    @property
    def populations(self) -> torch.Tensor:
        """(a, b, c) stacked."""
        return torch.stack(torch.broadcast_tensors(self.a, self.b, self.c), -1)


def _solve(p0, cot2, c, p_zz, p_xx, p_yy):
    bma = (2.0 / 3.0) * ((p_xx - p_yy) / p0)
    cotm1 = cot2 - 1.0
    b = (p_zz / p0 + c * cotm1 - bma / 2.0) / cotm1
    a = b - bma
    return a, b, c + a + 2.0 * b


def _any(x: torch.Tensor) -> bool:
    return bool(x.detach().any())


def _check_tensor(p0, p_zz, p_xx, p_yy, trace_tol: float) -> None:
    if _any(p0 == 0):
        raise ValueError("P0 must be nonzero")
    s = (p_zz + p_xx + p_yy).detach()
    if _any(s.abs() > trace_tol):
        raise ValueError(f"P tensor not traceless: sum = {float(s.abs().max()):.3g} (tolerance {trace_tol:g})")


def from_angle(p0, two_theta, c, p_zz, p_xx, p_yy, *, degrees: bool = False, trace_tol: float = 1e-6
               ) -> TownesDailey:
    """DTD option 'D': populations from the C-N-C angle 2 theta and the lone-pair population c.

    Raises ValueError for c > 2 or c < 0, for 2 theta outside (90, 180) degrees (at 90 the
    equations are singular; below, the lone pair would need more than 100 % s character), and
    for a tensor that is not traceless within ``trace_tol``.
    """
    p0, c, pz, px, py = (_t(x) for x in (p0, c, p_zz, p_xx, p_yy))
    tt_in = _t(two_theta)
    tt = _rad(tt_in, degrees)
    if _any(c > 2.0):
        raise ValueError("c <= 2")
    ttd = torch.rad2deg(tt.detach())
    if _any((ttd <= 90.0) | (ttd >= 180.0)):
        raise ValueError("2 theta must lie in (90, 180) degrees; at 90 the equations are singular")
    if _any(c < 0):
        raise ValueError("c must be >= 0")
    _check_tensor(p0, pz, px, py, trace_tol)
    cot2 = _cot2(tt)
    n = (1.0 - cot2) / cot2
    a, b, nt = _solve(p0, cot2, c, pz, px, py)
    return TownesDailey(tt_in, n, c, a, b, nt)


def from_alphas(p0, alpha2, alpha3, p_zz, p_xx, p_yy, *, degrees: bool = False, trace_tol: float = 1e-6
                ) -> TownesDailey:
    """DTD option 'A': from the N 2s (alpha2) and 2p_z (alpha3) coefficients in the singly occupied
    M-N sigma* orbital alpha1 d - sum_4N (alpha2 s + alpha3 p_z):

        alpha1^2 = 1 - 4 (alpha2^2 + alpha3^2),  n = alpha3^2/alpha2^2,  tan^2 theta = n + 1,
        c = 2 - (alpha2^2 + alpha3^2).

    ``two_theta`` is returned in radians (degrees with ``degrees=True``).  Raises ValueError for
    alpha > 1, alpha2 <= 0, alpha3 < 0, alpha1^2 <= 0 and a non-traceless tensor.
    """
    p0, al2, al3, pz, px, py = (_t(x) for x in (p0, alpha2, alpha3, p_zz, p_xx, p_yy))
    if _any((al2 > 1.0) | (al3 > 1.0)):
        raise ValueError("alpha <= 1")
    if _any((al2 <= 0) | (al3 < 0)):
        raise ValueError("need alpha2 > 0 and alpha3 >= 0")
    a22, a32 = al2 * al2, al3 * al3
    alpha12 = 1.0 - 4.0 * (a22 + a32)
    if _any(alpha12 <= 0.0):
        raise ValueError(f"alpha1^2 = {float(alpha12.detach().min()):.5g} <= 0; reduce alpha2, alpha3")
    _check_tensor(p0, pz, px, py, trace_tol)
    n = a32 / a22
    tan2 = n + 1.0
    cot2 = 1.0 / tan2
    two_theta = 2.0 * torch.atan(torch.sqrt(tan2))
    c = 2.0 - (a22 + a32)
    a, b, nt = _solve(p0, cot2, c, pz, px, py)
    return TownesDailey(torch.rad2deg(two_theta) if degrees else two_theta, n, c, a, b, nt, torch.sqrt(alpha12))

