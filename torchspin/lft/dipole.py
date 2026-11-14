"""DIPOLE: dipolar couplings of a metal 3d Slater orbital to a nucleus or a second electron (Telser's DIPOLE).

Ported from telser-tools (``telser_tools_diff.dipole``): the default (``mode="corrected"``,
exact 3d Slater-orbital tensor) path of ``telser_tools.dipole``.  Every continuous input may
carry gradients (distance, Z*, direction, the d orbital itself, g, alpha1^2, g_N, the partner g),
with leading batch dimensions.

Units and axes
--------------
Couplings (hyperfine A, electron-electron D, point-dipole values) are energies and come back in
``units`` (default ``'MHz'``; ``'cm^-1'``, ``'GHz'``, ``'K'``, ``'eV'`` on request).  The
dipolar tensor ``tensor_cm3`` is ``<(3 u u^T - u^2)/u^5>`` in cm^-3; the reduced tensors
(:func:`sto_tensor`, :func:`sto_tensor_full`) are dimensionless (R^3 T).  Distances are in
Angstrom.  g, alpha1^2 (metal-orbital population) and g_N are dimensionless.  The per-axis
outputs are on dipole.f's axes, ordered (y, x, z) = Fortran (3, 2, 1), z along the metal-point
vector; ``g`` is given in the same order.  Physical constants are CGS-Gaussian (as dipole.f),
derived from ``torchspin.constants`` (:data:`TORCHSPIN_CONSTANTS`); pass ``constants=``
(any object with the fields of :class:`Constants`, e.g. telser-tools' ``CODATA2018``) to
override.

The tensor, for any point direction and any real d orbital
-----------------------------------------------------------
The closed form ``R^3 T = P(A) exp(-4A/3) + Q(A)`` cancels its A^-4 ... A^-1 poles at small A
(telser-tools evaluates it in 60-digit Decimal arithmetic).  This module uses the multipole
expansion instead, which works for ANY point direction and ANY real d orbital
``Y(r) = r^T M r`` (M real symmetric, made traceless; d_xy is ``M = (e_x e_y^T + e_y e_x^T)/2``)::

    rho = N^2 Y(r)^2 exp(-2 zeta r),     Y^2 = h4 + r^2 h2 + r^4 h0   (harmonic h_L of degree L)
    V(P) = sum_L G_L(R) h_L(P),   T_ij(P) = d_i d_j V(P) + (4 pi/3) rho(P) delta_ij

(zeta = 1 units, ``R = zeta abs(P) = 2A/3``, ``A = Z* R/(2 a0)``).  The radial functions are
written through the lower incomplete gamma function, evaluated from its all-positive series
below ``2R = SERIES_SWITCH`` and from its complement above, so no step cancels.  Agreement with
the Decimal path from A = 1e-3 to 1e3: rel <= 3e-14 of ``max(abs(t))``.

Contents
--------
* :func:`sto_tensor_full` -> (..., 3, 3) R^3 T in the orbital frame
* :func:`sto_tensor` -> (..., 3) (t_y, t_x, t_z) for a dipole.f geometry
* :func:`point_dipole_full` -> (..., 3, 3) ``3 n n^T - 1``
* :func:`slater_A`, :func:`zstar`, :func:`orbital_matrix`, :func:`d_and_e`
* :func:`hyperfine` -> :class:`Hyperfine`; :func:`electron_electron` -> :class:`ElectronElectron`

Nuclear g values come from :func:`torchspin.nucgval` (``nucleus='14N'``); ``g_n=`` overrides it
(and may carry gradients).  telser-tools' table (Weil & Bolton 2007) differs from torchspin's
for some metal nuclides; pass ``g_n=`` to use a specific value.

Not ported: ``mode="fortran"`` / ``model="fortran"`` (dipole.f's polynomials, which are not the
dipolar tensor of a 3d Slater orbital; they exist to reproduce the original), the alpha1^2
clamping of that mode, nucleus lookup by bare mass number (use the isotope name), the CLI and
the report.

Examples
--------
14N at 2.0 Angstrom from Cu(II) (d9) in the orbital plane, axial g, 75 % metal character
(telser-tools' DIPOLE quick start); couplings on the (y, x, z) axes in MHz:

>>> from torchspin.lft import dipole
>>> h = dipole.hyperfine(9, 29, "14N", 2.0, g=(2.05, 2.05, 2.25), alpha2=0.75)
>>> round(float(h.zstar), 4), round(float(h.A), 5)
(13.201, 24.94627)
>>> [round(x, 4) for x in h.a_alpha.tolist()]
[-0.581, -0.5508, 1.2422]
>>> [round(x, 4) for x in h.point_dipole.tolist()]
[-0.5486, -0.5486, 1.2041]

Electron-electron coupling of a Ni(II) d electron to a spin 3.5 Angstrom away, in cm^-1:

>>> e = dipole.electron_electron(8, 28, 3.5, g=(2.2, 2.2, 2.0), units="cm^-1")
>>> [round(x, 4) for x in e.ddip_edip.tolist()]      # (Ddip, Edip)
[-0.123, -0.0004]
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch

from torchspin.constants import BMAGN, BOHRRAD, CLIGHT, GFREE, NMAGN, PLANCK

from . import units as _u
from ._engines.dipole import clementi_raimondi

__all__ = [
    "D_XY",
    "GEOMETRIES",
    "SERIES_SWITCH",
    "TORCHSPIN_CONSTANTS",
    "Constants",
    "ElectronElectron",
    "Hyperfine",
    "d_and_e",
    "electron_electron",
    "hyperfine",
    "orbital_matrix",
    "point_dipole_full",
    "slater_A",
    "sto_tensor",
    "sto_tensor_full",
    "zstar",
]

_F64 = torch.float64
GEOMETRIES = ("in_plane", "out_of_plane", "in_plane_lobe")


class Constants(NamedTuple):
    """CGS-Gaussian constants of dipole.f (field names as telser-tools' ``dipole.constants.Constants``).

    Attributes
    ----------
    g_e : float
        Free-electron g value.
    beta_n, beta_e : float
        Nuclear and Bohr magnetons in erg/G.
    h : float
        Planck constant in erg s.
    a0_cm : float
        Bohr radius in cm.
    mhz_to_cm : float
        1 MHz in cm^-1.
    ee_point_cm : float
        ``beta_e^2 / (h c)`` in cm^-1 cm^3 (point-dipole electron-electron prefactor).
    """

    g_e: float              # free-electron g value
    beta_n: float           # nuclear magneton, erg/G
    beta_e: float           # Bohr magneton, erg/G
    h: float                # Planck constant, erg s
    a0_cm: float            # Bohr radius, cm
    mhz_to_cm: float        # 1 MHz in cm^-1
    ee_point_cm: float      # beta_e^2 / (h c), cm^-1 cm^3 (point-dipole e-e prefactor)


#: From ``torchspin.constants`` (J/T -> erg/G is 1e3, J s -> erg s is 1e7, m -> cm is 1e2).
TORCHSPIN_CONSTANTS = Constants(
    g_e=GFREE, beta_n=NMAGN * 1e3, beta_e=BMAGN * 1e3, h=PLANCK * 1e7, a0_cm=BOHRRAD * 1e2,
    mhz_to_cm=1e6 / CLIGHT, ee_point_cm=(BMAGN * 1e3) ** 2 / (PLANCK * 1e7 * CLIGHT))


# ------------------------------------------------------------------------------ exact radial functions
#   G~_L(R)   = [R^-(2L+1) I_L(R) + J_L(R)]/(2L+1),   I_L = int_0^R r^m e^-2r dr (m = L + 6),
#                                                      J_L = int_R^inf r^(5-L) e^-2r dr
#   G~_L'(R)  = -R^-(2L+2) I_L(R)                      (the r^m terms cancel exactly)
#   G~_L''(R) = (2L+2) R^-(2L+3) I_L(R) - R^(4-L) e^-2R
# I_L is the lower incomplete gamma function, evaluated without cancellation:
#   2R < SERIES_SWITCH:  I = R^(m+1) e^-2R sum_k (2R)^k / ((m+1)(m+2)...(m+1+k))   (positive terms)
#   otherwise:           I = m!/2^(m+1) - e^-2R sum_j m!/(j! 2^(m-j+1)) R^j        (tail < m!/2^(m+1))
# and J_L = e^-2R sum_j (5-L)!/(j! 2^(5-L-j+1)) R^j (positive terms).
SERIES_SWITCH = 24.0
_N_SERIES = 90


def _upper_poly(m: int, R: torch.Tensor) -> torch.Tensor:
    """sum_j m!/(j! 2^(m-j+1)) R^j (Horner)."""
    out = torch.zeros_like(R)
    for j in range(m, -1, -1):
        out = out * R + math.factorial(m) / (math.factorial(j) * 2.0 ** (m - j + 1))
    return out


def _lower_gamma(m: int, R: torch.Tensor) -> torch.Tensor:
    x = 2.0 * R
    small = x < SERIES_SWITCH
    Rs = torch.where(small, R, torch.full_like(R, 1.0))                   # keep the unused branch finite
    xs = 2.0 * Rs
    s = torch.zeros_like(R)
    for k in range(_N_SERIES - 1, -1, -1):                                 # sum_k x^k / prod_{i=0..k}(m+1+i)
        s = (1.0 + xs * s) / (m + 1 + k)
    ser = Rs ** (m + 1) * torch.exp(-xs) * s
    Rl = torch.where(small, torch.full_like(R, SERIES_SWITCH), R)
    up = math.factorial(m) / 2.0 ** (m + 1) - torch.exp(-2.0 * Rl) * _upper_poly(m, Rl)
    return torch.where(small, ser, up)


def _radial(R: torch.Tensor) -> torch.Tensor:
    """(..., 3 L-values (0, 2, 4), 3 derivatives) of G~_L at zeta R = R; R > 0."""
    e = torch.exp(-2.0 * R)
    out = []
    for L in (0, 2, 4):
        I = _lower_gamma(L + 6, R)
        J = e * _upper_poly(5 - L, R)
        G = (R ** (-(2 * L + 1)) * I + J) / (2 * L + 1)
        G1 = -(R ** (-(2 * L + 2))) * I
        G2 = (2 * L + 2) * R ** (-(2 * L + 3)) * I - R ** (4 - L) * e
        out.append(torch.stack([G, G1, G2], -1))
    return torch.stack(out, -2)


# ------------------------------------------------------------------------------ tensor
def _t(x, device=None) -> torch.Tensor:
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=_F64, device=device)


D_XY = ((0.0, 0.5, 0.0), (0.5, 0.0, 0.0), (0.0, 0.0, 0.0))
_ORBITALS = {"xy": D_XY, "xz": ((0, 0, 0.5), (0, 0, 0), (0.5, 0, 0)), "yz": ((0, 0, 0), (0, 0, 0.5), (0, 0.5, 0)),
             "x2-y2": ((0.5, 0, 0), (0, -0.5, 0), (0, 0, 0)), "z2": ((-1, 0, 0), (0, -1, 0), (0, 0, 2.0))}


def orbital_matrix(name: str = "xy") -> torch.Tensor:
    """Matrix M of a real d orbital ``Y(r) = r^T M r``.

    Parameters
    ----------
    name : {'xy', 'xz', 'yz', 'x2-y2', 'z2'}, default 'xy'
        The orbital.

    Returns
    -------
    torch.Tensor, shape (3, 3), float64
        Symmetric; the overall scale is irrelevant.

    Raises
    ------
    KeyError
        For an unknown name.
    """
    return torch.tensor(_ORBITALS[name], dtype=_F64)


def _shapes(*batch):
    b = torch.broadcast_shapes(*batch)
    return (b, b, (*b, 3), (*b, 3, 3))


def sto_tensor_full(A, direction, orbital=None) -> torch.Tensor:
    """Reduced dipolar tensor of a 3d Slater orbital, ``t = R^3 <(3 u u^T - u^2)/u^5>``.

    Parameters
    ----------
    A : float or torch.Tensor, shape (...)
        Slater parameter ``A = Z* R/(2 a0)`` (> 0; :func:`slater_A`).
    direction : array_like or torch.Tensor, shape (..., 3)
        Metal -> point vector in the orbital frame (normalized here, so it may carry gradients
        through angles).
    orbital : array_like or torch.Tensor, shape (..., 3, 3), optional
        Real symmetric M of ``Y = r^T M r`` (made traceless here); default d_xy.

    Returns
    -------
    torch.Tensor, shape (..., 3, 3)
        Dimensionless.  The point-dipole limit (A -> inf) is ``3 n n^T - 1``.

    Examples
    --------
    >>> import torch
    >>> from torchspin.lft import dipole
    >>> t = dipole.sto_tensor_full(1e3, [1.0, 0.0, 0.0])
    >>> [round(x, 3) for x in torch.diagonal(t).tolist()]
    [2.0, -1.0, -1.0]
    """
    A = _t(A)
    n = _t(direction, A.device)
    n = n / torch.linalg.vector_norm(n, dim=-1, keepdim=True)
    M = orbital_matrix("xy").to(A.device) if orbital is None else _t(orbital, A.device)
    eye = torch.eye(3, dtype=_F64, device=A.device)
    M = 0.5 * (M + M.transpose(-1, -2))
    M = M - torch.diagonal(M, dim1=-2, dim2=-1).sum(-1)[..., None, None] / 3 * eye
    R = (2.0 / 3.0) * A                                                  # zeta R
    A, R, n, M = (torch.broadcast_to(x, s) for x, s in zip(
        (A, R, n, M), _shapes(A.shape, n.shape[:-1], M.shape[:-2])))
    M2 = M @ M
    tr2 = torch.diagonal(M2, dim1=-2, dim2=-1).sum(-1)
    h0 = 2.0 * tr2 / 15.0
    Q2 = (4.0 / 7.0) * (M2 - tr2[..., None, None] / 3.0 * eye)
    # work with unit vector n and P = R n: polynomials of degree L scale as R^L
    Mn = (M @ n.unsqueeze(-1)).squeeze(-1)
    Qn = (Q2 @ n.unsqueeze(-1)).squeeze(-1)
    y = (n * Mn).sum(-1)                         # Y(n)
    q = (n * Qn).sum(-1)                         # h2(n)
    nn = n.unsqueeze(-1) * n.unsqueeze(-2)
    outer = lambda a, b: a.unsqueeze(-1) * b.unsqueeze(-2)
    # h_L(n), grad h_L(n), Hessian h_L(n) (at unit radius; scale by R^L, R^(L-1), R^(L-2))
    h = [h0, q, y * y - q - h0]
    g = [torch.zeros_like(n), 2.0 * Qn,
         4.0 * y[..., None] * Mn - (2.0 * n * q[..., None] + 2.0 * Qn) - 4.0 * h0[..., None] * n]
    hess = [torch.zeros_like(M), 2.0 * Q2,
            8.0 * outer(Mn, Mn) + 4.0 * y[..., None, None] * M
            - (2.0 * q[..., None, None] * eye + 4.0 * (outer(n, Qn) + outer(Qn, n)) + 2.0 * Q2)
            - h0[..., None, None] * (4.0 * eye + 8.0 * nn)]
    rad = _radial(R)                                                     # (..., 3, 3)
    pref = 4.0 / (3.0 * tr2)                                             # 4 pi N^2
    T = torch.zeros_like(M)
    for i, L in enumerate((0, 2, 4)):
        G, G1, G2 = rad[..., i, 0], rad[..., i, 1], rad[..., i, 2]
        RL = R**L
        # R^3 d_i d_j [G(R) h(P)], h(P) = R^L h(n), grad = R^(L-1) g(n), Hessian = R^(L-2) H(n)
        term = (G2 * RL * R**3)[..., None, None] * h[i][..., None, None] * nn \
            + (G1 * RL * R**2)[..., None, None] * h[i][..., None, None] * (eye - nn) \
            + (G1 * RL * R**2)[..., None, None] * (outer(n, g[i]) + outer(g[i], n)) \
            + (G * RL * R)[..., None, None] * hess[i]
        T = T + term
    T = pref[..., None, None] * T
    rho = (4.0 / (9.0 * tr2)) * (y * R**2) ** 2 * torch.exp(-2.0 * R)    # (4 pi/3) N^2 Y(P)^2 e^-2R
    T = T + (rho * R**3)[..., None, None] * eye
    return 0.5 * (T + T.transpose(-1, -2))


_S2 = 1.0 / math.sqrt(2.0)
_GEOM = {  # direction of the point in the d_xy frame; dipole.f axes (y, x, z) as rows
    "in_plane": ((1.0, 0.0, 0.0), ((0, 0, 1.0), (0, 1.0, 0), (1.0, 0, 0))),
    "out_of_plane": ((0.0, 0.0, 1.0), ((0, 1.0, 0), (1.0, 0, 0), (0, 0, 1.0))),
    "in_plane_lobe": ((_S2, _S2, 0.0), ((0, 0, 1.0), (-_S2, _S2, 0), (_S2, _S2, 0))),
}


def sto_tensor(A, geometry: str = "in_plane") -> torch.Tensor:
    """Reduced dipolar tensor of d_xy on dipole.f's axes for a dipole.f geometry.

    Parameters
    ----------
    A : float or torch.Tensor, shape (...)
        Slater parameter (> 0).
    geometry : {'in_plane', 'out_of_plane', 'in_plane_lobe'}, default 'in_plane'
        ``'in_plane'`` (dipole.f 'I': the point on a d_xy nodal axis), ``'out_of_plane'`` ('O': on
        the orbital's normal) or ``'in_plane_lobe'`` (along a lobe; telser-tools addition).

    Returns
    -------
    torch.Tensor, shape (..., 3)
        (t_y, t_x, t_z), dimensionless, z along the metal-point vector.

    Raises
    ------
    ValueError
        For an unknown geometry.
    """
    if geometry not in GEOMETRIES:
        raise ValueError(f"geometry must be one of {GEOMETRIES}, got {geometry!r}")
    A = _t(A)
    d, axes = _GEOM[geometry]
    E = torch.tensor(axes, dtype=_F64, device=A.device)
    T = sto_tensor_full(A, torch.tensor(d, dtype=_F64, device=A.device))
    return torch.einsum("ai,...ij,aj->...a", E, T, E)


def point_dipole_full(direction) -> torch.Tensor:
    """Point-dipole tensor ``3 n n^T - 1``, shape (..., 3, 3), for the metal -> point direction (normalized here)."""
    n = _t(direction)
    n = n / torch.linalg.vector_norm(n, dim=-1, keepdim=True)
    return 3.0 * n.unsqueeze(-1) * n.unsqueeze(-2) - torch.eye(3, dtype=_F64, device=n.device)


def slater_A(zstar, r_angstrom, bohr_cm: float = TORCHSPIN_CONSTANTS.a0_cm) -> torch.Tensor:
    """Slater parameter ``A = Z* R/(2 a0)``.

    ``zstar`` is the effective nuclear charge Z*, ``r_angstrom`` the distance in Angstrom,
    ``bohr_cm`` the Bohr radius in cm.  Float or tensor inputs; returns a float64 tensor.
    """
    return _t(zstar) * _t(r_angstrom) * 1e-8 / (2.0 * bohr_cm)


def zstar(n_d: int, z: int) -> torch.Tensor:
    """Clementi-Raimondi effective nuclear charge Z* of a 3d electron (dipole.f CLEMRAIM).

    ``n_d`` is the number of d electrons and ``z`` the atomic number.  Returns a float64 scalar
    tensor (integer inputs: no gradient).
    """
    return torch.tensor(clementi_raimondi(n_d, z).value, dtype=_F64)


def d_and_e(d_yxz: torch.Tensor) -> torch.Tensor:
    """Axial and rhombic dipolar parameters ``(Ddip, Edip) = (3/2 D_z, (D_x - D_y)/2)``.

    ``d_yxz`` is a tensor (..., 3) ordered (D_y, D_x, D_z); returns (..., 2) in the same unit.
    """
    return torch.stack([1.5 * d_yxz[..., 2], 0.5 * (d_yxz[..., 1] - d_yxz[..., 0])], -1)


# ------------------------------------------------------------------------------ couplings
class Hyperfine(NamedTuple):
    """Electron-nuclear dipolar coupling (:func:`hyperfine`) on the axes (y, x, z).

    Attributes
    ----------
    A : torch.Tensor
        Slater parameter.
    zstar : torch.Tensor
        Effective nuclear charge Z*.
    tensor_cm3 : torch.Tensor, shape (..., 3)
        ``<(3 u u^T - u^2)/u^5>`` in cm^-3.
    a : torch.Tensor, shape (..., 3)
        Coupling without g correction, in ``units``.
    a_g : torch.Tensor, shape (..., 3)
        Times ``g_i/g_e``.
    a_alpha : torch.Tensor, shape (..., 3)
        Also times alpha1^2.
    point_dipole : torch.Tensor, shape (..., 3)
        Point-dipole value with g and alpha1^2.
    """

    A: torch.Tensor
    zstar: torch.Tensor
    tensor_cm3: torch.Tensor
    a: torch.Tensor
    a_g: torch.Tensor
    a_alpha: torch.Tensor
    point_dipole: torch.Tensor


class ElectronElectron(NamedTuple):
    """Electron-electron dipolar coupling ``H = S1.D.S2`` (:func:`electron_electron`) on the axes (y, x, z).

    Energies in the ``units`` asked for.  For two spins 1/2 combined to a triplet, the triplet
    zero-field splitting is half of these values.

    Attributes
    ----------
    A, zstar, tensor_cm3 : torch.Tensor
        As :class:`Hyperfine`.
    d : torch.Tensor, shape (..., 3)
        Coupling without g correction.
    d_g : torch.Tensor, shape (..., 3)
        Times ``g_i g' / g_e^2``.
    d_alpha : torch.Tensor, shape (..., 3)
        Also times alpha1^2.
    point_dipole : torch.Tensor, shape (..., 3)
        Point-dipole value with g (no alpha1^2, as dipole.f).
    """

    A: torch.Tensor
    zstar: torch.Tensor
    tensor_cm3: torch.Tensor
    d: torch.Tensor
    d_g: torch.Tensor
    d_alpha: torch.Tensor
    point_dipole: torch.Tensor

    @property
    def ddip_edip(self) -> torch.Tensor:
        """(Ddip, Edip) of the g-corrected tensor ``d_g``."""
        return d_and_e(self.d_g)

    @property
    def ddip_edip_point(self) -> torch.Tensor:
        """(Ddip, Edip) of the point-dipole tensor."""
        return d_and_e(self.point_dipole)

    @property
    def ddip_edip_alpha(self) -> torch.Tensor:
        """(Ddip, Edip) of ``d_alpha``."""
        return d_and_e(self.d_alpha)


_POINT = (-1.0, -1.0, 2.0)


def _check(r: torch.Tensor, alpha2: torch.Tensor) -> None:
    if bool((r.detach() <= 0).any()):
        raise ValueError("distance must be > 0")
    a = alpha2.detach()
    if bool(((a < 0) | (a > 1)).any()):
        raise ValueError("alpha1^2 must lie in [0, 1]")


def _setup(n_d, z, r_angstrom, geometry, zstar_, constants):
    zs = zstar(n_d, z) if zstar_ is None else _t(zstar_)
    r = _t(r_angstrom)
    A = slater_A(zs, r, constants.a0_cm)
    r3 = (r * 1e-8) ** 3
    T = sto_tensor(A, geometry) / r3.unsqueeze(-1)
    return zs, r3, A, T


def _g_n(nucleus, g_n) -> torch.Tensor:
    if g_n is not None:
        return _t(g_n)
    if nucleus is None:
        raise ValueError("give a nucleus (e.g. '14N') or g_n")
    from torchspin.nucdata import nucgval
    return torch.tensor(float(nucgval(str(nucleus))), dtype=_F64)


def hyperfine(n_d: int, z: int, nucleus: str | None, r_angstrom, g=(2.0023, 2.0023, 2.0023), alpha2=1.0,
              geometry: str = "in_plane", *, zstar=None, g_n=None, units: str = "MHz",
              constants: Constants = TORCHSPIN_CONSTANTS) -> Hyperfine:
    """Electron-nuclear dipolar (hyperfine) coupling of a metal d electron to a ligand nucleus.

    Parameters
    ----------
    n_d : int
        Number of d electrons (for the Clementi-Raimondi Z*).
    z : int
        Atomic number of the metal.
    nucleus : str or None
        Isotope name for :func:`torchspin.nucgval` (e.g. ``'14N'``); may be None when ``g_n`` is
        given.
    r_angstrom : float or torch.Tensor
        Metal-nucleus distance in Angstrom (> 0).
    g : array_like or torch.Tensor, shape (..., 3), default (2.0023, 2.0023, 2.0023)
        g values ordered (g_y, g_x, g_z).
    alpha2 : float or torch.Tensor, default 1.0
        alpha1^2, the metal-orbital population, in [0, 1].
    geometry : str, default 'in_plane'
        As :func:`sto_tensor`.
    zstar : float or torch.Tensor, optional
        Overrides the tabulated Z* (so the orbital exponent may carry gradients).
    g_n : float or torch.Tensor, optional
        Overrides the nuclear g value.
    units : str, default 'MHz'
        Energy unit of the couplings.
    constants : Constants, optional
        Default :data:`TORCHSPIN_CONSTANTS`.

    Returns
    -------
    Hyperfine

    Raises
    ------
    ValueError
        For a distance <= 0, alpha2 outside [0, 1], an unknown geometry or energy unit, or
        neither ``nucleus`` nor ``g_n``.
    """
    alpha2 = _t(alpha2)
    _check(_t(r_angstrom), alpha2)
    zs, r3, A, T = _setup(n_d, z, r_angstrom, geometry, zstar, constants)
    gn = _g_n(nucleus, g_n)
    g = _t(g)
    c = constants
    f = _u.energy_factor("MHz", units)
    pref = c.g_e * gn * c.beta_e * c.beta_n / c.h * 1e-6                  # MHz cm^3
    ad = pref[..., None] * T
    ag = ad * g / c.g_e
    aa = ag * alpha2[..., None]
    pt = (pref * alpha2)[..., None] * (g / c.g_e) * torch.tensor(_POINT, dtype=_F64) / r3[..., None]
    return Hyperfine(A, zs, T, ad * f, ag * f, aa * f, pt * f)


def electron_electron(n_d: int, z: int, r_angstrom, g=(2.0023, 2.0023, 2.0023), alpha2=1.0,
                      geometry: str = "in_plane", *, partner_g=None, zstar=None, units: str = "MHz",
                      constants: Constants = TORCHSPIN_CONSTANTS) -> ElectronElectron:
    """Electron-electron dipolar coupling between the metal d electron and a point spin.

    Parameters
    ----------
    n_d, z, r_angstrom, g, alpha2, geometry, zstar, constants
        As :func:`hyperfine` (``r_angstrom`` is the distance to the point spin).
    partner_g : float or torch.Tensor, optional
        g of the second (point) spin; default g_e.
    units : str, default 'MHz'
        Energy unit of the couplings (dipole.f printed MHz and cm^-1: use ``units='cm^-1'``).

    Returns
    -------
    ElectronElectron

    Raises
    ------
    ValueError
        For a distance <= 0, alpha2 outside [0, 1], or an unknown geometry or energy unit.
    """
    alpha2 = _t(alpha2)
    _check(_t(r_angstrom), alpha2)
    zs, r3, A, T = _setup(n_d, z, r_angstrom, geometry, zstar, constants)
    c = constants
    gp = _t(c.g_e if partner_g is None else partner_g)
    g = _t(g)
    pref = -(c.g_e**2) * c.beta_e**2 / c.h * 1e-6                         # MHz cm^3
    d = pref * T
    dg = d * g * gp[..., None] / c.g_e**2
    da = dg * alpha2[..., None]
    pt_cm = c.ee_point_cm * g * gp[..., None] * (-torch.tensor(_POINT, dtype=_F64)) / r3[..., None]
    f = _u.energy_factor("MHz", units)
    pt = pt_cm if _u.canonical_energy_unit(units) == "cm^-1" else pt_cm / c.mhz_to_cm * f
    return ElectronElectron(A, zs, T, d * f, dg * f, da * f, pt)
