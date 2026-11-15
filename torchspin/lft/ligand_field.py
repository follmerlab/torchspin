"""Differentiable one-electron d-shell ligand fields: AOM, Ballhausen crystal field, orbital energies.

Every constructor returns a real symmetric matrix (..., 5, 5) in the real-orbital basis
``(x2-y2, xy, xz, yz, z2)`` (Telser's DDN order and geometric phase convention).  Leading
dimensions batch, and every argument may be a tensor carrying gradients (tensor inputs must be
float64; Python floats and lists are converted).

Energy units pass straight through: the matrix is in whatever unit the bonding parameters
(e_sigma, Dq, ...) are given in.  :mod:`torchspin.lft.ddn` reads it in the unit named by its
``units=`` argument.  Angles are in radians (torchspin's convention); pass ``degrees=True``
for Telser-style input.

AOM entry points:

* :func:`aom_isotropic` (``n, e_sigma, e_pi, e_delta``): from bond vectors only.  With
  isotropic pi (and delta) bonding the AOM depends only on the bond direction, so there are no
  Euler angles and no gimbal singularity at theta = 0.
* :func:`aom_from_angles` (``theta, phi, psi, e``): the general (anisotropic pi/delta) path:
  frame ``R = Rz(phi) Ry(theta) Rz(psi)``, local orbital k -> ``c = <Q_j, R Q_k R^T>``,
  ``V = sum e c c^T``.  ``e`` columns are ``AOM_MODES``.
* :func:`aom_from_frames` (``R, e``): the same from rotation matrices (columns = local x', y',
  z').
* :func:`aom` (``ligands``): from a list of :class:`Ligand`.

Examples
--------
An octahedron with e_sigma = 5000 and e_pi = 1000 cm^-1: t2g at 4 e_pi, eg at 3 e_sigma.

>>> import torch
>>> from torchspin.lft import ligand_field as lf
>>> V = lf.aom(lf.octahedron(5000.0, 1000.0))                 # cm^-1 in, cm^-1 out
>>> [round(e, 6) for e in torch.linalg.eigvalsh(V).tolist()]
[4000.0, 4000.0, 4000.0, 15000.0, 15000.0]

The Ballhausen field with Dq = 1000 (e at +6 Dq, t2 at -4 Dq):

>>> V = lf.crystal_field(Dq=1000.0)                            # (x2-y2, xy, xz, yz, z2)
>>> [round(e, 6) for e in torch.diagonal(V).tolist()]
[6000.0, -4000.0, -4000.0, -4000.0, 6000.0]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch

from ._engines import ligand_field as _np_lf

__all__ = ["AOM_MODES", "MIX_PAIRS", "Ligand", "aom", "aom_from_angles", "aom_from_frames", "aom_isotropic",
           "crystal_field", "octahedron", "orbital_coefficients", "orbital_matrix", "rotation", "tetrahedron"]

AOM_MODES = ("sigma", "pi_x", "pi_y", "delta_x2y2", "delta_xy")
MIX_PAIRS = ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
_F64 = torch.float64


def _Q(device=None) -> torch.Tensor:
    return torch.as_tensor(_np_lf.Q, dtype=_F64, device=device)


def _t(x, device=None) -> torch.Tensor:
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=_F64, device=device)


# ------------------------------------------------------------------------------ frames
def rotation(theta, phi, psi=0.0, degrees: bool = False) -> torch.Tensor:
    """Ligand frame ``R = Rz(phi) Ry(theta) Rz(psi)``; columns are the ligand's x', y', z'.

    z' points from the metal to the ligand (polar angle ``theta``, azimuth ``phi``).  For
    ``psi = 0``, x' lies in the plane of z and z' and y' is along +phi-hat.

    Parameters
    ----------
    theta, phi : float or torch.Tensor
        Polar angle and azimuth of the metal -> ligand direction.
    psi : float or torch.Tensor, default 0.0
        Rotation of the ligand frame about z'.
    degrees : bool, default False
        Angles in degrees instead of radians.

    Returns
    -------
    torch.Tensor, shape (..., 3, 3)
        Broadcast over the angles.
    """
    theta, phi, psi = torch.broadcast_tensors(_t(theta), _t(phi), _t(psi))
    if degrees:
        theta, phi, psi = (torch.deg2rad(a) for a in (theta, phi, psi))

    def rz(a):
        c, s = torch.cos(a), torch.sin(a)
        z, o = torch.zeros_like(a), torch.ones_like(a)
        return torch.stack([torch.stack([c, -s, z], -1), torch.stack([s, c, z], -1), torch.stack([z, z, o], -1)], -2)

    c, s = torch.cos(theta), torch.sin(theta)
    z, o = torch.zeros_like(theta), torch.ones_like(theta)
    ry = torch.stack([torch.stack([c, z, s], -1), torch.stack([z, o, z], -1), torch.stack([-s, z, c], -1)], -2)
    return rz(phi) @ ry @ rz(psi)


def orbital_coefficients(R: torch.Tensor) -> torch.Tensor:
    """Real-orbital coefficients of rotated orbitals.

    ``R`` (..., 3, 3) is a frame from :func:`rotation`; returns ``C[..., k, j]``, the coefficient of
    global real orbital j in local orbital k, shape (..., 5, 5), orbitals in
    ``(x2-y2, xy, xz, yz, z2)`` order.
    """
    q = _Q(R.device)
    rq = R.unsqueeze(-3) @ q @ R.unsqueeze(-3).transpose(-2, -1)
    return torch.einsum("...kxy,jxy->...kj", rq, q)


# ------------------------------------------------------------------------------ AOM
def aom_from_frames(R: torch.Tensor, e) -> torch.Tensor:
    """Angular-overlap ligand field from ligand frames.

    ``V = sum_L sum_mode e[L, mode] c c^T``.

    Parameters
    ----------
    R : torch.Tensor, shape (..., nlig, 3, 3)
        Ligand frames (columns x', y', z'; z' metal -> ligand).
    e : array_like or torch.Tensor, shape (..., nlig, 5)
        Bonding parameters in ``AOM_MODES`` order (sigma, pi_x, pi_y, delta_x2y2, delta_xy), any
        energy unit.

    Returns
    -------
    torch.Tensor, shape (..., 5, 5)
        In the unit of ``e``.
    """
    C = orbital_coefficients(R)[..., list(_np_lf.AOM_LOCAL), :]
    return torch.einsum("...lm,...lma,...lmb->...ab", _t(e, R.device), C, C)


def aom_from_angles(theta, phi, psi, e, degrees: bool = False) -> torch.Tensor:
    """General angular-overlap ligand field from per-ligand angles.

    Parameters
    ----------
    theta, phi, psi : float or torch.Tensor, shape (..., nlig)
        Ligand directions and frame rotations, as :func:`rotation`.
    e : array_like or torch.Tensor, shape (..., nlig, 5)
        Bonding parameters in ``AOM_MODES`` order, any energy unit.
    degrees : bool, default False
        Angles in degrees.

    Returns
    -------
    torch.Tensor, shape (..., 5, 5)
        In the unit of ``e``.
    """
    return aom_from_frames(rotation(theta, phi, psi, degrees), _t(e))


def aom_isotropic(n, e_sigma, e_pi=0.0, e_delta=0.0) -> torch.Tensor:
    """Angular-overlap ligand field from bond vectors, with isotropic pi and delta bonding.

    The projectors per ligand, with ``s_a = <Q_a, (3 n n^T - I)/sqrt(6)>``::

        sigma:  s s^T
        pi:     P_pi[a, b] = 2 n^T Q_a (I - n n^T) Q_b n
        delta:  I - s s^T - P_pi

    Parameters
    ----------
    n : array_like or torch.Tensor, shape (..., nlig, 3)
        Metal -> ligand vectors (normalized here); float64 if a tensor.
    e_sigma : float or torch.Tensor
        sigma parameter; broadcasts against (..., nlig).
    e_pi, e_delta : float or torch.Tensor, default 0.0
        Isotropic pi and delta parameters.

    Returns
    -------
    torch.Tensor, shape (..., 5, 5)
        In the unit of the e parameters; smooth in ``n`` everywhere.
    """
    n = _t(n)
    n = n / torch.linalg.vector_norm(n, dim=-1, keepdim=True)
    q = _Q(n.device)
    eye3 = torch.eye(3, dtype=n.dtype, device=n.device)
    nn = n.unsqueeze(-1) * n.unsqueeze(-2)
    s = torch.einsum("...xy,jxy->...j", (3.0 * nn - eye3) / math.sqrt(6.0), q)
    qn = torch.einsum("jxy,...y->...jx", q, n)
    p_pi = 2.0 * torch.einsum("...ax,...xy,...by->...ab", qn, eye3 - nn, qn)
    p_sig = s.unsqueeze(-1) * s.unsqueeze(-2)
    p_del = torch.eye(5, dtype=n.dtype, device=n.device) - p_sig - p_pi
    es, ep, ed = (torch.broadcast_to(_t(x, n.device), n.shape[:-1]) for x in (e_sigma, e_pi, e_delta))
    v = es[..., None, None] * p_sig + ep[..., None, None] * p_pi + ed[..., None, None] * p_del
    return v.sum(-3)


@dataclass(frozen=True)
class Ligand:
    """One AOM ligand (:func:`aom`).

    Parameters
    ----------
    theta, phi : float or torch.Tensor
        Polar angle and azimuth of the metal -> ligand direction, radians (see
        :meth:`from_degrees`).
    psi : float or torch.Tensor, default 0.0
        Rotation of the local frame about the bond, radians.
    e_sigma : float or torch.Tensor, default 0.0
        sigma bonding parameter (any energy unit).
    e_pi_x, e_pi_y : float or torch.Tensor, default 0.0
        pi parameters acting on d(x'z') and d(y'z') of the local frame (DDN: "pi-c"/"pi-s").
    e_delta_x2y2, e_delta_xy : float or torch.Tensor, default 0.0
        delta parameters acting on d(x'2-y'2) and d(x'y') (DDN: "delta-c"/"delta-s").
    label : str, default ''
        Free-form name.
    """

    theta: float
    phi: float
    psi: float = 0.0
    e_sigma: float = 0.0
    e_pi_x: float = 0.0
    e_pi_y: float = 0.0
    e_delta_x2y2: float = 0.0
    e_delta_xy: float = 0.0
    label: str = ""

    @classmethod
    def isotropic(cls, theta, phi, e_sigma, e_pi=0.0, *, degrees: bool = False, label: str = "") -> "Ligand":
        """A ligand with ``e_pi_x = e_pi_y = e_pi`` and ``psi = 0`` (angles in degrees with ``degrees=True``)."""
        if degrees:
            theta, phi = _deg(theta), _deg(phi)
        return cls(theta, phi, 0.0, e_sigma, e_pi, e_pi, label=label)

    @classmethod
    def from_degrees(cls, theta, phi, psi=0.0, **e) -> "Ligand":
        """Telser-style input: ``theta``, ``phi``, ``psi`` in degrees; ``**e`` the bonding parameters."""
        return cls(_deg(theta), _deg(phi), _deg(psi), **e)


def _deg(x):
    return x * (math.pi / 180.0)


def aom(ligands: Iterable[Ligand], device=None) -> torch.Tensor:
    """Angular-overlap ligand field from :class:`Ligand` objects.

    ``V = sum_L sum_lambda e_lambda(L) |lambda_L><lambda_L|`` in the real-orbital basis.

    Parameters
    ----------
    ligands : iterable of Ligand
        The ligands (their bonding parameters in one energy unit).
    device : str or torch.device, optional
        Device of the result.

    Returns
    -------
    torch.Tensor, shape (5, 5)
        In the unit of the bonding parameters.
    """
    ligs = list(ligands)
    col = lambda f: torch.stack([_t(getattr(l, f), device) for l in ligs])  # noqa: E731
    e = torch.stack([col(f) for f in ("e_sigma", "e_pi_x", "e_pi_y", "e_delta_x2y2", "e_delta_xy")], -1)
    return aom_from_angles(col("theta"), col("phi"), col("psi"), e)


def octahedron(e_sigma, e_pi=0.0) -> list[Ligand]:
    """Six identical isotropic ligands on +-x, +-y, +-z (list of :class:`Ligand`)."""
    pos = [(0, 0), (180, 0), (90, 0), (90, 90), (90, 180), (90, 270)]
    return [Ligand.isotropic(t, p, e_sigma, e_pi, degrees=True) for t, p in pos]


def tetrahedron(e_sigma, e_pi=0.0) -> list[Ligand]:
    """Four identical isotropic ligands at alternate cube corners, S4 axes along x, y, z (list of :class:`Ligand`)."""
    t = math.degrees(math.acos(1 / math.sqrt(3.0)))
    pos = [(t, 45), (t, 225), (180 - t, 135), (180 - t, 315)]
    return [Ligand.isotropic(a, b, e_sigma, e_pi, degrees=True) for a, b in pos]


# ------------------------------------------------------------------------------ crystal field, orbitals
def _cf_units(geometry: str, axis: str, device) -> torch.Tensor:
    import numpy as np

    u = np.array([_np_lf.crystal_field(*np.eye(4)[i], geometry=geometry, axis=axis) for i in range(4)])
    return torch.as_tensor(u, dtype=_F64, device=device)


def crystal_field(Dq=0.0, Ds=0.0, Dt=0.0, Dr=0.0, geometry: str = "Oh", axis: str = "tetragonal") -> torch.Tensor:
    """Ballhausen crystal field in DDN's conventions (linear in Dq, Ds, Dt, Dr).

    Parameters
    ----------
    Dq, Ds, Dt, Dr : float or torch.Tensor, default 0.0
        Cubic, axial rank-2, axial rank-4 and rhombic parameters, any one energy unit; broadcast
        against each other.  For d1 (tetragonal) e(g) lies at +6Dq and t2(g) at -4Dq,
        ``E(x2-y2), E(xy) = 2Ds - Dt``, ``E(xz, yz) = -Ds + 4Dt``, ``E(z2) = -2Ds - 6Dt`` relative to
        the cubic levels.
    geometry : {'Oh', 'Td'}, default 'Oh'
        ``'Td'`` multiplies the whole matrix by -1, as DDN does.
    axis : {'tetragonal', 'trigonal'}, default 'tetragonal'
        z along C4 or along C3.

    Returns
    -------
    torch.Tensor, shape (..., 5, 5)
        In the unit of the parameters.

    Raises
    ------
    ValueError
        For an unknown ``geometry`` or ``axis``.
    """
    dq, ds, dt, dr = torch.broadcast_tensors(_t(Dq), _t(Ds), _t(Dt), _t(Dr))
    return torch.einsum("...i,iab->...ab", torch.stack([dq, ds, dt, dr], -1), _cf_units(geometry, axis, dq.device))


def orbital_matrix(energies, mixing=None) -> torch.Tensor:
    """Ligand-field matrix from d-orbital energies (DDN option D).

    Parameters
    ----------
    energies : array_like or torch.Tensor, shape (..., 5)
        Orbital energies in (x2-y2, xy, xz, yz, z2) order, any energy unit.
    mixing : array_like or torch.Tensor, shape (..., 10), optional
        Off-diagonal elements in ``MIX_PAIRS`` order.

    Returns
    -------
    torch.Tensor, shape (..., 5, 5)
        Real symmetric, in the unit of ``energies``.
    """
    e = _t(energies)
    v = torch.diag_embed(e)
    if mixing is not None:
        m = _t(mixing, e.device)
        rows = torch.tensor([a for a, _ in MIX_PAIRS], device=e.device)
        cols = torch.tensor([b for _, b in MIX_PAIRS], device=e.device)
        off = torch.zeros(*torch.broadcast_shapes(v.shape, m.shape[:-1] + (5, 5)), dtype=v.dtype, device=v.device)
        off[..., rows, cols] = m
        v = v + off + off.transpose(-2, -1)
    return v
