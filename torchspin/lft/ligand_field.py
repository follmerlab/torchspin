"""Differentiable one-electron d-shell ligand fields: AOM, Ballhausen crystal field, orbital energies.

Every constructor returns a real symmetric matrix (..., 5, 5) in the real-orbital basis
``(x2-y2, xy, xz, yz, z2)`` (Telser's DDN order and geometric phase convention).  Leading
dimensions batch, and every argument may be a tensor carrying gradients.

Energy units pass straight through: the matrix is in whatever unit the bonding parameters
(e_sigma, Dq, ...) are given in.  :mod:`torchspin.lft.ddn` reads it in the unit named by its
``units=`` argument.  Angles are in radians (torchspin's convention); pass ``degrees=True``
for Telser-style input.

AOM entry points
----------------
``aom_isotropic(n, e_sigma, e_pi, e_delta)``
    From bond vectors only.  With isotropic pi (and delta) bonding the AOM depends only on the
    bond direction, so there are no Euler angles and no gimbal singularity at theta = 0.
``aom_from_angles(theta, phi, psi, e)``
    General (anisotropic pi/delta) path: frame R = Rz(phi) Ry(theta) Rz(psi), local orbital k
    -> c = <Q_j, R Q_k R^T>, V = sum e c c^T.  ``e`` columns are ``AOM_MODES``.
``aom_from_frames(R, e)``
    The same from rotation matrices (columns = local x', y', z').
``aom(ligands)``
    From a list of :class:`Ligand`.
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
    """R = Rz(phi) Ry(theta) Rz(psi), shape (..., 3, 3); columns are the ligand's x', y', z'.

    z' points from the metal to the ligand (polar angle theta, azimuth phi).  For psi = 0,
    x' lies in the plane of z and z' and y' is along +phi-hat.
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
    """C[..., k, j]: global real-orbital coefficients j of local orbital k after rotation R."""
    q = _Q(R.device)
    rq = R.unsqueeze(-3) @ q @ R.unsqueeze(-3).transpose(-2, -1)
    return torch.einsum("...kxy,jxy->...kj", rq, q)


# ------------------------------------------------------------------------------ AOM
def aom_from_frames(R: torch.Tensor, e) -> torch.Tensor:
    """V = sum_L sum_mode e[L, mode] c c^T; R (..., nlig, 3, 3), e (..., nlig, 5) in AOM_MODES order."""
    C = orbital_coefficients(R)[..., list(_np_lf.AOM_LOCAL), :]
    return torch.einsum("...lm,...lma,...lmb->...ab", _t(e, R.device), C, C)


def aom_from_angles(theta, phi, psi, e, degrees: bool = False) -> torch.Tensor:
    """General AOM from per-ligand angles (..., nlig) and bonding parameters (..., nlig, 5)."""
    return aom_from_frames(rotation(theta, phi, psi, degrees), _t(e))


def aom_isotropic(n, e_sigma, e_pi=0.0, e_delta=0.0) -> torch.Tensor:
    """AOM from bond vectors n (..., nlig, 3) (normalized here) with isotropic pi and delta.

        sigma:  |s><s|, s_a = <Q_a, (3 n n^T - I)/sqrt6>
        pi:     P_pi[a, b] = 2 n^T Q_a (I - n n^T) Q_b n
        delta:  I - |s><s| - P_pi

    e_sigma, e_pi, e_delta broadcast against (..., nlig).  Smooth in n everywhere.
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
    """One AOM ligand.  Angles in radians unless built with ``degrees=True``.

    ``e_pi_x``/``e_pi_y`` act on d(x'z')/d(y'z') of the local frame (DDN: "pi-c"/"pi-s");
    ``e_delta_x2y2``/``e_delta_xy`` on d(x'2-y'2)/d(x'y') (DDN: "delta-c"/"delta-s").
    Values may be floats or tensors.
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
        if degrees:
            theta, phi = _deg(theta), _deg(phi)
        return cls(theta, phi, 0.0, e_sigma, e_pi, e_pi, label=label)

    @classmethod
    def from_degrees(cls, theta, phi, psi=0.0, **e) -> "Ligand":
        """Telser-style input: angles in degrees."""
        return cls(_deg(theta), _deg(phi), _deg(psi), **e)


def _deg(x):
    return x * (math.pi / 180.0)


def aom(ligands: Iterable[Ligand], device=None) -> torch.Tensor:
    """V = sum_L sum_lambda e_lambda(L) |lambda_L><lambda_L| (real-orbital matrix) from :class:`Ligand` s."""
    ligs = list(ligands)
    col = lambda f: torch.stack([_t(getattr(l, f), device) for l in ligs])  # noqa: E731
    e = torch.stack([col(f) for f in ("e_sigma", "e_pi_x", "e_pi_y", "e_delta_x2y2", "e_delta_xy")], -1)
    return aom_from_angles(col("theta"), col("phi"), col("psi"), e)


def octahedron(e_sigma, e_pi=0.0) -> list[Ligand]:
    """Six identical ligands on +-x, +-y, +-z."""
    pos = [(0, 0), (180, 0), (90, 0), (90, 90), (90, 180), (90, 270)]
    return [Ligand.isotropic(t, p, e_sigma, e_pi, degrees=True) for t, p in pos]


def tetrahedron(e_sigma, e_pi=0.0) -> list[Ligand]:
    """Four identical ligands at alternate cube corners (S4 axes along x, y, z)."""
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

    e(g) at +6Dq, t2(g) at -4Dq; ``axis`` 'tetragonal' (z = C4) or 'trigonal' (z = C3);
    ``geometry='Td'`` multiplies the whole matrix by -1, as DDN does.
    """
    dq, ds, dt, dr = torch.broadcast_tensors(_t(Dq), _t(Ds), _t(Dt), _t(Dr))
    return torch.einsum("...i,iab->...ab", torch.stack([dq, ds, dt, dr], -1), _cf_units(geometry, axis, dq.device))


def orbital_matrix(energies, mixing=None) -> torch.Tensor:
    """Real-orbital matrix from d-orbital energies (..., 5) in (x2-y2, xy, xz, yz, z2) order and
    optional mixing elements (..., 10) in ``MIX_PAIRS`` order (DDN option D)."""
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
