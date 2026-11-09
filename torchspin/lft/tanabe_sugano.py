"""Tanabe-Sugano: cubic d^n levels and diagrams, differentiable (Telser's DTANSUG via telser-tools).

Two independent routes to the levels of d^n (n = 1 ... 9) in an octahedral (Oh) or tetrahedral
(Td) field, both linear in (B, C, Dq) over fixed matrices:

Strong-field blocks (DTANSUG.F; McClure 1959, Griffith 1964 Table A.27)
    Every (2S+1)Gamma block in the t2^a e^b basis, with DTANSUG's hole-problem signs (SIGNXF,
    SIGNEL) and the DIAG offset of d6-d8.  The NumPy tables
    (:mod:`torchspin.lft._engines.tanabe_sugano`) are probed once at the unit vectors and checked
    for linearity, so ``H = B M_B + C M_C + Dq M_Dq`` with constant matrices.  The Racah A term
    is left out (add n(n-1)/2 A), and for d9 also the constant -56B + 28C, as in DTANSUG.
Full determinant basis
    :func:`spectrum`: all C(10, n) states from :mod:`torchspin.lft.ddn` (Racah A, B, C plus the
    Ballhausen cubic field; Td reverses the field).

Dq > 0 is the physical sign for both geometries (Oh: e above t2; Td: e below t2).

Units
-----
``B``, ``C``, ``Dq``, ``A`` and every returned energy are in ``units`` (default ``'MHz'``;
``'cm^-1'``, ``'GHz'``, ``'K'``, ``'eV'``).  The block matrices are homogeneous of degree one in
(B, C, Dq), so the strong-field route needs no conversion; ``units`` is still checked and
recorded.  The diagram uses the conventional reduced axes, E/B (relative to the ground state)
against Dq/B at fixed C/B, which carry no unit; pass ``B`` to :func:`tanabe_sugano` (or use
:meth:`TanabeSuganoDiagram.at`) for absolute energies and Dq in ``units``.

    >>> from torchspin.lft import tanabe_sugano as ts
    >>> e = ts.block_energies(3, "4T1", B=766.0, C=2855.0, Dq=1200.0, units="cm^-1")
    >>> d = ts.tanabe_sugano(5, c_over_b=4.5)                     # E/B vs Dq/B
    >>> d.e_over_b, d.labels, d.ground, d.crossover()
    >>> d.at(B=800.0, units="cm^-1").energy                       # absolute, cm^-1

Derivatives: energies through :func:`torchspin.lft._linalg.eigvalsh_gauge` (a degenerate
eigenvalue gets its level's mean derivative, exact for the symmetry-protected degeneracies of
the full determinant spectrum).  ``weights`` (|c_ij|^2) depend on the basis inside a degenerate
level and are differentiable only where the state is non-degenerate within its block.  The
order of :func:`all_levels` and the diagram's ground curve are decided on detached values.

Not ported: the DTANSUG text report, the Fortran grid rules (``fortran_grid``: overshooting and
always-ascending ranges, telser-tools findings 4-5) and the Fortran B-loop sign bug (finding 1;
:func:`scan_b` uses the correct sign, as the telser-tools NumPy port does).  Plots are separate.
"""

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import numpy as np
import torch

from . import units as _u
from ._engines import tanabe_sugano as _blocks
from ._engines.tanabe_sugano import BLOCKS, GEOMETRIES, Block, blocks_for, get_block
from ._linalg import eigh_gauge, eigvalsh_gauge
from .ddn import levels as _dn_levels
from .ligand_field import crystal_field

__all__ = ["BLOCKS", "GEOMETRIES", "Block", "BlockSolution", "CubicLevel", "Scan", "TanabeSuganoDiagram",
           "all_levels", "block_energies", "block_hamiltonian", "block_operators", "blocks_for", "get_block",
           "scan_b", "scan_dq", "solve_block", "spectrum", "tanabe_sugano"]

_F64 = torch.float64


def _t(x, device=None) -> torch.Tensor:
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=_F64, device=device)


def _block(n: int, label) -> Block:
    return label if isinstance(label, Block) else get_block(n, label)


@lru_cache(maxsize=None)
def _block_ops(n: int, key: tuple[int, int], geometry: str, device: str) -> torch.Tensor:
    blk = next(b for b in BLOCKS if (b.nh, b.first) == key)
    f = lambda b, c, d: _blocks.hamiltonian(n, blk, b, c, d, geometry)  # noqa: E731
    if np.any(f(0.0, 0.0, 0.0) != 0):
        raise AssertionError("block matrix has a constant part")
    M = np.array([f(1.0, 0.0, 0.0), f(0.0, 1.0, 0.0), f(0.0, 0.0, 1.0)])
    rng = np.random.default_rng(0)
    for _ in range(3):
        x = rng.normal(size=3) * np.array([1000.0, 4000.0, 1500.0])
        want = f(*x)
        if np.abs(want - np.tensordot(x, M, 1)).max() > 1e-9 * max(1.0, np.abs(want).max()):
            raise AssertionError(f"d{n} {blk.label}: block is not linear in (B, C, Dq)")
    return torch.as_tensor(M, dtype=_F64, device=device)


def block_operators(n: int, label, geometry: str = "Oh", device: str | torch.device = "cpu") -> torch.Tensor:
    """(3, size, size) float64, unit-free: H = B M[0] + C M[1] + Dq M[2] (configurations in ``Block.configs`` order)."""
    blk = _block(n, label)
    if blk.nh != _blocks._nh(n):
        raise ValueError(f"block {blk.label} (nh={blk.nh}) does not belong to d{n}")
    return _block_ops(int(n), (blk.nh, blk.first), _blocks._check_geometry(geometry), str(torch.device(device)))


def block_hamiltonian(n: int, label, B, C, Dq, geometry: str = "Oh", units: str = "MHz") -> torch.Tensor:
    """Real symmetric strong-field block (..., size, size) in ``units``; B, C, Dq (in ``units``) broadcast."""
    _u.canonical_energy_unit(units)
    b, c, d = torch.broadcast_tensors(_t(B), _t(C), _t(Dq))
    M = block_operators(n, label, geometry, b.device)
    return torch.einsum("...k,kij->...ij", torch.stack([b, c, d], -1).to(_F64), M)


def block_energies(n: int, label, B, C, Dq, geometry: str = "Oh", units: str = "MHz") -> torch.Tensor:
    """Ascending eigenvalues (..., size) of one block, in ``units``."""
    return eigvalsh_gauge(block_hamiltonian(n, label, B, C, Dq, geometry, units))


def spectrum(n: int, B, C, Dq, geometry: str = "Oh", A=0.0, units: str = "MHz") -> torch.Tensor:
    """All C(10, n) eigenvalues of d^n in a cubic field (determinant basis, :mod:`torchspin.lft.ddn`), ascending.

    Includes the Racah A term n(n-1)/2 A and, unlike the blocks, the full repulsion energy for
    every n.  ``Dq > 0`` is physical for both geometries.  All energies in ``units``.
    """
    geo = _blocks._check_geometry(geometry)
    return _dn_levels(n, A=A, B=B, C=C, V=crystal_field(_t(Dq), geometry=geo), units=units)


# ------------------------------------------------------------------------------ solutions and scans
class BlockSolution(NamedTuple):
    """One block at (B, C, Dq): ``matrix``, ``energies`` (ascending), ``zeroth_order`` (diagonal),
    ``vectors`` (columns), ``weights[..., i, j]`` = |<config j|state i>|^2; energies in ``units``."""

    n: int
    geometry: str
    block: Block
    B: torch.Tensor
    C: torch.Tensor
    Dq: torch.Tensor
    matrix: torch.Tensor
    energies: torch.Tensor
    zeroth_order: torch.Tensor
    vectors: torch.Tensor
    weights: torch.Tensor
    units: str


def solve_block(n: int, label, B, C, Dq, geometry: str = "Oh", units: str = "MHz") -> BlockSolution:
    """Diagonalize one block of d^n (batched over B, C, Dq; all in ``units``)."""
    geo = _blocks._check_geometry(geometry)
    blk = _block(n, label)
    h = block_hamiltonian(n, blk, B, C, Dq, geo, units)
    w, v = eigh_gauge(h)
    return BlockSolution(int(n), geo, blk, _t(B), _t(C), _t(Dq), h, w, torch.diagonal(h, dim1=-2, dim2=-1), v,
                         (v * v).transpose(-2, -1), _u.canonical_energy_unit(units))


class Scan(NamedTuple):
    """One block along a line of Dq (fixed B, C) or of B (fixed Dq, C/B); ``energies`` (npts, size)."""

    n: int
    geometry: str
    block: Block
    variable: str
    B: torch.Tensor
    C: torch.Tensor
    Dq: torch.Tensor
    energies: torch.Tensor
    zeroth_order: torch.Tensor
    weights: torch.Tensor
    units: str

    @property
    def x(self) -> torch.Tensor:
        return self.Dq if self.variable == "Dq" else self.B


def _scan(n, label, Bs, Cs, Dqs, variable, geometry, units):
    s = solve_block(n, label, Bs, Cs, Dqs, geometry, units)
    b, c, d = torch.broadcast_tensors(_t(Bs), _t(Cs), _t(Dqs))
    return Scan(int(n), s.geometry, s.block, variable, b, c, d, s.energies, s.zeroth_order, s.weights, s.units)


def scan_dq(n: int, label, B, C, dq_values, geometry: str = "Oh", units: str = "MHz") -> Scan:
    """Loop over Dq at fixed B, C (all in ``units``; any may carry gradients)."""
    dq = torch.atleast_1d(_t(dq_values))
    return _scan(n, label, _t(B).expand_as(dq), _t(C).expand_as(dq), dq, "Dq", geometry, units)


def scan_b(n: int, label, Dq, b_values, c_over_b, geometry: str = "Oh", units: str = "MHz") -> Scan:
    """Loop over Racah B with C = (C/B) B at fixed Dq (B, Dq in ``units``; C/B dimensionless).

    The cubic-field sign is that of the Dq loop (DTANSUG.F's B loop omits SIGNXF: telser-tools
    finding 1).
    """
    if not float(c_over_b) > 0:
        raise ValueError("c_over_b must be > 0 (DTANSUG.F silently substitutes 5)")
    b = torch.atleast_1d(_t(b_values))
    return _scan(n, label, b, b * _t(c_over_b), _t(Dq).expand_as(b), "B", geometry, units)


class CubicLevel(NamedTuple):
    """A level of d^n: (2S+1)Gamma ``label``, ``energy`` (in the units asked for), (2S+1)dim(Gamma)
    ``degeneracy``, ``root`` (0-based index within its block)."""

    label: str
    energy: torch.Tensor
    degeneracy: int
    root: int


def all_levels(n: int, B, C, Dq, geometry: str = "Oh", units: str = "MHz") -> list[CubicLevel]:
    """Every level of d^n (all blocks), ascending (order decided on detached energies); scalar B, C, Dq."""
    out = []
    for blk in blocks_for(n):
        e = block_energies(n, blk, B, C, Dq, geometry, units)
        out += [CubicLevel(blk.label, e[r], blk.degeneracy, r) for r in range(blk.size)]
    return sorted(out, key=lambda lv: float(lv.energy.detach()))


# ------------------------------------------------------------------------------ diagram
class TanabeSuganoDiagram(NamedTuple):
    """Tanabe-Sugano diagram data of d^n at fixed C/B.

    ``dq_over_b`` (npts,) the abscissa; ``e_over_b`` (ncurves, npts) E/B relative to the ground
    state at each point; ``labels[i]`` = (block label, root) of curve i (e.g. ('4T1', 1));
    ``degeneracies[i]`` = (2S+1) dim(Gamma) of curve i; ``ground[k]`` = the ground block at point k.
    ``B`` (None, or Racah B in ``units``) turns on the absolute axes :attr:`energy` and :attr:`dq`.
    """

    n: int
    geometry: str
    c_over_b: torch.Tensor
    dq_over_b: torch.Tensor
    labels: tuple[tuple[str, int], ...]
    degeneracies: tuple[int, ...]
    e_over_b: torch.Tensor
    ground: tuple[str, ...]
    B: torch.Tensor | None = None
    units: str = "MHz"

    @property
    def energy(self) -> torch.Tensor | None:
        """(ncurves, npts) energies above the ground state in ``units`` (E/B times B), or None without B."""
        return None if self.B is None else self.e_over_b * self.B[..., None, None]

    @property
    def dq(self) -> torch.Tensor | None:
        """(npts,) Dq in ``units`` (Dq/B times B), or None without B."""
        return None if self.B is None else self.dq_over_b * self.B[..., None]

    def at(self, B, units: str | None = None) -> "TanabeSuganoDiagram":
        """The same diagram with Racah ``B`` given in ``units`` (default: the diagram's units)."""
        u = _u.canonical_energy_unit(self.units if units is None else units)
        return self._replace(B=_t(B), units=u)

    def to(self, units: str) -> "TanabeSuganoDiagram":
        """Convert ``B`` (and so :attr:`energy`, :attr:`dq`) to ``units``; the reduced axes are unchanged."""
        u = _u.canonical_energy_unit(units)
        B = None if self.B is None else _u.convert_energy(self.B, self.units, u)
        return self._replace(B=B, units=u)

    def curve(self, label: str, root: int = 0) -> torch.Tensor:
        """E/B of one curve, e.g. ``curve('4T2')``."""
        lab = label.replace(" ", "").upper()
        for i, (l, r) in enumerate(self.labels):
            if l.upper() == lab and r == root:
                return self.e_over_b[i]
        raise KeyError(f"no curve ({label!r}, {root}); curves: {self.labels}")

    def crossover(self) -> list[float]:
        """Dq/B values (linear interpolation, detached) where the ground state changes."""
        out = []
        x = self.dq_over_b.detach()
        for k in range(1, len(self.ground)):
            a, b = self.ground[k - 1], self.ground[k]
            if a == b:
                continue
            ca, cb = self.curve(a).detach(), self.curve(b).detach()
            d0, d1 = float(cb[k - 1] - ca[k - 1]), float(cb[k] - ca[k])
            if abs(d0) < 1e-9:      # exact tie at the previous point (e.g. 5T2/5E of 5D at Dq = 0)
                continue
            out.append(float(x[k - 1] + (x[k] - x[k - 1]) * d0 / (d0 - d1)))
        return out


def tanabe_sugano(n: int, c_over_b=4.5, dq_over_b=None, geometry: str = "Oh", *, B=None,
                  units: str = "MHz") -> TanabeSuganoDiagram:
    """Tanabe-Sugano diagram: E/B relative to the ground state vs Dq/B at fixed C/B (n = 2 ... 8).

    ``dq_over_b`` defaults to 0 ... 4 (201 points).  E/B depends only on Dq/B and C/B (the
    blocks are linear in B, C, Dq), so B = 1 internally; the result is differentiable in
    ``c_over_b`` and ``dq_over_b``.  Optional ``B`` (Racah B in ``units``) adds the absolute
    axes ``.energy`` and ``.dq`` (differentiable in B).  The ground curve at each point is
    chosen on detached values; E/B then follows that curve.
    """
    if _blocks._check_n(n) in (1, 9):
        raise ValueError("d1/d9 have no electron repulsion; there is no Tanabe-Sugano diagram")
    geo = _blocks._check_geometry(geometry)
    u = _u.canonical_energy_unit(units)
    cb = _t(c_over_b)
    if not (float(cb) > 0 and np.isfinite(float(cb))):
        raise ValueError("c_over_b must be a positive number")
    x = torch.linspace(0.0, 4.0, 201, dtype=_F64) if dq_over_b is None else torch.atleast_1d(_t(dq_over_b))
    labels, degs, curves = [], [], []
    for blk in blocks_for(n):
        e = block_energies(n, blk, torch.ones_like(x), cb.expand_as(x), x, geo)      # (npts, size)
        for r in range(blk.size):
            labels.append((blk.label, r))
            degs.append(blk.degeneracy)
            curves.append(e[:, r])
    stack = torch.stack(curves)                                                       # (ncurves, npts)
    idx = stack.detach().argmin(0)
    ground_e = stack.gather(0, idx[None]).squeeze(0)
    ground = tuple(labels[int(i)][0] for i in idx)
    return TanabeSuganoDiagram(int(n), geo, cb, x, tuple(labels), tuple(degs), stack - ground_e, ground,
                               None if B is None else _t(B), u)
