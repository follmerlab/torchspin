"""Tanabe-Sugano: cubic d^n levels and diagrams, differentiable (Telser's DTANSUG via telser-tools).

Two independent routes to the levels of d^n (n = 1 ... 9) in an octahedral (Oh) or tetrahedral
(Td) field, both linear in (B, C, Dq) over fixed matrices:

* **Strong-field blocks** (DTANSUG.F; McClure 1959, Griffith 1964 Table A.27): every
  (2S+1)Gamma block in the t2^a e^b basis, with DTANSUG's hole-problem signs (SIGNXF, SIGNEL)
  and the DIAG offset of d6-d8.  The NumPy tables (``torchspin.lft._engines.tanabe_sugano``)
  are probed once at the unit vectors and checked for linearity, so
  ``H = B M_B + C M_C + Dq M_Dq`` with constant matrices.  The Racah A term is left out (add
  n(n-1)/2 A), and for d9 also the constant -56B + 28C, as in DTANSUG.
* **Full determinant basis**: :func:`spectrum`, all C(10, n) states from
  :mod:`torchspin.lft.ddn` (Racah A, B, C plus the Ballhausen cubic field; Td reverses the
  field).

Dq > 0 is the physical sign for both geometries (Oh: e above t2; Td: e below t2).  Blocks are
named by their (2S+1)Gamma label, e.g. ``'4T1'``; ``blocks_for(n)`` lists them and
``get_block(n, label)`` returns one (``Block``, from the NumPy tables).

Units
-----
``B``, ``C``, ``Dq``, ``A`` and every returned energy are in ``units`` (default ``'MHz'``;
``'cm^-1'``, ``'GHz'``, ``'K'``, ``'eV'``).  The block matrices are homogeneous of degree one in
(B, C, Dq), so the strong-field route needs no conversion; ``units`` is still checked and
recorded.  The diagram uses the conventional reduced axes, E/B (relative to the ground state)
against Dq/B at fixed C/B, which carry no unit; pass ``B`` to :func:`tanabe_sugano` (or use
:meth:`TanabeSuganoDiagram.at`) for absolute energies and Dq in ``units``.

Derivatives: energies through ``torchspin.lft._linalg.eigvalsh_gauge`` (a degenerate eigenvalue
gets its level's mean derivative, exact for the symmetry-protected degeneracies of the full
determinant spectrum).  ``weights`` (``abs(c_ij)**2``) depend on the basis inside a degenerate
level and are differentiable only where the state is non-degenerate within its block.  The
order of :func:`all_levels` and the diagram's ground curve are decided on detached values.

Not ported: the DTANSUG text report, the Fortran grid rules (``fortran_grid``: overshooting and
always-ascending ranges, telser-tools findings 4-5) and the Fortran B-loop sign bug (finding 1;
:func:`scan_b` uses the correct sign, as the telser-tools NumPy port does).  Plots are in
:mod:`torchspin.lft.plot`.

Examples
--------
Cr(III), d3, in cm^-1: the 4T2 block lies 10 Dq above 4A2 (the Racah A term and constants are
left out of the blocks).

>>> from torchspin.lft import tanabe_sugano as ts
>>> kw = dict(B=766.0, C=2855.0, Dq=1200.0, units="cm^-1")
>>> d10 = ts.block_energies(3, "4T2", **kw) - ts.block_energies(3, "4A2", **kw)
>>> round(float(d10[0]), 6)
12000.0
>>> [round(e, 1) for e in ts.block_energies(3, "4T1", **kw).tolist()]
[-7402.5, 3112.5]

The d5 diagram at C/B = 4.5: high spin (6A1) changes to low spin (2T2) at Dq/B = 2.84.

>>> d = ts.tanabe_sugano(5, c_over_b=4.5)                     # E/B vs Dq/B
>>> d.e_over_b.shape, d.ground[0], d.ground[-1], [round(x, 2) for x in d.crossover()]
(torch.Size([43, 201]), '6A1', '2T2', [2.84])
>>> d.at(B=800.0, units="cm^-1").energy.shape                # absolute, cm^-1
torch.Size([43, 201])
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
    """Constant matrices of one strong-field block: ``H = B M[0] + C M[1] + Dq M[2]``.

    Parameters
    ----------
    n : int
        Number of d electrons, 1 ... 9.
    label : str or Block
        Block label such as ``'4T1'``.
    geometry : {'Oh', 'Td'}, default 'Oh'
        Coordination geometry.
    device : str or torch.device, default 'cpu'
        Device of the result.

    Returns
    -------
    torch.Tensor, shape (3, nb, nb), float64
        Unit-free; configurations in ``Block.configs`` order.

    Raises
    ------
    ValueError
        If the block does not belong to d^n, or for an unknown geometry or label.
    """
    blk = _block(n, label)
    if blk.nh != _blocks._nh(n):
        raise ValueError(f"block {blk.label} (nh={blk.nh}) does not belong to d{n}")
    return _block_ops(int(n), (blk.nh, blk.first), _blocks._check_geometry(geometry), str(torch.device(device)))


def block_hamiltonian(n: int, label, B, C, Dq, geometry: str = "Oh", units: str = "MHz") -> torch.Tensor:
    """Real symmetric strong-field block of d^n.

    Parameters
    ----------
    n : int
        Number of d electrons, 1 ... 9.
    label : str or Block
        Block label such as ``'4T1'``.
    B, C, Dq : float or torch.Tensor
        Racah parameters and cubic field in ``units``; broadcast against each other.
    geometry : {'Oh', 'Td'}, default 'Oh'
        Coordination geometry.
    units : str, default 'MHz'
        Energy unit of the inputs and the result (checked; the blocks are homogeneous).

    Returns
    -------
    torch.Tensor, shape (..., nb, nb), float64
        In ``units``, without the Racah A term.
    """
    _u.canonical_energy_unit(units)
    b, c, d = torch.broadcast_tensors(_t(B), _t(C), _t(Dq))
    M = block_operators(n, label, geometry, b.device)
    return torch.einsum("...k,kij->...ij", torch.stack([b, c, d], -1).to(_F64), M)


def block_energies(n: int, label, B, C, Dq, geometry: str = "Oh", units: str = "MHz") -> torch.Tensor:
    """Ascending eigenvalues of one strong-field block.

    Arguments as :func:`block_hamiltonian`.  Returns a tensor of shape (..., nb) in ``units``
    (without the Racah A term).
    """
    return eigvalsh_gauge(block_hamiltonian(n, label, B, C, Dq, geometry, units))


def spectrum(n: int, B, C, Dq, geometry: str = "Oh", A=0.0, units: str = "MHz") -> torch.Tensor:
    """All C(10, n) eigenvalues of d^n in a cubic field (determinant basis, :mod:`torchspin.lft.ddn`).

    Parameters
    ----------
    n : int
        Number of d electrons, 1 ... 9.
    B, C, Dq : float or torch.Tensor
        Racah parameters and cubic field in ``units``; ``Dq > 0`` is physical for both
        geometries.
    geometry : {'Oh', 'Td'}, default 'Oh'
        Coordination geometry.
    A : float or torch.Tensor, default 0.0
        Racah A in ``units``.
    units : str, default 'MHz'
        Energy unit of the inputs and the result.

    Returns
    -------
    torch.Tensor, shape (..., C(10, n)), float64
        Ascending, including ``n(n-1)/2 A`` and, unlike the blocks, the full repulsion energy for
        every n.
    """
    geo = _blocks._check_geometry(geometry)
    return _dn_levels(n, A=A, B=B, C=C, V=crystal_field(_t(Dq), geometry=geo), units=units)


# ------------------------------------------------------------------------------ solutions and scans
class BlockSolution(NamedTuple):
    """One block of d^n diagonalized at (B, C, Dq) (:func:`solve_block`).

    Attributes
    ----------
    n : int
        Number of d electrons.
    geometry : str
        ``'Oh'`` or ``'Td'``.
    block : Block
        The block.
    B, C, Dq : torch.Tensor
        The parameters, in ``units``.
    matrix : torch.Tensor, shape (..., nb, nb)
        The block Hamiltonian.
    energies : torch.Tensor, shape (..., nb)
        Ascending eigenvalues.
    zeroth_order : torch.Tensor, shape (..., nb)
        Diagonal of ``matrix`` (strong-field configuration energies).
    vectors : torch.Tensor, shape (..., nb, nb)
        Eigenvectors as columns.
    weights : torch.Tensor, shape (..., nb, nb)
        ``weights[..., i, j] = abs(<config j|state i>)**2``.
    units : str
        Energy unit.
    """

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
    """Diagonalize one block of d^n.

    Arguments as :func:`block_hamiltonian` (batched over B, C, Dq, all in ``units``).  Returns a
    :class:`BlockSolution`.
    """
    geo = _blocks._check_geometry(geometry)
    blk = _block(n, label)
    h = block_hamiltonian(n, blk, B, C, Dq, geo, units)
    w, v = eigh_gauge(h)
    return BlockSolution(int(n), geo, blk, _t(B), _t(C), _t(Dq), h, w, torch.diagonal(h, dim1=-2, dim2=-1), v,
                         (v * v).transpose(-2, -1), _u.canonical_energy_unit(units))


class Scan(NamedTuple):
    """One block along a line of Dq (fixed B, C) or of B (fixed Dq, C/B) (:func:`scan_dq`, :func:`scan_b`).

    Attributes
    ----------
    variable : {'Dq', 'B'}
        The scanned parameter.
    B, C, Dq : torch.Tensor, shape (npts,)
        Parameters at every point, in ``units``.
    energies : torch.Tensor, shape (npts, nb)
        Ascending block energies.
    zeroth_order, weights : torch.Tensor
        As :class:`BlockSolution`.
    n, geometry, block, units
        As :class:`BlockSolution`.
    """

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
        """The scanned values: ``Dq`` or ``B``, shape (npts,)."""
        return self.Dq if self.variable == "Dq" else self.B


def _scan(n, label, Bs, Cs, Dqs, variable, geometry, units):
    s = solve_block(n, label, Bs, Cs, Dqs, geometry, units)
    b, c, d = torch.broadcast_tensors(_t(Bs), _t(Cs), _t(Dqs))
    return Scan(int(n), s.geometry, s.block, variable, b, c, d, s.energies, s.zeroth_order, s.weights, s.units)


def scan_dq(n: int, label, B, C, dq_values, geometry: str = "Oh", units: str = "MHz") -> Scan:
    """One block of d^n over a range of Dq at fixed B and C.

    ``B``, ``C`` and ``dq_values`` (1-D) are in ``units`` and may carry gradients; other
    arguments as :func:`block_hamiltonian`.  Returns a :class:`Scan`.
    """
    dq = torch.atleast_1d(_t(dq_values))
    return _scan(n, label, _t(B).expand_as(dq), _t(C).expand_as(dq), dq, "Dq", geometry, units)


def scan_b(n: int, label, Dq, b_values, c_over_b, geometry: str = "Oh", units: str = "MHz") -> Scan:
    """One block of d^n over a range of Racah B with ``C = (C/B) B`` at fixed Dq.

    ``Dq`` and ``b_values`` (1-D) are in ``units``; ``c_over_b`` is dimensionless.  The cubic-field
    sign is that of the Dq loop (DTANSUG.F's B loop omits SIGNXF: telser-tools finding 1).
    Returns a :class:`Scan`.

    Raises
    ------
    ValueError
        If ``c_over_b <= 0`` (DTANSUG.F silently substitutes 5).
    """
    if not float(c_over_b) > 0:
        raise ValueError("c_over_b must be > 0 (DTANSUG.F silently substitutes 5)")
    b = torch.atleast_1d(_t(b_values))
    return _scan(n, label, b, b * _t(c_over_b), _t(Dq).expand_as(b), "B", geometry, units)


class CubicLevel(NamedTuple):
    """One level of d^n (:func:`all_levels`).

    Attributes
    ----------
    label : str
        (2S+1)Gamma block label.
    energy : torch.Tensor
        Energy in the units asked for.
    degeneracy : int
        (2S+1) dim(Gamma).
    root : int
        0-based index within its block.
    """

    label: str
    energy: torch.Tensor
    degeneracy: int
    root: int


def all_levels(n: int, B, C, Dq, geometry: str = "Oh", units: str = "MHz") -> list[CubicLevel]:
    """Every level of d^n (all blocks), ascending.

    ``B``, ``C``, ``Dq`` are scalars in ``units``; the order is decided on detached energies.
    Returns a list of :class:`CubicLevel` (energies without the Racah A term).
    """
    out = []
    for blk in blocks_for(n):
        e = block_energies(n, blk, B, C, Dq, geometry, units)
        out += [CubicLevel(blk.label, e[r], blk.degeneracy, r) for r in range(blk.size)]
    return sorted(out, key=lambda lv: float(lv.energy.detach()))


# ------------------------------------------------------------------------------ diagram
class TanabeSuganoDiagram(NamedTuple):
    """Tanabe-Sugano diagram data of d^n at fixed C/B (:func:`tanabe_sugano`).

    Attributes
    ----------
    n : int
        Number of d electrons.
    geometry : str
        ``'Oh'`` or ``'Td'``.
    c_over_b : torch.Tensor
        C/B.
    dq_over_b : torch.Tensor, shape (npts,)
        The abscissa Dq/B.
    labels : tuple of (str, int)
        ``labels[i]`` = (block label, root) of curve i, e.g. ``('4T1', 1)``.
    degeneracies : tuple of int
        (2S+1) dim(Gamma) of each curve.
    e_over_b : torch.Tensor, shape (ncurves, npts)
        E/B relative to the ground state at each point.
    ground : tuple of str
        Ground block at each point.
    B : torch.Tensor or None
        Racah B in ``units``; turns on the absolute axes :attr:`energy` and :attr:`dq`.
    units : str
        Unit of ``B``.
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
        """Energies above the ground state in ``units`` (E/B times B), shape (ncurves, npts); None without B."""
        return None if self.B is None else self.e_over_b * self.B[..., None, None]

    @property
    def dq(self) -> torch.Tensor | None:
        """Dq in ``units`` (Dq/B times B), shape (npts,); None without B."""
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
        """E/B of one curve, e.g. ``curve('4T2')`` or ``curve('4T1', 1)``.

        Raises ``KeyError`` if there is no such (label, root).
        """
        lab = label.replace(" ", "").upper()
        for i, (l, r) in enumerate(self.labels):
            if l.upper() == lab and r == root:
                return self.e_over_b[i]
        raise KeyError(f"no curve ({label!r}, {root}); curves: {self.labels}")

    def crossover(self) -> list[float]:
        """Dq/B values (list of float, linear interpolation on detached values) where the ground state changes."""
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
    """Tanabe-Sugano diagram: E/B relative to the ground state vs Dq/B at fixed C/B.

    Parameters
    ----------
    n : int
        Number of d electrons, 2 ... 8.
    c_over_b : float or torch.Tensor, default 4.5
        C/B (> 0).
    dq_over_b : array_like or torch.Tensor, optional
        Abscissa Dq/B; default 0 ... 4 in 201 points.
    geometry : {'Oh', 'Td'}, default 'Oh'
        Coordination geometry.
    B : float or torch.Tensor, optional
        Racah B in ``units``; adds the absolute axes ``.energy`` and ``.dq`` (differentiable in B).
    units : str, default 'MHz'
        Unit of ``B``.

    Returns
    -------
    TanabeSuganoDiagram
        E/B depends only on Dq/B and C/B (the blocks are linear in B, C, Dq), so B = 1 internally;
        the result is differentiable in ``c_over_b`` and ``dq_over_b``.  The ground curve at each
        point is chosen on detached values; E/B then follows that curve.

    Raises
    ------
    ValueError
        For n = 1 or 9 (no electron repulsion, so no diagram), an invalid n or geometry, or a
        non-positive or non-finite ``c_over_b``.
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
