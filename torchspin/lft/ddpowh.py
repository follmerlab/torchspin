"""DDPOWH: powder EPR, FDMR, ENDOR and NMR/NQR spectra at field points, with Telser's level strain.

Torch port of Josh Telser's DDPOWH in its exact mode (via telser-tools, ``telser_tools_diff.ddpowh``),
on a torchspin :class:`~torchspin.SpinSystem`.  The Hamiltonian is torchspin's (``ham``: H = H0 - B n.mu),
plus two optional Telser terms that torchspin's core does not carry yet (``zeeman2`` for the second-order
Zeeman w (mu_B B.S)^2 and ``h0_extra`` for the I.S^3 hyperfine U; :mod:`torchspin.lft.telser2torchspin`
builds both).  Unlike ``pepper`` no resonance field is searched: for every orientation n of the grid and
every field B of the sweep, H(B n) is diagonalized and every selected level pair (i, j) adds::

    y(B) += w(n) P_ij pop_ij F(nu_ij) f(nu_ij(B) - nu_mw; W_ij(n, B))

* f: Telser's DLINE.F line (Gaussian or Lorentzian, HWHM W, cut at ``cutoff`` half widths), evaluated in
  the frequency domain at each field point.  The field-swept line is the frequency line mapped through the
  exact nu_ij(B), so curved and looping levels need no linearization (the 1/abs(d nu/dB) weight is implicit).
* W_ij(n, B) = W(n) (B/B_set)^b (1 + [sqrt(sum_b abs(c_bi)^2 w_b) + sqrt(sum_b abs(c_bj)^2 w_b)]/2) with
  W(n)^2 = sum_k W_k^2 n_k^2.  The square brackets are Telser's **level strain**: w_b is a weight per basis
  state ``|m_S, m_I>`` and c_bi the components of level i, so the width of a transition depends on what its two
  levels are made of.  :func:`strain_weights` builds DDPOWH's w_b = WMS(m_S) + WMI(m_I);
  :func:`torchspin.lft.ddpowj.strain_weights` DDPOWJ's product WMS1(m_1) WMS2(m_2).
* P_ij: the B1-averaged abs(<i|mu|j>)^2 / mu_B^2 (perpendicular mode; parallel for ``h1mode < 0``;
  perpendicular + h1mode x parallel for ``h1mode > 0``).  With ``moment='electron'`` (default, Telser's) mu
  is the electron Zeeman moment only (g.S: no nuclear moment, no third-order Zeeman u); with
  ``moment='full'`` it is torchspin's -dH/dB moment.
* pop_ij = p_i - p_j: Boltzmann populations at ``temperature`` (normalized; None: no populations).
* F(nu) = nu (Gaussian) or nu^3/nu_mw^2 (Lorentzian), nu in cm^-1: the Flygare frequency factor
  (``frequency_factor=False`` removes it, as in EasySpin/pepper).
* w(n): the grid weight times the partial-orientation factor exp(lam_x cos^2 phi + lam_z cos^2 theta)
  (``ordering=(lam_x, lam_z)``).

Functions: :func:`epr` (field sweep), :func:`fdmr` (frequency sweep at one field, zero field included),
:func:`endor` (rf sweep at one field, orientation selection by the EPR line) and :func:`nmr` (S = 0
NMR/NQR, Telser's DDPOWH NMR mode, with the chemical shielding sigma_k in kHz).

Units
-----
Fields in mT, microwave frequency in GHz, temperature in K, rf axes in MHz.  ``width``, ``nmr_width`` and
the FDMR axis are in ``units`` (default ``'MHz'``; ``'cm^-1'``, ``'GHz'``, ... via :mod:`torchspin.lft.units`).
Intensities are DDPOWH's exact-mode values (P as above, the line in 1/MHz, F in cm^-1), except that
field-derivative spectra are d/dB per mT (10x DDPOWH's per-gauss values).  ``constants`` (a
:class:`SpinConstants`) sets mu_B (P normalization), k_B and cm^-1 -> MHz; parity tests pass telser-tools'
CODATA 2018 values through :meth:`SpinConstants.from_telser`.

Example (an S = 3/2 system with D = 6000 MHz whose +-3/2 levels are broadened by level strain):

>>> import torch
>>> from torchspin import SpinSystem
>>> from torchspin.lft import ddpowh
>>> sys = SpinSystem(S=1.5, g=[2.0, 2.0, 2.0], D=[-2000., -2000., 4000.])   # MHz
>>> w = ddpowh.strain_weights(sys, ms=[1.0, 0.0, 0.0, 1.0])
>>> res = ddpowh.epr(sys, torch.linspace(50, 450, 201), 9.5, width=60.0,
...                  strain=ddpowh.Strain(w), igloo=6)
>>> res.spectrum.shape, res.n_orientations
(torch.Size([201]), 98)

Gradients
---------
Every tensor parameter of the SpinSystem that ``ham`` keeps on the graph (g, D, A, Q principal values,
full tensors, ``ee``, the Euler frames ...), the widths, the strain weights, b, B_set,
the microwave frequency, the temperature, ``ordering``, ``zeeman2``, ``h0_extra``, the shielding and
``scale`` carry gradients.  Eigenvalues and eigenvectors come from ``torchspin.lft._linalg.eigh_gauge``.
Discrete choices are made on detached values: which (orientation, field) points carry a line (the cutoff),
the window of each FDMR/ENDOR line and the ENDOR selection sign test.

Strain inside a degenerate level (e.g. zero field with isotropic exchange) uses per-state weights, which
are basis dependent there, as in Telser's program.

Not ported (Fortran reproduction, see telser-tools docs/FINDINGS.md): EPRTRANS/NMRTRANS probabilities, the
Fortran thresholds, the IEND/JEND strain bug (DDPOWH-06), the sqrt(ICUTOFF) cut is available as
``exact_cutoff=False``, the original phi windows as ``grid='fortran'``, d/dnu derivatives as
``field_derivative=False`` and unnormalized populations as ``normalized_populations=False``.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import torch

from torchspin.constants import BMAGN, BOLTZMANN, PLANCK
from torchspin.spinsystem import SpinSystem

from . import _powder
from . import units as _u
from ._linalg import eigh_gauge

__all__ = ["SpinConstants", "TORCHSPIN_CONSTANTS", "Strain", "PowderResult", "strain_weights",
           "manifold_pairs", "all_pairs", "epr", "fdmr", "endor", "nmr", "simulate"]

_F64, _C128 = torch.float64, torch.complex128
#: degrees -> radians for ``grid='fortran'``: pi/180, as telser-tools' exact mode (the Fortran's 1.745329252e-2
#: can add or drop the ring point at theta = 180 degrees).
_DEG = math.pi / 180.0


class SpinConstants(NamedTuple):
    """Physical constants used by the DDPOWH/DDPOWJ simulations.

    Attributes
    ----------
    mu_b : float
        Bohr magneton over Planck's constant, mu_B/h, in MHz per mT
        (normalizes the transition probabilities P).
    k_b : float
        Boltzmann constant over Planck's constant, k_B/h, in MHz per K.
    mhz_per_cm : float
        MHz per cm^-1 (energy conversion of widths, axes and the Flygare
        frequency factor).

    See Also
    --------
    TORCHSPIN_CONSTANTS : torchspin's CODATA values.
    """

    mu_b: float
    k_b: float
    mhz_per_cm: float

    @classmethod
    def from_telser(cls, c) -> "SpinConstants":
        """Build from a telser-tools ``Constants``-like object.

        Parameters
        ----------
        c : object
            Any object with attributes ``mu_b_mhz`` (MHz per gauss),
            ``mu_b_cm`` (cm^-1 per gauss) and ``k_b_cm`` (cm^-1 per K), e.g.
            telser-tools' CODATA 2018 constants (used by the parity tests).

        Returns
        -------
        SpinConstants
            ``mu_b = 10 * mu_b_mhz`` (MHz/mT), ``mhz_per_cm = mu_b_mhz /
            mu_b_cm`` and ``k_b = k_b_cm * mhz_per_cm``.
        """
        mhz_per_cm = float(c.mu_b_mhz) / float(c.mu_b_cm)
        return cls(float(c.mu_b_mhz) * 10.0, float(c.k_b_cm) * mhz_per_cm, mhz_per_cm)


#: Derived from ``torchspin.constants`` (CODATA 2022 mu_B).
TORCHSPIN_CONSTANTS = SpinConstants(BMAGN / PLANCK * 1e-9, BOLTZMANN / PLANCK * 1e-6, _u.energy_factor("cm^-1", "MHz"))


class Strain(NamedTuple):
    """Telser's level strain of a DDPOWH/DDPOWJ EPR line.

    The half width of the line between levels i and j at field B is
    ``W(n) (B/B_set)^b (1 + [sqrt(sum_b abs(c_bi)^2 w_b) + sqrt(sum_b
    abs(c_bj)^2 w_b)]/2)``, where ``c_bi`` are the components of level i in
    the basis state b and ``w_b`` are the basis weights.

    Attributes
    ----------
    weights : torch.Tensor, shape (nStates,)
        One weight per basis state of the SpinSystem (``nStates = sys.nStates``,
        states ``|m_S1, m_S2, ..., m_I1, ...>`` with every m descending,
        torchspin's order), dimensionless.  Built by :func:`strain_weights`
        (DDPOWH) or :func:`torchspin.lft.ddpowj.strain_weights` (DDPOWJ).
        May require grad.
    b : float or torch.Tensor, default 0.0
        Field exponent of the width.
    field : float or torch.Tensor, default 1.0
        Reference field B_set in mT; only used when ``b != 0``.
    """

    weights: torch.Tensor
    b: float | torch.Tensor = 0.0
    field: float | torch.Tensor = 1.0


class PowderResult(NamedTuple):
    """Result of :func:`epr`, :func:`fdmr`, :func:`endor` and :func:`nmr`.

    Attributes
    ----------
    axis : torch.Tensor, shape (npts,)
        The sweep axis as used: fields in mT (:func:`epr`), energies in
        ``units`` (:func:`fdmr`) or radio frequencies in MHz (:func:`endor`,
        :func:`nmr`); detached, ``float64``.
    spectrum : torch.Tensor, shape (npts,)
        The spectrum on ``axis`` (DDPOWH's exact-mode intensity scale, see
        the module docstring), on the autograd graph of differentiable
        inputs.
    n_orientations : int
        Number of grid orientations.
    grid : Grid
        The orientation grid (``theta``, ``phi`` in radians, ``weight``,
        unit ``directions`` (n_orientations, 3), ...).
    """

    axis: torch.Tensor
    spectrum: torch.Tensor
    n_orientations: int
    grid: _powder.Grid


# ----------------------------------------------------------------------------- helpers
def _t(x, device="cpu") -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x if x.dtype == _F64 else x.to(_F64)
    return torch.as_tensor(np.asarray(x, float), dtype=_F64, device=device)


def _det(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x


def _safe_sqrt(x: torch.Tensor) -> torch.Tensor:
    """sqrt with a zero (not infinite) derivative at 0."""
    pos = x > 0
    return torch.where(pos, torch.sqrt(torch.where(pos, x, torch.ones_like(x))), torch.zeros_like(x))


def _shape(shape: str) -> str:
    s = str(shape).strip().lower()
    if s in ("g", "gauss", "gaussian"):
        return "G"
    if s in ("l", "lorentz", "lorentzian"):
        return "L"
    raise ValueError("shape must be 'gaussian' or 'lorentzian'")


def _mixed(perp, par, h1mode: float):
    if h1mode == 0:
        return perp
    if h1mode < 0:
        return par
    return perp + h1mode * par


def _spins(sys: SpinSystem) -> tuple[list[float], int]:
    spins = [float(s) for s in sys.Spins]
    return spins, int(sys.nElectrons)


def _ms_values(s: float) -> np.ndarray:
    return s - np.arange(int(round(2 * s)) + 1, dtype=float)


def strain_weights(sys: SpinSystem, ms=None, mi=None, *, electron: int = 0, nucleus: int = 0) -> torch.Tensor:
    """DDPOWH's basis-state strain weights ``w_b = WMS(m_S) + WMI(m_I)``.

    Parameters
    ----------
    sys : SpinSystem
        The spin system; one weight is returned per basis state.
    ms : array_like or torch.Tensor, shape (2S+1,), optional
        WMS for ``m_S = S, S-1, ..., -S`` of electron ``electron`` (Telser's
        input order), dimensionless.  ``None``: zeros.
    mi : array_like or torch.Tensor, shape (2I+1,), optional
        WMI for ``m_I = I, ..., -I`` of nucleus ``nucleus``.  ``None``: zeros.
    electron : int, default 0
        0-based electron index for ``ms``.
    nucleus : int, default 0
        0-based nucleus index for ``mi``.

    Returns
    -------
    torch.Tensor, shape (nStates,)
        The weights in torchspin's basis order, ``float64``; tensor inputs
        keep their graph.

    Raises
    ------
    ValueError
        If ``ms``/``mi`` does not have ``2S+1``/``2I+1`` values, or if the
        electron or nucleus does not exist and more than one value is given
        (a single value for an absent spin, Telser's WMI(1) without a
        nucleus, is added to every state).

    See Also
    --------
    Strain, torchspin.lft.ddpowj.strain_weights

    Examples
    --------
    >>> from torchspin import SpinSystem
    >>> from torchspin.lft import ddpowh
    >>> sys = SpinSystem(S=0.5, Nucs='14N', A=[20.0, 20.0, 95.0])
    >>> ddpowh.strain_weights(sys, ms=[1.0, 0.0], mi=[0.5, 0.0, 0.5]).tolist()
    [1.5, 1.0, 1.5, 0.5, 0.0, 0.5]
    """
    spins, ne = _spins(sys)
    dims = [int(round(2 * s)) + 1 for s in spins]
    n = int(np.prod(dims)) if dims else 1
    w = torch.zeros(n, dtype=_F64)

    def lift(vec, k):
        v = _t(vec)
        if v.shape != (dims[k],):
            raise ValueError(f"spin {k}: need {dims[k]} weights (m = S ... -S), got {tuple(v.shape)}")
        shape = [1] * len(dims)
        shape[k] = dims[k]
        return v.reshape(shape).expand(dims).reshape(-1)

    def constant(vec, what):
        v = _t(vec).reshape(-1)
        if v.shape != (1,):
            raise ValueError(f"no such {what}: only one weight (a constant, Telser's 2S+1 = 1 or 2I+1 = 1) is allowed")
        return v.expand(n)

    if ms is not None:
        w = w + (lift(ms, electron) if 0 <= electron < ne else constant(ms, "electron spin"))
    if mi is not None:
        k = ne + nucleus
        w = w + (lift(mi, k) if 0 <= nucleus and k < len(spins) else constant(mi, "nucleus"))
    return w


def all_pairs(n: int) -> tuple[tuple[int, int], ...]:
    """Every level pair ``(i, j)``, ``i < j``, of ``n`` levels.

    Parameters
    ----------
    n : int
        Number of levels.

    Returns
    -------
    tuple of tuple of int
        0-based pairs of levels numbered by ascending energy.

    Examples
    --------
    >>> from torchspin.lft import ddpowh
    >>> ddpowh.all_pairs(3)
    ((0, 1), (0, 2), (1, 2))
    """
    return tuple((i, j) for i in range(n) for j in range(i + 1, n))


def manifold_pairs(ms_range, manifold_size: int) -> tuple[tuple[int, int], ...]:
    """DDPOWH's level-range pair selection.

    The levels, sorted by energy, are grouped in consecutive manifolds of
    ``manifold_size`` levels (Telser: 2I + 1, the m_S manifolds at high
    field).  Level i runs over manifolds a to b, level j over c to d, and
    only ``i < j`` is kept.

    Parameters
    ----------
    ms_range : sequence of 4 int
        ``(a, b, c, d)``, 1-based manifold numbers; ``(1, 1, 2, 2)`` selects
        the transitions from the lowest manifold to the next.
    manifold_size : int
        Levels per manifold.

    Returns
    -------
    tuple of tuple of int
        0-based level pairs ``(i, j)``.  (DDPOWJ's default, all pairs, is
        :func:`all_pairs`.)

    Examples
    --------
    >>> from torchspin.lft import ddpowh
    >>> ddpowh.manifold_pairs((1, 1, 2, 2), 2)
    ((0, 2), (0, 3), (1, 2), (1, 3))
    """
    a, b, c, d = (int(x) for x in ms_range)
    ni = int(manifold_size)
    return tuple((i - 1, j - 1) for i in range((a - 1) * ni + 1, b * ni + 1)
                 for j in range((c - 1) * ni + 1, d * ni + 1) if i < j)


def _nmr_pairs(lo: int, hi: int, ni: int):
    """DDPOWH's ENDOR pair set: every i < j inside manifolds lo ... hi (ll. 191-194 of the NumPy port)."""
    return tuple((i - 1, j - 1) for i in range((lo - 1) * ni + 1, hi * ni)
                 for j in range((lo - 1) * ni + 2, hi * ni + 1) if i < j)


# ----------------------------------------------------------------------------- the model
class _Model:
    """H(B n) = H0 + h0_extra - B n.mu + B^2 n.Z2.n of a SpinSystem, and the transition operators."""

    def __init__(self, sys: SpinSystem, *, zeeman2, h0_extra, moment: str, constants: SpinConstants, device):
        from torchspin.ham import ham
        from torchspin.ham_ez import ham_ez

        self.sys = sys
        self.c = constants
        H0, mx, my, mz = ham(sys, device=str(device))
        self.dev = H0.device
        self.n = H0.shape[-1]
        h0 = H0.to(_C128)
        if h0_extra is not None:
            x = _as(h0_extra)
            h0 = h0 + (x.to(self.dev, _C128) if torch.is_complex(x) else _t(x, self.dev).to(_C128))
        self.h0 = 0.5 * (h0 + h0.conj().transpose(-2, -1))
        self.mu = torch.stack([mx, my, mz]).to(_C128)                                  # MHz/mT, H = H0 - B n.mu
        self.z2 = None
        if zeeman2 is not None:
            x = _as(zeeman2)
            z2 = x.to(self.dev, _C128) if torch.is_complex(x) else _t(x, self.dev).to(_C128)
            if z2.shape != (3, 3, self.n, self.n):
                raise ValueError(f"zeeman2 must be (3, 3, {self.n}, {self.n}) (MHz/mT^2)")
            self.z2 = z2
        spins, ne = _spins(sys)
        self.spins, self.ne = spins, ne
        if moment not in ("electron", "full"):
            raise ValueError("moment must be 'electron' (g.S, Telser) or 'full' (-dH/dB, torchspin)")
        if moment == "full" or ne == 0:
            self.mu_tr = self.mu
        else:
            ex, ey, ez = ham_ez(sys, B0=None, device=str(device))
            self.mu_tr = torch.stack([ex, ey, ez]).to(_C128)

    def field_ops(self, d: torch.Tensor):
        """(-n.mu (c, n, n), n.Z2.n (c, n, n) or None) for unit directions d (c, 3)."""
        dc = d.to(_C128)
        m1 = -torch.einsum("ck,kij->cij", dc, self.mu)
        m2 = None if self.z2 is None else torch.einsum("ck,cl,klij->cij", dc, dc, self.z2)
        return m1, m2

    def hamiltonians(self, m1, m2, fields: torch.Tensor) -> torch.Tensor:
        """(c, nf, n, n) for field ops (c, n, n) and fields (nf,) in mT."""
        b = fields.to(_C128)[None, :, None, None]
        H = self.h0 + b * m1[:, None]
        if m2 is not None:
            H = H + b * b * m2[:, None]
        return H

    def nuclear_spin(self, nucleus: int) -> torch.Tensor:
        from torchspin.spinops import sop

        k = self.ne + int(nucleus)
        if not self.ne <= k < len(self.spins):
            raise ValueError("no such nucleus")
        return torch.stack([sop(self.spins, [k + 1, c + 1]) for c in range(3)]).to(_C128).to(self.dev)


def _as(x):
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x))


def _probabilities(v: torch.Tensor, op: torch.Tensor, d: torch.Tensor, h1mode: float, pi, pj) -> torch.Tensor:
    """B1-averaged |<i|op|j>|^2 for vectors (s, n, n), op (3, n, n), directions (s, 3): (s, P)."""
    vi, vj = v[..., pi], v[..., pj]
    el = torch.einsum("sap,kab,sbp->skp", vi.conj(), op, vj)
    tot = (el.real ** 2 + el.imag ** 2).sum(-2)
    a = torch.einsum("sk,skp->sp", d.to(_C128), el)
    par = a.real ** 2 + a.imag ** 2
    return _mixed(0.5 * (tot - par), par, h1mode)


class _Opts(NamedTuple):
    shape: str
    cutoff: int
    exact_cutoff: bool
    frequency_factor: bool
    normalized_populations: bool
    h1mode: float


def _line(o: _Opts, shape: str, deriv: bool, width, x):
    return _powder.line(shape, bool(deriv), o.cutoff, width, x, exact=o.exact_cutoff)


def _populations(e: torch.Tensor, bkt, normalized: bool, pi, pj):
    if bkt is None:
        return None
    p = torch.exp(-(e - e[..., :1]) / bkt)
    if normalized:
        p = p / p.sum(-1, keepdim=True)
    return p[..., pi] - p[..., pj]


def _level_strain(v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """sqrt(sum_b |c_bk|^2 w_b) for every level k: (..., n)."""
    return _safe_sqrt(torch.einsum("...bk,b->...k", v.real ** 2 + v.imag ** 2, w.to(v.real.dtype)))


def _pairs(pairs, n: int, dev):
    pr = all_pairs(n) if pairs is None else tuple((int(i), int(j)) for i, j in pairs)
    for i, j in pr:
        if not 0 <= i < j < n:
            raise ValueError(f"pair {(i, j)}: need 0 <= i < j < {n} (0-based level indices, ascending energy)")
    return (torch.tensor([i for i, _ in pr], dtype=torch.long, device=dev),
            torch.tensor([j for _, j in pr], dtype=torch.long, device=dev))


def _grid(igloo: int, extent: str, angle_range, degrees: bool, grid: str, dev) -> _powder.Grid:
    if extent not in ("octant", "hemisphere"):
        raise ValueError("extent must be 'octant' or 'hemisphere'")
    if grid not in ("exact", "fortran"):
        raise ValueError("grid must be 'exact' or 'fortran'")
    n_rot = 1 if extent == "octant" else 2
    rng = None
    if angle_range is not None:
        rng = tuple(float(a) for a in _det(angle_range))
        if len(rng) != 4:
            raise ValueError("angle_range is (theta_min, theta_max, phi_min, phi_max)")
        if not degrees:
            rng = tuple(math.degrees(a) for a in rng)
    return _powder.make_grid(int(igloo), n_rot, rng, exact=(grid == "exact"), deg=_DEG, device=dev)


def _orientation_weight(g: _powder.Grid, ordering) -> torch.Tensor:
    if ordering is None:
        return g.weight
    lx, lz = (_t(x, g.weight.device) for x in ordering)
    return g.weight * torch.exp(lx * g.cos_p ** 2 + lz * g.cos_t ** 2)


def _width(w: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    return _safe_sqrt(((w * d) ** 2).sum(-1))


def _efac(units: str, c: SpinConstants) -> float:
    """``units`` -> MHz, with ``c``'s cm^-1 (so that parity runs use one set of constants)."""
    return c.mhz_per_cm if _u.canonical_energy_unit(units) == "cm^-1" else _u.energy_factor(units, "MHz")


def _width3(width, units, dev, c: SpinConstants = None) -> torch.Tensor:
    w = _t(width, dev) * (_efac(units, c) if c is not None else _u.energy_factor(units, "MHz"))
    if w.shape == ():
        w = w.expand(3)
    if w.shape != (3,):
        raise ValueError("a width is (Wx, Wy, Wz) HWHM or one isotropic value")
    return w


def _chunks(n_items: int, per_item: int, budget: int):
    step = max(1, budget // max(per_item, 1))
    for a in range(0, n_items, step):
        yield slice(a, min(a + step, n_items))


def _bkt(temperature, c: SpinConstants, dev):
    return None if temperature is None else c.k_b * _t(temperature, dev)


def _opts(shape, cutoff, exact_cutoff, frequency_factor, normalized_populations, h1mode) -> _Opts:
    return _Opts(_shape(shape), int(cutoff), bool(exact_cutoff), bool(frequency_factor),
                 bool(normalized_populations), float(h1mode))


# ----------------------------------------------------------------------------- EPR field sweep
def epr(sys: SpinSystem, fields, mw_freq, *, width, shape: str = "gaussian", derivative: bool = False,
        cutoff: int = 8, temperature=None, h1mode: float = 0.0, pairs=None, strain: Strain | None = None,
        frequency_factor: bool = True, igloo: int = 10, extent: str = "hemisphere", angle_range=None,
        degrees: bool = False, ordering=None, grid: str = "exact", field_derivative: bool = True,
        normalized_populations: bool = True, exact_cutoff: bool = True, moment: str = "electron",
        zeeman2=None, h0_extra=None, scale=1.0, units: str = "MHz",
        constants: SpinConstants = TORCHSPIN_CONSTANTS, device="cpu",
        chunk_budget: int = 4_000_000) -> PowderResult:
    """Field-swept powder (or partial-orientation) EPR spectrum at the field points ``fields``.

    For every grid orientation and every field point the Hamiltonian is
    diagonalized and every selected level pair adds a line evaluated in the
    frequency domain (see the module docstring for the intensity formula).

    Parameters
    ----------
    sys : SpinSystem
        Spin system; its Hamiltonian is built by :func:`torchspin.ham`.
    fields : array_like or torch.Tensor, shape (npts,)
        Field points in mT (not differentiated).  Telser's axis is
        ``linspace(B_min, B_max, npts + 1)``.
    mw_freq : float or torch.Tensor
        Microwave frequency in GHz.
    width : float or array_like, shape (3,)
        Line half width at half maximum ``(Wx, Wy, Wz)`` along the molecular
        axes, in ``units`` (frequency domain); a scalar is isotropic.  The
        width for direction n is ``sqrt(sum_k W_k^2 n_k^2)``.
    shape : {'gaussian', 'lorentzian'}, default 'gaussian'
        Line shape (Telser's DLINE.F; ``'g'``/``'l'`` also accepted).
    derivative : bool, default False
        First-derivative spectrum: d/dB per mT (``field_derivative=False``:
        the Fortran's d/dnu).
    cutoff : int, default 8
        Lines are cut at ``cutoff`` half widths.
    temperature : float or torch.Tensor, optional
        Temperature in K for Boltzmann population differences.  ``None``
        (default): no population factor.
    h1mode : float, default 0.0
        Microwave mode: 0 perpendicular, ``< 0`` parallel, ``> 0``
        perpendicular + ``h1mode`` x parallel.
    pairs : sequence of (int, int), optional
        0-based level pairs ``(i, j)``, ``i < j``, levels numbered by
        ascending energy at each (orientation, field).  ``None`` (default):
        all pairs.  :func:`manifold_pairs` builds Telser's m_S ranges.
    strain : Strain, optional
        Telser's level strain; ``None`` (default): width ``W(n)`` only.
    frequency_factor : bool, default True
        Include the Flygare frequency factor (nu for Gaussian, nu^3/nu_mw^2
        for Lorentzian lines, nu in cm^-1); ``False`` as EasySpin/pepper.
    igloo : int, default 10
        Grid rings per 90 degrees of theta.
    extent : {'hemisphere', 'octant'}, default 'hemisphere'
        Powder grid extent: a hemisphere (general) or one octant (D2h
        systems; Telser's NROT = 1).
    angle_range : sequence of 4 float, optional
        ``(theta_min, theta_max, phi_min, phi_max)``: a restricted
        orientation range instead of the powder, in radians (degrees with
        ``degrees=True``).
    degrees : bool, default False
        ``angle_range`` is in degrees.
    ordering : tuple of 2 float or torch.Tensor, optional
        ``(lam_x, lam_z)``: orientation weight ``exp(lam_x cos^2(phi) +
        lam_z cos^2(theta))``, applied whenever given (Telser applies it only
        with ``angle_range``).
    moment : {'electron', 'full'}, default 'electron'
        Transition operator: the electron Zeeman moment ``g.S`` only
        (Telser) or torchspin's full ``-dH/dB`` moment.
    zeeman2 : array_like or torch.Tensor, shape (3, 3, nStates, nStates), optional
        Field-quadratic Zeeman operator in MHz/mT^2: ``H += B^2 n.Z2.n``
        (:func:`torchspin.lft.telser2torchspin.second_order_zeeman_operator`).
    h0_extra : array_like or torch.Tensor, shape (nStates, nStates), optional
        Extra field-independent Hamiltonian in MHz (e.g.
        :func:`torchspin.lft.telser2torchspin.u_term_operator`).
    scale : float or torch.Tensor, default 1.0
        Multiplies the spectrum.
    units : str, default 'MHz'
        Unit of ``width`` (``'MHz'``, ``'GHz'``, ``'cm^-1'``, ... of
        :mod:`torchspin.lft.units`).
    constants : SpinConstants, default TORCHSPIN_CONSTANTS
        mu_B/h, k_B/h and MHz per cm^-1.
    device : str or torch.device, default 'cpu'
        Device of the computation.

    Returns
    -------
    PowderResult
        ``axis`` = ``fields`` (mT), ``spectrum`` of shape ``(npts,)``.

    Other Parameters
    ----------------
    grid : {'exact', 'fortran'}, default 'exact'
        ``'fortran'`` reproduces the original phi windows of the grid.
    field_derivative : bool, default True
        With ``derivative``: d/dB per mT (``False``: d/dnu as the Fortran).
    normalized_populations : bool, default True
        Normalize the Boltzmann populations (``False``: as the Fortran).
    exact_cutoff : bool, default True
        Cut lines exactly at ``cutoff`` half widths (``False``: Telser's
        ``sqrt(ICUTOFF)`` rule).
    chunk_budget : int, default 4000000
        Approximate number of complex matrix elements per batch of
        orientations (memory control).

    Raises
    ------
    ValueError
        For an unknown ``shape``, ``extent``, ``grid`` or ``moment``; a
        ``width`` that is not a scalar or 3 values; ``angle_range`` without
        4 values; a pair outside ``0 <= i < j < n``; strain weights not of
        length ``n``; ``zeeman2`` not of shape ``(3, 3, n, n)``.

    See Also
    --------
    fdmr, endor, nmr, torchspin.lft.ddpowj.epr, torchspin.pepper

    Examples
    --------
    >>> import torch
    >>> from torchspin import SpinSystem
    >>> from torchspin.lft import ddpowh
    >>> sys = SpinSystem(S=0.5, g=[2.0, 2.0, 2.2])
    >>> res = ddpowh.epr(sys, torch.linspace(300, 350, 101), 9.5, width=50.0, igloo=6)
    >>> res.axis.shape, res.spectrum.shape, res.n_orientations
    (torch.Size([101]), torch.Size([101]), 98)
    """
    o = _opts(shape, cutoff, exact_cutoff, frequency_factor, normalized_populations, h1mode)
    m = _Model(sys, zeeman2=zeeman2, h0_extra=h0_extra, moment=moment, constants=constants, device=device)
    dev, c = m.dev, constants
    g = _grid(igloo, extent, angle_range, degrees, grid, dev)
    fnp = np.asarray(_det(fields), float).reshape(-1)
    fields_t = torch.as_tensor(fnp, dtype=_F64, device=dev)
    nf = len(fnp)
    pi, pj = _pairs(pairs, m.n, dev)
    y = torch.zeros(nf, dtype=_F64, device=dev)
    if len(pi) == 0:
        return PowderResult(fields_t, y, g.n_points, g)
    freq_mhz = _t(mw_freq, dev) * 1.0e3
    freq_cm = freq_mhz / c.mhz_per_cm
    fpos = float(freq_cm.detach()) ** 2 > 0
    bkt = _bkt(temperature, c, dev)
    wepr = _width(_width3(width, units, dev, c), g.directions)
    orf = _orientation_weight(g, ordering)
    lorentz = o.shape == "L"
    whstrn = None
    if strain is not None:
        sw = _t(strain.weights, dev)
        if sw.shape != (m.n,):
            raise ValueError(f"strain weights must have one value per basis state ({m.n})")
        whstrn = (fields_t / _t(strain.field, dev)) ** _t(strain.b, dev)
    for sl in _chunks(g.n_points, nf * m.n * m.n, chunk_budget):
        d = g.directions[sl]
        m1, m2 = m.field_ops(d)
        e, V = eigh_gauge(m.hamiltonians(m1, m2, fields_t))
        tmhz = e[..., pj] - e[..., pi]                                              # (c, nf, P) MHz
        if strain is not None:
            lev = _level_strain(V, sw)
            w = wepr[sl, None, None] * whstrn[None, :, None] * (1.0 + 0.5 * (lev[..., pi] + lev[..., pj]))
        else:
            w = wepr[sl, None, None].expand(tmhz.shape)
        wl = _line(o, o.shape, derivative, w, tmhz - freq_mhz)
        with torch.no_grad():
            keep = wl != 0
            ci, fi = torch.nonzero(keep.any(-1), as_tuple=True)
        if len(ci) == 0:
            continue
        dsel = d[ci]
        vs = V[ci, fi]
        P = _probabilities(vs, m.mu_tr, dsel, o.h1mode, pi, pj) / c.mu_b ** 2
        tr = tmhz[ci, fi] / c.mhz_per_cm                                             # cm^-1
        if not o.frequency_factor:
            amp = torch.ones_like(tr)
        elif lorentz:
            amp = tr ** 3 / freq_cm ** 2 if fpos else torch.zeros_like(tr)
        else:
            amp = tr
        kmask = keep[ci, fi].to(_F64)
        if derivative and field_derivative:
            dh = m1[ci]
            if m2 is not None:
                dh = dh + 2.0 * fields_t[fi].to(_C128)[:, None, None] * m2[ci]
            de = torch.einsum("sak,sab,sbk->sk", vs.conj(), dh, vs).real
            slope = de[:, pj] - de[:, pi]                                           # MHz per mT
            if not o.frequency_factor:
                damp = torch.zeros_like(tr)
            elif lorentz:
                damp = 3.0 * tr ** 2 / freq_cm ** 2 if fpos else torch.zeros_like(tr)
            else:
                damp = torch.ones_like(tr)
            f_abs = _line(o, o.shape, False, w[ci, fi], tmhz[ci, fi] - freq_mhz)
            val = P * slope * (amp * wl[ci, fi] + damp / c.mhz_per_cm * f_abs) * kmask
        else:
            val = P * amp * wl[ci, fi] * kmask
        pop = _populations(e[ci, fi], bkt, o.normalized_populations, pi, pj)
        if pop is not None:
            val = val * pop
        gsl = torch.arange(g.n_points, device=dev)[sl][ci]
        y = y.index_add(0, fi, val.sum(-1) * orf[gsl])
    return PowderResult(fields_t, _t(scale, dev) * y, g.n_points, g)


# ----------------------------------------------------------------------------- FDMR
def fdmr(sys: SpinSystem, axis, field, *, width, shape: str = "gaussian", derivative: bool = False, cutoff: int = 8,
         temperature=None, h1mode: float = 0.0, pairs=None, frequency_factor: bool = True, igloo: int = 10,
         extent: str = "hemisphere", angle_range=None, degrees: bool = False, ordering=None, grid: str = "exact",
         normalized_populations: bool = True, exact_cutoff: bool = True, moment: str = "electron", zeeman2=None,
         h0_extra=None, scale=1.0, units: str = "MHz", constants: SpinConstants = TORCHSPIN_CONSTANTS,
         device="cpu") -> PowderResult:
    """Frequency-domain (FDMR) powder spectrum at one field.

    ``field = 0`` is a true zero field: the levels are the same for every
    orientation, and the transition probability is still averaged over B1
    about each grid direction (as DDPOWH does).  There is no strain (Telser's
    FDMR input has none).

    Parameters
    ----------
    sys : SpinSystem
        Spin system.
    axis : array_like or torch.Tensor, shape (npts,)
        Transition energies at which the spectrum is evaluated, in
        ``units`` (not differentiated).
    field : float or torch.Tensor
        Static field in mT (0 allowed).
    width : float or array_like, shape (3,)
        HWHM ``(Wx, Wy, Wz)`` in ``units``, or one isotropic value.
    derivative : bool, default False
        d/dnu spectrum.
    frequency_factor : bool, default True
        Flygare factor: nu (Gaussian) or nu^3/x^2 (Lorentzian), with x the
        axis in cm^-1.
    shape, cutoff, temperature, h1mode, pairs, igloo, extent, angle_range, degrees, ordering, moment, zeeman2, h0_extra, scale, units, constants, device
        As for :func:`epr`.

    Returns
    -------
    PowderResult
        ``axis`` in ``units``, ``spectrum`` of shape ``(npts,)``.

    Other Parameters
    ----------------
    grid, normalized_populations, exact_cutoff
        As for :func:`epr`.

    Raises
    ------
    ValueError
        As for :func:`epr`.

    See Also
    --------
    epr

    Examples
    --------
    Zero-field FDMR of an S = 1 system (D = 3000 MHz, E = 300 MHz):

    >>> import torch
    >>> from torchspin import SpinSystem
    >>> from torchspin.lft import ddpowh
    >>> res = ddpowh.fdmr(SpinSystem(S=1, D=[3000.0, 300.0]), torch.linspace(2000, 4000, 201),
    ...                   0.0, width=50.0, igloo=6)
    >>> round(float(res.axis[res.spectrum.argmax()]))
    3300
    """
    o = _opts(shape, cutoff, exact_cutoff, frequency_factor, normalized_populations, h1mode)
    m = _Model(sys, zeeman2=zeeman2, h0_extra=h0_extra, moment=moment, constants=constants, device=device)
    dev, c = m.dev, constants
    g = _grid(igloo, extent, angle_range, degrees, grid, dev)
    x = _t(axis, dev).detach().reshape(-1)
    xmhz = x * _efac(units, c)
    pi, pj = _pairs(pairs, m.n, dev)
    if len(pi) == 0:
        return PowderResult(x, torch.zeros_like(x), g.n_points, g)
    wepr = _width(_width3(width, units, dev, c), g.directions)
    orf = _orientation_weight(g, ordering)
    d = g.directions
    m1, m2 = m.field_ops(d)
    b = _t(field, dev).reshape(1)
    e, V = eigh_gauge(m.hamiltonians(m1, m2, b)[:, 0])
    tmhz = e[:, pj] - e[:, pi]
    tr = tmhz / c.mhz_per_cm
    P = _probabilities(V, m.mu_tr, d, o.h1mode, pi, pj) / c.mu_b ** 2
    amp = P * ((tr ** 3 if o.shape == "L" else tr) if o.frequency_factor else 1.0)
    pop = _populations(e, _bkt(temperature, c, dev), o.normalized_populations, pi, pj)
    if pop is not None:
        amp = amp * pop
    amp = amp * orf[:, None]
    y = _powder.spectrum(xmhz, tmhz, wepr[:, None].expand(tmhz.shape), amp, o.shape, bool(derivative), o.cutoff,
                         exact=o.exact_cutoff)
    if o.shape == "L" and o.frequency_factor:
        q = (xmhz / c.mhz_per_cm) ** 2
        y = torch.where(q > 0, y / torch.where(q > 0, q, torch.ones_like(q)), torch.zeros_like(y))
    return PowderResult(x, _t(scale, dev) * y, g.n_points, g)


# ----------------------------------------------------------------------------- ENDOR / NMR
def _endor_core(m: _Model, o: _Opts, g: _powder.Grid, rf: torch.Tensor, field, *, nmr_width, nmr_shape, nmr_deriv,
                pairs, epr_pairs, mw_freq, width, epr_shape, epr_deriv, strain, field_strain, temperature,
                endor_shape: str, ordering, nucleus, mu=None, units="MHz", nmr_cutoff=None) -> torch.Tensor:
    dev, c = m.dev, m.c
    pi, pj = pairs
    if len(pi) == 0:
        return torch.zeros(len(rf), dtype=_F64, device=dev)
    wnmr = _width(_width3(nmr_width, units, dev, c), g.directions)
    orf = _orientation_weight(g, ordering)
    d = g.directions
    dc = d.to(_C128)
    mu = m.mu if mu is None else mu
    m1 = -torch.einsum("ck,kij->cij", dc, mu)
    m2 = None if m.z2 is None else torch.einsum("ck,cl,klij->cij", dc, dc, m.z2)
    b = _t(field, dev).reshape(1)
    e, V = eigh_gauge(m.hamiltonians(m1, m2, b)[:, 0])
    if m.ne > 0 and epr_pairs is not None:                                         # ENDOR orientation selection
        ei, ej = epr_pairs
        freq_mhz = _t(mw_freq, dev) * 1.0e3
        wepr = _width(_width3(width, units, dev, c), d)
        tm = e[:, ej] - e[:, ei]
        if strain is not None:
            lev = _level_strain(V, _t(strain.weights, dev))
            fac = (b[0] / _t(strain.field, dev)) ** _t(strain.b, dev) if field_strain else 1.0
            w = wepr[:, None] * fac * (1.0 + 0.5 * (lev[:, ei] + lev[:, ej]))
        else:
            w = wepr[:, None].expand(tm.shape)
        wl = _line(o, _shape(epr_shape), bool(epr_deriv), w, tm - freq_mhz)
        pe = _probabilities(V, m.mu_tr, d, o.h1mode, ei, ej) / c.mu_b ** 2
        sel_w = (pe * wl).sum(1)
    else:
        sel_w = torch.ones(e.shape[0], dtype=_F64, device=dev)
    treng = e[:, pj] - e[:, pi]                                                     # MHz
    Pn = _probabilities(V, m.nuclear_spin(nucleus), d, o.h1mode, pi, pj)
    lorentz_form = endor_shape == "L"
    amp = Pn * ((treng ** 3 if lorentz_form else treng) if o.frequency_factor else 1.0) * (sel_w * orf)[:, None]
    pop = _populations(e, _bkt(temperature, c, dev), o.normalized_populations, pi, pj)
    if pop is not None:
        amp = amp * pop
    live = sel_w.detach() > 0
    amp = torch.where(live[:, None], amp, torch.zeros_like(amp))
    sel = live.nonzero().squeeze(-1)
    nshape = _shape(nmr_shape)
    y = _powder.spectrum(rf, treng[sel], wnmr[sel][:, None].expand(-1, treng.shape[1]), amp[sel], nshape,
                         bool(nmr_deriv), o.cutoff if nmr_cutoff is None else int(nmr_cutoff), exact=o.exact_cutoff)
    if lorentz_form and o.frequency_factor:
        q = rf ** 2
        y = torch.where(q > 0, y / torch.where(q > 0, q, torch.ones_like(q)), torch.zeros_like(y))
    return y


def endor(sys: SpinSystem, rf, field, mw_freq, *, width, nmr_width, shape: str = "gaussian",
          nmr_shape: str = "gaussian", derivative: bool = False, epr_derivative: bool = False, cutoff: int = 8,
          nmr_cutoff: int | None = None,
          temperature=None, h1mode: float = 0.0, pairs=None, epr_pairs=None, nucleus: int = 0,
          strain: Strain | None = None, field_strain: bool = True, intensity_shape: str = "nmr",
          frequency_factor: bool = True, igloo: int = 10, extent: str = "hemisphere", angle_range=None,
          degrees: bool = False, ordering=None, grid: str = "exact", normalized_populations: bool = True,
          exact_cutoff: bool = True, moment: str = "electron", zeeman2=None, h0_extra=None, scale=1.0,
          units: str = "MHz", constants: SpinConstants = TORCHSPIN_CONSTANTS, device="cpu") -> PowderResult:
    """Powder ENDOR spectrum at one field, with orientation selection by the EPR line.

    Every orientation is weighted by its EPR excitation, the sum over
    ``epr_pairs`` of ``P_kl f(nu_kl - nu_mw; W_kl)`` with the EPR ``width``,
    ``shape`` and ``strain``; orientations with a non-positive weight are
    dropped.  Each NMR pair then adds ``P_NMR pop nu f(rf - nu_ij; w(n))``,
    where ``P_NMR`` is the B1-averaged ``abs(<i|I|j>)^2`` of nucleus
    ``nucleus`` (no g_N, no hyperfine enhancement; Telser's NMRTRANS in exact
    form).

    Parameters
    ----------
    sys : SpinSystem
        Spin system with at least one nucleus (without electron spins every
        orientation has weight 1).
    rf : array_like or torch.Tensor, shape (npts,)
        Radio frequencies in MHz (absolute, uniform).
    field : float or torch.Tensor
        Static field in mT.
    mw_freq : float or torch.Tensor
        Microwave frequency in GHz.
    width : float or array_like, shape (3,)
        EPR HWHM ``(Wx, Wy, Wz)`` in ``units`` for the orientation selection.
    nmr_width : float or array_like, shape (3,)
        NMR HWHM ``(wx, wy, wz)`` in ``units``.
    shape : {'gaussian', 'lorentzian'}, default 'gaussian'
        EPR (selection) line shape.
    nmr_shape : {'gaussian', 'lorentzian'}, default 'gaussian'
        NMR line shape.
    derivative : bool, default False
        d/d(rf) spectrum.
    epr_derivative : bool, default False
        Use the derivative EPR line for the selection weights.
    cutoff : int, default 8
        Cut of the EPR selection line in half widths.
    nmr_cutoff : int, optional
        Cut of the NMR lines; default ``cutoff``.
    pairs : sequence of (int, int), optional
        0-based NMR level pairs; ``None``: all.
    epr_pairs : sequence of (int, int), optional
        0-based EPR level pairs for the selection; ``None``: all.
    nucleus : int, default 0
        0-based nucleus whose spin operator drives the NMR transitions.
    strain : Strain, optional
        Level strain of the EPR selection line.
    field_strain : bool, default True
        Include the ``(B/B_set)^b`` factor of ``strain``.
    intensity_shape : {'nmr', 'epr'}, default 'nmr'
        Which line type selects the nu or nu^3/rf^2 Flygare factor
        (``'nmr'``: Telser exact; ``'epr'``: the Fortran's ELNTYP flag).
    temperature, h1mode, frequency_factor, igloo, extent, angle_range, degrees, ordering, moment, zeeman2, h0_extra, scale, units, constants, device
        As for :func:`epr`.

    Returns
    -------
    PowderResult
        ``axis`` = ``rf`` (MHz), ``spectrum`` of shape ``(npts,)``.

    Other Parameters
    ----------------
    grid, normalized_populations, exact_cutoff
        As for :func:`epr`.

    Raises
    ------
    ValueError
        For an unknown ``intensity_shape`` or a nonexistent ``nucleus``, and
        as for :func:`epr`.

    See Also
    --------
    epr, nmr, torchspin.salt

    Examples
    --------
    >>> import torch
    >>> from torchspin import SpinSystem
    >>> from torchspin.lft import ddpowh
    >>> sys = SpinSystem(S=0.5, g=[2.0, 2.0, 2.1], Nucs='1H', A=[5.0, 5.0, 12.0])
    >>> res = ddpowh.endor(sys, torch.linspace(5, 25, 201), 340.0, 9.5, width=30.0,
    ...                    nmr_width=0.2, igloo=6)
    >>> res.spectrum.shape
    torch.Size([201])
    """
    o = _opts(shape, cutoff, exact_cutoff, frequency_factor, normalized_populations, h1mode)
    m = _Model(sys, zeeman2=zeeman2, h0_extra=h0_extra, moment=moment, constants=constants, device=device)
    g = _grid(igloo, extent, angle_range, degrees, grid, m.dev)
    rf_t = _t(rf, m.dev).reshape(-1)
    if intensity_shape not in ("nmr", "epr"):
        raise ValueError("intensity_shape must be 'nmr' or 'epr'")
    endor_shape = _shape(nmr_shape if intensity_shape == "nmr" else shape)
    y = _endor_core(m, o, g, rf_t, field, nmr_width=nmr_width, nmr_shape=nmr_shape, nmr_deriv=derivative,
                    pairs=_pairs(pairs, m.n, m.dev), epr_pairs=_pairs(epr_pairs, m.n, m.dev), mw_freq=mw_freq,
                    width=width, epr_shape=shape, epr_deriv=epr_derivative, strain=strain, field_strain=field_strain,
                    temperature=temperature, endor_shape=endor_shape, ordering=ordering, nucleus=nucleus,
                    units=units, nmr_cutoff=nmr_cutoff)
    return PowderResult(rf_t.detach(), _t(scale, m.dev) * y, g.n_points, g)


def nmr(sys: SpinSystem, rf, field, *, nmr_width, shape: str = "gaussian", derivative: bool = False, cutoff: int = 8,
        temperature=None, h1mode: float = 0.0, pairs=None, shielding_khz=None, frequency_factor: bool = True,
        igloo: int = 10, extent: str = "hemisphere", angle_range=None, degrees: bool = False, ordering=None,
        grid: str = "exact", normalized_populations: bool = True, exact_cutoff: bool = True, h0_extra=None,
        scale=1.0, units: str = "MHz", constants: SpinConstants = TORCHSPIN_CONSTANTS,
        device="cpu") -> PowderResult:
    """Powder NMR/NQR spectrum of a system without electron spins (Telser's DDPOWH NMR mode).

    The nuclear levels (nuclear Zeeman + quadrupole) are computed at every
    grid orientation and each pair adds ``P pop nu f(rf - nu_ij; w(n))`` with
    ``P`` the B1-averaged ``abs(<i|I|j>)^2`` (no g_N).  ``field = 0`` gives the
    NQR spectrum.  Differentiable in Q, the shielding, the widths and the
    temperature.

    Parameters
    ----------
    sys : SpinSystem
        Spin system with ``S=[]`` and at least one nucleus, e.g.
        ``SpinSystem(S=[], Nucs='14N', Q=...)``; only the first nucleus is
        driven.
    rf : array_like or torch.Tensor, shape (npts,)
        Radio frequencies in MHz.
    field : float or torch.Tensor
        Static field in mT (0: NQR).
    nmr_width : float or array_like, shape (3,)
        HWHM ``(wx, wy, wz)`` in ``units``.
    shielding_khz : array_like, shape (3,), optional
        Telser's shielding ``(sigma_x, sigma_y, sigma_z)``: the shift of the
        Larmor frequency in kHz for the field along each molecular axis (a
        positive sigma lowers ``abs(nu_N)``); the nuclear Zeeman moment along
        k is scaled by ``1 - sigma_k/abs(nu_N(B))``.  Ignored at zero field.
    shape, derivative, cutoff, temperature, h1mode, pairs, frequency_factor, igloo, extent, angle_range, degrees, ordering, h0_extra, scale, units, constants, device
        As for :func:`epr` (``shape`` is the NMR line shape).

    Returns
    -------
    PowderResult
        ``axis`` = ``rf`` (MHz), ``spectrum`` of shape ``(npts,)``.

    Other Parameters
    ----------------
    grid, normalized_populations, exact_cutoff
        As for :func:`epr`.

    Raises
    ------
    ValueError
        If the system has electron spins or no nucleus, ``shielding_khz`` is
        not three values, and as for :func:`epr`.

    See Also
    --------
    endor

    Examples
    --------
    14N NQR (e2qQ/h = 3 MHz, eta = 0.2):

    >>> import torch
    >>> from torchspin import SpinSystem
    >>> from torchspin.lft import ddpowh
    >>> sys = SpinSystem(S=[], Nucs='14N', Q=[3.0, 0.2])
    >>> res = ddpowh.nmr(sys, torch.linspace(0.1, 3.0, 300), 0.0, nmr_width=0.02, igloo=6)
    >>> round(float(res.axis[res.spectrum.argmax()]), 1)
    2.4
    """
    if int(sys.nElectrons) != 0:
        raise ValueError("nmr is for S = 0 (SpinSystem(S=[], Nucs=...)); use endor for electron spins")
    if int(sys.nNuclei) < 1:
        raise ValueError("the SpinSystem has no nucleus")
    o = _opts(shape, cutoff, exact_cutoff, frequency_factor, normalized_populations, h1mode)
    m = _Model(sys, zeeman2=None, h0_extra=h0_extra, moment="full", constants=constants, device=device)
    g = _grid(igloo, extent, angle_range, degrees, grid, m.dev)
    rf_t = _t(rf, m.dev).reshape(-1)
    mu = m.mu
    if shielding_khz is not None:
        sig = _t(shielding_khz, m.dev).reshape(-1)
        if sig.shape != (3,):
            raise ValueError("shielding_khz is (sigma_x, sigma_y, sigma_z) in kHz")
        nu_n = _nuclear_larmor(m, field)
        if float(nu_n.detach()) > 0:
            mu = mu * (1.0 - (sig / 1.0e3) / nu_n).to(_C128)[:, None, None]
    y = _endor_core(m, o, g, rf_t, field, nmr_width=nmr_width, nmr_shape=shape, nmr_deriv=derivative,
                    pairs=_pairs(pairs, m.n, m.dev), epr_pairs=None, mw_freq=0.0, width=0.0, epr_shape=shape,
                    epr_deriv=False, strain=None, field_strain=False, temperature=temperature,
                    endor_shape=_shape(shape), ordering=ordering, nucleus=0, mu=mu, units=units)
    return PowderResult(rf_t.detach(), _t(scale, m.dev) * y, g.n_points, g)


def _nuclear_larmor(m: _Model, field) -> torch.Tensor:
    """|nu_N| in MHz of the first nucleus at ``field`` (mT): the norm of its moment's z row times B."""
    I = m.nuclear_spin(0)
    # moment operators are linear in I for S = 0: mu_z = k I_z (k = g_N mu_N / h, MHz/mT)
    k = (torch.einsum("ij,ji->", m.mu[2], I[2]) / torch.einsum("ij,ji->", I[2], I[2])).real
    return (k * _t(field, m.dev)).abs()


# ----------------------------------------------------------------------------- Telser inputs
_FUNCS = {"epr": epr, "fdmr": fdmr, "endor": endor, "nmr": nmr}


def simulate(conversion, sys: SpinSystem | None = None, **overrides) -> tuple[PowderResult, ...]:
    """Every spectrum of a converted Telser DDPOWH/DDPOWJ input.

    Parameters
    ----------
    conversion : TelserConversion
        Result of ``telser2torchspin.ddpowh2sys(inp, target='lft')`` or
        ``ddpowj2sys(..., target='lft')``; its
        ``extras['spectra']`` lists the spectra (function name and keywords).
    sys : SpinSystem, optional
        Replaces the converted SpinSystem (e.g. one rebuilt with tensor
        parameters for gradients).
    **overrides
        Keywords that replace those of every spectrum (``width=``,
        ``strain=``, ``temperature=``, ...).

    Returns
    -------
    tuple of PowderResult
        One result per spectrum, in input order.  The axis Telser writes is
        in ``conversion.extras['spectra'][k]['x']`` (G or T, cm^-1 or THz,
        MHz).

    Raises
    ------
    ValueError
        If ``conversion`` was not made with ``target='lft'``.

    See Also
    --------
    torchspin.lft.telser2torchspin.ddpowh2sys, torchspin.lft.telser2torchspin.ddpowj2sys
    """
    spectra = conversion.extras.get("spectra", ())
    if conversion.extras.get("target") != "lft":
        raise ValueError("convert with target='lft'")
    out = []
    for sp in spectra:
        kw = dict(sp["kwargs"])
        kw.update(overrides)
        out.append(_FUNCS[sp["func"]](conversion.sys if sys is None else sys, **kw))
    return tuple(out)
