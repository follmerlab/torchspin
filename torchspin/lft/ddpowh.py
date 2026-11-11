"""DDPOWH: powder EPR, FDMR, ENDOR and NMR/NQR spectra at field points, with Telser's level strain.

Torch port of Joshua Telser's DDPOWH in its exact mode (via telser-tools, ``telser_tools_diff.ddpowh``),
on a torchspin :class:`~torchspin.SpinSystem`.  The Hamiltonian is torchspin's (``ham``: H = H0 - B n.mu),
plus two optional Telser terms that torchspin's core does not carry yet (``zeeman2`` for the second-order
Zeeman w (mu_B B.S)^2 and ``h0_extra`` for the I.S^3 hyperfine U; :mod:`torchspin.lft.telser2torchspin`
builds both).  Unlike ``pepper`` no resonance field is searched: for every orientation n of the grid and
every field B of the sweep, H(B n) is diagonalized and every selected level pair (i, j) adds::

    y(B) += w(n) P_ij pop_ij F(nu_ij) f(nu_ij(B) - nu_mw; W_ij(n, B))

* f: Telser's DLINE.F line (Gaussian or Lorentzian, HWHM W, cut at ``cutoff`` half widths), evaluated in
  the frequency domain at each field point.  The field-swept line is the frequency line mapped through the
  exact nu_ij(B), so curved and looping levels need no linearization (the 1/|d nu/dB| weight is implicit).
* W_ij(n, B) = W(n) (B/B_set)^b (1 + [sqrt(sum_b |c_bi|^2 w_b) + sqrt(sum_b |c_bj|^2 w_b)]/2) with
  W(n)^2 = sum_k W_k^2 n_k^2.  The square brackets are Telser's **level strain**: w_b is a weight per basis
  state |m_S, m_I> and c_bi the components of level i, so the width of a transition depends on what its two
  levels are made of.  :func:`strain_weights` builds DDPOWH's w_b = WMS(m_S) + WMI(m_I);
  :func:`torchspin.lft.ddpowj.strain_weights` DDPOWJ's product WMS1(m_1) WMS2(m_2).
* P_ij: the B1-averaged |<i|mu|j>|^2 / mu_B^2 (perpendicular mode; parallel for ``h1mode < 0``;
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

    >>> import torch
    >>> from torchspin import SpinSystem
    >>> from torchspin.lft import ddpowh
    >>> sys = SpinSystem(S=1.5, g=[2.0, 2.0, 2.0], D=[-2000., -2000., 4000.])          # MHz
    >>> w = ddpowh.strain_weights(sys, ms=[1.0, 0.0, 0.0, 1.0])                         # broaden the +-3/2 levels
    >>> res = ddpowh.epr(sys, torch.linspace(50, 450, 401), 9.5, width=[60., 60., 60.],
    ...                  strain=ddpowh.Strain(w), igloo=20)
    >>> res.spectrum.shape                                                              # (401,)

Gradients
---------
Every tensor parameter of the SpinSystem that ``ham`` keeps on the graph (g, D, A, Q principal values,
full tensors, ``ee`` ...; not the Euler frames, mol2g bug B3), the widths, the strain weights, b, B_set,
the microwave frequency, the temperature, ``ordering``, ``zeeman2``, ``h0_extra``, the shielding and
``scale`` carry gradients.  Eigenvalues and eigenvectors come from :func:`torchspin.lft._linalg.eigh_gauge`.
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
    """mu_B/h in MHz per mT, k_B/h in MHz per K, and MHz per cm^-1."""

    mu_b: float
    k_b: float
    mhz_per_cm: float

    @classmethod
    def from_telser(cls, c) -> "SpinConstants":
        """From a telser-tools ``Constants``-like object (``mu_b_mhz`` MHz/G, ``mu_b_cm`` cm^-1/G, ``k_b_cm``)."""
        mhz_per_cm = float(c.mu_b_mhz) / float(c.mu_b_cm)
        return cls(float(c.mu_b_mhz) * 10.0, float(c.k_b_cm) * mhz_per_cm, mhz_per_cm)


#: Derived from :mod:`torchspin.constants` (CODATA 2022 mu_B).
TORCHSPIN_CONSTANTS = SpinConstants(BMAGN / PLANCK * 1e-9, BOLTZMANN / PLANCK * 1e-6, _u.energy_factor("cm^-1", "MHz"))


class Strain(NamedTuple):
    """Telser's strain width factor (B/B_set)^b (1 + [sqrt(sum |c_bi|^2 w_b) + sqrt(sum |c_bj|^2 w_b)]/2).

    ``weights`` (n,): one weight per basis state of the SpinSystem (|m_S1, m_S2, ..., m_I1, ...>, every m
    descending, torchspin's order); see :func:`strain_weights`.  ``b``: the field exponent; ``field``:
    B_set in mT (only used when b != 0).
    """

    weights: torch.Tensor
    b: float | torch.Tensor = 0.0
    field: float | torch.Tensor = 1.0


class PowderResult(NamedTuple):
    """``axis``: fields (mT, :func:`epr`), energies (``units``, :func:`fdmr`) or rf (MHz, :func:`endor`,
    :func:`nmr`); ``spectrum`` (len(axis),); ``grid``: the orientations (radians, weights)."""

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
    """DDPOWH's basis weights w_b = WMS(m_S) + WMI(m_I) for every basis state of ``sys``.

    ``ms`` (2S+1,): WMS for m_S = S, S-1, ..., -S of electron ``electron`` (Telser's input order);
    ``mi`` (2I+1,): WMI for m_I = I, ..., -I of nucleus ``nucleus``.  Missing vectors count as zeros; a
    single value for a spin that is absent (Telser's WMI(1) without a nucleus) is added to every state.
    Tensors keep their graph.
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
    """Every pair (i, j), i < j, of n levels (0-based, ascending energy)."""
    return tuple((i, j) for i in range(n) for j in range(i + 1, n))


def manifold_pairs(ms_range, manifold_size: int) -> tuple[tuple[int, int], ...]:
    """DDPOWH's level-range pair selection.

    Levels are sorted by energy and grouped in consecutive manifolds of ``manifold_size`` levels (Telser:
    2I + 1, the "m_S manifolds" at high field).  ``ms_range`` = (a, b, c, d), 1-based manifold numbers:
    i runs over manifolds a ... b, j over c ... d, and only i < j is kept.  (1, 1, 2, 2) is the lowest
    manifold to the next; DDPOWJ's default of all pairs is ``all_pairs(n)``.
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
    """Field-swept powder (or partial-orientation) EPR spectrum of ``sys`` at the field points ``fields``.

    ``fields`` (mT, 1-D, not differentiated; Telser's axis is ``linspace(B_min, B_max, npts + 1)``),
    ``mw_freq`` (GHz).  ``width`` (Wx, Wy, Wz) HWHM in ``units`` (frequency domain), or one value.
    ``shape``: ``'gaussian'``/``'lorentzian'``; ``derivative``: d/dB (per mT; ``field_derivative=False``:
    the Fortran's d/dnu).  ``cutoff``: lines cut at ``cutoff`` half widths.  ``temperature`` (K): Boltzmann
    population differences.  ``h1mode``: 0 perpendicular, < 0 parallel, > 0 perpendicular + h1mode x
    parallel.  ``pairs``: 0-based level pairs (i, j), i < j (None: all; :func:`manifold_pairs` for
    Telser's m_S ranges).  ``strain``: :class:`Strain` (None: W(n) only).  ``igloo``: grid rings per 90
    degrees; ``extent``: ``'hemisphere'`` (general) or ``'octant'`` (D2h systems; Telser's NROT = 1);
    ``angle_range`` (theta_min, theta_max, phi_min, phi_max): a restricted orientation range instead of the
    powder (radians unless ``degrees``); ``ordering`` (lam_x, lam_z): weight exp(lam_x cos^2 phi + lam_z
    cos^2 theta) (Telser applies it only with ``angle_range``).  ``moment``, ``zeeman2`` (3, 3, n, n) MHz/mT^2,
    ``h0_extra`` (n, n) MHz, ``constants``: module docstring.  ``scale`` multiplies the spectrum.
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
    """Frequency-domain (FDMR) powder spectrum on ``axis`` (energies in ``units``) at one ``field`` (mT).

    ``field = 0`` is a true zero field: the levels are the same for every orientation, and P is still
    B1-averaged about each grid direction (as DDPOWH does).  ``derivative``: d/dnu.  The Lorentzian Flygare
    factor is nu^3/x^2 with x the axis in cm^-1.  Other keywords as :func:`epr`; there is no strain in FDMR
    (Telser's input has none).
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
    """Powder ENDOR spectrum on ``rf`` (MHz, absolute, uniform) at one ``field`` (mT) and ``mw_freq`` (GHz).

    Every orientation is weighted by the EPR excitation sum_(k,l) P_kl f(nu_kl - nu_mw; W_kl) over
    ``epr_pairs`` (None: all), with the EPR ``width``/``shape``/``strain`` (``field_strain``: include
    (B/B_set)^b); orientations with a non-positive weight are dropped.  Each NMR pair (``pairs``, 0-based;
    None: all) then adds P_NMR pop nu f(rf - nu_ij; w(n)) with P_NMR the B1-averaged |<i|I|j>|^2 of
    nucleus ``nucleus`` (no g_N, no hyperfine enhancement; Telser's NMRTRANS in exact form), the NMR half
    width ``nmr_width`` (wx, wy, wz) and line ``nmr_shape``.  ``intensity_shape``: which line type selects
    the nu or nu^3/rf^2 Flygare factor (``'nmr'``, Telser exact; ``'epr'``, the Fortran's ELNTYP flag).
    ``cutoff`` cuts the EPR selection line, ``nmr_cutoff`` (default ``cutoff``) the NMR lines.
    Other keywords as :func:`epr`.
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
    """Powder NMR/NQR spectrum (S = 0: ``SpinSystem(S=[], Nucs=...)``) on ``rf`` (MHz) at ``field`` (mT).

    Telser's DDPOWH NMR mode: the nuclear levels (nuclear Zeeman + quadrupole Q) at every grid orientation,
    each pair (``pairs``; None: all) adding P pop nu f(rf - nu_ij; w(n)), P = B1-averaged |<i|I|j>|^2 (no
    g_N).  ``field = 0`` gives the NQR spectrum.  ``shielding_khz`` (sigma_x, sigma_y, sigma_z): Telser's
    shielding, the shift of the Larmor frequency in kHz for the field along each molecular axis (a positive
    sigma lowers |nu_N|): the nuclear Zeeman moment along k is scaled by 1 - sigma_k/|nu_N(B)|.  Differentiable
    in Q, the shielding, the widths and the temperature.  ``nmr_width`` (wx, wy, wz) HWHM in ``units``.
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
    """Every spectrum of a Telser input converted with ``telser2torchspin.ddpowh2sys(inp, target='lft')``
    (or ``ddpowj2sys(..., target='lft')``), one :class:`PowderResult` per spectrum.

    ``sys`` replaces the converted SpinSystem (e.g. one rebuilt with tensor parameters for gradients);
    ``overrides`` replace keywords of every spectrum (``width=``, ``strain=``, ``temperature=``, ...).
    The axis Telser writes is in ``conversion.extras['spectra'][k]['x']`` (G or T, cm^-1 or THz, MHz).
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
