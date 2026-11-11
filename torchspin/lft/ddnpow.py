"""DDNPOW: powder EPR (field-swept and frequency-domain) directly from the complete d^n model.

Torch port of Joshua Telser's DDNPOW in its exact mode (via telser-tools,
``telser_tools_diff.ddnpow``), built on :mod:`torchspin.lft.ddn`.  No spin Hamiltonian is
involved: for every orientation n of the grid and every field B of the sweep,
H = H0 + mu_B B n . (k L + g_e S) of the full d^n basis is diagonalized and every requested
pair (i, j) adds (Flygare eqs. 7-38 and 7-8)::

    y(B) += P_ij pop_ij nu_ij f(nu_ij(B) - nu_mw; W(n))              Gaussian
    y(B) += P_ij pop_ij nu_ij^3/nu_mw^2 f(nu_ij(B) - nu_mw; W(n))    Lorentzian

with f Telser's DLINE.F line, the HWHM W(n)^2 = sum_k W_k^2 n_k^2, P the B1-averaged
|<i|B1 . (k L + g_e S)|j>|^2, pop_ij = p_i - p_j (Boltzmann, normalized) and nu_ij in cm^-1.
:func:`fdmr` is the frequency-domain counterpart (one field, y(nu) += P pop nu_ij f(nu - nu_ij)).

Every Hamiltonian parameter, the line widths, the microwave frequency, the temperature and
``scale`` may carry gradients.

Units
-----
Energy-valued inputs (``B``, ``C``, ``A``, ``zeta``, ``alpha``, ``slater``, ``V``), the line
widths ``width`` (HWHM, x/y/z) and the FDMR axis are in ``units`` (default ``'MHz'``;
``'cm^-1'``, ``'GHz'``, ...).  The field axis is in mT, the microwave frequency in GHz, the
temperature in K, angles in radians (``degrees=True`` for Telser-style input).  Intensities are
arbitrary units defined exactly as DDNPOW's exact mode, independent of ``units`` (nu in cm^-1
in the Flygare factor, the line in 1/MHz), except that field-derivative spectra are d/dB per
mT (10x DDNPOW's per-gauss values).

    >>> import torch
    >>> from torchspin import lft
    >>> V = torch.diag(torch.tensor([9900., -2100., -5100., -5100., 2400.], dtype=torch.float64))  # cm^-1
    >>> res = lft.ddnpow.simulate(9, torch.linspace(250, 380, 261), 9.5, V=V, zeta=830.0,
    ...                           k=(0.8, 0.8, 0.8), width=(1.0e-3, 1.0e-3, 2.0e-3), units='cm^-1',
    ...                           igloo=10, derivative=True)
    >>> res.spectrum.shape                                     # (261,)

With ``units='cm^-1'`` the widths are in cm^-1 too (30 MHz = 1.0e-3 cm^-1).

Rayleigh-Ritz subspace (``subspace=True``, the default)
-------------------------------------------------------
The field sweep is solved in the span Q of the low eigenvectors at three fields of the sweep
and at zero field (orthonormalized by an SVD), as telser-tools does.  Q is a discrete choice and
is built on DETACHED values; Q^H H(theta) Q is linear in the parameters theta for fixed Q, so it
is assembled on the graph and diagonalized with :func:`torchspin.lft._linalg.eigh_gauge`.  The
forward values are the Ritz values, and their derivative is v_R^H dH v_R with the Ritz vector
v_R, which differs from the full-basis derivative by O(|v_R - v|) ~ (Ritz energy error / outer
gap)^(1/2).  Measured against ``subspace=False`` (full N x N eigh at every point; d^8, 45 states):
spectra agree to < 1e-7 and Jacobians to ~2e-6 of the largest response.

Discrete choices (all on detached values): the grid extent (``extent='auto'``: the octant when
the detached V has D2h symmetry about x, y, z, otherwise the hemisphere; a perturbation of V
that breaks D2h switches the grid, a discontinuity of the program itself), the number of states
kept, the (orientation, field) points inside the line cutoff and the FDMR line windows.

Accuracy.  The levels lie at 10^4-10^5 cm^-1, each transition is a difference of two of them,
and the lines are 10^-3-10^-2 cm^-1 wide, so the spectrum is only defined to about
2 N eps max|E| / W relative to its peak: telser-tools' NumPy DDNPOW itself moves by 2e-9 to 1e-8
when the energy origin is shifted.  Parity with it is therefore ~1e-8, not 1e-10.

Not ported (Fortran reproduction): DDN's hole problem and constants (``ddn_model``), the mirror
image (``mirror``; apply :func:`torchspin.lft.ddnloop.mirror_image` to V by hand), EPRTRANS
(``probabilities='fortran'``, DDNPOW-01), the H0 = 0 Fortran loop (``zero_field='fortran'``,
DDNPOW-05) and the 1e-10 thresholds; passing any of them raises ``NotImplementedError``.
Supported Fortran switches: ``grid='fortran'`` (DDNPOW-04), ``extent='octant'`` (DDNPOW-03),
``field_derivative=False`` (d/dnu, DDNPOW-06), ``normalized_populations=False`` (DDNPOW-07) and
``exact_cutoff=False`` (sqrt(ICUTOFF) cut, DDNPOW-08).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch

from . import _powder
from . import units as _u
from ._linalg import eigh_gauge
from ._operators import dn_operators
from .ddn import TORCHSPIN_CONSTANTS, Constants, hamiltonian, zeeman_operators

__all__ = ["PowderResult", "FORTRAN_DEG", "fdmr", "is_d2h", "simulate"]

_F64, _C128 = torch.float64, torch.complex128
#: ddnpow.f l. 76: the Fortran's degree constant, used by ``grid='fortran'``.
FORTRAN_DEG = 1.745329252e-2
_PARAMS = ("B", "C", "A", "zeta", "V", "k", "alpha", "slater")
_FORTRAN_ONLY = ("mode", "ddn_model", "mirror", "probabilities", "zero_field", "thresholds")


class PowderResult(NamedTuple):
    """``axis``: fields (mT) for :func:`simulate`, energies (``units``) for :func:`fdmr`;
    ``spectrum`` (len(axis),); ``subspace_used``: the Rayleigh-Ritz path was taken; ``grid``: the
    orientations (:class:`torchspin.lft._powder.Grid`, radians, weights)."""

    axis: torch.Tensor
    spectrum: torch.Tensor
    n_orientations: int
    subspace_used: bool
    grid: _powder.Grid


def _t(x, device="cpu") -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x if x.dtype == _F64 else x.to(_F64)
    return torch.as_tensor(np.asarray(x, float), dtype=_F64, device=device)


def _det(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x


def is_d2h(V, tol: float = 1e-9) -> bool:
    """True if the real-orbital ligand field V has D2h symmetry about x, y, z (only x2-y2 / z2 mix)."""
    if V is None:
        return True
    v = np.asarray(_det(V), float)
    off = v - np.diag(np.diag(v))
    off[0, 4] = off[4, 0] = 0.0
    return bool(np.abs(off).max() <= tol * max(1.0, np.abs(v).max()))


def _shape(shape: str) -> str:
    s = shape.lower()
    if s in ("g", "gaussian"):
        return "G"
    if s in ("l", "lorentzian"):
        return "L"
    raise ValueError("shape must be 'gaussian' ('G') or 'lorentzian' ('L')")


def _mixed_mode(perp, par, h1mode: float):
    if h1mode == 0:
        return perp
    if h1mode < 0:
        return par
    return perp + h1mode * par


class _Ctx:
    """Everything a powder run needs, in cm^-1 (energies) and MHz (line axis) internally."""

    def __init__(self, n, params: dict, *, pairs, width, units, temperature, h1mode, shape, cutoff,
                 normalized_populations, exact_cutoff, frequency_factor, subspace, constants: Constants, device):
        unknown = set(params) - set(_PARAMS)
        bad = sorted(k for k in unknown if k in _FORTRAN_ONLY)
        if bad:
            raise NotImplementedError("Fortran-reproduction options are not ported: " + ", ".join(bad)
                                      + " (use telser_tools.ddnpow for mode='fortran')")
        if unknown:
            raise TypeError(f"unexpected keywords: {sorted(unknown)}")
        self.c = constants
        self.dops = dn_operators(n, device)
        self.dev = self.dops.device
        self.n_full = self.dops.size
        f = _u.energy_factor(units, "cm^-1")
        self.mhz_per_cm = _u.energy_factor("cm^-1", "MHz")
        self.h0 = hamiltonian(self.dops, units=units, constants=constants, **params) * f       # (N, N) cm^-1
        if self.h0.ndim != 2:
            raise ValueError("ddnpow simulates one parameter set; pass unbatched parameters")
        k = params.get("k", (1.0, 1.0, 1.0))
        self.mu = zeeman_operators(self.dops, k, constants)                                   # (3, N, N)
        self.V_det = None if params.get("V") is None else np.asarray(_det(params["V"]), float)
        self.k_det = np.asarray(_det(k), float)
        self.width = _t(width, self.dev) * _u.energy_factor(units, "MHz")
        if self.width.shape != (3,):
            raise ValueError("width must be (Wx, Wy, Wz)")
        self.bkt = None if temperature is None else constants.k_b * _t(temperature, self.dev)
        self.h1mode = float(h1mode)
        self.shape = _shape(shape)
        self.cutoff = int(cutoff)
        self.normalized_populations = normalized_populations
        self.exact_cutoff = exact_cutoff
        self.frequency_factor = frequency_factor
        self.subspace = subspace
        pr = tuple((int(i), int(j)) for i, j in pairs)
        for i, j in pr:
            if not 0 <= i < j < self.n_full:
                raise ValueError(f"pair {(i, j)}: need 0 <= i < j < {self.n_full} (0-based state indices)")
        self.pi = torch.tensor([i for i, _ in pr], dtype=torch.long, device=self.dev)
        self.pj = torch.tensor([j for _, j in pr], dtype=torch.long, device=self.dev)
        with torch.no_grad():                                                                  # discrete: detached
            w0, v0 = torch.linalg.eigh(self.h0.detach())
        self.e0, self.v0 = w0.cpu().numpy(), v0
        top = max(max(j for _, j in pr) + 1, 1) if pr else 1
        if self.bkt is not None:
            top = max(top, int(np.searchsorted(self.e0 - self.e0[0], 46.0 * float(self.bkt.detach()))))
        self.k_need = min(top, self.n_full)

    def line(self, deriv, width, x):
        return _powder.line(self.shape, deriv, self.cutoff, width, x, exact=self.exact_cutoff)

    def populations(self, e):
        if self.bkt is None:
            return None
        p = torch.exp(-(e - e[..., :1]) / self.bkt)
        if self.normalized_populations:
            p = p / p.sum(-1, keepdim=True)
        return p[..., self.pi] - p[..., self.pj]

    def field_matrix(self, d: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...k,kab->...ab", d.to(_C128), self.mu)

    def probabilities(self, v: torch.Tensor, d: torch.Tensor | None) -> torch.Tensor:
        """B1-averaged P (S, P) for vectors (S, N, K) at unit directions d (S, 3) (None: isotropic)."""
        vi, vj = v[..., self.pi], v[..., self.pj]
        el = torch.einsum("sap,kab,sbp->skp", vi.conj(), self.mu, vj)
        tot = (el.real ** 2 + el.imag ** 2).sum(-2)
        if d is None:
            return tot / 3.0
        a = torch.einsum("sk,skp->sp", d.to(_C128), el)
        par = a.real ** 2 + a.imag ** 2
        return _mixed_mode(0.5 * (tot - par), par, self.h1mode)


def _width(w: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(((w * d) ** 2).sum(-1))


def _keff(ctx: _Ctx, bmax: float) -> int:
    """Number of low states in the Ritz space: at least k_need + 2, ending at a large zero-field gap."""
    e = ctx.e0
    zs = 4.0 * ctx.c.mu_b * max(bmax, 0.1) * max(1.0, float(np.max(ctx.k_det)))
    gap_min = max(20.0 * zs, 20.0)
    k = min(ctx.k_need + 2, ctx.n_full)
    while k < ctx.n_full and e[k] - e[k - 1] < gap_min:
        k += 1
    return k


@torch.no_grad()
def _ritz_basis(ctx: _Ctx, d: torch.Tensor, bs: np.ndarray, k: int) -> torch.Tensor:
    """(c, N, m) orthonormal basis, detached: low-k eigenvectors at the sample fields and at zero field."""
    h0 = ctx.h0.detach()
    nmu = torch.einsum("ck,kab->cab", d.to(_C128), ctx.mu.detach())
    b = torch.as_tensor(bs, dtype=_F64, device=ctx.dev).to(_C128)
    h = h0 + ctx.c.mu_b * b[None, :, None, None] * nmu[:, None]
    _, v = torch.linalg.eigh(h)
    blocks = [ctx.v0[None, :, :k].expand(d.shape[0], ctx.n_full, k)]
    blocks += [v[:, s, :, :k] for s in range(len(bs))]
    u, _, _ = torch.linalg.svd(torch.cat(blocks, -1), full_matrices=False)
    return u


def _use_subspace(ctx: _Ctx, fields: np.ndarray) -> bool:
    return ctx.subspace and 4 * _keff(ctx, float(np.max(np.abs(fields)))) < ctx.n_full


def _solve_sweep(ctx: _Ctx, d: torch.Tensor, fields: np.ndarray):
    """Energies (c, nf, K) and vectors (c, nf, N, K) of the K lowest states, on the graph."""
    k = ctx.k_need
    nmu = ctx.field_matrix(d)                                                      # (c, N, N)
    f = torch.as_tensor(fields, dtype=_F64, device=ctx.dev).to(_C128)[None, :, None, None]
    if not _use_subspace(ctx, fields):
        w, v = eigh_gauge(ctx.h0 + ctx.c.mu_b * f * nmu[:, None])
        return w[..., :k], v[..., :k]
    kk = _keff(ctx, float(np.max(np.abs(fields))))
    lo, hi = float(np.min(fields)), float(np.max(fields))
    bs = np.array([lo, 0.5 * (lo + hi), hi]) if hi > lo else np.array([lo])
    q = _ritz_basis(ctx, d, bs, kk)                                                # constant
    qh = q.conj().transpose(-2, -1)
    h0s = qh @ ctx.h0.to(_C128) @ q                                                # linear in the parameters
    zs = qh @ nmu @ q
    hs = h0s[:, None] + ctx.c.mu_b * f * zs[:, None]
    w, c = eigh_gauge(0.5 * (hs + hs.conj().transpose(-2, -1)))
    v = q[:, None] @ c[..., :k]
    return w[..., :k], v


def _grid(ctx: _Ctx, igloo: int, angle_range, degrees: bool, grid: str, extent: str) -> _powder.Grid:
    if extent == "auto":
        extent = "octant" if is_d2h(ctx.V_det) else "hemisphere"
    if extent not in ("octant", "hemisphere"):
        raise ValueError("extent must be 'auto', 'octant' or 'hemisphere'")
    if grid not in ("exact", "fortran"):
        raise ValueError("grid must be 'exact' or 'fortran'")
    n_rot = 1 if extent == "octant" else 2
    rng = None
    if angle_range is not None:
        rng = tuple(float(a) for a in angle_range)
        if len(rng) != 4:
            raise ValueError("angle_range is (theta_min, theta_max, phi_min, phi_max)")
        if not degrees:
            rng = tuple(float(np.degrees(a)) for a in rng)
    return _powder.make_grid(igloo, n_rot, rng, exact=(grid != "fortran"), deg=FORTRAN_DEG, device=ctx.dev)


def _chunk(n_items: int, per: int, budget: int = 6_000_000):
    step = max(1, budget // max(per, 1))
    for a in range(0, n_items, step):
        yield slice(a, min(a + step, n_items))


def _epr_field(ctx: _Ctx, g: _powder.Grid, fields: np.ndarray, freq_mhz: torch.Tensor, deriv: bool,
               field_derivative: bool) -> torch.Tensor:
    nf = len(fields)
    y = torch.zeros(nf, dtype=_F64, device=ctx.dev)
    if len(ctx.pi) == 0:
        return y
    freq_cm = freq_mhz / ctx.mhz_per_cm
    wepr = _width(ctx.width, g.directions)
    lorentz = ctx.shape == "L"
    ff = ctx.frequency_factor
    sub = _use_subspace(ctx, fields)
    per = nf * ctx.n_full * max(ctx.k_need, 1) * 2 + 4 * ctx.n_full ** 2 if sub else nf * ctx.n_full ** 2 * 2
    fpos = float(freq_cm.detach()) > 0
    for sl in _chunk(g.n_points, per):
        d = g.directions[sl]
        e, v = _solve_sweep(ctx, d, fields)
        tr = e[..., ctx.pj] - e[..., ctx.pi]
        tmhz = tr * ctx.mhz_per_cm
        w = wepr[sl, None, None].expand(tr.shape)
        wl = ctx.line(deriv, w, tmhz - freq_mhz)
        with torch.no_grad():
            keep = wl != 0
            ci, fi = torch.nonzero(keep.any(-1), as_tuple=True)
        if len(ci) == 0:
            continue
        dsel = d[ci]
        vs = v[ci, fi]
        P = ctx.probabilities(vs, dsel)
        t = tr[ci, fi]
        if not ff:
            amp = torch.ones_like(t)
        elif lorentz:
            amp = t ** 3 / freq_cm ** 2 if fpos else torch.zeros_like(t)
        else:
            amp = t
        mask = keep[ci, fi].to(_F64)
        if deriv and field_derivative:
            nmu = ctx.field_matrix(dsel)
            de = torch.einsum("sak,sab,sbk->sk", vs.conj(), nmu, vs).real
            slope = (de[:, ctx.pj] - de[:, ctx.pi]) * ctx.c.mu_b * ctx.mhz_per_cm               # MHz per mT
            if not ff:
                damp = torch.zeros_like(t)
            elif lorentz:
                damp = 3.0 * t ** 2 / freq_cm ** 2 if fpos else torch.zeros_like(t)
            else:
                damp = torch.ones_like(t)
            f_abs = ctx.line(False, w[ci, fi], tmhz[ci, fi] - freq_mhz)
            val = P * slope * (amp * wl[ci, fi] + damp / ctx.mhz_per_cm * f_abs) * mask
        else:
            val = P * amp * wl[ci, fi] * mask
        pop = ctx.populations(e[ci, fi])
        if pop is not None:
            val = val * pop
        y = y.index_add(0, fi, val.sum(-1) * g.weight[sl][ci])
    return y


def _common(n, params, pairs, width, units, temperature, h1mode, shape, cutoff, normalized_populations,
            exact_cutoff, frequency_factor, subspace, constants, device) -> _Ctx:
    return _Ctx(n, params, pairs=pairs, width=width, units=units, temperature=temperature, h1mode=h1mode,
                shape=shape, cutoff=cutoff, normalized_populations=normalized_populations,
                exact_cutoff=exact_cutoff, frequency_factor=frequency_factor, subspace=subspace,
                constants=constants, device=device)


def simulate(n, fields, mw_freq, *, width, pairs=((0, 1),), shape: str = "gaussian", derivative: bool = False,
             cutoff: int = 8, temperature=None, h1mode: float = 0.0, igloo: int = 10, angle_range=None,
             degrees: bool = False, grid: str = "exact", extent: str = "auto", field_derivative: bool = True,
             normalized_populations: bool = True, exact_cutoff: bool = True, frequency_factor: bool = True,
             subspace: bool = True, scale=1.0, units: str = "MHz", constants: Constants = TORCHSPIN_CONSTANTS,
             device="cpu", **params) -> PowderResult:
    """Field-swept powder (or partial-orientation) EPR spectrum of d^n from the full model.

    ``fields`` (mT, 1-D, not differentiated; DDNPOW's axis is ``linspace(B_min, B_max, npts + 1)``),
    ``mw_freq`` (GHz).  ``width`` (Wx, Wy, Wz) HWHM in ``units``.  ``params``: keywords of
    :func:`torchspin.lft.ddn.hamiltonian` (``B``, ``C``, ``A``, ``zeta``, ``V``, ``k``, ``alpha``,
    ``slater``) in ``units``.

    ``pairs``: 0-based (i, j), i < j (DDNPOW: i in [NSIMIN, NSIMAX], j in [NSJMIN, NSJMAX]).
    ``shape``: ``'gaussian'`` or ``'lorentzian'``; ``derivative``: first derivative;
    ``cutoff``: lines cut at ``cutoff`` half widths.  ``temperature`` (K): Boltzmann population
    differences.  ``h1mode``: 0 perpendicular, < 0 parallel, > 0 perpendicular + h1mode parallel.
    ``igloo``: grid size (rings per 90 degrees).  ``angle_range`` (theta_min, theta_max, phi_min,
    phi_max): a partial orientation range instead of the powder (radians unless ``degrees``).
    ``grid``, ``extent``, ``field_derivative``, ``normalized_populations``, ``exact_cutoff``,
    ``frequency_factor`` (off: no Flygare nu factor, as EasySpin/torchspin), ``subspace``: see the
    module docstring.  ``scale`` multiplies the spectrum.
    """
    ctx = _common(n, params, pairs, width, units, temperature, h1mode, shape, cutoff, normalized_populations,
                  exact_cutoff, frequency_factor, subspace, constants, device)
    g = _grid(ctx, igloo, angle_range, degrees, grid, extent)
    fnp = np.asarray(_det(fields), float).reshape(-1)
    freq_mhz = _t(mw_freq, ctx.dev) * 1.0e3
    y = _epr_field(ctx, g, fnp, freq_mhz, derivative, field_derivative)
    return PowderResult(torch.as_tensor(fnp, dtype=_F64, device=ctx.dev), _t(scale, ctx.dev) * y, g.n_points,
                        _use_subspace(ctx, fnp), g)


def fdmr(n, axis, *, width, field=0.0, pairs=((0, 1),), shape: str = "gaussian", cutoff: int = 8, temperature=None,
         h1mode: float = 0.0, igloo: int = 10, angle_range=None, degrees: bool = False, grid: str = "exact",
         extent: str = "auto", normalized_populations: bool = True, exact_cutoff: bool = True,
         frequency_factor: bool = True, scale=1.0, units: str = "MHz", constants: Constants = TORCHSPIN_CONSTANTS,
         device="cpu", **params) -> PowderResult:
    """Frequency-domain (FDMR) powder spectrum of d^n on ``axis`` (energies in ``units``).

    ``field`` (mT): the static field magnitude.  At ``field <= 0`` the levels and the isotropic
    probability (1/3) sum_k |<i|mu_k|j>|^2 are the same for every orientation (times 1 + h1mode for
    h1mode > 0) and only the widths W(n) are averaged over the grid.  Other keywords as
    :func:`simulate` (absorption only).
    """
    ctx = _common(n, params, pairs, width, units, temperature, h1mode, shape, cutoff, normalized_populations,
                  exact_cutoff, frequency_factor, True, constants, device)
    g = _grid(ctx, igloo, angle_range, degrees, grid, extent)
    x = _t(axis, ctx.dev).detach().reshape(-1)
    xmhz = x * _u.energy_factor(units, "MHz")
    if len(ctx.pi) == 0:
        return PowderResult(x, torch.zeros_like(x), g.n_points, False, g)
    lorentz = ctx.shape == "L"
    ff = ctx.frequency_factor
    k = ctx.k_need
    wid = _width(ctx.width, g.directions)
    h0 = float(_det(field))
    if h0 <= 0:
        w, v = eigh_gauge(ctx.h0)
        e, v = w[:k], v[:, :k]
        P = ctx.probabilities(v[None], None)[0]
        if ctx.h1mode > 0:
            P = P * (1.0 + ctx.h1mode)
        e = e[None].expand(g.n_points, k)
        P = P[None].expand(g.n_points, -1)
    else:
        d = g.directions
        w, vv = eigh_gauge(ctx.h0 + ctx.c.mu_b * h0 * ctx.field_matrix(d))
        e, v = w[:, :k], vv[:, :, :k]
        P = ctx.probabilities(v, d)
    tr = e[:, ctx.pj] - e[:, ctx.pi]
    amp = P * ((tr ** 3 if lorentz else tr) if ff else 1.0)
    pop = ctx.populations(e)
    if pop is not None:
        amp = amp * pop
    amp = amp * g.weight[:, None]
    y = _powder.spectrum(xmhz, tr * ctx.mhz_per_cm, wid[:, None].expand(tr.shape), amp, ctx.shape, False,
                         ctx.cutoff, exact=ctx.exact_cutoff)
    if lorentz and ff:
        q = (xmhz / ctx.mhz_per_cm) ** 2
        y = torch.where(q > 0, y / torch.where(q > 0, q, torch.ones_like(q)), torch.zeros_like(y))
    return PowderResult(x, _t(scale, ctx.dev) * y, g.n_points, False, g)
