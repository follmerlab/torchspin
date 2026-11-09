"""Powder building blocks for :mod:`torchspin.lft.ddnpow`: orientation grids and DLINE.F lines (torch).

Ported from telser-tools (``telser_tools_diff.spin.grid`` and ``telser_tools_diff.spin.lineshape``).
The grids do not depend on any fitted parameter: the NumPy grid of
:mod:`torchspin.lft._engines.igloo` (Fortran "igloo" loop or the exact solid-angle quadrature
on the same rings) is computed once per argument set, converted to float64 tensors and cached.
``Grid`` fields (all (N,) unless noted): ``theta``, ``phi`` (radians), ``weight`` (Fortran grid:
1 per point; exact grid: solid-angle weights summing to 1), ``directions`` (N, 3), ``sin_t``,
``cos_t``, ``sin_p``, ``cos_p``, ``ring`` (long); ``n_points`` = N.  The tensors are shared
between calls: do not modify them in place.  Grid angle ranges are in degrees here (Telser's
convention); :mod:`torchspin.lft.ddnpow` converts from radians.

``line(shape, deriv, cutoff, width, x, exact=False)``: x = W - W0 (offset from the line
center), ``width`` = HWHM WG, both broadcastable, same frequency unit::

    Gaussian      f(x)  = (1/WG) exp(-ln2 x^2/WG^2),      f'(x) = f(x) (-2 ln2 x / WG^2)
    Lorentzian    f(x)  = WG / (WG^2 + x^2),              f'(x) = -2 x WG / (WG^2 + x^2)^2

f = 0 when WG <= 0 or outside the cutoff: (x/WG)^2 > ICUTOFF (Fortran: sqrt(ICUTOFF) half
widths, 7-digit ln 2) or |x/WG| > ICUTOFF with full-precision ln 2 (``exact=True``).  The
cutoff is a discrete choice (a point exactly on the edge has a jump; its derivative is that of
the side kept); ``width <= 0`` lines are 0 with 0 gradient.

:func:`spectrum` superposes many lines on an axis, sum_k a_k f(axis - c_k; w_k), touching
only the axis points inside each line's cutoff window (decided on detached values).
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import NamedTuple

import numpy as np
import torch

from ._engines import igloo as _igloo
from ._engines.igloo import DL2, cut_radius

__all__ = ["Grid", "from_numpy", "fortran_grid", "exact_grid", "make_grid", "line", "spectrum"]

_F64 = torch.float64


def _t(x) -> torch.Tensor:
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=_F64)


class Grid(NamedTuple):
    theta: torch.Tensor
    phi: torch.Tensor
    weight: torch.Tensor
    directions: torch.Tensor
    sin_t: torch.Tensor
    cos_t: torch.Tensor
    sin_p: torch.Tensor
    cos_p: torch.Tensor
    ring: torch.Tensor

    @property
    def n_points(self) -> int:
        """Number of orientations (``len(grid)`` is the NamedTuple's field count)."""
        return self.theta.shape[0]


def from_numpy(g: _igloo.Grid, device: str | torch.device = "cpu") -> Grid:
    """Tensors of a NumPy :class:`torchspin.lft._engines.igloo.Grid`."""
    f = lambda a: torch.as_tensor(np.array(a, float).reshape(-1), dtype=torch.float64, device=device)  # noqa: E731
    d = torch.as_tensor(np.array(g.directions, float).reshape(-1, 3), dtype=torch.float64, device=device)
    return Grid(f(g.theta), f(g.phi), f(g.weight), d, f(g.sin_t), f(g.cos_t), f(g.sin_p), f(g.cos_p),
                torch.as_tensor(np.array(g.ring, int).reshape(-1), dtype=torch.long, device=device))


def _key(angle_range):
    return None if angle_range is None else tuple(float(a) for a in angle_range)


@lru_cache(maxsize=64)
def _fortran(igloo: int, n_rot: int, angle_range, deg: float, device: str) -> Grid:
    return from_numpy(_igloo.fortran_grid(igloo, n_rot, angle_range, deg), device)


@lru_cache(maxsize=64)
def _exact(igloo: int, n_rot: int, angle_range, device: str) -> Grid:
    return from_numpy(_igloo.exact_grid(igloo, n_rot, angle_range), device)


def fortran_grid(igloo: int, n_rot: int, angle_range=None, deg: float = 1.745329252e-2,
                 device: str | torch.device = "cpu") -> Grid:
    """The igloo loop of ddpowhea.f ll. 1067-1086 (weights 1)."""
    return _fortran(int(igloo), int(n_rot), _key(angle_range), float(deg), str(torch.device(device)))


def exact_grid(igloo: int, n_rot: int, angle_range=None, device: str | torch.device = "cpu") -> Grid:
    """Same rings and point counts, phi over the full range, solid-angle weights (sum 1)."""
    return _exact(int(igloo), int(n_rot), _key(angle_range), str(torch.device(device)))


def make_grid(igloo: int, n_rot: int, angle_range=None, *, exact: bool, deg: float = 1.745329252e-2,
              device: str | torch.device = "cpu") -> Grid:
    return exact_grid(igloo, n_rot, angle_range, device) if exact else fortran_grid(igloo, n_rot, angle_range,
                                                                                    deg, device)


def line(shape: str, deriv: bool, cutoff: int, width, x, *, exact: bool = False) -> torch.Tensor:
    """f(x) or f'(x) (``deriv``) for HWHM ``width`` and offsets ``x`` (broadcast); 0 outside the cutoff."""
    w, x = torch.broadcast_tensors(_t(width), _t(x))
    ok = w > 0
    ws = torch.where(ok, w, torch.ones_like(w))
    r2 = (x / ws) ** 2
    if exact:
        lim = cut_radius(cutoff, True)
        keep = ok & (r2 <= lim * lim)
    else:
        keep = ok & (r2 <= float(cutoff))
    dl2 = -math.log(2.0) if exact else DL2
    if shape == "L":
        den = ws * ws + x * x
        val = (-2.0 * x * ws) / (den * den) if deriv else ws / den
    else:
        val = (1.0 / ws) * torch.exp(dl2 * r2)
        if deriv:
            val = val * (2.0 * (dl2 / (ws * ws)) * x)
    return torch.where(keep, val, torch.zeros_like(val))


def spectrum(axis, centers, widths, amplitudes, shape: str, deriv: bool, cutoff: int, *, exact: bool = False,
             window: bool = True) -> torch.Tensor:
    """y(axis) = sum_k amplitudes_k f(axis - centers_k; widths_k), shape (nx,).

    ``axis`` (nx,) must be uniformly spaced when ``window`` is True (each line is evaluated
    only on the axis points inside its cutoff window, chosen on detached values); with
    ``window=False`` every line is evaluated on the whole axis (any axis).  ``centers``,
    ``widths``, ``amplitudes`` are (L,) (broadcast) and differentiable.
    """
    ax = _t(axis).to(_F64)
    c, w, a = torch.broadcast_tensors(_t(centers), _t(widths), _t(amplitudes))
    c, w, a = c.reshape(-1), w.reshape(-1), a.reshape(-1)
    nx = ax.shape[-1]
    if not window or nx < 2 or c.numel() == 0:
        f = line(shape, deriv, cutoff, w[:, None], ax[None, :] - c[:, None], exact=exact)
        return (a[:, None] * f).sum(0)
    with torch.no_grad():
        a0, da = float(ax[0]), float(ax[-1] - ax[0]) / (nx - 1)
        rad = cut_radius(cutoff, exact) * w.detach()
        cd = c.detach()
        sgn = 1.0 if da > 0 else -1.0
        lo = torch.floor((cd - sgn * rad - a0) / da).long() - 1
        hi = torch.ceil((cd + sgn * rad - a0) / da).long() + 1
        lo, hi = torch.minimum(lo, hi), torch.maximum(lo, hi)
        hit = (hi >= 0) & (lo <= nx - 1) & (w.detach() > 0)
        sel = torch.nonzero(hit).squeeze(-1)
    y = torch.zeros(nx, dtype=torch.promote_types(ax.dtype, a.dtype))
    if sel.numel() == 0:
        return y + 0.0 * (a.sum() + c.sum() + w.sum())
    lo, hi = lo[sel].clamp(0, nx - 1), hi[sel].clamp(0, nx - 1)
    span = int((hi - lo).max()) + 1
    idx = lo[:, None] + torch.arange(span)[None, :]
    ok = idx <= hi[:, None]
    idx = torch.where(ok, idx, torch.zeros_like(idx))
    f = line(shape, deriv, cutoff, w[sel, None], ax[idx] - c[sel, None], exact=exact)
    val = torch.where(ok, f * a[sel, None], torch.zeros_like(f))
    return y.index_add(0, idx.reshape(-1), val.reshape(-1))
