"""Field-direction grids for magnetometry: single crystal, axes, exact Gauss hemisphere, DDN igloo.

``E_i(-B) = E_i(B)`` (time reversal; the ligand field is real), so ``M(n) = M(-n)`` and a
hemisphere is a full powder average.  Every grid returns unit vectors and weights summing to 1
as NumPy arrays (the grids carry no gradients).  Angles are in radians (torchspin's convention)
except where DDN's own input is reproduced (:func:`ddn_igloo`, whose ranges are DDN's RTH/RPH
in degrees).

:func:`ddn_igloo` reproduces DDNSUSCALP.F / DDNMAGCALP.F ll. 262-312 exactly, including the
main-program clamps (DDNSUSFITP.F ll. 308-319).  Its (90, 90) octant is NOT a powder average
for a low-symmetry complex (off-diagonal chi in the molecular frame); (180, 180) is a
hemisphere.  ``gauss_hemisphere(n)`` is exact for every polynomial in the direction up to
degree 2n (a 3x3 tensor for any n >= 1); use it unless you are reproducing DDN.
Copied from telser-tools (``telser_tools.magnetometry.powder``).

Examples
--------
>>> from torchspin.lft import grids
>>> g = grids.gauss_hemisphere(2)
>>> len(g), round(float(g.weights.sum()), 12)
(10, 1.0)
>>> round(float((g.weights * g.vectors[:, 2] ** 2).sum()), 12)     # <n_z^2> = 1/3, exact
0.333333333333
>>> len(grids.ddn_igloo(90, 90, 5))
22
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["OrientationGrid", "axes", "ddn_igloo", "ddn_igloo_settings", "directions", "gauss_hemisphere", "single"]

DDN_PI = 1.745329252e-2   # DDNSUSCALP.F l. 71: "PI" is pi/180, truncated


@dataclass(frozen=True)
class OrientationGrid:
    """A set of field directions with weights.

    Attributes
    ----------
    vectors : numpy.ndarray, shape (N, 3)
        Unit field directions.
    weights : numpy.ndarray, shape (N,)
        Weights summing to 1.
    label : str
        Description.
    """

    vectors: np.ndarray
    weights: np.ndarray
    label: str = ""

    def __len__(self) -> int:
        return len(self.weights)

    @property
    def theta(self) -> np.ndarray:
        """Polar angles in radians, shape (N,)."""
        return np.arccos(np.clip(self.vectors[:, 2], -1, 1))

    @property
    def phi(self) -> np.ndarray:
        """Azimuths in radians, 0 ... 2 pi, shape (N,)."""
        return np.arctan2(self.vectors[:, 1], self.vectors[:, 0]) % (2 * np.pi)


def _vec(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    st = np.sin(theta)
    return np.column_stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)])


def single(theta: float = 0.0, phi: float = 0.0, degrees: bool = False) -> OrientationGrid:
    """One field direction (a single-crystal measurement).

    ``theta``, ``phi`` are the polar angle and azimuth (radians, or degrees with
    ``degrees=True``).  Returns an :class:`OrientationGrid` with weight 1.
    """
    t, p = (np.radians([theta]), np.radians([phi])) if degrees else (np.array([theta]), np.array([phi]))
    return OrientationGrid(_vec(t, p), np.ones(1), f"theta={theta}, phi={phi}")


def directions(vectors) -> OrientationGrid:
    """Arbitrary directions ``vectors`` (3,) or (N, 3) (normalized here) with equal weights; returns an :class:`OrientationGrid`."""
    v = np.atleast_2d(np.asarray(vectors, float))
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    return OrientationGrid(v, np.full(len(v), 1.0 / len(v)), "directions")


def axes() -> OrientationGrid:
    """The x, y, z directions with weight 1/3 each (equal to DDN's (90, 90, IGLOO=1) grid)."""
    return OrientationGrid(np.eye(3), np.full(3, 1.0 / 3.0), "x, y, z")


def gauss_hemisphere(n: int = 12) -> OrientationGrid:
    """Gauss product grid on the upper hemisphere.

    The positive nodes of 2n-point Gauss-Legendre in cos(theta) times (2n+1) equally spaced phi.
    For an even function of the direction this is the full-sphere product rule, exact for every
    polynomial in (n_x, n_y, n_z) up to degree 2n.

    Parameters
    ----------
    n : int, default 12
        Order; the grid has ``n (2n+1)`` directions.

    Returns
    -------
    OrientationGrid
    """
    u, w = np.polynomial.legendre.leggauss(2 * n)
    keep = u > 0
    ct, wt = u[keep], w[keep]
    nph = 2 * n + 1
    ph = 2 * np.pi * np.arange(nph) / nph
    T, P = np.meshgrid(np.arccos(ct), ph, indexing="ij")
    W = np.repeat(wt[:, None], nph, axis=1) / nph
    return OrientationGrid(_vec(T.ravel(), P.ravel()), W.ravel() / W.sum(), f"Gauss hemisphere n={n}")


def ddn_igloo_settings(rth: float, rph: float, igloo: int) -> tuple[float, float, int]:
    """DDN's clamps of the igloo settings (DDNSUSFITP.F ll. 308-319).

    ``rth``, ``rph`` are DDN's theta and phi ranges in degrees, ``igloo`` its grid parameter.
    Ranges <= 90 become 90; ranges > 90 become 180 and halve ``igloo``; ``igloo`` is then clamped to
    1 ... 50.  Returns ``(rth, rph, igloo)``.
    """
    igloo = int(igloo)
    rth = 90.0 if rth <= 90.0 else rth
    rph = 90.0 if rph <= 90.0 else rph
    if rth > 90.0:
        rth, igloo = 180.0, int(igloo / 2)
    if rph > 90.0:
        rph, igloo = 180.0, int(igloo / 2)
    return rth, rph, min(max(igloo, 1), 50)


def ddn_igloo(rth: float, rph: float, igloo: int, clamp: bool = True) -> OrientationGrid:
    """DDN's igloo powder grid (equal weights).

    Parameters
    ----------
    rth, rph : float
        DDN's theta and phi ranges in degrees (90 or 180 after clamping).
    igloo : int
        DDN's grid density parameter.
    clamp : bool, default True
        Apply :func:`ddn_igloo_settings` first, as DDN does.

    Returns
    -------
    OrientationGrid
    """
    if clamp:
        rth, rph, igloo = ddn_igloo_settings(rth, rph, igloo)
    nth = igloo if rth == 90.0 else 2 * igloo
    maxth = nth + 1 if rth == 90.0 else nth
    nph = igloo if rph == 90.0 else 2 * igloo
    th_l, ph_l = [], []
    for ith in range(1, maxth + 1):
        theta = rth * DDN_PI * float(ith - 1) / float(nth)
        if theta >= 180.0 * DDN_PI:
            continue
        mphm1 = int(float(nph) * np.sin(theta))
        for iph in range(1, mphm1 + 2):
            phi = rph * DDN_PI * ((1.0 - float(mphm1) / float(nph)) / 2.0 + float(iph - 1) / float(nph))
            if phi >= 180.0 * DDN_PI:
                continue
            th_l.append(theta)
            ph_l.append(phi)
    th, ph = np.array(th_l), np.array(ph_l)
    return OrientationGrid(_vec(th, ph), np.full(len(th), 1.0 / len(th)), f"DDN igloo {rth:g}/{rph:g}/{igloo}")


def as_grid(grid) -> OrientationGrid:
    """Normalize a grid argument: None -> field along z; an :class:`OrientationGrid` as is; anything else -> :func:`directions`."""
    if grid is None:
        return single(0.0, 0.0)
    if isinstance(grid, OrientationGrid):
        return grid
    return directions(grid)
