"""Orientation grids of Telser's powder programs (DDPOWH, DDNPOW), NumPy.

Copied from telser-tools (``telser_tools.ddpowh.grid``, unchanged code) so that torchspin has
no runtime dependency on telser-tools; the DLINE.F constants (``DL2``, ``cut_radius``,
``area_fraction``) are copied from ``telser_tools.ddpowh.lineshape``.  Parity with the
originals is checked in torchspin/tests/test_lft_ddnpow.py when telser-tools is installed.
Angles here are in degrees (Telser's input convention); the torch wrappers in
:mod:`torchspin.lft._powder` take radians.

The original "igloo" loop (ddpowhea.f ll. 1067-1086) and a correct quadrature on the same rings.

The Fortran grid
----------------
NROT = 1 (no rotation of A, Q): the octant theta, phi in [0, 90] deg; NROT = 2: the hemisphere
theta, phi in [0, 180).  With a user range (normal powder = N, ll. 432-444) RTH = thmax - thmin
and RPH = phmax - phmin replace NROT*90.  With N = NROT*IGLOO::

    theta_k = thmin + RTH (k - 1)/N,      k = 1 ... N + 2 - NROT       (skipped if >= 180 deg)
    m_k     = INT(N sin theta_k)                                        (truncation, l. 1077)
    phi_kj  = phmin + RPH [ (1 - m_k/N)/2 + (j - 1)/N ],  j = 1 ... m_k + 1   (skipped if >= 180)

and every point has weight 1 (the program just adds).  The number of points in a ring follows
sin theta, as an igloo grid should, but the phi spacing is always RPH/N, so the ring's points
cover only the centered window phi in phmin + RPH [(1 - m_k/N)/2, (1 + m_k/N)/2], i.e.
45 deg +- 45 sin(theta) deg for the octant: an equal-area (sinusoidal-projection) patch, not
the octant.  Orientations near (theta, phi) = (30, 0), (30, 90), (60, 5) ... are never
visited (finding DDPOWH-03).

The exact grid
--------------
Same rings theta_k and the same number of points per ring, but the points are spread over
the whole phi range at their own spacing RPH/(m_k + 1) (midpoints), and each point carries
the solid angle of its cell: the band [theta_k - dtheta/2, theta_k + dtheta/2] (clipped to
the theta range) divided by the number of points in the ring.  Weights are normalized to sum
to 1, so a spectrum is an orientation average.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["Grid", "fortran_grid", "exact_grid", "make_grid", "DL2", "cut_radius", "area_fraction"]


@dataclass(frozen=True)
class Grid:
    theta: np.ndarray       # radians
    phi: np.ndarray
    weight: np.ndarray
    sin_t: np.ndarray
    cos_t: np.ndarray
    sin_p: np.ndarray
    cos_p: np.ndarray
    ring: np.ndarray        # ring index k - 1

    @property
    def directions(self) -> np.ndarray:
        return np.stack([self.sin_t * self.cos_p, self.sin_t * self.sin_p, self.cos_t], axis=-1)

    def __len__(self) -> int:
        return len(self.theta)


def _ranges(n_rot: int, angle_range):
    rth = rph = n_rot * 90.0
    thmin = phmin = 0.0
    if angle_range is not None:
        thmin, thmax, phmin, phmax = angle_range
        rth, rph = thmax - thmin, phmax - phmin
    return thmin, rth, phmin, rph


def fortran_grid(igloo: int, n_rot: int, angle_range=None, deg: float = 1.745329252e-2) -> Grid:
    """The loop of ll. 1067-1086 with the Fortran's PI = pi/180 constant (``deg``)."""
    thmin, rth, phmin, rph = _ranges(n_rot, angle_range)
    nth = nph = n_rot * max(int(igloo), 1)
    maxth = nth + 2 - n_rot
    th_l, ph_l, ring = [], [], []
    for ith in range(1, maxth + 1):
        theta = thmin * deg + rth * deg * (ith - 1) / nth
        if theta >= 180.0 * deg:
            continue
        st = np.sin(theta)
        mphm1 = int(nph * st)                      # INTEGER = REAL*8: truncation toward zero
        for iph in range(1, mphm1 + 2):
            phi = phmin * deg + rph * deg * ((1.0 - mphm1 / nph) / 2.0 + (iph - 1) / nph)
            if phi >= 180.0 * deg:
                continue
            th_l.append(theta)
            ph_l.append(phi)
            ring.append(ith - 1)
    t, p = np.array(th_l), np.array(ph_l)
    return Grid(t, p, np.ones(len(t)), np.sin(t), np.cos(t), np.sin(p), np.cos(p), np.array(ring, int))


def exact_grid(igloo: int, n_rot: int, angle_range=None) -> Grid:
    """Same rings and point counts; phi spread over the full range; solid-angle weights."""
    deg = np.pi / 180.0
    thmin, rth, phmin, rph = _ranges(n_rot, angle_range)
    nth = nph = n_rot * max(int(igloo), 1)
    maxth = nth + 2 - n_rot
    dth = rth * deg / nth
    lo_lim, hi_lim = sorted((thmin * deg, (thmin + rth) * deg))
    th_l, ph_l, w_l, ring = [], [], [], []
    for ith in range(1, maxth + 1):
        theta = thmin * deg + rth * deg * (ith - 1) / nth
        if theta >= np.pi - 1e-12:
            continue
        a = min(max(theta - abs(dth) / 2, lo_lim), hi_lim)
        b = min(max(theta + abs(dth) / 2, lo_lim), hi_lim)
        band = abs(np.cos(a) - np.cos(b))
        npts = int(nph * abs(np.sin(theta))) + 1
        if rth == 0:                                # a single theta value: every ring is the same
            band = 1.0
        for j in range(npts):
            phi = phmin * deg + rph * deg * (j + 0.5) / npts
            if phi >= np.pi - 1e-12:
                continue
            th_l.append(theta)
            ph_l.append(phi)
            w_l.append(band / npts)
            ring.append(ith - 1)
    t, p, w = np.array(th_l), np.array(ph_l), np.array(w_l)
    if w.sum() > 0:
        w = w / w.sum()
    return Grid(t, p, w, np.sin(t), np.cos(t), np.sin(p), np.cos(p), np.array(ring, int))


def make_grid(igloo: int, n_rot: int, angle_range=None, *, exact: bool, deg: float = 1.745329252e-2) -> Grid:
    return exact_grid(igloo, n_rot, angle_range) if exact else fortran_grid(igloo, n_rot, angle_range, deg)


# ---------------------------------------------------------------- DLINE.F constants
DL2 = -0.6931472           # dline.f l. 12 (7-digit -ln 2)


def cut_radius(cutoff: int, exact: bool) -> float:
    """Half-width multiples where the line is cut: sqrt(ICUTOFF) (Fortran) or ICUTOFF (exact)."""
    return float(cutoff) if exact else math.sqrt(max(float(cutoff), 0.0))


def area_fraction(shape: str, cutoff: int, exact: bool = False) -> float:
    """Fraction of the absorption line's area kept inside the cutoff."""
    a = cut_radius(cutoff, exact)
    if shape == "L":
        return 2.0 / math.pi * math.atan(a)
    return math.erf(a * math.sqrt(math.log(2.0)))
