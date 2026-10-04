"""SOPHE spherical powder-averaging grid for torchspin.

Port of EasySpin's ``sphgrid.m`` (Wang & Hanson, J.Magn.Reson. 117, 1-8, 1995).

The grid produces a set of orientations (phi, theta) with solid-angle weights
for numerical integration over the orientational space.  Weights sum to 4π
(full-sphere solid angle).

Supported symmetries (subset of EasySpin):

    'Dinfh'  – axially symmetric: single meridian, phi=0, theta=0..π/2
    'D2h'    – rhombic: one octant, phi=0..π/2, theta=0..π/2
    'Ci'     – triclinic: hemisphere, phi=0..2π, theta=0..π/2
    'C1'     – no symmetry: full sphere, phi=0..2π, theta=0..π
"""
from __future__ import annotations

import functools
import math

import numpy as np
import torch


# Mapping from EasySpin symmetry name → (nOctants, maxPhi, closedPhi)
_SYM_PARAMS: dict[str, tuple[int, float, bool]] = {
    'O3':    (-1, 0.0,              False),  # single z point
    'Dinfh': (0,  0.0,              False),  # quarter meridian
    'D6h':   (1,  math.pi / 2,     True),
    'D4h':   (1,  math.pi / 2,     True),
    'Oh':    (1,  math.pi / 2,     True),
    'D3d':   (1,  math.pi / 2,     True),
    'Th':    (1,  math.pi / 2,     True),
    'D2h':   (1,  math.pi / 2,     True),
    'C4h':   (1,  math.pi / 2,     False),
    'C6h':   (1,  math.pi / 2,     False),
    'C2h':   (2,  math.pi,         False),
    'S6':    (2,  math.pi,         False),
    'Ci':    (4,  2 * math.pi,     False),
    'C1':    (8,  2 * math.pi,     False),
}


def sphgrid(
    symmetry: str,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a spherical powder-averaging grid.

    Parameters
    ----------
    symmetry:
        EasySpin point-group symmetry string, e.g. ``'Dinfh'``, ``'D2h'``,
        ``'Ci'``, ``'C1'``.
    N:
        Number of grid knots along a quarter meridian (``GridSize``).  Must
        be at least 2.

    Returns
    -------
    phi:
        Azimuthal angles in radians, shape ``(M,)``.
    theta:
        Polar angles in radians, shape ``(M,)``.
    weights:
        Solid-angle weights, shape ``(M,)``.  Sum is 4π.
    vecs:
        Unit vectors on the sphere, shape ``(3, M)``; rows are x, y, z.
    """
    if N < 2:
        raise ValueError("N (GridSize) must be at least 2.")
    if symmetry not in _SYM_PARAMS:
        raise ValueError(
            f"Unsupported symmetry '{symmetry}'. "
            f"Supported: {sorted(_SYM_PARAMS)}."
        )

    phi_t, theta_t, weights_t, vecs = _sphgrid_tensors(symmetry, int(N))
    # fresh tensors per call: callers may modify them in place
    return phi_t.clone(), theta_t.clone(), weights_t.clone(), vecs.clone()


@functools.lru_cache(maxsize=64)
def _sphgrid_tensors(symmetry: str, N: int):
    """Cached grid tensors (the grid is rebuilt many times per simulation:
    interpolation, facets, ordering weights)."""
    nOctants, maxPhi, closedPhi = _SYM_PARAMS[symmetry]
    phi, theta, weights = _sphgrid_core(nOctants, maxPhi, N, closedPhi)

    # Convert to tensors
    phi_t = torch.tensor(phi, dtype=torch.float64)
    theta_t = torch.tensor(theta, dtype=torch.float64)
    weights_t = torch.tensor(weights, dtype=torch.float64)

    # Unit vectors: cols are [sin(θ)cos(φ), sin(θ)sin(φ), cos(θ)]
    sin_th = torch.sin(theta_t)
    vecs = torch.stack([
        sin_th * torch.cos(phi_t),
        sin_th * torch.sin(phi_t),
        torch.cos(theta_t),
    ], dim=0)  # (3, M)

    return phi_t, theta_t, weights_t, vecs


# ---------------------------------------------------------------------------
# Core grid generation (faithful Python translation of sphgrid_.m)
# ---------------------------------------------------------------------------

def _sphgrid_core(
    nOctants: int,
    maxPhi: float,
    GridSize: int,
    closedPhi: bool,
) -> tuple[list[float], list[float], list[float]]:
    """Internal SOPHE grid generator.  Returns plain Python lists."""
    dtheta = (math.pi / 2) / (GridSize - 1)

    if nOctants > 0:
        # Determine effective octant count for phi sweep
        if nOctants == 8:
            nOct = 4
        else:
            nOct = nOctants

        nOrientations = GridSize + nOct * GridSize * (GridSize - 1) // 2
        phi = [0.0] * nOrientations
        theta = [0.0] * nOrientations
        Weights = [0.0] * nOrientations

        sindth2 = math.sin(dtheta / 2)

        # Weight for phi=0 boundary points (half-weight if closedPhi)
        w0 = 0.5 if closedPhi else 1.0

        # North pole (theta = 0)
        phi[0] = 0.0
        theta[0] = 0.0
        Weights[0] = maxPhi * (1 - math.cos(dtheta / 2))

        # Intermediate slices (theta from dtheta to pi/2 - dtheta)
        start = 1  # 0-based index into phi/theta/Weights
        for iSlice in range(2, GridSize):  # iSlice = 2..GridSize-1
            nPhi = nOct * (iSlice - 1) + 1
            dPhi = maxPhi / (nPhi - 1)
            th = (iSlice - 1) * dtheta
            w_slice = 2 * math.sin(th) * sindth2 * dPhi
            for k in range(nPhi):
                phi[start + k] = k * dPhi
                theta[start + k] = th
                if k == 0:
                    Weights[start + k] = w_slice * w0
                elif k == nPhi - 1:
                    Weights[start + k] = w_slice * 0.5
                else:
                    Weights[start + k] = w_slice
            start += nPhi

        # Equatorial slice (theta = pi/2)
        nPhi = nOct * (GridSize - 1) + 1
        dPhi = maxPhi / (nPhi - 1)
        w_eq = sindth2 * dPhi
        for k in range(nPhi):
            phi[start + k] = k * dPhi
            theta[start + k] = math.pi / 2
            if k == 0:
                Weights[start + k] = w_eq * w0
            elif k == nPhi - 1:
                Weights[start + k] = w_eq * 0.5
            else:
                Weights[start + k] = w_eq

        # Remove border (phi = maxPhi) points if open phi interval.
        # rmv_indices contains the 0-based indices to remove.
        # For each slice (including equatorial), the last point (phi=maxPhi)
        # is at 0-based index = cumulative sum of nPhi up to that slice minus 1,
        # plus the north pole (index 0).  The MATLAB formula gives 1-based
        # indices = cumsum(nOct*(1:GridSize-1)+1)+1, which corresponds to
        # 0-based = cumsum(nOct*(1:GridSize-1)+1) = our cumulative values below.
        if not closedPhi:
            rmv_indices = set()
            cumulative = 0
            for iSlice in range(1, GridSize):
                cumulative += nOct * iSlice + 1
                rmv_indices.add(cumulative)  # 0-based index to remove
            phi = [phi[i] for i in range(nOrientations) if i not in rmv_indices]
            theta = [theta[i] for i in range(nOrientations) if i not in rmv_indices]
            Weights = [Weights[i] for i in range(nOrientations) if i not in rmv_indices]

        # For C1 (8 octants), add lower hemisphere.
        # After border removal, the equatorial slice has nPhi-1 points (last
        # phi=maxPhi point was removed).  Reflect all non-equatorial points
        # (all points except the last nPhi-1) across the equator.
        if nOctants == 8:
            n_upper = len(phi)
            nPhi_eq = nPhi - 1  # equatorial points after removal
            # Non-equatorial points: 0-based indices 0 .. n_upper-nPhi_eq-1
            # Reversed (MATLAB: idx = length-nPhi+1:-1:1 → 0-based: reversed below)
            idx_lower = list(range(n_upper - nPhi_eq - 1, -1, -1))
            for i in idx_lower:
                Weights[i] /= 2.0  # halve reflected weights
            phi_lower = [phi[i] for i in idx_lower]
            theta_lower = [math.pi - theta[i] for i in idx_lower]
            Weights_lower = [Weights[i] for i in idx_lower]
            phi = phi + phi_lower
            theta = theta + theta_lower
            Weights = Weights + Weights_lower

        # Normalize: sum = 4*pi
        scale = 2 * (2 * math.pi / maxPhi)
        Weights = [w * scale for w in Weights]

    elif nOctants == 0:
        # Dinfh: quarter meridian
        phi = [0.0] * GridSize
        theta = [i * math.pi / 2 / (GridSize - 1) for i in range(GridSize)]
        # Weights: -2*(2*pi)*diff(cos([0, dtheta/2:dtheta:pi/2, pi/2]))
        # Integrate: w_j = 2*(2*pi) * (cos(theta_j - dtheta/2) - cos(theta_j + dtheta/2))
        # Endpoints need care; replicate MATLAB's formulation exactly.
        # cos boundary points: 0, dtheta/2, 3*dtheta/2, ..., (GridSize-2)*dtheta+dtheta/2, pi/2
        cos_bounds = (
            [math.cos(0)]
            + [math.cos((i - 0.5) * dtheta) for i in range(1, GridSize)]
            + [math.cos(math.pi / 2)]
        )
        Weights = [
            -2 * (2 * math.pi) * (cos_bounds[i + 1] - cos_bounds[i])
            for i in range(GridSize)
        ]

    elif nOctants == -1:
        # O3: single z-orientation
        phi = [0.0]
        theta = [0.0]
        Weights = [4 * math.pi]

    else:
        raise ValueError(f"Unsupported nOctants={nOctants}.")

    return phi, theta, Weights


# ---------------------------------------------------------------------------
# Triangle connectivity and areas for SOPHE non-axial (closedPhi, nOct=1) grids
# ---------------------------------------------------------------------------

def _d2h_triangulation(N: int) -> np.ndarray:
    """Build SOPHE triangle connectivity for a closedPhi, nOct=1 grid (e.g. D2h).

    This implements the triangle index formula from EasySpin's ``sphgrid.m``
    (case ``closedPhi=true, nOct=1``).  The grid has ``N*(N+1)//2`` points
    ordered as: north pole, then rows of increasing theta (1, 2, … N points).

    Parameters
    ----------
    N : int
        GridSize (number of knots along quarter meridian, same as passed to
        ``sphgrid``).

    Returns
    -------
    tri_idx : np.ndarray, shape (nTri, 3)
        0-based indices into the sphgrid point array.  Total triangles = (N-1)².
    """
    npts = N * (N + 1) // 2

    # 1-based indices for all points, then remove last of each row
    a = np.arange(1, npts + 1, dtype=np.int64)
    remove_1based = np.array([i * (i + 1) // 2 for i in range(1, N + 1)], dtype=np.int64)
    a = np.delete(a, remove_1based - 1)  # 0-based deletion, values stay 1-based

    # Upward triangles: [t, a[t-1], a[t-1]+1]  (1-based t = 1..n_up)
    n_up = N * (N - 1) // 2
    t1 = np.arange(1, n_up + 1, dtype=np.int64)
    t2 = a[:n_up]
    t3 = a[:n_up] + 1
    tri_up = np.column_stack([t1, t2, t3])

    # Downward triangles: [b, 2*(b+1)-k, b+1]  (1-based)
    n_down = (N - 1) * (N - 2) // 2
    if n_down > 0:
        k = np.arange(1, n_down + 1, dtype=np.int64)
        b = a[k - 1]
        tri_down = np.column_stack([b, 2 * (b + 1) - k, b + 1])
        tri = np.vstack([tri_up, tri_down]) - 1  # convert to 0-based
    else:
        tri = tri_up - 1

    return tri  # shape (nTri, 3), 0-based


def _triangle_areas(tri_idx: np.ndarray, vecs: torch.Tensor) -> np.ndarray:
    """Compute spherical triangle areas (solid angles) via d'Huilier's theorem.

    Parameters
    ----------
    tri_idx : np.ndarray, shape (nTri, 3)
        0-based point indices (from ``_d2h_triangulation``).
    vecs : torch.Tensor, shape (3, nPoints)
        Unit vectors on the sphere (from ``sphgrid``).

    Returns
    -------
    areas : np.ndarray, shape (nTri,)
        Solid angles (steradians) for each triangle.  Sum ≈ 2π for D2h octant
        before the 4π normalization that includes symmetry multiplicity.
    """
    v = vecs.numpy()  # (3, nPoints)
    i0, i1, i2 = tri_idx[:, 0], tri_idx[:, 1], tri_idx[:, 2]

    x1, x2, x3 = v[:, i0], v[:, i1], v[:, i2]

    # Edge arc lengths via dot products
    a1 = np.arccos(np.clip((x2 * x3).sum(axis=0), -1.0, 1.0))
    a2 = np.arccos(np.clip((x3 * x1).sum(axis=0), -1.0, 1.0))
    a3 = np.arccos(np.clip((x1 * x2).sum(axis=0), -1.0, 1.0))

    # d'Huilier's formula
    s = (a1 + a2 + a3) / 2.0
    # Clamp argument to [0, ∞) to avoid sqrt of tiny negative due to float errors
    arg = np.maximum(
        np.tan(s / 2) * np.tan((s - a1) / 2) * np.tan((s - a2) / 2) * np.tan((s - a3) / 2),
        0.0,
    )
    areas = 4.0 * np.arctan(np.sqrt(arg))
    return areas


# ---------------------------------------------------------------------------
# sphrand — random orientations on the sphere
# ---------------------------------------------------------------------------

def sphrand(N: int, k: int = 4):
    """Generate *N* random orientations uniformly distributed on the sphere.

    Port of EasySpin's ``sphrand.m``.

    Parameters
    ----------
    N : int
        Number of orientations to generate.
    k : int
        Octant coverage:

        * 1 — full sphere (4π sr)
        * 2 — upper hemisphere (θ ∈ [0, π/2])
        * 4 — upper octant (φ ∈ [0, π/2], θ ∈ [0, π/2]) [default]
        * 8 — single spherical triangle (φ ∈ [0, π/4], θ ∈ [0, π/2])

    Returns
    -------
    phi : ndarray, shape (N,)
        Azimuthal angles in radians.
    theta : ndarray, shape (N,)
        Polar angles in radians (from +z axis).

    Notes
    -----
    Orientations are drawn from a uniform distribution on the sphere
    restricted to the requested octant.  ``cos(theta)`` is drawn from
    a uniform distribution (not theta itself) to avoid polar crowding.

    Examples
    --------
    >>> phi, theta = sphrand(1000)          # 1000 orientations, upper octant
    >>> phi, theta = sphrand(500, k=2)     # upper hemisphere
    >>> phi, theta = sphrand(200, k=1)     # full sphere
    """
    if k not in (1, 2, 4, 8):
        raise ValueError(f"k must be 1, 2, 4, or 8; got {k}")

    rng = np.random.default_rng()

    # cos(theta) range: full sphere [-1,1], hemisphere [0,1]
    if k == 1:
        cos_theta = rng.uniform(-1.0, 1.0, N)
        phi_max = 2.0 * math.pi
    elif k == 2:
        cos_theta = rng.uniform(0.0, 1.0, N)
        phi_max = 2.0 * math.pi
    elif k == 4:
        cos_theta = rng.uniform(0.0, 1.0, N)
        phi_max = math.pi / 2.0
    else:  # k == 8
        cos_theta = rng.uniform(0.0, 1.0, N)
        phi_max = math.pi / 4.0

    theta = np.arccos(cos_theta)
    phi = rng.uniform(0.0, phi_max, N)

    return phi, theta


# ---------------------------------------------------------------------------
# Triangulation for all grid symmetries (EasySpin sphgrid.m, nargout > 1)
# ---------------------------------------------------------------------------

_GRIDPARAM = {
    # symmetry: (maxPhi, closedPhi, nOctants) — EasySpin private/gridparam.m
    'C1':  (2 * math.pi, False, 8), 'Ci': (2 * math.pi, False, 4),
    'C2h': (math.pi, False, 2),     'S6': (2 * math.pi / 3, False, 2),
    'C4h': (math.pi / 2, False, 1), 'C6h': (math.pi / 3, False, 1),
    'D2h': (math.pi / 2, True, 1),  'Th': (math.pi / 2, True, 1),
    'D3d': (math.pi / 3, True, 1),  'D4h': (math.pi / 4, True, 1),
    'Oh':  (math.pi / 4, True, 1),  'D6h': (math.pi / 6, True, 1),
    'Dinfh': (0.0, True, 0),        'O3': (0.0, True, -1),
}


def gridparam(symmetry: str) -> tuple[float, bool, int]:
    """(maxPhi, closedPhi, nOctants) for a grid symmetry (EasySpin gridparam.m)."""
    if symmetry not in _GRIDPARAM:
        raise ValueError(f"Unsupported symmetry group {symmetry!r}.")
    return _GRIDPARAM[symmetry]


def _ang2vec(phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])


def grid_triangulation(symmetry: str, N: int) -> tuple[np.ndarray, np.ndarray]:
    """Triangle connectivity and solid angles of a SOPHE grid (EasySpin sphgrid.m).

    Returns ``(tri, areas)`` with ``tri`` an ``(nTri, 3)`` array of 0-based
    knot indices into the grid returned by :func:`sphgrid` and ``areas`` the
    spherical triangle areas in steradians. Closed-φ one-octant grids (D2h
    family) use the analytic SOPHE connectivity; open-φ grids use a Delaunay
    triangulation of the (θcosφ, θsinφ) projection, with the wrap-around φ
    border handled as in EasySpin (extra border knots mapped back onto the
    φ = 0 knots) and near-degenerate triangles removed. Full-sphere (C1) grids
    mirror the upper-hemisphere triangulation to the lower hemisphere.
    """
    from scipy.spatial import Delaunay
    maxPhi, closedPhi, nOct = gridparam(symmetry)
    if nOct <= 0:
        return np.zeros((0, 3), dtype=int), np.zeros(0)
    phi, theta, _ = _sphgrid_core(nOct, maxPhi, N, closedPhi)
    phi = np.asarray(phi, dtype=float); theta = np.asarray(theta, dtype=float)
    vecs = _ang2vec(phi, theta)
    if nOct == 1 and closedPhi:
        tri = _d2h_triangulation(N)
        areas = _triangle_areas(tri, torch.from_numpy(vecs))
        return tri, areas
    if nOct in (1, 2):
        # open φ: append the φ = maxPhi border knots, triangulate, map them back
        phx = np.concatenate([phi, np.full(N - 1, maxPhi)])
        thx = np.concatenate([theta, math.pi / 2 * np.arange(1, N) / (N - 1)])
        tri = Delaunay(np.column_stack([thx * np.cos(phx), thx * np.sin(phx)])).simplices
        areas = _triangle_areas(tri, torch.from_numpy(_ang2vec(phx, thx)))
        rmv = np.arange(phi.size, phx.size)
        k = np.arange(N - 1)
        rpl = (k * (k + 1) // 2 + 1) if nOct == 1 else (k * k + k + 1)   # φ = 0 knot of each slice
        remap = np.arange(phx.size); remap[rmv] = rpl
        tri = remap[tri]
    elif nOct == 4:
        tri = Delaunay(np.column_stack([theta * np.cos(phi), theta * np.sin(phi)])).simplices
        areas = _triangle_areas(tri, torch.from_numpy(vecs))
    else:  # nOct == 8, full sphere
        phx, thx, _ = _sphgrid_core(4, 2 * math.pi, N, False)
        phx = np.asarray(phx); thx = np.asarray(thx)
        tri_up = Delaunay(np.column_stack([thx * np.cos(phx), thx * np.sin(phx)])).simplices
        n_total = 4 * N * N - 8 * N + 6
        tri_lo = n_total - 1 - tri_up
        n_eq = 4 * (N - 1)
        on_eq = tri_up >= thx.size - n_eq
        tri_lo[on_eq] = tri_up[on_eq]
        tri = np.concatenate([tri_up, tri_lo], axis=0)
        areas = _triangle_areas(tri, torch.from_numpy(vecs))
    keep = areas >= areas.mean() * 1e-3
    return tri[keep], areas[keep]
