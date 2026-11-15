"""SOPHE projection onto a uniform axis with centered, hat-weighted bins (B85, 0.9.35).

The powder projections (``pepper``, ``pepper_autograd``, ``salt``) turn every facet of the
orientation grid into a piecewise-linear density on the field (or frequency) axis: a
triangle gives a tent between its three vertex positions (EasySpin ``projecttriangles``), a
Dinfh meridian segment a uniform density between its two end positions (``projectzones``).

Convention (deliberate deviation from EasySpin, see KNOWN_LIMITATIONS.md):

* Axis point ``x_k`` *represents* the density around it, not the interval to its right.
  The value at ``x_k`` is the density integrated against the hat (linear interpolation,
  cloud-in-cell) kernel of half-width ``delta`` centered on ``x_k``::

      spec[k] = ∫ rho(x) Λ((x − x_k)/delta) dx / delta,   Λ(u) = max(0, 1 − |u|)

  (in units where the bin sum, times ``delta``, is the integrated intensity).  Because the
  hats form a partition of unity, ``sum(spec) · delta`` equals the projected intensity
  exactly up to roundoff, except for mass within one increment beyond the first or last
  point, which goes partly to the missing neighbor (dropped, as mass outside the range was
  before).
* A point mass (a facet of zero width: all vertices at one position) is split linearly
  between the two nearest axis points.  The hat-weighted integral of a facet of width w
  tends to that split as w → 0, so the spectrum is continuous in every vertex position, for
  any facet width, and its derivative with respect to the positions exists almost
  everywhere and is the true one (the old truncating deposit of sub-bin facets had none).

EasySpin's ``projecttriangles.c``/``projectzones.c`` integrate the density over the box
``[x_k, x_k + delta)`` and truncate sub-bin facets into one box.  That reports every
projected spectrum half an increment low (best-fit shift +0.49 delta against a converged
reference) and loses the position dependence of sub-bin facets.

The hat-weighted integral of a linear piece is computed exactly: on each half of a hat the
integrand is (linear density) × (linear weight), a quadratic, which Simpson's rule
integrates exactly.  Every term is formed from the overlap of the piece with the half-hat
and the relative position inside the piece (between 0 and 1), never from differences of
O(bin index) quantities, so narrow pieces (density ∝ 1/width) lose no precision.  Bins whose
hat lies entirely inside one linear piece take the density value at the point; those are
filled through running sums with an integer coverage count that keeps uncovered bins at
exactly 0 (B47).

Everything here is torch and differentiable in positions, amplitudes and weights (bin
indices are detached); the numpy wrappers in ``pepper`` call the same code, so the numpy and
torch projections agree to roundoff.
"""
from __future__ import annotations

import torch

_F64 = torch.float64

# Interiors of at most this many bins are filled bin by bin, longer ones by running sums.
_SHORT_INTERIOR = 64

__all__ = ['hat_points_t', 'hat_segments_t', 'project_tents_t', 'project_zones_t',
           'projecttriangles_t', 'projectzones_t']


def hat_points_t(n: int, u: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
    """Point masses at fractional axis positions ``u`` (axis point k at ``u = k``), each
    split linearly between the two nearest points: ``(1 − t)`` to ``floor(u)``, ``t`` to
    ``floor(u) + 1``.  Weights falling outside ``[0, n)`` are dropped."""
    spec = torch.zeros(n, dtype=_F64, device=u.device)
    if u.numel() == 0:
        return spec
    lo = torch.floor(u.detach()).long()
    t = u - lo.to(_F64)
    for k, w in ((lo, 1.0 - t), (lo + 1, t)):
        m = (k >= 0) & (k < n)
        if bool(m.any()):
            spec = spec.index_add(0, k[m], (mass * w)[m])
    return spec


def hat_segments_t(n: int, a: torch.Tensor, b: torch.Tensor, va: torch.Tensor, vb: torch.Tensor) -> torch.Tensor:
    """Hat-weighted projection of linear densities.

    Segment i has density ``va[i]`` at ``a[i]`` and ``vb[i]`` at ``b[i]`` (fractional axis
    coordinates, ``a ≤ b``), linear in between; its mass is ``(va + vb)/2 · (b − a)``.
    Returns ``spec[k] = Σ_i ∫ L_i(u) Λ(u − k) du``.  A segment of zero width is the point
    mass ``(va + vb)/2 · (b − a)`` (value 0) at ``a``, kept on the graph so that the
    derivative with respect to the width is right.
    """
    dev = a.device
    spec = torch.zeros(n, dtype=_F64, device=dev)
    if a.numel() == 0:
        return spec
    w = b - a
    zero = w.detach() <= 0.0
    if bool(zero.any()):
        spec = spec + hat_points_t(n, a[zero], 0.5 * (va[zero] + vb[zero]) * w[zero])
    nz = ~zero
    fa = torch.floor(a.detach()).long()
    fb = torch.floor(b.detach()).long()
    nz = nz & (fb + 1 >= 0) & (fa <= n - 1)          # some touched bin in range
    if not bool(nz.any()):
        return spec
    a, b, va, vb, w, fa, fb = a[nz], b[nz], va[nz], vb[nz], w[nz], fa[nz], fb[nz]
    # Edge bins: the hats of fa, fa+1 contain a, those of fb, fb+1 contain b.  For
    # fb <= fa + 2 the four bins fa .. fa+3 cover everything (unused ones get 0).
    k2 = torch.maximum(fb, fa + 2)
    cand = torch.stack([fa, fa + 1, k2, k2 + 1], dim=1)            # (m, 4)
    kf = cand.to(_F64)
    A, B, W = a[:, None], b[:, None], w[:, None]
    VA, VB = va[:, None], vb[:, None]
    dv = VB - VA
    val = torch.zeros_like(kf)
    for lo_off, rising in ((-1.0, True), (0.0, False)):
        lo = kf + lo_off
        hi = lo + 1.0
        p = torch.maximum(A, lo)
        q = torch.minimum(B, hi)
        L = torch.clamp(q - p, min=0.0)
        # positions inside the segment (0..1) and the hat weights at p, q
        sp = (p - A) / W
        sq = (q - A) / W
        if rising:                                   # weight u − (k − 1)
            wp, wq = p - lo, q - lo
        else:                                        # weight (k + 1) − u
            wp, wq = hi - p, hi - q
        Lp = VA + dv * sp
        Lq = VA + dv * sq
        Lm = 0.5 * (Lp + Lq)
        wm = 0.5 * (wp + wq)
        val = val + (L / 6.0) * (Lp * wp + 4.0 * Lm * wm + Lq * wq)
    inr = (cand >= 0) & (cand < n)
    spec = spec.index_add(0, cand[inr], val[inr])
    # Interior bins fa+2 .. fb-1: the hat lies inside the segment, value L(k).
    lo_i = torch.clamp(fa + 2, min=0)
    hi_i = torch.clamp(fb, max=n)                    # exclusive
    cnt_i = torch.clamp(hi_i - lo_i, min=0)
    # Short interiors are scattered bin by bin with the local form va + dv·(k − a)/w
    # (no cancellation).  Long ones go through running sums of c1·k + c0, whose
    # roundoff residual (∝ eps·|dv|·a/w) is small because w is large there; feeding the
    # many short ones through the same sums would leave a residual ∝ eps·|dv|·a/w of
    # each in every later covered bin.
    short = (cnt_i > 0) & (cnt_i <= _SHORT_INTERIOR)
    if bool(short.any()):
        reps = cnt_i[short]
        seg = torch.repeat_interleave(torch.arange(int(reps.numel()), device=dev), reps)
        start = torch.repeat_interleave(lo_i[short], reps)
        offs = torch.arange(int(reps.sum()), device=dev) - torch.repeat_interleave(torch.cumsum(reps, 0) - reps, reps)
        kk = start + offs
        a_s, w_s, va_s, dv_s = a[short][seg], w[short][seg], va[short][seg], dv[short, 0][seg]
        spec = spec.index_add(0, kk, va_s + dv_s * ((kk.to(_F64) - a_s) / w_s))
    has = cnt_i > _SHORT_INTERIOR
    if bool(has.any()):
        s1 = (dv[:, 0] / w)[has]
        s0 = va[has] - s1 * a[has]
        li, hj = lo_i[has], hi_i[has]
        S1 = torch.zeros(n + 1, dtype=_F64, device=dev).index_add(0, li, s1).index_add(0, hj, -s1)
        S0 = torch.zeros(n + 1, dtype=_F64, device=dev).index_add(0, li, s0).index_add(0, hj, -s0)
        c1 = torch.cumsum(S1, 0)[:n]
        c0 = torch.cumsum(S0, 0)[:n]
        # Past the last interior the +s/−s pairs leave a roundoff residual (multiplied
        # by k); an integer coverage count keeps uncovered bins exactly 0 (B47).
        cnt = torch.zeros(n + 1, dtype=torch.long, device=dev).index_add(
            0, li, torch.ones_like(li)).index_add(0, hj, -torch.ones_like(hj))
        cov = torch.cumsum(cnt, 0)[:n] > 0
        kk = torch.arange(n, dtype=_F64, device=dev)
        spec = torch.where(cov, spec + c1 * kk + c0, spec)
    return spec


def project_tents_t(P: torch.Tensor, mass: torch.Tensor, n: int) -> torch.Tensor:
    """Project tents.  ``P``: ``(m, 3)`` vertex positions in fractional axis coordinates
    (no NaN); ``mass``: ``(m,)`` integrated intensity of each tent (bin-sum units).  A tent
    rises linearly from its lowest to its middle vertex and falls to its highest one, with
    peak ``2·mass/width``; a tent of zero width is a point mass at the vertex centroid."""
    spec = torch.zeros(n, dtype=_F64, device=P.device)
    if P.shape[0] == 0:
        return spec
    P, _ = torch.sort(P, dim=1)
    left, mid, right = P[:, 0], P[:, 1], P[:, 2]
    width = right - left
    Pd = P.detach()
    inside = torch.floor(Pd[:, 0]) == torch.floor(Pd[:, 2])
    if bool(inside.any()):
        spec = spec + hat_points_t(n, P[inside].mean(dim=1), mass[inside])
    nz = ~inside
    if not bool(nz.any()):
        return spec
    left, mid, right, width, mass = left[nz], mid[nz], right[nz], width[nz], mass[nz]
    h = 2.0 * mass / width
    zeros = torch.zeros_like(h)
    a = torch.cat([left, mid]); b = torch.cat([mid, right])
    va = torch.cat([zeros, h]); vb = torch.cat([h, zeros])
    return spec + hat_segments_t(n, a, b, va, vb)


def project_zones_t(left: torch.Tensor, right: torch.Tensor, mass: torch.Tensor, n: int) -> torch.Tensor:
    """Project uniform densities on ``[left, right]`` (fractional axis coordinates, no NaN,
    ``left ≤ right``) with integrated intensity ``mass``; zero-width zones are point
    masses."""
    spec = torch.zeros(n, dtype=_F64, device=left.device)
    if left.numel() == 0:
        return spec
    width = right - left
    inside = torch.floor(left.detach()) == torch.floor(right.detach())
    if bool(inside.any()):
        spec = spec + hat_points_t(n, 0.5 * (left[inside] + right[inside]), mass[inside])
    nz = ~inside
    if not bool(nz.any()):
        return spec
    hgt = mass[nz] / width[nz]
    return spec + hat_segments_t(n, left[nz], right[nz], hgt, hgt)


def projecttriangles_t(tri_idx: torch.Tensor, areas: torch.Tensor, fun: torch.Tensor, amp: torch.Tensor,
                       x0: float, delta: float, n: int) -> torch.Tensor:
    """SOPHE triangle projection (EasySpin ``projecttriangles``) with centered hat bins.

    ``fun``/``amp``: ``(n_slots, n_knots)`` positions and amplitudes (NaN = no
    resonance); ``tri_idx`` ``(nTri, 3)`` knot indices; ``areas`` ``(nTri,)`` solid
    angles.  Every triangle is a tent of integrated intensity ``mean(amp)·area`` between
    its vertex positions; returns the spectral density on the ``n``-point axis
    ``x0 + k·delta`` (``spec[k]`` = hat-weighted integral around ``x_k`` / ``delta``),
    summed over slots and triangles.  Continuous in the vertex positions for every
    triangle width, so gradients exist almost everywhere and are the true ones."""
    dev = fun.device
    tri_idx = torch.as_tensor(tri_idx, dtype=torch.long, device=dev)
    P = fun[:, tri_idx].reshape(-1, 3)                       # (n_slots·nTri, 3)
    A = amp[:, tri_idx].reshape(-1, 3)
    area = torch.as_tensor(areas, dtype=_F64, device=dev).repeat(fun.shape[0])
    ok = ~torch.isnan(P.detach()).any(dim=1)
    if not bool(ok.any()):
        return torch.zeros(n, dtype=_F64, device=dev)
    P, A, area = P[ok], A[ok], area[ok]
    mass = A.mean(dim=1) * area / delta
    return project_tents_t((P - x0) / delta, mass, n)


def projectzones_t(pos: torch.Tensor, amp: torch.Tensor, seg_weights: torch.Tensor,
                   x0: float, delta: float, n: int) -> torch.Tensor:
    """Axial-grid projection (EasySpin ``projectzones``) with centered hat bins: each
    segment between consecutive knots spreads a uniform density of integrated intensity
    ``mean(amp)·weight`` over its field interval.  ``pos``/``amp``: ``(n_slots, n_knots)``
    (``amp`` may have one column: the same amplitude everywhere); ``seg_weights``:
    ``(n_knots-1,)``."""
    dev = pos.device
    nSeg = seg_weights.shape[0]
    p0, p1 = pos[:, :nSeg], pos[:, 1:nSeg + 1]
    left = torch.minimum(p0, p1).reshape(-1); right = torch.maximum(p0, p1).reshape(-1)
    if amp.shape[-1] == 1:
        mean_amp = amp[:, :1].expand(-1, nSeg).reshape(-1)
    else:
        mean_amp = ((amp[:, :nSeg] + amp[:, 1:nSeg + 1]) / 2.0).reshape(-1)
    sw = torch.as_tensor(seg_weights, dtype=_F64, device=dev).repeat(pos.shape[0])
    ok = ~(torch.isnan(left.detach()) | torch.isnan(right.detach()))
    if not bool(ok.any()):
        return torch.zeros(n, dtype=_F64, device=dev)
    mass = mean_amp[ok] * sw[ok] / delta
    return project_zones_t((left[ok] - x0) / delta, (right[ok] - x0) / delta, mass, n)
