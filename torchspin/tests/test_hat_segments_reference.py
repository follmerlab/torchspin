"""hat_segments_t (A4, one-pass edge bins) against the previous two-pass kernel.

The reference below is the 0.9.38 edge-bin arithmetic, frozen here.  The new kernel must
agree to roundoff in values and in gradients for every input family the projections feed
it, including the boundary cases whose derivatives come from zero-length overlaps."""
import pytest
import torch

from torchspin._projection import hat_segments_t, hat_points_t, _SHORT_INTERIOR

_F64 = torch.float64


def _reference(n, a, b, va, vb):
    spec = torch.zeros(n, dtype=_F64)
    if a.numel() == 0:
        return spec
    w = b - a
    zero = w.detach() <= 0.0
    if bool(zero.any()):
        spec = spec + hat_points_t(n, a[zero], 0.5 * (va[zero] + vb[zero]) * w[zero])
    fa = torch.floor(a.detach()).long()
    fb = torch.floor(b.detach()).long()
    nz = ~zero & (fb + 1 >= 0) & (fa <= n - 1)
    if not bool(nz.any()):
        return spec
    a, b, va, vb, w, fa, fb = a[nz], b[nz], va[nz], vb[nz], w[nz], fa[nz], fb[nz]
    k2 = torch.maximum(fb, fa + 2)
    cand = torch.stack([fa, fa + 1, k2, k2 + 1], dim=1)
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
        sp = (p - A) / W
        sq = (q - A) / W
        wp, wq = (p - lo, q - lo) if rising else (hi - p, hi - q)
        Lp = VA + dv * sp
        Lq = VA + dv * sq
        val = val + (L / 6.0) * (Lp * wp + 4.0 * 0.5 * (Lp + Lq) * 0.5 * (wp + wq) + Lq * wq)
    inr = (cand >= 0) & (cand < n)
    spec = spec.index_add(0, cand[inr], val[inr])
    lo_i = torch.clamp(fa + 2, min=0)
    hi_i = torch.clamp(fb, max=n)
    for i in range(a.numel()):                       # interiors: plain loop, density at k
        for k in range(int(lo_i[i]), int(hi_i[i])):
            spec = spec.index_add(0, torch.tensor([k]), (va[i] + dv[i, 0] * ((k - a[i]) / w[i]))[None])
    return spec


def _family(name, n, m, g):
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(m, dtype=_F64, generator=g)
    if name == 'narrow':                             # sub-bin pieces (tents of fine grids)
        a = u(-2, n + 1); b = a + u(0, 0.9)
    elif name == 'medium':                           # a few bins
        a = u(-3, n); b = a + u(0.5, 6)
    elif name == 'long':                             # interiors via running sums
        a = u(-20, n); b = a + u(_SHORT_INTERIOR + 1, _SHORT_INTERIOR + 40)
    elif name == 'integer_ends':                     # a, b exactly on axis points
        a = torch.randint(-2, n, (m,), generator=g).to(_F64); b = a + torch.randint(0, 5, (m,), generator=g).to(_F64)
    elif name == 'mixed_zero':                       # some zero widths (point masses)
        a = u(0, n); b = a + u(0, 3) * (torch.rand(m, dtype=_F64, generator=g) > 0.3)
    elif name == 'far_out':                          # entirely off the axis on both sides
        a = torch.cat([u(-50, -3)[: m // 2], u(n + 2, n + 50)[: m - m // 2]]); b = a + u(0, 2)
    va = u(0, 3); vb = u(0, 3)
    vb = torch.where(torch.rand(m, generator=g) < 0.2, torch.zeros_like(vb), vb)   # tent ends
    return a, b, va, vb


FAMILIES = ['narrow', 'medium', 'long', 'integer_ends', 'mixed_zero', 'far_out']


@pytest.mark.parametrize('family', FAMILIES)
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_values_match_reference(family, seed):
    g = torch.Generator().manual_seed(seed)
    n, m = 40, 60
    a, b, va, vb = _family(family, n, m, g)
    new = hat_segments_t(n, a, b, va, vb)
    ref = _reference(n, a, b, va, vb)
    scale = max(float(ref.abs().max()), 1.0)
    assert float((new - ref).abs().max()) <= 1e-13 * scale


@pytest.mark.parametrize('family', FAMILIES)
def test_gradients_match_reference(family):
    g = torch.Generator().manual_seed(7)
    n, m = 30, 25
    ins = [x.clone().requires_grad_(True) for x in _family(family, n, m, g)]
    wts = torch.rand(n, dtype=_F64, generator=g)
    out_new, out_ref = hat_segments_t(n, *ins), _reference(n, *ins)
    if not out_ref.requires_grad:                    # nothing on the axis: constant 0 in both
        assert not out_new.requires_grad and float(out_new.abs().max()) == 0.0
        return
    gn = torch.autograd.grad((hat_segments_t(n, *ins) * wts).sum(), ins, allow_unused=True)
    gr = torch.autograd.grad((_reference(n, *ins) * wts).sum(), ins, allow_unused=True)
    for x, y in zip(gn, gr):
        x = torch.zeros(m, dtype=_F64) if x is None else x
        y = torch.zeros(m, dtype=_F64) if y is None else y
        assert float((x - y).abs().max()) <= 1e-12 * max(float(y.abs().max()), 1.0)


def test_mass_is_conserved_for_interior_segments():
    g = torch.Generator().manual_seed(3)
    n = 200
    a = 5 + 150 * torch.rand(100, dtype=_F64, generator=g)
    b = a + 30 * torch.rand(100, dtype=_F64, generator=g)
    va = torch.rand(100, dtype=_F64, generator=g); vb = torch.rand(100, dtype=_F64, generator=g)
    mass = (0.5 * (va + vb) * (b - a)).sum()
    assert float(hat_segments_t(n, a, b, va, vb).sum()) == pytest.approx(float(mass), rel=1e-13)
