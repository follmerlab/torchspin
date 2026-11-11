"""SOPHE triangle projection of near-zero-width triangles straddling a bin edge (B18).

The edge-bin integrals of the tent ramps used to be written as differences of
O(bin index) terms, e.g. ``b − (a + first + 1)/2``.  With ``f0 ∝ 1/width²`` the
rounding of those terms (~ulp(index)) was amplified to a relative error of order
ulp(index)/width: vertices [10 − 1e-14, 10 − 1e-14, 10] on a 5 + 0.01·k axis gave
area 1.0526 instead of 1, and salt's zero-field 1H ENDOR line came out 2.6 % too
large.  Both the torch (``projecttriangles_t``) and the NumPy (``_projecttriangles``)
projections are checked against an exact rational reference.
"""
import importlib
from fractions import Fraction

import numpy as np
import pytest
import torch

from torchspin.pepper_autograd import projecttriangles_t

pep = importlib.import_module('torchspin.pepper')

_TRI = torch.tensor([[0, 1, 2]])


def _proj_t(P, amp=(1.0, 1.0, 1.0), area=1.0, x0=5.0, delta=0.01, n=1001):
    P = P if torch.is_tensor(P) else torch.tensor(P, dtype=torch.float64)
    amp = amp if torch.is_tensor(amp) else torch.tensor(amp, dtype=torch.float64)
    return projecttriangles_t(_TRI, torch.tensor([area], dtype=torch.float64), P[None], amp[None], x0, delta, n)


def _exact(bin_coords, n):
    """Exact bin contents (unit mass) of the tent on the float bin coordinates."""
    L, M, R = (Fraction(float(v)) for v in sorted(bin_coords))
    out = np.zeros(n)
    if R == L:
        k = int(np.trunc(float(L)))
        if 0 <= k < n:
            out[k] = 1.0
        return out
    W = R - L

    def cdf(u):
        if u <= L:
            return Fraction(0)
        if u >= R:
            return Fraction(1)
        if u <= M:
            return (u - L) ** 2 / (W * (M - L))
        return 1 - (R - u) ** 2 / (W * (R - M))
    for k in range(max(int(L) - 1, 0), min(int(R) + 2, n)):
        out[k] = float(cdf(Fraction(k + 1)) - cdf(Fraction(k)))
    return out


def _fuzz_triangles(seed, count, x0, delta, n):
    rng = np.random.default_rng(seed)
    for i in range(count):
        kind = i % 6
        if kind in (1, 2, 3):          # on / next to a bin edge
            c = x0 + delta * rng.integers(5, n - 30)
            c = c + c * rng.choice([0.0, 1.0, -1.0]) * 10 ** rng.uniform(-16, -13)
        else:
            c = x0 + delta * rng.uniform(5, n - 30)
        if kind == 0:                  # generic, a few bins wide
            P = c + delta * rng.uniform(0, 20, 3)
        elif kind == 1:                # near-degenerate, straddling an edge
            P = c + c * 10 ** rng.uniform(-16, -8, 3) * rng.choice([-1, 1], 3)
        elif kind == 2:                # two vertices tied
            e = c * 10 ** rng.uniform(-16, -8)
            P = [c - e, c - e, c] if rng.random() < 0.5 else [c, c, c + e]
        elif kind == 3:                # all three tied
            P = [c, c, c]
        elif kind == 4:                # sub-bin
            P = c + delta * rng.uniform(0, 1, 3)
        else:                          # one tiny side next to a wide one
            P = c + delta * np.array([0.0, 10 ** rng.uniform(-16, -1), rng.uniform(1, 3)])
        yield kind, np.asarray(P, dtype=float)


def test_repro_area_straddling_edge():
    """[10 − 1e-14, 10 − 1e-14, 10], x0 = 5, Δ = 0.01: area 1.0526 before the fix."""
    s = _proj_t([10 - 1e-14, 10 - 1e-14, 10.0])
    assert float(s.sum()) * 0.01 == pytest.approx(1.0, abs=1e-14)


@pytest.mark.parametrize('impl', ['torch', 'numpy'])
def test_fuzz_vs_exact_reference(impl):
    """Seeded fuzz incl. near-degenerate straddlers: every bin within 1e-12 of the
    exact rational result and the mass conserved (unit mass per triangle)."""
    n, x0 = 1001, 5.0
    x = x0 + 0.01 * np.arange(n)
    delta = float(x[1] - x[0])             # the NumPy projection takes Δ from the axis
    worst_bin = worst_mass = 0.0
    for kind, P in _fuzz_triangles(18, 600, x0, delta, n):
        bins = ((torch.tensor(P, dtype=torch.float64) - x0) / delta).numpy()
        ref = _exact(bins, n)
        if impl == 'torch':
            s = _proj_t(P, x0=x0, delta=delta, n=n).numpy() * delta
        else:
            s = pep._projecttriangles(np.array([[0, 1, 2]]), np.array([1.0]), P, np.ones(3), x) * delta
        worst_bin = max(worst_bin, float(np.abs(s - ref).max()))
        worst_mass = max(worst_mass, abs(float(s.sum()) - float(ref.sum())))
    assert worst_bin < 1e-12
    assert worst_mass < 1e-12


@pytest.mark.parametrize('P', [
    [7.0031, 7.0123, 7.0577], [7.0031, 7.0035, 7.0039], [7.00312, 7.00317, 7.00405],
    [9.1234, 9.3456, 9.9001], [6.00011, 6.0002, 6.0003]])
def test_gradcheck_nondegenerate(P):
    Pt = torch.tensor(P, dtype=torch.float64, requires_grad=True)
    At = torch.tensor([1.0, 1.3, 0.8], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda p, a: _proj_t(p, a, area=0.7), (Pt, At),
                                    eps=1e-7, atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize('P', [
    [10 - 1e-14, 10 - 1e-14, 10.0], [10.0, 10.0, 10.0], [7.003, 7.003, 7.003],
    [7.0, 7.0, 7.0 + 1e-15], [7.0 - 1e-13, 7.0, 7.0 + 1e-13], [7.00312, 7.00312, 7.004]])
def test_degenerate_area_and_finite_gradients(P):
    Pt = torch.tensor(P, dtype=torch.float64, requires_grad=True)
    At = torch.ones(3, dtype=torch.float64, requires_grad=True)
    s = _proj_t(Pt, At, area=0.7)
    assert float(s.detach().sum()) * 0.01 == pytest.approx(0.7, abs=1e-14)
    w = torch.cos(torch.arange(s.numel(), dtype=torch.float64))
    gP, gA = torch.autograd.grad((s * w).sum(), (Pt, At), allow_unused=True)
    for g in (gP, gA):
        assert g is None or bool(torch.isfinite(g).all())
