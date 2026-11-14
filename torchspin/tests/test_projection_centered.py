"""Centered, hat-weighted projection bins (B85, 0.9.35).

Before 0.9.35 the SOPHE projections (pepper, pepper_autograd, salt; EasySpin
projecttriangles.c / projectzones.c) integrated each facet over [x_k, x_k + dx) and
reported it at x_k, and dropped facets narrower than one bin into a single bin by
truncation.  Every projected spectrum sat dx/2 low, and the position gradient of
sub-bin facets was lost: for an isotropic g on a forced/auto D2h grid the gradient
along g = s*[1 1 1] was +999.7 (central differences of that spectrum agreed) against
-13770 on the O3 template path.

Axis point x_k now carries the facet density integrated against the hat of half-width
dx centered on it (a point mass is split linearly between its two nearest points).
These tests fail on 0.9.34 except the conservation and agreement checks (e, f).
"""
import importlib
import math

import numpy as np
import pytest
import torch

from torchspin import Experiment, Options, SpinSystem, pepper
from torchspin.pepper_autograd import pepper_autograd, projecttriangles_t, projectzones_t

pep = importlib.import_module('torchspin.pepper')

_F64 = torch.float64


# ---------------------------------------------------------------------------
# (a) no half-bin offset against a converged reference
# ---------------------------------------------------------------------------

def _rhombic(n, lw=1.0, grid=None, fn=pepper):
    sys = SpinSystem(S=0.5, g=[1.98, 2.05, 2.12], lw=lw)
    exp = Experiment(mwFreq=9.5, Range=[310, 350], nPoints=n, Harmonic=1)
    opt = Options(**({'GridSize': grid} if grid else {}))
    x, y = fn(sys, exp, opt)
    return x.detach().numpy(), y.detach().numpy()


@pytest.fixture(scope='module')
def rhombic_reference():
    # 16x finer axis: its own discretization error is (1/16)^2 of the coarse one
    return _rhombic(16 * 511 + 1, grid=[91, 10])


def _best_shift(x, y, xf, yf):
    """Shift s (units of dx) minimizing ||y(x) - ref(x + s dx)||; parabolic refinement."""
    dx = x[1] - x[0]
    s = np.arange(-1.0, 1.0001, 0.01)
    r = np.array([np.sum((y - np.interp(x + sk * dx, xf, yf)) ** 2) for sk in s])
    i = int(np.clip(np.argmin(r), 1, s.size - 2))
    den = r[i - 1] - 2 * r[i] + r[i + 1]
    return s[i] + 0.005 * 2 * (r[i - 1] - r[i + 1]) / (2 * den) if den > 0 else s[i]


@pytest.mark.parametrize('engine', ['pepper', 'pepper_autograd'])
def test_projected_spectrum_not_shifted(rhombic_reference, engine):
    """Rhombic S = 1/2, 512 points over 40 mT, lw 1 mT, default grid.  Measured: best-fit
    shift -0.000 dx (0.9.34: +0.49 dx); max error vs the reference 6.5e-3 of max
    (0.9.34: 9.4e-2)."""
    xf, yf = rhombic_reference
    x, y = _rhombic(512, fn=pepper if engine == 'pepper' else pepper_autograd)
    s = _best_shift(x, y, xf, yf)
    err = np.abs(y - np.interp(x, xf, yf)).max() / np.abs(yf).max()
    print(f'{engine}: best-fit shift {s:+.3f} dx, max error {err:.2e}')
    assert abs(s) < 0.1
    assert err < 1e-2


def test_narrow_line_error_vs_reference():
    """lw 0.3 mT (dx/sigma = 0.6): max error 3.8e-2 of max (0.9.34: 31 %)."""
    xf, yf = _rhombic(16 * 511 + 1, lw=0.3, grid=[91, 10])
    x, y = _rhombic(512, lw=0.3)
    err = np.abs(y - np.interp(x, xf, yf)).max() / np.abs(yf).max()
    print(f'lw 0.3: max error {err:.2e}, shift {_best_shift(x, y, xf, yf):+.3f} dx')
    assert err < 5e-2
    assert abs(_best_shift(x, y, xf, yf)) < 0.1


# ---------------------------------------------------------------------------
# (b) isotropic system on a forced D2h grid vs the O3 template path
# ---------------------------------------------------------------------------

def _iso(n, g, sym=None, fn=pepper):
    exp = Experiment(mwFreq=9.5, Range=[310, 370], nPoints=n, Harmonic=1)
    opt = Options(GridSymmetry=sym) if sym else Options()
    return fn(SpinSystem(S=0.5, g=g, lw=1.0), exp, opt)[1].detach().numpy()


@pytest.mark.parametrize('engine', ['pepper', 'pepper_autograd'])
@pytest.mark.parametrize('g', [2.0, 2.0001, 2.0003])
def test_isotropic_forced_d2h_matches_o3(engine, g):
    """All facets of an isotropic line have (near) zero width.  Measured max error
    relative to the O3 template spectrum: 4096 points 2.3e-4..5.2e-4 (0.9.34:
    2.8e-2..5.5e-2), 512 points 2.0e-2..3.2e-2 (0.9.34: 4.7e-2..3.4e-1; the rest is
    the point-mass discretization at dx/sigma = 0.28)."""
    fn = pepper if engine == 'pepper' else pepper_autograd
    for n, tol in ((4096, 1e-3), (512, 4e-2)):
        ref = _iso(n, g)
        d2h = _iso(n, g, 'D2h', fn)
        err = np.abs(d2h - ref).max() / np.abs(ref).max()
        print(f'{engine} g={g} n={n}: {err:.2e}')
        assert err < tol, (n, err)


# ---------------------------------------------------------------------------
# (c) isotropic-direction gradient (three equal g leaves -> D2h grid)
# ---------------------------------------------------------------------------

def _weighted(g, n, sym=None):
    exp = Experiment(mwFreq=9.5, Range=[310, 370], nPoints=n, Harmonic=1)
    x = torch.linspace(310, 370, n, dtype=_F64)
    w = torch.sin(x / 2.3) * (512 / n)
    opt = Options(GridSymmetry=sym) if sym else Options()
    return (w * pepper_autograd(SpinSystem(S=0.5, g=g, lw=1.0), exp, opt)[1]).sum()


@pytest.mark.parametrize('n,tol_o3', [(512, 0.2), (4096, 0.05)])
def test_isotropic_direction_gradient(n, tol_o3):
    """d/ds of g = s*[1 1 1] at s = 2 from three g leaves (auto grid symmetry: D2h) equals
    converged central differences of the same spectrum and approaches the O3 gradient.

    0.9.34 at 512 points: +999.7 (equal to its central differences: the truncating
    sub-bin deposit had no position dependence) against O3 -13770.  Now, measured:
    512 points -11452 (O3 -13770, 17 %), 4096 points -14237 (O3 -13801, 3.2 %); central
    differences agree to 1e-10.  All facets of an isotropic line coincide, so the line is
    one point mass split linearly between two axis points and its position derivative is
    the chord slope of the sampled line shape: first order in dx (dx/sigma = 0.28 and
    0.035 here), unlike the forward error (second order)."""
    leaves = [torch.tensor(2.0, dtype=_F64, requires_grad=True) for _ in range(3)]
    val = _weighted(torch.stack(leaves), n)
    grads = torch.autograd.grad(val, leaves)
    total = float(sum(grads))
    s = torch.tensor(2.0, dtype=_F64, requires_grad=True)
    (g_o3,) = torch.autograd.grad(_weighted(s, n), s)
    g_o3 = float(g_o3)

    def f(sv):
        return float(_weighted(sv * torch.ones(3, dtype=_F64), n, 'D2h'))
    cds = [(f(2.0 + h) - f(2.0 - h)) / (2 * h) for h in (1e-6, 1e-7)]
    print(f'n={n}: leaves {total:.6e} CD {cds} O3 {g_o3:.6e} ({abs(total / g_o3 - 1):.1e})')
    assert abs(cds[0] - cds[1]) < 1e-8 * abs(cds[0])            # converged
    assert abs(total - cds[0]) < 1e-8 * abs(cds[0])
    assert abs(total - g_o3) < tol_o3 * abs(g_o3)


# ---------------------------------------------------------------------------
# (d) zero-width facets: continuous, position-dependent deposit
# ---------------------------------------------------------------------------

def _tri_spec(c, width=0.0, n=16, x0=0.0, dx=1.0):
    """One triangle with vertices c, c + width/2, c + width (one slot)."""
    fun = torch.stack([c, c + 0.5 * width, c + width]).reshape(1, 3)
    amp = torch.ones(1, 3, dtype=_F64)
    return projecttriangles_t(torch.tensor([[0, 1, 2]]), torch.tensor([1.0], dtype=_F64), fun, amp, x0, dx, n)


@pytest.mark.parametrize('c0', [5.0, 5.25, 5.5, 5.75, 6.0 - 1e-12, 6.0])
def test_zero_width_triangle_linear_split_and_gradient(c0):
    """A zero-width triangle at c is split (1 - t, t) between floor(c) and floor(c) + 1
    (0.9.34: all of it into bin trunc(c), zero position gradient)."""
    c = torch.tensor(c0, dtype=_F64, requires_grad=True)
    spec = _tri_spec(c)
    k = math.floor(c0); t = c0 - k
    expect = np.zeros(16); expect[k] += 1 - t; expect[k + 1] += t
    assert np.allclose(spec.detach().numpy(), expect, rtol=0, atol=1e-12)
    # d spec[k] / dc = -1, d spec[k+1] / dc = +1
    w = torch.zeros(16, dtype=_F64); w[k] = 1.0; w[k + 1] = 2.0
    (g,) = torch.autograd.grad((w * spec).sum(), c)
    assert abs(float(g) - 1.0) < 1e-12


def test_narrow_triangles_continuous_in_width_and_position():
    """The deposit of a triangle of width w tends to the point-mass split as w -> 0
    (difference O(w)), across bin centers and bin midpoints, and its gradient with
    respect to the position matches central differences for every width."""
    for c0 in (5.0 - 1e-3, 5.0, 5.3, 5.5, 5.5 - 2e-7):
        point = _tri_spec(torch.tensor(c0, dtype=_F64)).numpy()
        for wdt in (1e-9, 1e-6, 1e-3, 0.3, 2.5):
            near = _tri_spec(torch.tensor(c0 - wdt / 2, dtype=_F64), wdt).numpy()
            assert np.abs(near - point).sum() < 2.0 * wdt + 1e-12, (c0, wdt)
            c = torch.tensor(c0 - wdt / 2, dtype=_F64, requires_grad=True)
            wts = torch.linspace(0.5, 3.0, 16, dtype=_F64) ** 2
            (g,) = torch.autograd.grad((wts * _tri_spec(c, wdt)).sum(), c)
            h = 1e-7 * max(wdt, 1e-3)
            fp = float((wts * _tri_spec(torch.tensor(c0 - wdt / 2 + h, dtype=_F64), wdt)).sum())
            fm = float((wts * _tri_spec(torch.tensor(c0 - wdt / 2 - h, dtype=_F64), wdt)).sum())
            assert abs(float(g) - (fp - fm) / (2 * h)) < 1e-5 * max(1.0, abs(float(g))), (c0, wdt)


def test_zero_width_zone_linear_split():
    pos = torch.tensor([[7.4, 7.4]], dtype=_F64, requires_grad=True)
    spec = projectzones_t(pos, torch.ones(1, 2, dtype=_F64), torch.tensor([1.0], dtype=_F64), 0.0, 1.0, 16)
    assert np.allclose(spec.detach().numpy()[7:9], [0.6, 0.4], atol=1e-12)
    (g,) = torch.autograd.grad(spec[8], pos)
    assert np.allclose(g.numpy(), [[0.5, 0.5]], atol=1e-12)


# ---------------------------------------------------------------------------
# (e) integrated intensity is conserved
# ---------------------------------------------------------------------------

def test_projection_conserves_intensity():
    """sum(spec) * dx equals the total facet intensity to roundoff for facets inside the
    range, for every facet width (measured <= 2.4e-14 relative; 0.9.34: 2.9e-12, the
    running-sum residual of short ramps)."""
    rng = np.random.default_rng(5)
    x = np.linspace(330, 350, 2048)
    dx = x[1] - x[0]
    for spread in (1e-9, 1e-3, 0.03, 1.0, 4.0):
        fun = np.clip(340 + spread * rng.standard_normal(500), 331, 349)
        fun[:30] = rng.uniform(331, 349, 30)
        amp = rng.random(500)
        tri = rng.integers(0, 500, (4000, 3))
        tri[:300, 1] = tri[:300, 0]; tri[:100, 2] = tri[:100, 0]
        areas = rng.random(4000)
        spec = pep._projecttriangles(tri, areas, fun, amp, x)
        mass = (amp[tri].mean(1) * areas).sum()
        print(spread, spec.sum() * dx / mass - 1)
        assert abs(spec.sum() * dx / mass - 1) < 2e-13, spread
        assert spec.min() >= 0.0
        segw = rng.random(499)
        spz = pep._projectzones(fun, amp, segw, x)
        mz = (0.5 * (amp[:-1] + amp[1:]) * segw).sum()
        assert abs(spz.sum() * dx / mz - 1) < 2e-13, spread


def test_pepper_stick_spectrum_area_independent_of_grid():
    """Projected stick spectrum (no lw, Harmonic 0) well inside the range: the area is the
    same on 512 and 4096 points (measured 1e-15)."""
    sys = SpinSystem(S=0.5, g=[1.98, 2.05, 2.12])
    areas = []
    for n in (512, 4096):
        x, y = pepper(sys, Experiment(mwFreq=9.5, Range=[310, 350], nPoints=n, Harmonic=0), Options())
        areas.append(float(y.sum()) * float(x[1] - x[0]))
    assert abs(areas[0] / areas[1] - 1) < 1e-12


# ---------------------------------------------------------------------------
# (f) numpy and torch paths agree; vectorised vs reference loop
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('lw', [1.0, 0.3])
def test_pepper_matches_pepper_autograd(lw):
    """Measured 1.5e-12 (lw 1) and 4.7e-12 (lw 0.3) of max, as before the change (the
    difference comes from interpolation and convolution, not from the projection)."""
    _, y = _rhombic(512, lw=lw)
    _, ya = _rhombic(512, lw=lw, fn=pepper_autograd)
    assert np.abs(y - ya).max() < 1e-11 * np.abs(y).max()


def test_numpy_front_end_equals_torch_projection():
    rng = np.random.default_rng(9)
    x = np.linspace(330, 350, 777)
    fun = 340 + 2 * rng.standard_normal(300)
    fun[:20] = x[rng.integers(0, 777, 20)]
    amp = rng.random(300)
    tri = rng.integers(0, 300, (2000, 3))
    tri[:200, 1] = tri[:200, 0]
    areas = rng.random(2000)
    a = pep._projecttriangles(tri, areas, fun, amp, x)
    b = projecttriangles_t(torch.from_numpy(tri), torch.from_numpy(areas), torch.from_numpy(fun)[None],
                           torch.from_numpy(amp)[None], float(x[0]), float(x[1] - x[0]), 777).numpy()
    assert np.array_equal(a, b)
