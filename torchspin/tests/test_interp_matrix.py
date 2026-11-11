"""The cached D2h interpolation matrix must reproduce EasySpin's G3 interpolation
(gridinterp.m rectify + esspline2d): each θ row resampled along φ with
``esspline1d(y, 1, nc)`` (zero end slopes, gridinterp.m:184), then the
not-a-knot bicubic spline (RectBivariateSpline).  The row rule is written out
here from esspline1d.m rather than taken from scipy."""
import sys

import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline

import torchspin  # noqa: F401
from torchspin.sphgrid import sphgrid

pep = sys.modules['torchspin.pepper']


def _esspline1d_endslopes0(y, nn):
    """EasySpin esspline1d(y, 1, nn): spline over x = 1..n with zero first
    derivatives at both ends, evaluated at linspace(1, n, nn)."""
    y = np.asarray(y, dtype=float)
    n = y.size
    xs = np.linspace(1.0, n, nn)
    dy = np.diff(y)
    if n == 2:            # esspline1d.m:55-60, cubic Hermite with zero slopes
        s = xs - 1.0
        return y[0] + dy[0] * (3 * s ** 2 - 2 * s ** 3)
    A = np.zeros((n, n)); b = np.zeros(n)
    A[0, 0] = A[-1, -1] = 1.0                       # esspline1d.m:89-91
    for i in range(1, n - 1):
        A[i, i - 1:i + 2] = [1.0, 4.0, 1.0]
        b[i] = 3 * (dy[i - 1] + dy[i])
    slp = np.linalg.solve(A, b)
    c4 = slp[:-1] + slp[1:] - 2 * dy
    c3 = dy - slp[:-1] - c4
    k = np.minimum(np.floor(xs).astype(int), n - 1) - 1
    s = xs - (k + 1)
    return ((c4[k] * s + c3[k]) * s + slp[k]) * s + y[k]


def _easyspin_g3(values, N_c, N_f):
    phi_rect = np.linspace(0, np.pi / 2, N_c)
    z = np.zeros((N_c, N_c)); idx = 0
    for r in range(1, N_c + 1):
        rv = values[idx:idx + r]; idx += r
        z[r - 1] = rv[0] if r == 1 else (rv if r == N_c else _esspline1d_endslopes0(rv, N_c))
    spl = RectBivariateSpline(np.linspace(0, np.pi / 2, N_c), phi_rect, z, kx=3, ky=3, s=0)
    rows = np.arange(1, N_f + 1)
    th = np.repeat((rows - 1) / (N_f - 1) * np.pi / 2, rows)
    ph = np.concatenate([np.array([0.0])] + [np.linspace(0, np.pi / 2, r) for r in range(2, N_f + 1)])
    return spl.ev(th, ph)


def test_esspline1d_reference_rows():
    """The reference row rule: ends are flat, knots are reproduced, and a
    2-point row is the zero-slope cubic Hermite polynomial."""
    y = _esspline1d_endslopes0([1.0, 3.0], 5)
    assert np.allclose(y, [1.0, 1.3125, 2.0, 2.6875, 3.0])
    for n in (3, 4, 7):
        v = np.random.default_rng(n).standard_normal(n)
        yy = _esspline1d_endslopes0(v, 1 + 100 * (n - 1))
        assert np.allclose(yy[::100], v, atol=1e-13)
        assert abs(yy[1] - yy[0]) < 1e-3 * np.ptp(v) and abs(yy[-1] - yy[-2]) < 1e-3 * np.ptp(v)


@pytest.mark.parametrize('N_c,N_f', [(4, 13), (7, 25), (19, 73), (31, 121)])
def test_d2h_interp_matrix_matches_easyspin_rule(N_c, N_f):
    rng = np.random.default_rng(N_c)
    _, _, _, vc = sphgrid('D2h', N_c)
    _, _, _, vf = sphgrid('D2h', N_f)
    for _ in range(3):
        v = 340 + 3 * rng.standard_normal(vc.shape[1])
        got = pep._interp_sph(vc, vf, v)
        ref = _easyspin_g3(v, N_c, N_f)
        assert got.shape == ref.shape
        assert np.abs(got - ref).max() < 1e-10


@pytest.mark.parametrize('N_c,N_f', [(7, 25), (19, 73)])
def test_large_grid_fallback_matches_matrix(monkeypatch, N_c, N_f):
    """The per-call scipy path used above _INTERP_MATRIX_MAX follows the same rule."""
    rng = np.random.default_rng(N_f)
    _, _, _, vc = sphgrid('D2h', N_c)
    _, _, _, vf = sphgrid('D2h', N_f)
    v = 340 + 3 * rng.standard_normal(vc.shape[1])
    monkeypatch.setattr(pep, '_INTERP_MATRIX_MAX', 0)
    got = pep._interp_sph(vc, vf, v)
    assert np.abs(got - _easyspin_g3(v, N_c, N_f)).max() < 1e-10


@pytest.mark.parametrize('N_c,N_f', [(7, 25), (19, 73)])
def test_interp_grid_closedphi_rows_match_easyspin_rule(N_c, N_f):
    """_interp_grid (the rectify port used when knots are missing) resamples
    closed-φ rows with the same zero-end-slope rule when it interpolates cubically."""
    rng = np.random.default_rng(N_c + 1)
    v = 340 + 3 * rng.standard_normal(N_c * (N_c + 1) // 2)
    got = pep._interp_grid(v, 'D2h', N_c, N_f)
    assert np.abs(got - _easyspin_g3(v, N_c, N_f)).max() < 1e-10
