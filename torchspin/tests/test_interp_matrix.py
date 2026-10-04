"""The cached D2h interpolation matrix must reproduce the row-spline + bicubic
(RectBivariateSpline) G3 interpolation it replaces."""
import sys

import numpy as np
import pytest
from scipy.interpolate import CubicSpline, RectBivariateSpline

import torchspin  # noqa: F401
from torchspin.sphgrid import sphgrid

pep = sys.modules['torchspin.pepper']


def _scipy_g3(values, N_c, N_f):
    phi_rect = np.linspace(0, np.pi / 2, N_c)
    z = np.zeros((N_c, N_c)); idx = 0
    for r in range(1, N_c + 1):
        rv = values[idx:idx + r]; idx += r
        z[r - 1] = rv[0] if r == 1 else (rv if r == N_c else CubicSpline(np.linspace(0, np.pi / 2, r), rv)(phi_rect))
    spl = RectBivariateSpline(np.linspace(0, np.pi / 2, N_c), phi_rect, z, kx=3, ky=3, s=0)
    rows = np.arange(1, N_f + 1)
    th = np.repeat((rows - 1) / (N_f - 1) * np.pi / 2, rows)
    ph = np.concatenate([np.array([0.0])] + [np.linspace(0, np.pi / 2, r) for r in range(2, N_f + 1)])
    return spl.ev(th, ph)


@pytest.mark.parametrize('N_c,N_f', [(7, 25), (19, 73), (31, 121)])
def test_d2h_interp_matrix_matches_scipy(N_c, N_f):
    rng = np.random.default_rng(N_c)
    _, _, _, vc = sphgrid('D2h', N_c)
    _, _, _, vf = sphgrid('D2h', N_f)
    for _ in range(3):
        v = 340 + 3 * rng.standard_normal(vc.shape[1])
        got = pep._interp_sph(vc, vf, v)
        ref = _scipy_g3(v, N_c, N_f)
        assert got.shape == ref.shape
        assert np.abs(got - ref).max() < 1e-10
