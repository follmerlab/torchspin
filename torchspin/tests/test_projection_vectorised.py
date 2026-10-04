"""Vectorised SOPHE projection (projecttriangles / projectzones) must reproduce
the loop port bin by bin, including degenerate triangles, bin-edge vertices,
out-of-range positions and NaN vertices."""
import importlib

import numpy as np
import pytest

pep = importlib.import_module('torchspin.pepper')


def _tri_case(rng, nPts, nTri, spread, shift=0.0, nan_frac=0.0, degen=0.0):
    fun = 340 + shift + spread * rng.standard_normal(nPts)
    amp = rng.random(nPts)
    tri = rng.integers(0, nPts, size=(nTri, 3))
    if degen:
        k = rng.random(nTri) < degen
        tri[k, 1] = tri[k, 0]
        k2 = rng.random(nTri) < degen / 2
        tri[k2, 2] = tri[k2, 0]
    if nan_frac:
        fun[rng.random(nPts) < nan_frac] = np.nan
    areas = rng.random(nTri) * 4 * np.pi / nTri
    return tri, areas, fun, amp


@pytest.mark.parametrize('spread,shift,nan_frac,degen', [
    (3, 0, 0, 0), (0.01, 0, 0, 0.3), (0.001, 0, 0, 0.5), (30, 0, 0.05, 0),
    (3, -12, 0, 0), (3, 12, 0.02, 0.1), (1e-6, 0, 0, 0.5), (0.003, 0, 0, 0)])
def test_projecttriangles_matches_loop(spread, shift, nan_frac, degen):
    rng = np.random.default_rng(1)
    x = np.linspace(330, 350, 2048)
    for _ in range(3):
        tri, areas, fun, amp = _tri_case(rng, 500, 4000, spread, shift, nan_frac, degen)
        fun[:50] = x[rng.integers(0, 2048, 50)]          # vertices exactly on bin edges
        ref = pep._projecttriangles_loop(tri, areas, fun, amp, x)
        new = pep._projecttriangles(tri, areas, fun, amp, x)
        scale = max(np.max(np.abs(ref)), 1e-300)
        assert np.allclose(ref, new, rtol=1e-9, atol=1e-9 * scale)


@pytest.mark.parametrize('spread,shift,nan_frac', [
    (3, 0, 0), (0.01, 0, 0), (1e-7, 0, 0), (30, 0, 0.05), (3, -12, 0), (3, 12, 0.02)])
def test_projectzones_matches_loop(spread, shift, nan_frac):
    rng = np.random.default_rng(2)
    x = np.linspace(330, 350, 2048)
    n = 600
    for _ in range(3):
        pos = 340 + shift + spread * rng.standard_normal(n)
        pos[:40] = x[rng.integers(0, 2048, 40)]
        pos[rng.random(n) < nan_frac] = np.nan
        segw = rng.random(n - 1)
        for amp in (rng.random(n), np.array([0.7])):
            ref = pep._projectzones_loop(pos, amp, segw, x)
            new = pep._projectzones(pos, amp, segw, x)
            scale = max(np.max(np.abs(ref)), 1e-300)
            assert np.allclose(ref, new, rtol=1e-9, atol=1e-9 * scale)


def test_projection_empty_inputs():
    x = np.linspace(330, 350, 64)
    assert np.all(pep._projecttriangles(np.zeros((0, 3), dtype=int), np.zeros(0), np.zeros(0), np.zeros(0), x) == 0)
    assert np.all(pep._projectzones(np.array([np.nan, np.nan]), np.array([1.0]), np.array([1.0]), x) == 0)
