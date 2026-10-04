"""Tests for torchspin.sphgrid."""
import math

import numpy as np
import pytest
import torch

from torchspin.sphgrid import sphgrid


def _check_unit_vectors(vecs: torch.Tensor):
    """All columns of vecs must be unit vectors."""
    norms = vecs.norm(dim=0)
    assert (norms - 1.0).abs().max().item() < 1e-12, \
        f"vecs not unit vectors: max norm error = {(norms - 1).abs().max():.2e}"


def test_sphgrid_weights_sum_dinfh():
    """Dinfh weights sum to 4π."""
    _, _, w, _ = sphgrid('Dinfh', 10)
    assert abs(w.sum().item() - 4 * math.pi) < 1e-10, \
        f"Dinfh weights sum = {w.sum():.6f}, expected {4*math.pi:.6f}"


def test_sphgrid_weights_sum_d2h():
    """D2h weights sum to 4π."""
    _, _, w, _ = sphgrid('D2h', 10)
    assert abs(w.sum().item() - 4 * math.pi) < 1e-8, \
        f"D2h weights sum = {w.sum():.6f}, expected {4*math.pi:.6f}"


def test_sphgrid_weights_sum_ci():
    """Ci weights sum to 4π."""
    _, _, w, _ = sphgrid('Ci', 10)
    assert abs(w.sum().item() - 4 * math.pi) < 1e-8, \
        f"Ci weights sum = {w.sum():.6f}, expected {4*math.pi:.6f}"


def test_sphgrid_weights_sum_c1():
    """C1 weights sum to 4π."""
    _, _, w, _ = sphgrid('C1', 10)
    assert abs(w.sum().item() - 4 * math.pi) < 1e-8, \
        f"C1 weights sum = {w.sum():.6f}, expected {4*math.pi:.6f}"


def test_sphgrid_unit_vectors_dinfh():
    """Dinfh: all vecs are unit vectors."""
    _, _, _, vecs = sphgrid('Dinfh', 15)
    _check_unit_vectors(vecs)


def test_sphgrid_unit_vectors_ci():
    """Ci: all vecs are unit vectors."""
    _, _, _, vecs = sphgrid('Ci', 15)
    _check_unit_vectors(vecs)


def test_sphgrid_theta_range_dinfh():
    """Dinfh: theta in [0, pi/2]."""
    _, theta, _, _ = sphgrid('Dinfh', 15)
    assert theta.min().item() >= -1e-12
    assert theta.max().item() <= math.pi / 2 + 1e-12


def test_sphgrid_theta_range_ci():
    """Ci: theta in [0, pi/2]."""
    _, theta, _, _ = sphgrid('Ci', 15)
    assert theta.min().item() >= -1e-12
    assert theta.max().item() <= math.pi / 2 + 1e-12


def test_sphgrid_phi_range_dinfh():
    """Dinfh: phi = 0 for all points."""
    phi, _, _, _ = sphgrid('Dinfh', 10)
    assert phi.abs().max().item() < 1e-12


def test_sphgrid_phi_range_ci():
    """Ci: phi in [0, 2π)."""
    phi, _, _, _ = sphgrid('Ci', 10)
    assert phi.min().item() >= -1e-12
    assert phi.max().item() <= 2 * math.pi + 1e-12


def test_sphgrid_gridsize_2():
    """GridSize=2 is the minimum; should not raise."""
    phi, theta, w, vecs = sphgrid('Dinfh', 2)
    assert w.sum().item() > 0
    _check_unit_vectors(vecs)


def test_sphgrid_n_points_dinfh():
    """Dinfh: N grid points for GridSize=N (one meridian)."""
    N = 17
    phi, theta, w, _ = sphgrid('Dinfh', N)
    assert phi.shape[0] == N


def test_sphgrid_weights_positive():
    """All weights must be positive."""
    for sym in ('Dinfh', 'D2h', 'Ci'):
        _, _, w, _ = sphgrid(sym, 12)
        assert w.min().item() > 0, f"{sym}: found non-positive weight"


# ---------------------------------------------------------------------------
# sphrand
# ---------------------------------------------------------------------------

class TestSphrand:
    def test_count(self):
        from torchspin.sphgrid import sphrand
        phi, theta = sphrand(100)
        assert len(phi) == 100
        assert len(theta) == 100

    def test_k4_phi_range(self):
        """k=4: phi in [0, pi/2]."""
        from torchspin.sphgrid import sphrand
        import numpy as np
        phi, theta = sphrand(2000, k=4)
        assert phi.min() >= 0
        assert phi.max() <= np.pi / 2 + 1e-10

    def test_k4_theta_range(self):
        """k=4: theta in [0, pi/2] (upper hemisphere)."""
        from torchspin.sphgrid import sphrand
        import numpy as np
        phi, theta = sphrand(2000, k=4)
        assert theta.min() >= 0
        assert theta.max() <= np.pi / 2 + 1e-10

    def test_k2_full_azimuth(self):
        """k=2: phi in [0, 2*pi]."""
        from torchspin.sphgrid import sphrand
        import numpy as np
        phi, theta = sphrand(5000, k=2)
        assert phi.max() > np.pi  # must span past pi

    def test_k1_full_sphere(self):
        """k=1: theta spans [0, pi]."""
        from torchspin.sphgrid import sphrand
        import numpy as np
        phi, theta = sphrand(5000, k=1)
        assert theta.max() > np.pi / 2  # must have theta > pi/2

    def test_k8_phi_range(self):
        """k=8: phi in [0, pi/4]."""
        from torchspin.sphgrid import sphrand
        import numpy as np
        phi, theta = sphrand(2000, k=8)
        assert phi.max() <= np.pi / 4 + 1e-10

    def test_invalid_k(self):
        from torchspin.sphgrid import sphrand
        with pytest.raises(ValueError, match="k must be"):
            sphrand(10, k=3)

    def test_uniform_cos_theta(self):
        """cos(theta) should be roughly uniform (not crowded at poles)."""
        from torchspin.sphgrid import sphrand
        import numpy as np
        phi, theta = sphrand(10000, k=1)
        cos_t = np.cos(theta)
        # cos(theta) should have roughly equal counts in each quartile
        n1 = np.sum((cos_t >= -1) & (cos_t < -0.5))
        n2 = np.sum((cos_t >= -0.5) & (cos_t < 0))
        n3 = np.sum((cos_t >= 0) & (cos_t < 0.5))
        n4 = np.sum((cos_t >= 0.5) & (cos_t <= 1))
        # All quartiles should have roughly equal count (within 15%)
        mean = np.mean([n1, n2, n3, n4])
        for n in [n1, n2, n3, n4]:
            assert abs(n - mean) / mean < 0.15
