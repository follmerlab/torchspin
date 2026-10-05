"""Tests for torchspin.orisel — orientation selectivity weights."""
import math
import numpy as np
import pytest

from torchspin.orisel import orisel, OriselOptions
from torchspin.spinsystem import SpinSystem
from torchspin.experiment import Experiment


def _make_sys_exp():
    """Simple S=1/2 rhombic spin system + X-band experiment."""
    sys = SpinSystem(S=[0.5], g=[[2.0, 2.1, 2.2]], lw=[0.5, 0.0])
    exp = Experiment(mwFreq=9.5, Range=[295, 380])
    return sys, exp


class TestOriselBasic:
    def test_returns_three_arrays(self):
        """Returns (weights, phi, theta) tuple."""
        sys, exp = _make_sys_exp()
        result = orisel(sys, exp)
        assert len(result) == 3

    def test_shapes_match(self):
        """weights, phi, theta all have the same length."""
        sys, exp = _make_sys_exp()
        weights, phi, theta = orisel(sys, exp)
        assert weights.shape == phi.shape == theta.shape

    def test_weights_nonnegative(self):
        """All weights are >= 0."""
        sys, exp = _make_sys_exp()
        weights, _, _ = orisel(sys, exp)
        assert np.all(weights >= 0)

    def test_max_weight_is_one(self):
        """Weights are normalized so max = 1.0."""
        sys, exp = _make_sys_exp()
        weights, _, _ = orisel(sys, exp)
        assert abs(weights.max() - 1.0) < 1e-10

    def test_phi_theta_range(self):
        """phi in [0, 2π), theta in [0, π]."""
        sys, exp = _make_sys_exp()
        _, phi, theta = orisel(sys, exp)
        assert np.all(phi >= -1e-10)
        assert np.all(phi <= 2 * math.pi + 1e-10)
        assert np.all(theta >= -1e-10)
        assert np.all(theta <= math.pi + 1e-10)

    def test_narrow_bandwidth_more_selective(self):
        """Narrow bandwidth → lower fraction of orientations with weight > 0.5."""
        sys, exp = _make_sys_exp()
        opt_broad = OriselOptions(GridSize=15, ExcitBandwidth=5000.0)
        opt_narrow = OriselOptions(GridSize=15, ExcitBandwidth=10.0)
        w_broad, _, _ = orisel(sys, exp, opt_broad)
        w_narrow, _, _ = orisel(sys, exp, opt_narrow)
        frac_broad = (w_broad > 0.5).sum() / len(w_broad)
        frac_narrow = (w_narrow > 0.5).sum() / len(w_narrow)
        assert frac_narrow <= frac_broad

    def test_default_options(self):
        """Calling without opt uses defaults without error."""
        sys, exp = _make_sys_exp()
        weights, phi, theta = orisel(sys, exp)
        assert len(weights) > 0

    def test_threshold_zeroes_small_weights(self):
        """Threshold > 0 sets small weights to zero."""
        sys, exp = _make_sys_exp()
        opt = OriselOptions(GridSize=15, Threshold=0.5)
        weights, _, _ = orisel(sys, exp, opt)
        # No weights between 0 and threshold
        nonzero = weights[weights > 0]
        if len(nonzero) > 0:
            assert np.all(nonzero >= 0.5)
