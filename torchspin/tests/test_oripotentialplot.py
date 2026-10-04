"""Tests for oripotentialplot — orientation potential visualization.

Verifies:
1. Basic plotting functionality
2. Input validation
3. Both potential and population modes
4. Alpha/beta and beta/gamma angle pairs
"""

import numpy as np
import pytest

from torchspin.oripotentialplot import oripotentialplot, _lmk_sum


class TestLMKSum:
    """Test the Wigner D-matrix potential sum."""

    def test_zero_coefficients(self):
        """Zero lambda should give zero potential."""
        alpha = np.array([0.0, 1.0, 2.0])
        beta = np.array([0.5, 1.0, 1.5])
        gamma = np.zeros(3)
        result = _lmk_sum(alpha, beta, gamma,
                          lam=np.array([0.0]),
                          Lp=np.array([2]),
                          Mp=np.array([0]),
                          Kp=np.array([0]))
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_isotropic_term(self):
        """L=0 term should be constant over all orientations."""
        alpha = np.linspace(0, 2 * np.pi, 20)
        beta = np.ones(20) * np.pi / 3
        gamma = np.zeros(20)
        result = _lmk_sum(alpha, beta, gamma,
                          lam=np.array([1.0]),
                          Lp=np.array([0]),
                          Mp=np.array([0]),
                          Kp=np.array([0]))
        # D^0_00 = 1, so sum should be constant = 1.0
        np.testing.assert_allclose(result, 1.0, atol=1e-8)

    def test_l2_anisotropy(self):
        """L=2, M=0, K=0 term should vary with beta."""
        alpha = np.zeros(20)
        beta = np.linspace(0, np.pi, 20)
        gamma = np.zeros(20)
        result = _lmk_sum(alpha, beta, gamma,
                          lam=np.array([1.0]),
                          Lp=np.array([2]),
                          Mp=np.array([0]),
                          Kp=np.array([0]))
        # D^2_00(beta) = (3*cos^2(beta) - 1)/2
        # Should not be constant
        assert np.std(result) > 0.1


class TestOriPotentialPlot:
    """Test the plotting function."""

    @pytest.fixture(autouse=True)
    def setup_matplotlib(self):
        """Use non-interactive backend."""
        import matplotlib
        matplotlib.use('Agg')

    def test_basic_potential(self):
        """Basic potential plot should not crash."""
        potential = np.array([[2, 0, 0, 1.5]])
        fig, ax, vals = oripotentialplot(potential, grid_size=10)
        assert fig is not None
        assert vals.size > 0
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_population_mode(self):
        """Population mode should produce non-negative values."""
        potential = np.array([[2, 0, 0, 1.0]])
        fig, ax, vals = oripotentialplot(potential, plot_type='population',
                                          grid_size=10)
        # Population should be non-negative
        assert np.all(vals >= -1e-10)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_beta_gamma_angles(self):
        """Beta/gamma angle pair should work."""
        potential = np.array([[2, 0, 0, 1.0]])
        fig, ax, vals = oripotentialplot(potential, angles='beta_gamma',
                                          grid_size=10)
        assert vals.size > 0
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_multi_term_potential(self):
        """Multiple potential terms should work."""
        potential = np.array([
            [2, 0, 0, 1.5],
            [2, 2, 0, 0.5],
        ])
        fig, ax, vals = oripotentialplot(potential, grid_size=10)
        assert vals.size > 0
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_invalid_potential_shape(self):
        """Non-Nx4 potential should raise ValueError."""
        with pytest.raises(ValueError, match="Nx4"):
            oripotentialplot(np.array([[1, 2, 3]]))

    def test_negative_L(self):
        """Negative L should raise ValueError."""
        with pytest.raises(ValueError, match="nonnegative"):
            oripotentialplot(np.array([[-1, 0, 0, 1.0]]))

    def test_K_exceeds_L(self):
        """K > L should raise ValueError."""
        with pytest.raises(ValueError, match="K"):
            oripotentialplot(np.array([[2, 0, 5, 1.0]]))
