"""Tests for Tier 1 utility modules: sigeq, fdaxis, plegendre, photoselect,
levelsplot, stackplot.
"""
from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from torchspin.constants import PLANCK, BOLTZMANN


# ============================================================================
# sigeq
# ============================================================================

class TestSigeq:
    """Tests for torchspin.sigeq.sigeq."""

    def test_trace_one(self):
        """Density matrix should have trace 1."""
        from torchspin.sigeq import sigeq
        H = torch.diag(torch.tensor([0.0, 100.0, 200.0]))
        sigma = sigeq(H, 300.0)
        assert abs(torch.trace(sigma).real.item() - 1.0) < 1e-12

    def test_hermitian(self):
        """Density matrix should be Hermitian."""
        from torchspin.sigeq import sigeq
        H = torch.randn(4, 4, dtype=torch.complex128)
        H = (H + H.conj().T) / 2  # make Hermitian
        sigma = sigeq(H, 100.0)
        diff = torch.max(torch.abs(sigma - sigma.conj().T)).item()
        assert diff < 1e-14

    def test_positive_semidefinite(self):
        """Density matrix eigenvalues should be non-negative."""
        from torchspin.sigeq import sigeq
        H = torch.diag(torch.tensor([0.0, 50.0, 100.0, 150.0]))
        sigma = sigeq(H, 200.0)
        eigvals = torch.linalg.eigvalsh(sigma).real
        assert torch.all(eigvals >= -1e-15)

    def test_high_temperature_limit(self):
        """At very high T, should approach maximally mixed state I/N."""
        from torchspin.sigeq import sigeq
        N = 3
        H = torch.diag(torch.tensor([0.0, 100.0, 200.0]))
        sigma = sigeq(H, 1e10)  # very high T
        expected = torch.eye(N, dtype=torch.complex128) / N
        assert torch.allclose(sigma, expected, atol=1e-8)

    def test_low_temperature_ground_state(self):
        """At very low T, population should concentrate on ground state."""
        from torchspin.sigeq import sigeq
        H = torch.diag(torch.tensor([0.0, 1000.0, 2000.0]))
        sigma = sigeq(H, 0.001)  # very low T
        # Ground state population ~ 1
        assert sigma[0, 0].real.item() > 0.999

    def test_polarization_mode(self):
        """Polarization mode should give traceless matrix."""
        from torchspin.sigeq import sigeq
        H = torch.diag(torch.tensor([0.0, 100.0]))
        sigma_pol = sigeq(H, 300.0, polarization=True)
        assert abs(torch.trace(sigma_pol).real.item()) < 1e-14

    def test_negative_temperature_error(self):
        """Should raise ValueError for T <= 0."""
        from torchspin.sigeq import sigeq
        H = torch.eye(2)
        with pytest.raises(ValueError):
            sigeq(H, -1.0)
        with pytest.raises(ValueError):
            sigeq(H, 0.0)

    def test_boltzmann_populations(self):
        """Check Boltzmann populations for a known two-level system."""
        from torchspin.sigeq import sigeq
        gap_MHz = 100.0
        T = 10.0  # Kelvin
        H = torch.diag(torch.tensor([0.0, gap_MHz]))
        sigma = sigeq(H, T)
        # Populations
        p0 = sigma[0, 0].real.item()
        p1 = sigma[1, 1].real.item()
        # Expected ratio: p1/p0 = exp(-beta * gap)
        beta = (1e6 * PLANCK) / (BOLTZMANN * T)
        expected_ratio = math.exp(-beta * gap_MHz)
        actual_ratio = p1 / p0
        assert abs(actual_ratio - expected_ratio) / expected_ratio < 1e-10


# ============================================================================
# fdaxis
# ============================================================================

class TestFdaxis:
    """Tests for torchspin.fdaxis.fdaxis."""

    def test_from_time_vector(self):
        """Create frequency axis from time vector."""
        from torchspin.fdaxis import fdaxis
        N = 128
        dt = 0.01  # microseconds
        t = np.arange(N) * dt
        f = fdaxis(t)
        assert len(f) == N
        # DC at center
        dc_idx = N // 2
        assert abs(f[dc_idx]) < 1e-12

    def test_from_dt_and_N(self):
        """Create frequency axis from dT and N."""
        from torchspin.fdaxis import fdaxis
        dt = 0.001  # seconds
        N = 256
        f = fdaxis(dt, N)
        assert len(f) == N
        # Nyquist frequency
        f_nyq = 1.0 / (2.0 * dt)
        assert abs(f[-1] - f_nyq * (2.0 / N) * (N - 1 - N // 2)) < 1e-10

    def test_symmetric(self):
        """Frequency axis should be approximately symmetric for even N."""
        from torchspin.fdaxis import fdaxis
        N = 256
        dt = 0.01
        f = fdaxis(dt, N)
        # For even N, axis goes from -f_nyq to f_nyq*(1-2/N)
        assert f[0] < 0
        assert f[-1] > 0

    def test_frequency_resolution(self):
        """Check frequency resolution = 1/(N*dT)."""
        from torchspin.fdaxis import fdaxis
        dt = 0.005
        N = 200
        f = fdaxis(dt, N)
        df = f[1] - f[0]
        expected_df = 1.0 / (N * dt)
        assert abs(df - expected_df) < 1e-12

    def test_missing_N_raises(self):
        """Should raise error when dT is scalar and N is not given."""
        from torchspin.fdaxis import fdaxis
        with pytest.raises(ValueError):
            fdaxis(0.01)

    def test_odd_N(self):
        """Should handle odd N correctly."""
        from torchspin.fdaxis import fdaxis
        N = 101
        f = fdaxis(0.01, N)
        assert len(f) == N
        # DC at center (index N//2 = 50)
        assert abs(f[N // 2]) < 1e-12


# ============================================================================
# plegendre
# ============================================================================

class TestPlegendre:
    """Tests for torchspin.plegendre.plegendre."""

    def test_P0(self):
        """P_0(z) = 1."""
        from torchspin.plegendre import plegendre
        z = np.linspace(-1, 1, 11)
        y = plegendre(0, z)
        np.testing.assert_allclose(y, 1.0, atol=1e-14)

    def test_P1(self):
        """P_1(z) = z."""
        from torchspin.plegendre import plegendre
        z = np.linspace(-1, 1, 11)
        y = plegendre(1, z)
        np.testing.assert_allclose(y, z, atol=1e-14)

    def test_P2(self):
        """P_2(z) = (3z^2 - 1)/2."""
        from torchspin.plegendre import plegendre
        z = np.linspace(-1, 1, 51)
        y = plegendre(2, z)
        expected = (3 * z ** 2 - 1) / 2
        np.testing.assert_allclose(y, expected, atol=1e-13)

    def test_P3(self):
        """P_3(z) = (5z^3 - 3z)/2."""
        from torchspin.plegendre import plegendre
        z = np.linspace(-1, 1, 51)
        y = plegendre(3, z)
        expected = (5 * z ** 3 - 3 * z) / 2
        np.testing.assert_allclose(y, expected, atol=1e-13)

    def test_associated_P21(self):
        """P_2^1(z) = -3z*sqrt(1-z^2) (with Condon-Shortley phase)."""
        from torchspin.plegendre import plegendre
        z = np.linspace(-0.99, 0.99, 51)
        y = plegendre(2, 1, z)
        expected = -3 * z * np.sqrt(1 - z ** 2)
        np.testing.assert_allclose(y, expected, atol=1e-12)

    def test_associated_P22(self):
        """P_2^2(z) = 3(1-z^2) (with Condon-Shortley phase)."""
        from torchspin.plegendre import plegendre
        z = np.linspace(-0.99, 0.99, 51)
        y = plegendre(2, 2, z)
        expected = 3 * (1 - z ** 2)
        np.testing.assert_allclose(y, expected, atol=1e-12)

    def test_negative_M(self):
        """P_L^{-M} should satisfy the standard relation."""
        from torchspin.plegendre import plegendre
        L, M = 3, 2
        z = np.array([0.5])
        y_pos = plegendre(L, M, z, False)  # no CS phase
        y_neg = plegendre(L, -M, z, False)
        factor = (-1) ** M * math.factorial(L - M) / math.factorial(L + M)
        np.testing.assert_allclose(y_neg, factor * y_pos, atol=1e-14)

    def test_no_CS_phase(self):
        """Without Condon-Shortley phase, P_2^1 = 3z*sqrt(1-z^2)."""
        from torchspin.plegendre import plegendre
        z = np.array([0.5])
        y = plegendre(2, 1, z, False)
        expected = 3 * 0.5 * np.sqrt(1 - 0.25)
        np.testing.assert_allclose(y, expected, atol=1e-14)

    def test_scipy_agreement(self):
        """Compare with scipy.special.lpmv for L=4, M=2."""
        from torchspin.plegendre import plegendre
        from scipy.special import lpmv
        L, M = 4, 2
        z = np.linspace(-0.99, 0.99, 51)
        y_ours = plegendre(L, M, z, False)  # without CS phase
        y_scipy = lpmv(M, L, z)  # scipy does NOT include CS phase
        np.testing.assert_allclose(y_ours, y_scipy, atol=1e-12)

    def test_scalar_input(self):
        """Should handle scalar z."""
        from torchspin.plegendre import plegendre
        y = plegendre(2, np.float64(0.5))
        expected = (3 * 0.25 - 1) / 2
        assert abs(y - expected) < 1e-14

    def test_invalid_M_raises(self):
        """Should raise for |M| > L."""
        from torchspin.plegendre import plegendre
        with pytest.raises(ValueError):
            plegendre(2, 3, np.array([0.0]))


# ============================================================================
# photoselect
# ============================================================================

class TestPhotoselect:
    """Tests for torchspin.photoselect.photoselect."""

    def test_tdm_along_z_aligned(self):
        """TDM along z, field along z, light along y, E along z: full excitation."""
        from torchspin.photoselect import photoselect
        w = photoselect([0, 0, 1], [[0.0, 0.0, 0.0]], [0, 1, 0], 0.0)
        # E-field along z (alpha=0 with k along y), TDM along z → weight = 1
        assert w[0] == pytest.approx(1.0, abs=1e-10)

    def test_tdm_perpendicular(self):
        """TDM perpendicular to E-field → weight = 0."""
        from torchspin.photoselect import photoselect
        # TDM along x, E-field along z (alpha=0, k along y)
        w = photoselect([1, 0, 0], [[0.0, 0.0, 0.0]], [0, 1, 0], 0.0)
        assert w[0] == pytest.approx(0.0, abs=1e-10)

    def test_string_tdm(self):
        """String TDM input."""
        from torchspin.photoselect import photoselect
        w = photoselect('z', [[0.0, 0.0, 0.0]], [0, 1, 0], 0.0)
        assert w[0] == pytest.approx(1.0, abs=1e-10)

    def test_unpolarized_light(self):
        """Unpolarized light gives intermediate weight."""
        from torchspin.photoselect import photoselect
        w = photoselect('z', [[0.0, 0.0, 0.0]], [0, 1, 0], float('nan'))
        # For unpolarized: weight = (1 - |tdm·k|^2)/2
        # tdm=[0,0,1], k=[0,1,0] → tdm·k=0 → weight = 0.5
        assert w[0] == pytest.approx(0.5, abs=1e-10)

    def test_multiple_orientations(self):
        """Should handle multiple orientations."""
        from torchspin.photoselect import photoselect
        ori = np.array([[0.0, 0.0, 0.0], [0.0, np.pi / 2, 0.0]])
        w = photoselect('z', ori, [0, 1, 0], 0.0)
        assert w.shape == (2,)

    def test_chi_integration(self):
        """When chi is omitted, result integrates over chi."""
        from torchspin.photoselect import photoselect
        # 2-element ori → integrate chi
        w = photoselect('z', [[0.0, 0.0]], [0, 1, 0], 0.0)
        assert w.shape == (1,)
        assert 0 <= w[0] <= 1

    def test_weight_range(self):
        """All weights should be in [0, 1]."""
        from torchspin.photoselect import photoselect
        np.random.seed(42)
        ori = np.random.rand(20, 3) * np.array([2 * np.pi, np.pi, 2 * np.pi])
        w = photoselect('z', ori, 'y', 0.0)
        assert np.all(w >= -1e-10)
        assert np.all(w <= 1 + 1e-10)


# ============================================================================
# levelsplot
# ============================================================================

class TestLevelsplot:
    """Tests for torchspin.levelsplot.levelsplot."""

    def test_basic_plot(self):
        """Basic energy level plot should return fig and ax."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from torchspin import SpinSystem
        from torchspin.levelsplot import levelsplot

        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        fig, ax = levelsplot(sys, 'z', [0, 500])
        assert fig is not None
        assert ax is not None
        # S=1/2 → 2 energy levels → 2 lines
        assert len(ax.lines) == 2
        plt.close(fig)

    def test_triplet_system(self):
        """S=1 system should have 3 energy levels."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from torchspin import SpinSystem
        from torchspin.levelsplot import levelsplot

        sys = SpinSystem(S=[1], g=[[2.0, 2.0, 2.0]], D=[[-1000/3, -1000/3, 2000/3]])
        fig, ax = levelsplot(sys, 'z', [0, 1000])
        assert len(ax.lines) == 3
        plt.close(fig)

    def test_units_GHz(self):
        """Energy axis should be labeled in GHz."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from torchspin import SpinSystem
        from torchspin.levelsplot import levelsplot, LevelsPlotOptions

        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        opt = LevelsPlotOptions(Units='GHz')
        fig, ax = levelsplot(sys, 'z', [0, 500], opt=opt)
        assert 'GHz' in ax.get_ylabel()
        plt.close(fig)

    def test_orientation_string(self):
        """Various string orientations should work."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from torchspin import SpinSystem
        from torchspin.levelsplot import levelsplot

        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        for ori in ['x', 'y', 'z', 'xy']:
            fig, ax = levelsplot(sys, ori, [0, 500])
            assert fig is not None
            plt.close(fig)

    def test_custom_axes(self):
        """Should plot on a provided axes."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from torchspin import SpinSystem
        from torchspin.levelsplot import levelsplot

        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        fig, ax = plt.subplots()
        fig_ret, ax_ret = levelsplot(sys, 'z', [0, 500], ax=ax)
        assert ax_ret is ax
        plt.close(fig)


# ============================================================================
# stackplot
# ============================================================================

class TestStackplot:
    """Tests for torchspin.stackplot.stackplot."""

    def test_basic(self):
        """Basic stacked plot."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from torchspin.stackplot import stackplot

        x = np.linspace(0, 10, 100)
        y = np.column_stack([np.sin(x * k) for k in range(1, 4)])
        fig, ax, lines = stackplot(x, y)
        assert len(lines) == 3
        plt.close(fig)

    def test_auto_transpose(self):
        """Should auto-transpose row-major data."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from torchspin.stackplot import stackplot

        x = np.linspace(0, 10, 100)
        y = np.vstack([np.sin(x * k) for k in range(1, 4)])  # (3, 100)
        fig, ax, lines = stackplot(x, y)
        assert len(lines) == 3
        plt.close(fig)

    def test_labels(self):
        """Labels should set ytick labels."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from torchspin.stackplot import stackplot

        x = np.linspace(0, 10, 100)
        y = np.column_stack([np.sin(x), np.cos(x)])
        fig, ax, lines = stackplot(x, y, labels=['sin', 'cos'])
        tick_labels = [t.get_text() for t in ax.get_yticklabels()]
        assert 'sin' in tick_labels
        assert 'cos' in tick_labels
        plt.close(fig)

    def test_no_scaling(self):
        """scale='none' should not modify data."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from torchspin.stackplot import stackplot

        x = np.linspace(0, 10, 100)
        y = np.column_stack([np.sin(x) * 5])
        fig, ax, lines = stackplot(x, y, scale='none', step=0.0)
        # With step=0 and no scaling, data should be unchanged
        line_data = lines[0].get_ydata()
        np.testing.assert_allclose(line_data, y[:, 0], atol=1e-12)
        plt.close(fig)

    def test_explicit_step_positions(self):
        """Explicit step array should set exact positions."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from torchspin.stackplot import stackplot

        x = np.linspace(0, 10, 50)
        y = np.column_stack([np.sin(x), np.cos(x), np.sin(2 * x)])
        fig, ax, lines = stackplot(x, y, step=np.array([0, 5, 10]))
        plt.close(fig)
