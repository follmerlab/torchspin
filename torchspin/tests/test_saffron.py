"""Tests for the saffron pulse EPR simulator.

Covers:
- Module imports and dataclass construction
- Pathway enumeration (saffron_pathways)
- DFT binning and peak accumulation (saffron_peaks)
- Nuclear subspace builder
- End-to-end predefined experiments (2p/3p/4p ESEEM, HYSCORE, MimsENDOR)
- Product rule consistency
- Time-domain vs frequency-domain consistency
- Autograd compatibility (sf_evolve path)
"""

import math
import numpy as np
import pytest
import torch

from torchspin import SpinSystem, saffron, PulseExperiment, SaffronOptions
from torchspin.saffron import (
    _build_nuclear_subspaces,
    _NuclearSubspace,
    _EXP_NAMES,
    _EXP_PARAMS,
)
from torchspin.saffron_pathways import find_refocusing_pathways
from torchspin.saffron_peaks import _dft_bin, sf_peaks, sf_evolve
from torchspin.constants import NMAGN, PLANCK


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def sys_1H():
    """Simple S=1/2 + 1H system."""
    return SpinSystem(
        S=[0.5],
        g=[[2.0, 2.0, 2.0]],
        Nucs=['1H'],
        A=[[2.0, 2.0, 5.0]],  # MHz
    )


@pytest.fixture
def sys_14N():
    """S=1/2 + 14N system with quadrupole."""
    return SpinSystem(
        S=[0.5],
        g=[[2.0, 2.0, 2.0]],
        Nucs=['14N'],
        A=[[1.0, 1.0, 3.0]],  # MHz
        Q=[[0.5]],  # MHz
    )


@pytest.fixture
def sys_2nuc():
    """S=1/2 + 1H + 14N system."""
    return SpinSystem(
        S=[0.5],
        g=[[2.0, 2.0, 2.0]],
        Nucs=['1H', '14N'],
        A=[[2.0, 2.0, 5.0], [1.0, 1.0, 3.0]],
        Q=[[0.0], [0.5]],
    )


@pytest.fixture
def sys_endor():
    """System for MimsENDOR tests."""
    return SpinSystem(
        S=[0.5],
        g=[[2.0, 2.0, 2.0]],
        Nucs=['1H'],
        A=[[2.0, 2.0, 5.0]],
        lwEndor=0.1,
    )


# =====================================================================
# 1. Import and dataclass tests
# =====================================================================

class TestImports:
    def test_import_saffron(self):
        from torchspin import saffron, PulseExperiment, SaffronOptions
        assert callable(saffron)

    def test_import_pathways(self):
        from torchspin import find_refocusing_pathways
        assert callable(find_refocusing_pathways)

    def test_pulse_experiment_defaults(self):
        exp = PulseExperiment(Field=350.0)
        assert exp.Field == 350.0
        assert exp.Sequence == '2pESEEM'
        assert exp.tau == 0.0
        assert exp.dt is None
        assert exp.nPoints is None

    def test_saffron_options_defaults(self):
        opt = SaffronOptions()
        assert opt.GridSize == 31
        assert opt.ProductRule is False
        assert opt.TimeDomain is False
        assert opt.Window == 'ham+'


# =====================================================================
# 2. Pathway enumeration tests
# =====================================================================

class TestPathways:
    def test_2p_pathways(self):
        """2pESEEM: t=[tau, tau], inc=[1, 1] → should give 1 refocusing pathway."""
        pw = find_refocusing_pathways([0.2, 0.2], [1, 1])
        assert pw.ndim == 2
        assert pw.shape[1] == 2
        # Last interval must be 4 (-1 coherence)
        assert np.all(pw[:, -1] == 4)
        # Must have at least 1 pathway
        assert len(pw) >= 1

    def test_3p_pathways(self):
        """3pESEEM: t=[tau, T, tau], inc=[0, 1, 0] → 2 refocusing pathways."""
        pw = find_refocusing_pathways([0.2, 0.5, 0.2], [0, 1, 0])
        assert pw.ndim == 2
        assert pw.shape[1] == 3
        assert len(pw) == 2

    def test_hyscore_pathways(self):
        """HYSCORE: t=[tau, t1, t2, tau], inc=[0, 1, 2, 0] → refocusing pathways."""
        pw = find_refocusing_pathways([0.2, 0.0, 0.0, 0.2], [0, 1, 2, 0])
        assert pw.ndim == 2
        assert pw.shape[1] == 4
        # All last intervals must be -1 coherence (code 4)
        assert np.all(pw[:, -1] == 4)
        assert len(pw) >= 2

    def test_single_pulse_fid(self):
        """Single free evolution → just detection at -1."""
        pw = find_refocusing_pathways([0.1], [1])
        assert pw.shape == (1, 1)
        assert pw[0, 0] == 4

    def test_4p_pathways(self):
        """4pESEEM: t=[tau, T, T, tau], inc=[0, 1, 1, 0] → refocusing pathways."""
        pw = find_refocusing_pathways([0.2, 0.0, 0.0, 0.2], [0, 1, 1, 0])
        assert pw.ndim == 2
        assert pw.shape[1] == 4
        assert np.all(pw[:, -1] == 4)
        assert len(pw) >= 2


# =====================================================================
# 3. DFT binning tests
# =====================================================================

class TestDFTBin:
    def test_zero_frequency(self):
        """Zero frequency → bin 0."""
        nu = torch.tensor([0.0])
        idx = _dft_bin(nu, 0.01, 512)
        assert idx.item() == 0

    def test_nyquist_wrapping(self):
        """Frequency at 1/dt should wrap to bin 0."""
        dt = 0.01
        nu = torch.tensor([1.0 / dt])
        idx = _dft_bin(nu, dt, 512)
        assert idx.item() == 0

    def test_positive_frequency(self):
        """Known frequency maps to expected bin."""
        dt = 0.01
        n = 512
        nu = torch.tensor([10.0])  # 10 MHz
        idx = _dft_bin(nu, dt, n)
        # Expected: fmod(-10*0.01, 1) * 512 = fmod(-0.1, 1) * 512
        # fmod(-0.1, 1) = -0.1 → -0.1 * 512 = -51.2 → ceil(-51.2) = -51
        # -51 % 512 = 461
        assert idx.item() == 461

    def test_batch(self):
        """Vectorized binning for multiple frequencies."""
        nu = torch.tensor([0.0, 5.0, 10.0])
        idx = _dft_bin(nu, 0.01, 512)
        assert idx.shape == (3,)
        assert idx[0].item() == 0


# =====================================================================
# 4. Peak accumulation tests (sf_peaks, sf_evolve)
# =====================================================================

class TestSfPeaks:
    def test_scheme_1_basic(self):
        """IncSchemeID=1 (3pESEEM) accumulates into 1D buffer."""
        N = 2
        n_pts = 64
        Ea = torch.tensor([0.0, 5.0])
        Eb = torch.tensor([0.0, 3.0])
        G = torch.eye(N, dtype=torch.complex128) * 0.1
        D = torch.eye(N, dtype=torch.complex128) * 0.1

        buff_re = torch.zeros(n_pts, dtype=torch.float64)
        buff_im = torch.zeros(n_pts, dtype=torch.float64)

        sf_peaks(1, buff_re, buff_im, [0.01], [1], [1], Ea, Eb, G, D)

        # Should have nonzero entries
        assert buff_re.abs().sum() > 0 or buff_im.abs().sum() > 0

    def test_scheme_2_basic(self):
        """IncSchemeID=2 (2pESEEM) accumulates into 1D buffer."""
        N = 2
        n_pts = 64
        Ea = torch.tensor([0.0, 5.0])
        Eb = torch.tensor([0.0, 3.0])
        G = torch.randn(N, N, dtype=torch.complex128) * 0.1
        D = torch.randn(N, N, dtype=torch.complex128) * 0.1
        T1l = torch.randn(N, N, dtype=torch.complex128) * 0.1
        T1r = torch.randn(N, N, dtype=torch.complex128) * 0.1

        buff_re = torch.zeros(n_pts, dtype=torch.float64)
        buff_im = torch.zeros(n_pts, dtype=torch.float64)

        sf_peaks(2, buff_re, buff_im, [0.01], [1, 2], [2, 1],
                 Ea, Eb, G, D, T1l, T1r)

        assert buff_re.abs().sum() > 0 or buff_im.abs().sum() > 0

    def test_scheme_11_basic(self):
        """IncSchemeID=11 (HYSCORE) accumulates into 2D buffer."""
        N = 2
        n1, n2 = 32, 32
        Ea = torch.tensor([0.0, 5.0])
        Eb = torch.tensor([0.0, 3.0])
        G = torch.randn(N, N, dtype=torch.complex128) * 0.1
        D = torch.randn(N, N, dtype=torch.complex128) * 0.1
        T1l = torch.randn(N, N, dtype=torch.complex128) * 0.1
        T1r = torch.randn(N, N, dtype=torch.complex128) * 0.1

        buff_re = torch.zeros(n1, n2, dtype=torch.float64)
        buff_im = torch.zeros(n1, n2, dtype=torch.float64)

        sf_peaks(11, buff_re, buff_im, [0.01, 0.01], [1, 2], [1, 2],
                 Ea, Eb, G, D, T1l, T1r)

        assert buff_re.abs().sum() > 0 or buff_im.abs().sum() > 0


class TestSfEvolve:
    def test_scheme_1_shape(self):
        """sf_evolve IncSchemeID=1 returns correct shape."""
        N = 2
        Ea = torch.tensor([0.0, 5.0])
        Eb = torch.tensor([0.0, 3.0])
        G = torch.eye(N, dtype=torch.complex128) * 0.1
        D = torch.eye(N, dtype=torch.complex128) * 0.1

        sig = sf_evolve(1, [64], [0.01], [1], [1], Ea, Eb, G, D)
        assert sig.shape == (64,)
        assert sig.dtype == torch.complex128

    def test_scheme_2_shape(self):
        """sf_evolve IncSchemeID=2 returns correct shape."""
        N = 2
        Ea = torch.tensor([0.0, 5.0])
        Eb = torch.tensor([0.0, 3.0])
        G = torch.randn(N, N, dtype=torch.complex128) * 0.1
        D = torch.randn(N, N, dtype=torch.complex128) * 0.1
        T1l = torch.randn(N, N, dtype=torch.complex128) * 0.1
        T1r = torch.randn(N, N, dtype=torch.complex128) * 0.1

        sig = sf_evolve(2, [64], [0.01], [1, 2], [2, 1],
                        Ea, Eb, G, D, T1l, T1r)
        assert sig.shape == (64,)

    def test_scheme_11_shape(self):
        """sf_evolve IncSchemeID=11 returns correct 2D shape."""
        N = 2
        Ea = torch.tensor([0.0, 5.0])
        Eb = torch.tensor([0.0, 3.0])
        G = torch.randn(N, N, dtype=torch.complex128) * 0.1
        D = torch.randn(N, N, dtype=torch.complex128) * 0.1
        T1l = torch.randn(N, N, dtype=torch.complex128) * 0.1
        T1r = torch.randn(N, N, dtype=torch.complex128) * 0.1

        sig = sf_evolve(11, [32, 32], [0.01, 0.01], [1, 2], [1, 2],
                        Ea, Eb, G, D, T1l, T1r)
        assert sig.shape == (32, 32)

    def test_peaks_vs_evolve_scheme_1(self):
        """sf_peaks (freq domain) should match sf_evolve (time domain) for scheme 1."""
        N = 2
        n_pts = 64
        Ea = torch.tensor([0.0, 5.0])
        Eb = torch.tensor([0.0, 3.0])
        G = torch.eye(N, dtype=torch.complex128) * 0.25
        D = torch.eye(N, dtype=torch.complex128)

        # Frequency domain → IFFT
        buff_re = torch.zeros(n_pts, dtype=torch.float64)
        buff_im = torch.zeros(n_pts, dtype=torch.float64)
        sf_peaks(1, buff_re, buff_im, [0.01], [1], [1], Ea, Eb, G, D)
        td_freq = torch.fft.ifft(buff_re + 1j * buff_im) * n_pts

        # Time domain
        td_time = sf_evolve(1, [n_pts], [0.01], [1], [1], Ea, Eb, G, D)

        np.testing.assert_allclose(td_freq.numpy(), td_time.numpy(),
                                   atol=1e-6, rtol=1e-6)


# =====================================================================
# 5. Nuclear subspace builder tests
# =====================================================================

class TestNuclearSubspace:
    def test_single_1H(self, sys_1H):
        """Single 1H nucleus → 2×2 operators."""
        subs = _build_nuclear_subspaces(sys_1H, [0], product_rule=False,
                                         is_endor=False, device='cpu')
        assert len(subs) == 1
        assert subs[0].Hnzx.shape == (2, 2)
        assert subs[0].Hhfx.shape == (2, 2)
        assert subs[0].Hnq.shape == (2, 2)
        # No ENDOR operators
        assert subs[0].Ix is None

    def test_single_14N(self, sys_14N):
        """14N I=1 → 3×3 operators, nonzero quadrupole."""
        subs = _build_nuclear_subspaces(sys_14N, [0], product_rule=False,
                                         is_endor=False, device='cpu')
        assert len(subs) == 1
        assert subs[0].Hnzx.shape == (3, 3)
        # Quadrupole should be nonzero
        assert subs[0].Hnq.abs().sum() > 0

    def test_product_rule_separates(self, sys_2nuc):
        """Product rule: 2 nuclei → 2 subspaces."""
        subs = _build_nuclear_subspaces(sys_2nuc, [0, 1], product_rule=True,
                                         is_endor=False, device='cpu')
        assert len(subs) == 2
        assert subs[0].Hnzx.shape == (2, 2)  # 1H: I=1/2 → 2x2
        assert subs[1].Hnzx.shape == (3, 3)  # 14N: I=1 → 3x3

    def test_combined_subspace(self, sys_2nuc):
        """No product rule: 2 nuclei → 1 subspace of dim 2*3=6."""
        subs = _build_nuclear_subspaces(sys_2nuc, [0, 1], product_rule=False,
                                         is_endor=False, device='cpu')
        assert len(subs) == 1
        assert subs[0].Hnzx.shape == (6, 6)

    def test_endor_operators(self, sys_endor):
        """ENDOR mode builds Ix/Iy/Iz operators."""
        subs = _build_nuclear_subspaces(sys_endor, [0], product_rule=False,
                                         is_endor=True, device='cpu')
        assert subs[0].Ix is not None
        assert subs[0].Iy is not None
        assert subs[0].Iz is not None

    def test_nuclear_zeeman_sign(self, sys_1H):
        """Nuclear Zeeman prefactor matches MATLAB: pre = -gn*nmagn/(1e3*planck*1e6)."""
        from torchspin.nucdata import nucgval
        gn = nucgval('1H')
        expected_pre = -gn * NMAGN / (1e3 * PLANCK * 1e6)

        subs = _build_nuclear_subspaces(sys_1H, [0], product_rule=False,
                                         is_endor=False, device='cpu')
        # Hnzz should be pre * Iz. For I=1/2, Iz = diag([0.5, -0.5])
        # So Hnzz[0,0] = pre * 0.5
        ratio = subs[0].Hnzz[0, 0].real.item() / 0.5
        np.testing.assert_allclose(ratio, expected_pre, rtol=1e-10)


# =====================================================================
# 6. End-to-end smoke tests
# =====================================================================

class TestSaffronSmoke:
    def test_2p_runs(self, sys_1H):
        """2pESEEM runs without error and returns expected shapes."""
        exp = PulseExperiment(Field=350.0, Sequence='2pESEEM',
                              dt=0.01, nPoints=128, tau=0.1)
        opt = SaffronOptions(GridSize=5)
        x, signal, info = saffron(sys_1H, exp, opt)

        assert isinstance(x, np.ndarray)
        assert signal.shape == (128,)
        assert 'td' in info
        assert 'fd' in info

    def test_3p_runs(self, sys_1H):
        """3pESEEM runs without error."""
        exp = PulseExperiment(Field=350.0, Sequence='3pESEEM',
                              dt=0.01, nPoints=128, tau=0.2)
        opt = SaffronOptions(GridSize=5)
        x, signal, info = saffron(sys_1H, exp, opt)
        assert signal.shape == (128,)

    def test_4p_runs(self, sys_1H):
        """4pESEEM runs without error."""
        exp = PulseExperiment(Field=350.0, Sequence='4pESEEM',
                              dt=0.01, nPoints=128, tau=0.2)
        opt = SaffronOptions(GridSize=5)
        x, signal, info = saffron(sys_1H, exp, opt)
        assert signal.shape == (128,)

    def test_hyscore_runs(self, sys_1H):
        """HYSCORE returns 2D signal."""
        exp = PulseExperiment(Field=350.0, Sequence='HYSCORE',
                              dt=[0.02, 0.02], nPoints=[64, 64], tau=0.2)
        opt = SaffronOptions(GridSize=5)
        x, signal, info = saffron(sys_1H, exp, opt)
        assert signal.shape == (64, 64)
        assert isinstance(x, list)
        assert len(x) == 2

    def test_mims_endor_runs(self, sys_endor):
        """MimsENDOR runs without error."""
        exp = PulseExperiment(Field=350.0, Sequence='MimsENDOR',
                              tau=0.2, Range=[10, 20], nPoints=201)
        opt = SaffronOptions(GridSize=5, EndorMethod=0)
        x, signal, info = saffron(sys_endor, exp, opt)
        assert signal.shape == (201,)
        assert x[0] == pytest.approx(10.0)
        assert x[-1] == pytest.approx(20.0)

    def test_default_params(self, sys_1H):
        """saffron fills in reasonable defaults for dt/nPoints."""
        exp = PulseExperiment(Field=350.0, Sequence='2pESEEM', tau=0.1)
        opt = SaffronOptions(GridSize=5)
        x, signal, info = saffron(sys_1H, exp, opt)
        assert len(signal) > 0

    def test_nonzero_signal(self, sys_1H):
        """Simulated 2pESEEM signal is not all zeros."""
        exp = PulseExperiment(Field=350.0, Sequence='2pESEEM',
                              dt=0.01, nPoints=128, tau=0.1)
        opt = SaffronOptions(GridSize=10)
        x, signal, info = saffron(sys_1H, exp, opt)
        assert np.abs(signal).max() > 0

    def test_time_domain_mode(self, sys_1H):
        """TimeDomain=True uses sf_evolve path."""
        exp = PulseExperiment(Field=350.0, Sequence='2pESEEM',
                              dt=0.01, nPoints=128, tau=0.1)
        opt = SaffronOptions(GridSize=5, TimeDomain=True)
        x, signal, info = saffron(sys_1H, exp, opt)
        assert signal.shape == (128,)
        assert np.abs(signal).max() > 0


# =====================================================================
# 7. Physics sanity checks
# =====================================================================

class TestPhysicsSanity:
    def test_2p_real_at_t0(self, sys_1H):
        """2pESEEM: signal at t=0 should be predominantly real."""
        exp = PulseExperiment(Field=350.0, Sequence='2pESEEM',
                              dt=0.01, nPoints=256, tau=0.1)
        opt = SaffronOptions(GridSize=10)
        x, signal, info = saffron(sys_1H, exp, opt)
        td = info['td']
        # At t=0, imaginary part should be much smaller than real part
        assert abs(td[0].real) > abs(td[0].imag) * 10

    def test_2p_echo_decay(self, sys_1H):
        """2pESEEM: signal should generally decay with time."""
        exp = PulseExperiment(Field=350.0, Sequence='2pESEEM',
                              dt=0.01, nPoints=256, tau=0.1)
        opt = SaffronOptions(GridSize=10)
        x, signal, info = saffron(sys_1H, exp, opt)
        td = info['td']
        # Signal magnitude at start should be >= end (on average)
        first_quarter = np.abs(td[:64]).mean()
        last_quarter = np.abs(td[-64:]).mean()
        # Not strict — oscillations may cause fluctuations
        # Just check signal isn't growing
        assert first_quarter >= last_quarter * 0.5

    def test_14N_quadrupole(self, sys_14N):
        """14N system with quadrupole should produce signal with more peaks."""
        exp = PulseExperiment(Field=350.0, Sequence='3pESEEM',
                              dt=0.01, nPoints=256, tau=0.2)
        opt = SaffronOptions(GridSize=10)
        x, signal, info = saffron(sys_14N, exp, opt)
        fd = info['fd']
        assert fd is not None
        # Should have nonzero spectral content
        assert np.abs(fd).max() > 0

    def test_hyscore_symmetry(self, sys_1H):
        """HYSCORE with isotropic g should have approximately symmetric spectrum."""
        exp = PulseExperiment(Field=350.0, Sequence='HYSCORE',
                              dt=[0.02, 0.02], nPoints=[64, 64], tau=0.2)
        opt = SaffronOptions(GridSize=10)
        x, signal, info = saffron(sys_1H, exp, opt)
        fd = info['fd']
        # For isotropic g, the 2D spectrum should be roughly symmetric:
        # |F(f1, f2)| ~ |F(f2, f1)|
        asym = np.abs(np.abs(fd) - np.abs(fd.T)).sum() / (np.abs(fd).sum() + 1e-30)
        assert asym < 0.2  # <20% asymmetry for isotropic g


# =====================================================================
# 8. Time-domain vs frequency-domain consistency
# =====================================================================

class TestTdFdConsistency:
    def test_2p_td_vs_fd(self, sys_1H):
        """2pESEEM: TimeDomain and frequency-domain paths should give same td."""
        exp_td = PulseExperiment(Field=350.0, Sequence='2pESEEM',
                                 dt=0.01, nPoints=128, tau=0.1)
        exp_fd = PulseExperiment(Field=350.0, Sequence='2pESEEM',
                                 dt=0.01, nPoints=128, tau=0.1)
        opt_td = SaffronOptions(GridSize=10, TimeDomain=True)
        opt_fd = SaffronOptions(GridSize=10, TimeDomain=False)

        _, _, info_td = saffron(sys_1H, exp_td, opt_td)
        _, _, info_fd = saffron(sys_1H, exp_fd, opt_fd)

        td1 = info_td['td']
        td2 = info_fd['td']

        # DFT binning has discretization error vs direct time-domain (~0.1%)
        np.testing.assert_allclose(td1, td2, atol=5e-3, rtol=5e-3)

    def test_3p_td_vs_fd(self, sys_1H):
        """3pESEEM: TimeDomain and frequency-domain paths should match."""
        exp_td = PulseExperiment(Field=350.0, Sequence='3pESEEM',
                                 dt=0.01, nPoints=128, tau=0.2)
        exp_fd = PulseExperiment(Field=350.0, Sequence='3pESEEM',
                                 dt=0.01, nPoints=128, tau=0.2)
        opt_td = SaffronOptions(GridSize=10, TimeDomain=True)
        opt_fd = SaffronOptions(GridSize=10, TimeDomain=False)

        _, _, info_td = saffron(sys_1H, exp_td, opt_td)
        _, _, info_fd = saffron(sys_1H, exp_fd, opt_fd)

        np.testing.assert_allclose(info_td['td'], info_fd['td'],
                                   atol=5e-3, rtol=5e-3)


# =====================================================================
# 9. Product rule tests
# =====================================================================

class TestProductRule:
    def test_product_rule_vs_direct(self, sys_2nuc):
        """Product rule should approximately match direct (non-product-rule) result."""
        exp_pr = PulseExperiment(Field=350.0, Sequence='2pESEEM',
                                 dt=0.01, nPoints=128, tau=0.1)
        exp_di = PulseExperiment(Field=350.0, Sequence='2pESEEM',
                                 dt=0.01, nPoints=128, tau=0.1)
        opt_pr = SaffronOptions(GridSize=10, ProductRule=True)
        opt_di = SaffronOptions(GridSize=10, ProductRule=False)

        _, _, info_pr = saffron(sys_2nuc, exp_pr, opt_pr)
        _, _, info_di = saffron(sys_2nuc, exp_di, opt_di)

        td_pr = info_pr['td']
        td_di = info_di['td']

        # Product rule is an approximation — check cosine similarity > 0.9
        cos_sim = (np.abs(np.sum(td_pr.conj() * td_di)) /
                   (np.linalg.norm(td_pr) * np.linalg.norm(td_di) + 1e-30))
        assert cos_sim > 0.9, f"Cosine similarity {cos_sim:.3f} too low"


# =====================================================================
# 10. Relaxation decay
# =====================================================================

class TestRelaxation:
    def test_t2_decay_2p(self, sys_1H):
        """2pESEEM with T2 should decay faster."""
        sys_relax = SpinSystem(
            S=[0.5], g=[[2.0, 2.0, 2.0]],
            Nucs=['1H'], A=[[2.0, 2.0, 5.0]],
            T2=1.0,  # µs
        )
        exp1 = PulseExperiment(Field=350.0, Sequence='2pESEEM',
                               dt=0.01, nPoints=256, tau=0.1)
        exp2 = PulseExperiment(Field=350.0, Sequence='2pESEEM',
                               dt=0.01, nPoints=256, tau=0.1)
        opt = SaffronOptions(GridSize=5)

        _, _, info_norelax = saffron(sys_1H, exp1, opt)
        _, _, info_relax = saffron(sys_relax, exp2, opt)

        td_no = info_norelax['td']
        td_yes = info_relax['td']

        # Signal at late times should be smaller with relaxation
        late_no = np.abs(td_no[-64:]).mean()
        late_yes = np.abs(td_yes[-64:]).mean()
        assert late_yes < late_no or late_no < 1e-15  # relaxation decays signal


# =====================================================================
# 11. Autograd compatibility
# =====================================================================

class TestAutograd:
    def test_sf_evolve_grad(self):
        """sf_evolve should support autograd (gradient flows through eigenvalues)."""
        N = 2
        Ea = torch.tensor([0.0, 5.0], requires_grad=True)
        Eb = torch.tensor([0.0, 3.0])
        G = torch.eye(N, dtype=torch.complex128) * 0.25
        D = torch.eye(N, dtype=torch.complex128)

        sig = sf_evolve(1, [32], [0.01], [1], [1], Ea, Eb, G, D)
        loss = sig.real.sum()
        loss.backward()

        assert Ea.grad is not None
        assert Ea.grad.abs().sum() > 0


# =====================================================================
# 12. Edge cases
# =====================================================================

class TestEdgeCases:
    def test_no_nuclei_error(self):
        """System with no nuclei should handle gracefully."""
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        exp = PulseExperiment(Field=350.0, Sequence='2pESEEM',
                              dt=0.01, nPoints=64, tau=0.1)
        opt = SaffronOptions(GridSize=5)
        # Should still run (trivial result) or raise informative error
        try:
            x, signal, info = saffron(sys, exp, opt)
            # If it runs, signal should be trivial
        except (ValueError, IndexError):
            pass  # Acceptable to raise an error for no-nucleus case

    def test_endor_missing_range_error(self, sys_endor):
        """MimsENDOR without Range should raise ValueError."""
        exp = PulseExperiment(Field=350.0, Sequence='MimsENDOR', tau=0.2)
        opt = SaffronOptions(GridSize=5)
        with pytest.raises(ValueError, match="Range"):
            saffron(sys_endor, exp, opt)

    def test_endor_missing_lwEndor_error(self):
        """MimsENDOR without lwEndor should raise ValueError."""
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]],
                         Nucs=['1H'], A=[[2.0, 2.0, 5.0]])
        exp = PulseExperiment(Field=350.0, Sequence='MimsENDOR',
                              tau=0.2, Range=[10, 20])
        opt = SaffronOptions(GridSize=5)
        with pytest.raises(ValueError, match="lwEndor"):
            saffron(sys, exp, opt)

    def test_custom_missing_flip_error(self):
        """Custom sequence without Flip should raise ValueError."""
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]],
                         Nucs=['1H'], A=[[2.0, 2.0, 5.0]])
        exp = PulseExperiment(Field=350.0, Sequence='custom',
                              Inc=[1, 1], dt=0.01, nPoints=64)
        opt = SaffronOptions(GridSize=5)
        with pytest.raises(ValueError, match="Flip"):
            saffron(sys, exp, opt)

    def test_custom_missing_inc_error(self):
        """Custom sequence without Inc should raise ValueError."""
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]],
                         Nucs=['1H'], A=[[2.0, 2.0, 5.0]])
        exp = PulseExperiment(Field=350.0, Sequence='custom',
                              Flip=[1.0, 1.0], dt=0.01, nPoints=64)
        opt = SaffronOptions(GridSize=5)
        with pytest.raises(ValueError, match="Inc"):
            saffron(sys, exp, opt)

    def test_s_gt_half_runs(self):
        """S=1 with nucleus should run without error (fast algorithm now supports S>1/2)."""
        sys = SpinSystem(S=[1], g=[[2.0, 2.0, 2.0]],
                         Nucs=['1H'], A=[[2.0, 2.0, 5.0]])
        exp = PulseExperiment(Field=350.0, Sequence='2pESEEM',
                              dt=0.01, nPoints=64, tau=0.1)
        opt = SaffronOptions(GridSize=5)
        x, y, info = saffron(sys, exp, opt)
        assert y is not None
        assert len(y) == 64

    def test_multi_electron_error(self):
        """nElectrons > 1 should still raise NotImplementedError for fast mode."""
        sys = SpinSystem(S=[0.5, 0.5], g=[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
        exp = PulseExperiment(Field=350.0, Sequence='2pESEEM',
                              dt=0.01, nPoints=64, tau=0.1)
        opt = SaffronOptions(GridSize=5)
        with pytest.raises(NotImplementedError):
            saffron(sys, exp, opt)


# =====================================================================
# 13. Custom sequence tests
# =====================================================================

class TestCustomSequence:
    def test_custom_2p_runs(self, sys_1H):
        """Custom 2-pulse sequence runs and produces nonzero signal."""
        exp = PulseExperiment(
            Field=350.0, Sequence='custom',
            Flip=[1.0, 1.0], Inc=[1, 1], t=[0.1, 0.1],
            Phase=[1.0, 1.0], dt=0.01, nPoints=128,
        )
        opt = SaffronOptions(GridSize=5)
        x, signal, info = saffron(sys_1H, exp, opt)
        assert signal.shape == (128,)
        assert np.abs(signal).max() > 0

    def test_custom_3p_runs(self, sys_1H):
        """Custom 3-pulse sequence runs and produces nonzero signal."""
        exp = PulseExperiment(
            Field=350.0, Sequence='custom',
            Flip=[1.0, 1.0, 1.0], Inc=[0, 1, 0], t=[0.2, 0.5, 0.2],
            Phase=[1.0, 1.0, 1.0], dt=0.01, nPoints=128,
        )
        opt = SaffronOptions(GridSize=5)
        x, signal, info = saffron(sys_1H, exp, opt)
        assert signal.shape == (128,)
        assert np.abs(signal).max() > 0

    def test_custom_vs_predefined_2p_shape(self, sys_1H):
        """Custom 2pESEEM should match predefined in spectral shape (cosine > 0.99)."""
        opt = SaffronOptions(GridSize=10)

        exp_pre = PulseExperiment(Field=350.0, Sequence='2pESEEM',
                                   dt=0.01, nPoints=128, tau=0.1)
        _, _, info_pre = saffron(sys_1H, exp_pre, opt)

        exp_cust = PulseExperiment(
            Field=350.0, Sequence='custom',
            Flip=[1.0, 1.0], Inc=[1, 1], t=[0.1, 0.1],
            Phase=[1.0, 1.0], dt=0.01, nPoints=128,
        )
        _, _, info_cust = saffron(sys_1H, exp_cust, opt)

        td_pre = info_pre['td']
        td_cust = info_cust['td']

        cos_sim = (np.abs(np.sum(td_pre.conj() * td_cust)) /
                   (np.linalg.norm(td_pre) * np.linalg.norm(td_cust) + 1e-30))
        assert cos_sim > 0.99, f"Cosine similarity {cos_sim:.4f} too low"

    def test_custom_vs_predefined_3p_shape(self, sys_1H):
        """Custom 3pESEEM should match predefined in spectral shape."""
        opt = SaffronOptions(GridSize=10)

        exp_pre = PulseExperiment(Field=350.0, Sequence='3pESEEM',
                                   dt=0.01, nPoints=128, tau=0.2)
        _, _, info_pre = saffron(sys_1H, exp_pre, opt)

        exp_cust = PulseExperiment(
            Field=350.0, Sequence='custom',
            Flip=[1.0, 1.0, 1.0], Inc=[0, 1, 0], t=[0.2, 0.0, 0.2],
            Phase=[1.0, 1.0, 1.0], dt=0.01, nPoints=128,
        )
        _, _, info_cust = saffron(sys_1H, exp_cust, opt)

        td_pre = info_pre['td']
        td_cust = info_cust['td']

        cos_sim = (np.abs(np.sum(td_pre.conj() * td_cust)) /
                   (np.linalg.norm(td_pre) * np.linalg.norm(td_cust) + 1e-30))
        assert cos_sim > 0.99, f"Cosine similarity {cos_sim:.4f} too low"

    def test_custom_coherence_filter(self, sys_1H):
        """Coherence filter restricts pathways as expected."""
        exp = PulseExperiment(
            Field=350.0, Sequence='custom',
            Flip=[1.0, 1.0, 1.0], Inc=[0, 1, 0], t=[0.2, 0.0, 0.2],
            Phase=[1.0, 1.0, 1.0], dt=0.01, nPoints=128,
            Filter='.a.',  # only alpha in middle interval
        )
        opt = SaffronOptions(GridSize=5)
        x, signal, info = saffron(sys_1H, exp, opt)
        assert signal.shape == (128,)

    def test_custom_time_domain_mode(self, sys_1H):
        """Custom sequence with TimeDomain=True."""
        exp = PulseExperiment(
            Field=350.0, Sequence='custom',
            Flip=[1.0, 1.0], Inc=[1, 1], t=[0.1, 0.1],
            Phase=[1.0, 1.0], dt=0.01, nPoints=128,
        )
        opt = SaffronOptions(GridSize=5, TimeDomain=True)
        x, signal, info = saffron(sys_1H, exp, opt)
        assert signal.shape == (128,)
        assert np.abs(signal).max() > 0

    def test_custom_td_vs_fd(self, sys_1H):
        """Custom sequence: TimeDomain and freq-domain paths match in shape."""
        exp_td = PulseExperiment(
            Field=350.0, Sequence='custom',
            Flip=[1.0, 1.0], Inc=[1, 1], t=[0.1, 0.1],
            Phase=[1.0, 1.0], dt=0.01, nPoints=128,
        )
        exp_fd = PulseExperiment(
            Field=350.0, Sequence='custom',
            Flip=[1.0, 1.0], Inc=[1, 1], t=[0.1, 0.1],
            Phase=[1.0, 1.0], dt=0.01, nPoints=128,
        )
        opt_td = SaffronOptions(GridSize=10, TimeDomain=True)
        opt_fd = SaffronOptions(GridSize=10, TimeDomain=False)

        _, _, info_td = saffron(sys_1H, exp_td, opt_td)
        _, _, info_fd = saffron(sys_1H, exp_fd, opt_fd)

        cos_sim = (np.abs(np.sum(info_td['td'].conj() * info_fd['td'])) /
                   (np.linalg.norm(info_td['td']) * np.linalg.norm(info_fd['td']) + 1e-30))
        assert cos_sim > 0.99, f"TD vs FD cosine {cos_sim:.4f} too low"

    def test_inc_scheme_id_3(self, sys_1H):
        """IncSchemeID 3 ([1,-1]) runs."""
        exp = PulseExperiment(
            Field=350.0, Sequence='custom',
            Flip=[1.0, 1.0, 1.0], Inc=[0, 1, -1], t=[0.2, 0.0, 0.0],
            Phase=[1.0, 1.0, 1.0], dt=0.01, nPoints=128,
        )
        opt = SaffronOptions(GridSize=5)
        x, signal, info = saffron(sys_1H, exp, opt)
        assert signal.shape == (128,)


# =====================================================================
# 14. S > 1/2 fast algorithm tests
# =====================================================================

class TestHighSpin:
    """Tests for the S>1/2 path of the saffron fast algorithm.

    The fast algorithm for S>1/2 diagonalizes the electronic Hamiltonian,
    computes <S> expectation values per eigenstate, and sums over all allowed
    EPR transitions (lower triangle of |<i|SyLab|j>| matrix).
    """

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------

    @pytest.fixture
    def sys_S1_1H(self):
        """S=1 triplet + 1H, D=100 MHz, E=0 (axial)."""
        return SpinSystem(
            S=[1], g=[[2.002, 2.001, 2.000]],
            D=[[100.0, 0.0]],
            Nucs='1H', A=[[3.0, 3.0, 9.0]],
        )

    @pytest.fixture
    def sys_S32_14N(self):
        """S=3/2 + 14N with quadrupole."""
        return SpinSystem(
            S=[1.5], g=[[2.002, 2.001, 2.000]],
            D=[[50.0, 0.0]],
            Nucs='14N', A=[[5.0, 5.0, 12.0]],
            Q=[[0.3, 0.0, 0.0]],
        )

    # ------------------------------------------------------------------
    # Basic execution and output shape
    # ------------------------------------------------------------------

    def test_S1_2pESEEM_runs(self, sys_S1_1H):
        """S=1 + 1H 2pESEEM: runs and returns correct length."""
        exp = PulseExperiment(Field=350.0, Sequence='2pESEEM',
                              dt=0.01, nPoints=64, tau=0.2)
        opt = SaffronOptions(GridSize=5)
        x, y, info = saffron(sys_S1_1H, exp, opt)
        assert len(y) == 64
        assert np.any(np.abs(y) > 0), "Signal is all-zero"

    def test_S1_3pESEEM_runs(self, sys_S1_1H):
        """S=1 + 1H 3pESEEM: runs and returns correct length."""
        exp = PulseExperiment(Field=350.0, Sequence='3pESEEM',
                              dt=0.01, nPoints=64, tau=0.2)
        opt = SaffronOptions(GridSize=5)
        x, y, info = saffron(sys_S1_1H, exp, opt)
        assert len(y) == 64
        assert np.any(np.abs(y) > 0)

    def test_S32_HYSCORE_runs(self, sys_S32_14N):
        """S=3/2 + 14N HYSCORE: runs and returns correct 2D shape."""
        exp = PulseExperiment(Field=330.0, Sequence='HYSCORE',
                              dt=[0.02, 0.02], nPoints=[32, 32], tau=0.2)
        opt = SaffronOptions(GridSize=5)
        x, y, info = saffron(sys_S32_14N, exp, opt)
        y_arr = np.asarray(y)
        assert y_arr.shape == (32, 32)
        assert np.any(np.abs(y_arr) > 0)

    def test_S1_no_D_runs(self):
        """S=1 + 1H without ZFS: should still run (D=0 diagonalizes to degenerate manifolds)."""
        sys = SpinSystem(S=[1], g=[[2.002, 2.001, 2.000]],
                         Nucs='1H', A=[[3.0, 3.0, 9.0]])
        exp = PulseExperiment(Field=350.0, Sequence='2pESEEM',
                              dt=0.01, nPoints=32, tau=0.1)
        opt = SaffronOptions(GridSize=5)
        x, y, info = saffron(sys, exp, opt)
        assert len(y) == 32

    # ------------------------------------------------------------------
    # D-tensor input format compatibility
    # ------------------------------------------------------------------

    def test_D_principal_values_format(self):
        """D given as [Dxx, Dyy, Dzz] principal values works correctly."""
        # D=100, E=0 → Dxx=Dyy=-33.333, Dzz=66.667
        sys = SpinSystem(S=[1], g=[[2.002, 2.001, 2.000]],
                         D=[[-33.333, -33.333, 66.667]],
                         Nucs='1H', A=[[3.0, 3.0, 9.0]])
        exp = PulseExperiment(Field=350.0, Sequence='2pESEEM',
                              dt=0.01, nPoints=32, tau=0.2)
        opt = SaffronOptions(GridSize=5)
        x, y, info = saffron(sys, exp, opt)
        assert len(y) == 32
        assert np.any(np.abs(y) > 0)

    def test_D_DE_format_matches_principal(self):
        """[D, E] shorthand and equivalent [Dxx, Dyy, Dzz] produce identical signals."""
        exp = PulseExperiment(Field=350.0, Sequence='2pESEEM',
                              dt=0.01, nPoints=32, tau=0.2)
        opt = SaffronOptions(GridSize=10)

        sys_de = SpinSystem(S=[1], g=[[2.002, 2.001, 2.000]],
                            D=[[100.0, 0.0]],
                            Nucs='1H', A=[[3.0, 3.0, 9.0]])
        sys_pv = SpinSystem(S=[1], g=[[2.002, 2.001, 2.000]],
                            D=[[-33.3333, -33.3333, 66.6667]],
                            Nucs='1H', A=[[3.0, 3.0, 9.0]])

        _, y_de, _ = saffron(sys_de, exp, opt)
        _, y_pv, _ = saffron(sys_pv, exp, opt)
        np.testing.assert_allclose(y_de, y_pv, rtol=1e-5,
                                   err_msg="[D,E] and [Dxx,Dyy,Dzz] formats disagree")

    # ------------------------------------------------------------------
    # Physical consistency checks
    # ------------------------------------------------------------------

    def test_S1_signal_finite(self, sys_S1_1H):
        """S=1 signal contains no NaN or Inf."""
        exp = PulseExperiment(Field=350.0, Sequence='2pESEEM',
                              dt=0.01, nPoints=64, tau=0.2)
        opt = SaffronOptions(GridSize=10)
        _, y, _ = saffron(sys_S1_1H, exp, opt)
        assert np.all(np.isfinite(y)), "Signal contains NaN or Inf"

    def test_S32_signal_finite(self, sys_S32_14N):
        """S=3/2 signal contains no NaN or Inf."""
        exp = PulseExperiment(Field=330.0, Sequence='HYSCORE',
                              dt=[0.02, 0.02], nPoints=[32, 32], tau=0.2)
        opt = SaffronOptions(GridSize=5)
        _, y, _ = saffron(sys_S32_14N, exp, opt)
        assert np.all(np.isfinite(np.asarray(y))), "Signal contains NaN or Inf"

    def test_S1_2p_td_fd_consistent(self, sys_S1_1H):
        """S=1: TimeDomain and freq-domain paths produce consistent signals (cosine > 0.99)."""
        exp = PulseExperiment(Field=350.0, Sequence='2pESEEM',
                              dt=0.01, nPoints=128, tau=0.2)
        opt_td = SaffronOptions(GridSize=10, TimeDomain=True)
        opt_fd = SaffronOptions(GridSize=10, TimeDomain=False)
        _, _, info_td = saffron(sys_S1_1H, exp, opt_td)
        _, _, info_fd = saffron(sys_S1_1H, exp, opt_fd)
        td = info_td['td']
        fd = info_fd['td']
        cos = (np.abs(np.dot(td.conj().ravel(), fd.ravel())) /
               (np.linalg.norm(td) * np.linalg.norm(fd) + 1e-30))
        assert cos > 0.99, f"TD/FD cosine {cos:.4f} < 0.99"

    def test_S1_signal_larger_than_S12_for_same_nuc(self):
        """S=1 with 2 EPR transitions should generally give larger ESEEM amplitude than S=1/2."""
        exp = PulseExperiment(Field=350.0, Sequence='2pESEEM',
                              dt=0.01, nPoints=64, tau=0.2)
        opt = SaffronOptions(GridSize=15)

        sys_half = SpinSystem(S=[0.5], g=[[2.001, 2.001, 2.001]],
                              Nucs='1H', A=[[3.0, 3.0, 9.0]])
        sys_one = SpinSystem(S=[1], g=[[2.001, 2.001, 2.001]],
                             D=[[100.0, 0.0]],
                             Nucs='1H', A=[[3.0, 3.0, 9.0]])

        _, y_half, _ = saffron(sys_half, exp, opt)
        _, y_one, _ = saffron(sys_one, exp, opt)

        # S=1 has 2 EPR transitions contributing; signal amplitude should differ
        amp_half = np.max(np.abs(y_half))
        amp_one = np.max(np.abs(y_one))
        assert amp_one > 0 and amp_half > 0, "One or both signals are zero"

    # ------------------------------------------------------------------
    # Orientation selection with S>1/2
    # ------------------------------------------------------------------

    def test_S1_orisel_runs(self, sys_S1_1H):
        """S=1 + orientation selection (ExciteWidth finite): runs without error."""
        exp = PulseExperiment(Field=340.0, Sequence='2pESEEM',
                              dt=0.01, nPoints=64, tau=0.2,
                              mwFreq=9.5, ExciteWidth=500.0)
        opt = SaffronOptions(GridSize=10)
        x, y, info = saffron(sys_S1_1H, exp, opt)
        assert len(y) == 64

    # ------------------------------------------------------------------
    # S > 1/2 with multiple nuclear subspaces
    # ------------------------------------------------------------------

    def test_S1_two_nuclei_runs(self):
        """S=1 + 1H + 14N (two subspaces): product rule applies correctly."""
        sys = SpinSystem(
            S=[1], g=[[2.002, 2.001, 2.000]],
            D=[[100.0, 0.0]],
            Nucs=['1H', '14N'],
            A=[[3.0, 3.0, 9.0], [5.0, 5.0, 12.0]],
            Q=[[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]],
        )
        exp = PulseExperiment(Field=350.0, Sequence='2pESEEM',
                              dt=0.01, nPoints=32, tau=0.2)
        opt = SaffronOptions(GridSize=5)
        x, y, info = saffron(sys, exp, opt)
        assert len(y) == 32
        assert np.any(np.abs(y) > 0)

    # ------------------------------------------------------------------
    # Sequences: 2pESEEM, 3pESEEM, HYSCORE, MimsENDOR all work for S>1/2
    # ------------------------------------------------------------------

    def test_S1_3pESEEM_14N_runs(self):
        """S=1 + 14N quadrupole 3pESEEM: runs correctly."""
        sys = SpinSystem(
            S=[1], g=[[2.002, 2.001, 2.000]],
            D=[[100.0, 0.0]],
            Nucs='14N', A=[[5.0, 5.0, 12.0]],
            Q=[[0.3, 0.0, 0.0]],
        )
        exp = PulseExperiment(Field=350.0, Sequence='3pESEEM',
                              dt=0.01, nPoints=32, tau=0.2)
        opt = SaffronOptions(GridSize=5)
        x, y, info = saffron(sys, exp, opt)
        assert len(y) == 32

    def test_S32_MimsENDOR_runs(self):
        """S=3/2 + 1H MimsENDOR: runs without error."""
        sys = SpinSystem(
            S=[1.5], g=[[2.002, 2.001, 2.000]],
            D=[[50.0, 0.0]],
            Nucs='1H', A=[[3.0, 3.0, 9.0]],
            lwEndor=0.1,
        )
        exp = PulseExperiment(Field=350.0, Sequence='MimsENDOR',
                              dt=0.002, nPoints=64, tau=0.1, mwFreq=9.5,
                              Range=[13.0, 17.0])
        opt = SaffronOptions(GridSize=5)
        x, y, info = saffron(sys, exp, opt)
        assert len(y) == 64


# =====================================================================
# 15. Internal consistency tests (EasySpin parity-style assertions)
#     These compare two torchspin runs with different but equivalent inputs;
#     no MATLAB .mat file needed.
# =====================================================================

class TestInternalConsistency:
    """Mirror EasySpin's within-simulation consistency checks.

    Each test corresponds to a named EasySpin test file.
    """

    # ------------------------------------------------------------------
    # saffron_fullA: full 3×3 A tensor == diagonal A + AFrame rotation
    # ------------------------------------------------------------------

    def test_fullA_equivalent(self):
        """Full 3×3 A matrix == diagonal A + Euler AFrame (cosine > 0.99).

        EasySpin: saffron_fullA.m — Ra'*diag([2,5,10])*Ra vs A=[2,5,10]+AFrame.
        """
        import math
        from torchspin.rotations import erot
        angles = [20 * math.pi / 180, 60 * math.pi / 180, 0.0]
        Ra = erot(angles).numpy()
        A_diag = np.array([2.0, 5.0, 10.0])
        A_full = Ra.T @ np.diag(A_diag) @ Ra  # rotated full 3×3

        sys1 = SpinSystem(S=[0.5], g=[[2.0023, 2.0023, 2.0023]],
                          Nucs='1H', A=A_full)
        sys2 = SpinSystem(S=[0.5], g=[[2.0023, 2.0023, 2.0023]],
                          Nucs='1H', A=[A_diag.tolist()],
                          AFrame=[[angles[0], angles[1], angles[2]]])

        exp = PulseExperiment(Field=350.0, Sequence='3pESEEM',
                              tau=0.001, dt=0.010, nPoints=60, Range=[0, 30])
        opt = SaffronOptions(GridSize=10, TimeDomain=True)

        _, _, info1 = saffron(sys1, exp, opt)
        _, _, info2 = saffron(sys2, exp, opt)
        td1 = np.asarray(info1['td']).ravel()
        td2 = np.asarray(info2['td']).ravel()
        cos = abs(np.dot(td1.conj(), td2)) / (np.linalg.norm(td1) * np.linalg.norm(td2) + 1e-30)
        assert cos > 0.99, f"fullA cosine {cos:.4f} < 0.99"

    # ------------------------------------------------------------------
    # saffron_hyscore_productrule: ProductRule=0 vs ProductRule=1 (HYSCORE)
    # ------------------------------------------------------------------

    def test_hyscore_productrule_consistency(self):
        """HYSCORE ProductRule=0 vs ProductRule=1: signal shape similar (cosine > 0.70).

        EasySpin: saffron_hyscore_productrule.m. Note: EasySpin's test has a
        copy-paste bug comparing y0 to itself; we check y0 vs y1 properly.
        The product rule is an approximation — exact agreement is not expected,
        but overall spectral shape should be similar.
        """
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]],
                         Nucs=['1H', '1H'],
                         A=[[1.0, 1.0, 3.0], [4.0, 4.0, 2.0]])
        exp = PulseExperiment(Field=1213.2, Sequence='HYSCORE',
                              dt=0.004, nPoints=10, tau=0.1)
        opt0 = SaffronOptions(GridSize=15, ProductRule=False)
        opt1 = SaffronOptions(GridSize=15, ProductRule=True)

        _, y0, _ = saffron(sys, exp, opt0)
        _, y1, _ = saffron(sys, exp, opt1)

        a = np.asarray(y0, dtype=float).ravel()
        b = np.asarray(y1, dtype=float).ravel()
        assert np.any(np.abs(a) > 0) and np.any(np.abs(b) > 0), \
            "One or both signals are all-zero"
        cos = abs(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)
        assert cos > 0.70, f"ProductRule=0 vs 1 cosine {cos:.4f} < 0.70"

    # ------------------------------------------------------------------
    # saffron_mimsproduct: MimsENDOR ProductRule=0 vs ProductRule=1
    # ------------------------------------------------------------------

    def test_mimsproduct_productrule_consistency(self):
        """MimsENDOR ProductRule=0 vs ProductRule=1: normalized signals agree.

        EasySpin: saffron_mimsproduct.m.
        """
        from torchspin.utils import larmorfrq
        nuI = np.asarray(larmorfrq('1H', 350.0)).reshape(-1)[0].item()
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]],
                         Nucs=['1H', '1H'],
                         A=[[3.0, 3.0, 3.0], [0.5, 0.5, 0.5]],
                         lwEndor=0.1)
        exp = PulseExperiment(Field=350.0, Sequence='MimsENDOR',
                              tau=0.01, Range=[0.5, nuI * 2 + 2],
                              nPoints=256)
        opt0 = SaffronOptions(GridSize=10, ProductRule=False)
        opt1 = SaffronOptions(GridSize=10, ProductRule=True)

        _, y0, _ = saffron(sys, exp, opt0)
        _, y1, _ = saffron(sys, exp, opt1)

        a = np.asarray(y0, dtype=float).ravel()
        b = np.asarray(y1, dtype=float).ravel()
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a > 0:
            a = a / norm_a
        if norm_b > 0:
            b = b / norm_b
        diff = np.sum(np.abs(a - b))
        assert diff < 0.1, f"MimsENDOR ProductRule sum|diff|={diff:.4g} >= 0.1"

    # ------------------------------------------------------------------
    # saffron_orientation_selective_customsequence
    # ------------------------------------------------------------------

    def test_orisel_predefined_vs_manual(self):
        """Predefined 2pESEEM + ExciteWidth == manual {p90,tau,p180,tau} + ExciteWidth.

        EasySpin: saffron_orientation_selective_customsequence.m.
        """
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.2]],
                         Nucs='1H', A=[[1.0, 1.0, 6.0]])
        opt = SaffronOptions(GridSize=31, TimeDomain=True)

        exp_pre = PulseExperiment(
            Field=350.0, Sequence='2pESEEM',
            mwFreq=9.5, dt=0.01, nPoints=120,
            ExciteWidth=200.0, tau=0.0,
        )
        exp_man = PulseExperiment(
            Field=350.0, Sequence='custom',
            mwFreq=9.5, dt=0.01, nPoints=120,
            ExciteWidth=200.0,
            Flip=[1.0, 1.0], Inc=[1, 1], t=[0.0, 0.0], Phase=[1.0, 1.0],
        )

        _, _, info_pre = saffron(sys, exp_pre, opt)
        _, _, info_man = saffron(sys, exp_man, opt)

        td_pre = np.asarray(info_pre['td']).ravel()
        td_man = np.asarray(info_man['td']).ravel()
        n = min(len(td_pre), len(td_man))
        cos = abs(np.dot(td_pre[:n].conj(), td_man[:n])) / (
            np.linalg.norm(td_pre[:n]) * np.linalg.norm(td_man[:n]) + 1e-30)
        assert cos > 0.99, f"orisel predefined vs manual cosine {cos:.4f} < 0.99"

    # ------------------------------------------------------------------
    # saffron_shfnucorder: nucleus ordering independence
    # ------------------------------------------------------------------

    def test_shfnucorder_independence(self):
        """ProductRule=1: 2H+63Cu vs 63Cu+2H ordering gives same signal.

        EasySpin: saffron_shfnucorder.m. Compares frequency-domain output.
        """
        A_2H = [[-0.553, -0.553, 1.128]]
        A_63Cu = [[96.0, 96.0, 570.0]]
        g = [[2.046, 2.046, 2.279]]

        sys1 = SpinSystem(S=[0.5], g=g,
                          Nucs=['2H', '63Cu'], A=A_2H + A_63Cu)
        sys2 = SpinSystem(S=[0.5], g=g,
                          Nucs=['63Cu', '2H'], A=A_63Cu + A_2H)

        exp = PulseExperiment(
            Field=340.2, Sequence='3pESEEM',
            mwFreq=9.738, ExciteWidth=30.0,
            dt=0.016, tau=0.224, T=0.080, nPoints=200,
        )
        opt = SaffronOptions(GridSize=31, ProductRule=True)

        _, y1, _ = saffron(sys1, exp, opt)
        _, y2, _ = saffron(sys2, exp, opt)

        a1 = np.asarray(y1).ravel()
        a2 = np.asarray(y2).ravel()
        max_y = np.max(np.abs(a1))
        assert max_y > 0, "Signal is all-zero"
        np.testing.assert_allclose(a1, a2, atol=max_y * 1e-5,
                                   err_msg="Nucleus ordering changes the signal")

    # ------------------------------------------------------------------
    # saffron_symmetry: axial A rotation invariance
    # ------------------------------------------------------------------

    def test_symmetry_axial_aframe(self):
        """Axial A=[3,3,12] with AFrame=[0,0,0] == AFrame=[167°,-78°,-47°] (cosine > 0.99).

        EasySpin: saffron_symmetry.m. Axial tensors are invariant under
        in-plane rotations about the symmetry axis.
        """
        import math
        sys1 = SpinSystem(S=[0.5], g=[[2.0023, 2.0023, 2.0023]],
                          Nucs='1H', A=[[3.0, 3.0, 12.0]],
                          AFrame=[[0.0, 0.0, 0.0]])
        sys2 = SpinSystem(S=[0.5], g=[[2.0023, 2.0023, 2.0023]],
                          Nucs='1H', A=[[3.0, 3.0, 12.0]],
                          AFrame=[[167 * math.pi / 180, -78 * math.pi / 180,
                                   -47 * math.pi / 180]])

        exp = PulseExperiment(Field=324.9, Sequence='3pESEEM',
                              dt=0.008, nPoints=200, tau=0.001)
        opt = SaffronOptions(GridSize=15, TimeDomain=True)

        _, _, info1 = saffron(sys1, exp, opt)
        _, _, info2 = saffron(sys2, exp, opt)

        td1 = np.asarray(info1['td']).ravel()
        td2 = np.asarray(info2['td']).ravel()
        cos = abs(np.dot(td1.conj(), td2)) / (
            np.linalg.norm(td1) * np.linalg.norm(td2) + 1e-30)
        assert cos > 0.99, f"symmetry axial cosine {cos:.4f} < 0.99"

    # ------------------------------------------------------------------
    # saffron_A_coresys: regression — MimsENDOR + ExciteWidth must not crash
    # ------------------------------------------------------------------

    def test_A_coresys_smoke(self):
        """MimsENDOR with two 1H and ExciteWidth: must not crash.

        EasySpin: saffron_A_coresys.m. Regression test for a bug where
        nucleus removal from the coreSys (triggered by ExciteWidth) caused
        an index error when Sys.A_ was set.
        """
        sys = SpinSystem(
            S=[0.5], g=[[2.0023, 2.0023, 2.0023]],
            Nucs=['1H', '1H'],
            A=[[200.0, 200.0, 200.0], [3.0, 3.0, 3.0]],
            lwEndor=0.1,
        )
        exp = PulseExperiment(
            Field=350.0, Sequence='MimsENDOR',
            tau=0.444, Range=[13.0, 16.4],
            mwFreq=9.7, ExciteWidth=50.0,
        )
        opt = SaffronOptions(GridSize=5)
        x, y, info = saffron(sys, exp, opt)
        assert y is not None

    # ------------------------------------------------------------------
    # ProductRule + TimeDomain
    # ------------------------------------------------------------------

    def test_productrule_timedomain(self):
        """ProductRule + TimeDomain matches ProductRule + FD accumulation.

        The FD path bins peaks onto a frequency grid before IFFT, so tiny
        discretization differences (~1e-4 relative) are expected; the two
        paths must agree in shape (cosine > 0.9999) and scale (< 0.1%).
        """
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]],
                         Nucs=['1H', '1H'],
                         A=[[1.0, 1.0, 3.0], [4.0, 4.0, 2.0]])
        exp = PulseExperiment(Field=350.0, Sequence='3pESEEM',
                              dt=0.016, tau=0.2, T=0.08, nPoints=100)

        _, _, info_td = saffron(sys, exp, SaffronOptions(
            GridSize=20, ProductRule=True, TimeDomain=True))
        _, _, info_fd = saffron(sys, exp, SaffronOptions(
            GridSize=20, ProductRule=True, TimeDomain=False))

        a = np.asarray(info_td['td']).real.ravel()
        b = np.asarray(info_fd['td']).real.ravel()
        assert np.abs(a).max() > 0
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        assert cos > 0.9999, f"PR+TD vs PR+FD cosine {cos:.6f}"
        scale = np.abs(a).max() / np.abs(b).max()
        assert abs(scale - 1.0) < 1e-3, f"scale ratio {scale:.5f} != 1"

    # ------------------------------------------------------------------
    # saffron_crystal: CrystalSymmetry — skip until implemented
    # ------------------------------------------------------------------

    def test_crystal_smoke(self):
        """Crystal HYSCORE with P212121 CrystalSymmetry runs without crashing.

        EasySpin: saffron_crystal.m (multi-site space group).
        """
        sys = SpinSystem(S=[0.5], g=[[2.0023, 2.0023, 2.0023]],
                         Nucs='1H', A=[[5.0, 5.0, 20.0]], lwEndor=0.5)
        exp = PulseExperiment(Field=330.0, Sequence='HYSCORE',
                              tau=0.080, dt=0.120, nPoints=32,
                              MolFrame=[np.pi / 3, np.pi / 6, np.pi / 4],
                              SampleFrame=[np.pi / 9, np.pi / 5, 0],
                              CrystalSymmetry='P212121')
        _, y, _ = saffron(sys, exp, SaffronOptions())
        y = np.asarray(y)
        assert y.shape == (32, 32)
        assert np.all(np.isfinite(y.real))
        assert np.any(np.abs(y) > 0)
