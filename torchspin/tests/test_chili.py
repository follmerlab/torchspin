"""Tests for torchspin.chili — slow-motion EPR via SLE.

Tests are organized from unit-level (physics helpers) to integration-level
(full spectrum shapes). All tests must be fast enough to run in CI.
"""
import math
import numpy as np
import pytest

from torchspin import SpinSystem
from torchspin.chili import (
    ChiliOptions,
    _parse_diffusion,
    _tensor_cart2sph,
    _build_ori_basis,
    _precompute_jjj2,
    _starting_vector,
    _lanczos,
    _spectral_function,
    chili,
)
from torchspin.experiment import Experiment
from torchspin.constants import BMAGN, PLANCK


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_sys(tcorr=1e-10, lw=None, nuc=None, A=None):
    """S=1/2 system with optional nucleus."""
    kw = dict(S=[0.5], g=[[2.006, 2.006, 2.002]], tcorr=tcorr)
    if lw is not None:
        kw['lw'] = lw
    if nuc is not None:
        kw['Nucs'] = [nuc]
        kw['A'] = [A or [10.0, 10.0, 50.0]]
    return SpinSystem(**kw)


def _simple_exp(Range=(330, 350), nPoints=256, Harmonic=0):
    return Experiment(mwFreq=9.5, Range=list(Range), nPoints=nPoints, Harmonic=Harmonic)


# ---------------------------------------------------------------------------
# Unit tests: _parse_diffusion
# ---------------------------------------------------------------------------

class TestParseDiffusion:
    def test_tcorr(self):
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]], tcorr=1e-9)
        D = _parse_diffusion(sys)
        assert abs(D - 1.0 / (6 * 1e-9)) < 1e-3

    def test_logtcorr(self):
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]], logtcorr=-9)
        D = _parse_diffusion(sys)
        expected = 1.0 / (6 * 1e-9)
        assert abs(D - expected) / expected < 1e-6

    def test_Diff(self):
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]], Diff=1e8)
        D = _parse_diffusion(sys)
        assert abs(D - 1e8) < 1.0

    def test_logDiff(self):
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]], logDiff=8)
        D = _parse_diffusion(sys)
        assert abs(D - 1e8) / 1e8 < 1e-6

    def test_logtcorr_overrides_tcorr(self):
        """logtcorr takes precedence over tcorr."""
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]], tcorr=1e-8, logtcorr=-9)
        D = _parse_diffusion(sys)
        expected = 1.0 / (6 * 1e-9)  # logtcorr=-9 → tcorr=1e-9
        assert abs(D - expected) / expected < 1e-6

    def test_no_diffusion_raises(self):
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        with pytest.raises(ValueError, match="chili"):
            _parse_diffusion(sys)


# ---------------------------------------------------------------------------
# Unit tests: _tensor_cart2sph
# ---------------------------------------------------------------------------

class TestTensorCart2Sph:
    def test_isotropic_tensor_zero_F2(self):
        """Isotropic tensor has F2=0."""
        F0, F2 = _tensor_cart2sph(np.array([5.0, 5.0, 5.0]))
        assert np.all(np.abs(F2) < 1e-12)

    def test_isotropic_F0(self):
        """F0 = -sqrt(1/3)*(Axx+Ayy+Azz)."""
        a = 5.0
        F0, _ = _tensor_cart2sph(np.array([a, a, a]))
        expected = -math.sqrt(1.0 / 3.0) * 3 * a
        assert abs(F0 - expected) < 1e-12

    def test_axial_tensor(self):
        """Axial tensor: Axx=Ayy, F2[q=+2]=F2[q=-2]=0."""
        F0, F2 = _tensor_cart2sph(np.array([5.0, 5.0, 10.0]))
        assert abs(F2[0].real) < 1e-12  # q=+2
        assert abs(F2[4].real) < 1e-12  # q=-2
        # F2[q=0] = sqrt(2/3)*(10 - 0.5*(5+5)) = sqrt(2/3)*5
        expected_F2_0 = math.sqrt(2.0 / 3.0) * 5.0
        assert abs(F2[2].real - expected_F2_0) < 1e-10

    def test_rhombic_tensor(self):
        """Rhombic tensor: F2[q=+2] = F2[q=-2] = 0.5*(Axx-Ayy)."""
        F0, F2 = _tensor_cart2sph(np.array([3.0, 7.0, 10.0]))
        assert abs(F2[0].real - 0.5 * (3.0 - 7.0)) < 1e-10  # q=+2
        assert abs(F2[4].real - 0.5 * (3.0 - 7.0)) < 1e-10  # q=-2

    def test_traceless_part(self):
        """F0 captures the trace; F2 captures the anisotropy."""
        A = np.array([2.0, 4.0, 6.0])
        F0, F2 = _tensor_cart2sph(A)
        assert abs(F0 - (-math.sqrt(1.0 / 3.0) * 12.0)) < 1e-10


# ---------------------------------------------------------------------------
# Unit tests: _build_ori_basis
# ---------------------------------------------------------------------------

class TestBuildOriBasis:
    def test_L0_state_exists(self):
        """L=0, K=0, jK=+1 state must exist."""
        basis = _build_ori_basis(4, 3, 2)
        l0 = [s for s in basis if s['L'] == 0 and s['K'] == 0 and s['jK'] == 1]
        assert len(l0) == 1

    def test_K0_jK_parity(self):
        """For K=0, jK = (-1)^L."""
        basis = _build_ori_basis(6, 5, 3)
        for s in basis:
            if s['K'] == 0:
                expected_jK = (-1) ** s['L']
                assert s['jK'] == expected_jK, f"L={s['L']}: jK={s['jK']} != {expected_jK}"

    def test_K_positive_both_jK(self):
        """For K>0, both jK=+1 and jK=-1 must appear."""
        basis = _build_ori_basis(4, 3, 2)
        k1_states = [s for s in basis if s['K'] == 1]
        jKs = set(s['jK'] for s in k1_states)
        assert +1 in jKs and -1 in jKs

    def test_M_always_zero(self):
        """M=0 for all basis states (Meirovitch)."""
        basis = _build_ori_basis(6, 5, 3)
        assert all(s['M'] == 0 for s in basis)

    def test_Kmax_respected(self):
        """K never exceeds Kmax."""
        Kmax = 2
        basis = _build_ori_basis(6, 5, Kmax)
        assert all(s['K'] <= Kmax for s in basis)

    def test_basis_size_grows_with_Lmax(self):
        """Larger Lmax → more basis states."""
        b1 = _build_ori_basis(4, 3, 2)
        b2 = _build_ori_basis(8, 7, 2)
        assert len(b2) > len(b1)


# ---------------------------------------------------------------------------
# Unit tests: 3j symbols
# ---------------------------------------------------------------------------

class TestJjj2:
    def test_triangle_rule_zero(self):
        """3j(0,2,1;0,0,0) should be zero (triangle rule fails: |0-1|>2? no, 0+1<2)."""
        jjj2 = _precompute_jjj2(4)
        def idx(L, MK): return L*L+L-MK
        # (0,2,1;0,0,0): triangle 0+1=1 < 2, so zero
        assert abs(jjj2[idx(0, 0), idx(1, 0)]) < 1e-10

    def test_known_3j_symbol(self):
        """3j(2,2,2;0,0,0) = -sqrt(2/35) ≈ -0.2390."""
        jjj2 = _precompute_jjj2(4)
        def idx(L, MK): return L*L+L-MK
        val = jjj2[idx(2, 0), idx(2, 0)]
        expected = -math.sqrt(2.0 / 35.0)  # known value
        assert abs(val - expected) < 1e-6

    def test_symmetry(self):
        """3j(L1,2,L2;M1,m,M2) is related to (L2,2,L1;M2,m,M1) by (-1)^(L1+2+L2)."""
        jjj2 = _precompute_jjj2(4)
        def idx(L, MK): return L*L+L-MK
        # 3j(2,2,4;0,0,0) and 3j(4,2,2;0,0,0): same by symmetry (L1+2+L2=8, even)
        v1 = jjj2[idx(2, 0), idx(4, 0)]
        v2 = jjj2[idx(4, 0), idx(2, 0)]
        # They differ by (-1)^(2+2+4) = (-1)^8 = +1
        assert abs(v1 - v2) < 1e-8


# ---------------------------------------------------------------------------
# Unit tests: Lanczos
# ---------------------------------------------------------------------------

class TestLanczos:
    def test_diagonal_matrix(self):
        """For a diagonal matrix, Lanczos should return eigenvalues on diagonal."""
        diag_vals = np.array([1.0+0j, 2.0+0j, 3.0+0j, 4.0+0j])
        L = sp.diags(diag_vals)
        b = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        alphas, betas = _lanczos(L, b, 100, 1e-12)
        # For b = e_1, Lanczos should converge immediately (b is already eigenvector)
        assert len(alphas) >= 1
        # alpha[0] should be L[0,0] = 1.0
        assert abs(alphas[0] - 1.0) < 1e-10

    def test_spectral_function_Lorentzian(self):
        """For a 1×1 Liouvillian with L=a, G(z) = 1/(z+a) (MATLAB convention)."""
        alphas = np.array([2.0 + 0j])
        betas = np.array([], dtype=complex)
        z = 2.0 + 1j
        G = _spectral_function(alphas, betas, z)
        expected = 1.0 / (z + 2.0)  # MATLAB: z + alpha (not z - alpha)
        assert abs(G - expected) < 1e-12

    def test_spectral_function_imaginary_part(self):
        """Im(G(ω+iε)) should be a Lorentzian peaked at ω=0 when alpha=0."""
        alphas = np.array([0.0 + 0j])
        betas = np.array([], dtype=complex)
        eps = 1.0
        z = 0.0 + 1j * eps
        G = _spectral_function(alphas, betas, z)
        # G = 1/(iε) → Im(G) = -1/ε (negative)
        assert abs(G.imag - (-1.0 / eps)) < 1e-12


# ---------------------------------------------------------------------------
# Integration tests: chili spectrum
# ---------------------------------------------------------------------------

class TestChiliIntegration:
    def test_returns_correct_shapes(self):
        """chili returns (B_axis, spec) with correct shapes."""
        sys = _simple_sys(tcorr=1e-10)
        exp = _simple_exp(nPoints=64)
        B, spec = chili(sys, exp, ChiliOptions(LLMK=[4, 3, 0, 0], max_iter=50))
        assert B.shape == (64,)
        assert spec.shape == (64,)

    def test_B_axis_range(self):
        """B axis spans the requested range."""
        sys = _simple_sys(tcorr=1e-10)
        exp = _simple_exp(Range=(330, 350), nPoints=64)
        B, spec = chili(sys, exp, ChiliOptions(LLMK=[4, 3, 0, 0], max_iter=50))
        assert abs(B[0] - 330.0) < 1e-6
        assert abs(B[-1] - 350.0) < 1e-6

    def test_absorption_spectrum_nonzero(self):
        """Absorption spectrum should have nonzero values in the field range."""
        sys = _simple_sys(tcorr=1e-10)
        exp = _simple_exp(Range=(330, 350), nPoints=128, Harmonic=0)
        B, spec = chili(sys, exp, ChiliOptions(LLMK=[6, 5, 0, 2], max_iter=100))
        spec = np.asarray(spec)
        assert np.any(np.abs(spec) > 1e-10)

    def test_derivative_spectrum_has_positive_negative(self):
        """First-derivative spectrum should have both positive and negative parts."""
        sys = _simple_sys(tcorr=1e-10)
        exp = _simple_exp(Range=(330, 350), nPoints=256, Harmonic=1)
        B, spec = chili(sys, exp, ChiliOptions(LLMK=[6, 5, 0, 2], max_iter=200))
        spec = np.asarray(spec)
        assert np.any(spec > 0)
        assert np.any(spec < 0)

    def test_fast_motion_limit_narrower_than_slow(self):
        """Very short tcorr (fast motion) should give narrower spectrum than long tcorr."""
        exp = _simple_exp(Range=(330, 350), nPoints=256, Harmonic=0)
        opt = ChiliOptions(LLMK=[8, 7, 0, 2], max_iter=200)
        sys_fast = _simple_sys(tcorr=1e-11)
        sys_slow = _simple_sys(tcorr=1e-8)
        _, spec_fast = chili(sys_fast, exp, opt)
        _, spec_slow = chili(sys_slow, exp, opt)
        spec_fast = np.asarray(spec_fast)
        spec_slow = np.asarray(spec_slow)
        # Fast motion: narrower → higher peak / smaller FWHM
        # Measure "width" as FWHM of the peak
        def half_width(spec):
            pk = spec.max()
            half = pk / 2
            above = np.where(spec > half)[0]
            if len(above) < 2:
                return float('inf')
            return above[-1] - above[0]
        w_fast = half_width(spec_fast)
        w_slow = half_width(spec_slow)
        assert w_fast <= w_slow

    def test_nucleus_shifts_spectrum(self):
        """Adding a nucleus (A_iso ≠ 0) should change the spectrum."""
        exp = _simple_exp(Range=(330, 360), nPoints=256, Harmonic=0)
        opt = ChiliOptions(LLMK=[6, 5, 0, 2], max_iter=150)
        sys_no_nuc = _simple_sys(tcorr=1e-10)
        sys_with_nuc = _simple_sys(tcorr=1e-10, nuc='1H', A=[10.0, 10.0, 10.0])
        _, spec_no = chili(sys_no_nuc, exp, opt)
        _, spec_with = chili(sys_with_nuc, exp, opt)
        spec_no = np.asarray(spec_no)
        spec_with = np.asarray(spec_with)
        # Spectra should differ
        diff = np.max(np.abs(spec_no - spec_with))
        assert diff > 1e-6 * np.max(np.abs(spec_no))

    def test_isotropic_system_raises(self):
        """An isotropic system has no slow-motion spectrum (EasySpin error)."""
        sys = SpinSystem(S=[1.0], g=[[2.0, 2.0, 2.0]], tcorr=1e-10)
        exp = _simple_exp()
        with pytest.raises(ValueError, match="isotropic"):
            chili(sys, exp)

    def test_high_spin_supported(self):
        """S=1 with a zero-field splitting runs through the general Liouvillian."""
        sys = SpinSystem(S=[1.0], g=[[2.01, 2.005, 2.002]], D=100.0, tcorr=1e-8)
        exp = Experiment(mwFreq=9.5, Range=[331, 346], nPoints=128, Harmonic=0)
        _, y = chili(sys, exp, ChiliOptions(LLMK=[6, 3, 2, 2]))
        assert np.all(np.isfinite(y.numpy())) and y.abs().max() > 0

    def test_no_diffusion_raises(self):
        """Missing tcorr/Diff should raise ValueError."""
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        exp = _simple_exp()
        with pytest.raises(ValueError):
            chili(sys, exp)

    def test_harmonic_0_1_2_different(self):
        """Harmonic 0, 1, 2 should give different spectra."""
        sys = _simple_sys(tcorr=1e-10)
        opt = ChiliOptions(LLMK=[6, 5, 0, 2], max_iter=150)
        spectra = []
        for h in [0, 1, 2]:
            exp = _simple_exp(nPoints=256, Harmonic=h)
            _, spec = chili(sys, exp, opt)
            spectra.append(spec)
        assert not np.allclose(spectra[0], spectra[1])
        assert not np.allclose(spectra[1], spectra[2])


# ---------------------------------------------------------------------------
# Import needed for diagonal test
# ---------------------------------------------------------------------------

import scipy.sparse as sp
