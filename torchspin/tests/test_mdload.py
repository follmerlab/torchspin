"""Tests for mdload, mdhmm, and mdtraj2oripot — cardamom Phase 5.

Uses real MD trajectory data from tests/mdfiles/:
  - A10R1_polyAla.{dcd, psf} — R1 (MTSSL) on polyalanine (DCD/PSF)
  - A10TOAC_polyAla.{dcd, psf} — TOAC on polyalanine (DCD/PSF)
  - 1ubq_A28R1_noSol.{trr, gro} — R1 on ubiquitin (TRR/GRO)
"""
import os
import numpy as np
import pytest

from torchspin.mdload import (
    mdload, MDData, MDInfo, MDOptions,
    _read_dcd, _read_psf, _read_gro, _read_trr,
)
from torchspin.mdhmm import (
    mdhmm, HMMResult, HMMOptions,
    _circular_kmeans, _dist_pbc, _viterbi, _forward_backward,
    _gaussian_prob_circular,
)
from torchspin.mdtraj2oripot import mdtraj2oripot


MDFILES = os.path.join(
    os.path.dirname(__file__), '..', '..', 'tests', 'mdfiles'
)

HAS_R1_DCD = os.path.exists(os.path.join(MDFILES, 'A10R1_polyAla.dcd'))
HAS_TOAC_DCD = os.path.exists(os.path.join(MDFILES, 'A10TOAC_polyAla.dcd'))
HAS_GRO = os.path.exists(os.path.join(MDFILES, '1ubq_A28R1_noSol.gro'))
HAS_TRR = os.path.exists(os.path.join(MDFILES, '1ubq_A28R1_noSol.trr'))


# ---------------------------------------------------------------------------
# DCD reader
# ---------------------------------------------------------------------------

class TestReadDCD:

    @pytest.mark.skipif(not HAS_R1_DCD, reason="DCD file not available")
    def test_read_r1_dcd(self):
        """Read R1 polyAla DCD file."""
        xyz, nAtoms, nFrames, dt = _read_dcd(
            os.path.join(MDFILES, 'A10R1_polyAla.dcd')
        )
        assert nAtoms > 0
        assert nFrames > 0
        assert dt > 0
        assert xyz.shape == (nFrames, 3, nAtoms)

    @pytest.mark.skipif(not HAS_R1_DCD, reason="DCD file not available")
    def test_read_dcd_with_indices(self):
        """Read DCD with atom index filter."""
        xyz_full, nAtoms, nFrames, dt = _read_dcd(
            os.path.join(MDFILES, 'A10R1_polyAla.dcd')
        )
        # Read only first 10 atoms
        xyz_sub, _, nf2, _ = _read_dcd(
            os.path.join(MDFILES, 'A10R1_polyAla.dcd'),
            atom_indices=list(range(10))
        )
        assert xyz_sub.shape[2] == 10
        assert nf2 == nFrames

    @pytest.mark.skipif(not HAS_TOAC_DCD, reason="TOAC DCD file not available")
    def test_read_toac_dcd(self):
        """Read TOAC polyAla DCD file."""
        xyz, nAtoms, nFrames, dt = _read_dcd(
            os.path.join(MDFILES, 'A10TOAC_polyAla.dcd')
        )
        assert nAtoms > 0
        assert nFrames > 0


# ---------------------------------------------------------------------------
# PSF reader
# ---------------------------------------------------------------------------

class TestReadPSF:

    @pytest.mark.skipif(not HAS_R1_DCD, reason="PSF file not available")
    def test_read_r1_psf(self):
        """Read R1 polyAla PSF file."""
        data = _read_psf(
            os.path.join(MDFILES, 'A10R1_polyAla.psf'),
            seg_name='PROT', res_name='CYR1', label_name='R1'
        )
        assert data['nAtoms'] > 0
        assert len(data['idx_SpinLabel']) > 0
        assert len(data['idx_ProteinCA']) > 0

    @pytest.mark.skipif(not HAS_TOAC_DCD, reason="TOAC PSF file not available")
    def test_read_toac_psf(self):
        """Read TOAC polyAla PSF file."""
        data = _read_psf(
            os.path.join(MDFILES, 'A10TOAC_polyAla.psf'),
            seg_name='PROA', res_name='TOC', label_name='TOAC'
        )
        assert data['nAtoms'] > 0
        assert len(data['idx_SpinLabel']) > 0


# ---------------------------------------------------------------------------
# GRO/TRR readers
# ---------------------------------------------------------------------------

class TestReadGRO:

    @pytest.mark.skipif(not HAS_GRO, reason="GRO file not available")
    def test_read_gro(self):
        """Read ubiquitin GRO file."""
        data = _read_gro(
            os.path.join(MDFILES, '1ubq_A28R1_noSol.gro'),
            res_name='CYR1', label_name='R1'
        )
        assert data['nAtoms'] > 0


class TestReadTRR:

    @pytest.mark.skipif(not HAS_TRR, reason="TRR file not available")
    def test_read_trr(self):
        """Read ubiquitin TRR file."""
        xyz, nAtoms, nFrames, dt = _read_trr(
            os.path.join(MDFILES, '1ubq_A28R1_noSol.trr')
        )
        assert nAtoms > 0
        assert nFrames > 0
        assert xyz.shape == (nFrames, 3, nAtoms)


# ---------------------------------------------------------------------------
# Circular distance
# ---------------------------------------------------------------------------

class TestCircularDistance:

    def test_zero_distance(self):
        assert _dist_pbc(0.0, 0.0) == 0.0

    def test_wrap_around(self):
        """Distance across 2π boundary."""
        d = _dist_pbc(0.1, 2 * np.pi - 0.1)
        assert abs(d - 0.2) < 1e-10

    def test_symmetric(self):
        d1 = _dist_pbc(1.0, 2.0)
        d2 = _dist_pbc(2.0, 1.0)
        assert abs(d1 - d2) < 1e-10

    def test_max_distance(self):
        d = _dist_pbc(0, np.pi)
        assert abs(d - np.pi) < 1e-10


# ---------------------------------------------------------------------------
# Circular K-means
# ---------------------------------------------------------------------------

class TestCircularKMeans:

    def test_basic_clustering(self):
        """Cluster two well-separated groups on a circle."""
        rng = np.random.default_rng(42)
        group1 = rng.normal(0.5, 0.1, (1, 100))
        group2 = rng.normal(3.0, 0.1, (1, 100))
        data = np.hstack([group1, group2])

        idx, centroids = _circular_kmeans(data, 2, n_repeats=3, seed=42)
        assert len(np.unique(idx)) == 2
        assert centroids.shape == (1, 2)

    def test_correct_number_clusters(self):
        rng = np.random.default_rng(42)
        data = rng.uniform(-np.pi, np.pi, (2, 300))
        idx, centroids = _circular_kmeans(data, 5, n_repeats=2, seed=42)
        assert centroids.shape[1] == 5


# ---------------------------------------------------------------------------
# Gaussian probability
# ---------------------------------------------------------------------------

class TestGaussianProbCircular:

    def test_peak_at_mean(self):
        """Max probability should be at the mean."""
        mu = np.array([0.0])
        sigma = np.array([[0.1]])
        x = np.linspace(-np.pi, np.pi, 100).reshape(1, -1)
        probs = _gaussian_prob_circular(x, mu, sigma)
        peak_idx = np.argmax(probs)
        assert abs(x[0, peak_idx]) < 0.1


# ---------------------------------------------------------------------------
# Forward-backward algorithm
# ---------------------------------------------------------------------------

class TestForwardBackward:

    def test_gamma_sums_to_one(self):
        """State occupancies should sum to 1 at each time step."""
        n_states, n_steps = 3, 50
        trans = np.array([[0.7, 0.2, 0.1],
                          [0.1, 0.8, 0.1],
                          [0.2, 0.1, 0.7]])
        init = np.array([0.4, 0.3, 0.3])
        obs = np.random.rand(n_states, n_steps) + 0.1

        gamma, xi, ll = _forward_backward(obs, trans, init)
        sums = gamma.sum(axis=0)
        assert np.allclose(sums, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Viterbi
# ---------------------------------------------------------------------------

class TestViterbi:

    def test_known_sequence(self):
        """Viterbi should recover an obvious state sequence."""
        n_states = 2
        n_steps = 100
        trans = np.array([[0.95, 0.05], [0.05, 0.95]])
        init = np.array([0.5, 0.5])

        # Create observation probabilities that strongly indicate state 0 for
        # first half and state 1 for second half
        obs = np.ones((n_states, n_steps)) * 0.01
        obs[0, :50] = 0.99
        obs[1, 50:] = 0.99

        path = _viterbi(obs, trans, init)
        assert np.all(path[:45] == 0)
        assert np.all(path[55:] == 1)


# ---------------------------------------------------------------------------
# mdhmm
# ---------------------------------------------------------------------------

class TestMDHMM:

    def test_synthetic_hmm(self):
        """Build HMM from synthetic dihedral data with 2 states."""
        rng = np.random.default_rng(42)
        n_steps = 1000

        # Two-state system with well-separated dihedral distributions
        states = np.zeros(n_steps, dtype=int)
        state = 0
        for t in range(1, n_steps):
            if rng.random() < 0.02:
                state = 1 - state
            states[t] = state

        dihedrals = np.zeros((1, n_steps))
        dihedrals[0, states == 0] = rng.normal(1.0, 0.2, np.sum(states == 0))
        dihedrals[0, states == 1] = rng.normal(-1.0, 0.2, np.sum(states == 1))

        result = mdhmm(dihedrals, dt=1e-12, n_states=2, n_lag=1,
                        opt=HMMOptions(nTrials=2))

        assert isinstance(result, HMMResult)
        assert result.nStates == 2
        assert result.TransProb.shape == (2, 2)
        assert abs(result.TransProb.sum(axis=1) - 1.0).max() < 1e-6
        assert len(result.viterbiTraj) == n_steps

    def test_transition_matrix_stochastic(self):
        """Transition matrix rows should sum to 1."""
        rng = np.random.default_rng(123)
        data = rng.normal(0, 1, (2, 500))
        result = mdhmm(data, dt=1e-12, n_states=3, n_lag=1)
        row_sums = result.TransProb.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# mdtraj2oripot
# ---------------------------------------------------------------------------

class TestMdtraj2oripot:

    def test_basic(self):
        """Test with synthetic rotation matrices."""
        nSteps = 1000
        rng = np.random.default_rng(42)
        frames = np.zeros((3, 3, nSteps))
        for t in range(nSteps):
            # Random rotation via Gram-Schmidt
            A = rng.standard_normal((3, 3))
            Q, _ = np.linalg.qr(A)
            if np.linalg.det(Q) < 0:
                Q[:, 0] = -Q[:, 0]
            frames[:, :, t] = Q

        potential, pdf = mdtraj2oripot(frames, n_bins=30)
        assert potential.shape == (30, 30, 30)
        assert pdf.shape == (30, 30, 30)
        assert np.min(potential) == 0.0  # Shifted to min=0
        assert np.all(pdf >= 0)

    def test_potential_minimum_at_peak(self):
        """Potential should be minimal where PDF is maximal."""
        nSteps = 5000
        # All frames near identity
        frames = np.zeros((3, 3, nSteps))
        for t in range(nSteps):
            # Small random perturbation from identity
            noise = np.random.randn(3, 3) * 0.1
            Q, _ = np.linalg.qr(np.eye(3) + noise)
            if np.linalg.det(Q) < 0:
                Q[:, 0] = -Q[:, 0]
            frames[:, :, t] = Q

        potential, pdf = mdtraj2oripot(frames, n_bins=20)
        # The potential minimum should correspond to PDF maximum
        pot_min_idx = np.unravel_index(np.argmin(potential), potential.shape)
        pdf_max_idx = np.unravel_index(np.argmax(pdf), pdf.shape)
        assert pot_min_idx == pdf_max_idx


# ---------------------------------------------------------------------------
# Integration: mdload with real data
# ---------------------------------------------------------------------------

class TestMDLoadIntegration:

    @pytest.mark.skipif(not HAS_R1_DCD, reason="R1 DCD/PSF files not available")
    def test_load_r1_polyala(self):
        """Full mdload pipeline for R1 on polyalanine."""
        info = MDInfo(SegName='PROT', ResName='CYR1', LabelName='R1')
        md = mdload(
            os.path.join(MDFILES, 'A10R1_polyAla.dcd'),
            os.path.join(MDFILES, 'A10R1_polyAla.psf'),
            info=info,
        )
        assert isinstance(md, MDData)
        assert md.nSteps == 20
        assert md.FrameTraj.shape == (3, 3, 20)
        assert md.FrameTrajwrtProt.shape == (3, 3, 20)
        assert md.RProtDiff.shape == (3, 3, 20)
        assert md.dihedrals.shape == (5, 20)  # R1 has 5 chi angles
        # Frame axes should be orthonormal
        F = md.FrameTraj[:, :, 0]
        assert abs(np.linalg.det(F)) - 1.0 < 0.01

    @pytest.mark.skipif(not HAS_TOAC_DCD, reason="TOAC DCD/PSF files not available")
    def test_load_toac_polyala(self):
        """Full mdload pipeline for TOAC on polyalanine."""
        info = MDInfo(SegName='PROA', ResName='TOC', LabelName='TOAC')
        md = mdload(
            os.path.join(MDFILES, 'A10TOAC_polyAla.dcd'),
            os.path.join(MDFILES, 'A10TOAC_polyAla.psf'),
            info=info,
        )
        assert isinstance(md, MDData)
        assert md.nSteps == 4
        assert md.FrameTraj.shape == (3, 3, 4)
        assert md.dihedrals.shape == (2, 4)  # TOAC has 2 chi angles
