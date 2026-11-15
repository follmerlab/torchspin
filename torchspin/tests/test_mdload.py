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
# DCD reader: CHARMM unit-cell block (B97)
# ---------------------------------------------------------------------------

DATA = os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'data')
DCD_REF = os.path.join(DATA, 'ref_md_readdcd.mat')
HAS_DCD_REF = os.path.exists(DCD_REF)


def _write_dcd(path, xyz, version=24, extra_block=False, four_dims=False,
               delta=0.04, nsavc=1000, nset=None):
    """Write a little-endian DCD file (NAMD/CHARMM layout, or X-PLOR with
    version=0) with coordinates xyz of shape (nFrames, 3, nAtoms)."""
    import struct
    nFrames, _, nAtoms = xyz.shape
    nset = nFrames if nset is None else nset

    def rec(payload):
        return struct.pack('<i', len(payload)) + payload + struct.pack('<i', len(payload))

    icntrl = [0] * 20
    icntrl[0], icntrl[1], icntrl[2], icntrl[3] = nset, 0, nsavc, nset * nsavc
    if version > 0:
        # file offsets 44 (DELTA), 48 (unit-cell flag), 52 (4-dims flag)
        icntrl[9] = struct.unpack('<i', struct.pack('<f', delta))[0]
        icntrl[10] = 1 if extra_block else 0
        icntrl[11] = 1 if four_dims else 0
        hdr = b'CORD' + struct.pack('<20i', *icntrl)
    else:  # X-PLOR: DELTA is a float64 at file offsets 44-51
        hdr = (b'CORD' + struct.pack('<9i', *icntrl[:9])
               + struct.pack('<d', delta) + struct.pack('<9i', *icntrl[11:]))
    hdr = hdr[:80] + struct.pack('<i', version)  # VERSION at file offset 84
    title = struct.pack('<i', 1) + b'REMARKS synthetic test file'.ljust(80)
    out = rec(hdr) + rec(title) + rec(struct.pack('<i', nAtoms))
    for k in range(nFrames):
        if extra_block:  # unit cell A, gamma, B, beta, alpha, C
            out += rec(struct.pack('<6d', 30.0 + k, 0.0, 31.0, 0.0, 0.0, 32.0))
        for dim in range(3):
            out += rec(xyz[k, dim].astype('<f4').tobytes())
        if four_dims:
            out += rec(np.full(nAtoms, 99.0, dtype='<f4').tobytes())
    with open(path, 'wb') as f:
        f.write(out)


def _first_short_read(monkeypatch, path):
    """Run _read_dcd on path and return the file offset at which its first
    short read (fewer bytes than requested) started; the reader stops there."""
    import builtins
    import importlib
    # the package re-exports the mdload function under the module's name
    mdl = importlib.import_module('torchspin.mdload')
    starts = []

    class _Tracked:
        def __init__(self, f):
            self._f = f

        def read(self, n=-1):
            pos = self._f.tell()
            data = self._f.read(n)
            if n >= 0 and len(data) < n:
                starts.append(pos)
            return data

        def __getattr__(self, name):
            return getattr(self._f, name)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._f.close()

    monkeypatch.setattr(mdl, 'open',
                        lambda *a, **k: _Tracked(builtins.open(*a, **k)),
                        raising=False)
    try:
        mdl._read_dcd(path)
    finally:
        monkeypatch.undo()
    assert starts, "reader never hit the end of the file"
    return starts[0]


class TestReadDCDUnitCell:
    """B97: the CHARMM unit-cell flag is the int32 at file offset 48 (read
    only when the VERSION at offset 84 is > 0), not bit 0 of VERSION. Before
    the fix the 56-byte unit-cell record of every frame was read as coordinates,
    so both bundled trajectories came out shifted (first x value 0.0 instead
    of -14.474 for A10R1, max error 60.5 A; up to 1.8e34 for A10TOAC)."""

    @staticmethod
    def _xyz(nFrames, nAtoms):
        rng = np.random.default_rng(97)
        return rng.uniform(-20, 20, (nFrames, 3, nAtoms)).astype(np.float32)

    @pytest.mark.parametrize('extra_block', [False, True])
    @pytest.mark.parametrize('four_dims', [False, True])
    def test_synthetic_charmm(self, tmp_path, extra_block, four_dims,
                              monkeypatch):
        xyz = self._xyz(5, 7)
        path = str(tmp_path / 'synthetic.dcd')
        _write_dcd(path, xyz, extra_block=extra_block, four_dims=four_dims)
        out, nAtoms, nFrames, dt = _read_dcd(path)
        assert (nAtoms, nFrames) == (7, 5)
        np.testing.assert_array_equal(out, xyz.astype(np.float64))
        np.testing.assert_allclose(
            dt, float(np.float32(0.04)) * 1000 * 48.88821e-15, rtol=1e-12)
        assert _first_short_read(monkeypatch, path) == os.path.getsize(path)
        sub, _, nf, _ = _read_dcd(path, atom_indices=[6, 0, 3])
        assert nf == 5
        np.testing.assert_array_equal(sub, xyz[:, :, [6, 0, 3]])

    def test_synthetic_xplor(self, tmp_path):
        """VERSION == 0 (X-PLOR): no extra records; DELTA is a float64, and the
        int32 at offset 48 is its upper half, not a unit-cell flag."""
        xyz = self._xyz(3, 4)
        path = str(tmp_path / 'xplor.dcd')
        _write_dcd(path, xyz, version=0, delta=0.002)
        out, nAtoms, nFrames, dt = _read_dcd(path)
        assert (nAtoms, nFrames) == (4, 3)
        np.testing.assert_array_equal(out, xyz.astype(np.float64))
        np.testing.assert_allclose(dt, 0.002 * 1000 * 48.88821e-15, rtol=1e-12)

    @pytest.mark.skipif(not (HAS_DCD_REF and HAS_R1_DCD and HAS_TOAC_DCD),
                        reason="DCD files or EasySpin reference not available")
    @pytest.mark.parametrize('name', ['A10R1_polyAla', 'A10TOAC_polyAla'])
    def test_bundled_vs_easyspin(self, name):
        """Coordinates of all frames (atom subset) and of the first frame (all
        atoms) equal EasySpin md_readdcd exactly (float32 data)."""
        import scipy.io as sio
        ref = sio.loadmat(DCD_REF, squeeze_me=True)[name]
        fname = os.path.join(MDFILES, name + '.dcd')
        xyz, nAtoms, nFrames, dt = _read_dcd(fname)
        assert nAtoms == int(ref['nAtoms'].item())
        assert nFrames == int(ref['nFrames'].item())
        np.testing.assert_allclose(dt, float(ref['dt'].item()), rtol=1e-12)
        np.testing.assert_array_equal(
            xyz[0], np.asarray(ref['xyz_full_frame1'].item()))
        idx = np.atleast_1d(ref['idx'].item()).astype(int) - 1
        np.testing.assert_array_equal(
            xyz[:, :, idx], np.asarray(ref['xyz'].item()))

    @pytest.mark.skipif(not (HAS_R1_DCD and HAS_TOAC_DCD),
                        reason="DCD files not available")
    @pytest.mark.parametrize('name, n_expected',
                             [('A10R1_polyAla', 20), ('A10TOAC_polyAla', 4)])
    def test_bundled_layout_to_eof(self, name, n_expected, monkeypatch):
        """Every frame is a 56-byte unit-cell record plus three coordinate
        records, and the frames read fill the file exactly to EOF. (The
        NAMD headers store NSET = 0, so the count is checked against the
        file size and EasySpin's count instead.)"""
        import struct
        fname = os.path.join(MDFILES, name + '.dcd')
        with open(fname, 'rb') as f:
            raw = f.read()
        version, = struct.unpack('<i', raw[84:88])
        unitcell, = struct.unpack('<i', raw[48:52])
        assert version > 0 and unitcell == 1
        title_len, = struct.unpack('<i', raw[92:96])
        header_size = 92 + 4 + title_len + 4 + 12
        _, nAtoms, nFrames, _ = _read_dcd(fname)
        frame_size = 56 + 3 * (8 + 4 * nAtoms)
        # the reader's first short read (its end-of-trajectory signal)
        # starts exactly at EOF: no frame or partial frame is left unread
        assert _first_short_read(monkeypatch, fname) == len(raw)
        assert nFrames == n_expected
        assert header_size + nFrames * frame_size == len(raw)
        # each frame's unit-cell record starts with its 48-byte marker
        for k in range(nFrames):
            start = header_size + k * frame_size
            assert struct.unpack('<i', raw[start:start + 4])[0] == 48

    @pytest.mark.skipif(not (HAS_R1_DCD and HAS_TOAC_DCD),
                        reason="DCD files not available")
    @pytest.mark.parametrize('name, seg',
                             [('A10R1_polyAla', 'PROT'),
                              ('A10TOAC_polyAla', 'PROA')])
    def test_bundled_protein_matches_pdb(self, name, seg):
        """The protein atoms do not move in these runs: in every frame they
        equal the PDB coordinates to the PDB's 3 decimals. (Solvent atoms
        move and are wrapped, so they are not compared.)"""
        with open(os.path.join(MDFILES, name + '.pdb')) as f:
            atoms = [l for l in f if l.startswith(('ATOM', 'HETATM'))]
        pdb = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                        for l in atoms]).T
        mask = np.array([l[72:76].strip() == seg for l in atoms])
        xyz, nAtoms, nFrames, _ = _read_dcd(os.path.join(MDFILES, name + '.dcd'))
        assert nAtoms == len(atoms) and mask.sum() > 200
        for k in range(nFrames):
            np.testing.assert_allclose(xyz[k][:, mask], pdb[:, mask],
                                       rtol=0, atol=5.01e-4)


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

    @pytest.mark.skipif(not (HAS_R1_DCD and HAS_TOAC_DCD),
                        reason="DCD/PSF files not available")
    @pytest.mark.parametrize('label, name, info', [
        ('R1', 'A10R1_polyAla',
         MDInfo(SegName='PROT', ResName='CYR1', LabelName='R1')),
        ('TOAC', 'A10TOAC_polyAla',
         MDInfo(SegName='PROA', ResName='TOC', LabelName='TOAC')),
    ])
    def test_frames_vs_easyspin(self, label, name, info):
        """Label frames and protein rotation vs EasySpin's own regression data
        (mdload_files_R1/TOAC.mat, from EasySpin's tests/data). Before B97 the
        misread coordinates gave errors up to 1.8 here. dihedrals and ProtCAxyz
        are not compared (they differ from EasySpin for reasons unrelated to
        the DCD reader)."""
        import scipy.io as sio
        ref = sio.loadmat(os.path.join(DATA, f'mdload_files_{label}.mat'),
                          squeeze_me=True)['data']
        md = mdload(os.path.join(MDFILES, name + '.dcd'),
                    os.path.join(MDFILES, name + '.psf'), info=info)
        assert md.nSteps == int(ref['nSteps'].item())
        np.testing.assert_allclose(md.dt, float(ref['dt'].item()), rtol=1e-12)
        for field in ('FrameTraj', 'FrameTrajwrtProt', 'RProtDiff'):
            np.testing.assert_allclose(
                getattr(md, field), np.asarray(ref[field].item()),
                rtol=0, atol=1e-10, err_msg=field)
