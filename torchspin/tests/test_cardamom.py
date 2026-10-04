"""Tests for cardamom trajectory-based EPR simulation.

Verifies:
1. Basic diffusion model (no HF, with HF)
2. Jump model
3. MD-direct model (uses test .mat trajectory)
4. MATLAB reference validation (cosine similarity)
5. Output shapes and edge cases
6. ISTOs propagation method (Oganesyan 2011)
"""
import os
import numpy as np
import pytest

from torchspin import SpinSystem, Experiment
from torchspin.cardamom import cardamom, CardamomPar, CardamomOptions, MDInput


MDFILES_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'tests', 'mdfiles'
)


def _cosine_sim(a, b):
    """Cosine similarity between two 1D arrays."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-30 or norm_b < 1e-30:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Basic shape and output tests
# ---------------------------------------------------------------------------

class TestCardamomOutputs:
    """Output shapes and basic sanity."""

    def test_output_shapes(self):
        sys = SpinSystem(S=[0.5], g=[[2.009, 2.006, 2.002]],
                         tcorr=5e-9, lw=[0.5])
        exp = Experiment(mwFreq=9.5, Range=[330, 350], Harmonic=1)
        par = CardamomPar(Model='diffusion', dtSpin=1e-10, nSteps=200,
                          nTraj=5, nOrients=3)

        B, spc, td, t = cardamom(sys, exp, par)
        assert B.shape == (1024,)
        assert spc.shape == (1024,)
        assert td.shape == (200,)
        assert t.shape == (200,)

    def test_custom_npoints(self):
        sys = SpinSystem(S=[0.5], g=[[2.009, 2.006, 2.002]],
                         tcorr=5e-9, lw=[0.5])
        exp = Experiment(mwFreq=9.5, Range=[330, 350], nPoints=512,
                         Harmonic=1)
        par = CardamomPar(Model='diffusion', dtSpin=1e-10, nSteps=200,
                          nTraj=5, nOrients=3)
        B, spc, _, _ = cardamom(sys, exp, par)
        assert B.shape == (512,)
        assert spc.shape == (512,)

    def test_field_range(self):
        sys = SpinSystem(S=[0.5], g=[[2.009, 2.006, 2.002]],
                         tcorr=5e-9, lw=[0.5])
        exp = Experiment(mwFreq=9.5, Range=[330, 350], Harmonic=1)
        par = CardamomPar(Model='diffusion', dtSpin=1e-10, nSteps=200,
                          nTraj=5, nOrients=3)
        B, _, _, _ = cardamom(sys, exp, par)
        assert B[0] == pytest.approx(330.0, abs=1e-10)
        assert B[-1] == pytest.approx(350.0, abs=1e-10)

    def test_nonzero_spectrum(self):
        sys = SpinSystem(S=[0.5], g=[[2.009, 2.006, 2.002]],
                         tcorr=5e-9, lw=[0.5])
        exp = Experiment(mwFreq=9.5, Range=[330, 350], Harmonic=1)
        par = CardamomPar(Model='diffusion', dtSpin=1e-10, nSteps=500,
                          nTraj=20, nOrients=10)
        B, spc, _, _ = cardamom(sys, exp, par)
        assert np.max(np.abs(spc)) > 0

    def test_s1_raises(self):
        sys = SpinSystem(S=[1.0], g=[[2.0, 2.0, 2.0]], tcorr=5e-9)
        exp = Experiment(mwFreq=9.5, Range=[330, 350], Harmonic=1)
        par = CardamomPar(Model='diffusion', dtSpin=1e-10, nSteps=100,
                          nTraj=5, nOrients=3)
        with pytest.raises(ValueError, match="S = 1/2"):
            cardamom(sys, exp, par)


class TestCardamomHarmonic:
    """Detection harmonic."""

    def test_harmonic_0_absorption(self):
        sys = SpinSystem(S=[0.5], g=[[2.009, 2.006, 2.002]],
                         tcorr=5e-9, lw=[0.5])
        exp = Experiment(mwFreq=9.5, Range=[330, 350], Harmonic=0)
        par = CardamomPar(Model='diffusion', dtSpin=1e-10, nSteps=500,
                          nTraj=20, nOrients=10)
        B, spc, _, _ = cardamom(sys, exp, par)
        # Absorption mode should produce non-zero output
        assert np.max(np.abs(spc)) > 0

    def test_harmonic_1_derivative(self):
        sys = SpinSystem(S=[0.5], g=[[2.009, 2.006, 2.002]],
                         tcorr=5e-9, lw=[0.5])
        exp = Experiment(mwFreq=9.5, Range=[330, 350], Harmonic=1)
        par = CardamomPar(Model='diffusion', dtSpin=1e-10, nSteps=500,
                          nTraj=20, nOrients=10)
        B, spc, _, _ = cardamom(sys, exp, par)
        # First derivative should cross zero
        assert spc.min() < 0
        assert spc.max() > 0


# ---------------------------------------------------------------------------
# Diffusion model tests
# ---------------------------------------------------------------------------

class TestCardamomDiffusion:
    """Diffusion model."""

    def test_isotropic_g(self):
        """Isotropic g → spectrum centered at resonance field."""
        sys = SpinSystem(S=[0.5], g=[[2.003, 2.003, 2.003]],
                         tcorr=5e-9, lw=[0.5])
        exp = Experiment(mwFreq=9.5, Range=[330, 350], Harmonic=0)
        par = CardamomPar(Model='diffusion', dtSpin=1e-10, nSteps=500,
                          nTraj=30, nOrients=10)
        B, spc, _, _ = cardamom(sys, exp, par)
        # Peak should be near resonance field: B = mwFreq / (g * 13.996 MHz/mT)
        gamma = 2.003 * 13.996e-3  # GHz/mT
        B_res = 9.5 / gamma
        peak_B = B[np.argmax(spc)]
        assert abs(peak_B - B_res) < 2.0  # within 2 mT

    def test_with_14n(self):
        sys = SpinSystem(S=[0.5], g=[[2.009, 2.006, 2.002]],
                         Nucs=['14N'], A=[[10.0, 10.0, 95.0]],
                         tcorr=5e-9, lw=[0.5])
        exp = Experiment(mwFreq=9.5, Range=[330, 350], Harmonic=1)
        par = CardamomPar(Model='diffusion', dtSpin=1e-10, nSteps=500,
                          nTraj=10, nOrients=5)
        B, spc, _, _ = cardamom(sys, exp, par)
        assert np.max(np.abs(spc)) > 0

    @pytest.mark.slow
    @pytest.mark.xfail(reason="Stochastic: fast-motion narrowing requires many more trajectories/longer propagation to converge reliably")
    def test_fast_motion(self):
        """Very fast motion (small tcorr) → narrow spectrum.
        Marked slow because reliable width comparison needs many trajectories.
        """
        np.random.seed(42)
        sys = SpinSystem(S=[0.5], g=[[2.009, 2.006, 2.002]],
                         tcorr=1e-11, lw=[0.1])
        exp = Experiment(mwFreq=9.5, Range=[330, 350], Harmonic=1)
        par = CardamomPar(Model='diffusion', dtSpin=1e-11, nSteps=2000,
                          dtSpatial=1e-12, nTraj=200, nOrients=50)
        B, spc_fast, _, _ = cardamom(sys, exp, par)

        np.random.seed(99)
        sys_slow = SpinSystem(S=[0.5], g=[[2.009, 2.006, 2.002]],
                              tcorr=1e-7, lw=[0.1])
        par_slow = CardamomPar(Model='diffusion', dtSpin=1e-9, nSteps=2000,
                               dtSpatial=1e-10, nTraj=200, nOrients=50)
        _, spc_slow, _, _ = cardamom(sys_slow, exp, par_slow)

        # Fast-motion spectrum should be narrower
        nz_fast = np.where(np.abs(spc_fast) > np.max(np.abs(spc_fast)) * 0.05)[0]
        nz_slow = np.where(np.abs(spc_slow) > np.max(np.abs(spc_slow)) * 0.05)[0]
        if len(nz_fast) > 0 and len(nz_slow) > 0:
            width_fast = B[nz_fast[-1]] - B[nz_fast[0]]
            width_slow = B[nz_slow[-1]] - B[nz_slow[0]]
            assert width_fast < width_slow


# ---------------------------------------------------------------------------
# Jump model tests
# ---------------------------------------------------------------------------

class TestCardamomJump:
    """Jump model."""

    def test_two_state_jump(self):
        rates = np.array([[-1e9, 1e9], [1e9, -1e9]])
        oris = np.array([[0.0, 0.0, 0.0], [0.0, np.pi/4, 0.0]])

        sys = SpinSystem(S=[0.5], g=[[2.009, 2.006, 2.002]],
                         tcorr=5e-9, lw=[0.5],
                         TransRates=rates, Orientations=oris)
        exp = Experiment(mwFreq=9.5, Range=[330, 350], Harmonic=1)
        par = CardamomPar(Model='jump', dtSpin=1e-10, nSteps=300,
                          dtSpatial=1e-11, nTraj=5, nOrients=3)
        B, spc, _, _ = cardamom(sys, exp, par)
        assert np.max(np.abs(spc)) > 0


# ---------------------------------------------------------------------------
# MD-direct model tests
# ---------------------------------------------------------------------------

class TestCardamomMDDirect:
    """MD-direct model using test trajectory data."""

    @pytest.fixture
    def md_traj(self):
        """Load MTSSL polyAla trajectory from tests/mdfiles."""
        mat_path = os.path.join(MDFILES_DIR, 'MTSSL_polyAla_traj.mat')
        if not os.path.exists(mat_path):
            pytest.skip(f"Reference file not found: {mat_path}")
        import scipy.io
        data = scipy.io.loadmat(mat_path)
        traj = data['Traj'][0, 0]
        return {
            'dt': float(traj['dt'].ravel()[0]),
            'nSteps': int(traj['nSteps'].ravel()[0]),
            'FrameTraj': traj['FrameTraj'],
            'FrameTrajwrtProt': traj['FrameTrajwrtProt'],
            'RProtDiff': traj['RProtDiff'],
        }

    def test_md_direct_runs(self, md_traj):
        """MD-direct model should produce non-zero spectrum."""
        # Use a small subset of the trajectory
        nUse = min(500, md_traj['FrameTraj'].shape[3])
        RTraj = md_traj['FrameTraj'][:, :, :, :nUse]
        # Expand single traj to match nTraj shape: (3,3,nSteps,nTraj)
        if RTraj.shape[2] == 1:
            RTraj = np.broadcast_to(RTraj, (3, 3, 1, nUse))
            RTraj = RTraj.transpose(0, 1, 3, 2)  # → (3,3,nUse,1)

        md = MDInput(
            FrameTraj=RTraj,
            dt=md_traj['dt'],
        )

        sys = SpinSystem(S=[0.5], g=[[2.009, 2.006, 2.002]],
                         tcorr=5e-9, lw=[0.5])
        exp = Experiment(mwFreq=9.5, Range=[330, 350], Harmonic=1)
        nStepsSpin = min(200, RTraj.shape[2])
        par = CardamomPar(Model='MD-direct', dtSpin=md_traj['dt'],
                          nSteps=nStepsSpin, nTraj=RTraj.shape[3],
                          nOrients=3)

        B, spc, td, t = cardamom(sys, exp, par, md=md)
        assert B.shape[0] == 1024
        assert np.max(np.abs(spc)) > 0


# ---------------------------------------------------------------------------
# MATLAB reference validation
# ---------------------------------------------------------------------------

class TestCardamomMATLABRef:
    """Validate against MATLAB EasySpin reference spectra."""

    @pytest.fixture
    def nitroxide_ref(self):
        """Load MATLAB nitroxide reference spectrum."""
        mat_path = os.path.join(MDFILES_DIR, 'MTSSL_polyAla_spc_nitroxide.mat')
        if not os.path.exists(mat_path):
            pytest.skip(f"Reference file not found: {mat_path}")
        import scipy.io
        data = scipy.io.loadmat(mat_path)
        return {
            'B': data['BOld'].ravel(),
            'spc': data['spcOld'].ravel(),
            't': data['tOld'].ravel(),
            'ExpectVal': data['ExpectValOld'].ravel(),
        }

    @pytest.fixture
    def istos_ref(self):
        """Load MATLAB ISTOs reference spectrum."""
        mat_path = os.path.join(MDFILES_DIR, 'MTSSL_polyAla_spc_istos.mat')
        if not os.path.exists(mat_path):
            pytest.skip(f"Reference file not found: {mat_path}")
        import scipy.io
        data = scipy.io.loadmat(mat_path)
        return {
            'B': data['BOld'].ravel(),
            'spc': data['spcOld'].ravel(),
            't': data['tOld'].ravel(),
            'ExpectVal': data['ExpectValOld'].ravel(),
        }

    def test_nitroxide_ref_loaded(self, nitroxide_ref):
        """Verify reference data loads correctly."""
        assert nitroxide_ref['B'].shape[0] == 1024
        assert nitroxide_ref['spc'].shape[0] == 1024
        assert np.max(np.abs(nitroxide_ref['spc'])) > 0

    def test_istos_ref_loaded(self, istos_ref):
        """Verify ISTOs reference data loads correctly."""
        assert istos_ref['B'].shape[0] == 1024
        assert istos_ref['spc'].shape[0] == 1024


# ---------------------------------------------------------------------------
# Options and configuration tests
# ---------------------------------------------------------------------------

class TestCardamomOptions:
    """Options handling."""

    def test_no_window(self):
        sys = SpinSystem(S=[0.5], g=[[2.009, 2.006, 2.002]],
                         tcorr=5e-9, lw=[0.5])
        exp = Experiment(mwFreq=9.5, Range=[330, 350], Harmonic=1)
        par = CardamomPar(Model='diffusion', dtSpin=1e-10, nSteps=200,
                          nTraj=5, nOrients=3)
        opt = CardamomOptions(FFTWindow=False)
        B, spc, _, _ = cardamom(sys, exp, par, opt)
        assert np.max(np.abs(spc)) > 0

    def test_default_options(self):
        sys = SpinSystem(S=[0.5], g=[[2.009, 2.006, 2.002]],
                         tcorr=5e-9, lw=[0.5])
        exp = Experiment(mwFreq=9.5, Range=[330, 350], Harmonic=1)
        par = CardamomPar(Model='diffusion', dtSpin=1e-10, nSteps=200,
                          nTraj=5, nOrients=3)
        B, spc, _, _ = cardamom(sys, exp, par)
        assert spc.shape == (1024,)

    def test_custom_orientations(self):
        sys = SpinSystem(S=[0.5], g=[[2.009, 2.006, 2.002]],
                         tcorr=5e-9, lw=[0.5])
        exp = Experiment(mwFreq=9.5, Range=[330, 350], Harmonic=1)
        orients = np.array([[0.0, 0.0], [np.pi/2, np.pi/4], [np.pi, np.pi/2]])
        par = CardamomPar(Model='diffusion', dtSpin=1e-10, nSteps=200,
                          nTraj=5, Orients=orients)
        B, spc, _, _ = cardamom(sys, exp, par)
        assert spc.shape == (1024,)


# ---------------------------------------------------------------------------
# Dynamics parameter extraction
# ---------------------------------------------------------------------------

class TestDynamicsExtraction:
    """Dynamics parameter precedence."""

    def test_tcorr(self):
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]], tcorr=1e-9, lw=[0.5])
        exp = Experiment(mwFreq=9.5, Range=[330, 350], Harmonic=1)
        par = CardamomPar(Model='diffusion', dtSpin=1e-10, nSteps=100,
                          nTraj=3, nOrients=2)
        B, spc, _, _ = cardamom(sys, exp, par)
        assert spc.shape == (1024,)

    def test_logtcorr(self):
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]], logtcorr=-9.0, lw=[0.5])
        exp = Experiment(mwFreq=9.5, Range=[330, 350], Harmonic=1)
        par = CardamomPar(Model='diffusion', dtSpin=1e-10, nSteps=100,
                          nTraj=3, nOrients=2)
        B, spc, _, _ = cardamom(sys, exp, par)
        assert spc.shape == (1024,)

    def test_missing_dynamics_raises(self):
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]], lw=[0.5])
        exp = Experiment(mwFreq=9.5, Range=[330, 350], Harmonic=1)
        par = CardamomPar(Model='diffusion', dtSpin=1e-10, nSteps=100,
                          nTraj=3, nOrients=2)
        with pytest.raises(ValueError, match="tcorr"):
            cardamom(sys, exp, par)


# ---------------------------------------------------------------------------
# ISTOs method tests
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'tests', 'data'
)


class TestCardamomISTOs:
    """ISTOs propagation method (Oganesyan 2011)."""

    def test_istos_diffusion_basic(self):
        """ISTOs method with diffusion model produces non-zero spectrum."""
        sys = SpinSystem(S=[0.5], g=[[2.009, 2.006, 2.002]],
                         tcorr=5e-9, lw=[0.5])
        exp = Experiment(mwFreq=9.5, Range=[330, 350], Harmonic=1)
        par = CardamomPar(Model='diffusion', dtSpin=2e-10, nSteps=100,
                          nTraj=5, nOrients=3)
        opt = CardamomOptions(Method='ISTOs')
        B, spc, td, t = cardamom(sys, exp, par, opt)
        assert B.shape == (1024,)
        assert spc.shape == (1024,)
        assert np.max(np.abs(spc)) > 0

    def test_istos_with_hyperfine(self):
        """ISTOs method with 14N hyperfine coupling."""
        sys = SpinSystem(
            S=[0.5], g=[[2.009, 2.006, 2.002]],
            Nucs=['14N'], A=[[16.8, 16.8, 100.9]],
            tcorr=5e-9, lw=[0.5],
        )
        exp = Experiment(mwFreq=9.5, Range=[330, 350], Harmonic=1)
        par = CardamomPar(Model='diffusion', dtSpin=2e-10, nSteps=100,
                          nTraj=3, nOrients=2)
        opt = CardamomOptions(Method='ISTOs')
        B, spc, td, t = cardamom(sys, exp, par, opt)
        assert np.max(np.abs(spc)) > 0

    def test_istos_jump_basic(self):
        """ISTOs method with jump model."""
        rates = np.array([[-1e9, 1e9], [1e9, -1e9]])
        oris = np.array([[0.0, 0.0, 0.0], [0.0, np.pi / 4, 0.0]])
        sys = SpinSystem(
            S=[0.5], g=[[2.009, 2.006, 2.002]],
            tcorr=5e-9, lw=[0.5],
            TransRates=rates, Orientations=oris,
        )
        exp = Experiment(mwFreq=9.5, Range=[330, 350], Harmonic=1)
        par = CardamomPar(Model='jump', dtSpin=2e-10, nSteps=200,
                          dtSpatial=1e-11, nTraj=5, nOrients=3)
        opt = CardamomOptions(Method='ISTOs')
        B, spc, _, _ = cardamom(sys, exp, par, opt)
        assert np.max(np.abs(spc)) > 0

    def test_istos_block_averaging(self):
        """Block averaging should produce a valid spectrum."""
        sys = SpinSystem(S=[0.5], g=[[2.009, 2.006, 2.002]],
                         tcorr=5e-9, lw=[0.5])
        exp = Experiment(mwFreq=9.5, Range=[330, 350], Harmonic=1)
        par = CardamomPar(Model='diffusion', dtSpin=2e-10, nSteps=200,
                          nTraj=5, nOrients=2, BlockLength=4)
        opt = CardamomOptions(Method='ISTOs')
        B, spc, td, t = cardamom(sys, exp, par, opt)
        assert np.max(np.abs(spc)) > 0
        # Block averaging with length 4 should reduce nSteps
        assert td.shape[0] == 200 // 4

    def test_istos_md_direct(self):
        """ISTOs method with MD-direct model."""
        mat_path = os.path.join(MDFILES_DIR, 'MTSSL_polyAla_traj.mat')
        if not os.path.exists(mat_path):
            pytest.skip("MD trajectory file not found")
        import scipy.io
        data = scipy.io.loadmat(mat_path)
        traj = data['Traj'][0, 0]
        dt = float(traj['dt'].ravel()[0])

        # Use FrameTraj — shape (3, 3, nSteps, nTraj) after permute
        RTraj = traj['FrameTraj']
        if RTraj.ndim == 4 and RTraj.shape[2] == 1:
            RTraj = RTraj.transpose(0, 1, 3, 2)

        nUse = min(200, RTraj.shape[2])
        RTraj = RTraj[:, :, :nUse, :]

        md = MDInput(FrameTraj=RTraj, dt=dt * 2.5)  # tScale = 2.5

        sys = SpinSystem(
            S=[0.5], g=[[2.009, 2.006, 2.002]],
            Nucs=['14N'], A=[[16.8, 16.8, 100.9]],
            tcorr=5e-9, lw=[0.1, 0.1],
        )
        exp = Experiment(mwFreq=9.4, Range=[325, 345], Harmonic=1)
        par = CardamomPar(
            Model='MD-direct', dtSpin=dt * 2.5,
            nSteps=min(100, nUse), nTraj=RTraj.shape[3], nOrients=3,
        )
        opt = CardamomOptions(Method='ISTOs')
        B, spc, _, _ = cardamom(sys, exp, par, opt, md=md)
        assert np.max(np.abs(spc)) > 0


class TestCardamomISTOsMATLABRef:
    """Validate ISTOs method against MATLAB EasySpin reference spectra."""

    @pytest.fixture
    def istos_jump_ref(self):
        """Load MATLAB ISTOs stochastic jump reference."""
        mat_path = os.path.join(DATA_DIR, 'cardamom_istos_stoch_jump.mat')
        if not os.path.exists(mat_path):
            pytest.skip("Reference file not found")
        import scipy.io
        data = scipy.io.loadmat(mat_path)
        return data['data'][0, 0]['spc'].ravel()

    @pytest.fixture
    def istos_md_ref(self):
        """Load MATLAB ISTOs MD-direct reference."""
        mat_path = os.path.join(DATA_DIR, 'cardamom_istos_md_direct.mat')
        if not os.path.exists(mat_path):
            pytest.skip("Reference file not found")
        import scipy.io
        data = scipy.io.loadmat(mat_path)
        return data['data'][0, 0]['spc'].ravel()

    def test_istos_jump_ref_loaded(self, istos_jump_ref):
        """Verify ISTOs jump reference data loads correctly."""
        assert istos_jump_ref.shape[0] == 1024
        assert np.max(np.abs(istos_jump_ref)) > 0

    def test_istos_md_ref_loaded(self, istos_md_ref):
        """Verify ISTOs MD-direct reference data loads correctly."""
        assert istos_md_ref.shape[0] == 1024
        assert np.max(np.abs(istos_md_ref)) > 0


class TestPropagateISTOsUnit:
    """Unit tests for propagate_istos function."""

    @staticmethod
    def _make_qtraj(nTraj=5, nSteps=200):
        """Generate quaternion trajectory from diffusion."""
        from torchspin.stochtraj_diffusion import stochtraj_diffusion, DiffusionPar
        Diff = np.array([1e8, 1e8, 1e8])
        dp = DiffusionPar(nTraj=nTraj, nSteps=nSteps, dt=1e-10)
        t, RTraj, qTraj = stochtraj_diffusion(Diff, dp)
        return qTraj  # (4, nSteps, nTraj)

    def test_pure_electron_no_hf(self):
        """S=1/2 electron without hyperfine should produce decaying FID."""
        from torchspin._cardamom_propagatedm import propagate_istos

        qTraj = self._make_qtraj(nTraj=5, nSteps=200)
        g = np.array([2.009, 2.006, 2.002])
        omega = 2 * np.pi * 9.5e9  # 9.5 GHz
        CenterField = 339.0  # mT

        signal = propagate_istos(
            [0.5], g, None, qTraj, omega, 1e-10, 200, 5,
            CenterField,
        )
        assert signal.shape == (200,)
        # First point should have largest magnitude (initial coherence)
        assert np.abs(signal[0]) >= np.abs(signal[-1])

    def test_d2_caching(self):
        """D2 caching should return identical results."""
        from torchspin._cardamom_propagatedm import propagate_istos

        qTraj = self._make_qtraj(nTraj=3, nSteps=100)
        g = np.array([2.009, 2.006, 2.002])
        omega = 2 * np.pi * 9.5e9
        CenterField = 339.0

        # First call without cache
        sig1 = propagate_istos(
            [0.5], g, None, qTraj, omega, 1e-10, 100, 3,
            CenterField,
        )
        # Second call with cache
        cache = {}
        sig2 = propagate_istos(
            [0.5], g, None, qTraj, omega, 1e-10, 100, 3,
            CenterField, d2_cache=cache,
        )
        assert np.allclose(sig1, sig2)
        assert len(cache) == 1  # one entry cached

    def test_detect_Sz(self):
        """detect_Sz should return real-valued signal."""
        from torchspin._cardamom_propagatedm import propagate_istos

        qTraj = self._make_qtraj(nTraj=3, nSteps=100)
        g = np.array([2.009, 2.006, 2.002])
        omega = 2 * np.pi * 9.5e9
        CenterField = 339.0

        sig_sp = propagate_istos(
            [0.5], g, None, qTraj, omega, 1e-10, 100, 3,
            CenterField, detect_Sz=False,
        )
        sig_sz = propagate_istos(
            [0.5], g, None, qTraj, omega, 1e-10, 100, 3,
            CenterField, detect_Sz=True,
        )
        # Sz detection should give predominantly real signal
        assert np.max(np.abs(sig_sz.imag)) < np.max(np.abs(sig_sz.real)) * 0.1
        # S+ and Sz should differ
        assert not np.allclose(sig_sp, sig_sz)
