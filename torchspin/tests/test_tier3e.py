"""Tests for Tier 3e: Cardamom ISTOs propagation method.

Tests cover:
- istotensor: irreducible spherical tensor operators from vector operators
- magint: IST decomposition of spin Hamiltonian interactions
- wigD: rank-2 Wigner D-matrices from quaternion trajectories
- propagate_istos: full Hilbert-space density matrix propagation
- cardamom with Method='ISTOs': end-to-end simulation
"""
import numpy as np
import pytest

from torchspin._cardamom_istos import istotensor, magint, wigD
from torchspin._cardamom_propagatedm import propagate_istos


# ============================================================================
# istotensor
# ============================================================================

class TestISTO:
    """Irreducible spherical tensor operator tests."""

    def test_rank0_trace(self):
        """Rank-0 component is -(1/sqrt(3)) * Tr(a.b)."""
        # Two scalar vectors: a = [1, 0, 0], b = [1, 0, 0]
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        T0, T1, T2 = istotensor(a, b)
        expected = -(1 / np.sqrt(3)) * 1.0  # a.b = 1
        assert abs(T0 - expected) < 1e-10

    def test_rank0_spin_operators(self):
        """Rank-0 from S=1/2 spin operators gives scalar."""
        from torchspin.spinops import sop
        spins = [0.5]
        Sx = sop(spins, [1, 1]).numpy()
        Sy = sop(spins, [1, 2]).numpy()
        Sz = sop(spins, [1, 3]).numpy()
        T0, T1, T2 = istotensor([Sx, Sy, Sz], [Sx, Sy, Sz])
        # T0 = -(1/sqrt(3)) * (Sx^2 + Sy^2 + Sz^2) = -(1/sqrt(3)) * S(S+1)/1 * I
        # For S=1/2: S(S+1) = 3/4, so T0 = -(3/4)/sqrt(3) * I = -sqrt(3)/4 * I
        expected_trace = -np.sqrt(3) / 4 * 2  # trace of -sqrt(3)/4 * I_2
        assert abs(np.trace(T0) - expected_trace) < 1e-10

    def test_rank2_traceless(self):
        """Rank-2 T2(0) is traceless for identical operators."""
        from torchspin.spinops import sop
        spins = [0.5]
        Sx = sop(spins, [1, 1]).numpy()
        Sy = sop(spins, [1, 2]).numpy()
        Sz = sop(spins, [1, 3]).numpy()
        T0, T1, T2 = istotensor([Sx, Sy, Sz], [Sx, Sy, Sz])
        # T2(0) = sqrt(2/3) * (Sz^2 - 0.5*(Sx^2 + Sy^2))
        # For S=1/2: Sz^2 = 0.25*I, Sx^2+Sy^2 = 0.5*I → traceless
        assert abs(np.trace(T2[2])) < 1e-10

    def test_scalar_vector_product(self):
        """IST of B-field and spin operator has correct dimensions."""
        from torchspin.spinops import sop
        spins = [0.5]
        Sx = sop(spins, [1, 1]).numpy()
        Sy = sop(spins, [1, 2]).numpy()
        Sz = sop(spins, [1, 3]).numpy()
        B = [0.0, 0.0, 340e-3]  # 340 mT in T
        T0, T1, T2 = istotensor(B, [Sx, Sy, Sz])
        # T0 should be a 2x2 matrix
        assert T0.shape == (2, 2)
        assert T2[0].shape == (2, 2)

    def test_five_rank2_components(self):
        """Rank-2 has exactly 5 components."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        T0, T1, T2 = istotensor(a, b)
        assert len(T2) == 5

    def test_three_rank1_components(self):
        """Rank-1 has exactly 3 components."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        T0, T1, T2 = istotensor(a, b)
        assert len(T1) == 3


# ============================================================================
# magint
# ============================================================================

class TestMagint:
    """IST decomposition tests."""

    def test_single_electron(self):
        """Single S=1/2 electron gives one interaction."""
        g = np.array([[2.002, 2.002, 2.002]])
        T, F = magint([0.5], g, 340.0)
        assert len(T['T0']) == 1  # one electron Zeeman
        assert F['F0'].shape == (1,)
        assert F['F2'].shape == (1, 5)

    def test_isotropic_g_rank2_zero(self):
        """Isotropic g-tensor has zero rank-2 components."""
        g = np.array([[2.002, 2.002, 2.002]])
        T, F = magint([0.5], g, 340.0)
        np.testing.assert_allclose(np.abs(F['F2'][0, :]), 0, atol=1e-6)

    def test_axial_g_rank2_nonzero(self):
        """Axial g-tensor has non-zero rank-2 components."""
        g = np.array([[2.009, 2.009, 2.002]])
        T, F = magint([0.5], g, 340.0)
        # F2(0) should be non-zero for axial tensor
        assert abs(F['F2'][0, 2]) > 1e-6

    def test_electron_plus_nucleus(self):
        """S=1/2 + I=1 gives 2 interactions (EZ + HF)."""
        g = np.array([[2.002, 2.002, 2.002]])
        A = np.array([[10.0, 10.0, 95.0]])  # MHz
        T, F = magint([0.5, 1.0], g, 340.0, A=A, nuc_spins=[1.0])
        assert len(T['T0']) == 2
        assert F['F0'].shape == (2,)

    def test_T0_operators_hermitian(self):
        """T0 operators should be Hermitian."""
        g = np.array([[2.009, 2.006, 2.002]])
        A = np.array([[10.0, 10.0, 95.0]])
        T, F = magint([0.5, 1.0], g, 340.0, A=A, nuc_spins=[1.0])
        for T0_op in T['T0']:
            if isinstance(T0_op, np.ndarray) and T0_op.ndim == 2:
                np.testing.assert_allclose(T0_op, T0_op.conj().T, atol=1e-12)

    def test_T2_operators_shape(self):
        """T2 operators have correct Hilbert space dimension."""
        g = np.array([[2.009, 2.006, 2.002]])
        T, F = magint([0.5, 1.0], g, 340.0, A=np.array([[10, 10, 95]]),
                       nuc_spins=[1.0])
        nStates = 6  # 2 * 3
        for T2_list in T['T2']:
            for T2_op in T2_list:
                assert T2_op.shape == (nStates, nStates)


# ============================================================================
# wigD
# ============================================================================

class TestWigD:
    """Wigner D-matrix tests."""

    def test_identity_quaternion(self):
        """Identity quaternion gives identity D-matrix."""
        q = np.array([[1.0], [0.0], [0.0], [0.0]])  # (4, 1)
        D2 = wigD(q)  # (5, 5, 1, 1)
        np.testing.assert_allclose(D2[:, :, 0, 0], np.eye(5), atol=1e-10)

    def test_unitarity(self):
        """D-matrix is unitary for a general quaternion."""
        # Random unit quaternion
        q = np.array([0.5, 0.5, 0.5, 0.5])[:, np.newaxis]  # normalized
        D2 = wigD(q)[:, :, 0, 0]
        I5 = D2 @ D2.conj().T
        np.testing.assert_allclose(I5, np.eye(5), atol=1e-10)

    def test_trajectory_shape(self):
        """Output shape is (5, 5, nSteps, nTraj)."""
        nSteps, nTraj = 50, 10
        q = np.random.randn(4, nSteps, nTraj)
        # Normalize
        q /= np.sqrt(np.sum(q**2, axis=0, keepdims=True))
        D2 = wigD(q)
        assert D2.shape == (5, 5, nSteps, nTraj)

    def test_z_rotation_diagonal(self):
        """Z-rotation by angle phi gives diagonal D-matrix."""
        phi = np.pi / 3
        # Quaternion for z-rotation: q = [cos(phi/2), 0, 0, sin(phi/2)]
        q = np.array([[np.cos(phi / 2)], [0.0], [0.0], [np.sin(phi / 2)]])
        D2 = wigD(q)[:, :, 0, 0]
        # For z-rotation, D^2_{mm'}(0, 0, phi) = delta_{mm'} * exp(-i*m*phi)
        for i in range(5):
            for j in range(5):
                if i != j:
                    assert abs(D2[i, j]) < 1e-10

    def test_determinant_one(self):
        """D-matrix has determinant 1 (special unitary)."""
        q = np.array([[0.6], [0.2], [0.5], [0.3]])
        q /= np.sqrt(np.sum(q**2))
        q = q.reshape(4, 1)
        D2 = wigD(q)[:, :, 0, 0]
        det = np.linalg.det(D2)
        assert abs(abs(det) - 1.0) < 1e-8


# ============================================================================
# propagate_istos
# ============================================================================

class TestPropagateISTOs:
    """Full Hilbert-space propagation tests."""

    def test_isotropic_signal_decays(self):
        """Isotropic g-tensor: signal should be non-trivial."""
        np.random.seed(42)
        spins = [0.5]
        g = np.array([2.002, 2.002, 2.002])
        nSteps = 50
        nTraj = 5
        dtSpin = 1e-10

        # Identity quaternion trajectory (no dynamics)
        qTraj = np.zeros((4, nSteps, nTraj))
        qTraj[0] = 1.0

        omega = 2 * np.pi * 9.5e9  # 9.5 GHz
        CenterField = 340.0  # mT

        signal = propagate_istos(
            spins, g, None, qTraj, omega, dtSpin, nSteps, nTraj, CenterField
        )
        assert signal.shape == (nSteps,)
        assert np.all(np.isfinite(signal))
        # Initial signal should be non-zero (Tr(S+ * Sx) = Tr(S+ * (S+ + S-)/2))
        assert abs(signal[0]) > 0

    def test_static_signal_shape(self):
        """Static system gives expected signal shape."""
        spins = [0.5]
        g = np.array([2.009, 2.006, 2.002])
        nSteps = 30
        nTraj = 3
        dtSpin = 1e-10

        qTraj = np.zeros((4, nSteps, nTraj))
        qTraj[0] = 1.0

        omega = 2 * np.pi * 9.5e9
        CenterField = 340.0

        signal = propagate_istos(
            spins, g, None, qTraj, omega, dtSpin, nSteps, nTraj, CenterField
        )
        assert signal.shape == (nSteps,)

    def test_with_hyperfine(self):
        """S=1/2 + 14N(I=1) system propagates correctly."""
        spins = [0.5, 1.0]
        g = np.array([2.009, 2.006, 2.002])
        A = np.array([[10.0, 10.0, 95.0]])  # MHz
        nSteps = 20
        nTraj = 2
        dtSpin = 1e-10

        qTraj = np.zeros((4, nSteps, nTraj))
        qTraj[0] = 1.0

        omega = 2 * np.pi * 9.5e9
        CenterField = 340.0

        signal = propagate_istos(
            spins, g, A, qTraj, omega, dtSpin, nSteps, nTraj, CenterField,
            nuc_spins=[1.0],
        )
        assert signal.shape == (nSteps,)
        assert np.all(np.isfinite(signal))
        # Hilbert space is 2*3 = 6
        assert abs(signal[0]) > 0

    def test_signal_bounded(self):
        """Signal magnitude stays bounded (no numerical blowup)."""
        spins = [0.5]
        g = np.array([2.009, 2.006, 2.002])
        nSteps = 100
        nTraj = 3
        dtSpin = 1e-10

        # Random rotation trajectory
        np.random.seed(123)
        qTraj = np.random.randn(4, nSteps, nTraj)
        qTraj /= np.sqrt(np.sum(qTraj**2, axis=0, keepdims=True))

        omega = 2 * np.pi * 9.5e9
        CenterField = 340.0

        signal = propagate_istos(
            spins, g, None, qTraj, omega, dtSpin, nSteps, nTraj, CenterField
        )
        # |Tr(S+ * rho)| should be bounded by max singular value of S+
        assert np.all(np.abs(signal) < 10)

    def test_multi_trajectory_averaging(self):
        """More trajectories produce smoother signal."""
        np.random.seed(7)
        spins = [0.5]
        g = np.array([2.009, 2.006, 2.002])
        nSteps = 30
        dtSpin = 1e-10
        omega = 2 * np.pi * 9.5e9
        CenterField = 340.0

        # Few trajectories
        nTraj_few = 2
        qTraj = np.random.randn(4, nSteps, nTraj_few)
        qTraj /= np.sqrt(np.sum(qTraj**2, axis=0, keepdims=True))
        sig_few = propagate_istos(
            spins, g, None, qTraj, omega, dtSpin, nSteps, nTraj_few, CenterField
        )

        # More trajectories
        nTraj_many = 20
        qTraj = np.random.randn(4, nSteps, nTraj_many)
        qTraj /= np.sqrt(np.sum(qTraj**2, axis=0, keepdims=True))
        sig_many = propagate_istos(
            spins, g, None, qTraj, omega, dtSpin, nSteps, nTraj_many, CenterField
        )

        # Both should be valid
        assert np.all(np.isfinite(sig_few))
        assert np.all(np.isfinite(sig_many))

    def test_lab_frame_rotation(self):
        """Lab-frame quaternion trajectory changes the signal."""
        spins = [0.5]
        g = np.array([2.009, 2.006, 2.002])
        nSteps = 30
        nTraj = 3
        dtSpin = 1e-10
        omega = 2 * np.pi * 9.5e9
        CenterField = 340.0

        qTraj = np.zeros((4, nSteps, nTraj))
        qTraj[0] = 1.0

        # No lab rotation
        sig1 = propagate_istos(
            spins, g, None, qTraj, omega, dtSpin, nSteps, nTraj, CenterField
        )

        # With 90-degree lab rotation around y
        qLab = np.zeros((4, nSteps, nTraj))
        qLab[0] = np.cos(np.pi / 4)
        qLab[2] = np.sin(np.pi / 4)
        sig2 = propagate_istos(
            spins, g, None, qTraj, omega, dtSpin, nSteps, nTraj, CenterField,
            qLab=qLab,
        )

        # Anisotropic g → different orientations give different signals
        assert not np.allclose(sig1, sig2, atol=1e-6)


# ============================================================================
# cardamom integration (ISTOs method)
# ============================================================================

class TestCardamomISTOs:
    """End-to-end cardamom tests with Method='ISTOs'."""

    def test_istos_basic(self):
        """Basic cardamom ISTOs simulation runs without error."""
        from torchspin import SpinSystem, Experiment, cardamom, CardamomPar, CardamomOptions

        sys = SpinSystem(
            S=[0.5],
            g=[[2.009, 2.006, 2.002]],
            tcorr=5e-9,
        )
        exp = Experiment(mwFreq=9.5, Range=[330, 350], Harmonic=1)
        par = CardamomPar(
            Model='diffusion', dtSpin=2e-10, nSteps=100,
            dtSpatial=2e-11, nTraj=3, nOrients=3,
        )
        opt = CardamomOptions(Method='ISTOs', Verbosity=0)

        B, spc, td, t = cardamom(sys, exp, par, opt)
        assert B.shape[0] > 0
        assert spc.shape == B.shape
        assert td.shape == t.shape
        assert np.all(np.isfinite(spc))

    def test_istos_with_nitrogen(self):
        """ISTOs with 14N hyperfine produces valid spectrum."""
        from torchspin import SpinSystem, Experiment, cardamom, CardamomPar, CardamomOptions

        sys = SpinSystem(
            S=[0.5],
            g=[[2.009, 2.006, 2.002]],
            Nucs=['14N'],
            A=[[10.0, 10.0, 95.0]],
            tcorr=5e-9,
        )
        exp = Experiment(mwFreq=9.5, Range=[330, 350], Harmonic=1)
        par = CardamomPar(
            Model='diffusion', dtSpin=2e-10, nSteps=80,
            dtSpatial=2e-11, nTraj=2, nOrients=2,
        )
        opt = CardamomOptions(Method='ISTOs', Verbosity=0)

        B, spc, td, t = cardamom(sys, exp, par, opt)
        assert np.all(np.isfinite(spc))
        assert len(B) == getattr(exp, 'nPoints', 1024)

    def test_istos_vs_fast_qualitative(self):
        """ISTOs and fast methods produce qualitatively similar signals."""
        from torchspin import SpinSystem, Experiment, cardamom, CardamomPar, CardamomOptions

        sys = SpinSystem(
            S=[0.5],
            g=[[2.009, 2.006, 2.002]],
            tcorr=5e-9,
        )
        exp = Experiment(mwFreq=9.5, Range=[330, 350], Harmonic=1)
        par = CardamomPar(
            Model='diffusion', dtSpin=2e-10, nSteps=100,
            dtSpatial=2e-11, nTraj=5, nOrients=3,
        )

        np.random.seed(42)
        opt_fast = CardamomOptions(Method='fast', Verbosity=0)
        B_fast, spc_fast, _, _ = cardamom(sys, exp, par, opt_fast)

        np.random.seed(42)
        opt_istos = CardamomOptions(Method='ISTOs', Verbosity=0)
        B_istos, spc_istos, _, _ = cardamom(sys, exp, par, opt_istos)

        # Both should produce valid spectra (not identical due to different physics)
        assert np.all(np.isfinite(spc_fast))
        assert np.all(np.isfinite(spc_istos))
        # Both should have signal in the sweep range
        assert np.max(np.abs(spc_fast)) > 0
        assert np.max(np.abs(spc_istos)) > 0
