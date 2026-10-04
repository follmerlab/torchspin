"""Tests for torchspin.diptensor — dipolar coupling tensor."""
import math
import numpy as np
import pytest

from torchspin.diptensor import diptensor


class TestDiptensor:
    def test_shape(self):
        """Returns (3, 3) array."""
        T = diptensor(2.0, 2.0, [0, 0, 1.0])
        assert T.shape == (3, 3)

    def test_symmetric(self):
        """Tensor is symmetric."""
        T = diptensor(2.0, 2.0, [1, 1, 1])
        assert np.allclose(T, T.T, atol=1e-12)

    def test_traceless(self):
        """Point-dipole tensor is traceless: Txx + Tyy + Tzz = 0."""
        T = diptensor(2.0, 2.0, [0.5, 0.3, 0.7])
        assert abs(np.trace(T)) < 1e-10

    def test_axial_z(self):
        """Along z: Tzz = -2 * Txx = -2 * Tyy (axial symmetry)."""
        T = diptensor(2.0, 2.0, [0, 0, 1.0])
        assert abs(T[0, 0] - T[1, 1]) < 1e-10       # Txx = Tyy
        assert abs(T[2, 2] + 2 * T[0, 0]) < 1e-10   # Tzz = -2*Txx

    def test_sign_ee_matches_matlab(self):
        """Audit P3G-1 fix: e-e along ẑ gives T_zz < 0 (matches MATLAB).

        MATLAB diptensor.m:109 uses ``T = -mu0/(4π)·mug1'·d·mug2/r³`` with
        ``mug_e = +bmagn·g``, producing a negative T_zz for two electrons
        along ẑ. Prior to this fix, torchspin produced a positive T_zz
        (sign error masked by tests using ``abs(T_zz)``).
        """
        T = diptensor(2.0, 2.0, [0, 0, 1.0])
        assert T[2, 2] < 0, f"e-e T_zz must be negative; got {T[2, 2]:.4f}"
        # Magnitude check (μ₀·g²·μ_B²/4π·h·r³·1e6 ≈ 52 MHz for 1 nm, factor 2 from 3-1)
        assert abs(T[2, 2]) > 50.0, "magnitude unreasonably small"
        assert abs(T[2, 2]) < 250.0, "magnitude unreasonably large"

    def test_sign_en_matches_matlab(self):
        """Audit P3G-1 fix: electron-1H along ẑ gives T_zz > 0."""
        T = diptensor(2.0, '1H', [0, 0, 1.0])
        assert T[2, 2] > 0, f"e-1H T_zz must be positive; got {T[2, 2]:.4f}"

    def test_sign_nn_matches_matlab(self):
        """Audit P3G-1 fix: 1H-1H along ẑ gives T_zz < 0 (both gn>0)."""
        T = diptensor('1H', '1H', [0, 0, 0.3])
        assert T[2, 2] < 0, f"1H-1H T_zz must be negative; got {T[2, 2]:.4e}"

    def test_axial_x(self):
        """Along x: Txx = -2 * Tyy = -2 * Tzz."""
        T = diptensor(2.0, 2.0, [1, 0, 0])
        assert abs(T[1, 1] - T[2, 2]) < 1e-10
        assert abs(T[0, 0] + 2 * T[1, 1]) < 1e-10

    def test_inverse_cube_scaling(self):
        """Magnitude scales as 1/r³."""
        T1 = diptensor(2.0, 2.0, [0, 0, 1.0])
        T2 = diptensor(2.0, 2.0, [0, 0, 2.0])
        ratio = T1[2, 2] / T2[2, 2]
        assert abs(ratio - 8.0) < 1e-8  # (2/1)³ = 8

    def test_electron_electron_order_of_magnitude(self):
        """e-e at 1 nm: Tzz ~ 52 MHz."""
        from torchspin.constants import GFREE
        T = diptensor(GFREE, GFREE, [0, 0, 1.0])
        assert 40 < abs(T[2, 2]) < 200

    def test_proton_proton(self):
        """1H-1H at 0.3 nm (typical H-H distance): Tzz ~ 60 kHz."""
        T = diptensor('1H', '1H', [0, 0, 0.3])
        # 1H gn ≈ 5.585, so much smaller than e-e; expect ~ 0.05-0.1 MHz
        assert abs(T[2, 2]) < 1.0
        assert abs(T[2, 2]) > 1e-3

    def test_electron_proton(self):
        """e-H coupling at 0.5 nm: Tzz ~ 3-10 MHz."""
        from torchspin.constants import GFREE
        T = diptensor(GFREE, '1H', [0, 0, 0.5])
        assert 0.5 < abs(T[2, 2]) < 30

    def test_zero_vector_raises(self):
        with pytest.raises(ValueError, match="non-zero"):
            diptensor(2.0, 2.0, [0, 0, 0])

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            diptensor(2.0, 2.0, [0, 1])

    def test_g_tensor_input(self):
        """g-tensor (3,) or (3,3) input accepted."""
        T1 = diptensor([2.0, 2.0, 2.0], 2.0, [0, 0, 1.0])
        T2 = diptensor(2.0, 2.0, [0, 0, 1.0])
        assert np.allclose(T1, T2, atol=1e-10)


class TestZfsframes:
    def test_axial(self):
        """Axial D-tensor: D≠0, E=0."""
        from torchspin.spinsystem import SpinSystem
        from torchspin.ham_zf import zfsframes
        sys = SpinSystem(S=[0.5], D=[[-500, -500, 1000]])
        D_vals, E_vals, _, _ = zfsframes(sys)
        assert abs(D_vals[0] - 1500) < 1e-6
        assert abs(E_vals[0]) < 1e-6

    def test_rhombic(self):
        """Rhombic D-tensor: D≠0, E≠0."""
        from torchspin.spinsystem import SpinSystem
        from torchspin.ham_zf import zfsframes
        sys = SpinSystem(S=[0.5], D=[[-600, -400, 1000]])
        D_vals, E_vals, _, E0 = zfsframes(sys)
        assert abs(D_vals[0] - 1500) < 1e-6
        assert abs(abs(E_vals[0]) - 100) < 1e-6  # E = ±100 depending on axis ordering

    def test_no_zfs(self):
        """No D → D=E=0 for all electrons."""
        from torchspin.spinsystem import SpinSystem
        from torchspin.ham_zf import zfsframes
        sys = SpinSystem(S=[1.0])
        D_vals, E_vals, _, _ = zfsframes(sys)
        assert D_vals[0] == 0.0
        assert E_vals[0] == 0.0

    def test_traceless_E0(self):
        """Traceless D-tensor: E0 (isotropic offset) ≈ 0."""
        from torchspin.spinsystem import SpinSystem
        from torchspin.ham_zf import zfsframes
        sys = SpinSystem(S=[1.0], D=[[-500, -500, 1000]])
        _, _, _, E0 = zfsframes(sys)
        assert abs(E0[0]) < 1e-6

    def test_euler_angles_shape(self):
        """Euler angles returned per electron."""
        from torchspin.spinsystem import SpinSystem
        from torchspin.ham_zf import zfsframes
        sys = SpinSystem(S=[1.0], D=[[-500, -500, 1000]])
        _, _, Euler, _ = zfsframes(sys)
        assert len(Euler) == 1
        assert len(Euler[0]) == 3
