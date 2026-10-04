"""Tests for torchspin.rotutils — rotation utilities.

All tests use EasySpin's z-y'-z'' passive convention.
"""

import math
import numpy as np
import pytest

from torchspin.rotations import erot   # reference implementation already validated
from torchspin.rotutils import (
    eulang,
    euler2quat,
    quat2euler,
    quat2rotmat,
    rotmat2quat,
    rotaxi2mat,
    rotmat2axi,
    quatinv,
    quatmult,
    quatvecmult,
    vec2ang,
    ang2vec,
    tensor_cart2sph,
    tensor_sph2cart,
    rotateframe,
)

# Convenience: a set of test angle triplets (radians)
_ANGLES = [
    (0.0, 0.0, 0.0),
    (math.pi / 4, math.pi / 3, math.pi / 6),
    (1.2, 0.7, 2.1),
    (0.0, math.pi / 2, 0.0),
    (math.pi, math.pi / 4, math.pi),
    (2.5, 0.1, 0.3),
]


def _make_quat(a, b, g):
    """Build reference unit quaternion from euler2quat."""
    return euler2quat(np.array([a, b, g]))


def _identity_quat():
    return np.array([1.0, 0.0, 0.0, 0.0])


# ===========================================================================
# quatinv
# ===========================================================================

class TestQuatinv:
    def test_identity_inverted(self):
        q = _identity_quat()
        assert np.allclose(quatinv(q), q)

    def test_conjugate_structure(self):
        q = np.array([0.6, 0.2, 0.7, 0.327])
        q /= np.linalg.norm(q)
        qinv = quatinv(q)
        assert np.isclose(qinv[0], q[0])
        assert np.allclose(qinv[1:], -q[1:])

    def test_double_inverse_identity(self):
        q = np.array([0.5, 0.5, 0.5, 0.5])
        assert np.allclose(quatinv(quatinv(q)), q)

    def test_batch(self):
        q = np.column_stack([_make_quat(*a) for a in _ANGLES[:3]])
        qi = quatinv(q)
        assert qi.shape == (4, 3)
        assert np.allclose(qi[0], q[0])
        assert np.allclose(qi[1:], -q[1:])


# ===========================================================================
# quatmult
# ===========================================================================

class TestQuatmult:
    def test_identity_left(self):
        """q ⊗ identity = q."""
        for a in _ANGLES:
            q = _make_quat(*a)
            e = _identity_quat()
            assert np.allclose(quatmult(q, e), q, atol=1e-12)

    def test_identity_right(self):
        """identity ⊗ q = q."""
        for a in _ANGLES:
            q = _make_quat(*a)
            e = _identity_quat()
            assert np.allclose(quatmult(e, q), q, atol=1e-12)

    def test_inverse_gives_identity(self):
        """q ⊗ q⁻¹ = identity."""
        for a in _ANGLES:
            q = _make_quat(*a)
            t = quatmult(q, quatinv(q))
            assert np.allclose(np.abs(t), [1, 0, 0, 0], atol=1e-12)

    def test_normalised_output(self):
        """Product of unit quaternions is unit quaternion."""
        q = _make_quat(1.0, 0.5, 2.0)
        r = _make_quat(0.3, 1.2, 0.7)
        t = quatmult(q, r)
        assert abs(np.linalg.norm(t) - 1.0) < 1e-12

    def test_not_commutative(self):
        q = _make_quat(0.3, 0.7, 1.1)
        r = _make_quat(1.5, 0.2, 0.4)
        assert not np.allclose(quatmult(q, r), quatmult(r, q))


# ===========================================================================
# quatvecmult
# ===========================================================================

class TestQuatvecmult:
    def test_identity_no_rotation(self):
        """Identity quaternion leaves vector unchanged."""
        v = np.array([1.0, 2.0, 3.0])
        r = quatvecmult(_identity_quat(), v)
        assert np.allclose(r, v, atol=1e-12)

    def test_z_rotation_90(self):
        """90° rotation around z: x→y, y→-x."""
        q = euler2quat(np.array([math.pi / 2, 0.0, 0.0]), convention='active')
        v = np.array([1.0, 0.0, 0.0])
        r = quatvecmult(q, v)
        assert np.allclose(r, [0.0, 1.0, 0.0], atol=1e-12)

    def test_preserves_length(self):
        """Rotation preserves vector length."""
        for a in _ANGLES:
            q = euler2quat(np.array(a), convention='active')
            v = np.array([1.0, 2.0, -0.5])
            r = quatvecmult(q, v)
            assert abs(np.linalg.norm(r) - np.linalg.norm(v)) < 1e-10

    def test_batch_vectors(self):
        q = _make_quat(0.5, 0.3, 1.0)
        vs = np.random.default_rng(0).standard_normal((3, 5))
        r = quatvecmult(q[:, np.newaxis], vs)
        assert r.shape == (3, 5)
        # Check each column matches single-vector call
        for k in range(5):
            rk = quatvecmult(q, vs[:, k])
            assert np.allclose(r[:, k], rk, atol=1e-12)


# ===========================================================================
# quat2rotmat / rotmat2quat
# ===========================================================================

class TestQuat2Rotmat:
    def test_identity_quat(self):
        R = quat2rotmat(_identity_quat())
        assert np.allclose(R, np.eye(3), atol=1e-12)

    def test_orthogonal(self):
        """quat2rotmat output is orthogonal (R.T @ R = I)."""
        for a in _ANGLES:
            q = _make_quat(*a)
            R = quat2rotmat(q)
            assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)

    def test_det_plus_one(self):
        for a in _ANGLES:
            q = _make_quat(*a)
            R = quat2rotmat(q)
            assert abs(np.linalg.det(R) - 1.0) < 1e-12

    def test_roundtrip(self):
        """rotmat2quat(quat2rotmat(q)) recovers original q."""
        for a in _ANGLES:
            q = _make_quat(*a)
            R = quat2rotmat(q)
            q2 = rotmat2quat(R)
            # q and -q give same rotation; both q0>=0 here so should match
            assert np.allclose(q, q2, atol=1e-12) or np.allclose(q, -q2, atol=1e-12)

    def test_matches_erot(self):
        """quat2rotmat(euler2quat(angles)) should match erot(angles)."""
        for a, b, g in _ANGLES:
            q = euler2quat(np.array([a, b, g]))
            R_q = quat2rotmat(q)
            R_e = erot(a, b, g).numpy()
            assert np.allclose(R_q, R_e, atol=1e-12), f"Mismatch at {(a,b,g)}"

    def test_batch(self):
        q_batch = np.column_stack([_make_quat(*a) for a in _ANGLES[:4]])
        R_batch = quat2rotmat(q_batch)
        assert R_batch.shape == (3, 3, 4)


class TestRotmat2Quat:
    def test_identity_matrix(self):
        q = rotmat2quat(np.eye(3))
        assert np.allclose(q, [1, 0, 0, 0], atol=1e-12)

    def test_q0_non_negative(self):
        for a in _ANGLES:
            R = erot(*a).numpy()
            q = rotmat2quat(R)
            assert q[0] >= -1e-12

    def test_normalised(self):
        for a in _ANGLES:
            R = erot(*a).numpy()
            q = rotmat2quat(R)
            assert abs(np.linalg.norm(q) - 1.0) < 1e-12


# ===========================================================================
# eulang
# ===========================================================================

class TestEulang:
    def test_identity_matrix(self):
        angles = eulang(np.eye(3))
        # Identity: beta=0, alpha=gamma=0 (degenerate case)
        assert np.isclose(angles[1], 0.0)

    def test_roundtrip_via_erot(self):
        """eulang(erot(angles)) recovers the original angles."""
        for a, b, g in _ANGLES:
            R = erot(a, b, g).numpy()
            angles = eulang(R)
            R2 = erot(*angles).numpy()
            # Both rotation matrices must be identical (angles may differ in degenerate cases)
            assert np.allclose(R, R2, atol=1e-10), f"Failed for {(a,b,g)}: got {angles}"

    def test_beta_non_negative(self):
        for a, b, g in _ANGLES:
            R = erot(a, b, g).numpy()
            angles = eulang(R)
            assert angles[1] >= -1e-10

    def test_non_orthogonal_raises(self):
        R = np.ones((3, 3)) * 2.0
        with pytest.raises(ValueError, match="orthogonal"):
            eulang(R)

    def test_negative_det_raises(self):
        R = -np.eye(3)
        with pytest.raises(ValueError, match="determinant"):
            eulang(R)


# ===========================================================================
# euler2quat / quat2euler
# ===========================================================================

class TestEuler2Quat:
    def test_zero_angles_identity(self):
        q = euler2quat(np.array([0.0, 0.0, 0.0]))
        assert np.allclose(q, [1, 0, 0, 0], atol=1e-12)

    def test_unit_norm(self):
        for a in _ANGLES:
            q = euler2quat(np.array(a))
            assert abs(np.linalg.norm(q) - 1.0) < 1e-12

    def test_q0_non_negative(self):
        for a in _ANGLES:
            q = euler2quat(np.array(a))
            assert q[0] >= -1e-12

    def test_three_arg_form(self):
        a, b, g = 0.5, 0.3, 1.2
        q1 = euler2quat(np.array([a, b, g]))
        q2 = euler2quat(a, b, g)
        assert np.allclose(q1, q2, atol=1e-15)

    def test_passive_active_differ(self):
        a, b, g = 0.5, 0.7, 1.1
        q_pas = euler2quat(np.array([a, b, g]), convention='passive')
        q_act = euler2quat(np.array([a, b, g]), convention='active')
        # They should give different quaternions for non-zero angles
        assert not np.allclose(q_pas, q_act)

    def test_active_passive_are_inverses(self):
        """euler2quat(angles, active) = quatinv(euler2quat(angles, passive))."""
        for a in _ANGLES:
            q_pas = euler2quat(np.array(a), convention='passive')
            q_act = euler2quat(np.array(a), convention='active')
            # Active = inverse of passive (up to global sign)
            diff = np.abs(np.abs(quatmult(q_pas, q_act)[0]) - 1.0)
            assert diff < 1e-12


class TestQuat2Euler:
    def test_zero_angles_roundtrip(self):
        angles_in = np.array([0.0, 0.0, 0.0])
        q = euler2quat(angles_in)
        a, b, g = quat2euler(q)
        assert np.isclose(float(b), 0.0, atol=1e-10)

    def test_roundtrip(self):
        """euler2quat → quat2euler should recover the original rotation."""
        for a, b, g in _ANGLES:
            q = euler2quat(np.array([a, b, g]))
            a2, b2, g2 = quat2euler(q)
            # Verify via rotation matrix (angles may differ in degenerate cases)
            R_in = erot(a, b, g).numpy()
            R_out = erot(float(a2), float(b2), float(g2)).numpy()
            assert np.allclose(R_in, R_out, atol=1e-10), \
                f"Input ({a:.3f},{b:.3f},{g:.3f}) → ({float(a2):.3f},{float(b2):.3f},{float(g2):.3f})"

    def test_beta_non_negative(self):
        for a in _ANGLES:
            q = euler2quat(np.array(a))
            _, b, _ = quat2euler(q)
            assert float(b) >= -1e-10

    def test_unnormalised_raises(self):
        q = np.array([1.0, 0.1, 0.0, 0.0])  # not normalised
        with pytest.raises(ValueError, match="normalised"):
            quat2euler(q)


# ===========================================================================
# rotaxi2mat
# ===========================================================================

class TestRotaxi2mat:
    def test_zero_angle_identity(self):
        R = rotaxi2mat('z', 0.0)
        assert np.allclose(R, np.eye(3), atol=1e-10)

    def test_z_axis_90(self):
        """90° rotation around z: passive matrix, apply as R.T @ v (EasySpin convention)."""
        R = rotaxi2mat('z', math.pi / 2)
        # MATLAB: v_rot = R.'*v  (transpose); x → y after 90° rotation
        assert np.allclose(R.T @ np.array([1, 0, 0]), [0, 1, 0], atol=1e-12)

    def test_x_axis_180(self):
        """180° rotation around x: y → -y, z → -z."""
        R = rotaxi2mat('x', math.pi)
        assert np.allclose(R @ np.array([0, 1, 0]), [0, -1, 0], atol=1e-10)
        assert np.allclose(R @ np.array([0, 0, 1]), [0, 0, -1], atol=1e-10)

    def test_string_shortcuts(self):
        for key in ['x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz']:
            R = rotaxi2mat(key, math.pi / 3)
            assert R.shape == (3, 3)
            assert np.allclose(R.T @ R, np.eye(3), atol=1e-10)

    def test_orthogonal(self):
        for a, b, g in _ANGLES:
            R = rotaxi2mat([1, 1, 0], b)
            assert np.allclose(R.T @ R, np.eye(3), atol=1e-10)
            assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_unnormalized_axis(self):
        """Unnormalized axis should give same result as normalized."""
        R1 = rotaxi2mat([2.0, 0.0, 0.0], math.pi / 4)
        R2 = rotaxi2mat([1.0, 0.0, 0.0], math.pi / 4)
        assert np.allclose(R1, R2, atol=1e-12)

    def test_bad_axis_raises(self):
        with pytest.raises(ValueError):
            rotaxi2mat([1, 2], 1.0)


# ===========================================================================
# rotmat2axi
# ===========================================================================

class TestRotmat2axi:
    def test_identity_zero_angle(self):
        n, rho = rotmat2axi(np.eye(3))
        assert abs(rho) < 1e-10

    def test_roundtrip(self):
        """rotaxi2mat → rotmat2axi → rotaxi2mat should give same matrix."""
        for ax in [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]:
            for rho in [0.3, 1.0, 2.0, math.pi / 2]:
                R = rotaxi2mat(ax, rho)
                n2, rho2 = rotmat2axi(R)
                R2 = rotaxi2mat(n2, rho2)
                assert np.allclose(R, R2, atol=1e-10), \
                    f"Roundtrip failed for axis={ax}, rho={rho:.2f}"

    def test_unit_axis(self):
        for ax in [[1, 1, 1], [0.5, 0.5, 0.0]]:
            R = rotaxi2mat(ax, 1.0)
            n, rho = rotmat2axi(R)
            assert abs(np.linalg.norm(n) - 1.0) < 1e-10

    def test_rho_non_negative(self):
        for ax in ['x', 'y', 'z']:
            for rho_in in [0.1, 1.0, 2.5]:
                R = rotaxi2mat(ax, rho_in)
                _, rho_out = rotmat2axi(R)
                assert rho_out >= -1e-12


# ---------------------------------------------------------------------------
# vec2ang / ang2vec
# ---------------------------------------------------------------------------

class TestVec2ang:
    def test_z_axis(self):
        """[0,0,1] → theta=0, phi=0."""
        phi, theta = vec2ang([0, 0, 1])
        assert abs(theta) < 1e-10

    def test_x_axis(self):
        """[1,0,0] → phi=0, theta=pi/2."""
        phi, theta = vec2ang([1, 0, 0])
        assert abs(phi) < 1e-10
        assert abs(theta - np.pi / 2) < 1e-10

    def test_y_axis(self):
        """[0,1,0] → phi=pi/2, theta=pi/2."""
        phi, theta = vec2ang([0, 1, 0])
        assert abs(phi - np.pi / 2) < 1e-10
        assert abs(theta - np.pi / 2) < 1e-10

    def test_minus_z(self):
        """[0,0,-1] → theta=pi."""
        phi, theta = vec2ang([0, 0, -1])
        assert abs(theta - np.pi) < 1e-10

    def test_phi_range(self):
        """phi should be in [0, 2*pi)."""
        phi, theta = vec2ang([-1, 0, 0])
        assert 0 <= phi < 2 * np.pi + 1e-10

    def test_unnormalized_vector(self):
        """Non-unit vectors should give same angles as unit vectors."""
        phi1, theta1 = vec2ang([3, 0, 0])
        phi2, theta2 = vec2ang([1, 0, 0])
        assert abs(phi1 - phi2) < 1e-10
        assert abs(theta1 - theta2) < 1e-10

    def test_batch(self):
        """Array input → array output."""
        v = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).T  # shape (3,3)
        phi, theta = vec2ang(v)
        assert phi.shape == (3,)
        assert theta.shape == (3,)


class TestAng2vec:
    def test_z_axis(self):
        """phi=0, theta=0 → [0,0,1]."""
        v = ang2vec(0, 0)
        assert abs(v[0]) < 1e-10
        assert abs(v[1]) < 1e-10
        assert abs(v[2] - 1) < 1e-10

    def test_x_axis(self):
        """phi=0, theta=pi/2 → [1,0,0]."""
        v = ang2vec(0, np.pi / 2)
        assert abs(v[0] - 1) < 1e-10
        assert abs(v[1]) < 1e-10
        assert abs(v[2]) < 1e-10

    def test_unit_length(self):
        """Output vector should have unit norm."""
        for phi in [0, 0.5, 1.0, 2.0]:
            for theta in [0, 0.3, 1.0, np.pi]:
                v = ang2vec(phi, theta)
                assert abs(np.linalg.norm(v) - 1) < 1e-10

    def test_roundtrip(self):
        """ang2vec(vec2ang(v)) ≈ v (for unit vectors)."""
        import itertools
        for phi0, theta0 in itertools.product([0.1, 1.0, 2.5], [0.3, 1.0, 2.0]):
            v0 = ang2vec(phi0, theta0)
            phi1, theta1 = vec2ang(v0)
            v1 = ang2vec(phi1, theta1)
            assert np.linalg.norm(v0 - v1) < 1e-10

    def test_batch(self):
        """Array phi/theta → (3, N) output."""
        phi = np.array([0.0, np.pi / 2, np.pi])
        theta = np.array([np.pi / 2, np.pi / 2, np.pi / 2])
        v = ang2vec(phi, theta)
        assert v.shape == (3, 3)
        # All should be unit vectors
        norms = np.linalg.norm(v, axis=0)
        assert np.allclose(norms, 1.0)


# ---------------------------------------------------------------------------
# tensor_cart2sph / tensor_sph2cart
# ---------------------------------------------------------------------------

class TestTensorConversions:
    def test_roundtrip_axial(self):
        """Axial traceless tensor: exact roundtrip."""
        T = np.diag([-1., -1., 2.])
        T2 = tensor_cart2sph(T)
        T_back = tensor_sph2cart(T2)
        assert np.allclose(T_back, T, atol=1e-12)

    def test_roundtrip_rhombic(self):
        """Rhombic traceless tensor: exact roundtrip."""
        T = np.diag([-1., -2., 3.])
        T2 = tensor_cart2sph(T)
        T_back = tensor_sph2cart(T2)
        assert np.allclose(T_back, T, atol=1e-12)

    def test_roundtrip_offdiagonal(self):
        """Traceless tensor with off-diagonal: roundtrip."""
        T = np.array([[1., 0.5, 0.3],
                      [0.5, -2., 0.1],
                      [0.3, 0.1, 1.]])
        T2 = tensor_cart2sph(T)
        T_back = tensor_sph2cart(T2)
        assert np.allclose(T_back, T, atol=1e-12)

    def test_spherical_components_count(self):
        """tensor_cart2sph returns 5 components."""
        T = np.diag([1., -1., 0.])
        T2 = tensor_cart2sph(T)
        assert T2.shape == (5,)

    def test_isotropic_gives_zero_T2(self):
        """Isotropic traceless tensor (zero) → all T2m = 0."""
        T = np.zeros((3, 3))
        T2 = tensor_cart2sph(T)
        assert np.allclose(T2, 0.0, atol=1e-14)

    def test_D_tensor_T20(self):
        """Axial D-tensor (D=1000, E=0): T20 ∝ Dzz."""
        T = np.diag([-500., -500., 1000.])
        T2 = tensor_cart2sph(T)
        # Only T20 should be non-zero for axial tensor
        assert abs(T2[0]) < 1e-10  # T2m2 = 0
        assert abs(T2[1]) < 1e-10  # T2m1 = 0
        assert abs(T2[2]) > 1.0    # T20 ≠ 0
        assert abs(T2[3]) < 1e-10  # T2p1 = 0
        assert abs(T2[4]) < 1e-10  # T2p2 = 0


# ---------------------------------------------------------------------------
# rotateframe
# ---------------------------------------------------------------------------

class TestRotateframe:
    def test_identity_rotation(self):
        """Rotating by 0 returns original Euler angles."""
        ang0 = np.array([0.3, 0.5, 0.1])
        ang1 = rotateframe(ang0, [0, 0, 1], 0.0)
        assert np.allclose(ang1, ang0, atol=1e-10)

    def test_full_rotation(self):
        """Rotating by 2π returns original orientation."""
        ang0 = np.array([0.3, 0.5, 0.1])
        ang1 = rotateframe(ang0, [1, 0, 0], 2 * np.pi)
        # Euler angles may differ by 2π in alpha/gamma — check rotation matrix
        from torchspin.rotations import erot
        R0 = np.array(erot(ang0.tolist()))
        R1 = np.array(erot(ang1.tolist()))
        assert np.allclose(R0, R1, atol=1e-8)

    def test_z_rotation_alpha(self):
        """Rotating the identity frame by π/2 around z → alpha=π/2."""
        ang0 = np.array([0., 0., 0.])
        ang1 = rotateframe(ang0, [0, 0, 1], np.pi / 2)
        assert abs(ang1[0] - np.pi / 2) < 1e-8
        assert abs(ang1[1]) < 1e-8  # beta unchanged

    def test_array_rho(self):
        """Array of rho values returns (M, 3) array."""
        ang0 = np.array([0., 0., 0.])
        rhos = np.linspace(0, np.pi, 5)
        result = rotateframe(ang0, [0, 1, 0], rhos)
        assert result.shape == (5, 3)

    def test_normalization_of_axis(self):
        """Non-unit nRot gives same result as unit nRot."""
        ang0 = np.array([0.3, 0.5, 0.1])
        r1 = rotateframe(ang0, [0, 0, 1], 0.5)
        r2 = rotateframe(ang0, [0, 0, 3], 0.5)
        assert np.allclose(r1, r2, atol=1e-10)
