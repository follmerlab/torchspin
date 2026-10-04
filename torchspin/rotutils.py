"""Rotation utilities for EPR spectroscopy.

Port of EasySpin's rotation-related functions:
- ``eulang.m``       — rotation matrix → Euler angles
- ``euler2quat.m``   — Euler angles → unit quaternion
- ``quat2euler.m``   — unit quaternion → Euler angles
- ``quat2rotmat.m``  — unit quaternion → rotation matrix
- ``rotmat2quat.m``  — rotation matrix → unit quaternion
- ``rotaxi2mat.m``   — axis-angle → rotation matrix
- ``rotmat2axi.m``   — rotation matrix → rotation axis and angle
- ``quatinv.m``      — quaternion inverse (conjugate)
- ``quatmult.m``     — quaternion multiplication (Hamilton product)
- ``quatvecmult.m``  — rotate a 3-vector by a quaternion
- ``vec2ang.m``      — Cartesian 3-vector → (phi, theta) polar angles
- ``ang2vec.m``      — (phi, theta) → unit Cartesian 3-vector

Convention notes
----------------
All functions follow EasySpin's conventions:

* **Euler angles**: z-y'-z'' passive rotation (same as ``erot``).
* **Quaternion layout**: ``q = [q0, q1, q2, q3]`` where q0 is the scalar
  (real) part and ``[q1, q2, q3]`` is the vector part.  Arrays of
  quaternions have shape ``(4, N)``; a single quaternion is ``(4,)`` or
  ``(4, 1)``.
* **Passive vs active**: all functions default to the *passive* (alias)
  convention used throughout EasySpin.

References
----------
EasySpin source files: euler2quat.m, quat2euler.m, quat2rotmat.m,
rotmat2quat.m, rotaxi2mat.m, rotmat2axi.m, eulang.m, quatinv.m,
quatmult.m, quatvecmult.m.
"""

import math
import numpy as np

__all__ = [
    'eulang',
    'euler2quat',
    'quat2euler',
    'quat2rotmat',
    'rotmat2quat',
    'rotaxi2mat',
    'rotmat2axi',
    'quatinv',
    'quatmult',
    'quatvecmult',
    'vec2ang',
    'ang2vec',
    'tensor_cart2sph',
    'tensor_sph2cart',
    'rotateframe',
]


# ---------------------------------------------------------------------------
# eulang — rotation matrix → Euler angles
# ---------------------------------------------------------------------------

def eulang(R: np.ndarray) -> np.ndarray:
    """Extract z-y'-z'' Euler angles from a rotation matrix.

    Parameters
    ----------
    R : array_like, shape (3, 3)
        Real orthogonal rotation matrix with det(R) = +1.

    Returns
    -------
    angles : ndarray, shape (3,)
        ``[alpha, beta, gamma]`` in radians.  ``beta >= 0``;
        ``alpha`` and ``gamma`` are in ``[0, 2π)`` when possible.

    Notes
    -----
    If the matrix is not exactly orthogonal (orthogonality error < 1e-2)
    it is orthogonalised via SVD before extraction.

    Degenerate cases (β = 0 or β = π): the full z-rotation is collected
    in ``alpha`` and ``gamma`` is set to 0.
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3) or not np.isrealobj(R):
        raise ValueError("R must be a real 3×3 matrix")

    err = np.linalg.norm(R.T @ R - np.eye(3))
    if err > 1e-2:
        raise ValueError(f"R is not orthogonal (error = {err:.4g})")
    if err > 1e-6:
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt

    if np.linalg.det(R) < 0:
        raise ValueError("R has negative determinant; not a proper rotation")

    _EPS = 1e-8
    r33 = float(R[2, 2])

    if abs(r33 - 1.0) <= _EPS:          # β = 0 (degenerate)
        alpha = math.atan2(R[0, 1], R[1, 1])
        beta = 0.0
        gamma = 0.0
    elif abs(r33 + 1.0) <= _EPS:        # β = π (degenerate)
        alpha = math.atan2(-R[0, 1], R[1, 1])
        beta = math.pi
        gamma = 0.0
    else:
        alpha = math.atan2(R[2, 1], R[2, 0])
        beta = math.atan2(math.sqrt(R[2, 0] ** 2 + R[2, 1] ** 2), r33)
        gamma = math.atan2(R[1, 2], -R[0, 2])

    # Shift α, γ to [0, 2π) when possible
    _THR = -1e-8
    if alpha < _THR:
        alpha += 2 * math.pi
    if gamma < _THR:
        gamma += 2 * math.pi

    # Ensure β ≥ 0
    if beta < 0:
        beta = -beta
        if alpha < 0:
            alpha += math.pi
            gamma += math.pi
        else:
            alpha -= math.pi
            gamma -= math.pi

    return np.array([alpha, beta, gamma])


# ---------------------------------------------------------------------------
# euler2quat — Euler angles → unit quaternion
# ---------------------------------------------------------------------------

def euler2quat(alpha, beta=None, gamma=None, convention: str = 'passive') -> np.ndarray:
    """Convert z-y'-z'' Euler angles to a unit quaternion.

    Parameters
    ----------
    alpha : float or array_like
        First Euler angle (radians), or a ``(3, …)`` array of all angles.
    beta : float or array_like, optional
        Second Euler angle (radians).
    gamma : float or array_like, optional
        Third Euler angle (radians).
    convention : {'passive', 'active'}
        Rotation convention.  Default ``'passive'``.

    Returns
    -------
    q : ndarray, shape (4,) or (4, N)
        Unit quaternion(s) with ``q[0] >= 0`` (scalar part non-negative).
        Layout: ``q = [q0, q1, q2, q3]``.
    """
    scalar_input = False
    if beta is None and gamma is None:
        angles = np.asarray(alpha, dtype=float)
        if angles.ndim == 1 and angles.size == 3:
            a, b, g = angles
        elif angles.ndim == 2 and angles.shape[0] == 3:
            a = angles[0]
            b = angles[1]
            g = angles[2]
        else:
            raise ValueError("When called with one argument, it must be "
                             "a (3,) vector or (3, N) array")
    else:
        scalar_input = (np.ndim(alpha) == 0 and np.ndim(beta) == 0
                        and np.ndim(gamma) == 0)
        a = np.asarray(alpha, dtype=float).ravel()
        b = np.asarray(beta, dtype=float).ravel()
        g = np.asarray(gamma, dtype=float).ravel()

    convention = convention.lower()
    if convention not in ('passive', 'active'):
        raise ValueError("convention must be 'passive' or 'active'")

    if convention == 'passive':
        # EasySpin passive: negate and swap α↔γ before Hamilton product
        a, b, g = -g, -b, -a

    q0 = np.cos(b / 2) * np.cos((g + a) / 2)
    q1 = np.sin(b / 2) * np.sin((g - a) / 2)
    q2 = np.sin(b / 2) * np.cos((g - a) / 2)
    q3 = np.cos(b / 2) * np.sin((g + a) / 2)

    q = np.array([q0, q1, q2, q3])

    # Enforce q0 ≥ 0 (q and -q represent the same rotation)
    if q.ndim == 1:
        if q[0] < 0:
            q = -q
    else:
        neg = q[0] < 0
        q[:, neg] = -q[:, neg]

    # Squeeze (4,1) → (4,) when all inputs were scalars
    if scalar_input and q.ndim == 2:
        return q[:, 0]
    return q


# ---------------------------------------------------------------------------
# quat2euler — unit quaternion → Euler angles
# ---------------------------------------------------------------------------

def quat2euler(q: np.ndarray, convention: str = 'passive'):
    """Convert a unit quaternion to z-y'-z'' Euler angles.

    Parameters
    ----------
    q : array_like, shape (4,) or (4, N)
        Unit quaternion(s) with layout ``[q0, q1, q2, q3]``.
    convention : {'passive', 'active'}
        Default ``'passive'``.

    Returns
    -------
    alpha, beta, gamma : ndarray
        Euler angles in radians, each shape ``()`` (scalar) or ``(N,)``.
    """
    q = np.asarray(q, dtype=float)
    if q.shape[0] != 4:
        raise ValueError("q must have shape (4,) or (4, N)")

    norm_err = np.abs(1.0 - np.sqrt(np.sum(q ** 2, axis=0)))
    if np.any(norm_err > 1e-8):
        raise ValueError("q is not normalised")

    convention = convention.lower()
    if convention == 'passive':
        q = quatinv(q) + 0.0   # + 0.0 flushes IEEE 754 negative zeros (-0.0 → 0.0)
    elif convention != 'active':
        raise ValueError("convention must be 'passive' or 'active'")

    # Enforce q0 ≥ 0
    if q.ndim == 1:
        if q[0] < 0:
            q = -q
    else:
        neg = q[0] < 0
        q[:, neg] = -q[:, neg]

    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]

    sy = 2 * np.sqrt((q2 * q3 + q1 * q0) ** 2 + (q1 * q3 - q2 * q0) ** 2)

    # +0.0 in arctan2 arguments flushes IEEE 754 negative zeros (-0.0→0.0)
    # to avoid atan2(+0, -0) = π (correct IEEE 754 but wrong for us)
    alpha = np.arctan2(2 * (q2 * q3 - q0 * q1) + 0.0,
                       2 * (q1 * q3 + q0 * q2) + 0.0)
    alpha = np.where(alpha < 0, alpha + 2 * math.pi, alpha)

    beta = np.real(np.arctan2(sy + 0.0, 1 - 2 * q1 ** 2 - 2 * q2 ** 2))

    gamma = np.arctan2(2 * (q2 * q3 + q0 * q1) + 0.0,
                       -2 * (q1 * q3 - q0 * q2) + 0.0)
    gamma = np.where(gamma < 0, gamma + 2 * math.pi, gamma)

    return alpha, beta, gamma


# ---------------------------------------------------------------------------
# quat2rotmat — unit quaternion → rotation matrix
# ---------------------------------------------------------------------------

def quat2rotmat(q: np.ndarray) -> np.ndarray:
    """Convert a unit quaternion to a rotation matrix.

    Parameters
    ----------
    q : array_like, shape (4,) or (4, N)
        Unit quaternion(s).

    Returns
    -------
    R : ndarray, shape (3, 3) or (3, 3, N)
        Rotation matrix/matrices.
    """
    q = np.asarray(q, dtype=float)
    if q.shape[0] != 4:
        raise ValueError("q must have shape (4,) or (4, N)")

    norm_err = np.abs(1.0 - np.sqrt(np.sum(q ** 2, axis=0)))
    if np.any(norm_err > 1e-8):
        raise ValueError("q is not normalised")

    single = q.ndim == 1
    if single:
        q = q[:, np.newaxis]

    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    N = q.shape[1]
    R = np.zeros((3, 3, N))

    R[0, 0] = 1 - 2 * q2 ** 2 - 2 * q3 ** 2
    R[0, 1] = 2 * q1 * q2 - 2 * q0 * q3
    R[0, 2] = 2 * q1 * q3 + 2 * q0 * q2
    R[1, 0] = 2 * q1 * q2 + 2 * q0 * q3
    R[1, 1] = 1 - 2 * q1 ** 2 - 2 * q3 ** 2
    R[1, 2] = 2 * q2 * q3 - 2 * q0 * q1
    R[2, 0] = 2 * q1 * q3 - 2 * q0 * q2
    R[2, 1] = 2 * q2 * q3 + 2 * q0 * q1
    R[2, 2] = 1 - 2 * q1 ** 2 - 2 * q2 ** 2

    if single:
        return R[:, :, 0]
    return R


# ---------------------------------------------------------------------------
# rotmat2quat — rotation matrix → unit quaternion
# ---------------------------------------------------------------------------

def rotmat2quat(R: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a unit quaternion.

    Parameters
    ----------
    R : array_like, shape (3, 3) or (3, 3, N)
        Proper rotation matrix/matrices.

    Returns
    -------
    q : ndarray, shape (4,) or (4, N)
        Unit quaternion(s) with q[0] >= 0.
    """
    R = np.asarray(R, dtype=float)
    if R.shape[:2] != (3, 3):
        raise ValueError("R must have shape (3, 3) or (3, 3, N)")

    single = R.ndim == 2
    if single:
        R = R[:, :, np.newaxis]

    N = R.shape[2]
    q = np.zeros((4, N))

    for k in range(N):
        M = R[:, :, k]
        tr = M[0, 0] + M[1, 1] + M[2, 2]
        if tr > 0:
            S = 2 * math.sqrt(tr + 1)
            q[0, k] = S / 4
            q[1, k] = (M[2, 1] - M[1, 2]) / S
            q[2, k] = (M[0, 2] - M[2, 0]) / S
            q[3, k] = (M[1, 0] - M[0, 1]) / S
        elif M[0, 0] > M[1, 1] and M[0, 0] > M[2, 2]:
            S = 2 * math.sqrt(1 + M[0, 0] - M[1, 1] - M[2, 2])
            q[0, k] = (M[2, 1] - M[1, 2]) / S
            q[1, k] = S / 4
            q[2, k] = (M[0, 1] + M[1, 0]) / S
            q[3, k] = (M[0, 2] + M[2, 0]) / S
        elif M[1, 1] > M[2, 2]:
            S = 2 * math.sqrt(1 + M[1, 1] - M[0, 0] - M[2, 2])
            q[0, k] = (M[0, 2] - M[2, 0]) / S
            q[1, k] = (M[0, 1] + M[1, 0]) / S
            q[2, k] = S / 4
            q[3, k] = (M[1, 2] + M[2, 1]) / S
        else:
            S = 2 * math.sqrt(1 + M[2, 2] - M[0, 0] - M[1, 1])
            q[0, k] = (M[1, 0] - M[0, 1]) / S
            q[1, k] = (M[0, 2] + M[2, 0]) / S
            q[2, k] = (M[1, 2] + M[2, 1]) / S
            q[3, k] = S / 4

    # Enforce q0 ≥ 0
    neg = q[0] < 0
    q[:, neg] = -q[:, neg]

    if single:
        return q[:, 0]
    return q


# ---------------------------------------------------------------------------
# rotaxi2mat — axis-angle → rotation matrix  (Rodrigues formula)
# ---------------------------------------------------------------------------

_AXIS_SHORTCUTS = {
    'x':   np.array([1.0, 0.0, 0.0]),
    'y':   np.array([0.0, 1.0, 0.0]),
    'z':   np.array([0.0, 0.0, 1.0]),
    'xy':  np.array([1.0, 1.0, 0.0]),
    'xz':  np.array([1.0, 0.0, 1.0]),
    'yz':  np.array([0.0, 1.0, 1.0]),
    'xyz': np.array([1.0, 1.0, 1.0]),
}


def rotaxi2mat(n, rho: float) -> np.ndarray:
    """Rotation matrix from an axis and angle (Rodrigues formula).

    Parameters
    ----------
    n : str or array_like, shape (3,)
        Rotation axis.  Accepts string shortcuts ``'x'``, ``'y'``, ``'z'``,
        ``'xy'``, ``'xz'``, ``'yz'``, ``'xyz'``.  Does not need to be normalised.
    rho : float
        Rotation angle in radians.

    Returns
    -------
    R : ndarray, shape (3, 3)
        Rotation matrix.  Apply as ``v_rot = R.T @ v`` (passive convention).

    Notes
    -----
    Uses the Rodrigues formula: ``R = I + N·sin(ρ) + N²·(1−cos(ρ))``,
    where N is the skew-symmetric cross-product matrix.
    """
    if isinstance(n, str):
        key = n.lower()
        if key not in _AXIS_SHORTCUTS:
            raise ValueError(f"Unknown axis shortcut '{n}'. "
                             "Valid: 'x','y','z','xy','xz','yz','xyz'")
        n_vec = _AXIS_SHORTCUTS[key].copy()
    else:
        n_vec = np.asarray(n, dtype=float).ravel()
        if n_vec.size != 3:
            raise ValueError("Rotation axis must be a 3-element vector")

    norm = np.linalg.norm(n_vec)
    if norm == 0:
        raise ValueError("Rotation axis must be non-zero")
    n_vec = n_vec / norm

    # EasySpin's N matrix: transpose of standard cross-product matrix.
    # Rodrigues formula with this N gives the PASSIVE rotation matrix
    # such that v_rot = R.T @ v rotates v by rho around n.
    nx, ny, nz = n_vec
    N = np.array([[0,  nz, -ny],
                  [-nz, 0,  nx],
                  [ny, -nx,  0]], dtype=float)

    R = np.eye(3) + N * math.sin(rho) + N @ N * (1 - math.cos(rho))

    # Clean up near-exact values
    _EPS = 1e-10
    R[np.abs(R) < _EPS] = 0.0
    R[np.abs(R - 1) < _EPS] = 1.0
    R[np.abs(R + 1) < _EPS] = -1.0

    return R


# ---------------------------------------------------------------------------
# rotmat2axi — rotation matrix → axis and angle
# ---------------------------------------------------------------------------

def rotmat2axi(R: np.ndarray):
    """Extract the rotation axis and angle from a rotation matrix.

    Parameters
    ----------
    R : array_like, shape (3, 3)
        Proper rotation matrix.

    Returns
    -------
    n : ndarray, shape (3,)
        Unit rotation axis.
    rho : float
        Rotation angle in radians (in ``[0, π]``).
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3) or not np.isrealobj(R):
        raise ValueError("R must be a real 3×3 matrix")

    tr = float(np.trace(R))
    if abs(tr - 3.0) < 1e-10:   # identity: no rotation
        return np.array([0.0, 0.0, 1.0]), 0.0

    rho = float(np.real(np.arccos(np.clip((tr - 1) / 2, -1, 1))))
    n = np.array([R[1, 2] - R[2, 1],
                  R[2, 0] - R[0, 2],
                  R[0, 1] - R[1, 0]], dtype=float)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-12:
        # ρ = π case: find eigenvector for eigenvalue +1
        # Fallback: axis from diagonal
        n = np.array([R[0, 0] + 1, R[1, 0], R[2, 0]], dtype=float)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            n = np.array([0.0, 0.0, 1.0])
        else:
            n = n / n_norm
    else:
        n = n / n_norm

    return n, rho


# ---------------------------------------------------------------------------
# quatinv — quaternion inverse (conjugate for unit quaternions)
# ---------------------------------------------------------------------------

def quatinv(q: np.ndarray) -> np.ndarray:
    """Invert a unit quaternion (= conjugate).

    Parameters
    ----------
    q : array_like, shape (4,) or (4, N)
        Unit quaternion(s).

    Returns
    -------
    qinv : ndarray, same shape as *q*
        Inverse: ``[q0, -q1, -q2, -q3]``.
    """
    q = np.asarray(q, dtype=float)
    if q.shape[0] != 4:
        raise ValueError("q must have shape (4,) or (4, N)")
    sign = np.array([1.0, -1.0, -1.0, -1.0])
    if q.ndim == 1:
        return q * sign
    return q * sign[:, np.newaxis]


# ---------------------------------------------------------------------------
# quatmult — Hamilton product of two quaternions
# ---------------------------------------------------------------------------

def quatmult(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Quaternion multiplication (Hamilton product) q ⊗ r.

    Parameters
    ----------
    q, r : array_like, shape (4,) or (4, N)
        Unit quaternions.  Both must have the same shape.

    Returns
    -------
    t : ndarray, same shape as *q*
        Product quaternion.
    """
    q = np.asarray(q, dtype=float)
    r = np.asarray(r, dtype=float)
    if q.shape != r.shape or q.shape[0] != 4:
        raise ValueError("q and r must both have shape (4,) or (4, N) and be equal size")

    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    r0, r1, r2, r3 = r[0], r[1], r[2], r[3]

    t = np.empty_like(q)
    t[0] = r0 * q0 - r1 * q1 - r2 * q2 - r3 * q3
    t[1] = r0 * q1 + r1 * q0 - r2 * q3 + r3 * q2
    t[2] = r0 * q2 + r1 * q3 + r2 * q0 - r3 * q1
    t[3] = r0 * q3 - r1 * q2 + r2 * q1 + r3 * q0

    return t


# ---------------------------------------------------------------------------
# quatvecmult — rotate a vector by a quaternion
# ---------------------------------------------------------------------------

def quatvecmult(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a 3-vector (or batch of vectors) by a unit quaternion.

    Parameters
    ----------
    q : array_like, shape (4,) or (4, N)
        Unit quaternion(s).
    v : array_like, shape (3,) or (3, N)
        Vector(s) to rotate.

    Returns
    -------
    r : ndarray, shape (3,) or (3, N)
        Rotated vector(s).

    Notes
    -----
    Computes ``r = q ⊗ [0,v] ⊗ q⁻¹`` (pure quaternion sandwich product).
    """
    q = np.asarray(q, dtype=float)
    v = np.asarray(v, dtype=float)

    if q.shape[0] != 4:
        raise ValueError("q must have leading dimension 4")
    if v.shape[0] != 3:
        raise ValueError("v must have leading dimension 3")

    single_q = q.ndim == 1
    single_v = v.ndim == 1

    if single_q:
        q = q[:, np.newaxis]
    if single_v:
        v = v[:, np.newaxis]

    N = max(q.shape[1], v.shape[1])
    if q.shape[1] == 1 and N > 1:
        q = np.repeat(q, N, axis=1)
    if v.shape[1] == 1 and N > 1:
        v = np.repeat(v, N, axis=1)

    # Pure quaternion qv = [0, v]
    qv = np.vstack([np.zeros((1, N)), v])

    result = quatmult(quatmult(q, qv), quatinv(q))
    r = result[1:, :]

    if single_q and single_v:
        return r[:, 0]
    return r


# ---------------------------------------------------------------------------
# vec2ang — Cartesian 3-vector → (phi, theta) polar angles
# ---------------------------------------------------------------------------

def vec2ang(v) -> tuple:
    """Convert a Cartesian 3-vector to polar angles (phi, theta).

    Parameters
    ----------
    v : array_like, shape (3,) or (3, N)
        Unit or non-unit Cartesian vector(s).  Non-unit vectors are
        normalised internally.

    Returns
    -------
    phi : float or ndarray
        Azimuthal angle(s) in radians, in ``[0, 2π)``.
    theta : float or ndarray
        Polar angle(s) from +z axis, in ``[0, π]``.

    Notes
    -----
    Matches EasySpin's ``vec2ang.m`` convention:
    ``v = [sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)]``.

    Examples
    --------
    >>> vec2ang([0, 0, 1])   # +z axis
    (0.0, 0.0)
    >>> vec2ang([1, 0, 0])   # +x axis
    (0.0, 1.5707963...)
    """
    v = np.asarray(v, dtype=float)
    single = v.ndim == 1
    if single:
        v = v[:, np.newaxis]

    # Normalise
    norms = np.linalg.norm(v, axis=0, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    v = v / norms

    x, y, z = v[0], v[1], v[2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.mod(np.arctan2(y, x), 2 * math.pi)

    if single:
        return float(phi[0]), float(theta[0])
    return phi, theta


# ---------------------------------------------------------------------------
# ang2vec — (phi, theta) → unit Cartesian 3-vector
# ---------------------------------------------------------------------------

def ang2vec(phi, theta) -> np.ndarray:
    """Convert polar angles to a unit Cartesian 3-vector.

    Parameters
    ----------
    phi : float or array_like
        Azimuthal angle(s) in radians.
    theta : float or array_like
        Polar angle(s) from +z axis, in radians.

    Returns
    -------
    v : ndarray, shape (3,) or (3, N)
        Unit Cartesian vector(s):
        ``[sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)]``.

    Notes
    -----
    Inverse of ``vec2ang``.  Matches EasySpin's ``ang2vec.m``.

    Examples
    --------
    >>> ang2vec(0, 0)          # +z axis
    array([0., 0., 1.])
    >>> ang2vec(0, np.pi/2)    # +x axis
    array([1., 0., 0.])
    """
    phi = np.asarray(phi, dtype=float)
    theta = np.asarray(theta, dtype=float)
    scalar = phi.ndim == 0 and theta.ndim == 0
    phi = np.atleast_1d(phi)
    theta = np.atleast_1d(theta)

    st = np.sin(theta)
    v = np.array([st * np.cos(phi), st * np.sin(phi), np.cos(theta)])

    if scalar:
        return v[:, 0]
    return v


# ---------------------------------------------------------------------------
# tensor_cart2sph — 3×3 Cartesian tensor → rank-2 spherical components
# ---------------------------------------------------------------------------

def tensor_cart2sph(T) -> np.ndarray:
    """Convert a 3×3 Cartesian tensor to rank-2 spherical tensor components.

    Port of EasySpin's ``tensor_cart2sph.m``.

    Parameters
    ----------
    T : array_like, shape (3, 3)
        Symmetric 3×3 Cartesian tensor.

    Returns
    -------
    T2 : ndarray, shape (5,)
        Spherical tensor components ``[T2m2, T2m1, T20, T2p1, T2p2]``
        corresponding to q = −2, −1, 0, +1, +2 (complex in general).

    Notes
    -----
    Conversion formulae (Edmonds convention, same as EasySpin)::

        T20  = sqrt(3/2) * Tzz - (Txx + Tyy) / sqrt(6)
        T2p1 = -(Txz + i*Tyz)         [= -T_{+1}]
        T2m1 =  (Txz - i*Tyz)         [= -T_{-1}*]
        T2p2 =  (Txx - Tyy)/2 + i*Txy
        T2m2 =  (Txx - Tyy)/2 - i*Txy
    """
    T = np.asarray(T, dtype=complex)
    Txx, Txy, Txz = T[0, 0], T[0, 1], T[0, 2]
    Tyy, Tyz = T[1, 1], T[1, 2]
    Tzz = T[2, 2]

    T20  = math.sqrt(3.0 / 2.0) * Tzz - (Txx + Tyy) / math.sqrt(6.0)
    T2p1 = -(Txz + 1j * Tyz)
    T2m1 =  (Txz - 1j * Tyz)
    T2p2 = (Txx - Tyy) / 2.0 + 1j * Txy
    T2m2 = (Txx - Tyy) / 2.0 - 1j * Txy

    return np.array([T2m2, T2m1, T20, T2p1, T2p2])


# ---------------------------------------------------------------------------
# tensor_sph2cart — rank-2 spherical components → 3×3 Cartesian tensor
# ---------------------------------------------------------------------------

def tensor_sph2cart(T2) -> np.ndarray:
    """Convert rank-2 spherical tensor components to a 3×3 Cartesian tensor.

    Port of EasySpin's ``tensor_sph2cart.m``.  Inverse of ``tensor_cart2sph``.

    Parameters
    ----------
    T2 : array_like, shape (5,)
        Spherical components ``[T2m2, T2m1, T20, T2p1, T2p2]``
        (q = −2, −1, 0, +1, +2).

    Returns
    -------
    T : ndarray, shape (3, 3)
        Symmetric **traceless** 3×3 Cartesian tensor (real if input is a
        real traceless tensor).

    Notes
    -----
    The rank-2 spherical components capture only the **traceless** part of a
    symmetric tensor (5 independent components).  The isotropic (trace) part
    is not recoverable from T2 alone.  The reconstruction assumes Tr(T) = 0.

    Inverse formulae (derived from ``tensor_cart2sph`` with Tr(T)=0):

    * ``Tzz  = T20 * sqrt(6) / 4``
    * ``Txx  = -Tzz/2 + Re(T2m2 + T2p2) / 2``
    * ``Tyy  = -Tzz/2 - Re(T2m2 + T2p2) / 2``
    * ``Txy  = Im(T2p2 - T2m2) / 2``
    * ``Txz  = Re(T2m1 - T2p1) / 2``
    * ``Tyz  = -Im(T2m1 + T2p1) / 2``
    """
    T2 = np.asarray(T2, dtype=complex)
    T2m2, T2m1, T20, T2p1, T2p2 = T2

    # Diagonal: derived from traceless constraint (Txx + Tyy = -Tzz)
    # c = sqrt(3/2) + 1/sqrt(6) = 4/sqrt(6), so Tzz = T20/c = T20*sqrt(6)/4
    Tzz = float(np.real(T20)) * math.sqrt(6.0) / 4.0
    diag_half = -Tzz / 2.0
    cross = np.real(T2m2 + T2p2) / 2.0  # = (Txx - Tyy) / 2
    Txx = diag_half + cross
    Tyy = diag_half - cross

    # Off-diagonal
    Txy = np.imag(T2p2 - T2m2) / 2.0
    Txz = np.real(T2m1 - T2p1) / 2.0
    Tyz = -np.imag(T2p1 + T2m1) / 2.0

    T = np.array([
        [Txx, Txy, Txz],
        [Txy, Tyy, Tyz],
        [Txz, Tyz, Tzz],
    ], dtype=complex)

    if np.max(np.abs(T.imag)) < 1e-12:
        return T.real
    return T


# ---------------------------------------------------------------------------
# rotateframe — rotate Euler frame around an axis by angles rho
# ---------------------------------------------------------------------------

def rotateframe(ang0, nRot, rho) -> np.ndarray:
    """Rotate an Euler frame around a rotation axis.

    Port of EasySpin's ``rotateframe.m``.

    Parameters
    ----------
    ang0 : array_like, shape (3,)
        Initial Euler angles ``[alpha, beta, gamma]`` in radians defining the
        starting orientation.
    nRot : array_like, shape (3,)
        Unit vector (or non-unit — normalised internally) defining the
        rotation axis in the molecular frame.
    rho : float or array_like, shape (M,)
        Rotation angle(s) in radians around *nRot*.

    Returns
    -------
    ang_new : ndarray, shape (3,) or (M, 3)
        New Euler angles after the rotation(s).  Shape ``(3,)`` for scalar
        *rho*, ``(M, 3)`` for array *rho*.

    Notes
    -----
    The rotation is applied as:
    ``R_new = R_nRot(rho) @ R_ang0``
    where all rotations are passive (alias) z-y'-z'' convention.

    The resulting matrix is decomposed back to Euler angles via ``eulang``.

    Examples
    --------
    >>> rotateframe([0, 0, 0], [0, 0, 1], np.pi/2)   # rotate by 90° around z
    array([1.5707..., 0., 0.])
    """
    ang0 = np.asarray(ang0, dtype=float)
    nRot = np.asarray(nRot, dtype=float)
    nRot = nRot / np.linalg.norm(nRot)

    rho = np.asarray(rho, dtype=float)
    scalar = rho.ndim == 0
    rho = np.atleast_1d(rho)

    # Build starting rotation matrix from Euler angles
    # Import here to avoid circular import
    from torchspin.rotations import erot as _erot
    R0 = np.array(_erot(ang0.tolist()))

    results = []
    for r in rho:
        # Rotation matrix for angle r around nRot
        R_rot = rotaxi2mat(nRot, r)
        # Combined rotation: new = R_rot @ R0
        R_new = R_rot @ R0
        ang_new = eulang(R_new)
        results.append(ang_new)

    out = np.array(results)
    if scalar:
        return out[0]
    return out
