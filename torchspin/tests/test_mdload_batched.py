"""B18: mdload builds the spin-label frames, the Kabsch rotations that remove the global
protein motion and the frames relative to the protein for all MD frames at once.  Must equal
the former per-frame loops (copied here) to roundoff, including the reflection branch."""
import numpy as np
import pytest

from torchspin.mdload import _calc_label_frame, _remove_global_rotation


def _loop_frames(xyz, topo, label):
    n = xyz.shape[0]
    nrm = lambda v: v / max(np.linalg.norm(v), 1e-30)
    F = np.zeros((3, 3, n))
    for t in range(n):
        NNNO = nrm(xyz[t, :, topo['idx_ON']] - xyz[t, :, topo['idx_NN']])
        if label == 'R1':
            a = nrm(xyz[t, :, topo['idx_C1']] - xyz[t, :, topo['idx_NN']])
            b = nrm(xyz[t, :, topo['idx_C2']] - xyz[t, :, topo['idx_NN']])
        else:
            a = nrm(xyz[t, :, topo['idx_CGR']] - xyz[t, :, topo['idx_NN']])
            b = nrm(xyz[t, :, topo['idx_CGS']] - xyz[t, :, topo['idx_NN']])
        z = nrm(np.cross(a, NNNO) + np.cross(NNNO, b))
        F[:, 0, t], F[:, 1, t], F[:, 2, t] = NNNO, np.cross(z, NNNO), z
    return F


def _loop_rotation(xyz, ca, n):
    R = np.zeros((3, 3, n)); R[:, :, 0] = np.eye(3)
    ref = xyz[0][:, ca]
    for t in range(1, n):
        ref_c = ref - ref.mean(axis=1, keepdims=True)
        cur = xyz[t][:, ca]; cur_c = cur - cur.mean(axis=1, keepdims=True)
        U, _, Vt = np.linalg.svd(cur_c @ ref_c.T)
        S = np.eye(3)
        if np.linalg.det(Vt.T @ U.T) < 0:
            S[2, 2] = -1
        R[:, :, t] = Vt.T @ S @ U.T
    return R


TOPO = {'idx_ON': 3, 'idx_NN': 4, 'idx_C1': 5, 'idx_C2': 6, 'idx_CGR': 7, 'idx_CGS': 8}


@pytest.mark.parametrize('label', ['R1', 'TOAC'])
@pytest.mark.parametrize('seed', [0, 1])
def test_label_frames_equal_loop(label, seed):
    xyz = np.random.default_rng(seed).standard_normal((300, 3, 12)) * 2
    np.testing.assert_allclose(_calc_label_frame(xyz, TOPO, label), _loop_frames(xyz, TOPO, label),
                               rtol=0, atol=1e-13)


@pytest.mark.parametrize('seed', [0, 1])
def test_global_rotation_equals_loop_incl_reflection(seed):
    rng = np.random.default_rng(seed)
    n, ca = 200, np.arange(10, 30)
    xyz = rng.standard_normal((n, 3, 30))
    xyz[::7, 0, :] *= -1                      # mirrored frames: Kabsch takes the reflection branch
    R = _remove_global_rotation(xyz, ca, n)
    np.testing.assert_allclose(R, _loop_rotation(xyz, ca, n), rtol=0, atol=1e-13)
    assert np.allclose(np.linalg.det(np.moveaxis(R, 2, 0)), 1.0)
