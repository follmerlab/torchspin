"""Performance train P1: each faster path must give exactly what the code it
replaced gave (bit-identical), and the native-thread controls must leave the
process as they found it."""
import sys

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from torchspin import _linalg


# ----------------------------------------------------------------------------- BLAS cap
def test_serial_blas_restores_and_nests():
    ctl = _linalg._blas_controller()
    if ctl is None:
        pytest.skip('threadpoolctl cannot inspect this process')
    before = [p['num_threads'] for p in ctl.select(user_api='blas').info()]
    with _linalg.serial_blas():
        with _linalg.serial_blas():
            assert all(p['num_threads'] == 1 for p in ctl.select(user_api='blas').info())
        assert all(p['num_threads'] == 1 for p in ctl.select(user_api='blas').info())
    assert [p['num_threads'] for p in ctl.select(user_api='blas').info()] == before
    assert _linalg._blas_state['depth'] == 0


@pytest.mark.parametrize('n,nb', [(6, 64), (12, 64), (36, 40)])
def test_eigh_paths_unchanged(n, nb):
    """Pooled, unpooled and BLAS-capped eigh give the plain batched LAPACK result."""
    g = torch.Generator().manual_seed(n)
    A = torch.randn(nb, n, n, dtype=torch.complex128, generator=g)
    H = A + A.mH
    E0, V0 = torch.linalg.eigh(H)
    E1, V1 = _linalg.eigh(H)
    assert torch.equal(E0, E1) and torch.equal(V0, V1)


# ----------------------------------------------------------------------------- chili mat-vec
def test_row_block_matvec_identical(monkeypatch):
    cs = sys.modules.get('torchspin.chili_sle') or __import__('torchspin.chili_sle', fromlist=['x'])
    rng = np.random.default_rng(3)
    A = sp.random(3000, 3000, density=0.01, format='csr', random_state=4, dtype=complex)
    A.data = A.data + 1j * rng.standard_normal(A.nnz)
    q = rng.standard_normal(3000) + 1j * rng.standard_normal(3000)
    monkeypatch.setattr(cs, '_MATVEC_POOL_NNZ', 1)
    prev = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        mv = cs._row_block_matvec(A)
        assert np.array_equal(mv(q), A @ q)
    finally:
        torch.set_num_threads(prev)


# ----------------------------------------------------------------------------- cardamom free diffusion
def _ref_propagate_free(q, rand_steps, nSteps):
    """The 0.9.38 step loop of stochtraj_diffusion._propagate without a potential."""
    for iStep in range(1, nSteps):
        qLast = q[:, :, iStep - 1]
        ang_step = rand_steps[:, :, iStep]
        theta = np.sqrt(np.sum(ang_step ** 2, axis=0, keepdims=True))
        theta_safe = np.where(theta < 1e-30, 1e-30, theta)
        ux = ang_step[0:1] / theta_safe
        uy = ang_step[1:2] / theta_safe
        uz = ang_step[2:3] / theta_safe
        st = np.sin(theta / 2.0)
        ct = np.cos(theta / 2.0)
        q1, q2, q3, q4 = qLast[0:1], qLast[1:2], qLast[2:3], qLast[3:4]
        q[0, :, iStep] = (q1 * ct - q2 * ux * st - q3 * uy * st - q4 * uz * st).ravel()
        q[1, :, iStep] = (q2 * ct + q1 * ux * st - q4 * uy * st + q3 * uz * st).ravel()
        q[2, :, iStep] = (q3 * ct + q4 * ux * st + q1 * uy * st - q2 * uz * st).ravel()
        q[3, :, iStep] = (q4 * ct - q3 * ux * st + q2 * uy * st + q1 * uz * st).ravel()
    return q


@pytest.mark.parametrize('ntraj,nsteps', [(1, 50), (17, 200), (64, 31)])
def test_cardamom_free_propagation_identical(ntraj, nsteps):
    from torchspin.stochtraj_diffusion import _propagate_free
    rng = np.random.default_rng(ntraj)
    rand = 0.05 * rng.standard_normal((3, ntraj, nsteps))
    rand[:, 0, 3] = 0.0                                   # a zero step (theta below the floor)
    q0 = rng.standard_normal((4, ntraj))
    q0 /= np.linalg.norm(q0, axis=0)
    qa = np.zeros((4, ntraj, nsteps)); qa[:, :, 0] = q0
    qb = qa.copy()
    assert np.array_equal(_ref_propagate_free(qa, rand, nsteps), _propagate_free(qb, rand, nsteps))


# ----------------------------------------------------------------------------- spidyan propagator table
def test_propagator_table_levels_subset():
    from torchspin.spidyan import _build_propagator_table
    rng = np.random.default_rng(0)
    n = 4
    H0 = rng.standard_normal((n, n)); H0 = H0 + H0.T
    X = rng.standard_normal((n, n)); X = X + X.T
    full = _build_propagator_table(H0, X, 1e-3, 64, 0.5)
    part = _build_propagator_table(H0, X, 1e-3, 64, 0.5, levels=[0, 7, 32, 64])
    for i in range(65):
        if i in (0, 7, 32, 64):
            assert np.array_equal(full[i], part[i])
        else:
            assert part[i] is None


# ----------------------------------------------------------------------------- triangulation cache
def test_grid_triangulation_cache_returns_copies():
    from torchspin.sphgrid import grid_triangulation, _grid_triangulation
    tri, areas = grid_triangulation('C2h', 9)
    tri[0, 0] = -1; areas[0] = -1.0
    tri2, areas2 = grid_triangulation('C2h', 9)
    assert tri2[0, 0] != -1 and areas2[0] != -1.0
    _grid_triangulation.cache_clear()
    tri3, areas3 = grid_triangulation('C2h', 9)
    assert np.array_equal(tri2, tri3) and np.array_equal(areas2, areas3)


# ----------------------------------------------------------------------------- esfit memo
def test_esfit_memo_after_optimizer_only(monkeypatch):
    """The final point is evaluated by the optimizer, again for the residuals and as
    the center of the uncertainty stencil: with the memo the last two are served
    (two fewer model calls), with identical results and evaluation counts."""
    from torchspin.esfit import esfit, FitOptions, ModelWrapper

    def run():
        calls = []

        def model(p):
            calls.append(np.asarray(p, float).tobytes())
            x = np.linspace(-1, 1, 64)
            return np.exp(-(x - p[0]) ** 2 / p[1])       # center and width (amplitude is autoscaled)

        data = model(np.array([0.1, 0.1])); calls.clear()
        r = esfit(data, model, p0=np.array([0.0, 0.08]), vary=np.array([0.3, 0.05]),
                  options=FitOptions(method='levmar', max_iter=6, verbosity=0, compute_uncertainties=True))
        return r, len(calls)

    r1, n1 = run()
    with monkeypatch.context() as m:
        m.setattr(ModelWrapper, 'serve_memo', property(lambda self: False, lambda self, v: None))
        r0, n0 = run()
    assert n0 - n1 == 2
    assert np.array_equal(r1.pfit, r0.pfit) and np.array_equal(r1.pstd, r0.pstd)
    assert r1.n_evaluations == r0.n_evaluations
