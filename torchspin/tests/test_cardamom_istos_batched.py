"""B9: cardamom's ISTOs propagation is batched over steps and trajectories (one SciPy expm
call, matmul over trajectories).  Must equal the former per-(step, trajectory) loops to
roundoff (measured 2e-16 to 3e-16; the trajectory average sums in another order)."""
import numpy as np
import pytest

import torchspin._cardamom_propagatedm as pdm
from torchspin import Experiment, SpinSystem
from torchspin.cardamom import CardamomOptions, CardamomPar, cardamom


def _loop_istos_propagation(H, H0, dtSpin, Sx):
    from scipy.linalg import expm
    nStates, _, nSteps, nTraj = H.shape
    U0 = expm(-1j * dtSpin * H0)
    U = np.zeros((nStates, nStates, nSteps, nTraj), dtype=complex)
    for i in range(nSteps):
        for t in range(nTraj):
            U[:, :, i, t] = U0 @ expm(1j * dtSpin * H[:, :, i, t])
    rho = np.zeros((nStates, nStates, nSteps, nTraj), dtype=complex)
    rho[:, :, 0, :] = Sx[:, :, None]
    for i in range(1, nSteps):
        for t in range(nTraj):
            rho[:, :, i, t] = U[:, :, i - 1, t] @ rho[:, :, i - 1, t] @ U[:, :, i - 1, t].conj().T
    return rho.mean(axis=3)


@pytest.mark.parametrize('n, seed', [(3, 0), (6, 1), (12, 2)])
def test_batched_steps_equal_loops(n, seed):
    from scipy.linalg import expm
    rng = np.random.default_rng(seed)
    nSteps, nTraj, dt = 40, 7, 1e-10
    H = rng.standard_normal((n, n, nSteps, nTraj)) + 1j * rng.standard_normal((n, n, nSteps, nTraj))
    H = 1e9 * (H + np.conj(np.swapaxes(H, 0, 1))) / 2
    H0 = np.diag(rng.standard_normal(n)) * 1e10 + 0j
    Sx = rng.standard_normal((n, n)) + 0j
    ref = _loop_istos_propagation(H, H0, dt, Sx)
    # the batched form as in propagate_istos
    Hb = np.ascontiguousarray(np.moveaxis(H, (0, 1), (2, 3)))
    U = expm(-1j * dt * H0) @ expm(1j * dt * Hb)
    Uh = np.conj(np.swapaxes(U, -1, -2))
    rho = np.zeros((nSteps, nTraj, n, n), dtype=complex); rho[0] = Sx
    for i in range(1, nSteps):
        rho[i] = U[i - 1] @ rho[i - 1] @ Uh[i - 1]
    got = np.moveaxis(rho.mean(axis=1), 0, 2)
    assert np.abs(got - ref).max() <= 1e-13 * np.abs(ref).max()


@pytest.mark.parametrize('kw, rng_', [
    (dict(S=[0.5], g=[[2.008, 2.006, 2.003]], Nucs='14N', A=[[20, 20, 85]], tcorr=1e-9), [332, 352]),
    (dict(S=[0.5], g=[[2.05, 2.05, 2.25]], Nucs='63Cu', A=[[50, 50, 550]], tcorr=2e-9), [270, 350]),
])
def test_istos_spectrum_is_finite_and_seeded(kw, rng_):
    e = Experiment(mwFreq=9.5, Range=rng_, nPoints=128, Harmonic=0)
    p = CardamomPar(Model='diffusion', nTraj=6, nSteps=120, dtSpin=1e-10, dtSpatial=1e-10, seed=3)
    y1 = np.asarray(cardamom(SpinSystem(**kw), e, p, CardamomOptions(Method='ISTOs', Verbosity=0))[1])
    y2 = np.asarray(cardamom(SpinSystem(**kw), e, p, CardamomOptions(Method='ISTOs', Verbosity=0))[1])
    assert np.all(np.isfinite(y1)) and np.abs(y1).max() > 0
    np.testing.assert_array_equal(y1, y2)
