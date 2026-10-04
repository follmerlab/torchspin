"""
mdhmm — Hidden Markov Model for MD dihedral trajectories.

Builds an HMM from dihedral angle trajectories using:
1. Circular K-means clustering for initialization
2. Baum-Welch (EM) algorithm for parameter estimation
3. Viterbi decoding for state assignment

Based on EasySpin's mdhmm.m, mdhmm_em.m, mdhmm_kmeans.m.

All angles are in **radians**.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple
from scipy.optimize import minimize


@dataclass
class HMMResult:
    """Hidden Markov Model result.

    Attributes
    ----------
    TransProb : ndarray, shape (nStates, nStates)
        Transition probability matrix.
    eqDistr : ndarray, shape (nStates,)
        Equilibrium distribution.
    mu : ndarray, shape (nDihedrals, nStates)
        State center vectors (dihedral angles).
    Sigma : ndarray, shape (nDihedrals, nDihedrals, nStates)
        Covariance matrices per state.
    viterbiTraj : ndarray, shape (nSteps,) or (nTraj, nSteps)
        Most-likely state sequence.
    tauRelax : ndarray
        Relaxation times from eigenvalues of TransProb.
    tLag : float
        Lag time (dt * nLag).
    nStates : int
    dt : float
    nLag : int
    """
    TransProb: np.ndarray = None
    eqDistr: np.ndarray = None
    mu: np.ndarray = None
    Sigma: np.ndarray = None
    viterbiTraj: np.ndarray = None
    tauRelax: np.ndarray = None
    tLag: float = 0.0
    nStates: int = 0
    dt: float = 0.0
    nLag: int = 0


@dataclass
class HMMOptions:
    """Options for mdhmm."""
    Verbosity: int = 0
    isSeeded: bool = False
    nTrials: int = 5


# ---------------------------------------------------------------------------
# Circular distance helpers
# ---------------------------------------------------------------------------

def _dist_pbc(x1, x2, W=2 * np.pi):
    """Circular distance between angles."""
    d = np.abs(x1 - x2)
    return np.minimum(d, W - d)


def _mean_pbc(angles, axis=None, W=2 * np.pi):
    """Circular mean of angles using unwrapping."""
    if len(angles) == 0:
        return 0.0
    ref = angles.flat[0] if axis is None else np.take(angles, 0, axis=axis)
    # Unwrap relative to reference
    diff = angles - ref
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    mean_diff = np.mean(diff, axis=axis)
    return (ref + mean_diff + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Circular K-means (mdhmm_kmeans.m)
# ---------------------------------------------------------------------------

def _circular_kmeans(data, n_clusters, n_repeats=5, centroids0=None, seed=None):
    """K-means clustering with circular distance metric.

    Parameters
    ----------
    data : ndarray, shape (nDihedrals, nPoints)
        Dihedral angle data.
    n_clusters : int
    n_repeats : int
    centroids0 : ndarray, shape (nDihedrals, n_clusters), optional
    seed : int, optional

    Returns
    -------
    idx : ndarray, shape (nPoints,)
        Cluster assignments.
    centroids : ndarray, shape (nDihedrals, n_clusters)
    """
    n_dim, n_points = data.shape
    rng = np.random.default_rng(seed)

    best_cost = np.inf
    best_idx = None
    best_centroids = None

    for trial in range(n_repeats):
        # Initialize centroids
        if centroids0 is not None and trial == 0:
            centroids = centroids0.copy()
        else:
            # K-means++ initialization with circular distance
            centroids = np.zeros((n_dim, n_clusters))
            # First centroid: random
            idx0 = rng.integers(0, n_points)
            centroids[:, 0] = data[:, idx0]

            for k in range(1, n_clusters):
                # Distance to nearest centroid
                dists = np.full(n_points, np.inf)
                for j in range(k):
                    d = np.sum(_dist_pbc(data, centroids[:, j:j + 1]) ** 2, axis=0)
                    dists = np.minimum(dists, d)
                # Probability proportional to distance²
                probs = dists / dists.sum()
                idx_k = rng.choice(n_points, p=probs)
                centroids[:, k] = data[:, idx_k]

        # Iterate
        max_iter = 500
        for iteration in range(max_iter):
            # Assignment step
            dists = np.zeros((n_clusters, n_points))
            for k in range(n_clusters):
                dists[k] = np.sum(_dist_pbc(data, centroids[:, k:k + 1]) ** 2, axis=0)
            idx = np.argmin(dists, axis=0)

            # Update step
            new_centroids = np.zeros_like(centroids)
            for k in range(n_clusters):
                members = data[:, idx == k]
                if members.shape[1] == 0:
                    new_centroids[:, k] = data[:, rng.integers(0, n_points)]
                else:
                    for d in range(n_dim):
                        new_centroids[d, k] = _mean_pbc(members[d])

            # Check convergence
            centroid_change = np.max(_dist_pbc(new_centroids, centroids))
            centroids = new_centroids
            if centroid_change < 1e-4:
                break

        # Cost
        cost = 0
        for k in range(n_clusters):
            members = data[:, idx == k]
            cost += np.sum(_dist_pbc(members, centroids[:, k:k + 1]) ** 2)

        if cost < best_cost:
            best_cost = cost
            best_idx = idx
            best_centroids = centroids

    return best_idx, best_centroids


# ---------------------------------------------------------------------------
# Gaussian probability (circular)
# ---------------------------------------------------------------------------

def _gaussian_prob_circular(data, mu, sigma):
    """Multivariate Gaussian probability for circular data.

    Parameters
    ----------
    data : ndarray, shape (nDim, nPoints)
    mu : ndarray, shape (nDim,)
    sigma : ndarray, shape (nDim, nDim)

    Returns
    -------
    prob : ndarray, shape (nPoints,)
    """
    n_dim = data.shape[0]
    diff = _dist_pbc(data, mu[:, np.newaxis])
    # Use signed difference for proper Gaussian
    diff_signed = data - mu[:, np.newaxis]
    diff_signed = (diff_signed + np.pi) % (2 * np.pi) - np.pi

    try:
        sigma_inv = np.linalg.inv(sigma)
        det_sigma = np.linalg.det(sigma)
    except np.linalg.LinAlgError:
        sigma_reg = sigma + 1e-6 * np.eye(n_dim)
        sigma_inv = np.linalg.inv(sigma_reg)
        det_sigma = np.linalg.det(sigma_reg)

    det_sigma = max(det_sigma, 1e-300)

    exponent = -0.5 * np.sum(diff_signed * (sigma_inv @ diff_signed), axis=0)
    norm = 1.0 / ((2 * np.pi) ** (n_dim / 2) * np.sqrt(det_sigma))
    prob = norm * np.exp(exponent)

    return np.maximum(prob, 1e-300)


# ---------------------------------------------------------------------------
# Forward-backward algorithm
# ---------------------------------------------------------------------------

def _forward_backward(obs_prob, trans_prob, init_distr):
    """Forward-backward algorithm for HMM.

    Parameters
    ----------
    obs_prob : ndarray, shape (nStates, nSteps)
        Observation probabilities B(state, time).
    trans_prob : ndarray, shape (nStates, nStates)
        Transition matrix A(i,j) = P(state_j at t+1 | state_i at t).
    init_distr : ndarray, shape (nStates,)

    Returns
    -------
    gamma : ndarray, shape (nStates, nSteps)
        State occupancy probabilities.
    xi : ndarray, shape (nStates, nStates, nSteps-1)
        Transition counts.
    log_lik : float
        Log-likelihood.
    """
    n_states, n_steps = obs_prob.shape

    # Forward pass with scaling
    alpha = np.zeros((n_states, n_steps))
    scale = np.zeros(n_steps)

    alpha[:, 0] = init_distr * obs_prob[:, 0]
    scale[0] = np.sum(alpha[:, 0])
    if scale[0] > 0:
        alpha[:, 0] /= scale[0]

    for t in range(1, n_steps):
        alpha[:, t] = (trans_prob.T @ alpha[:, t - 1]) * obs_prob[:, t]
        scale[t] = np.sum(alpha[:, t])
        if scale[t] > 0:
            alpha[:, t] /= scale[t]

    # Log-likelihood
    log_lik = np.sum(np.log(np.maximum(scale, 1e-300)))

    # Backward pass
    beta = np.zeros((n_states, n_steps))
    beta[:, -1] = 1.0

    for t in range(n_steps - 2, -1, -1):
        beta[:, t] = trans_prob @ (obs_prob[:, t + 1] * beta[:, t + 1])
        if scale[t + 1] > 0:
            beta[:, t] /= scale[t + 1]

    # Gamma and Xi
    gamma = alpha * beta
    gamma_sum = np.sum(gamma, axis=0, keepdims=True)
    gamma = gamma / np.maximum(gamma_sum, 1e-300)

    xi = np.zeros((n_states, n_states, n_steps - 1))
    for t in range(n_steps - 1):
        xi_t = (alpha[:, t:t + 1] * trans_prob) * (obs_prob[:, t + 1] * beta[:, t + 1])
        xi_sum = np.sum(xi_t)
        if xi_sum > 0:
            xi[:, :, t] = xi_t / xi_sum

    return gamma, xi, log_lik


# ---------------------------------------------------------------------------
# Viterbi algorithm
# ---------------------------------------------------------------------------

def _viterbi(obs_prob, trans_prob, init_distr):
    """Viterbi algorithm for most-likely state sequence.

    Parameters
    ----------
    obs_prob : ndarray, shape (nStates, nSteps)
    trans_prob : ndarray, shape (nStates, nStates)
    init_distr : ndarray, shape (nStates,)

    Returns
    -------
    path : ndarray, shape (nSteps,)
        Most-likely state indices.
    """
    n_states, n_steps = obs_prob.shape

    log_trans = np.log(np.maximum(trans_prob, 1e-300))
    log_obs = np.log(np.maximum(obs_prob, 1e-300))
    log_init = np.log(np.maximum(init_distr, 1e-300))

    # Viterbi forward
    V = np.zeros((n_states, n_steps))
    backptr = np.zeros((n_states, n_steps), dtype=int)

    V[:, 0] = log_init + log_obs[:, 0]

    for t in range(1, n_steps):
        for j in range(n_states):
            scores = V[:, t - 1] + log_trans[:, j]
            backptr[j, t] = np.argmax(scores)
            V[j, t] = scores[backptr[j, t]] + log_obs[j, t]

    # Backtrack
    path = np.zeros(n_steps, dtype=int)
    path[-1] = np.argmax(V[:, -1])
    for t in range(n_steps - 2, -1, -1):
        path[t] = backptr[path[t + 1], t + 1]

    return path


# ---------------------------------------------------------------------------
# Transition matrix estimation via L-BFGS-B
# ---------------------------------------------------------------------------

def _estimate_transition_matrix(xi, gamma):
    """Estimate transition matrix from EM sufficient statistics.

    Parameters
    ----------
    xi : ndarray, shape (nStates, nStates)
        Sum of transition counts.
    gamma : ndarray, shape (nStates,)
        Sum of state occupancies.

    Returns
    -------
    T : ndarray, shape (nStates, nStates)
        Row-stochastic transition matrix.
    """
    n = xi.shape[0]
    T = xi.copy()

    # Normalize rows
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-300)
    T = T / row_sums

    return T


# ---------------------------------------------------------------------------
# Baum-Welch (EM) algorithm (mdhmm_em.m)
# ---------------------------------------------------------------------------

def _baum_welch(data, n_states, init_distr, trans_prob, mu, sigma,
                max_iter=200, tol=1e-4):
    """Baum-Welch algorithm for HMM parameter estimation.

    Parameters
    ----------
    data : ndarray, shape (nDim, nSteps)
    n_states : int
    init_distr : ndarray, shape (nStates,)
    trans_prob : ndarray, shape (nStates, nStates)
    mu : ndarray, shape (nDim, nStates)
    sigma : ndarray, shape (nDim, nDim, nStates)
    max_iter : int
    tol : float

    Returns
    -------
    log_lik_history : list
    eq_distr : ndarray
    trans_prob : ndarray
    mu : ndarray
    sigma : ndarray
    """
    n_dim, n_steps = data.shape
    log_lik_history = []
    prev_log_lik = -np.inf

    for iteration in range(max_iter):
        # E-step: compute observation probabilities
        obs_prob = np.zeros((n_states, n_steps))
        for k in range(n_states):
            obs_prob[k] = _gaussian_prob_circular(data, mu[:, k], sigma[:, :, k])

        # Forward-backward
        gamma, xi, log_lik = _forward_backward(obs_prob, trans_prob, init_distr)
        log_lik_history.append(log_lik)

        # Check convergence
        if iteration > 0:
            rel_change = abs(log_lik - prev_log_lik) / max(abs(prev_log_lik), 1)
            if rel_change < tol:
                break
        prev_log_lik = log_lik

        # M-step: update parameters
        # Transition matrix
        xi_sum = np.sum(xi, axis=2)
        trans_prob = _estimate_transition_matrix(xi_sum, np.sum(gamma[:, :-1], axis=1))

        # State means (circular)
        for k in range(n_states):
            weights = gamma[k]
            total_weight = np.sum(weights)
            if total_weight > 1e-10:
                for d in range(n_dim):
                    # Weighted circular mean
                    diff = data[d] - mu[d, k]
                    diff = (diff + np.pi) % (2 * np.pi) - np.pi
                    mu[d, k] = mu[d, k] + np.sum(weights * diff) / total_weight
                    mu[d, k] = (mu[d, k] + np.pi) % (2 * np.pi) - np.pi

        # Covariance matrices
        for k in range(n_states):
            weights = gamma[k]
            total_weight = np.sum(weights)
            if total_weight > 1e-10:
                diff = data - mu[:, k:k + 1]
                diff = (diff + np.pi) % (2 * np.pi) - np.pi
                sigma[:, :, k] = (diff * weights) @ diff.T / total_weight
                # Regularization
                sigma[:, :, k] += 1e-4 * np.eye(n_dim)

        # Update initial distribution
        init_distr = gamma[:, 0]
        init_distr = init_distr / np.sum(init_distr)

    # Equilibrium distribution from eigenvector of TransProb
    eigenvalues, eigenvectors = np.linalg.eig(trans_prob.T)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    eq_distr = np.real(eigenvectors[:, idx])
    eq_distr = np.abs(eq_distr)
    eq_distr = eq_distr / np.sum(eq_distr)

    return log_lik_history, eq_distr, trans_prob, mu, sigma


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def mdhmm(
    dihedrals: np.ndarray,
    dt: float,
    n_states: int,
    n_lag: int,
    opt: Optional[HMMOptions] = None,
) -> HMMResult:
    """Build Hidden Markov Model from dihedral angle trajectories.

    Parameters
    ----------
    dihedrals : ndarray, shape (nDihedrals, nSteps) or (nDihedrals, nTraj, nSteps)
        Spin-label dihedral angles in radians.
    dt : float
        Dihedral trajectory time step.
    n_states : int
        Number of HMM states.
    n_lag : int
        Lag step (number of time steps for downsampling).
    opt : HMMOptions, optional

    Returns
    -------
    hmm : HMMResult
    """
    if opt is None:
        opt = HMMOptions()

    # Handle multi-trajectory input
    if dihedrals.ndim == 3:
        n_dim, n_traj, n_steps_per_traj = dihedrals.shape
        # Concatenate trajectories for clustering and EM
        data = dihedrals.reshape(n_dim, -1)
    else:
        n_dim, total_steps = dihedrals.shape
        n_traj = 1
        data = dihedrals

    # Downsample by lag
    if n_lag > 1:
        data = data[:, ::n_lag]

    n_steps = data.shape[1]

    # Step 1: K-means clustering
    idx, centroids = _circular_kmeans(
        data, n_states, n_repeats=opt.nTrials, seed=42
    )

    # Initialize HMM parameters from k-means
    mu = centroids.copy()  # (nDim, nStates)
    sigma = np.zeros((n_dim, n_dim, n_states))
    init_distr = np.zeros(n_states)

    for k in range(n_states):
        members = data[:, idx == k]
        n_members = members.shape[1]
        init_distr[k] = n_members / n_steps

        if n_members > 1:
            diff = members - mu[:, k:k + 1]
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            sigma[:, :, k] = (diff @ diff.T) / n_members + 1e-4 * np.eye(n_dim)
        else:
            sigma[:, :, k] = 0.1 * np.eye(n_dim)

    # Initialize transition matrix
    trans_prob = np.zeros((n_states, n_states))
    for t in range(n_steps - 1):
        trans_prob[idx[t], idx[t + 1]] += 1
    row_sums = trans_prob.sum(axis=1, keepdims=True)
    trans_prob = trans_prob / np.maximum(row_sums, 1)

    # Step 2: Baum-Welch
    log_lik_history, eq_distr, trans_prob, mu, sigma = _baum_welch(
        data, n_states, init_distr, trans_prob, mu, sigma
    )

    # Step 3: Viterbi decoding
    obs_prob = np.zeros((n_states, n_steps))
    for k in range(n_states):
        obs_prob[k] = _gaussian_prob_circular(data, mu[:, k], sigma[:, :, k])

    viterbi_traj = _viterbi(obs_prob, trans_prob, eq_distr)

    # Eliminate unvisited states
    visited = np.unique(viterbi_traj)
    if len(visited) < n_states:
        # Remap
        state_map = {old: new for new, old in enumerate(visited)}
        viterbi_traj = np.array([state_map[s] for s in viterbi_traj])
        n_states_new = len(visited)
        mu = mu[:, visited]
        sigma = sigma[:, :, visited]
        trans_prob = trans_prob[np.ix_(visited, visited)]
        eq_distr = eq_distr[visited]
        eq_distr /= eq_distr.sum()
        n_states = n_states_new

    # Relaxation times
    eigenvalues = np.sort(np.real(np.linalg.eigvals(trans_prob)))[::-1]
    # Skip the eigenvalue = 1 (equilibrium)
    tau_relax = np.zeros(len(eigenvalues) - 1)
    for i in range(1, len(eigenvalues)):
        if eigenvalues[i] > 0 and eigenvalues[i] < 1:
            tau_relax[i - 1] = -n_lag * dt / np.log(eigenvalues[i])

    # Build result
    result = HMMResult()
    result.TransProb = trans_prob
    result.eqDistr = eq_distr
    result.mu = mu
    result.Sigma = sigma
    result.viterbiTraj = viterbi_traj
    result.tauRelax = tau_relax
    result.tLag = n_lag * dt
    result.nStates = n_states
    result.dt = dt
    result.nLag = n_lag

    return result
