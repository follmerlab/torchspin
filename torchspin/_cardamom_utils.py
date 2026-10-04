"""Shared helpers for cardamom trajectory-based EPR simulation.

Ports of MATLAB EasySpin private helpers:
- ``cardamom_tensortraj.m`` → ``tensor_traj()``
- spiral grid generation → ``spiral_grid()``
- Gelman-Rubin convergence diagnostic → ``gelman_rubin()``
"""
from __future__ import annotations

import numpy as np
import torch


def tensor_traj(
    T_diag: torch.Tensor,
    R: torch.Tensor,
) -> torch.Tensor:
    """Rotate an interaction tensor along a trajectory of rotation matrices.

    Computes ``R @ diag(T) @ R^T`` for each time step and trajectory.

    This is a direct port of ``cardamom_tensortraj.m``.

    Parameters
    ----------
    T_diag:
        Principal values of the interaction tensor, shape ``(3,)``.
    R:
        Rotation matrix trajectory, shape ``(3, 3, nSteps, nTraj)``.

    Returns
    -------
    T_traj:
        Rotated tensor trajectory, shape ``(3, 3, nSteps, nTraj)``.
    """
    # Build diagonal tensor
    if T_diag.ndim == 1 and T_diag.shape[0] == 3:
        T = torch.diag(T_diag)
    elif T_diag.ndim == 2 and T_diag.shape == (3, 3):
        T = T_diag
    else:
        raise ValueError("T must be a 3-vector or 3x3 matrix.")

    nSteps = R.shape[2]
    nTraj = R.shape[3]

    # R @ T @ R^T  — batched over (nSteps, nTraj)
    # Reshape R to (nSteps*nTraj, 3, 3) for batch matmul
    R_flat = R.permute(2, 3, 0, 1).reshape(-1, 3, 3)  # (N, 3, 3)
    R_inv = R_flat.transpose(-2, -1)  # R^T
    T_exp = T.unsqueeze(0).expand(R_flat.shape[0], -1, -1)

    T_rot = torch.bmm(R_flat, torch.bmm(T_exp, R_inv))
    return T_rot.reshape(nSteps, nTraj, 3, 3).permute(2, 3, 0, 1)


def spiral_grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a spherical spiral grid of *n* points.

    Returns (phi, theta) arrays suitable for powder averaging in cardamom.
    This is the same grid used in MATLAB's ``cardamom.m`` (lines 592-595).

    Parameters
    ----------
    n:
        Number of grid points.

    Returns
    -------
    phi:
        Azimuthal angles in radians, shape ``(n,)``.
    theta:
        Polar angles in radians, shape ``(n,)``.
    """
    pts = np.linspace(-1, 1, n)
    theta = np.arccos(pts)
    phi = np.sqrt(np.pi * n) * np.arcsin(pts)
    return phi, theta


def gelman_rubin(chains: np.ndarray) -> float:
    """Compute the Gelman-Rubin R-hat convergence diagnostic.

    Parameters
    ----------
    chains:
        Array of shape ``(n_chains, n_samples)`` containing scalar
        summary statistics (e.g. autocorrelation values) from
        independent trajectories.

    Returns
    -------
    R_hat:
        Gelman-Rubin R statistic. Values close to 1.0 indicate
        convergence; R < 1.1 is the typical threshold.
    """
    n_chains, n_samples = chains.shape
    if n_chains < 2:
        return 1.0

    chain_means = chains.mean(axis=1)
    chain_vars = chains.var(axis=1, ddof=1)

    grand_mean = chain_means.mean()
    B = n_samples * np.var(chain_means, ddof=1)  # between-chain variance
    W = np.mean(chain_vars)  # within-chain variance

    if W < 1e-30:
        return 1.0

    var_hat = ((n_samples - 1) / n_samples) * W + (1.0 / n_samples) * B
    R_hat = np.sqrt(var_hat / W)
    return float(R_hat)
