"""
mdtraj2oripot — Convert MD trajectory to orientation potential.

Bins rotation matrices into Euler angle histograms and computes
an orientation potential as -log(PDF).

Based on EasySpin's mdtraj2oripot.m.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Tuple, Optional


def mdtraj2oripot(
    frame_traj: np.ndarray,
    n_bins: int = 90,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert frame trajectory to orientation potential.

    Parameters
    ----------
    frame_traj : ndarray, shape (3, 3, nSteps)
        Label frame rotation matrices at each time step.
    n_bins : int or tuple of 3 ints
        Number of histogram bins per Euler angle.

    Returns
    -------
    potential : ndarray, shape (nBinsPhi+1, nBinsTheta+1, nBinsPsi+1)
        Orientation potential: -log(pdf).
    pdf : ndarray
        Normalized 3D histogram of Euler angles.
    """
    nSteps = frame_traj.shape[2]

    if isinstance(n_bins, (int, float)):
        n_bins_phi = n_bins_theta = n_bins_psi = int(n_bins)
    else:
        n_bins_phi, n_bins_theta, n_bins_psi = n_bins

    # Convert rotation matrices to Euler angles (ZYZ convention)
    phi = np.zeros(nSteps)
    theta = np.zeros(nSteps)
    psi = np.zeros(nSteps)

    for t in range(nSteps):
        R = frame_traj[:, :, t]
        # ZYZ Euler angles from rotation matrix
        theta[t] = np.arccos(np.clip(R[2, 2], -1, 1))

        if abs(np.sin(theta[t])) > 1e-10:
            phi[t] = np.arctan2(R[1, 2], R[0, 2])
            psi[t] = np.arctan2(R[2, 1], -R[2, 0])
        else:
            phi[t] = np.arctan2(-R[0, 1], R[0, 0])
            psi[t] = 0.0

    # Wrap to [0, 2π) for phi and psi, [0, π] for theta
    phi = phi % (2 * np.pi)
    psi = psi % (2 * np.pi)

    # 3D histogram
    bins_phi = np.linspace(0, 2 * np.pi, n_bins_phi + 1)
    bins_theta = np.linspace(0, np.pi, n_bins_theta + 1)
    bins_psi = np.linspace(0, 2 * np.pi, n_bins_psi + 1)

    pdf, _ = np.histogramdd(
        np.column_stack([phi, theta, psi]),
        bins=[bins_phi, bins_theta, bins_psi],
    )

    # Gaussian smoothing
    sigma_smooth = max(1, n_bins_phi // 30)
    pdf = gaussian_filter(pdf.astype(float), sigma=sigma_smooth)

    # Normalize
    pdf_sum = pdf.sum()
    if pdf_sum > 0:
        pdf = pdf / pdf_sum

    # Compute potential
    floor = 1e-10
    pdf_floored = np.maximum(pdf, floor)
    potential = -np.log(pdf_floored)

    # Shift so minimum is zero
    potential -= potential.min()

    return potential, pdf
