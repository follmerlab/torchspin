"""HYSCORE powder spectrum outline.

Port of EasySpin's ``nucfrq2d.m``.

Computes powder HYSCORE peak positions (without amplitudes) by
diagonalizing the spin Hamiltonian at many orientations and extracting
nuclear transition frequencies in the alpha and beta electron-spin
manifolds.

Only S=1/2 systems are supported.

Example
-------
>>> from torchspin import SpinSystem
>>> from torchspin.nucfrq2d import nucfrq2d
>>> sys = SpinSystem(S=[0.5], g=[[2, 2, 2]], Nucs=['1H'], A=[[3-5, 3-5, 3+10]])
>>> data = nucfrq2d(sys, 350.0)
>>> data['FreqsAlpha'].shape  # (nOri, nfreq)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from torchspin.ham import ham
from torchspin.hamsymm import hamsymm
from torchspin.sphgrid import sphgrid
from torchspin.nucdata import nucgval
from torchspin.constants import NMAGN, PLANCK


def nucfrq2d(
    sys,
    B0: float,
    tau: Optional[np.ndarray | list | float] = None,
    *,
    GridSize: int = 31,
    plot: bool = False,
    ax=None,
) -> dict:
    """Compute HYSCORE powder spectrum peak positions.

    Parameters
    ----------
    sys:
        Spin system (:class:`~torchspin.SpinSystem`).  Must have S=1/2.
    B0:
        External magnetic field magnitude in mT.
    tau:
        Tau delay(s) in microseconds for blind-spot visualization.
        If ``None`` or 0, no blind spots are computed.
    GridSize:
        Number of orientational grid knots (default 31).
    plot:
        If ``True``, create a matplotlib plot.
    ax:
        Matplotlib axes for plotting.

    Returns
    -------
    data:
        Dictionary with keys:

        * ``'FreqsAlpha'``: shape ``(nOri, nfreq)`` — nuclear frequencies
          in the alpha (electron spin-up) manifold, in MHz.
        * ``'FreqsBeta'``: shape ``(nOri, nfreq)`` — nuclear frequencies
          in the beta (electron spin-down) manifold, in MHz.
        * ``'LarmorFreqs'``: Larmor frequencies of the nuclei at B0, in MHz.
        * ``'Modulation'``: blind-spot pattern (if tau given), or ``None``.
        * ``'maxFreq'``: maximum transition frequency in MHz.
    """
    # Validate S=1/2
    S_vals = sys.S if isinstance(sys.S, list) else [sys.S]
    if len(S_vals) != 1 or S_vals[0] != 0.5:
        raise ValueError("nucfrq2d only supports S=1/2 systems")

    # Build Hamiltonian
    H0, mux, muy, muz = ham(sys, B0=None)
    N = H0.shape[0]
    nn = N // 2  # states per manifold

    # Masks for alpha/beta manifold transitions
    # Alpha: upper-left block upper triangle; Beta: lower-right block upper triangle
    alpha_mask = np.zeros((N, N), dtype=bool)
    beta_mask = np.zeros((N, N), dtype=bool)
    for i in range(nn):
        for j in range(i + 1, nn):
            beta_mask[i, j] = True
            alpha_mask[nn + i, nn + j] = True
    nfreq = np.sum(beta_mask)

    # Orientational grid
    symm_result = hamsymm(sys)
    symm_str = symm_result[0] if isinstance(symm_result, tuple) else symm_result
    phi_grid, theta_grid, weights, vecs = sphgrid(symm_str, GridSize)

    # vecs is (3, nOri) — unit vectors on the sphere
    if hasattr(vecs, 'numpy'):
        vecs = vecs.detach().cpu().numpy()
    else:
        vecs = np.asarray(vecs)
    nOri = vecs.shape[1]

    FreqsAlpha = np.zeros((nOri, nfreq))
    FreqsBeta = np.zeros((nOri, nfreq))

    H0_np = H0.detach().cpu().numpy()
    mux_np = mux.detach().cpu().numpy()
    muy_np = muy.detach().cpu().numpy()
    muz_np = muz.detach().cpu().numpy()

    for k in range(nOri):
        Bx = B0 * vecs[0, k]
        By = B0 * vecs[1, k]
        Bz = B0 * vecs[2, k]
        H = H0_np - (Bx * mux_np + By * muy_np + Bz * muz_np)
        E = np.sort(np.linalg.eigvalsh(H).real)
        # Transition energy matrix
        EE = E[:, None] - E[None, :]
        FreqsAlpha[k, :] = EE[alpha_mask]
        FreqsBeta[k, :] = EE[beta_mask]

    maxFreq = max(np.max(np.abs(FreqsAlpha)), np.max(np.abs(FreqsBeta)))

    # Larmor frequencies
    nucs = sys.Nucs if isinstance(sys.Nucs, list) else [sys.Nucs]
    larmor_freqs = []
    for nuc in nucs:
        gn = nucgval(nuc)
        # Larmor freq in MHz: nu_L = gn * nmagn * B0 / (planck * 1e9)
        # B0 in mT = 1e-3 T
        nu_L = abs(gn) * NMAGN * (B0 * 1e-3) / (PLANCK * 1e6)
        larmor_freqs.append(nu_L)

    # Blind spot modulation
    modulation = None
    if tau is not None:
        tau_arr = np.atleast_1d(np.asarray(tau, dtype=np.float64))
        if np.all(tau_arr > 0):
            n_pts = 201
            expand = 1.1
            freq_axis = expand * maxFreq * np.linspace(-1, 1, n_pts)
            modulation = np.zeros((n_pts, n_pts))
            for t_val in tau_arr:
                mod_1d = np.abs(np.sin(np.pi * t_val * freq_axis))
                modulation += np.outer(mod_1d, mod_1d)
            mod_max = np.max(modulation)
            if mod_max > 0:
                modulation /= mod_max

    result = {
        'FreqsAlpha': FreqsAlpha,
        'FreqsBeta': FreqsBeta,
        'LarmorFreqs': np.array(larmor_freqs),
        'Modulation': modulation,
        'maxFreq': maxFreq,
    }

    # Plotting
    if plot:
        _plot_nucfrq2d(result, B0, tau, ax=ax)

    return result


def _plot_nucfrq2d(data: dict, B0: float, tau, *, ax=None):
    """Plot HYSCORE powder spectrum outline."""
    import matplotlib.pyplot as plt

    expand = 1.1
    maxFreq = data['maxFreq']
    FreqsAlpha = data['FreqsAlpha']
    FreqsBeta = data['FreqsBeta']
    nfreq = FreqsAlpha.shape[1]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    # Blind spot background
    if data['Modulation'] is not None:
        n_pts = data['Modulation'].shape[0]
        freq_axis = expand * maxFreq * np.linspace(-1, 1, n_pts)
        ax.pcolormesh(freq_axis, freq_axis, data['Modulation'],
                      cmap='gray_r', shading='auto', vmin=0, vmax=1)

    # Diagonals
    lim = expand * maxFreq
    ax.plot([-lim, lim], [-lim, lim], 'k-', linewidth=0.5)
    ax.plot([-lim, lim], [lim, -lim], 'k-', linewidth=0.5)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)

    # Larmor frequency antidiagonals
    for nu_L in data['LarmorFreqs']:
        ax.plot([-2 * nu_L, 0, 2 * nu_L], [0, 2 * nu_L, 0],
                'k--', linewidth=0.5)

    ba_color = [0.6, 0, 0]  # red
    ab_color = [0, 0, 0.6]  # blue

    # Plot beta-alpha correlations
    for i1 in range(nfreq):
        for i2 in range(nfreq):
            ax.plot(FreqsBeta[:, i1], FreqsAlpha[:, i2],
                    '.', color=ba_color, markersize=1)
            ax.plot(-FreqsBeta[:, i1], FreqsAlpha[:, i2],
                    '.', color=ba_color, markersize=1)

    # Plot alpha-beta correlations
    for i1 in range(nfreq):
        for i2 in range(nfreq):
            ax.plot(FreqsAlpha[:, i1], FreqsBeta[:, i2],
                    '.', color=ab_color, markersize=1)
            ax.plot(-FreqsAlpha[:, i1], FreqsBeta[:, i2],
                    '.', color=ab_color, markersize=1)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$\nu_1$ (MHz)')
    ax.set_ylabel(r'$\nu_2$ (MHz)')

    return fig, ax
