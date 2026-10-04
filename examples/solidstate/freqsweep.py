"""Frequency-Swept EPR at Fixed Field
Comparison of field-swept and frequency-swept EPR for an S=1/2 axial spin system.

EasySpin equivalent: examples/solidstate/freqsweep.m
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torchspin import SpinSystem, Experiment, Options, pepper
from torchspin import resfreqs_matrix, makespec, convspec


def run():
    Sys = SpinSystem(
        S=[0.5],
        g=[[2.05, 2.05, 2.22]],
        lw=[0.5, 0.0],
    )

    # Field-swept spectrum (standard mode)
    exp = Experiment(mwFreq=9.5, Range=[300, 380], nPoints=1024, Harmonic=1)
    opt = Options(GridSize=40, GridSymmetry='Ci', Verbosity=0)
    B, spc_field = pepper(Sys, exp, opt)

    # Frequency-swept: fix field, accumulate resonance frequencies
    B_fixed_mT = 335.0
    n_phi, n_theta = 30, 30
    phi_arr = np.linspace(0, np.pi / 2, n_phi)
    theta_arr = np.linspace(0, np.pi / 2, n_theta)
    freq_min_GHz, freq_max_GHz = 8.0, 11.0
    freq_all, intens_all = [], []

    for phi in phi_arr:
        for theta in theta_arr:
            f, iv = resfreqs_matrix(
                Sys, phi, theta, B_fixed_mT,
                freq_range=(freq_min_GHz, freq_max_GHz),
            )
            freq_all.extend(f.tolist())
            intens_all.extend(iv.tolist())

    nPts = 512
    freq_min_MHz = freq_min_GHz * 1e3
    freq_max_MHz = freq_max_GHz * 1e3
    freq_axis = np.linspace(freq_min_MHz, freq_max_MHz, nPts)
    spc_freq = None

    if len(freq_all) > 0:
        freq_MHz_arr = np.array(freq_all)
        intens_arr = np.array(intens_all)
        spc_sticks = torch.zeros(nPts)
        for f_val, i_val in zip(freq_MHz_arr, intens_arr):
            idx = int((f_val - freq_min_MHz) / (freq_max_MHz - freq_min_MHz) * (nPts - 1))
            if 0 <= idx < nPts:
                spc_sticks[idx] += i_val
        lw_MHz = 50.0
        spc_freq = convspec(spc_sticks, freq_axis[1] - freq_axis[0], lw_MHz, 0.0, deriv=0)
        spc_freq = spc_freq.numpy()

    return B.numpy(), spc_field.numpy(), freq_axis / 1e3, spc_freq, B_fixed_mT


def _plot(B_np, spc_field, freq_GHz_axis, spc_freq, B_fixed_mT):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    y_field = spc_field / np.abs(spc_field).max() if np.abs(spc_field).max() > 0 else spc_field
    axes[0].plot(B_np, y_field, 'b', lw=1.5)
    axes[0].set_xlabel('Magnetic field (mT)')
    axes[0].set_ylabel('Intensity (norm.)')
    axes[0].set_title('Field-swept EPR (\u03bd = 9.5 GHz)')
    axes[0].grid(True, alpha=0.3)

    if spc_freq is not None and np.abs(spc_freq).max() > 0:
        y_freq = spc_freq / np.abs(spc_freq).max()
        axes[1].plot(freq_GHz_axis, y_freq, 'r', lw=1.5)
        axes[1].axvline(9.5, color='k', lw=0.5, ls='--', label='9.5 GHz')
        axes[1].legend(fontsize=8)
    axes[1].set_xlabel('Microwave frequency (GHz)')
    axes[1].set_ylabel('Intensity (norm.)')
    axes[1].set_title(f'Frequency-swept EPR (B = {B_fixed_mT} mT)')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Field-swept vs frequency-swept EPR: S=1/2, axial g', fontsize=12)
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
