"""¹H ENDOR — Summed and Separate Nucleus Contributions
Simulates a system with three ¹H nuclei (each with a different A tensor)
and shows both the total ENDOR spectrum and the contribution from each nucleus.
Equivalent to the MATLAB nucspinadd() pattern: specify all A tensors directly.

EasySpin equivalent: examples/endor/endorseparate.m
"""
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchspin import SpinSystem
from torchspin.salt import salt
from torchspin.experiment import Experiment, Options
from torchspin.constants import NMAGN, PLANCK


def run():
    # MATLAB: three 1H nuclei added via nucspinadd
    #   A1 = [-1  2]*1   + 7   = [6,  9]  -> [Ax,Az] axial -> [6, 6, 9]
    #   A2 = [-1  2]*1.5 + 0.8 = [-0.7, 3.8]
    #   A3 = [-1  2]*0.5 + 0.7 = [0.2, 1.7]
    # g = [2.25, 2.25, 2.0]  (axial)
    # Three identical 1H nuclei, each with its own axial A tensor
    A1 = [6.0, 6.0, 9.0]           # [-1*1+7, 2*1+7] -> [6, 9] MHz, axial
    A2 = [-0.7, -0.7, 3.8]         # [-1*1.5+0.8, 2*1.5+0.8] MHz, axial
    A3 = [0.2, 0.2, 1.7]           # [-1*0.5+0.7, 2*0.5+0.7] MHz, axial

    # Simulate each nucleus separately, then sum
    B_field = 308.46   # mT (MATLAB Experiment.Field)
    mwFreq  = 9.681    # GHz

    # Proton Larmor frequency at B_field
    gn_1H = 5.58569
    nu_H = gn_1H * NMAGN * (B_field * 1e-3) / PLANCK * 1e-6  # MHz
    freq_range = (6.0, 20.0)   # MHz (MATLAB Experiment.Range)

    exp = Experiment(
        mwFreq=mwFreq,
        Range=[B_field - 5.0, B_field + 5.0],
        nPoints=128,
    )
    opt = Options(GridSize=31, GridSymmetry='Ci', Verbosity=0)

    spectra = []
    for A in [A1, A2, A3]:
        Sys = SpinSystem(
            S=[0.5],
            g=[[2.25, 2.25, 2.0]],
            Nucs=['1H'],
            A=[A],
            lw=[0.0, 0.0],
        )
        freq, spc = salt(Sys, exp, opt, freq_range=freq_range, n_points=512, lw_mhz=0.1)
        spectra.append(spc.numpy())

    freq_np = freq.numpy()
    spec_sum = np.sum(spectra, axis=0)
    return freq_np, spectra, spec_sum, nu_H


def _plot(freq_np, spectra, spec_sum, nu_H):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    # Normalise sum for display
    pk = np.abs(spec_sum).max()
    ax1.plot(freq_np, spec_sum / pk if pk > 0 else spec_sum, 'k', lw=1.5)
    ax1.set_ylabel('Intensity (norm.)')
    ax1.set_title('Total ENDOR spectrum (3 × ¹H)')
    ax1.grid(True, alpha=0.3)

    colors = ['b', 'r', 'g']
    labels = ['¹H nucleus 1', '¹H nucleus 2', '¹H nucleus 3']
    for spc, col, lbl in zip(spectra, colors, labels):
        pk_i = np.abs(spc).max()
        ax2.plot(freq_np, spc / pk_i if pk_i > 0 else spc, color=col, lw=1.2, label=lbl)
    ax2.set_xlabel('ENDOR frequency (MHz)')
    ax2.set_ylabel('Intensity (norm.)')
    ax2.set_title('Separate nucleus contributions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
