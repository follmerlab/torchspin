"""Simple ¹H ENDOR Powder Spectrum
Demonstrates simulation of a ¹H ENDOR powder spectrum using salt().
A single ¹H coupled to an S=1/2 electron gives a Pake doublet at
ν_ENDOR = ν_H ± A/2, where ν_H is the proton Larmor frequency.

EasySpin equivalent: examples/endor/endorsimple.m
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
    Sys = SpinSystem(
        S=[0.5],
        g=[[2.0, 2.0, 2.0]],
        Nucs=['1H'],
        A=[[5.0, 5.0, 20.0]],   # MHz: axial, Aiso = (5+5+20)/3 ≈ 10 MHz
    )

    B0_mT = 330.0
    # 1H nuclear Larmor frequency (MHz) at B0_mT
    gn_1H = 5.58569   # nuclear g-factor for 1H
    nu_H = gn_1H * NMAGN * (B0_mT * 1e-3) / PLANCK * 1e-6  # MHz

    exp = Experiment(
        mwFreq=9.5,
        Range=[B0_mT - 10.0, B0_mT + 10.0],
        nPoints=256,
    )
    opt = Options(GridSize=40, GridSymmetry='Ci', Verbosity=0)
    endor_range = (nu_H - 15.0, nu_H + 15.0)

    freq, spc = salt(Sys, exp, opt, freq_range=endor_range, n_points=512, lw_mhz=0.3)
    freq_np = freq.numpy()
    spc_np = spc.numpy()
    return freq_np, spc_np, nu_H


def _plot(freq_np, spc_np, nu_H):
    y = spc_np / np.abs(spc_np).max()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(freq_np, y, 'b', lw=1.5)
    ax.axvline(nu_H, color='k', lw=0.8, ls='--', label=f'ν$_H$ = {nu_H:.2f} MHz')
    ax.axvline(nu_H + 10.0, color='r', lw=0.8, ls=':', label='ν$_H$ ± A$_{iso}$/2')
    ax.axvline(nu_H - 10.0, color='r', lw=0.8, ls=':')
    ax.set_xlabel('ENDOR frequency (MHz)')
    ax.set_ylabel('Intensity (norm.)')
    ax.set_title('¹H ENDOR powder: A = [5, 5, 20] MHz')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
