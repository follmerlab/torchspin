"""ENDOR Matrix Diagonalization — Four-Proton System
Simulates a ¹H ENDOR powder spectrum for a system with four protons using
salt() (matrix diagonalization). The MATLAB original compares matrix vs.
perturbation methods; torchspin's salt() uses the matrix method only.

EasySpin equivalent: examples/endor/endorperturb.m
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
    # Parameters from MATLAB endorperturb.m (nucspinadd calls)
    # Proton 1: A = [3, 6, 2]  MHz
    # Proton 2: A = [5, 3, 1]  MHz
    # Proton 3: A = [8, 6,-4]  MHz
    # Proton 4: A = [1,-1, 1.6] MHz
    Sys = SpinSystem(
        S=[0.5],
        g=[[2.3, 2.1, 2.0]],
        Nucs=['1H', '1H', '1H', '1H'],
        A=[
            [3.0, 6.0, 2.0],
            [5.0, 3.0, 1.0],
            [8.0, 6.0, -4.0],
            [1.0, -1.0, 1.6],
        ],
        HStrain=[200.0, 200.0, 200.0],   # MHz
    )

    B_field = 300.0   # mT  (MATLAB Exp.Field)
    mwFreq  = 9.5     # GHz

    # ¹H Larmor frequency at B_field
    gn_1H = 5.58569
    nu_H = gn_1H * NMAGN * (B_field * 1e-3) / PLANCK * 1e-6   # MHz
    freq_range = (nu_H - 10.0, nu_H + 10.0)   # ±10 MHz around ν_H

    exp = Experiment(
        mwFreq=mwFreq,
        Range=[B_field - 8.0, B_field + 8.0],
        nPoints=128,
    )
    opt = Options(GridSize=31, GridSymmetry='Ci', Verbosity=1)

    print("Computing ENDOR spectrum (matrix diagonalization)...")
    freq, spc = salt(Sys, exp, opt,
                     freq_range=freq_range, n_points=512, lw_mhz=0.1)

    freq_np = freq.numpy() - nu_H   # shift to ν - ν_H axis (MATLAB-style)
    spc_np  = spc.numpy()
    return freq_np, spc_np, nu_H


def _plot(freq_np, spc_np, nu_H):
    pk = np.abs(spc_np).max()
    y = spc_np / pk if pk > 0 else spc_np
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(freq_np, y, 'b', lw=1.5, label='Matrix diagonalization')
    ax.axvline(0, color='k', lw=0.8, ls='--', label=f'ν$_H$ = {nu_H:.1f} MHz')
    ax.set_xlabel('ν − ν$_H$ (MHz)')
    ax.set_ylabel('Intensity (norm.)')
    ax.set_title('¹H ENDOR: 4-proton system (matrix method)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
