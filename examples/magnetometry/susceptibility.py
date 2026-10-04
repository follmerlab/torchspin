"""Temperature Dependence of Magnetic Susceptibility
Computes chi_mol(T) and chi_mol*T for S=7/2 with axial ZFS D=2 cm^-1 at
B=1 T, showing the deviation from Curie law at low temperature due to ZFS.

EasySpin equivalent: examples/magnetometry/susceptibility.m
"""
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchspin import SpinSystem, unitconvert
from torchspin.curry import curry


def run():
    # MATLAB: Sys.S = 7/2; D = 2 cm^-1; Sys.D = D*30e3 (cm^-1 -> MHz)
    D_cm1 = 2.0                     # cm^-1
    D_MHz = unitconvert(D_cm1, 'cm^-1->MHz')   # cm^-1 -> MHz

    # EasySpin axial D convention: D = Dzz - (Dxx+Dyy)/2
    # Traceless tensor: Dxx = Dyy = -D/3, Dzz = 2*D/3
    Sys = SpinSystem(
        S=[3.5],
        g=[[2.0, 2.0, 2.0]],
        D=[[-D_MHz / 3.0, -D_MHz / 3.0, 2.0 * D_MHz / 3.0]],
    )

    # Temperature grid matching MATLAB: [0.1:0.1:1, 1:0.5:50] K
    T_fine   = np.arange(0.1, 1.01, 0.1)
    T_coarse = np.arange(1.0, 50.5,  0.5)
    T_arr = np.unique(np.concatenate([T_fine, T_coarse]))

    B_fields = np.array([1000.0])   # 1 T in mT

    _, chi_mol = curry(Sys, B_fields, T_arr, grid_size=15)

    # chi_mol shape (nB, nT), in m³/mol
    chi = chi_mol[0, :].numpy()     # m³/mol
    chiT = chi * T_arr               # m³·K/mol

    return T_arr, chi, chiT, D_cm1


def _plot(T_arr, chi, chiT, D_cm1):
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    axes[0].plot(T_arr, chi, 'b', lw=1.5)
    axes[0].set_ylabel('$\\chi_{mol}$ (m$^3$ mol$^{-1}$)')
    axes[0].set_title(f'S=7/2, D = {D_cm1} cm$^{{-1}}$, B = 1 T')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(T_arr, chiT, 'r', lw=1.5)
    axes[1].set_xlabel('Temperature (K)')
    axes[1].set_ylabel('$\\chi_{mol} T$ (K m$^3$ mol$^{-1}$)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
