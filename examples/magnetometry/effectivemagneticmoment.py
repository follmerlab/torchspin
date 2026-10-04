"""Effective Magnetic Moment vs Temperature
Compares the single-molecule magnetic moment mu (in Bohr magnetons) and
the effective magnetic moment mu_eff for S=1, g=2.05, D=5 cm^-1. At high T
mu_eff plateaus at g*sqrt(S*(S+1)); ZFS causes a downturn at low T.

EasySpin equivalent: examples/magnetometry/effectivemagneticmoment.m
"""
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchspin import SpinSystem, unitconvert, BMAGN, BOLTZMANN, AVOGADRO
from torchspin.curry import curry


# mu0 for susceptibility -> mu_eff conversion
_MU0 = 4.0 * np.pi * 1e-7   # H/m


def run():
    # MATLAB: Sys.S=1; Sys.g=2.05; Sys.D = 5*30e4 cm^-1->MHz (5 cm^-1)
    g_val = 2.05
    D_cm1 = 5.0
    D_MHz = unitconvert(D_cm1, 'cm^-1->MHz')

    # EasySpin axial D convention: D = Dzz - (Dxx+Dyy)/2
    # Traceless tensor: Dxx = Dyy = -D/3, Dzz = 2*D/3
    Sys = SpinSystem(
        S=[1.0],
        g=[[g_val, g_val, g_val]],
        D=[[-D_MHz / 3.0, -D_MHz / 3.0, 2.0 * D_MHz / 3.0]],
    )

    # T = 0:100 K — skip T=0 (not physical); use 1 K as minimum
    T_arr    = np.arange(1, 101, dtype=float)
    B_fields = np.array([100.0])   # 0.1 T in mT

    mu_BM, chi_mol = curry(Sys, B_fields, T_arr, grid_size=15)

    # mu_BM shape (nB, nT): magnetic moment in Bohr magnetons
    mu = mu_BM[0, :].numpy()      # mu_B / molecule

    # mu_eff from susceptibility (SI formula):
    # mu_eff = sqrt(3 k_B T chi_mol / (mu0 N_A mu_B^2))
    chi = chi_mol[0, :].numpy()   # m³/mol
    mu_eff = np.sqrt(3.0 * BOLTZMANN * T_arr * chi / (_MU0 * AVOGADRO * BMAGN**2))

    # High-temperature theoretical limit
    mu_eff_lim = g_val * np.sqrt(1.0 * 2.0)   # g * sqrt(S*(S+1)) for S=1

    return T_arr, mu, mu_eff, mu_eff_lim, g_val, D_cm1


def _plot(T_arr, mu, mu_eff, mu_eff_lim, g_val, D_cm1):
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    axes[0].plot(T_arr, mu, 'b', lw=1.5)
    axes[0].set_ylabel('$\\mu$ ($\\mu_B$)')
    axes[0].set_title(f'S=1, g={g_val}, D={D_cm1} cm$^{{-1}}$, B=0.1 T')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(T_arr, mu_eff, 'r', lw=1.5, label='$\\mu_{eff}$ (simulation)')
    axes[1].axhline(mu_eff_lim, color='k', lw=1.2, ls='--',
                    label=f'High-T limit: {mu_eff_lim:.3f}')
    axes[1].set_xlabel('Temperature (K)')
    axes[1].set_ylabel('$\\mu_{eff}$')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
