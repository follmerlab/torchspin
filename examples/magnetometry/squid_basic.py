"""Magnetization vs Field (SQUID-like)
Computes M(B) saturation curves at low temperature for S=1/2, 1, 3/2, 5/2,
demonstrating Brillouin-function behaviour.

EasySpin equivalent: examples/magnetometry/squid_basic.m
"""
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchspin import SpinSystem
from torchspin.curry import curry


def run():
    T_K      = 2.0
    B_fields = np.linspace(0.0, 7000.0, 200)   # 0–7 T in mT

    systems = [
        (SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]]), 'S=1/2'),
        (SpinSystem(S=[1.0], g=[[2.0, 2.0, 2.0]]), 'S=1'),
        (SpinSystem(S=[1.5], g=[[2.0, 2.0, 2.0]]), 'S=3/2'),
        (SpinSystem(S=[2.5], g=[[2.0, 2.0, 2.0]]), 'S=5/2'),
    ]

    T_arr  = np.array([T_K])
    curves = []
    for Sys, label in systems:
        mu_BM, _ = curry(Sys, B_fields, T_arr, grid_size=10)
        M = mu_BM[:, 0].numpy()
        curves.append((M, label, Sys.Spins[0]))

    return B_fields, curves, T_K


def _plot(B_fields, curves, T_K):
    colors = ['b', 'r', 'g', 'm']
    fig, ax = plt.subplots(figsize=(8, 5))

    for (M, label, S_val), col in zip(curves, colors):
        M_sat = 2 * S_val  # g=2 saturation
        ax.plot(B_fields / 1000, M, color=col, lw=2,
                label=f'{label}  (M$_{{sat}}$={M_sat:.0f} $\\mu_B$)')

    ax.set_xlabel('Magnetic field (T)')
    ax.set_ylabel('Magnetization ($\\mu_B$ / molecule)')
    ax.set_title(f'M(B) saturation curves at T = {T_K} K')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
