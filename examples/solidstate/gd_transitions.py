"""Gadolinium Complex: Individual Transitions at W-Band
S=7/2 Gd(III) at 95 GHz with D strain and T=10 K Boltzmann populations.

EasySpin equivalent: examples/solidstate/gd_transitions.m
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchspin import SpinSystem, Experiment, Options, pepper


def run():
    Gd = SpinSystem(
        S=[3.5],   # S = 7/2
        g=[[1.992, 1.992, 1.992]],
        D=[1200.0, 0.0, -1200.0],   # D=1200 MHz, E=0 -> [Dxx=0, Dyy=0, Dzz=0]... use D/E form
        DStrain=[200.0],
        lw=[2.0, 0.0],
    )

    # D=1200 MHz, E=0: Dxx = -D/3+E = -400, Dyy = -D/3-E = -400, Dzz = 2D/3 = 800
    D_val = 1200.0
    E_val = 0.0
    Dxx = -D_val / 3 + E_val
    Dyy = -D_val / 3 - E_val
    Dzz = 2 * D_val / 3

    Gd = SpinSystem(
        S=[3.5],
        g=[[1.992, 1.992, 1.992]],
        D=[[Dxx, Dyy, Dzz]],
        DStrain=[200.0],
        lw=[2.0, 0.0],
    )

    exp = Experiment(
        mwFreq=95.0,
        Range=[2900, 3900],
        nPoints=2048,
        Temperature=10.0,
        Harmonic=0,
    )
    opt = Options(GridSize=[19, 4], Verbosity=0)

    B, spc = pepper(Gd, exp, opt)
    return B.numpy(), spc.numpy()


def _plot(B_np, spc):
    y = spc / np.abs(spc).max() if np.abs(spc).max() > 0 else spc

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(B_np, y, 'k', lw=1.5)
    ax.set_xlabel('Magnetic field (mT)')
    ax.set_ylabel('Absorption (norm.)')
    ax.set_title('Gd(III) S=7/2, W-band (95 GHz), T=10 K, D=1200 MHz')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
