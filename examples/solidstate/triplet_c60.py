"""Triplet-State C60 Fullerene EPR
S=1 triplet of C60 with near-spherical symmetry and very small ZFS (D ~ 1e-4 cm^-1).

EasySpin equivalent: examples/solidstate/triplet_C60fullerene.m
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchspin import SpinSystem, Experiment, Options, pepper

CM1_TO_MHZ = 29979.2458


def run():
    D_cm = 1.14e-4
    E_cm = 0.0
    D_MHz = D_cm * CM1_TO_MHZ
    E_MHz = E_cm * CM1_TO_MHZ

    Dxx = -D_MHz / 3 + E_MHz
    Dyy = -D_MHz / 3 - E_MHz
    Dzz = 2 * D_MHz / 3

    Sys = SpinSystem(
        S=[1.0],
        g=[[2.0023, 2.0023, 2.0023]],
        D=[[Dxx, Dyy, Dzz]],
        lw=[0.3, 0.0],
    )

    exp = Experiment(mwFreq=9.5, Range=[335, 345], nPoints=2048, Harmonic=1)
    opt = Options(GridSize=50, GridSymmetry='Ci', Verbosity=0)

    B, spc = pepper(Sys, exp, opt)
    return B.numpy(), spc.numpy(), D_cm


def _plot(B_np, spc, D_cm):
    y = spc / np.abs(spc).max() if np.abs(spc).max() > 0 else spc

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(B_np, y, 'b', lw=1.5)
    ax.set_xlabel('Magnetic field (mT)')
    ax.set_ylabel('Intensity (norm.)')
    ax.set_title(f'C\u2086\u2080 triplet state: S=1, D = {D_cm:.2e} cm\u207b\u00b9, X-band')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
