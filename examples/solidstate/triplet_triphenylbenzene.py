"""Ground-State Triplet of Triphenylbenzene Dianion
S=1 organic triplet with ZFS converted from cm^-1 to MHz via unitconvert.

EasySpin equivalent: examples/solidstate/triplet_triphenylbenzene.m
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchspin import SpinSystem, Experiment, Options, pepper, unitconvert


def run():
    # ZFS parameters (Jesse et al, Mol.Phys. 6, 633 (1963))
    D_cminv = 0.0435   # cm^-1
    E_cminv = 0.0      # cm^-1

    # Convert cm^-1 -> MHz
    D_MHz = float(unitconvert(D_cminv, 'cm^-1->MHz'))
    E_MHz = float(unitconvert(E_cminv, 'cm^-1->MHz'))

    # Convert [D, E] -> traceless diagonal ZFS tensor [Dxx, Dyy, Dzz]
    Dxx = -D_MHz / 3 + E_MHz
    Dyy = -D_MHz / 3 - E_MHz
    Dzz = 2 * D_MHz / 3

    Sys = SpinSystem(
        S=[1.0],
        g=[[2.0, 2.0, 2.0]],   # isotropic g
        D=[[Dxx, Dyy, Dzz]],
        lw=[4.0, 0.0],         # Sys.lwpp = 4 mT Gaussian
    )

    exp = Experiment(mwFreq=9.150, Range=[100, 500], nPoints=1024, Harmonic=1)
    opt = Options(GridSize=[19, 4], Verbosity=0)

    B, spc = pepper(Sys, exp, opt)
    return B.numpy(), spc.numpy(), D_cminv, E_cminv, D_MHz


def _plot(B_np, spc, D_cminv, E_cminv, D_MHz):
    y = spc / np.abs(spc).max() if np.abs(spc).max() > 0 else spc

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(B_np, y, 'k', lw=1.5)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel('Magnetic field (mT)')
    ax.set_ylabel('dI/dB (norm.)')
    ax.set_title(
        f'Triphenylbenzene dianion triplet: D = {D_cminv} cm\u207b\u00b9 = {D_MHz:.0f} MHz'
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
