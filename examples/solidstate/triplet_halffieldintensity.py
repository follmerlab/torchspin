"""Triplet Half-Field Transition and D Strain
S=1 triplet: built-in DStrain vs explicit loop over D distribution vs HStrain only.

EasySpin equivalent: examples/solidstate/triplet_halffieldintensity.m
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchspin import SpinSystem, Experiment, Options, pepper, gaussian


def run():
    D0 = 2000.0    # center of D distribution, MHz
    Dfwhm = 500.0  # FWHM of Gaussian D distribution, MHz

    exp = Experiment(mwFreq=9.5, Range=[100, 500], nPoints=1024, Harmonic=1)
    opt = Options(GridSize=90, GridSymmetry='auto', Verbosity=0)

    # (1) Built-in DStrain
    Triplet_dstrain = SpinSystem(
        S=[1.0],
        g=[[2.0, 2.0, 2.0]],
        D=[[-D0/3, -D0/3, 2*D0/3]],
        DStrain=[Dfwhm],
        HStrain=[50.0, 50.0, 50.0],
        lw=[0.5, 0.0],
    )
    B, spc_DStrain = pepper(Triplet_dstrain, exp, opt)
    B_np = B.numpy()

    # (2) Explicit loop over D distribution
    D_vals = np.linspace(-1, 1, 51) * 2 * Dfwhm + D0
    weights = gaussian(D_vals, D0, Dfwhm)[0]  # absorption component
    weights = weights / weights.sum()

    spc_Dloop = np.zeros(len(B_np))
    for D_k, w in zip(D_vals, weights):
        Sys_k = SpinSystem(
            S=[1.0],
            g=[[2.0, 2.0, 2.0]],
            D=[[-D_k/3, -D_k/3, 2*D_k/3]],
            HStrain=[50.0, 50.0, 50.0],
            lw=[0.5, 0.0],
        )
        _, spc_k = pepper(Sys_k, exp, opt)
        spc_Dloop += w * spc_k.numpy()

    # (3) HStrain only (no DStrain)
    Triplet_hstrain = SpinSystem(
        S=[1.0],
        g=[[2.0, 2.0, 2.0]],
        D=[[-D0/3, -D0/3, 2*D0/3]],
        HStrain=[280.0, 280.0, 280.0],
        lw=[0.5, 0.0],
    )
    _, spc_HStrain = pepper(Triplet_hstrain, exp, opt)

    # Normalize all spectra to their peak
    def norm(s):
        pk = np.abs(s).max()
        return s / pk if pk > 0 else s

    return B_np, norm(spc_DStrain.numpy()), norm(spc_Dloop), norm(spc_HStrain.numpy())


def _plot(B_np, spc_DStrain, spc_Dloop, spc_HStrain):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(B_np, spc_DStrain, 'b', lw=1.5, label='DStrain (built-in)')
    ax.plot(B_np, spc_Dloop, 'r--', lw=1.5, label='loop over D distribution')
    ax.plot(B_np, spc_HStrain, 'g:', lw=1.5, label='HStrain only')
    ax.set_xlabel('Magnetic field (mT)')
    ax.set_ylabel('d\u03c7\u2032\u2032/dB (norm.)')
    ax.set_title('S=1 triplet: half-field transition and D strain')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
