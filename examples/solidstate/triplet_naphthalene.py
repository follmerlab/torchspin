"""Photo-Excited Triplet State of Naphthalene
X-band powder EPR of the S=1 triplet with ZFS from Hutchison et al. 1961.

EasySpin equivalent: examples/solidstate/triplet_naphthalene.m
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchspin import SpinSystem, Experiment, Options, pepper

CM1_TO_MHZ = 29979.2458


def run():
    D_cm = 0.1003
    E_cm = 0.0137
    D_MHz = D_cm * CM1_TO_MHZ
    E_MHz = E_cm * CM1_TO_MHZ

    Dxx = -D_MHz / 3 + E_MHz
    Dyy = -D_MHz / 3 - E_MHz
    Dzz = 2 * D_MHz / 3

    naphthalene = SpinSystem(
        S=[1.0],
        g=[[2.003, 2.003, 2.003]],
        D=[[Dxx, Dyy, Dzz]],
        lw=[6.0, 0.0],
    )

    exp = Experiment(
        mwFreq=9.5,
        Range=[0, 500],
        nPoints=1024,
        Harmonic=1,
    )
    opt = Options(GridSize=50, GridSymmetry='Ci', Verbosity=0)

    B, spc = pepper(naphthalene, exp, opt)
    return B.numpy(), spc.numpy(), D_cm, E_cm


def _plot(B_np, spc, D_cm, E_cm):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(B_np, spc, 'k', lw=1.0)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel('Magnetic field (mT)')
    ax.set_ylabel('dI/dB (arb. u.)')
    ax.set_title(f'Naphthalene triplet, X-band\nD = {D_cm} cm\u207b\u00b9, E = {E_cm} cm\u207b\u00b9')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
