"""Two Coupled Electron Spins
Two S=1/2 spins with isotropic exchange coupling J; strong vs. weak exchange limits.

EasySpin equivalent: examples/solidstate/eespins.m
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchspin import SpinSystem, Experiment, Options, pepper


def run():
    exp = Experiment(mwFreq=9.5, Range=[200, 500], nPoints=2048, Harmonic=1)
    opt = Options(GridSize=40, GridSymmetry='Ci', Verbosity=0)

    # Strong exchange: large J -> triplet + singlet well-separated
    # ee field takes shape (nPairs, 3) for diagonal tensors
    Sys_strong = SpinSystem(
        S=[0.5, 0.5],
        g=[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
        ee=[[800.0, 800.0, 800.0]],
        lw=[1.0, 0.0],
    )

    # Weak exchange: small J -> two nearly independent S=1/2 centers
    Sys_weak = SpinSystem(
        S=[0.5, 0.5],
        g=[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
        ee=[[10.0, 10.0, 10.0]],
        lw=[1.0, 0.0],
    )

    B, spc_strong = pepper(Sys_strong, exp, opt)
    _, spc_weak = pepper(Sys_weak, exp, opt)

    return B.numpy(), spc_strong.numpy(), spc_weak.numpy()


def _plot(B_np, spc_strong, spc_weak):
    def norm(s):
        pk = np.abs(s).max()
        return s / pk if pk > 0 else s

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(B_np, norm(spc_strong), 'b', lw=1.5)
    axes[0].set_title('Strong exchange (J = 800 MHz): triplet-like spectrum')
    axes[0].set_ylabel('Intensity (norm.)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(B_np, norm(spc_weak), 'r', lw=1.5)
    axes[1].set_title('Weak exchange (J = 10 MHz): two nearly independent S=1/2 centers')
    axes[1].set_xlabel('Magnetic field (mT)')
    axes[1].set_ylabel('Intensity (norm.)')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Two coupled S=1/2 spins: isotropic exchange', fontsize=12)
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
