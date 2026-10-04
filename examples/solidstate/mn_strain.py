"""Mn(II) X-Band EPR with D Strain and Field Modulation
S=5/2 Mn(II) with 55Mn hyperfine, D strain; absorption and first-harmonic derivative.

EasySpin equivalent: examples/solidstate/mn_strain.m
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchspin import SpinSystem, Experiment, Options, pepper, fieldmod


def run():
    Sys = SpinSystem(
        S=[2.5],   # S = 5/2
        g=[[2.0, 2.0, 2.0]],
        Nucs=['55Mn'],
        A=[[253.0, 253.0, 253.0]],   # isotropic A, MHz
        D=[[-170.0/3, -170.0/3, 2*170.0/3]],  # D=170 MHz axial: [Dxx,Dyy,Dzz]
        DStrain=[120.0],              # FWHM of Gaussian D distribution, MHz
        lw=[1.0, 0.0],
    )

    exp = Experiment(
        mwFreq=9.4,
        Range=[270, 400],
        nPoints=2601,
        Harmonic=0,   # absorption spectrum
    )
    opt = Options(GridSize=[19, 4], Verbosity=0)

    B, spec0 = pepper(Sys, exp, opt)
    B_np = B.numpy()
    spec0_np = spec0.numpy()

    # Generate first-harmonic spectrum with field modulation (0.1 mT amplitude)
    modamp = 0.1  # mT
    spec1 = fieldmod(B_np, spec0_np, modamp, harmonic=1)

    return B_np, spec0_np, spec1


def _plot(B_np, spec0, spec1):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    y0 = spec0 / np.abs(spec0).max() if np.abs(spec0).max() > 0 else spec0
    ax1.plot(B_np, y0, 'k', lw=1.2)
    ax1.set_ylabel('Absorption (norm.)')
    ax1.set_title('Mn(II) S=5/2 absorption spectrum (Harmonic=0)')
    ax1.grid(True, alpha=0.3)

    y1 = spec1 / np.abs(spec1).max() if np.abs(spec1).max() > 0 else spec1
    ax2.plot(B_np, y1, 'b', lw=1.2)
    ax2.set_xlabel('Magnetic field (mT)')
    ax2.set_ylabel('d\u03c7\u2032\u2032/dB (norm.)')
    ax2.set_title('First-harmonic derivative (field modulation, A_mod=0.1 mT)')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Mn(II) X-band EPR with D strain (D=170, \u03b4D=120 MHz)', fontsize=12)
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
