"""Temperature-Dependent EPR of an S=1 Triplet
Boltzmann-weighted powder EPR at multiple temperatures (80, 40, 20, 10, 5, 2 K).

EasySpin equivalent: examples/solidstate/temperature.m
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchspin import SpinSystem, Experiment, Options, pepper


def run():
    Sys = SpinSystem(
        S=[1.0],
        g=[[2.0, 2.0, 2.0]],
        D=[200.0, 200.0, -400.0],
        lw=[1.0, 0.0],
    )

    exp_base = dict(mwFreq=9.5, Range=[300, 380], nPoints=1024, Harmonic=0)
    opt = Options(GridSize=40, GridSymmetry='Ci', Verbosity=0)
    Temps = [80.0, 40.0, 20.0, 10.0, 5.0, 2.0]

    B_ref = None
    spectra = []
    for T in Temps:
        exp = Experiment(**exp_base, Temperature=T)
        B, spc = pepper(Sys, exp, opt)
        if B_ref is None:
            B_ref = B.numpy()
        pk = float(spc.abs().max())
        y = spc.numpy() / pk if pk > 0 else spc.numpy()
        spectra.append(y)

    return B_ref, spectra, Temps


def _plot(B_ref, spectra, Temps):
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(Temps)))
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (T, y, col) in enumerate(zip(Temps, spectra, colors)):
        ax.plot(B_ref, y + 0.3 * i, color=col, lw=1.5, label=f'{T} K')
    ax.set_xlabel('Magnetic field (mT)')
    ax.set_ylabel('Intensity (norm., offset)')
    ax.set_title('Temperature-dependent EPR: S=1 triplet, 9.5 GHz')
    ax.legend(title='Temperature')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
