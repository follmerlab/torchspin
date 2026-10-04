"""Cobalt Trimer Solution EPR
Isotropic fast-motion solution EPR of a cobalt trimer with 3 equivalent
59Co nuclei (I=7/2); demonstrates the complex hyperfine pattern arising
from three equivalent high-spin nuclei.

EasySpin equivalent: examples/liquids/cobalttrimer.m
"""
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchspin import SpinSystem, Experiment, garlic


def run():
    # 3 equivalent Co nuclei (59Co, I=7/2), A=100 MHz isotropic
    # Specify each Co separately (n>1 not supported in fast-motion garlic)
    sys = SpinSystem(
        S=[0.5],
        g=[[2.0, 2.0, 2.0]],
        Nucs=['59Co', '59Co', '59Co'],
        A=[[100.0, 100.0, 100.0],
           [100.0, 100.0, 100.0],
           [100.0, 100.0, 100.0]],
        lw=[0.0, 0.8],
    )
    exp = Experiment(mwFreq=9.5, Range=[310, 370], nPoints=4096, Harmonic=1)
    B, spec = garlic(sys, exp)
    return B, spec


def _plot(B, spec):
    y = spec.numpy(); y /= np.abs(y).max()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(B.numpy(), y, 'b', lw=0.8)
    ax.set_xlabel('Magnetic field (mT)')
    ax.set_ylabel('Intensity (norm.)')
    ax.set_title('Cobalt trimer — 3×⁵⁹Co (I=7/2), A=100 MHz, X-band')
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
