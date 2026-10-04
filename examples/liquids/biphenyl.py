"""Biphenyl Radical Anion Solution EPR
Simulates the fast-motion solution EPR spectrum of the biphenyl radical anion.
Biphenyl has two sets of 4 equivalent protons; literature couplings A ≈ 7.43
and 2.74 MHz. Demonstrates a multi-line solution spectrum from 8 protons.

EasySpin equivalent: examples/liquids/biphenyl.m
"""
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchspin import SpinSystem
from torchspin.garlic import garlic
from torchspin.experiment import Experiment


def run():
    # Literature hyperfine couplings for biphenyl radical anion (MHz)
    # Group 1: 4 ortho protons, A ≈ 7.43 MHz
    # Group 2: 4 meta protons, A ≈ 2.74 MHz
    A1_MHz = 7.43   # ortho H
    A2_MHz = 2.74   # meta H

    Sys = SpinSystem(
        S=[0.5],
        g=[[2.0027, 2.0027, 2.0027]],
        Nucs=['1H'] * 8,
        A=(
            [[A1_MHz, A1_MHz, A1_MHz]] * 4
            + [[A2_MHz, A2_MHz, A2_MHz]] * 4
        ),
        lw=[0.0, 0.1],
    )

    exp = Experiment(mwFreq=9.5, Range=[336, 342], nPoints=2048, Harmonic=1)

    B, spc = garlic(Sys, exp)

    return B.numpy(), spc.numpy()


def _plot(B_np, y):
    pk = np.abs(y).max()
    y_norm = y / pk if pk > 0 else y

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(B_np, y_norm, 'b', lw=0.9)
    ax.set_xlabel('Magnetic field (mT)')
    ax.set_ylabel('Intensity (norm.)')
    ax.set_title('Biphenyl radical anion: 4+4 equivalent $^1$H, X-band solution EPR')
    ax.set_yticks([])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
