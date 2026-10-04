"""Fremy's Salt 14N Solution EPR
Classic nitroxide radical (potassium nitrosodisulfonate) showing the 3-line
14N hyperfine pattern in fast-motion solution.

EasySpin equivalent: examples/liquids/fremysalt.m
"""
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchspin import SpinSystem, unitconvert
from torchspin.garlic import garlic
from torchspin.experiment import Experiment


def run():
    # Literature values: g = 2.0055, A(14N) = 13.09 G
    # Convert G -> mT -> MHz
    A_G   = 13.09          # Gauss
    A_mT  = A_G * 0.1     # 1 G = 0.1 mT
    A_MHz = A_G * 2.80250   # MHz per gauss at g ≈ 2 (as in the EasySpin example)

    Sys = SpinSystem(
        S=[0.5],
        g=[[2.0055, 2.0055, 2.0055]],
        Nucs=['14N'],
        A=[[A_MHz, A_MHz, A_MHz]],
        lw=[0.0, 0.15],
    )

    exp = Experiment(mwFreq=9.5, Range=[335, 342], nPoints=2048, Harmonic=1)

    B, spc = garlic(Sys, exp)

    return B.numpy(), spc.numpy(), A_G, A_MHz


def _plot(B_np, y, A_G, A_MHz):
    pk = np.abs(y).max()
    y_norm = y / pk if pk > 0 else y

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(B_np, y_norm, 'b', lw=1.2)
    ax.set_xlabel('Magnetic field (mT)')
    ax.set_ylabel('Intensity (norm.)')
    ax.set_title(
        f"Fremy's salt: S=1/2, 14N triplet (fast-motion), X-band\n"
        f"A(14N) = {A_G:.2f} G = {A_MHz:.2f} MHz"
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
