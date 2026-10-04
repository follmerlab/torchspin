"""Fast-Motion Nitroxide EPR vs Correlation Time
Demonstrates motional narrowing in a nitroxide radical as a function of
rotational correlation time, from fast (1e-11 s) to near-slow (1e-9 s).

EasySpin equivalent: examples/liquids/fast_fnb.m
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
    # MATLAB source: g = [2.0088, 2.0061, 2.0027]
    # A = unitconvert([5.8, 5.8, 30.8]/10, 'mT->MHz')  (G -> mT -> MHz)
    g = [2.0088, 2.0061, 2.0027]
    A_G = np.array([5.8, 5.8, 30.8])
    A_MHz = unitconvert(A_G / 10.0, 'mT->MHz', g=np.mean(g))  # G -> mT -> MHz

    logtcorr_list = [-10.5, -10.0, -9.5, -9.0]

    exp = Experiment(mwFreq=9.5, Range=[336, 341], nPoints=1024, Harmonic=1)

    spectra = []
    B_axis  = None
    for ltc in logtcorr_list:
        Sys = SpinSystem(
            S=[0.5],
            g=[g],
            Nucs=['14N'],
            A=[list(A_MHz)],
            lw=[0.0, 0.1],
            logtcorr=ltc,
        )
        B, spc = garlic(Sys, exp)
        if B_axis is None:
            B_axis = B.numpy()
        spectra.append(spc.numpy())

    return B_axis, spectra, logtcorr_list


def _plot(B_axis, spectra, logtcorr_list):
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(logtcorr_list)))
    fig, ax = plt.subplots(figsize=(8, 5))

    for k, (y, ltc, col) in enumerate(zip(spectra, logtcorr_list, colors)):
        pk = np.abs(y).max()
        y_norm = y / pk if pk > 0 else y
        offset = k * 1.2
        ax.plot(B_axis, y_norm + offset, color=col, lw=1.2,
                label=f'log(τ_c) = {ltc:.1f}')

    ax.set_xlabel('Magnetic field (mT)')
    ax.set_ylabel('Intensity (norm., offset)')
    ax.set_title('Nitroxide fast-motion EPR: correlation time dependence')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_yticks([])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
