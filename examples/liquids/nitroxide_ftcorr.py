"""Nitroxide Fast-Motion EPR vs. Correlation Time
Demonstrates how the EPR lineshape of a nitroxide changes with rotational
correlation time (log10 scale), from narrow solution triplet to broad lines.

EasySpin equivalent: examples/liquids/nitroxide_ftcorr.m
"""
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchspin import SpinSystem, Experiment, garlic, unitconvert


def run():
    g = [2.0088, 2.0061, 2.0027]
    # MATLAB: A = unitconvert([5.8, 5.8, 30.8]/10, 'mT->MHz')
    # Values are in Gauss; divide by 10 to get mT, then convert to MHz
    A_G = np.array([5.8, 5.8, 30.8])
    A = unitconvert(A_G / 10.0, 'mT->MHz', g=np.mean(g))

    exp = Experiment(mwFreq=9.5, Range=[336, 341], nPoints=2048, Harmonic=1)
    logtcorr_vals = [-10.5, -10.0, -9.5, -9.0]

    spectra = []
    B_ref = None
    for lgt in logtcorr_vals:
        sys = SpinSystem(
            S=[0.5],
            g=[g],
            Nucs=['14N'],
            A=[list(A)],
            lw=[0.0, 0.1],
            logtcorr=lgt,
        )
        B, spec = garlic(sys, exp)
        if B_ref is None:
            B_ref = B
        spectra.append(spec.numpy())

    return B_ref, spectra, logtcorr_vals


def _plot(B, spectra, logtcorr_vals):
    B_np = B.numpy()
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(spectra)))
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, (y, lgt, col) in enumerate(zip(spectra, logtcorr_vals, colors)):
        y = y.copy(); pk = np.abs(y).max()
        if pk > 0: y /= pk
        ax.plot(B_np, y + k * 0.7, color=col, lw=1.2,
                label=f'log τ = {lgt}')
    ax.set_xlabel('Magnetic field (mT)')
    ax.set_ylabel('Intensity (offset)')
    ax.set_title('Nitroxide fast-motion EPR — correlation time series')
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
