"""Nitroxide EPR vs. Rotational Correlation Time
Shows how the EPR lineshape of a nitroxide changes with τ_c from
slow-motion (powder-like) to fast-motion (narrow triplet).

EasySpin equivalent: examples/slowmotion/nitroxide_tcorr.m
"""
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchspin import SpinSystem
from torchspin.chili import chili, ChiliOptions
from torchspin.experiment import Experiment


def run():
    exp = Experiment(mwFreq=9.5, Range=[328, 352], nPoints=2048, Harmonic=1)
    opt = ChiliOptions(Verbosity=0)

    # Correlation times from slow (powder-like) to fast (solution-like)
    tcorr_list = [1e-7, 1e-8, 1e-9, 1e-10]
    results = []
    for tcorr in tcorr_list:
        Sys = SpinSystem(
            S=[0.5],
            g=[[2.0089, 2.0058, 2.0021]],
            Nucs=['14N'],
            A=[[16.0, 16.0, 100.0]],   # MHz
            tcorr=tcorr,
            lw=[0.0, 0.05],
        )
        B, spc = chili(Sys, exp, opt)
        results.append((B, spc, tcorr))
    return results


def _plot(results):
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(results)))
    fig, axes = plt.subplots(len(results), 1, figsize=(9, 10), sharex=True)
    for ax, (B, spc, tcorr), col in zip(axes, results, colors):
        y = np.asarray(spc, dtype=float).copy()
        pk = np.abs(y).max()
        if pk > 0:
            y /= pk
        ax.plot(B, y, color=col, lw=1.5)
        ax.set_ylabel('I (norm.)')
        ax.set_title(f'τ$_c$ = {tcorr:.0e} s')
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Magnetic field (mT)')
    plt.suptitle('Nitroxide EPR: slow-motion to fast-motion regime', fontsize=12)
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(result)
