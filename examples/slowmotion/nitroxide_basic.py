"""Nitroxide Basic Slow-Motion EPR
Demonstrates field-swept slow-motion CW EPR of a nitroxide radical (TEMPO)
using the Stochastic Liouville Equation (SLE) approach in chili().

EasySpin equivalent: examples/slowmotion/nitroxide_basic.m
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
    # TEMPO-like nitroxide: g and A tensors, 3 ns correlation time
    Sys = SpinSystem(
        S=[0.5],
        g=[[2.0089, 2.0058, 2.0021]],
        Nucs=['14N'],
        A=[[16.0, 16.0, 100.0]],   # MHz
        tcorr=3e-9,                  # 3 ns — slow motion
        lw=[0.0, 0.05],              # mT Lorentzian FWHM
    )
    exp = Experiment(mwFreq=9.5, Range=[328, 352], nPoints=2048, Harmonic=1)
    opt = ChiliOptions(Verbosity=0)

    B, spc = chili(Sys, exp, opt)
    return B, spc


def _plot(B, spc):
    y = spc / np.abs(spc).max()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(B, y, 'b', lw=1.5)
    ax.set_xlabel('Magnetic field (mT)')
    ax.set_ylabel('Intensity (norm.)')
    ax.set_title('Nitroxide slow-motion EPR: S=1/2 + ¹⁴N, τ$_c$ = 3 ns')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
