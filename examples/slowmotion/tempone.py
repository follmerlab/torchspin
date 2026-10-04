"""TEMPONE Slow-Motion EPR Spectrum
Reproduces Figure 5 (p. 38) of Schneider & Freed, Biol. Magn. Reson. 8 (1989).
Uses Diff = 1e6 s⁻¹ (slow tumbling) and typical TEMPONE g and A tensors.

EasySpin equivalent: examples/slowmotion/tempone.m
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
from torchspin.utils import unitconvert


def run():
    # TEMPONE parameters from Schneider & Freed Table 2, entry 2
    # A values originally in Gauss: 5.8, 5.8, 30.8 G -> mT (divide by 10) -> MHz
    A_mT = np.array([5.8, 5.8, 30.8]) / 10.0   # mT
    A_MHz = unitconvert(A_mT, 'mT->MHz')          # uses g=GFREE by default

    Sys = SpinSystem(
        S=[0.5],
        g=[[2.0088, 2.0061, 2.0027]],
        Nucs=['14N'],
        A=[A_MHz.tolist()],
        Diff=1e6,           # rotational diffusion coefficient, s^-1
        lw=[0.0, 0.2],      # Lorentzian broadening, FWHM, mT
    )
    exp = Experiment(
        mwFreq=9.2646,
        Range=[320, 345],   # mT — covers the TEMPONE pattern
        nPoints=1024,
        Harmonic=1,
    )
    opt = ChiliOptions(LLMK=[14, 7, 2, 6], Verbosity=0)

    B, spc = chili(Sys, exp, opt)
    return B, spc


def _plot(B, spc):
    B0 = 330.0   # mT reference (center, as in MATLAB original)
    pk = np.abs(spc).max()
    y = spc / pk if pk > 0 else spc
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(B - B0, y, 'b', lw=1.5)
    ax.set_xlabel('B − B₀ (mT)')
    ax.set_ylabel('Intensity (norm.)')
    ax.set_title('TEMPONE slow-motion EPR (Diff = 10⁶ s⁻¹)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
