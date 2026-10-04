"""Effect of Tumbling on EPR Lineshape
Simulates a nitroxide-like radical with an axial g tensor at six correlation
times from 1 ns to 300 ns, showing the transition from near-fast-motion to
deep slow-motion. Spectra are stacked vertically for comparison.

EasySpin equivalent: examples/slowmotion/tumbling.m
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
    exp = Experiment(
        mwFreq=9.5,
        Range=[337.0, 340.0],
        nPoints=512,
        Harmonic=1,
    )
    # Larger basis for slow tumbling (MATLAB Opt.LLMK = [14 3 6 6])
    opt = ChiliOptions(LLMK=[14, 3, 6, 6], Verbosity=0)

    tcorr_values = np.array([1, 3, 10, 30, 100, 300]) * 1e-9   # seconds

    results = []
    for tcorr in tcorr_values:
        Sys = SpinSystem(
            S=[0.5],
            g=[[2.008, 2.008, 2.003]],
            lw=[0.05, 0.0],    # 0.05 mT Gaussian broadening
            tcorr=float(tcorr),
        )
        B, spc = chili(Sys, exp, opt)
        results.append((B, spc, float(tcorr)))
    return results


def _plot(results):
    n = len(results)
    colors = plt.cm.viridis(np.linspace(0, 0.85, n))
    fig, ax = plt.subplots(figsize=(8, 8))

    offset = 0.0
    gap = 2.2
    for i, (B, spc, tcorr) in enumerate(results):
        pk = np.abs(spc).max()
        y = spc / pk if pk > 0 else spc
        ax.plot(B, y + offset, color=colors[i], lw=1.5,
                label=f'τ$_c$ = {tcorr*1e9:.0f} ns')
        offset += gap

    ax.set_xlabel('Magnetic field (mT)')
    ax.set_ylabel('Intensity (stacked, arb. u.)')
    ax.set_title('Tumbling nitroxide: axial g tensor vs. τ$_c$')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(result)
