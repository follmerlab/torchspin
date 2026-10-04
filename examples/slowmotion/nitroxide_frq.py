"""Frequency Dependence of Nitroxide Slow-Motion EPR
Simulates a nitroxide radical at four microwave frequencies (3, 9.5, 35, 95 GHz)
with τ_c = 1 ns using chili() and displays all four field-swept spectra.

EasySpin equivalent: examples/slowmotion/nitroxide_frq.m
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
    Sys = SpinSystem(
        S=[0.5],
        g=[[2.008, 2.006, 2.003]],
        Nucs=['14N'],
        A=[[20.0, 20.0, 90.0]],   # MHz
        tcorr=1e-9,                # 1 ns correlation time
    )
    opt = ChiliOptions(Verbosity=0)

    mw_freqs = [3.0, 9.5, 35.0, 95.0]   # GHz
    results = []
    for mw in mw_freqs:
        # Center field ≈ mwFreq * 1e3 / (gfree * 13.996) ≈ mwFreq * 35.7 mT/GHz
        B_center = mw * 1e3 / (2.006 * 13.996)   # mT
        sweep = max(15.0, B_center * 0.06)         # ±sweep/2
        exp = Experiment(
            mwFreq=mw,
            Range=[B_center - sweep, B_center + sweep],
            nPoints=1024,
            Harmonic=1,
        )
        B, spc = chili(Sys, exp, opt)
        results.append((B, spc, mw))
    return results


def _plot(results):
    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=False)
    for ax, (B, spc, mw) in zip(axes, results):
        pk = np.abs(spc).max()
        y = spc / pk if pk > 0 else spc
        ax.plot(B, y, 'b', lw=1.2)
        ax.set_ylabel('I (norm.)')
        ax.set_title(f'{mw:g} GHz')
        ax.set_xlim(B[0], B[-1])
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Magnetic field (mT)')
    plt.suptitle('Frequency dependence of nitroxide slow-motion EPR (τ$_c$ = 1 ns)',
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(result)
