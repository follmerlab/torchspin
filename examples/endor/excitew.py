"""ENDOR Lineshape vs. Broadening
Simulates a ¹H ENDOR spectrum for an axial S=1/2 system and shows how
the apparent lineshape changes as the ENDOR linewidth (lw_mhz) is varied.

Note: The MATLAB original (excitew.m) varies Exp.ExciteWidth, which controls
orientation selectivity by limiting the EPR excitation bandwidth. torchspin's
salt() accumulates all EPR-resonant orientations uniformly; this example
instead varies the ENDOR Gaussian broadening (lw_mhz) to illustrate lineshape
sensitivity — a related concept.

EasySpin equivalent: examples/endor/excitew.m
"""
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchspin import SpinSystem
from torchspin.salt import salt
from torchspin.experiment import Experiment, Options


def run():
    # Axial S=1/2 system with one proton (from MATLAB excitew.m)
    # Sys.A = [-1 2]*2 + 3  ->  Ax = -1*2+3 = 1 MHz, Az = 2*2+3 = 7 MHz
    Sys = SpinSystem(
        S=[0.5],
        g=[[2.0, 2.0, 2.2]],
        Nucs=['1H'],
        A=[[1.0, 1.0, 7.0]],   # MHz: axial A tensor
    )

    exp = Experiment(
        mwFreq=9.5,
        Range=[305.0, 325.0],   # mT around 315 mT EPR field
        nPoints=128,
    )
    opt = Options(GridSize=31, GridSymmetry='Ci', Verbosity=0)

    freq_range = (8.0, 19.0)   # MHz (MATLAB Experiment.Range)

    # Vary ENDOR linewidth: narrow to broad
    lw_values = [0.05, 0.2, 0.5, 1.0]   # MHz FWHM
    results = []
    for lw in lw_values:
        freq, spc = salt(Sys, exp, opt,
                         freq_range=freq_range, n_points=512, lw_mhz=lw)
        results.append((freq.numpy(), spc.numpy(), lw))
    return results


def _plot(results):
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(results)))
    fig, ax = plt.subplots(figsize=(8, 5))
    for (freq, spc, lw), col in zip(results, colors):
        pk = np.abs(spc).max()
        y = spc / pk if pk > 0 else spc
        ax.plot(freq, y, color=col, lw=1.5, label=f'lw = {lw:.2f} MHz')
    ax.set_xlabel('ENDOR frequency (MHz)')
    ax.set_ylabel('Intensity (norm.)')
    ax.set_title('¹H ENDOR: effect of ENDOR linewidth broadening')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(result)
