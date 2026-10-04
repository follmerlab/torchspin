"""Isotope Mixture Solution EPR
Compares solution EPR spectra for a radical containing a carbon and boron
nucleus, using pure spin-active isotopes (13C + 11B) vs spin-inactive carbon
(12C, I=0) with pure 11B. Illustrates how isotopic substitution collapses
hyperfine structure.

EasySpin equivalent: examples/liquids/isotopemix.m
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
    # MATLAB: Sys.A = [12, 50] MHz, Sys.n = [1, 1]
    # Case 1: 12C (I=0, NMR-silent) + 11B (I=3/2) — only boron splits the spectrum
    # Case 2: 13C (I=1/2) + 11B (I=3/2) — both nuclei split the spectrum
    exp = Experiment(mwFreq=9.5, Range=[330, 348], nPoints=10000, Harmonic=1)

    # 12C has I=0; not NMR-active — no hyperfine splitting from carbon
    # Use only 11B for case 1 (pure spin-inactive C)
    sys_12C_11B = SpinSystem(
        S=[0.5],
        g=[[2.0, 2.0, 2.0]],
        Nucs=['11B'],                       # 12C (I=0) omitted — no HF
        A=[[50.0, 50.0, 50.0]],            # only boron coupling, MHz
        lw=[0.0, 0.05],
    )

    # 13C (I=1/2) + 11B (I=3/2): both nuclei contribute hyperfine splitting
    sys_13C_11B = SpinSystem(
        S=[0.5],
        g=[[2.0, 2.0, 2.0]],
        Nucs=['13C', '11B'],
        A=[[12.0, 12.0, 12.0],
           [50.0, 50.0, 50.0]],
        lw=[0.0, 0.05],
    )

    B, spec_12C = garlic(sys_12C_11B, exp)
    _, spec_13C = garlic(sys_13C_11B, exp)

    return B.numpy(), spec_12C.numpy(), spec_13C.numpy()


def _plot(B_np, spec_12C, spec_13C):
    def norm(y):
        pk = np.abs(y).max()
        return y / pk if pk > 0 else y

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(B_np, norm(spec_12C), 'b',   lw=0.9, label='$^{12}$C (I=0) + $^{11}$B')
    ax.plot(B_np, norm(spec_13C), 'r--', lw=0.9, label='$^{13}$C (I=1/2) + $^{11}$B')
    ax.set_xlabel('Magnetic field (mT)')
    ax.set_ylabel('Intensity (norm.)')
    ax.set_title('Isotope effect: $^{12}$C vs $^{13}$C with $^{11}$B — X-band solution EPR')
    ax.set_yticks([])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
