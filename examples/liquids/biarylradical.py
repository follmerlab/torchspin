"""Bridged Biaryl Cation Radical Solution EPR
Large solution EPR spectrum (9216 resonance lines) from
6-hydrodipyrido[1,2-c:2',1'-e]-imidazole cation radical with 12 nuclei
in 6 groups of 2 (14N + 5×1H); coupling constants in Gauss from literature.

EasySpin equivalent: examples/liquids/biarylradical.m
"""
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchspin import SpinSystem, Experiment, garlic, unitconvert, GFREE


def run():
    # Hyperfine couplings in Gauss (from literature)
    A_Gauss = [+4.34, -2.39, -0.65, -2.81, -0.23, +24.24]
    # Convert Gauss → mT (/10) → MHz with free-electron g
    A_MHz = unitconvert(np.array(A_Gauss) / 10.0, 'mT->MHz', g=GFREE)

    # MATLAB: Sys.n = [2 2 2 2 2 2] — 2 of each nucleus type
    # Expand to explicit list: each of the 6 types appears twice, interleaved
    nucs = ['14N', '1H', '1H', '1H', '1H', '1H',
            '14N', '1H', '1H', '1H', '1H', '1H']
    A_list = [[A_MHz[i % 6]] * 3 for i in range(6)] + [[A_MHz[i % 6]] * 3 for i in range(6)]
    sys = SpinSystem(
        S=[0.5],
        g=[[2.00316, 2.00316, 2.00316]],
        Nucs=nucs,
        A=A_list,
        lwpp=[0.0, 0.006],
    )
    exp = Experiment(mwFreq=9.532, Range=[333, 346], nPoints=8192, Harmonic=1)
    B, spec = garlic(sys, exp)
    return B, spec


def _plot(B, spec):
    y = spec.numpy(); y /= np.abs(y).max()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(B.numpy(), y, 'b', lw=0.5)
    ax.set_xlabel('Magnetic field (mT)')
    ax.set_ylabel('Intensity (norm.)')
    ax.set_title('Biaryl cation radical — 12 nuclei (6×¹⁴N+¹H pairs), 9216 lines')
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
