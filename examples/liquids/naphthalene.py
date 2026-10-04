"""Naphthalene Radical Anion and Cation Solution EPR
Fast-motion solution EPR of naphthalene radical (anion and cation) with
two groups of 4 equivalent protons; coupling constants from Gerson Table 8.8.

EasySpin equivalent: examples/liquids/naphthalene.m
"""
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchspin import SpinSystem, Experiment, garlic, unitconvert


def run():
    exp = Experiment(mwFreq=9.5, Range=[337, 342], nPoints=8192, Harmonic=1)

    # Hyperfine values in mT (from Gerson Table 8.8), converted to MHz
    A_anion_mT  = [-0.495, -0.183]   # two groups of 4H
    A_cation_mT = [-0.587, -0.167]

    A_anion_MHz  = unitconvert(A_anion_mT,  'mT->MHz')
    A_cation_MHz = unitconvert(A_cation_mT, 'mT->MHz')

    # Each group has 4 equivalent protons — list individually
    # (garlic requires n=1 per nucleus in fast-motion mode)
    def _naph_sys(A_MHz):
        return SpinSystem(
            S=[0.5],
            g=[[2.0, 2.0, 2.0]],
            Nucs=['1H', '1H', '1H', '1H', '1H', '1H', '1H', '1H'],
            A=[
                [A_MHz[0]]*3, [A_MHz[0]]*3, [A_MHz[0]]*3, [A_MHz[0]]*3,
                [A_MHz[1]]*3, [A_MHz[1]]*3, [A_MHz[1]]*3, [A_MHz[1]]*3,
            ],
            lw=[0.0, 0.01],
        )

    sys_anion  = _naph_sys(A_anion_MHz)
    sys_cation = _naph_sys(A_cation_MHz)

    B, spec_anion  = garlic(sys_anion,  exp)
    _,  spec_cation = garlic(sys_cation, exp)

    return B, spec_anion, spec_cation


def _plot(B, spec_anion, spec_cation):
    B_np = B.numpy()
    ya = spec_anion.numpy();  ya /= np.abs(ya).max()
    yc = spec_cation.numpy(); yc /= np.abs(yc).max()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(B_np, ya, 'b', lw=0.8, label='radical anion')
    ax.plot(B_np, yc, 'r', lw=0.8, label='radical cation')
    ax.set_xlabel('Magnetic field (mT)')
    ax.set_ylabel('Intensity (norm.)')
    ax.set_title('Naphthalene radical anion/cation — 2×4¹H, X-band')
    ax.set_yticks([])
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
