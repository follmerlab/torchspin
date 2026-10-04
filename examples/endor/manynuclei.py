"""ENDOR with Many Proton Nuclei
Simulates an orientation-selective ENDOR spectrum with 5 × ¹H nuclei,
each having a random hyperfine tensor. Demonstrates that salt() handles
multi-nucleus systems at Q-band fields.

Note: The MATLAB original uses Opt.Method='perturb1' for speed. torchspin
salt() uses matrix diagonalization throughout; for 5 protons the Hilbert
space is 2 × 2^5 = 64 dimensional, which is manageable.

EasySpin equivalent: examples/endor/manynuclei.m
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
    rng = np.random.default_rng(42)   # fixed seed for reproducibility

    # 5 × ¹H with random A tensors (principal values, MHz)
    A_list = (rng.random((5, 3)) * 16.0).tolist()   # 0..16 MHz each component

    Sys = SpinSystem(
        S=[0.5],
        g=[[2.0, 2.1, 2.2]],
        Nucs=['1H', '1H', '1H', '1H', '1H'],
        A=A_list,
    )

    # Q-band experiment: ~94 GHz, B ≈ 3280 mT
    mwFreq = 94.0   # GHz
    B_field = 3280.0  # mT
    exp = Experiment(
        mwFreq=mwFreq,
        Range=[B_field - 15.0, B_field + 15.0],
        nPoints=128,
    )
    opt = Options(GridSize=31, GridSymmetry='Ci', Verbosity=1)

    # ¹H Larmor frequency at B_field (~140 MHz at 3280 mT)
    from torchspin.constants import NMAGN, PLANCK
    gn_1H = 5.58569
    nu_H = gn_1H * NMAGN * (B_field * 1e-3) / PLANCK * 1e-6   # MHz

    freq_range = (nu_H - 10.0, nu_H + 10.0)   # 20 MHz window

    freq, spc = salt(Sys, exp, opt, freq_range=freq_range, n_points=512, lw_mhz=0.1)
    return freq.numpy(), spc.numpy(), nu_H, A_list


def _plot(freq, spc, nu_H, A_list):
    pk = np.abs(spc).max()
    y = spc / pk if pk > 0 else spc
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(freq, y, 'b', lw=1.5)
    ax.axvline(nu_H, color='k', lw=0.8, ls='--', label=f'ν$_H$ = {nu_H:.1f} MHz')
    ax.set_xlabel('ENDOR frequency (MHz)')
    ax.set_ylabel('Intensity (norm.)')
    ax.set_title('Q-band ENDOR: 5 × ¹H with random hyperfine tensors')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
