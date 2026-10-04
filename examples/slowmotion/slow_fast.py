"""Slow-Motion vs Fast-Motion EPR Comparison
Compares the SLE-based chili() spectrum with the fast-motion garlic() result
for the same nitroxide at τ_c = 1 ns. The spectra are overlaid on the same
axis — at 1 ns the two methods should be close but not identical.

EasySpin equivalent: examples/slowmotion/slow_fast.m
"""
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchspin import SpinSystem
from torchspin.chili import chili, ChiliOptions
from torchspin.garlic import garlic
from torchspin.experiment import Experiment, Options
from torchspin.utils import unitconvert


def run():
    # Parameters from MATLAB slow_fast.m (TEMPONE-like nitroxide)
    A_mT = np.array([5.8, 5.8, 30.8]) / 10.0   # Gauss -> mT
    A_MHz = unitconvert(A_mT, 'mT->MHz')

    tcorr = 1e-9   # 1 ns
    lw = 0.03      # mT Lorentzian FWHM

    # Field range: center ≈ 338 mT, sweep 10 mT
    B_center = 338.0
    half_sweep = 5.0
    exp_range = [B_center - half_sweep, B_center + half_sweep]

    # Shared spin system parameters
    g = [2.0088, 2.0061, 2.0027]
    Nucs = ['14N']
    A = [A_MHz.tolist()]

    exp = Experiment(mwFreq=9.5, Range=exp_range, nPoints=1024, Harmonic=1)

    # --- chili: SLE slow-motion ---
    Sys_chili = SpinSystem(
        S=[0.5], g=[g], Nucs=Nucs, A=A,
        tcorr=tcorr, lw=[0.0, lw],
    )
    opt_chili = ChiliOptions(Verbosity=0)
    B_s, spc_s = chili(Sys_chili, exp, opt_chili)

    # --- garlic: fast-motion perturbation ---
    Sys_garlic = SpinSystem(
        S=[0.5], g=[g], Nucs=Nucs, A=A,
        tcorr=tcorr, lw=[0.0, lw],
    )
    opt_garlic = Options(Verbosity=0)
    B_f, spc_f = garlic(Sys_garlic, exp, opt_garlic)

    B_s_np = B_s if isinstance(B_s, np.ndarray) else B_s.numpy()
    B_f_np = B_f if isinstance(B_f, np.ndarray) else B_f.numpy()
    spc_s_np = spc_s if isinstance(spc_s, np.ndarray) else spc_s.numpy()
    spc_f_np = spc_f if isinstance(spc_f, np.ndarray) else spc_f.numpy()

    # Normalize each to max absolute value
    spc_s_norm = spc_s_np / np.abs(spc_s_np).max()
    spc_f_norm = spc_f_np / np.abs(spc_f_np).max()

    return B_s_np, spc_s_norm, B_f_np, spc_f_norm, tcorr


def _plot(B_s, spc_s, B_f, spc_f, tcorr):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(B_s, spc_s, 'k', lw=1.5, label='chili (SLE)')
    ax.plot(B_f, spc_f, 'r', lw=1.5, ls='--', label='garlic (fast-motion)')
    ax.set_xlabel('Magnetic field (mT)')
    ax.set_ylabel('Intensity (norm.)')
    ax.set_title(f'SLE vs fast-motion: τ$_c$ = {tcorr*1e9:.0f} ns')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
