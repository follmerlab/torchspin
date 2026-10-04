"""Two-Stage Global/Local Fitting
Fits a rhombic g-tensor powder EPR spectrum (95 GHz) using a two-stage
strategy: (1) coarse Monte Carlo global search to escape local minima,
then (2) local Nelder-Mead refinement from the best global solution.

Note: EasySpin uses a genetic algorithm for the global stage. torchspin
provides 'montecarlo' and 'simplex' methods; we use 'montecarlo' for the
broad search and 'simplex' for local refinement.

EasySpin equivalent: examples/fitting/globallocal.m
"""
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchspin import SpinSystem, addnoise
from torchspin.pepper import pepper
from torchspin.experiment import Experiment, Options
from torchspin.esfit import esfit, FitOptions


def run():
    # True parameters (from MATLAB globallocal.m)
    g_true = [2.01, 2.04, 2.07]
    sys_true = SpinSystem(
        S=[0.5],
        Nucs=['1H'],
        g=[g_true],
        A=[[150.0, 90.0, 130.0]],   # MHz
        lw=[2.0, 0.0],
    )

    # W-band (95 GHz), center 3330 mT, sweep 240 mT
    B_center = 3330.0
    half = 120.0
    exp = Experiment(
        mwFreq=95.0,
        Range=[B_center - half, B_center + half],
        nPoints=1024,
        Harmonic=1,
    )
    opt = Options(GridSize=31, GridSymmetry='auto', Verbosity=0)

    B, spc_clean = pepper(sys_true, exp, opt)
    spc_exp = addnoise(spc_clean.numpy(), 200, 'n')
    B_np = B.numpy()

    rng = np.random.default_rng(7)
    g_start = np.array(g_true) + rng.random(3) * 0.005
    p0   = g_start
    vary = np.array([0.02, 0.02, 0.02])

    def model(params):
        gx, gy, gz = params
        s = SpinSystem(
            S=[0.5],
            Nucs=['1H'],
            g=[[gx, gy, gz]],
            A=[[150.0, 90.0, 130.0]],
            lw=[2.0, 0.0],
        )
        _, y = pepper(s, exp, opt)
        return y.numpy()

    # Stage 1: coarse Monte Carlo global search
    print("Stage 1: Monte Carlo global search...")
    fit_opt_global = FitOptions(
        method='montecarlo', max_iter=300, mc_samples=300,
        verbosity=1, compute_uncertainties=False,
    )
    result_global = esfit(spc_exp, model, p0, vary, options=fit_opt_global)

    # Stage 2: local simplex refinement from best global solution
    print("Stage 2: Simplex local refinement...")
    fit_opt_local = FitOptions(
        method='simplex', max_iter=500, verbosity=1,
    )
    result_local = esfit(spc_exp, model, result_global.pfit,
                         vary * 0.3, options=fit_opt_local)

    return B_np, spc_exp, result_global, result_local, g_true


def _plot(B_np, spc_exp, result_global, result_local, g_true):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    p_glob = result_global.pfit
    p_loc  = result_local.pfit

    axes[0].plot(B_np, spc_exp, 'k', lw=0.6, alpha=0.7, label='Data')
    axes[0].plot(B_np, result_global.fit, 'g', lw=1.2, ls='--',
                 label=f'Global: g=[{p_glob[0]:.4f},{p_glob[1]:.4f},{p_glob[2]:.4f}]')
    axes[0].plot(B_np, result_local.fit, 'r', lw=1.5,
                 label=f'Local:  g=[{p_loc[0]:.4f},{p_loc[1]:.4f},{p_loc[2]:.4f}]')
    axes[0].set_ylabel('Intensity (arb. u.)')
    axes[0].set_title(f'Two-stage fit: true g = {g_true}')
    axes[0].legend(fontsize=8)

    axes[1].plot(B_np, spc_exp - result_local.fit, 'b', lw=0.8)
    axes[1].axhline(0, color='k', lw=0.5)
    axes[1].set_xlabel('Magnetic field (mT)')
    axes[1].set_ylabel('Residual (local)')

    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
