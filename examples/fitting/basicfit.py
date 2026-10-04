"""Basic Spectral Fitting
Generates a noisy S=1/2 + ¹H powder spectrum and fits g_x and A_x
using esfit() with the Nelder-Mead simplex optimizer.

EasySpin equivalent: examples/fitting/basicfit.m
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
    # True spin system — used to generate synthetic data
    sys_true = SpinSystem(
        S=[0.5],
        g=[[2.00, 2.10, 2.20]],
        Nucs=['1H'],
        A=[[120.0, 50.0, 78.0]],   # MHz
        lw=[1.0, 0.0],
    )
    exp = Experiment(mwFreq=10.0, Range=[300, 380], nPoints=1024, Harmonic=1)
    opt = Options(GridSize=31, GridSymmetry='C2h', Verbosity=0)

    B, spc_clean = pepper(sys_true, exp, opt)
    spc_exp = addnoise(spc_clean.numpy(), 150, 'n')
    B_np = B.numpy()

    # Starting parameters: [g_x, A_x] — slightly off from true values
    p0 = np.array([1.98, 100.0])
    vary = np.array([0.10, 30.0])

    def model(params):
        gx, Ax = params
        s = SpinSystem(
            S=[0.5],
            g=[[gx, 2.10, 2.20]],
            Nucs=['1H'],
            A=[[Ax, 50.0, 78.0]],
            lw=[1.0, 0.0],
        )
        _, y = pepper(s, exp, opt)
        return y.numpy()

    fit_opt = FitOptions(method='simplex', max_iter=500, verbosity=1)
    result = esfit(spc_exp, model, p0, vary, options=fit_opt)
    return B_np, spc_exp, result


def _plot(B_np, spc_exp, result):
    p_fit = result.pfit
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(B_np, spc_exp, 'k', lw=0.8, label='Data (noisy)')
    axes[0].plot(B_np, result.fit, 'r', lw=1.5,
                 label=f'Fit: g$_x$={p_fit[0]:.4f}, A$_x$={p_fit[1]:.1f} MHz')
    axes[0].set_ylabel('Intensity (arb. u.)')
    axes[0].set_title('Basic spectral fit: S=1/2 + ¹H')
    axes[0].legend()
    axes[1].plot(B_np, spc_exp - result.fit, 'b', lw=0.8)
    axes[1].axhline(0, color='k', lw=0.5)
    axes[1].set_xlabel('Magnetic field (mT)')
    axes[1].set_ylabel('Residual')
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
