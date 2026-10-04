"""Fitting a Two-Component EPR Spectrum
Generates a noisy two-component S=1/2 powder spectrum and fits the
g-value, linewidth, and weight of the minor component with esfit().

EasySpin equivalent: examples/fitting/twocompfit.m
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
    sys1_true = SpinSystem(S=[0.5], g=[[2.00, 2.00, 2.20]], lw=[1.0, 0.0], weight=0.7)
    sys2_true = SpinSystem(S=[0.5], g=[[1.85, 1.85, 2.05]], lw=[2.0, 0.0], weight=0.3)

    exp = Experiment(mwFreq=10.0, Range=[300, 400], nPoints=512, Harmonic=1)
    opt = Options(GridSize=31, GridSymmetry='auto', Verbosity=0)

    B, spc_true = pepper([sys1_true, sys2_true], exp, opt)
    spc_exp = addnoise(spc_true.numpy(), 50, 'n')
    B_np = B.numpy()

    # Free parameters: [g1_x(=y), lw2, weight2]
    p0 = np.array([1.98, 1.3, 0.6])
    vary = np.array([0.03, 0.9, 0.3])

    def model(params):
        g1x, lw2, w2 = params
        s1 = SpinSystem(S=[0.5], g=[[g1x, g1x, 2.20]], lw=[1.0, 0.0], weight=0.7)
        s2 = SpinSystem(S=[0.5], g=[[1.85, 1.85, 2.05]], lw=[lw2, 0.0], weight=w2)
        _, y = pepper([s1, s2], exp, opt)
        return y.numpy()

    fit_opt = FitOptions(method='simplex', max_iter=400, verbosity=1)
    result = esfit(spc_exp, model, p0, vary, options=fit_opt)
    return B_np, spc_exp, result


def _plot(B_np, spc_exp, result):
    p_fit = result.pfit
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(B_np, spc_exp, 'k', lw=0.8, label='Data (noisy)')
    ax1.plot(B_np, result.fit, 'r', lw=1.5, label='Two-component fit')
    ax1.set_ylabel('Intensity (arb. u.)')
    ax1.set_title('Two-component fit: two S=1/2 species at different g-values')
    ax1.legend()
    ax2.plot(B_np, spc_exp - result.fit, 'b', lw=0.8)
    ax2.axhline(0, color='k', lw=0.5)
    ax2.set_xlabel('Magnetic field (mT)')
    ax2.set_ylabel('Residual')
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
