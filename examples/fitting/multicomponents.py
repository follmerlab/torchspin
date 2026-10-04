"""Fitting a Multi-Component Spectrum
Generates a noisy two-component S=1/2 powder spectrum and recovers
the g-tensor principal values of each component with esfit().

EasySpin equivalent: examples/fitting/multicomponents.m
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
    # Component 1: dominant (weight=1), axial-like g
    sys1 = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.2]], lw=[1.0, 0.0], weight=1.0)
    # Component 2: minor (weight=0.3), different g
    sys2 = SpinSystem(S=[0.5], g=[[2.1, 2.1, 2.15]], lw=[1.0, 0.0], weight=0.3)

    exp = Experiment(mwFreq=9.5, Range=[280, 370], nPoints=1024, Harmonic=1)
    opt = Options(GridSize=31, GridSymmetry='Ci', Verbosity=0)

    # the multi-component call applies each system's ``weight`` (1.0 and 0.3);
    # summing separately weighted single-component spectra would weight twice
    B, spc_true = pepper([sys1, sys2], exp, opt)
    spc_true = spc_true.numpy()
    spc_noisy = addnoise(spc_true, 40, 'n', rng=np.random.default_rng(0))   # seeded: reproducible example
    B_np = B.numpy()

    # Starting parameters (perturbed): [g1_x(=y), g1_z, g2_x(=y), g2_z]
    p0 = np.array([1.98, 2.22, 2.08, 2.17])
    vary = np.array([0.1, 0.1, 0.1, 0.1])

    def model(params):
        g1x, g1z, g2x, g2z = params
        s1 = SpinSystem(S=[0.5], g=[[g1x, g1x, g1z]], lw=[1.0, 0.0], weight=1.0)
        s2 = SpinSystem(S=[0.5], g=[[g2x, g2x, g2z]], lw=[1.0, 0.0], weight=0.3)
        _, y = pepper([s1, s2], exp, opt)
        return y.numpy()

    fit_opt = FitOptions(method='global', swarm_size=16, max_iter=20, seed=0, verbosity=0)   # swarm + simplex polish: robust from the perturbed start
    result = esfit(spc_noisy, model, p0, vary, options=fit_opt)
    return B_np, spc_noisy, result


def _plot(B_np, spc_noisy, result):
    p_fit = result.pfit
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(B_np, spc_noisy, 'k', lw=0.8, label='Data (noisy)')
    ax1.plot(B_np, result.fit, 'r', lw=1.5, label='Fit')
    ax1.set_ylabel('Intensity (arb. u.)')
    ax1.set_title('Two-component spectrum fit')
    ax1.legend()
    ax2.plot(B_np, spc_noisy - result.fit, 'b', lw=0.8)
    ax2.axhline(0, color='k', lw=0.5)
    ax2.set_xlabel('Magnetic field (mT)')
    ax2.set_ylabel('Residual')
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
