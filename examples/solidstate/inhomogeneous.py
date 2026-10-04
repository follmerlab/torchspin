"""Custom Inhomogeneous gx Distribution
S=1/2 axial system with a Gaussian distribution of gx values summed to give a broadened spectrum.

EasySpin equivalent: examples/solidstate/inhomogeneous.m
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchspin import SpinSystem, Experiment, Options, pepper, gaussian


def run():
    N = 100          # number of gx values
    gx0 = 2.0        # central gx value
    gx_lw = 0.03     # FWHM of the distribution
    gx_vals = gx0 + np.linspace(-1, 1, N) * gx_lw * 2.5
    # gaussian(x, x0, fwhm) -> weights
    weights = gaussian(gx_vals, gx0, gx_lw)[0]  # absorption component
    weights = weights / weights.sum()

    exp = Experiment(mwFreq=9.5, Range=[300, 360], nPoints=1024, Harmonic=1)
    opt = Options(GridSize=[19, 4], Verbosity=0)

    spec_total = None
    B_ref = None
    for k, (gx, w) in enumerate(zip(gx_vals, weights)):
        Sys = SpinSystem(
            S=[0.5],
            g=[[gx, 2.0, 2.1]],
            lw=[0.8, 0.0],
        )
        B, spc = pepper(Sys, exp, opt)
        if B_ref is None:
            B_ref = B.numpy()
            spec_total = np.zeros_like(B_ref)
        spec_total += w * spc.numpy()

    return B_ref, spec_total, gx0, gx_lw


def _plot(B_np, spec_total, gx0, gx_lw):
    fig, ax = plt.subplots(figsize=(9, 4))
    y = spec_total / np.abs(spec_total).max() if np.abs(spec_total).max() > 0 else spec_total
    ax.plot(B_np, y, 'k', lw=1.5)
    ax.set_xlabel('Magnetic field (mT)')
    ax.set_ylabel('Intensity (norm.)')
    ax.set_title(f'Gaussian distribution of gx (gx0={gx0}, FWHM={gx_lw})')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
