"""Fitting Pentacene Anion Radical Hyperfine Constants
Generates a noisy solution-EPR spectrum of the pentacene radical anion with
four sets of equivalent protons and recovers all four hyperfine constants
simultaneously using esfit() with the Nelder-Mead simplex optimizer.

EasySpin equivalent: examples/fitting/pentacenefit.m
"""
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchspin import SpinSystem, addnoise
from torchspin.garlic import garlic
from torchspin.experiment import Experiment, Options
from torchspin.esfit import esfit, FitOptions


def run():
    # True parameters (MATLAB pentacenefit.m)
    # 4 groups of equivalent 1H: n=[2,4,4,4], A=[11.9,8.5,2.6,2.4] MHz
    A_true = [11.9, 8.5, 2.6, 2.4]

    # n=[2,4,4,4] -> expand to 14 individual protons (torchspin lists each explicitly)
    ns = [2, 4, 4, 4]
    nucs_true = sum([['1H'] * n for n in ns], [])
    A_list_true = sum([[[a, a, a]] * n for a, n in zip(A_true, ns)], [])
    sys_true = SpinSystem(
        S=[0.5],
        g=[[2.0, 2.0, 2.0]],
        Nucs=nucs_true,
        A=A_list_true,
        lw=[0.0, 0.01],
    )

    # Center 339.4 mT, sweep 5 mT → Range [337, 342] roughly
    B_center = 339.4
    half = 2.5
    exp = Experiment(
        mwFreq=9.5,
        Range=[B_center - half, B_center + half],
        nPoints=3000,
        Harmonic=1,
    )
    opt = Options(Verbosity=0)

    B, spc_clean = garlic(sys_true, exp, opt)
    spc_clean_np = spc_clean.numpy() if hasattr(spc_clean, 'numpy') else spc_clean
    spc_exp = addnoise(spc_clean_np, 150, 'n')
    B_np = B.numpy() if hasattr(B, 'numpy') else B

    # Starting parameters: slightly perturbed A values
    p0   = np.array([12.0, 8.5, 2.6, 2.4])
    vary = np.array([2.0,  2.0, 1.0, 1.0])

    def model(params):
        A1, A2, A3, A4 = params
        A_vals = [A1, A2, A3, A4]
        nucs_m = sum([['1H'] * n for n in ns], [])
        A_list_m = sum([[[a, a, a]] * n for a, n in zip(A_vals, ns)], [])
        s = SpinSystem(
            S=[0.5],
            g=[[2.0, 2.0, 2.0]],
            Nucs=nucs_m,
            A=A_list_m,
            lw=[0.0, 0.01],
        )
        _, y = garlic(s, exp, opt)
        return y.numpy() if hasattr(y, 'numpy') else y

    fit_opt = FitOptions(method='simplex', max_iter=1000, verbosity=1)
    result = esfit(spc_exp, model, p0, vary, options=fit_opt)
    return B_np, spc_exp, result, A_true


def _plot(B_np, spc_exp, result, A_true):
    p_fit = result.pfit
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(B_np, spc_exp, 'k', lw=0.6, label='Data (noisy)', alpha=0.7)
    axes[0].plot(B_np, result.fit, 'r', lw=1.5, label='Fit')
    axes[0].set_ylabel('Intensity (arb. u.)')
    axes[0].set_title('Pentacene anion radical: hyperfine fit (4 × ¹H groups)')
    lines = [f'A{i+1}: fit={p_fit[i]:.2f}, true={A_true[i]:.1f} MHz' for i in range(4)]
    axes[0].legend([axes[0].lines[0], axes[0].lines[1]] , ['Data', 'Fit'])
    axes[0].text(0.01, 0.97, '\n'.join(lines), transform=axes[0].transAxes,
                 va='top', fontsize=8, family='monospace')
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
