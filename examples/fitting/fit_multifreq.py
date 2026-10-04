"""Global Fit of Multi-Frequency Slow-Motion EPR Spectra
Generates synthetic slow-motion EPR data at X-band (9.4 GHz) and Q-band
(34 GHz) for a nitroxide, concatenates both spectra into a single residual
vector, and fits the rotational diffusion coefficient (logDiff) and linewidths
with esfit() using a custom model wrapper.

EasySpin equivalent: examples/fitting/fit_multifreq.m
"""
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchspin import SpinSystem, addnoise
from torchspin.chili import chili, ChiliOptions
from torchspin.experiment import Experiment
from torchspin.esfit import esfit, FitOptions
from torchspin.utils import unitconvert


def run():
    # Nitroxide parameters (MATLAB fit_multifreq.m)
    A_mT = np.array([0.5, 3.6])          # [Ax, Az] in mT
    A_MHz = unitconvert(A_mT, 'mT->MHz')  # convert to MHz

    g_vals   = [2.009, 2.006, 2.002]
    logDiff_true = 8.0   # log10(D_rot / s^-1)
    lw_X_true = [0.0, 0.1]   # mT
    lw_Q_true = [0.0, 0.1]

    # X-band: center 335.6 mT, sweep 12 mT
    exp_X = Experiment(
        mwFreq=9.4,
        Range=[335.6 - 6.0, 335.6 + 6.0],
        nPoints=1024,
        Harmonic=1,
    )
    # Q-band: center 1214.0 mT, sweep 20 mT
    exp_Q = Experiment(
        mwFreq=34.0,
        Range=[1214.0 - 10.0, 1214.0 + 10.0],
        nPoints=1024,
        Harmonic=1,
    )
    opt = ChiliOptions(Verbosity=0)

    def _make_sys(logDiff, lw_x, lw_q, for_x=True):
        lw = lw_x if for_x else lw_q
        return SpinSystem(
            S=[0.5],
            g=[g_vals],
            Nucs=['14N'],
            A=[[A_MHz[0], A_MHz[0], A_MHz[1]]],
            logDiff=logDiff,
            lw=lw,
        )

    # Simulate true spectra
    sys_X = _make_sys(logDiff_true, lw_X_true, lw_Q_true, for_x=True)
    sys_Q = _make_sys(logDiff_true, lw_X_true, lw_Q_true, for_x=False)

    B_X, spc_X_clean = chili(sys_X, exp_X, opt)
    B_Q, spc_Q_clean = chili(sys_Q, exp_Q, opt)

    SNR = 20
    spc_X_exp = addnoise(np.asarray(spc_X_clean), SNR, 'n')
    spc_Q_exp = addnoise(np.asarray(spc_Q_clean), SNR, 'n')

    # Ensure numpy
    B_X_np = np.asarray(B_X)
    B_Q_np = np.asarray(B_Q)
    spc_X_exp = np.asarray(spc_X_exp)
    spc_Q_exp = np.asarray(spc_Q_exp)

    # Concatenated experimental data vector
    data_concat = np.concatenate([spc_X_exp, spc_Q_exp])

    # Model wrapper: params = [logDiff, lw_x_lor, lw_q_lor]
    def model(params):
        logD, lwX1, lwQ1 = params
        sys_x = SpinSystem(
            S=[0.5], g=[g_vals], Nucs=['14N'],
            A=[[A_MHz[0], A_MHz[0], A_MHz[1]]],
            logDiff=logD, lw=[0.0, max(lwX1, 0.0)],
        )
        sys_q = SpinSystem(
            S=[0.5], g=[g_vals], Nucs=['14N'],
            A=[[A_MHz[0], A_MHz[0], A_MHz[1]]],
            logDiff=logD, lw=[0.0, max(lwQ1, 0.0)],
        )
        _, y_x = chili(sys_x, exp_X, opt)
        _, y_q = chili(sys_q, exp_Q, opt)
        return np.concatenate([np.asarray(y_x), np.asarray(y_q)])

    p0   = np.array([logDiff_true + 0.3, 0.08, 0.08])
    vary = np.array([0.5, 0.05, 0.05])

    fit_opt = FitOptions(method='simplex', max_iter=300, verbosity=1)
    result = esfit(data_concat, model, p0, vary, options=fit_opt)

    return B_X_np, B_Q_np, spc_X_exp, spc_Q_exp, result, logDiff_true


def _plot(B_X, B_Q, spc_X_exp, spc_Q_exp, result, logDiff_true):
    n = len(spc_X_exp)
    fit_X = result.fit[:n]
    fit_Q = result.fit[n:]
    p_fit = result.pfit

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=False)
    axes[0].plot(B_X, spc_X_exp, 'k', lw=0.7, alpha=0.7, label='Data X-band')
    axes[0].plot(B_X, fit_X, 'r', lw=1.5, label='Fit')
    axes[0].set_ylabel('Intensity')
    axes[0].set_title(f'X-band (9.4 GHz)  —  logDiff fit: {p_fit[0]:.3f} (true: {logDiff_true})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(B_Q, spc_Q_exp, 'k', lw=0.7, alpha=0.7, label='Data Q-band')
    axes[1].plot(B_Q, fit_Q, 'b', lw=1.5, label='Fit')
    axes[1].set_ylabel('Intensity')
    axes[1].set_xlabel('Magnetic field (mT)')
    axes[1].set_title('Q-band (34 GHz)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
