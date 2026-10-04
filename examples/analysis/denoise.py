"""Signal Denoising: Four Smoothing Methods Compared
Compares flat moving average, binomial moving average, Savitzky-Golay, and
RC filter applied to a noisy Gaussian signal. Shows that binomial and
Savitzky-Golay preserve amplitude and width better than flat/RC.

EasySpin equivalent: examples/analysis/denoise.m
"""
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchspin import gaussian, addnoise, datasmooth, rcfilt


def run():
    # MATLAB: x = linspace(-1, 1, 801); gaussian(x, 0, 0.2)
    x = np.linspace(-1.0, 1.0, 801)

    # gaussian() returns (ya, yd); use absorption component
    y_pure, _ = gaussian(x, x0=0.0, fwhm=0.2)
    y_pure = y_pure / y_pure.max()

    # Add uniform noise (MATLAB uses rand-0.5, SNR equivalent ~20)
    rng = np.random.default_rng(42)
    noise_amp = 0.1
    y_noisy = y_pure + noise_amp * (rng.random(x.size) - 0.5)

    # MATLAB: m = 50 (half-width of smoothing window)
    m = 50

    y_flat  = datasmooth(y_noisy, m, method='flat')
    y_binom = datasmooth(y_noisy, m, method='binom')
    y_sg    = datasmooth(y_noisy, m, method='savgol', poly_order=2)

    # RC filter: sample_time=1 point, time_constant=m points
    y_rc = rcfilt(y_noisy, sample_time=1.0, time_constant=float(m))

    return x, y_pure, y_noisy, y_flat, y_binom, y_sg, y_rc


def _plot(x, y_pure, y_noisy, y_flat, y_binom, y_sg, y_rc):
    labels  = ['noisy (offset)', 'flat', 'binomial', 'Savitzky-Golay', 'RC filter']
    curves  = [y_noisy + 0.2, y_flat, y_binom, y_sg, y_rc]
    colors  = ['0.6', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    lwidths = [0.8,   1.2,        1.4,          1.4,          1.2]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Reference pure signal
    ax.plot(x, y_pure, 'k--', lw=1.0, label='pure signal', zorder=5)

    for label, y, col, lw in zip(labels, curves, colors, lwidths):
        ax.plot(x, y, color=col, lw=lw, label=label)

    ax.set_xlabel('x')
    ax.set_ylabel('Intensity')
    ax.set_title('Smoothing methods compared: flat, binomial, Savitzky-Golay, RC filter')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
