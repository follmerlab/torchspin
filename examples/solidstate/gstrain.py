"""Effect of g-Strain at Various Microwave Frequencies
g-strain broadening scales with frequency; demonstrated at S, X, Q, W, and 350 GHz bands.

EasySpin equivalent: examples/solidstate/gstrain.m
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchspin import SpinSystem, Experiment, Options, pepper


def run():
    g_tensor = [2.0104, 2.0074, 2.0026]
    gStrain = [0.001, 0.0008, 0.0005]

    freqs  = [3.0,     9.5,      35.0,       95.0,        350.0]
    ranges = [[102, 112], [334, 344], [1240, 1255], [3372, 3395], [12425, 12505]]
    labels = ['S-band (3 GHz)', 'X-band (9.5 GHz)', 'Q-band (35 GHz)',
              'W-band (95 GHz)', '350 GHz']

    opt = Options(GridSize=50, GridSymmetry='Ci', Verbosity=0)

    results = []
    for freq, rng in zip(freqs, ranges):
        exp = Experiment(mwFreq=freq, Range=rng, nPoints=512, Harmonic=0)
        sys_no = SpinSystem(S=[0.5], g=[g_tensor])
        B, spc_no = pepper(sys_no, exp, opt)
        sys_gs = SpinSystem(S=[0.5], g=[g_tensor], gStrain=gStrain)
        _, spc_gs = pepper(sys_gs, exp, opt)
        results.append((B.numpy(), spc_no.numpy(), spc_gs.numpy()))

    return results, labels


def _plot(results, labels):
    def norm(s):
        pk = np.abs(s).max()
        return s / pk if pk > 0 else s

    fig, axes = plt.subplots(len(results), 1, figsize=(8, 10))
    for i, ((B_np, spc_no, spc_gs), label) in enumerate(zip(results, labels)):
        axes[i].plot(B_np, norm(spc_no), 'r', lw=1.2, label='no gStrain')
        axes[i].plot(B_np, norm(spc_gs), 'k', lw=1.2, label='gStrain')
        axes[i].set_ylabel('I (norm.)')
        axes[i].text(0.02, 0.75, label, transform=axes[i].transAxes, fontsize=8)
        if i == 0:
            axes[i].legend(fontsize=8)
    axes[-1].set_xlabel('Magnetic field (mT)')
    fig.suptitle('g-Strain effect at increasing microwave frequencies')
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
