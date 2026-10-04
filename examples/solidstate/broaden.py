"""Line Broadening Models for CW EPR
Three broadening mechanisms compared for an S=1/2 powder spectrum:
HStrain (unresolved hyperfine), lw (global Gaussian), and gStrain.

EasySpin equivalent: examples/solidstate/broaden.m
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchspin import SpinSystem, Experiment, Options, pepper, deriv


def run():
    g = [1.9, 2.01, 2.3]
    exp = Experiment(
        mwFreq=9.5,
        Range=[280, 370],
        nPoints=1024,
        Harmonic=0,
    )
    opt = Options(GridSize=50, GridSymmetry='Ci', Verbosity=0)

    # (1) HStrain: unresolved hyperfine along each g principal axis
    sys1 = SpinSystem(S=[0.5], g=[g], HStrain=[110.0, 40.0, 50.0])
    B, spc1 = pepper(sys1, exp, opt)

    # (2) Global Gaussian convolution (lw in mT)
    sys2 = SpinSystem(S=[0.5], g=[g], lw=[5.0, 0.0])
    _, spc2 = pepper(sys2, exp, opt)

    # (3) g-strain: Gaussian distribution of g principal values
    sys3 = SpinSystem(S=[0.5], g=[g], gStrain=[0.05, 0.01, 0.01])
    _, spc3 = pepper(sys3, exp, opt)

    return B.numpy(), spc1.numpy(), spc2.numpy(), spc3.numpy()


def _plot(B_np, s1, s2, s3):
    def norm(s):
        pk = np.abs(s).max()
        return s / pk if pk > 0 else s

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1.plot(B_np, norm(s1), label='HStrain [110, 40, 50] MHz')
    ax1.plot(B_np, norm(s2), label='lw = 5 mT (Gaussian)')
    ax1.plot(B_np, norm(s3), label='gStrain [0.05, 0.01, 0.01]')
    ax1.set_ylabel('Intensity (norm.)')
    ax1.set_title('Different broadening models — absorption')
    ax1.legend()

    ax2.plot(B_np, norm(deriv(s1)), label='HStrain')
    ax2.plot(B_np, norm(deriv(s2)), label='lw')
    ax2.plot(B_np, norm(deriv(s3)), label='gStrain')
    ax2.set_xlabel('Magnetic field (mT)')
    ax2.set_ylabel('dI/dB (norm.)')
    ax2.set_title('First-derivative spectra')
    ax2.legend()

    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
