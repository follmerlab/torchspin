"""CF3 Radical: Three 19F Nuclei with Tilted AFrame
Axial g-tensor, three 19F nuclei with identical principal A values but different orientations.

EasySpin equivalent: examples/solidstate/trifluormethyl.m
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchspin import SpinSystem, Experiment, Options, pepper


def run():
    # Hyperfine principal values (MHz); from Edlund et al, Mol. Phys. 32 (1976) 49
    # A = [80 87 262]*2.8 MHz
    Ax = 80 * 2.8
    Ay = 87 * 2.8
    Az = 262 * 2.8

    # Three 19F nuclei have identical A principal values but different frame orientations.
    # beta = 17.8 deg tilt; AFrame rows = [0, beta, 0], [-120, beta, 0], [120, beta, 0] in deg->rad
    beta = 17.8 * np.pi / 180.0
    deg120 = 120.0 * np.pi / 180.0

    Sys = SpinSystem(
        S=[0.5],
        g=[[2.0042, 2.0042, 2.0024]],   # axial: g_perp=2.0042, g_par=2.0024
        Nucs=['19F', '19F', '19F'],
        A=[
            [Ax, Ay, Az],
            [Ax, Ay, Az],
            [Ax, Ay, Az],
        ],
        # AFrame: shape (nNuclei, 3*nElectrons) = (3, 3) for 1 electron, 3 nuclei
        # Each row = [alpha, beta, gamma] Euler angles in radians
        AFrame=[
            [0.0,        beta, 0.0],
            [-deg120,    beta, 0.0],
            [ deg120,    beta, 0.0],
        ],
        lw=[0.4, 0.0],
    )

    exp = Experiment(mwFreq=9.27, Range=[285, 375], nPoints=1024, Harmonic=1)
    opt = Options(GridSize=[19, 4], Verbosity=0)

    B, spc = pepper(Sys, exp, opt)
    return B.numpy(), spc.numpy()


def _plot(B_np, spc):
    y = spc / np.abs(spc).max() if np.abs(spc).max() > 0 else spc

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(B_np, y, 'k', lw=1.2)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel('Magnetic field (mT)')
    ax.set_ylabel('dI/dB (norm.)')
    ax.set_title('CF\u2083 radical, randomly oriented, 9.27 GHz')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
