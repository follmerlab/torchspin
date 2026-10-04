"""High-Spin Fe(III) X-Band CW EPR
Full S=5/2 treatment with ZFS versus effective S=1/2 approximation at 10 K.

EasySpin equivalent: examples/solidstate/iron_highspin.m
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchspin import SpinSystem, Experiment, Options, pepper

# 1 cm^-1 = 29979.2458 MHz
CM1_TO_MHZ = 29979.2458


def run():
    D_cm = 10.0
    E_cm = 0.5
    D_MHz = D_cm * CM1_TO_MHZ
    E_MHz = E_cm * CM1_TO_MHZ

    # Convert [D, E] -> diagonal ZFS: Dzz = 2D/3, Dxx = -D/3+E, Dyy = -D/3-E
    Dxx = -D_MHz / 3 + E_MHz
    Dyy = -D_MHz / 3 - E_MHz
    Dzz = 2 * D_MHz / 3

    fe_full = SpinSystem(
        S=[2.5],
        g=[[2.0, 2.0, 2.0]],
        D=[[Dxx, Dyy, Dzz]],
        lw=[5.0, 0.0],
    )
    fe_eff = SpinSystem(
        S=[0.5],
        g=[[7.13, 4.78, 1.919]],
        lw=[5.0, 0.0],
    )

    exp = Experiment(
        mwFreq=9.5,
        Range=[0, 700],
        nPoints=5000,
        Harmonic=0,
        Temperature=10.0,
    )
    opt = Options(GridSize=31, GridSymmetry='Ci', Verbosity=0)

    B, spc1 = pepper(fe_full, exp, opt)
    _, spc2 = pepper(fe_eff, exp, opt)

    return B.numpy(), spc1.numpy(), spc2.numpy(), D_cm, E_cm


def _plot(B_np, spc1, spc2, D_cm, E_cm):
    def norm(s):
        pk = np.abs(s).max()
        return s / pk if pk > 0 else s

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(B_np, norm(spc1), 'b', lw=1.5,
            label=f'S=5/2  (D={D_cm} cm\u207b\u00b9, E={E_cm} cm\u207b\u00b9)')
    ax.plot(B_np, norm(spc2), 'r--', lw=1.5,
            label='S_eff=1/2  (g=[7.13, 4.78, 1.919])')
    ax.set_xlabel('Magnetic field (mT)')
    ax.set_ylabel('Intensity (norm.)')
    ax.set_title('High-spin Fe(III), X-band (9.5 GHz), 10 K')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
