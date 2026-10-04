"""Chromium(III) Q-Band EPR
S=3/2 Cr(III) at 34 GHz; compare no nucleus / pure 53Cr / natural Cr mixture.

EasySpin equivalent: examples/solidstate/chromium_iii.m
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchspin import SpinSystem, Experiment, Options, pepper


def run():
    exp = Experiment(
        mwFreq=34.0,
        Range=[900, 1550],
        nPoints=4096,
        Harmonic=1,
    )
    # Opt.GridSize = [46 1] in MATLAB -> plain integer 46 (no interpolation)
    opt = Options(GridSize=46, GridSymmetry='Ci', Verbosity=0)

    # System without magnetic Cr nucleus
    Sys0 = SpinSystem(
        S=[1.5],
        g=[[1.990, 1.990, 1.990]],
        D=[3000.0, 750.0, -3750.0],   # D=3000 MHz, E=750 MHz -> traceless [Dxx,Dyy,Dzz]
        lw=[1.0, 0.0],
    )

    # EasySpin D=[3000 750] means D-value=3000 MHz, E-value=750 MHz
    # Dxx = -D/3+E, Dyy = -D/3-E, Dzz = 2D/3
    D_val = 3000.0
    E_val = 750.0
    Dxx = -D_val / 3 + E_val
    Dyy = -D_val / 3 - E_val
    Dzz = 2 * D_val / 3

    Sys0 = SpinSystem(
        S=[1.5],
        g=[[1.990, 1.990, 1.990]],
        D=[[Dxx, Dyy, Dzz]],
        lw=[1.0, 0.0],
    )

    B, spec0 = pepper(Sys0, exp, opt)

    # System with pure 53Cr nucleus (I=3/2, A=[360, 720] MHz axial)
    Sys_53 = SpinSystem(
        S=[1.5],
        g=[[1.990, 1.990, 1.990]],
        D=[[Dxx, Dyy, Dzz]],
        Nucs=['53Cr'],
        A=[[360.0, 360.0, 720.0]],   # [1 2]*360 MHz
        lw=[1.0, 0.0],
    )
    _, spec_53 = pepper(Sys_53, exp, opt)

    # Natural-abundance Cr mixture:
    #   52Cr (83.79%, I=0), 50Cr (4.35%, I=0), 54Cr (2.37%, I=0), 53Cr (9.50%, I=3/2)
    # I=0 isotopes contribute no hyperfine → equivalent to Sys0
    # Weighted sum: 90.5% * spec0 + 9.5% * spec_53
    frac_53 = 0.09501
    spec_na = ((1.0 - frac_53) * spec0 + frac_53 * spec_53).numpy()

    return B.numpy(), spec0.numpy(), spec_53.numpy(), spec_na


def _plot(B_np, spec0, spec_53, spec_na):
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    def norm(s):
        pk = np.abs(s).max()
        return s / pk if pk > 0 else s

    axes[0].plot(B_np, norm(spec0), 'k', lw=1.2)
    axes[0].set_title('Without \u00b3\u00b3Cr nucleus')
    axes[0].set_ylabel('Intensity (norm.)')

    axes[1].plot(B_np, norm(spec_53), 'b', lw=1.2)
    axes[1].set_title('With \u2075\u00b3Cr nucleus (isotopically pure)')
    axes[1].set_ylabel('Intensity (norm.)')

    axes[2].plot(B_np, norm(spec_na), 'r', lw=1.2)
    axes[2].set_title('Natural isotope mixture (Cr)')
    axes[2].set_xlabel('Magnetic field (mT)')
    axes[2].set_ylabel('Intensity (norm.)')

    plt.suptitle('Cr(III) S=3/2, Q-band (34 GHz)', fontsize=12)
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
