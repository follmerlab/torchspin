"""Cu-EDTA Two-Component Spectrum
Two copper-EDTA complexes with different g/A tensors simulated in one pepper call.

EasySpin equivalent: examples/solidstate/cuedta.m
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchspin import SpinSystem, Experiment, Options, pepper


def run():
    # Component 1: CuEDTA1.g = [2.04 2.321], A = [15 136]*2.8 MHz
    CuEDTA1 = SpinSystem(
        S=[0.5],
        g=[[2.04, 2.04, 2.321]],   # axial: g_perp=2.04, g_par=2.321
        Nucs=['63Cu'],    # natural Cu ≈ 69% 63Cu; torchspin requires explicit isotope
        A=[[15 * 2.8, 15 * 2.8, 136 * 2.8]],   # MHz
        lw=[2.0, 0.0],
        weight=1.0,
    )

    # Component 2: CuEDTA2.g = [2.032 2.288], A = [15 144]*2.8 MHz
    CuEDTA2 = SpinSystem(
        S=[0.5],
        g=[[2.032, 2.032, 2.288]],
        Nucs=['63Cu'],    # natural Cu ≈ 69% 63Cu; torchspin requires explicit isotope
        A=[[15 * 2.8, 15 * 2.8, 144 * 2.8]],   # MHz
        lw=[2.0, 0.0],
        weight=0.6,
    )

    exp = Experiment(mwFreq=9.5, Range=[250, 360], nPoints=1024, Harmonic=1)
    opt = Options(GridSize=[19, 4], Verbosity=0)

    # Pass list of two spin systems -> pepper sums weighted components
    B, spc = pepper([CuEDTA1, CuEDTA2], exp, opt)

    # Also simulate each component separately for display
    _, spc1 = pepper(CuEDTA1, exp, opt)
    _, spc2 = pepper(CuEDTA2, exp, opt)

    return B.numpy(), spc.numpy(), spc1.numpy(), spc2.numpy()


def _plot(B_np, spc, spc1, spc2):
    def norm(s):
        pk = np.abs(s).max()
        return s / pk if pk > 0 else s

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(B_np, norm(spc1), 'b', lw=1.2)
    axes[0].set_title('Component 1: Cu-EDTA (g\u22a5=2.04, g\u2225=2.321)')
    axes[0].set_ylabel('Intensity (norm.)')

    axes[1].plot(B_np, norm(spc2), 'r', lw=1.2)
    axes[1].set_title('Component 2: Cu-EDTA (g\u22a5=2.032, g\u2225=2.288), weight=0.6')
    axes[1].set_ylabel('Intensity (norm.)')

    axes[2].plot(B_np, norm(spc), 'k', lw=1.5)
    axes[2].set_title('Two-component mixture')
    axes[2].set_xlabel('Magnetic field (mT)')
    axes[2].set_ylabel('Intensity (norm.)')

    plt.suptitle('Cu-EDTA two-component powder EPR, X-band (9.5 GHz)', fontsize=12)
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
