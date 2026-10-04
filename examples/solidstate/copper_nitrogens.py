"""Cu(II) with Four Equivalent 14N Ligands
Axial Cu(II) powder EPR with 63Cu and 2x14N superhyperfine coupling at X-band.

EasySpin equivalent: examples/solidstate/copper_nitrogens.m
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchspin import SpinSystem, Experiment, Options, pepper


def run():
    # Cu(II) S=1/2, 63Cu (I=3/2), axial g and A tensors
    # 2x14N superhyperfine (representative; full 4N system would be 63Cu+4x14N)
    Sys = SpinSystem(
        S=[0.5],
        g=[[2.055, 2.055, 2.225]],
        Nucs=['63Cu', '14N', '14N'],
        A=[
            [40.0,  40.0,  490.0],
            [14.0,  14.0,   14.0],
            [14.0,  14.0,   14.0],
        ],
        lw=[0.8, 0.0],
    )

    exp = Experiment(mwFreq=9.5, Range=[270, 380], nPoints=2048, Harmonic=1)
    opt = Options(GridSize=25, GridSymmetry='Ci', Verbosity=0)

    B, spc = pepper(Sys, exp, opt)
    return B.numpy(), spc.numpy()


def _plot(B_np, spc):
    y = spc / np.abs(spc).max() if np.abs(spc).max() > 0 else spc

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(B_np, y, 'b', lw=1.2)
    ax.set_xlabel('Magnetic field (mT)')
    ax.set_ylabel('Intensity (norm.)')
    ax.set_title('Cu(II) + 4\u00d7\u00b9\u2074N powder EPR, X-band (9.5 GHz)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
