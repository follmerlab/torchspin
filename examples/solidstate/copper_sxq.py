"""Cu(II) at S, X, and Q Band
Natural-abundance Cu(II) powder EPR at three microwave frequencies using CenterSweep.

EasySpin equivalent: examples/solidstate/copper_sxq.m
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchspin import SpinSystem, Experiment, Options, pepper


def run():
    # Natural-abundance Cu mixture (63Cu + 65Cu handled by Nucs='Cu')
    SysCu = SpinSystem(
        S=[0.5],
        g=[[2.0, 2.0, 2.2]],
        Nucs=['63Cu'],    # natural Cu ≈ 69% 63Cu; torchspin requires explicit isotope
        A=[[50.0, 50.0, 400.0]],   # for 63Cu, MHz; axial
        lw=[1.0, 0.0],
    )

    opt = Options(GridSize=[19, 4], Verbosity=0)

    # S-band: 2 GHz, center=65 mT, sweep=80 mT
    exp_S = Experiment(mwFreq=2.0, Range=[65.0 - 40.0, 65.0 + 40.0], nPoints=1024, Harmonic=1)
    Bs, specS = pepper(SysCu, exp_S, opt)

    # X-band: 9.5 GHz, center=315 mT, sweep=80 mT
    exp_X = Experiment(mwFreq=9.5, Range=[315.0 - 40.0, 315.0 + 40.0], nPoints=1024, Harmonic=1)
    Bx, specX = pepper(SysCu, exp_X, opt)

    # Q-band: 34 GHz, center=1150 mT, sweep=150 mT
    exp_Q = Experiment(mwFreq=34.0, Range=[1150.0 - 75.0, 1150.0 + 75.0], nPoints=1024, Harmonic=1)
    Bq, specQ = pepper(SysCu, exp_Q, opt)

    return Bs.numpy(), specS.numpy(), Bx.numpy(), specX.numpy(), Bq.numpy(), specQ.numpy()


def _plot(Bs, specS, Bx, specX, Bq, specQ):
    def norm(s):
        pk = np.abs(s).max()
        return s / pk if pk > 0 else s

    fig, axes = plt.subplots(3, 1, figsize=(9, 9))

    axes[0].plot(Bs, norm(specS), 'k', lw=1.2)
    axes[0].set_title('S-band (2 GHz)')
    axes[0].set_ylabel('Intensity (norm.)')

    axes[1].plot(Bx, norm(specX), 'b', lw=1.2)
    axes[1].set_title('X-band (9.5 GHz)')
    axes[1].set_ylabel('Intensity (norm.)')

    axes[2].plot(Bq, norm(specQ), 'r', lw=1.2)
    axes[2].set_title('Q-band (34 GHz)')
    axes[2].set_xlabel('Magnetic field (mT)')
    axes[2].set_ylabel('Intensity (norm.)')

    plt.suptitle('Cu(II) powder EPR at S, X, Q band', fontsize=12)
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
