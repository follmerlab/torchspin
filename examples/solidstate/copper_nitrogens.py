"""Cu(II) with Four Imidazole Nitrogens
Axial Cu(II) powder EPR with four resolved 14N superhyperfine couplings, X-band.

Parameters from S. K. Buchanan, G. C. Dismukes, Biochemistry 1987, 26(16),
5049-5055 (Fig. 1 and its legend), https://doi.org/10.1021/bi00390a025

The four nitrogen A tensors point in four different directions, so this is not
a set of equivalent nuclei: each needs its own AFrame.

EasySpin equivalent: examples/solidstate/copper_nitrogens.m
"""
from pathlib import Path
import math
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torchspin import SpinSystem, Experiment, Options, pepper


def _system() -> SpinSystem:
    A_Cu = [0.0019 * 30e3, 0.0019 * 30e3, 0.0203 * 30e3]   # cm^-1 -> MHz
    A_N = [0.00145 * 30e3, 0.00145 * 30e3, 0.0017 * 30e3]
    h = math.pi / 2
    return SpinSystem(
        S=[0.5],
        g=[[2.05, 2.05, 2.19]],
        Nucs=['Cu', '14N', '14N', '14N', '14N'],   # natural-abundance copper
        A=[A_Cu] + [A_N] * 4,
        AFrame=[[0, 0, 0],
                [0, -h, 0],
                [-h, -h, 0],
                [-2 * h, -h, 0],
                [-3 * h, -h, 0]],
        lwpp=[0.0, 0.65],                          # mT, Lorentzian
    )


def run():
    Sys = _system()
    exp = Experiment(mwFreq=9.05, Range=[250, 350], nPoints=2048, Harmonic=1)

    # EasySpin's script uses second-order perturbation theory, which is what
    # makes four nitrogens affordable there.
    t0 = time.perf_counter()
    B, spc_pt = pepper(Sys, exp, Options(Method='perturb', GridSize=61, Verbosity=0))
    t_pt = time.perf_counter() - t0

    # The hybrid method keeps the electron and the copper exact and treats the
    # nitrogens perturbationally (EasySpin Opt.Method='hybrid' with
    # Opt.HybridCoreNuclei=1).  The copper hyperfine is ~600 MHz, far too large
    # for perturbation theory, so this is markedly more accurate here at a
    # comparable cost; the exact matrix method would need a 648-dimensional
    # Hilbert space.
    t0 = time.perf_counter()
    _, spc_hy = pepper(Sys, exp, Options(Method='hybrid', HybridCoreNuclei=[1],
                                         GridSize=61, Verbosity=0))
    t_hy = time.perf_counter() - t0

    return B.numpy(), spc_pt.numpy(), spc_hy.numpy(), t_pt, t_hy


def _plot(B_np, spc_pt, spc_hy, t_pt, t_hy):
    def norm(s):
        pk = np.abs(s).max()
        return s / pk if pk > 0 else s

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(B_np, norm(spc_pt), 'r', lw=1.2, label=f'perturb ({t_pt:.2f} s)')
    ax.plot(B_np, norm(spc_hy), 'b', lw=1.2, label=f'hybrid ({t_hy:.2f} s)')
    ax.set_xlabel('Magnetic field (mT)')
    ax.set_ylabel('Intensity (norm.)')
    ax.set_title('Cu(II) + 4×¹⁴N powder EPR, X-band (9.05 GHz)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
