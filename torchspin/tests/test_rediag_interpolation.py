"""B133: ``Options.RediagLimit``.  For spin systems of dimension >= 16 on the pepper matrix
route (forward runs only), ``resfields_batch`` interpolates the two transition eigenvectors
between the neighboring field knots (cubic Hermite, derivatives by perturbation theory) when
both states keep an overlap >= RediagLimit across the segment, and moves the field by a
Rayleigh-quotient Newton step; other resonances, and those whose levels are degenerate at
a knot, are diagonalized as before.

``RediagLimit >= 1`` is the exact path (identical to disabling interpolation); smaller
systems and runs on the autograd graph never interpolate; at the default (0.999) spectra
stay within 5e-7 relative of exact while diagonalizing fewer matrices."""
import math

import numpy as np
import pytest
import torch

import torchspin.resfields_batch as _rb
from torchspin import Experiment, Options, SpinSystem, pepper
from torchspin.pepper_autograd import pepper_autograd

DT = torch.float64

# dim >= 16 systems: (SpinSystem kwargs, Experiment kwargs, GridSize)
LARGE = {
    'cu_2n': (dict(S=[0.5], g=[2.05, 2.05, 2.25], Nucs='63Cu,14N,14N',
                   A=[[30, 30, 500], [40, 40, 40], [40, 40, 40]], lwpp=[0.5, 0]),
              dict(mwFreq=9.5, Range=[260, 360]), [7, 4]),
    'mn_S52_5K': (dict(S=[2.5], g=2.0, D=[[600, 100]], Nucs='55Mn', A=[[250, 250, 250]], lwpp=[1, 0]),
                  dict(mwFreq=9.5, Range=[260, 360], Temperature=5.0), [7, 4]),
    'nitroxide_birad': (dict(S=[0.5, 0.5], g=[[2.008, 2.006, 2.003], [2.008, 2.006, 2.003]],
                             ee=[[80, 80, -160]], Nucs='14N,14N',
                             A=[[20, 20, 85, 0, 0, 0], [0, 0, 0, 20, 20, 85]], lw=[0.3, 0]),
                        dict(mwFreq=9.5, Range=[325, 355]), [7, 4]),
}


def _run(name, rl, nPoints=1024):
    sk, ek, gs = LARGE[name]
    opt = Options(Verbosity=0, GridSize=gs, RediagLimit=rl)
    return np.asarray(pepper(SpinSystem(**sk), Experiment(nPoints=nPoints, **ek), opt)[1])


def _count_eigh(monkeypatch):
    n = [0]
    orig = _rb.eigh

    def spy(H, *a, **k):
        n[0] += H.shape[:-2].numel()
        return orig(H, *a, **k)
    monkeypatch.setattr(_rb, 'eigh', spy)
    return n


@pytest.mark.parametrize('name', list(LARGE))
def test_default_close_to_exact_with_fewer_diagonalizations(name, monkeypatch):
    n = _count_eigh(monkeypatch)
    y_exact = _run(name, 1.0)
    n_exact, n[0] = n[0], 0
    y_int = _run(name, Options().RediagLimit)
    assert Options().RediagLimit == 0.999
    assert np.isfinite(y_int).all()
    assert np.abs(y_int - y_exact).max() <= 5e-7 * np.abs(y_exact).max()
    assert n[0] < n_exact                                   # the interpolation actually ran


def test_degenerate_levels_are_diagonalized():
    """Symmetric biradical (2 equivalent 14N): many transition levels are exactly degenerate,
    and each knot's eigh returns its own basis of the subspace.  Those resonances must be
    diagonalized; interpolating them gave 5e-6 relative here (grouped intensities 2e-4)."""
    assert np.abs(_run('nitroxide_birad', 0.999) - _run('nitroxide_birad', 1.0)).max() \
        <= 1e-9 * np.abs(_run('nitroxide_birad', 1.0)).max()


def test_isolated_levels():
    E = torch.tensor([[-3.0, -1.0, -1.0, 2.0], [-3.0, -1.0, -1.0 + 1e-3, 2.0]], dtype=DT)
    for n, want in ((0, [True, True]), (1, [False, True]), (2, [False, True]), (3, [True, True])):
        got = _rb._isolated(E, torch.tensor([n, n]))
        assert got.tolist() == want


def test_limit_one_is_the_exact_path(monkeypatch):
    """RediagLimit >= 1 must give the pre-B133 result bit for bit: compare against the
    default with interpolation switched off by the dimension gate."""
    y1 = _run('cu_2n', 1.0)
    y5 = _run('cu_2n', 5.0)
    monkeypatch.setattr(_rb, '_INTERP_MIN_DIM', 10 ** 9)
    y_off = _run('cu_2n', 0.999)
    assert np.array_equal(y1, y_off) and np.array_equal(y5, y_off)


def test_limit_zero_interpolates_everything_and_stays_sane():
    y0 = _run('cu_2n', 0.0)
    y1 = _run('cu_2n', 1.0)
    assert np.isfinite(y0).all()
    assert np.abs(y0 - y1).max() <= 1e-2 * np.abs(y1).max()


def test_small_systems_never_interpolate():
    sys = SpinSystem(S=[0.5], g=[2.05, 2.06, 2.25], Nucs='63Cu', A=[[40, 50, 550]], lw=[1.0, 0])  # dim 8
    exp = Experiment(mwFreq=9.5, Range=[250, 360], nPoints=1024)
    ys = [np.asarray(pepper(sys, exp, Options(Verbosity=0, GridSize=[10, 4], RediagLimit=rl))[1])
          for rl in (1.0, 0.999, 0.0)]
    assert np.array_equal(ys[0], ys[1]) and np.array_equal(ys[0], ys[2])


def test_gradient_runs_never_interpolate():
    """On the autograd graph the resonances are always diagonalized (B14 linearization), so
    the value and the gradient do not depend on RediagLimit."""
    def run(rl):
        g = torch.tensor([2.05, 2.05, 2.25], dtype=DT, requires_grad=True)
        sys = SpinSystem(S=[0.5], g=[g], Nucs='63Cu,14N', A=[[30.0, 30.0, 500.0], [40.0, 40.0, 40.0]],
                         lw=[1.0, 0.0])                                           # dim 24
        _, y = pepper_autograd(sys, Experiment(mwFreq=9.5, Range=[260, 360], nPoints=256, Harmonic=1),
                               Options(Verbosity=0, GridSize=[7, 4], RediagLimit=rl))
        (gg,) = torch.autograd.grad((y * torch.linspace(-1, 1, 256, dtype=DT)).sum(), [g])
        return y.detach(), gg
    y1, g1 = run(1.0)
    y0, g0 = run(0.0)
    assert torch.equal(y1, y0) and torch.equal(g1, g0)


@pytest.mark.parametrize('bad', [-0.1, True, 'x', None, float('nan'), [0.9]])
def test_validation_rejects(bad):
    with pytest.raises(ValueError, match='RediagLimit'):
        Options(RediagLimit=bad)


@pytest.mark.parametrize('ok', [0, 0.5, 0.95, 1, 2.0, math.inf, np.float64(0.99)])
def test_validation_accepts(ok):
    assert Options(RediagLimit=ok).RediagLimit == ok
