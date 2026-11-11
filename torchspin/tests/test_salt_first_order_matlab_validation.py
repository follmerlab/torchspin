"""Absolute intensity scale of salt's first-order perturbation path (bug B16, 0.8.1).

For one electron without quadrupole couplings ``salt`` uses first-order
perturbation theory by itself.  Before 0.8.1 that path gave every
``(nucleus, mS)`` row unit intensity, while the matrix path uses
|<i|mu_perp|j>|^2 in (MHz/mT)^2: the spectrum scale jumped ~2200x for 1H with
the path, and nuclei with different gn or I got wrong relative weights.  The
path now carries EasySpin ``endorfrq_perturb.m``'s intensities.

References: tests/data/salt_first_order_refs.mat from
tests/data/generate_salt_first_order_refs.m (EasySpin salt with
``Opt.Method = 'matrix'`` (EasySpin's default) and ``'perturb1'``).

Global convention.  torchspin's fixed-field salt returns the per-bin,
orientation-averaged amplitude: projected density (per MHz, solid angle 4 pi)
times ``dx / (4 pi)``.  EasySpin returns the density times 2 pi (chi
integral).  So for every path

    y_torchspin = y_EasySpin * dx / (8 pi^2),     dx = rf bin width (MHz),

which is what the ratio checks below divide out.
"""
import math
import os

import numpy as np
import pytest
import torch

from torchspin.experiment import Experiment, Options
from torchspin.salt import salt
from torchspin.spinsystem import SpinSystem

try:
    from scipy.io import loadmat
except ImportError:  # pragma: no cover
    loadmat = None

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "data")


def _refs():
    if loadmat is None:
        pytest.skip("scipy not installed")
    path = os.path.join(DATA_DIR, "salt_first_order_refs.mat")
    if not os.path.exists(path):
        pytest.skip(f"Reference file not found: {path}")
    return list(np.atleast_1d(loadmat(path, simplify_cells=True)["ref"]))


def _sys_from(S):
    A = np.atleast_2d(np.asarray(S['A'], float)).tolist()
    return SpinSystem(S=[float(S['S'])], g=[np.asarray(S['g'], float).tolist()],
                      Nucs=S['Nucs'], A=A)


def _run(r):
    S, E = r['Sys'], r['Exp']
    x, y = salt(_sys_from(S), Experiment(Field=float(E['Field']), mwFreq=9.7),
                Options(GridSize=[int(v) for v in np.ravel(r['Opt']['GridSize'])]),
                freq_range=tuple(np.asarray(E['Range'], float)), n_points=int(E['nPoints']),
                lw_mhz=float(S['lwEndor']))
    x, y = x.numpy(), y.numpy()
    dx = x[1] - x[0]
    return x, y / (dx / (8 * math.pi ** 2))    # on EasySpin's scale


def _cos(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


@pytest.mark.parametrize("case", [0, 1], ids=["1H", "1H+14N"])
def test_first_order_path_matches_easyspin_perturb1(case):
    """Default (first-order) path = EasySpin perturb1 in shape and absolute scale."""
    r = _refs()[case]
    x, y = _run(r)
    yp = np.asarray(r['yperturb1'], float)
    np.testing.assert_allclose(x, np.asarray(r['x'], float), rtol=0, atol=1e-12)
    assert _cos(y, yp) > 0.999999
    # before 0.8.1 (unit intensities): ratio 2206 (1H); 1451 at cosine 0.649 (1H+14N)
    assert abs(y.sum() / yp.sum() - 1) < 1e-6


@pytest.mark.parametrize("case", [0, 1], ids=["1H", "1H+14N"])
def test_first_order_path_matches_easyspin_default_matrix(case):
    """Default path vs EasySpin's default (matrix) method: first-order vs exact."""
    r = _refs()[case]
    _, y = _run(r)
    ym = np.asarray(r['ymatrix'], float)
    assert _cos(y, ym) > 0.99999
    # the remaining difference is first-order theory vs diagonalization (0.3 %)
    assert abs(y.sum() / ym.sum() - 1) < 5e-3


def test_field_swept_first_order_matches_matrix_scale():
    """Field-swept (Experiment.Range) mode: the first-order path has the matrix path's scale.

    A negligible Q (1e-7 MHz) sends the same system down the matrix path.
    """
    def mk(Q):
        kw = dict(S=[0.5], g=[[2.0, 2.01, 2.02]], Nucs='1H,14N', A=[[-2, -2, 4], [5, 5, 7]])
        if Q:
            kw['Q'] = [[0, 0, 0], [1e-7, 1e-7, -2e-7]]
        return SpinSystem(**kw)
    exp = Experiment(Range=[340, 350], mwFreq=9.7)
    y = {}
    for Q in (False, True):
        _, yy = salt(mk(Q), exp, Options(GridSize=31), freq_range=(0, 25), n_points=501, lw_mhz=0.2)
        y[Q] = yy.numpy()
    assert _cos(y[False], y[True]) > 0.99999
    assert abs(y[False].sum() / y[True].sum() - 1) < 5e-3


def test_first_order_gradient_two_nuclei():
    """The per-nucleus weights are constants: gradients w.r.t. A vs central differences."""
    A0 = [[-2.0, -2.0, 4.0], [5.0, 5.0, 7.0]]

    def run(A):
        s = SpinSystem(S=[0.5], g=[[2.0, 2.01, 2.02]], Nucs='1H,14N', A=A)
        return salt(s, Experiment(Field=326.5, mwFreq=9.7), Options(GridSize=[11, 3]),
                    freq_range=(0, 25), n_points=200, lw_mhz=0.5)[1]

    w = torch.linspace(0, 1, 200, dtype=torch.float64)
    A = torch.tensor(A0, dtype=torch.float64, requires_grad=True)
    (g,) = torch.autograd.grad((run(A) * w).sum(), A)
    h = 1e-5
    for i, j in ((0, 2), (1, 0)):
        Ap = torch.tensor(A0, dtype=torch.float64); Ap[i, j] += h
        Am = torch.tensor(A0, dtype=torch.float64); Am[i, j] -= h
        fd = float(((run(Ap) - run(Am)) * w).sum()) / (2 * h)
        assert abs(float(g[i, j]) - fd) < 1e-6 * max(abs(fd), 1e-30) + 1e-12 * float(run(Ap).abs().max())
