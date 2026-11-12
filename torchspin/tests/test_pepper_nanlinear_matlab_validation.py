"""Position interpolation when a transition has no resonance at some knots,
against EasySpin (B31; tests/data/ref_pepper_nanlinear.mat,
generate_pepper_nanlinear_refs.m).

EasySpin sets one flag when any transition lacks a resonance somewhere and
then interpolates the positions of every transition with ``L1`` (2-D grids)
or ``L3`` (axial grids) instead of the global cubic ``G3``
(pepper.m:701, 763-772).  Up to 0.9.3 ``pepper`` switched only the slots
that carried NaNs and still splined the others (``pepper_autograd`` was
already right): S = 5/2 with B4 on an explicit D2h grid was cosine 0.9587.

Measured with the fix (pepper and pepper_autograd agree): the three closed-phi
cases have cosine >= 1 - 3e-8 and amplitude within 4e-5.  On main they were
cosine 0.95870 / 0.99882 / 0.99971 and amplitude 1.0082 / 0.9841 / 0.9953.
"""
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io import loadmat

from torchspin import SpinSystem, Experiment, Options, pepper
from torchspin.pepper_autograd import pepper_autograd

REF = Path(__file__).resolve().parents[2] / 'tests' / 'data' / 'ref_pepper_nanlinear.mat'

COS_MIN = 1 - 1e-7
AMP_TOL = 1e-4

# Open-phi grid (C2h): the fix makes pepper agree with pepper_autograd
# (cosine 0.93949 -> 0.99635), but both stay off EasySpin (amplitude 1.053).
# Without interpolation (GridSize [19 1]) or without NaNs (Range [0 800]) the
# same system agrees to 2e-8 / 2e-6, so the residual is in the L1 interpolation
# or projection of open-phi grids with missing knots.  Not traced.
OPEN_PHI = {'s52_b4_C2h_h1'}


def _load():
    m = loadmat(REF, squeeze_me=True, struct_as_record=False)
    return {str(c.name): c for c in np.atleast_1d(m['cases'])}


@pytest.fixture(scope='module')
def cases():
    if not REF.exists():
        pytest.skip(f'reference file missing: {REF}')
    return _load()


def _case_names():
    return list(_load().keys()) if REF.exists() else []


def _inputs(c):
    S = c.Sys
    D = np.atleast_1d(np.asarray(S.D, dtype=float))
    kw = dict(S=[float(S.S)], g=float(S.g), D=[[D[0], D[1] if D.size > 1 else 0.0]],
              lw=[float(v) for v in np.atleast_1d(S.lw)])
    if hasattr(S, 'B4'):
        kw['B'] = [None, [[float(v) for v in np.atleast_1d(S.B4)]]]
    E = c.Exp
    exp = Experiment(mwFreq=float(E.mwFreq), Range=[float(v) for v in np.atleast_1d(E.Range)],
                     nPoints=int(E.nPoints), Harmonic=int(E.Harmonic))
    O = c.Opt
    okw = {'Verbosity': 0}
    if hasattr(O, 'GridSize'):
        okw['GridSize'] = [int(v) for v in np.atleast_1d(O.GridSize)]
    if hasattr(O, 'GridSymmetry'):
        okw['GridSymmetry'] = str(O.GridSymmetry)
    return SpinSystem(**kw), exp, Options(**okw)


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _params():
    out = []
    for n in _case_names():
        for f in ('pepper', 'pepper_autograd'):
            marks = ()
            if n in OPEN_PHI:
                marks = pytest.mark.xfail(strict=True, reason='open-phi L1 residual (not traced)')
            out.append(pytest.param(n, f, marks=marks, id=f'{n}-{f}'))
    return out


@pytest.mark.parametrize('name,func', _params())
def test_nanlinear_vs_matlab(cases, name, func):
    c = cases[name]
    sys, exp, opt = _inputs(c)
    fn = pepper if func == 'pepper' else pepper_autograd
    x, spc = fn(sys, exp, opt)
    x = x.detach().cpu().numpy().ravel(); spc = spc.detach().cpu().numpy().ravel()
    ref = np.asarray(c.spc, dtype=float).ravel(); ref_x = np.asarray(c.x, dtype=float).ravel()
    assert np.allclose(x, ref_x, rtol=1e-9), 'sweep axis differs'
    cs = _cos(spc, ref)
    amp = abs(spc).max() / abs(ref).max()
    print(f'{name} {func}: cosine {cs:.9f}, amplitude ratio {amp:.6f}')
    assert cs > COS_MIN, (name, cs)
    assert amp == pytest.approx(1.0, abs=AMP_TOL), (name, amp)


@pytest.mark.parametrize('name', _case_names())
def test_pepper_matches_autograd(cases, name):
    """Both implementations take the same interpolation path, open-phi included."""
    sys, exp, opt = _inputs(cases[name])
    _, y1 = pepper(sys, exp, opt)
    _, y2 = pepper_autograd(sys, exp, opt)
    y1 = y1.detach().cpu().numpy().ravel(); y2 = y2.detach().cpu().numpy().ravel()
    assert np.allclose(y1, y2, rtol=1e-8, atol=1e-8 * abs(y2).max())
