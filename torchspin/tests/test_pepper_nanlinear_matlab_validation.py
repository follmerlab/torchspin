"""Position interpolation when a transition has no resonance at some knots
(B31, B88): against converged references, and the documented deviation from
EasySpin (tests/data/ref_pepper_nanlinear.mat, generate_pepper_nanlinear_refs.m).

EasySpin sets one flag when any transition lacks a resonance somewhere and
then interpolates the positions of every transition with ``L1`` (2-D grids)
or ``L3`` (axial grids) instead of the global cubic ``G3`` (pepper.m:701,
763-772).  0.9.4 copied that (S = 5/2 with B4 on D2h was cosine 0.9587 with
EasySpin before, 1 - 2e-10 after).  In all four cases here every missing knot
is a transition that leaves Exp.Range, and the all-linear mode reproduces the
coarse-grid facets: against a converged reference (no interpolation,
GridSize [181 1], wide sweep cropped; tests/data/ref_missing_knots.npz) the
EasySpin-identical spectra were 1 - cosine 0.0434 / 1.25e-3 / 0.0434 / 3.2e-4.

Since 0.9.28 the level pairs that leave the range are searched again over a
padded window and only slots that still miss knots go linear (B88), so these
spectra deliberately differ from EasySpin.  Measured (1 - cosine / max|dev|
relative to the maximum, interior of the range, pepper and pepper_autograd
alike), 0.9.27 -> 0.9.28:

* s52_b4_D2h_h1: vs converged 4.34e-2 / 0.236 -> 9.61e-5 / 0.0135;
  vs EasySpin 2.3e-10 -> 4.13e-2 (amplitude 1.0082);
* s52_b4_D2h_h0: 1.25e-3 / 0.0863 -> 2.33e-6 / 0.0033; EasySpin 8e-11 -> 1.18e-3;
* s52_b4_C2h_h1: 4.34e-2 / 0.236 -> 9.71e-5 / 0.0135; EasySpin 3.6e-3 -> 6.05e-2
  (was a strict xfail before: B58, EasySpin's open-phi L1 rows);
* s52_rhombic_h1: 3.21e-4 / 0.0256 -> 1.16e-6 / 0.0017; EasySpin 3e-11 -> 2.9e-4.

The EasySpin comparisons are strict xfails with these numbers.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io import loadmat

from torchspin import SpinSystem, Experiment, Options, pepper
from torchspin.pepper_autograd import pepper_autograd

DATA = Path(__file__).resolve().parents[2] / 'tests' / 'data'
REF = DATA / 'ref_pepper_nanlinear.mat'
CONVERGED = DATA / 'ref_missing_knots.npz'

_spec = importlib.util.spec_from_file_location('_gen_missing_knots', DATA / 'generate_missing_knots_refs.py')
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

COS_MIN = 1 - 1e-7
AMP_TOL = 1e-4

# EasySpin case -> converged reference, and bounds on (1 - cosine, max|dev|)
# against it (interior; measured x 1.05, see the module docstring)
CONV = {
    's52_b4_D2h_h1': ('case3', 1.01e-4, 1.42e-2),
    's52_b4_D2h_h0': ('case3_h0', 2.45e-6, 3.47e-3),
    's52_b4_C2h_h1': ('case3_c2h', 1.02e-4, 1.42e-2),
    's52_rhombic_h1': ('case3r', 1.22e-6, 1.81e-3),
}
# measured vs EasySpin since 0.9.28 (1 - cosine, amplitude ratio)
DEVIATION = {
    's52_b4_D2h_h1': (4.13e-2, 1.0082),
    's52_b4_D2h_h0': (1.18e-3, 0.9841),
    's52_b4_C2h_h1': (6.05e-2, 1.0617),
    's52_rhombic_h1': (2.86e-4, 0.9953),
}


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
            c, a = DEVIATION[n]
            reason = (f'B88 (0.9.28): the range cuts transitions; torchspin searches them again over a '
                      f'padded window and interpolates only slots with missing knots linearly, EasySpin '
                      f'interpolates every slot linearly (pepper.m:701, 763-772). Measured 1 - cosine '
                      f'{c:.3g}, amplitude {a:.4f}; torchspin is closer to the converged reference '
                      f'(test_nanlinear_vs_converged)')
            out.append(pytest.param(n, f, marks=pytest.mark.xfail(strict=True, reason=reason), id=f'{n}-{f}'))
    return out


@pytest.mark.parametrize('func', ['pepper', 'pepper_autograd'])
@pytest.mark.parametrize('name', list(CONV))
def test_nanlinear_vs_converged(cases, name, func):
    """The four EasySpin cases against converged references (no
    interpolation, GridSize [181 1], wide sweep cropped)."""
    sys, exp, opt = _inputs(cases[name])
    rn, cmax, dmax = CONV[name]
    fn = pepper if func == 'pepper' else pepper_autograd
    x, spc = fn(sys, exp, opt)
    x = x.detach().cpu().numpy().ravel(); spc = spc.detach().cpu().numpy().ravel()
    r = np.load(CONVERGED)
    assert np.allclose(x, r[rn + '_x'], rtol=0, atol=1e-9)
    c, d = gen.metrics(spc, r[rn], gen.interior(x, sys))
    print(f'{name} {func}: 1 - cosine {c:.4e}, max dev {d:.4e}')
    assert c <= cmax and d <= dmax, (name, func, c, d)


@pytest.mark.parametrize('name,func', _params())
def test_nanlinear_vs_matlab(cases, name, func):
    """EasySpin bit-compatibility of the all-linear mode (0.9.4-0.9.27); a
    documented deviation since 0.9.28 (strict xfails, see _params)."""
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
    """Both implementations take the same re-search and interpolation path, open-phi included."""
    sys, exp, opt = _inputs(cases[name])
    _, y1 = pepper(sys, exp, opt)
    _, y2 = pepper_autograd(sys, exp, opt)
    y1 = y1.detach().cpu().numpy().ravel(); y2 = y2.detach().cpu().numpy().ravel()
    assert np.allclose(y1, y2, rtol=1e-8, atol=1e-8 * abs(y2).max())
