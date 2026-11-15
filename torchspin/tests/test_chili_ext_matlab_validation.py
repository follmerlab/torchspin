"""
MATLAB validation of the chili stochastic-Liouville port (PORT_SPEC_2 Phase 2).

Reference: tests/data/ref_chili_ext.mat from tests/data/generate_chili_ext_refs.m,
mirroring 25 EasySpin chili_*.m tests (simple, fieldsweeps, nucspins,
simplepotential, rhombicdiff, general_appfield, fullA/fullg, twonuclei,
postconvolution, potential_general, twomethods, freqsweep, freqderiv, frqdep,
mwphase, temperature, magnetictilt, jkmin_psmin, largebasis_nan, twocomponents,
momdsimple, solvers, rhombicg, pepper limit, potential frequency sweep).
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from torchspin.spinsystem import SpinSystem
from torchspin.experiment import Experiment
from torchspin.chili import chili, ChiliOptions
from torchspin.constants import GFREE

REF_FILE = Path(__file__).parent.parent.parent / "tests" / "data" / "ref_chili_ext.mat"


@pytest.fixture(scope="module")
def cases():
    if not REF_FILE.exists():
        pytest.skip(f"Reference data not found: {REF_FILE}")
    refs = loadmat(str(REF_FILE), squeeze_me=True, struct_as_record=False)
    return {str(c.name): c for c in np.atleast_1d(refs['cases'])}


def _fields(o):
    return getattr(o, '_fieldnames', [])


def _get(o, name, default=None):
    return getattr(o, name) if name in _fields(o) else default


def _axial3(v):
    a = np.atleast_1d(np.asarray(v, dtype=float))
    if a.size == 1:
        return [float(a[0])] * 3
    if a.size == 2:
        return [float(a[0]), float(a[0]), float(a[1])]
    return a.tolist()


def sys_from_mat(S) -> SpinSystem:
    f = _fields(S)
    Sv = np.atleast_1d(np.asarray(_get(S, 'S', 0.5), dtype=float)).tolist()
    n_e = len(Sv)
    kw = {'S': Sv}
    g = _get(S, 'g')
    if g is None:
        kw['g'] = GFREE
    else:
        g = np.asarray(g, dtype=float)
        kw['g'] = g.tolist() if (g.ndim == 2 and g.shape == (3, 3)) else _axial3(g)
    D = _get(S, 'D')
    if D is not None:
        D = np.asarray(D, dtype=float)
        kw['D'] = float(D) if D.ndim == 0 else D.tolist()
    nucs = _get(S, 'Nucs')
    if nucs is not None and str(nucs):
        kw['Nucs'] = str(nucs)
        n_n = len(str(nucs).split(','))
        A = np.asarray(_get(S, 'A'), dtype=float)
        if A.ndim == 2 and A.shape == (3 * n_n, 3):
            kw['A'] = A.tolist()                                 # full matrices
        elif A.ndim == 2:
            kw['A'] = [_axial3(row) for row in A]
        elif n_n > 1 and A.size == n_n:
            kw['A'] = [[float(v)] * 3 for v in A]
        else:
            kw['A'] = [_axial3(A)]
        AF = _get(S, 'AFrame')
        if AF is not None:
            kw['AFrame'] = np.asarray(AF, dtype=float).tolist()
    for name in ('lw', 'lwpp'):
        v = _get(S, name)
        if v is not None:
            v = np.atleast_1d(np.asarray(v, dtype=float)).tolist()
            kw[name] = v if len(v) == 2 else [v[0], 0.0]
    for name in ('tcorr', 'logtcorr', 'Diff', 'logDiff'):
        v = _get(S, name)
        if v is not None:
            v = np.atleast_1d(np.asarray(v, dtype=float))
            kw[name] = float(v[0]) if v.size == 1 else v.tolist()
    pot = _get(S, 'Potential')
    if pot is not None:
        kw['Potential'] = np.asarray(pot, dtype=float).reshape(-1, 4).tolist()
    w = _get(S, 'weight')
    if w is not None:
        kw['weight'] = float(w)
    return SpinSystem(**kw)


def exp_from_mat(E) -> Experiment:
    f = _fields(E)
    kw = {}
    if 'Field' in f and 'mwFreq' not in f:
        kw['Field'] = float(E.Field)
        kw['mwRange'] = np.asarray(E.mwRange, dtype=float).tolist()
        kw['Harmonic'] = int(_get(E, 'Harmonic', 0))
    else:
        kw['mwFreq'] = float(E.mwFreq)
        if 'Range' in f:
            kw['Range'] = np.asarray(E.Range, dtype=float).tolist()
        else:
            kw['CenterSweep'] = np.asarray(E.CenterSweep, dtype=float).tolist()
        kw['Harmonic'] = int(_get(E, 'Harmonic', 1))
    for name in ('nPoints',):
        v = _get(E, name)
        if v is not None:
            kw[name] = int(v)
    for name in ('Temperature', 'mwPhase'):
        v = _get(E, name)
        if v is not None:
            kw[name] = float(v)
    sf = _get(E, 'SampleFrame')
    if sf is not None:
        kw['SampleFrame'] = np.asarray(sf, dtype=float).reshape(-1).tolist()
    return Experiment(**kw)


def opt_from_mat(O) -> ChiliOptions:
    kw = {'Verbosity': 0}
    for name, conv in (('LLMK', lambda v: np.asarray(v, dtype=float).astype(int).tolist()),
                       ('highField', bool), ('jKmin', int), ('evenK', bool), ('MpSymm', bool),
                       ('GridSize', int), ('Solver', str), ('FieldSweepMethod', str)):
        v = _get(O, name)
        if v is not None:
            kw[name] = conv(v)
    pc = _get(O, 'PostConvNucs')
    if pc is not None and np.asarray(pc).size:
        kw['PostConvNucs'] = np.atleast_1d(np.asarray(pc, dtype=int)).tolist()
    return ChiliOptions(**kw)


def cosine(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# name -> (cosine threshold, amplitude tolerance, reason)  — default 0.999 / 2 %
LOOSE = {}
XFAIL = {}
# EasySpin reference known to be wrong: held to the default tolerance as a strict xfail
# (the test must keep failing), with the objective replacement named in the reason.
STRICT_XFAIL = {
    'simple_tc1e-07': "EasySpin's default 'fast' output stopped its continued fraction early "
                      "(MATLAB check ref_chili_mpsymm_d: at Opt.Threshold 1e-10 it moves by "
                      "cosine 0.9979 / amplitude 1.069). Measured against this reference: cosine "
                      "0.99798, amplitude 1.066. Objective check: test_chili_lanczos.py::"
                      "test_simple_tc1e7_matches_converged_easyspin_fast (cosine 0.9999966)",
}


def _run(c):
    exp = exp_from_mat(c.Exp)
    if _get(c, 'Sys2') is not None:
        return chili([sys_from_mat(c.Sys), sys_from_mat(c.Sys2)], exp, opt_from_mat(c.Opt))
    return chili(sys_from_mat(c.Sys), exp, opt_from_mat(c.Opt))


def _case_names():
    if not REF_FILE.exists():
        return []
    refs = loadmat(str(REF_FILE), squeeze_me=True, struct_as_record=False)
    return [str(c.name) for c in np.atleast_1d(refs['cases'])]


@pytest.mark.parametrize("name", _case_names())
def test_chili_vs_matlab(cases, name, request):
    if name in XFAIL:
        pytest.xfail(XFAIL[name])
    if name in STRICT_XFAIL:
        request.applymarker(pytest.mark.xfail(strict=True, reason=STRICT_XFAIL[name]))
    c = cases[name]
    x, y = _run(c)
    y = y.numpy(); ref = np.asarray(c.y, dtype=float)
    assert np.all(np.isfinite(y)), f"{name}: non-finite spectrum"
    np.testing.assert_allclose(x.numpy(), np.asarray(c.x, dtype=float), rtol=0, atol=1e-6)
    cs = cosine(y, ref)
    ratio = np.abs(y).max() / np.abs(ref).max()
    cs_min, amp_tol, _ = LOOSE.get(name, (0.999, 0.02, ''))
    assert cs > cs_min, f"{name}: cosine {cs:.5f}"
    assert abs(ratio - 1) < amp_tol, f"{name}: amplitude ratio {ratio:.4f}"
