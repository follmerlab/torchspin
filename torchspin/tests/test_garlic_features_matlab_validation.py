"""garlic feature validation against EasySpin (PORT_SPEC_2 item 4.1).

Reference: tests/data/ref_garlic_features.mat from tests/data/generate_garlic_features_refs.m
(EasySpin 6, MATLAB R2024b), mirroring garlic_isotopemix, garlic_multicomponents,
garlic_separate_components, garlic_fieldrange, garlic_freqsweep_autorange and
garlic_equivnuclei: natural-abundance mixtures ('B', 'Cl' with n=2, 'C'),
multi-component sums and separate rows, automatic field/frequency sweep ranges
(axis compared to EasySpin's), equivalent nuclei vs explicit lists.
"""
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from torchspin.spinsystem import SpinSystem
from torchspin.experiment import Experiment, Options
from torchspin.garlic import garlic
from torchspin.constants import GFREE

REF = Path(__file__).resolve().parents[2] / 'tests' / 'data' / 'ref_garlic_features.mat'


def _fields(o):
    return getattr(o, '_fieldnames', [])


def _get(o, name, default=None):
    return getattr(o, name) if name in _fields(o) else default


def sys_from_mat(S) -> SpinSystem:
    kw = {'S': [0.5]}
    g = _get(S, 'g')
    kw['g'] = GFREE if g is None else float(g)
    nucs = _get(S, 'Nucs')
    if nucs is not None:
        kw['Nucs'] = str(nucs)
        A = np.atleast_1d(np.asarray(_get(S, 'A'), dtype=float))
        kw['A'] = [[float(a)] * 3 for a in A]
        n = _get(S, 'n')
        if n is not None:
            kw['n'] = np.atleast_1d(np.asarray(n, dtype=float)).astype(int).tolist()
    for name in ('lw', 'lwpp'):
        v = _get(S, name)
        if v is not None:
            v = np.atleast_1d(np.asarray(v, dtype=float)).tolist()
            kw[name] = v if len(v) == 2 else [v[0], 0.0]
    w = _get(S, 'weight')
    if w is not None:
        kw['weight'] = float(w)
    return SpinSystem(**kw)


def exp_from_mat(E) -> Experiment:
    kw = {}
    for name, conv in (('mwFreq', float), ('Field', float), ('Harmonic', int), ('nPoints', int)):
        v = _get(E, name)
        if v is not None:
            kw[name] = conv(v)
    for name in ('Range', 'CenterSweep', 'mwRange', 'mwCenterSweep'):
        v = _get(E, name)
        if v is not None:
            kw[name] = np.asarray(v, dtype=float).tolist()
    return Experiment(**kw)


def opt_from_mat(O) -> Options:
    kw = {'Verbosity': 0}
    for name in ('Method', 'separate'):
        v = _get(O, name)
        if v is not None:
            kw[name] = str(v)
    return Options(**kw)


def cosine(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


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


def _run(c):
    systems = [sys_from_mat(c.Sys)]
    for extra in ('Sys2', 'Sys3'):
        if _get(c, extra) is not None:
            systems.append(sys_from_mat(getattr(c, extra)))
    sys = systems if len(systems) > 1 else systems[0]
    return garlic(sys, exp_from_mat(c.Exp), opt_from_mat(c.Opt))


@pytest.mark.parametrize('name', _case_names())
def test_garlic_features_vs_matlab(cases, name):
    c = cases[name]
    x, spc = _run(c)
    x = x.numpy(); spc = spc.numpy()
    ref = np.asarray(c.spc, dtype=float); ref_x = np.asarray(c.x, dtype=float)
    assert np.allclose(x, ref_x, rtol=1e-9, atol=1e-9 * abs(ref_x).max()), 'sweep axis differs'
    assert spc.shape == ref.shape, (spc.shape, ref.shape)
    rows = [(spc, ref)] if ref.ndim == 1 else list(zip(spc, ref))
    for k, (a, b) in enumerate(rows):
        assert cosine(a, b) > 0.999, (name, k, cosine(a, b))
        amp = abs(a).max() / abs(b).max()
        assert abs(amp - 1) < 0.02, (name, k, amp)
