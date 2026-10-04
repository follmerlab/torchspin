"""
MATLAB validation of pepper single-crystal simulations (PORT_SPEC_2 item 3.1).

Reference: tests/data/ref_pepper_crystal.mat from generate_pepper_crystal_refs.m,
mirroring EasySpin's pepper_crystal_molframe/multiori/samplerot/sites/th/
twocrystals, pepper_samplerotation, pepper_intensity_crystal_mx,
pepper_intensity_isopowder_crystal and pepper_smallgdiff_crystal tests.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from torchspin.spinsystem import SpinSystem
from torchspin.experiment import Experiment, Options
from torchspin.pepper import pepper
from torchspin.constants import GFREE

REF_FILE = Path(__file__).parent.parent.parent / "tests" / "data" / "ref_pepper_crystal.mat"


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
    kw = {'S': [0.5]}
    g = _get(S, 'g')
    kw['g'] = GFREE if g is None else _axial3(g)
    for name in ('gFrame', 'AFrame', 'HStrain'):
        v = _get(S, name)
        if v is not None:
            kw[name] = np.asarray(v, dtype=float).reshape(-1).tolist()
    nucs = _get(S, 'Nucs')
    if nucs is not None and str(nucs):
        kw['Nucs'] = str(nucs)
        kw['A'] = [_axial3(_get(S, 'A'))]
    for name in ('lw', 'lwpp'):
        v = _get(S, name)
        if v is not None:
            v = np.atleast_1d(np.asarray(v, dtype=float)).tolist()
            kw[name] = v if len(v) == 2 else [v[0], 0.0]
    return SpinSystem(**kw)


def exp_from_mat(E) -> Experiment:
    kw = {'mwFreq': float(E.mwFreq), 'Range': np.asarray(E.Range, dtype=float).tolist()}
    for name, conv in (('nPoints', int), ('Harmonic', int)):
        v = _get(E, name)
        if v is not None:
            kw[name] = conv(v)
    if 'Harmonic' not in kw:
        kw['Harmonic'] = 1
    sf = _get(E, 'SampleFrame')
    if sf is not None:
        sf = np.asarray(sf, dtype=float)
        kw['SampleFrame'] = sf.reshape(-1, 3).tolist() if sf.size else [[0.0, 0.0, 0.0]]
    cs = _get(E, 'CrystalSymmetry')
    if cs is not None:
        kw['CrystalSymmetry'] = str(cs) if isinstance(cs, str) else int(cs)
    mf = _get(E, 'MolFrame')
    if mf is not None:
        kw['MolFrame'] = np.asarray(mf, dtype=float).reshape(-1).tolist()
    rot = _get(E, 'SampleRotation')
    if rot is not None and np.asarray(rot, dtype=object).size:
        axis, rho = rot[0], rot[1]
        axis = str(axis) if isinstance(axis, str) else np.asarray(axis, dtype=float).reshape(-1).tolist()
        kw['SampleRotation'] = (axis, np.atleast_1d(np.asarray(rho, dtype=float)).tolist())
    return Experiment(**kw)


def opt_from_mat(O) -> Options:
    kw = {'Verbosity': 0}
    sites = _get(O, 'Sites')
    if sites is not None and np.asarray(sites).size:
        kw['Sites'] = np.atleast_1d(np.asarray(sites, dtype=int)).tolist()
    sep = _get(O, 'separate')
    if sep:
        kw['separate'] = str(sep)
    m = _get(O, 'Method')
    if m is not None:
        kw['Method'] = str(m)
    return Options(**kw)


def cosine(a, b):
    a = np.asarray(a, dtype=float).ravel(); b = np.asarray(b, dtype=float).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


LOOSE = {}


@pytest.mark.parametrize("name", [
    'molframe_gFrame', 'molframe_MolFrame', 'multiori_D2h', 'samplerot_sum', 'samplerot_separate',
    'sites_all', 'sites_1', 'sites_23', 'th', 'twocrystals_sum', 'twocrystals_separate',
    'samplerotation_rotated', 'samplerotation_viaRotation', 'intensity_crystal_mx',
    'isopowder_powder', 'isopowder_crystal', 'smallgdiff_crystal_A_n100', 'smallgdiff_crystal_B_n100',
    'smallgdiff_crystal_A_n20000', 'smallgdiff_crystal_B_n20000', 'crystal_hstrain',
])
def test_crystal_vs_matlab(cases, name):
    c = cases[name]
    x, y = pepper(sys_from_mat(c.Sys), exp_from_mat(c.Exp), opt_from_mat(c.Opt))
    y = y.numpy(); ref = np.asarray(c.spc, dtype=float)
    assert y.shape == ref.shape, (y.shape, ref.shape)
    assert np.all(np.isfinite(y))
    cs_min, amp_tol, _ = LOOSE.get(name, (0.999, 0.02, ''))
    cs = cosine(y, ref); ratio = np.abs(y).max() / np.abs(ref).max()
    assert cs > cs_min, f"{name}: cosine {cs:.5f}"
    assert abs(ratio - 1) < amp_tol, f"{name}: amplitude ratio {ratio:.4f}"


def test_isotropic_crystal_equals_powder_integral(cases):
    """pepper_intensity_isopowder_crystal: the integrated intensity of an
    isotropic-g crystal spectrum equals that of the powder spectrum."""
    cp, cc = cases['isopowder_powder'], cases['isopowder_crystal']
    xp, yp = pepper(sys_from_mat(cp.Sys), exp_from_mat(cp.Exp), opt_from_mat(cp.Opt))
    xc, yc = pepper(sys_from_mat(cc.Sys), exp_from_mat(cc.Exp), opt_from_mat(cc.Opt))
    dx = float(xp[1] - xp[0])
    assert abs(float(yp.sum() * dx) / float(yc.sum() * dx) - 1) < 1e-6
