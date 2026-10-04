"""
MATLAB validation of pepper's D-strain spectrum construction (PORT_SPEC_2 1.3).

Reference: tests/data/ref_pepper_dstrain.mat from generate_pepper_dstrain_refs.m,
mirroring EasySpin tests pepper_dstrain_explicit, pepper_dstraincorrelated
(r = 0, +1, -1, at GridSize 15/31/61 and without strain) and pepper_multidstrain,
plus two exchange-coupled triplets with correlated D/E strain.

These exercise EasySpin's "summation" accumulation (interpolate positions,
intensities and strain widths per transition to the fine grid, one Gaussian
per facet with Lambda smoothing) and the level-pair transition bookkeeping.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from torchspin.spinsystem import SpinSystem
from torchspin.experiment import Experiment, Options
from torchspin.pepper import pepper
from torchspin.constants import GFREE

REF_FILE = Path(__file__).parent.parent.parent / "tests" / "data" / "ref_pepper_dstrain.mat"


@pytest.fixture(scope="module")
def cases():
    if not REF_FILE.exists():
        pytest.skip(f"Reference data not found: {REF_FILE}")
    refs = loadmat(str(REF_FILE), squeeze_me=True, struct_as_record=False)
    return {str(c.name): c for c in np.atleast_1d(refs['cases'])}


def _get(o, name, default=None):
    return getattr(o, name) if name in getattr(o, '_fieldnames', []) else default


def sys_from_mat(S) -> SpinSystem:
    Sv = np.atleast_1d(np.asarray(S.S, dtype=float)).tolist()
    n_e = len(Sv)
    kw = {'S': Sv}
    g = _get(S, 'g')
    if g is None:
        kw['g'] = GFREE
    else:
        g = np.asarray(g, dtype=float)
        kw['g'] = g.tolist() if g.ndim else float(g)
    D = np.asarray(S.D, dtype=float)
    if D.ndim == 0:
        kw['D'] = float(D)                      # scalar D -> [D, E=0]
    elif D.ndim == 1 and D.size == 2 and n_e == 1:
        kw['D'] = [D.tolist()]                  # [[D, E]]
    elif D.ndim == 1 and D.size == n_e:
        kw['D'] = D.reshape(n_e, 1).tolist()    # one scalar D per electron
    else:
        kw['D'] = D.tolist()
    Ds = np.asarray(S.DStrain, dtype=float)
    kw['DStrain'] = Ds.tolist() if Ds.ndim else float(Ds)
    r = _get(S, 'DStrainCorr')
    if r is not None:
        kw['DStrainCorr'] = np.atleast_1d(np.asarray(r, dtype=float)).tolist()
    ee = _get(S, 'ee')
    if ee is not None:
        kw['ee'] = [float(ee)]
    lwpp = _get(S, 'lwpp')
    if lwpp is not None:
        kw['lwpp'] = [float(lwpp), 0.0]
    else:
        kw['lw'] = [0.0, 0.0]
    return SpinSystem(**kw)


def exp_from_mat(E) -> Experiment:
    kw = {'mwFreq': float(E.mwFreq)}
    if 'Range' in E._fieldnames:
        kw['Range'] = np.asarray(E.Range, dtype=float).tolist()
    else:
        kw['CenterSweep'] = np.asarray(E.CenterSweep, dtype=float).tolist()
    for name in ('nPoints', 'Harmonic'):
        v = _get(E, name)
        if v is not None:
            kw[name] = int(v)
    if 'Harmonic' not in kw:
        kw['Harmonic'] = 1
    return Experiment(**kw)


def opt_from_mat(O) -> Options:
    gs = np.atleast_1d(np.asarray(O.GridSize, dtype=float)).astype(int).tolist()
    return Options(GridSize=gs if len(gs) == 2 else [gs[0], 4], Verbosity=0)  # EasySpin scalar N == [N 4]


def cosine(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


CASES = [
    'dstrain_explicit', 'dstraincorr_+0', 'dstraincorr_+1', 'dstraincorr_-1',
    'dstraincorr_+0_gs31', 'dstraincorr_+1_gs31', 'dstraincorr_+0_gs61', 'dstraincorr_+1_gs61',
    'dstraincorr_nostrain_gs15', 'multidstrain', 'multidstrain_31', 'two_triplets_dstrain_corr_61',
]


@pytest.mark.parametrize("name", CASES)
def test_dstrain_spectra_vs_matlab(cases, name):
    c = cases[name]
    B, spc = pepper(sys_from_mat(c.Sys), exp_from_mat(c.Exp), opt_from_mat(c.Opt))
    spc = spc.numpy()
    ref = np.asarray(c.spc, dtype=float)
    assert np.all(np.isfinite(spc))
    cs = cosine(spc, ref)
    ratio = np.abs(spc).max() / np.abs(ref).max()
    assert cs > 0.999, f"{name}: cosine {cs:.5f}"
    assert abs(ratio - 1) < 0.03, f"{name}: amplitude ratio {ratio:.4f}"
