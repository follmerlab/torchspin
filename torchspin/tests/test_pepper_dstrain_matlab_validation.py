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


# multidstrain / multidstrain_31: two S = 1 spins with axial D and E strain.
# EasySpin's hamsymm ignores strains and picks Dinfh (one meridian), so its
# spectrum is 5.9 % of the maximum off the converged D2h one; torchspin (B82,
# 0.9.24) takes the strains into account and picks D2h.  The EasySpin comparison
# is kept as a strict xfail; test_multidstrain_converged_reference checks the new
# default against a fine D2h grid, and the same Dinfh grid still reproduces EasySpin.
EASYSPIN_GRID_B82 = ('multidstrain', 'multidstrain_31')
_B82_XFAIL = pytest.mark.xfail(
    strict=True, reason='B82: EasySpin picks Dinfh and ignores the E strain '
                        '(KNOWN_LIMITATIONS, automatic grid symmetry and strains)')

CASES = [
    'dstrain_explicit', 'dstraincorr_+0', 'dstraincorr_+1', 'dstraincorr_-1',
    'dstraincorr_+0_gs31', 'dstraincorr_+1_gs31', 'dstraincorr_+0_gs61', 'dstraincorr_+1_gs61',
    'dstraincorr_nostrain_gs15',
    *(pytest.param(n, marks=_B82_XFAIL) for n in EASYSPIN_GRID_B82),
    'two_triplets_dstrain_corr_61',
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


@pytest.mark.parametrize("name", EASYSPIN_GRID_B82)
def test_multidstrain_converged_reference(cases, name):
    """Default grid (now D2h) vs a fine D2h grid [61 4] (which agrees with [121 4]
    to 2.3e-4 of the maximum): 0.22 % / 0.07 % off; the Dinfh grid that EasySpin
    and torchspin < 0.9.24 used is 5.9 % off.  With that Dinfh grid forced,
    torchspin still reproduces EasySpin's spectrum (1e-4)."""
    c = cases[name]
    sys, exp, opt = sys_from_mat(c.Sys), exp_from_mat(c.Exp), opt_from_mat(c.Opt)
    y_auto = pepper(sys, exp, opt)[1].numpy()
    y_fine = pepper(sys, exp, Options(GridSize=[61, 4], GridSymmetry='D2h', Verbosity=0))[1].numpy()
    y_dinfh = pepper(sys, exp, Options(GridSize=opt.GridSize, GridSymmetry='Dinfh', Verbosity=0))[1].numpy()
    scale = np.abs(y_fine).max()
    assert np.abs(y_auto - y_fine).max() < 5e-3 * scale
    assert np.abs(y_dinfh - y_fine).max() > 5e-2 * scale         # the old grid
    ref = np.asarray(c.spc, dtype=float)
    assert cosine(y_dinfh, ref) > 0.999
    assert abs(np.abs(y_dinfh).max() / np.abs(ref).max() - 1) < 0.03
