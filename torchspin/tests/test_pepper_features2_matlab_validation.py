"""pepper feature validation against EasySpin (PORT_SPEC_2 items 3.4-3.6).

Reference: tests/data/ref_pepper_features2.mat from tests/data/generate_pepper_features2_refs.m
(EasySpin 6, MATLAB R2024b), mirroring EasySpin's pepper_noneqpop_* (zero-field, eigen,
xyz, coupled and density-matrix initial states; field and frequency sweeps),
pepper_ordering_* (built-in λ, user functions, tilted sample) and
pepper_photoselection* (lightBeam strings and {k alpha}, lightScatter, tdm forms).
"""
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from torchspin.tests.test_pepper_matlab_validation_ext2 import (
    sys_from_mat, exp_from_mat, opt_from_mat, cosine, _get, fine_grid_options, max_rel_dev)
from torchspin.pepper import pepper

REF = Path(__file__).resolve().parents[2] / 'tests' / 'data' / 'ref_pepper_features2.mat'


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


# photosel_custom_beam and photosel_tdmangles_* run on an interpolated C1 grid
# (default for Sys.tdm with low symmetry, [19 4]). EasySpin's C1 interpolation
# mirrors the lower hemisphere (gridinterp.m:211-219, B74), torchspin's does not
# since 0.9.21, so EasySpin is a known-wrong reference there and the EasySpin
# comparison is removed (0.9.10 agreed with it at cosine >= 0.99998). Reference:
# torchspin on the same fine knots without interpolation ([73 1]); name -> bound on
# max |dev| / max, measured x ~1.05. Measured 0.9.21 (0.9.10): custom_beam 7.792e-4
# (1.934e-3), tdmangles_parallel 3.132e-3 (3.556e-3), _perpendicular 4.280e-3 (6.132e-3).
C1_INTERP = {
    'photosel_custom_beam': 8.2e-4,
    'photosel_tdmangles_parallel': 3.3e-3,
    'photosel_tdmangles_perpendicular': 4.5e-3,
}

# name -> (cosine threshold, amplitude tolerance, reason)
LOOSE = {}


def _init_state(c):
    if _get(c, 'initDens') is not None:
        return np.asarray(c.initDens, dtype=complex)
    if _get(c, 'initPops') is not None:
        return (np.asarray(c.initPops, dtype=float).reshape(-1).tolist(), str(c.initBasis))
    return None


def _run(c, fine=False):
    sys = sys_from_mat(c.Sys, init_state=_init_state(c))
    exp = exp_from_mat(c.Exp, c.Sys, x_axis=np.asarray(c.B, dtype=float))
    opt = opt_from_mat(c.Opt)
    return pepper(sys, exp, fine_grid_options(opt) if fine else opt)


@pytest.mark.parametrize('name', [n for n in _case_names() if n not in C1_INTERP])
def test_pepper_features2_vs_matlab(cases, name):
    c = cases[name]
    B, spc = _run(c)
    B = B.detach().cpu().numpy(); spc = spc.detach().cpu().numpy()
    ref = np.asarray(c.spc, dtype=float)
    ref_B = np.asarray(c.B, dtype=float)
    assert np.allclose(B, ref_B, rtol=0, atol=1e-6 * max(1.0, abs(ref_B).max())), 'axis differs'
    cos_thr, amp_tol, _ = LOOSE.get(name, (0.999, 0.02, ''))
    assert spc.shape == ref.shape
    assert cosine(spc, ref) > cos_thr, (name, cosine(spc, ref))
    amp = abs(spc).max() / abs(ref).max()
    assert abs(amp - 1) < amp_tol, (name, amp)


@pytest.mark.parametrize('name', sorted(C1_INTERP))
def test_pepper_features2_c1_vs_no_interpolation(cases, name):
    """Interpolated C1 grid vs the same fine knots without interpolation (B74)."""
    c = cases[name]
    assert opt_from_mat(c.Opt).grid_size_interp > 1
    y = _run(c)[1].detach().cpu().numpy()
    ref = _run(c, fine=True)[1].detach().cpu().numpy()
    dev = max_rel_dev(y, ref)
    assert dev < C1_INTERP[name], (name, dev)
