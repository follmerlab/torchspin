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
    sys_from_mat, exp_from_mat, opt_from_mat, cosine, _get)
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


# photosel_custom_beam and photosel_tdmangles_* run on a C1 grid with interpolation,
# where torchspin deliberately deviates from EasySpin since 0.9.21 (B74: EasySpin
# mirrors the interpolated lower hemisphere). Measured cosine / amplitude 0.9.10 ->
# 0.9.21: custom_beam 0.9999983/1.0073 -> 0.9999981/1.0085, tdmangles_parallel
# 0.9999889/1.0057 -> 0.9999890/1.0043, _perpendicular 0.9999973/0.9980 ->
# 0.9999966/0.9995; vs the same fine grid without interpolation (torchspin [73 1])
# 1.9e-3 -> 7.8e-4, 3.6e-3 -> 3.1e-3, 6.1e-3 -> 4.3e-3 of the maximum.
# name -> (cosine threshold, amplitude tolerance, reason)
LOOSE = {}


def _init_state(c):
    if _get(c, 'initDens') is not None:
        return np.asarray(c.initDens, dtype=complex)
    if _get(c, 'initPops') is not None:
        return (np.asarray(c.initPops, dtype=float).reshape(-1).tolist(), str(c.initBasis))
    return None


def _run(c):
    sys = sys_from_mat(c.Sys, init_state=_init_state(c))
    exp = exp_from_mat(c.Exp, c.Sys, x_axis=np.asarray(c.B, dtype=float))
    return pepper(sys, exp, opt_from_mat(c.Opt))


@pytest.mark.parametrize('name', _case_names())
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
