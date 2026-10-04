"""chili's automatic basis symmetry must reproduce EasySpin's default (fast-builder)
output in the slow-motion limit, where the plain general basis differs (cosine 0.937)."""
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from torchspin import SpinSystem, Experiment, chili
from torchspin.chili import ChiliOptions

REF = Path(__file__).resolve().parents[2] / 'tests' / 'data' / 'example_nitroxide_tcorr.mat'


def _cos(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


@pytest.mark.skipif(not REF.exists(), reason='reference file missing')
def test_default_basis_matches_easyspin_default_across_tcorr():
    ref = loadmat(str(REF), squeeze_me=True)
    exp = Experiment(mwFreq=9.5, Range=[328, 352], nPoints=2048, Harmonic=1)
    for tc, y_ref in zip(ref['tcorr_list'], ref['y_all']):
        sys = SpinSystem(S=[0.5], g=[[2.0089, 2.0058, 2.0021]], Nucs=['14N'], A=[[16.0, 16.0, 100.0]], tcorr=float(tc), lw=[0.0, 0.05])
        _, y = chili(sys, exp)
        assert _cos(y, y_ref) > 0.9999, f'tcorr {tc:.0e}: cosine {_cos(y, y_ref):.5f}'
    # the explicit general basis reproduces EasySpin's general method, which differs at 100 ns
    sys = SpinSystem(S=[0.5], g=[[2.0089, 2.0058, 2.0021]], Nucs=['14N'], A=[[16.0, 16.0, 100.0]], tcorr=1e-7, lw=[0.0, 0.05])
    _, y_gen = chili(sys, exp, ChiliOptions(MpSymm=False))
    assert 0.93 < _cos(y_gen, ref['y_all'][0]) < 0.95
