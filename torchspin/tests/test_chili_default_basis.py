"""chili's automatic basis symmetry must reproduce EasySpin's default (fast-builder)
output in the slow-motion limit, where the plain general basis differs (cosine 0.937)."""
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from torchspin import SpinSystem, Experiment, chili
from torchspin.chili import ChiliOptions

REF = Path(__file__).resolve().parents[2] / 'tests' / 'data' / 'example_nitroxide_tcorr.mat'
# measured 8.5e-4 (0.9.32: 0.40); the default (symmetry-basis Lanczos) is 2.1e-4 from exact
PLAIN_LANCZOS_RELMAX = 2e-3


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
    # The plain general basis (MpSymm=False) gives the same spectrum.  Until the B116 fix this
    # asserted 0.93 < cosine < 0.95: the window encoded the Lanczos stopping early at a
    # near-breakdown (step 40), as EasySpin's general method also does.  The converged
    # plain-basis result matches EasySpin's default output: cosine 0.9999925 (measured).
    sys = SpinSystem(S=[0.5], g=[[2.0089, 2.0058, 2.0021]], Nucs=['14N'], A=[[16.0, 16.0, 100.0]], tcorr=1e-7, lw=[0.0, 0.05])
    _, y_gen = chili(sys, exp, ChiliOptions(MpSymm=False))
    y_gen = y_gen.numpy()
    assert _cos(y_gen, ref['y_all'][0]) > 0.9999, f'plain basis: cosine {_cos(y_gen, ref["y_all"][0]):.7f}'
    # ... and its Lanczos stops where the spectrum is converged: compare with the exact
    # solution (eigenvalue solver on the symmetry basis, equal to the plain basis solved
    # directly to roundoff, see test_chili_mpsymm.py).
    _, y_exact = chili(sys, exp, ChiliOptions(MpSymm=True, Solver='E'))
    y_exact = y_exact.numpy()
    relmax = float(np.max(np.abs(y_gen - y_exact)) / np.max(np.abs(y_exact)))
    assert relmax < PLAIN_LANCZOS_RELMAX, f'plain-basis Lanczos vs exact: relmax {relmax:.2e}'
