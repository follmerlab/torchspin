"""pepper with non-equilibrium populations (``Sys.initState``) and strain broadening.

Up to 0.9.39 any HStrain, gStrain or DStrain together with ``initState`` dropped
every emissive (negative) line: the strain summation kept only facets of positive
intensity. EasySpin keeps signed intensities when the populations are not thermal.

Reference: tests/data/ref_pepper_initstate_strain.mat from
tests/data/generate_pepper_initstate_strain_refs.m (EasySpin 6, MATLAB R2026a).
The self-consistency tests below need no reference file.
"""
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from torchspin import Experiment, Options, SpinSystem, pepper
from torchspin.tests.test_pepper_matlab_validation_ext2 import (
    _get, cosine, exp_from_mat, opt_from_mat, sys_from_mat)

REF = Path(__file__).resolve().parents[2] / 'tests' / 'data' / 'ref_pepper_initstate_strain.mat'

EXP = Experiment(mwFreq=9.7, Range=[290, 400], nPoints=1024, Harmonic=0)
POPS = ([0.1, 0.3, 0.6], 'xyz')


def _triplet(pops=POPS, **kw):
    return SpinSystem(S=1, g=2.003, D=[1500, 300], lwpp=[1, 0], initState=pops, **kw)


def _spc(sys, exp=EXP, **opt):
    return pepper(sys, exp, Options(Verbosity=0, **opt))[1].numpy()


@pytest.mark.parametrize('strain', [{'HStrain': [30, 30, 30]}, {'DStrain': [30, 10]},
                                    {'gStrain': [0.002, 0.002, 0.004]}])
def test_strain_keeps_emission(strain):
    """The polarized triplet has emissive lines with and without strain."""
    ref = _spc(_triplet())
    spc = _spc(_triplet(**strain))
    assert ref.min() < -0.5 * ref.max()
    assert spc.min() < -0.5 * spc.max()
    # net intensity (absorption minus emission) is unchanged by a small strain
    assert spc.sum() == pytest.approx(ref.sum(), rel=0.01)


def test_vanishing_strain_matches_no_strain():
    """A strain far below the line width is the strain-free spectrum."""
    ref = _spc(_triplet())
    spc = _spc(_triplet(HStrain=[1e-6, 1e-6, 1e-6]))
    assert cosine(spc, ref) > 0.9999
    assert np.max(np.abs(spc - ref)) < 0.01 * np.max(np.abs(ref))


def test_population_mirror_negates_spectrum():
    """Populations p' = 2/3 - p reverse every population difference, so the
    spectrum (with strain) changes sign exactly."""
    p = np.array(POPS[0])
    a = _spc(_triplet(HStrain=[30, 30, 30]))
    b = _spc(_triplet(pops=((2 / 3 - p).tolist(), 'xyz'), HStrain=[30, 30, 30]))
    np.testing.assert_allclose(b, -a, rtol=0, atol=1e-9 * np.max(np.abs(a)))


def _cases():
    if not REF.exists():
        return {}
    m = loadmat(REF, squeeze_me=True, struct_as_record=False)
    return {str(c.name): c for c in np.atleast_1d(m['cases'])}


@pytest.mark.parametrize('name', list(_cases()))
def test_initstate_strain_vs_matlab(name):
    c = _cases()[name]
    sys = sys_from_mat(c.Sys, init_state=(np.asarray(c.initPops, float).reshape(-1).tolist(),
                                           str(c.initBasis)))
    exp = exp_from_mat(c.Exp, c.Sys, x_axis=np.asarray(c.B, dtype=float))
    B, spc = pepper(sys, exp, opt_from_mat(c.Opt))
    spc = spc.numpy()
    ref = np.asarray(c.spc, dtype=float)
    assert np.allclose(B.numpy(), np.asarray(c.B, float), rtol=0, atol=1e-6 * np.abs(c.B).max())
    assert ref.min() < 0 and spc.min() < 0
    assert cosine(spc, ref) > 0.999
    assert np.max(np.abs(spc)) == pytest.approx(np.max(np.abs(ref)), rel=0.02)
