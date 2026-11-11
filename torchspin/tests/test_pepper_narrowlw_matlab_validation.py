"""Final ``Sys.lw`` convolution in pepper against EasySpin (B13;
tests/data/ref_pepper_narrowlw.mat, generate_pepper_narrowlw_refs.m).

* A line width of at most two increments with harmonic 0 is not convolved
  (pepper.m:1195-1223): an under-resolved triplet frequency sweep (default and
  ``Transitions [1 2]``) and field sweeps with a narrow Gaussian / Lorentzian.
  Before the fix torchspin convolved these and its amplitudes were 1-7 % low.
* With a Voigt width the Lorentzian carries the harmonic and ``Exp.mwPhase``
  (pepper.m:1152-1163), and a narrow Gaussian next to it is skipped.

Also: autograd vs central differences through the Voigt path, and the
torchspin extension that a narrow width with a harmonic is still convolved
(EasySpin raises)."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io import loadmat

from torchspin.tests.test_pepper_matlab_validation_ext2 import (
    sys_from_mat, exp_from_mat, opt_from_mat, cosine, _get)
from torchspin.pepper import pepper
from torchspin.spinsystem import SpinSystem
from torchspin.experiment import Experiment, Options

REF = Path(__file__).resolve().parents[2] / 'tests' / 'data' / 'ref_pepper_narrowlw.mat'


def _load():
    m = loadmat(REF, squeeze_me=True, struct_as_record=False)
    return {str(c.name): c for c in np.atleast_1d(m['cases'])}


@pytest.fixture(scope='module')
def cases():
    if not REF.exists():
        pytest.skip(f'reference file missing: {REF}')
    return _load()


# Measured since 0.8.9 (B25, closed-phi row spline with zero end slopes):
# |amplitude - 1| <= 2e-7 and cosine 1 - 1e-9 for all eight cases.  Before,
# the field-sweep Voigt cases at Harmonic 1 were 1.0020, 1.0020 and 1.0048
# (tolerance 0.006, the others 0.002) and the rest within 4.1e-5.  The skip
# itself is checked exactly in test_narrow_gaussian_next_to_lorentzian_is_skipped.
AMP_TOL = 1e-5
COS_MIN = 1 - 1e-7


def _case_names():
    return list(_load().keys()) if REF.exists() else []


def _transitions(O):
    t = _get(O, 'Transitions')
    if t is None:
        return None
    a = np.atleast_2d(np.asarray(t, dtype=int))
    return [tuple(int(v) for v in row) for row in a]


@pytest.mark.parametrize('name', _case_names())
def test_pepper_narrowlw_vs_matlab(cases, name):
    c = cases[name]
    exp = exp_from_mat(c.Exp, c.Sys, x_axis=np.asarray(c.x, dtype=float))
    opt = replace(opt_from_mat(c.Opt), Transitions=_transitions(c.Opt))
    x, spc = pepper(sys_from_mat(c.Sys), exp, opt)
    x = x.detach().cpu().numpy(); spc = spc.detach().cpu().numpy()
    ref = np.asarray(c.spc, dtype=float); ref_x = np.asarray(c.x, dtype=float)
    assert np.allclose(x, ref_x, rtol=1e-9, atol=1e-9 * abs(ref_x).max()), 'sweep axis differs'
    cs = cosine(spc, ref)
    amp = abs(spc).max() / abs(ref).max()
    print(f'{name}: cosine {cs:.7f}, amplitude ratio {amp:.5f}')
    assert cs > COS_MIN, (name, cs)
    assert amp == pytest.approx(1.0, abs=AMP_TOL), (name, amp)


def test_narrow_gaussian_next_to_lorentzian_is_skipped():
    """With a Voigt width the Gaussian has harmonic 0, so a Gaussian of at most
    two increments is not convolved: same spectrum as the Lorentzian alone."""
    exp = Experiment(mwFreq=9.5, Range=[320, 345], nPoints=1024, Harmonic=1)
    opt = Options(Verbosity=0)
    voigt = pepper(SpinSystem(S=0.5, g=[2.0, 2.05, 2.1], lw=[0.03, 0.8]), exp, opt)[1]
    lor = pepper(SpinSystem(S=0.5, g=[2.0, 2.05, 2.1], lw=[0.0, 0.8]), exp, opt)[1]
    assert torch.equal(voigt, lor)


def test_narrow_width_with_harmonic_still_convolved():
    """torchspin extension: EasySpin raises for FWHM <= 2 increments with a
    harmonic; torchspin convolves (the derivative needs the line shape)."""
    sys = SpinSystem(S=0.5, g=[2.0, 2.05, 2.1], lw=[0.03, 0.0])
    exp = Experiment(mwFreq=9.5, Range=[320, 345], nPoints=1024, Harmonic=1)
    _, spc = pepper(sys, exp, Options(Verbosity=0))
    assert torch.isfinite(spc).all() and spc.abs().max() > 0


def test_voigt_autograd_vs_central_differences():
    """pepper_autograd follows the same assignment (Lorentzian carries the
    harmonic); d spc / d lw matches central differences."""
    from torchspin.pepper_autograd import pepper_autograd
    exp = Experiment(mwFreq=9.5, Range=[320, 345], nPoints=512, Harmonic=1)
    opt = Options(Verbosity=0)

    def run(lw_g, lw_l):
        sys = SpinSystem(S=0.5, g=[2.0, 2.05, 2.1], lw=[lw_g, lw_l])
        return pepper_autograd(sys, exp, opt)[1]

    lw = torch.tensor([1.0, 0.8], dtype=torch.float64, requires_grad=True)
    spc = run(lw[0], lw[1])
    ref = pepper(SpinSystem(S=0.5, g=[2.0, 2.05, 2.1], lw=[1.0, 0.8]), exp, opt)[1]
    assert torch.allclose(spc.detach(), ref, rtol=1e-9, atol=1e-9 * ref.abs().max())
    w = torch.linspace(-1.0, 1.0, spc.numel(), dtype=torch.float64)
    (grad,) = torch.autograd.grad((spc * w).sum(), lw)
    h = 1e-5
    for k in range(2):
        e = [1.0, 0.8]; e[k] += h; fp = (run(*e).detach() * w).sum()
        e = [1.0, 0.8]; e[k] -= h; fm = (run(*e).detach() * w).sum()
        fd = (fp - fm) / (2 * h)
        assert float(grad[k]) == pytest.approx(float(fd), rel=1e-5), (k, float(grad[k]), float(fd))
