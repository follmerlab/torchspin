"""A one-element D is one D value with E = 0, as in EasySpin (B52).

EasySpin's validatespinsys.m:208-210 reads ``numel(Sys.D) == nElectrons`` as
one D value per electron (E = 0), so for one electron ``Sys.D = [300]`` is the
same as ``Sys.D = 300``; for several electrons a single value is
'Sys.D has wrong size.'  Before 0.9.12 ``SpinSystem(S=[2.5], D=[300.])`` kept
D as a length-1 vector and pepper failed in ham_zf with
'IndexError: too many indices for tensor of dimension 1'.  A D of any other
shape that is not normalized to (nElectrons, 3) or (3*nElectrons, 3) now raises
a ValueError in the constructor instead of an IndexError later.
"""
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io import loadmat

from torchspin.experiment import Experiment, Options
from torchspin.pepper import pepper
from torchspin.spinsystem import SpinSystem
from torchspin.tests.test_pepper_matlab_validation_ext2 import (
    sys_from_mat, exp_from_mat, opt_from_mat, cosine)

REF = Path(__file__).resolve().parents[2] / 'tests' / 'data' / 'ref_pepper_ext2.mat'


@pytest.mark.parametrize('D1', [[300.0], torch.tensor([300.0])])
def test_one_element_D_equals_scalar_D(D1):
    a = SpinSystem(S=[2.5], D=D1)
    b = SpinSystem(S=[2.5], D=300.0)
    assert a.D.shape == (1, 3)
    assert torch.equal(a.D, b.D)
    exp = Experiment(Range=[0, 800], mwFreq=9.5, nPoints=512)
    opt = Options(GridSize=10)
    xa, ya = pepper(a, exp, opt)
    xb, yb = pepper(b, exp, opt)
    assert ya.abs().max() > 0
    assert torch.equal(ya, yb)


def test_one_element_D_keeps_gradient():
    D = torch.tensor([300.0], dtype=torch.float64, requires_grad=True)
    s = SpinSystem(S=[2.5], D=D)
    (s.D * torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)).sum().backward()
    # d/dD of D*(-1/3*1 - 1/3*2 + 2/3*3) = 1
    assert torch.allclose(D.grad, torch.tensor([1.0], dtype=torch.float64))


@pytest.mark.parametrize('name', ['perturb_highspin_matrix', 'fulld_scalar'])
def test_one_element_D_vs_easyspin(name):
    """EasySpin cases with a scalar Sys.D (a 1x1 array in MATLAB), run with D = [D]."""
    if not REF.exists():
        pytest.skip(f'reference file missing: {REF}')
    m = loadmat(REF, squeeze_me=True, struct_as_record=False)
    c = {str(k.name): k for k in np.atleast_1d(m['cases'])}[name]
    Dval = float(c.Sys.D)
    s = sys_from_mat(c.Sys)
    s.D = [Dval]
    exp = exp_from_mat(c.Exp, c.Sys)
    opt = opt_from_mat(c.Opt)
    x, spc = pepper(s, exp, opt)
    spc = spc.numpy()
    ref = np.asarray(c.spc, dtype=float)
    # same thresholds as test_pepper_matlab_validation_ext2
    assert cosine(spc, ref) > 0.999
    assert abs(np.abs(spc).max() / np.abs(ref).max() - 1) < 0.02
    # and identical to the scalar-D system
    s0 = sys_from_mat(c.Sys)
    _, spc0 = pepper(s0, exp, opt)
    assert np.array_equal(spc, spc0.numpy())


@pytest.mark.parametrize('D', [[300.0], [[300.0]]])
def test_one_element_D_several_electrons_raises(D):
    """EasySpin: 'Sys.D has wrong size.' (one value for two electrons)."""
    with pytest.raises(ValueError, match='D has wrong size'):
        SpinSystem(S=[1, 1], D=D, ee=[0.0, 0.0, 0.0])


def test_unnormalizable_D_raises():
    """A length-6 vector for two electrons was kept as is (IndexError in ham_zf)."""
    with pytest.raises(ValueError, match='D has wrong size'):
        SpinSystem(S=[1, 1], D=[-100.0, -100.0, 200.0, -50.0, -50.0, 100.0], ee=[0.0, 0.0, 0.0])
