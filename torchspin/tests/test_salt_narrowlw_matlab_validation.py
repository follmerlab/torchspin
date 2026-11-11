"""Final ENDOR line-width convolution in salt against EasySpin (B24, 0.8.11;
tests/data/salt_narrowlw_refs.mat, generate_salt_narrowlw_refs.m).

EasySpin's salt.m (884-911) does not convolve a ``Sys.lwEndor`` component whose
FWHM is at most two increments when its harmonic is 0: such a line is
delta-like.  Before 0.8.11 torchspin convolved ``lw_mhz`` anyway, which lowered
the peak of an under-resolved spectrum (1H, g = [2 2.05 2.1], A = [5 8 12] MHz,
2-28 MHz in 0.02 MHz increments): 0.933 of EasySpin at 1.5 increments, 0.900
at 2.0.  Wider lines (2.5 and 10 increments) were already exact.

The references are fixed-field powder spectra (EasySpin salt has no field-swept
mode; torchspin's ``Experiment.Range`` mode applies the same rule and is checked
here for the exact skip).  torchspin's fixed-field salt returns EasySpin's
spectrum times ``dx / (8 pi^2)`` (test_salt_first_order_matlab_validation.py).
"""
import math
import os

import numpy as np
import pytest
import torch

from torchspin.experiment import Experiment, Options
from torchspin.salt import salt
from torchspin.spinsystem import SpinSystem

try:
    from scipy.io import loadmat
except ImportError:  # pragma: no cover
    loadmat = None

REF = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "data", "salt_narrowlw_refs.mat")
NAMES = ['lw_0', 'lw_1p5dx', 'lw_2p0dx', 'lw_2p5dx', 'lw_10dx']


@pytest.fixture(scope='module')
def cases():
    if loadmat is None:
        pytest.skip("scipy not installed")
    if not os.path.exists(REF):
        pytest.skip(f"Reference file not found: {REF}")
    m = loadmat(REF, simplify_cells=True)
    return {str(c['name']): c for c in np.atleast_1d(m['cases'])}


def _sys(S):
    return SpinSystem(S=[float(S['S'])], g=[np.asarray(S['g'], float).tolist()], Nucs=S['Nucs'],
                      A=[np.asarray(S['A'], float).tolist()])


@pytest.mark.parametrize('name', NAMES)
def test_salt_narrowlw_vs_easyspin(cases, name):
    c = cases[name]
    S, E = c['Sys'], c['Exp']
    lw = np.ravel(np.asarray(S['lwEndor'], float))
    assert lw[1] == 0.0                                     # Gaussian widths only (salt has no Lorentzian)
    x, y = salt(_sys(S), Experiment(Field=float(E['Field'])), Options(GridSize=int(c['Opt']['GridSize'])),
                freq_range=tuple(np.asarray(E['Range'], float)), n_points=int(E['nPoints']),
                lw_mhz=float(lw[0]))
    x, y = x.numpy(), y.numpy()
    ref_x, ref = np.asarray(c['x'], float), np.asarray(c['spc'], float)
    np.testing.assert_allclose(x, ref_x, rtol=0, atol=1e-12)
    dx = x[1] - x[0]
    y = y / (dx / (8 * math.pi ** 2))                       # on EasySpin's scale
    # measured: |y - ref| <= 4e-13 of the peak in every case (before the fix the
    # 1.5- and 2.0-increment widths were off by 0.067 and 0.100 of the peak)
    err = float(np.abs(y - ref).max() / np.abs(ref).max())
    print(f'{name}: lw {lw[0]:g} MHz = {lw[0] / dx:.1f} increments, max |diff| / peak {err:.1e}')
    assert err < 1e-9, (name, err)


def _fixed(lw, n_points=1301):
    sys = SpinSystem(S=[0.5], g=[[2.0, 2.05, 2.1]], Nucs='1H', A=[[5.0, 8.0, 12.0]])
    return salt(sys, Experiment(Field=340.0), Options(GridSize=11), freq_range=(2, 28),
                n_points=n_points, lw_mhz=lw)[1]


def test_fixed_field_narrow_width_is_skipped_exactly():
    """A width of at most two increments gives the unbroadened spectrum, bit for bit."""
    dx = 26 / 1300
    y0 = _fixed(0.0)
    for k in (0.5, 1.5, 2.0):
        assert torch.equal(_fixed(k * dx), y0), k
    assert not torch.equal(_fixed(2.0001 * dx), y0)


def test_field_swept_narrow_width_is_skipped_exactly():
    """Field-swept (Experiment.Range) mode, a torchspin extension: same rule."""
    sys = SpinSystem(S=[0.5], g=[[2.0, 2.01, 2.02]], Nucs='1H', A=[[-2.0, -2.0, 4.0]])
    exp = Experiment(Range=[340, 345], mwFreq=9.6)

    def run(lw):
        return salt(sys, exp, Options(GridSize=7), freq_range=(10, 20), n_points=251, lw_mhz=lw)[1]
    dx = 10 / 250
    y0 = run(0.0)
    assert torch.equal(run(1.5 * dx), y0)
    assert torch.equal(run(2.0 * dx), y0)
    assert not torch.equal(run(2.5 * dx), y0)


def test_skipped_width_has_zero_gradient():
    """A skipped width stays in the autograd graph with zero gradient (the
    spectrum does not depend on it), so autograd.grad does not raise."""
    dx = 26 / 1300
    w = torch.linspace(-1.0, 1.0, 1301, dtype=torch.float64)
    lw = torch.tensor(1.5 * dx, dtype=torch.float64, requires_grad=True)
    y = _fixed(lw)
    assert y.requires_grad
    (g,) = torch.autograd.grad((y * w).sum(), lw)
    assert float(g) == 0.0
    h = 1e-5
    assert float(((_fixed(1.5 * dx + h) - _fixed(1.5 * dx - h)) * w).sum()) == 0.0


def test_width_above_threshold_gradient_vs_central_differences():
    """Just above the threshold (2.5 increments) d spc / d lw matches central differences."""
    dx = 26 / 1300
    lw0 = 2.5 * dx
    w = torch.linspace(-1.0, 1.0, 1301, dtype=torch.float64) ** 2
    lw = torch.tensor(lw0, dtype=torch.float64, requires_grad=True)
    (g,) = torch.autograd.grad((_fixed(lw) * w).sum(), lw)
    h = 1e-6
    fd = float(((_fixed(lw0 + h) - _fixed(lw0 - h)) * w).sum()) / (2 * h)
    assert fd != 0.0
    assert float(g) == pytest.approx(fd, rel=1e-6), (float(g), fd)
