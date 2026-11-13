"""Range padding before the final line-width convolution against EasySpin (B34,
0.9.6; tests/data/range_padding_refs.mat, generate_range_padding_refs.m).

When the spectrum is nonzero at the first or last point of the sweep range,
EasySpin's salt.m (858-882) and pepper.m (1171-1193) extend both ends by
``nPad = round(max([fwhmG fwhmL])/deltaX*Opt.PaddingMultiplier)`` copies of the
edge values before convolving and crop afterwards.  Before 0.9.6 torchspin's
convolution zero-extended, so the points within about two widths of such an end
came out low: end points 0.45-0.56 of EasySpin's in salt and in a pepper
frequency sweep, and 2.9-3.2 times EasySpin's in a derivative field sweep with
HStrain.  ``Options.PaddingMultiplier`` (default 3, 0 = off) is EasySpin's
``Opt.PaddingMultiplier``; each case is checked at 3, 0 and 1.

torchspin's fixed-field salt returns EasySpin's spectrum times ``dx / (8 pi^2)``
(test_salt_first_order_matlab_validation.py).
"""
import math
import os

import numpy as np
import pytest
import torch

from torchspin.experiment import Experiment, Options
from torchspin.pepper import pepper
from torchspin.pepper_autograd import pepper_autograd
from torchspin.salt import salt
from torchspin.spinsystem import SpinSystem

try:
    from scipy.io import loadmat
except ImportError:  # pragma: no cover
    loadmat = None

REF = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "data", "range_padding_refs.mat")
PM = {'spc': 3, 'spc_pm0': 0, 'spc_pm1': 1}

# Measured max |torchspin - EasySpin| / peak with the fix (all three multipliers);
# before the fix (default multiplier) in parentheses:
#   salt_1H_perturb1    3.3e-13  (0.44; end points 0.562, 0.552 of EasySpin's)
#   salt_1H14N_matrix   1.8e-4   (0.23; end points 0.454, 0.562); the 1.8e-4 is
#                       in the interior and is the same with and without padding
#   pepper_freq_L_h1    3.0e-13  (0.32; end points 0.538, 0.548)
#   pepper_strain_G_h1  1.8e-4   (0.80; end points 2.94, 3.21); interior HStrain
#                       floor, the same with and without padding
TOL = {'salt_1H_perturb1': 1e-9, 'salt_1H14N_matrix': 1e-3,
       'pepper_freq_L_h1': 1e-9, 'pepper_strain_G_h1': 1e-3}


@pytest.fixture(scope='module')
def cases():
    if loadmat is None:
        pytest.skip("scipy not installed")
    if not os.path.exists(REF):
        pytest.skip(f"Reference file not found: {REF}")
    m = loadmat(REF, simplify_cells=True)
    return {str(c['name']): c for c in np.atleast_1d(m['cases'])}


def _a2(v):
    return np.atleast_2d(np.asarray(v, float)).tolist()


def _opt(c, pm):
    return Options(GridSize=[int(v) for v in np.ravel(c['Opt']['GridSize'])], PaddingMultiplier=pm)


def _salt(c, pm):
    S, E = c['Sys'], c['Exp']
    kw = dict(S=[0.5], g=_a2(S['g']), Nucs=S['Nucs'], A=_a2(S['A']))
    if 'Q' in S:
        kw['Q'] = _a2(S['Q'])
    x, y = salt(SpinSystem(**kw), Experiment(Field=float(E['Field'])), _opt(c, pm),
                freq_range=tuple(np.asarray(E['Range'], float)), n_points=int(E['nPoints']),
                lw_mhz=float(S['lwEndor']))
    x, y = x.numpy(), y.numpy()
    return x, y / ((x[1] - x[0]) / (8 * math.pi ** 2))      # on EasySpin's scale


def _pepper_inputs(c):
    S, E = c['Sys'], c['Exp']
    skw = dict(S=[0.5], g=_a2(S['g']), lw=np.ravel(S['lw']).tolist())
    if 'HStrain' in S:
        skw['HStrain'] = np.ravel(S['HStrain']).tolist()
    ekw = {k: (np.asarray(v, float).tolist() if np.ndim(v) else float(v)) for k, v in E.items()}
    ekw['nPoints'] = int(ekw['nPoints'])
    return SpinSystem(**skw), Experiment(**ekw)


def _pepper(c, pm, fun=pepper):
    s, e = _pepper_inputs(c)
    x, y = fun(s, e, _opt(c, pm))
    return np.asarray(x.detach()).ravel(), np.asarray(y.detach()).ravel()


@pytest.mark.parametrize('key', list(PM))
@pytest.mark.parametrize('name', list(TOL))
def test_range_padding_vs_easyspin(cases, name, key):
    c = cases[name]
    runs = [('salt', _salt(c, PM[key]))] if name.startswith('salt') else [('pepper', _pepper(c, PM[key]))]
    if name == 'pepper_strain_G_h1':
        runs.append(('pepper_autograd', _pepper(c, PM[key], pepper_autograd)))
    ref_x, ref = np.asarray(c['x'], float), np.asarray(c[key], float)
    # EasySpin itself pads here: the multiplier changes its spectrum
    assert np.abs(np.asarray(c['spc'], float) - np.asarray(c['spc_pm0'], float)).max() > 0.1 * np.abs(ref).max()
    for fn, (x, y) in runs:
        np.testing.assert_allclose(x, ref_x, rtol=0, atol=1e-9)
        err = float(np.abs(y - ref).max() / np.abs(ref).max())
        print(f'{name} {fn} PaddingMultiplier {PM[key]}: max |diff| / peak {err:.1e}, '
              f'end points {y[0] / ref[0]:.4f} {y[-1] / ref[-1]:.4f} of EasySpin')
        assert err < TOL[name], (name, fn, key, err)


@pytest.mark.parametrize('fun', [pepper, pepper_autograd])
def test_voigt_field_sweep_padded_since_b88(cases, fun):
    """Rhombic Voigt field sweep cut by the range (lw = [0.5 1.5] mT).

    In EasySpin (and torchspin up to 0.9.27) the spectrum before the convolution
    is zero at both ends, because the triangles that cross the range ends are
    dropped (B75), so it is not padded and does not depend on
    Opt.PaddingMultiplier.  Since 0.9.28 (B88) those triangles are clipped, the
    ends carry intensity, and the padding applies.  Against the same system swept
    60 mT wider on both sides without interpolation (D2h [121 1], cropped): cosine
    0.994571 padded vs 0.908229 unpadded and 0.856951 for EasySpin."""
    c = cases['pepper_V_h1']
    assert np.array_equal(np.asarray(c['spc'], float), np.asarray(c['spc_pm0'], float))
    _, y3 = _pepper(c, 3, fun)
    _, y0 = _pepper(c, 0, fun)
    assert not np.array_equal(y3, y0)
    s, e = _pepper_inputs(c)
    r0, r1 = e.Range
    n = e.nPoints
    dx = (r1 - r0) / (n - 1)
    k = int(round(60 / dx))
    wide = Experiment(mwFreq=e.mwFreq, Range=[r0 - k * dx, r1 + k * dx], nPoints=n + 2 * k, Harmonic=e.Harmonic)
    ref = np.asarray(pepper(s, wide, Options(GridSymmetry='D2h', GridSize=[121, 1]))[1], float)[k:k + n]

    def cos(a):
        return float(a @ ref / np.linalg.norm(a) / np.linalg.norm(ref))
    assert cos(y3) > 0.9945, cos(y3)
    assert cos(y3) > cos(y0) and cos(y3) > cos(np.asarray(c['spc'], float))


# ── decision and count ─────────────────────────────────────────────────────

def _count(*a, **k):
    import sys
    return sys.modules['torchspin.pepper']._edge_pad_count(*a, **k)


def test_pad_count_matlab_rounding_and_tolerance():
    spec = torch.zeros(50, dtype=torch.float64)
    spec[10:40] = 1.0
    assert _count(spec, 1.0, 2.5, 0.0, 1.0) == 0              # zero ends: no padding
    spec[0] = 1e-12                                            # round-off floor: not an edge
    assert _count(spec, 1.0, 2.5, 0.0, 1.0) == 0
    spec[0] = 1e-6
    assert _count(spec, 1.0, 2.5, 0.0, 1.0) == 3              # MATLAB round(2.5) = 3 (Python: 2)
    assert _count(spec, 1.0, 0.0, 2.5, 3.0) == 8              # round(7.5), Lorentzian width
    assert _count(spec, 1.0, 1.0, 2.0, 3.0) == 6              # max of the two widths
    assert _count(spec, 1.0, 2.5, 0.0, 0.0) == 0              # PaddingMultiplier 0: off
    assert _count(-spec, 1.0, 2.5, 0.0, 1.0) == 3             # sign does not matter


def test_pad_decision_over_all_rows():
    """As EasySpin's any(spec(:,1)~=0) over the rows of spec: one row reaching an
    end pads every row by the same count (each row relative to its own maximum)."""
    import sys
    pep = sys.modules['torchspin.pepper']
    a = torch.zeros(80, dtype=torch.float64); a[30:50] = 1.0
    b = torch.zeros(80, dtype=torch.float64); b[:20] = 1e-3        # reaches the low end
    two = torch.stack([a, b])
    assert _count(two, 0.1, 0.5, 0.0) == 15
    assert _count(a, 0.1, 0.5, 0.0) == 0
    exp = Experiment(mwFreq=9.5, Range=[300, 307.9], nPoints=80, Harmonic=1)
    x = torch.linspace(300, 307.9, 80, dtype=torch.float64)
    out = pep._finish_field_sweep(two, x, 0.1, exp, 0.5, 0.0, False)
    for k, row in enumerate((a, b)):
        ref = pep._lw_convolve(row, 0.1, 0.5, 0.0, 1, n_pad=15)
        assert torch.equal(out[k], ref)
    # row a alone is not padded, so its result differs from the batched one
    assert not torch.equal(pep._finish_field_sweep(a, x, 0.1, exp, 0.5, 0.0, False), out[0])


@pytest.mark.parametrize('bad', [-1, -0.5, float('nan'), float('inf'), True, '3', None])
def test_padding_multiplier_validation(bad):
    with pytest.raises(ValueError, match='PaddingMultiplier'):
        Options(PaddingMultiplier=bad)


def test_padding_multiplier_default():
    assert Options().PaddingMultiplier == 3
    Options(PaddingMultiplier=0)
    Options(PaddingMultiplier=np.float64(1.5))


# ── gradients with padding active ──────────────────────────────────────────

def _w(n):
    return torch.linspace(0, 1, n, dtype=torch.float64) ** 2   # weights the high end


def _salt_y(A, lw):
    s = SpinSystem(S=[0.5], g=[[2, 2.05, 2.1]], Nucs='1H', A=A)
    return salt(s, Experiment(Field=330), Options(GridSize=[19, 4]), freq_range=(10, 19),
                n_points=451, lw_mhz=lw)[1]


def test_salt_gradients_with_padding():
    """Fixed-field salt on 10-19 MHz (spectrum reaches both ends, nPad = 30):
    autograd vs central differences (h = 1e-4) for A and lw.  Measured relative
    errors 2.0e-6, 2.6e-6, 4.1e-10 (A) and 7.0e-9 (lw); the first five points
    alone (edge region, lw) 6.6e-8."""
    A0 = [[5.0, 8.0, 12.0]]
    n = 451
    spec0 = _salt_y(A0, 0.0)
    assert _count(spec0, 9 / 450, 0.2, 0.0) == 30            # padding is active
    A = torch.tensor(A0, dtype=torch.float64, requires_grad=True)
    lw = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    w = _w(n)
    gA, glw = torch.autograd.grad((_salt_y(A, lw) * w).sum(), (A, lw))
    h = 1e-4
    for j in range(3):
        Ap = torch.tensor(A0, dtype=torch.float64); Ap[0, j] += h
        Am = torch.tensor(A0, dtype=torch.float64); Am[0, j] -= h
        fd = float(((_salt_y(Ap, 0.2) - _salt_y(Am, 0.2)) * w).sum()) / (2 * h)
        assert abs(float(gA[0, j]) - fd) <= 1e-4 * abs(fd), (j, float(gA[0, j]), fd)
    A_c = torch.tensor(A0, dtype=torch.float64)
    fd = float(((_salt_y(A_c, 0.2 + h) - _salt_y(A_c, 0.2 - h)) * w).sum()) / (2 * h)
    assert abs(float(glw) - fd) <= 1e-6 * abs(fd), (float(glw), fd)
    w2 = torch.zeros(n, dtype=torch.float64); w2[:5] = 1.0
    g2 = torch.autograd.grad((_salt_y(torch.tensor(A0, dtype=torch.float64), lw) * w2).sum(), lw)[0]
    fd = float(((_salt_y(A_c, 0.2 + h) - _salt_y(A_c, 0.2 - h)) * w2).sum()) / (2 * h)
    assert abs(float(g2) - fd) <= 1e-5 * abs(fd), (float(g2), fd)


def _pep_y(lw, g):
    s = SpinSystem(S=[0.5], g=g, HStrain=[150, 150, 150], lw=lw)
    return pepper_autograd(s, Experiment(mwFreq=9.5, Range=[333, 342], nPoints=451),
                           Options(GridSize=[31, 4]))[1]


def test_pepper_autograd_gradients_with_padding():
    """pepper_autograd HStrain field sweep cut by the range (Gaussian lw 1 mT,
    nPad = 150): autograd vs central differences for lw (h = 1e-5, measured
    relative error 1.1e-10) and g (h = 1e-6: 5.8e-10, 1.1e-9, 7.8e-7; the g_z
    derivative is 1e-3 of the g_x one)."""
    g0 = [[2.0, 2.01, 2.02]]
    w = _w(451)
    lw = torch.tensor([1.0, 0.0], dtype=torch.float64, requires_grad=True)
    g = torch.tensor(g0, dtype=torch.float64, requires_grad=True)
    y = _pep_y(lw, g)
    assert _count(_pep_y([0.0], g0).detach(), 9 / 450, 1.0, 0.0) == 150   # padding is active
    glw, gg = torch.autograd.grad((y * w).sum(), (lw, g))
    h = 1e-5
    fd = float(((_pep_y([1.0 + h], g0) - _pep_y([1.0 - h], g0)) * w).sum()) / (2 * h)
    assert abs(float(glw[0]) - fd) <= 1e-6 * abs(fd), (float(glw[0]), fd)
    h = 1e-6
    for j in range(3):
        gp = [list(g0[0])]; gp[0][j] += h
        gm = [list(g0[0])]; gm[0][j] -= h
        fd = float(((_pep_y([1.0], gp) - _pep_y([1.0], gm)) * w).sum()) / (2 * h)
        assert abs(float(gg[0, j]) - fd) <= 1e-5 * abs(fd), (j, float(gg[0, j]), fd)
