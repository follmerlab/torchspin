"""C1 grid interpolation of the lower hemisphere (B74, 0.9.21).

``sphgrid`` builds the lower hemisphere of the C1 grid from the upper knots in
reverse order (EasySpin sphgrid.m:320-325), so each lower θ row is stored with
φ decreasing. The rectify step of the grid interpolation (EasySpin
private/gridinterp.m:204-224, torchspin ``pepper._interp_grid``) read those rows
as φ increasing, so a lower knot at φ took the value from -φ - Δφ (mirrored
and shifted by one coarse φ step of that row).
Only C1 (8 octants) has lower rows; C1 is chosen by default for systems with
``Sys.tdm`` (photoselection) and low symmetry, or with ``GridSymmetry='C1'``.
The rows are now reversed at the rectify step; the ``sphgrid`` order is
unchanged (``grid_triangulation`` depends on it).

torchspin follows the exact (no-interpolation) result here and deliberately
deviates from EasySpin, which has the same error (KNOWN_LIMITATIONS.md).

Measured (before -> after the fix):

* f = 12x + 15y + 8z + 4xy - 3yz + 2xz (mT) on C1 [19 4]: max |error| L1
  32.5 -> 0.52 mT, G3 35.7 -> 0.064 mT; [31 4]: L1 32.1 -> 0.31, G3 35.1 -> 0.035.
* Spectra, C1 [19 4] vs C1 [73 1] (the same fine knots without interpolation),
  max deviation / max: Cu(II) non-coaxial g/A 0.342 -> 7.5e-3 (pepper and
  pepper_autograd), photoselection triplet (default grid C1) 6.9e-2 -> 1.8e-3,
  salt 1H 2.1e-2 -> 5.4e-4.
"""
import hashlib
import sys

import numpy as np
import pytest
import torch

import torchspin.pepper  # noqa: F401  (module, not the re-exported function)
from torchspin.experiment import Experiment, Options
from torchspin.hamsymm import hamsymm
from torchspin.pepper import pepper
from torchspin.pepper_autograd import pepper_autograd
from torchspin.salt import salt
from torchspin.spinsystem import SpinSystem
from torchspin.sphgrid import sphgrid

_pep = sys.modules['torchspin.pepper']


def _rel(a, ref):
    return float(np.abs(np.asarray(a) - np.asarray(ref)).max() / np.abs(ref).max())


# ---------------------------------------------------------------------------
# Direct interpolation against the exact function
# ---------------------------------------------------------------------------
def _f(n):
    """Smooth test function (mT) without symmetry in φ."""
    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    return 12 * x + 15 * y + 8 * z + 4 * x * y - 3 * y * z + 2 * x * z


def _knots(sym, N):
    ph, th, _, _ = sphgrid(sym, N)
    ph = ph.numpy(); th = th.numpy()
    return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1), th


# (N_c, interpolation factor, linear) -> max |error| bound (mT); measured after
# the fix [before]: [19 4] L1 0.5205 [32.52], G3 0.0639 [35.72];
# [31 4] L1 0.3090 [32.12], G3 0.0347 [35.14].
_DIRECT = {
    (19, 4, True): 0.6, (19, 4, False): 0.08,
    (31, 4, True): 0.36, (31, 4, False): 0.045,
}


@pytest.mark.parametrize('N_c,n_i,linear', list(_DIRECT), ids=lambda v: str(v))
def test_c1_interpolation_matches_exact_function(N_c, n_i, linear):
    N_f = (N_c - 1) * n_i + 1
    nc, _ = _knots('C1', N_c)
    nf, thf = _knots('C1', N_f)
    yi = _pep._interp_grid(_f(nc), 'C1', N_c, N_f, linear=linear)
    err = np.abs(yi - _f(nf))
    lower = thf > np.pi / 2 + 1e-9
    assert np.isfinite(err).all()
    assert err.max() < _DIRECT[(N_c, n_i, linear)], err.max()
    # the lower hemisphere is as accurate as the upper one (before: 30-70x worse)
    assert err[lower].max() < 1.2 * err[~lower].max(), (err[lower].max(), err[~lower].max())


def test_c1_interpolation_with_missing_knots():
    """L1 with a missing coarse knot in the lower hemisphere: the NaN stays
    local (79 of 20738 fine knots) and the remaining fine knots are as accurate
    as without it (0.52 mT; before 32.5 mT)."""
    N_c, N_f = 19, 73
    nc, thc = _knots('C1', N_c)
    nf, _ = _knots('C1', N_f)
    y = _f(nc)
    k = int(np.nonzero(thc > 2.0)[0][5])           # a lower-hemisphere knot
    y[k] = np.nan
    yi = _pep._interp_grid(y, 'C1', N_c, N_f, linear=True)
    nan = np.isnan(yi)
    assert 0 < nan.sum() < 120, int(nan.sum())
    assert np.abs(yi[~nan] - _f(nf)[~nan]).max() < 0.6


# ---------------------------------------------------------------------------
# Other open-φ grids: unchanged bit for bit
# ---------------------------------------------------------------------------
# sha256 of the float64 (little-endian) output of _interp_grid(y, sym, 19, 73)
# for y = default_rng(74).standard_normal(n_knots) (L1nan: every 7th knot NaN),
# computed with 0.9.10 (before the fix) and identical after it.
_HASHES = {
    ('Ci', 'G3'): 'd4898b8ddf0c04afff38d9c7dcdf1fb2ce23ae8c8f2e43d76c6961bb841211e0',
    ('Ci', 'L1'): '1c9b0fe11e1850c3a85a6baec6b7090663f2b0fb3b0846d94d17f6608ac87666',
    ('Ci', 'L1nan'): '5d0dfceb7463a8a291900b2ef88b9410a2ecee63dbb41c64aa54f0c08832ad5f',
    ('C2h', 'G3'): '0c2b30d920102483e8a29c6d160c59a8c3e5dbc201358463457799d68f2a2fb5',
    ('C2h', 'L1'): '3cd56071e797533f23b202a9bb1eb0eeeb159025dd8f053be595ca8f542ffe5d',
    ('C2h', 'L1nan'): '62dbb2f41d43ba2f38faf6f7950db6dc6e6b315a01005c252e60a349aacc51fa',
    ('C4h', 'G3'): '7766bd0aa9998143a121419704aecaaeab7459b18755a4908856a5d3525c97c6',
    ('C4h', 'L1'): 'eac73888bd7d5a5d4371fc8b67735ddd476b8dae0f7ea432ba236c6ce8bef867',
    ('C4h', 'L1nan'): '835785452855d680773e5ee920190264ca3890563e29c85f287dde8a71d8bdf0',
}


@pytest.mark.parametrize('sym,mode', list(_HASHES), ids=lambda v: str(v))
def test_other_open_phi_grids_unchanged(sym, mode):
    n = sphgrid(sym, 19)[0].numel()
    y = np.random.default_rng(74).standard_normal(n)
    if mode == 'L1nan':
        y[::7] = np.nan
    out = np.ascontiguousarray(_pep._interp_grid(y, sym, 19, 73, linear=(mode != 'G3')), dtype='<f8')
    assert hashlib.sha256(out.tobytes()).hexdigest() == _HASHES[(sym, mode)]


# ---------------------------------------------------------------------------
# Spectra against the same fine grid without interpolation
# ---------------------------------------------------------------------------
def _cu(gs, func=pepper):
    s = SpinSystem(S=[0.5], g=[2.05, 2.08, 2.30], gFrame=[0.3, 0.5, 0.7], Nucs='63Cu',
                   A=[60, 90, 500], AFrame=[1.1, 0.6, 0.4], lw=[1.5, 0])
    e = Experiment(mwFreq=9.5, Range=[260, 345], Harmonic=1, nPoints=1024)
    return func(s, e, Options(GridSize=list(gs), GridSymmetry='C1', Verbosity=0))[1].detach().numpy().ravel()


def _photosel_sys():
    return SpinSystem(S=[1], g=[2.0, 2.03, 2.06], gFrame=[0.2, 0.4, 0.6], D=[1500, 300],
                      DFrame=[0.3, 0.6, 0.9], tdm=[1.0, 0.5, 0.3], lw=[2, 0])


def _photosel(gs):
    e = Experiment(mwFreq=9.5, Range=[180, 500], Harmonic=0, nPoints=1024, lightBeam='perpendicular')
    return pepper(_photosel_sys(), e, Options(GridSize=list(gs), Verbosity=0))[1].numpy().ravel()


def _salt(gs):
    s = SpinSystem(S=[0.5], g=[[2.0, 2.1, 2.3]], gFrame=[0.3, 0.5, 0.7], Nucs='1H',
                   A=[[-6, 4, 12]], AFrame=[1.1, 0.6, 0.4])
    return salt(s, Experiment(Field=310.0, mwFreq=9.5), Options(GridSize=list(gs), GridSymmetry='C1'),
                freq_range=(0, 30), n_points=1024, lw_mhz=0.2)[1].detach().numpy().ravel()


@pytest.fixture(scope='module')
def cu_ref():
    return _cu((73, 1))


@pytest.mark.parametrize('func', ['pepper', 'pepper_autograd'])
def test_cu_spectrum_vs_no_interpolation(cu_ref, func):
    """Cu(II), non-coaxial g and A, C1 [19 4] vs C1 [73 1]: 7.5e-3 (before
    0.342). A Ci grid with the same sizes gives 7.8e-3."""
    y = _cu((19, 4), pepper if func == 'pepper' else pepper_autograd)
    assert _rel(y, cu_ref) < 1e-2


def test_cu_pepper_autograd_matches_pepper():
    """2.0e-11 (unchanged by the fix)."""
    assert _rel(_cu((19, 4), pepper_autograd), _cu((19, 4))) < 1e-9


def test_photoselection_default_grid_vs_no_interpolation():
    """Triplet with Sys.tdm and non-coaxial g, D: the default grid is C1.
    [19 4] vs [73 1]: 1.8e-3 (before 6.9e-2)."""
    assert hamsymm(_photosel_sys())[0] == 'C1'
    assert _rel(_photosel((19, 4)), _photosel((73, 1))) < 2.5e-3


def test_salt_c1_vs_no_interpolation():
    """1H ENDOR, non-coaxial g and A, C1 [19 4] vs [73 1]: 5.4e-4 (before 2.1e-2)."""
    assert _rel(_salt((19, 4)), _salt((73, 1))) < 1e-3


def test_easyspin_fullsphere_c1_vs_no_interpolation():
    """EasySpin's pepper_fullsphere system (coaxial rhombic g) on C1 [10 2] vs
    C1 [19 1]: 6.0e-4 (before 9.0e-3; EasySpin's own output 9.4e-3). The one-knot
    φ shift of the old row order also moved systems that are symmetric in φ."""
    s = SpinSystem(S=[0.5], g=[2.0, 2.05, 2.2], lw=[1.0, 0.0])
    e = Experiment(mwFreq=9.5, Harmonic=0, Range=[305, 345])

    def run(gs):
        return pepper(s, e, Options(GridSize=gs, GridSymmetry='C1', Verbosity=0))[1].numpy().ravel()

    assert _rel(run([10, 2]), run([19, 1])) < 1.5e-3


# ---------------------------------------------------------------------------
# Gradient on a C1 grid
# ---------------------------------------------------------------------------
def test_c1_gradient_vs_central_differences():
    """pepper_autograd d/dg and d/dD on C1 [19 4] vs central differences of
    pepper. Measured (relative): g 8.7e-8, 2.1e-7, 1.4e-7 (h = 1e-5; 1e-7 is
    roundoff-limited, 3e-6); D 5.3e-7, 8.2e-7 (h = 1e-2 MHz; 1 MHz is
    truncation-limited, 1.2e-2). Before the fix the same check gave 2.4e-6 for
    d/dg_y (consistency only: both sides used the same interpolation)."""
    e = Experiment(mwFreq=9.5, Range=[180, 500], Harmonic=1, nPoints=1024)
    o = Options(GridSize=[19, 4], GridSymmetry='C1', Verbosity=0)
    w = torch.linspace(0.0, 1.0, 1024, dtype=torch.float64) ** 2

    def mk(p):
        return SpinSystem(S=[1], g=p[:3].reshape(1, 3), gFrame=[0.4, 0.8, 0.2], D=p[3:].reshape(1, 2), lw=[2, 0])

    p0 = [2.0, 2.05, 2.1, 1500.0, 300.0]
    p = torch.tensor(p0, dtype=torch.float64, requires_grad=True)
    grad = torch.autograd.grad((pepper_autograd(mk(p), e, o)[1].ravel() * w).sum(), p)[0].numpy()

    def L(q):
        return float((pepper(mk(torch.tensor(q, dtype=torch.float64)), e, o)[1].ravel() * w).sum())

    for j, h in enumerate([1e-5, 1e-5, 1e-5, 1e-2, 1e-2]):
        pp = list(p0); pp[j] += h
        pm = list(p0); pm[j] -= h
        fd = (L(pp) - L(pm)) / (2 * h)
        assert abs(grad[j] - fd) < 2e-6 * abs(fd), (j, grad[j], fd)
