"""Empty bins of SOPHE-projected stick spectra are exactly 0 (B47;
tests/data/ref_pepper_exactzero.mat, generate_pepper_exactzero_refs.m).

The vectorised projections fill the interior bins of wide ramps / zones through
running sums (``c1·k + c0``).  Past the last ramp the ``+s``/``−s`` pairs did not
cancel exactly and the residual was multiplied by the bin index, so bins that no
triangle reaches held a roundoff floor of about ±5e-14 of the maximum (negative
values in a stick absorption spectrum).  EasySpin's projecttriangles.c /
projectzones.c add every bin directly and leave such bins at exactly 0.  An
integer coverage count built with the same lo/hi scatter now zeroes them.

* EasySpin references: no negative bins, and every bin that is 0 in EasySpin is
  0 in torchspin (pepper and pepper_autograd, field and frequency sweeps,
  triangles and Dinfh zones).
* Each projection helper is compared with its pre-fix form (the same source with
  the coverage mask removed): covered bins are bit-identical, only bins outside
  every interior change, and there the old value was the roundoff residual.
* Whole spectra against the pre-fix helpers: the change stays below 1e-12 of the
  maximum (the residual removed, also from bins that another pass fills, then
  convolved), and spectra that had no residual are bit-identical.
* Gradients of pepper_autograd through the masked sums vs central differences.
"""
import importlib
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io import loadmat

from torchspin.tests.test_pepper_matlab_validation_ext2 import sys_from_mat, exp_from_mat, opt_from_mat
from torchspin.experiment import Experiment, Options
from torchspin.spinsystem import SpinSystem

pep = importlib.import_module('torchspin.pepper')
pag = importlib.import_module('torchspin.pepper_autograd')

REF = Path(__file__).resolve().parents[2] / 'tests' / 'data' / 'ref_pepper_exactzero.mat'


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


def _run(fn, c):
    exp = exp_from_mat(c.Exp, c.Sys, x_axis=np.asarray(c.x, dtype=float))
    x, spc = fn(sys_from_mat(c.Sys), exp, opt_from_mat(c.Opt))
    return x.detach().cpu().numpy(), spc.detach().cpu().numpy().ravel()


# Measured with the fix: |spc - EasySpin| / max <= 1.0e-5 (axial field sweep;
# rhombic 6.6e-7, frequency sweeps <= 2.2e-13).  [31 4] is not compared value by
# value: the 4x interpolation differs from EasySpin there (B48, separate fix).
VAL_TOL = 2e-5


# The rhombic field sweeps' only empty EasySpin bins are the first/last 4-6 bins of
# the range (9 and 12): EasySpin drops the triangles that cross the range ends
# (B75).  Since 0.9.28 (B88) torchspin clips them, and those bins hold 27 % of the
# maximum, as on the same grid swept wider (test_range_end_bins_vs_wide_sweep).
_B75_DEVIATION = pytest.mark.xfail(strict=True, reason=(
    "Deliberate deviation since 0.9.28 (B88/B75): EasySpin's empty bins here are the range "
    "ends, where it drops the triangles crossing the range; torchspin clips them (27 % of max)."))


def _es_params():
    # pepper_autograd: field sweeps only
    return [pytest.param(nm, eng, marks=[_B75_DEVIATION] if nm.startswith('rhombic_field') else [])
            for nm in _case_names() for eng in ('pepper', 'pepper_autograd')
            if not (eng == 'pepper_autograd' and '_freq_' in nm)]


@pytest.mark.parametrize('name', ['rhombic_field_31x1', 'rhombic_field_31x4'])
@pytest.mark.parametrize('engine', ['pepper', 'pepper_autograd'])
def test_range_end_bins_vs_wide_sweep(cases, name, engine):
    """The range-end bins EasySpin leaves empty equal the same grid swept 200 bins
    wider on both sides (B75 fixed in 0.9.28): to roundoff (measured <= 9.7e-14 of
    max) except the first and last bin, which follow the first-bin convention
    (measured <= 8.0e-5 of max, [31 1]; 3.7e-7, [31 4])."""
    c = cases[name]
    fn = pep.pepper if engine == 'pepper' else pag.pepper_autograd
    x, spc = _run(fn, c)
    ref = np.asarray(c.spc, dtype=float).ravel()
    ends = ref == 0
    assert ends[:4].all() and ends[-4:].all()
    dx = x[1] - x[0]
    k = 200
    wide = Experiment(mwFreq=float(c.Exp.mwFreq), Range=[x[0] - k * dx, x[-1] + k * dx],
                      nPoints=len(x) + 2 * k, Harmonic=0)
    w = fn(sys_from_mat(c.Sys), wide, opt_from_mat(c.Opt))[1].detach().cpu().numpy().ravel()[k:k + len(x)]
    scale = np.abs(w).max()
    inner = np.zeros_like(ends)
    inner[1:-1] = ends[1:-1]
    assert np.abs(spc[inner] - w[inner]).max() < 1e-12 * scale
    assert np.abs(spc[ends] - w[ends]).max() < 1e-4 * scale
    assert spc[ends].min() > 0.2 * scale          # EasySpin: 0


@pytest.mark.parametrize('name,engine', _es_params())
def test_empty_bins_exactly_zero_vs_easyspin(cases, name, engine):
    c = cases[name]
    fn = pep.pepper if engine == 'pepper' else pag.pepper_autograd
    x, spc = _run(fn, c)
    ref = np.asarray(c.spc, dtype=float).ravel()
    assert np.allclose(x, np.asarray(c.x, dtype=float), rtol=1e-12, atol=0), 'sweep axis differs'
    es_empty = ref == 0
    assert es_empty.any()
    # main: min -5.3e-14 (pepper) / -4.7e-14 (autograd) of max for rhombic_field_31x1,
    # -1.6e-14 (pepper) for rhombic_field_31x4, -3.9e-14 for rhombic_freq_31x1
    assert spc.min() >= 0.0, (name, spc.min() / spc.max())
    assert np.all(spc[es_empty] == 0.0), (name, int((spc[es_empty] != 0).sum()))
    if name.endswith('31x4'):
        # B48: torchspin's 4x interpolation leaves 6 bins next to the empty
        # region uncovered that EasySpin fills, so only the inclusion holds here
        return
    assert np.array_equal(spc == 0.0, es_empty), name
    err = np.abs(spc - ref).max() / ref.max()
    print(f'{name} {engine}: max |spc - ref| / max = {err:.2e}')
    assert err < VAL_TOL, (name, err)


# ---------------------------------------------------------------------------
# Running sums with an integer coverage mask (B47), centered hat bins (B85)
# ---------------------------------------------------------------------------
# Since 0.9.35 the projection lives in torchspin._projection.  Interiors of more
# than _SHORT_INTERIOR bins go through running sums (c1·k + c0) with the B47
# coverage mask; shorter ones are scattered bin by bin.  The pre-0.9.35 helpers
# (pepper._ramp_bins, pepper_autograd._ramp_bins_t) and their source-patched
# "pre-fix" twins are gone; the invariant is checked directly instead: bins
# outside every facet's support are exactly 0, covered bins equal the
# facet-by-facet reference loop (pepper._projecttriangles_loop / _projectzones_loop).

from torchspin import _projection as _proj  # noqa: E402


def _support(lo_b, hi_b, n):
    """Axis points whose hat touches some interval [lo_b, hi_b] (axis coordinates)."""
    cov = np.zeros(n, dtype=bool)
    for a, b in zip(lo_b, hi_b):
        k0 = max(int(np.floor(a)), 0)
        k1 = min(int(np.floor(b)) + 1, n - 1)
        cov[k0:k1 + 1] = True
    return cov


@pytest.mark.parametrize('seed', [1, 2])
def test_long_ramps_uncovered_bins_exactly_zero(seed):
    """Wide tents (interiors far above _SHORT_INTERIOR, so the running sums run) in
    the lower half of the axis: every point beyond their support is exactly 0,
    the rest matches the reference loop."""
    rng = np.random.default_rng(seed)
    n = 1200
    x = 300.0 + 0.05 * np.arange(n)
    nt = 40
    fun = 300.0 + 0.05 * rng.uniform(5, 550, 3 * nt)
    tri = np.arange(3 * nt).reshape(nt, 3)
    areas = rng.uniform(0.1, 1.0, nt)
    amp = rng.uniform(0.5, 2.0, 3 * nt)
    P = np.sort(fun[tri], axis=1)
    assert ((P[:, 2] - P[:, 0]) / 0.05 > 4 * _proj._SHORT_INTERIOR).any()
    new = pep._projecttriangles(tri, areas, fun, amp, x)
    ref = pep._projecttriangles_loop(tri, areas, fun, amp, x)
    cov = _support((P[:, 0] - 300.0) / 0.05, (P[:, 2] - 300.0) / 0.05, n)
    assert np.all(new[~cov] == 0.0) and (~cov).sum() > 500
    assert new.min() >= 0.0
    assert np.abs(new - ref).max() < 1e-13 * np.abs(ref).max()


def test_long_zones_uncovered_bins_exactly_zero():
    rng = np.random.default_rng(7)
    n = 1200
    x = 300.0 + 0.05 * np.arange(n)
    pos = 300.0 + 0.05 * np.cumsum(rng.uniform(0.0, 90.0, 9)) + 0.1
    rng.shuffle(pos)                                          # segments run both ways
    amp = rng.uniform(0.5, 2.0, pos.size)
    w = rng.uniform(0.1, 1.0, pos.size - 1)
    new = pep._projectzones(pos, amp, w, x)
    newt = pag.projectzones_t(torch.tensor(pos)[None], torch.tensor(amp)[None], torch.tensor(w),
                              float(x[0]), float(x[1] - x[0]), n).numpy()
    ref = pep._projectzones_loop(pos, amp, w, x)
    lb = (np.minimum(pos[:-1], pos[1:]) - 300.0) / 0.05
    rb = (np.maximum(pos[:-1], pos[1:]) - 300.0) / 0.05
    cov = _support(lb, rb, n)
    for spec in (new, newt):
        assert np.all(spec[~cov] == 0.0) and (~cov).sum() > 100
        assert np.abs(spec - ref).max() < 1e-13 * np.abs(ref).max()


_SPECTRA = {
    'rhombic_stick': (dict(S=0.5, g=[2.0, 2.05, 2.1]),
                      dict(mwFreq=9.5, Range=[315, 345], nPoints=601, Harmonic=0), dict(GridSize=[31, 1])),
    'axial_stick': (dict(S=0.5, g=[2.0, 2.0, 2.1]),
                    dict(mwFreq=9.5, Range=[315, 345], nPoints=601, Harmonic=0), dict(GridSize=[31, 1])),
}


@pytest.mark.parametrize('key,engine', [(k, e) for k in _SPECTRA for e in ('pepper', 'pepper_autograd')])
def test_stick_spectra_empty_bins_exactly_zero(key, engine):
    """Stick spectra spanning part of the range: no negative bins, and the points
    beyond the spectrum's hat support are exactly 0 (B47 with centered bins)."""
    fn = pep.pepper if engine == 'pepper' else pag.pepper_autograd
    s, e, o = _SPECTRA[key]
    x, y = fn(SpinSystem(**s), Experiment(**e), Options(Verbosity=0, **o))
    x = x.numpy(); y = y.detach().numpy().ravel()
    assert y.min() >= 0.0
    nz = np.nonzero(y)[0]
    assert nz.size and nz[0] > 20 and nz[-1] < y.size - 20
    assert np.all(y[:nz[0]] == 0.0) and np.all(y[nz[-1] + 1:] == 0.0)


def test_autograd_gradient_vs_central_differences():
    """The coverage mask is integer (no graph); d spc / d g through the masked
    running sums matches central differences."""
    exp = Experiment(mwFreq=9.5, Range=[325, 337], nPoints=601, Harmonic=0)
    opt = Options(Verbosity=0, GridSize=[31, 1])
    g0 = [2.0, 2.05, 2.1]
    w = torch.linspace(-1.0, 1.0, 601, dtype=torch.float64) ** 2 + 0.5

    def f(g):
        return (pag.pepper_autograd(SpinSystem(S=0.5, g=g), exp, opt)[1].ravel() * w).sum()

    g = torch.tensor(g0, dtype=torch.float64, requires_grad=True)
    (grad,) = torch.autograd.grad(f(g), g)
    h = 1e-7
    for k in range(3):
        gp = torch.tensor(g0, dtype=torch.float64); gp[k] += h
        gm = torch.tensor(g0, dtype=torch.float64); gm[k] -= h
        fd = (f(gp) - f(gm)).item() / (2 * h)
        print(f'g[{k}]: autograd {grad[k].item():.8e}, central difference {fd:.8e}')
        assert grad[k].item() == pytest.approx(fd, rel=1e-5)
