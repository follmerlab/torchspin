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
import inspect
import re
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


def _es_params():
    # pepper_autograd: field sweeps only
    return [(nm, eng) for nm in _case_names() for eng in ('pepper', 'pepper_autograd')
            if not (eng == 'pepper_autograd' and '_freq_' in nm)]


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
# Pre-fix helpers: the current source with the coverage mask removed
# ---------------------------------------------------------------------------

_PREFIX = {
    (pep, '_ramp_bins'): (
        r"        # Past the last wide ramp.*?spec \+= np\.where\(cov, (c1 \* np\.arange\(nPoints, dtype=np\.float64\) \+ c0), 0\.0\)\n",
        r"        spec += \1\n"),
    (pep, '_projectzones'): (
        r"        # bins no interior covers stay exactly 0 \(see _ramp_bins\)\n.*?"
        r"spec \+= np\.where\(np\.cumsum\(cnt\)\[:nPoints\] > 0, (np\.cumsum\(S0\)\[:nPoints\]), 0\.0\)\n",
        r"        spec += \1\n"),
    (pag, '_ramp_bins_t'): (
        r"        # Bins that no interior covers.*?"
        r"spec = torch\.where\(cov, (spec \+ c1 \* torch\.arange\(n, dtype=_F64, device=dev\) \+ c0), spec\)\n",
        r"        spec = \1\n"),
    (pag, 'projectzones_t'): (
        r"        # bins no interior covers stay exactly 0 \(see _ramp_bins_t\)\n.*?"
        r"spec = spec \+ torch\.where\(torch\.cumsum\(cnt, 0\)\[:n\] > 0, c0, torch\.zeros_like\(c0\)\)\n",
        r"        spec = spec + torch.cumsum(S0, 0)[:n]\n"),
}


def _prefix(mod, name):
    pattern, repl = _PREFIX[(mod, name)]
    src = inspect.getsource(getattr(mod, name))
    old, n = re.subn(pattern, repl, src, flags=re.S)
    assert n == 1, f'pre-fix pattern for {name} no longer matches'
    assert 'cnt' not in old
    ns = dict(vars(mod))
    exec(compile(old, f'<pre-fix {name}>', 'exec'), ns)
    return ns[name]


def _wide_ramps(seed, n, count=40, hi_bin=None):
    """Ramps whose interiors span >= 3 bins, all inside [0, hi_bin)."""
    rng = np.random.default_rng(seed)
    hi_bin = n if hi_bin is None else hi_bin
    a = rng.uniform(1.0, hi_bin - 12.0, count)
    b = np.minimum(a + rng.uniform(4.0, 40.0, count), hi_bin - 1e-3)
    return a, b


def _interior_and_edges(a, b, n):
    first = np.trunc(a).astype(int); last = np.trunc(b).astype(int)
    cov = np.zeros(n, dtype=bool); edge = np.zeros(n, dtype=bool)
    for f, l in zip(first, last):
        cov[f + 1:l] = True
        edge[[f, l]] = True
    return cov, edge


@pytest.mark.parametrize('rising', [True, False])
def test_ramp_bins_covered_bins_bit_identical(rising):
    n = 400
    a, b = _wide_ramps(1 if rising else 2, n, hi_bin=300)
    f0 = np.random.default_rng(3).uniform(0.1, 10.0, a.size)
    new = np.zeros(n); pep._ramp_bins(new, f0, a, b, rising)
    old = np.zeros(n); _prefix(pep, '_ramp_bins')(old, f0, a, b, rising)
    cov, edge = _interior_and_edges(a, b, n)
    assert np.array_equal(new[cov], old[cov])
    # an edge bin outside every interior lost only the residual
    assert np.abs(new - old)[edge & ~cov].max() < 1e-13 * np.abs(old).max()
    empty = ~cov & ~edge
    assert np.all(new[empty] == 0.0)
    assert np.any(old[empty] != 0.0)              # the residual the fix removes
    assert np.abs(old[empty]).max() < 1e-13 * np.abs(old).max()


@pytest.mark.parametrize('rising', [True, False])
def test_ramp_bins_t_covered_bins_bit_identical(rising):
    n = 400
    a, b = _wide_ramps(4 if rising else 5, n, hi_bin=300)
    base = np.random.default_rng(6).uniform(0.1, 10.0, a.size)
    args = (n, torch.tensor(base), torch.tensor(a), torch.tensor(b), 0.01, rising)
    new = pag._ramp_bins_t(*args).numpy()
    old = _prefix(pag, '_ramp_bins_t')(*args).numpy()
    cov, edge = _interior_and_edges(a, b, n)
    assert np.array_equal(new[cov], old[cov])
    # an edge bin outside every interior lost only the residual
    assert np.abs(new - old)[edge & ~cov].max() < 1e-13 * np.abs(old).max()
    empty = ~cov & ~edge
    assert np.all(new[empty] == 0.0)
    assert np.any(old[empty] != 0.0)
    assert np.abs(old[empty]).max() < 1e-13 * np.abs(old).max()


def _zones(seed, n):
    rng = np.random.default_rng(seed)
    x = 300.0 + 0.05 * np.arange(n)
    nseg = 60
    pos = 300.0 + 0.05 * np.cumsum(rng.uniform(0.0, 6.0, nseg + 1)) + 0.1
    pos = pos[pos < 300.0 + 0.05 * (n - 60)][:nseg + 1]
    rng.shuffle(pos)                                          # segments run both ways
    amp = rng.uniform(0.5, 2.0, pos.size)
    w = rng.uniform(0.1, 1.0, pos.size - 1)
    return pos, amp, w, x


def test_projectzones_covered_bins_bit_identical():
    n = 600
    pos, amp, w, x = _zones(7, n)
    new = pep._projectzones(pos, amp, w, x)
    old = _prefix(pep, '_projectzones')(pos, amp, w, x)
    newt = pag.projectzones_t(torch.tensor(pos)[None], torch.tensor(amp)[None], torch.tensor(w),
                              float(x[0]), float(x[1] - x[0]), n).numpy()
    oldt = _prefix(pag, 'projectzones_t')(torch.tensor(pos)[None], torch.tensor(amp)[None], torch.tensor(w),
                                          float(x[0]), float(x[1] - x[0]), n).numpy()
    lb = (np.minimum(pos[:-1], pos[1:]) - x[0]) / (x[1] - x[0])
    rb = (np.maximum(pos[:-1], pos[1:]) - x[0]) / (x[1] - x[0])
    cov, edge = _interior_and_edges(lb, rb, n)
    for nw, od in ((new, old), (newt, oldt)):
        assert np.array_equal(nw[cov], od[cov])
        assert np.abs(nw - od)[edge & ~cov].max() < 1e-13 * np.abs(od).max()
        empty = ~cov & ~edge
        assert np.all(nw[empty] == 0.0)
        assert np.any(od[empty] != 0.0)
        assert np.abs(od[empty]).max() < 1e-13 * np.abs(od).max()


# ---------------------------------------------------------------------------
# Whole spectra against the pre-fix helpers
# ---------------------------------------------------------------------------

_SPECTRA = {
    'rhombic_stick': (dict(S=0.5, g=[2.0, 2.05, 2.1]),
                      dict(mwFreq=9.5, Range=[325, 337], nPoints=601, Harmonic=0), dict(GridSize=[31, 1])),
    'rhombic_lw_h1': (dict(S=0.5, g=[2.0, 2.05, 2.1], lw=1.5),
                      dict(mwFreq=9.5, Range=[320, 345], nPoints=1024, Harmonic=1), dict()),
    'cu_hyperfine': (dict(S=0.5, g=[2.05, 2.06, 2.25], Nucs='63Cu', A=[[60.0, 70.0, 550.0]], lwpp=1.0),
                     dict(mwFreq=9.5, Range=[270, 345], nPoints=1024, Harmonic=1), dict()),
    'axial_stick': (dict(S=0.5, g=[2.0, 2.0, 2.1]),
                    dict(mwFreq=9.5, Range=[315, 345], nPoints=601, Harmonic=0), dict(GridSize=[31, 1])),
    'triplet_freq': (dict(S=1, g=2.0, D=[[3000.0, 300.0]], lwpp=10.0),
                     dict(Field=340, mwRange=[5, 15], nPoints=1024), dict()),
}


def _spectrum(fn, key):
    s, e, o = _SPECTRA[key]
    return fn(SpinSystem(**s), Experiment(**e), Options(Verbosity=0, **o))[1].detach().numpy().ravel()


@pytest.mark.parametrize('key,engine', [(k, e) for k in _SPECTRA for e in ('pepper', 'pepper_autograd')
                                         if not (e == 'pepper_autograd' and 'mwRange' in _SPECTRA[k][1])])
def test_spectra_vs_prefix_helpers(monkeypatch, key, engine):
    fn = pep.pepper if engine == 'pepper' else pag.pepper_autograd
    new = _spectrum(fn, key)
    with monkeypatch.context() as mp:
        if engine == 'pepper':
            mp.setattr(pep, '_ramp_bins', _prefix(pep, '_ramp_bins'))
            mp.setattr(pep, '_projectzones', _prefix(pep, '_projectzones'))
        else:
            mp.setattr(pag, '_ramp_bins_t', _prefix(pag, '_ramp_bins_t'))
            mp.setattr(pag, 'projectzones_t', _prefix(pag, 'projectzones_t'))
        old = _spectrum(fn, key)
    scale = np.abs(old).max()
    diff = np.abs(new - old).max() / scale
    print(f'{key} {engine}: max |new - old| / max = {diff:.2e}, bins changed {(new != old).sum()}')
    if key.startswith('axial'):
        assert np.array_equal(new, old)          # Dinfh zones: no residual before either
        return
    # measured: stick 5.3e-14 (pepper) / 4.7e-14 (autograd); after convolution
    # <= 8.8e-14 (pepper) and 1.2e-13 / 5.2e-13 (autograd, rhombic / Cu)
    assert diff < 1e-12, (key, diff)
    if key == 'rhombic_stick':
        assert old.min() < 0.0 and new.min() >= 0.0
        assert np.all(np.abs(old[new == 0.0]) < 1e-13 * scale)


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
