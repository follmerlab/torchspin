"""Field sweeps whose range cuts transitions at some orientations (B88, B75, B91).

The matrix resonance search covers Exp.Range only, so a transition that leaves
the window at some orientations has knots without a resonance (NaN).  Up to
0.9.27 pepper and pepper_autograd then (a) interpolated the positions of EVERY
transition linearly, as EasySpin does (pepper.m:701, 763-772), which reproduces
the coarse-grid facets, and (b) dropped every triangle or zone with a NaN
vertex, so triangles crossing the range ends were lost even without
interpolation (B75; EasySpin too).  Since 0.9.28 the level pairs with partly
missing resonances are searched again over Range +- 0.5 W (W = sweep width,
lower edge clamped at 0; doubling up to 4 W while a pair keeps improving)
and replaced wholesale; projection stays on Exp.Range and clips the
out-of-range vertices; only slots that still miss knots go linear.

References: tests/data/ref_missing_knots.npz, generate_missing_knots_refs.py
(no interpolation, GridSize [181 1], on an aligned sweep wide enough that no
transition leaves it, cropped).  Spectra are compared on the interior of the
range, edges +- 5 FWHM dropped: at the range ends the final line-width
convolution sees lines centered outside the window only through the spectrum
itself, which the edge padding of Options.PaddingMultiplier (#53, 0.9.6)
handles; with it the full-range numbers become the interior ones (case 1
[19 4] 0.94750 / 0.328, case 2 0.999865 / 0.077, B75 case 1 [181 1]
1.000000 / 3e-4).

Measured (1 - cosine / max|dev|/max, interior), 0.9.27 -> 0.9.28, pepper and
pepper_autograd alike:

* case 1, 63Cu with tilted AFrame, auto C2h [19 4] (B88): 0.20125 / 0.624 ->
  0.052500 / 0.328 (the rest is the C2h grid at [19 4]; a wide sweep at the
  same grid gives the same spectrum to 3e-11, below);
* case 2, 91Zr D2h [19 4] (B91): 6.72e-4 / 0.0497 -> 4.42e-6 / 0.0049;
* case 3, S = 5/2 with B4, D2h [19 4] (B31): 0.043361 / 0.236 ->
  9.609e-5 / 0.0135;
* same-grid consistency (narrow range vs the crop of a wide range at the same
  GridSize, interior): case 1 [19 4] 0.197 / 0.604 -> 0 / 2.8e-11; B75 case 1
  [31 1] (no interpolation) 2.2e-4 / 0.0435 -> 0 / 2.6e-11; stick spectrum
  (no convolution) [31 1], whole range but the first bin, 0.059 -> 1.6e-11.
"""
import importlib.util
import sys as _sys
from pathlib import Path

import numpy as np
import pytest
import torch

import torchspin.pepper  # noqa: F401  (module, not the re-exported function)
from torchspin import Experiment, Options, SpinSystem, pepper
from torchspin.pepper_autograd import _interp_slots_t, pepper_autograd

_pep = _sys.modules['torchspin.pepper']
DATA = Path(__file__).resolve().parents[2] / 'tests' / 'data'
REF = DATA / 'ref_missing_knots.npz'

_spec = importlib.util.spec_from_file_location('_gen_missing_knots', DATA / 'generate_missing_knots_refs.py')
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


@pytest.fixture(scope='module')
def ref():
    assert REF.exists(), f'reference file missing: {REF}'
    return np.load(REF)


_metrics = gen.metrics


def _interior(x, sys):
    return gen.interior(x, sys)


@pytest.fixture
def research_calls(monkeypatch):
    """Records the calls of pepper._replace_pairs (non-empty iff a re-search ran)."""
    calls = []
    orig = _pep._replace_pairs

    def spy(lists, new, bad):
        calls.append(list(bad))
        return orig(lists, new, bad)
    monkeypatch.setattr(_pep, '_replace_pairs', spy)
    return calls


# (case, GridSize): bounds on (1 - cosine, max|dev|/max) vs the converged
# reference, interior; measured x 1.05 (see the module docstring)
REF_BOUNDS = {
    ('case1', (19, 4)): (0.0552, 0.345),
    ('case2', (19, 4)): (4.7e-6, 5.2e-3),
    ('case3', (19, 4)): (1.01e-4, 1.42e-2),
}


@pytest.mark.parametrize('func', ['pepper', 'pepper_autograd'])
@pytest.mark.parametrize('name,grid', list(REF_BOUNDS), ids=[f'{n}-{g[0]}x{g[1]}' for n, g in REF_BOUNDS])
def test_vs_converged_reference(ref, research_calls, name, grid, func):
    s, e, okw = gen.CASES[name][0]()
    fn = pepper if func == 'pepper' else pepper_autograd
    x, y = fn(s, e, Options(GridSize=list(grid), Verbosity=0, **okw))
    x = x.numpy(); y = y.detach().numpy().ravel()
    assert np.allclose(x, ref[name + '_x'], rtol=0, atol=1e-9)
    c, d = _metrics(y, ref[name], _interior(x, s))
    print(f'{name} {grid} {func}: 1 - cosine {c:.4e}, max dev {d:.4e}')
    assert research_calls, 'the range cuts transitions: the re-search must run'
    cmax, dmax = REF_BOUNDS[(name, grid)]
    assert c <= cmax and d <= dmax, (name, func, c, d)


def _cu_stick(rng=(290.0, 350.0), n=1024):
    s = SpinSystem(S=0.5, g=[2.05, 2.05, 2.25], Nucs='63Cu', A=[50, 50, 500], AFrame=[0, 0.4, 0])
    return s, Experiment(mwFreq=9.5, Range=list(rng), nPoints=n, Harmonic=0), {}


def _cu_hstrain(rng=(290.0, 350.0), n=1024):
    s = SpinSystem(S=0.5, g=[2.05, 2.05, 2.25], Nucs='63Cu', A=[50, 50, 500], AFrame=[0, 0.4, 0],
                   HStrain=[20, 20, 40])
    return s, Experiment(mwFreq=9.5, Range=list(rng), nPoints=n, Harmonic=1), {}


# narrow range vs the crop of an aligned wide range at the same GridSize:
# (builder, GridSize, lower extension, upper extension, mask) -> max|dev|/max.
# 'interior' drops the edges +- 5 FWHM (final convolution, see above); the
# stick and strain spectra have no convolution and are compared everywhere but
# in the first bin (it collects the projected density just below the range).
SAME_GRID = {
    'case1-19x4': (gen.case1, [19, 4], 60.0, 0.0, 'interior'),
    'case1-31x1-B75': (gen.case1, [31, 1], 60.0, 0.0, 'interior'),
    'case3-19x4': (gen.case3, [19, 4], 200.0, 200.0, 'interior'),
    'cu-stick-31x1-B75': (_cu_stick, [31, 1], 60.0, 0.0, 'bins'),
    'cu-stick-19x4': (_cu_stick, [19, 4], 60.0, 0.0, 'bins'),
    'cu-hstrain-19x4': (_cu_hstrain, [19, 4], 60.0, 0.0, 'bins'),
    'cu-hstrain-19x1-B75': (_cu_hstrain, [19, 1], 60.0, 0.0, 'bins'),
}


@pytest.mark.parametrize('label', list(SAME_GRID))
def test_narrow_equals_wide_crop(research_calls, label):
    """With every resonance found, a narrow sweep is the crop of a wide one at
    the same grid (projection clips out-of-range vertices).  Measured max|dev|
    1.6e-11..2.8e-11; on 0.9.27 0.044..0.60 (interior; stick [31 1]: 0.059)."""
    build, grid, lo, hi, how = SAME_GRID[label]
    s, e, okw = build()
    wr, wn, i0 = gen.wide_axis(tuple(e.Range), int(e.nPoints), lo, hi)
    s2, e2, _ = build(wr, wn)
    x, y = pepper(s, e, Options(GridSize=grid, Verbosity=0, **okw))
    assert research_calls, 'the narrow range cuts transitions: the re-search must run'
    _, y2 = pepper(s2, e2, Options(GridSize=grid, Verbosity=0, **okw))
    x = x.numpy(); y = y.numpy(); y2 = y2.numpy()[i0:i0 + y.size]
    if how == 'interior':
        mask = _interior(x, s)
    else:
        mask = np.ones(x.size, bool); mask[0] = False
    d = _metrics(y, y2, mask)[1]
    print(f'{label}: max dev {d:.2e}')
    assert d <= 1e-10, (label, d)


@pytest.mark.parametrize('name', ['case1', 'case2', 'case3'])
def test_pepper_matches_autograd(name):
    """pepper and pepper_autograd take the same re-search and per-slot
    interpolation: measured 1.3e-11 (case 1), 3.9e-12, 7.9e-12 of the maximum."""
    s, e, okw = gen.CASES[name][0]()
    opt = Options(GridSize=[19, 4], Verbosity=0, **okw)
    y1 = pepper(s, e, opt)[1].numpy()
    y2 = pepper_autograd(s, e, opt)[1].detach().numpy()
    assert np.abs(y1 - y2).max() <= 5e-11 * np.abs(y1).max()


_GRAD_CASES = {
    # case 1 on an explicit Ci grid: on the auto C2h grid a finite step in one
    # component of the axial g or A changes the grid symmetry (B1)
    'case1': (dict(S=0.5, g=[2.05, 2.05, 2.25], Nucs='63Cu', A=[50.0, 50.0, 500.0], AFrame=[0, 0.4, 0]),
              0.3, dict(mwFreq=9.5, Range=[290, 350], nPoints=1024, Harmonic=1), 'Ci'),
    'case2': (dict(S=0.5, g=[2.0, 1.95, 1.9], Nucs='91Zr', A=[780.0, 780.0, 780.0]),
              1.0, dict(mwFreq=9.65, Range=[300, 400], nPoints=256, Harmonic=1), 'D2h'),
}
# relative central-difference step and bound on max|AD - CD|/max|CD| per
# parameter.  Measured (case 1 / case 2): A 1.0e-6 / 9.9e-9 (h = 1e-6 A;
# 1e-5: 7e-4, truncation), g 3.9e-8 / 1.9e-8 (h = 1e-7 g; 1e-6: 8e-4; 1e-8:
# 1.7e-7, roundoff), lw 4.2e-9 / 1.2e-8 (h = 1e-4 lw)
_GRAD_STEPS = {'A': (1e-6, 3e-6), 'g': (1e-7, 1.2e-7), 'lw': (1e-4, 4e-8)}


@pytest.mark.parametrize('field', list(_GRAD_STEPS))
@pytest.mark.parametrize('name', list(_GRAD_CASES))
def test_gradient_vs_central_differences(research_calls, name, field):
    """d/dA, d/dg, d/dlw (Gaussian FWHM) of a weighted sum of the spectrum on
    the re-searched cases vs converged central differences."""
    base, lw0, ekw, grid = _GRAD_CASES[name]
    opt = Options(GridSize=[19, 4], GridSymmetry=grid, Verbosity=0)
    exp = Experiment(**ekw)
    p0 = torch.tensor([lw0] if field == 'lw' else base[field], dtype=torch.float64)

    def spec(p):
        kw = dict(base)
        if field == 'lw':
            kw['lw'] = torch.cat([p, torch.zeros(1, dtype=torch.float64)])
        else:
            kw['lw'] = [lw0, 0.0]
            kw[field] = p
        return pepper_autograd(SpinSystem(**kw), exp, opt)[1]

    w = torch.from_numpy(np.random.default_rng(0).standard_normal(exp.nPoints))
    p = p0.clone().requires_grad_(True)
    grad = torch.autograd.grad((spec(p) * w).sum(), p)[0].numpy()
    assert research_calls, 'the range cuts transitions: the re-search must run'
    rel, bound = _GRAD_STEPS[field]
    cd = np.empty(p0.numel())
    with torch.no_grad():
        for i in range(p0.numel()):
            h = rel * abs(float(p0[i]))
            pp = p0.clone(); pp[i] += h
            pm = p0.clone(); pm[i] -= h
            cd[i] = float(((spec(pp) - spec(pm)) * w).sum()) / (2 * h)
    err = np.abs(grad - cd).max() / np.abs(cd).max()
    print(f'{name} d/d{field}: AD {grad}, CD {cd}, rel err {err:.2e}')
    assert err <= bound, (name, field, grad, cd, err)


@pytest.mark.parametrize('grid', [[19, 4], [19, 1]])
@pytest.mark.parametrize('func', ['pepper', 'pepper_autograd'])
def test_no_research_without_missing_knots(research_calls, monkeypatch, func, grid):
    """A range that holds every resonance takes the 0.9.27 path unchanged: no
    re-search (the coarse results are passed on as the same objects), and no
    slot has a NaN, so none goes linear.  (A sweep of 29 such spectra, pepper
    and pepper_autograd, 9 grid symmetries, strain, perturbation and frequency
    sweeps, was bit-identical to 0.9.27.)"""
    seen = []
    orig = _pep._research_missing_knots

    def spy(lists, rng, search, verbose=False):
        out = orig(lists, rng, search, verbose)
        seen.append(out is lists)
        return out
    monkeypatch.setattr(_pep, '_research_missing_knots', spy)
    s = SpinSystem(S=0.5, g=[2.0, 2.05, 2.1], Nucs='14N', A=[20, 20, 90], lw=0.5)
    e = Experiment(mwFreq=9.5, Range=[305, 350], nPoints=512, Harmonic=1)
    fn = pepper if func == 'pepper' else pepper_autograd
    _, y = fn(s, e, Options(GridSize=grid, Verbosity=0))
    assert seen == [True]
    assert not research_calls
    assert np.isfinite(y.detach().numpy()).all()


def test_research_window_widens():
    """_research_missing_knots: windows Range +- 0.5 W, 1 W, 2 W, 4 W (lower
    edge clamped at 0) while a pair improves; a pair's new resonances are kept
    only if fewer are missing; stops when complete or nothing improves."""
    t = lambda *v: torch.tensor(v, dtype=torch.float64)
    pr = lambda *p: torch.tensor(p, dtype=torch.long).reshape(-1, 2)
    E = (t(), t(), None, None)                  # no resonance at an orientation

    def ori(*lines):                            # lines: (u, v, B)
        if not lines:
            return E
        return (t(*[b for _, _, b in lines]), t(*[1.0] * len(lines)), None,
                pr(*[x for u, v, _ in lines for x in (u, v)]))

    def lists_of(*oris):
        return tuple(list(c) for c in zip(*oris))
    # pair (0, 1) resonates at orientation 0 only, (0, 2) everywhere
    start = lists_of(ori((0, 1, 10.0), (0, 2, 20.0)), ori((0, 2, 21.0)), ori((0, 2, 22.0)))
    windows = []

    def search(found_at):
        def _s(window, pairs):
            windows.append(list(window))
            assert pairs == [(0, 1)]
            n = found_at[min(len(windows), len(found_at)) - 1]
            return lists_of(*[ori((0, 1, 10.0 + 0.1 * k)) if k < n else E for k in range(3)])
        return _s
    # improves (2 of 3, then 3 of 3): two rounds
    out = _pep._research_missing_knots(start, (5.0, 25.0), search([2, 3]))
    assert windows == [[0.0, 35.0], [0.0, 45.0]]
    assert _pep._missing_resonances(out[0], out[3]) == {}
    assert sorted(out[0][2].tolist()) == [10.2, 22.0]
    # improves once, then never again: all four windows, then stop
    windows.clear()
    out = _pep._research_missing_knots(start, (100.0, 120.0), search([2, 2, 2, 2]))
    assert windows == [[90.0, 130.0]] + [[80.0, 140.0]]
    assert _pep._missing_resonances(out[0], out[3]) == {(0, 1): 1}
    windows.clear()
    out = _pep._research_missing_knots(start, (100.0, 120.0), search([2, 2, 2, 3]))
    assert windows == [[90.0, 130.0], [80.0, 140.0]]
    # no improvement at all: the very same objects come back
    windows.clear()
    out = _pep._research_missing_knots(start, (100.0, 120.0), search([1]))
    assert windows == [[90.0, 130.0]] and out is start
    # a worse search result is not taken
    out = _pep._research_missing_knots(start, (100.0, 120.0), search([0]))
    assert out is start
    # nothing missing: no search, same objects
    full = lists_of(ori((0, 1, 10.0)), ori((0, 1, 11.0)))
    assert _pep._research_missing_knots(full, (0.0, 50.0), lambda w, p: 1 / 0) is full


@pytest.mark.parametrize('sym,N_c,N_f', [('D2h', 7, 19), ('Ci', 7, 19), ('Dinfh', 7, 19)])
def test_per_slot_linear(sym, N_c, N_f):
    """Positions: a slot with missing knots goes linear (L1; L3 axial), the
    others keep the global cubic, in pepper._interp_slot and
    pepper_autograd._interp_slots_t (rows mixed in one call)."""
    from torchspin.sphgrid import sphgrid
    phi, theta, _, vc = sphgrid(sym, N_c)
    _, _, _, vf = sphgrid(sym, N_f)
    vc = vc if vc.shape[0] == 3 else vc.T
    vf = vf if vf.shape[0] == 3 else vf.T
    smooth = (300.0 + 20.0 * np.cos(theta.numpy()) ** 2 + 5.0 * np.sin(theta.numpy()) ** 2 * np.cos(2 * phi.numpy()))
    holes = smooth.copy(); holes[3] = np.nan
    vals = np.stack([smooth, holes, smooth + 1.0])
    out = _interp_slots_t(torch.from_numpy(vals), sym, N_c, N_f, 'pos').numpy()
    for r in range(3):
        exp_ = _pep._interp_slot(vals[r], sym, N_c, N_f, vc, vf)
        assert np.array_equal(np.isnan(out[r]), np.isnan(exp_))
        ok = ~np.isnan(exp_)
        assert np.allclose(out[r][ok], exp_[ok], rtol=0, atol=1e-10)
    lin = (_pep._interp_l3(smooth, N_f) if sym == 'Dinfh'
           else _pep._interp_grid(smooth, sym, N_c, N_f, linear=True))
    # the NaN-free rows are cubic, not linear
    assert np.abs(out[0] - lin).max() > 1e-3
    assert np.abs(out[0] - _pep._interp_slot(smooth, sym, N_c, N_f, vc, vf)).max() < 1e-10
