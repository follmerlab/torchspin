"""Phase B fix train (0.9.39): silent wrong results and late errors, each on more than
one example.  B100 R², B101 Harmonic default, B102 Temperature <= 0, B103 'Ham' keys,
B104 public isotopologues, B105 fast-motion frames, B106 full Q, B107 garlic Diff/mwMode,
B108 A without Nucs, B109 grid_convergence on the graph, B110 grid budget, B111 infinite
bounds, B112/B113 option validation before any model evaluation."""
import math

import numpy as np
import pytest
import torch

import torchspin
from torchspin import Experiment, Options, SpinSystem, garlic, pepper
from torchspin.esfit import FitOptions, esfit
from torchspin.fastmotion import fastmotion
from torchspin.rotations import erot

X = np.linspace(320.0, 360.0, 200)


def _gauss(p):
    return np.exp(-((X - p[0]) / p[1]) ** 2)


def _lor(p):
    return 1.0 / (1.0 + ((X - p[0]) / p[1]) ** 2)


# B100 ------------------------------------------------------------------------------
@pytest.mark.parametrize('model, ptrue, p0', [(_gauss, [340.0, 3.0], [339.0, 3.5]),
                                              (_lor, [335.0, 1.5], [336.0, 2.0])])
@pytest.mark.parametrize('noise', [0.01, 0.5])
def test_r_squared_is_about_the_data(model, ptrue, p0, noise):
    rng = np.random.default_rng(0)
    data = model(ptrue) + noise * rng.standard_normal(X.size)
    r = esfit(data, model, p0=np.array(p0), vary=np.array([3.0, 1.0]),
              options=FitOptions(method='levmar', verbosity=0, compute_uncertainties=False))
    ref = 1 - np.sum(r.residuals ** 2) / np.sum((data - data.mean()) ** 2)
    assert r.r_squared == pytest.approx(ref, rel=1e-12, abs=1e-12)
    assert -0.5 < r.r_squared <= 1.0


# B101 ------------------------------------------------------------------------------
@pytest.mark.parametrize('kw, harmonic', [
    (dict(Field=340.0, mwRange=[9.0, 10.0]), 0),
    (dict(Field=340.0, mwRange=[9.0, 10.0], mwFreq=9.5), 0),
    (dict(Field=340.0, mwCenterSweep=[9.5, 0.5], mwFreq=9.5), 0),
    (dict(mwFreq=9.5, Range=[330.0, 350.0]), 1),
    (dict(mwFreq=9.5, Range=[330.0, 350.0], Field=340.0), 1),
    (dict(Field=340.0, mwRange=[9.0, 10.0], mwFreq=9.5, Harmonic=2), 2),
])
def test_harmonic_default_by_sweep_mode(kw, harmonic):
    assert Experiment(**kw).Harmonic == harmonic


# B102 ------------------------------------------------------------------------------
@pytest.mark.parametrize('T', [0.0, -5.0, float('nan'), torch.tensor(0.0)])
def test_nonpositive_temperature_raises(T):
    with pytest.raises(ValueError, match='Temperature'):
        Experiment(mwFreq=9.5, Range=[330.0, 350.0], Temperature=T)


@pytest.mark.parametrize('T', [1e-3, 4.2, float('inf')])
def test_positive_temperature_accepted(T):
    assert Experiment(mwFreq=9.5, Range=[330.0, 350.0], Temperature=T).Temperature == T


# B103 ------------------------------------------------------------------------------
@pytest.mark.parametrize('S, key, val', [(1.5, '112', [[1.0, 0, 0.2, 0, 0.5]]),
                                         (1.0, '222', [[1e-4, 0, 1e-4, 0, 0]])])
def test_ham_prefixed_and_bare_keys_are_the_same_term(S, key, val):
    kw = {} if key.startswith('11') else {'g': [2.0]}
    e = Experiment(mwFreq=9.5, Range=[200.0, 450.0], nPoints=256)
    a = SpinSystem(S=[S], Ham={key: val}, lwpp=[2.0, 0], **kw)
    b = SpinSystem(S=[S], Ham={'Ham' + key: val}, lwpp=[2.0, 0], **kw)
    assert list(b.Ham) == [key]
    ya = np.asarray(pepper(a, e, Options(GridSize=7))[1]); yb = np.asarray(pepper(b, e, Options(GridSize=7))[1])
    np.testing.assert_array_equal(ya, yb)


@pytest.mark.parametrize('ham', [{'Ham11': [[1.0]]}, {'abc': [[1.0]]}, {'112': [[1, 0, 0, 0, 0]], 'Ham112': [[1, 0, 0, 0, 0]]}])
def test_bad_ham_keys_raise(ham):
    with pytest.raises(ValueError, match='Ham'):
        SpinSystem(S=[1.5], Ham=ham)


# B104 ------------------------------------------------------------------------------
@pytest.mark.parametrize('nucs, n_iso', [('Cu', 2), ('Cu,1H', 2), ('14N', 1)])
def test_public_isotopologues_is_the_easyspin_port(nucs, n_iso):
    from torchspin.isotopologues import isotopologues as port
    assert torchspin.isotopologues is port
    k = len(nucs.split(','))
    out = torchspin.isotopologues(SpinSystem(S=[0.5], Nucs=nucs, A=[[50.0, 50.0, 50.0]] * k))
    assert len(out) == n_iso
    if n_iso > 1:
        assert sum(float(s.weight) for s in out) == pytest.approx(1.0, abs=1e-3)


# B105, B106 ------------------------------------------------------------------------
NX = dict(S=[0.5], g=[2.0088, 2.0061, 2.0027], Nucs='14N')
A_PV = np.array([16.0, 16.0, 100.0])


def _lw(**kw):
    return fastmotion(SpinSystem(**NX, **kw), 340.0, 1e-10)[0]


@pytest.mark.parametrize('fr', [[0.3, 0.7, 0.2], [0.0, math.pi / 2, 0.0], [1.1, -0.4, 2.5]])
def test_fastmotion_frames(fr):
    l0 = _lw(A=[A_PV])
    lj = _lw(A=[A_PV], AFrame=[fr], gFrame=[fr])              # joint rotation: invariant
    np.testing.assert_allclose(lj, l0, rtol=1e-12)
    R = erot(fr).numpy().T
    la = _lw(A=[A_PV], AFrame=[fr])
    np.testing.assert_allclose(la, _lw(A=R @ np.diag(A_PV) @ R.T), rtol=1e-12)   # = full matrix
    assert np.abs(la - l0).max() > 1e-3 * l0.max()            # and it does change the widths


@pytest.mark.parametrize('ang', [[0.4, 1.0, -0.3], [0.0, 0.5, 0.0]])
def test_fastmotion_full_q_matches_principal(ang):
    Q = np.array([-1.0, -0.5, 1.5]); R = erot(ang).numpy().T
    np.testing.assert_allclose(_lw(A=[A_PV], Q=R @ np.diag(Q) @ R.T), _lw(A=[A_PV], Q=[Q]), rtol=1e-12)


# B107 ------------------------------------------------------------------------------
E_FM = Experiment(mwFreq=9.5, Range=[325.0, 355.0], nPoints=512)
NXL = dict(NX, A=[[16.0, 16.0, 100.0]], lwpp=[0.05, 0])


@pytest.mark.parametrize('tc', [1e-10, 3e-11])
def test_garlic_diff_equals_tcorr(tc):
    ref = np.asarray(garlic(SpinSystem(**NXL, tcorr=tc), E_FM)[1])
    for kw in ({'Diff': 1 / (6 * tc)}, {'logDiff': math.log10(1 / (6 * tc))}):
        np.testing.assert_allclose(np.asarray(garlic(SpinSystem(**NXL, **kw), E_FM)[1]), ref,
                                   rtol=1e-12, atol=1e-12 * np.abs(ref).max())


@pytest.mark.parametrize('kw', [dict(Diff=[1e9, 2e9]), dict(tcorr=1e-10, Diff=1e9)])
def test_garlic_diff_errors(kw):
    with pytest.raises(ValueError):
        garlic(SpinSystem(**NXL, **kw), E_FM)


def test_garlic_mwmode():
    import dataclasses
    ref = np.asarray(garlic(SpinSystem(**NXL, tcorr=1e-10), E_FM)[1])
    perp = np.asarray(garlic(SpinSystem(**NXL, tcorr=1e-10), dataclasses.replace(E_FM, mwMode='perpendicular'))[1])
    par = np.asarray(garlic(SpinSystem(**NXL, tcorr=1e-10), dataclasses.replace(E_FM, mwMode='parallel'))[1])
    np.testing.assert_array_equal(perp, ref)
    assert np.abs(par).max() == 0.0
    with pytest.raises(NotImplementedError):
        garlic(SpinSystem(**NXL, tcorr=1e-10), dataclasses.replace(E_FM, mwMode=('y', 'circular+')))


# B108 ------------------------------------------------------------------------------
@pytest.mark.parametrize('method', ['pepper', 'broadband'])
def test_differentiable_spectrum_rejects_A_without_nucs(method):
    g = torch.tensor([2.0, 2.05, 2.1], dtype=torch.float64)
    with pytest.raises(ValueError, match='Nucs'):
        torchspin.differentiable_spectrum(g, 0.5, 9.5, (300.0, 360.0), nPoints=64,
                                          A=torch.tensor([50.0, 50.0, 50.0], dtype=torch.float64), method=method)


# B109 ------------------------------------------------------------------------------
@pytest.mark.parametrize('gv', [[2.0, 2.05, 2.1], [2.003, 2.003, 2.2]])
def test_grid_convergence_on_the_graph(gv):
    e = Experiment(mwFreq=9.5, Range=[300.0, 345.0], nPoints=256)
    g = torch.tensor(gv, dtype=torch.float64, requires_grad=True)
    s = SpinSystem(S=[0.5], g=g, lwpp=[0.5, 0])
    c = torchspin.grid_convergence(s, e, Options(GridSize=[7, 4]))
    # the check describes the spectrum this system gives (pepper's graph route)
    a_ = pepper(s, e, Options(GridSize=[7, 4]))[1].detach().numpy()
    b_ = pepper(s, e, Options(GridSize=[28, 4]))[1].detach().numpy()
    assert c.cosine == pytest.approx(float(a_ @ b_ / np.linalg.norm(a_) / np.linalg.norm(b_)), abs=1e-12)
    assert c.amplitude_ratio == pytest.approx(float(np.abs(a_).max() / np.abs(b_).max()), abs=1e-12)


# B110 ------------------------------------------------------------------------------
def test_grid_over_budget_raises():
    with pytest.raises(ValueError, match='max_iter'):
        esfit(_gauss([340.0, 3.0]), _gauss, p0=np.array([335.0, 3.0]), vary=np.array([10.0, 1.0]),
              options=FitOptions(method='grid', grid_points=11, max_iter=100, verbosity=0))


@pytest.mark.parametrize('ptrue', [[344.0, 3.6], [336.0, 2.4]])
def test_grid_searches_the_whole_grid(ptrue):
    # optimum at the upper/lower end of the first axis: a truncated grid never gets there
    r = esfit(_gauss(ptrue), _gauss, p0=np.array([340.0, 3.0]), vary=np.array([4.0, 0.6]),
              options=FitOptions(method='grid', grid_points=11, max_iter=121, verbosity=0,
                                 compute_uncertainties=False))
    np.testing.assert_allclose(r.pfit, ptrue, atol=1e-9)


# B111 ------------------------------------------------------------------------------
@pytest.mark.parametrize('method', ['simplex', 'levmar', 'grid', 'swarm'])
@pytest.mark.parametrize('lb, ub', [([330.0, -np.inf], [350.0, np.inf]), ([330.0, 1.0], None)])
def test_infinite_bounds_raise_for_bounded_methods(method, lb, ub):
    with pytest.raises(ValueError, match='finite bounds'):
        esfit(_gauss([340.0, 3.0]), _gauss, p0=np.array([335.0, 2.5]), lb=np.array(lb),
              ub=None if ub is None else np.array(ub), options=FitOptions(method=method, verbosity=0))


def test_infinite_bounds_still_work_for_lbfgsb():
    r = esfit(_gauss([340.0, 3.0]), _gauss, p0=np.array([339.0, 2.8]), lb=np.array([330.0, 0.5]),
              options=FitOptions(method='lbfgsb', verbosity=0, compute_uncertainties=False))
    np.testing.assert_allclose(r.pfit, [340.0, 3.0], atol=1e-3)


# B112, B113 ------------------------------------------------------------------------
@pytest.mark.parametrize('kw', [dict(method='nonsense'), dict(autoscale='LSQ'), dict(target='integral'),
                                dict(lsq_negative_scale='mirror'), dict(workers='gpu'),
                                dict(n_workers='bogus'), dict(n_workers=0), dict(baseline=-2)])
def test_invalid_options_raise_before_any_model_evaluation(kw):
    calls = []

    def model(p):
        calls.append(1)
        return _gauss(p)
    with pytest.raises(ValueError, match='FitOptions'):
        esfit(_gauss([340.0, 3.0]), model, p0=np.array([335.0, 2.5]), vary=np.array([5.0, 1.0]),
              options=FitOptions(verbosity=0, **kw))
    assert calls == []


# B129 ------------------------------------------------------------------------------
# Rescale is a numerical device (L/scale): with it on, every orientation must still be
# divided by its own scale.  EasySpin's chili.m divides the powder sum by the last
# orientation's scale.  dev before the fix (LLMK [8 4 4 2], GridSize 7, 256 points):
# nitroxide 2.7e-4, Cu 7.5e-3 Rescale on vs off; after: 9e-8, 3.8e-5 (Lanczos threshold).
@pytest.mark.parametrize('skw, ekw', [
    (dict(S=[0.5], g=[2.0088, 2.0061, 2.0027], Nucs='14N', A=[[16, 16, 100]], tcorr=20e-9,
          Potential=[2, 0, 0, 1.5], lw=[0, 0.1]), dict(mwFreq=9.5, Range=[325, 355], nPoints=128)),
    (dict(S=[0.5], g=[2.05, 2.05, 2.25], Nucs='63Cu', A=[[50, 50, 550]], tcorr=5e-9,
          Potential=[2, 0, 0, 2.0], lw=[0, 0.5]), dict(mwFreq=9.5, Range=[260, 360], nPoints=128)),
])
def test_chili_rescale_does_not_change_the_powder_spectrum(skw, ekw):
    from torchspin import ChiliOptions, chili
    y = {r: np.asarray(chili(SpinSystem(**skw), Experiment(**ekw),
                             ChiliOptions(LLMK=[8, 4, 4, 2], GridSize=5, Rescale=r, Verbosity=0,
                                          Threshold=1e-10))[1]) for r in (True, False)}
    assert np.abs(y[True] - y[False]).max() / np.abs(y[False]).max() < 1e-5
