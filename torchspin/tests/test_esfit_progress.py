"""esfit progress reporting, resource summary and early stop (all methods)."""
import os
import pickle
import time

import numpy as np
import pytest

from torchspin.esfit import (FitOptions, FitResult, esfit, _FitProgress, _TrackedObjective,
                             _eval_population, _CloudPickled)

X_AXIS = np.linspace(0, 1, 64)
TRUE = np.array([1.5, -0.7])


def _quadratic_model(p):
    """Module-level (picklable by reference) model: p0 + p1 * x."""
    return p[0] + p[1] * X_AXIS


def _data():
    return _quadratic_model(TRUE)


METHODS = ['simplex', 'lbfgsb', 'powell', 'grid', 'montecarlo', 'genetic', 'swarm', 'global', 'levmar', 'trf']


def _opts(method, **kw):
    small = dict(swarm_size=6, ga_population=6, mc_samples=10, grid_points=3, max_iter=30,
                 verbosity=0, compute_uncertainties=False, seed=0)
    small.update(kw)
    return FitOptions(method=method, **small)


def test_tracker_record_counts_and_best():
    tr = _FitProgress(FitOptions(), 'simplex')
    obj = _TrackedObjective(lambda x: float(np.sum(np.asarray(x) ** 2)), tr)
    xs = [np.array([1.0, 1.0]), np.array([0.5, 0.0]), np.array([2.0, 2.0]), np.array([0.1, 0.1]), np.array([0.3, 0.0])]
    vals = [obj(x) for x in xs]
    assert tr.n_evaluations == 5 and tr.best_f == min(vals)
    assert np.allclose(tr.best_x, [0.1, 0.1])
    xs[3][0] = 99.0                                   # best_x must be a copy
    assert tr.best_x[0] == 0.1
    tr.record_batch(np.array([[0.0, 0.05], [1.0, 1.0]]), np.array([0.0025, 2.0]))
    assert tr.n_evaluations == 7 and tr.best_f == 0.0025 and np.allclose(tr.best_x, [0.0, 0.05])


class _RaiseAfter:
    """Model that raises KeyboardInterrupt on its k-th call and records what it saw."""

    def __init__(self, k):
        self.k, self.calls, self.seen = k, 0, []

    def __call__(self, p):
        self.calls += 1
        if self.calls == self.k:
            raise KeyboardInterrupt
        self.seen.append(np.array(p, float))
        return _quadratic_model(p)


@pytest.mark.parametrize('method', METHODS)
def test_interrupt_returns_best_so_far(method):
    k = 7
    model = _RaiseAfter(k)
    res = esfit(_data(), model, np.array([1.0, 0.0]), vary=np.array([1.0, 1.0]), options=_opts(method))
    assert isinstance(res, FitResult)
    assert res.success is False and res.interrupted is True
    assert res.message.startswith('interrupted') and 'KeyboardInterrupt' in res.message
    assert res.n_evaluations == k - 1
    # best-so-far among the parameter sets the model actually evaluated
    rmsds = [np.sqrt(np.mean((_quadratic_model(p) - _data()) ** 2)) for p in model.seen]
    # esfit rescales (autoscale) so compare through the objective: pfit must be one of the seen points
    assert any(np.allclose(res.pfit, p) for p in model.seen), (res.pfit, model.seen[:3])
    assert res.elapsed_s is not None and res.elapsed_s >= 0


def test_max_time_stops():
    def slow(p):
        time.sleep(0.005)
        return _quadratic_model(p)
    res = esfit(_data(), slow, np.array([1.0, 0.0]), vary=np.array([1.0, 1.0]),
                options=_opts('swarm', swarm_size=8, max_iter=10 ** 5, max_time=0.2))
    assert res.interrupted and 'max_time' in res.message and res.elapsed_s < 2.0


def test_stop_when_predicate():
    res = esfit(_data(), _quadratic_model, np.array([1.0, 0.0]), vary=np.array([1.0, 1.0]),
                options=_opts('simplex', max_iter=500, stop_when=lambda info: info['n_evaluations'] >= 20))
    assert res.interrupted and res.n_evaluations == 20 and 'stop_when' in res.message


def test_progress_callable_false_stops():
    infos = []

    def cb(info):
        infos.append(info)
        return False
    res = esfit(_data(), _quadratic_model, np.array([1.0, 0.0]), vary=np.array([1.0, 1.0]),
                options=_opts('swarm', progress=cb, progress_interval=0.0))
    assert res.interrupted and 'callback' in res.message
    keys = {'method', 'phase', 'iter', 'iter_total', 'n_evaluations', 'eval_total', 'rmsd', 'best_f', 'best_x', 'elapsed_s', 'evals_per_s', 'eta_s'}
    assert keys <= set(infos[0])


def test_progress_text_output(capsys):
    esfit(_data(), _quadratic_model, np.array([1.0, 0.0]), vary=np.array([1.0, 1.0]),
          options=_opts('swarm', max_iter=5, progress='text', progress_interval=0.0))
    out = capsys.readouterr().out
    assert 'torch threads' in out and 'cpus' in out
    assert '  best ' in out and 'ev/s' in out and 'done:' in out


def test_progress_off_is_silent(capsys):
    esfit(_data(), _quadratic_model, np.array([1.0, 0.0]), vary=np.array([1.0, 1.0]), options=_opts('swarm', max_iter=3))
    assert capsys.readouterr().out == ''


def test_legacy_callback_unchanged():
    seen = []

    def cb(info):
        seen.append(info)
        return False                                   # legacy callback cannot stop the fit
    res = esfit(_data(), _quadratic_model, np.array([1.0, 0.0]), vary=np.array([1.0, 1.0]),
                options=_opts('swarm', max_iter=4, callback=cb))
    assert not res.interrupted and len(seen) == 4
    assert set(seen[0]) == {'method', 'iter', 'max_iter', 'best_f', 'best_x'}


@pytest.mark.skipif((os.cpu_count() or 1) < 2, reason='needs 2 cores')
def test_progress_with_process_pool(monkeypatch, capsys):
    pytest.importorskip('cloudpickle')
    shipped = []
    orig_init = _CloudPickled.__init__

    def spy(self, fn):
        shipped.append(fn)
        orig_init(self, fn)
    monkeypatch.setattr(_CloudPickled, '__init__', spy)
    res = esfit(_data(), _quadratic_model, np.array([1.0, 0.0]), vary=np.array([1.0, 1.0]),
                options=_opts('swarm', swarm_size=8, max_iter=3, n_workers=2, workers='processes',
                              progress='text', progress_interval=0.0))
    assert not res.interrupted and res.n_evaluations == 8 * 4 + 1
    assert shipped and all(not isinstance(f, _TrackedObjective) for f in shipped)
    assert 'procs' in capsys.readouterr().out
    with pytest.raises(TypeError):
        pickle.dumps(_TrackedObjective(_quadratic_model, _FitProgress(FitOptions(), 'swarm')))


def test_global_shares_tracker():
    infos = []
    res = esfit(_data(), _quadratic_model, np.array([1.0, 0.0]), vary=np.array([1.0, 1.0]),
                options=_opts('global', max_iter=40, progress=infos.append, progress_interval=0.0))
    phases = [i['phase'] for i in infos]
    assert 'global/swarm' in phases and 'global/simplex' in phases
    assert phases.index('global/simplex') > phases.index('global/swarm')
    evals = [i['n_evaluations'] for i in infos]
    assert evals == sorted(evals) and res.n_evaluations == evals[-1]


def test_fitresult_new_fields_backward_compatible():
    r = FitResult(pfit=np.zeros(1), pnames=['a'], p_start=np.zeros(1), p_fixed=np.zeros(1, bool), fit=np.zeros(2),
                  fit_raw=np.zeros(2), residuals=np.zeros(2), scale=1.0, baseline=np.zeros(2), rmsd=0.0, ssr=0.0,
                  n_params=1, n_data=2, algorithm='simplex', n_iterations=0, success=True, message='')
    assert r.interrupted is False and r.n_evaluations is None and r.elapsed_s is None
    res = esfit(_data(), _quadratic_model, np.array([1.0, 0.0]), vary=np.array([1.0, 1.0]), options=_opts('levmar'))
    assert res.n_evaluations > 0 and res.elapsed_s >= 0 and not res.interrupted


def test_grid_progress_eval_total():
    infos = []
    esfit(_data(), _quadratic_model, np.array([1.0, 0.0]), vary=np.array([1.0, 1.0]),
          options=_opts('grid', grid_points=6, max_iter=1000, progress=infos.append, progress_interval=0.0))
    totals = {i['eval_total'] for i in infos if i['eval_total']}
    assert totals == {37}
    assert infos[-1]['n_evaluations'] >= 37
