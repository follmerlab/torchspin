"""esfit ``method='trf'``: bounded trust-region reflective least squares (SciPy).

Parity with a direct ``scipy.optimize.least_squares`` call on the same residual
(amplitude and baseline fitted by least squares inside the residual), bounds,
fixed parameters, the evaluation-budget and stop semantics, progress accounting,
and that the default method is unchanged.  Synthetic data only."""
import numpy as np
import pytest
from scipy.optimize import least_squares

from torchspin.esfit import FitOptions, FitResult, esfit

X = np.linspace(0.0, 500.0, 350)
TRUE = np.array([145.0, 95.0, 25.0])
START, LB, UB = [120.0, 130.0, 40.0], [80.0, 50.0, 10.0], [180.0, 180.0, 60.0]


def _model(p):
    """Two Gaussians (centre, separation, width); amplitude/baseline come from esfit's autoscale."""
    c, d, w = p
    return np.exp(-((X - c) / w) ** 2) - 0.6 * np.exp(-((X - c - d) / (1.4 * w)) ** 2)


def _opts(**kw):
    o = dict(method='trf', target='fcn', max_iter=100, compute_uncertainties=False)
    o.update(kw)
    return FitOptions(**o)


def test_default_method_unchanged():
    assert FitOptions().method == 'simplex'


def test_trf_matches_direct_scipy_with_scale_baseline_and_bounds():
    data = 2.4 * _model(TRUE) + 0.12 + 0.03 * X / X.max()

    def residual(p):                    # esfit's lsq amplitude + linear baseline, written out by hand
        s = _model(p)
        D = np.column_stack([s, np.ones_like(X), X / X.max()])
        c = np.linalg.lstsq(D, data, rcond=None)[0]
        c[0] = abs(c[0])
        return D @ c - data
    ref = least_squares(residual, START, bounds=(LB, UB), x_scale='jac', diff_step=1e-4,
                        max_nfev=100, ftol=1e-9, xtol=1e-9, gtol=1e-9)
    fit = esfit(data, _model, p0=START, lb=LB, ub=UB,
                options=_opts(autoscale='lsq', baseline=1, trf_diff_step=1e-4,
                              tol_fun=1e-9, tol_x=1e-9, trf_gradient_tol=1e-9))
    assert isinstance(fit, FitResult) and fit.algorithm == 'trf'
    np.testing.assert_allclose(fit.pfit, ref.x, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(fit.pfit, TRUE, rtol=1e-5)
    np.testing.assert_allclose(fit.fit, data, atol=1e-8)
    assert fit.success and not fit.interrupted and ref.success
    assert fit.n_iterations == ref.njev                       # one Jacobian per trust-region iteration
    assert fit.n_evaluations > ref.nfev                       # tracker also counts the Jacobian probes


def test_budget_exhaustion_reports_failure_not_interruption():
    fit = esfit(_model(TRUE), _model, p0=START, lb=LB, ub=UB, options=_opts(max_iter=1))
    assert not fit.success and not fit.interrupted
    assert 'maximum number' in fit.message.lower()
    assert np.isfinite(fit.pfit).all() and np.isfinite(fit.fit).all()


def test_respects_parameter_bound():
    fit = esfit(_model(TRUE), _model, p0=START, lb=LB, ub=[130.0, 180.0, 60.0], options=_opts())
    assert 129.9 < fit.pfit[0] <= 130.0


def test_every_evaluation_stays_within_bounds():
    seen = []

    def observed(p):
        seen.append(np.array(p, dtype=float))
        return _model(p)
    esfit(_model(TRUE), observed, p0=START, lb=LB, ub=UB, options=_opts(max_iter=20))
    seen = np.array(seen)
    assert np.all(seen >= LB) and np.all(seen <= UB)


def test_fixed_parameter_stays_fixed_and_can_be_released():
    data = _model(TRUE)
    seen = []

    def observed(p):
        seen.append(np.array(p, dtype=float))
        return _model(p)
    fit = esfit(data, observed, p0=START, vary=[40.0, 0.0, 20.0], options=_opts())
    assert np.array_equal(fit.p_fixed, [False, True, False]) and fit.pfit.size == 2
    assert all(p[1] == 130.0 for p in seen)                   # the model only ever saw the fixed value
    fit2 = esfit(data, _model, p0=START, vary=[40.0, 40.0, 20.0], options=_opts())
    assert not fit2.p_fixed.any() and fit2.pfit.size == 3
    assert abs(fit2.pfit[1] - TRUE[1]) < 1e-3


def test_stop_when_returns_best_so_far():
    fit = esfit(_model(TRUE), _model, p0=START, lb=LB, ub=UB,
                options=_opts(stop_when=lambda info: info['n_evaluations'] >= 6))
    assert fit.interrupted and not fit.success and fit.n_evaluations == 6
    assert 'stop_when' in fit.message and np.isfinite(fit.fit).all()


def test_progress_counts_jacobian_probes():
    infos = []
    fit = esfit(_model(TRUE), _model, p0=START, lb=LB, ub=UB,
                options=_opts(max_iter=5, progress=infos.append, progress_interval=0.0))
    evals = [i['n_evaluations'] for i in infos]
    assert evals == sorted(evals) and fit.n_evaluations == evals[-1]
    assert all(i['method'] == 'trf' for i in infos)
    # each iteration: one step evaluation plus one finite-difference probe per active parameter
    assert fit.n_evaluations >= fit.n_iterations * (1 + 3)


def test_x_scale_array_and_uncertainties():
    data = _model(TRUE) + 1e-3 * np.sin(X / 7.0)              # deterministic misfit so the covariance is finite
    fit = esfit(data, _model, p0=START, lb=LB, ub=UB,
                options=_opts(trf_x_scale=np.array([10.0, 10.0, 5.0]), compute_uncertainties=True))
    assert fit.success and fit.pstd is not None and fit.pstd.shape == (3,) and np.all(fit.pstd > 0)


def test_unknown_method_still_rejected():
    with pytest.raises(ValueError, match='Unknown method'):
        esfit(_model(TRUE), _model, p0=START, lb=LB, ub=UB, options=_opts(method='trust'))
