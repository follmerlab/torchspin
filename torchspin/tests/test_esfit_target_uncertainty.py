"""esfit target default and uncertainties of transformed targets (B131).

The default target is 'fcn' (EasySpin 6). With target='int' the optimizer
minimizes the cumulative sum of the residuals, which turns white noise into a
random walk; the uncertainties must account for that correlation instead of
treating the integrated residuals as independent (which understated sigma by
2-25x on a three-line derivative spectrum)."""
import warnings

import numpy as np
import pytest

from torchspin.esfit import FitOptions, _apply_target_transform, _target_transform_adjoint, esfit


def test_default_target_is_fcn():
    assert FitOptions().target == 'fcn'


@pytest.mark.parametrize('target', ['fcn', 'int', 'diff'])
def test_adjoint_matches_explicit_matrix(target):
    n = 7
    T = np.column_stack([_apply_target_transform(e, target) for e in np.eye(n)])
    X = np.random.default_rng(0).standard_normal((n, 3))
    assert np.allclose(_target_transform_adjoint(X, target), T.T @ X, rtol=0, atol=1e-13)


_x = np.linspace(-1, 1, 300)
_true = np.array([0.02, 0.06, 0.3])


def _three_line_derivative(p):
    c, w, a = p
    u = (_x[:, None] - c - np.array([-1, 0, 1]) * a) / w
    return -(u * np.exp(-u * u)).sum(1)


@pytest.mark.parametrize('target', ['fcn', 'int'])
def test_uncertainties_cover_the_scatter(target):
    """Over noise realizations, (pfit - true) / pstd has a spread near 1.

    A fit that ends on a bound has no curvature error bar (pstd = nan since
    0.9.40); with target='int' a few realizations do, and they are left out."""
    Z = []
    for seed in range(60):
        data = _three_line_derivative(_true) + 0.05 * np.random.default_rng(seed).standard_normal(_x.size)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = esfit(data, _three_line_derivative, p0=_true * [1, 1.05, 0.97] + [0.005, 0, 0],
                      vary=np.array([0.05, 0.02, 0.05]),
                      options=FitOptions(method='levmar', target=target, verbosity=0, max_iter=200))
        lb, ub = r.p_start - [0.05, 0.02, 0.05], r.p_start + [0.05, 0.02, 0.05]
        on_bound = np.isclose(r.pfit, lb, rtol=0, atol=1e-6) | np.isclose(r.pfit, ub, rtol=0, atol=1e-6)
        assert np.array_equal(np.isnan(r.pstd), on_bound)
        Z.append((r.pfit - _true) / r.pstd)
    Z = np.array(Z)
    assert np.all(np.isfinite(Z).sum(0) >= 50)
    spread = np.nanstd(Z, 0)
    assert np.all((spread > 0.6) & (spread < 1.5)), spread
