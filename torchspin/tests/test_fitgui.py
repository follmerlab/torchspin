"""Headless checks of the fitting panel: session runs in a thread, panel builds without a display."""
import time

import numpy as np
import pytest

X = np.linspace(0, 1, 64)


def _model(p):
    # Gaussian peak: position and width are not absorbed by esfit's amplitude/baseline autoscale
    return np.exp(-((X - p[0]) / p[1]) ** 2)


def test_session_runs_in_thread_and_records_history():
    from torchspin.esfit import FitOptions
    from torchspin.fitgui import FitSession
    data = _model([0.4, 0.15])
    s = FitSession(data, _model, [0.5, 0.25], vary=[0.3, 0.2], names=['a', 'b'],
                   options=FitOptions(method='simplex', max_iter=200, compute_uncertainties=False), progress_interval=0.0)
    s.start(); s.join(30)
    assert not s.running and s.state == 'done' and s.error is None
    assert s.result is not None and np.allclose(s.result.pfit, [0.4, 0.15], atol=2e-3)
    assert len(s.history) > 0 and s.history[-1][0] == s.result.n_evaluations


def test_session_stop_returns_best_so_far():
    from torchspin.esfit import FitOptions
    from torchspin.fitgui import FitSession

    def slow(p):
        time.sleep(0.005)
        return _model(p)
    data = _model([0.4, 0.15])
    s = FitSession(data, slow, [0.5, 0.25], vary=[0.3, 0.2], options=FitOptions(method='swarm', swarm_size=8, max_iter=10 ** 5, compute_uncertainties=False), progress_interval=0.0)
    s.start(); time.sleep(0.3); s.stop(); s.join(30)
    assert s.state == 'interrupted' and s.result.interrupted and 'stop_when' in s.result.message


def test_panel_builds_headless_and_refreshes():
    pytest.importorskip('ipywidgets')
    import matplotlib
    matplotlib.use('Agg')
    from torchspin.esfit import FitOptions
    from torchspin.fitgui import FitSession, FitPanel
    data = _model([0.4, 0.15])
    s = FitSession(data, _model, [0.5, 0.25], vary=[0.3, 0.2], names=['a', 'b'],
                   options=FitOptions(method='levmar', max_iter=50, compute_uncertainties=False), progress_interval=0.0)
    panel = FitPanel(s, x=X, xlabel='x')
    assert np.allclose(panel.p_current, [0.5, 0.25]) and panel.result is None
    panel.fixes[1].value = True                     # fix b, fit a only
    panel.start(); s.join(30); panel.refresh()
    assert panel.result is not None and s.best_x[1] == 0.25 and abs(s.best_x[0] - 0.4) < 0.1   # b fixed, a fitted
    assert not panel.b_start.disabled and panel.b_stop.disabled
    assert panel.sliders[0].value == pytest.approx(s.best_x[0])


def test_panel_trf_preserves_absolute_bounds_and_unfixes():
    pytest.importorskip('ipywidgets')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from torchspin.esfit import FitOptions
    from torchspin.fitgui import FitSession, FitPanel
    seen = []

    def observed(p):
        seen.append(np.array(p, dtype=float))
        return _model(p)
    # explicit lb/ub; the true position 0.9 lies outside ub = 0.8
    s = FitSession(_model([0.9, 0.18]), observed, [0.7, 0.15], lb=[0.1, 0.05], ub=[0.8, 0.3],
                   options=FitOptions(method='trf', target='fcn', max_iter=50, compute_uncertainties=False),
                   progress_interval=0.0)
    panel = FitPanel(s, x=X)
    assert panel.method.value == 'trf' and 'trf' in panel.method.options
    panel.fixes[1].value = True                                  # fix the width
    panel.start(); s.join(30); panel.refresh()
    assert s.error is None and not s.running
    assert s.best_x[0] > 0.799 and s.best_x[1] == 0.15          # position pinned at ub, width untouched
    assert np.array_equal(s.result.p_fixed, [False, True])
    panel.fixes[1].value = False                                 # release the width for the next run
    panel.start(); s.join(30); panel.refresh()
    assert s.error is None and not s.running
    assert np.array_equal(s.result.p_fixed, [False, False])
    assert abs(s.best_x[1] - 0.15) > 1e-3                        # it actually moved
    seen = np.array(seen)                                        # bounds held across both runs
    assert np.all(seen >= [0.1, 0.05]) and np.all(seen <= [0.8, 0.3])
    np.testing.assert_allclose(panel.l_res.get_ydata(), s.data - s.result.fit, atol=1e-12)   # residual = data - fit
    plt.close(panel.fig)


def test_session_vary_mode_fix_is_per_run():
    from torchspin.esfit import FitOptions
    from torchspin.fitgui import FitSession
    s = FitSession(_model([0.4, 0.15]), _model, [0.5, 0.25], vary=[0.3, 0.2],
                   options=FitOptions(method='trf', target='fcn', max_iter=50, compute_uncertainties=False),
                   progress_interval=0.0)
    s.start(vary=[0.3, 0.0]); s.join(30)
    assert s.state == 'done' and np.array_equal(s.result.p_fixed, [False, True]) and s.best_x[1] == 0.25
    assert np.array_equal(s.vary, [0.3, 0.2])                    # start(vary=...) must not overwrite the session's vary
    s.start(); s.join(30)                                        # default: everything free again
    assert s.state == 'done' and not s.result.p_fixed.any() and abs(s.best_x[1] - 0.15) < 2e-3


def test_session_rejects_start_outside_explicit_bounds():
    from torchspin.esfit import FitOptions
    from torchspin.fitgui import FitSession
    s = FitSession(_model([0.4, 0.15]), _model, [0.5, 0.25], lb=[0.1, 0.05], ub=[0.8, 0.3],
                   options=FitOptions(method='trf', compute_uncertainties=False), progress_interval=0.0)
    with pytest.raises(ValueError, match='lb/ub'):
        s.start(p0=[0.9, 0.25])
    assert s.state == 'idle' and not s.running and np.array_equal(s.p0, [0.5, 0.25])   # untouched


def test_panel_reenables_controls_when_start_fails(monkeypatch):
    pytest.importorskip('ipywidgets')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from torchspin.esfit import FitOptions
    from torchspin.fitgui import FitSession, FitPanel
    s = FitSession(_model([0.4, 0.15]), _model, [0.5, 0.25], lb=[0.1, 0.05], ub=[0.8, 0.3],
                   options=FitOptions(method='trf', compute_uncertainties=False), progress_interval=0.0)
    panel = FitPanel(s, x=X)

    def refuse(**kw):
        raise ValueError('starting parameters must lie within lb/ub')
    monkeypatch.setattr(s, 'start', refuse)
    with pytest.raises(ValueError):
        panel.start()
    assert not panel.b_start.disabled and panel.b_stop.disabled
    assert all(not sl.disabled for sl in panel.sliders) and 'error' in panel.status.value
    plt.close(panel.fig)
