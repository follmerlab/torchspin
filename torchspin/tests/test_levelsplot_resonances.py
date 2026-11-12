"""Resonance transitions in levelsplot with ``mwFreq`` (B78;
tests/data/ref_levelsplot.mat, generate_levelsplot_refs.m).

Before 0.9.19 levelsplot called ``resfields(sys, exp, dict)`` with the wrong
signature and swallowed the TypeError in a bare ``except Exception: pass``, so
no transition was ever drawn.  It now follows EasySpin levelsplot.m:244-272:
resonances of one crystal orientation (``Exp.SampleFrame = [-chi -theta
-phi]``, ``Threshold 0``, ``Freq2Field 0``), intensities relative to the
largest, lines below ``PlotThreshold`` skipped, each drawn between its two
levels at the resonance field and colored between the allowed and forbidden
colors by its relative intensity.

EasySpin's own intensities interpolate the eigenvectors linearly between
segment knots (``Opt.RediagLimit`` 0.95, resfields.m:1116-1160); torchspin
diagonalizes at every resonance field.  The reference therefore also stores
resfields with ``RediagLimit 2`` (always rediagonalize), which torchspin matches
to 6e-9; the drawn lines match EasySpin's levelsplot to 1.9e-4 (relative
intensity) and 2.9e-5 mT (field; EasySpin's fields are spline-model roots).
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from scipy.io import loadmat  # noqa: E402

import importlib  # noqa: E402
from torchspin.levelsplot import levelsplot, LevelsPlotOptions  # noqa: E402
from torchspin.spinsystem import SpinSystem  # noqa: E402
from torchspin.constants import PLANCK, BMAGN  # noqa: E402

# the package exports the function under the module's name: fetch the module itself
lp_mod = importlib.import_module('torchspin.levelsplot')

REF = Path(__file__).resolve().parents[2] / 'tests' / 'data' / 'ref_levelsplot.mat'

ALLOWED = np.array([1.0, 0.0, 0.0])
FORBIDDEN = np.array([0.8, 0.8, 0.8])


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _transition_lines(ax):
    """Vertical two-point lines on the level diagram (EasySpin Tag 'transition')."""
    return [ln for ln in ax.get_lines()
            if len(ln.get_xdata()) == 2 and ln.get_xdata()[0] == ln.get_xdata()[1]]


def _vec3(x):
    x = np.atleast_1d(np.asarray(x, dtype=float))
    return [float(x[0])] * 3 if x.size == 1 else [float(v) for v in x]


def _sys_from_mat(S):
    kw = dict(S=[float(S.S)], g=[_vec3(S.g)])
    if hasattr(S, 'D'):
        kw['D'] = [[float(v) for v in np.atleast_1d(S.D)]]
    if hasattr(S, 'Nucs'):
        kw['Nucs'] = str(S.Nucs)
        kw['A'] = [_vec3(S.A)]
    return SpinSystem(**kw)


def _ori_from_mat(ori):
    return ori if isinstance(ori, str) else [float(v) for v in np.atleast_1d(ori)]


def _load_cases():
    if not REF.exists():
        pytest.skip(f'reference file missing: {REF}')
    m = loadmat(REF, squeeze_me=True, struct_as_record=False)
    return {str(c.name): c for c in np.atleast_1d(m['cases'])}


CASE_NAMES = ['triplet_axial', 'triplet_rhombic', 'triplet_xy', 'quartet', 'doublet_14N']


@pytest.fixture(scope='module')
def cases():
    return _load_cases()


def test_doublet_single_line_at_h_nu_over_g_muB():
    """S = 1/2, g = 2, 9.5 GHz: one transition at h nu / (g muB) spanning 9.5 GHz."""
    sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
    fig, ax = levelsplot(sys, 'z', [300, 400], mwFreq=9.5)
    lines = _transition_lines(ax)
    assert len(lines) == 1
    B_expected = PLANCK * 9.5e9 / (2.0 * BMAGN) * 1e3            # 339.3767 mT
    assert B_expected == pytest.approx(339.3767, abs=1e-4)
    assert lines[0].get_xdata()[0] == pytest.approx(B_expected, abs=1e-6)
    E = sorted(lines[0].get_ydata())                              # GHz: -4.75, +4.75
    assert E[1] - E[0] == pytest.approx(9.5, abs=1e-9)
    assert E[0] == pytest.approx(-4.75, abs=1e-9)
    np.testing.assert_allclose(matplotlib.colors.to_rgb(lines[0].get_color()), ALLOWED)
    assert not any('no EPR resonances' in t.get_text() for t in ax.texts)


def test_no_resonances_in_range_message():
    sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
    fig, ax = levelsplot(sys, 'z', [0, 100], mwFreq=9.5)
    assert _transition_lines(ax) == []
    assert any('no EPR resonances' in t.get_text() for t in ax.texts)


@pytest.mark.parametrize('name', CASE_NAMES)
def test_resonances_match_easyspin_rediagonalized(cases, name):
    """Fields and relative intensities vs EasySpin resfields with RediagLimit 2."""
    c = cases[name]
    phi, theta, chi = lp_mod._parse_orientation(_ori_from_mat(c.Ori))
    B = np.atleast_1d(c.B).astype(float)
    B_res, I, tr = lp_mod._resonances(_sys_from_mat(c.Sys), phi, theta, chi, B, float(c.mwFreq))
    es_B = np.atleast_1d(c.BresRediag).astype(float)
    es_I = np.atleast_1d(c.IntRediag).astype(float)
    es_tr = np.atleast_2d(c.TrRediag).astype(int) - 1
    assert len(B_res) == len(es_B)
    got = {(int(u), int(v)): (b, i) for b, i, (u, v) in zip(B_res, I, tr)}
    ref = {(int(u), int(v)): (b, i) for b, i, (u, v) in zip(es_B, es_I, es_tr)}
    if len(got) == len(B_res) and len(ref) == len(es_B):
        assert set(got) == set(ref)
        keys = sorted(ref)
        gB = np.array([got[k][0] for k in keys]); rB = np.array([ref[k][0] for k in keys])
        gI = np.array([got[k][1] for k in keys]); rI = np.array([ref[k][1] for k in keys])
    else:   # a level pair with two resonances (quartet): match by field
        o_g, o_r = np.argsort(B_res), np.argsort(es_B)
        assert [tuple(t) for t in tr[o_g]] == [tuple(t) for t in es_tr[o_r]]
        gB, rB, gI, rI = B_res[o_g], es_B[o_r], I[o_g], es_I[o_r]
    # fields: max 2.9e-5 mT (triplet_xy); torchspin's are exact roots (gap - mwFreq
    # ~1e-12 MHz), EasySpin's within its spline model accuracy (8e-4 MHz there)
    np.testing.assert_allclose(gB, rB, rtol=0, atol=5e-5)
    # relative intensities: max 6e-9
    np.testing.assert_allclose(gI / gI.max(), rI / rI.max(), rtol=0, atol=1e-8)


@pytest.mark.parametrize('name', CASE_NAMES)
def test_drawn_transitions_match_easyspin_levelsplot(cases, name):
    """The drawn lines vs the 'transition' lines of EasySpin's levelsplot figure."""
    c = cases[name]
    fig, ax = levelsplot(_sys_from_mat(c.Sys), _ori_from_mat(c.Ori), np.atleast_1d(c.B).astype(float),
                         mwFreq=float(c.mwFreq))
    lines = _transition_lines(ax)
    ud = np.atleast_2d(c.lineUserData)                    # [level1 level2 B absintensity]
    es_Y = np.atleast_2d(c.lineY)
    assert len(lines) == ud.shape[0]
    o_es = np.argsort(ud[:, 2])
    o_py = np.argsort([ln.get_xdata()[0] for ln in lines])
    for k_es, k_py in zip(o_es, o_py):
        ln = lines[k_py]
        assert ln.get_xdata()[0] == pytest.approx(ud[k_es, 2], abs=5e-5)              # max 2.9e-5 mT
        np.testing.assert_allclose(ln.get_ydata(), es_Y[k_es], rtol=0, atol=2e-6)   # GHz, max 8e-7
        rgb = np.array(matplotlib.colors.to_rgb(ln.get_color()))
        a = ud[k_es, 3]
        # EasySpin's relative intensities carry its eigenvector interpolation: max 1.5e-4
        np.testing.assert_allclose(rgb, a * ALLOWED + (1 - a) * FORBIDDEN, rtol=0, atol=2e-4)


def test_plot_threshold_skips_weak_lines(cases):
    """doublet_14N: two lines ~2e-8 relative (below PlotThreshold 1e-6) are not drawn."""
    c = cases['doublet_14N']
    sys = _sys_from_mat(c.Sys)
    B = np.atleast_1d(c.B).astype(float)
    fig, ax = levelsplot(sys, _ori_from_mat(c.Ori), B, mwFreq=float(c.mwFreq))
    assert len(_transition_lines(ax)) == 7 == np.atleast_2d(c.lineUserData).shape[0]
    fig2, ax2 = levelsplot(sys, _ori_from_mat(c.Ori), B, mwFreq=float(c.mwFreq),
                           opt=LevelsPlotOptions(PlotThreshold=0.0))
    assert len(_transition_lines(ax2)) == 9


def test_stick_spectrum(cases):
    c = cases['triplet_axial']
    fig, axs = levelsplot(_sys_from_mat(c.Sys), _ori_from_mat(c.Ori), np.atleast_1d(c.B).astype(float),
                          mwFreq=float(c.mwFreq), opt=LevelsPlotOptions(StickSpectrum=True))
    sticks = sorted((ln.get_xdata()[0], ln.get_ydata()[1]) for ln in _transition_lines(axs[1]))
    ud = np.atleast_2d(c.lineUserData)
    ref = sorted(zip(ud[:, 2], ud[:, 3]))
    np.testing.assert_allclose([s[0] for s in sticks], [r[0] for r in ref], rtol=0, atol=1e-5)    # max 2.9e-6 mT
    np.testing.assert_allclose([s[1] for s in sticks], [r[1] for r in ref], rtol=0, atol=1e-4)    # max 6.4e-5


def test_resonance_errors_propagate(monkeypatch):
    """An error in the resonance computation is raised, not silently skipped."""
    import torchspin.resfields_batch as rb

    def boom(*args, **kwargs):
        raise RuntimeError('resonance solver failed')

    monkeypatch.setattr(rb, 'resfields_batch', boom)
    sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
    with pytest.raises(RuntimeError, match='resonance solver failed'):
        levelsplot(sys, 'z', [300, 400], mwFreq=9.5)
    # without mwFreq the resonance code is not reached
    fig, ax = levelsplot(sys, 'z', [300, 400])
    assert _transition_lines(ax) == []


def test_no_grad_leak_from_resonances():
    sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
    B_res, I, tr = lp_mod._resonances(sys, 0.0, 0.0, 0.0, np.array([300.0, 400.0]), 9.5)
    assert isinstance(B_res, np.ndarray) and B_res.shape == (1,)
    assert tr.tolist() == [[0, 1]]
    assert torch.is_grad_enabled()
