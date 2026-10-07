"""Usability defects reported against 0.3.0 by users fitting a Cu(II) spectrum.

These are the issues that cost a user time without producing a wrong number:
a silent convergence trap, a bare ModuleNotFoundError, bounds that let a
linewidth go negative, and an abscissa whose units are not where you look for
them.
"""

import warnings

import numpy as np
import pytest

from torchspin import Experiment, Options, SpinSystem, pepper


# --- Options.GridSize convergence warning -----------------------------------

A_CU = [15.3311, 15.3311, 646.629]
A_N = [45.0, 45.0, 45.0]


def _cupc(n_nitrogen=4):
    return SpinSystem(S=[0.5], g=[[2.04894, 2.04894, 2.181]],
                      Nucs=['63Cu'] + ['14N'] * n_nitrogen,
                      A=[A_CU] + [A_N] * n_nitrogen, lw=[0.542209, 0.0])


def _cupc_exp():
    return Experiment(mwFreq=9.347144, Range=[233.8, 433.8], nPoints=2667, Harmonic=1)


def _hybrid(**kw):
    return Options(Method='hybrid', HybridCoreNuclei=[1], Verbosity=0, **kw)


def test_default_grid_is_flagged_for_narrow_lines_on_a_large_hyperfine():
    """GridSize=[19,4] is badly unconverged for a 0.54 mT line on a 647 MHz
    hyperfine coupling, in EasySpin too, and a single spectrum does not show it."""
    from torchspin import grid_convergence
    c = grid_convergence(_cupc(), _cupc_exp(), _hybrid(GridSize=[19, 4]))
    assert not c.converged, c
    assert c.amplitude_ratio < 0.99 or c.cosine < 0.999, c
    assert 'NOT converged' in c.report()
    assert str(c.refined_grid_size[0]) in c.report(), 'the report should name a grid to use'


def test_converged_grid_is_reported_as_converged():
    from torchspin import grid_convergence
    c = grid_convergence(_cupc(), _cupc_exp(), _hybrid(GridSize=[361, 4]))
    assert c.converged, c
    assert c.cosine > 0.9999 and abs(c.amplitude_ratio - 1) < 0.01


@pytest.mark.parametrize('grid', [[19, 4], 50, [31, 4]])
def test_no_false_positive_on_an_ordinary_nitroxide(grid):
    """An ordinary system on the default grid must come back converged, or the
    check is noise.  A step-size heuristic fails exactly here: it flags dozens
    of simulations that agree with EasySpin to cosine 0.9999, which is why this
    is a two-grid comparison instead."""
    from torchspin import grid_convergence
    sys = SpinSystem(S=[0.5], g=[[2.0088, 2.0061, 2.0027]], Nucs=['14N'],
                     A=[[16.0, 16.0, 86.0]], lw=[1.0, 0.0])
    exp = Experiment(mwFreq=9.5, Range=[330, 350], nPoints=1024, Harmonic=1)
    c = grid_convergence(sys, exp, Options(GridSize=grid, Verbosity=0))
    assert c.converged, c


def test_grid_convergence_rejects_a_pointless_refinement():
    from torchspin import grid_convergence
    with pytest.raises(ValueError, match='at least 2'):
        grid_convergence(_cupc(), _cupc_exp(), _hybrid(GridSize=[19, 4]), refine=1)


def test_pepper_itself_stays_quiet():
    """pepper must not warn about the grid on its own: the only trustworthy
    check costs a second simulation, so it is opt-in rather than automatic."""
    sys = SpinSystem(S=[0.5], g=[[2.0088, 2.0061, 2.0027]], Nucs=['14N'],
                     A=[[16.0, 16.0, 86.0]], lw=[1.0, 0.0])
    exp = Experiment(mwFreq=9.5, Range=[330, 350], nPoints=1024, Harmonic=1)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        pepper(sys, exp, Options(GridSize=[19, 4], Verbosity=0))
    assert [str(w.message) for w in rec if 'grid' in str(w.message).lower()] == []


# --- fitgui optional dependencies -------------------------------------------

def test_missing_gui_dependency_names_the_extra():
    """The failure used to be a bare ModuleNotFoundError with no pointer."""
    from torchspin import fitgui
    with pytest.raises(ImportError) as exc:
        fitgui._require('a_module_that_does_not_exist', 'testing')
    msg = str(exc.value)
    assert 'torchspin[gui]' in msg, msg
    assert 'a_module_that_does_not_exist' in msg, msg


def _extra_line(name):
    """The `name = [...]` line from [project.optional-dependencies], or None.

    Read as text rather than parsed: tomllib is stdlib only from Python 3.11
    and this package supports 3.10, and the assertions below only need to know
    which names appear on which line.
    """
    from pathlib import Path
    pyproject = Path(__file__).parent.parent.parent / 'pyproject.toml'
    if not pyproject.exists():
        return None
    in_extras = False
    for raw in pyproject.read_text().splitlines():
        line = raw.strip()
        if line.startswith('['):
            in_extras = line == '[project.optional-dependencies]'
            continue
        if in_extras and line.split('=')[0].strip() == name:
            return line.lower()
    return ''


def test_gui_extra_covers_what_the_panel_imports():
    """ipython and cloudpickle were missing from the gui extra even though
    FitPanel.show() and its n_workers control need them."""
    gui = _extra_line('gui')
    if gui is None:
        pytest.skip('pyproject.toml not available (installed package)')
    assert gui, 'no gui extra found in [project.optional-dependencies]'
    for pkg in ('ipywidgets', 'matplotlib', 'ipympl', 'ipython', 'cloudpickle'):
        assert pkg in gui, f'{pkg} missing from the gui extra: {gui}'
    # and the test extra must let the fitgui tests actually run
    assert 'ipywidgets' in (_extra_line('test') or '')


# --- esfit vary bounds ------------------------------------------------------

def test_vary_does_not_push_a_linewidth_negative():
    """EasySpin's esfit clips the lower bound at zero for fields that cannot be
    negative (esfit.m nonnegFieldNames); the dict style now does the same."""
    from torchspin.esfit import TorchSpinParameterHandler
    sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]], lw=[0.5, 0.0], weight=0.001)
    exp = Experiment(mwFreq=9.5, Range=[330, 350], nPoints=256, Harmonic=1)
    h = TorchSpinParameterHandler(
        p0={'Sys': sys, 'Exp': exp},
        vary={'Sys': {'lw': [2.0, 0.0], 'weight': 0.2, 'g': [[0.05, 0.05, 0.05]]}},
    )
    bounds = {name: (e['lb'], e['ub']) for name, e in zip(h.pnames, h.param_map)}
    # lw and weight are clipped at zero ...
    assert bounds['Sys.lw[0]'][0] == 0.0, bounds['Sys.lw[0]']
    assert bounds['Sys.weight'][0] == 0.0, bounds['Sys.weight']
    # ... and the upper bounds are untouched
    assert bounds['Sys.lw[0]'][1] == pytest.approx(2.5)
    assert bounds['Sys.weight'][1] == pytest.approx(0.201)
    # g may legitimately be anything, so it is not clipped
    assert bounds['Sys.g[0,0]'][0] == pytest.approx(1.95)


def test_explicit_bounds_are_never_silently_moved():
    """Clipping applies to vary only: an explicit lb is the user's statement."""
    from torchspin.esfit import TorchSpinParameterHandler
    sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]], lw=[0.5, 0.0])
    exp = Experiment(mwFreq=9.5, Range=[330, 350], nPoints=256, Harmonic=1)
    h = TorchSpinParameterHandler(
        p0={'Sys': sys, 'Exp': exp},
        lb={'Sys': {'lw': [-1.0, 0.0]}},
        ub={'Sys': {'lw': [2.0, 0.0]}},
    )
    lb = {name: e['lb'] for name, e in zip(h.pnames, h.param_map)}
    assert lb['Sys.lw[0]'] == pytest.approx(-1.0)


# --- eprload units ----------------------------------------------------------

def test_eprload_docstring_states_the_units():
    """The units live in KNOWN_LIMITATIONS but bit users at the API."""
    from torchspin import eprload
    doc = eprload.__doc__
    assert 'gauss' in doc.lower(), 'the Bruker abscissa unit is not in the docstring'
    assert 'mT' in doc, 'the docstring should say what the rest of torchspin uses'
    assert '/ 10' in doc, 'the docstring should show the conversion'


def test_eprload_info_labels_the_axis_units(tmp_path, capsys):
    """eprload_info printed a bare range, so a gauss axis looked like mT."""
    from torchspin import eprload_info
    # Minimal BES3T pair, field axis in gauss as Bruker writes it.
    n = 32
    (tmp_path / 'f.DTA').write_bytes(
        np.linspace(-1.0, 1.0, n).astype('>f8').tobytes())
    (tmp_path / 'f.DSC').write_text(
        "#DESC\t1.2\nDSRC\tEXP\nBSEQ\tBIG\nIKKF\tREAL\nXTYP\tIDX\n"
        "YTYP\tNODATA\nZTYP\tNODATA\nIRFMT\tD\n"
        f"XPTS\t{n}\nXMIN\t3400.000000\nXWID\t100.000000\nXUNI\t'G'\n")
    eprload_info(tmp_path / 'f.DTA')
    out = capsys.readouterr().out
    assert ' G ' in out, out
    assert 'divide by 10 for mT' in out, out
