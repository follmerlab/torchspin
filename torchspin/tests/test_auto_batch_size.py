"""Automatic orientation batch of pepper's matrix route (B138).

The orientations of a batch share one field-knot set, so a batch of many
orientations of a large spin system holds orientations x 100+ knots x dim^2
numbers.  pepper budgets 200 knots per orientation; the spectrum must not
depend on that choice for these systems.
"""
import numpy as np
import pytest

from torchspin import Experiment, Options, SpinSystem, pepper
from torchspin.experiment import auto_batch_size


class TestRule:
    def test_default_budget_is_unchanged(self):
        # pepper_autograd, salt and the hybrid method keep the 10-knot budget
        assert auto_batch_size(None, 36, 5000) == int(2.5e7 // (10 * 36 * 36))
        assert auto_batch_size(None, 72, 5000) == int(2.5e7 // (10 * 72 * 72))

    @pytest.mark.parametrize('dim, expected', [(3, 5000), (8, 1953), (36, 96), (72, 24), (216, 10), (648, 10)])
    def test_matrix_budget(self, dim, expected):
        assert auto_batch_size(None, dim, 5000, knots=200) == expected

    def test_never_more_than_the_grid(self):
        assert auto_batch_size(None, 36, 55, knots=200) == 55

    def test_floor_of_ten(self):
        assert auto_batch_size(None, 1000, 4, knots=200) == 10

    @pytest.mark.parametrize('knots', [10, 200])
    def test_explicit_batch_wins(self, knots):
        assert auto_batch_size(7, 36, 5000, knots=knots) == 7
        assert auto_batch_size(4000, 72, 5000, knots=knots) == 4000


CASES = {
    # name: (system, experiment, options, orientations, automatic batch)
    'mn_ci': (dict(S=2.5, g=2.0, D=[400.0, 400.0, -800.0], Nucs='55Mn', A=250.0, lw=1.5),
              dict(mwFreq=9.5, Range=[250, 450]), dict(GridSize=9, GridSymmetry='Ci'), 96),
    # inequivalent nitrogens: equivalent ones run as total-spin blocks of dimension <= 40
    'cu2n_d2h': (dict(S=0.5, g=[2.04, 2.06, 2.25], Nucs='63Cu,14N,14N',
                      A=[[30, 40, 500], [40, 40, 40], [36, 36, 44]], lwpp=0.5),
                 dict(mwFreq=9.5, Range=[260, 360]), dict(GridSize=[9, 2]), 24),
    'triplet_ci': (dict(S=1, g=[2.0, 2.01, 2.02], D=[900, 150], lwpp=1.0),
                   dict(mwFreq=9.5, Range=[250, 430]), dict(GridSize=15, GridSymmetry='Ci'), None),
}


def _spec(name, **opt):
    sk, ek, ok, _ = CASES[name]
    return pepper(SpinSystem(**sk), Experiment(nPoints=512, Harmonic=0, **ek),
                  Options(Verbosity=0, **ok, **opt))[1].numpy()


@pytest.mark.parametrize('name', list(CASES))
def test_automatic_batch_gives_the_one_batch_spectrum(name):
    auto = _spec(name)
    one = _spec(name, BatchSize=100000)
    assert auto.max() > 0
    assert np.abs(auto - one).max() <= 1e-6 * one.max()


@pytest.fixture
def batch_sizes(monkeypatch):
    """Orientations per call of the batched resonance search."""
    import importlib
    pp = importlib.import_module('torchspin.pepper')
    sizes = []
    orig = pp.resfields_batch

    def spy(H0, mux, muy, muz, phi, *a, **k):
        sizes.append(len(phi))
        return orig(H0, mux, muy, muz, phi, *a, **k)

    monkeypatch.setattr(pp, 'resfields_batch', spy)
    return sizes


@pytest.mark.parametrize('name', ['mn_ci', 'cu2n_d2h'])
def test_automatic_batch_splits_large_systems(name, batch_sizes):
    """The grid is larger than the automatic batch, so the search runs in several calls."""
    _spec(name)
    assert max(batch_sizes) == CASES[name][3]
    assert len(batch_sizes) > 1


def test_explicit_batch_is_used(batch_sizes):
    _spec('triplet_ci', BatchSize=7)
    assert max(batch_sizes) == 7
