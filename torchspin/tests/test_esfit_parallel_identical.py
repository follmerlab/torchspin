"""B4: with n_workers > 1 levmar evaluates its trial point and column-refresh probe in one
pool round.  Every method must give the serial fit bit for bit with workers (same pfit,
rmsd and number of model evaluations); simplex is the guard for the serial default."""
import numpy as np
import pytest

from torchspin import Experiment, Options, SpinSystem, pepper
from torchspin.esfit import FitOptions, esfit

X = np.linspace(320.0, 360.0, 256)


def two_lines(p):
    return (np.exp(-((X - p[0]) / p[1]) ** 2) + p[2] * np.exp(-((X - p[0] - 8.0) / p[1]) ** 2))


EXP = Experiment(mwFreq=9.5, Range=[330.0, 350.0], nPoints=256)


def nitroxide(p):
    s = SpinSystem(S=[0.5], g=[float(p[0]), float(p[1]), 2.003], Nucs='14N', A=[[20, 20, float(p[2])]],
                   lwpp=[0.4, 0])
    return pepper(s, EXP, Options(GridSize=[11, 4]))[1].numpy()


CASES = [
    (two_lines, [340.0, 2.0, 0.6], [338.5, 2.6, 0.45], [4.0, 1.5, 0.4]),
    (nitroxide, [2.008, 2.006, 85.0], [2.0075, 2.0066, 80.0], [0.002, 0.002, 10.0]),
]


@pytest.mark.parametrize('method', ['levmar', 'simplex'])
@pytest.mark.parametrize('model, ptrue, p0, vary', CASES, ids=['gauss', 'nitroxide'])
@pytest.mark.parametrize('workers', ['threads', 'processes'])
def test_parallel_fit_is_bit_identical(method, model, ptrue, p0, vary, workers):
    if workers == 'processes' and model is not two_lines:
        pytest.skip('one process-pool case is enough (spawn cost)')
    rng = np.random.default_rng(1)
    clean = model(np.array(ptrue))
    data = clean + 0.02 * np.abs(clean).max() * rng.standard_normal(clean.size)
    kw = dict(method=method, verbosity=0, compute_uncertainties=False, max_iter=200)
    r1 = esfit(data, model, p0=np.array(p0), vary=np.array(vary), options=FitOptions(**kw))
    r2 = esfit(data, model, p0=np.array(p0), vary=np.array(vary),
               options=FitOptions(n_workers=2, workers=workers, **kw))
    np.testing.assert_array_equal(r2.pfit, r1.pfit)
    assert r2.rmsd == r1.rmsd
    assert r2.n_evaluations == r1.n_evaluations
