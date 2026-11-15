"""Converged references for the coarse-vs-converged checks moved by the centered
projection bins (B85, 0.9.35): test_missing_knots.py, test_interp_nan_edges_matlab_validation.py,
test_sphgrid_maxphi_matlab_validation.py and test_pepper_nanlinear_matlab_validation.py.

Each reference is the case of generate_missing_knots_refs.py computed on a field axis F
times finer than the test's (F = 32; 16 for the 2048/2801-point cases), extended by whole
fine increments on both sides so that no transition leaves it (as there), on the
GridSize [181 1] orientation grid without interpolation, and then sampled at the test's
points x_k (point values of the converged spectrum, not cell averages: the projection
bins of the test spectrum and of ref_missing_knots.npz -- same axis -- share their
discretization, so those references are not converged in the field axis).

Run from the repository root (about 5 minutes):

    OMP_NUM_THREADS=4 PYTHONPATH=. python tests/data/generate_converged_b85_refs.py

Writes tests/data/ref_converged_b85.npz (keys ``<case>``, ``<case>_x``).
"""
import importlib.util
from pathlib import Path

import numpy as np

from torchspin import Options, pepper

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('gen', HERE / 'generate_missing_knots_refs.py')
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

FINE = {'case3': 32, 'case3_c2h': 32, 'case3r': 16, 'b50_h1': 32, 'b50_stick_wide': 16, 'oh_auto': 16}
GRID = [181, 1]


def reference(name, F):
    build, lo_ext, hi_ext = gen.CASES[name]
    s, e, okw = build()
    rng, n = tuple(e.Range), int(e.nPoints)
    nf = F * (n - 1) + 1
    wr, wn, i0 = gen.wide_axis(rng, nf, lo_ext, hi_ext)
    s, ew, okw = build(wr, wn)
    x, y = pepper(s, ew, Options(GridSize=GRID, Verbosity=0, **okw))
    x = x.numpy()[i0:i0 + nf:F]
    y = y.detach().numpy().ravel()[i0:i0 + nf:F]
    assert np.allclose(x, np.linspace(rng[0], rng[1], n), rtol=0, atol=1e-9), name
    print(f'{name}: {wn} fine points ({F}x), {GRID}')
    return x, y


if __name__ == '__main__':
    out = {}
    for nm, F in FINE.items():
        out[nm + '_x'], out[nm] = reference(nm, F)
    np.savez_compressed(HERE / 'ref_converged_b85.npz', **out)
    print('wrote tests/data/ref_converged_b85.npz')
