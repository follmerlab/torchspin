"""Converged references for torchspin/tests/test_missing_knots.py (B88, B75, B91).

Each reference is a no-interpolation spectrum (GridSize [181 1]) computed on a
sweep axis that extends the test's Exp.Range by whole field increments on both
sides (same increment, so the test's points are a subset), wide enough that no
transition leaves it at any knot, and then cropped back to the test's axis.
The script checks that the re-search of missing knots finds nothing on the wide
axis (no level pair leaves it; pairs that do not resonate at some orientations
at any field, as in s52_parallel, stay as they are), so the references do not
depend on the missing-knot handling they test.  The wide
sweep holds every resonance of the test window and keeps the tails of lines centered
outside the test window, so it is the converged target for the narrow sweep
(up to the grid: [181 1] vs [121 1] differ by cosine 1 - 5e-4 for case 1).

The B48/B50, oh_auto and s52_parallel entries are the inputs of the EasySpin
cases in test_interp_nan_edges_matlab_validation.py,
test_sphgrid_maxphi_matlab_validation.py and
test_pepper_parallel_matlab_validation.py whose ranges cut transitions.

Run from the repository root (a few minutes, single thread):

    OMP_NUM_THREADS=1 PYTHONPATH=. python tests/data/generate_missing_knots_refs.py

Writes tests/data/ref_missing_knots.npz (keys ``<case>``: cropped spectrum,
``<case>_x``: its field axis).
"""
import os
import sys
from pathlib import Path

import numpy as np

import torchspin.pepper  # noqa: F401
from torchspin import SpinSystem, Experiment, Options, pepper

P = sys.modules['torchspin.pepper']


def case1(rng=(290.0, 350.0), n=1024):
    """B88: 63Cu with a tilted hyperfine frame (auto grid C2h)."""
    s = SpinSystem(S=0.5, g=[2.05, 2.05, 2.25], Nucs='63Cu', A=[50, 50, 500], lw=0.3, AFrame=[0, 0.4, 0])
    return s, Experiment(mwFreq=9.5, Range=list(rng), nPoints=n, Harmonic=1), {}


def case2(rng=(300.0, 400.0), n=256):
    """B91: 91Zr with a large isotropic hyperfine on an explicit D2h grid."""
    s = SpinSystem(S=0.5, g=[2.0, 1.95, 1.9], Nucs='91Zr', A=[780, 780, 780], lw=[1.0, 0.0])
    return s, Experiment(mwFreq=9.65, Range=list(rng), nPoints=n, Harmonic=1), {'GridSymmetry': 'D2h'}


def case3(rng=(200.0, 480.0), n=1024):
    """B31: S = 5/2 with D and B4 on an explicit D2h grid
    (test_pepper_nanlinear_matlab_validation.py, s52_b4_D2h_h1)."""
    s = SpinSystem(S=[2.5], g=2.0, D=[[300.0, 0.0]], lw=[1.0, 0.0],
                   B=[None, [[30.0, 0, 0, 0, 10.0, 0, 0, 0, 0]]])
    return s, Experiment(mwFreq=9.5, Range=list(rng), nPoints=n, Harmonic=1), {'GridSymmetry': 'D2h'}


def case3r(rng=(150.0, 450.0), n=2048):
    """Rhombic S = 5/2 on the default grid
    (test_pepper_nanlinear_matlab_validation.py, s52_rhombic_h1)."""
    s = SpinSystem(S=[2.5], g=2.0, D=[[400.0, 100.0]], lw=[1.5, 0.0])
    return s, Experiment(mwFreq=9.5, Range=list(rng), nPoints=n, Harmonic=1), {}


def case3_h0(rng=(200.0, 480.0), n=1024):
    """case3 at Harmonic 0 (s52_b4_D2h_h0)."""
    s, e, o = case3(rng, n)
    return s, Experiment(mwFreq=9.5, Range=list(rng), nPoints=n, Harmonic=0), o


def case3_c2h(rng=(200.0, 480.0), n=1024):
    """case3 on an explicit C2h grid (s52_b4_C2h_h1)."""
    s, e, _ = case3(rng, n)
    return s, e, {'GridSymmetry': 'C2h'}


def b48(rng=(325.0, 337.0), n=601, lw=None, harmonic=0):
    """B48 (test_interp_nan_edges_matlab_validation.py): rhombic S = 1/2, the
    range cuts the g = 2.1 edge; stick (no lw) or lw 1.5 mT."""
    kw = dict(S=[0.5], g=[2.0, 2.05, 2.1])
    if lw:
        kw['lw'] = [lw]
    return SpinSystem(**kw), Experiment(mwFreq=9.5, Range=list(rng), nPoints=n, Harmonic=harmonic), {}


def b48_h0(rng=(325.0, 337.0), n=601):
    return b48(rng, n, lw=1.5, harmonic=0)


def b48_h1(rng=(325.0, 337.0), n=601):
    return b48(rng, n, lw=1.5, harmonic=1)


def b50_h1(rng=(290.0, 390.0), n=1024):
    """B50: the case3 system on [290 390] mT."""
    return case3(rng, n)


def b50_stick(rng=(290.0, 390.0), n=1024):
    """B50 stick spectrum (no lw, Harmonic 0)."""
    s = SpinSystem(S=[2.5], g=2.0, D=[[300.0, 0.0]], B=[None, [[30.0, 0, 0, 0, 10.0, 0, 0, 0, 0]]])
    return s, Experiment(mwFreq=9.5, Range=list(rng), nPoints=n, Harmonic=0), {'GridSymmetry': 'D2h'}


def b50_stick_wide(rng=(200.0, 480.0), n=2801):
    return b50_stick(rng, n)


def oh_auto(rng=(250.0, 430.0), n=1024):
    """test_sphgrid_maxphi_matlab_validation.py oh_auto: cubic B4 only (Oh grid)."""
    s = SpinSystem(S=[2.5], g=2.0, B=[None, [[150.0, 0, 0, 0, 30.0, 0, 0, 0, 0]]], lw=[1.0, 0.0])
    return s, Experiment(mwFreq=9.5, Range=list(rng), nPoints=n, Harmonic=1), {}


def s52_parallel(rng=(20.0, 500.0), n=2048):
    """test_pepper_parallel_matlab_validation.py S52_parallel: S = 5/2,
    D = [1500 300] MHz, parallel-mode detection, default grid."""
    s = SpinSystem(S=[2.5], g=2.0, D=[[1500.0, 300.0]], lw=[3.0, 0.0])
    return s, Experiment(mwFreq=9.5, Range=list(rng), nPoints=n, Harmonic=1, Mode='parallel'), {}


# name: (builder, extension below, extension above) in mT; the lower extension
# stops at the last whole increment above 0 mT
CASES = {
    'case1': (case1, 60.0, 0.0),
    'case2': (case2, 300.0, 150.0),
    'case3': (case3, 200.0, 200.0),
    'case3_h0': (case3_h0, 200.0, 200.0),
    'case3_c2h': (case3_c2h, 200.0, 200.0),
    'case3r': (case3r, 150.0, 150.0),
    'b48_stick': (b48, 15.0, 15.0),
    'b48_h0': (b48_h0, 15.0, 15.0),
    'b48_h1': (b48_h1, 15.0, 15.0),
    'b50_h1': (b50_h1, 300.0, 300.0),
    'b50_stick': (b50_stick, 300.0, 300.0),
    'b50_stick_wide': (b50_stick_wide, 200.0, 200.0),
    'oh_auto': (oh_auto, 250.0, 300.0),
    's52_parallel': (s52_parallel, 20.0, 1000.0),
}
REF_GRID = [181, 1]


def wide_axis(rng, n, lo_ext, hi_ext):
    """Axis with the same increment, extended by whole increments.
    Returns (range, nPoints, index of rng[0])."""
    dx = (rng[1] - rng[0]) / (n - 1)
    k0 = min(int(round(lo_ext / dx)), int(np.floor(rng[0] / dx + 1e-9)))
    k1 = int(round(hi_ext / dx))
    return (rng[0] - k0 * dx, rng[1] + k1 * dx), n + k0 + k1, k0


def interior(x, sys):
    """Mask of the comparison window: the sweep without its ends +- 5 FWHM
    (at most 20 % of the width each), where the final line-width convolution
    sees lines centered outside the range only through the spectrum itself
    (handled by the edge padding of Options.PaddingMultiplier, #53).  Without
    a line width (stick spectra) every point."""
    lw = float(max(np.atleast_1d(sys.get_lw())))
    if lw <= 0:
        return np.ones(np.size(x), bool)
    k = min(5.0 * lw, 0.2 * (x[-1] - x[0]))
    return (x > x[0] + k) & (x < x[-1] - k)


def metrics(y, r, mask=None):
    """(1 - cosine, max|y - r| / max|r|) on ``mask``."""
    y = np.asarray(y, float).ravel(); r = np.asarray(r, float).ravel()
    if mask is not None:
        y, r = y[mask], r[mask]
    return (1.0 - float(y @ r / (np.linalg.norm(y) * np.linalg.norm(r))),
            float(np.abs(y - r).max() / np.abs(r).max()))


def reference(name):
    build, lo_ext, hi_ext = CASES[name]
    s, e, okw = build()
    rng, n = tuple(e.Range), int(e.nPoints)
    wr, wn, i0 = wide_axis(rng, n, lo_ext, hi_ext)
    s, ew, okw = build(wr, wn)
    taken = []
    orig = P._research_missing_knots

    def spy(lists, *a, **k):
        out = orig(lists, *a, **k)
        taken.append(out is not lists)
        return out
    P._research_missing_knots = spy
    try:
        x, y = pepper(s, ew, Options(GridSize=REF_GRID, Verbosity=0, **okw))
    finally:
        P._research_missing_knots = orig
    # The re-search must take nothing on the wide axis: pairs that still miss
    # resonances there do not resonate at some orientations at any field (e.g.
    # looping ones); anything it finds means the axis still cuts a transition.
    if any(taken):
        raise RuntimeError(f'{name}: the wide reference axis {wr} still cuts transitions; widen it')
    x = x.detach().numpy().ravel()[i0:i0 + n]
    y = y.detach().numpy().ravel()[i0:i0 + n]
    assert np.allclose(x, np.linspace(rng[0], rng[1], n), rtol=0, atol=1e-9), name
    print(f'{name}: wide axis {wr[0]:.4f}-{wr[1]:.4f} mT ({wn} points), cropped at {i0}')
    return x, y


if __name__ == '__main__':
    path = Path(__file__).resolve().parent / 'ref_missing_knots.npz'
    out = dict(np.load(path)) if path.exists() and len(sys.argv) > 1 else {}
    for nm in (sys.argv[1:] or CASES):
        out[nm + '_x'], out[nm] = reference(nm)
    np.savez_compressed(path, **out)
    print(f'wrote {os.path.relpath(path)}')
