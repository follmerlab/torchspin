"""Zero-field lines exactly on bin edges with an interpolated powder grid (0.8.14).

S = 1, D = [-700, -1300, 2000] MHz has its lines at 0.6, 2.7 and 3.3 GHz, which coincide with points of
the 10 MHz axis 0:0.01:5 GHz, i.e. with bin edges (bin k covers [x_k, x_k + dx), as in EasySpin).  With
GridSize [19 4] or [10 8] the line positions are interpolated from the coarse to the fine grid.  At zero
field every orientation has the same position, and EasySpin puts each line into one bin.  torchspin does
the same because ``pepper._zf_const_pos`` gives every interpolated vertex the coarse position; without it
the interpolation round-off scatters the vertices by a few ulp around the edge and 5-39 % of each line
moves into the lower neighbor bin (1 - cosine 0.07-0.08 for the sticks, maximum difference 1.9e-2 even at
a 300 MHz Gaussian width).  These tests fail when ``_zf_const_pos`` is replaced by the identity.

Which of the two bins a line on an edge lands in is decided by the last bit of the eigenvalues, which can
differ between machines (B19).  The comparison therefore does not assume EasySpin's bin: it requires a
single-bin deposit in bin k or k - 1 with EasySpin's line area, and rebuilds the reference from EasySpin's
spectra of the single level pairs (Opt.Transitions), each moved down one bin where torchspin's stick
landed in bin k - 1.  Moving a Gaussian line by one bin equals broadening the moved stick to round-off
(checked with D moved by +-1e-9 MHz, which puts lines into bin k - 1: 6.9e-15), so one tight tolerance holds
on every machine.  Measured agreement with EasySpin here: max |difference| / max <= 6.9e-15.
"""
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from torchspin import SpinSystem
from torchspin.experiment import Experiment, Options
from torchspin.pepper import pepper

REF_FILE = Path(__file__).parent.parent.parent / "tests" / "data" / "ref_pepper_zerofield_edge.mat"
LINE_BINS = (60, 270, 330)          # 0.6, 2.7, 3.3 GHz on the 10 MHz axis; same order as ref['pairs']
TOL = 1e-12                         # measured <= 6.9e-15 (max-norm, relative to the maximum)


@pytest.fixture(scope="module")
def ref():
    return loadmat(REF_FILE, squeeze_me=True)


def _simulate(ref, grid, lw):
    kw = dict(S=1.0, g=[2.0, 2.0, 2.0], D=[float(d) for d in ref["D"]])
    if np.any(lw):
        kw["lw"] = [float(w) for w in lw]
    x, y = pepper(SpinSystem(**kw), Experiment(Field=0.0, mwRange=[0, 5], nPoints=501, Harmonic=0),
                  Options(ZeroField=True, GridSize=[int(g) for g in grid]))
    return x.numpy(), y.numpy()


def _down(y, s):
    """Move a spectrum s (0 or -1) bins, filling with zeros."""
    if s == 0:
        return y
    z = np.zeros_like(y)
    z[:s] = y[-s:]
    return z


def _line_shifts(ref, ig):
    """torchspin's bin of each stick relative to x_k (0 or -1); asserts a single-bin deposit with
    EasySpin's area and nothing outside the three lines."""
    x, y = _simulate(ref, ref["GridSize"][ig], ref["lw"][0])
    np.testing.assert_allclose(x, ref["nu"], rtol=0, atol=1e-12)
    sticks = ref["spc"][ig, 0]
    shifts = []
    rest = y.copy()
    for k in LINE_BINS:
        w, w_es = y[k - 3:k + 4], sticks[k - 3:k + 4]
        assert np.count_nonzero(w_es) == 1 and w_es[3] > 0           # EasySpin: one bin, at x_k
        assert w.sum() == pytest.approx(w_es.sum(), rel=TOL)          # line area
        j = int(np.argmax(w)) - 3
        assert j in (0, -1), f"line at bin {k} landed {j} bins away"
        assert w.max() == pytest.approx(w.sum(), rel=TOL), \
            f"line at bin {k} split over bins {np.nonzero(w)[0] - 3}: {w[w != 0] / w.sum()}"
        shifts.append(j)
        rest[k - 3:k + 4] = 0
    assert np.abs(rest).max() <= TOL * np.abs(y).max()
    return shifts


@pytest.mark.parametrize("ig", [0, 1], ids=["grid19x4", "grid10x8"])
def test_zero_field_lines_on_bin_edges_single_bin(ref, ig):
    """Sticks: each line lands in one bin with EasySpin's area (bin k or, by round-off, k - 1)."""
    _line_shifts(ref, ig)


@pytest.mark.parametrize("il", [0, 1, 2], ids=["sticks", "gauss30", "gauss300"])
@pytest.mark.parametrize("ig", [0, 1], ids=["grid19x4", "grid10x8"])
def test_zero_field_lines_on_bin_edges_match_easyspin(ref, ig, il):
    """Spectrum vs EasySpin, rebuilt from its single-pair spectra at torchspin's bins."""
    shifts = _line_shifts(ref, ig)
    _, y = _simulate(ref, ref["GridSize"][ig], ref["lw"][il])
    expected = sum(_down(ref["spc_line"][ig, il, j], s) for j, s in enumerate(shifts))
    if not any(shifts):
        # EasySpin's own bins: the pair spectra add up to its full spectrum (measured 5.6e-17)
        np.testing.assert_allclose(expected, ref["spc"][ig, il], rtol=0, atol=1e-14 * ref["spc"][ig, il].max())
    err = np.abs(y - expected).max() / np.abs(expected).max()
    assert err < TOL, f"max |torchspin - EasySpin| / max = {err:.3e} (bins {shifts})"
