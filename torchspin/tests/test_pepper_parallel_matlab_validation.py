"""MATLAB validation of parallel-mode (B1 ∥ B0) pepper spectra (PORT_SPEC_2 item 3.2)."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from scipy.io import loadmat
from torchspin.spinsystem import SpinSystem
from torchspin.experiment import Experiment, Options
from torchspin.pepper import pepper

REF_FILE = Path(__file__).parent.parent.parent / "tests" / "data" / "ref_pepper_parallel.mat"
DATA = REF_FILE.parent

# S52_parallel cuts transitions at the range ends.  Since 0.9.28 (B88) they are
# searched again over a padded window and only the slot that still misses knots
# (looping level pairs that do not resonate at some orientations at any
# field) are interpolated linearly, while EasySpin interpolates every slot
# linearly: cosine with EasySpin 0.99989 -> 0.99582, amplitude 1.0121.
# Against a converged reference (no interpolation, GridSize [181 1], 0-1500 mT
# cropped; tests/data/ref_missing_knots.npz) 1 - cosine / max|dev| on the
# interior: 0.9.27 5.07e-3 / 0.130 (EasySpin 5.08e-3 / 0.131), now 6.96e-4 / 0.036.
_S52_XFAIL = pytest.mark.xfail(strict=True, reason=(
    'B88 (0.9.28): range-cut transitions re-searched, per-slot linear interpolation; EasySpin '
    'interpolates every slot linearly. Measured cosine 0.99582, amplitude 1.0121; torchspin is '
    'closer to the converged reference (test_s52_parallel_vs_converged)'))


@pytest.fixture(scope="module")
def cases():
    if not REF_FILE.exists():
        pytest.skip(f"Reference data not found: {REF_FILE}")
    refs = loadmat(str(REF_FILE), squeeze_me=True, struct_as_record=False)
    return {str(c.name): c for c in np.atleast_1d(refs['cases'])}


def _get(o, name, default=None):
    return getattr(o, name) if name in getattr(o, '_fieldnames', []) else default


def _run(c):
    S = c.Sys
    Sv = float(_get(S, 'S', 0.5))
    g = np.asarray(S.g, dtype=float)
    kw = {'S': [Sv], 'g': g.tolist() if g.ndim else float(g), 'lw': [float(S.lw), 0.0]}
    D = _get(S, 'D')
    if D is not None:
        kw['D'] = [np.asarray(D, dtype=float).tolist()]
    if _get(S, 'Nucs') is not None:
        kw['Nucs'] = str(S.Nucs); kw['A'] = [np.asarray(S.A, dtype=float).tolist()]
    E = c.Exp
    exp = Experiment(mwFreq=float(E.mwFreq), Range=np.asarray(E.Range, dtype=float).tolist(), Harmonic=int(E.Harmonic),
                     nPoints=int(_get(E, 'nPoints', 1024)), Mode=str(E.mwMode))
    return pepper(SpinSystem(**kw), exp, Options(Verbosity=0))


@pytest.mark.parametrize("name", ['triplet_parallel', 'triplet_perpendicular',
                                  pytest.param('S52_parallel', marks=_S52_XFAIL), 'Cu_parallel'])
def test_parallel_mode_vs_matlab(cases, name):
    c = cases[name]
    x, y = _run(c)
    y = y.numpy(); ref = np.asarray(c.spc, dtype=float)
    cs = float(np.dot(y, ref) / np.linalg.norm(y) / np.linalg.norm(ref))
    ratio = np.abs(y).max() / np.abs(ref).max()
    # Cu_parallel is a purely forbidden-transition spectrum (S=1/2 + 63Cu); its
    # weak lines depend on transition pre-selection (EasySpin: global maximum
    # transition rate over orientations; torchspin: per-orientation threshold).
    # S52_parallel: some transitions have no resonance at some knots, so EasySpin
    # interpolates every position linearly (pepper.m:763-772).  Up to 0.9.3 pepper splined
    # the slots without NaNs (B31) and was cosine 0.99587, amplitude 1.0112; since 0.9.4
    # cosine 0.99989, amplitude 0.99984 (threshold raised from 0.995).
    cs_min = {'Cu_parallel': 0.98, 'S52_parallel': 0.9998}.get(name, 0.999)
    assert cs > cs_min, f"{name}: cosine {cs:.5f}"
    assert abs(ratio - 1) < 0.03, f"{name}: amplitude ratio {ratio:.4f}"


def test_s52_parallel_vs_converged(cases):
    """S52_parallel against a converged reference (bounds: measured x 1.05)."""
    spec = importlib.util.spec_from_file_location('_gen_missing_knots', DATA / 'generate_missing_knots_refs.py')
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    x, y = _run(cases['S52_parallel'])
    x = x.numpy(); y = y.numpy()
    r = np.load(DATA / 'ref_missing_knots.npz')
    assert np.allclose(x, r['s52_parallel_x'], rtol=0, atol=1e-9)
    c, d = gen.metrics(y, r['s52_parallel'], gen.interior(x, gen.s52_parallel()[0]))
    print(f'S52_parallel: 1 - cosine {c:.4e}, max dev {d:.4e}')
    assert c <= 7.31e-4 and d <= 3.78e-2, (c, d)
