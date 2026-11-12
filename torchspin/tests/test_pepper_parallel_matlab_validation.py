"""MATLAB validation of parallel-mode (B1 ∥ B0) pepper spectra (PORT_SPEC_2 item 3.2)."""
from pathlib import Path
import numpy as np
import pytest
from scipy.io import loadmat
from torchspin.spinsystem import SpinSystem
from torchspin.experiment import Experiment, Options
from torchspin.pepper import pepper

REF_FILE = Path(__file__).parent.parent.parent / "tests" / "data" / "ref_pepper_parallel.mat"


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


@pytest.mark.parametrize("name", ['triplet_parallel', 'triplet_perpendicular', 'S52_parallel', 'Cu_parallel'])
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
