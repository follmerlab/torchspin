"""chili: Lanczos stopping rule (B116) and non-convergence handling.

EasySpin's rule stops at the first check step whose Lentz change is below Opt.Threshold. In
the pseudo-norm recursion a near-breakdown makes that one-step change small while the
continued fraction is still far from converged, so the plain-basis spectrum came out wrong
(nitroxide, tcorr = 100 ns, default LLMK: cosine 0.965 against the direct solve). EasySpin's
own default (fast) output has the same early stop at 256 points (tests/data/
generate_chili_mpsymm_refs.m, part c).
"""
from pathlib import Path
import warnings

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.io import loadmat

import torchspin.chili_sle as sle
from torchspin import SpinSystem, Experiment, chili
from torchspin.chili import ChiliOptions

DATA = Path(__file__).resolve().parents[2] / 'tests' / 'data'

NX_ES = dict(S=[0.5], g=[[2.0089, 2.0058, 2.0021]], Nucs=['14N'], A=[[16.0, 16.0, 100.0]], lw=[0.0, 0.05], tcorr=1e-7)
EXP_ES = dict(mwFreq=9.5, Range=[328, 352], nPoints=256, Harmonic=1)
RELMAX_SMALL = 2e-4      # measured 9.8e-5 (verified stop accepts a spectrum that moves < 1e-4)


def _cos(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _relmax(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def _sim(skw, ekw, **o):
    _, y = chili(SpinSystem(**skw), Experiment(**ekw), ChiliOptions(Verbosity=0, **o))
    return y.numpy()


def test_plain_basis_lanczos_matches_direct_small_basis():
    """B116 on a small basis (LLMK [8 0 2 2]): 0.9.32 stopped at step 60, relmax 4.8e-2."""
    ekw = dict(EXP_ES, nPoints=128)
    lanczos = _sim(NX_ES, ekw, LLMK=[8, 0, 2, 2], MpSymm=False)
    direct = _sim(NX_ES, ekw, LLMK=[8, 0, 2, 2], MpSymm=False, Solver='\\')
    assert _relmax(lanczos, direct) < RELMAX_SMALL, f'relmax {_relmax(lanczos, direct):.2e}'


@pytest.mark.skipif(not (DATA / 'ref_chili_mpsymm_a.mat').exists(), reason='reference file missing')
def test_plain_basis_lanczos_matches_easyspin_general_direct():
    """B116, default LLMK (5616 functions): 0.9.32 gave cosine 0.964772 against EasySpin's
    general method solved directly (and so does EasySpin's own general Lanczos)."""
    ref = loadmat(str(DATA / 'ref_chili_mpsymm_a.mat'), squeeze_me=True)
    y = _sim(NX_ES, EXP_ES, MpSymm=False)
    assert _cos(y, ref['yd']) > 0.99999, f"cosine {_cos(y, ref['yd']):.6f}"
    assert abs(np.abs(y).max() / np.abs(ref['yd']).max() - 1) < 1e-3


@pytest.mark.skipif(not (DATA / 'ref_chili_mpsymm_c.mat').exists(), reason='reference file missing')
def test_default_matches_converged_easyspin():
    """The default output equals EasySpin's general-direct and its fast builder run with a
    tight Lanczos threshold (1e-12); EasySpin's fast default (1e-6) stops early here."""
    a = loadmat(str(DATA / 'ref_chili_mpsymm_a.mat'), squeeze_me=True)
    c = loadmat(str(DATA / 'ref_chili_mpsymm_c.mat'), squeeze_me=True)
    y = _sim(NX_ES, EXP_ES)
    for name, ref in (('general direct', a['yd']), ('fast, Threshold 1e-12', c['yf12'])):
        assert _cos(y, ref) > 0.99999, f'{name}: cosine {_cos(y, ref):.6f}'
        assert abs(np.abs(y).max() / np.abs(ref).max() - 1) < 1e-3, name
    assert _cos(a['yf'], a['yd']) < 0.97          # documents EasySpin's early stop


def _complex_symmetric(n, seed=0):
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return sp.csr_matrix(M + M.T + 1j * n * np.eye(n))


def test_full_length_tridiagonalization_is_converged():
    """A full-length run (Krylov space = whole space) terminates the continued fraction."""
    n = 12
    A = _complex_symmetric(n)
    b = np.random.default_rng(1).normal(size=n)
    z = np.linspace(-3, 3, 7) * 1j
    s, converged, steps = sle.chili_lanczos(A, b, z, 1e-14)
    ref = np.array([b @ np.linalg.solve(A.toarray() + zz * np.eye(n), b) for zz in z]) / (b @ b)
    assert converged and steps == n
    assert _relmax(s, ref) < 1e-10
    _, converged, steps = sle.chili_lanczos(A, b, z, 1e-14, max_steps=n - 3)
    assert not converged and steps == n - 3


def test_nonconverged_lanczos_falls_back_to_direct_solver(monkeypatch):
    """EasySpin returns ones(nPoints) when the Lanczos does not converge; torchspin warns
    once and solves directly."""
    skw = dict(NX_ES, tcorr=1e-8)
    ekw = dict(EXP_ES, nPoints=64)
    direct = _sim(skw, ekw, LLMK=[6, 0, 2, 2], Solver='\\')
    calls = []

    def never_converges(A, b, z, threshold, max_steps=None):
        calls.append(1)
        return np.ones(z.shape, complex), False, 5

    monkeypatch.setattr(sle, 'chili_lanczos', never_converges)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        y = _sim(skw, ekw, LLMK=[6, 0, 2, 2])
    msgs = [str(x.message) for x in w if issubclass(x.category, RuntimeWarning) and 'Lanczos' in str(x.message)]
    assert calls and len(msgs) == 1, msgs
    assert 'using the direct solver' in msgs[0]
    np.testing.assert_array_equal(y, direct)


@pytest.mark.skipif(not (DATA / 'ref_chili_mpsymm_d.mat').exists(), reason='reference file missing')
def test_simple_tc1e7_matches_converged_easyspin_fast():
    """The loosened ext case simple_tc1e-07 (cosine 0.998, amplitude 1.066 against EasySpin's
    default output) is EasySpin's fast builder stopping early: run with Opt.Threshold=1e-10
    it moves by cosine 0.9979 / amplitude 1.069, and agrees with torchspin."""
    ref = loadmat(str(DATA / 'ref_chili_mpsymm_d.mat'), squeeze_me=True)
    skw = dict(S=[0.5], g=[[2.008, 2.0061, 2.0027]], Nucs=['14N'], A=[[16.0, 16.0, 86.0]], lw=[0.1, 0.0], tcorr=1e-7)
    x, y = chili(SpinSystem(**skw), Experiment(mwFreq=9.8, Range=[343, 358], nPoints=int(np.size(ref['x'])), Harmonic=1))
    np.testing.assert_allclose(x.numpy(), ref['x'], rtol=0, atol=1e-6)
    y = y.numpy()
    assert _cos(y, ref['yf10']) > 0.99999, f"cosine {_cos(y, ref['yf10']):.7f}"
    assert abs(np.abs(y).max() / np.abs(ref['yf10']).max() - 1) < 5e-3
