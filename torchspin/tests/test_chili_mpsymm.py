"""chili: M-pS-pI (Meirovitch) symmetry basis and its amplitude (B114, B115).

Without director tilt (and with a potential that has only M = 0 terms) the Liouvillian
conserves pS + pI - M, so the Sx starting vector splits into two decoupled halves
(pS + pI - M = +1 and -1). ``MpSymm`` keeps the +1 block and adds the -1 block's response
exactly as the conjugate mirror of the +1 block, I(z) = 1/2 [I+(z) + conj(I+(conj z))];
with ``highField`` (pSmin = +1) there is no -1 half. The pruned and the plain basis
therefore give the same spectrum.
"""
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import torchspin.chili_sle as sle
from torchspin import SpinSystem, Experiment, chili
from torchspin.chili import ChiliOptions

DATA = Path(__file__).resolve().parents[2] / 'tests' / 'data'

NX = dict(S=[0.5], g=[[2.008, 2.006, 2.003]], Nucs=['14N'], A=[[20.0, 20.0, 85.0]], lw=[0.0, 0.1], tcorr=1e-7)
EXP_NX = dict(mwFreq=9.5, Range=[330, 346], nPoints=64, Harmonic=0)


def _cos(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _sim(skw, ekw, **o):
    _, y = chili(SpinSystem(**skw), Experiment(**ekw), ChiliOptions(Verbosity=0, **o))
    return y.numpy()


def _relmax(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


# label, SpinSystem kwargs, Experiment kwargs, ChiliOptions kwargs
IDENTITY_CASES = [
    ('rhombic, odd L, Mmax > Kmax', NX, EXP_NX, dict(LLMK=[6, 3, 4, 2])),
    ('axial, Mmax = 0', NX, EXP_NX, dict(LLMK=[8, 0, 0, 4])),
    ('axial, Kmax = 0', NX, EXP_NX, dict(LLMK=[8, 3, 2, 0])),
    ('evenK=False, jKmin=-1', NX, EXP_NX, dict(LLMK=[6, 3, 2, 2], evenK=False, jKmin=-1)),
    ('pImax=[0]', NX, EXP_NX, dict(LLMK=[6, 3, 2, 2], pImax=[0])),
    ('highField', NX, EXP_NX, dict(LLMK=[6, 3, 4, 2], highField=True)),
    ('two nuclei 15N + 1H',
     dict(S=[0.5], g=[[2.008, 2.006, 2.002]], Nucs=['15N', '1H'], A=[[20, 20, 85], [5, 10, 20]], tcorr=3e-8, lw=[0, 0.1]),
     dict(mwFreq=9.8, Range=[344, 354], nPoints=64, Harmonic=0), dict(LLMK=[4, 0, 2, 2])),
    ('S = 1, no nucleus',
     dict(S=[1.0], g=[[2.01, 2.005, 2.002]], D=[[100.0, 20.0]], tcorr=1e-8, lw=[0, 0.1]),
     dict(mwFreq=9.5, Range=[331, 346], nPoints=64, Harmonic=0), dict(LLMK=[6, 0, 2, 2])),
    ('M = 0 potential, GridSize 1', dict(NX, tcorr=1e-8, Potential=[[2, 0, 0, 1.0]]), EXP_NX,
     dict(LLMK=[6, 0, 2, 2], GridSize=1)),
]


@pytest.mark.parametrize('label,skw,ekw,okw', IDENTITY_CASES, ids=[c[0] for c in IDENTITY_CASES])
def test_mpsymm_equals_plain_basis(label, skw, ekw, okw):
    """Plain, pruned and automatic basis give the same spectrum (direct solver).

    On 0.9.32 an explicit MpSymm=True returned twice the plain spectrum (EasySpin's general
    method does the same) and the automatic choice returned half of it with highField (B114).
    """
    plain = _sim(skw, ekw, MpSymm=False, Solver='\\', **okw)
    pruned = _sim(skw, ekw, MpSymm=True, Solver='\\', **okw)
    auto = _sim(skw, ekw, Solver='\\', **okw)
    # measured <= 1.5e-13 (S = 1/2), 4.8e-11 (S = 1); 0.9.32: pruned = 2 x plain
    assert _relmax(pruned, plain) < 1e-10, f'{label}: MpSymm=True vs plain {_relmax(pruned, plain):.2e}'
    np.testing.assert_array_equal(auto, pruned)          # automatic choice prunes here


def test_highfield_amplitude_matches_plain_basis_default_solver():
    """B114: the automatic choice with highField=True returned 0.5 x the plain result."""
    okw = dict(LLMK=[6, 3, 4, 2], highField=True)
    auto = _sim(NX, EXP_NX, **okw)
    plain = _sim(NX, EXP_NX, MpSymm=False, Solver='\\', **okw)
    assert _cos(auto, plain) > 0.99999
    assert abs(np.abs(auto).max() / np.abs(plain).max() - 1) < 1e-3


# Mirror term: the dropped pS + pI - M = -1 block is the conjugate mirror of the kept one,
# I(z) = 1/2 [I+(z) + conj(I+(conj z))].  Without it (1/2 I+ only) the error grows with the
# non-secular couplings: S = 1, D = 500 MHz at X band 1.0e-4 (measured).
# label, SpinSystem kwargs, Experiment kwargs, LLMK, bound.  Measured (direct / eigen solver):
# nitroxide 5e-16 / 2.9e-12, 63Cu 5.9e-16 / 2.7e-13, S = 1 D 100 MHz 4.8e-11, D 500 MHz 2.1e-8.
MIRROR_CASES = [
    ('nitroxide', NX, EXP_NX, [6, 3, 4, 2], 1e-11),
    ('63Cu, A_z 600 MHz', dict(S=[0.5], g=[[2.05, 2.05, 2.25]], Nucs=['63Cu'], A=[[60, 60, 600]], tcorr=1e-8, lw=[0, 0.5]),
     dict(mwFreq=9.5, Range=[270, 350], nPoints=64, Harmonic=0), [6, 0, 2, 2], 1e-11),
    ('S = 1, D 100 MHz', dict(S=[1.0], g=[[2.01, 2.005, 2.002]], D=[[100.0, 20.0]], tcorr=1e-8, lw=[0, 0.1]),
     dict(mwFreq=9.5, Range=[331, 346], nPoints=64, Harmonic=0), [6, 0, 2, 2], 2e-10),
    ('S = 1, D 500 MHz', dict(S=[1.0], g=[[2.01, 2.005, 2.002]], D=[[500.0, 0.0]], tcorr=1e-8, lw=[0, 0.1]),
     dict(mwFreq=9.5, Range=[300, 380], nPoints=64, Harmonic=0), [6, 0, 2, 2], 1e-7),
]


@pytest.mark.parametrize('label,skw,ekw,llmk,bound', MIRROR_CASES, ids=[c[0] for c in MIRROR_CASES])
def test_mirror_term_equals_plain_basis(label, skw, ekw, llmk, bound):
    plain = _sim(skw, ekw, LLMK=llmk, MpSymm=False, Solver='\\')
    for solver in ('\\', 'E'):
        pruned = _sim(skw, ekw, LLMK=llmk, MpSymm=True, Solver=solver)
        assert _relmax(pruned, plain) < bound, f'{label}, Solver {solver}: {_relmax(pruned, plain):.2e}'


def test_mirror_term_is_needed(monkeypatch):
    """Dropping the mirror half (EasySpin's general MpSymm, rescaled) misses 1e-4 for S = 1, D 500 MHz."""
    label, skw, ekw, llmk, bound = MIRROR_CASES[3]
    plain = _sim(skw, ekw, LLMK=llmk, MpSymm=False, Solver='\\')
    orig = sle.chili_lanczos

    def no_mirror(A, b, z, threshold, max_steps=None):
        s, conv, k = orig(A, b, z, threshold, max_steps)
        n = z.size // 2
        s[n:] = 0.0                                      # drop conj(I+(conj z))
        return s, conv, k

    monkeypatch.setattr(sle, 'chili_lanczos', no_mirror)
    half = _sim(skw, ekw, LLMK=llmk, MpSymm=True, Threshold=1e-12)
    assert _relmax(half, plain) > 50 * bound


TILT_POT = dict(NX, tcorr=1e-8, Potential=[[2, 0, 0, 1.0]])


def test_mpsymm_true_with_director_tilt_raises():
    """B115: under director tilt the symmetry does not hold (0.9.32: cosine 0.67, 6.9 x amplitude)."""
    with pytest.raises(ValueError, match='M-pS-pI symmetry does not hold.*director is tilted'):
        _sim(TILT_POT, EXP_NX, LLMK=[4, 0, 2, 2], GridSize=3, MpSymm=True)
    with pytest.raises(ValueError, match='director is tilted'):
        _sim(NX, dict(EXP_NX, SampleFrame=[0, 0.4, 0]), LLMK=[4, 0, 2, 2], MpSymm=True)


def test_mpsymm_true_with_m_nonzero_potential_raises():
    with pytest.raises(ValueError, match='M != 0'):
        _sim(dict(NX, tcorr=1e-8, Potential=[[2, 1, 0, 1.0]]), EXP_NX, LLMK=[4, 0, 2, 2], GridSize=1, MpSymm=True)


def test_automatic_basis_with_director_tilt_is_plain():
    okw = dict(LLMK=[4, 0, 2, 2], GridSize=3, Solver='\\')
    np.testing.assert_array_equal(_sim(TILT_POT, EXP_NX, **okw), _sim(TILT_POT, EXP_NX, MpSymm=False, **okw))


# ---- EasySpin references (tests/data/generate_chili_mpsymm_refs.m) -------------------------
NX_ES = dict(S=[0.5], g=[[2.0089, 2.0058, 2.0021]], Nucs=['14N'], A=[[16.0, 16.0, 100.0]], lw=[0.0, 0.05], tcorr=1e-7)
EXP_ES = dict(mwFreq=9.5, Range=[328, 352], nPoints=256, Harmonic=1)


@pytest.mark.skipif(not (DATA / 'ref_chili_mpsymm_b.mat').exists(), reason='reference file missing')
def test_highfield_matches_easyspin():
    """B114: EasySpin's fast and general (direct) highField outputs agree (ratio 1.00000);
    torchspin 0.9.32 returned half of them."""
    ref = loadmat(str(DATA / 'ref_chili_mpsymm_b.mat'), squeeze_me=True)
    y = _sim(NX_ES, EXP_ES, highField=True)
    for name in ('yf', 'yd'):
        assert _cos(y, ref[name]) > 0.99999, f'{name}: cosine {_cos(y, ref[name]):.6f}'
        assert abs(np.abs(y).max() / np.abs(ref[name]).max() - 1) < 1e-3
