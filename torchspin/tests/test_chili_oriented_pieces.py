"""B7: chili builds the orientation-dependent Liouville Hamiltonian from fixed pieces
(_OrientedLiouvillian) when there are many orientations.  The spectrum must equal the
per-orientation rbos + liouvhamiltonian route to roundoff, for powders with and without an
ordering potential, rank-1 terms, S = 1, and the explicit field sweep (B and G parts).

With the direct solver the two routes agree to roundoff (measured 7e-15; the Liouvillians
to 4e-16).  The Lanczos continued fraction amplifies last-digit differences of the
Liouvillian (a random 1-ulp perturbation moves its spectrum by 3e-8 to 2e-6, depending on
the case), and at the default Threshold 1e-6 a last-digit change can move the step at which
it stops: Cu + potential, LLMK [4 2 2 2], GridSize 4, 128 points: 3.6e-5 at Threshold 1e-6,
5e-9 at 1e-9, 3e-9 at 1e-12.  The Lanczos check therefore uses Threshold 1e-10."""
import numpy as np
import pytest

import torchspin.chili_sle as cs
from torchspin import ChiliOptions, Experiment, SpinSystem, chili

CASES = [
    ('nitroxide_potential', dict(S=[0.5], g=[2.0088, 2.0061, 2.0027], Nucs='14N', A=[[16, 16, 100]],
                                 tcorr=20e-9, Potential=[2, 0, 0, 1.5], lw=[0, 0.1]),
     dict(mwFreq=9.5, Range=[325, 355], nPoints=12), dict(LLMK=[4, 2, 2, 2], GridSize=4, Solver='\\')),
    ('cu_potential', dict(S=[0.5], g=[2.05, 2.05, 2.25], Nucs='63Cu', A=[[50, 50, 550]], tcorr=5e-9,
                          Potential=[2, 0, 0, 2.0], lw=[0, 0.5]),
     dict(mwFreq=9.5, Range=[260, 360], nPoints=12), dict(LLMK=[4, 2, 2, 2], GridSize=4, Solver='\\')),
    ('nitroxide_tilted_A', dict(S=[0.5], g=[2.0088, 2.0061, 2.0027], Nucs='14N', A=[[16, 16, 100]],
                                AFrame=[[0.3, 0.7, 0.2]], tcorr=10e-9, Potential=[2, 0, 0, 1.0], lw=[0, 0.1]),
     dict(mwFreq=9.5, Range=[325, 355], nPoints=12), dict(LLMK=[4, 2, 2, 2], GridSize=4, Solver='\\')),
    ('triplet_potential', dict(S=[1.0], g=[2.0], D=[300.0, 30.0], tcorr=5e-9, Potential=[2, 0, 0, 1.0],
                               lw=[0, 0.5]),
     dict(mwFreq=9.5, Range=[300, 380], nPoints=12), dict(LLMK=[4, 2, 2, 2], GridSize=4, Solver='\\')),
    ('nitroxide_explicit_sweep', dict(S=[0.5], g=[2.0088, 2.0061, 2.0027], Nucs='14N', A=[[16, 16, 100]],
                                      tcorr=20e-9, Potential=[2, 0, 0, 1.5], lw=[0, 0.1]),
     dict(mwFreq=9.5, Range=[325, 355], nPoints=48),
     dict(LLMK=[4, 2, 2, 2], GridSize=4, FieldSweepMethod='explicit', Solver='\\')),
]


@pytest.mark.parametrize('name, skw, ekw, okw', CASES, ids=[c[0] for c in CASES])
def test_pieces_equal_per_orientation_build(name, skw, ekw, okw, monkeypatch):
    run = lambda: np.asarray(chili(SpinSystem(**skw), Experiment(**ekw), ChiliOptions(Verbosity=0, **okw))[1])
    monkeypatch.setattr(cs, '_ORIENTED_PIECES_MIN', 10 ** 9)
    ref = run()
    monkeypatch.setattr(cs, '_ORIENTED_PIECES_MIN', 1)
    new = run()
    assert np.abs(ref).max() > 0
    assert np.abs(new - ref).max() <= 1e-9 * np.abs(ref).max()


def test_pieces_lanczos_within_its_roundoff_sensitivity(monkeypatch):
    name, skw, ekw, okw = CASES[1]
    okw = dict(okw, Threshold=1e-10); okw.pop('Solver')
    ekw = dict(ekw, nPoints=128)
    run = lambda: np.asarray(chili(SpinSystem(**skw), Experiment(**ekw), ChiliOptions(Verbosity=0, **okw))[1])
    monkeypatch.setattr(cs, '_ORIENTED_PIECES_MIN', 10 ** 9)
    ref = run()
    monkeypatch.setattr(cs, '_ORIENTED_PIECES_MIN', 1)
    assert np.abs(run() - ref).max() <= 1e-7 * np.abs(ref).max()
