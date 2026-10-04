"""The vectorised chili Liouvillian build must reproduce the loop port entry by entry."""
import time

import numpy as np
import torch
import pytest

import torchspin.chili_sle as sle
from torchspin import SpinSystem, Experiment, chili


def _check_equivalence(monkeypatch, sys, exp, expected_calls=1):
    stats = {'calls': 0, 'max_rel': 0.0, 't_loop': 0.0, 't_vec': 0.0}
    loop_fn, vec_fn = sle._liouvhamiltonian_loop, sle.liouvhamiltonian

    def checked(basis, Q0, Q1, Q2, jjj0, jjj1, jjj2):
        t = time.perf_counter(); ref = loop_fn(basis, Q0, Q1, Q2, jjj0, jjj1, jjj2); stats['t_loop'] += time.perf_counter() - t
        t = time.perf_counter(); new = vec_fn(basis, Q0, Q1, Q2, jjj0, jjj1, jjj2); stats['t_vec'] += time.perf_counter() - t
        d = abs(ref - new).max()
        scale = max(abs(ref).max(), 1e-300)
        stats['max_rel'] = max(stats['max_rel'], d / scale)
        stats['calls'] += 1
        assert d <= 1e-12 * scale, f"liouvhamiltonian mismatch: {d} (scale {scale})"
        assert ref.shape == new.shape
        return new

    monkeypatch.setattr(sle, 'liouvhamiltonian', checked)
    x, y = chili(sys, exp)
    assert stats['calls'] >= expected_calls
    assert bool(torch.isfinite(torch.as_tensor(y)).all())
    return stats


def test_liouv_vectorised_nitroxide_two_nuclei(monkeypatch):
    sys = SpinSystem(S=[0.5], g=[2.008, 2.006, 2.003], Nucs='14N,1H', A=[[20, 20, 85], [5, 5, 8]], tcorr=3e-9, lw=[0.1, 0.1])
    exp = Experiment(mwFreq=9.5, Range=[330, 350], nPoints=256)
    st = _check_equivalence(monkeypatch, sys, exp)
    assert st['max_rel'] <= 1e-12


def test_liouv_vectorised_powder_potential(monkeypatch):
    sys = SpinSystem(S=[0.5], g=[2.008, 2.006, 2.003], Nucs='14N', A=[[20, 20, 85]], tcorr=3e-9, lw=[0.1, 0.1],
                     Potential=[[2, 0, 0, 1.0]])
    exp = Experiment(mwFreq=9.5, Range=[330, 350], nPoints=256)
    st = _check_equivalence(monkeypatch, sys, exp, expected_calls=2)
    assert st['max_rel'] <= 1e-12


def test_liouv_vectorised_rank1_tilted_frames(monkeypatch):
    # tilted A frame → rank-1 ISTO components present (Q1 not None)
    sys = SpinSystem(S=[0.5], g=[2.008, 2.006, 2.003], Nucs='14N', A=[[20, 20, 85]], AFrame=[[0.3, 0.6, 0.2]],
                     tcorr=2e-9, lw=[0.1, 0.1])
    exp = Experiment(mwFreq=9.5, Range=[330, 350], nPoints=256)
    st = _check_equivalence(monkeypatch, sys, exp)
    assert st['max_rel'] <= 1e-12
