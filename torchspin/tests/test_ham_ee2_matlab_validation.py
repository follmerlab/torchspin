"""Sys.ee2 (biquadratic exchange) and Sys.D_ ([D, E/D]) against EasySpin ham()
(tests/data/ref_ham_ee2.mat from tests/data/generate_ham_ee2_refs.m)."""
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from torchspin.spinsystem import SpinSystem
from torchspin.ham import ham

REF = Path(__file__).resolve().parents[2] / 'tests' / 'data' / 'ref_ham_ee2.mat'


def _get(o, name, default=None):
    return getattr(o, name) if name in getattr(o, '_fieldnames', []) else default


def _sys(S) -> SpinSystem:
    Sv = np.atleast_1d(np.asarray(S.S, dtype=float)).tolist()
    kw = {'S': Sv, 'g': [[float(v)] * 3 for v in np.atleast_1d(np.asarray(S.g, dtype=float))]}
    for name in ('ee', 'ee2', 'eeFrame', 'D_'):
        v = _get(S, name)
        if v is not None:
            v = np.asarray(v, dtype=float)
            if name == 'ee':
                v = v.reshape(-1) if v.size % 3 else v
                kw['ee'] = [float(x) for x in v.reshape(-1)] if v.size in (1, len(Sv) * (len(Sv) - 1) // 2) else v.reshape(-1, 3).tolist()
            elif name == 'D_':
                kw['D_'] = v.reshape(-1, 2).tolist()
            else:
                kw[name] = v.reshape(-1).tolist() if name == 'ee2' else v.tolist()
    return SpinSystem(**kw)


def _cases():
    if not REF.exists():
        return []
    m = loadmat(REF, squeeze_me=True, struct_as_record=False)
    return [(str(c.name), c) for c in np.atleast_1d(m['cases'])]


@pytest.mark.parametrize('name,c', _cases(), ids=[n for n, _ in _cases()])
def test_ham_ee2_D_vs_matlab(name, c):
    sys = _sys(c.Sys)
    H0, mux, muy, muz = ham(sys, B0=None)
    F = np.asarray(c.F, dtype=complex)
    scale = max(1.0, abs(F).max())
    assert np.allclose(H0.numpy(), F, atol=1e-8 * scale), name
    # torchspin's moment operators follow EasySpin's G convention (H = F + B·G)
    for mu, G in ((mux, c.Gx), (muy, c.Gy), (muz, c.Gz)):
        G = np.asarray(G, dtype=complex)
        assert np.allclose(mu.numpy(), G, atol=1e-10 * max(1.0, abs(G).max())), name
