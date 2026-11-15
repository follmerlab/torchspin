"""Regression tests for B80: errors that used to be swallowed silently.

spidyan (pulse overlap, non-finite Hamiltonian / relaxation / eqState),
ham_ee / ham_nn (invalid spin indices), esfit (model, callback, progress and
stop_when errors), eprload (JEOL loader errors), mdhmm (singular covariance)
and mdload (truncated DCD / TRR files). Each case raised nothing, or raised a
misleading error, in 0.9.29.
"""
import os
import pickle
import shutil
import threading
import warnings

import numpy as np
import pytest
import scipy.linalg as la

from torchspin import SpinSystem
import torchspin.spidyan  # noqa: F401  (module object, not the function)
import sys as _sys
from torchspin.spidyan import spidyan, SpidyanOptions, _liouvillian
from torchspin.ham_ee import ham_ee
from torchspin.ham_nn import ham_nn
from torchspin.esfit import esfit, FitOptions, _FailedEval
from torchspin.eprload import eprload
from torchspin.mdhmm import _gaussian_prob_circular
from torchspin.mdload import _read_dcd, _read_trr

sp = _sys.modules['torchspin.spidyan']

MDFILES = os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'mdfiles')
TRR = os.path.join(MDFILES, '1ubq_A28R1_noSol.trr')


# ---------------------------------------------------------------------------
# spidyan
# ---------------------------------------------------------------------------

def _s12():
    return SpinSystem(S=[0.5], g=[[2.0] * 3])


def test_spidyan_pulse_overlap_raises():
    # second pulse starts 10 ns before the first (16 ns) one ends
    exp = {'Field': 340.0, 'mwFreq': 9.5,
           'Sequence': [{'tp': 0.016, 'Flip': np.pi / 2}, -0.010,
                        {'tp': 0.016, 'Flip': np.pi}, 0.2]}
    with pytest.raises(ValueError, match=r'[Pp]ulse overlap detected between events \[1, 3\] \(1-based'):
        spidyan(_s12(), exp)


def test_spidyan_negative_delay_without_overlap_still_runs():
    # a negative delay that moves the second pulse but does not overlap: valid
    exp = {'Field': 340.0, 'mwFreq': 9.5,
           'Sequence': [{'tp': 0.016, 'Flip': np.pi / 2}, 0.1,
                        {'tp': 0.016, 'Flip': np.pi}, -0.05, 0.2]}
    t, s, _ = spidyan(_s12(), exp)
    assert np.all(np.isfinite(s)) and np.size(s) > 0


def test_spidyan_nan_eqstate_raises():
    sys_ = _s12()
    sys_.T1 = 1.0
    sys_.T2 = 0.5
    sys_.eqState = np.array([[np.nan, 0], [0, 0.5]])
    exp = {'Field': 340.0, 'mwFreq': 9.5,
           'Sequence': [{'tp': 0.016, 'Flip': np.pi / 2}, 0.5]}
    with pytest.raises(ValueError, match='eqState'):
        spidyan(sys_, exp, SpidyanOptions(Relaxation=True))


def test_spidyan_overflowing_relaxation_rate_raises():
    sys_ = _s12()
    sys_.T1 = 1.0
    sys_.T2 = 1e-320          # 1/T2 overflows to inf
    exp = {'Field': 340.0, 'mwFreq': 9.5,
           'Sequence': [{'tp': 0.016, 'Flip': np.pi / 2}, 0.5]}
    with np.errstate(all='ignore'):
        with pytest.raises(ValueError, match='T1 and Sys.T2'):
            spidyan(sys_, exp, SpidyanOptions(Relaxation=True))


def _gamma(T1, T2):
    with np.errstate(all='ignore'):
        return sp._relaxation_superoperator([0.5], np.array(T1), np.array(T2))


RHO_EQ = np.array([[-0.5, 0], [0, 0.5]], complex).reshape(-1)


@pytest.mark.parametrize('ham, gamma, eq, what', [
    (np.diag([np.nan, 0.0]).astype(complex), _gamma(1.0, 0.5), RHO_EQ, 'Hamiltonian'),
    (np.zeros((2, 2), complex), _gamma(1.0, 1e-320), RHO_EQ, 'relaxation'),
    (np.zeros((2, 2), complex), _gamma(1.0, 0.5),
     np.array([np.inf, 0, 0, 0.5], complex), 'equilibrium'),
])
def test_liouvillian_nonfinite_input_raises(ham, gamma, eq, what):
    with pytest.raises(ValueError, match=what):
        _liouvillian(ham, gamma, eq, 1e-3)


def test_liouvillian_singular_pinv_steady_state(monkeypatch):
    """Guard: T2 = inf with degenerate levels makes -L exactly singular; the
    pseudo-inverse fallback must still give sigma_ss = rho_eq (the null-space
    components of the right-hand side vanish)."""
    calls = {'solve_failed': 0}
    orig = la.solve

    def spy(*a, **k):
        try:
            return orig(*a, **k)
        except la.LinAlgError:
            calls['solve_failed'] += 1
            raise
    monkeypatch.setattr(sp.la, 'solve', spy)
    H = np.zeros((2, 2), complex)
    G = _gamma(1.0, np.inf)
    L, ss = _liouvillian(H, G, RHO_EQ, 1e-3)
    assert calls['solve_failed'] == 1          # the pinv path was taken
    np.testing.assert_allclose(ss, RHO_EQ, rtol=0, atol=1e-14)
    L_full = -1j * (np.kron(H, np.eye(2)) - np.kron(np.eye(2), H.T)) - G
    resid = np.linalg.norm(-L_full @ ss - G @ RHO_EQ)
    assert resid < 1e-14
    assert np.all(np.isfinite(L))


def test_liouvillian_regular_steady_state_unchanged():
    H = np.diag([-1.0, 1.0]).astype(complex)
    G = _gamma(1.0, 0.5)
    _, ss = _liouvillian(H, G, RHO_EQ, 1e-3)
    L_full = -1j * (np.kron(H, np.eye(2)) - np.kron(np.eye(2), H.T)) - G
    np.testing.assert_array_equal(ss, la.solve(-L_full, G @ RHO_EQ))


# ---------------------------------------------------------------------------
# ham_ee / ham_nn
# ---------------------------------------------------------------------------

def _two_electrons():
    return SpinSystem(S=[0.5, 0.5], g=[[2.0] * 3, [2.0] * 3], ee=[100.0, 100.0, 100.0])


def _two_nuclei():
    return SpinSystem(S=0.5, g=[2.0] * 3, Nucs='1H,1H', A=[[1.0] * 3, [1.0] * 3],
                      nn=[[5.0, 5.0, 5.0]])


@pytest.mark.parametrize('idx, msg', [
    ([1, 5], 'between 1 and 2'),
    ([0, 2], 'between 1 and 2'),       # 0-based input
    ([1, 1], 'duplicate'),
    ([1, 2, 2], 'duplicate'),
    ([1, 1.5], 'integer'),
    ([1], 'at least 2'),
])
def test_ham_ee_invalid_indices_raise(idx, msg):
    with pytest.raises(ValueError, match=msg):
        ham_ee(_two_electrons(), idx)


@pytest.mark.parametrize('idx, msg', [
    ([1, 7], 'between 1 and 2'),
    ([0, 1], 'between 1 and 2'),
    ([2, 2], 'duplicate'),
    ([1], 'at least 2'),
])
def test_ham_nn_invalid_indices_raise(idx, msg):
    with pytest.raises(ValueError, match=msg):
        ham_nn(_two_nuclei(), idx)


def test_ham_ee_nn_valid_indices_unchanged():
    s = _two_electrons()
    assert np.array_equal(ham_ee(s, [2, 1]).numpy(), ham_ee(s).numpy())
    assert np.array_equal(ham_ee(s, np.array([1, 2])).numpy(), ham_ee(s).numpy())
    n = _two_nuclei()
    assert np.array_equal(ham_nn(n, [2, 1]).numpy(), ham_nn(n).numpy())
    assert float(np.abs(ham_nn(n).numpy()).max()) > 0


def test_ham_nn_single_nucleus_returns_zero_like_easyspin():
    # EasySpin ham_nn.m returns zeros before checking the indices when there are
    # fewer than 2 nuclei
    s = SpinSystem(S=0.5, g=[2.0] * 3, Nucs='1H', A=[1.0] * 3)
    H = ham_nn(s, [1, 5])
    assert not np.any(H.numpy())
    s0 = SpinSystem(S=0.5, g=[2.0] * 3, Nucs='1H,1H', A=[[1.0] * 3, [1.0] * 3],
                    nn=[[0.0, 0.0, 0.0]])
    assert not np.any(ham_nn(s0, [1, 5]).numpy())


# ---------------------------------------------------------------------------
# esfit
# ---------------------------------------------------------------------------

X = np.linspace(0, 6, 50)
Y = np.sin(X)


def _partly_failing_model(counter, lock):
    def model(p):
        if p[0] > 0.95:
            with lock:
                counter[0] += 1
            raise ValueError("simulator: parameter out of physical range")
        return np.sin(p[0] * X)
    return model


def _model_warnings(w):
    return [x for x in w if 'model raised' in str(x.message)]


def test_esfit_counts_and_warns_model_failures_serial():
    counter, lock = [0], threading.Lock()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        r = esfit(Y, _partly_failing_model(counter, lock), np.array([0.5]),
                  vary=np.array([0.6]),
                  options=FitOptions(method='simplex', max_iter=200,
                                     compute_uncertainties=False))
    assert counter[0] > 0
    assert r.n_failed_evals == counter[0]
    mw = _model_warnings(w)
    assert len(mw) == 1                                 # once per fit
    assert issubclass(mw[0].category, RuntimeWarning)
    assert 'parameter out of physical range' in str(mw[0].message)


def test_esfit_counts_model_failures_thread_pool():
    counter, lock = [0], threading.Lock()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        r = esfit(Y, _partly_failing_model(counter, lock), np.array([0.5]),
                  vary=np.array([0.6]),
                  options=FitOptions(method='grid', grid_points=13, n_workers=2,
                                     workers='threads', compute_uncertainties=False))
    assert counter[0] > 0
    assert r.n_failed_evals == counter[0]
    assert len(_model_warnings(w)) == 1


def test_esfit_no_failures_reports_zero():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        r = esfit(Y, lambda p: np.sin(p[0] * X), np.array([0.5]), vary=np.array([0.4]),
                  options=FitOptions(method='simplex', max_iter=100,
                                     compute_uncertainties=False))
    assert r.n_failed_evals == 0
    assert not _model_warnings(w)


def test_failed_eval_marker_pickles():
    f = pickle.loads(pickle.dumps(_FailedEval(1e10, 'ValueError: x')))
    assert isinstance(f, _FailedEval) and float(f) == 1e10 and f.error == 'ValueError: x'


def test_esfit_raising_callback_warns_once():
    def cb(info):
        raise TypeError("typo in my callback")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        r = esfit(Y, lambda p: p[0] * Y, np.array([0.5]), vary=np.array([0.6]),
                  options=FitOptions(method='swarm', max_iter=5, swarm_size=6, seed=1,
                                     callback=cb, compute_uncertainties=False))
    cw = [x for x in w if 'callback raised' in str(x.message)]
    assert len(cw) == 1 and 'typo in my callback' in str(cw[0].message)
    assert np.all(np.isfinite(r.pfit))


def test_esfit_raising_progress_warns_once():
    def prog(info):
        raise KeyError("bad key")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        r = esfit(Y, lambda p: p[0] * Y, np.array([0.5]), vary=np.array([0.6]),
                  options=FitOptions(method='simplex', max_iter=50, progress=prog,
                                     progress_every=1, compute_uncertainties=False))
    pw = [x for x in w if 'progress raised' in str(x.message)]
    assert len(pw) == 1 and 'bad key' in str(pw[0].message)
    assert not r.interrupted


def test_esfit_raising_stop_when_propagates():
    def stop(info):
        raise TypeError("typo in my stop condition")
    with pytest.raises(TypeError, match='typo in my stop condition'):
        esfit(Y, lambda p: p[0] * Y, np.array([0.5]), vary=np.array([0.6]),
              options=FitOptions(method='simplex', max_iter=20, stop_when=stop,
                                 compute_uncertainties=False))


# ---------------------------------------------------------------------------
# eprload
# ---------------------------------------------------------------------------

def test_eprload_truncated_jeol_reports_jeol_error(tmp_path):
    p = tmp_path / 'trunc_jeol.xyz'
    p.write_bytes(b'spin' + b'\x00' * 12 + b'name' + b'\x00' * 20)   # JEOL magic, truncated
    with pytest.raises(ValueError, match='JEOL file') as ei:
        eprload(str(p))
    assert ei.value.__cause__ is not None


def test_eprload_unknown_extension_still_unsupported(tmp_path):
    p = tmp_path / 'notes.xyz'
    p.write_bytes(b'hello world, not a spectrum')
    with pytest.raises(ValueError, match='Unsupported file extension'):
        eprload(str(p))


# ---------------------------------------------------------------------------
# mdhmm
# ---------------------------------------------------------------------------

def test_mdhmm_singular_covariance_warns_and_regularizes():
    data = np.array([[0.1, 0.2, 0.3], [0.5, 0.5, 0.5]])
    mu = np.array([0.2, 0.5])
    sig = np.array([[0.01, 0.0], [0.0, 0.0]])    # frozen second dihedral
    with pytest.warns(RuntimeWarning, match='singular state covariance'):
        pr = _gaussian_prob_circular(data, mu, sig)
    sig_reg = sig + 1e-6 * np.eye(2)
    d = (data - mu[:, None] + np.pi) % (2 * np.pi) - np.pi
    ref = (np.exp(-0.5 * np.sum(d * (np.linalg.inv(sig_reg) @ d), axis=0))
           / (2 * np.pi * np.sqrt(np.linalg.det(sig_reg))))
    np.testing.assert_allclose(pr, ref, rtol=1e-12)


# ---------------------------------------------------------------------------
# mdload
# ---------------------------------------------------------------------------

def _truncated_copy(src, dst, n_cut):
    shutil.copyfile(src, dst)
    size = os.path.getsize(dst)
    with open(dst, 'r+b') as f:
        f.truncate(size - n_cut)
    return dst


def _write_dcd(path, xyz):
    """Minimal DCD (no CHARMM unit-cell block): xyz shape (nFrames, 3, nAtoms)."""
    import struct
    n_frames, _, n_atoms = xyz.shape

    def rec(payload):
        return struct.pack('<i', len(payload)) + payload + struct.pack('<i', len(payload))
    hdr = bytearray(84)
    hdr[0:4] = b'CORD'
    struct.pack_into('<iii', hdr, 4, n_frames, 0, 1)
    struct.pack_into('<f', hdr, 40, 1.0)
    out = rec(bytes(hdr)) + rec(struct.pack('<i', 1) + b' ' * 80) + rec(struct.pack('<i', n_atoms))
    for fr in xyz.astype('<f4'):
        for dim in range(3):
            out += rec(fr[dim].tobytes())
    with open(path, 'wb') as f:
        f.write(out)


def test_read_dcd_truncated_warns(tmp_path):
    xyz = np.random.default_rng(0).normal(size=(4, 3, 7)).astype(np.float32)
    full = str(tmp_path / 'full.dcd')
    _write_dcd(full, xyz)
    with warnings.catch_warnings():
        warnings.simplefilter('error')                # complete file: no warning
        xyz_r, _, n_full, _ = _read_dcd(full)
    assert n_full == 4
    np.testing.assert_array_equal(xyz_r, xyz)
    p = _truncated_copy(full, str(tmp_path / 't.dcd'), 10)
    with pytest.warns(RuntimeWarning, match='DCD.*truncated'):
        xyz_t, _, n_t, _ = _read_dcd(p)
    assert n_t == 3
    np.testing.assert_array_equal(xyz_t, xyz[:3])


@pytest.mark.skipif(not os.path.exists(TRR), reason='TRR file not available')
def test_read_trr_truncated_warns(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        xyz, _, n_full, _ = _read_trr(TRR)
    p = _truncated_copy(TRR, str(tmp_path / 't.trr'), 100)
    with pytest.warns(RuntimeWarning, match='TRR.*truncated'):
        xyz_t, _, n_t, _ = _read_trr(p)
    assert n_t == n_full - 1
    np.testing.assert_array_equal(xyz_t, xyz[:-1])
