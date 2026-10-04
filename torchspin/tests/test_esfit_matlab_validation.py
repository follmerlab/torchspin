"""esfit against EasySpin esfit (tests/data/ref_esfit.mat from tests/data/generate_esfit_refs.m):
deterministic Levenberg–Marquardt and Nelder–Mead fits of noisy synthetic pepper spectra
from offset starting values.  The models are built exactly like EasySpin's parameter
vectors (axial g = 2 parameters, then lwpp, then A), so both implementations must reach
the same minimum."""
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from torchspin.spinsystem import SpinSystem
from torchspin.experiment import Experiment
from torchspin.esfit import esfit, FitOptions
from torchspin.pepper import pepper

REF = Path(__file__).resolve().parents[2] / 'tests' / 'data' / 'ref_esfit.mat'


def _fields(o):
    return getattr(o, '_fieldnames', [])


def _get(o, name, default=None):
    return getattr(o, name) if name in _fields(o) else default


def _exp(E):
    kw = {}
    for name, conv in (('mwFreq', float), ('Field', float), ('nPoints', int)):
        v = _get(E, name)
        if v is not None:
            kw[name] = conv(v)
    for name in ('Range', 'mwRange'):
        v = _get(E, name)
        if v is not None:
            kw[name] = np.asarray(v, dtype=float).tolist()
    return Experiment(**kw)


def _model_and_params(S0, V):
    """EasySpin-style parameter vector: g (2 axial or 3), lwpp, A (in Vary order)."""
    g0 = np.atleast_1d(np.asarray(S0.g, dtype=float)); ng = g0.size
    p0, vary = list(g0), list(np.atleast_1d(np.asarray(V.g, dtype=float)))
    has_lw = _get(V, 'lwpp') is not None
    if has_lw:
        p0.append(float(S0.lwpp)); vary.append(float(V.lwpp))
    has_A = _get(V, 'A') is not None
    if has_A:
        p0 += list(np.atleast_1d(np.asarray(S0.A, dtype=float))); vary += list(np.atleast_1d(np.asarray(V.A, dtype=float)))
    lw = _get(S0, 'lw'); lwpp = _get(S0, 'lwpp'); nucs = _get(S0, 'Nucs'); A0 = _get(S0, 'A')
    exp_holder = {}

    def model(p):
        p = np.asarray(p, dtype=float)
        g = [p[0], p[0], p[1]] if ng == 2 else p[:3].tolist()
        k = ng
        kw = {'S': [0.5], 'g': g}
        if has_lw:
            kw['lwpp'] = [float(p[k]), 0.0]; k += 1
        elif lwpp is not None:
            kw['lwpp'] = [float(lwpp), 0.0]
        if lw is not None:
            kw['lw'] = [float(lw), 0.0]
        if nucs is not None:
            kw['Nucs'] = str(nucs)
            kw['A'] = [p[k:k + 3].tolist()] if has_A else [np.atleast_1d(np.asarray(A0, dtype=float)).tolist()]
        _, y = pepper(SpinSystem(**kw), exp_holder['exp'])
        return y.detach().cpu().numpy()
    return model, np.array(p0), np.array(vary), exp_holder


def _cases():
    if not REF.exists():
        return []
    m = loadmat(REF, squeeze_me=True, struct_as_record=False)
    return [(str(c.name), c) for c in np.atleast_1d(m['cases'])]


@pytest.mark.parametrize('name,c', _cases(), ids=[n for n, _ in _cases()])
def test_esfit_vs_matlab(name, c):
    model, p0, vary, holder = _model_and_params(c.Sys0, c.Vary)
    holder['exp'] = _exp(c.Exp)
    method, target = str(c.method).split()
    opts = FitOptions(method=method, target=target, verbosity=0, compute_uncertainties=False)
    data = np.asarray(c.spc, dtype=float)
    res = esfit(data, model, p0, vary=vary, options=opts)
    ref_p = np.asarray(c.pfit, dtype=float).reshape(-1)
    py = np.asarray(res.pfit, dtype=float)
    assert py.shape == ref_p.shape, (name, py, ref_p)
    # fit quality: torchspin must reach at least EasySpin's residual (within 5 %)
    ref_fit = np.asarray(c.fit, dtype=float)
    rms = lambda f: np.sqrt(np.mean((f - data) ** 2))
    py_rms, ref_rms = rms(np.asarray(res.fit, dtype=float)), rms(ref_fit)
    assert py_rms < 1.05 * ref_rms + 1e-12, (name, py_rms, ref_rms)
    if py_rms < 0.95 * ref_rms:
        # EasySpin's run stalled at a worse point (its levmar stops at the symmetric
        # starting values in some cases); the parameters cannot be compared
        return
    ng = np.atleast_1d(np.asarray(c.Sys0.g, dtype=float)).size
    g_ok = np.allclose(py[:ng], ref_p[:ng], atol=3e-4)
    if not g_ok and ng == 3 and _get(c.Sys0, 'Nucs') is None:
        # A powder spectrum without hyperfine coupling is invariant under a
        # permutation of the g principal values; when the start has gx0 == gy0
        # the two mirror minima are identical and which one a simplex reaches is
        # decided by rounding noise (EasySpin and torchspin can differ here).
        g0 = np.atleast_1d(np.asarray(c.Sys0.g, dtype=float))
        if g0[0] == g0[1]:
            g_ok = np.allclose(py[[1, 0, 2]], ref_p[:ng], atol=3e-4)
    assert g_ok, (name, py[:ng], ref_p[:ng])
    k = ng
    if _get(c.Vary, 'lwpp') is not None:
        assert abs(py[k] - ref_p[k]) < 0.03, (name, py[k], ref_p[k]); k += 1
    if _get(c.Vary, 'A') is not None:
        assert np.allclose(py[k:k + 3], ref_p[k:k + 3], atol=0.5), (name, py[k:k + 3], ref_p[k:k + 3])
