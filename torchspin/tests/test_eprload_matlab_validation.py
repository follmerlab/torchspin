"""eprload against the EasySpin oracle for every file in tests/eprfiles
(tests/data/ref_eprload.mat from tests/data/generate_eprload_refs.m, PORT_SPEC_2 item 4.5).

For each file EasySpin could read, torchspin's eprload must return the same
abscissa (or abscissae, for 2D data) and the same data array (real or complex),
up to floating-point round-off.
"""
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from torchspin.eprload import eprload

ROOT = Path(__file__).resolve().parents[2] / 'tests'
REF = ROOT / 'data' / 'ref_eprload.mat'

# files torchspin cannot read yet -> xfail reason
XFAIL = {}


def _entries():
    if not REF.exists():
        return []
    m = loadmat(REF, squeeze_me=True, struct_as_record=False)
    return [e for e in np.atleast_1d(m['entries'])]


def _ids():
    return [str(e.file) for e in _entries()]


def _as_axes(x, xcell):
    if xcell:
        return [np.asarray(a, dtype=float).ravel() for a in np.atleast_1d(x)]
    return [np.asarray(x, dtype=float).ravel()]


@pytest.mark.parametrize('entry', _entries(), ids=_ids())
def test_eprload_vs_matlab(entry):
    name = str(entry.file)
    if name in XFAIL:
        pytest.xfail(XFAIL[name])
    x, y, pars = eprload(ROOT / 'eprfiles' / name)
    # data: one array, or one array per data value (EasySpin cell array)
    if isinstance(y, list):
        refs = [np.asarray(v) for v in np.atleast_1d(entry.y)]
        assert len(y) == len(refs), (name, len(y), len(refs))
        pairs = list(zip(y, refs))
    else:
        pairs = [(y, np.asarray(entry.y))]
    for y_k, r_k in pairs:
        y_k = np.squeeze(np.asarray(y_k)); r_k = np.squeeze(np.asarray(r_k))
        assert y_k.shape == r_k.shape, (name, y_k.shape, r_k.shape)
        scale = max(1.0, float(np.abs(r_k).max()))
        assert np.allclose(y_k, r_k, rtol=1e-6, atol=1e-9 * scale), name
    # abscissae: one per data dimension; EasySpin returns [] for some formats
    ref_axes = _as_axes(entry.x, bool(entry.xcell))
    axes = list(x) if isinstance(x, (list, tuple)) else [x]
    axes = [np.asarray(a, dtype=float).ravel() for a in axes]
    assert len(axes) == len(ref_axes) or all(r.size == 0 for r in ref_axes), (name, len(axes), len(ref_axes))
    for a, r in zip(axes, ref_axes):
        if r.size == 0:
            continue
        assert a.shape == r.shape, (name, a.shape, r.shape)
        assert np.allclose(a, r, rtol=1e-7, atol=1e-9 * max(1.0, float(np.abs(r).max()))), name
