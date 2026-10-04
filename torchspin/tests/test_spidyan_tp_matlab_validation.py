"""spidyan: indirect dimension varying the pulse length tp (PORT_SPEC_2 item 4.2),
against tests/data/spidyan_onedimensional_tp.mat (generate_spidyan_tp_refs.m)."""
import os

import numpy as np
import pytest

from torchspin.tests.test_spidyan_matlab_validation import _load_data, _cosine, _zsys, DATA_DIR
from torchspin.spidyan import spidyan, SpidyanOptions


def test_spidyan_tp_sweep_cosine():
    if not os.path.exists(os.path.join(DATA_DIR, 'spidyan_onedimensional_tp.mat')):
        pytest.skip('reference missing')
    ref = _load_data('spidyan_onedimensional_tp.mat')
    p = {'Type': 'rectangular', 'tp': 0.1, 'Flip': np.pi}
    exp = {'Field': 1240.0, 'mwFreq': 33.5, 'Sequence': [p, 0.5, p], 'DetSequence': [1],
           'nPoints': 3, 'Dim1': ['p1.tp', 0.02], 'DetOperator': ['z1'], 'DetPhase': 0.0}
    t, sig, _ = spidyan(_zsys(), exp, SpidyanOptions(IntTimeStep=0.0001, SimFreq=32))
    ref_re = np.atleast_2d(np.asarray(ref['sig_re'], dtype=float))
    rows = [np.asarray(s).real.ravel() for s in sig] if isinstance(sig, (list, tuple)) \
        else [np.asarray(sig)[i].real.ravel() for i in range(np.asarray(sig).shape[0])]
    assert len(rows) == ref_re.shape[0]
    for i, row in enumerate(rows):
        r = ref_re[i][np.isfinite(ref_re[i])]
        n = min(len(row), len(r))
        assert n > 0 and _cosine(row[:n], r[:n]) > 0.99, f'step {i}: cosine {_cosine(row[:n], r[:n]):.4f}'
