"""PORT_SPEC_3 Phase 1 pepper items against EasySpin (tests/data/ref_pepper_round3.mat,
generate_pepper_round3_refs.m): ordering in frequency sweeps, photoselection with
perturbation theory, Exp.mwMode excitation modes (unpolarized, circular, tilted linear;
powders and crystals), Opt.separate='transitions', automatic sweep ranges."""
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from torchspin.tests.test_pepper_matlab_validation_ext2 import (
    sys_from_mat, exp_from_mat, opt_from_mat, cosine, _get)
from torchspin.pepper import pepper

REF = Path(__file__).resolve().parents[2] / 'tests' / 'data' / 'ref_pepper_round3.mat'


def _load():
    m = loadmat(REF, squeeze_me=True, struct_as_record=False)
    return {str(c.name): c for c in np.atleast_1d(m['cases'])}


@pytest.fixture(scope='module')
def cases():
    if not REF.exists():
        pytest.skip(f'reference file missing: {REF}')
    return _load()


def _case_names():
    return list(_load().keys()) if REF.exists() else []


LOOSE = {
    # EasySpin's matrix path interpolates eigenvectors linearly across the field segment;
    # for 63Cu+14N its intensities differ from exact diagonalisation (and from its own
    # perturb2 spectrum, cosine 0.951). torchspin (exact) agrees with EasySpin perturb2 at 0.988.
    'septrans_CuN_matrix': (0.93, 0.15),
    'autorange_field': (0.999, 0.03),   # same system as ext2 'fieldrange_Q' (amplitude 0.976, documented)
}
XFAIL = {
    'septrans_CuN_matrix': "EasySpin's matrix path interpolates eigenvectors across the field segment; "
                           "for 63Cu+14N its per-orientation intensities are 3-4 % off the exact value "
                           "(MATLAB check: exact 0.5692 vs resfields 0.5880; torchspin 0.5692) and its "
                           "spectrum disagrees with its own perturb2 (cosine 0.951). torchspin matches "
                           "EasySpin perturb2 at 0.988.",
}


def _run(c):
    auto = _get(c.Exp, 'Range') is None and _get(c.Exp, 'CenterSweep') is None and _get(c.Exp, 'mwRange') is None
    exp = exp_from_mat(c.Exp, c.Sys, x_axis=None if auto else np.asarray(c.x, dtype=float))
    systems = [sys_from_mat(c.Sys)]
    if _get(c, 'Sys2') is not None:
        systems.append(sys_from_mat(c.Sys2))
    return pepper(systems if len(systems) > 1 else systems[0], exp, opt_from_mat(c.Opt))


@pytest.mark.parametrize('name', _case_names())
def test_pepper_round3_vs_matlab(cases, name):
    if name in XFAIL:
        pytest.xfail(XFAIL[name])
    c = cases[name]
    x, spc = _run(c)
    x = x.detach().cpu().numpy(); spc = spc.detach().cpu().numpy()
    ref = np.asarray(c.spc, dtype=float); ref_x = np.asarray(c.x, dtype=float)
    assert np.allclose(x, ref_x, rtol=1e-9, atol=1e-9 * abs(ref_x).max()), 'sweep axis differs'
    cos_thr, amp_tol = LOOSE.get(name, (0.999, 0.02))
    if name.startswith('septrans'):
        # separate transitions: EasySpin orders transitions by pre-selection rate
        # and keeps weak ones the torchspin slot bookkeeping may drop; match each
        # significant reference row to the closest torchspin row.  Perturbation
        # methods must give exactly the same number of rows (pepper_separate_transitions).
        assert spc.ndim == 2 and ref.ndim == 2 and spc.shape[1] == ref.shape[1], (spc.shape, ref.shape)
        if str(_get(c.Opt, 'Method', 'matrix')).startswith('perturb'):
            assert spc.shape[0] == ref.shape[0], (name, spc.shape, ref.shape)
        gmax = abs(ref).max()
        used = set()
        for k, b in enumerate(ref):
            if abs(b).max() < 1e-3 * gmax:
                continue
            best = max((j for j in range(spc.shape[0]) if j not in used), key=lambda j: cosine(spc[j], b))
            used.add(best)
            assert cosine(spc[best], b) > cos_thr, (name, k, cosine(spc[best], b))
            amp = abs(spc[best]).max() / abs(b).max()
            assert abs(amp - 1) < amp_tol, (name, k, amp)
        return
    assert spc.shape == ref.shape, (name, spc.shape, ref.shape)
    rows = [(spc, ref)] if ref.ndim == 1 else list(zip(spc, ref))
    for k, (a, b) in enumerate(rows):
        if abs(b).max() == 0:
            assert abs(a).max() < 1e-12 * max(1.0, abs(ref).max()), (name, k)
            continue
        assert cosine(a, b) > cos_thr, (name, k, cosine(a, b))
        amp = abs(a).max() / abs(b).max()
        assert abs(amp - 1) < amp_tol, (name, k, amp)
