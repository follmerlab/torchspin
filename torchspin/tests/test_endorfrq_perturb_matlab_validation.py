"""endorfrq_perturb against EasySpin's endorfrq_perturb.m, element by element
(B20, B27; tests/data/endorfrq_perturb_refs.mat,
generate_endorfrq_perturb_refs.m).

* B20 intensities: the matrix element |<mI+1|I+|mI>|^2/4 = k(2I+1-k)/4 for
  k = 1..2I (.m:367) and the factor prod(2I+1) over all nuclei (.m:385).
  Before 0.8.8 torchspin used the wrong level (the k = 1 line was 0) and
  dropped prod(2I+1).
* B20 order: EasySpin lists, per nucleus, the lower-level mI major and mS
  ascending within it (.m:303-304, 377-381); before 0.8.8 torchspin was
  mS-major with mS descending, so only sorted frequencies agreed.
* B27: a full 3x3 Q (.m:84-85) was read row-wise as principal values.

Eight systems at four orientations, 340 mT, PerturbOrder 2 and 1: 1H with
AFrame; 14N with Q and QFrame; 1H+14N; 55Mn with Q (I = 5/2); 1H at strong
coupling (A = 200 MHz); 14N with a full Q; 1H+14N with full (non-symmetric) A
and full Q; S = 3/2 with D and 14N (four mS manifolds).

Measured agreement after the fix (all 8 x 4 x 2 calls): frequencies within
7.4e-16 and intensities within 4.9e-16 relative (the same formulas evaluated
in the same order), so rtol = 1e-12 leaves three orders of magnitude of
headroom for platform rounding. The smallest reference frequency is 8.5 MHz,
so no absolute tolerance is needed.
"""
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from torchspin.spinsystem import SpinSystem
from torchspin.endorfrq_perturb import endorfrq_perturb

REF = Path(__file__).resolve().parents[2] / 'tests' / 'data' / 'endorfrq_perturb_refs.mat'

RTOL = 1e-12


def _load():
    m = loadmat(REF, squeeze_me=True, struct_as_record=False)
    return {str(c.name): c for c in np.atleast_1d(m['cases'])}


_CASES = _load() if REF.exists() else {}


def _sys_from_mat(S):
    kw = dict(S=[float(S.S)], g=[np.atleast_1d(S.g).astype(float).tolist()],
              Nucs=str(S.Nucs))
    A = np.atleast_2d(np.asarray(S.A, dtype=float))
    kw['A'] = A.tolist()
    for fld in ('AFrame', 'QFrame'):
        if hasattr(S, fld):
            kw[fld] = np.atleast_2d(np.asarray(getattr(S, fld), dtype=float)).tolist()
    if hasattr(S, 'Q'):
        kw['Q'] = np.atleast_2d(np.asarray(S.Q, dtype=float)).tolist()
    if hasattr(S, 'D'):
        kw['D'] = [np.atleast_1d(S.D).astype(float).tolist()]
    return SpinSystem(**kw)


@pytest.mark.skipif(not REF.exists(), reason=f'reference file missing: {REF}')
@pytest.mark.parametrize('order', [2, 1])
@pytest.mark.parametrize('name', sorted(_CASES) or ['none'])
def test_elementwise_vs_easyspin(name, order):
    c = _CASES[name]
    sys = _sys_from_mat(c.Sys)
    fref = np.atleast_2d(c.f if order == 2 else c.f1)
    iref = np.atleast_2d(c.i if order == 2 else c.i1)
    oris = np.atleast_2d(c.ori)
    for j, (phi, theta) in enumerate(oris):
        f, i = endorfrq_perturb(sys, float(c.Field), phi=float(phi), theta=float(theta),
                                order=order)
        np.testing.assert_allclose(f.numpy(), fref[:, j], rtol=RTOL, atol=0,
                                   err_msg=f'{name} ori {j} frequencies')
        np.testing.assert_allclose(i.numpy(), iref[:, j], rtol=RTOL, atol=0,
                                   err_msg=f'{name} ori {j} intensities')


@pytest.mark.skipif(not REF.exists(), reason=f'reference file missing: {REF}')
def test_nuclei_subset_keeps_full_state_count():
    """``nuclei=[1]`` returns the 14N block of the full call unchanged: the
    prod(2I+1) factor is over all nuclei of the system (.m:385), as in
    EasySpin's Opt.Nuclei."""
    sys = _sys_from_mat(_CASES['H1_N14'].Sys)
    f_all, i_all = endorfrq_perturb(sys, 340.0, phi=0.3, theta=0.7)
    f_N, i_N = endorfrq_perturb(sys, 340.0, phi=0.3, theta=0.7, nuclei=[1])
    np.testing.assert_array_equal(f_N.numpy(), f_all[2:].numpy())
    np.testing.assert_array_equal(i_N.numpy(), i_all[2:].numpy())
