"""orca2torchspin against EasySpin orca2easyspin (tests/data/ref_orca2easyspin.mat from
tests/data/generate_orca_refs.m) on the ORCA files shipped with torchspin (tests/orca)
and with EasySpin (EasySpin/tests/orca, when that checkout is present)."""
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from torchspin.orca2torchspin import orca2torchspin

HERE = Path(__file__).resolve().parents[2]
REF = HERE / 'tests' / 'data' / 'ref_orca2easyspin.mat'
DIRS = {1: HERE / 'tests' / 'orca', 2: HERE.parent / 'EasySpin' / 'tests' / 'orca'}


def _entries():
    if not REF.exists():
        return []
    m = loadmat(REF, squeeze_me=True, struct_as_record=False)
    return list(np.atleast_1d(m['entries']))


def _ids():
    return [str(e.file) for e in _entries()]


def _fields(o):
    return getattr(o, '_fieldnames', [])


def _tensor_mol(pv, frame):
    """Full tensor in the molecular frame from principal values and Euler angles
    (EasySpin: T_M = R_M2T' * diag(pv) * R_M2T with R_M2T = erot(frame))."""
    from torchspin.rotations import erot
    R = erot(np.asarray(frame, dtype=float).reshape(-1).tolist()).numpy()
    return R.T @ np.diag(np.asarray(pv, dtype=float).reshape(-1)) @ R


def _rows(v, n):
    v = np.asarray(v, dtype=float)
    return v.reshape(n, -1)


@pytest.mark.parametrize('entry', _entries(), ids=_ids())
def test_orca2torchspin_vs_easyspin(entry):
    d, name = str(entry.file).split('/', 1)
    path = DIRS[int(d)] / name
    if not path.exists():
        pytest.skip(f'{path} not available')
    out = orca2torchspin(path)
    structs = out if isinstance(out, list) else [out]
    assert len(structs) == int(entry.n), (name, len(structs), int(entry.n))
    tol = dict(rtol=1e-6, atol=1e-8)
    for s_idx, (sys, data) in enumerate(structs, start=1):
        pre = f's{s_idx}_'
        ref = {f[len(pre):]: getattr(entry, f) for f in _fields(entry) if f.startswith(pre)}
        if 'S' in ref:
            assert np.allclose(np.asarray(sys.S, dtype=float).reshape(-1), np.asarray(ref['S'], dtype=float).reshape(-1)), (name, 'S')
        # nuclei: EasySpin lists element symbols and may carry atoms without data
        # (zero rows, e.g. an O with no EFG block); torchspin keeps only atoms with
        # hyperfine/quadrupole data and uses the reference isotope.  Align by atom index.
        ref_idx = np.atleast_1d(np.asarray(ref.get('NucsIdx', []), dtype=int)).tolist() if 'NucsIdx' in ref else None
        py_idx = [int(i) for i in np.atleast_1d(getattr(data, 'nucs_idx', []))]
        if 'Nucs' in ref and ('A' in ref or 'Q' in ref) and ref_idx:
            ref_nucs = [n.strip() for n in str(ref['Nucs']).split(',') if n.strip()]
            py_nucs = [''.join(ch for ch in n if ch.isalpha()) for n in sys.Nucs]
            assert set(py_idx) <= set(ref_idx), (name, py_idx, ref_idx)
            for i_py, idx in enumerate(py_idx):
                assert py_nucs[i_py] == ref_nucs[ref_idx.index(idx)], (name, py_nucs, ref_nucs)
        if 'xyz' in ref:
            xyz = np.asarray(getattr(data, 'xyz'), dtype=float)
            r = np.asarray(ref['xyz'], dtype=float)
            if xyz.shape == r.T.shape and xyz.shape != r.shape:
                xyz = xyz.T
            assert np.allclose(xyz, r, **tol), (name, 'xyz')
        # tensors compared in the molecular frame (invariant to axis permutation / frame choice)
        for key, frame_key in (('g', 'gFrame'), ('D', 'DFrame')):
            if key in ref:
                val = getattr(sys, key).detach().cpu().numpy()
                r_pv = np.asarray(ref[key], dtype=float); r_fr = np.asarray(ref.get(frame_key, [0, 0, 0]), dtype=float)
                v_fr = getattr(sys, frame_key).detach().cpu().numpy()
                assert np.allclose(_tensor_mol(val, v_fr), _tensor_mol(r_pv, r_fr), **tol), (name, key)
        n_n = len(sys.Nucs)
        for key, frame_key in (('A', 'AFrame'), ('Q', 'QFrame')):
            if key in ref and n_n > 0 and ref_idx:
                n_ref = len(ref_idx)
                val = _rows(getattr(sys, key).detach().cpu().numpy(), n_n)
                v_fr = _rows(getattr(sys, frame_key).detach().cpu().numpy(), n_n)
                r_pv = _rows(ref[key], n_ref); r_fr = _rows(ref.get(frame_key, np.zeros((n_ref, 3))), n_ref)
                for i_py, idx in enumerate(py_idx):
                    i_ref = ref_idx.index(idx)
                    assert np.allclose(_tensor_mol(val[i_py], v_fr[i_py]), _tensor_mol(r_pv[i_ref], r_fr[i_ref]), **tol), (name, key, idx)
