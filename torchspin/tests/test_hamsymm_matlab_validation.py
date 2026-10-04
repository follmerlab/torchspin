"""MATLAB EasySpin cross-validation of hamsymm.

Reference data: ``tests/data/ref_hamsymm.mat``, produced by
``tests/data/generate_hamsymm_refs.m`` (EasySpin ``hamsymm``). For every case
the point group string must match exactly and the symmetry frame must match
EasySpin's ``RMatrix`` element-wise.

Frame convention: torchspin's ``hamsymm`` returns the mol→sym rotation
``R`` (rows = symmetry axes in molecular coordinates); EasySpin's
``RMatrix`` has the symmetry axes along its *columns*. Hence the check is
``R.T == RMatrix``. No sign/permutation freedom is granted: both
implementations enumerate candidate frames in the same order and keep the
first frame that reaches the highest symmetry, so the frames must coincide.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io import loadmat

from torchspin.constants import CLIGHT
from torchspin.hamsymm import hamsymm
from torchspin.rotations import erot
from torchspin.spinsystem import SpinSystem

DATA_DIR = Path(__file__).parent.parent.parent / "tests" / "data"
REF_FILE = DATA_DIR / "ref_hamsymm.mat"


@pytest.fixture(scope="module")
def refs():
    if not REF_FILE.exists():
        pytest.skip(f"Reference data not found: {REF_FILE} (run generate_hamsymm_refs.m)")
    return loadmat(str(REF_FILE), squeeze_me=True, struct_as_record=False)["ref"]


# --- spin systems, identical to generate_hamsymm_refs.m ----------------------

_b20, _b22 = 100.0, 150.0
_B6 = 12.87
_Rg = erot([0.3, 0.7, 0.2]).numpy()
_gfull = (_Rg.T @ np.diag([2.0, 2.1, 2.2]) @ _Rg).tolist()
_RA = erot([0.0, 24 * math.pi / 180, 0.0]).numpy()
_Afull = _RA @ np.diag([1.0, 1.0, 2.0]) @ _RA.T
_J1, _J2, _dz1 = -100.0, -89.9, 4.85
_dz2 = -_dz1
_ee12 = np.array([[-_J1, _dz1, 0], [-_dz1, -_J1, 0], [0, 0, -_J1]])
_ee13 = np.array([[-_J2, _dz2, 0], [-_dz2, -_J2, 0], [0, 0, -_J2]])
_EE = (np.vstack([_ee12, _ee13, _ee12]) * 100 * CLIGHT / 1e6).tolist()

CASES = {
    # --- eigenvalue branch ---
    "cubic_B4": dict(S=[3.5], g=[[2, 2, 2]], B=[None, [[50, 0, 0, 0, 10, 0, 0, 0, 0]]]),
    "cubic_B6": dict(S=[3.5], g=[[2, 2, 2]],
                     B=[None, None, [[0, 0, -210, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0]]]),
    "cubic_B4B6": dict(S=[3.5], g=[[2, 2, 2]],
                       B=[None, [[50, 0, 0, 0, 10, 0, 0, 0, 0]],
                          [[0, 0, -21 * _B6, 0, 0, 0, _B6, 0, 0, 0, 0, 0, 0]]]),
    "stevens_B2_rhombic": dict(S=[1], g=[[2, 2, 2]], B=[[[_b22, 0, _b20, 0, 0]]]),
    "stevens_B40_axial": dict(S=[2], g=[[2, 2, 2]], B=[None, [[0, 0, 0, 0, 30, 0, 0, 0, 0]]]),
    "stevens_B40_B44": dict(S=[2], g=[[2, 2, 2]], B=[None, [[3, 0, 0, 0, 30, 0, 0, 0, 0]]]),
    "stevens_B40_B43": dict(S=[2], g=[[2, 2, 2]], B=[None, [[0, 4, 0, 0, 30, 0, 0, 0, 0]]]),
    "stevens_B60_B66": dict(S=[3.5], g=[[2, 2, 2]],
                            B=[None, None, [[2, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0]]]),
    "stevens_B2_tilted": dict(S=[1], g=[[2, 2, 2]], B=[[[_b22, 0, _b20, 0, 0]]],
                              BFrame=[[[0.3, 0.7, 0.2]]]),
    "stevens_B2_tilted_with_g": dict(S=[1], g=[[2.0, 2.1, 2.2]], gFrame=[[0.3, 0.7, 0.2]],
                                     B=[[[_b22, 0, _b20, 0, 0]]], BFrame=[[[0.3, 0.7, 0.2]]]),
    "fullg_axial_x": dict(S=[0.5], g=np.diag([2.3, 2.0, 2.0]).tolist()),
    "fullg_rhombic_tilted": dict(S=[0.5], g=_gfull),
    "fullA_axial_tilted": dict(S=[0.5], Nucs="1H", A=_Afull.tolist()),
    "fullA_axial_tilted_small": dict(S=[0.5], Nucs="1H", A=(_Afull * 1e-3).tolist()),
    "fullee_threespins": dict(S=[0.5, 0.5, 0.5], g=[2, 2, 2], ee=_EE),
    "ham132_axial": dict(S=[1.5], g=[[2, 2, 2]], Ham={"132": [[0, 0, 0.05, 0, 0]]}),
    # --- geometric branch ---
    "geom_rhombic_tilted": dict(S=[0.5], g=[[2.0, 2.1, 2.2]], gFrame=[[0.3, 0.7, 0.2]]),
    "geom_two_axial_ztilt": dict(S=[0.5], Nucs="1H", g=[[2, 2, 3]], A=[[3, 3, 1]],
                                 AFrame=[[0, 0.4, 0]]),
    "geom_rhombic_axial_plane": dict(S=[0.5], Nucs="1H", g=[[2, 2, 3]], A=[[3, 2, 1]],
                                     gFrame=[[0.5, math.pi / 2, 0]]),
    "geom_two_rhombic_one_axis": dict(S=[0.5], Nucs="1H", g=[[2, 2.5, 3]], A=[[3, 2, 1]],
                                      AFrame=[[0.6, 0, 0]]),
    "geom_axial_x": dict(S=[0.5], g=[[2.2, 2, 2]]),
    "geom_twospinA": dict(S=[0.5, 0.5], g=[2, 2], Nucs="1H", A=[[1, 1, 1, 2, 2, 3]], ee=[6]),
}

# Expected groups (documented for readability; the .mat is authoritative)
EXPECTED = {
    "cubic_B4": "Oh", "cubic_B6": "Oh", "cubic_B4B6": "Oh",
    "stevens_B2_rhombic": "D2h", "stevens_B40_axial": "Dinfh",
    "stevens_B40_B44": "D4h", "stevens_B40_B43": "D3d", "stevens_B60_B66": "D6h",
    "stevens_B2_tilted": "Ci", "stevens_B2_tilted_with_g": "D2h",
    "fullg_axial_x": "Dinfh", "fullg_rhombic_tilted": "Ci",
    "fullA_axial_tilted": "C2h", "fullA_axial_tilted_small": "C2h",
    "fullee_threespins": "Dinfh", "ham132_axial": "Dinfh",
    "geom_rhombic_tilted": "D2h", "geom_two_axial_ztilt": "C2h",
    "geom_rhombic_axial_plane": "C2h", "geom_two_rhombic_one_axis": "C2h",
    "geom_axial_x": "Dinfh", "geom_twospinA": "Dinfh",
}


@pytest.mark.parametrize("name", list(CASES.keys()))
def test_hamsymm_vs_matlab(refs, name):
    ref = getattr(refs, name)
    sys = SpinSystem(**CASES[name])
    pgroup, R = hamsymm(sys)

    assert pgroup == str(ref.PGroup), f"{name}: torchspin {pgroup} vs EasySpin {ref.PGroup}"
    assert pgroup == EXPECTED[name]

    RMatrix = np.asarray(ref.RMatrix, dtype=float)
    R_S2M = R.numpy().T
    assert np.allclose(R_S2M, RMatrix, atol=1e-10), (
        f"{name}: symmetry frame mismatch\n torchspin R.T=\n{R_S2M}\n EasySpin RMatrix=\n{RMatrix}"
    )


def test_reference_case_count(refs):
    """Every reference case in the .mat is exercised above."""
    names = set(refs._fieldnames)
    assert names == set(CASES.keys())
