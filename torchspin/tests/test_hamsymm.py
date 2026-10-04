"""Tests for hamsymm (Hamiltonian symmetry detection)."""
import pytest
import torch

from torchspin.hamsymm import hamsymm
from torchspin.spinsystem import SpinSystem


def test_hamsymm_isotropic():
    """Isotropic g → O3 symmetry."""
    sys = SpinSystem(S=1/2, g=2.0)
    pgroup, R = hamsymm(sys)
    assert pgroup == 'O3'
    assert R.shape == (3, 3)


def test_hamsymm_axial():
    """Axial g-tensor → Dinfh symmetry."""
    sys = SpinSystem(S=1/2, g=[2.0, 2.0, 2.2])
    pgroup, R = hamsymm(sys)
    assert pgroup == 'Dinfh'


def test_hamsymm_rhombic():
    """Rhombic g-tensor → D2h symmetry."""
    sys = SpinSystem(S=1/2, g=[2.0, 2.1, 2.2])
    pgroup, R = hamsymm(sys)
    assert pgroup == 'D2h'


def test_hamsymm_axial_tilted_frame():
    """Axial g with tilted frame still gives Dinfh."""
    sys = SpinSystem(S=1/2, g=[2.2, 2.0, 2.0], gFrame=[0.1, 0.2, 0.3])
    pgroup, R = hamsymm(sys)
    assert pgroup == 'Dinfh'
    # Frame orientation should be rotated


def test_hamsymm_two_axial_collinear():
    """Two collinear axial tensors → Dinfh."""
    sys = SpinSystem(S=1/2, g=[2.0, 2.0, 2.2], Nucs='1H', A=[10, 10, 30])
    pgroup, R = hamsymm(sys)
    assert pgroup == 'Dinfh'


def test_hamsymm_two_axial_perpendicular():
    """Two perpendicular axial tensors → D2h."""
    sys = SpinSystem(
        S=1/2,
        g=[2.0, 2.0, 2.2],
        Nucs='1H',
        A=[30, 10, 10],  # Axial along x
    )
    pgroup, R = hamsymm(sys)
    assert pgroup == 'D2h'


def test_hamsymm_two_axial_oblique():
    """Two oblique axial tensors → C2h."""
    sys = SpinSystem(
        S=1/2,
        g=[2.0, 2.0, 2.2],
        Nucs='1H',
        A=[10, 10, 30],
        AFrame=[0.0, 0.5, 0.0],  # Tilted 0.5 rad about y (beta angle)
    )
    pgroup, R = hamsymm(sys)
    assert pgroup == 'C2h'


def test_hamsymm_two_rhombic_collinear():
    """Two collinear rhombic tensors → D2h."""
    sys = SpinSystem(
        S=1/2,
        g=[2.0, 2.1, 2.2],
        Nucs='1H',
        A=[10, 20, 30],
    )
    pgroup, R = hamsymm(sys)
    assert pgroup == 'D2h'


def test_hamsymm_two_rhombic_one_axis_common():
    """Two rhombic tensors sharing one axis → C2h."""
    sys = SpinSystem(
        S=1/2,
        g=[2.0, 2.1, 2.2],
        Nucs='1H',
        A=[10, 20, 30],
        AFrame=[0.3, 0.0, 0.0],  # One axis coincides (z)
    )
    pgroup, R = hamsymm(sys)
    assert pgroup == 'C2h'


def test_hamsymm_two_rhombic_general():
    """Two rhombic tensors, general misalignment → Ci."""
    sys = SpinSystem(
        S=1/2,
        g=[2.0, 2.1, 2.2],
        Nucs='1H',
        A=[10, 20, 30],
        AFrame=[0.3, 0.2, 0.1],  # Arbitrary angles
    )
    pgroup, R = hamsymm(sys)
    assert pgroup == 'Ci'


def test_hamsymm_with_zfs_axial():
    """S=1 with axial ZFS → Dinfh."""
    sys = SpinSystem(S=1, g=2.0, D=[-100, -100, 200])
    pgroup, R = hamsymm(sys)
    assert pgroup == 'Dinfh'


def test_hamsymm_with_zfs_rhombic():
    """S=1 with rhombic ZFS → D2h."""
    sys = SpinSystem(S=1, g=2.0, D=[-150, -50, 200])
    pgroup, R = hamsymm(sys)
    assert pgroup == 'D2h'


def test_hamsymm_with_zfs_and_hyperfine_collinear():
    """S=1 with D and A collinear → Dinfh."""
    sys = SpinSystem(
        S=1,
        g=2.0,
        D=[-100, -100, 200],
        Nucs='14N',
        A=[20, 20, 80],
    )
    pgroup, R = hamsymm(sys)
    assert pgroup == 'Dinfh'


def test_hamsymm_with_zfs_and_hyperfine_perpendicular():
    """S=1 with D (axial z) and A (axial x) → D2h."""
    sys = SpinSystem(
        S=1,
        g=2.0,
        D=[-100, -100, 200],
        Nucs='14N',
        A=[80, 20, 20],  # Axial along x
    )
    pgroup, R = hamsymm(sys)
    assert pgroup == 'D2h'


def test_hamsymm_rotation_matrix_shape():
    """Rotation matrix has correct shape."""
    sys = SpinSystem(S=1/2, g=[2.0, 2.1, 2.2])
    pgroup, R = hamsymm(sys)
    assert R.shape == (3, 3)
    assert R.dtype == torch.float64


def test_hamsymm_rotation_matrix_orthonormal():
    """Rotation matrix is orthonormal."""
    sys = SpinSystem(S=1/2, g=[2.0, 2.1, 2.2])
    pgroup, R = hamsymm(sys)

    # Check orthonormality: R^T @ R = I
    eye = torch.matmul(R.T, R)
    assert torch.allclose(eye, torch.eye(3, dtype=torch.float64), atol=1e-10)

    # Check det(R) = 1 (proper rotation)
    det = torch.linalg.det(R)
    assert torch.allclose(det, torch.tensor(1.0, dtype=torch.float64), atol=1e-10)


def test_hamsymm_nitroxide():
    """Typical nitroxide radical (14N) → D2h or C2h depending on tilt."""
    # Collinear case
    sys = SpinSystem(
        S=1/2,
        g=[2.008, 2.006, 2.002],
        Nucs='14N',
        A=[15, 15, 90],
    )
    pgroup, R = hamsymm(sys)
    assert pgroup == 'D2h'

    # Tilted A-frame (beta rotation tilts z-axis)
    sys_tilted = SpinSystem(
        S=1/2,
        g=[2.008, 2.006, 2.002],
        Nucs='14N',
        A=[15, 15, 90],
        AFrame=[0.0, 0.2, 0.0],  # Beta rotation tilts the axial axis
    )
    pgroup_tilted, R_tilted = hamsymm(sys_tilted)
    assert pgroup_tilted == 'C2h'


def test_hamsymm_triplet_zfs():
    """Photoexcited triplet with ZFS → D2h."""
    sys = SpinSystem(S=1, g=2.0, D=[-100, -50, 150])
    pgroup, R = hamsymm(sys)
    assert pgroup == 'D2h'


def test_hamsymm_high_spin():
    """High-spin S=3/2 with axial ZFS → Dinfh."""
    sys = SpinSystem(S=3/2, g=2.0, D=[-100, -100, 200])
    pgroup, R = hamsymm(sys)
    assert pgroup == 'Dinfh'


def test_hamsymm_two_electrons_isotropic():
    """Two electrons, isotropic g → O3."""
    sys = SpinSystem(S=[1/2, 1/2], g=[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
    pgroup, R = hamsymm(sys)
    assert pgroup == 'O3'


def test_hamsymm_two_electrons_same_g():
    """Two electrons with identical rhombic g → D2h."""
    sys = SpinSystem(
        S=[1/2, 1/2],
        g=[[2.0, 2.1, 2.2], [2.0, 2.1, 2.2]],
    )
    pgroup, R = hamsymm(sys)
    assert pgroup == 'D2h'


def test_hamsymm_multiple_nuclei():
    """Multiple nuclei with same A-tensor → maintains symmetry."""
    sys = SpinSystem(
        S=1/2,
        g=[2.0, 2.0, 2.2],
        Nucs='1H,1H',
        A=[[10, 10, 30], [10, 10, 30]],
    )
    pgroup, R = hamsymm(sys)
    assert pgroup == 'Dinfh'


def test_hamsymm_ee_coupling():
    """Electron-electron coupling contributes to symmetry."""
    sys = SpinSystem(
        S=[1/2, 1/2],
        g=[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
        ee=[10, 10, 30],  # Axial dipolar coupling
    )
    pgroup, R = hamsymm(sys)
    assert pgroup == 'Dinfh'


def test_hamsymm_Q_tensor():
    """Nuclear quadrupole tensor (Q) contributes to symmetry."""
    sys = SpinSystem(
        S=1/2,
        g=2.0,
        Nucs='2H',  # Deuterium, I=1
        A=[10, 10, 10],
        Q=[-1, -1, 2],  # Axial
    )
    pgroup, R = hamsymm(sys)
    assert pgroup == 'Dinfh'


# ===========================================================================
# Ports of EasySpin's hamsymm_*.m test suite (tests/hamsymm_*.m).
#
# Each EasySpin test also re-runs with ``Sys.ZB11.l=0; Sys.ZB11.vals=1``
# (a legacy isotropic higher-order Zeeman term). Current EasySpin ignores the
# unknown ``ZB11`` field entirely, so that half of every test is a no-op and
# is not reproduced here.
# ===========================================================================

import math
import numpy as np

# Fixed stand-ins for MATLAB's rand() so the tests are deterministic.
_RAND3 = [0.8147, 0.9058, 0.1270]
_RAND1 = 0.6324


def test_es_hamsymm_axial():
    """hamsymm_axial: axial g -> Dinfh."""
    assert hamsymm(SpinSystem(S=[0.5], g=[[2, 2, 3]]))[0] == 'Dinfh'


def test_es_hamsymm_axialtilted():
    """hamsymm_axialtilted: axial g with arbitrary gFrame -> Dinfh."""
    assert hamsymm(SpinSystem(S=[0.5], g=[[2, 2, 3]], gFrame=[_RAND3]))[0] == 'Dinfh'


def test_es_hamsymm_axialzfs():
    """hamsymm_axialzfs: S=1 with D=[100 0] -> Dinfh."""
    assert hamsymm(SpinSystem(S=[1], g=2.0, D=[100, 0]))[0] == 'Dinfh'


@pytest.mark.parametrize("B4,B6", [(0, 10), (10, 0), (10, 12.87)])
def test_es_hamsymm_cubic(B4, B6):
    """hamsymm_cubic: cubic B4/B6 Stevens terms for S=7/2 -> Oh (eigenvalue branch)."""
    sys = SpinSystem(
        S=[3.5], g=[[2, 2, 2]],
        B=[None,
           [[5 * B4, 0, 0, 0, B4, 0, 0, 0, 0]],
           [[0, 0, -21 * B6, 0, 0, 0, B6, 0, 0, 0, 0, 0, 0]]],
    )
    assert hamsymm(sys)[0] == 'Oh'


def test_es_hamsymm_fullee():
    """hamsymm_fullee: three S=1/2 with full antisymmetric ee matrices runs (Dinfh)."""
    from torchspin.constants import CLIGHT
    J1, J2, dz1 = -100.0, -89.9, 4.85
    dz2 = -dz1
    ee12 = np.array([[-J1, dz1, 0], [-dz1, -J1, 0], [0, 0, -J1]])
    ee13 = np.array([[-J2, dz2, 0], [-dz2, -J2, 0], [0, 0, -J2]])
    EE = np.vstack([ee12, ee13, ee12]) * 100 * CLIGHT / 1e6
    sys = SpinSystem(S=[0.5, 0.5, 0.5], g=[2, 2, 2], ee=EE.tolist())
    pgroup, R = hamsymm(sys)
    assert pgroup == 'Dinfh'   # value recorded from EasySpin (ref_hamsymm.mat)


@pytest.mark.parametrize("scale", [100, 10, 1, 0.1, 0.001])
def test_es_hamsymm_fullhf_magnitude(scale):
    """hamsymm_fullhf_magnitude: full A tilted 24 deg -> C2h regardless of magnitude."""
    from torchspin.rotations import erot
    R = erot([0, 24 * math.pi / 180, 0]).numpy()
    A = R @ np.diag([1.0, 1.0, 2.0]) @ R.T
    sys = SpinSystem(S=[0.5], Nucs='1H', A=(A * scale).tolist())
    assert hamsymm(sys)[0] == 'C2h'


def test_es_hamsymm_isotropic():
    """hamsymm_isotropic: isotropic g -> O3."""
    assert hamsymm(SpinSystem(S=[0.5], g=[[2, 2, 2]]))[0] == 'O3'


def test_es_hamsymm_isotropictilted():
    """hamsymm_isotropictilted: isotropic g with arbitrary gFrame -> O3."""
    assert hamsymm(SpinSystem(S=[0.5], g=[[2, 2, 2]], gFrame=[_RAND3]))[0] == 'O3'


def test_es_hamsymm_rhombicaxial():
    """hamsymm_rhombicaxial: axial g + rhombic A, collinear -> D2h."""
    sys = SpinSystem(S=[0.5], Nucs='1H', g=[[2, 2, 3]], A=[[3, 2, 1]])
    assert hamsymm(sys)[0] == 'D2h'


def test_es_hamsymm_rhombicaxialgeneral():
    """hamsymm_rhombicaxialgeneral: axial g (general tilt) + rhombic A -> Ci."""
    gFrame = [[r + math.pi / 10 for r in _RAND3]]
    sys = SpinSystem(S=[0.5], Nucs='1H', g=[[2, 2, 3]], A=[[3, 2, 1]], gFrame=gFrame)
    assert hamsymm(sys)[0] == 'Ci'


def test_es_hamsymm_rhombicaxialplane():
    """hamsymm_rhombicaxialplane: axial g axis lying in a sigma plane of rhombic A -> C2h."""
    sys = SpinSystem(S=[0.5], Nucs='1H', g=[[2, 2, 3]], A=[[3, 2, 1]],
                     gFrame=[[_RAND1 + math.pi / 10, math.pi / 2, 0]])
    assert hamsymm(sys)[0] == 'C2h'


def test_es_hamsymm_stevensops():
    """hamsymm_stevensops: B2 = [b22 0 b20 0 0] and the equivalent D both give D2h."""
    b20, b22 = 100.0, 150.0
    sys_B = SpinSystem(S=[1], g=[[2, 2, 2]], B=[[[b22, 0, b20, 0, 0]]])
    D = [-b20 + b22, -b20 - b22, 2 * b20]
    sys_D = SpinSystem(S=[1], g=[[2, 2, 2]], D=[D])
    assert hamsymm(sys_B)[0] == 'D2h'
    assert hamsymm(sys_D)[0] == 'D2h'


def test_es_hamsymm_twoaxial():
    """hamsymm_twoaxial: two collinear axial tensors -> Dinfh."""
    sys = SpinSystem(S=[0.5], Nucs='1H', g=[[2, 2, 3]], A=[[3, 3, 1]])
    assert hamsymm(sys)[0] == 'Dinfh'


def test_es_hamsymm_twoaxial90():
    """hamsymm_twoaxial90: two axial tensors at 90 deg -> D2h."""
    sys = SpinSystem(S=[0.5], Nucs='1H', g=[[2, 2, 3]], A=[[3, 3, 1]],
                     AFrame=[[0, -math.pi / 2, 0]])
    assert hamsymm(sys)[0] == 'D2h'


def test_es_hamsymm_twoaxialgeneral():
    """hamsymm_twoaxialgeneral: two axial tensors, general tilt -> C2h (or Ci)."""
    AFrame = [[math.pi * r for r in _RAND3]]
    sys = SpinSystem(S=[0.5], Nucs='1H', g=[[2, 2, 3]], A=[[3, 3, 10]], AFrame=AFrame)
    assert hamsymm(sys)[0] in ('C2h', 'Ci')


def test_es_hamsymm_twoaxialztilt():
    """hamsymm_twoaxialztilt: axial A tilted about y -> C2h."""
    sys = SpinSystem(S=[0.5], Nucs='1H', g=[[2, 2, 3]], A=[[3, 3, 1]],
                     AFrame=[[0, _RAND1 + math.pi / 20, 0]])
    assert hamsymm(sys)[0] == 'C2h'


def test_es_hamsymm_tworhombic():
    """hamsymm_tworhombic: two collinear rhombic tensors -> D2h."""
    sys = SpinSystem(S=[0.5], Nucs='1H', g=[[2, 2.5, 3]], A=[[3, 2, 1]])
    assert hamsymm(sys)[0] == 'D2h'


def test_es_hamsymm_tworhombicgeneral():
    """hamsymm_tworhombicgeneral: two rhombic tensors, general tilt -> Ci."""
    AFrame = [[r + math.pi / 30 for r in _RAND3]]
    sys = SpinSystem(S=[0.5], Nucs='1H', g=[[2, 2.5, 3]], A=[[3, 2, 1]], AFrame=AFrame)
    assert hamsymm(sys)[0] == 'Ci'


def test_es_hamsymm_tworhombiconeaxis():
    """hamsymm_tworhombiconeaxis: two rhombic tensors sharing z -> C2h."""
    sys = SpinSystem(S=[0.5], Nucs='1H', g=[[2, 2.5, 3]], A=[[3, 2, 1]],
                     AFrame=[[_RAND1 + math.pi / 20, 0, 0]])
    assert hamsymm(sys)[0] == 'C2h'


def test_es_hamsymm_twospinA():
    """hamsymm_twospinA: two S=1/2, one proton coupled to both -> Dinfh."""
    sys = SpinSystem(S=[0.5, 0.5], g=[2, 2], Nucs='1H', A=[[1, 1, 1, 2, 2, 3]], ee=[6])
    assert hamsymm(sys)[0] == 'Dinfh'
    sys2 = SpinSystem(S=[0.5, 0.5], g=[2, 2], Nucs='1H', A=[[1, 1, 1, 2, 2, 3]], ee=[6],
                      AFrame=[[0, 0, 0, 0, 0, 0]])
    assert hamsymm(sys2)[0] == 'Dinfh'


# ===========================================================================
# Frame-convention and combiner regression tests
# ===========================================================================

def test_hamsymm_frame_convention_rows_are_axes():
    """R maps mol->sym; its rows are the symmetry axes in molecular coordinates.

    For an axial tensor with gFrame=(a,b,c) the unique axis in the molecular
    frame is the third row of erot(gFrame) (= third column of erot(gFrame).T,
    EasySpin's RMatrix)."""
    from torchspin.rotations import erot
    angles = [0.3, 0.7, 0.2]
    sys = SpinSystem(S=[0.5], g=[[2, 2, 3]], gFrame=[angles])
    pgroup, R = hamsymm(sys)
    assert pgroup == 'Dinfh'
    expected_axis = erot(angles)[2]
    assert torch.allclose(R[2], expected_axis, atol=1e-12)


def test_hamsymm_axial_x_frame_maps_z_to_x():
    """Axial tensor along molecular x: the symmetry z-axis must be x.

    Regression: the previous implementation permuted the columns of the
    mol->sym matrix, which sent the symmetry z-axis to molecular y."""
    sys = SpinSystem(S=[0.5], g=[[2.2, 2.0, 2.0]])
    pgroup, R = hamsymm(sys)
    assert pgroup == 'Dinfh'
    z_sym_in_mol = R.T @ torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    assert torch.allclose(z_sym_in_mol.abs(), torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))


def test_hamsymm_eigs_and_geom_agree_on_tilted_axial():
    """A zero Stevens term forces the eigenvalue branch; the frame must match
    the geometric branch for a tilted axial g."""
    angles = [0.3, 0.7, 0.2]
    sys_geom = SpinSystem(S=[0.5], g=[[2, 2, 3]], gFrame=[angles])
    sys_eigs = SpinSystem(S=[0.5], g=[[2, 2, 3]], gFrame=[angles], B=[[[0, 0, 0, 0, 0]]])
    pg1, R1 = hamsymm(sys_geom)
    pg2, R2 = hamsymm(sys_eigs)
    assert pg1 == pg2 == 'Dinfh'
    assert torch.allclose(R1, R2, atol=1e-12)


def test_hamsymm_c2h_plus_d2h_keeps_c2h_when_axis_shared():
    """C2h (from two oblique axial tensors) + rhombic tensor whose axis is
    parallel to the C2h axis -> stays C2h (EasySpin combinesymms C2h+D2h)."""
    # g axial along z, A1 axial tilted about y -> C2h with z_sym = y (mol)
    # A2 rhombic with its principal y axis along molecular y (untilted) -> C2h
    sys = SpinSystem(
        S=[0.5], Nucs='1H,1H',
        g=[[2, 2, 3]],
        A=[[3, 3, 1], [1, 2, 3]],
        AFrame=[[0, 0.4, 0], [0, 0, 0]],
    )
    assert hamsymm(sys)[0] == 'C2h'


def test_hamsymm_c2h_plus_d2h_falls_to_ci_otherwise():
    """C2h + rhombic tensor sharing no axis -> Ci."""
    sys = SpinSystem(
        S=[0.5], Nucs='1H,1H',
        g=[[2, 2, 3]],
        A=[[3, 3, 1], [1, 2, 3]],
        AFrame=[[0, 0.4, 0], [0.3, 0.5, 0.1]],
    )
    assert hamsymm(sys)[0] == 'Ci'


def test_hamsymm_nn_tensor_counts():
    """Anisotropic nuclear-nuclear coupling lowers the symmetry."""
    sys_iso = SpinSystem(S=[0.5], g=[[2, 2, 3]], Nucs='1H,1H', A=[[1, 1, 1], [1, 1, 1]])
    sys_nn = SpinSystem(S=[0.5], g=[[2, 2, 3]], Nucs='1H,1H', A=[[1, 1, 1], [1, 1, 1]],
                        nn=[[0.1, 0.1, 0.3]], nnFrame=[[0, 0.5, 0]])
    assert hamsymm(sys_iso)[0] == 'Dinfh'
    assert hamsymm(sys_nn)[0] == 'C2h'


def test_hamsymm_scalar_Q_stays_geometric():
    """A 1-D Q triple for one nucleus is a principal-value list, not a full
    matrix: hamsymm must stay on the geometric path (regression for the
    SpinSystem.fullQ misclassification)."""
    sys = SpinSystem(S=[0.5], g=2.0, Nucs='2H', A=[10, 10, 10], Q=[-1, -1, 2])
    assert hamsymm(sys)[0] == 'Dinfh'
