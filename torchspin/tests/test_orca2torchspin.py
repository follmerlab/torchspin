"""Tests for torchspin.orca2torchspin — ORCA QC output → SpinSystem.

Mirrors the 18 MATLAB test files in tests/orca2easyspin_*.m.
Test data: 47+ files in tests/orca/.
"""
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from torchspin.orca2torchspin import (
    orca2torchspin,
    OrcaData,
    _element_to_number,
    _number_to_element,
    _reference_isotope,
    _reference_q_isotope,
    _symmetrize_g,
    _ensure_righthanded,
    _CM1_TO_MHZ,
)
from torchspin.rotutils import eulang
from torchspin.rotations import erot

# Root of the test data directory
ORCA_DIR = Path(__file__).resolve().parents[2] / 'tests' / 'orca'


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _rotation_matrix(euler_angles):
    """Convert Euler angles (tensor or array) to 3x3 rotation matrix."""
    if isinstance(euler_angles, torch.Tensor):
        angles = euler_angles.detach().numpy()
    else:
        angles = np.asarray(euler_angles)
    if angles.ndim == 2:
        angles = angles[0]  # first nucleus/electron
    return erot(angles)


def _assert_rotation_close(R1, R2, atol=1e-2):
    """Assert two rotation matrices are close (up to sign ambiguity)."""
    # Rotation matrices can differ by sign of individual columns
    assert np.allclose(np.abs(R1), np.abs(R2), atol=atol), (
        f"Rotation matrices differ:\n{R1}\nvs\n{R2}"
    )


# =========================================================================
# Category 1: Utility functions
# =========================================================================
class TestElementMapping:
    def test_hydrogen(self):
        assert _element_to_number('H') == 1

    def test_nitrogen(self):
        assert _element_to_number('N') == 7

    def test_copper(self):
        assert _element_to_number('Cu') == 29

    def test_reverse_hydrogen(self):
        assert _number_to_element(1) == 'H'

    def test_reverse_copper(self):
        assert _number_to_element(29) == 'Cu'

    def test_roundtrip(self):
        for Z in [1, 6, 7, 8, 26, 29, 79]:
            sym = _number_to_element(Z)
            assert _element_to_number(sym) == Z


class TestReferenceIsotope:
    def test_hydrogen(self):
        assert _reference_isotope('H') == '1H'

    def test_nitrogen(self):
        assert _reference_isotope('N') == '14N'

    def test_carbon(self):
        assert _reference_isotope('C') == '13C'

    def test_copper(self):
        assert _reference_isotope('Cu') == '63Cu'

    def test_oxygen(self):
        iso = _reference_isotope('O')
        assert iso == '17O'

    def test_q_nitrogen(self):
        result = _reference_q_isotope('N')
        assert result is not None
        assert result[0] == '14N'
        assert result[1] == 1.0  # I=1


class TestGSymmetrize:
    def test_identity(self):
        g = np.eye(3) * 2.0023
        g_sym = _symmetrize_g(g)
        np.testing.assert_allclose(g_sym, g, atol=1e-10)

    def test_asymmetric(self):
        # An asymmetric g-matrix should be symmetrised
        g_raw = np.array([[2.003, 0.001, 0.0],
                          [0.0005, 2.006, 0.0],
                          [0.0, 0.0, 2.002]])
        g_sym = _symmetrize_g(g_raw)
        np.testing.assert_allclose(g_sym, g_sym.T, atol=1e-14)


# =========================================================================
# Category 2: File format reading (smoke tests)
# Mirrors: orca2easyspin_basic.m, orca2easyspin_hydroxyl.m
# =========================================================================
class TestReadFormats:
    """Smoke tests: files parse without error and return valid structures."""

    def test_read_prop_basic(self):
        """Mirrors orca2easyspin_basic.m — read binary .prop files."""
        files = ['tp051512_opt.prop', 'dioxygen_g.prop', 'nx4me.prop']
        for fname in files:
            sys, data = orca2torchspin(ORCA_DIR / fname)
            assert isinstance(data, OrcaData)
            assert len(sys.S) >= 1

    def test_read_prop_with_cutoff(self):
        files = ['tp051512_opt.prop', 'dioxygen_g.prop', 'nx4me.prop']
        for fname in files:
            sys, data = orca2torchspin(ORCA_DIR / fname, hf_cutoff=1.0)
            assert isinstance(data, OrcaData)

    def test_read_hydroxyl_oof(self):
        """Mirrors orca2easyspin_hydroxyl.m — read .oof files."""
        files = [
            'hydroxyl_g.oof', 'hydroxyl_gA.oof', 'hydroxyl_gAiso.oof',
            'hydroxyl_gQ.oof', 'hydroxyl_Q.oof', 'hydroxyl_HO.oof',
        ]
        for fname in files:
            sys, data = orca2torchspin(ORCA_DIR / fname)
            assert isinstance(data, OrcaData)

    def test_read_hydroxyl_prop(self):
        files = [
            'hydroxyl_g.prop', 'hydroxyl_gA.prop', 'hydroxyl_gAiso.prop',
            'hydroxyl_gQ.prop', 'hydroxyl_Q.prop', 'hydroxyl_HO.prop',
        ]
        for fname in files:
            sys, data = orca2torchspin(ORCA_DIR / fname)
            assert isinstance(data, OrcaData)

    def test_read_main_output(self):
        sys, data = orca2torchspin(ORCA_DIR / 'quinoline_triplet.out')
        assert isinstance(data, OrcaData)
        assert data.orca_version == '5.0.4'

    def test_read_property_txt(self):
        sys, data = orca2torchspin(ORCA_DIR / 'methyl_all_property.txt')
        assert isinstance(data, OrcaData)

    def test_read_triplet_oof(self):
        """Mirrors orca2easyspin_triplet.m."""
        sys, data = orca2torchspin(ORCA_DIR / 'dioxygen_g.oof')
        assert isinstance(data, OrcaData)

    def test_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            orca2torchspin('nonexistent_file.out')


# =========================================================================
# Category 3: Spin extraction
# Mirrors: orca2easyspin_spin.m
# =========================================================================
class TestSpinExtraction:
    def test_doublet(self):
        """Hydroxyl radical (OH) is a doublet (S=1/2)."""
        sys, _ = orca2torchspin(ORCA_DIR / 'hydroxyl_098_v400.oof')
        assert sys.S[0] == pytest.approx(0.5)

    def test_triplet(self):
        """Dioxygen is a triplet (S=1)."""
        sys, _ = orca2torchspin(ORCA_DIR / 'dioxygen_gD_v400.oof')
        assert sys.S[0] == pytest.approx(1.0)

    def test_singlet(self):
        """Water is a singlet (S=0)."""
        sys, _ = orca2torchspin(ORCA_DIR / 'water.oof')
        assert sys.S[0] == pytest.approx(0.0)


# =========================================================================
# Category 4: Tensor extraction accuracy
# =========================================================================
class TestGTensor:
    def test_quinoline_g_values(self):
        """Quinoline triplet g principal values from .out file."""
        sys, _ = orca2torchspin(ORCA_DIR / 'quinoline_triplet.out')
        g = sys.g.numpy().flatten()
        # From ORCA output: g(tot) = 2.0022043, 2.0030905, 2.0038265
        np.testing.assert_allclose(g, [2.0022043, 2.0030905, 2.0038265],
                                   atol=1e-4)

    def test_hydroxyl_g_values(self):
        """Hydroxyl g principal values from .prop file."""
        sys, _ = orca2torchspin(ORCA_DIR / 'hydroxyl_098_v400.prop')
        g = sys.g.numpy().flatten()
        # g values should be near free-electron with small anisotropy
        assert np.all(g > 2.0) and np.all(g < 2.1)
        # Verify ordering (ascending from eigh)
        assert g[0] <= g[1] <= g[2]

    def test_g_finiteness(self):
        """All g values must be finite."""
        sys, _ = orca2torchspin(ORCA_DIR / 'Pr_EPR.out')
        assert torch.all(torch.isfinite(sys.g))


class TestDTensor:
    def test_quinoline_D_values(self):
        """Quinoline triplet D tensor in MHz from .out file."""
        sys, _ = orca2torchspin(ORCA_DIR / 'quinoline_triplet.out')
        D = sys.D.numpy().flatten()
        # raw D eigenvalues in cm⁻¹: [-0.020438, -0.014204, 0.034642]
        D_expected = np.array([-0.020438, -0.014204, 0.034642]) * _CM1_TO_MHZ
        np.testing.assert_allclose(D, D_expected, rtol=1e-3)

    def test_dioxygen_D_from_prop(self):
        """Dioxygen D tensor from binary .prop file."""
        sys, _ = orca2torchspin(ORCA_DIR / 'dioxygen_gD.prop')
        assert sys.D is not None
        D = sys.D.numpy().flatten()
        assert np.all(np.isfinite(D))

    def test_no_D_when_absent(self):
        """Hydroxyl (doublet) has no ZFS."""
        sys, _ = orca2torchspin(ORCA_DIR / 'hydroxyl_g.oof')
        assert sys.D is None


class TestATensor:
    def test_quinoline_A_nitrogen(self):
        """Quinoline 14N hyperfine from .out file."""
        sys, _ = orca2torchspin(ORCA_DIR / 'quinoline_triplet.out')
        assert sys.nNuclei == 1
        assert sys.Nucs == ['14N']
        A = sys.A.numpy().flatten()
        # From ORCA: A(Tot) = -1.3530, -1.7576, 22.3594 MHz
        np.testing.assert_allclose(A, [-1.3530, -1.7576, 22.3594], atol=0.01)

    def test_methyl_A_all_nuclei(self):
        """Methyl radical: 1 C + 3 H from property.txt."""
        sys, _ = orca2torchspin(ORCA_DIR / 'methyl_all_property.txt')
        assert sys.nNuclei == 4
        assert sys.Nucs[0] == '13C'
        assert all(n == '1H' for n in sys.Nucs[1:])

    def test_A_finiteness(self):
        """All A values must be finite."""
        sys, _ = orca2torchspin(ORCA_DIR / 'Pr_EPR.out')
        if sys.nNuclei > 0:
            assert torch.all(torch.isfinite(sys.A))


class TestQTensor:
    def test_quinoline_Q_nitrogen(self):
        """Quinoline 14N quadrupole from .out file."""
        sys, _ = orca2torchspin(ORCA_DIR / 'quinoline_triplet.out')
        assert sys.Q is not None
        Q = sys.Q.numpy().flatten()
        assert np.all(np.isfinite(Q))
        # Q should be small (order of ~1 MHz for 14N)
        assert np.max(np.abs(Q)) < 10.0

    def test_hydroxyl_Q(self):
        """Hydroxyl Q tensor from .oof file."""
        sys, _ = orca2torchspin(ORCA_DIR / 'hydroxyl_gQ.oof')
        assert sys.Q is not None
        Q = sys.Q.numpy().flatten()
        assert np.all(np.isfinite(Q))


class TestCoordinates:
    def test_quinoline_nAtoms(self):
        """Quinoline C9H6N has 16+1=17 atoms (including N)."""
        _, data = orca2torchspin(ORCA_DIR / 'quinoline_triplet.out')
        assert data.xyz.shape == (17, 3)
        assert len(data.elements) == 17
        assert data.elements[0] == 'N'

    def test_water_coordinates(self):
        """Water has 3 atoms."""
        _, data = orca2torchspin(ORCA_DIR / 'water.oof')
        assert data.xyz.shape == (3, 3)
        assert 'O' in data.elements
        assert data.elements.count('H') == 2


# =========================================================================
# Category 5: Frame orientation (critical for spectral accuracy)
# Mirrors: orca2easyspin_frames_main.m
# =========================================================================
class TestFrameOrientation:
    """Test that tensor frames are extracted correctly for quinoline triplet.

    The quinoline triplet is a planar molecule. Key physics:
    - g tensor x-axis should be along the out-of-plane direction
    - D tensor z-axis should be along the out-of-plane direction
    - A tensor z-axis should be along the out-of-plane direction
    """

    @pytest.fixture
    def quinoline_tilted(self):
        sys, data = orca2torchspin(ORCA_DIR / 'quinoline_triplet_tilted.out')
        return sys, data

    def _get_oop_axis(self, data):
        """Calculate out-of-plane axis from N, C4, C9 positions."""
        xyz = data.xyz
        N = xyz[0]   # atom 0 = N
        C4 = xyz[3]  # atom 3 = C4
        C9 = xyz[8]  # atom 8 = C9
        oop = np.cross(C4 - N, C9 - N)
        return oop / np.linalg.norm(oop)

    def test_gx_along_oop(self, quinoline_tilted):
        """g tensor x-axis should be along out-of-plane direction."""
        sys, data = quinoline_tilted
        oop = self._get_oop_axis(data)

        R_M2g = _rotation_matrix(sys.gFrame)
        R_g2M = R_M2g.T
        gx_M = R_g2M @ np.array([1, 0, 0])  # x-axis in molecular frame

        assert abs(abs(gx_M @ oop) - 1.0) < 0.02

    def test_Dz_along_oop(self, quinoline_tilted):
        """D tensor z-axis should be along out-of-plane direction."""
        sys, data = quinoline_tilted
        oop = self._get_oop_axis(data)

        R_M2D = _rotation_matrix(sys.DFrame)
        R_D2M = R_M2D.T
        Dz_M = R_D2M @ np.array([0, 0, 1])

        assert abs(abs(Dz_M @ oop) - 1.0) < 0.01

    def test_Az_along_oop(self, quinoline_tilted):
        """A tensor z-axis should be along out-of-plane direction."""
        sys, data = quinoline_tilted
        oop = self._get_oop_axis(data)

        AFrame = sys.AFrame.numpy()[0]  # first nucleus
        R_M2A = erot(AFrame)
        R_A2M = R_M2A.T
        Az_M = R_A2M @ np.array([0, 0, 1])

        assert abs(abs(Az_M @ oop) - 1.0) < 0.01

    def test_Qy_along_oop(self, quinoline_tilted):
        """Q tensor y-axis should be along out-of-plane direction."""
        sys, data = quinoline_tilted
        if sys.Q is None:
            pytest.skip("No Q tensor")
        oop = self._get_oop_axis(data)

        QFrame = sys.QFrame.numpy()[0]
        R_M2Q = erot(QFrame)
        R_Q2M = R_M2Q.T
        Qy_M = R_Q2M @ np.array([0, 1, 0])

        assert abs(abs(Qy_M @ oop) - 1.0) < 0.01


# =========================================================================
# Category 6: Cross-format consistency
# Mirrors: orca2easyspin_dioxygen.m, orca2easyspin_consistency.m
# =========================================================================
class TestCrossFormatConsistency:
    def test_dioxygen_out_vs_proptxt(self):
        """Mirrors orca2easyspin_dioxygen.m — .oof vs _property.txt."""
        sys1, _ = orca2torchspin(ORCA_DIR / 'dioxygen_singlepoint.oof')
        sys2, _ = orca2torchspin(ORCA_DIR / 'dioxygen_singlepoint_property.txt')

        # g values
        np.testing.assert_allclose(
            sys1.g.numpy(), sys2.g.numpy(), rtol=1e-4)

        # g frame (compare rotation matrices for sign ambiguity)
        R1 = _rotation_matrix(sys1.gFrame)
        R2 = _rotation_matrix(sys2.gFrame)
        _assert_rotation_close(R1, R2, atol=1e-4)

        # D values
        np.testing.assert_allclose(
            sys1.D.numpy(), sys2.D.numpy(), rtol=1e-4)

        # A values
        np.testing.assert_allclose(
            sys1.A.numpy(), sys2.A.numpy(), rtol=1e-4)

    def test_quinoline_v5_vs_v4_out(self):
        """Mirrors orca2easyspin_consistency.m — ORCA v5 vs v4 .out files."""
        sys1, _ = orca2torchspin(ORCA_DIR / 'quinoline_triplet_tilted.out')
        sys2, _ = orca2torchspin(ORCA_DIR / 'quinoline_triplet_tilted_v4.out')

        # Principal values should agree within tolerances
        # Different ORCA versions give slightly different results
        np.testing.assert_allclose(
            sys1.g.numpy(), sys2.g.numpy(), rtol=1e-2)
        np.testing.assert_allclose(
            sys1.D.numpy(), sys2.D.numpy(), rtol=1e-2)
        # A-tensor can differ up to ~3% between ORCA versions
        np.testing.assert_allclose(
            sys1.A.numpy(), sys2.A.numpy(), rtol=5e-2)

        # Frame rotation matrices should agree
        R1_D = _rotation_matrix(sys1.DFrame)
        R2_D = _rotation_matrix(sys2.DFrame)
        _assert_rotation_close(R1_D, R2_D, atol=1e-2)

    def test_quinoline_v4_out_vs_prop(self):
        """ORCA v4 main output vs binary property file."""
        sys1, _ = orca2torchspin(ORCA_DIR / 'quinoline_triplet_tilted_v4.out')
        sys2, _ = orca2torchspin(ORCA_DIR / 'quinoline_triplet_tilted_v4.prop')

        np.testing.assert_allclose(
            sys1.g.numpy(), sys2.g.numpy(), rtol=1e-2)
        np.testing.assert_allclose(
            sys1.D.numpy(), sys2.D.numpy(), rtol=1e-2)
        np.testing.assert_allclose(
            sys1.A.numpy(), sys2.A.numpy(), rtol=1e-2)


# =========================================================================
# Category 7: Multi-structure files and filtering
# Mirrors: orca2easyspin_proptxt_scans.m
# =========================================================================
class TestMultiStructure:
    def test_optscan_oof(self):
        """Relaxed scan should return multiple structures."""
        result = orca2torchspin(ORCA_DIR / 'hydroxyl_optscan.oof')
        assert isinstance(result, list)
        assert len(result) >= 2
        for sys, data in result:
            assert len(sys.S) >= 1

    def test_parascan_oof(self):
        """Parameter scan should return multiple structures."""
        result = orca2torchspin(ORCA_DIR / 'hydroxyl_parascan.oof')
        assert isinstance(result, list)
        assert len(result) >= 2


class TestHyperfineCutoff:
    def test_cutoff_filters_nuclei(self):
        """High cutoff should remove weakly-coupled nuclei."""
        sys_all, _ = orca2torchspin(ORCA_DIR / 'hydroxyl_098_v400.prop')
        sys_cut, _ = orca2torchspin(ORCA_DIR / 'hydroxyl_098_v400.prop',
                                    hf_cutoff=200.0)
        # Original has hydrogen, cutoff removes it if max|A| < 200
        A_max = float(torch.max(torch.abs(sys_all.A)))
        if A_max <= 200.0:
            assert sys_cut.nNuclei == 0
        else:
            assert sys_cut.nNuclei <= sys_all.nNuclei

    def test_zero_cutoff_keeps_all(self):
        """Zero cutoff should keep all nuclei with any HFC."""
        sys, _ = orca2torchspin(ORCA_DIR / 'Pr_EPR.out', hf_cutoff=0)
        assert sys.nNuclei >= 1

    def test_large_cutoff_removes_all(self):
        """Very large cutoff should remove all nuclei."""
        sys, _ = orca2torchspin(ORCA_DIR / 'quinoline_triplet.out',
                                hf_cutoff=1e6)
        assert sys.nNuclei == 0


# =========================================================================
# Category 8: ORCA version handling
# Mirrors: orca2easyspin_main_versions.m
# =========================================================================
class TestVersionHandling:
    def test_v303(self):
        """ORCA 3.0.3 output should parse."""
        sys, _ = orca2torchspin(ORCA_DIR / 'hydroxyl_098_v303.oof')
        assert sys.S[0] == pytest.approx(0.5)

    def test_v400(self):
        """ORCA 4.0.0 output should parse."""
        sys, _ = orca2torchspin(ORCA_DIR / 'hydroxyl_098_v400.oof')
        assert sys.S[0] == pytest.approx(0.5)

    def test_v503(self):
        """ORCA 5.0.3 output should parse."""
        sys, _ = orca2torchspin(ORCA_DIR / 'hydroxyl_098_v503.oof')
        assert sys.S[0] == pytest.approx(0.5)


# =========================================================================
# Category 9: Integration with torchspin simulators
# =========================================================================
class TestIntegration:
    def test_pepper_from_orca(self):
        """Load ORCA → SpinSystem → pepper() produces valid spectrum."""
        from torchspin.pepper import pepper
        from torchspin.experiment import Experiment

        sys, _ = orca2torchspin(ORCA_DIR / 'hydroxyl_098_v400.oof')
        exp = Experiment(mwFreq=9.5, Range=[300, 360], nPoints=512)
        B, spc = pepper(sys, exp)
        spc = np.asarray(spc)
        assert len(B) == 512
        assert np.all(np.isfinite(spc))
        assert np.max(np.abs(spc)) > 0

    def test_autograd_from_orca(self):
        """Loaded SpinSystem tensors support requires_grad."""
        sys, _ = orca2torchspin(ORCA_DIR / 'hydroxyl_098_v400.oof')

        # Clone g with gradient tracking
        g_param = sys.g.clone().detach().requires_grad_(True)
        assert g_param.requires_grad

        # Simple operation to verify gradient flows
        loss = g_param.sum()
        loss.backward()
        assert g_param.grad is not None
        assert torch.all(torch.isfinite(g_param.grad))


# =========================================================================
# Category 10: OrcaData metadata
# =========================================================================
class TestOrcaData:
    def test_metadata_fields(self):
        _, data = orca2torchspin(ORCA_DIR / 'quinoline_triplet.out')
        assert data.orca_version == '5.0.4'
        assert data.charge == 0
        assert data.multiplicity == 3
        assert data.xyz.shape[0] == 17
        assert len(data.elements) == 17

    def test_mulliken(self):
        _, data = orca2torchspin(ORCA_DIR / 'quinoline_triplet.out')
        assert data.mulliken_charge is not None
        assert len(data.mulliken_charge) == 17
        assert data.mulliken_spin is not None

    def test_raw_g_matrix(self):
        _, data = orca2torchspin(ORCA_DIR / 'quinoline_triplet.out')
        assert data.graw is not None
        assert data.graw.shape == (3, 3)


# =========================================================================
# Category 11: Edge cases and robustness
# =========================================================================
class TestEdgeCases:
    def test_all_orca_files_parseable(self):
        """Every file in tests/orca/ (except .inp/.oif) should parse."""
        skip_exts = {'.inp', '.oif'}
        failures = []
        for f in sorted(ORCA_DIR.iterdir()):
            if f.is_dir() or f.suffix in skip_exts:
                continue
            try:
                orca2torchspin(str(f))
            except Exception as e:
                failures.append(f"{f.name}: {e}")
        assert not failures, f"Failed files:\n" + "\n".join(failures)

    def test_g_tensor_positive(self):
        """g principal values should be positive (physical constraint)."""
        sys, _ = orca2torchspin(ORCA_DIR / 'quinoline_triplet.out')
        assert torch.all(sys.g > 0)

    def test_dtype_float64(self):
        """All tensors should be float64."""
        sys, _ = orca2torchspin(ORCA_DIR / 'quinoline_triplet.out')
        assert sys.g.dtype == torch.float64
        if sys.D is not None:
            assert sys.D.dtype == torch.float64
        if sys.nNuclei > 0:
            assert sys.A.dtype == torch.float64
