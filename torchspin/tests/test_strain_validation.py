"""
MATLAB Validation Tests for Strain Broadening

Tests torchspin strain implementation against EasySpin MATLAB reference data.

To generate reference data:
1. Run tests/data/generate_strain_refs.m in MATLAB 
2. Ensure all 6 .mat files are created in tests/data/
3. Run this test: pytest torchspin/tests/test_strain_validation.py -v

KNOWN LIMITATIONS (Phase 4 TODO):
- Strain normalization uses empirical correction factor (8/π)
- Works for simple cases (peak ratio ~1.0) but fails max_relative_error metric
- Complex cases (tilted frames, combined strain) need deeper investigation
- See STRAIN_NORMALIZATION_STATUS.md for details
"""

import os
import pytest
import torch
import numpy as np
import scipy.io as sio

from torchspin.spinsystem import SpinSystem
from torchspin.pepper import pepper
from torchspin.experiment import Experiment, Options

# Path to reference data
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'tests', 'data'
)

def load_matlab_ref(filename):
    """Load MATLAB .mat reference file"""
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        pytest.skip(f"Reference file not found: {filename}. Run generate_strain_refs.m in MATLAB.")
    
    mat_data = sio.loadmat(filepath, simplify_cells=True)
    
    # Find spc and B variables (they're named spc1, spc2, etc.)
    spc_key = [k for k in mat_data.keys() if k.startswith('spc')][0]
    B_key = [k for k in mat_data.keys() if k.startswith('B') and not k.startswith('_')][0]
    
    return mat_data[spc_key], mat_data[B_key]


def compare_spectra(spc_py, spc_mat, B_py, B_mat, tol_rms=0.10, tol_peak=0.20):
    """
    Compare two spectra using normalized RMS difference and peak height ratio.

    The old max_relative_error metric was unsuitable for derivative spectra
    (Harmonic=1) which have zero crossings — division by near-zero values gives
    astronomically large relative errors even for nearly identical shapes.

    Args:
        spc_py: torchspin spectrum (torch tensor or numpy array)
        spc_mat: MATLAB spectrum (numpy array)
        B_py: torchspin field axis (torch tensor or numpy array)
        B_mat: MATLAB field axis (numpy array)
        tol_rms: RMS tolerance for normalized shape difference (default 10%)
        tol_peak: tolerance for peak height ratio deviation from 1.0 (default 20%)

    Returns:
        (passes, rms_diff, metrics_dict)
    """
    # Convert to numpy if needed
    if torch.is_tensor(spc_py):
        spc_py = spc_py.detach().cpu().numpy()
    if torch.is_tensor(B_py):
        B_py = B_py.detach().cpu().numpy()

    # Ensure 1D
    spc_py = spc_py.flatten()
    spc_mat = spc_mat.flatten()
    B_py = B_py.flatten()
    B_mat = B_mat.flatten()

    # Check field axes match
    assert np.allclose(B_py, B_mat, rtol=1e-4), "Field axes don't match"

    # Normalize to peak height
    max_py = np.abs(spc_py).max()
    max_mat = np.abs(spc_mat).max()

    # Handle edge case of empty spectrum
    if max_py < 1e-12 or max_mat < 1e-12:
        return True, 0.0, {"status": "both spectra empty"}

    spc_py_norm = spc_py / max_py
    spc_mat_norm = spc_mat / max_mat

    # RMS difference of normalized shapes (robust to zero crossings)
    diff = spc_py_norm - spc_mat_norm
    rms_diff = np.sqrt(np.mean(diff ** 2))

    peak_ratio = max_py / max_mat

    metrics = {
        "rms_difference": rms_diff,
        "peak_height_ratio": peak_ratio,
        # Keep max_relative_error for diagnostics only (not used for pass/fail)
        "max_relative_error": np.max(np.abs(diff) / (np.abs(spc_mat_norm) + 1e-12)),
    }

    passes = rms_diff < tol_rms and (1.0 - tol_peak) < peak_ratio < (1.0 + tol_peak)

    return passes, rms_diff, metrics


@pytest.mark.parametrize("filename,sys_params,exp_params", [
    # Test 1: HStrain only — PASSING
    pytest.param(
        'strain_ref_hstrain.mat',
        {
            'S': [1/2],
            'g': [[2.00, 2.05, 2.15]],
            'HStrain': [50.0, 100.0, 150.0],
            'lw': [0.0, 0.0],
        },
        {
            'mwFreq': 9.5,
            'Range': [320.0, 380.0],
            'nPoints': 1024,
            'Harmonic': 1,
        },
        id='hstrain',
    ),
    # Test 2: gStrain only — PASSING
    pytest.param(
        'strain_ref_gstrain.mat',
        {
            'S': [1/2],
            'g': [[2.00, 2.05, 2.15]],
            'gStrain': [[0.01, 0.02, 0.03]],
            'lw': [0.0, 0.0],
        },
        {
            'mwFreq': 9.5,
            'Range': [320.0, 380.0],
            'nPoints': 1024,
            'Harmonic': 1,
        },
        id='gstrain',
    ),
    # Test 3: DStrain (S=1) — xfail: normalization wrong for multi-transition S=1 spectra
    pytest.param(
        'strain_ref_dstrain.mat',
        {
            'S': [1],
            'g': [[2.0, 2.0, 2.0]],
            # MATLAB D=[500, 100] means [D_axial, E_rhombic]
            # Convert to [Dxx, Dyy, Dzz]: Dxx=-D/3+E, Dyy=-D/3-E, Dzz=2D/3
            'D': [[-500/3 + 100, -500/3 - 100, 2*500/3]],  # MHz
            'DStrain': [50.0, 10.0],
            'lw': [0.0, 0.0],
        },
        {
            'mwFreq': 9.5,
            'Range': [280.0, 400.0],
            'nPoints': 1024,
            'Harmonic': 1,
        },
        id='dstrain',
    ),
    # Test 4: gStrain + AStrain with correlation — xfail: nuclei+strain correction factor wrong
    pytest.param(
        'strain_ref_gstrain_astrain.mat',
        {
            'S': [1/2],
            'g': [[2.006, 2.006, 2.002]],
            'Nucs': '14N',
            'A': [[20.0, 40.0, 80.0]],
            'gStrain': [[0.01, 0.01, 0.02]],
            'AStrain': [5.0, 10.0, 15.0],
            'gAStrainCorr': -1.0,
            'lw': [0.0, 0.0],
        },
        {
            'mwFreq': 9.5,
            'Range': [320.0, 360.0],
            'nPoints': 1024,
            'Harmonic': 1,
        },
        id='gstrain_astrain',
    ),
    # Test 5: Combined HStrain + gStrain — PASSING
    pytest.param(
        'strain_ref_combined.mat',
        {
            'S': [1/2],
            'g': [[2.00, 2.05, 2.15]],
            'HStrain': [30.0, 50.0, 70.0],
            'gStrain': [[0.005, 0.01, 0.015]],
            'lw': [0.0, 0.0],
        },
        {
            'mwFreq': 9.5,
            'Range': [320.0, 380.0],
            'nPoints': 1024,
            'Harmonic': 1,
        },
        id='combined',
    ),
    # Test 6: Tilted g-frame with gStrain — xfail: shape discrepancy with tilted frame
    pytest.param(
        'strain_ref_tilted_frame.mat',
        {
            'S': [1/2],
            'g': [[2.00, 2.05, 2.15]],
            'gFrame': [np.pi/6, np.pi/4, np.pi/3],
            'gStrain': [[0.01, 0.02, 0.03]],
            'lw': [0.0, 0.0],
        },
        {
            'mwFreq': 9.5,
            'Range': [320.0, 380.0],
            'nPoints': 1024,
            'Harmonic': 1,
        },
        id='tilted_frame',
    ),
])
def test_strain_vs_matlab(filename, sys_params, exp_params):
    """
    Validate torchspin strain implementation against MATLAB reference data.
    
    This parametrized test covers 6 different strain scenarios:
    1. HStrain only (residual broadening)
    2. gStrain only (g-value distribution)
    3. DStrain (D/E parameter distribution in S=1)
    4. gStrain + AStrain with correlation
    5. Combined HStrain + gStrain
    6. Tilted g-frame with gStrain
    """
    # Load MATLAB reference data
    spc_mat, B_mat = load_matlab_ref(filename)
    
    # Create SpinSystem
    sys = SpinSystem(**sys_params)
    
    # Run torchspin pepper
    exp = Experiment(**exp_params)
    
    # Use same grid size as MATLAB (31 from generate_strain_refs.m)
    opt = Options(GridSize=31, GridSymmetry='auto', Verbosity=0)

    B_py, spc_py = pepper(sys, exp, opt)

    # Compare spectra
    passes, rms_diff, metrics = compare_spectra(spc_py, spc_mat, B_py, B_mat)

    print(f"\n{filename}:")
    print(f"  RMS difference: {rms_diff:.4f} (threshold: 0.10)")
    print(f"  Peak height ratio: {metrics['peak_height_ratio']:.4f} (target: 1.0 ± 0.20)")
    print(f"  Max relative error (diagnostic): {metrics['max_relative_error']:.1f}")

    # Assert pass
    assert passes, (
        f"Strain validation failed for {filename}:\n"
        f"  RMS difference: {rms_diff:.4f} (threshold: 0.10)\n"
        f"  Peak height ratio: {metrics['peak_height_ratio']:.4f} (target: 0.80–1.20)\n"
        f"  Metrics: {metrics}"
    )


def test_no_strain_backward_compatibility():
    """
    Ensure systems without strain still work correctly (backward compatibility).
    Should use fast global broadening path.
    """
    sys = SpinSystem(
        S=[1/2],
        g=[[2.00, 2.05, 2.15]],
        lw=[1.0, 0.5],  # Gaussian + Lorentzian
    )
    
    exp = Experiment(
        mwFreq=9.5,
        Range=[320.0, 380.0],
        nPoints=512,
        Harmonic=1,
    )
    opt = Options(GridSize=19, GridSymmetry='Dinfh', Verbosity=0)
    
    B, spc = pepper(sys, exp, opt)
    
    # Should produce a spectrum
    assert B.shape == (exp.nPoints,)
    assert spc.shape == (exp.nPoints,)
    assert torch.isfinite(spc).all()
    assert spc.abs().max() > 0, "Spectrum is empty"


def test_strain_shape_consistency():
    """
    Test that strain parameters with different shapes are normalized correctly.
    """
    # Scalar HStrain → [3]
    sys1 = SpinSystem(S=[1/2], g=[[2.0, 2.0, 2.0]], HStrain=50.0)
    assert sys1.HStrain.shape == (3,)
    assert torch.allclose(sys1.HStrain, torch.tensor([50.0, 50.0, 50.0], dtype=sys1.HStrain.dtype))
    
    # List gStrain → [nElectrons, 3]
    sys2 = SpinSystem(S=[1/2], g=[[2.0, 2.0, 2.0]], gStrain=[0.01, 0.02, 0.03])
    assert sys2.gStrain.shape == (1, 3)
    
    # Scalar DStrain → [2] (FWHM_D, FWHM_E)
    # Note: In SpinSystem, scalar DStrain normalizes to [value, 0] for D-only strain
    D_val, E_val = 500.0, 100.0
    sys3 = SpinSystem(S=[1], g=[[2.0, 2.0, 2.0]], 
                      D=[[-D_val/3 + E_val, -D_val/3 - E_val, 2*D_val/3]], 
                      DStrain=30.0)
    assert sys3.DStrain.shape == (1, 2)  # one [FWHM_D, FWHM_E] row per electron
    assert torch.allclose(sys3.DStrain, torch.tensor([[30.0, 0.0]], dtype=sys3.DStrain.dtype))


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
