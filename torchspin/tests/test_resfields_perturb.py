"""Tests for perturbation theory resonance field calculation."""
import pytest
import torch
import numpy as np

from torchspin.resfields_perturb import resfields_perturb
from torchspin.resfields import resfields
from torchspin import ham
from torchspin.spinsystem import SpinSystem
from torchspin.experiment import Experiment, Options


@pytest.fixture
def exp_defaults():
    """Default experiment parameters."""
    return Experiment(mwFreq=9.5, Range=[320, 360])


@pytest.fixture
def opt_defaults():
    """Default options."""
    return Options(Threshold=1e-6)


def test_perturb_s_half_isotropic_matches_matrix(exp_defaults, opt_defaults):
    """Perturbation theory matches matrix method for S=1/2, isotropic g."""
    sys = SpinSystem(S=1/2, g=2.0)
    phi, theta = 0.0, 0.0

    # Perturbation theory
    B_pert, I_pert = resfields_perturb(sys, phi, theta, exp_defaults, opt_defaults)

    # Matrix method
    H0, mux, muy, muz = ham(sys)
    B_mat, I_mat, _ = resfields(H0, mux, muy, muz, phi, theta, exp_defaults, opt_defaults, sys=sys)

    # Should have 1 transition
    assert len(B_pert) == 1
    assert len(B_mat) == 1

    # Fields should match within numerical tolerance
    assert torch.allclose(B_pert, B_mat, rtol=1e-10, atol=1e-12)
    
    # Intensities should match
    assert torch.allclose(I_pert, I_mat, rtol=1e-6)


def test_perturb_s_half_anisotropic_matches_matrix(exp_defaults, opt_defaults):
    """Perturbation theory matches matrix method for anisotropic g."""
    sys = SpinSystem(S=1/2, g=[2.002, 2.005, 2.008])
    phi, theta = 0.3, 0.7

    B_pert, I_pert = resfields_perturb(sys, phi, theta, exp_defaults, opt_defaults)
    
    H0, mux, muy, muz = ham(sys)
    B_mat, I_mat, _ = resfields(H0, mux, muy, muz, phi, theta, exp_defaults, opt_defaults, sys=sys)

    assert len(B_pert) == 1
    assert torch.allclose(B_pert, B_mat, rtol=1e-9, atol=1e-11)


def test_perturb_s_half_with_14N_matches_matrix(exp_defaults, opt_defaults):
    """Perturbation theory with hyperfine matches matrix method."""
    sys = SpinSystem(S=1/2, g=2.0, Nucs="14N", A=[20, 20, 80])
    phi, theta = 0.0, 0.0

    B_pert, I_pert = resfields_perturb(sys, phi, theta, exp_defaults, opt_defaults, order=2)
    
    H0, mux, muy, muz = ham(sys)
    B_mat, I_mat, _ = resfields(H0, mux, muy, muz, phi, theta, exp_defaults, opt_defaults, sys=sys)

    # S=1/2 with I=1 (14N) gives 3 transitions
    assert len(B_pert) == 3
    assert len(B_mat) == 3

    # Sort both arrays before comparison
    B_pert_sorted, idx_p = torch.sort(B_pert)
    B_mat_sorted, idx_m = torch.sort(B_mat)
    I_pert_sorted = I_pert[idx_p]
    I_mat_sorted = I_mat[idx_m]

    # Fields should match
    assert torch.allclose(B_pert_sorted, B_mat_sorted, rtol=1e-6, atol=1e-8)
    
    # Intensities should match
    assert torch.allclose(I_pert_sorted, I_mat_sorted, rtol=1e-5)


def test_perturb_s_half_two_nuclei_matches_matrix(exp_defaults, opt_defaults):
    """Perturbation theory with multiple nuclei."""
    sys = SpinSystem(S=1/2, g=2.0, Nucs="14N,1H", A=[[20, 20, 80], [10, 10, 15]])
    phi, theta = 0.0, 0.0

    B_pert, I_pert = resfields_perturb(sys, phi, theta, exp_defaults, opt_defaults, order=2)
    
    H0, mux, muy, muz = ham(sys)
    B_mat, I_mat, _ = resfields(H0, mux, muy, muz, phi, theta, exp_defaults, opt_defaults, sys=sys)

    # S=1/2 with I1=1 (14N), I2=1/2 (1H) gives 3*2=6 transitions
    assert len(B_pert) == 6
    assert len(B_mat) == 6

    B_pert_sorted, _ = torch.sort(B_pert)
    B_mat_sorted, _ = torch.sort(B_mat)

    assert torch.allclose(B_pert_sorted, B_mat_sorted, rtol=1e-6, atol=1e-8)


def test_perturb_s1_with_zfs_matches_matrix(exp_defaults, opt_defaults):
    """Perturbation theory for S=1 with ZFS."""
    sys = SpinSystem(S=1, g=2.0, D=[-100, -100, 200])
    phi, theta = 0.0, 0.0

    B_pert, I_pert = resfields_perturb(sys, phi, theta, exp_defaults, opt_defaults, order=2)
    
    H0, mux, muy, muz = ham(sys)
    B_mat, I_mat, _ = resfields(H0, mux, muy, muz, phi, theta, exp_defaults, opt_defaults, sys=sys)

    # S=1 gives 2 transitions
    assert len(B_pert) == 2
    assert len(B_mat) == 2

    B_pert_sorted, _ = torch.sort(B_pert)
    B_mat_sorted, _ = torch.sort(B_mat)

    # With ZFS, perturbation theory is approximate - allow ~0.1 mT error
    assert torch.allclose(B_pert_sorted, B_mat_sorted, rtol=2e-3, atol=0.1)


def test_perturb_s1_rhombic_zfs_matches_matrix(exp_defaults, opt_defaults):
    """Perturbation theory for S=1 with rhombic ZFS."""
    sys = SpinSystem(S=1, g=2.0, D=[-150, -50, 200])
    phi, theta = 0.5, 1.0

    B_pert, I_pert = resfields_perturb(sys, phi, theta, exp_defaults, opt_defaults, order=2)
    
    H0, mux, muy, muz = ham(sys)
    B_mat, I_mat, _ = resfields(H0, mux, muy, muz, phi, theta, exp_defaults, opt_defaults, sys=sys)

    assert len(B_pert) == 2
    
    B_pert_sorted, _ = torch.sort(B_pert)
    B_mat_sorted, _ = torch.sort(B_mat)

    # Rhombic ZFS - perturbation theory approximate, allow ~0.1 mT error
    assert torch.allclose(B_pert_sorted, B_mat_sorted, rtol=2e-3, atol=0.1)


def test_perturb_second_order_more_accurate_than_first():
    """Second-order perturbation is more accurate than first-order."""
    # System with significant hyperfine where second-order matters
    sys = SpinSystem(S=1/2, g=2.0, Nucs="14N", A=[30, 30, 100])
    exp = Experiment(mwFreq=9.5, Range=[300, 380])
    opt = Options(Threshold=1e-6)
    phi, theta = 0.0, 0.0

    # Matrix method (reference)
    H0, mux, muy, muz = ham(sys)
    B_mat, _, _ = resfields(H0, mux, muy, muz, phi, theta, exp, opt)
    B_mat_sorted, _ = torch.sort(B_mat)

    # First-order perturbation
    B_pert1, _ = resfields_perturb(sys, phi, theta, exp, opt, order=1)
    B_pert1_sorted, _ = torch.sort(B_pert1)

    # Second-order perturbation
    B_pert2, _ = resfields_perturb(sys, phi, theta, exp, opt, order=2)
    B_pert2_sorted, _ = torch.sort(B_pert2)

    # Second-order error should be smaller than first-order error
    err1 = torch.abs(B_pert1_sorted - B_mat_sorted).max()
    err2 = torch.abs(B_pert2_sorted - B_mat_sorted).max()

    assert err2 < err1, f"Second-order error {err2} >= first-order error {err1}"
    assert err2 < 0.01, f"Second-order error {err2} too large"


def test_perturb_raises_for_two_electrons():
    """Should raise error for multi-electron systems."""
    sys = SpinSystem(S=[1/2, 1/2], g=[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
    exp = Experiment(mwFreq=9.5, Range=[320, 360])
    
    with pytest.raises(ValueError, match="requires exactly 1 electron"):
        resfields_perturb(sys, 0, 0, exp)


def test_perturb_high_spin_s3half():
    """Test S=3/2 system."""
    sys = SpinSystem(S=3/2, g=2.0, D=[-100, -100, 200])
    exp = Experiment(mwFreq=9.5, Range=[200, 500])
    opt = Options(Threshold=1e-6)
    phi, theta = 0.0, 0.0

    B_pert, I_pert = resfields_perturb(sys, phi, theta, exp, opt, order=2)
    
    # S=3/2 gives 3 transitions
    assert len(B_pert) == 3

    # Compare to matrix method
    H0, mux, muy, muz = ham(sys)
    B_mat, I_mat, _ = resfields(H0, mux, muy, muz, phi, theta, exp, opt)
    
    B_pert_sorted, _ = torch.sort(B_pert)
    B_mat_sorted, _ = torch.sort(B_mat)

    # High spin ZFS - perturbation theory less accurate, allow ~0.5 mT error
    assert torch.allclose(B_pert_sorted, B_mat_sorted, rtol=2e-3, atol=0.5)


def test_perturb_tilted_frames():
    """Test with tilted g and A frames."""
    sys = SpinSystem(
        S=1/2,
        g=[2.002, 2.005, 2.008],
        gFrame=[0.1, 0.2, 0.3],
        Nucs="14N",
        A=[20, 25, 85],
        AFrame=[0.15, 0.25, 0.35]
    )
    exp = Experiment(mwFreq=9.5, Range=[320, 360])
    opt = Options(Threshold=1e-6)
    phi, theta = 0.4, 0.9

    B_pert, I_pert = resfields_perturb(sys, phi, theta, exp, opt, order=2)
    
    H0, mux, muy, muz = ham(sys)
    B_mat, I_mat, _ = resfields(H0, mux, muy, muz, phi, theta, exp, opt)

    assert len(B_pert) == 3
    
    # Matrix method may find weak forbidden transitions with tilted frames
    # Keep only the 3 strongest transitions for comparison
    if len(B_mat) > 3:
        top_indices = torch.topk(I_mat, k=3).indices
        B_mat = B_mat[top_indices]
        I_mat = I_mat[top_indices]
    
    B_pert_sorted, _ = torch.sort(B_pert)
    B_mat_sorted, _ = torch.sort(B_mat)

    assert torch.allclose(B_pert_sorted, B_mat_sorted, rtol=1e-6, atol=1e-8)


def test_perturb_intensity_threshold():
    """Test that intensity threshold filtering works."""
    sys = SpinSystem(S=1/2, g=[2.0, 2.0, 2.3], Nucs="14N", A=[20, 20, 80])
    exp = Experiment(mwFreq=9.5, Range=[320, 360])
    
    # Low threshold - should get all transitions
    opt_low = Options(Threshold=1e-10)
    B_low, I_low = resfields_perturb(sys, 0, 0, exp, opt_low)
    
    # High threshold - should filter some out
    opt_high = Options(Threshold=1e-2)
    B_high, I_high = resfields_perturb(sys, 0, 0, exp, opt_high)
    
    # High threshold should give fewer or equal transitions
    assert len(B_high) <= len(B_low)


def test_perturb_field_range_filtering():
    """Test that field range filtering works correctly."""
    sys = SpinSystem(S=1/2, g=2.0, Nucs="14N", A=[20, 20, 80])
    
    # Wide range
    exp_wide = Experiment(mwFreq=9.5, Range=[0, 500])
    B_wide, _ = resfields_perturb(sys, 0, 0, exp_wide)
    
    # Narrow range
    exp_narrow = Experiment(mwFreq=9.5, Range=[330, 345])
    B_narrow, _ = resfields_perturb(sys, 0, 0, exp_narrow)
    
    # Narrow range should give fewer transitions
    assert len(B_narrow) <= len(B_wide)
    
    # All narrow range fields should be in wide range
    for B in B_narrow:
        assert torch.any(torch.isclose(B_wide, B, rtol=1e-10))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
