"""B79 (0.9.23): zero hyperfine principal value in second-order resfreqs_perturb/endorfrq_perturb.

The second-order hyperfine terms use inv(A).  torchspin wrapped torch.linalg.inv
in ``except Exception`` and substituted a zero matrix, so a zero principal value
gave a silently wrong second-order result (endorfrq_perturb, S=1/2, 1H
A=[5 8 0]: 15.59326 MHz vs the A_z -> 0 limit 15.59382 MHz).  EasySpin
resfreqs_perturb.m / resfields_perturb.m:120-127 raise 'All hyperfine coupling
constants must be non-zero.'; endorfrq_perturb.m has no check and returns NaN.
torchspin now raises a ValueError with EasySpin's message at second order (the
same check as resfields_perturb, B65).  First order needs no inverse and runs.
"""
import pytest
import torch

from torchspin import SpinSystem
from torchspin.experiment import Experiment
from torchspin.endorfrq_perturb import endorfrq_perturb
from torchspin.resfreqs_perturb import resfreqs_perturb
from torchspin.rotations import erot

MSG = "All hyperfine coupling constants must be non-zero"
B0 = 340.0
PHI, THETA = 0.3, 0.7
EXP = Experiment(Field=B0, mwFreq=9.7)


def _sys_diag(Az, **kw):
    return SpinSystem(S=0.5, g=[2.0, 2.05, 2.1], Nucs='1H', A=[5.0, 8.0, Az], **kw)


def _sys_full_singular():
    """Full (non-diagonal) A matrix with rank 2: rotated diag(5, 8, 0)."""
    R = torch.as_tensor(erot([0.3, 0.5, 0.2]), dtype=torch.float64)
    A_full = R.T @ torch.diag(torch.tensor([5.0, 8.0, 0.0], dtype=torch.float64)) @ R
    assert (A_full.abs() > 1e-3).all()          # no zero entry: only the rank shows it
    return SpinSystem(S=0.5, g=[2.0, 2.05, 2.1], Nucs='1H', A=A_full.tolist())


# --- second order raises -----------------------------------------------------

def test_resfreqs_perturb_zero_diagonal_A_raises():
    with pytest.raises(ValueError, match=MSG):
        resfreqs_perturb(_sys_diag(0.0), PHI, THETA, EXP, order=2)
    with pytest.raises(ValueError, match=MSG):   # default order is 2
        resfreqs_perturb(_sys_diag(0.0), PHI, THETA, EXP)


def test_endorfrq_perturb_zero_diagonal_A_raises():
    with pytest.raises(ValueError, match=MSG):
        endorfrq_perturb(_sys_diag(0.0), B0, PHI, THETA, order=2)
    with pytest.raises(ValueError, match=MSG):
        endorfrq_perturb(_sys_diag(0.0), B0, PHI, THETA)


def test_zero_A_with_frame_raises():
    s = _sys_diag(0.0, AFrame=[[0.3, 0.5, 0.2]])
    with pytest.raises(ValueError, match=MSG):
        resfreqs_perturb(s, PHI, THETA, EXP, order=2)
    with pytest.raises(ValueError, match=MSG):
        endorfrq_perturb(s, B0, PHI, THETA, order=2)


def test_singular_full_A_raises():
    s = _sys_full_singular()
    assert s.fullA
    with pytest.raises(ValueError, match=MSG):
        resfreqs_perturb(s, PHI, THETA, EXP, order=2)
    with pytest.raises(ValueError, match=MSG):
        endorfrq_perturb(s, B0, PHI, THETA, order=2)


def test_second_nucleus_named_in_error():
    s = SpinSystem(S=0.5, g=[2.0, 2.05, 2.1], Nucs='1H,13C', A=[[5.0, 8.0, 12.0], [3.0, 0.0, 6.0]])
    with pytest.raises(ValueError, match=r"nucleus 2, 13C"):
        resfreqs_perturb(s, PHI, THETA, EXP, order=2)
    with pytest.raises(ValueError, match=r"nucleus 2, 13C"):
        endorfrq_perturb(s, B0, PHI, THETA, order=2)
    # endorfrq_perturb only checks the nuclei it computes
    nu, _ = endorfrq_perturb(s, B0, PHI, THETA, order=2, nuclei=[0])
    assert torch.isfinite(nu).all()


# --- first order still runs and is continuous in A_z -> 0 -------------------

def test_first_order_runs_and_matches_limit():
    nu0, int0 = endorfrq_perturb(_sys_diag(0.0), B0, PHI, THETA, order=1)
    nu1, int1 = endorfrq_perturb(_sys_diag(1e-9), B0, PHI, THETA, order=1)
    torch.testing.assert_close(nu0, nu1, rtol=0, atol=1e-8)
    torch.testing.assert_close(int0, int1, rtol=1e-12, atol=0)

    f0, i0 = resfreqs_perturb(_sys_diag(0.0), PHI, THETA, EXP, order=1)
    f1, i1 = resfreqs_perturb(_sys_diag(1e-9), PHI, THETA, EXP, order=1)
    torch.testing.assert_close(f0, f1, rtol=0, atol=1e-11)
    torch.testing.assert_close(i0, i1, rtol=1e-12, atol=0)


def test_first_order_singular_full_A_runs():
    s = _sys_full_singular()
    nu, _ = endorfrq_perturb(s, B0, PHI, THETA, order=1)
    f, _ = resfreqs_perturb(s, PHI, THETA, EXP, order=1)
    assert torch.isfinite(nu).all() and torch.isfinite(f).all()


# --- nonzero A unchanged ------------------------------------------------------
# Reference values from main (ba7e962, 0.9.0); bit-identical there on this
# change's development machine, compared with a tight tolerance for portability.

_SYS_HN = dict(S=0.5, g=[2.0, 2.05, 2.1], Nucs='1H,14N',
               A=[[5.0, 8.0, 12.0], [20.0, 25.0, 40.0]],
               AFrame=[[0.1, 0.4, 0.2], [0, 0.3, 0]], Q=[[0, 0, 0], [-0.5, -0.3, 0.8]])
_SYS_HS = dict(S=1.5, g=[2.0, 2.0, 2.0], D=[300.0, 30.0], Nucs='14N', A=[[10.0, 12.0, 15.0]])

_REF = {
    ('HN', 'endor'): (
        [20.195277111908652, 8.854635208993852, 20.658565729543522, 17.07021411051249,
         19.193397272830765, 18.608022698458548],
        [0.0013596312495587769, 0.0013596312495587769, 9.472273364588366e-06,
         9.472273364588366e-06, 9.472273364588366e-06, 9.472273364588366e-06]),
    ('HN', 'resfreq'): (
        [9.763588040487766, 9.801373071848644, 9.839115273570071, 9.775168933460108,
         9.812953964820986, 9.850696166542413],
        [34.10284850789795] * 6),
    ('HS', 'endor'): (
        [21.045049134796304, 7.483547597762093, 5.354304976979575, 18.993303106243964,
         21.063552852816116, 7.489840189499944, 5.348500414745557, 18.975273197510695],
        [4.736136682294183e-06] * 8),
    ('HS', 'resfreq'): (
        [9.773586969657167, 9.786443489618419, 9.799287821914415, 9.49473278893382,
         9.507956964904025, 9.521168953208974, 9.258226217963172, 9.27181804994233,
         9.285397694256234],
        [195.8948717782413, 195.8948717782413, 195.8948717782413, 261.19316237098843,
         261.19316237098843, 261.19316237098843, 195.8948717782413, 195.8948717782413,
         195.8948717782413]),
}


@pytest.mark.parametrize('name, kw', [('HN', _SYS_HN), ('HS', _SYS_HS)])
def test_nonzero_A_second_order_unchanged(name, kw):
    s = SpinSystem(**kw)
    nu, it = endorfrq_perturb(s, B0, PHI, THETA, order=2)
    f, fi = resfreqs_perturb(s, PHI, THETA, EXP, order=2)
    for (val, ref) in ((nu, _REF[(name, 'endor')][0]), (it, _REF[(name, 'endor')][1]),
                       (f, _REF[(name, 'resfreq')][0]), (fi, _REF[(name, 'resfreq')][1])):
        torch.testing.assert_close(val, torch.tensor(ref, dtype=val.dtype), rtol=1e-13, atol=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
