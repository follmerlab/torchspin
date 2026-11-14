"""Rotation covariance of the field-sweep gStrain/AStrain widths (B84, 0.9.33).

The g-strain model is a Gaussian distribution of the principal g values in the
g frame: for a field direction z (molecular frame) the width^2 is
``sum_i (R z)_i^2 (gStrain_i/g_i * mwFreq)^2 = z' G G z = |G z|^2`` with
``G = R' diag(gStrain/g * mwFreq) R`` (``R = erot(gFrame)``), and likewise
``|(G + c*mI*A) z|^2`` with AStrain.  EasySpin (``resfields.m`` 787/792/1309,
``resfields_perturb.m`` 523/532) and torchspin before 0.9.33 used the
element-wise square ``z' (G.^2) z``, which equals the model only when G is
diagonal (gFrame = 0).  With a tilted gFrame the widths then depended on how
the same physical system was described: per-orientation widths differed by up
to ~100 % between ``gFrame = [0.4 0.9 -0.3]`` and the equivalent untilted
system, powder spectra by 9-27 % of the peak.

The covariance and hamsymm tests compare the tilted description with the
equivalent untilted one (field direction rotated accordingly), so the
reference is exact; all fail before the fix.  The last test checks the
differentiable path against converged central differences.
"""
from __future__ import annotations

import math

import pytest
import torch

from torchspin.experiment import Experiment, Options
from torchspin.ham import ham
from torchspin.hamsymm import hamsymm
from torchspin.pepper import pepper
from torchspin.pepper_autograd import pepper_autograd
from torchspin.rotations import erot
from torchspin.spinsystem import SpinSystem
from torchspin.strainwidth import compute_strain_widths

F64 = torch.float64
FRAME = [0.4, 0.9, -0.3]
MWFREQ = 9.5


def _angles(z):
    z = z / z.norm()
    return math.atan2(float(z[1]), float(z[0])), math.acos(max(-1.0, min(1.0, float(z[2]))))


def _directions(n=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    d = torch.randn(n, 3, dtype=F64, generator=g)
    return d / d.norm(dim=1, keepdim=True)


def _widths(kw, z, pairs, B0, Range=None):
    sys = SpinSystem(**kw)
    H0, mux, muy, muz = ham(sys)
    phi, theta = _angles(z)
    return compute_strain_widths(sys, H0, mux, muy, muz, phi, theta, torch.tensor(pairs),
                                 MWFREQ, B0, Range=Range)


CASES = {
    'axial': (dict(S=0.5, g=[2.0, 2.0, 2.2], gStrain=[0.01, 0.01, 0.05]),
              dict(gFrame=FRAME), [[0, 1]], 320.0, None),
    'rhombic': (dict(S=0.5, g=[2.0, 2.1, 2.3], gStrain=[0.01, 0.03, 0.06]),
                dict(gFrame=FRAME), [[0, 1]], 310.0, None),
    # both g frames tilted identically; isotropic exchange is rotation invariant
    'two_electron': (dict(S=[0.5, 0.5], g=[[2.0, 2.0, 2.2], [1.98, 2.05, 2.1]], ee=10.0,
                          gStrain=[[0.01, 0.01, 0.04], [0.02, 0.01, 0.015]]),
                     dict(gFrame=[FRAME, FRAME]), [[0, 1], [0, 2], [1, 3], [2, 3]], 320.0, None),
    # isotropic g and A: the AStrain mI labels (<Iz> along the molecular z axis)
    # do not depend on the frame.  With anisotropic g or A they do (B96, not
    # covariant: 6e-4 / 3e-3 relative here with g = [2 2 2.2]), separately from B84.
    'g_A_strain_isoA': (dict(S=0.5, g=[2.0, 2.0, 2.0], Nucs='63Cu', A=[400.0, 400.0, 400.0],
                             gStrain=[0.01, 0.01, 0.04], AStrain=[10.0, 10.0, 40.0], gAStrainCorr=1.0),
                        dict(gFrame=FRAME, AFrame=[FRAME]),
                        [[0, 7], [1, 6], [2, 5], [3, 4]], 320.0, [270.0, 370.0]),
    'A_strain_isoA': (dict(S=0.5, g=[2.0, 2.0, 2.0], Nucs='63Cu', A=[400.0, 400.0, 400.0],
                           AStrain=[10.0, 10.0, 40.0]),
                      dict(AFrame=[FRAME]),
                      [[0, 7], [1, 6], [2, 5], [3, 4]], 320.0, [270.0, 370.0]),
}


@pytest.mark.parametrize("name", list(CASES))
def test_per_orientation_widths_rotation_covariant(name):
    """Tilted system at z_M == untilted system at n = R z_M, transition by transition.

    Max relative difference over 12 directions, before -> after the fix:
    axial 1.04 -> 1.1e-15, rhombic 1.08 -> 5.9e-16, two-electron 0.82 ->
    7.0e-15, g+A strain 0.92 -> 1.6e-15, AStrain alone 0.92 -> 2.0e-15.
    """
    base, tilt, pairs, B0, rng = CASES[name]
    R = erot(FRAME).to(F64)                       # molecular -> g frame
    for n in _directions():
        w0 = _widths(base, n, pairs, B0, rng)                   # untilted, g-frame direction
        w1 = _widths({**base, **tilt}, R.T @ n, pairs, B0, rng)  # tilted, molecular direction
        assert torch.all(w0 > 0)
        torch.testing.assert_close(w1, w0, rtol=1e-12, atol=0.0)


def _rel(a, b):
    return float((a - b).abs().max() / b.abs().max())


@pytest.mark.parametrize("kw", [
    dict(S=0.5, g=[2.0, 2.0, 2.2], gStrain=[0.01, 0.01, 0.05]),
    dict(S=0.5, g=[2.0, 2.1, 2.3], gStrain=[0.01, 0.03, 0.06]),
], ids=['axial', 'rhombic'])
def test_powder_rotation_covariant(kw):
    """pepper: tilted and untilted descriptions give the same powder spectrum.

    With the fix the strain symmetry follows the g frame (Dinfh/D2h there), so
    both use the same physical grid.  Before the fix the element-wise square
    had a lower symmetry (Ci) and different widths: 0.27 (axial) / 0.092
    (rhombic) of the peak on this grid.
    """
    exp = Experiment(mwFreq=MWFREQ, Range=[280.0, 360.0], nPoints=1024, Harmonic=0)
    opt = Options(GridSize=[19, 4])
    y0 = pepper(SpinSystem(**kw), exp, opt)[1]
    y1 = pepper(SpinSystem(**kw, gFrame=FRAME), exp, opt)[1]
    assert _rel(y1, y0) < 1e-10


@pytest.mark.parametrize("kw,tilt,expected", [
    (dict(S=0.5, g=[2.0, 2.0, 2.2], gStrain=[0.01, 0.01, 0.05]), dict(gFrame=FRAME), 'Dinfh'),
    (dict(S=0.5, g=[2.0, 2.1, 2.3], gStrain=[0.01, 0.03, 0.06]), dict(gFrame=FRAME), 'D2h'),
    (dict(S=0.5, g=[2.0, 2.0, 2.2], Nucs='63Cu', A=[50.0, 50.0, 500.0], gStrain=[0.01, 0.01, 0.04],
          AStrain=[10.0, 10.0, 40.0]), dict(gFrame=FRAME, AFrame=[FRAME]), 'Dinfh'),
    (dict(S=0.5, g=[2.0, 2.0, 2.2], Nucs='63Cu', A=[50.0, 50.0, 500.0], AStrain=[10.0, 10.0, 40.0]),
     dict(gFrame=FRAME, AFrame=[FRAME]), 'Dinfh'),
], ids=['gStrain_axial', 'gStrain_rhombic', 'g_A_strain', 'A_strain'])
def test_hamsymm_tilted_collinear_strain(kw, tilt, expected):
    """Strain tensors collinear with g (B82 strain symmetry): the symmetry of
    the untilted system, now in the tilted frame.  Before the fix the
    element-wise squares had lower symmetry (Ci/C2h)."""
    assert hamsymm(SpinSystem(**kw))[0] == expected
    assert hamsymm(SpinSystem(**kw, **tilt))[0] == expected


def test_autograd_widths_rotation_covariant():
    """pepper_autograd at user orientations: tilted (z_M) and untilted (R z_M)
    give the same spectrum and the same d/dgStrain (before the fix 0.38 /
    1.26 off relative to the maximum; after 1.2e-13 / 3.1e-13)."""
    kw = dict(S=0.5, g=[2.0, 2.1, 2.3])
    exp_kw = dict(mwFreq=MWFREQ, Range=[280.0, 360.0], nPoints=801, Harmonic=1)
    R = erot(FRAME).to(F64)
    n = _directions(8, seed=1)
    zM = n @ R                                     # rows R.T @ n
    ori0 = [list(_angles(v)) for v in n]
    ori1 = [list(_angles(v)) for v in zM]
    r = torch.randn(801, dtype=F64, generator=torch.Generator().manual_seed(0))
    out = []
    for frame, ori in ((None, ori0), (FRAME, ori1)):
        s = torch.tensor([0.01, 0.03, 0.06], dtype=F64, requires_grad=True)
        extra = {} if frame is None else dict(gFrame=frame)
        y = pepper_autograd(SpinSystem(**kw, gStrain=s, **extra), Experiment(**exp_kw, Orientations=ori),
                            Options())[1]
        (y @ r).backward()
        out.append((y.detach(), s.grad))
    assert _rel(out[1][0], out[0][0]) < 1e-10
    assert _rel(out[1][1], out[0][1]) < 1e-10


def test_gradients_vs_central_differences():
    """d/dgStrain and d/dgFrame of a tilted rhombic powder (fixed Ci grid, as the
    autograd grid with a requires-grad gFrame) against converged central
    differences (h = 1e-6).  Guards the differentiable matrix-square path; the
    rotation covariance of the gradient is tested above (this check alone
    passes before the fix too: there autograd matched the old function).

    max|autograd - CD| / max|CD| (OMP_NUM_THREADS=4, torch 2.10):
      h      gStrain   gFrame
      1e-3   1.2e-2    1.2e-2
      1e-4   1.2e-4    3.2e-4
      1e-5   1.2e-6    3.1e-6
      1e-6   1.2e-8    1.2e-6
      1e-7   2.6e-9    9.6e-6   (round-off)
    """
    exp = Experiment(mwFreq=MWFREQ, Range=[280.0, 360.0], nPoints=801, Harmonic=1)
    opt = Options(GridSize=[19, 4], GridSymmetry='Ci')
    r = torch.randn(801, dtype=F64, generator=torch.Generator().manual_seed(0))
    s0 = torch.tensor([0.01, 0.03, 0.06], dtype=F64)
    f0 = torch.tensor(FRAME, dtype=F64)

    def loss(s, fr):
        return pepper_autograd(SpinSystem(S=0.5, g=[2.0, 2.1, 2.3], gStrain=s, gFrame=fr), exp, opt)[1] @ r

    s = s0.clone().requires_grad_()
    fr = f0.clone().requires_grad_()
    loss(s, fr).backward()
    h = 1e-6
    for x0, grad, which in ((s0, s.grad, 0), (f0, fr.grad, 1)):
        fd = torch.empty(3, dtype=F64)
        for i in range(3):
            e = torch.zeros(3, dtype=F64)
            e[i] = h
            args_p = (x0 + e, f0) if which == 0 else (s0, x0 + e)
            args_m = (x0 - e, f0) if which == 0 else (s0, x0 - e)
            fd[i] = (loss(*args_p) - loss(*args_m)) / (2 * h)
        assert float((grad - fd).abs().max() / fd.abs().max()) < 1e-5
