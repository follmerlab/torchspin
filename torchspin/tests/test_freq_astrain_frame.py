"""Frequency-sweep AStrain along the AFrame axes, on the autograd graph (B90),
and g/A strain with more than one electron spin (B89), 0.9.31.

B90: before 0.9.31 :func:`compute_strain_widths_freq_batch` built the AStrain
width operators as ``float(AStrain[c]) * I_c S_c`` along the *molecular* axes:
``AFrame`` was ignored and ``float()`` cut the gradient.  The width term of
principal value k is ``AStrain_k * |dnu/dA_k|`` with
``dnu/dA_k = m((R'S)_k (R'I)_k)``, ``R = R_A2M = erot(AFrame).'``.  The
references here are objective: eigenvalue central differences in A_k (unit
test), the permuted untilted system for an axis-swap AFrame (spectrum), and
converged central differences (gradients).  EasySpin ``resfreqs_matrix.m``
rotates S with ``R(k,c)`` and I with ``R(c,k)``, which differs from dnu/dA_k
for a general tilt; torchspin deliberately does not copy that
(KNOWN_LIMITATIONS).

B89: AStrain with more than one electron spin (field and frequency sweeps) and
gStrain with more than one electron spin in a frequency sweep raise
``ValueError``, as in EasySpin (``resfields.m`` 89-92,
``resfreqs_matrix.m`` 102-106).  Before, the field sweep dropped AStrain
silently without gStrain, and the frequency sweep used the first electron's
strains.
"""
import math

import numpy as np
import pytest
import torch

from torchspin import Experiment, Options, SpinSystem, ham, pepper
from torchspin.pepper_autograd import pepper_autograd
from torchspin.rotations import erot
from torchspin.spinops import sop
from torchspin.strainwidth import compute_strain_widths_freq_batch

F64 = torch.float64
B0 = 340.0
A0 = [30.0, 60.0, 200.0]
AS0 = [10.0, 20.0, 40.0]
FR0 = [0.4, 0.9, -0.3]


def _cu(A=A0, AS=AS0, FR=FR0, g=(2.0, 2.0, 2.0)):
    return SpinSystem(S=0.5, g=list(g), Nucs='63Cu', A=A, AFrame=FR, AStrain=AS)


def _muzL(sys, ph, th):
    H0, mx, my, mz = ham(sys, B0=None)
    ph = torch.as_tensor(ph, dtype=F64).reshape(-1)
    th = torch.as_tensor(th, dtype=F64).reshape(-1)
    z = torch.stack([torch.sin(th) * torch.cos(ph), torch.sin(th) * torch.sin(ph), torch.cos(th)], 1)
    muzL = z[:, 0, None, None] * mx + z[:, 1, None, None] * my + z[:, 2, None, None] * mz
    return H0, muzL, ph, th


def _all_pairs(n):
    iu, ju = torch.triu_indices(n, n, 1)
    return torch.stack([iu, ju], 1)


def test_unit_widths_equal_eigenvalue_derivatives_generic_frame():
    """Width of every transition = sqrt(sum_k (AStrain_k dnu/dA_k)^2), dnu/dA_k
    from central differences of the eigenvalues in the principal value A_k at a
    generic AFrame.  Main before 0.9.31: 19.0 MHz instead of 56.9 MHz for the
    outermost allowed line, 6.5 instead of 18.9 MHz for the next (error 38 MHz)."""
    ph, th = 0.7, 1.1
    s = _cu()
    H0, muzL, ph_t, th_t = _muzL(s, ph, th)
    P = _all_pairs(H0.shape[0])
    w = compute_strain_widths_freq_batch(s, H0, muzL, ph_t, th_t, B0, [P])[0]

    def eig(Avec):
        H0_, muzL_, _, _ = _muzL(_cu(A=Avec), ph, th)
        return torch.linalg.eigvalsh(H0_[None] - B0 * muzL_)[0]

    h = 1e-3          # eigenvalues are smooth in A; CD error ~1e-8 MHz here
    ref2 = torch.zeros(P.shape[0], dtype=F64)
    for k in range(3):
        Ap = list(A0); Am = list(A0); Ap[k] += h; Am[k] -= h
        dE = (eig(Ap) - eig(Am)) / (2 * h)
        ref2 += (AS0[k] * (dE[P[:, 1]] - dE[P[:, 0]])) ** 2
    ref = ref2.sqrt()
    E = torch.linalg.eigvalsh(H0 - B0 * muzL[0])
    nu = E[P[:, 1]] - E[P[:, 0]]
    allowed = nu > 5000.0
    assert float(ref[allowed].max()) > 50.0          # the case is not trivial
    err = float((w - ref).abs().max())
    assert err < 1e-6, f"max |width - dnu/dA reference| = {err:.3g} MHz (widths {w[allowed].tolist()})"


def test_zero_aframe_equals_molecular_axis_formula():
    """AFrame = 0: values unchanged from the pre-0.9.31 formula
    AStrain_c * m(I_c S_c) along the molecular axes, also with AFrame on the
    graph (identity rotation computed by erot)."""
    ph, th = [0.7, 0.2, 1.3], [1.1, 0.4, 2.0]
    for rg in (False, True):
        s = _cu(FR=torch.zeros(3, dtype=F64, requires_grad=rg), g=(2.0, 2.05, 2.2))
        H0, muzL, ph_t, th_t = _muzL(s, ph, th)
        P = _all_pairs(H0.shape[0])
        w = compute_strain_widths_freq_batch(s, H0, muzL, ph_t, th_t, B0, [P] * 3)
        S = [sop(s.Spins, [[1, c]]).to(torch.complex128) for c in (1, 2, 3)]
        I = [sop(s.Spins, [[2, c]]).to(torch.complex128) for c in (1, 2, 3)]
        for m in range(3):
            _, V = torch.linalg.eigh(H0 - B0 * muzL[m])
            V = V.to(torch.complex128)
            lw2 = torch.zeros(P.shape[0], dtype=F64)
            for c in range(3):
                d = torch.real((V.conj() * ((AS0[c] * (I[c] @ S[c])) @ V)).sum(0))
                lw2 += (d[P[:, 1]] - d[P[:, 0]]) ** 2
            np.testing.assert_allclose(w[m].detach().numpy(), lw2.sqrt().detach().numpy(), rtol=1e-12, atol=1e-12)


def test_freq_sweep_axis_swap_frame_equals_permuted_system():
    """pepper frequency sweep: AFrame [0 pi/2 0] (A-frame z along molecular x)
    equals the untilted system with A and AStrain permuted accordingly.
    Main before 0.9.31: relative difference 0.31 (AStrain left on the molecular axes);
    fixed: 1.4e-13."""
    A = torch.tensor([30.0, 60.0, 450.0], dtype=F64)
    AS = torch.tensor([10.0, 20.0, 80.0], dtype=F64)
    FR = [0.0, math.pi / 2, 0.0]
    R = erot(torch.tensor(FR, dtype=F64)).T
    A_mol = torch.diagonal(R @ torch.diag(A) @ R.T)
    AS_mol = torch.diagonal(R @ torch.diag(AS) @ R.T)
    assert torch.allclose(A_mol, A.flip(0), atol=1e-9)
    exp = Experiment(Field=340.0, mwRange=[8.8, 10.4], nPoints=1024, Harmonic=0)
    opt = Options(GridSize=[19, 4], GridSymmetry='Ci')
    tilt = pepper(SpinSystem(S=0.5, g=[2.0] * 3, Nucs='63Cu', A=A.tolist(), AFrame=FR,
                             AStrain=AS.tolist()), exp, opt)[1].detach().numpy()
    flat = pepper(SpinSystem(S=0.5, g=[2.0] * 3, Nucs='63Cu', A=A_mol.tolist(),
                             AStrain=AS_mol.tolist()), exp, opt)[1].detach().numpy()
    rel = float(np.abs(tilt - flat).max() / np.abs(flat).max())
    assert rel < 1e-10, f"tilted vs permuted system: relative difference {rel:.3g}"


_PH, _TH = [0.7, 0.2, 1.3], [1.1, 0.4, 2.0]


def _weighted_width_sum(AS, FR):
    """sum_t t * width_t over 3 orientations, all 28 transitions (torchspin)."""
    s = _cu(AS=AS, FR=FR, g=(2.0, 2.05, 2.2))
    H0, muzL, ph_t, th_t = _muzL(s, _PH, _TH)
    P = _all_pairs(H0.shape[0])
    w = compute_strain_widths_freq_batch(s, H0, muzL, ph_t, th_t, B0, [P] * 3)
    return sum((torch.arange(1, x.numel() + 1, dtype=F64) * x).sum() for x in w)


def _weighted_width_sum_ref(AS, FR):
    """Independent reference of the same quantity (plain floats): Hellmann-Feynman
    dnu/dA_k = m(dH/dA_k), dH/dA_k = sum_ab R[a,k] R[b,k] S_a I_b, R = erot(AFrame).'"""
    s = _cu(AS=list(AS), FR=list(FR), g=(2.0, 2.05, 2.2))
    H0, muzL, _, _ = _muzL(s, _PH, _TH)
    P = _all_pairs(H0.shape[0])
    R = erot(torch.tensor(FR, dtype=F64)).T
    S = [sop(s.Spins, [[1, c]]).to(torch.complex128) for c in (1, 2, 3)]
    I = [sop(s.Spins, [[2, c]]).to(torch.complex128) for c in (1, 2, 3)]
    dH = [sum(float(R[a, k] * R[b, k]) * (S[a] @ I[b]) for a in range(3) for b in range(3))
          for k in range(3)]
    total = 0.0
    for m in range(3):
        _, V = torch.linalg.eigh(H0 - B0 * muzL[m])
        V = V.to(torch.complex128)
        lw2 = torch.zeros(P.shape[0], dtype=F64)
        for k in range(3):
            d = torch.real((V.conj() * (dH[k] @ V)).sum(0))
            lw2 += (float(AS[k]) * (d[P[:, 1]] - d[P[:, 0]])) ** 2
        total += float((torch.arange(1, P.shape[0] + 1, dtype=F64) * lw2.sqrt()).sum())
    return total


@pytest.mark.parametrize("which", ["AStrain", "AFrame"])
def test_gradients_match_converged_central_differences(which):
    """Autograd d/dAStrain and d/dAFrame of a weighted sum of torchspin widths
    (3 orientations, all 28 transitions) against central differences of the
    independent Hellmann-Feynman reference.  Relative error of the CD gradient
    vs autograd for steps h (torch 2.15; older torch alike):

        h        1e-1     1e-2     1e-3     1e-4     1e-5     1e-6
        AStrain  2.2e-7   2.2e-9   2.6e-11  2.4e-11  5.7e-10  3.5e-9
        AFrame   4.9e-2   5.0e-4   5.0e-6   5.0e-8   6.9e-10  3.1e-9

    so h = 1e-3 (AStrain) and h = 1e-5 (AFrame) are in the converged range.
    Main before 0.9.31: no gradient for AStrain (float()), and an AFrame
    gradient of a different function (strain operators on the molecular axes)."""
    AS = torch.tensor(AS0, dtype=F64, requires_grad=True)
    FR = torch.tensor(FR0, dtype=F64, requires_grad=True)
    y = _weighted_width_sum(AS, FR)
    y_ref = _weighted_width_sum_ref(AS0, FR0)
    assert abs(float(y.detach()) - y_ref) <= 1e-9 * abs(y_ref), (float(y.detach()), y_ref)
    g = torch.autograd.grad(y, [AS if which == "AStrain" else FR], allow_unused=True)[0]
    assert g is not None, f"no gradient for {which}"
    h, tol = (1e-3, 1e-8) if which == "AStrain" else (1e-5, 1e-7)
    x0 = AS0 if which == "AStrain" else FR0
    cd = []
    for k in range(3):
        p = list(x0); m = list(x0); p[k] += h; m[k] -= h
        if which == "AStrain":
            fp, fm = _weighted_width_sum_ref(p, FR0), _weighted_width_sum_ref(m, FR0)
        else:
            fp, fm = _weighted_width_sum_ref(AS0, p), _weighted_width_sum_ref(AS0, m)
        cd.append((fp - fm) / (2 * h))
    cd = torch.tensor(cd, dtype=F64)
    rel = float((cd - g).norm() / g.norm())
    assert rel < tol, f"{which}: autograd {g.tolist()} vs CD {cd.tolist()} (rel {rel:.3g})"


def test_zero_width_transition_has_finite_gradient():
    """A transition with exactly zero width (AStrain only along an axis whose
    operator vanishes on it) must not give NaN gradients (safe sqrt)."""
    AS = torch.tensor([0.0, 0.0, 40.0], dtype=F64, requires_grad=True)
    s = SpinSystem(S=0.5, g=[2.0] * 3, Nucs='1H', A=[0.0, 0.0, 0.0], AStrain=AS)
    H0, muzL, ph_t, th_t = _muzL(s, 0.0, 0.0)
    P = _all_pairs(H0.shape[0])
    w = compute_strain_widths_freq_batch(s, H0, muzL, ph_t, th_t, B0, [P])[0]
    assert bool((w.detach() == 0).any())
    g, = torch.autograd.grad(w.sum(), AS)
    assert bool(torch.isfinite(g).all()), g


# --- B89: g/A strain with more than one electron spin -----------------------

def _two_e(**kw):
    base = dict(S=[0.5, 0.5], g=[[2.0, 2.0, 2.0], [2.1, 2.1, 2.1]], ee=[300.0, 300.0, 300.0],
                Nucs='1H', A=[[20.0, 20.0, 80.0, 0, 0, 0]], HStrain=[10, 10, 10])
    base.update(kw)
    return SpinSystem(**base)


_EXP_B = dict(mwFreq=9.5, Range=[310, 350], nPoints=256)
_EXP_B_EMPTY = dict(mwFreq=9.5, Range=[10, 20], nPoints=256)        # no resonance in range
_EXP_F = dict(Field=330.0, mwRange=[9.0, 9.9], nPoints=256, Harmonic=0)
_A = dict(AStrain=[0, 0, 60.0])
_G = dict(gStrain=[[0.01] * 3, [0.0] * 3])


@pytest.mark.parametrize("fn,exp,kw,label", [
    (pepper, _EXP_B, _A, "field sweep, AStrain without gStrain (was silently dropped)"),
    (pepper, _EXP_B, {**_A, **_G}, "field sweep, gStrain + AStrain"),
    (pepper, _EXP_B_EMPTY, _A, "field sweep without resonances, AStrain"),
    (pepper_autograd, _EXP_B, _A, "pepper_autograd, AStrain without gStrain"),
    (pepper, _EXP_F, _A, "frequency sweep, AStrain (was applied to electron 1)"),
    (pepper, _EXP_F, _G, "frequency sweep, gStrain (was applied to electron 1)"),
    (pepper, _EXP_F, {**_A, **_G}, "frequency sweep, gStrain + AStrain"),
])
def test_multi_electron_strain_raises(fn, exp, kw, label):
    with pytest.raises(ValueError, match="more than one electron"):
        fn(_two_e(**kw), Experiment(**exp), Options(GridSize=[7, 1]))


def test_multi_electron_strain_raises_in_width_function():
    s = _two_e(**_A)
    H0, muzL, ph_t, th_t = _muzL(s, 0.3, 0.5)
    with pytest.raises(ValueError, match="more than one electron"):
        compute_strain_widths_freq_batch(s, H0, muzL, ph_t, th_t, B0, [_all_pairs(H0.shape[0])])


def test_multi_electron_gstrain_field_sweep_still_runs():
    """The field sweep keeps its per-electron g strain model (no AStrain)."""
    x, y = pepper(_two_e(**_G), Experiment(**_EXP_B), Options(GridSize=[7, 1]))
    assert np.isfinite(y.detach().numpy()).all() and float(np.abs(y.detach().numpy()).max()) > 0
