"""Zero hyperfine principal values in second-order perturbation theory: the exact A -> 0 limit (0.9.29).

The second-order hyperfine invariant of Iwasaki (1974) is A2 = det(A) u'·inv(A)·k.  Only
det(A) inv(A) = adj(A) enters, a polynomial in the entries of A, so resfields_perturb(_batch),
resfreqs_perturb and endorfrq_perturb now use the adjugate (torchspin._adjugate._adjugate3) and
are defined, continuous and differentiable at a zero principal value or a singular full A.
Before, resfields_perturb raised torch's LinAlgError from inv(A) (B65), and resfreqs_perturb and
endorfrq_perturb raised ValueError (0.9.23, B79).  EasySpin resfields_perturb.m:120-127 and
resfreqs_perturb.m raise 'All hyperfine coupling constants must be non-zero.';
endorfrq_perturb.m has no check and returns NaN.

In the EPR modules k = A u/|A u|, so A2 = det(A)/|A u| and vanishes for a singular A; it enters
the fields with the factor (1 - 2 mS), i.e. only for S > 1/2 (the tests use S = 1 there).  In
endorfrq_perturb k is the direction of the field at the nucleus (hyperfine + nuclear Zeeman) and
A2 is non-zero at S = 1/2.
"""
import math

import pytest
import torch

from torchspin import SpinSystem, Experiment, ham
from torchspin.experiment import Options
from torchspin.constants import PLANCK, BMAGN
from torchspin.rotations import erot
from torchspin._adjugate import _adjugate3
from torchspin.endorfrq_perturb import endorfrq_perturb
from torchspin.resfreqs_perturb import resfreqs_perturb
from torchspin.resfields_perturb import resfields_perturb, resfields_perturb_batch
from torchspin.resfields import resfields
from torchspin.resfreqs_matrix import resfreqs_matrix
from torchspin.endorfrq import endorfrq

DT = torch.float64
G = [2.0, 2.05, 2.1]
B0 = 340.0                        # mT (frequency sweeps, ENDOR)
MW = 9.7                          # GHz (field sweeps)
E0 = MW * 1e3                     # MHz
PHI, THETA = 0.3, 0.7
EXP_B = Experiment(mwFreq=MW, Range=[250, 400])
EXP_F = Experiment(Field=B0, mwFreq=MW)
PRINCIPAL = {'1H': (5.0, 8.0), '14N': (20.0, 15.0), '63Cu': (50.0, 30.0)}   # MHz
AFRAME = [0.4, 0.2, 0.1]
R_FULL = torch.as_tensor(erot([0.3, 0.5, 0.2]), dtype=DT)
FUNCS = ('resfields_perturb', 'resfields_perturb_batch', 'resfreqs_perturb', 'endorfrq_perturb')
SHAPES = ('one_zero_noframe', 'one_zero', 'two_zeros', 'full_rank2')


def _system(nuc, shape, z, S, **kw):
    """S = 1/2 or 1 (D = [60 6] MHz); A with the principal value(s) z (0 or eps)."""
    a, b = PRINCIPAL[nuc]
    if S == 1:
        kw.update(D=[60.0, 6.0])
    if shape == 'one_zero_noframe':      # exactly singular: main's inv(A) raised LinAlgError (B65)
        return SpinSystem(S=S, g=G, Nucs=nuc, A=[a, b, z], **kw)
    if shape == 'one_zero':              # with a frame, R diag(a, b, 0) R' is singular to roundoff
        return SpinSystem(S=S, g=G, Nucs=nuc, A=[a, b, z], AFrame=AFRAME, **kw)
    if shape == 'two_zeros':
        return SpinSystem(S=S, g=G, Nucs=nuc, A=[a, z, z], AFrame=AFRAME, **kw)
    A = R_FULL.T @ torch.diag(torch.tensor([a, b, z], dtype=DT)) @ R_FULL
    if z == 0.0:
        assert (A.abs() > 1e-3).all()     # no zero entry: only the rank shows it
    s = SpinSystem(S=S, g=G, Nucs=nuc, A=A.tolist(), **kw)
    assert s.fullA
    return s


def _run(fn, s):
    if fn == 'resfields_perturb':
        return resfields_perturb(s, PHI, THETA, EXP_B, order=2)[0]               # mT
    if fn == 'resfields_perturb_batch':
        B, _ = resfields_perturb_batch(s, torch.tensor([PHI, 1.1], dtype=DT),
                                       torch.tensor([THETA, 2.0], dtype=DT), EXP_B, Options(),
                                       order=2, return_full=True)
        return torch.cat(B)                                                       # mT
    if fn == 'resfreqs_perturb':
        return resfreqs_perturb(s, PHI, THETA, EXP_F, order=2)[0]                 # GHz
    return endorfrq_perturb(s, B0, PHI, THETA, order=2)[0]                        # MHz


# Slope bound C of |f(0) - f(eps)| <= C eps, in output units per MHz of eps.  Measured maxima
# over 1H/14N/63Cu x the three shapes x S = 1/2, 1 (eps = 1e-6 and 1e-8 agree to a few %):
# fields 4.6e-4 mT/MHz (63Cu full, S = 1; batch identical), resfreqs 1.3e-5 GHz/MHz (same),
# endorfrq 0.98 MHz/MHz (1H full, S = 1).  S = 1/2 EPR and all 'two_zeros' EPR systems: 0 to
# roundoff (A2 enters with 1 - 2 mS = 0, and adj(A) = 0 for rank 1; |A u| is O(eps^2)).
SLOPE = {'resfields_perturb': 1e-3, 'resfields_perturb_batch': 1e-3,
         'resfreqs_perturb': 3e-5, 'endorfrq_perturb': 2.0}


@pytest.mark.parametrize('S', [0.5, 1])
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('nuc', list(PRINCIPAL))
@pytest.mark.parametrize('fn', FUNCS)
def test_zero_principal_value_is_continuous_limit(fn, nuc, shape, S):
    f0 = _run(fn, _system(nuc, shape, 0.0, S))
    assert f0.numel() > 0 and torch.isfinite(f0).all()
    floor = 64 * torch.finfo(DT).eps * float(f0.abs().max())     # roundoff of the output
    d = {}
    for eps in (1e-6, 1e-8):
        fe = _run(fn, _system(nuc, shape, eps, S))
        assert fe.shape == f0.shape
        d[eps] = float((f0 - fe).abs().max())
        assert d[eps] <= SLOPE[fn] * eps + floor, (eps, d[eps])
    # O(eps): a 100x smaller eps gives a ~100x smaller change (or both at the roundoff floor)
    assert d[1e-8] <= d[1e-6] / 50 + floor


def test_endorfrq_b79_example_is_the_limit():
    """B79: S = 1/2, 1H A = [5 8 0] MHz.  Main's zero matrix for inv(A) gave 15.59326 MHz for
    the first line; 0.9.23 raised; the A_z -> 0 limit is 15.593819600 MHz."""
    s0 = SpinSystem(S=0.5, g=G, Nucs='1H', A=[5.0, 8.0, 0.0])
    s9 = SpinSystem(S=0.5, g=G, Nucs='1H', A=[5.0, 8.0, 1e-9])
    nu0, int0 = endorfrq_perturb(s0, B0, PHI, THETA, order=2)
    nu9, int9 = endorfrq_perturb(s9, B0, PHI, THETA, order=2)
    torch.testing.assert_close(nu0, nu9, rtol=0, atol=1e-9)
    torch.testing.assert_close(int0, int9, rtol=1e-12, atol=0)
    assert abs(float(nu0[0]) - 15.593819600) < 1e-8
    assert abs(float(nu0[0]) - 15.59326) > 5e-4


# --- sanity: matrix diagonalization at A -> 0 ---------------------------------------
# Neglected third-order terms are of order Amax^3/E0^2 (Amax: the largest principal value,
# E0 = 9700 MHz).  Measured |perturbation - matrix| / (Amax^3/E0^2), S = 1/2: endorfrq <= 1.55
# (63Cu full), resfields (fields, as MHz) <= 0.48, resfreqs <= 0.59; bound 3.  EasySpin's
# perturbation EPR has no nuclear Zeeman term, so the EPR matrix references use gnscale = 0;
# endorfrq_perturb includes it.  S = 1 is left to the continuity test: its D/ZFS terms and
# the S = 1 EPR perturbation error (~1e-4 relative, module docstring) mask the A -> 0 term.
# Main's zero-matrix ENDOR result was off by 5.6e-4 MHz for 1H (bound 1.6e-5 MHz).

@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('nuc', list(PRINCIPAL))
def test_matrix_agrees_at_zero(nuc, shape):
    c3 = max(PRINCIPAL[nuc]) ** 3 / E0 ** 2                    # MHz
    s = _system(nuc, shape, 0.0, 0.5)
    nu_p = torch.sort(_run('endorfrq_perturb', s)).values
    nu_m, _, _ = endorfrq(s, B0, PHI, THETA)
    nu_m = torch.sort(nu_m).values
    assert nu_m.shape == nu_p.shape
    assert float((nu_p - nu_m).abs().max()) <= 3 * c3

    s0 = _system(nuc, shape, 0.0, 0.5, gnscale=[0.0])
    B_p = torch.sort(_run('resfields_perturb', s0)).values
    H0, mux, muy, muz = ham(s0)
    B_m, _, _ = resfields(H0, mux, muy, muz, PHI, THETA, EXP_B, Options(Threshold=1e-2), sys=s0)
    B_m = torch.sort(B_m).values
    assert B_m.shape == B_p.shape
    mT_per_MHz = 1e9 * PLANCK / (min(G) * BMAGN)
    assert float((B_p - B_m).abs().max()) <= 3 * c3 * mT_per_MHz

    f_p = torch.sort(_run('resfreqs_perturb', s0)).values * 1e3    # MHz
    f_m, _ = resfreqs_matrix(s0, PHI, THETA, B0, threshold=1e-2)
    f_m = torch.sort(f_m).values
    assert f_m.shape == f_p.shape
    assert float((f_p - f_m).abs().max()) <= 3 * c3


# --- gradient at a zero principal value ------------------------------------------------
# resfields_perturb(_batch) are differentiable (resfreqs_perturb and endorfrq_perturb go
# through float()).  S = 1 so that A2 enters.  Central differences with h = 1e-3 MHz: measured
# |autograd - FD| <= 2.7e-10 (h = 1e-2 and 1e-4 agree to 1e-8 and 3e-9: converged).

def _weighted_fields(fn, A, full, frame=True):
    kw = dict(A=A) if full or not frame else dict(A=A, AFrame=AFRAME)
    s = SpinSystem(S=1, g=G, D=[60.0, 6.0], Nucs='63Cu', **kw)
    B = _run(fn, s)
    # Non-linear weights: a linear weight cancels the A2 term, odd in mI and in 1 - 2 mS.
    w = torch.cos(1.7 * torch.arange(B.numel(), dtype=DT) ** 1.3)
    return (w * B).sum()


@pytest.mark.parametrize('fn', ['resfields_perturb', 'resfields_perturb_batch'])
@pytest.mark.parametrize('shape', SHAPES)
def test_gradient_at_zero_matches_central_differences(fn, shape):
    full = shape == 'full_rank2'
    if full:
        A0 = R_FULL.T @ torch.diag(torch.tensor([50.0, 30.0, 0.0], dtype=DT)) @ R_FULL
    elif shape in ('one_zero', 'one_zero_noframe'):
        A0 = torch.tensor([[50.0, 30.0, 0.0]], dtype=DT)
    else:
        A0 = torch.tensor([[50.0, 0.0, 0.0]], dtype=DT)
    frame = shape != 'one_zero_noframe'
    A = A0.clone().requires_grad_(True)
    grad, = torch.autograd.grad(_weighted_fields(fn, A, full, frame), A)
    assert torch.isfinite(grad).all()
    h = 1e-3
    for i in range(A0.numel()):
        Ap, Am = A0.clone(), A0.clone()
        Ap.view(-1)[i] += h
        Am.view(-1)[i] -= h
        fd = float(_weighted_fields(fn, Ap, full, frame)
                   - _weighted_fields(fn, Am, full, frame)) / (2 * h)
        ad = float(grad.view(-1)[i])
        assert abs(ad - fd) <= 2e-9 + 1e-7 * abs(ad), (i, ad, fd)
    if shape in ('one_zero', 'one_zero_noframe'):
        assert abs(float(grad[0, 2])) > 1e-6     # the zero principal value does enter


# --- the helper ---------------------------------------------------------------------------

def test_adjugate3_matches_det_inv_and_is_defined_when_singular():
    torch.manual_seed(0)
    M = torch.randn(64, 3, 3, dtype=DT) * 30
    adj = _adjugate3(M)
    ref = torch.linalg.det(M)[:, None, None] * torch.linalg.inv(M)
    assert float(((adj - ref).abs().amax((1, 2)) / ref.abs().amax((1, 2))).max()) < 1e-13
    I = torch.eye(3, dtype=DT)
    for d in ([5.0, 8.0, 0.0], [50.0, 0.0, 0.0], [0.0, 0.0, 0.0]):
        S = R_FULL.T @ torch.diag(torch.tensor(d, dtype=DT)) @ R_FULL
        a = _adjugate3(S)
        scale = max(d) ** 2 + 1.0
        torch.testing.assert_close(a @ S, torch.zeros(3, 3, dtype=DT), rtol=0, atol=1e-12 * scale)
        torch.testing.assert_close(S @ a, torch.zeros(3, 3, dtype=DT), rtol=0, atol=1e-12 * scale)
        # rank 2: adj = (product of the non-zero values) n n', n the null vector
        if d[2] == 0 and d[1] != 0:
            n = R_FULL.T @ torch.tensor([0.0, 0.0, 1.0], dtype=DT)
            torch.testing.assert_close(a, d[0] * d[1] * torch.outer(n, n), rtol=0, atol=1e-12)
    assert torch.equal(_adjugate3(I), I)
    X = torch.randn(3, 3, dtype=DT, requires_grad=True)
    assert torch.autograd.gradcheck(_adjugate3, (X,))
    with pytest.raises(ValueError):
        _adjugate3(torch.zeros(2, 2, dtype=DT))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
