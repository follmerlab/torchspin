"""Single-orientation ``resfields`` and ``compute_strain_widths`` (B76, B5; 0.9.20).

Both are thin wrappers over the batched solvers (``resfields_batch``,
``compute_strain_widths_batch``) with one orientation.  Before 0.9.20 they had
their own per-orientation code (Python-float Brent search, ``.item()``):

* the outputs carried no gradient, or, for S = 1 with DStrain + HStrain, a
  gradient through the eigenvectors only that had the wrong sign
  (dW/dg0 = +0.771 against -0.807 from finite differences);
* a two-point bracket found at most one resonance per level pair, so a
  looping transition lost its second line;
* full g + gStrain read ``g[0, :]`` (the first row of the matrix) and gave an
  infinite width instead of EasySpin's error (B5).

Gradients are checked against a 4-point central difference
(-f(x+2h) + 8 f(x+h) - 8 f(x-h) + f(x-2h)) / 12h with steps for which the
difference is converged (truncation O(h^4) and round-off both below 1e-9
relative).  Frame-angle gradients need #64 and are not tested here.
"""
from __future__ import annotations

import pytest
import torch

from torchspin import SpinSystem
from torchspin.experiment import Experiment, Options
from torchspin.ham import ham
from torchspin.resfields import resfields
from torchspin.resfields_batch import resfields_batch
from torchspin.strainwidth import compute_strain_widths, compute_strain_widths_batch

DT = torch.float64
PHI, THETA = 0.4, 0.9
RTOL = 1e-6
STRAIN_KEYS = ('gStrain', 'HStrain', 'AStrain')


def T(*a):
    return torch.tensor(a, dtype=DT)


# name -> (builder(params) -> SpinSystem, default params, field range, params to differentiate)
def _s12(p):
    return SpinSystem(S=[0.5], g=p['g'].view(1, 3), gStrain=p['gStrain'].view(1, 3),
                      gFrame=T(0.3, 0.5, 0.2).view(1, 3), HStrain=p['HStrain'].view(3))


def _cu(p):
    return SpinSystem(S=[0.5], g=p['g'].view(1, 3), Nucs='63Cu', A=p['A'].view(1, 3),
                      gStrain=p['gStrain'].view(1, 3), AStrain=p['AStrain'].view(3),
                      gFrame=T(0.2, 0.4, 0.1).view(1, 3))


def _s1(p):
    return SpinSystem(S=[1.0], g=p['g'].view(1, 3), D=p['D'].view(1, 2),
                      DStrain=T(300., 100.).view(1, 2), HStrain=p['HStrain'].view(3))


def _two_e(p):
    return SpinSystem(S=[0.5, 0.5], g=torch.stack([p['g'], T(2.05, 2.05, 2.0)]),
                      ee=p['J'].view(1, 1) * torch.ones(1, 3, dtype=DT),
                      gStrain=torch.stack([p['gStrain'], T(0.01, 0.01, 0.01)]))


CASES = {
    'S12_gStrain_HStrain': (_s12, dict(g=T(2.0, 2.1, 2.2), gStrain=T(0.01, 0.02, 0.03),
                                       HStrain=T(10., 20., 30.)), [250., 400.]),
    'Cu_A_AStrain': (_cu, dict(g=T(2.05, 2.05, 2.25), A=T(60., 60., 500.),
                               gStrain=T(0.01, 0.01, 0.02), AStrain=T(20., 20., 40.)), [230., 380.]),
    'S1_D_DStrain_HStrain': (_s1, dict(g=T(2.0, 2.0, 2.0), D=T(1500., 200.),
                                       HStrain=T(30., 40., 50.)), [100., 600.]),
    'two_e_J_gStrain': (_two_e, dict(g=T(2.0, 2.0, 2.1), J=T(500.),
                                     gStrain=T(0.02, 0.02, 0.02)), [250., 400.]),
}

# (case, parameter, component, FD step)
GRAD_PARAMS = [
    ('S12_gStrain_HStrain', 'g', 0, 1e-5),
    ('S12_gStrain_HStrain', 'g', 2, 1e-5),
    ('S12_gStrain_HStrain', 'gStrain', 1, 1e-5),
    ('S12_gStrain_HStrain', 'HStrain', 2, 1e-2),
    ('Cu_A_AStrain', 'g', 2, 1e-5),
    ('Cu_A_AStrain', 'A', 2, 1e-2),
    ('Cu_A_AStrain', 'A', 0, 1e-2),
    ('Cu_A_AStrain', 'gStrain', 2, 1e-5),
    ('Cu_A_AStrain', 'AStrain', 2, 1e-2),
    ('S1_D_DStrain_HStrain', 'g', 0, 1e-5),
    ('S1_D_DStrain_HStrain', 'D', 0, 1e-2),
    ('S1_D_DStrain_HStrain', 'D', 1, 1e-2),
    ('S1_D_DStrain_HStrain', 'HStrain', 1, 1e-2),
    ('two_e_J_gStrain', 'J', 0, 1e-2),
    ('two_e_J_gStrain', 'g', 2, 1e-5),
    ('two_e_J_gStrain', 'gStrain', 0, 1e-5),
]


def _params(name, override=None):
    p = {k: v.clone() for k, v in CASES[name][1].items()}
    if override:
        p.update(override)
    return p


def _exp(name):
    return Experiment(mwFreq=9.5, Range=CASES[name][2])


def _run_resfields(name, p, fn=resfields):
    sys = CASES[name][0](p)
    H0, mux, muy, muz = ham(sys)
    return fn(H0, mux, muy, muz, PHI, THETA, _exp(name), Options(), sys=sys)


def _run_csw(name, p, B0=320.0):
    sys = CASES[name][0](p)
    H0, mux, muy, muz = ham(sys)
    n = H0.shape[0]
    tr = torch.tensor([(i, j) for i in range(n) for j in range(i + 1, n)])
    return compute_strain_widths(sys, H0, mux, muy, muz, PHI, THETA, tr, 9.5, B0)


def _fd4(f, x0, h):
    """4-point central difference of a list-of-scalars function f(x).

    Returns the derivatives and the round-off floor of each: 10 eps |f(x0)| / h
    (the observed function-value noise of the eigensolver and root search is
    about 0.03 eps |f| / h, so the floor has a margin of 300).
    """
    vals = [f(x0 + k * h) for k in (2, 1, -1, -2)]
    d = [(-a + 8 * b - 8 * c + e) / (12 * h) for a, b, c, e in zip(*vals)]
    eps = torch.finfo(DT).eps
    floor = [10 * eps * max(abs(a), abs(e)) / h for a, e in zip(vals[0], vals[3])]
    return d, floor


def _check(ad, fd_floor, what):
    fd, floor = fd_floor
    for k, (a, b, fl) in enumerate(zip(ad, fd, floor)):
        assert abs(a - b) <= RTOL * abs(b) + fl, (
            f"{what}[{k}]: autograd {a:+.10e} vs FD {b:+.10e} (round-off floor {fl:.1e})")


# ---------------------------------------------------------------------------
# resfields: gradients of B, I and W vs finite differences
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,key,idx,h", GRAD_PARAMS,
                         ids=[f"{c}-{k}{i}" for c, k, i, _ in GRAD_PARAMS])
def test_resfields_gradients_match_finite_differences(name, key, idx, h):
    base = _params(name)
    x0 = float(base[key].view(-1)[idx])

    def outputs(x, grad=False):
        v = base[key].clone()
        if grad:
            v.requires_grad_(True)
            vv = v
        else:
            vv = v.clone()
            vv.view(-1)[idx] = x
        B, I, W = _run_resfields(name, _params(name, {key: vv}))
        return (B, I, W), v

    (B, I, W), leaf = outputs(x0, grad=True)
    assert B.numel() > 0 and W is not None and W.shape == B.shape
    # strains enter the widths only; Hamiltonian parameters enter B, I and W
    assert W.requires_grad
    if key not in STRAIN_KEYS:
        assert B.requires_grad and I.requires_grad
    ad = []
    for out in (B, I, W):
        if not out.requires_grad:
            ad.append(0.0)
            continue
        g, = torch.autograd.grad(out.sum(), leaf, retain_graph=True, allow_unused=True)
        ad.append(0.0 if g is None else float(g.view(-1)[idx]))

    n_lines = B.numel()

    def f(x):
        (Bx, Ix, Wx), _ = outputs(x)
        assert Bx.numel() == n_lines, "line count changed within the FD stencil"
        return [float(Bx.sum()), float(Ix.sum()), float(Wx.sum())]

    _check(ad, _fd4(f, x0, h), f"{name} d(B,I,W)/d{key}[{idx}]")


def test_resfields_S1_dstrain_hstrain_width_gradient_sign():
    """The case that was silently wrong: dW/dg0 = +0.771 (eigenvectors only) vs FD -0.807."""
    name = 'S1_D_DStrain_HStrain'
    g = T(2.0, 2.0, 2.0).requires_grad_(True)
    B, I, W = _run_resfields(name, _params(name, {'g': g}))
    dW, = torch.autograd.grad(W.sum(), g)

    def f(x):
        _, _, Wx = _run_resfields(name, _params(name, {'g': T(x, 2.0, 2.0)}))
        return [float(Wx.sum())]

    fd = _fd4(f, 2.0, 1e-5)
    assert fd[0][0] < -0.5
    _check([float(dW[0])], fd, "S=1 dW/dg0")


# ---------------------------------------------------------------------------
# compute_strain_widths: gradients vs finite differences
# ---------------------------------------------------------------------------

CSW_PARAMS = [p for p in GRAD_PARAMS if p[0] != 'two_e_J_gStrain' or p[1] != 'g']


@pytest.mark.parametrize("name,key,idx,h", CSW_PARAMS,
                         ids=[f"{c}-{k}{i}" for c, k, i, _ in CSW_PARAMS])
def test_compute_strain_widths_gradients_match_finite_differences(name, key, idx, h):
    base = _params(name)
    x0 = float(base[key].view(-1)[idx])
    leaf = base[key].clone().requires_grad_(True)
    W = _run_csw(name, _params(name, {key: leaf}))
    assert W.requires_grad
    # finite widths only (forbidden pairs with zero slope give 0, not differentiable)
    mask = torch.isfinite(W.detach()) & (W.detach() > 0)
    g, = torch.autograd.grad(W[mask].sum(), leaf, allow_unused=True)
    ad = [0.0 if g is None else float(g.view(-1)[idx])]

    def f(x):
        v = base[key].clone()
        v.view(-1)[idx] = x
        return [float(_run_csw(name, _params(name, {key: v}))[mask].sum())]

    _check(ad, _fd4(f, x0, h), f"{name} dW/d{key}[{idx}]")


def test_compute_strain_widths_gradient_in_B0_and_theta():
    name = 'S1_D_DStrain_HStrain'
    p = _params(name)
    sys = CASES[name][0](p)
    H0, mux, muy, muz = ham(sys)
    tr = torch.tensor([[0, 1], [1, 2]])
    B0 = torch.tensor(320.0, dtype=DT, requires_grad=True)
    th = torch.tensor(THETA, dtype=DT, requires_grad=True)
    W = compute_strain_widths(sys, H0, mux, muy, muz, PHI, th, tr, 9.5, B0)
    gB, gth = torch.autograd.grad(W.sum(), (B0, th))

    def fB(x):
        return [float(compute_strain_widths(sys, H0, mux, muy, muz, PHI, THETA, tr, 9.5, x).sum())]

    def fth(x):
        return [float(compute_strain_widths(sys, H0, mux, muy, muz, PHI, x, tr, 9.5, 320.0).sum())]

    _check([float(gB)], _fd4(fB, 320.0, 1e-2), "dW/dB0")
    _check([float(gth)], _fd4(fth, THETA, 1e-5), "dW/dtheta")


# ---------------------------------------------------------------------------
# Values: identical to the batched solvers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(CASES))
def test_resfields_equals_resfields_batch(name):
    p = _params(name)
    B, I, W = _run_resfields(name, p)
    sys = CASES[name][0](p)
    H0, mux, muy, muz = ham(sys)
    Bb, Ib, Wb = resfields_batch(H0, mux, muy, muz, T(PHI), T(THETA), _exp(name), Options(), sys=sys)
    assert B.numel() > 0
    assert torch.equal(B, Bb[0]) and torch.equal(I, Ib[0]) and torch.equal(W, Wb[0])


@pytest.mark.parametrize("name", list(CASES))
def test_compute_strain_widths_equals_batch(name):
    p = _params(name)
    sys = CASES[name][0](p)
    H0, mux, muy, muz = ham(sys)
    n = H0.shape[0]
    tr = torch.tensor([(i, j) for i in range(n) for j in range(i + 1, n)])
    W = compute_strain_widths(sys, H0, mux, muy, muz, PHI, THETA, tr, 9.5, 320.0)
    st, ct, sp, cp = map(float, (torch.sin(T(THETA)), torch.cos(T(THETA)), torch.sin(T(PHI)), torch.cos(T(PHI))))
    muzL = (st * cp * mux + st * sp * muy + ct * muz).unsqueeze(0)
    Wb = compute_strain_widths_batch(sys, H0, muzL, T(PHI), T(THETA), T(320.0), [tr], 9.5)[0]
    assert W.shape == (tr.shape[0],)
    torch.testing.assert_close(W, Wb, rtol=1e-14, atol=0.0)


def test_no_strain_gives_none_and_strain_without_lines_gives_empty():
    s = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
    H0, mux, muy, muz = ham(s)
    B, I, W = resfields(H0, mux, muy, muz, PHI, THETA, Experiment(mwFreq=9.5, Range=[300., 400.]),
                        Options(), sys=s)
    assert B.numel() == 1 and W is None
    s = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]], HStrain=[10., 10., 10.])
    H0, mux, muy, muz = ham(s)
    B, I, W = resfields(H0, mux, muy, muz, PHI, THETA, Experiment(mwFreq=9.5, Range=[400., 500.]),
                        Options(), sys=s)
    assert B.numel() == 0 and I.numel() == 0
    assert W is not None and W.numel() == 0


# ---------------------------------------------------------------------------
# Looping transition: all resonances of a level pair
# ---------------------------------------------------------------------------

def test_looping_transition_returns_both_resonances():
    """S = 1, D = 10 GHz > h nu: the lowest level pair (0, 1) is resonant twice in [5, 1000] mT.

    The old two-point bracket found at most one resonance per level pair (here
    18.8 mT, without the line at 622.8 mT).  The fields are checked against an
    independent dense scan of the exact eigenvalues.
    """
    s = SpinSystem(S=[1.0], g=[[2.0, 2.0, 2.0]], D=[[10000.0, 0.0]])
    H0, mux, muy, muz = ham(s)
    exp = Experiment(mwFreq=9.5, Range=[5.0, 1000.0])
    phi, theta = 0.0, 0.3
    B, I, W = resfields(H0, mux, muy, muz, phi, theta, exp, Options(Threshold=0), sys=s)
    Bb, Ib, _, P = resfields_batch(H0, mux, muy, muz, T(phi), T(theta), exp, Options(Threshold=0),
                                   sys=s, return_pairs=True)
    assert torch.equal(B, Bb[0]) and torch.equal(I, Ib[0])

    # Independent reference: sign changes of E_1 - E_0 - h nu on a dense grid, refined by bisection.
    st, ct = float(torch.sin(T(theta))), float(torch.cos(T(theta)))
    muzL = st * mux + ct * muz

    def f(b):
        E = torch.linalg.eigvalsh(H0 - b * muzL)
        return float(E[1] - E[0]) - 9500.0

    grid = torch.linspace(5.0, 1000.0, 20001, dtype=DT).tolist()
    roots = []
    for a, b in zip(grid[:-1], grid[1:]):
        fa, fb = f(a), f(b)
        if fa * fb < 0:
            for _ in range(60):
                m = 0.5 * (a + b)
                if f(m) * fa > 0:
                    a, fa = m, f(m)
                else:
                    b = m
            roots.append(0.5 * (a + b))
    assert len(roots) == 2
    pair01 = (P[0][:, 0] == 0) & (P[0][:, 1] == 1)
    B01 = B[pair01]
    assert B01.numel() == 2
    torch.testing.assert_close(B01, T(*roots), rtol=0, atol=1e-6)


# ---------------------------------------------------------------------------
# B5: full g matrix + gStrain raises
# ---------------------------------------------------------------------------

def test_full_g_with_gstrain_raises():
    gfull = torch.tensor([[2.0, 0.05, 0.0], [0.05, 2.1, 0.0], [0.0, 0.0, 2.2]], dtype=DT)
    s = SpinSystem(S=[0.5], g=gfull, gStrain=T(0.01, 0.02, 0.03).view(1, 3))
    H0, mux, muy, muz = ham(s)
    with pytest.raises(ValueError, match="full g"):
        resfields(H0, mux, muy, muz, PHI, THETA, Experiment(mwFreq=9.5, Range=[250., 400.]),
                  Options(), sys=s)
    with pytest.raises(ValueError, match="full g"):
        compute_strain_widths(s, H0, mux, muy, muz, PHI, THETA, torch.tensor([[0, 1]]), 9.5, 320.0)
