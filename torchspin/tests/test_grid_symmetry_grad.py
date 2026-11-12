"""B1: the automatic grid symmetry must hold for the differentiated parameters.

The powder grid is chosen outside the autograd graph.  A symmetry that holds at
the current values only by accident (an axial or isotropic g given as three
independent numbers, a tensor rotated about the common z axis) is broken by the
perturbation the gradient measures, so a grid of that symmetry gives a wrong
gradient.  ``torchspin.hamsymm.grid_symmetry`` chooses the symmetry of the
differentiated parameterization (autograd leaves perturbed, fields moved along
J @ dp): a scalar g keeps O3, [g_perp, g_perp, g_par] from two leaves keeps
Dinfh, independent components lose the accidental symmetry; Ci when the symmetry
frame itself moves with the parameters.

Reference: central differences of the forward model on the grid the gradient
differentiates (D2h / Ci forced), steps h = 1e-7 for g and 1e-6 for A (MHz):
converged (h = 3e-8 resp. 1e-5 change the values by < 1e-6 relative).  For the
components that break the accidental symmetry, the central difference of the
default ('auto') forward model is the same number.
"""
import pytest
import torch

from torchspin import SpinSystem, Experiment, Options, pepper
from torchspin.hamsymm import hamsymm, grid_symmetry
from torchspin.pepper_autograd import pepper_autograd
from torchspin.salt import salt

F64 = torch.float64


def _t(v, grad=True):
    return torch.tensor(v, dtype=F64, requires_grad=grad)

X_CU = Experiment(mwFreq=9.5, Range=[250., 360.], nPoints=1024, Harmonic=1)
X_SMALL = Experiment(mwFreq=9.5, Range=[300., 360.], nPoints=512, Harmonic=1)


def _cu(p):
    return SpinSystem(S=[0.5], g=[p['g']], Nucs=['63Cu'], A=[p['A']], lw=[0.6, 0.])


def _iso(p):
    return SpinSystem(S=[0.5], g=[p['g']], lw=[1.0, 0.])


def _rot_z(p):
    # rhombic 1H hyperfine rotated about z: D2h with the axial g at these values,
    # C2h as soon as g is rhombic; its frame differs from g's, so the safe grid is Ci
    return SpinSystem(S=[0.5], g=[p['g']], Nucs=['1H'], A=[p['A']], AFrame=[[0.5, 0., 0.]], lw=[1.0, 0.])


def _loss(fn, make, exp, p, keys, grad, sym='auto', N=(31, 4)):
    ts = {k: torch.tensor(v, dtype=F64, requires_grad=grad and k in keys) for k, v in p.items()}
    y = fn(make(ts), exp, Options(GridSize=list(N), GridSymmetry=sym))[1]
    loss = (y ** 2).sum()
    if not grad:
        return float(loss)
    return [g.detach() for g in torch.autograd.grad(loss, [ts[k] for k in keys])]


def _fd(fn, make, exp, p, key, h, sym, N):
    out = []
    for i in range(len(p[key])):
        vals = []
        for s in (1, -1):
            q = dict(p)
            v = list(p[key]); v[i] += s * h; q[key] = v
            vals.append(_loss(fn, make, exp, q, [], False, sym, N))
        out.append((vals[0] - vals[1]) / (2 * h))
    return torch.tensor(out, dtype=F64)


def _rel(a, b):
    return float((a - b).abs().max() / b.abs().max())


def test_axial_g_and_A_pepper_autograd():
    """Axial g and A (63Cu): the default grid was Dinfh, dL/dg_x = -11.5e6 vs -5.72e6."""
    p = dict(g=[2.05, 2.05, 2.25], A=[40., 40., 550.])
    assert hamsymm(_cu(p))[0] == 'Dinfh'
    dg, dA = _loss(pepper_autograd, _cu, X_CU, p, ['g', 'A'], True)
    fd_g = _fd(pepper_autograd, _cu, X_CU, p, 'g', 1e-7, 'D2h', (31, 4))
    fd_A = _fd(pepper_autograd, _cu, X_CU, p, 'A', 1e-6, 'D2h', (31, 4))
    assert _rel(dg, fd_g) < 1e-5, (dg, fd_g)
    assert _rel(dA, fd_A) < 1e-4, (dA, fd_A)
    # gx, gy break the axial symmetry: the default forward model (D2h at g +- h)
    # has the same central difference
    fd_auto = _fd(pepper_autograd, _cu, X_CU, p, 'g', 1e-7, 'auto', (31, 4))
    assert _rel(dg[:2], fd_auto[:2]) < 1e-5, (dg, fd_auto)


def test_axial_g_via_pepper_routing():
    """pepper() routes grad-requiring inputs to pepper_autograd: same gradient."""
    p = dict(g=[2.05, 2.05, 2.25], A=[40., 40., 550.])
    (dg,) = _loss(pepper, _cu, X_CU, p, ['g'], True)
    fd_g = _fd(pepper, _cu, X_CU, p, 'g', 1e-7, 'D2h', (31, 4))
    assert _rel(dg, fd_g) < 1e-5, (dg, fd_g)


def test_isotropic_g_three_components():
    """g = [2, 2, 2] as three independent numbers: O3 gave [4.57e6, 4.57e6, -4.57e6]
    vs ~1.596e6 each.  (A scalar g keeps O3: test_scalar_g_keeps_o3.)"""
    p = dict(g=[2.0, 2.0, 2.0])
    assert hamsymm(_iso(p))[0] == 'O3'
    (dg,) = _loss(pepper_autograd, _iso, X_SMALL, p, ['g'], True, N=(19, 4))
    fd_g = _fd(pepper_autograd, _iso, X_SMALL, p, 'g', 1e-7, 'D2h', (19, 4))
    # 1e-3, not 1e-5: at exactly coincident resonance fields the projection's
    # gradient is off by ~8e-4 on ANY fixed grid (forced D2h or Ci alike; exact
    # 1e-4 away from isotropy) -- a separate pepper_autograd limitation.  The
    # old O3 gradient was off by a factor 3 with the wrong sign in z.
    assert _rel(dg, fd_g) < 1e-3, (dg, fd_g)
    # every component breaks O3: within the Dinfh-vs-D2h discretization of the
    # default forward model (0.3 %), far from the old O3 gradient
    fd_auto = _fd(pepper_autograd, _iso, X_SMALL, p, 'g', 1e-7, 'auto', (19, 4))
    assert _rel(dg, fd_auto) < 5e-3, (dg, fd_auto)


def test_axial_g_with_A_rotated_about_z():
    """D2h at the values; rhombic g shares only z with the rotated A: C2h about z
    (same group and frame for two independent perturbations)."""
    p = dict(g=[2.0, 2.0, 2.1], A=[20., 60., 120.])
    sys = _rot_z(p)
    assert hamsymm(sys)[0] == 'D2h'
    sys_g = _rot_z(dict(g=torch.tensor(p['g'], requires_grad=True), A=p['A']))
    pg, R = grid_symmetry(sys_g)
    assert pg == 'C2h' and torch.allclose(R.abs(), torch.eye(3, dtype=F64))
    (dg,) = _loss(pepper_autograd, _rot_z, X_SMALL, p, ['g'], True, N=(19, 4))
    fd_g = _fd(pepper_autograd, _rot_z, X_SMALL, p, 'g', 1e-7, 'C2h', (19, 4))
    assert _rel(dg, fd_g) < 1e-5, (dg, fd_g)
    # the old D2h grid: gx was 9x off (-249499 vs 28623)
    fd_ci = _fd(pepper_autograd, _rot_z, X_SMALL, p, 'g', 1e-7, 'Ci', (19, 4))
    assert _rel(dg, fd_ci) < 5e-3, (dg, fd_ci)


def test_salt_axial_A():
    """salt ENDOR, axial A and Q (14N): dA was [-0.217, 0, 0.053] vs -0.1079 for x and y."""
    exp = Experiment(Field=3394, mwFreq=95)

    def run(A, grad, sym='auto'):
        At = torch.tensor([A], dtype=F64, requires_grad=grad)
        S = SpinSystem(S=[0.5], g=[[2.2, 2.2, 2.0]], Nucs='14N', A=At, Q=[[0.84, 0.84, -1.68]])
        y = salt(S, exp, Options(GridSize=19, GridSymmetry=sym), freq_range=(0, 35),
                 n_points=512, lw_mhz=0.2)[1]
        loss = (torch.as_tensor(y) ** 2).sum() * 1e12
        return torch.autograd.grad(loss, At)[0][0] if grad else float(loss)

    A0, h = [4.0, 4.0, 5.0], 1e-6
    dA = run(A0, True)

    def fd(sym):
        return torch.tensor([(run([a + h * (i == j) for j, a in enumerate(A0)], False, sym)
                              - run([a - h * (i == j) for j, a in enumerate(A0)], False, sym)) / (2 * h)
                             for i in range(3)], dtype=F64)
    # h = 1e-6 MHz: the central difference agrees with h = 1e-7 to ~5e-4
    fd_d2h = fd('D2h')
    assert _rel(dA, fd_d2h) < 1e-3, (dA, fd_d2h)
    fd_auto = fd('auto')
    assert _rel(dA[:2], fd_auto[:2]) < 1e-3, (dA, fd_auto)


# ---------------------------------------------------------------------------
# grid_symmetry decisions
# ---------------------------------------------------------------------------

def _stack(*vals):
    return torch.stack(list(vals))


_GP, _GZ = _t(2.0), _t(2.2)


@pytest.mark.parametrize('make, expected', [
    (lambda: SpinSystem(S=0.5, g=_t(2.1)), 'O3'),                                  # scalar g: one leaf
    (lambda: SpinSystem(S=0.5, g=_stack(_GP, _GP, _GZ)), 'Dinfh'),                 # [gp, gp, gz]: two leaves
    (lambda: SpinSystem(S=0.5, g=_t([2.0, 2.0, 2.2])), 'D2h'),                    # three independent numbers
    (lambda: SpinSystem(S=[0.5], g=_t([[2.0, 2.0, 2.2]])), 'D2h'),                # a leaf passed directly
    (lambda: SpinSystem(S=0.5, g=_t([2.0, 2.0, 2.0])), 'D2h'),                    # O3 only at the values
    (lambda: SpinSystem(S=0.5, g=_t([2.0, 2.05, 2.1])), 'D2h'),                   # rhombic: unchanged
    (lambda: SpinSystem(S=0.5, g=_t([2.0, 2.05, 2.1], False), gFrame=_t([0.3, 0.4, 0.5])), 'Ci'),  # frame moves
    (lambda: SpinSystem(S=0.5, g=_t([2.0, 2.0, 2.0], False), gFrame=_t([0.3, 0.4, 0.5])), 'O3'),   # frame of an isotropic g
    (lambda: SpinSystem(S=0.5, g=_t([2.0, 2.0, 2.2], False), Nucs='1H', A=[[1., 1., 5.]], gnscale=_t([1.0])), 'Dinfh'),  # gnscale: no symmetry input
    (lambda: SpinSystem(S=0.5, g=_t([2.0, 2.0, 2.2], False), HStrain=_t(10.)), 'Dinfh'),          # isotropic HStrain from one leaf
    (lambda: SpinSystem(S=0.5, g=_t([2.0, 2.0, 2.2], False), HStrain=_t([10., 10., 10.])), 'D2h'),  # three strain numbers
    (lambda: SpinSystem(S=1, g=2.0, D=_t([300., 0.])), 'D2h'),                    # E is a parameter
    (lambda: SpinSystem(S=0.5, g=_t([[2.0, 0, 0], [0, 2.0, 0], [0, 0, 2.2]])), 'Ci'),  # full tensor
])
def test_grid_symmetry_decisions(make, expected):
    assert grid_symmetry(make())[0] == expected


def test_grid_symmetry_frame_angle_grad_on_second_tensor():
    """A differentiated AFrame collinear with g within hamsymm's 0.2 deg tolerance
    must still be detected (the angle perturbation exceeds that tolerance)."""
    sys = SpinSystem(S=0.5, g=[2.0, 2.05, 2.1], Nucs='1H', A=[[2., 5., 9.]],
                     AFrame=_t([[0., 0., 0.]]))
    assert hamsymm(sys)[0] == 'D2h'
    assert grid_symmetry(sys)[0] == 'Ci'


def test_grid_symmetry_without_grad_is_hamsymm():
    """No symmetry input requires grad: exactly hamsymm (same group, same frame)."""
    systems = [
        SpinSystem(S=0.5, g=[2.0, 2.0, 2.2], gFrame=[0.1, 0.7, 0.2]),
        SpinSystem(S=0.5, g=[2.0, 2.05, 2.2], Nucs='63Cu', A=[40., 50., 550.], AFrame=[0.2, 0.3, 0.]),
        SpinSystem(S=1, g=2.0, D=[300., 50.], lw=0.5),
        SpinSystem(S=[3.5], g=[[2, 2, 2]], B=[None, [[50., 0, 0, 0, 10., 0, 0, 0, 0]], None]),
    ]
    for s in systems:
        pg0, R0 = hamsymm(s)
        pg1, R1 = grid_symmetry(s)
        assert pg0 == pg1
        assert torch.equal(R0, R1)


def test_scalar_g_keeps_o3():
    """A scalar g that requires grad is an isotropic parameterization: O3 (one
    orientation, exact line templates), forward equal to plain pepper."""
    g = _t(2.0)
    sys = SpinSystem(S=[0.5], g=g, lw=[1.0, 0.])
    assert grid_symmetry(sys)[0] == 'O3'
    y = pepper(sys, X_SMALL, Options(GridSize=[19, 4]))[1]
    y0 = pepper(SpinSystem(S=[0.5], g=2.0, lw=[1.0, 0.]), X_SMALL, Options(GridSize=[19, 4]))[1]
    assert y.requires_grad
    assert float(y0.abs().max()) > 0
    assert _rel(y.detach(), y0) < 1e-9          # pepper_autograd's O3 path vs pepper (#65)


def test_two_leaf_axial_g_keeps_dinfh():
    """g = [g_perp, g_perp, g_par] from two leaves: Dinfh, and the gradient is the
    central difference of the (Dinfh) forward model."""
    def run(gp, gz, grad):
        tp, tz = _t(gp, grad), _t(gz, grad)
        sys = SpinSystem(S=[0.5], g=torch.stack([tp, tp, tz]).unsqueeze(0), Nucs=['63Cu'],
                         A=[[40., 40., 550.]], lw=[0.6, 0.])
        y = pepper_autograd(sys, X_CU, Options(GridSize=[31, 4]))[1]
        loss = (y ** 2).sum()
        return torch.stack(torch.autograd.grad(loss, [tp, tz])) if grad else float(loss)

    tp, tz = _t(2.05), _t(2.25)
    assert grid_symmetry(SpinSystem(S=0.5, g=torch.stack([tp, tp, tz]))) [0] == 'Dinfh'
    d = run(2.05, 2.25, True)
    h = 1e-7
    fd = torch.tensor([(run(2.05 + h, 2.25, False) - run(2.05 - h, 2.25, False)) / (2 * h),
                       (run(2.05, 2.25 + h, False) - run(2.05, 2.25 - h, False)) / (2 * h)], dtype=F64)
    assert _rel(d, fd) < 1e-5, (d, fd)


def test_field_leaf_and_graph_mixed():
    """A field passed as a leaf (its own parameterization) next to one with a graph."""
    gz = _t(2.2)
    A = _t([[40., 40., 550.]])                     # leaf: three independent numbers
    sys = SpinSystem(S=0.5, g=torch.stack([_t(2.0, False), _t(2.0, False), gz]), Nucs='63Cu', A=A)
    assert grid_symmetry(sys)[0] == 'D2h'
    sys2 = SpinSystem(S=0.5, g=torch.stack([_t(2.0, False), _t(2.0, False), gz]), Nucs='63Cu',
                      A=[[40., 40., 550.]])
    assert grid_symmetry(sys2)[0] == 'Dinfh'


def test_no_double_backward_falls_back_to_fields(monkeypatch):
    """Without double backward each grad-requiring field is perturbed on its own,
    which is never less safe (scalar g then gets D2h)."""
    import sys as _sys
    hs = _sys.modules['torchspin.hamsymm']

    def no_jvp(items, gen):
        raise RuntimeError('double backward not supported')
    monkeypatch.setattr(hs, '_field_directions', no_jvp)
    assert grid_symmetry(SpinSystem(S=0.5, g=_t(2.1)))[0] == 'D2h'


def test_graph_survives_grid_symmetry():
    """The JVP uses the graph without consuming it or touching .grad."""
    gz = _t(2.2)
    g = torch.stack([_t(2.0, False), _t(2.0, False), gz])
    sys = SpinSystem(S=0.5, g=g)
    grid_symmetry(sys)
    assert gz.grad is None
    (sys.g ** 2).sum().backward()
    assert gz.grad is not None
