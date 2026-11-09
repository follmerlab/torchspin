"""torchspin.lft.dlsd5 (low-spin d5 g <-> ligand field), milestone M4.

The NumPy oracle is the solver copied into ``torchspin.lft._engines.dlsd5``, so forward parity,
gradients against central differences and the inverse-of-forward Jacobian checks run without
telser-tools.  When telser-tools is installed, the copied engine is checked to reproduce it
exactly on every fixture (``tests/data/telser/dlsd5_fortran.json``, inputs of Telser's DLSD5 runs).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.autograd.forward_ad as fwAD

from torchspin import lft
from torchspin.lft import dlsd5 as td
from torchspin.lft._engines import dlsd5 as E

F64 = torch.float64
REF = json.loads((Path(__file__).parent / "data" / "telser" / "dlsd5_fortran.json").read_text())
TET_G = [c["input"] for c in REF["tetragonal_from_g"]]
TRI_G = [c["input"] for c in REF["trigonal_from_g"]]       # g_par, g_perp, 'B'|'K', k, twist (degrees)


def close(a, b, rtol=1e-10, atol=1e-12):
    np.testing.assert_allclose(np.asarray(a, float), np.asarray(b, float), rtol=rtol, atol=atol)


def fd_jacobian(fun, x, steps, lower=None, upper=None):
    """Central differences (one-sided next to a bound)."""
    x = np.asarray(x, float)
    cols = []
    for j, h in enumerate(steps):
        xp, xm = x.copy(), x.copy()
        up = upper is None or x[j] + h <= upper[j]
        dn = lower is None or x[j] - h >= lower[j]
        if up and dn:
            xp[j] += h
            xm[j] -= h
            cols.append((fun(xp) - fun(xm)) / (2 * h))
        elif up:
            xp[j] += h
            cols.append((fun(xp) - fun(x)) / h)
        else:
            xm[j] -= h
            cols.append((fun(x) - fun(xm)) / h)
    return np.array(cols).T


def best_fd(fun, x, J, steps=(1e-4, 1e-5, 1e-6), lower=None, upper=None):
    """max |J - J_fd| / max |J|, best of a few step sizes (per column)."""
    err = np.full(len(x), np.inf)
    for h in steps:
        Jf = fd_jacobian(fun, np.asarray(x, float), np.full(len(x), h), lower, upper)
        err = np.minimum(err, np.abs(J - Jf).max(0))
    return err.max() / np.abs(J).max()


def rev_jacobian(y, x):
    return torch.stack([torch.autograd.grad(y[i], x, retain_graph=True)[0] for i in range(len(y))]).numpy()


def _tri_kw(mode, val):
    return {"fix": "beta"} if mode == "B" else {"fix": "k", "k": val}


# ------------------------------------------------------------------------------ the copied engine == telser-tools
def _telser():
    pytest.importorskip("telser_tools.dlsd5")
    from telser_tools.dlsd5 import hamiltonian, tetragonal, trigonal
    return hamiltonian, tetragonal, trigonal


def _same(a, b):
    """Dataclass solutions equal field by field (floats exactly, NaN == NaN)."""
    assert type(a).__name__ == type(b).__name__
    for f in a.__dataclass_fields__:
        x, y = getattr(a, f), getattr(b, f)
        if isinstance(x, float) and isinstance(y, float) and math.isnan(x):
            assert math.isnan(y), f
        else:
            assert x == y, (f, x, y)


@pytest.mark.parametrize("mode", ["mcgarvey", "all"])
def test_engine_copy_identical_to_telser_tools_tetragonal(mode):
    _, T, _ = _telser()
    for g in TET_G if mode == "mcgarvey" else TET_G[:2]:
        for standardize in (True, False):
            ref, got = T.from_g(*g, mode=mode, standardize=standardize), E.tetragonal_from_g(*g, mode=mode,
                                                                                               standardize=standardize)
            assert len(ref) == len(got)
            for r, s in zip(ref, got):
                _same(r, s)
    for a, b, k in [c["input"] for c in REF["tetragonal_from_abc"]] + [(0.3, 0.4, 0.9), (1.2, 0.3, 0.9)]:
        for r, s in zip(T.from_complex_coefficients(a, b, k), E.from_complex_coefficients(a, b, k)):
            _same(r, s)
        for r, s in zip(T.from_real_coefficients(a, b, k), E.from_real_coefficients(a, b, k)):
            _same(r, s)


def test_engine_copy_identical_to_telser_tools_trigonal_and_hamiltonian():
    H, _, TR = _telser()
    assert E.BETA_OCT == TR.BETA_OCT
    for gpar, gperp, mode, val, phi in TRI_G:
        for r, s in zip(TR.from_g(gpar, gperp, twist_deg=phi, **_tri_kw(mode, val)),
                        E.trigonal_from_g(gpar, gperp, twist_deg=phi, **_tri_kw(mode, val))):
            _same(r, s)
    for at, k in [c["input"] for c in REF["trigonal_from_coefficients"]]:
        for r, s in zip(TR.from_coefficients(at, k), E.trigonal_from_coefficients(at, k)):
            _same(r, s)
    for i in range(3):
        np.testing.assert_array_equal(E._L[i], H._L[i])
        np.testing.assert_array_equal(E._SPIN[i], H._SPIN[i])
    for cf in (H.tetragonal_cf(3.0, 1.0), H.trigonal_cf(-2.0)):
        a, b = H.solve_t2g_hole(cf, 0.9), E.solve_t2g_hole(cf, 0.9)
        np.testing.assert_array_equal(a.g_tensor, b.g_tensor)
        np.testing.assert_array_equal(a.doublet_energies, b.doublet_energies)


# ------------------------------------------------------------------------------ against Telser's Fortran printout
@pytest.mark.parametrize("case", REF["tetragonal_from_g"][:4], ids=lambda c: str(c["input"]))
def test_tetragonal_branches_reproduce_dlsd5_fortran_k(case):
    """k and the input |g| of each McGarvey branch match DLSD5.F's printout (5-6 digits); the
    Fortran prints its own (unrelabeled, E0-sign) conventions, so only k is compared directly."""
    got = [b for b in td.tetragonal_from_g(*case["input"]) if isinstance(b, td.TetragonalBranch)]
    ks = sorted(float(b.k) for b in got)
    ref = sorted(s["k"] for s in case["solutions"] if s.get("k") is not None)
    assert len(ks) == len(ref)
    np.testing.assert_allclose(ks, ref, atol=2e-5)


# ------------------------------------------------------------------------------ exact Hamiltonian
def _np_hole(cf, k):
    st = E.solve_t2g_hole(cf, k)
    return st, np.array([st.hole_weights[x] for x in E.T2G_LABELS])


def _check_hole(cf_np, cf_t, k):
    st, w = _np_hole(cf_np, k)
    ts = td.solve_t2g_hole(cf_t, k)
    close(ts.g_principal, st.g_principal)
    close(ts.ggT, st.g_tensor @ st.g_tensor.T, atol=1e-12)
    close(ts.doublet_energies, st.doublet_energies)
    close(ts.hole_weights, w)
    assert float(ts.g_sign_product) == st.g_sign_product
    close(ts.g_det, np.linalg.det(st.g_tensor))


def _fixture_ligand_fields():
    """(Delta, V, k) of every converged branch of every tetragonal fixture, and trigonal (Delta, k)."""
    tet, tri = [], []
    for g in TET_G:
        for s in E.tetragonal_from_g(*g):
            if s.converged and s.delta is not None:
                tet.append((s.delta, s.v, s.k))
    for gpar, gperp, mode, val, phi in TRI_G:
        for s in E.trigonal_from_g(gpar, gperp, **_tri_kw(mode, val)):
            if s.converged and s.delta is not None and abs(s.beta - E.BETA_OCT) < 1e-12:
                tri.append((s.delta, s.k))
    return tet, tri


FIX_TET, FIX_TRI = _fixture_ligand_fields()


@pytest.mark.parametrize("d,v,k", FIX_TET + [(0.0, 0.0, 1.0), (3.0, 0.0, 0.8), (-4.0, 2.0, 1.1)])
def test_hole_state_parity_tetragonal(d, v, k):
    _check_hole(E.tetragonal_cf(d, v), td.tetragonal_cf(d, v), k)


@pytest.mark.parametrize("d,k", FIX_TRI + [(-2.0, 0.9), (0.0, 1.0)])
def test_hole_state_parity_trigonal(d, k):
    _check_hole(E.trigonal_cf(d), td.trigonal_cf(d), k)


def test_hole_state_closed_forms_and_units():
    """Cubic t2g with k = 1: the hole doublet is Gamma7 with g = -2 (|g| = 2, det < 0); xi = 1
    splits t2g into Gamma8 at -xi/2 (4-fold) and Gamma7 at +xi (2-fold).  The exact Hamiltonian
    is homogeneous in (cf, xi): energies scale with the unit, g does not."""
    s = td.solve_t2g_hole(torch.zeros(3, 3, dtype=F64), 1.0)
    close(s.g_principal, [2.0, 2.0, 2.0], atol=1e-12)
    assert float(s.g_det) < 0
    close(s.doublet_energies, [-0.5, -0.5, 1.0], atol=1e-12)
    close(s.hole_weights, [1 / 3, 1 / 3, 1 / 3], atol=1e-12)
    f = lft.units.energy_factor("cm^-1", "MHz")
    a = td.solve_t2g_hole(td.tetragonal_cf(3.0, 1.0), 0.9, 1.0)
    b = td.solve_t2g_hole(td.tetragonal_cf(3.0 * f, 1.0 * f), 0.9, f)
    close(b.doublet_energies / f, a.doublet_energies, rtol=1e-12, atol=1e-12)
    close(b.g_principal, a.g_principal, rtol=1e-11, atol=1e-12)


def test_hole_state_batched():
    d = torch.tensor([3.0, -2.5, 6.0], dtype=F64)
    v = torch.tensor([1.0, 1.2, -3.0], dtype=F64)
    s = td.solve_t2g_hole(td.tetragonal_cf(d, v), torch.tensor([0.9, 1.05, 0.7], dtype=F64))
    assert s.g_principal.shape == (3, 3) and s.projector.shape == (3, 6, 6)
    for i in range(3):
        one = td.solve_t2g_hole(td.tetragonal_cf(d[i], v[i]), [0.9, 1.05, 0.7][i])
        close(s.g_principal[i], one.g_principal, rtol=1e-13)


def _np_hole_vec(x):
    d, v, k = x
    st, w = _np_hole(E.tetragonal_cf(d, v), k)
    g = st.g_tensor @ st.g_tensor.T
    return np.concatenate([st.g_principal, st.doublet_energies, w, g[np.triu_indices(3)]])


def _t_hole_vec(x):
    s = td.solve_t2g_hole(td.tetragonal_cf(x[0], x[1]), x[2])
    return torch.cat([s.g_principal, s.doublet_energies, s.hole_weights, s.ggT[tuple(torch.triu_indices(3, 3))]])


@pytest.mark.parametrize("x", [(3.0, 1.0, 0.9), (-2.5, 1.2, 1.05), (6.0, -3.0, 0.7)])
def test_hole_state_gradient_vs_numpy_fd(x):
    xt = torch.tensor(x, dtype=F64, requires_grad=True)
    J = rev_jacobian(_t_hole_vec(xt), xt)
    assert best_fd(_np_hole_vec, x, J) < 1e-7
    Jf = torch.func.jacfwd(_t_hole_vec)(xt.detach()).numpy()
    np.testing.assert_allclose(Jf, J, atol=1e-12 * np.abs(J).max())


@pytest.mark.parametrize("x", [(3.0, 0.0, 0.9), (0.0, 0.0, 1.0)])
def test_hole_state_gradients_at_axial_and_cubic(x):
    """Axial (gx = gy) and cubic (t2g degenerate): g g^T, sum g and the doublet energies stay
    exactly differentiable; gradcheck in both modes through the Kramers degeneracy."""
    xt = torch.tensor(x, dtype=F64, requires_grad=True)

    def f(z):
        s = td.solve_t2g_hole(td.tetragonal_cf(z[0], z[1]), z[2])
        return torch.cat([s.ggT.reshape(-1), s.g_principal.sum(-1, keepdim=True), s.hole_weights,
                          s.doublet_energies[..., -1:]])

    assert torch.autograd.gradcheck(f, (xt,), eps=1e-6, atol=1e-6, check_forward_ad=True)
    J = rev_jacobian(f(xt), xt)

    def npf(z):
        st, w = _np_hole(E.tetragonal_cf(z[0], z[1]), z[2])
        return np.concatenate([(st.g_tensor @ st.g_tensor.T).reshape(-1), [st.g_principal.sum()], w,
                               st.doublet_energies[-1:]])

    assert best_fd(npf, x, J) < 1e-7


# ------------------------------------------------------------------------------ closed forms (fixtures)
@pytest.mark.parametrize("case", REF["tetragonal_from_abc"], ids=lambda c: str(c["input"]))
@pytest.mark.parametrize("standardize", [False, True])
def test_from_complex_coefficients_parity(case, standardize):
    a, b, k = case["input"]
    for n, t in zip(E.from_complex_coefficients(a, b, k, standardize), td.from_complex_coefficients(a, b, k, standardize)):
        close([t.a, t.b, t.c], [n.a, n.b, n.c])
        close(t.g, n.g)
        if n.delta is not None:
            close([t.delta, t.v, t.e0], [n.delta, n.v, n.e0])


@pytest.mark.parametrize("case", REF["tetragonal_from_abc"], ids=lambda c: str(c["input"]))
def test_from_complex_coefficients_matches_fortran_g(case):
    a, b, k = case["input"]
    p, m = td.from_complex_coefficients(a, b, k, standardize=False)
    close(p.g, case["g_plus"], rtol=0, atol=2e-5)
    close(m.g, case["g_minus"], rtol=0, atol=2e-5)


@pytest.mark.parametrize("A,B", [(0.3, 0.4), (0.9, 0.2), (0.1, 0.95), (1.2, 0.3), (0.8, 0.8)])
def test_from_real_coefficients_parity_including_normalization(A, B):
    for n, t in zip(E.from_real_coefficients(A, B, 0.9), td.from_real_coefficients(A, B, 0.9)):
        close([t.a, t.b, t.c], [n.a, n.b, n.c])
        close(t.g, n.g)
        if n.delta is not None:
            close([t.delta, t.v, t.e0], [n.delta, n.v, n.e0])


@pytest.mark.parametrize("case", REF["trigonal_from_coefficients"], ids=lambda c: str(c["input"]))
def test_trigonal_from_coefficients_parity(case):
    at, k = case["input"]
    out = td.trigonal_from_coefficients(at, k).numpy()
    for row, s in zip(out, E.trigonal_from_coefficients(at, k)):
        close(row, [s.g_perp, s.g_par, s.delta])
    for row, key in zip(out, ("plus", "minus")):
        f = case[key]
        close(row, [f["g_perp"], f["g_par"], f["delta"]], rtol=0, atol=2e-5)


def test_closed_form_gradients_vs_numpy_fd():
    x = np.array([0.35, 0.9, 0.85])                                      # a, b, k (+c branch)

    def npf(z):
        s = E.from_complex_coefficients(*z, standardize=False)[0]
        return np.array([*s.g, s.delta, s.v, s.e0])

    xt = torch.tensor(x, requires_grad=True)
    s = td.from_complex_coefficients(*xt.unbind(), standardize=False)[0]
    J = rev_jacobian(torch.cat([s.g, torch.stack([s.delta, s.v, s.e0])]), xt)
    assert best_fd(npf, x, J) < 1e-8


def test_epi_from_twist_radians_and_degrees():
    d = torch.tensor(0.8, dtype=F64)
    close(td.epi_from_twist(d, math.radians(30.0)), td.epi_from_twist(d, 30.0, degrees=True), rtol=1e-15)
    close(td.epi_from_twist(d, math.pi / 4), 0.8 / 3.0, rtol=1e-15)
    close(td.epi_from_twist(d, 30.0, degrees=True), E.epi_from_twist(0.8, 30.0), rtol=1e-14)


# ------------------------------------------------------------------------------ inverse: parity
def _vals(b):
    return np.array([float(b.a), float(b.b), float(b.c), float(b.k), float(b.delta), float(b.v), float(b.e0),
                     *b.g.detach().numpy(), *b.weights.detach().numpy()])


def _np_vals(s):
    w = s.weights
    return np.array([s.a, s.b, s.c, s.k, s.delta, s.v, s.e0, s.gx, s.gy, s.gz, w["xy"], w["yz"], w["xz"]])


@pytest.mark.parametrize("g", TET_G, ids=str)
@pytest.mark.parametrize("standardize", [True, False])
def test_tetragonal_from_g_parity_every_fixture(g, standardize):
    ref = E.tetragonal_from_g(*g, standardize=standardize)
    got = td.tetragonal_from_g(*g, standardize=standardize)
    assert len(ref) == len(got)
    for s, b in zip(ref, got):
        if not (s.converged and s.delta is not None):
            assert not isinstance(b, td.TetragonalBranch) and b.label == s.label
            continue
        assert b.numpy.label == s.label and b.numpy.ground_doublet == s.ground_doublet
        close(_vals(b), _np_vals(s))
        close(np.sort(np.abs(b.g.detach().numpy())), np.sort(np.abs(g)), rtol=1e-9)


def test_tetragonal_from_g_parity_mode_all():
    g = TET_G[0]
    ref, got = E.tetragonal_from_g(*g, mode="all"), td.tetragonal_from_g(*g, mode="all")
    assert len(ref) == len(got) > 2
    for s, b in zip(ref, got):
        close(_vals(b), _np_vals(s))


def test_tetragonal_from_g_rejects_batches_and_bad_order():
    with pytest.raises(ValueError):
        td.tetragonal_from_g(torch.tensor([2.6, 2.5]), 2.3, 1.6)
    with pytest.raises(ValueError):
        td.tetragonal_from_g(1.6, 2.38, 2.62)                          # McGarvey mode needs g1 >= g2 >= g3


@pytest.mark.parametrize("case", TRI_G, ids=str)
def test_trigonal_from_g_parity_every_fixture(case):
    gpar, gperp, mode, val, phi = case
    ref = E.trigonal_from_g(gpar, gperp, twist_deg=phi, **_tri_kw(mode, val))
    got = td.trigonal_from_g(gpar, gperp, twist=phi, degrees=True, **_tri_kw(mode, val))
    got_rad = td.trigonal_from_g(gpar, gperp, twist=math.radians(phi), **_tri_kw(mode, val))
    for s, b, br in zip(ref, got, got_rad):
        if not (s.converged and s.delta is not None):
            assert not isinstance(b, td.TrigonalBranch) and b.label == s.label
            continue
        close([b.at, b.bt, b.k, b.beta, b.g_perp, b.g_par, b.delta, b.epi],
              [s.at, s.bt, s.k, s.beta, s.g_perp, s.g_par, s.delta, s.epi])
        close(br.epi, s.epi)


# ------------------------------------------------------------------------------ inverse: gradients
def _np_branch(g, i, **kw):
    s = E.tetragonal_from_g(*g, **kw)[i]
    return np.array([s.a, s.b, s.c, s.k, s.delta, s.v, s.e0])


def _t_branch(b):
    return torch.stack([b.a, b.b, b.c, b.k, b.delta, b.v, b.e0])


@pytest.mark.parametrize("g", TET_G[:6], ids=str)
def test_tetragonal_inverse_gradients_vs_numpy_fd_each_branch(g):
    gt = torch.tensor(g, dtype=F64, requires_grad=True)
    for i, b in enumerate(td.tetragonal_from_g(*gt.unbind())):
        if not isinstance(b, td.TetragonalBranch):
            continue
        J = rev_jacobian(_t_branch(b), gt)
        assert best_fd(lambda x: _np_branch(x, i), g, J) < 1e-6, b.numpy.label
        cols = []                                                            # forward mode (dual numbers)
        for j in range(3):
            with fwAD.dual_level():
                gd = fwAD.make_dual(torch.tensor(g, dtype=F64), torch.eye(3, dtype=F64)[j])
                cols.append(fwAD.unpack_dual(_t_branch(td.tetragonal_from_g(*gd.unbind())[i])).tangent)
        np.testing.assert_allclose(torch.stack(cols, -1).numpy(), J, atol=1e-12 * np.abs(J).max())


def test_tetragonal_inverse_gradients_mode_all_each_branch():
    g = TET_G[0]
    cache = {}

    def npall(x):
        key = tuple(np.round(x, 15))
        if key not in cache:
            cache[key] = E.tetragonal_from_g(*x, mode="all")
        return cache[key]

    gt = torch.tensor(g, dtype=F64, requires_grad=True)
    for i, b in enumerate(td.tetragonal_from_g(*gt.unbind(), mode="all")):
        J = rev_jacobian(_t_branch(b), gt)
        err = best_fd(lambda x: (lambda s: np.array([s.a, s.b, s.c, s.k, s.delta, s.v, s.e0]))(npall(x)[i]), g, J,
                      steps=(1e-5, 1e-6))
        assert err < 1e-6, (i, err)


@pytest.mark.parametrize("seed", range(6))
def test_tetragonal_inverse_is_inverse_of_forward_jacobian(seed):
    """d(Delta, V, k)/d(g) of the branch that recovers the truth = (d g/d(Delta, V, k))^-1 (exact Hamiltonian)."""
    rng = np.random.default_rng(seed)
    d = rng.uniform(1.5, 6) * rng.choice([-1, 1])
    v, k = rng.uniform(-0.6, 0.6) * d, rng.uniform(0.7, 1.1)
    x = torch.tensor([d, v, k], dtype=F64)
    fwd = lambda z: td.solve_t2g_hole(td.tetragonal_cf(z[0], z[1]), z[2]).g_principal.flip(-1)
    Jf = torch.func.jacrev(fwd)(x).numpy()
    g = fwd(x).numpy().copy()
    gt = torch.tensor(g, requires_grad=True)
    hits = [b for b in td.tetragonal_from_g(*gt.unbind()) if isinstance(b, td.TetragonalBranch)
            and abs(b.delta.item() - d) + abs(abs(b.v.item()) - abs(v)) + abs(b.k.item() - k) < 1e-7]
    assert hits
    b = hits[0]
    J = rev_jacobian(torch.stack([b.delta, b.v, b.k]), gt)
    if np.sign(b.v.item()) != np.sign(v):                                 # x<->y relabeled: V -> -V
        J[1] *= -1
    np.testing.assert_allclose(J @ Jf, np.eye(3), atol=1e-9)


def test_tetragonal_inverse_axial_g_derivatives():
    """g1 = g2: V = 0 and finite derivatives with the x<->y symmetry; one-sided NumPy differences agree."""
    g = [2.5, 2.5, 1.7]
    gt = torch.tensor(g, dtype=F64, requires_grad=True)
    for i, b in enumerate(td.tetragonal_from_g(*gt.unbind())):
        J = rev_jacobian(torch.stack([b.delta, b.v, b.k]), gt)
        assert np.isfinite(J).all() and abs(b.v.item()) < 1e-12
        np.testing.assert_allclose(J[[0, 2], 0], J[[0, 2], 1], rtol=1e-9)
        np.testing.assert_allclose(J[1, 0], -J[1, 1], rtol=1e-9)
        Jf = fd_jacobian(lambda x: _np_branch(x, i)[[4, 5, 3]], np.array(g), np.full(3, 1e-7),
                         lower=np.array([g[0], -9, -9]), upper=np.array([9, g[1], 9]))
        assert np.abs(J - Jf).max() < 1e-5 * np.abs(J).max(), b.numpy.label


@pytest.mark.parametrize("case", TRI_G, ids=str)
def test_trigonal_inverse_gradients_vs_numpy_fd(case):
    """Including d/d(beta), d/dk and d/d(twist) (radians)."""
    gpar, gperp, mode, val, phi = case
    fixed = E.BETA_OCT if mode == "B" else val
    x0 = np.array([gpar, gperp, fixed, math.radians(phi)])

    def kw(x):
        return {"fix": "beta", "beta": x[2]} if mode == "B" else {"fix": "k", "k": x[2]}

    xt = torch.tensor(x0, requires_grad=True)
    for i, b in enumerate(td.trigonal_from_g(xt[0], xt[1], twist=xt[3], **kw(xt))):
        if not isinstance(b, td.TrigonalBranch):
            continue

        def npf(x):
            s = E.trigonal_from_g(x[0], x[1], twist_deg=math.degrees(x[3]), **kw(x))[i]
            return np.array([s.at, s.bt, s.k, s.beta, s.delta, s.epi])

        J = rev_jacobian(torch.stack([b.at, b.bt, b.k, b.beta, b.delta, b.epi]), xt)
        assert best_fd(npf, x0, J) < 1e-6, b.numpy.label


@pytest.mark.parametrize("seed", range(4))
def test_trigonal_inverse_is_inverse_of_forward_jacobian(seed):
    rng = np.random.default_rng(100 + seed)
    d, k = rng.uniform(1.0, 5.0) * rng.choice([-1, 1]), rng.uniform(0.7, 1.1)
    c3 = torch.ones(3, dtype=F64) / 3**0.5

    def fwd(z):
        s = td.solve_t2g_hole(td.trigonal_cf(z[0]), z[1])
        g_par = torch.sqrt(c3 @ s.ggT @ c3)
        return torch.stack([g_par, torch.sqrt((s.ggT.trace() - g_par**2) / 2)])

    x = torch.tensor([d, k], dtype=F64)
    Jf = torch.func.jacrev(fwd)(x).numpy()
    g = fwd(x).detach().numpy()
    gt = torch.tensor(g, requires_grad=True)
    hits = [b for b in td.trigonal_from_g(gt[0], gt[1]) if isinstance(b, td.TrigonalBranch)
            and abs(b.delta.item() - d) + abs(b.k.item() - k) < 1e-7]
    assert hits
    J = rev_jacobian(torch.stack([hits[0].delta, hits[0].k]), gt)
    np.testing.assert_allclose(J @ Jf, np.eye(2), atol=1e-9)


def test_implicit_attach_gradcheck():
    """The implicit-function attachment is exact (second-order FD) in both modes."""
    from scipy.optimize import fsolve

    def F(x, th):
        return torch.stack([x[0] ** 3 + th[0] * x[1] - th[1], torch.sin(x[1]) + x[0] * th[0] - 0.3])

    def solve(th):
        thd = th.detach().numpy()
        r = fsolve(lambda x: F(torch.tensor(x), torch.tensor(thd)).numpy(), [0.5, 0.2], xtol=1e-13)
        return td.implicit_attach(F, torch.tensor(r), th)

    th = torch.tensor([0.7, 1.1], dtype=F64, requires_grad=True)
    assert torch.autograd.gradcheck(solve, (th,), eps=1e-6, atol=1e-7, check_forward_ad=True)
