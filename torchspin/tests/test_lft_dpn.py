"""torchspin.lft.dpn: complete-basis p^n levels and g (port of telser_tools_diff.dpn).

Self-contained checks (closed forms, unit consistency, gradients against central differences
and gradcheck with forward AD), and parity with telser-tools (engine copy bit-identical; the 27
DPN reference runs, exact and as DPN/DPNFIT compute them; ligand fields; doublet g; gradients
against central differences of the NumPy oracle) when telser-tools is installed.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import torch

from torchspin import lft
from torchspin.lft import dpn
from torchspin.lft._engines import pn
from torchspin.lft.units import energy_factor

F64 = torch.float64
CM_TO_MHZ = energy_factor("cm^-1", "MHz")
DATA = Path(__file__).parent / "data" / "telser"


@pytest.fixture(autouse=True)
def _one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def _cm(n, **kw):
    return dpn.levels(n, units="cm^-1", **kw)


# ------------------------------------------------------------------------------ closed forms
def test_basis_sizes():
    assert [dpn.pn_operators(n).size for n in range(7)] == [1, 6, 15, 20, 15, 6, 1]
    with pytest.raises(ValueError):
        dpn.pn_operators(7)


def test_p1_crystal_field_orbital_energies():
    """E(px) = Ds - Dr, E(py) = Ds + Dr, E(pz) = -2 Ds, each a Kramers pair."""
    E = _cm(1, V=dpn.crystal_field(300.0, -70.0))
    want = sorted([300.0 + 70.0] * 2 + [300.0 - 70.0] * 2 + [-600.0] * 2)
    np.testing.assert_allclose(E.numpy(), want, atol=1e-9)


def test_p1_spin_orbit_splitting_is_three_halves_zeta():
    E = _cm(1, zeta=400.0)
    np.testing.assert_allclose(E.numpy(), [-400.0] * 2 + [200.0] * 4, atol=1e-9)    # 2P1/2, 2P3/2


def test_p2_free_ion_terms():
    """3P = -3/25 F2, 1D = +3/25 F2, 1S = +12/25 F2 (DPN), Trees alpha L(L+1)."""
    F2, a = 46000.0, 50.0
    E = _cm(2, F2=F2, alpha=a)
    want = sorted([-3 / 25 * F2 + 2 * a] * 9 + [3 / 25 * F2 + 6 * a] * 5 + [12 / 25 * F2])
    np.testing.assert_allclose(E.numpy(), want, atol=1e-8)
    E3 = _cm(3, F2=F2, relative=True)                                   # 2D - 4S = 9 F2/25 (p3)
    assert int(((E3 - 9 / 25 * F2).abs() < 1e-8).sum()) == 10
    EA = _cm(2, F2=F2, A=1000.0)
    torch.testing.assert_close(EA, _cm(2, F2=F2) + 1000.0, rtol=0, atol=1e-9)    # A n(n-1)/2, one pair


def test_p_orbitals_are_not_split_by_an_octahedron():
    V = dpn.aom(lft.octahedron(6000.0, 1000.0))
    torch.testing.assert_close(V, (2 * 6000.0 + 4 * 1000.0) * torch.eye(3, dtype=F64), rtol=0, atol=1e-9)
    V = dpn.aom(lft.tetrahedron(6000.0, 1000.0))
    torch.testing.assert_close(V, (4 / 3 * 6000.0 + 8 / 3 * 1000.0) * torch.eye(3, dtype=F64), rtol=0, atol=1e-9)


def test_aom_constructors_agree_and_delta_is_refused():
    ligs = [lft.Ligand.isotropic(0.3, 0.2, 1000.0, 300.0), lft.Ligand.isotropic(1.9, 3.5, 800.0, 150.0)]
    nvec = torch.tensor([[math.sin(l.theta) * math.cos(l.phi), math.sin(l.theta) * math.sin(l.phi), math.cos(l.theta)]
                         for l in ligs], dtype=F64)
    torch.testing.assert_close(dpn.aom(ligs), dpn.aom_isotropic(nvec, torch.tensor([1000.0, 800.0], dtype=F64),
                                                                  torch.tensor([300.0, 150.0], dtype=F64)),
                               rtol=0, atol=1e-10)
    lig = lft.Ligand.from_degrees(30.0, 20.0, 10.0, e_sigma=1000.0, e_pi_x=300.0, e_pi_y=200.0)
    th = torch.tensor([30.0], dtype=F64)
    v1 = dpn.aom_from_angles(th, th * 2 / 3, th / 3, torch.tensor([[1000.0, 300.0, 200.0]], dtype=F64), degrees=True)
    torch.testing.assert_close(dpn.aom([lig]), v1, rtol=0, atol=1e-10)
    with pytest.raises(ValueError, match="delta"):
        dpn.aom([lft.Ligand(0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 50.0)])


def test_crystal_field_and_orbital_energies_batch():
    V = dpn.crystal_field(torch.tensor([100.0, 200.0], dtype=F64), 10.0)
    assert V.shape == (2, 3, 3)
    torch.testing.assert_close(torch.diagonal(V[1]), torch.tensor([190.0, 210.0, -400.0], dtype=F64))
    E = _cm(1, V=V)
    assert E.shape == (2, 6)
    torch.testing.assert_close(dpn.orbital_energies([1.0, 2.0, 3.0]), torch.diag(torch.tensor([1.0, 2.0, 3.0],
                                                                                             dtype=F64)))


def test_to_m_basis_is_unitary_and_crystal_field_matches_vl():
    V = dpn.crystal_field(300.0, -70.0)
    vm = dpn.to_m_basis(V)
    np.testing.assert_allclose(np.diag(vm.numpy()).real, [300.0, -600.0, 300.0], atol=1e-12)   # Ds, -2Ds, Ds
    assert abs(complex(vm[2, 0]) - (-70.0)) < 1e-12 or abs(complex(vm[2, 0]) - 70.0) < 1e-12   # |<1|V|-1>| = |Dr|


def test_p1_free_ion_g_is_lande_and_free_electron_without_soc():
    g = dpn.g_tensor(1, zeta=500.0, units="cm^-1")                         # 2P1/2: g_J = (2 - g_e/2) / 1.5
    ge = dpn.TORCHSPIN_CONSTANTS.g_e
    np.testing.assert_allclose(g.principal.numpy(), [abs(2 - ge / 2) / 1.5] * 3, rtol=1e-10)
    assert float(g.gap) == pytest.approx(750.0, rel=1e-12)
    g0 = dpn.g_tensor(1, V=dpn.crystal_field(-500.0, 100.0), units="cm^-1")   # orbital singlet ground, no SOC
    np.testing.assert_allclose(g0.principal.numpy(), [ge] * 3, rtol=1e-12)


def test_p1_axial_g_orders():
    """p1, axial field with (px, py) lowest (Ds < 0) and SOC: axial g (g_x = g_y), g = (g g^T)^(1/2)."""
    g = dpn.g_tensor(1, zeta=300.0, V=dpn.crystal_field(-2000.0), k=(0.9, 0.9, 0.9), units="cm^-1")
    gx, gy, gz = g.principal.numpy()
    assert gx == pytest.approx(gy, rel=1e-9)
    torch.testing.assert_close(g.g @ g.g, g.ggT, rtol=1e-12, atol=1e-12)


def test_zeeman_free_spin_p1():
    """p1, zeta = 0, orbital singlet ground: splitting g_e mu_B B."""
    B = 500.0                                                         # mT
    E = _cm(1, V=dpn.crystal_field(3000.0), field=(0.0, 0.0, B))
    c = dpn.TORCHSPIN_CONSTANTS
    assert float(E[1] - E[0]) == pytest.approx(c.g_e * c.mu_b * B, rel=1e-10)
    Ev = _cm(1, V=dpn.crystal_field(3000.0), field=dpn.field_vector(B, math.pi / 2, 0.3))
    assert float(Ev[1] - Ev[0]) == pytest.approx(c.g_e * c.mu_b * B, rel=1e-10)


def test_field_vector_degrees():
    torch.testing.assert_close(dpn.field_vector(2.0, 90.0, 90.0, degrees=True),
                               torch.tensor([0.0, 2.0, 0.0], dtype=F64), rtol=0, atol=1e-15)


# ------------------------------------------------------------------------------ units
def test_units_mhz_vs_cm1_consistent():
    kw = dict(F2=41000.0, zeta=350.0, alpha=30.0, A=500.0)
    V = dpn.crystal_field(900.0, 140.0)
    f = (0.0, 100.0, 300.0)
    E_cm = dpn.levels(2, V=V, units="cm^-1", field=f, **kw)
    E_mhz = dpn.levels(2, V=V * CM_TO_MHZ, field=f, **{k: v * CM_TO_MHZ for k, v in kw.items()})
    torch.testing.assert_close(E_mhz, E_cm * CM_TO_MHZ, rtol=1e-12, atol=1e-3)
    kK = energy_factor("cm^-1", "K")
    E_K = dpn.levels(2, V=V * kK, units="K", field=f, **{k: v * kK for k, v in kw.items()})
    torch.testing.assert_close(E_K, E_cm * kK, rtol=1e-12, atol=1e-9)
    g_cm = dpn.g_tensor(1, zeta=400.0, V=V, units="cm^-1")
    g_mhz = dpn.PnSystem(1).g_tensor(zeta=400.0 * CM_TO_MHZ, V=V * CM_TO_MHZ)
    torch.testing.assert_close(g_mhz.ggT, g_cm.ggT, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(g_mhz.gap, g_cm.gap * CM_TO_MHZ, rtol=1e-12, atol=0)
    h1 = dpn.dpn_hamiltonian(4, F2=41000.0, zeta=500.0, V=V, units="cm^-1", lf_type="X")
    h2 = dpn.dpn_hamiltonian(4, F2=41000.0 * CM_TO_MHZ, zeta=500.0 * CM_TO_MHZ, V=V * CM_TO_MHZ, lf_type="X")
    torch.testing.assert_close(h2, h1 * CM_TO_MHZ, rtol=1e-12, atol=1e-6)


def test_g_tensor_refuses_field_and_unknown_keywords():
    with pytest.raises(ValueError):
        dpn.g_tensor(1, zeta=1.0, field=(0, 0, 1.0))
    with pytest.raises(TypeError):
        dpn.g_tensor(1, B=1.0)
    with pytest.raises(ValueError):
        dpn.dpn_hamiltonian(2, program="ddn")


# ------------------------------------------------------------------------------ DPN's matrix (self-contained)
@pytest.mark.parametrize("n", [4, 5])
def test_hole_route_equals_direct_pn_in_crystal_field(n):
    """DPN's p^(6-n) hole problem (-V, -zeta) gives the direct p^n levels up to a constant."""
    kw = dict(F2=41000.0, zeta=600.0, alpha=25.0, V=dpn.crystal_field(900.0, 140.0), units="cm^-1")
    direct = _cm(n, **{k: v for k, v in kw.items() if k != "units"}, relative=True)
    for prog in ("dpn", "dpnfit"):
        E = torch.linalg.eigvalsh(dpn.dpn_hamiltonian(n, lf_type="X", program=prog, **kw))
        torch.testing.assert_close(E - E[0], direct, rtol=0, atol=1e-8)


def test_internal_m_matrix_drops_o21_for_aom_only():
    V = dpn.aom([lft.Ligand.from_degrees(30.0, 20.0, 10.0, e_sigma=1000.0, e_pi_x=300.0, e_pi_y=200.0)])
    vm = dpn.internal_m_matrix(V, "A", True, 3)
    assert float(vm[2, 1].abs()) == 0.0 and float(vm[1, 0].abs()) == 0.0
    torch.testing.assert_close(dpn.internal_m_matrix(V, "X"), dpn.to_m_basis(V))
    torch.testing.assert_close(dpn.internal_m_matrix(V, "A", False, 1), dpn.to_m_basis(V).conj())


# ------------------------------------------------------------------------------ gradients (self-contained)
def _cd(fun, x, rels=(1e-3, 1e-4, 1e-5)):
    out = []
    for r in rels:
        cols = []
        for j in range(len(x)):
            h = r * max(abs(float(x[j])), 1.0)
            e = torch.zeros_like(x)
            e[j] = h
            cols.append((fun(x + e) - fun(x - e)) / (2 * h))
        out.append(torch.stack(cols, -1))
    return out


def _assert_grad(J, fds, tol=1e-6):
    J = J.detach()
    err = torch.stack([(J - f).abs().amax(0) for f in fds]).amin(0)
    scale = float(J.abs().max())
    assert bool((err <= tol * scale + 1e-6 * J.abs().amax(0)).all()), (err, scale)


def _kw(x):
    return dict(F2=x[0], zeta=x[1], alpha=x[2], V=dpn.crystal_field(x[3], x[4]), k=x[5:8])


def test_levels_gradient_vs_central_differences_and_gradcheck():
    x0 = torch.tensor([42000.0, 350.0, 30.0, 900.0, 140.0, 0.9, 0.85, 0.8], dtype=F64)
    f = torch.tensor([30.0, 120.0, 250.0], dtype=F64)
    fun = lambda x: dpn.levels(2, field=f, units="cm^-1", **_kw(x))  # noqa: E731
    J = torch.func.jacfwd(fun)(x0)
    _assert_grad(J, _cd(fun, x0))
    # gradcheck (reverse and forward AD) on a well-separated spectrum of O(1) cm^-1 levels
    fs = torch.tensor([300.0, 1200.0, 2500.0], dtype=F64)
    xs = torch.tensor([4.2, 0.9, 0.3, 1.1, 0.4, 0.9, 0.85, 0.8], dtype=F64, requires_grad=True)
    for n in (1, 2):
        g = lambda x, n=n: dpn.levels(n, field=fs, units="cm^-1", **_kw(x))  # noqa: E731
        assert torch.autograd.gradcheck(g, (xs,), eps=1e-6, atol=1e-6, rtol=1e-6, check_forward_ad=True)


def test_kramers_g_gradient_self_contained():
    def gg(x):
        V = dpn.aom_from_angles(x[[3, 5]], x[[4, 6]], torch.zeros(2, dtype=F64),
                                torch.tensor([[2000.0, 300.0, 100.0], [1500.0, 200.0, 400.0]], dtype=F64))
        return dpn.g_tensor(1, zeta=x[0], V=V, k=torch.stack([x[1], x[1], x[2]]), units="cm^-1").ggT.reshape(-1)

    x0 = torch.tensor([400.0, 0.9, 0.95, 0.35, 0.17, 1.92, 1.4], dtype=F64)
    J = torch.func.jacfwd(gg)(x0)
    _assert_grad(J, _cd(gg, x0))
    xr = x0.clone().requires_grad_(True)
    (gr,) = torch.autograd.grad(gg(xr).sum(), xr)
    torch.testing.assert_close(gr, J.sum(0), rtol=1e-9, atol=1e-12)


def test_degenerate_level_derivative_is_level_mean():
    """p1 at zero field, Dr = 0: Kramers and M_L degeneracies; every member carries its level's derivative."""
    x0 = torch.tensor([0.0, 350.0, 0.0, 900.0, 0.0, 1.0, 1.0, 1.0], dtype=F64)
    fun = lambda x: dpn.levels(1, units="cm^-1", **_kw(x))  # noqa: E731
    J = torch.func.jacfwd(fun)(x0)
    E = fun(x0)
    for i in range(0, 6, 2):
        assert float((E[i + 1] - E[i]).abs()) < 1e-9
        torch.testing.assert_close(J[i], J[i + 1], rtol=0, atol=1e-10)
    # level means are smooth: compare with central differences of the pair means
    mean = lambda x: fun(x).reshape(3, 2).mean(-1)  # noqa: E731
    _assert_grad(torch.func.jacfwd(mean)(x0), _cd(mean, x0))


# ------------------------------------------------------------------------------ parity with telser-tools
@lru_cache(maxsize=None)
def _ref():
    return json.loads((DATA / "dpn_inputs.json").read_text())["cases"]


def _c(C):
    return dpn.Constants(mu_b=C.mu_b * 10.0, g_e=C.g_e, k_b=C.k_b)          # cm^-1/G -> cm^-1/mT


def test_engine_copy_is_bit_identical():
    engine = pytest.importorskip("telser_tools.dpn.engine")
    terms = pytest.importorskip("telser_tools.dpn.terms")
    lf = pytest.importorskip("telser_tools.dpn.ligand_field")
    assert np.array_equal(pn.U_REAL, lf.U_REAL)
    for n in range(7):
        a, b = pn.operators(n), engine.operators(n)
        assert a.dets == b.dets
        for f in ("coulomb", "excitation", "so", "L", "S", "L2", "S2", "n_orb"):
            assert np.array_equal(getattr(a, f), getattr(b, f)), (n, f)
        ta, pa = pn.term_projectors(n)
        tb, pb = terms.term_projectors(n)
        assert [t.label for t in ta] == [t.label for t in tb]
        np.testing.assert_allclose(pa, pb, rtol=0, atol=1e-12)


def test_dpn_constants_match_telser():
    h = pytest.importorskip("telser_tools.dpn.hamiltonian")
    c = _c(h.DPN_CONSTANTS)
    assert dpn.DPN_CONSTANTS.mu_b == pytest.approx(c.mu_b, rel=1e-15)
    assert dpn.DPN_CONSTANTS.g_e == c.g_e and dpn.DPN_CONSTANTS.k_b == c.k_b


_RUNS = [k for k, c in _ref().items() if c["returncode"] == 0]


@pytest.mark.parametrize("name", _RUNS)
def test_reference_runs_exact_and_dpn_match_numpy(name):
    tdpn = pytest.importorskip("telser_tools.dpn")
    from telser_tools.dpn import compat
    from telser_tools.dpn.hamiltonian import DPN_CONSTANTS
    from telser_tools.dpn.inputfile import parse_input

    d = parse_input(_ref()[name]["stdin"])
    p = d.parameters()
    fields = [None] if d.H0 <= 0 else [d.H0 * np.eye(3)[i] for i in range(3)]
    for f in fields:
        fm = None if f is None else f / 10.0                           # G -> mT
        want = tdpn.PnSystem(d.n, DPN_CONSTANTS).solve(p, f).energies
        got = dpn.levels(d.n, F2=p.F2, zeta=p.zeta, alpha=p.alpha, A=p.A, V=p.ligand_field, k=p.k, field=fm,
                         units="cm^-1", constants=_c(DPN_CONSTANTS)).numpy()
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-9)
        for prog in ("dpn", "dpnfit"):
            want = compat.dpn_solution(d.n, p, f, d.lf_type, prog).energies
            h = dpn.dpn_hamiltonian(d.n, F2=p.F2, zeta=p.zeta, V=p.ligand_field, k=p.k, alpha=p.alpha, field=fm,
                                    lf_type=d.lf_type, program=prog, units="cm^-1", constants=_c(DPN_CONSTANTS))
            np.testing.assert_allclose(torch.linalg.eigvalsh(h).numpy(), want, rtol=1e-10, atol=1e-9)


def test_ligand_field_constructors_match_numpy():
    lf = pytest.importorskip("telser_tools.dpn.ligand_field")
    from telser_tools.dpn import compat

    ligs = [lf.Ligand(30, 20, 10, 1000, 300, 200), lf.Ligand(100, 200, 0, 800, 100, 150)]
    mine = [lft.Ligand.from_degrees(l.theta, l.phi, l.psi, e_sigma=l.e_sigma, e_pi_x=l.e_pi_x, e_pi_y=l.e_pi_y)
            for l in ligs]
    v = lf.aom(ligs)
    np.testing.assert_allclose(dpn.aom(mine).numpy(), v, atol=1e-12)
    for lig in (lf.octahedron(900, 200), lf.tetrahedron(900, 200)):
        n = np.array([[np.sin(np.radians(l.theta)) * np.cos(np.radians(l.phi)),
                       np.sin(np.radians(l.theta)) * np.sin(np.radians(l.phi)), np.cos(np.radians(l.theta))]
                      for l in lig])
        np.testing.assert_allclose(dpn.aom_isotropic(torch.tensor(n), 900.0, 200.0).numpy(), lf.aom(lig), atol=1e-10)
    np.testing.assert_allclose(dpn.crystal_field(300.0, -70.0).numpy(), lf.crystal_field(300, -70), atol=0)
    np.testing.assert_allclose(dpn.to_m_basis(torch.tensor(v)).numpy(), lf.to_m_basis(v), atol=1e-12)
    for np_ in (1, 2, 3):
        np.testing.assert_allclose(dpn.internal_m_matrix(torch.tensor(v), "A", True, np_).numpy(),
                                   compat.internal_m_matrix(v, "A", True, np_), atol=1e-12)


@pytest.mark.parametrize("n", [1, 5])   # p3: 4S ground, doublet gap ~0.03 cm^-1 (ill-conditioned in both codes)
def test_g_tensor_matches_numpy(n):
    tdpn = pytest.importorskip("telser_tools.dpn")
    from telser_tools.dpn import ligand_field as lf
    from telser_tools.dpn.hamiltonian import CODATA2018

    v = lf.aom([lf.Ligand(20, 10, 5, 2000, 300, 100), lf.Ligand(110, 80, 0, 1500, 200, 400)])
    p = tdpn.Parameters(F2=45000, zeta=400, ligand_field=v, k=(0.9, 0.8, 0.95))
    g_np, gg_np = tdpn.PnSystem(n).solve(p).g_tensor()
    g = dpn.g_tensor(n, F2=p.F2, zeta=p.zeta, V=v, k=p.k, units="cm^-1", constants=_c(CODATA2018))
    np.testing.assert_allclose(g.ggT.numpy(), gg_np, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(g.principal.numpy(), g_np, rtol=1e-10, atol=1e-12)


def _np_fd(fun, x, rels=(1e-3, 1e-4, 1e-5)):
    from telser_tools.fitting import jacobian

    return [jacobian(fun, x, r * np.maximum(np.abs(x), 1.0)) for r in rels]


def _assert_np_grad(J, fds, tol=1e-6):
    err = np.min([np.abs(J - f).max(axis=0) for f in fds], axis=0)
    scale = np.abs(J).max()
    assert np.all(err <= tol * scale + 1e-6 * np.abs(J).max(axis=0)), (err, scale)


def _level_mean_matrix(e, tol=1e-6):
    grp = np.cumsum(np.r_[0, np.diff(e) > tol * np.maximum(1.0, np.abs(e[1:]))])
    return np.array([(grp == g) / np.sum(grp == g) for g in range(grp[-1] + 1)])


@pytest.mark.parametrize("n,field,deg", [(2, (300.0, 1200.0, 2500.0), False), (3, None, True), (1, None, True),
                                         (2, None, True), (4, (0.0, 0.0, 4000.0), False)])
def test_energy_gradients_vs_numpy_including_degenerate_levels(n, field, deg):
    tdpn = pytest.importorskip("telser_tools.dpn")
    from telser_tools.dpn import ligand_field as lf
    from telser_tools.dpn.hamiltonian import CODATA2018

    def params(x):
        return tdpn.Parameters(F2=x[0], zeta=x[1], alpha=x[2], ligand_field=lf.crystal_field(x[3], x[4]),
                               k=tuple(x[5:8]))

    x0 = np.array([42000.0, 350.0, 30.0, 900.0, 0.0 if deg else 140.0, 0.9, 0.85, 0.8])
    sysn = tdpn.PnSystem(n)
    e0 = sysn.energies(params(x0), field)
    M = _level_mean_matrix(e0)
    if deg:
        assert M.shape[0] < len(e0)
    fm = None if field is None else torch.tensor(field, dtype=F64) / 10.0
    J = torch.func.jacfwd(lambda x: dpn.levels(n, field=fm, units="cm^-1", constants=_c(CODATA2018), **_kw(x)))(
        torch.tensor(x0)).numpy()
    assert np.isfinite(J).all()
    Jl = M @ J
    np.testing.assert_allclose(J, (M > 0).T.astype(float) @ Jl, atol=1e-8)
    _assert_np_grad(Jl, _np_fd(lambda x: M @ sysn.energies(params(x), field), x0))


@pytest.mark.parametrize("n", [1, 5])
def test_kramers_g_gradient_vs_numpy(n):
    tdpn = pytest.importorskip("telser_tools.dpn")
    from telser_tools.dpn import ligand_field as lf
    from telser_tools.dpn.hamiltonian import CODATA2018

    def numpy_gg(x):
        v = lf.aom([lf.Ligand(x[4], x[5], 0, 2000, 300, 100), lf.Ligand(x[6], x[7], 0, 1500, 200, 400)])
        p = tdpn.Parameters(F2=x[0], zeta=x[1], ligand_field=v, k=(x[2], x[2], x[3]))
        return tdpn.PnSystem(n).solve(p).g_tensor()[1].ravel()

    def torch_gg(x):
        e = torch.tensor([[2000.0, 300.0, 100.0], [1500.0, 200.0, 400.0]], dtype=F64)
        V = dpn.aom_from_angles(x[[4, 6]], x[[5, 7]], torch.zeros(2, dtype=F64), e, degrees=True)
        return dpn.g_tensor(n, F2=x[0], zeta=x[1], V=V, k=torch.stack([x[2], x[2], x[3]]), units="cm^-1",
                            constants=_c(CODATA2018)).ggT.reshape(-1)

    x0 = np.r_[45000.0, 400.0, 0.9, 0.95, 20.0, 10.0, 110.0, 80.0]
    np.testing.assert_allclose(torch_gg(torch.tensor(x0)).numpy(), numpy_gg(x0), rtol=1e-10, atol=1e-12)
    J = torch.func.jacfwd(torch_gg)(torch.tensor(x0)).numpy()
    _assert_np_grad(J, _np_fd(numpy_gg, x0))


@pytest.mark.parametrize("n,lft_type,prog", [(4, "A", "dpn"), (5, "X", "dpnfit"), (2, "A", "dpnfit")])
def test_dpn_matrix_gradients_vs_numpy(n, lft_type, prog):
    tdpn = pytest.importorskip("telser_tools.dpn")
    from telser_tools.dpn import compat
    from telser_tools.dpn import ligand_field as lf

    f = np.array([0.0, 0.0, 3000.0])

    def V_np(x):
        if lft_type == "A":
            return lf.aom([lf.Ligand(0, 0, 0, x[3], x[4], x[4]), lf.Ligand(90, 0, 0, x[3], x[4], x[4]),
                           lf.Ligand(90, 90, 0, 0.8 * x[3], 0.0, 0.0)])
        return lf.crystal_field(x[3], x[4])

    def V_t(x):
        if lft_type == "A":
            th = torch.tensor([0.0, 90.0, 90.0], dtype=F64)
            ph = torch.tensor([0.0, 0.0, 90.0], dtype=F64)
            z = torch.zeros((), dtype=F64)
            e = torch.stack([torch.stack([x[3], x[4], x[4]]), torch.stack([x[3], x[4], x[4]]),
                             torch.stack([0.8 * x[3], z, z])])
            return dpn.aom_from_angles(th, ph, torch.zeros(3, dtype=F64), e, degrees=True)
        return dpn.crystal_field(x[3], x[4])

    def e_np(x):
        p = tdpn.Parameters(F2=x[0], zeta=x[1], alpha=x[2], ligand_field=V_np(x), k=(0.9, 0.9, x[5]))
        return compat.dpn_solution(n, p, f, lft_type, prog).energies

    def e_t(x):
        k = torch.stack([torch.tensor(0.9, dtype=F64), torch.tensor(0.9, dtype=F64), x[5]])
        return torch.linalg.eigvalsh(dpn.dpn_hamiltonian(
            n, F2=x[0], zeta=x[1], alpha=x[2], V=V_t(x), k=k, field=torch.tensor(f / 10.0), lf_type=lft_type,
            program=prog, units="cm^-1"))

    def e_t_gauge(x):
        from torchspin.lft._linalg import eigvalsh_gauge

        k = torch.stack([torch.tensor(0.9, dtype=F64), torch.tensor(0.9, dtype=F64), x[5]])
        return eigvalsh_gauge(dpn.dpn_hamiltonian(n, F2=x[0], zeta=x[1], alpha=x[2], V=V_t(x), k=k,
                                                  field=torch.tensor(f / 10.0), lf_type=lft_type, program=prog,
                                                  units="cm^-1"))

    x0 = np.array([41000.0, 500.0, 20.0, 1500.0, 350.0, 0.85])
    np.testing.assert_allclose(e_t(torch.tensor(x0)).numpy(), e_np(x0), atol=1e-9)
    M = _level_mean_matrix(e_np(x0))
    J = M @ torch.func.jacfwd(e_t_gauge)(torch.tensor(x0)).numpy()
    _assert_np_grad(J, _np_fd(lambda x: M @ e_np(x), x0))
