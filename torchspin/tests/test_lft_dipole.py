"""torchspin.lft.dipole (dipolar couplings of a 3d Slater orbital), milestone M4.

Self-contained: exact constraints (traceless, symmetric, rotation covariance, C4 symmetry, far
field), brute-force quadrature of the d_xy density at general points, absolute magnitudes from
SI constants, unit consistency, gradients (gradcheck in both modes and central differences).
With telser-tools installed: every fixture of ``tests/data/telser/dipole_fortran.json`` in the
default (corrected) mode, and the reduced tensor against telser-tools' 60-digit Decimal path on
A = 1e-3 .. 1e3, both at telser-tools' CODATA 2018 constants.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from torchspin.constants import BMAGN, GFREE, NMAGN, PLANCK
from torchspin.lft import dipole as tdp
from torchspin.lft._engines import dipole as E
from torchspin.lft.units import energy_factor
from torchspin.nucdata import nucgval

F64 = torch.float64
CASES = json.loads((Path(__file__).parent / "data" / "telser" / "dipole_fortran.json").read_text())["cases"]


def close(a, b, rtol=1e-10):
    a, b = np.asarray(a, float), np.asarray(b, float)
    np.testing.assert_allclose(a, b, rtol=rtol, atol=rtol * max(np.abs(b).max(), 1e-300))


def _fd(f, x, h):
    cols = []
    for j in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        cols.append((f(xp) - f(xm)) / (2 * h))
    return np.array(cols).T


def best_fd(f, x, J, steps):
    err = np.full(J.shape[1], np.inf)
    for h in steps:
        err = np.minimum(err, np.abs(J - _fd(f, np.asarray(x, float), h)).max(0))
    return err.max() / np.abs(J).max()


def rev_jacobian(y, x):
    return torch.stack([torch.autograd.grad(y[i], x, retain_graph=True)[0] for i in range(len(y))]).numpy()


# ------------------------------------------------------------------------------ brute-force quadrature
# Independent of the module: T_ij(P) = int s ds dOmega d_i d_j rho(P + s n) + (4 pi/3) rho(P) delta_ij
# for the normalized 3d Slater d_xy density, zeta = 1 (from telser-tools tests/test_dipole_vs_physics.py).
N2 = 1.0 / ((4 * np.pi / 15) * 720 / 2**7)


def _rho_and_hessian(x, y, z):
    r = np.maximum(np.sqrt(x * x + y * y + z * z), 1e-300)
    e = np.exp(-2 * r)
    f, g = x * x * y * y, (2 * x * y * y, 2 * x * x * y, 0 * x)
    H = ((2 * y * y, 4 * x * y, 0 * x), (4 * x * y, 2 * x * x, 0 * x), (0 * x, 0 * x, 0 * x))
    u = (x / r, y / r, z / r)
    out = np.empty((3, 3) + x.shape)
    for i in range(3):
        for j in range(3):
            d2e = e * (4 * u[i] * u[j] - 2 * ((i == j) - u[i] * u[j]) / r)
            out[i, j] = N2 * (H[i][j] * e - 2 * e * (g[i] * u[j] + g[j] * u[i]) + f * d2e)
    return N2 * f * e, out


def _quadrature_tensor(P, ns=32, nt=64, nphi=96):
    R = np.linalg.norm(P)
    u = P / R
    a = np.array([0, 0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0, 0])
    ez = np.cross(u, a)
    ez /= np.linalg.norm(ez)
    M = np.stack([-u, np.cross(ez, -u), ez], 1)
    edges = sorted({0, R / 2, R, 1.5 * R, 2 * R, *(R + k for k in (2, 4, 8, 16, 30))})
    s, ws = np.polynomial.legendre.leggauss(ns)
    S = np.concatenate([(b - a) / 2 * s + (a + b) / 2 for a, b in zip(edges[:-1], edges[1:])])
    W = np.concatenate([(b - a) / 2 * ws for a, b in zip(edges[:-1], edges[1:])])
    ct, wt = np.polynomial.legendre.leggauss(nt)
    ph = (np.arange(nphi) + 0.5) * 2 * np.pi / nphi
    st = np.sqrt(1 - ct * ct)
    n = M @ np.stack([np.outer(st, np.cos(ph)), np.outer(st, np.sin(ph)), np.outer(ct, np.ones(nphi))]).reshape(3, -1)
    w_ang = np.repeat(wt, nphi) * 2 * np.pi / nphi
    _, h = _rho_and_hessian(*(P[:, None, None] + S[None, :, None] * n[:, None, :]))
    hess = np.einsum("ijsa,s,a->ij", h, W * S, w_ang)
    rho_P, _ = _rho_and_hessian(*P)
    return hess + 4 * np.pi / 3 * rho_P * np.eye(3)


@pytest.mark.parametrize("A,direction", [(1.5, (0.3, 0.8, 0.5)), (3.0, (-0.7, 0.2, 0.6)), (3.0, (1.0, 0.0, 0.0)),
                                         (4.0, (0.0, 0.0, 1.0))])
def test_tensor_matches_brute_force_quadrature(A, direction):
    n = np.array(direction) / np.linalg.norm(direction)
    R = 2 * A / 3
    t_quad = _quadrature_tensor(R * n) * R**3
    t = tdp.sto_tensor_full(A, torch.tensor(n)).numpy()
    assert np.abs(t - t_quad).max() < 3e-4 * np.abs(t).max()


# ------------------------------------------------------------------------------ exact constraints
def _rot(seed):
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=F64, generator=torch.Generator().manual_seed(seed)))
    return q * torch.sign(torch.linalg.det(q))


@pytest.mark.parametrize("A", [0.02, 0.7, 3.0, 17.9, 18.1, 60.0])
def test_rotation_covariance_trace_and_symmetry(A):
    g = torch.Generator().manual_seed(int(A * 10))
    n = torch.randn(3, dtype=F64, generator=g)
    M = torch.randn(3, 3, dtype=F64, generator=g)
    M = M + M.T
    R = _rot(int(A * 10))
    t = tdp.sto_tensor_full(A, n, M)
    t_rot = tdp.sto_tensor_full(A, R @ n, R @ M @ R.T)
    torch.testing.assert_close(t_rot, R @ t @ R.T, rtol=0, atol=1e-13 * float(t.abs().max()))
    assert abs(float(t.trace())) < 1e-13 * float(t.abs().max())
    torch.testing.assert_close(t, t.T, rtol=0, atol=0)


def test_named_orbitals_axial_symmetry_and_far_field():
    c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
    Rz = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1.0]], dtype=F64)
    n = torch.tensor([0.3, -0.5, 0.8], dtype=F64)
    t1 = tdp.sto_tensor_full(4.0, n, tdp.orbital_matrix("x2-y2"))           # d(x2-y2) = d(xy) rotated by 45 deg
    t2 = Rz.T @ tdp.sto_tensor_full(4.0, Rz @ n, tdp.orbital_matrix("xy")) @ Rz
    torch.testing.assert_close(t1, t2, rtol=0, atol=1e-14)
    tz = tdp.sto_tensor_full(4.0, torch.tensor([0, 0, 1.0], dtype=F64), tdp.orbital_matrix("z2"))
    assert abs(float(tz[0, 0] - tz[1, 1])) < 1e-14 and float(tz[0, 1].abs()) < 1e-14
    far = tdp.sto_tensor_full(5e4, n, tdp.orbital_matrix("z2"))
    torch.testing.assert_close(far, tdp.point_dipole_full(n), rtol=0, atol=1e-6)
    t_oop = tdp.sto_tensor(torch.tensor([0.5, 3.0, 20.0], dtype=F64), "out_of_plane")
    torch.testing.assert_close(t_oop[:, 0], t_oop[:, 1], rtol=1e-13, atol=1e-15)    # C4: t_x = t_y on the normal
    close(tdp.sto_tensor(1e5, "in_plane"), [-1.0, -1.0, 2.0], rtol=1e-6)


def test_batched_shapes_and_torch_func():
    A = torch.tensor([[0.5, 2.0], [10.0, 40.0]], dtype=F64)
    assert tdp.sto_tensor(A, "in_plane").shape == (2, 2, 3)
    n = torch.randn(4, 3, dtype=F64, generator=torch.Generator().manual_seed(0))
    assert tdp.sto_tensor_full(torch.tensor(3.0, dtype=F64), n).shape == (4, 3, 3)
    J = torch.func.jacfwd(lambda a: tdp.sto_tensor(a, "out_of_plane"))(torch.tensor(5.0, dtype=F64))
    Jr = torch.func.jacrev(lambda a: tdp.sto_tensor(a, "out_of_plane"))(torch.tensor(5.0, dtype=F64))
    torch.testing.assert_close(J, Jr, rtol=1e-13, atol=0)
    r = torch.tensor([2.0, 2.5, 3.0], dtype=F64)
    h = tdp.hyperfine(9, 29, "14N", r, alpha2=0.8)
    assert h.a_alpha.shape == (3, 3)
    close(h.a_alpha[1], tdp.hyperfine(9, 29, "14N", 2.5, alpha2=0.8).a_alpha, rtol=1e-14)


def test_geometry_validation():
    with pytest.raises(ValueError):
        tdp.sto_tensor(2.0, "nowhere")
    with pytest.raises(ValueError):
        tdp.hyperfine(9, 29, "14N", 2.0, alpha2=1.2)
    with pytest.raises(ValueError):
        tdp.electron_electron(9, 29, -1.0)
    with pytest.raises(ValueError):
        tdp.hyperfine(9, 29, None, 2.0)


# ------------------------------------------------------------------------------ magnitudes and units
def test_point_dipole_magnitudes_from_si_constants():
    """mu0/4pi = 1e-7: 1H at 1 A gives g_e g_N mu_B mu_N/(h r^3) = 79.07 MHz per unit of (3nn - 1);
    two g_e spins at 1 nm give 52.04 MHz (Jeschke, Annu. Rev. Phys. Chem. 2012, 63, 419)."""
    h = tdp.hyperfine(9, 29, "1H", 1.0, g=(GFREE,) * 3)
    unit = 1e-7 * GFREE * float(nucgval("1H")) * BMAGN * NMAGN / (PLANCK * 1e-30) / 1e6
    close(h.point_dipole, [-unit, -unit, 2 * unit], rtol=1e-12)
    assert unit == pytest.approx(79.07, abs=0.01)
    e = tdp.electron_electron(9, 29, 10.0, g=(GFREE,) * 3)
    close(e.point_dipole, [52.04, 52.04, -104.08], rtol=2e-4)
    close(e.ddip_edip_point, [-156.12, 0.0], rtol=2e-4)


@pytest.mark.parametrize("units", ["cm^-1", "GHz", "K", "eV"])
def test_units_are_a_pure_rescaling(units):
    f = energy_factor("MHz", units)
    for fn, kw in ((tdp.hyperfine, {"nucleus": "14N"}), (tdp.electron_electron, {})):
        a = fn(9, 29, r_angstrom=2.2, g=(2.05, 2.06, 2.25), alpha2=0.7, geometry="out_of_plane", **kw)
        b = fn(9, 29, r_angstrom=2.2, g=(2.05, 2.06, 2.25), alpha2=0.7, geometry="out_of_plane", units=units, **kw)
        for x, y in zip(a[3:], b[3:]):
            close(y, x * f, rtol=1e-12)
        close(b.tensor_cm3, a.tensor_cm3, rtol=0)


def test_nucleus_name_uses_torchspin_nucdata():
    a = tdp.hyperfine(9, 29, "15N", 2.0)
    b = tdp.hyperfine(9, 29, None, 2.0, g_n=float(nucgval("15N")))
    close(a.a, b.a, rtol=0)
    assert float(a.a[2]) < 0                                              # g_N(15N) < 0


def test_torchspin_constants_close_to_codata_2018():
    """CODATA 2022 (torchspin) vs CODATA 2018 (telser-tools): mu_B, mu_N, a0 differ by ~1e-8 relative."""
    c = tdp.TORCHSPIN_CONSTANTS
    ref = dict(g_e=2.00231930436256, beta_n=5.0507837461e-24, beta_e=9.2740100783e-21, h=6.62607015e-27,
               a0_cm=5.29177210903e-9, mhz_to_cm=1e6 / 2.99792458e10)
    for k, v in ref.items():
        assert getattr(c, k) == pytest.approx(v, rel=5e-8), k


# ------------------------------------------------------------------------------ gradients
@pytest.mark.parametrize("A", [0.3, 17.99, 18.01])
def test_gradcheck_direction_orbital_and_A(A):
    g = torch.Generator().manual_seed(7)
    n = torch.randn(3, dtype=F64, generator=g).requires_grad_()
    M = torch.randn(3, 3, dtype=F64, generator=g).requires_grad_()
    a = torch.tensor(A, dtype=F64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a_, n_, m_: tdp.sto_tensor_full(a_, n_, m_), (a, n, M),
                                    eps=1e-6, atol=1e-7, rtol=1e-6, check_forward_ad=True)


def test_gradcheck_through_the_pole_direction():
    """Directions along the orbital axes (a node or the C4 axis) are not special: smooth there."""
    for d in ([0, 0, 1.0], [1.0, 0, 0], [0.5**0.5, 0.5**0.5, 0]):
        n = torch.tensor(d, dtype=F64, requires_grad=True)
        assert torch.autograd.gradcheck(lambda n_: tdp.sto_tensor_full(2.5, n_), (n,), eps=1e-6, atol=1e-8,
                                        check_forward_ad=True)


def _t_hyperfine(x, geometry, units="MHz"):
    r, zs, gy, gx, gz, al, gn = x.unbind()
    h = tdp.hyperfine(9, 29, None, r, torch.stack([gy, gx, gz]), al, geometry, zstar=zs, g_n=gn, units=units)
    return torch.cat([h.a, h.a_g, h.a_alpha, h.point_dipole])


@pytest.mark.parametrize("geometry", tdp.GEOMETRIES)
def test_hyperfine_gradients_vs_central_differences(geometry):
    """In (r, Z*, g_y, g_x, g_z, alpha1^2, g_N); the differences are of the float64 module itself."""
    x0 = np.array([2.0, 13.201, 2.05, 2.06, 2.25, 0.75, 0.40376100])
    xt = torch.tensor(x0, requires_grad=True)
    J = rev_jacobian(_t_hyperfine(xt, geometry), xt)
    f = lambda z: _t_hyperfine(torch.tensor(z), geometry).numpy()
    assert best_fd(f, x0, J, [1e-4, 1e-5, 1e-6]) < 1e-7
    Jf = torch.func.jacfwd(lambda z: _t_hyperfine(z, geometry))(xt.detach()).numpy()
    np.testing.assert_allclose(Jf, J, atol=1e-13 * np.abs(J).max())


def test_electron_electron_gradients_vs_central_differences():
    x0 = np.array([4.0, 2.05, 2.06, 2.25, 0.8, 2.0023])

    def tf(x, units="cm^-1"):
        r, gy, gx, gz, al, gp = x.unbind()
        e = tdp.electron_electron(9, 29, r, torch.stack([gy, gx, gz]), al, "in_plane", partner_g=gp, units=units)
        return torch.cat([e.d_g, e.d_alpha, e.point_dipole, e.ddip_edip])

    xt = torch.tensor(x0, requires_grad=True)
    J = rev_jacobian(tf(xt), xt)
    assert best_fd(lambda z: tf(torch.tensor(z)).numpy(), x0, J, [1e-4, 1e-5, 1e-6]) < 1e-7


# ------------------------------------------------------------------------------ parity with telser-tools
def _telser():
    pytest.importorskip("telser_tools.dipole")
    from telser_tools import dipole
    from telser_tools.dipole import tensor, zeff
    from telser_tools.dipole.constants import CODATA2018
    from telser_tools.dipole.coupling import _nucleus
    from telser_tools.nuclear import Isotope
    return dipole, tensor, zeff, CODATA2018, _nucleus, Isotope


def test_zstar_copy_identical_to_telser_tools():
    _, _, zeff, *_ = _telser()
    assert E.TABULATED == zeff.TABULATED
    for z in range(1, 60):
        for n_d in range(11):
            a, b = E.clementi_raimondi(n_d, z), zeff.clementi_raimondi(n_d, z)
            assert (a.value, a.method, a.notes) == (b.value, b.method, b.notes)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_fixture_parity_default_mode(case):
    dipole, _, _, C18, _nucleus, Isotope = _telser()
    i = case["input"]
    nuc = i["nucleus"]
    if isinstance(nuc, list):
        mass, name, spin, gn = nuc
        nuc = Isotope(mass, name, spin, 0.0 if gn is None else gn)
    if not 0.0 <= i["alpha2"] <= 1.0:
        with pytest.raises(ValueError):
            dipole.hyperfine(i["n_d"], i["z"], nuc, 2.0, alpha2=i["alpha2"])
        with pytest.raises(ValueError):
            tdp.hyperfine(i["n_d"], i["z"], None, 2.0, alpha2=i["alpha2"], g_n=1.0)
        return
    for r in i["r"]:
        if i["kind"] == "N":
            ref = dipole.hyperfine(i["n_d"], i["z"], nuc, r, tuple(i["g"]), i["alpha2"], i["geometry"])
            got = tdp.hyperfine(i["n_d"], i["z"], None, r, i["g"], i["alpha2"], i["geometry"],
                                g_n=_nucleus(nuc, False).g_n, constants=C18)
            for f, ft in (("a_mhz", "a"), ("a_g_mhz", "a_g"), ("a_alpha_mhz", "a_alpha"),
                          ("point_dipole_mhz", "point_dipole")):
                if np.abs(getattr(ref, f)).max() > 0:
                    close(getattr(got, ft), getattr(ref, f))
        else:
            ref = dipole.electron_electron(i["n_d"], i["z"], r, tuple(i["g"]), i["alpha2"], i["geometry"])
            mhz = tdp.electron_electron(i["n_d"], i["z"], r, i["g"], i["alpha2"], i["geometry"], constants=C18)
            cm = tdp.electron_electron(i["n_d"], i["z"], r, i["g"], i["alpha2"], i["geometry"], constants=C18,
                                       units="cm^-1")
            for got, sfx in ((mhz, "mhz"), (cm, "cm")):
                for f in ("d", "d_g", "d_alpha"):
                    close(getattr(got, f), getattr(ref, f"{f}_{sfx}"))
            close(cm.point_dipole, ref.point_dipole_cm)
            close(cm.ddip_edip, ref.ddip_edip_cm)
            close(cm.ddip_edip_point, ref.ddip_edip_point_cm)
            close(cm.ddip_edip_alpha, ref.ddip_edip_alpha_cm)
            got = mhz
        close(got.tensor_cm3, ref.tensor_cm3)
        close(got.A, ref.A)
        close(got.zstar, ref.zstar.value)


@pytest.mark.parametrize("geometry", tdp.GEOMETRIES)
def test_reduced_tensor_matches_decimal_path_everywhere(geometry):
    """rel 1e-13 of max|t| (abs 1e-15 floor at the out_of_plane zero crossing, A ~ 3.19) over A = 1e-3 .. 1e3."""
    _, tensor, *_ = _telser()
    A_grid = np.geomspace(1e-3, 1e3, 121)
    got = tdp.sto_tensor(torch.tensor(A_grid), geometry).numpy()
    for A, t in zip(A_grid, got):
        ref = np.array(tensor.sto_tensor(float(A), geometry))
        assert np.abs(t - ref).max() <= 1e-13 * np.abs(ref).max() + 1e-15, A


@pytest.mark.parametrize("geometry", tdp.GEOMETRIES)
@pytest.mark.parametrize("A", [0.05, 0.5, 3.0, 17.99, 18.01, 40.0])
def test_dA_vs_decimal_central_differences(geometry, A):
    _, tensor, *_ = _telser()
    a = torch.tensor(A, dtype=F64, requires_grad=True)
    t = tdp.sto_tensor(a, geometry)
    J = torch.stack([torch.autograd.grad(t[i], a, retain_graph=True)[0] for i in range(3)]).numpy()[:, None]
    err = min(np.abs(J - _fd(lambda x: np.array(tensor.sto_tensor(float(x[0]), geometry)), np.array([A]), A * h)).max()
              for h in (1e-3, 1e-4, 1e-5))
    assert err / np.abs(J).max() < 1e-8


def test_nucdata_vs_telser_table_for_ligand_nuclei():
    """torchspin.nucdata and telser-tools' table (Weil & Bolton 2007) agree for common ligand nuclei;
    some metal nuclides differ (63Cu 1.7e-3, 55Mn 4.5e-3, 149Sm 5.7e-3 relative, both negative;
    165Ho 1.668 vs 1.19).  Use ``g_n=`` for a specific value."""
    pytest.importorskip("telser_tools.nuclear")
    from telser_tools.nuclear import find
    for n in ("1H", "2H", "13C", "14N", "15N", "17O", "19F", "31P", "33S", "35Cl"):
        assert float(nucgval(n)) == pytest.approx(find(n).g_n, rel=1e-6), n
    assert float(nucgval("149Sm")) < 0 and find("149Sm").g_n < 0
