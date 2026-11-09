"""torchspin.lft milestone M2: DDN levels and g, ligand-field constructors, dn2sys -> pepper.

Self-contained physics checks (closed forms), unit handling, gradients against finite
differences, and parity with telser-tools (the source of the port) when it is installed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from torchspin import Experiment, Options, SpinSystem, lft
from torchspin.lft import ddn
from torchspin.lft.units import energy_factor

F64 = torch.float64
CM_TO_MHZ = energy_factor("cm^-1", "MHz")


def _elongated_cu(e_sigma=6000.0, e_pi=1000.0, ax=0.8, tilt=0.0):
    """Cu(II): four equatorial ligands (optionally tilted, radians) and two weaker axial ones."""
    eq = [lft.Ligand.isotropic(math.pi / 2 + (-1) ** i * tilt, i * math.pi / 2, e_sigma, e_pi) for i in range(4)]
    axl = [lft.Ligand.isotropic(0.0, 0.0, ax * e_sigma, 0.0), lft.Ligand.isotropic(math.pi, 0.0, ax * e_sigma, 0.0)]
    return lft.aom(eq + axl)


# ------------------------------------------------------------------------------ closed forms
def test_d1_octahedral_splitting_is_10Dq():
    E = ddn.levels(1, V=lft.crystal_field(Dq=1500.0), units="cm^-1")
    np.testing.assert_allclose(E.numpy(), [-6000.0] * 6 + [9000.0] * 4, atol=1e-9)


def test_d3_first_spin_allowed_band_is_10Dq():
    """4A2g -> 4T2g = 10Dq for any B, C (Tanabe-Sugano); weak field, so 4T2 lies below 2E."""
    E = ddn.levels(3, B=800.0, C=3200.0, V=lft.crystal_field(Dq=800.0), units="cm^-1", relative=True)
    assert float(E[4]) == pytest.approx(8000.0, abs=1e-8)
    assert float(E[15]) == pytest.approx(8000.0, abs=1e-8)        # 4T2 is 12-fold
    assert float(E[3]) == pytest.approx(0.0, abs=1e-8)            # 4A2 is 4-fold


def test_d2_free_ion_3P_minus_3F_is_15B():
    E = ddn.levels(2, B=900.0, C=3600.0, units="cm^-1", relative=True)
    assert int((E[:21].abs() < 1e-8).sum()) == 21                  # 3F ground term, 21 states
    assert int(((E - 15 * 900.0).abs() < 1e-8).sum()) == 9         # 3P at 15B, 9 states
    assert int(((E - (5 * 900.0 + 2 * 3600.0)).abs() < 1e-8).sum()) == 5   # 1D at 5B + 2C


def test_octahedron_aom_equals_crystal_field():
    """Octahedral AOM: 10Dq = 3 e_sigma - 4 e_pi."""
    v = lft.aom(lft.octahedron(6000.0, 1000.0))
    w = torch.linalg.eigvalsh(v)
    assert float(w[-1] - w[0]) == pytest.approx(3 * 6000.0 - 4 * 1000.0, abs=1e-9)


def test_d9_without_spin_orbit_has_free_electron_g():
    g = ddn.g_tensor(9, V=_elongated_cu(), units="cm^-1")
    np.testing.assert_allclose(g.principal.numpy(), [ddn.TORCHSPIN_CONSTANTS.g_e] * 3, rtol=1e-12)


def test_d9_elongated_g_order_and_frame():
    g = ddn.g_tensor(9, V=_elongated_cu(), zeta=830.0, k=(0.8, 0.8, 0.8), units="cm^-1")
    gx, gy, gz = torch.diagonal(g.g).tolist()
    assert gx == pytest.approx(gy, rel=1e-12)
    assert gz > gx > 2.0                                           # dx2-y2 ground state: g|| > g_perp > g_e
    torch.testing.assert_close(g.g @ g.g, g.ggT, rtol=1e-12, atol=1e-12)


# ------------------------------------------------------------------------------ units
def test_units_mhz_vs_cm1_consistent():
    V = lft.crystal_field(Dq=1200.0, Ds=300.0, Dt=50.0)
    kw = dict(B=800.0, C=3200.0, zeta=200.0)
    E_cm = ddn.levels(3, V=V, units="cm^-1", **kw)
    E_mhz = ddn.levels(3, V=V * CM_TO_MHZ, units="MHz", **{k: v * CM_TO_MHZ for k, v in kw.items()})
    torch.testing.assert_close(E_mhz, E_cm * CM_TO_MHZ, rtol=1e-12, atol=1e-3)
    E_K = ddn.levels(3, V=V, units="cm^-1", **kw) * energy_factor("cm^-1", "K")
    E_K2 = ddn.levels(3, V=V * energy_factor("cm^-1", "K"), units="K",
                      **{k: v * energy_factor("cm^-1", "K") for k, v in kw.items()})
    torch.testing.assert_close(E_K2, E_K, rtol=1e-12, atol=1e-8)


def test_gap_reported_in_requested_units():
    g_cm = ddn.g_tensor(9, V=_elongated_cu(), zeta=830.0, units="cm^-1")
    g_mhz = ddn.g_tensor(9, V=_elongated_cu() * CM_TO_MHZ, zeta=830.0 * CM_TO_MHZ, units="MHz")
    assert float(g_mhz.gap) == pytest.approx(float(g_cm.gap) * CM_TO_MHZ, rel=1e-12)
    torch.testing.assert_close(g_mhz.principal, g_cm.principal, rtol=1e-12, atol=1e-12)


def test_zeeman_field_in_mT():
    """d1, no ligand field or SO: the 2D splits by g_e mu_B B for the spin and mu_B B per m_L."""
    E = ddn.levels(1, field=torch.tensor([0.0, 0.0, 1000.0], dtype=F64), units="cm^-1")
    mu = ddn.TORCHSPIN_CONSTANTS.mu_b * 1000.0
    expected = sorted(ml * mu + ms * ddn.TORCHSPIN_CONSTANTS.g_e * mu for ml in range(-2, 3) for ms in (-0.5, 0.5))
    np.testing.assert_allclose(E.numpy(), expected, rtol=1e-12, atol=1e-14)


def test_dnsystem_wrapper_defaults():
    s = ddn.DnSystem(3, units="cm^-1")
    assert s.size == 120
    torch.testing.assert_close(s.levels(B=800.0, C=3200.0), ddn.levels(3, B=800.0, C=3200.0, units="cm^-1"))


# ------------------------------------------------------------------------------ gradients
def _fd(f, x, h):
    cols = []
    for i in range(x.numel()):
        e = torch.zeros_like(x)
        e[i] = h[i]
        cols.append((f(x + e) - f(x - e)) / (2 * h[i]))
    return torch.stack(cols, -1)


def test_level_gradients_vs_central_differences():
    def f(p):
        B, C, Dq, Ds, Dt, zeta = p
        return ddn.levels(3, B=B, C=C, zeta=zeta, V=lft.crystal_field(Dq, Ds, Dt), units="cm^-1")[[0, 4, 12, 30]]

    x = torch.tensor([800.0, 3200.0, 1500.0, 250.0, 60.0, 180.0], dtype=F64, requires_grad=True)
    J = torch.autograd.functional.jacobian(f, x)
    Jfd = _fd(f, x.detach(), [1e-3] * 6)
    torch.testing.assert_close(J, Jfd, rtol=1e-6, atol=1e-7)


def test_g_gradients_vs_central_differences_including_axial():
    def f(p):
        es, ep, tilt, zeta = p
        return ddn.g_tensor(9, V=_elongated_cu(es, ep, tilt=tilt), zeta=zeta, k=(0.8, 0.8, 0.8),
                            units="cm^-1").principal

    for tilt in (0.2, 0.0):                                            # 0.0: axial, g_x = g_y
        x = torch.tensor([6000.0, 1000.0, tilt, 830.0], dtype=F64, requires_grad=True)
        J = torch.autograd.functional.jacobian(f, x)
        assert torch.isfinite(J).all()
        if tilt:
            torch.testing.assert_close(J, _fd(f, x.detach(), [1e-2, 1e-2, 1e-6, 1e-3]), rtol=1e-5, atol=1e-10)
        else:                                                          # degenerate pair: check the sum
            s = lambda p: f(p)[:2].sum(-1, keepdim=True)  # noqa: E731
            torch.testing.assert_close(torch.autograd.functional.jacobian(s, x),
                                       _fd(s, x.detach(), [1e-2, 1e-2, 1e-6, 1e-3]), rtol=1e-5, atol=1e-8)


def test_isotropic_aom_matches_angle_path():
    g = torch.Generator().manual_seed(0)
    n = torch.randn(5, 3, dtype=F64, generator=g)
    n = n / n.norm(dim=-1, keepdim=True)
    th, ph = torch.arccos(n[:, 2]), torch.atan2(n[:, 1], n[:, 0])
    e = torch.tensor([6000.0, 900.0, 900.0, 50.0, 50.0], dtype=F64).expand(5, 5)
    torch.testing.assert_close(lft.aom_isotropic(n, 6000.0, 900.0, 50.0),
                               lft.aom_from_angles(th, ph, torch.zeros(5, dtype=F64), e), rtol=1e-12, atol=1e-9)


# ------------------------------------------------------------------------------ bridge -> pepper
def test_dn2sys_builds_full_g_spin_system():
    sys, gt = lft.dn2sys(9, dict(V=_elongated_cu(), zeta=830.0, k=(0.8, 0.8, 0.8)), units="cm^-1",
                         Nucs="63Cu", A=[60.0, 60.0, 500.0], lw=[1.0, 0.0])
    assert isinstance(sys, SpinSystem) and sys.fullg
    torch.testing.assert_close(sys.g, gt.g)
    with pytest.raises(ValueError):
        lft.dn2sys(9, dict(V=lft.aom(lft.octahedron(6000.0, 1000.0)), zeta=830.0), units="cm^-1", min_gap=100.0)
    with pytest.raises(TypeError):
        lft.dn2sys(9, dict(zeta=830.0), units="cm^-1", g=2.0)


def test_ligand_field_gradient_flows_through_pepper():
    """d(spectrum)/d(e_pi): ligand field -> g -> pepper, against central differences."""
    from torchspin.pepper_autograd import pepper_autograd

    exp = Experiment(mwFreq=9.5, Range=[255.0, 360.0], nPoints=512, Harmonic=1)
    opt = Options(GridSize=[19, 4], GridSymmetry="D2h", Method="matrix")

    def spec(e_pi):
        sys, _ = lft.dn2sys(9, dict(V=_elongated_cu(6000.0, e_pi, tilt=0.15), zeta=830.0, k=(0.8, 0.8, 0.8)),
                            units="cm^-1", lw=[3.0, 0.0])
        return pepper_autograd(sys, exp, opt)[1]

    x = torch.tensor(1000.0, dtype=F64, requires_grad=True)
    y = spec(x)
    w = torch.linspace(-1.0, 1.0, y.numel(), dtype=F64)
    (g,) = torch.autograd.grad((y * w).sum(), x)
    h = 0.5
    fd = (((spec(torch.tensor(1000.0 + h, dtype=F64)) - spec(torch.tensor(1000.0 - h, dtype=F64))) * w).sum() / (2 * h))
    assert float(g) != 0.0
    assert float(g) == pytest.approx(float(fd), rel=1e-4)


# ------------------------------------------------------------------------------ parity with telser-tools
def _telser():
    tt = pytest.importorskip("telser_tools.ddn")
    from telser_tools.ddn import ligand_field as tlf
    from telser_tools.ddn.hamiltonian import CODATA2018
    return tt, tlf, CODATA2018


@pytest.mark.parametrize("n", range(1, 10))
def test_levels_parity_with_telser_tools(n):
    tt, tlf, C18 = _telser()
    rng = np.random.default_rng(n)
    v = rng.normal(size=(5, 5)) * 800.0
    v = v + v.T
    p = tt.Parameters(B=820.0, C=3300.0, zeta=310.0, ligand_field=v, k=(0.9, 0.85, 0.8), alpha=60.0)
    consts = ddn.Constants(mu_b=C18.mu_b * 10.0, g_e=C18.g_e, k_b=C18.k_b)    # cm^-1/G -> cm^-1/mT
    kw = dict(B=820.0, C=3300.0, zeta=310.0, V=torch.tensor(v), k=(0.9, 0.85, 0.8), alpha=60.0,
              units="cm^-1", constants=consts)
    for field_G in (None, np.array([1200.0, -500.0, 3000.0])):
        ref = tt.DnSystem(n).energies(p, field_G)
        E = ddn.levels(n, field=None if field_G is None else torch.tensor(field_G / 10.0), **kw)
        assert np.all(np.abs(E.numpy() - ref) <= np.maximum(1e-9, 1e-10 * np.abs(ref)))


def test_ligand_field_parity_with_telser_tools():
    _, tlf, _ = _telser()
    ligs_deg = [(90, 0, 10, 6000, 900, 700, 40, 30), (35, 120, 0, 5000, 800, 800, 0, 0), (150, 250, 70, 4000, 300, 500, 20, 10)]
    ref = tlf.aom([tlf.Ligand(*l) for l in ligs_deg])
    ours = lft.aom([lft.Ligand.from_degrees(t, p, s, e_sigma=a, e_pi_x=b, e_pi_y=c, e_delta_x2y2=d, e_delta_xy=f)
                    for t, p, s, a, b, c, d, f in ligs_deg])
    np.testing.assert_allclose(ours.numpy(), ref, atol=1e-9)
    for geo in ("Oh", "Td"):
        for axis in ("tetragonal", "trigonal"):
            np.testing.assert_allclose(lft.crystal_field(1200.0, 300.0, 80.0, 40.0, geometry=geo, axis=axis).numpy(),
                                       tlf.crystal_field(1200.0, 300.0, 80.0, 40.0, geometry=geo, axis=axis), atol=1e-10)


@pytest.mark.parametrize("n", [1, 3, 5, 7, 9])
def test_kramers_g_parity_with_telser_tools(n):
    tt, tlf, C18 = _telser()
    consts = ddn.Constants(mu_b=C18.mu_b * 10.0, g_e=C18.g_e, k_b=C18.k_b)
    for V in (tlf.crystal_field(1500.0), tlf.crystal_field(1500.0, 300.0, 50.0, 20.0)):
        p = tt.Parameters(B=800, C=3200, zeta=300, ligand_field=V, k=(0.9, 0.85, 0.8))
        sol = tt.DnSystem(n).solve(p)
        if sol.energies[2] - sol.energies[1] < 1.0:
            continue
        g = ddn.g_tensor(n, B=800.0, C=3200.0, zeta=300.0, V=torch.tensor(V), k=(0.9, 0.85, 0.8),
                         units="cm^-1", constants=consts)
        np.testing.assert_allclose(g.ggT.numpy(), sol.g_tensor((0, 1))[1], atol=1e-10)
