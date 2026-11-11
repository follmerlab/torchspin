"""torchspin.lft milestone M3a: DDNSUS / DDNMAG (exact thermal averages over the full d^n model)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from torchspin import SpinSystem, curry, lft
from torchspin.lft import ddnmag, ddnsus, grids

F64 = torch.float64
GE = 2.00231930436092


def _co_kw(**over):
    kw = dict(B=820.0, C=3690.0, zeta=480.0, k=(0.85, 0.85, 0.85), V=lft.crystal_field(950.0, 250.0, 60.0),
              units="cm^-1")
    kw.update(over)
    return kw


# ------------------------------------------------------------------------------ physics
@pytest.mark.parametrize("n,S,Dq", [(5, 2.5, 100.0), (3, 1.5, 1500.0)])
def test_spin_only_limit_equals_curry(n, S, Dq):
    """k = 0 and zeta = 0 leave g_e S only: chi must equal curry's for a free spin S (linear regime)."""
    T = [5.0, 50.0, 300.0]
    chi = ddnsus.susceptibility(n, T, field=1.0, grid=grids.gauss_hemisphere(4), B=800.0, C=3200.0,
                                V=lft.crystal_field(Dq=Dq), k=(0.0, 0.0, 0.0), units="cm^-1").powder
    _, ref = curry(SpinSystem(S=S, g=GE), [1.0], T, delta_B=0.01)
    torch.testing.assert_close(chi, ref, rtol=1e-6, atol=0)


def test_curie_law_and_mu_eff():
    chiT = ddnsus.chi_T(5, [300.0], field=1.0, grid=grids.axes(), B=800.0, C=3200.0, V=lft.crystal_field(Dq=100.0),
                        units="cm^-1", chi_units="cgs").powder
    expected = ddnsus.curie_constant("cgs") * GE ** 2 * 2.5 * 3.5
    assert float(chiT) == pytest.approx(expected, rel=1e-5)
    assert ddnsus.mu_eff(float(chiT), "cgs") == pytest.approx(GE * math.sqrt(35 / 4), rel=1e-5)
    assert ddnsus.curie_constant("cgs") == pytest.approx(0.12505, rel=1e-4)


def test_d3_van_vleck_tip_is_8NmuB2_over_10Dq():
    kw = dict(B=800.0, C=3200.0, V=lft.crystal_field(Dq=1500.0), units="cm^-1", chi_units="cgs",
              grid=grids.gauss_hemisphere(4), field=10.0)
    tip = ddnsus.susceptibility(3, [300.0], **kw).powder - ddnsus.susceptibility(3, [300.0], k=(0.0, 0.0, 0.0), **kw).powder
    expected = 8 * 3 * ddnsus.curie_constant("cgs") * ddnmag.MAG_CONSTANTS.k_b / 15000.0   # N mu_B^2 = 3 C k_B
    assert float(tip) == pytest.approx(expected, rel=1e-4)


def test_saturation_of_free_spin_brillouin():
    """d5 6S at 2 K, 7 T: M -> g_e S Brillouin function."""
    M = ddnmag.magnetization(5, [7000.0], [2.0], B=800.0, C=3200.0, V=lft.crystal_field(Dq=100.0), units="cm^-1").powder
    c = ddnmag.MAG_CONSTANTS
    x = GE * c.mu_b * 7000.0 / (c.k_b * 2.0)
    S = 2.5
    brill = (2 * S + 1) / 2 * 1 / math.tanh((2 * S + 1) / 2 * x) - 0.5 / math.tanh(x / 2)
    assert float(M) == pytest.approx(GE * brill, rel=1e-7)


def test_magnetization_vanishes_at_zero_field_and_is_even_in_direction():
    kw = _co_kw()
    M = ddnmag.magnetization(7, [0.0, 500.0], [5.0], grid=grids.directions([[0.3, 0.2, 0.9], [-0.3, -0.2, -0.9]]), **kw)
    assert torch.all(M.per_direction[:, 0] == 0)
    torch.testing.assert_close(M.per_direction[0], M.per_direction[1], rtol=1e-9, atol=0)


# ------------------------------------------------------------------------------ units
def test_units_consistency():
    kw = _co_kw()
    T = [2.0, 20.0, 300.0]
    si = ddnsus.susceptibility(7, T, chi_units="SI", grid=grids.axes(), **kw).powder
    cgs = ddnsus.susceptibility(7, T, chi_units="cgs", grid=grids.axes(), **kw).powder
    torch.testing.assert_close(si, cgs * 10 * ddnmag.MAG_CONSTANTS.mu0, rtol=1e-14, atol=0)
    mu = ddnmag.magnetization(7, [1000.0], T, grid=grids.axes(), **kw).powder
    emu = ddnmag.magnetization(7, [1000.0], T, grid=grids.axes(), M_units="emu/mol", **kw).powder
    torch.testing.assert_close(emu / mu, torch.full_like(mu, 5585.0), rtol=1e-3, atol=0)
    f = lft.units.energy_factor("cm^-1", "MHz")
    kw_mhz = dict(B=820.0 * f, C=3690.0 * f, zeta=480.0 * f, k=(0.85,) * 3, V=lft.crystal_field(950.0, 250.0, 60.0) * f)
    # MHz vs cm^-1 inputs: same physics, different round-off through the thermal average (CI BLAS: up to
    # 8e-11 relative; 1e-12 on macOS Accelerate)
    torch.testing.assert_close(ddnsus.susceptibility(7, T, grid=grids.axes(), **kw_mhz).powder, si, rtol=1e-9, atol=0)


def test_chi0_adds_tip_in_both_quantities():
    kw = _co_kw()
    base = ddnsus.susceptibility(7, [10.0], grid=grids.axes(), chi_units="cgs", **kw).powder
    with_tip = ddnsus.susceptibility(7, [10.0], grid=grids.axes(), chi_units="cgs", chi0=1e-3, **kw).powder
    assert float(with_tip - base) == pytest.approx(1e-3, rel=1e-12)
    M0 = ddnmag.magnetization(7, [1000.0], [10.0], grid=grids.axes(), M_units="emu/mol", **kw).powder
    M1 = ddnmag.magnetization(7, [1000.0], [10.0], grid=grids.axes(), M_units="emu/mol", chi0=1e-3,
                              chi_units="cgs", **kw).powder
    assert float(M1 - M0) == pytest.approx(1e-3 * 10000.0, rel=1e-9)     # chi0 * H(G)


# ------------------------------------------------------------------------------ gradients
def test_gradients_vs_central_differences_through_kramers_degeneracy():
    T = torch.tensor([2.0, 10.0, 100.0], dtype=F64)

    def f(p):
        zeta, Ds, k = p
        return ddnsus.chi_T(7, T, grid=grids.gauss_hemisphere(2), chi_units="cgs",
                            **_co_kw(zeta=zeta, V=lft.crystal_field(950.0, Ds, 60.0), k=k.expand(3))).powder[0]

    x = torch.tensor([480.0, 250.0, 0.85], dtype=F64)
    J = torch.autograd.functional.jacobian(f, x)
    E = torch.eye(3, dtype=F64)
    # best of several steps: small steps are dominated by eigensolver round-off, large ones by truncation
    err = torch.stack([torch.stack([((f(x + E[i] * h) - f(x - E[i] * h)) / (2 * h) - J[:, i]).abs().max()
                                    for i, h in enumerate(hs)]) for hs in ([1e-1, 1e-1, 1e-4], [3e-1, 3e-1, 3e-4],
                                                                           [3e-2, 3e-2, 3e-5])]).min(0).values
    assert torch.all(err <= 1e-6 * J.abs().max(0).values), err
    Jf = torch.func.jacfwd(f)(x)
    torch.testing.assert_close(Jf, J, rtol=1e-10, atol=1e-14)


# ------------------------------------------------------------------------------ parity with telser-tools
def _telser():
    tdiff = pytest.importorskip("telser_tools_diff.magnetometry")
    from telser_tools import ddn as tddn
    from telser_tools.magnetometry import powder
    from telser_tools.magnetometry.constants import CODATA2018 as M18
    consts = ddnmag.MagConstants(mu_b=M18.mu_b * 10, g_e=M18.g_e, k_b=M18.k_b, N_A=M18.N_A,
                                 mu_b_SI=M18.mu_b_erg * 1e-3, mu0=4e-7 * math.pi)
    return tdiff, tddn, powder, consts


@pytest.mark.parametrize("mirror", [False, True])
def test_parity_with_telser_tools(mirror):
    tdiff, tddn, powder, C = _telser()
    V = tddn.ligand_field.crystal_field(1500.0, 300.0, 50.0, 20.0) + np.array(
        [[0, 40, 0, 10, 0], [40, 0, 5, 0, 0], [0, 5, 0, 0, 30], [10, 0, 0, 0, 0], [0, 0, 30, 0, 0]], float)
    p = tddn.Parameters(B=800, C=3200, zeta=300, ligand_field=V, k=(0.9, 0.85, 0.8))
    ms = tdiff.MagneticSystem(7, ddn_mirror=mirror)
    kw = dict(units="cm^-1", constants=C, ddn_mirror=mirror, B=800.0, C=3200.0, zeta=300.0, k=(0.9, 0.85, 0.8),
              V=torch.tensor(V))
    for tg, g in ((powder.gauss_hemisphere(3), grids.gauss_hemisphere(3)), (powder.ddn_igloo(90, 90, 4),
                                                                             grids.ddn_igloo(90, 90, 4))):
        ref = ms.magnetization(p, H=[1000.0, 50000.0], T=[2.0, 300.0], grid=tg).per_direction
        ours = ddnmag.magnetization(7, [100.0, 5000.0], [2.0, 300.0], grid=g, M_units="emu/mol", **kw).per_direction
        torch.testing.assert_close(ours, ref, rtol=1e-11, atol=1e-12)
        ref = ms.susceptibility(p, T=[2.0, 10.0, 300.0], H=1000.0, grid=tg).powder
        ours = ddnsus.susceptibility(7, [2.0, 10.0, 300.0], field=100.0, grid=g, chi_units="cgs", **kw).powder
        torch.testing.assert_close(ours, ref, rtol=1e-11, atol=0)


def test_grids_identical_to_telser_tools():
    _, _, powder, _ = _telser()
    for args in [(90, 90, 10), (180, 180, 12), (90, 180, 7), (180, 90, 50)]:
        a, b = grids.ddn_igloo(*args), powder.ddn_igloo(*args)
        np.testing.assert_array_equal(a.vectors, b.vectors)
        np.testing.assert_array_equal(a.weights, b.weights)
    np.testing.assert_array_equal(grids.gauss_hemisphere(5).vectors, powder.gauss_hemisphere(5).vectors)
