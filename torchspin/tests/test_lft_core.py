"""torchspin.lft milestone M1: unit factors, degenerate-safe linear algebra, d^n operator engine.

Parity with telser-tools (the source of the port) runs only when telser-tools is installed;
everything else is self-contained.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from torchspin.lft import _linalg as tl
from torchspin.lft import units
from torchspin.lft._engines import dn
from torchspin.lft._operators import dn_operators

F64, C128 = torch.float64, torch.complex128


# ------------------------------------------------------------------------------ units
def test_energy_factors_against_definitions():
    assert units.energy_factor("cm^-1", "MHz") == pytest.approx(29979.2458, rel=1e-12)
    assert units.energy_factor("cm^-1", "K") == pytest.approx(1.438776877, rel=1e-9)
    assert units.energy_factor("cm^-1", "eV") == pytest.approx(1.239841984e-4, rel=1e-9)
    assert units.energy_factor("GHz", "MHz") == pytest.approx(1e3, rel=1e-15)
    assert units.energy_factor("cm-1", "cm⁻¹") == 1.0


@pytest.mark.parametrize("a", units.ENERGY_UNITS)
@pytest.mark.parametrize("b", units.ENERGY_UNITS)
def test_energy_round_trip(a, b):
    assert units.energy_factor(a, b) * units.energy_factor(b, a) == pytest.approx(1.0, rel=1e-14)


def test_field_and_chi_factors():
    assert units.field_factor("T", "mT") == 1e3
    assert units.field_factor("G", "mT") == pytest.approx(0.1, rel=1e-15)
    # S = 1/2, g = 2: chi T = 0.375 cm^3 K/mol (cgs) = 4.71e-6 m^3 K/mol (SI)
    assert units.convert_chi(0.375, "cgs", "SI") == pytest.approx(0.375 * 4 * math.pi * 1e-6, rel=1e-15)
    assert units.chi_factor("SI", "cgs") * units.chi_factor("cgs", "SI") == pytest.approx(1.0, rel=1e-15)


def test_conversions_keep_the_graph():
    x = torch.tensor([1000.0, 2.0], dtype=F64, requires_grad=True)
    y = units.convert_energy(x, "cm^-1", "MHz")
    (g,) = torch.autograd.grad(y.sum(), x)
    torch.testing.assert_close(g, torch.full_like(x, units.energy_factor("cm^-1", "MHz")))


def test_unknown_units_raise():
    with pytest.raises(ValueError):
        units.energy_factor("furlong", "MHz")
    with pytest.raises(ValueError):
        units.field_factor("Oe", "mT")
    with pytest.raises(ValueError):
        units.chi_factor("emu", "SI")


def test_plot_default_and_labels():
    assert units.PLOT_ENERGY_UNITS == "cm^-1"
    assert units.energy_label("cm-1") == "cm⁻¹"


# ------------------------------------------------------------------------------ linear algebra
def _herm_with_degeneracies(n=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    U, _ = torch.linalg.qr(torch.randn(n, n, dtype=C128, generator=g))
    w = torch.tensor([0.0, 0.0, 1.0, 1.0, 2.5, 3.0, 3.0, 4.0], dtype=F64)[:n]
    return (U * w.to(C128)) @ U.conj().T


def test_ground_projector_gradcheck_at_kramers_degeneracy():
    H = _herm_with_degeneracies().requires_grad_()
    W = torch.randn(8, 8, dtype=C128, generator=torch.Generator().manual_seed(2))
    W = W + W.conj().T
    f = lambda h: (tl.ground_manifold(0.5 * (h + h.conj().T), 2).projector * W).real.sum()  # noqa: E731
    assert torch.autograd.gradcheck(f, (H,), eps=1e-6, atol=1e-6, check_forward_ad=True)


def test_thermal_average_gradcheck_at_degeneracies():
    H = _herm_with_degeneracies().requires_grad_()
    A = torch.randn(2, 8, 8, dtype=C128, generator=torch.Generator().manual_seed(3))
    A = A + A.conj().transpose(-1, -2)
    beta = torch.tensor([0.3, 2.0], dtype=F64)
    f = lambda h: tl.thermal_average(0.5 * (h + h.conj().T), A, beta)  # noqa: E731
    assert torch.autograd.gradcheck(f, (H,), eps=1e-6, atol=1e-6, check_forward_ad=True)
    Ar = A.clone().requires_grad_()
    g = lambda a: tl.thermal_average(H.detach(), 0.5 * (a + a.conj().transpose(-1, -2)), beta)  # noqa: E731
    assert torch.autograd.gradcheck(g, (Ar,), eps=1e-6, atol=1e-6, check_forward_ad=True)


def test_eigh_gauge_level_sums_gradcheck():
    H = _herm_with_degeneracies().requires_grad_()
    W = torch.randn(8, 8, dtype=C128, generator=torch.Generator().manual_seed(7))
    W = W + W.conj().T

    def f(h):
        L, V = tl.eigh_gauge(0.5 * (h + h.conj().T))
        P = V[:, 2:4] @ V[:, 2:4].conj().T
        return torch.stack([L[:2].sum(), L[2:4].sum(), L[4], (P * W).real.sum()])

    assert torch.autograd.gradcheck(f, (H,), eps=1e-6, atol=1e-6, check_forward_ad=True)


def test_gauge_zeroing_is_wrong_for_thermal_averages():
    """Why lft does not route thermal averages through a gauge-zeroing eigh backward."""
    H = _herm_with_degeneracies().requires_grad_()
    A = torch.randn(8, 8, dtype=C128, generator=torch.Generator().manual_seed(4))
    A = A + A.conj().T
    L, V = tl.eigh_gauge(0.5 * (H + H.conj().T))
    naive = (torch.softmax(-0.3 * L, -1) * torch.diagonal(V.conj().T @ A @ V).real).sum()
    g_naive = torch.autograd.grad(naive, H)[0]
    g_dk = torch.autograd.grad(tl.thermal_average(0.5 * (H + H.conj().T), A[None], 0.3).sum(), H)[0]
    assert (g_naive - g_dk).abs().max() > 1e-2


def test_newton_schulz_sqrt_exact_gradient_at_axial():
    R = torch.linalg.qr(torch.randn(3, 3, dtype=F64, generator=torch.Generator().manual_seed(5)))[0]
    gp = torch.tensor([2.07, 2.07, 2.36], dtype=F64, requires_grad=True)
    W = torch.randn(3, 3, dtype=F64, generator=torch.Generator().manual_seed(6))
    g = R @ torch.diag(gp) @ R.T
    got = torch.autograd.grad((tl.sqrtm_psd(g @ g.T) * W).sum(), gp)[0]
    exact = torch.autograd.grad((R @ torch.diag(gp) @ R.T * W).sum(), gp)[0]
    torch.testing.assert_close(got, exact, rtol=1e-10, atol=1e-12)


def test_thermal_average_truncation_to_populated_states():
    g = torch.Generator().manual_seed(8)
    U, _ = torch.linalg.qr(torch.randn(12, 12, dtype=C128, generator=g))
    w = torch.tensor([0.0, 0.0, 3.0, 3.0, 5.0, 900.0, 900.0, 1200.0, 2000.0, 2500.0, 2500.0, 4000.0], dtype=F64)
    H = ((U * w.to(C128)) @ U.conj().T).requires_grad_()
    A = torch.randn(2, 12, 12, dtype=C128, generator=g)
    A = A + A.conj().transpose(-1, -2)
    beta = torch.tensor([1 / 2.0, 1 / 10.0], dtype=F64)
    assert tl._n_populated(w, beta) == 5
    f = lambda h: tl.thermal_average(0.5 * (h + h.conj().T), A, beta)  # noqa: E731
    assert torch.autograd.gradcheck(f, (H,), eps=1e-6, atol=1e-7, check_forward_ad=True)
    old = tl.THERMAL_CUTOFF
    try:
        tl.THERMAL_CUTOFF = 1e9
        full = torch.autograd.grad(f(H).sum(), H)[0]
    finally:
        tl.THERMAL_CUTOFF = old
    torch.testing.assert_close(torch.autograd.grad(f(H).sum(), H)[0], full, rtol=0, atol=1e-15)


def test_jacfwd_through_eigh_gauge():
    H0 = _herm_with_degeneracies()
    D = torch.randn(8, 8, dtype=C128, generator=torch.Generator().manual_seed(9))
    D = D + D.conj().T

    def f(t):
        L, _ = tl.eigh_gauge(H0 + t[0] * D)
        return L

    J = torch.func.jacfwd(f)(torch.zeros(1, dtype=F64))
    Jr = torch.autograd.functional.jacobian(f, torch.zeros(1, dtype=F64))
    torch.testing.assert_close(J, Jr, rtol=1e-10, atol=1e-12)


# ------------------------------------------------------------------------------ d^n engine
@pytest.mark.parametrize("n", range(11))
def test_dn_engine_invariants(n):
    ops = dn.operators(n)
    assert ops.size == math.comb(10, n)
    terms, P = dn.term_projectors(n)
    np.testing.assert_allclose(P.sum(0), np.eye(ops.size), atol=1e-9)
    for t, p in zip(terms, P):
        assert np.trace(p) == pytest.approx(t.dimension, abs=1e-9)
        np.testing.assert_allclose(p @ ops.L2 @ p, t.L * (t.L + 1) * p, atol=1e-9)
        np.testing.assert_allclose(p @ ops.S2 @ p, t.S * (t.S + 1) * p, atol=1e-9)


def test_dn_free_ion_ground_terms():
    expected = {1: "2D", 2: "3F", 3: "4F", 4: "5D", 5: "6S", 6: "5D", 7: "4F", 8: "3F", 9: "2D"}
    for n, lab in expected.items():
        assert dn.ddn_terms(n)[0].label == lab


def test_torch_operators_match_numpy():
    ops, t = dn.operators(5), dn_operators(5)
    np.testing.assert_array_equal(t.so.numpy(), ops.so)
    np.testing.assert_array_equal(t.L.numpy(), ops.L)
    lf = np.einsum("ma,nb,mnij->abij", dn.U_REAL, dn.U_REAL.conj(), ops.excitation)
    np.testing.assert_array_equal(t.lf_real.numpy(), lf)
    assert dn_operators(5) is t                     # cached


@pytest.mark.parametrize("n", range(11))
def test_dn_engine_parity_with_telser_tools(n):
    engine = pytest.importorskip("telser_tools.ddn.engine")
    terms = pytest.importorskip("telser_tools.ddn.terms")
    a, b = dn.operators(n), engine.operators(n)
    for f in ("coulomb", "excitation", "so", "L", "S", "L2", "S2", "pair"):
        np.testing.assert_array_equal(getattr(a, f), getattr(b, f))
    ta, pa = dn.term_projectors(n)
    tb, pb = terms.term_projectors(n)
    assert [x.label for x in ta] == [x.label for x in tb]
    np.testing.assert_allclose(pa, pb, atol=1e-12)
