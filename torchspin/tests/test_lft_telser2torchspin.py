"""torchspin.lft milestone M5: Telser's spin-Hamiltonian programs through torchspin (``lft.telser2torchspin``).

1. Self-contained checks of the converters (closed forms, unit and constant handling, Euler frames vs full
   tensors, gradients, errors for unmapped features).
2. Parity with telser-tools (skipped when it is absent) on its fixtures (stdin texts in
   ``tests/data/telser/spin_family_inputs.json``):

   * DDZHQ, all 31 golden inputs: levels at every cycle (rel 1e-12 with telser-tools' constants passed
     through ``constants=``), all-pair B1-averaged probabilities (summed over degenerate levels), resonance
     fields and intensities (``resfields``), ENDOR frequencies and intensities (``endorfrq``);
   * DJDZE, all 50 golden inputs: levels (with the biquadratic constant) and probabilities;
   * DDPOWH and DDPOWJ powder spectra against ``pepper``/``salt`` after normalization to the maximum.
     Telser's igloo grid with solid-angle weights and pepper's interpolated grid integrate the same
     orientation average differently, so the tolerance is set by the grid convergence measured in
     telser-tools' own torchspin comparisons (docs/ddpowh.md, docs/ddpowj.md): 0.3-1 % of the peak for
     smooth S = 1/2 patterns, a few % where sharp D-edges meet a 10-20 G field step.
   * Where conventions differ, the documented relation is tested instead: Telser's EPR moment is g.S
     (torchspin's is -dH/dB, which includes the third-order Zeeman u and the nuclear moment); Telser's
     Flygare frequency factor; mixed H1 modes.
"""

from __future__ import annotations

import dataclasses
import json
import math
import types
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from torchspin import Options, SpinSystem, pepper
from torchspin.ham import ham
from torchspin.lft import telser2torchspin as t2t
from torchspin.lft.units import energy_factor

F64 = torch.float64
CM = energy_factor("cm^-1", "MHz")
DATA = json.loads((Path(__file__).parent / "data" / "telser" / "spin_family_inputs.json").read_text())

pytestmark = pytest.mark.filterwarnings("ignore:Sparse CSR tensor support:UserWarning")


def _levels(sy, b_mt=0.0, n=(0.0, 0.0, 1.0)):
    H0, mx, my, mz = ham(sy)
    n = [float(x) for x in n]
    return torch.linalg.eigvalsh(H0 - b_mt * (n[0] * mx + n[1] * my + n[2] * mz))


def _levels_t(sy, b_mt, n):
    """Graph-keeping levels (n a tensor)."""
    H0, mx, my, mz = ham(sy)
    return torch.linalg.eigvalsh(H0 - b_mt * (n[0] * mx + n[1] * my + n[2] * mz))


# ============================================================================== 1. self-contained
def test_abf_and_stevens_round_trip():
    a, F, b = 0.012, -0.03, 0.006
    B40, B42, B44 = t2t.abf_to_stevens(a, F, b)
    assert (B40, B42, B44) == pytest.approx((a / 120 + F / 180, b / 3, a / 24))
    assert t2t.stevens_to_abf(B40, B42, B44) == pytest.approx((a, F, b))


def test_cubic_a_term_s52_gamma7_gamma8():
    """S = 5/2, (a/120)(O40 + 5 O44): a Gamma7 doublet at -2a and a Gamma8 quartet at +a; isotropic Zeeman."""
    a = 0.02
    sy = t2t.spin2sys(s=2.5, a=a, g=(2.0, 2.0, 2.0))
    e = _levels(sy).numpy() / CM
    np.testing.assert_allclose(np.sort(e), [-2 * a] * 2 + [a] * 4, atol=1e-14)
    ref = _levels(sy, 300.0, (0, 0, 1)).numpy()
    for n in ((1, 0, 0), (0, 1, 0)):
        np.testing.assert_allclose(_levels(sy, 300.0, n).numpy(), ref, atol=1e-9)


def test_third_order_zeeman_closed_form_and_cubic_invariance():
    """mu_B u B_z C_z is diagonal: E_m = mu_B B (g m + u (m^3 - (3X - 1)/5 m)); u_x = u_y = u_z is cubic."""
    s, g, u, b = 1.5, 2.0, 0.05, 500.0
    sy = t2t.spin2sys(s=s, g=(g, g, g), u=(u, u, u))
    x = s * (s + 1)
    m = np.arange(-s, s + 1)
    ref = np.sort(t2t.MU_B_MHZ_PER_MT * b * (g * m + u * (m ** 3 - (3 * x - 1) / 5 * m)))
    np.testing.assert_allclose(_levels(sy, b, (0, 0, 1)).numpy(), ref, rtol=1e-12)
    for n in ((1, 0, 0), (0, 1, 0), (1 / math.sqrt(3),) * 3):
        e = _levels(sy, b, n).numpy()
        if n[0] == n[1] == n[2]:          # not a C4 axis: the levels differ from B || z
            assert np.abs(e - ref).max() > 1e-3
        else:
            np.testing.assert_allclose(e, ref, rtol=1e-12)
    rows = t2t.third_order_zeeman_ham(s, (u, u, u))
    assert np.abs(rows["132"]).max() < 1e-12 * np.abs(rows["134"]).max()       # cubic: only l = 4


def test_exchange_conventions_and_triplet_singlet_gap():
    """-2J' S1.S2 with J' = -5 cm^-1 is Telser's (EasySpin's) J = +10: triplet 10 cm^-1 above the singlet."""
    assert t2t.telser_J(-5.0, "-2J") == 10.0 and t2t.telser_J(3.0, "+J") == 3.0 and t2t.telser_J(2.0, "-J") == -2.0
    with pytest.raises(ValueError):
        t2t.telser_J(1.0, "J/2")
    sy = t2t.dimer2sys(site1=dict(s=0.5), site2=dict(s=0.5), J=-5.0, J_convention="-2J")
    e = _levels(sy).numpy() / CM
    np.testing.assert_allclose(e, [-7.5, 2.5, 2.5, 2.5], atol=1e-12)


def test_antisymmetric_exchange_closed_form():
    """d || z, two S = 1/2: E = J/4 (x2), -J/4 +- sqrt(J^2 + d^2)/2."""
    J, d = 1.3, 0.4
    sy = t2t.dimer2sys(site1=dict(s=0.5), site2=dict(s=0.5), J=J, d=(0.0, 0.0, d))
    e = np.sort(_levels(sy).numpy() / CM)
    r = math.sqrt(J * J + d * d) / 2
    np.testing.assert_allclose(e, np.sort([J / 4, J / 4, -J / 4 + r, -J / 4 - r]), atol=1e-12)
    T = t2t.antisymmetric_tensor((0.3, -0.2, 0.1))
    torch.testing.assert_close(T, -T.T)


@pytest.mark.parametrize("s1,s2", [(1.0, 1.0), (1.5, 1.0), (2.5, 2.5)])
def test_biquadratic_and_quadrupolar_lande(s1, s2):
    """Zero field, isotropic couplings: Telser's E(S) = J x + j (x^2 - X1X2/3) + 24 JQ (x^2 + x/2 - X1X2/3),
    x = [S(S+1) - X1 - X2]/2; torchspin's levels + dimer_energy_offset reproduce it."""
    J, j, JQ = 1.7, 0.21, 0.013
    sy = t2t.dimer2sys(site1=dict(s=s1), site2=dict(s=s2), J=J, j=j, JQ=JQ)
    off = t2t.dimer_energy_offset(s1, s2, j, JQ)
    e = (_levels(sy).numpy() + off) / CM
    x1, x2 = s1 * (s1 + 1), s2 * (s2 + 1)
    ref = []
    for S in np.arange(abs(s1 - s2), s1 + s2 + 1):
        x = (S * (S + 1) - x1 - x2) / 2
        ref += [J * x + j * (x * x - x1 * x2 / 3) + 24 * JQ * (x * x + x / 2 - x1 * x2 / 3)] * int(2 * S + 1)
    np.testing.assert_allclose(e, np.sort(ref), atol=1e-11)


def test_units_and_constants():
    a = t2t.spin2sys(s=1.0, D=0.3, E=0.05)
    b = t2t.spin2sys(s=1.0, D=0.3 * CM, E=0.05 * CM, units="MHz")
    c = t2t.spin2sys(s=1.0, D=0.3 * CM / 1e3, E=0.05 * CM / 1e3, units="GHz")
    for o in (b, c):
        torch.testing.assert_close(o.D, a.D, rtol=1e-14, atol=0)
    # telser-tools-like constants: g rescaled so mu_B g is the same, cm^-1 -> MHz from mu_b_mhz/mu_b_cm
    k = types.SimpleNamespace(mu_b_mhz=1.3996, mu_b_cm=4.6686437e-5, mu_n_mhz=7.6225914e-4)
    d = t2t.spin2sys(s=1.0, g=(2.0, 2.0, 2.0), D=0.3, constants=k)
    assert float(d.g[0, 0]) == pytest.approx(2.0 * 13.996 / t2t.MU_B_MHZ_PER_MT, rel=1e-15)
    assert float(d.D[0, 2]) == pytest.approx(0.2 * 1.3996 / 4.6686437e-5, rel=1e-15)
    with pytest.raises(ValueError):
        t2t.spin2sys(s=1.0, D=0.3, units="furlong")


def test_s0_nuclear_system_closed_form():
    """S = 0 (DDPOWH NMR/NQR): I = 1, axial traceless Q, zero field: I.Q.I = (3/2) Qzz Iz^2 - Qzz, so
    m = 0 at -Qzz and m = +-1 at Qzz/2."""
    sy = t2t.spin2sys(s=0.0, I=1.0, g_n=0.403761, Q=(-0.4, -0.4, 0.8))
    assert sy.S == [] and sy.Nucs == ["14N"]
    np.testing.assert_allclose(_levels(sy).numpy(), [-0.8, 0.4, 0.4], atol=1e-13)


def test_isotope_for():
    assert t2t.isotope_for(1.0, "14N") == "14N"
    assert t2t.isotope_for(1.0, "2H (D)") == "2H"
    assert t2t.isotope_for(1.5, "4Xx") == "7Li"          # unknown label: stand-in with the same I
    assert t2t.isotope_for(0.5, "14N") == "1H"           # label disagrees with I: stand-in
    sy = t2t.spin2sys(s=0.5, I=1.5, g_n=0.5, A=(10.0, 10.0, 10.0), nucleus=None)
    from torchspin.nucdata import nucgval
    assert sy.Nucs == ["7Li"] and float(sy.gnscale[0]) * nucgval("7Li") == pytest.approx(0.5)


def test_unmapped_features_raise_or_drop():
    with pytest.raises(NotImplementedError, match="second-order Zeeman"):
        t2t.spin2sys(s=1.0, D=0.1, w=0.5)
    with pytest.raises(NotImplementedError, match="I.S\\^3"):
        t2t.spin2sys(s=1.5, I=0.5, g_n=5.58, A=(1.0, 1.0, 1.0), U=2.0)
    with pytest.raises(ValueError, match="S = 0 needs a nucleus"):
        t2t.spin2sys(s=0.0)
    dropped = []
    with pytest.warns(UserWarning, match="dropped w"):
        sy = t2t.spin2sys(s=1.0, D=0.1, w=0.5, strict=False, dropped=dropped)
    assert dropped == ["w (second-order Zeeman)"] and sy.S == [1.0]
    with pytest.raises(NotImplementedError, match="mixed H1 mode"):
        t2t.epr_experiment(9.5, (3000, 3600), 100, h1mode=1.0)
    with pytest.raises(NotImplementedError, match="molecular frame"):
        t2t.dimer2sys(site1=dict(s=2.0, B40=0.01), site2=dict(s=2.0, B40=0.01, D=0.1, D_euler=(10.0, 0.0, 0.0)))


def test_frames_equal_full_tensors():
    """Euler frames (gFrame/DFrame/AFrame/QFrame, the torchspin convention) and R diag R^T give the same H."""
    kw = dict(s=1.0, g=(2.0, 2.1, 2.2), D=0.2, E=0.03, I=1.0, g_n=0.403761, A=(5.0, 10.0, 40.0),
              A_euler=(20.0, 50.0, -30.0), Q=(-0.5, -0.3, 0.8), Q_euler=(-10.0, 70.0, 15.0))
    hs = [ham(t2t.spin2sys(**kw, full_tensors=f)) for f in (False, True)]
    for x, y in zip(*hs):
        torch.testing.assert_close(x, y, rtol=0, atol=1e-10)
    site1 = dict(s=1.0, g=(2.0, 2.05, 2.1), D=0.3, E=0.05)
    site2 = dict(s=1.5, g=(1.95, 2.1, 2.3), g_euler=(10.0, 40.0, 70.0), D=-0.4, E=0.1, D_euler=(-20.0, 30.0, 5.0),
                 mu=0.02)
    hs = [ham(t2t.dimer2sys(site1=site1, site2=site2, J=0.7, Ddip=0.1, Edip=0.02, dip_euler=(5.0, 15.0, 25.0),
                            d=(0.1, -0.05, 0.02), full_tensors=f)) for f in (False, True)]
    for x, y in zip(*hs):
        torch.testing.assert_close(x, y, rtol=0, atol=1e-9)
    r = t2t.euler_matrix(0.3, 1.1, -0.4)
    torch.testing.assert_close(r @ r.T, torch.eye(3, dtype=F64))
    assert float(torch.linalg.det(r)) == pytest.approx(1.0)


def _fd(f, x0, h=1e-3):
    """Richardson-extrapolated central difference (eigenvalues of ~1e4 MHz: plain small steps hit round-off)."""
    return (8 * (f(x0 + h) - f(x0 - h)) - (f(x0 + 2 * h) - f(x0 - 2 * h))) / (12 * h)


def test_gradients_through_couplings_and_full_tensor_angles():
    n = torch.tensor([0.3, -0.5, 0.81], dtype=F64)
    n = n / n.norm()

    def e_beta(beta):
        sy = t2t.spin2sys(s=0.5, g=(2.0, 2.05, 2.2), I=1.0, g_n=0.403761, A=(5.0, 10.0, 40.0),
                          A_euler=torch.stack([torch.tensor(20.0, dtype=F64), beta, torch.tensor(-30.0, dtype=F64)]),
                          full_tensors=True)
        return _levels_t(sy, 340.0, n)[0]

    beta = torch.tensor(50.0, dtype=F64, requires_grad=True)
    (gr,) = torch.autograd.grad(e_beta(beta), beta)
    ref = _fd(lambda x: float(e_beta(torch.tensor(x, dtype=F64))), 50.0)
    assert float(gr) == pytest.approx(ref, rel=1e-6)

    def e_dimer(J, Ddip):
        sy = t2t.dimer2sys(site1=dict(s=0.5, g=(2.05, 2.06, 2.25)), site2=dict(s=0.5, g=(2.05, 2.06, 2.25)), J=J,
                           Ddip=Ddip, Edip=0.01, dip_euler=(10.0, 30.0, 0.0))
        return _levels_t(sy, 330.0, n)

    J = torch.tensor(-3.0, dtype=F64, requires_grad=True)
    Dd = torch.tensor(0.08, dtype=F64, requires_grad=True)
    gJ, gD = torch.autograd.grad(e_dimer(J, Dd)[1], (J, Dd))
    with torch.no_grad():
        fJ = _fd(lambda x: float(e_dimer(torch.tensor(x, dtype=F64), Dd)[1]), -3.0)
        fD = _fd(lambda x: float(e_dimer(J, torch.tensor(x, dtype=F64))[1]), 0.08)
    assert float(gJ) == pytest.approx(fJ, rel=1e-6)
    assert float(gD) == pytest.approx(fD, rel=1e-6)


def test_epr_experiment_and_ordering():
    exp = t2t.epr_experiment(9.5, (2900.0, 3600.0), 350, deriv=1, temperature=4.0, h1mode=-1.0)
    assert exp.Range == [290.0, 360.0] and exp.nPoints == 351 and exp.Harmonic == 1 and exp.Mode == "parallel"
    assert exp.Temperature == 4.0
    exp = t2t.epr_experiment(240.0, (8.0, 9.0), 100, field_units="T")
    assert exp.Range == [8000.0, 9000.0]
    f = t2t.ordering(0.5, -1.2)
    beta, gamma = 0.7, 2.1                          # field at theta = beta, phi = pi - gamma
    th, ph = beta, math.pi - gamma
    assert float(f(np.array([0.3]), beta, gamma)[0]) == pytest.approx(
        math.exp(0.5 * math.cos(ph) ** 2 - 1.2 * math.cos(th) ** 2))


def test_ham_full_field_path_agrees_with_moment_path_for_ham1xx():
    """Bug B11, fixed in torchspin 0.6.7: ham(sys, B0) == H0 - B.mu for the converter's Ham1xx rows."""
    sy = t2t.spin2sys(s=1.5, u=(0.05, 0.02, 0.03))
    H0, mx, my, mz = ham(sy)
    B = [10.0, 20.0, 30.0]
    torch.testing.assert_close(ham(sy, B0=B), H0 - (B[0] * mx + B[1] * my + B[2] * mz))


# ============================================================================== 2. parity with telser-tools
def _tt(name):
    return pytest.importorskip(f"telser_tools.{name}")


@pytest.fixture(scope="module")
def C():
    return pytest.importorskip("telser_tools.spin").CODATA2018


def _clusters(e, tol):
    lab = np.zeros(len(e), int)
    for k in range(1, len(e)):
        lab[k] = lab[k - 1] + (0 if e[k] - e[k - 1] < tol else 1)
    return lab


def _block_sum(P, lab):
    M = np.zeros((len(lab), lab.max() + 1))
    M[np.arange(len(lab)), lab] = 1.0
    return M.T @ P @ M


def _ts_probabilities(sy, b_mt, n, h1mode, mub):
    """B1-averaged P from torchspin's moment operators, on Telser's scale (S = 1/2, g = 2 -> 1)."""
    H0, mx, my, mz = ham(sy)
    e, v = torch.linalg.eigh(H0 - b_mt * (n[0] * mx + n[1] * my + n[2] * mz))
    mu = [(v.conj().T @ m @ v).numpy() for m in (mx, my, mz)]
    nmu = n[0] * mu[0] + n[1] * mu[1] + n[2] * mu[2]
    tot = sum(np.abs(m) ** 2 for m in mu)
    if b_mt == 0:
        p = tot / 3
    elif h1mode < 0:
        p = np.abs(nmu) ** 2
    else:
        p = (tot - np.abs(nmu) ** 2) / 2 + (h1mode * np.abs(nmu) ** 2 if h1mode > 0 else 0.0)
    return e.numpy(), p / mub ** 2


def _telser_probabilities_full_moment(ops, vectors, n, h1mode, constants):
    """Telser's B1 average with the moment -dH/dB (z1/mu_B: g.S, the u term and the nuclear moment)."""
    m = ops.z1 / constants.mu_b_cm
    el = np.swapaxes(vectors.conj(), -1, -2)[None] @ m @ vectors[None]
    tot = np.sum(np.abs(el) ** 2, axis=0)
    if n is None:
        return tot / 3
    nm = np.abs(np.einsum("k,kij->ij", n, el)) ** 2
    if h1mode < 0:
        return nm
    return (tot - nm) / 2 + (h1mode * nm if h1mode > 0 else 0.0)


def _ddzhq_input(name):
    ddzhq = _tt("ddzhq")
    inp = ddzhq.read_input(DATA["ddzhq"][name])
    return ddzhq, inp


@pytest.mark.parametrize("name", sorted(DATA["ddzhq"]))
def test_ddzhq_levels_and_probabilities(name, C):
    """Every DDZHQ fixture: torchspin levels at every cycle = ddzhq.run(exact) (rel 1e-12); B1-averaged
    probabilities = Telser's with the full moment -dH/dB (rel 1e-9 of the largest, summed over degenerate
    levels), and = telser-tools' g.S + nuclear moment probabilities when u = 0."""
    ddzhq, inp = _ddzhq_input(name)
    spin = _tt("spin")
    has_w_u = (inp.s >= 1 and inp.w != 0) or (inp.s >= 1.5 and inp.nucleus_spin >= 0.5 and inp.U != 0)
    if has_w_u:
        with pytest.raises(NotImplementedError):
            t2t.ddzhq2sys(inp, constants=C)
        inp = dataclasses.replace(inp, w=0.0, U=0.0)            # the documented gap: compare without w and U
    cv = t2t.ddzhq2sys(inp, constants=C)
    res = ddzhq.run(inp)
    ops = spin.build(res.params, C)
    mub = C.mu_b_mhz * 10
    assert len(res.cycles) == len(cv.extras["cycles"])
    for cy, c in zip(res.cycles, cv.extras["cycles"]):
        n = c["direction"].numpy()
        assert c["field_mT"] * 10 == pytest.approx(cy.field_g, rel=1e-13, abs=1e-12)
        e, P = _ts_probabilities(cv.sys, c["field_mT"], n, inp.h1mode, mub)
        np.testing.assert_allclose(e * C.mhz_to_cm, cy.energies, rtol=0, atol=1e-12 * max(1.0, np.abs(cy.energies).max()))
        lab = _clusters(cy.energies, 1e-7 * max(1.0, np.abs(cy.energies).max()))
        nn = None if c["zero_field"] else n
        ref = _telser_probabilities_full_moment(ops, cy.vectors, nn, inp.h1mode, C)
        a, b = _block_sum(P, lab), _block_sum(ref, lab)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-9 * max(1e-30, np.abs(b).max()))
        if not any(inp.u) or inp.s < 1.5:
            gs = spin.epr_probabilities(ops, spin.Eigensystem(cy.energies, cy.vectors), nn, inp.h1mode,
                                        nuclear_moment=True)
            np.testing.assert_allclose(a, _block_sum(gs, lab), rtol=0, atol=1e-9 * max(1e-30, np.abs(b).max()))


def test_ddzhq_u_term_moment_convention(C):
    """With u != 0 Telser's EPRTRANS moment g.S differs from -dH/dB (torchspin's): measurable, not round-off."""
    ddzhq, inp = _ddzhq_input("s32_u_de")
    spin = _tt("spin")
    cv = t2t.ddzhq2sys(inp, constants=C)
    res = ddzhq.run(inp)
    ops = spin.build(res.params, C)
    cy, c = res.cycles[0], cv.extras["cycles"][0]
    _, P = _ts_probabilities(cv.sys, c["field_mT"], c["direction"].numpy(), inp.h1mode, C.mu_b_mhz * 10)
    gs = spin.epr_probabilities(ops, spin.Eigensystem(cy.energies, cy.vectors), c["direction"].numpy(), inp.h1mode)
    assert np.abs(P - gs).max() > 1e-4 * np.abs(gs).max()


def test_ddzhq_default_constants_codata_2022(C):
    """Without constants=, torchspin's CODATA 2022 mu_B: Zeeman levels 1.4e-9 apart (telser-tools' own bound)."""
    ddzhq, inp = _ddzhq_input("example_run1")
    cv = t2t.ddzhq2sys(inp)
    res = ddzhq.run(inp)
    for cy in res.cycles:
        n = t2t._direction(math.radians(cy.cycle.theta), math.radians(cy.cycle.phi)).numpy()
        e = _levels(cv.sys, cy.field_g / 10, n).numpy() / CM
        np.testing.assert_allclose(e, cy.energies, rtol=0, atol=5e-9 * np.abs(cy.energies).max())


def _random_params(spin, nuclear, rng, s, iso):
    nuc = nuclear.lookup(iso) if iso else None
    g = tuple(rng.uniform(1.9, 2.3, 3))
    kw = dict(D=rng.uniform(-3, 3), E=rng.uniform(0, 0.5)) if s >= 1 else {}
    if s >= 2:
        kw.update(B40=0.01, B42=-0.006, B44=0.012)
    if s >= 1.5:
        kw.update(u=tuple(rng.uniform(-0.03, 0.03, 3)))
    if nuc:
        Q = (-0.5, -0.4, 0.9) if nuc.spin >= 1 else (0.0, 0.0, 0.0)
        kw["nucleus"] = spin.Nucleus(nuc.spin, nuc.g_n, tuple(rng.uniform(-150, 150, 3)), tuple(rng.uniform(-90, 90, 3)),
                                     0.0, Q, tuple(rng.uniform(-90, 90, 3)), nuc.label)
    return spin.SpinParameters(s, g, **kw)


@pytest.mark.parametrize("s,iso,freq,rng_mt,D", [(0.5, 14, 9.5, (250.0, 400.0), None),
                                                  (0.5, 63, 9.5, (250.0, 400.0), None),
                                                  (1.5, 1, 9.5, (130.0, 420.0), 5.0),
                                                  (2.5, 0, 34.0, (1000.0, 1400.0), 0.05)])
def test_resfields_match_telser_resonance_fields(s, iso, freq, rng_mt, D, C):
    """torchspin.resfields on the converted system = telser_tools.spin.resonance_fields: fields rel 1e-7 (resfields
    convergence), intensities = mu_B^2 P/(2I+1)/|dE/dB| rel 1e-6 (P with the moment -dH/dB: nuclear moment and u term)."""
    spin, nuclear = _tt("spin"), _tt("nuclear")
    import torchspin as ts
    from torchspin.resfields import resfields

    rng = np.random.default_rng(int(10 * s) + iso)
    p = _random_params(spin, nuclear, rng, s, iso)
    if D is not None:
        p = dataclasses.replace(p, D=D, E=D / 20)
    sy = t2t.spin2sys(p, constants=C)
    H0, mx, my, mz = ham(sy)
    ops = spin.build(p, C)
    dirs = rng.normal(size=(4, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    mine = spin.resonance_fields(ops, dirs, freq, (rng_mt[0] * 10, rng_mt[1] * 10), n_grid=800, tol_g=1e-10,
                                 nuclear_moment=True)
    exp = ts.Experiment(mwFreq=freq, Range=list(rng_mt), nPoints=256)
    pol = 1.0 / (2 * p.nucleus.spin + 1) if p.nucleus else 1.0
    mub = C.mu_b_mhz * 10
    found = 0
    for k, d in enumerate(dirs):
        th, ph = math.acos(max(-1.0, min(1.0, d[2]))), math.atan2(d[1], d[0])
        bres, inten, _ = resfields(H0, mx, my, mz, ph, th, exp, ts.Options(Threshold=0.0), sys=sy)
        sel = mine.orientation == k
        mf = mine.field[sel] / 10
        # Telser's P with the full moment -dH/dB (the u term is in torchspin's moment operators)
        prob = np.array([_telser_probabilities_full_moment(ops, spin.solve(ops, bf, d).vectors, d, 0.0, C)[i, j]
                         for bf, i, j in zip(mine.field[sel], mine.lower[sel], mine.upper[sel])])
        mexp = mub ** 2 * prob * pol / (np.abs(mine.slope[sel]) * 10 / C.mhz_to_cm)
        it = inten.numpy()
        for bt, i_t in zip(bres.numpy()[it > 1e-6 * it.max()], it[it > 1e-6 * it.max()]):
            j = int(np.argmin(np.abs(mf - bt)))
            assert mf[j] == pytest.approx(bt, rel=1e-7)
            assert mexp[j] == pytest.approx(i_t, rel=1e-6, abs=1e-9 * it.max())
            found += 1
    assert found >= 4


@pytest.mark.parametrize("name", ["example_run1", "s05_14n_quad", "s05_1h_rot"])
def test_ddzhq_nmr_lines_match_endorfrq(name, C):
    """NMR lines: endorfrq frequencies = level differences; intensities = (g_N mu_N)^2 x Telser's NMR P."""
    ddzhq, inp = _ddzhq_input(name)
    spin = _tt("spin")
    from torchspin import endorfrq

    cv = t2t.ddzhq2sys(inp, constants=C)
    res = ddzhq.run(inp)
    ops = spin.build(res.params, C)
    gn_mu = inp.g_n * C.mu_n_mhz * 10
    for cy, c in zip(res.cycles, cv.extras["cycles"]):
        if c["zero_field"]:
            continue
        fr, it, tr = endorfrq(cv.sys, c["field_mT"], phi=c["phi"], theta=c["theta"], threshold=0.0)
        P = spin.nmr_probabilities(ops, spin.Eigensystem(cy.energies, cy.vectors), c["direction"].numpy())
        e = cy.energies / C.mhz_to_cm
        for f, i_t, (a, b) in zip(fr.numpy(), it.numpy(), tr.numpy()):
            assert e[b] - e[a] == pytest.approx(f, abs=1e-8 * max(1.0, np.abs(e).max()))
            if i_t > 1e-6 * it.numpy().max():
                assert gn_mu ** 2 * P[a, b] == pytest.approx(i_t, rel=1e-8)


@pytest.mark.parametrize("name", sorted(DATA["djdze"]))
def test_djdze_levels_and_probabilities(name, C):
    """Every DJDZE fixture: torchspin levels + dimer_energy_offset = djdze.run(exact) (rel 1e-12) at every cycle;
    probabilities summed over degenerate levels = Telser's with -dH/dB (rel 1e-9; = g1.S1 + g2.S2 unless mu != 0)."""
    djdze, dimer = _tt("djdze"), _tt("dimer")
    inp = djdze.read_input(DATA["djdze"][name])
    cv = t2t.djdze2sys(inp, constants=C)
    res = djdze.run(inp)
    ops = dimer.build(res.params, C)
    mub = C.mu_b_mhz * 10
    for cy, c in zip(res.cycles, cv.extras["cycles"]):
        assert c["field_mT"] * 10 == pytest.approx(cy.field_g, rel=1e-13, abs=1e-12)
        n = c["direction"].numpy()
        e, P = _ts_probabilities(cv.sys, c["field_mT"], n, inp.h1mode, mub)
        e = (e + cv.extras["energy_offset"]) * C.mhz_to_cm
        np.testing.assert_allclose(e, cy.energies, rtol=0, atol=1e-12 * max(1.0, np.abs(cy.energies).max()))
        lab = _clusters(cy.energies, 1e-7 * max(1.0, np.abs(cy.energies).max()))
        nn = None if c["zero_field"] else n
        ref = _telser_probabilities_full_moment(ops, cy.vectors, nn, inp.h1mode, C)
        np.testing.assert_allclose(_block_sum(P, lab), _block_sum(ref, lab), rtol=0,
                                   atol=1e-9 * max(1e-30, np.abs(ref).max()))


# ------------------------------------------------------------------------------ powder spectra
def _norm_dev(ref, y):
    a, b = np.asarray(ref) / np.abs(ref).max(), np.asarray(y) / np.abs(y).max()
    return float(np.abs(a - b).max()), float(np.corrcoef(a, b)[0, 1])


def _ddpowh(name, **replace):
    ddpowh = _tt("ddpowh")
    inp = ddpowh.read_input(DATA["ddpowh"][name])
    return ddpowh, dataclasses.replace(inp, **replace) if replace else inp


def _w(w):
    return dict(epr_width=(w, w, w))


# Tolerances: 1.5-2x the difference measured with both grids refined (telser-tools igloo N vs 2N, pepper
# GridSize 91 vs 181), see docs/lft_gap_analysis.md.  Line widths are >= ~5 field steps (pepper bins, Telser
# samples: the difference scales as (step/FWHM)^2).
@pytest.mark.parametrize("name,replace,tol", [
    ("gen_rhombic_abs", {}, 3e-3),                              # measured 0.15 %
    ("gen_cu_abs", _w(40.0), 1e-2),                             # 0.5 % (2.6 % at W = 20 MHz: 6 G steps)
    ("rhombic_s12", dict(igloo=60, epr_deriv=1), 4e-2),         # derivative = Harmonic 1: 2.5 % (linearized line)
    ("s1_D", dict(igloo=60), 1.5e-2),                           # 0.6 %
    ("gen_s1_pop", _w(200.0), 3e-2),                            # 2 K, normalized populations: 1.6 %
])
def test_ddpowh_field_sweeps_match_pepper(name, replace, tol):
    ddpowh, inp = _ddpowh(name, **replace)
    sp = ddpowh.simulate(inp, frequency_factor=False).spectra[0]
    cv = t2t.ddpowh2sys(inp)
    x, y = pepper(cv.sys, cv.exp, Options(GridSize=91, Threshold=0.0))
    np.testing.assert_allclose(x.numpy(), np.asarray(sp.x) * (1e3 if inp.field_units == "T" else 0.1), rtol=1e-12)
    dev, corr = _norm_dev(sp.y, y.numpy())
    assert dev < tol and corr > 0.999, (dev, corr)


def test_ddpowh_third_order_zeeman_powder():
    """S = 3/2, D = 2 cm^-1, u = (0.05, 0.03, 0.02): pepper with the converted Ham term is within 7 % of DDPOWH
    (4.7 % measured; the same system with u = 0 differs by 4.0 % in both programs: D-edge integration), and
    leaving u out of the torchspin system is 3x worse (16 %).  The per-orientation map of u is exact (levels,
    resfields); the remaining u-dependent part is Telser's g.S moment vs torchspin's -dH/dB."""
    ddpowh, inp = _ddpowh("gen_s32_u", **_w(150.0))
    sp = ddpowh.simulate(inp, frequency_factor=False).spectra[0]
    opt = Options(GridSize=91, Threshold=0.0)
    cv = t2t.ddpowh2sys(inp)
    dev, corr = _norm_dev(sp.y, pepper(cv.sys, cv.exp, opt)[1].numpy())
    cv0 = t2t.ddpowh2sys(dataclasses.replace(inp, u=(0.0, 0.0, 0.0)))
    dev0, _ = _norm_dev(sp.y, pepper(cv0.sys, cv0.exp, opt)[1].numpy())
    assert dev < 7e-2 and corr > 0.995, (dev, corr)
    assert dev0 > 2 * dev


def test_ddpowh_parallel_mode_is_pepper_parallel():
    """H1 mode < 0 (B1 || B0): S = 1, D = 0.05 cm^-1 has Delta m = 2 intensity; pepper Mode='parallel'."""
    ddpowh = _tt("ddpowh")
    inp = dataclasses.replace(ddpowh.read_input(DATA["ddpowh"]["s1_D"]), h1mode=-1.0, igloo=60)
    sp = ddpowh.simulate(inp, frequency_factor=False).spectra[0]
    cv = t2t.ddpowh2sys(inp)
    assert cv.exp.Mode == "parallel"
    _, y = pepper(cv.sys, cv.exp, Options(GridSize=91, Threshold=0.0))
    dev, corr = _norm_dev(sp.y, y.numpy())
    assert dev < 1.5e-2 and corr > 0.999, (dev, corr)            # measured 0.8 % (telser igloo 60 vs 120: 0.7 %)


def test_ddpowh_mixed_mode_relation():
    """Telser's mixed mode r: y = y_perp + r y_par (exact mode), so pepper(perp) + r pepper(par) with one
    common normalization reproduces it; the converter refuses the mixed mode itself (strict)."""
    ddpowh = _tt("ddpowh")
    base = dataclasses.replace(ddpowh.read_input(DATA["ddpowh"]["s1_D"]), igloo=60)
    r = 2.5
    with pytest.raises(NotImplementedError, match="mixed H1 mode"):
        t2t.ddpowh2sys(dataclasses.replace(base, h1mode=r))
    ym = ddpowh.simulate(dataclasses.replace(base, h1mode=r), frequency_factor=False).spectra[0].y
    yp = ddpowh.simulate(base, frequency_factor=False).spectra[0].y
    yq = ddpowh.simulate(dataclasses.replace(base, h1mode=-1.0), frequency_factor=False).spectra[0].y
    np.testing.assert_allclose(ym, yp + r * yq, rtol=1e-10, atol=1e-12 * np.abs(ym).max())
    cv = t2t.ddpowh2sys(base)
    opt = Options(GridSize=91, Threshold=0.0)
    _, tp = pepper(cv.sys, cv.exp, opt)
    _, tq = pepper(cv.sys, dataclasses.replace(cv.exp, Mode="parallel"), opt)
    # common scale from the perpendicular spectra, then the mixture
    k = float(np.abs(yp).max() / tp.abs().max())
    dev, corr = _norm_dev(ym, k * (tp.numpy() + r * tq.numpy()))
    assert dev < 2e-2 and corr > 0.999, (dev, corr)


def test_ddpowh_frequency_factor_relation():
    """Telser's Gaussian Flygare factor multiplies each line by its transition energy (cm^-1), which is h nu_mw
    within a line width: for a field sweep y_ff = h nu_mw y to O(W/nu)."""
    ddpowh, inp = _ddpowh("gen_rhombic_abs", igloo=20)
    a = ddpowh.simulate(inp).spectra[0].y
    b = ddpowh.simulate(inp, frequency_factor=False).spectra[0].y
    hv = inp.freq_ghz * 1e3 / CM
    assert float(np.sum(a) / np.sum(b)) == pytest.approx(hv, rel=2 * 20.0 / (inp.freq_ghz * 1e3))


def test_ddpowh_fdmr_matches_pepper_frequency_sweep():
    ddpowh, inp = _ddpowh("fdmr_s1_cm", igloo=40)
    with pytest.raises(NotImplementedError, match="zero-field FDMR"):        # spectrum 1 is at zero field
        t2t.ddpowh2sys(inp)
    sp = ddpowh.simulate(inp, frequency_factor=False).spectra[1]               # spectrum 2: 20000 G
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = t2t.ddpowh2sys(inp, strict=False)
    ex = cv.extras["spectra"][1]["exp"]
    assert ex.Field == pytest.approx(2000.0)
    assert ex.mwRange == pytest.approx([x * CM / 1e3 for x in inp.energy_range], rel=1e-14)
    x, y = pepper(cv.sys, ex, Options(GridSize=91, Threshold=0.0))
    np.testing.assert_allclose(x.numpy(), np.asarray(sp.x) * CM / 1e3, rtol=1e-12)
    dev, corr = _norm_dev(sp.y, y.numpy())
    assert dev < 3e-2 and corr > 0.999, (dev, corr)


def test_ddpowh_endor_matches_salt():
    """14N ENDOR (Q != 0: salt's matrix path).  salt labels each frequency bin by its lower edge, so its
    spectrum is compared half a bin later (telser-tools docs/ddpowh.md)."""
    from torchspin.salt import salt

    ddpowh, inp = _ddpowh("gen_endor_n14")
    sp = ddpowh.simulate(inp, frequency_factor=False).spectra[0]
    cv = t2t.ddpowh2sys(inp)
    s0 = cv.extras["spectra"][0]
    assert s0["exp"].Field == pytest.approx(331.1) and s0["salt"]["lw_mhz"] == pytest.approx(0.4)
    f, y = salt(cv.sys, s0["exp"], Options(GridSize=[61, 4], Threshold=0.0, Verbosity=0), **s0["salt"])
    f, y = f.numpy(), y.numpy()
    y = np.interp(f, f + 0.5 * (f[1] - f[0]), y)
    dev, corr = _norm_dev(sp.y, y)
    assert dev < 6e-3 and corr > 0.99999, (dev, corr)


@pytest.mark.parametrize("name,match", [("rhombic_s12_lorentz", "Lorentzian"), ("s32_strain", "strain"),
                                        ("mode_1_nrho8", "mixed H1"), ("s52_pop_w_u", "second-order Zeeman"),
                                        ("s72_I12_U", "I.S\\^3"), ("nqr_n14", "NMR/NQR")])
def test_ddpowh_unmapped_features(name, match):
    ddpowh, inp = _ddpowh(name)
    with pytest.raises(NotImplementedError, match=match):
        t2t.ddpowh2sys(inp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = t2t.ddpowh2sys(inp, strict=False)
    assert cv.extras["dropped"]


def test_ddpowh_partial_orientation_is_listed():
    ddpowh, inp = _ddpowh("partial_orient")
    cv = t2t.ddpowh2sys(inp)
    assert any("ORFACT" in d for d in cv.extras["dropped"]) == bool(inp.lam_x or inp.lam_z)


def _ddpowj(name, **replace):
    ddpowj = _tt("ddpowj")
    inp = ddpowj.read_input(DATA["ddpowj"][name])
    return ddpowj, dataclasses.replace(inp, **replace) if replace else inp


@pytest.mark.parametrize("name,replace,tol", [
    ("cu2_triplet", dict(igloo=40, **_w(60.0)), 3e-3),          # measured 0.06 %
    ("cu2_triplet_rot", dict(igloo=40, **_w(60.0)), 5e-3),      # 0.16 %
    # S = 1 with D ~ h nu (0.3, 0.2 vs 0.32 cm^-1): looping transitions with turning points (d nu/dB = 0), where
    # pepper's linearized field width W/|d nu/dB| breaks down and Telser's field-point line does not; 4.5-5.6 %
    # with both grids refined.  With D << h nu (s1_D above) the same mapping agrees to 0.6 %.
    ("s1_s12", dict(igloo=40, **_w(100.0)), 8e-2),
    ("s1_s0", dict(igloo=40, **_w(100.0)), 8e-2),               # S2 = 0: one-electron SpinSystem
    ("cu2_dm_hemisphere", dict(igloo=40), 2.5e-2),              # general d, hemisphere; telser igloo 30 vs 60: 1.8 %
    ("cu2_pop_4K", dict(igloo=40, **_w(60.0)), 1e-2),           # 0.4 %
    ("cu2_halffield", dict(igloo=40), 5e-3),                    # half-field (Delta M = 2) region: 0.2 %
])
def test_ddpowj_matches_pepper(name, replace, tol):
    ddpowj, inp = _ddpowj(name, **replace)
    sp = ddpowj.simulate(inp, frequency_factor=False).spectra[0]
    cv = t2t.ddpowj2sys(inp)
    x, y = pepper(cv.sys, cv.exp, Options(GridSize=91, Threshold=0.0))
    np.testing.assert_allclose(x.numpy(), np.asarray(sp.x) * 0.1, rtol=1e-12)
    dev, corr = _norm_dev(sp.y, y.numpy())
    assert dev < tol and corr > (0.995 if tol > 0.05 else 0.999), (dev, corr)


@pytest.mark.parametrize("name,match", [("cu2_strain", "strain"), ("cu2_bq_lorentz", "Lorentzian")])
def test_ddpowj_unmapped_features(name, match):
    ddpowj, inp = _ddpowj(name)
    with pytest.raises(NotImplementedError, match=match):
        t2t.ddpowj2sys(inp)


def test_ddpowj_s12_strain_is_a_width_doubling_in_telser_exact_mode():
    """Documented relation (DDPOWJ-10 and the WMS1*WMS2 note): with strain on, b = 0 and an S = 1/2 partner, the
    exact-mode basis weights WMS1*WMS2 are 0, so the strain factor is 1 and the spectrum equals the no-strain one,
    which the converter (strict=False, strain dropped) maps."""
    ddpowj, inp = _ddpowj("cu2_strain", igloo=20)
    a = ddpowj.simulate(inp, frequency_factor=False).spectra[0].y
    b = ddpowj.simulate(dataclasses.replace(inp, strain=False), frequency_factor=False).spectra[0].y
    if float(inp.strain_b) == 0.0:
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14 * np.abs(b).max())


# ------------------------------------------------------------------------------ small gaps filled in lft
def test_quadrupole_echo_closed_form_and_torchspin_q_form():
    """P = 3/2 Qzz, eta = (Qxx - Qyy)/Qzz with |Qzz| >= |Qyy| >= |Qxx|; [e2qQ, eta] is SpinSystem's Q shorthand."""
    q = (-0.5, -0.3, 0.8)
    echo = t2t.quadrupole_echo(q, 1.0)
    assert float(echo.p_par) == pytest.approx(1.2) and float(echo.eta) == pytest.approx(0.25)
    assert float(echo.e2qQ) == pytest.approx(1.6)
    sy = SpinSystem(S=[0.5], Nucs=["14N"], Q=[[float(echo.e2qQ), float(echo.eta)]])
    np.testing.assert_allclose(np.sort(sy.Q.numpy().ravel()), np.sort(q), atol=1e-14)
    assert t2t.quadrupole_echo((0.0, 0.0, 0.0), 1.0) is None
    e = t2t.quadrupole_echo((3.0, -3.0, 0.0), 1.5)                    # |Qmax| = |Qmid|: eta = 1
    assert abs(float(e.eta)) == pytest.approx(1.0)
    Q = torch.tensor([-0.5, -0.3, 0.8], dtype=F64, requires_grad=True)
    (gq,) = torch.autograd.grad(t2t.quadrupole_echo(Q, 1.0).eta, Q)
    np.testing.assert_allclose(gq.numpy(), [-1 / 0.8, 1 / 0.8, -(-0.3 + 0.5) / 0.8 ** 2], rtol=1e-12)


@pytest.mark.parametrize("q", [(-0.5, -0.3, 0.8), (5.0, 6.0, -11.0), (3.0, -3.0, 0.0), (0.2, 0.7, -0.9)])
def test_quadrupole_echo_matches_ddzhq(q):
    ref = _tt("ddzhq.program").quadrupole_echo(q, 1.5, "exact")
    mine = t2t.quadrupole_echo(q, 1.5)
    assert float(mine.p_par) == pytest.approx(ref.p_par) and float(mine.e2qQ) == pytest.approx(ref.e2qq)
    assert abs(float(mine.eta)) == pytest.approx(abs(ref.eta))


def test_spin_projections_closed_form():
    """Isotropic triplet (two S = 1/2, g = 2): each site has <S_i>.n = m/2 for every field direction."""
    sy = t2t.dimer2sys(site1=dict(s=0.5), site2=dict(s=0.5), J=-5.0)
    for n in ((0, 0, 1), (0.6, 0.0, 0.8), (1 / math.sqrt(3),) * 3):
        sp = t2t.spin_projections(sy, 340.0, n).numpy()
        np.testing.assert_allclose(np.sort(sp[:, 0]), [-0.5, 0.0, 0.0, 0.5], atol=1e-12)
        np.testing.assert_allclose(sp[:, 0], sp[:, 1], atol=1e-12)


def test_spin_projections_match_djdze(C):
    djdze = _tt("djdze")
    n_checked = 0
    for name, st in DATA["djdze"].items():
        inp = djdze.read_input(st)
        if not any(cy.spin_magnitudes for cy in inp.cycles):
            continue
        cv = t2t.djdze2sys(inp, constants=C)
        res = djdze.run(inp)
        for cy, c in zip(res.cycles, cv.extras["cycles"]):
            if not cy.spin_magnitudes or len(cv.sys.S) < 2:
                continue
            mine = t2t.spin_projections(cv.sys, c["field_mT"], c["direction"]).numpy()
            lab = _clusters(cy.energies, 1e-7 * max(1.0, np.abs(cy.energies).max()))
            for cl in np.unique(lab):
                levels = [k + 1 for k in np.nonzero(lab == cl)[0]]
                if not all(k in cy.spin_magnitudes for k in levels):
                    continue
                ref = np.sum([cy.spin_magnitudes[k] for k in levels], axis=0)
                np.testing.assert_allclose(mine[np.array(levels) - 1].sum(axis=0), ref, atol=1e-9)
                n_checked += 1
    assert n_checked > 10
