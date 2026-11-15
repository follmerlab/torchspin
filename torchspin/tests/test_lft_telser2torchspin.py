"""torchspin.lft milestone M5: Telser's spin-Hamiltonian programs through torchspin (``lft.telser2torchspin``).

1. Self-contained checks of the converters (closed forms, unit and constant handling, Euler frames vs full
   tensors, gradients, errors for unmapped features).
2. Parity with telser-tools on its fixtures (stdin texts in ``tests/data/telser/spin_family_inputs.json``):
   live when telser-tools is importable, else against its frozen outputs (data/telser/refs, ``_telser_refs``):

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
import inspect
import json
import math
import types
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import torch

from torchspin import Options, SpinSystem, pepper
from torchspin.ham import ham
from torchspin.lft import telser2torchspin as t2t
from torchspin.lft.units import energy_factor
from torchspin.tests import _telser_refs as R

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


def test_w_and_u_are_mapped():
    """w -> Sys.Ham lB = 2 rows (needs Options.QuadraticZeeman), U -> Sys.H0Extra (keeps its graph)."""
    from torchspin.spinops import sop

    sy = t2t.spin2sys(s=1.0, D=0.1, w=0.5)
    assert set(sy.Ham) >= {"222"} and t2t.quadratic_zeeman_options(sy) == {"QuadraticZeeman": True}
    assert t2t.quadratic_zeeman_options(t2t.spin2sys(s=1.0, D=0.1)) == {}
    U = torch.tensor(2.0, dtype=F64, requires_grad=True)
    su = t2t.spin2sys(s=1.5, I=0.5, g_n=5.58, A=(1.0, 1.0, 1.0), U=U)
    assert len(su.H0Extra) == 1 and su.H0Extra[0][0].requires_grad
    # w (mu_B B.S)^2 on the full-field path, up to the scalar w mu_B^2 B^2 X/3
    for s in (1.0, 2.5):
        sy = t2t.spin2sys(s=s, g=(2.0, 2.0, 2.0), w=0.3)
        n = torch.tensor([0.3, 0.5, 0.81], dtype=F64)
        n = n / n.norm()
        b = 400.0
        nS = sum(float(n[c]) * sop([s], [1, c + 1]) for c in range(3))
        H = 2.0 * t2t.MU_B_MHZ_PER_MT * b * nS + 0.3 * CM * (t2t.MU_B_MHZ_PER_MT / CM * b) ** 2 * (nS @ nS)
        ref = torch.linalg.eigvalsh(H) - float(t2t.second_order_zeeman_offset(s, 0.3, b))
        np.testing.assert_allclose(torch.linalg.eigvalsh(ham(sy, (b * n).tolist())).numpy(), ref.numpy(),
                                   atol=1e-10 * float(ref.abs().max()))


def test_unmapped_features_raise_or_drop():
    with pytest.raises(ValueError, match="S = 0 needs a nucleus"):
        t2t.spin2sys(s=0.0)
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
# Every telser-tools value comes from an oracle: live when telser-tools is importable, else the frozen copy in
# data/telser/refs/telser2torchspin.npz (see _telser_refs).  The converters read telser-tools inputs by
# attribute: inputs are stored as data and rebuilt as frozen dataclasses with the same fields and the input
# classes' properties (checked against the stored values); constants as a namespace of telser-tools' values.

# the properties of telser-tools' input classes (the oracle checks that this list is complete)
_PROPERTIES = {
    "ddpowh.infile.DdpowhInput": dict(ns=lambda i: int(2 * i.s + 1.01), ni=lambda i: int(2 * i.nucleus_spin + 1.01),
                                      n_rot=lambda i: 2 if i.rotate else 1),
    "ddpowj.infile.DdpowjInput": dict(s1=lambda i: i.sites[0].s, s2=lambda i: i.sites[1].s,
                                      n=lambda i: int(round(2 * i.s1 + 1)) * int(round(2 * i.s2 + 1)),
                                      n_rot=lambda i: 2 if (i.rotate2 or i.rotate_dip) else 1),
    "ddzhq.infile.DdzhqInput": dict(ns=lambda i: int(2 * i.s + 1.01), ni=lambda i: int(2 * i.nucleus_spin + 1.01)),
    "ddzhq.infile.Cycle": dict(computed=lambda c: not (c.entered == "G" and c.value <= 0),
                               zero_field=lambda c: c.entered == "H" and c.value <= 0),
    "djdze.infile.DjdzeInput": dict(s1=lambda i: i.sites[0].s, s2=lambda i: i.sites[1].s,
                                    n=lambda i: int(round(2 * i.s1 + 1)) * int(round(2 * i.s2 + 1))),
}
_CONSTANTS = ("mu_b_cm", "mu_b_mhz", "mu_n_mhz", "k_b_cm", "deg", "mhz_to_cm")


def _class_name(cls) -> str:
    return cls.__module__.removeprefix("telser_tools.") + "." + cls.__name__


def _props_of(obj, name_of) -> list:
    """[class name, {property: value}] of every dataclass in ``obj`` with properties, depth first."""
    out = []

    def walk(o):
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            name = name_of(o)
            if _PROPERTIES.get(name):
                out.append([name, {p: getattr(o, p) for p in sorted(_PROPERTIES[name])}])
            for f in dataclasses.fields(o):
                walk(getattr(o, f.name))
        elif isinstance(o, (list, tuple)):
            for x in o:
                walk(x)
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(k)
                walk(v)
    walk(obj)
    return out


def freeze_input(obj) -> dict:
    """A telser-tools input (nested frozen dataclasses) as plain data, with its property values."""
    def fr(o):
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            name = _class_name(type(o))
            props = {n for n, v in inspect.getmembers(type(o)) if isinstance(v, property)}
            assert props == set(_PROPERTIES.get(name, {})), (name, props)
            return {"_dataclass": name, "fields": {f.name: fr(getattr(o, f.name)) for f in dataclasses.fields(o)}}
        if isinstance(o, tuple):
            return {"_tuple": [fr(x) for x in o]}
        if isinstance(o, list):
            return [fr(x) for x in o]
        if isinstance(o, dict):
            return {"_dict": [[fr(k), fr(v)] for k, v in o.items()]}
        if o is None or isinstance(o, (bool, int, float, str)):
            return o
        raise TypeError(f"unexpected {type(o).__name__} in a telser-tools input")
    return dict(obj=fr(obj), props=_props_of(obj, lambda o: _class_name(type(o))))


@lru_cache(maxsize=None)
def _thawed_class(name: str, fields: tuple):
    ns = {p: property(f) for p, f in _PROPERTIES.get(name, {}).items()}
    ns["_telser_class"] = name
    return dataclasses.make_dataclass(name.rsplit(".", 1)[1], list(fields), namespace=ns, frozen=True)


def thaw_input(d: dict):
    """The object :func:`freeze_input` stored (attribute access, properties, ``dataclasses.replace``)."""
    def th(v):
        if isinstance(v, list):
            return [th(x) for x in v]
        if isinstance(v, dict):
            if "_dataclass" in v:
                f = v["fields"]
                return _thawed_class(v["_dataclass"], tuple(f))(**{k: th(x) for k, x in f.items()})
            if "_tuple" in v:
                return tuple(th(x) for x in v["_tuple"])
            if "_dict" in v:
                return {th(k): th(x) for k, x in v["_dict"]}
        return v
    obj = th(d["obj"])
    assert _props_of(obj, lambda o: type(o)._telser_class) == d["props"]
    return obj


def freeze_constants(c) -> dict:
    return {a: float(getattr(c, a)) for a in _CONSTANTS}


def _telser_direction(theta_deg, phi_deg, c):
    """Telser's field direction for a cycle (ddzhq/djdze program: degrees * c.deg, sin/cos)."""
    t, p = theta_deg * c.deg, phi_deg * c.deg
    return np.array([math.sin(t) * math.cos(p), math.sin(t) * math.sin(p), math.cos(t)])


_INPUT_CASES = [f"{prog}/{name}" for prog in ("ddzhq", "djdze", "ddpowh", "ddpowj") for name in DATA[prog]]


@R.oracle("telser2torchspin", cases=_INPUT_CASES)
def _telser_input(case):
    import importlib
    prog, name = case.split("/")
    return freeze_input(importlib.import_module(f"telser_tools.{prog}").read_input(DATA[prog][name]))


def _input(prog, name, **replace):
    inp = thaw_input(R.ref(_telser_input, f"{prog}/{name}"))
    return dataclasses.replace(inp, **replace) if replace else inp


@R.oracle("telser2torchspin")
def _telser_constants():
    from telser_tools.spin import CODATA2018
    return freeze_constants(CODATA2018)


@pytest.fixture(scope="module")
def C():
    """telser-tools' CODATA 2018 constants (spin.CODATA2018) as a namespace."""
    return types.SimpleNamespace(**R.ref(_telser_constants))


def _clusters(e, tol):
    lab = np.zeros(len(e), int)
    for k in range(1, len(e)):
        lab[k] = lab[k - 1] + (0 if e[k] - e[k - 1] < tol else 1)
    return lab


def _level_clusters(e):
    return _clusters(e, 1e-7 * max(1.0, np.abs(e).max()))


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


def _has_w_u(inp):
    return (inp.s >= 1 and inp.w != 0) or (inp.s >= 1.5 and inp.nucleus_spin >= 0.5 and inp.U != 0)


@R.oracle("telser2torchspin", cases=sorted(DATA["ddzhq"]))
def _telser_ddzhq(name):
    """Per computed cycle: field, levels and the B1-averaged probabilities summed over degenerate levels
    (invariant under the eigenvectors' choice), with the moment -dH/dB (P) and, where it equals it, g.S + the
    nuclear moment (P_gS); at Telser's field direction.  With w or U: the full-model levels (w_energies), and
    the probabilities of the linear-moment model (w = U = 0)."""
    from telser_tools import ddzhq, spin
    C = spin.CODATA2018
    inp = ddzhq.read_input(DATA["ddzhq"][name])
    out = {}
    if _has_w_u(inp):
        out["w_energies"] = [cy.energies for cy in ddzhq.run(inp).cycles]
        inp = dataclasses.replace(inp, w=0.0, U=0.0)
    res = ddzhq.run(inp)
    ops = spin.build(res.params, C)
    cycles = []
    for cy in res.cycles:
        zero = bool(cy.cycle.zero_field)
        n = None if zero else _telser_direction(cy.cycle.theta, cy.cycle.phi, C)
        lab = _level_clusters(cy.energies)
        d = dict(field_g=cy.field_g, energies=cy.energies, zero_field=zero,
                 P=_block_sum(_telser_probabilities_full_moment(ops, cy.vectors, n, inp.h1mode, C), lab))
        if not any(inp.u) or inp.s < 1.5:
            d["P_gS"] = _block_sum(spin.epr_probabilities(ops, spin.Eigensystem(cy.energies, cy.vectors), n,
                                                          inp.h1mode, nuclear_moment=True), lab)
        cycles.append(d)
    out["cycles"] = cycles
    return out


@pytest.mark.parametrize("name", sorted(DATA["ddzhq"]))
def test_ddzhq_levels_and_probabilities(name, C):
    """Every DDZHQ fixture: torchspin levels at every cycle = ddzhq.run(exact) (rel 1e-12); B1-averaged
    probabilities = Telser's with the full moment -dH/dB (rel 1e-9 of the largest, summed over degenerate
    levels), and = telser-tools' g.S + nuclear moment probabilities when u = 0."""
    ref = R.ref(_telser_ddzhq, name)
    inp = _input("ddzhq", name)
    if _has_w_u(inp):                                           # w -> Sys.Ham lB = 2, U -> H0Extra: full-field levels
        cw = t2t.ddzhq2sys(inp, constants=C)
        for want, c in zip(ref["w_energies"], cw.extras["cycles"]):
            n = c["direction"].numpy()
            e = torch.linalg.eigvalsh(ham(cw.sys, (c["field_mT"] * n).tolist())).numpy()
            if inp.s >= 1 and inp.w != 0:
                e = e + float(t2t.second_order_zeeman_offset(inp.s, inp.w, c["field_mT"], constants=C,
                                                             mu_b=C.mu_b_mhz * 10))
            np.testing.assert_allclose(e * C.mhz_to_cm, want, rtol=0, atol=1e-12 * max(1.0, np.abs(want).max()))
        inp = dataclasses.replace(inp, w=0.0, U=0.0)            # probabilities below: the linear-moment model
    cv = t2t.ddzhq2sys(inp, constants=C)
    mub = C.mu_b_mhz * 10
    assert len(ref["cycles"]) == len(cv.extras["cycles"])
    for cy, c in zip(ref["cycles"], cv.extras["cycles"]):
        n = c["direction"].numpy()
        E = cy["energies"]
        assert c["field_mT"] * 10 == pytest.approx(cy["field_g"], rel=1e-13, abs=1e-12)
        assert c["zero_field"] == cy["zero_field"]
        e, P = _ts_probabilities(cv.sys, c["field_mT"], n, inp.h1mode, mub)
        np.testing.assert_allclose(e * C.mhz_to_cm, E, rtol=0, atol=1e-12 * max(1.0, np.abs(E).max()))
        a, b = _block_sum(P, _level_clusters(E)), cy["P"]
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-9 * max(1e-30, np.abs(b).max()))
        if not any(inp.u) or inp.s < 1.5:
            np.testing.assert_allclose(a, cy["P_gS"], rtol=0, atol=1e-9 * max(1e-30, np.abs(b).max()))


@R.oracle("telser2torchspin")
def _telser_ddzhq_u_term():
    """s32_u_de, first cycle: telser-tools' g.S probabilities (EPRTRANS moment) at Telser's field direction."""
    from telser_tools import ddzhq, spin
    C = spin.CODATA2018
    inp = ddzhq.read_input(DATA["ddzhq"]["s32_u_de"])
    res = ddzhq.run(inp)
    ops = spin.build(res.params, C)
    cy = res.cycles[0]
    n = _telser_direction(cy.cycle.theta, cy.cycle.phi, C)
    return dict(gs=spin.epr_probabilities(ops, spin.Eigensystem(cy.energies, cy.vectors), n, inp.h1mode))


def test_ddzhq_u_term_moment_convention(C):
    """With u != 0 Telser's EPRTRANS moment g.S differs from -dH/dB (torchspin's): measurable, not round-off."""
    inp = _input("ddzhq", "s32_u_de")
    gs = R.ref(_telser_ddzhq_u_term)["gs"]
    cv = t2t.ddzhq2sys(inp, constants=C)
    c = cv.extras["cycles"][0]
    _, P = _ts_probabilities(cv.sys, c["field_mT"], c["direction"].numpy(), inp.h1mode, C.mu_b_mhz * 10)
    assert np.abs(P - gs).max() > 1e-4 * np.abs(gs).max()


@R.oracle("telser2torchspin")
def _telser_ddzhq_example_run1():
    from telser_tools import ddzhq
    res = ddzhq.run(ddzhq.read_input(DATA["ddzhq"]["example_run1"]))
    return [dict(theta=cy.cycle.theta, phi=cy.cycle.phi, field_g=cy.field_g, energies=cy.energies)
            for cy in res.cycles]


def test_ddzhq_default_constants_codata_2022():
    """Without constants=, torchspin's CODATA 2022 mu_B: Zeeman levels 1.4e-9 apart (telser-tools' own bound)."""
    cv = t2t.ddzhq2sys(_input("ddzhq", "example_run1"))
    for cy in R.ref(_telser_ddzhq_example_run1):
        n = t2t._direction(math.radians(cy["theta"]), math.radians(cy["phi"])).numpy()
        e = _levels(cv.sys, cy["field_g"] / 10, n).numpy() / CM
        np.testing.assert_allclose(e, cy["energies"], rtol=0, atol=5e-9 * np.abs(cy["energies"]).max())


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


_RESFIELDS_CASES = [(0.5, 14, 9.5, (250.0, 400.0), None), (0.5, 63, 9.5, (250.0, 400.0), None),
                    (1.5, 1, 9.5, (130.0, 420.0), 5.0), (2.5, 0, 34.0, (1000.0, 1400.0), 0.05)]


@R.oracle("telser2torchspin", cases=range(len(_RESFIELDS_CASES)))
def _telser_resonance_fields(k):
    """Random spin parameters (telser-tools' nuclear table), 4 random directions, spin.resonance_fields and, per
    resonance, Telser's probability with the full moment -dH/dB at the resonance field."""
    from telser_tools import nuclear, spin
    C = spin.CODATA2018
    s, iso, freq, rng_mt, D = _RESFIELDS_CASES[k]
    rng = np.random.default_rng(int(10 * s) + iso)
    p = _random_params(spin, nuclear, rng, s, iso)
    if D is not None:
        p = dataclasses.replace(p, D=D, E=D / 20)
    ops = spin.build(p, C)
    dirs = rng.normal(size=(4, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    mine = spin.resonance_fields(ops, dirs, freq, (rng_mt[0] * 10, rng_mt[1] * 10), n_grid=800, tol_g=1e-10,
                                 nuclear_moment=True)
    prob = np.array([_telser_probabilities_full_moment(ops, spin.solve(ops, bf, dirs[o]).vectors, dirs[o], 0.0, C)[i, j]
                     for bf, o, i, j in zip(mine.field, mine.orientation, mine.lower, mine.upper)])
    return dict(params=freeze_input(p), dirs=dirs, orientation=np.asarray(mine.orientation),
                field=np.asarray(mine.field), slope=np.asarray(mine.slope), prob=prob)


@pytest.mark.parametrize("case", range(len(_RESFIELDS_CASES)),
                         ids=["s{}-{}-{}-{}-{}".format(*c[:3], *c[3]) for c in _RESFIELDS_CASES])
def test_resfields_match_telser_resonance_fields(case, C):
    """torchspin.resfields on the converted system = telser_tools.spin.resonance_fields: fields rel 1e-7 (resfields
    convergence), intensities = mu_B^2 P/(2I+1)/|dE/dB| rel 1e-6 (P with the moment -dH/dB: nuclear moment and u term)."""
    import torchspin as ts
    from torchspin.resfields import resfields

    _, _, freq, rng_mt, _ = _RESFIELDS_CASES[case]
    ref = R.ref(_telser_resonance_fields, case)
    p = thaw_input(ref["params"])
    sy = t2t.spin2sys(p, constants=C)
    H0, mx, my, mz = ham(sy)
    exp = ts.Experiment(mwFreq=freq, Range=list(rng_mt), nPoints=256)
    pol = 1.0 / (2 * p.nucleus.spin + 1) if p.nucleus else 1.0
    mub = C.mu_b_mhz * 10
    found = 0
    for k, d in enumerate(ref["dirs"]):
        th, ph = math.acos(max(-1.0, min(1.0, d[2]))), math.atan2(d[1], d[0])
        bres, inten, _ = resfields(H0, mx, my, mz, ph, th, exp, ts.Options(Threshold=0.0), sys=sy)
        sel = ref["orientation"] == k
        mf = ref["field"][sel] / 10
        # Telser's P with the full moment -dH/dB (the u term is in torchspin's moment operators)
        mexp = mub ** 2 * ref["prob"][sel] * pol / (np.abs(ref["slope"][sel]) * 10 / C.mhz_to_cm)
        it = inten.numpy()
        for bt, i_t in zip(bres.numpy()[it > 1e-6 * it.max()], it[it > 1e-6 * it.max()]):
            j = int(np.argmin(np.abs(mf - bt)))
            assert mf[j] == pytest.approx(bt, rel=1e-7)
            assert mexp[j] == pytest.approx(i_t, rel=1e-6, abs=1e-9 * it.max())
            found += 1
    assert found >= 4


_NMR_NAMES = ["example_run1", "s05_14n_quad", "s05_1h_rot"]


@R.oracle("telser2torchspin", cases=_NMR_NAMES)
def _telser_ddzhq_nmr(name):
    """Per cycle: levels and spin.nmr_probabilities at Telser's field direction (None at zero field)."""
    from telser_tools import ddzhq, spin
    C = spin.CODATA2018
    res = ddzhq.run(ddzhq.read_input(DATA["ddzhq"][name]))
    ops = spin.build(res.params, C)
    out = []
    for cy in res.cycles:
        zero = bool(cy.cycle.zero_field)
        P = None if zero else spin.nmr_probabilities(ops, spin.Eigensystem(cy.energies, cy.vectors),
                                                     _telser_direction(cy.cycle.theta, cy.cycle.phi, C))
        out.append(dict(zero_field=zero, energies=cy.energies, P=P))
    return out


@pytest.mark.parametrize("name", _NMR_NAMES)
def test_ddzhq_nmr_lines_match_endorfrq(name, C):
    """NMR lines: endorfrq frequencies = level differences; intensities = (g_N mu_N)^2 x Telser's NMR P."""
    from torchspin import endorfrq

    inp = _input("ddzhq", name)
    cv = t2t.ddzhq2sys(inp, constants=C)
    gn_mu = inp.g_n * C.mu_n_mhz * 10
    ref = R.ref(_telser_ddzhq_nmr, name)
    assert len(ref) == len(cv.extras["cycles"])
    for cy, c in zip(ref, cv.extras["cycles"]):
        assert c["zero_field"] == cy["zero_field"]
        if c["zero_field"]:
            continue
        fr, it, tr = endorfrq(cv.sys, c["field_mT"], phi=c["phi"], theta=c["theta"], threshold=0.0)
        P = cy["P"]
        e = cy["energies"] / C.mhz_to_cm
        for f, i_t, (a, b) in zip(fr.numpy(), it.numpy(), tr.numpy() - 1):     # endorfrq pairs are 1-based
            assert e[b] - e[a] == pytest.approx(f, abs=1e-8 * max(1.0, np.abs(e).max()))
            if i_t > 1e-6 * it.numpy().max():
                assert gn_mu ** 2 * P[a, b] == pytest.approx(i_t, rel=1e-8)


@R.oracle("telser2torchspin", cases=sorted(DATA["djdze"]))
def _telser_djdze(name):
    """Per cycle: field, levels, and the B1-averaged probabilities with -dH/dB summed over degenerate levels,
    at Telser's field direction (None at zero field, where DJDZE keeps the previous orientation)."""
    from telser_tools import dimer, djdze, spin
    C = spin.CODATA2018
    inp = djdze.read_input(DATA["djdze"][name])
    res = djdze.run(inp)
    ops = dimer.build(res.params, C)
    out = []
    for cy in res.cycles:
        zero = cy.field_g == 0
        n = None if zero else _telser_direction(cy.theta, cy.phi, C)
        P = _telser_probabilities_full_moment(ops, cy.vectors, n, inp.h1mode, C)
        out.append(dict(field_g=cy.field_g, energies=cy.energies, zero_field=bool(zero),
                        P=_block_sum(P, _level_clusters(cy.energies))))
    return out


@pytest.mark.parametrize("name", sorted(DATA["djdze"]))
def test_djdze_levels_and_probabilities(name, C):
    """Every DJDZE fixture: torchspin levels + dimer_energy_offset = djdze.run(exact) (rel 1e-12) at every cycle;
    probabilities summed over degenerate levels = Telser's with -dH/dB (rel 1e-9; = g1.S1 + g2.S2 unless mu != 0)."""
    inp = _input("djdze", name)
    cv = t2t.djdze2sys(inp, constants=C)
    mub = C.mu_b_mhz * 10
    for cy, c in zip(R.ref(_telser_djdze, name), cv.extras["cycles"]):
        assert c["field_mT"] * 10 == pytest.approx(cy["field_g"], rel=1e-13, abs=1e-12)
        assert c["zero_field"] == cy["zero_field"]
        n = c["direction"].numpy()
        E = cy["energies"]
        e, P = _ts_probabilities(cv.sys, c["field_mT"], n, inp.h1mode, mub)
        e = (e + cv.extras["energy_offset"]) * C.mhz_to_cm
        np.testing.assert_allclose(e, E, rtol=0, atol=1e-12 * max(1.0, np.abs(E).max()))
        ref = cy["P"]
        np.testing.assert_allclose(_block_sum(P, _level_clusters(E)), ref, rtol=0,
                                   atol=1e-9 * max(1e-30, np.abs(ref).max()))


# ------------------------------------------------------------------------------ powder spectra
def _norm_dev(ref, y):
    a, b = np.asarray(ref) / np.abs(ref).max(), np.asarray(y) / np.abs(y).max()
    return float(np.abs(a - b).max()), float(np.corrcoef(a, b)[0, 1])


def _ddpowh(name, **replace):
    return _input("ddpowh", name, **replace)


def _w(w):
    return dict(epr_width=(w, w, w))


def _spectra(prog, inp, **kw):
    """telser-tools' exact-mode spectra (x, y) of a (replaced) input of DATA[prog]."""
    import importlib
    res = importlib.import_module(f"telser_tools.{prog}").simulate(inp, **kw)
    return [dict(x=np.asarray(s.x, dtype=float), y=np.asarray(s.y, dtype=float)) for s in res.spectra]


def _telser_read(prog, name, **replace):
    import importlib
    inp = importlib.import_module(f"telser_tools.{prog}").read_input(DATA[prog][name])
    return dataclasses.replace(inp, **replace) if replace else inp


# Tolerances: 1.5-2x the difference measured with both grids refined (telser-tools igloo N vs 2N, pepper
# GridSize 91 vs 181), see docs/lft_gap_analysis.md.  Line widths are >= ~5 field steps (pepper bins, Telser
# samples: the difference scales as (step/FWHM)^2).
_SWEEPS = {
    "gen_rhombic_abs": ({}, 3e-3),                              # measured 0.15 %
    "gen_cu_abs": (_w(40.0), 1e-2),                             # 0.5 % (2.6 % at W = 20 MHz: 6 G steps)
    "rhombic_s12": (dict(igloo=60, epr_deriv=1), 4e-2),         # derivative = Harmonic 1: 2.5 % (linearized line)
    "s1_D": (dict(igloo=60), 1.5e-2),                           # 0.6 %
    "gen_s1_pop": (_w(200.0), 3e-2),                            # 2 K, normalized populations: 1.6 %
}


@R.oracle("telser2torchspin", cases=list(_SWEEPS))
def _telser_ddpowh_sweep(name):
    return _spectra("ddpowh", _telser_read("ddpowh", name, **_SWEEPS[name][0]), frequency_factor=False)[0]


@pytest.mark.parametrize("name", list(_SWEEPS))
def test_ddpowh_field_sweeps_match_pepper(name):
    replace, tol = _SWEEPS[name]
    inp = _ddpowh(name, **replace)
    sp = R.ref(_telser_ddpowh_sweep, name)
    cv = t2t.ddpowh2sys(inp)
    x, y = pepper(cv.sys, cv.exp, Options(GridSize=91, Threshold=0.0))
    np.testing.assert_allclose(x.numpy(), sp["x"] * (1e3 if inp.field_units == "T" else 0.1), rtol=1e-12)
    dev, corr = _norm_dev(sp["y"], y.numpy())
    assert dev < tol and corr > 0.999, (dev, corr)


@R.oracle("telser2torchspin")
def _telser_ddpowh_u():
    return _spectra("ddpowh", _telser_read("ddpowh", "gen_s32_u", **_w(150.0)), frequency_factor=False)[0]["y"]


def test_ddpowh_third_order_zeeman_powder():
    """S = 3/2, D = 2 cm^-1, u = (0.05, 0.03, 0.02): pepper with the converted Ham term is within 7 % of DDPOWH
    (4.7 % measured; the same system with u = 0 differs by 4.0 % in both programs: D-edge integration), and
    leaving u out of the torchspin system is 3x worse (16 %).  The per-orientation map of u is exact (levels,
    resfields); the remaining u-dependent part is Telser's g.S moment vs torchspin's -dH/dB."""
    inp = _ddpowh("gen_s32_u", **_w(150.0))
    y_ref = R.ref(_telser_ddpowh_u)
    opt = Options(GridSize=91, Threshold=0.0)
    cv = t2t.ddpowh2sys(inp)
    dev, corr = _norm_dev(y_ref, pepper(cv.sys, cv.exp, opt)[1].numpy())
    cv0 = t2t.ddpowh2sys(dataclasses.replace(inp, u=(0.0, 0.0, 0.0)))
    dev0, _ = _norm_dev(y_ref, pepper(cv0.sys, cv0.exp, opt)[1].numpy())
    assert dev < 7e-2 and corr > 0.995, (dev, corr)
    assert dev0 > 2 * dev


@R.oracle("telser2torchspin")
def _telser_ddpowh_parallel():
    inp = _telser_read("ddpowh", "s1_D", h1mode=-1.0, igloo=60)
    return _spectra("ddpowh", inp, frequency_factor=False)[0]["y"]


def test_ddpowh_parallel_mode_is_pepper_parallel():
    """H1 mode < 0 (B1 || B0): S = 1, D = 0.05 cm^-1 has Delta m = 2 intensity; pepper Mode='parallel'."""
    inp = _ddpowh("s1_D", h1mode=-1.0, igloo=60)
    cv = t2t.ddpowh2sys(inp)
    assert cv.exp.Mode == "parallel"
    _, y = pepper(cv.sys, cv.exp, Options(GridSize=91, Threshold=0.0))
    dev, corr = _norm_dev(R.ref(_telser_ddpowh_parallel), y.numpy())
    assert dev < 1.5e-2 and corr > 0.999, (dev, corr)            # measured 0.8 % (telser igloo 60 vs 120: 0.7 %)


_MIXED_R = 2.5


@R.oracle("telser2torchspin")
def _telser_ddpowh_mixed():
    base = _telser_read("ddpowh", "s1_D", igloo=60)
    y = {k: _spectra("ddpowh", dataclasses.replace(base, h1mode=h), frequency_factor=False)[0]["y"]
         for k, h in (("ym", _MIXED_R), ("yq", -1.0))}
    return dict(y, yp=_spectra("ddpowh", base, frequency_factor=False)[0]["y"])


def test_ddpowh_mixed_mode_relation():
    """Telser's mixed mode r: y = y_perp + r y_par (exact mode), so pepper(perp) + r pepper(par) with one
    common normalization reproduces it; the converter refuses the mixed mode itself (strict)."""
    base = _ddpowh("s1_D", igloo=60)
    r = _MIXED_R
    with pytest.raises(NotImplementedError, match="mixed H1 mode"):
        t2t.ddpowh2sys(dataclasses.replace(base, h1mode=r))
    ref = R.ref(_telser_ddpowh_mixed)
    ym, yp, yq = ref["ym"], ref["yp"], ref["yq"]
    np.testing.assert_allclose(ym, yp + r * yq, rtol=1e-10, atol=1e-12 * np.abs(ym).max())
    cv = t2t.ddpowh2sys(base)
    opt = Options(GridSize=91, Threshold=0.0)
    _, tp = pepper(cv.sys, cv.exp, opt)
    _, tq = pepper(cv.sys, dataclasses.replace(cv.exp, Mode="parallel"), opt)
    # common scale from the perpendicular spectra, then the mixture
    k = float(np.abs(yp).max() / tp.abs().max())
    dev, corr = _norm_dev(ym, k * (tp.numpy() + r * tq.numpy()))
    assert dev < 2e-2 and corr > 0.999, (dev, corr)


@R.oracle("telser2torchspin")
def _telser_ddpowh_frequency_factor():
    inp = _telser_read("ddpowh", "gen_rhombic_abs", igloo=20)
    return dict(a=_spectra("ddpowh", inp)[0]["y"], b=_spectra("ddpowh", inp, frequency_factor=False)[0]["y"])


def test_ddpowh_frequency_factor_relation():
    """Telser's Gaussian Flygare factor multiplies each line by its transition energy (cm^-1), which is h nu_mw
    within a line width: for a field sweep y_ff = h nu_mw y to O(W/nu)."""
    inp = _ddpowh("gen_rhombic_abs", igloo=20)
    ref = R.ref(_telser_ddpowh_frequency_factor)
    a, b = ref["a"], ref["b"]
    hv = inp.freq_ghz * 1e3 / CM
    assert float(np.sum(a) / np.sum(b)) == pytest.approx(hv, rel=2 * 20.0 / (inp.freq_ghz * 1e3))


@R.oracle("telser2torchspin")
def _telser_ddpowh_fdmr():
    return _spectra("ddpowh", _telser_read("ddpowh", "fdmr_s1_cm", igloo=40), frequency_factor=False)


def test_ddpowh_fdmr_matches_pepper_frequency_sweep():
    inp = _ddpowh("fdmr_s1_cm", igloo=40)
    res = R.ref(_telser_ddpowh_fdmr)
    cv = t2t.ddpowh2sys(inp)                                                    # spectrum 1 is at zero field
    z = cv.extras["spectra"][0]
    assert z["exp"].Field == 0.0 and z["options"] == {"ZeroField": True}
    _, yz = pepper(cv.sys, z["exp"], Options(GridSize=91, Threshold=0.0, **z["options"]))
    dz, cz = _norm_dev(res[0]["y"], yz.numpy())
    assert dz < 3e-2 and cz > 0.999, (dz, cz)
    sp = res[1]                                                                 # spectrum 2: 20000 G
    ex = cv.extras["spectra"][1]["exp"]
    assert ex.Field == pytest.approx(2000.0)
    assert ex.mwRange == pytest.approx([x * CM / 1e3 for x in inp.energy_range], rel=1e-14)
    x, y = pepper(cv.sys, ex, Options(GridSize=91, Threshold=0.0))
    np.testing.assert_allclose(x.numpy(), sp["x"] * CM / 1e3, rtol=1e-12)
    dev, corr = _norm_dev(sp["y"], y.numpy())
    assert dev < 3e-2 and corr > 0.999, (dev, corr)


@R.oracle("telser2torchspin")
def _telser_ddpowh_endor():
    return _spectra("ddpowh", _telser_read("ddpowh", "gen_endor_n14"), frequency_factor=False)[0]["y"]


def test_ddpowh_endor_matches_salt():
    """14N ENDOR (Q != 0: salt's matrix path).  salt labels each frequency bin by its lower edge, so its
    spectrum is compared half a bin later (telser-tools docs/ddpowh.md)."""
    from torchspin.salt import salt

    inp = _ddpowh("gen_endor_n14")
    y_ref = R.ref(_telser_ddpowh_endor)
    cv = t2t.ddpowh2sys(inp)
    s0 = cv.extras["spectra"][0]
    assert s0["exp"].Field == pytest.approx(331.1) and s0["salt"]["lw_mhz"] == pytest.approx(0.4)
    f, y = salt(cv.sys, s0["exp"], Options(GridSize=[61, 4], Threshold=0.0, Verbosity=0), **s0["salt"])
    f, y = f.numpy(), y.numpy()
    y = np.interp(f, f + 0.5 * (f[1] - f[0]), y)
    dev, corr = _norm_dev(y_ref, y)
    assert dev < 6e-3 and corr > 0.99999, (dev, corr)


@pytest.mark.parametrize("name,match", [("s32_strain", "strain"),
                                        ("mode_1_nrho8", "mixed H1"), ("nqr_n14", "NMR/NQR")])
def test_ddpowh_unmapped_features(name, match):
    inp = _ddpowh(name)
    with pytest.raises(NotImplementedError, match=match):
        t2t.ddpowh2sys(inp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = t2t.ddpowh2sys(inp, strict=False)
    assert cv.extras["dropped"]


# Lorentzian EPR lines -> SpinSystem.HStrainL = 2 W (torchspin 0.7.7).  Telser's cutoff is lifted (pepper does
# not cut lines) and the Flygare factor is off.  Tolerances 2x the difference measured with both grids refined
# (telser igloo 30/60, pepper GridSize 91/181); the Gaussian counterparts are in the tests above.
_LORENTZ = {
    "isotropic": (dict(igloo=60), 3e-3),                                               # measured 0.14 %
    "anisotropic": (dict(igloo=40, epr_width=(40.0, 80.0, 150.0)), 1e-3),              # anisotropic W: 0.04 %
    "derivative": (dict(igloo=40, epr_width=(40.0, 80.0, 150.0), epr_deriv=1), 4e-2),  # 1.9 % (Gaussian: 2.5 %)
}


@R.oracle("telser2torchspin", cases=list(_LORENTZ))
def _telser_ddpowh_lorentz(case):
    inp = _telser_read("ddpowh", "rhombic_s12_lorentz", epr_cutoff=10000, **_LORENTZ[case][0])
    return _spectra("ddpowh", inp, frequency_factor=False)[0]["y"]


@pytest.mark.parametrize("case", list(_LORENTZ))
def test_ddpowh_lorentzian_matches_pepper_hstrainl(case):
    replace, tol = _LORENTZ[case]
    inp = _ddpowh("rhombic_s12_lorentz", epr_cutoff=10000, **replace)
    cv = t2t.ddpowh2sys(inp)
    assert cv.sys.HStrain is None
    np.testing.assert_allclose(cv.sys.HStrainL.numpy(), 2.0 * np.asarray(inp.epr_width), rtol=1e-15)
    assert any("ICUTOFF" in n for n in cv.extras["notes"])
    x, y = pepper(cv.sys, cv.exp, Options(GridSize=91, Threshold=0.0))
    dev, corr = _norm_dev(R.ref(_telser_ddpowh_lorentz, case), y.numpy())
    assert dev < tol and corr > 0.999, (dev, corr)


@R.oracle("telser2torchspin")
def _telser_ddpowh_lorentz_fdmr():
    inp = _telser_read("ddpowh", "fdmr_s1_cm", igloo=40, epr_shape="L", epr_cutoff=10000)
    return [s["y"] for s in _spectra("ddpowh", inp, frequency_factor=False)]


def test_ddpowh_lorentzian_fdmr_matches_pepper_hstrainl():
    """Zero field (Options.ZeroField) and 2 T; measured 1.07 % / 0.57 % (Gaussian lines: 0.68 % / 0.47 %)."""
    inp = _ddpowh("fdmr_s1_cm", igloo=40, epr_shape="L", epr_cutoff=10000)
    res = R.ref(_telser_ddpowh_lorentz_fdmr)
    cv = t2t.ddpowh2sys(inp)
    assert cv.sys.HStrainL is not None and cv.sys.HStrain is None
    assert len(res) == len(cv.extras["spectra"])
    for y_ref, z, tol in zip(res, cv.extras["spectra"], (2.5e-2, 1.2e-2)):
        _, y = pepper(cv.sys, z["exp"], Options(GridSize=91, Threshold=0.0, **z["options"]))
        dev, corr = _norm_dev(y_ref, y.numpy())
        assert dev < tol and corr > 0.999, (dev, corr)


def test_ddpowh_lorentzian_endor_is_not_mapped():
    """salt's orientation selection is Gaussian (it ignores HStrainL): Lorentzian ENDOR stays a strict refusal."""
    inp = _ddpowh("gen_endor_n14", epr_shape="L")
    with pytest.raises(NotImplementedError, match="Lorentzian"):
        t2t.ddpowh2sys(inp)


def test_ddpowh_partial_orientation_is_listed():
    inp = _ddpowh("partial_orient")
    cv = t2t.ddpowh2sys(inp)
    assert any("ORFACT" in d for d in cv.extras["dropped"]) == bool(inp.lam_x or inp.lam_z)


def _ddpowj(name, **replace):
    return _input("ddpowj", name, **replace)


_DDPOWJ_PEPPER = {
    "cu2_triplet": (dict(igloo=40, **_w(60.0)), 3e-3),          # measured 0.06 %
    "cu2_triplet_rot": (dict(igloo=40, **_w(60.0)), 5e-3),      # 0.16 %
    # S = 1 with D ~ h nu (0.3, 0.2 vs 0.32 cm^-1): looping transitions with turning points (d nu/dB = 0), where
    # pepper's linearized field width W/|d nu/dB| breaks down and Telser's field-point line does not; 4.5-5.6 %
    # with both grids refined.  With D << h nu (s1_D above) the same mapping agrees to 0.6 %.
    "s1_s12": (dict(igloo=40, **_w(100.0)), 8e-2),
    "s1_s0": (dict(igloo=40, **_w(100.0)), 8e-2),               # S2 = 0: one-electron SpinSystem
    "cu2_dm_hemisphere": (dict(igloo=40), 2.5e-2),              # general d, hemisphere; telser igloo 30 vs 60: 1.8 %
    "cu2_pop_4K": (dict(igloo=40, **_w(60.0)), 1e-2),           # 0.4 %
    "cu2_halffield": (dict(igloo=40), 5e-3),                    # half-field (Delta M = 2) region: 0.2 %
}


@R.oracle("telser2torchspin", cases=list(_DDPOWJ_PEPPER))
def _telser_ddpowj_sweep(name):
    return _spectra("ddpowj", _telser_read("ddpowj", name, **_DDPOWJ_PEPPER[name][0]), frequency_factor=False)[0]


@pytest.mark.parametrize("name", list(_DDPOWJ_PEPPER))
def test_ddpowj_matches_pepper(name):
    replace, tol = _DDPOWJ_PEPPER[name]
    inp = _ddpowj(name, **replace)
    sp = R.ref(_telser_ddpowj_sweep, name)
    cv = t2t.ddpowj2sys(inp)
    x, y = pepper(cv.sys, cv.exp, Options(GridSize=91, Threshold=0.0))
    np.testing.assert_allclose(x.numpy(), sp["x"] * 0.1, rtol=1e-12)
    dev, corr = _norm_dev(sp["y"], y.numpy())
    assert dev < tol and corr > (0.995 if tol > 0.05 else 0.999), (dev, corr)


@pytest.mark.parametrize("name,match", [("cu2_strain", "strain")])
def test_ddpowj_unmapped_features(name, match):
    inp = _ddpowj(name)
    with pytest.raises(NotImplementedError, match=match):
        t2t.ddpowj2sys(inp)


@R.oracle("telser2torchspin")
def _telser_ddpowj_lorentz():
    inp = _telser_read("ddpowj", "cu2_bq_lorentz", igloo=40, epr_cutoff=10000)
    return _spectra("ddpowj", inp, frequency_factor=False)[0]["y"]


def test_ddpowj_lorentzian_matches_pepper_hstrainl():
    """Lorentzian lines -> HStrainL = 2 W (0.7.7); cutoff lifted, Flygare off.  Measured 0.15 % (igloo 40/80,
    pepper GridSize 91/181: 0.144-0.150 %)."""
    inp = _ddpowj("cu2_bq_lorentz", igloo=40, epr_cutoff=10000)
    cv = t2t.ddpowj2sys(inp)
    assert cv.sys.HStrain is None and cv.sys.HStrainL is not None
    assert any("ICUTOFF" in n for n in cv.extras["notes"])
    x, y = pepper(cv.sys, cv.exp, Options(GridSize=91, Threshold=0.0))
    dev, corr = _norm_dev(R.ref(_telser_ddpowj_lorentz), y.numpy())
    assert dev < 3e-3 and corr > 0.999, (dev, corr)


@R.oracle("telser2torchspin")
def _telser_ddpowj_s12_strain():
    inp = _telser_read("ddpowj", "cu2_strain", igloo=20)
    return dict(a=_spectra("ddpowj", inp, frequency_factor=False)[0]["y"],
                b=_spectra("ddpowj", dataclasses.replace(inp, strain=False), frequency_factor=False)[0]["y"])


def test_ddpowj_s12_strain_is_a_width_doubling_in_telser_exact_mode():
    """Documented relation (DDPOWJ-10 and the WMS1*WMS2 note): with strain on, b = 0 and an S = 1/2 partner, the
    exact-mode basis weights WMS1*WMS2 are 0, so the strain factor is 1 and the spectrum equals the no-strain one,
    which the converter (strict=False, strain dropped) maps."""
    inp = _ddpowj("cu2_strain", igloo=20)
    ref = R.ref(_telser_ddpowj_s12_strain)
    a, b = ref["a"], ref["b"]
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


_ECHO_Q = [(-0.5, -0.3, 0.8), (5.0, 6.0, -11.0), (3.0, -3.0, 0.0), (0.2, 0.7, -0.9)]


@R.oracle("telser2torchspin", cases=range(len(_ECHO_Q)))
def _telser_quadrupole_echo(k):
    from telser_tools.ddzhq.program import quadrupole_echo
    ref = quadrupole_echo(_ECHO_Q[k], 1.5, "exact")
    return dict(p_par=ref.p_par, e2qq=ref.e2qq, eta=ref.eta)


@pytest.mark.parametrize("k", range(len(_ECHO_Q)), ids=[str(q) for q in _ECHO_Q])
def test_quadrupole_echo_matches_ddzhq(k):
    ref = R.ref(_telser_quadrupole_echo, k)
    mine = t2t.quadrupole_echo(_ECHO_Q[k], 1.5)
    assert float(mine.p_par) == pytest.approx(ref["p_par"]) and float(mine.e2qQ) == pytest.approx(ref["e2qq"])
    assert abs(float(mine.eta)) == pytest.approx(abs(ref["eta"]))


def test_spin_projections_closed_form():
    """Isotropic triplet (two S = 1/2, g = 2): each site has <S_i>.n = m/2 for every field direction."""
    sy = t2t.dimer2sys(site1=dict(s=0.5), site2=dict(s=0.5), J=-5.0)
    for n in ((0, 0, 1), (0.6, 0.0, 0.8), (1 / math.sqrt(3),) * 3):
        sp = t2t.spin_projections(sy, 340.0, n).numpy()
        np.testing.assert_allclose(np.sort(sp[:, 0]), [-0.5, 0.0, 0.0, 0.5], atol=1e-12)
        np.testing.assert_allclose(sp[:, 0], sp[:, 1], atol=1e-12)


@R.oracle("telser2torchspin")
def _telser_djdze_spin_projections():
    """Per DJDZE fixture with spin magnitudes, per cycle: the level clusters (degenerate levels) all of whose
    levels have DJDZE's <S_1>.n, <S_2>.n, and the sums over each such cluster (invariant under the
    eigenvectors' choice); None for a cycle without spin magnitudes."""
    from telser_tools import djdze
    out = {}
    for name, st in DATA["djdze"].items():
        inp = djdze.read_input(st)
        if not any(cy.spin_magnitudes for cy in inp.cycles):
            continue
        cycles = []
        for cy in djdze.run(inp).cycles:
            if not cy.spin_magnitudes:
                cycles.append(None)
                continue
            lab = _level_clusters(cy.energies)
            cl = []
            for c in np.unique(lab):
                levels = [k + 1 for k in np.nonzero(lab == c)[0]]
                if all(k in cy.spin_magnitudes for k in levels):
                    cl.append(dict(levels=levels, ref=np.sum([cy.spin_magnitudes[k] for k in levels], axis=0)))
            cycles.append(cl)
        out[name] = cycles
    return out


def test_spin_projections_match_djdze(C):
    n_checked = 0
    for name, cycles in R.ref(_telser_djdze_spin_projections).items():
        cv = t2t.djdze2sys(_input("djdze", name), constants=C)
        for clusters, c in zip(cycles, cv.extras["cycles"]):
            if clusters is None or len(cv.sys.S) < 2:
                continue
            mine = t2t.spin_projections(cv.sys, c["field_mT"], c["direction"]).numpy()
            for cl in clusters:
                np.testing.assert_allclose(mine[np.array(cl["levels"]) - 1].sum(axis=0), cl["ref"], atol=1e-9)
                n_checked += 1
    assert n_checked > 10


def test_w_and_u_powder_through_pepper_match_the_exact_field_point_model():
    """pepper with the converted w (Sys.Ham lB = 2, Options.QuadraticZeeman) and U (Sys.H0Extra) vs lft.ddpowh
    (Telser's exact field-point model, = telser-tools to 1e-10), frequency factor off.

    * S = 1, D = 0.08 cm^-1, w = 1 /cm^-1, W = 150 MHz: mapped 0.4 % of the peak, w dropped 57 %.
    * S = 3/2 + 1H, D = 5 cm^-1, U = 60 MHz, W = 60 MHz: mapped 2.5 % (pepper vs the field-point model without U:
      1.8 %, the grid/linearization baseline), U dropped 18 %.
    """
    from torchspin import Experiment
    from torchspin.lft import ddpowh

    cases = [(dict(s=1.0, g=(2.0, 2.0, 2.0), D=0.08, E=0.01), dict(w=1.0), np.linspace(150.0, 550.0, 401), 150.0,
              1e-2),
             (dict(s=1.5, g=(2.0, 2.0, 2.0), D=5.0, I=0.5, g_n=5.58569, A=(40.0, 40.0, 40.0)), dict(U=60.0),
              np.linspace(150.0, 400.0, 501), 60.0, 4e-2)]
    for base, term, f, W, tol in cases:
        s_l = t2t.spin2sys(**base)
        extra = (dict(zeeman2=t2t.second_order_zeeman_operator(s_l, term["w"])) if "w" in term
                 else dict(h0_extra=t2t.u_term_operator(s_l, term["U"])))
        ref = ddpowh.epr(s_l, f, 9.5, width=W, igloo=60, extent="octant", frequency_factor=False, **extra).spectrum
        ex = Experiment(mwFreq=9.5, Range=[f[0], f[-1]], nPoints=len(f), Harmonic=0)
        sm = t2t.spin2sys(**base, **term, HStrain=[2 * W] * 3)
        y = pepper(sm, ex, Options(GridSize=91, Threshold=0.0, **t2t.quadratic_zeeman_options(sm)))[1]
        y0 = pepper(t2t.spin2sys(**base, HStrain=[2 * W] * 3), ex, Options(GridSize=91, Threshold=0.0))[1]
        dev, corr = _norm_dev(ref.numpy(), y.numpy())
        dev0, _ = _norm_dev(ref.numpy(), y0.numpy())
        assert dev < tol and corr > 0.998, (term, dev, corr)
        assert dev0 > 4 * dev, (term, dev0, dev)
