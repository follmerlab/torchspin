"""torchspin.lft.dsoxf (milestone M4): one free-ion term in a crystal field (Telser's DSOXF).

Self-contained checks (one-electron fields against lft.ddn, Lande intervals, Zeeman in mT,
unit consistency), gradients against central differences (degenerate levels as level means,
Kramers-doublet g, level-summed observables, gradcheck with forward AD), and parity with
telser-tools when it is installed: the copied NumPy engine is bit-identical to the original,
and all 56 DSOXF reference cases (both conventions, zero field and along x, y, z) match.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import torch

from torchspin import lft
from torchspin.lft import ddn, dsoxf
from torchspin.lft._engines import dsoxf as eng
from torchspin.lft.units import energy_factor

F64 = torch.float64
CM_TO_MHZ = energy_factor("cm^-1", "MHz")
DATA = Path(__file__).parent / "data" / "telser"


def _lm(e, tol=1e-6):
    """0/1 level matrix of sorted energies (chained criterion)."""
    lab = np.cumsum(np.r_[0, np.diff(e) > tol])
    return (lab[:, None] == lab[None, :]).astype(float)


def _close(a, b, rtol=1e-10, atol=1e-10):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol)


# ------------------------------------------------------------------------------ closed forms
@pytest.mark.parametrize("symmetry,axis", [("cubic", "tetragonal"), ("axial", "tetragonal"),
                                           ("rhombic", "tetragonal"), ("axial", "trigonal")])
def test_d1_is_ballhausen_one_electron_field(symmetry, axis):
    """d1 DSOXF (lambda = zeta, Delta4 = -Dt, V2 = Dr) equals the d^n determinant model of lft.ddn, field included."""
    Dq, Ds, Dt, Dr, zeta = 1500.0, 300.0, 80.0, 40.0, 150.0
    use = dict(cubic=(0, 0, 0), axial=(Ds, Dt, 0), rhombic=(Ds, Dt, Dr))[symmetry]
    V = lft.crystal_field(Dq, *use, axis=axis)
    field = torch.tensor([30.0, -50.0, 400.0], dtype=F64)
    ops = dsoxf.dsoxf_operators("d", 1, symmetry=symmetry, axis=axis)
    for B in (None, field):
        E = dsoxf.levels(ops, lam=zeta, dq=Dq, ds=Ds, dt4=-Dt, v2=Dr, k=(0.9, 0.9, 0.8), field=B, units="cm^-1")
        ref = ddn.levels(1, zeta=zeta, V=V, k=(0.9, 0.9, 0.8), field=B, units="cm^-1")
        _close(E.numpy(), ref.numpy(), atol=1e-9)


def test_d9_is_the_one_hole_problem():
    """d9 (2D, SIGNXF = SIGNLAM = -1) equals lft.ddn d9 with the same field and zeta."""
    Dq, Ds, Dt, zeta = 1200.0, 2000.0, 500.0, 830.0
    E = dsoxf.levels(dsoxf.dsoxf_operators("d", 9, symmetry="axial"), lam=zeta, dq=Dq, ds=Ds, dt4=-Dt,
                     units="cm^-1", relative=True)
    ref = ddn.levels(9, zeta=zeta, V=lft.crystal_field(Dq, Ds, Dt), units="cm^-1", relative=True)
    _close(E.numpy(), ref.numpy(), atol=1e-9)


@pytest.mark.parametrize("shell,code,sign", [("d", 2, 1.0), ("d", 8, -1.0), ("f", 1, 1.0), ("f", 13, -1.0)])
def test_free_term_lande_intervals(shell, code, sign):
    """No crystal field: E(J) - E(J-1) = lambda J, lambda = +|lambda| below half filling, - above."""
    lam = 200.0
    ops = dsoxf.dsoxf_operators(shell, code)
    E = dsoxf.levels(ops, lam=lam, units="cm^-1").numpy()
    L, S = ops.L, ops.S
    Js = np.arange(abs(L - S), L + S + 1)
    want = np.sort(np.concatenate([[sign * lam / 2 * (J * (J + 1) - L * (L + 1) - S * (S + 1))] * int(2 * J + 1)
                                   for J in Js]))
    _close(E, want, atol=1e-9)


def test_zeeman_field_in_mT():
    """2D, no field terms: m_L mu_B B + g_e m_S mu_B B."""
    E = dsoxf.levels(dsoxf.dsoxf_operators("d", 1), field=(0.0, 0.0, 1000.0), units="cm^-1")
    c = ddn.TORCHSPIN_CONSTANTS
    want = sorted(c.mu_b * 1000.0 * (ml + c.g_e * ms) for ml in range(-2, 3) for ms in (-0.5, 0.5))
    _close(E.numpy(), want, rtol=1e-12, atol=1e-14)


def test_orbital_occupancies_and_spin():
    """Elongated d9: the ground doublet is d(x2-y2) with <S_z> = +-1/2 in a z field; rows sum to 1."""
    s = dsoxf.DsoxfSystem("d", 9, symmetry="axial", units="cm^-1", dq=1200.0, ds=2000.0, dt4=-500.0)
    sol = s.solve(field=(0.0, 0.0, 300.0))
    occ = sol.orbital_occupancies()
    _close(occ.sum(-1).numpy(), np.ones(10), atol=1e-12)
    assert eng.D_ORBITALS[int(occ[0].argmax())] == "x2-y2"
    _close(sorted(sol.spin_along_field()[:2].tolist()), [-0.5, 0.5], atol=1e-12)
    _close(sol.ml_occupancies().sum(-1).numpy(), np.ones(10), atol=1e-12)


def test_units_mhz_vs_cm1_consistent():
    p = dict(lam=180.0, dq=1300.0, ds=150.0, dt4=-60.0, rho=0.4)
    a = dsoxf.DsoxfSystem("d", 4, symmetry="axial", units="cm^-1", **p)
    b = dsoxf.DsoxfSystem("d", 4, symmetry="axial", units="MHz", **{k: v * CM_TO_MHZ for k, v in p.items()})
    field = (100.0, 0.0, 250.0)
    sa, sb = a.solve(field=field), b.solve(field=field)
    torch.testing.assert_close(sb.energies, sa.energies * CM_TO_MHZ, rtol=1e-12, atol=1e-4)
    torch.testing.assert_close(sb.populations(20.0), sa.populations(20.0), rtol=1e-10, atol=1e-14)
    torch.testing.assert_close(sb.field_g(0, 1), sa.field_g(0, 1), rtol=1e-10, atol=0.0)
    ka, kb = a.solve().kramers_g(), b.solve().kramers_g()
    assert len(ka) == len(kb)
    K = energy_factor("cm^-1", "K")
    sK = dsoxf.DsoxfSystem("d", 4, symmetry="axial", units="K", **{k: v * K for k, v in p.items()})
    torch.testing.assert_close(sK.levels(), a.levels() * K, rtol=1e-12, atol=1e-8)
    with pytest.raises(TypeError):
        dsoxf.DsoxfSystem("d", 1, dz=1.0)
    with pytest.raises(ValueError):
        dsoxf.dsoxf_operators("d", 10)


def test_kramers_g_of_cubic_d1():
    """Cubic d1 with SO: one Kramers doublet (Gamma7; Gamma8 is a quartet), with isotropic g."""
    sol = dsoxf.DsoxfSystem("d", 1, units="cm^-1", lam=150.0, dq=1800.0).solve()
    kg = sol.kramers_g()
    assert len(kg) == 1                                     # the Gamma7 doublet (Gamma8 is a quartet)
    p = kg[0].principal
    _close(p.numpy(), [float(p[0])] * 3, atol=1e-10)


# ------------------------------------------------------------------------------ gradients
def _fd(fun, x, rels=(1e-3, 1e-4, 1e-5)):
    out = []
    for r in rels:
        h = r * np.maximum(np.abs(x), 1.0)
        cols = []
        for i in range(len(x)):
            e = np.zeros_like(x)
            e[i] = h[i]
            cols.append((fun(x + e) - fun(x - e)) / (2 * h[i]))
        out.append(np.stack(cols, -1))
    return out


def _assert_grad(J, fds, tol=1e-6, floor=1e-11):
    err = np.min([np.abs(J - f).max(axis=0) for f in fds], axis=0)
    scale = np.abs(J).max()
    assert np.all(err <= tol * scale + floor), (err, scale)


def _level_means(e, tol=1e-6):
    D = _lm(e, tol)
    rows = D[np.sort(np.unique(D, axis=0, return_index=True)[1])]
    return rows / rows.sum(1, keepdims=True)


CASES = [
    # (system kwargs, parameter names, field in mT); energies in cm^-1
    (dict(shell="d", code=9, lam=830.0, dq=1200.0, ds=2000.0, dt4=-500.0, symmetry="axial", k=(0.8, 0.8, 0.7)),
     ["lam", "dq", "ds", "dt4"], (50.0, 150.0, 300.0)),
    (dict(shell="d", code=1, lam=150.0, dq=1800.0), ["lam", "dq"], None),                         # cubic, Kramers
    (dict(shell="d", code=6, lam=100.0, dq=1000.0, rho=0.5, geometry="Td"), ["lam", "dq", "rho"], None),
    (dict(shell="d", code=2, lam=200.0, dq=1500.0, ds=300.0, dt4=50.0, v2=40.0, v4=20.0, symmetry="rhombic",
          axis="trigonal", rho=0.3), ["lam", "dq", "ds", "dt4", "v2", "v4", "rho"], (0.0, 0.0, 800.0)),
    (dict(shell="d", code=-5, lam=400.0, ds=-800.0, v2=200.0, symmetry="rhombic"), ["lam", "ds", "v2"], None),
    (dict(shell="f", code=1, lam=600.0, b4=2000.0, b6=800.0, b20=300.0, b43=100.0, b63=50.0, b66=40.0,
          symmetry="axial", axis="trigonal"), ["lam", "b4", "b6", "b20", "b43", "b63", "b66"], (100.0, 0.0, 200.0)),
    (dict(shell="f", code=13, lam=2500.0, b4=1500.0, b6=500.0, b22=150.0, b42=80.0, b62=60.0, b44=30.0,
          symmetry="rhombic"), ["lam", "b4", "b6", "b22", "b42", "b62", "b44"], None),
]
IDS = [f"{c[0]['shell']}{c[0]['code']}" for c in CASES]
_DISC = ("shell", "code", "geometry", "symmetry", "axis")


def _system(kw):
    disc = {k: kw[k] for k in _DISC if k in kw}
    return dsoxf.DsoxfSystem(units="cm^-1", **disc, **{k: v for k, v in kw.items() if k not in disc})


def _solution(kw, names, field, x):
    return _system(kw).solve(field=field, **{n: x[i] for i, n in enumerate(names)})


def _np(fun):
    return lambda x: fun(torch.tensor(x)).detach().numpy()


@pytest.mark.parametrize("kw,names,field", CASES, ids=IDS)
def test_energy_gradients(kw, names, field):
    x0 = np.array([kw[n] for n in names], float)
    E = lambda x: _solution(kw, names, field, x).energies  # noqa: E731
    e0 = _np(E)(x0)
    M = _level_means(e0)
    J = torch.func.jacfwd(E)(torch.tensor(x0)).numpy()
    assert np.isfinite(J).all()
    if field is None:
        assert M.shape[0] < len(e0)                       # degenerate levels present
    _assert_grad(M @ J, _fd(lambda x: M @ _np(E)(x), x0))
    # reverse mode agrees with forward mode
    x = torch.tensor(x0, requires_grad=True)
    (g,) = torch.autograd.grad((torch.as_tensor(M[0]) @ E(x)), x)
    _close(g.numpy(), (M @ J)[0], rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("kw,names,field", [c for c in CASES if c[2] is None],
                         ids=[i for i, c in zip(IDS, CASES) if c[2] is None])
def test_kramers_g_gradients(kw, names, field):
    """g g^T of every zero-field Kramers doublet (exactly degenerate) vs central differences."""
    x0 = np.array([kw[n] for n in names], float)
    doublets = [idx for _, idx in _solution(kw, names, None, torch.tensor(x0)).levels(1e-6) if len(idx) == 2]
    if not doublets:
        pytest.skip("no Kramers doublet")

    def gg(x):
        s = _solution(kw, names, None, x)
        return torch.cat([s.g_tensor(idx).ggT.reshape(-1) for idx in doublets])

    J = torch.func.jacfwd(gg)(torch.tensor(x0)).numpy()
    _assert_grad(J, _fd(_np(gg), x0))


def test_level_summed_observables_gradients():
    """m_L occupancies and transition moments summed over levels: differentiable through the cubic degeneracy."""
    kw = dict(shell="d", code=4, lam=180.0, dq=1300.0, ds=150.0, dt4=-60.0, symmetry="axial", rho=0.4)
    names = ["lam", "dq", "ds", "dt4", "rho"]
    x0 = np.array([kw[n] for n in names], float)
    b1 = torch.tensor([1.0, 0.0, 0.0], dtype=F64)
    D = torch.as_tensor(_lm(_solution(kw, names, None, torch.tensor(x0)).energies.numpy()))

    def obs(x):
        s = _solution(kw, names, None, x)
        return torch.cat([D @ s.ml_occupancies()[:, 0], s.level_transition_moments(b1)[0]])

    J = torch.func.jacfwd(obs)(torch.tensor(x0)).numpy()
    _assert_grad(J, _fd(_np(obs), x0), tol=1e-5)


def test_gradcheck_forward_ad():
    ops = dsoxf.dsoxf_operators("d", 2, symmetry="rhombic")

    def f(lam, dq, ds, v2):
        return dsoxf.levels(ops, lam=lam, dq=dq, ds=ds, v2=v2, rho=0.3, field=(20.0, 10.0, 80.0), units="cm^-1")

    args = tuple(torch.tensor(v, dtype=F64, requires_grad=True) for v in (200.0, 1500.0, 300.0, 40.0))
    assert torch.autograd.gradcheck(f, args, eps=1e-4, atol=1e-6, rtol=1e-5, check_forward_ad=True)


def test_batched_parameters():
    ops = dsoxf.dsoxf_operators("d", 9, symmetry="axial")
    ds = torch.tensor([1000.0, 2000.0, 3000.0], dtype=F64)
    E = dsoxf.levels(ops, lam=830.0, dq=1200.0, ds=ds, units="cm^-1")
    assert E.shape == (3, 10)
    for i in range(3):
        _close(E[i].numpy(), dsoxf.levels(ops, lam=830.0, dq=1200.0, ds=float(ds[i]), units="cm^-1").numpy(), atol=1e-9)


# ------------------------------------------------------------------------------ parity with telser-tools
def _telser():
    return pytest.importorskip("telser_tools.dsoxf")


_SETTINGS = [dict(geometry=g, symmetry=s, axis=a, conventions=c) for g in ("Oh", "Td")
             for s in ("cubic", "axial", "rhombic") for a in ("tetragonal", "trigonal") for c in ("corrected", "dsoxf")]


@pytest.mark.parametrize("shell,code", [("d", c) for c in eng.D_CODES] + [("f", c) for c in eng.F_CODES])
def test_engine_copy_is_bit_identical(shell, code):
    td = _telser()
    from telser_tools.dsoxf import configurations as tconf
    from telser_tools.dsoxf.hamiltonian import orbital_vectors as t_orbital_vectors
    rng = np.random.default_rng(abs(code) + (100 if shell == "f" else 0))
    vals = dict(zip(dsoxf.CF_PARAMETERS, (rng.normal(size=len(dsoxf.CF_PARAMETERS)) * 300.0).tolist()))
    for st in _SETTINGS:
        a = eng.DsoxfModel(shell=shell, code=code, lam=-210.0, rho=0.7, **st, **vals)
        b = td.DsoxfModel(shell=shell, code=code, lam=-210.0, rho=0.7, **st, **vals)
        assert np.array_equal(a.zero_field, b.zero_field)
        assert np.array_equal(a.spin_spin, b.spin_spin)
        assert (a.spin_orbit, a.orbital_zeeman_sign) == (b.spin_orbit, b.orbital_zeeman_sign)
        if shell == "d":                     # d_signs is d-shell only (it raises KeyError for f terms with L = 3)
            assert a.d_signs == b.d_signs
    ca, cb = eng.configuration(shell, code), tconf.configuration(shell, code)
    assert (ca.L, ca.S, ca.sign_cf, ca.sign_lambda, ca.b2x, ca.b4x, ca.term) == \
        (cb.L, cb.S, cb.sign_cf, cb.sign_lambda, cb.b2x, cb.b4x, cb.term)
    ba, bb = a.basis, b.basis
    for x, y in zip((*ba.Lv, *ba.Sv, ba.LS, ba.ml, ba.ms), (*bb.Lv, *bb.Sv, bb.LS, bb.ml, bb.ms)):
        assert np.array_equal(x, y)
    if ca.L in (2, 3):
        for kind in ("real", "general", "cubic"):
            assert np.array_equal(eng.orbital_vectors(ca.L, kind)[1], t_orbital_vectors(ca.L, kind)[1])
    for n in range(1, 8):
        assert eng.stevens_ratios(3, n) == tconf.stevens_ratios(3, n)


def test_engine_stevens_operators_bit_identical():
    td = _telser()
    for j in (1, 2, 3, 5, 6, 2.5, 3.5):
        for k, qs in ((2, (0, 2)), (4, (0, 2, 3, 4)), (6, (0, 2, 3, 4, 6))):
            for q in qs:
                if 2 * j >= k:
                    assert np.array_equal(eng.stevens(j, k, q), td.stevens(j, k, q))


@lru_cache(maxsize=None)
def _ref_cases():
    return json.loads((DATA / "dsoxf_cases.json").read_text())["cases"]


@lru_cache(maxsize=None)
def _inp(name):
    from telser_tools.dsoxf.inputfile import parse_input
    return parse_input(_ref_cases()[name]["stdin"])


UNIT = {"x": (1.0, 0, 0), "y": (0, 1.0, 0), "z": (0, 0, 1.0)}


@pytest.mark.parametrize("conv", ["dsoxf", "corrected"])
@pytest.mark.parametrize("name", sorted(json.loads((DATA / "dsoxf_cases.json").read_text())["cases"]))
def test_reference_cases_match_telser_tools(name, conv):
    """Every DSOXF reference case: energies, <S>, occupancies, transition moments, populations, g(obs), Kramers g."""
    td = _telser()
    from telser_tools.dsoxf.hamiltonian import CODATA2018, DSOXF_CONSTANTS, solve
    d = _inp(name)
    model = d.model(conv)
    tc = DSOXF_CONSTANTS if conv == "dsoxf" else CODATA2018
    consts = ddn.Constants(mu_b=tc.mu_b * 10.0, g_e=tc.g_e, k_b=tc.k_b)        # cm^-1/G -> cm^-1/mT
    tsys = dsoxf.DsoxfSystem.from_model(model, units="cm^-1", constants=consts)
    assert isinstance(model, td.DsoxfModel)
    fields = [None] + ([d.H0 * np.array(UNIT[a]) for a in "xyz"] if d.H0 > 0 else [])
    for f in fields:
        a = solve(model, f, tc)
        b = tsys.solve(field=None if f is None else torch.tensor(f / 10.0))      # G -> mT
        e = a.energies
        _close(b.energies, e, rtol=1e-12, atol=1e-9)
        assert np.array_equal(_lm(e), dsoxf.level_matrix(b.energies, units="cm^-1").numpy())
        # per-state values where the state is isolated (gap > 0.1 cm^-1), cluster sums otherwise
        D = _lm(e, 1e-1)
        single = D.sum(1) == 1
        for got, want in ((b.spin_expectation(), a.spin_expectation()), (b.ml_occupancies(), a.ml_occupancies())):
            _close(D @ got.numpy(), D @ want, atol=1e-9)
            _close(got.numpy()[single], want[single], atol=1e-9)
        L = model.config.L
        for kind in (["real"] if L == 2 else ["general", "cubic"] if L == 3 else []):
            _close(D @ b.orbital_occupancies(kind).numpy(), D @ a.orbital_occupancies(kind), atol=1e-9)
        for b1 in ([None] if f is not None else []) + [np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])]:
            want = D @ a.transition_moments(b1) @ D
            got = b.level_transition_moments(None if b1 is None else torch.tensor(b1), tol=1e-1).numpy()
            _close(got, want, rtol=1e-9, atol=1e-9 * max(1.0, np.abs(want).max()))
        if d.temperature:
            kT = tc.k_b * d.temperature
            _close(b.populations(d.temperature).numpy(), a.populations(d.temperature),
                   rtol=1e-10 / min(kT, 1.0) + 1e-12, atol=1e-300)
        if f is not None and len(e) > 1:
            _close(float(b.field_g(0, 1)), a.field_g(0, 1), rtol=1e-9, atol=1e-9)
        if f is None:
            ka, kb = a.kramers_g(), b.kramers_g()
            assert len(ka) == len(kb)
            for (ea, ga), gb in zip(ka, kb):
                _close(float(gb.energy), ea, atol=1e-9)
                # squares: sqrt amplifies the round-off of a vanishing g
                _close(gb.principal.numpy() ** 2, ga ** 2, rtol=1e-9, atol=1e-10 * max(1.0, (ga ** 2).max()))
