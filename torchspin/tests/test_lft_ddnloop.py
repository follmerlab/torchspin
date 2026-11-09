"""torchspin.lft.ddnloop: d^n levels and EPR transitions over field, angle and parameter scans.

Self-contained checks (closed forms, the effective-spin g limit, units, gauge-invariant level
sums), gradients (forward-mode AD against central differences, best of several steps), and
forward parity with telser-tools' NumPy DDNLOOP (exact mode) on all 21 DDNLOOP reference
inputs when telser-tools is installed.
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
from torchspin.lft import ddn, ddnloop
from torchspin.lft.units import energy_factor

F64 = torch.float64
CM_TO_MHZ = energy_factor("cm^-1", "MHz")
INPUTS = json.loads((Path(__file__).parent / "data" / "telser" / "ddnloop_inputs.json").read_text())["cases"]


def _cu_V(tilt=0.0):
    eq = [lft.Ligand.isotropic(math.pi / 2 + (-1) ** i * tilt, i * math.pi / 2, 6000.0, 1000.0) for i in range(4)]
    ax = [lft.Ligand.isotropic(0.0, 0.0, 4800.0, 0.0), lft.Ligand.isotropic(math.pi, 0.0, 4800.0, 0.0)]
    return lft.aom(eq + ax)


# ------------------------------------------------------------------------------ self-contained
def test_d1_zero_spin_orbit_zeeman_is_free_electron():
    """d^1, zeta = 0, k = 0: the ground orbital is a pure spin doublet split by g_e mu_B B."""
    B = torch.linspace(0.0, 1000.0, 11, dtype=F64)
    r = ddnloop.field_scan(1, B, theta=0.4, phi=1.1, V=lft.crystal_field(1500.0, 300.0, 50.0, 100.0), k=(0.0, 0.0, 0.0),
                           units="cm^-1", pairs=[(0, 1)])
    c = ddn.TORCHSPIN_CONSTANTS
    # eigenvalue round-off: ~N eps max|E| = 10 * 2.2e-16 * 1e4 cm^-1 ~ 2e-11 cm^-1
    torch.testing.assert_close(r.transition_energies[:, 0], c.g_e * c.mu_b * B, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(r.g_obs[1:, 0].numpy(), c.g_e, rtol=1e-8)
    assert torch.isnan(r.g_obs[0, 0])
    # perpendicular mode, pure spin 1/2: |M|^2 = g_e^2/2, n.M = 0, so P = (|M|^2 - |n.M|^2)/2 = g_e^2/4
    np.testing.assert_allclose(r.probabilities[1:, 0].numpy(), c.g_e ** 2 / 4, rtol=1e-10)


def test_low_field_g_obs_matches_doublet_g_tensor():
    """Kramers doublet of Cu(II): g_obs(n) -> sqrt(n^T g g^T n) as B -> 0 (corrections ~ mu_B B / gap)."""
    kw = dict(V=_cu_V(0.2), zeta=830.0, k=(0.8, 0.8, 0.8), units="cm^-1")
    gt = ddn.g_tensor(9, **kw)
    th = torch.tensor([0.0, 0.5, 1.2, math.pi / 2], dtype=F64)
    ph = torch.tensor([0.0, 0.7, 2.0], dtype=F64)
    r = ddnloop.angle_scan(9, 1.0, th, ph, pairs=[(0, 1)], **kw)
    n = r.directions
    want = torch.sqrt(torch.einsum("...i,ij,...j->...", n, gt.ggT, n))
    torch.testing.assert_close(r.g_obs[..., 0], want, rtol=1e-6, atol=0.0)
    assert r.energies.shape == (4, 3, 10)


def test_units_mhz_vs_cm1():
    V = lft.crystal_field(Dq=1200.0, Ds=300.0, Dt=50.0)
    kw = dict(B=800.0, C=3200.0, zeta=200.0)
    common = dict(fields=torch.tensor([0.0, 300.0, 1000.0], dtype=F64), theta=0.3, phi=0.2, pairs=[(0, 1), (0, 3)],
                  temperature=4.2, mw_freq=9.5)
    a = ddnloop.scan(3, V=V, units="cm^-1", **kw, **common)
    b = ddnloop.scan(3, V=V * CM_TO_MHZ, units="MHz", **{k: v * CM_TO_MHZ for k, v in kw.items()}, **common)
    assert b.units == "MHz" and a.units == "cm^-1"
    torch.testing.assert_close(b.energies, a.energies * CM_TO_MHZ, rtol=1e-12, atol=1e-4)
    torch.testing.assert_close(b.transition_energies, a.transition_energies * CM_TO_MHZ, rtol=1e-8, atol=1e-6)
    for f in ("populations", "level_probabilities", "population_factors", "g_obs"):
        torch.testing.assert_close(getattr(b, f), getattr(a, f), rtol=1e-7, atol=1e-10, equal_nan=True)
    # per-pair quantities only off the B = 0 Kramers degeneracies (basis dependent there)
    torch.testing.assert_close(b.mw_weights[1:], a.mw_weights[1:], rtol=1e-7, atol=1e-12)


def test_populations_and_mesh_order():
    r = ddnloop.scan(2, B=900.0, C=3600.0, V=lft.crystal_field(Dq=1500.0, Ds=200.0), zeta=180.0, units="cm^-1",
                     fields=torch.tensor([0.0, 500.0], dtype=F64), theta=torch.tensor([0.0, 30.0, 60.0], dtype=F64),
                     phi=torch.tensor([0.0, 45.0], dtype=F64), mesh=True, degrees=True, temperature=10.0)
    assert r.energies.shape == (3, 2, 2, 45)
    torch.testing.assert_close(r.populations.sum(-1), torch.ones(3, 2, 2, dtype=F64))
    assert float(r.theta[1, 0, 0]) == pytest.approx(math.radians(30.0))
    assert float(r.fields[0, 0, 1]) == 500.0
    raw = ddnloop.scan(2, B=900.0, C=3600.0, V=lft.crystal_field(Dq=1500.0, Ds=200.0), zeta=180.0, units="cm^-1",
                       fields=500.0, temperature=10.0, normalized_populations=False).populations
    assert float(raw[0]) == 1.0


def test_level_probabilities_equal_pairs_off_degeneracy_and_are_basis_free_at_zero_field():
    kw = dict(V=lft.crystal_field(Dq=1500.0, Ds=300.0, Dt=40.0), zeta=150.0, k=(0.9, 0.9, 0.9), units="cm^-1",
              pairs="all")
    r = ddnloop.field_scan(1, torch.tensor([400.0], dtype=F64), theta=0.6, phi=0.3, **kw)
    torch.testing.assert_close(r.level_probabilities, r.probabilities, rtol=1e-10, atol=1e-12)
    z = ddnloop.field_scan(1, torch.tensor([0.0], dtype=F64), theta=0.6, phi=0.3, **kw)
    # rotate the eigenvectors inside every Kramers doublet: the level sums do not move
    U = torch.linalg.qr(torch.randn(2, 2, dtype=torch.complex128, generator=torch.Generator().manual_seed(1)))[0]
    vec = z.vectors[0].clone()
    for a in range(0, 10, 2):
        vec[:, a:a + 2] = vec[:, a:a + 2] @ U
    mu = ddn.zeeman_operators(1, (0.9, 0.9, 0.9))
    pm = ddnloop.pair_probability_matrix(vec, mu, z.directions[0])
    pi, pj = zip(*z.pairs)
    torch.testing.assert_close(ddnloop.level_probabilities(z.energies[0], pm, pi, pj), z.level_probabilities[0],
                               rtol=1e-10, atol=1e-12)


def test_parameter_scan_matches_levels():
    B = torch.tensor([600.0, 800.0, 1000.0], dtype=F64)
    kw = dict(C=3600.0, V=lft.crystal_field(Dq=1500.0), zeta=200.0, units="cm^-1")
    field = torch.tensor([0.0, 0.0, 300.0], dtype=F64)
    E = ddnloop.parameter_scan(3, "B", B, relative=True, field=field, **kw)
    for b, e in zip(B, E):
        torch.testing.assert_close(e, ddn.levels(3, B=b, relative=True, field=field, **kw))
    Vs = torch.stack([lft.crystal_field(Dq=d) for d in (1000.0, 1500.0)])
    Ev = ddnloop.parameter_scan(1, "V", Vs, units="cm^-1")
    torch.testing.assert_close(Ev[:, -1] - Ev[:, 0], torch.tensor([10000.0, 15000.0], dtype=F64))
    with pytest.raises(ValueError):
        ddnloop.parameter_scan(1, "slater", Vs)
    with pytest.raises(ValueError):
        ddnloop.parameter_scan(1, "V", B)


def test_mirror_image_is_y_reflection():
    V = lft.aom([lft.Ligand.from_degrees(30, 20, 10, e_sigma=5000, e_pi_x=900, e_pi_y=700),
                 lft.Ligand.from_degrees(120, 250, 0, e_sigma=4000, e_pi_x=800, e_pi_y=800)])
    D = torch.diag(torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0], dtype=F64))
    torch.testing.assert_close(ddnloop.mirror_image(V), D @ V @ D, rtol=0, atol=1e-11)
    # the mirror-image ligand set (phi -> -phi, psi -> -psi) gives the same field
    W = lft.aom([lft.Ligand.from_degrees(30, -20, -10, e_sigma=5000, e_pi_x=900, e_pi_y=700),
                 lft.Ligand.from_degrees(120, -250, 0, e_sigma=4000, e_pi_x=800, e_pi_y=800)])
    torch.testing.assert_close(ddnloop.mirror_image(V), W, rtol=0, atol=1e-9)


def test_bad_arguments():
    with pytest.raises(ValueError):
        ddnloop.scan(1, fields=1.0, pairs=[(1, 1)])
    with pytest.raises(TypeError):
        ddnloop.scan(1, fields=1.0, Dq=100.0)
    with pytest.raises(ValueError):
        ddnloop.scan(1, fields=torch.ones(2, 2), mesh=True)


# ------------------------------------------------------------------------------ gradients
def _fd_best(fun, x0, J, rels=(1e-3, 1e-4, 1e-5)):
    """Per-column error of J against central differences, best of several steps."""
    errs = []
    for r in rels:
        cols = []
        for j in range(len(x0)):
            h = r * max(abs(float(x0[j])), 1.0)
            e = torch.zeros_like(x0)
            e[j] = h
            cols.append((fun(x0 + e) - fun(x0 - e)) / (2 * h))
        errs.append((J - torch.stack(cols, -1)).abs().amax(0))
    return torch.stack(errs).amin(0)


def _assert_grad(fun, x0, tol=1e-6):
    J = torch.func.jacfwd(fun)(x0)
    assert torch.isfinite(J).all()
    err = _fd_best(fun, x0, J)
    scale = J.abs().max()
    assert scale > 0
    assert bool((err <= tol * scale + 1e-12).all()), (err, scale)


def _xf(n, x):
    kw = dict(V=lft.crystal_field(x[0], x[1], x[2], x[3]), zeta=x[4], k=x[5:8], units="cm^-1")
    if n > 1:
        kw.update(B=x[8], C=x[9])
    return kw


@pytest.mark.parametrize("n,loop", [(1, "field"), (3, "field"), (8, "field"), (9, "angle"), (1, "angle")])
def test_scan_gradients(n, loop):
    """d(energies, transition energies, probabilities, level sums, population factors, mw weights, g_obs)
    / d(Dq, Ds, Dt, Dr, zeta, k[, B, C]); field scans start at B = 0 (Kramers degeneracies)."""
    x0 = torch.tensor([1500.0, 313.0, 57.0, 25.0, 211.0, 0.97, 0.88, 0.79] + ([805.0, 3220.0] if n > 1 else []),
                      dtype=F64)
    if loop == "field":
        grid = dict(fields=torch.tensor([0.0, 150.0, 400.0], dtype=F64), theta=0.7, phi=0.4)
    else:
        grid = dict(fields=350.0, theta=torch.tensor([0.1, 0.8, 1.4], dtype=F64)[:, None],
                    phi=torch.tensor([0.2, 1.0], dtype=F64))
    pairs = [(0, 1), (0, 2), (1, 3)]
    ref = ddnloop.scan(n, pairs=pairs, temperature=6.0, mw_freq=9.5, h1mode=0.3, **grid, **_xf(n, x0))
    on = (ref.fields > 0).reshape(-1)

    def fun(x):
        r = ddnloop.scan(n, pairs=pairs, temperature=6.0, mw_freq=9.5, h1mode=0.3, **grid, **_xf(n, x))
        flat = lambda t: t.reshape(on.numel(), -1)  # noqa: E731
        return torch.cat([flat(r.energies).reshape(-1), flat(r.transition_energies).reshape(-1),
                          flat(r.level_probabilities).reshape(-1), flat(r.population_factors).reshape(-1),
                          flat(r.probabilities)[on].reshape(-1), flat(r.mw_weights)[on].reshape(-1),
                          flat(r.g_obs)[on].reshape(-1)])

    _assert_grad(fun, x0)


def test_temperature_and_angle_gradients():
    V = _cu_V(0.15)

    def fun(x):
        r = ddnloop.scan(9, V=V, zeta=830.0, k=(0.8, 0.8, 0.8), units="cm^-1", fields=330.0, theta=x[0], phi=x[1],
                         pairs="all", temperature=x[2], mw_freq=9.5)
        return torch.cat([r.energies, r.level_probabilities, r.populations, r.population_factors, r.g_obs])

    _assert_grad(fun, torch.tensor([0.6, 0.35, 3.0], dtype=F64))


def test_aom_bond_vector_gradients():
    """Gradients w.r.t. the AOM bond directions (d^1, 300 mT, two orientations)."""
    n0 = torch.tensor([[0.1, 0.05, 1.0], [-0.05, 0.1, -1.0], [1.0, 0.1, 0.0], [-1.0, 0.0, 0.1], [0.0, 1.0, 0.05],
                       [0.1, -1.0, 0.0]], dtype=F64)

    def fun(x):
        V = lft.aom_isotropic(x.reshape(6, 3), 6000.0, 1200.0)
        r = ddnloop.scan(1, V=V, zeta=600.0, k=(0.9, 0.9, 0.9), units="cm^-1", fields=300.0,
                         theta=torch.tensor([0.4, 1.2], dtype=F64), phi=torch.tensor([0.3, 0.7], dtype=F64),
                         pairs="all")
        return torch.cat([r.energies.reshape(-1), r.level_probabilities.reshape(-1)])

    _assert_grad(fun, n0.reshape(-1))


def test_zero_field_level_probability_gradient():
    """At B = 0 every level of d^1 is a Kramers doublet (or quartet): the level sum is differentiable."""
    def fun(x):
        r = ddnloop.scan(1, fields=0.0, theta=0.5, phi=0.2, pairs=[(0, 2), (0, 4), (2, 5)], **_xf(1, x))
        return r.level_probabilities

    _assert_grad(fun, torch.tensor([1500.0, 313.0, 57.0, 25.0, 211.0, 0.97, 0.88, 0.79], dtype=F64))


# ------------------------------------------------------------------------------ parity with telser-tools
@lru_cache(maxsize=None)
def _telser():
    dl = pytest.importorskip("telser_tools.ddnloop")
    from telser_tools.ddn.hamiltonian import CODATA2018
    from telser_tools.ddnloop.model import exact_probabilities, pair_elements
    from telser_tools.ddnloop.program import _points, model_for
    return dl, CODATA2018, exact_probabilities, pair_elements, _points, model_for


def _isolated(e, rtol=1e-6):
    """States whose gap to every other exceeds rtol * max|E| (eigenvectors conditioned to ~eps/rtol)."""
    d = np.abs(e[..., :, None] - e[..., None, :]) + np.eye(e.shape[-1]) * 1e300
    return d.min(-1) > np.maximum(rtol * np.abs(e).max(-1, keepdims=True), 1e-3)


def _ours(inp, C18):
    p = inp.parameters()
    kw = dict(zeta=p.zeta, V=None if p.ligand_field is None else torch.tensor(p.ligand_field), k=p.k, alpha=p.alpha)
    if p.slater is not None:
        kw["slater"] = tuple(p.slater)
    else:
        kw.update(B=p.B, C=p.C, A=p.A)
    pairs = None
    if inp.epr:
        pairs = [(i - 1, j - 1) for i in range(inp.i_min, inp.i_max + 1) for j in range(inp.j_min, inp.j_max + 1)
                 if j > i]
    consts = ddn.Constants(mu_b=C18.mu_b * 10.0, g_e=C18.g_e, k_b=C18.k_b)       # cm^-1/G -> cm^-1/mT
    return ddnloop.scan(inp.n, fields=inp.fields() / 10.0, theta=inp.thetas(), phi=inp.phis(), mesh=True,
                        degrees=True, pairs=pairs, temperature=inp.temperature, h1mode=inp.h1mode,
                        mw_freq=inp.mw_ghz, units="cm^-1", constants=consts, **kw)


@pytest.mark.parametrize("name", sorted(INPUTS))
def test_parity_with_telser_tools(name):
    """Energies, populations, every listed row (energy, probability where both states are isolated,
    population factor, mw weight, g_obs) and the level-summed probabilities everywhere."""
    dl, C18, np_exact, np_pairs, points, model_for = _telser()
    inp = dl.parse_input(INPUTS[name])
    a = dl.run(inp, "exact")
    b = _ours(inp, C18)
    npts = len(a.fields)
    E = b.energies.reshape(npts, -1).numpy()
    np.testing.assert_allclose(E, a.energies, rtol=1e-12, atol=1e-9)
    th, ph, hh = points(inp)
    np.testing.assert_allclose(np.degrees(b.theta.reshape(-1).numpy()), th, atol=1e-12)
    np.testing.assert_allclose(b.fields.reshape(-1).numpy() * 10.0, hh, atol=1e-12)
    iso = _isolated(a.energies)
    flat = lambda t: None if t is None else t.reshape(npts, -1).numpy()  # noqa: E731
    pops = flat(b.populations)
    if not inp.epr:
        for r in a.rows:
            k = int(np.flatnonzero((th == r.theta) & (ph == r.phi) & (hh == r.h0))[0])
            assert E[k, r.i - 1] == pytest.approx(r.energy, rel=1e-12, abs=1e-9)
            if r.population is not None:
                assert pops[k, r.i - 1] == pytest.approx(r.population, rel=1e-9, abs=1e-12)
        return
    if not b.pairs:
        return
    q_of = {pr: q for q, pr in enumerate(b.pairs)}
    tr, P, pf, mw, g = (flat(x) for x in (b.transition_energies, b.probabilities, b.population_factors,
                                          b.mw_weights, b.g_obs))
    pmax = max([1.0] + [r.probability for r in a.rows])
    for r in a.rows:
        k = int(np.flatnonzero((th == r.theta) & (ph == r.phi) & (hh == r.h0))[0])
        q = q_of[(r.i - 1, r.j - 1)]
        assert tr[k, q] == pytest.approx(r.energy, rel=1e-12, abs=1e-9)
        if r.g_obs is not None:
            assert g[k, q] == pytest.approx(r.g_obs, rel=1e-9)
        if r.population is not None:
            assert pf[k, q] == pytest.approx(r.population, rel=1e-9, abs=1e-9 * pmax)
        if iso[k, [r.i - 1, r.j - 1]].all():
            assert P[k, q] == pytest.approx(r.probability, rel=1e-9, abs=1e-9 * pmax)
            if r.mw_weight is not None:
                # E_j - E_i agrees to ~1e-11 cm^-1 (eigenvalue round-off), which moves a weight on the
                # flank of the 2 % line by ~1e-9 relative
                assert mw[k, q] == pytest.approx(r.mw_weight, rel=1e-7, abs=1e-9 * pmax)
    # level sums at every point (including B = 0) vs the NumPy per-pair values summed over levels
    pi, pj = [i for i, _ in b.pairs], [j for _, j in b.pairs]
    model = model_for(inp, a.options)
    t, p = np.radians(th), np.radians(ph)
    dirs = np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], -1)
    w, v = model.solve(hh[:, None] * dirs)
    N = w.shape[-1]
    m = np.swapaxes(v.conj(), -1, -2)[:, None] @ model.mu[None] @ v[:, None]
    Pall = np_exact(m.reshape(npts, 3, N * N), dirs, inp.h1mode).reshape(npts, N, N)
    D = (np.abs(w[:, :, None] - w[:, None, :]) <= 1e-11 * np.abs(w).max(-1)[:, None, None] + 1e-9).astype(float)
    want = (D @ Pall @ D)[:, pi, pj]
    atol = 1e-9 * max(1.0, np.abs(want).max())
    np.testing.assert_allclose(flat(b.level_probabilities), want, rtol=1e-9, atol=atol)
    ok = iso[:, pi] & iso[:, pj]
    want_pp = np_exact(np_pairs(v, model.mu, np.array(pi), np.array(pj)), dirs, inp.h1mode)
    np.testing.assert_allclose(P[ok], want_pp[ok], rtol=1e-9, atol=atol)


def test_mirror_image_parity_with_telser_tools():
    pytest.importorskip("telser_tools")
    from telser_tools.ddn import ligand_field as tlf
    v = tlf.aom([tlf.Ligand(30, 20, 10, 5000, 900, 700, 0, 0), tlf.Ligand(120, 250, 0, 4000, 800, 800, 0, 0)])
    want = tlf.from_m_basis(np.conj(tlf.to_m_basis(v)))
    np.testing.assert_allclose(ddnloop.mirror_image(torch.tensor(v)).numpy(), want, atol=1e-11)
