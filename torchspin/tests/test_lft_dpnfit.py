"""torchspin.lft.dpnfit: p^n band-energy fits with an exact (forward-AD) Jacobian.

Self-contained: one-band closed forms, noise-free recovery, units, the Jacobian against
central differences, LOWSPN, the hole route.  With telser-tools installed: on every DPNFIT
reference input the NumPy program accepts (both modes), the residuals equal NumPy's, the
Jacobian equals telser_tools_diff's and converged central differences of NumPy's residuals,
and fits reach the same objective, uncertainties and null directions.
"""

from __future__ import annotations

import json
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import torch

from torchspin.lft import dpn, dpnfit
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


def _cd(fun, x, rels=(1e-3, 1e-4, 1e-5)):
    out = []
    for r in rels:
        cols = []
        for j in range(len(x)):
            h = r * max(abs(x[j]), 1.0)
            e = np.zeros(len(x))
            e[j] = h
            cols.append((fun(x + e) - fun(x - e)) / (2 * h))
        out.append(np.array(cols).T)
    return out


def _assert_grad(J, fds, tol=1e-7):
    err = np.min([np.abs(J - f).max(axis=0) for f in fds], axis=0)
    scale = max(np.abs(J).max(), 1e-300)
    assert np.all(err <= tol * scale + 1e-6 * np.abs(J).max(axis=0)), (err, scale)


# ------------------------------------------------------------------------------ closed forms
def test_p2_one_band_fit_F2():
    """1D - 3P = 6 F2/25 (zeta = alpha = 0): states 1-9 are 3P, 10-14 1D."""
    m = dpnfit.DpnFitModel(2, units="cm^-1")
    prob = dpnfit.DpnFit(m, [10], [12000.0], x0=[0.0, 30000.0, 0.0, 0.0, 0.0],
                         free=[False, True, False, False, False])
    res = prob.fit()
    assert res.x[1] == pytest.approx(25 * 12000.0 / 6, rel=1e-10)
    assert res.objective < 1e-16


def test_p3_one_band_fit_F2():
    """2D - 4S = 9 F2/25: states 1-4 are 4S."""
    m = dpnfit.DpnFitModel(3, units="cm^-1")
    prob = dpnfit.DpnFit(m, [5], [15000.0], x0=[0.0, 30000.0, 0.0], free=[False, True, False])
    assert prob.fit().x[1] == pytest.approx(25 * 15000.0 / 9, rel=1e-10)


def test_p1_crystal_field_and_option_p():
    """p1: E(pz) = -2 Ds, E(px, py) = Ds, so the band is 3 Ds; option P (px = py at 0): E(pz) directly (states 5, 6)."""
    prob = dpnfit.DpnFit(dpnfit.DpnFitModel(1, units="cm^-1"), [3], [2400.0], x0=[0.0, 500.0, 0.0],
                         free=[False, True, False])
    assert prob.fit().x[1] == pytest.approx(800.0, rel=1e-10)
    mp = dpnfit.DpnFitModel(1, lf_type="P", orbital_isymm=[1, 1, 3], units="cm^-1")
    assert mp.parameter_names() == ["zeta", "E_px", "E_py", "E_pz"]
    prob = dpnfit.DpnFit(mp, [5], [1500.0], x0=[0.0, 0.0, 0.0, 400.0], free=[False, False, False, True])
    assert prob.fit().x[3] == pytest.approx(1500.0, rel=1e-10)


def test_noise_free_recovery_p2_and_units():
    truth = np.array([600.0, 38000.0, 0.0, 1200.0, 250.0])
    m = dpnfit.DpnFitModel(2, units="cm^-1")
    probe = dpnfit.DpnFit(m, [1], [0.0], x0=truth)
    E = probe.energies(truth).detach().numpy()
    states = np.arange(2, 16)
    prob = dpnfit.DpnFit(m, states, E[states - 1], x0=truth * [0.9, 1.03, 1, 0.8, 1.2],
                         free=[True, True, False, True, True], lower=[0, 0, -np.inf, -np.inf, 0])
    res = prob.fit()
    np.testing.assert_allclose(res.x, truth, rtol=1e-6)
    assert res.null == [] and res.success
    # the same problem in MHz: same solution in MHz, residuals scale with the unit
    mm = dpnfit.DpnFitModel(2)
    probm = dpnfit.DpnFit(mm, states, E[states - 1] * CM_TO_MHZ, x0=truth * [0.9, 1.03, 1, 0.8, 1.2] * CM_TO_MHZ,
                          free=[True, True, False, True, True], lower=[0, 0, -np.inf, -np.inf, 0])
    np.testing.assert_allclose(probm.fit().x / CM_TO_MHZ, truth, rtol=1e-6)
    x = truth * 1.01
    np.testing.assert_allclose(probm.residuals(x * CM_TO_MHZ).detach().numpy(),
                               prob.residuals(x).detach().numpy() * CM_TO_MHZ, rtol=1e-9, atol=1e-6)


def test_aom_identifiability_reported():
    """An isotropic-pi ligand on z: only e_sigma - e_pi is determined (one null direction)."""
    m = dpnfit.DpnFitModel(1, lf_type="A", nlam=3, angles=[(0.0, 0.0, 0.0)], isymm=[1], pi_factors=[(0.0, 0.0, 1.0)],
                           units="cm^-1")
    assert m.parameter_names() == ["zeta", "esigma1", "epi_s1", "epi_c1"]
    truth = np.array([300.0, 5000.0, 1500.0, 0.0])
    E = dpnfit.DpnFit(m, [1], [0.0], x0=truth).energies(truth).detach().numpy()
    prob = dpnfit.DpnFit(m, [3, 5], E[[2, 4]], x0=[250.0, 4000.0, 1000.0, 0.0], free=[True, True, True, False])
    res = prob.fit()
    assert len(res.null) == 1 and set(res.null[0]) == {"esigma1", "epi_s1"}
    assert res.x[1] - res.x[2] == pytest.approx(3500.0, rel=1e-8)
    assert np.isnan(res.stderr[res.free]).all()


def test_measurement_groups_count_degenerate_rows_once():
    m = dpnfit.DpnFitModel(3, units="cm^-1")
    truth = np.array([4800.0, 33000.0, 0.0])
    E = dpnfit.DpnFit(m, [1], [0.0], x0=truth).energies(truth).detach().numpy()
    rng = np.random.default_rng(0)
    bands = [(5, 6), (11, 6), (17, 4)]
    states = np.concatenate([np.arange(s, s + g) for s, g in bands])
    obs = np.concatenate([np.full(g, E[s - 1] + rng.normal(0, 5.0)) for s, g in bands])
    prob = dpnfit.DpnFit(m, states, obs, x0=[4500.0, 32000.0, 0.0], free=[True, True, False])
    res = prob.fit()
    ms = res.measurements
    assert len(ms.groups) == 3 and ms.n_measurements == 3 and ms.dof == 1
    assert np.all(ms.stderr_unscaled[:2] > res.stderr_unscaled[:2] * 1.5)    # ~ sqrt(degeneracy)


# ------------------------------------------------------------------------------ Jacobian, modes, options
@pytest.mark.parametrize("mode", ["exact", "dpn"])
def test_jacobian_vs_central_differences_aom_in_field(mode):
    m = dpnfit.DpnFitModel(4, lf_type="A", nlam=3, degrees=True, units="cm^-1", mode=mode, B0=500.0, theta=30.0,
                           phi=20.0, k=(0.9, 0.9, 0.8),
                           angles=[(10.0, 0.0, 0.0), (95.0, 10.0, 0.0), (90.0, 100.0, 5.0)], isymm=[1, 2, 3])
    x = np.array([1500.0, 41000.0, 20.0, 5000.0, 900.0, 700.0, 4500.0, 800.0, 600.0, 3000.0, 300.0, 200.0])
    prob = dpnfit.DpnFit(m, np.arange(2, 16), np.linspace(100, 60000, 14), x0=x)
    J = prob.jacobian(x)
    assert J.shape == (14, 12)
    fun = lambda v: prob.residuals(v).detach().numpy()  # noqa: E731
    _assert_grad(J, _cd(fun, x))
    Jf = prob.jacobian(x, free=[True, False] * 6)
    np.testing.assert_allclose(Jf, J[:, ::2], rtol=1e-12, atol=1e-12)


def test_zeta_kink_has_one_sided_derivative_at_zero():
    m = dpnfit.DpnFitModel(1, units="cm^-1")
    prob = dpnfit.DpnFit(m, [3], [100.0], x0=[0.0, 500.0, 0.0])
    J0 = prob.jacobian([0.0, 500.0, 0.0])
    Jp = prob.jacobian([1e-9, 500.0, 0.0])
    np.testing.assert_allclose(J0, Jp, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(prob.residuals([-300.0, 500.0, 0.0]).detach().numpy(),
                               prob.residuals([300.0, 500.0, 0.0]).detach().numpy(), rtol=1e-12)


@pytest.mark.parametrize("n", [4, 5])
def test_hole_route_mode_dpn_equals_exact_at_zero_field(n):
    kw = dict(units="cm^-1")
    x = [700.0, 40000.0, 30.0, 1100.0, 200.0] if n == 4 else [700.0, 1100.0, 200.0]
    e = [dpnfit.DpnFit(dpnfit.DpnFitModel(n, mode=mode, **kw), [1], [0.0], x0=x).energies(x) for mode in
         ("exact", "dpn")]
    torch.testing.assert_close(e[0], e[1], rtol=0, atol=1e-8)


def test_lowspn_penalty_and_zero_derivative():
    """p2 with a strong axial field: the ground state is a singlet; LOWSPN with SPNGRND 1 rejects it."""
    m = dpnfit.DpnFitModel(2, units="cm^-1")
    x = np.array([300.0, 20000.0, 0.0, 12000.0, 0.0])
    prob = dpnfit.DpnFit(m, [2, 5], [100.0, 200.0], [1.0, 4.0], x0=x, spngrnd=1.0, lowspn=True)
    ev = prob.evaluate(x)
    assert not ev.ground_spin_ok and ev.spins[0] == 0.0
    np.testing.assert_allclose(ev.residuals, [1e5, 2e5])
    assert np.all(prob.jacobian(x) == 0.0)
    ign = dpnfit.DpnFit(m, [2, 5], [100.0, 200.0], [1.0, 4.0], x0=x, spngrnd=1.0, lowspn=True,
                        lowspn_policy="ignore")
    np.testing.assert_allclose(ign.residuals(x).detach().numpy(), np.sqrt([1.0, 4.0]) * (ev.calc - [100.0, 200.0]))
    pm = dpnfit.DpnFit(dpnfit.DpnFitModel(2), [2], [100.0], x0=x * CM_TO_MHZ, spngrnd=1.0, lowspn=True)
    assert float(pm.residuals(x * CM_TO_MHZ)[0]) == pytest.approx(1e5 * CM_TO_MHZ, rel=1e-12)


def test_input_validation():
    with pytest.raises(ValueError, match="DPNFIT-04"):
        dpnfit.DpnFit(dpnfit.DpnFitModel(1), [7], [1.0], x0=[0.0])
    with pytest.raises(ValueError):
        dpnfit.DpnFitModel(6)
    with pytest.raises(ValueError):
        dpnfit.DpnFitModel(2, mode="fortran")
    with pytest.raises(ValueError):
        dpnfit.DpnFit(dpnfit.DpnFitModel(2), [2], [1.0], x0=[0.0], lowspn_policy="stale")
    p = dpnfit.DpnFit(dpnfit.DpnFitModel(2, units="cm^-1"), [2, 3], [-5.0, 10.0], [-1.0, 2.0], x0=[1.0])
    assert p.exp[0] == 0.0 and p.w[0] == 0.0                  # negative energies and weights -> 0, as DPNFIT
    assert p.parameter_names() == ["zeta"]
    res = dpnfit.DpnFit(dpnfit.DpnFitModel(2, units="cm^-1"), [2], [1.0], x0=[1.0, 2.0], free=[False, False]).fit()
    assert res.message == "all parameters fixed"


# ------------------------------------------------------------------------------ parity with telser-tools
@lru_cache(maxsize=None)
def _ref():
    return json.loads((DATA / "dpnfit_inputs.json").read_text())["cases"]


def _numpy_prob(name, mode):
    pytest.importorskip("telser_tools.dpnfit")
    from telser_tools.dpnfit import parse_input

    return parse_input(_ref()[name]["stdin"]).program(mode)


@lru_cache(maxsize=None)
def _cases():
    try:
        from telser_tools.dpnfit import parse_input
    except ImportError:
        return ["telser-tools-missing"]
    out = []
    for k, c in _ref().items():
        try:
            parse_input(c["stdin"]).program()
            out.append(k)
        except Exception:  # noqa: BLE001  (inputs the original rejects, findings DPNFIT-04/-05)
            pass
    return out


def _convert(p) -> dpnfit.DpnFit:
    """A torchspin DpnFit from a telser_tools DpnFit (cm^-1, degrees, gauss -> mT, the oracle's constants)."""
    from telser_tools.dpn.hamiltonian import CODATA2018, DPN_CONSTANTS

    m = p.model
    C = CODATA2018 if m.mode == "exact" else DPN_CONSTANTS
    tm = dpnfit.DpnFitModel(m.n, m.lf_type, m.nlam, angles=list(m.angles), isymm=list(m.isymm),
                            blfact=list(m.blfact), pi_factors=list(m.pi_factors), orbital_isymm=list(m.orbital_isymm),
                            B0=m.H0 / 10.0, theta=m.theta, phi=m.phi, k=tuple(m.k), mode=m.mode, degrees=True,
                            units="cm^-1", constants=dpn.Constants(C.mu_b * 10.0, C.g_e, C.k_b))
    fp = p.fit_parameters()
    return dpnfit.DpnFit(tm, [o.state for o in p.observations], [o.energy for o in p.observations],
                         [o.weight for o in p.observations], x0=[q.value for q in fp], free=[not q.fixed for q in fp],
                         lower=[q.lower for q in fp], upper=[q.upper for q in fp],
                         step=[q.step if q.step else max(abs(q.value), 1.0) for q in fp], spngrnd=p.spngrnd,
                         lowspn=bool(p.lowspn))


def test_cases_cover_every_configuration_and_mode():
    pytest.importorskip("telser_tools.dpnfit")
    cases = _cases()
    assert len(cases) >= 28
    probs = [_numpy_prob(k, "exact") for k in cases]
    assert {q.model.n for q in probs} == {1, 2, 3, 4, 5}
    assert {q.model.lf_type.upper() for q in probs} == {"A", "X", "P"}
    assert any(q.model.has_field for q in probs)


@pytest.mark.parametrize("mode", ["exact", "dpn"])
@pytest.mark.parametrize("name", _cases())
def test_residuals_and_jacobian_match_numpy(name, mode):
    p = _numpy_prob(name, mode)
    from telser_tools.fitting import jacobian

    q = _convert(p)
    # a generic point: at the symmetric start values single parameters split degenerate levels,
    # where ranked energies have no derivative (Kramers degeneracies remain and are tested)
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    x = np.array(p.x0, float)
    x = x + rng.uniform(0.02, 0.08, len(x)) * np.maximum(np.abs(x), 50.0)
    np.testing.assert_allclose(q.residuals(x).detach().numpy(), p.residuals(x), rtol=1e-10, atol=1e-8)
    hn = np.asarray(p.model.hamiltonian(x), complex)
    np.testing.assert_allclose(q.model.hamiltonian(x).numpy(), hn, rtol=0, atol=1e-9 * max(1.0, np.abs(hn).max()))
    J = q.jacobian(x)
    fds = [jacobian(p.residuals, x, r * np.maximum(np.abs(x), 1.0)) for r in (1e-3, 1e-4, 1e-5)]
    _assert_grad(J, fds)
    diff = pytest.importorskip("telser_tools_diff.dpnfit")
    Jd = diff.dpnfit_jacobian(p)(x, np.ones(len(x), bool))
    np.testing.assert_allclose(J, Jd, rtol=0, atol=1e-9 * max(np.abs(J).max(), 1e-300))


def test_lowspn_case_matches_numpy():
    name = "p2_lowspn_wrong_ground_fit"
    if name not in _cases():
        pytest.skip("reference case unavailable")
    p = _numpy_prob(name, "exact")
    q = _convert(p)
    x = np.asarray(p.x0, float)
    ev_n, ev_t = p.evaluate(x), q.evaluate(x)
    assert ev_n.ground_spin_ok == ev_t.ground_spin_ok is False
    np.testing.assert_allclose(ev_t.residuals, ev_n.residuals, rtol=1e-12)
    np.testing.assert_allclose(ev_t.spins, ev_n.spins)
    assert ev_t.dominant == ev_n.dominant


FITS = ["dpn_fit_AOM", "p2_fit_XF", "p3_fit_XF", "p4_fit_XF", "p2_aom_fit", "p1_porb_fit"]


@pytest.mark.parametrize("name", FITS)
def test_fit_matches_numpy(name):
    p = _numpy_prob(name, "exact")
    if not any(not q.fixed for q in p.fit_parameters()):
        pytest.skip("no free parameter")
    a = p.fit(monitor=False, track_crossings=False)
    b = _convert(p).fit()
    assert b.objective == pytest.approx(a.result.objective, rel=1e-6, abs=1e-8)
    from telser_tools.fitting import null_directions

    na = null_directions(a.result)
    assert len(b.null) == len(na)             # p2_aom_fit: e_sigma - e_pi of one ligand only
    if na:
        return
    f = a.result.free
    np.testing.assert_allclose(b.stderr[f], a.result.stderr[f], rtol=1e-3)
    if a.measurements is not None:
        np.testing.assert_allclose(b.measurements.stderr_unscaled[f], a.measurements.stderr_stepit[f], rtol=1e-3)
        assert b.measurements.n_measurements == a.measurements.n_measurements
