"""torchspin.lft milestone M3b: joint fits with exact Jacobians (bands, g, chi, M, EPR spectrum)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from torchspin import Experiment, Options, lft
from torchspin.lft import ddnmag, ddnsus, grids
from torchspin.lft.fit import (Bands, FitParameter, GValues, JointFit, Magnetization, Spectrum, Susceptibility,
                               null_directions)

F64 = torch.float64


# ------------------------------------------------------------------------------ synthetic Co(II), as telser-tools
def _co_model(p):
    return dict(B=p["B"], C=4.5 * p["B"], zeta=p["zeta"], k=p["k"].expand(3),
                V=lft.crystal_field(p["Dq"], p["Ds"], p["Dt"]))


_CO_TRUTH = dict(B=820.0, Dq=950.0, Ds=250.0, Dt=60.0, zeta=480.0, k=0.85)
_CO_BOUNDS = dict(B=(500, 1200), Dq=(500, 1600), Ds=(-800, 800), Dt=(-300, 300), zeta=(100, 700), k=(0.5, 1.0))
_T = np.array([2, 4, 6, 10, 15, 20, 30, 50, 80, 120, 170, 230, 300.0])
_STATES = [13, 27, 39, 45, 53, 59]


def _co_params(values=None):
    v = _CO_TRUTH if values is None else dict(zip(_CO_TRUTH, values))
    return [FitParameter(n, float(v[n]), *_CO_BOUNDS[n]) for n in _CO_TRUTH]


def _fd_best(fun, x, free, rels=(1e-4, 1e-5, 1e-6)):
    """Columnwise best-of-steps central differences."""
    J = []
    for rel in rels:
        cols = []
        for j in np.flatnonzero(free):
            h = rel * max(abs(x[j]), 1.0)
            e = np.zeros_like(x)
            e[j] = h
            cols.append((fun(x + e) - fun(x - e)) / (2 * h))
        J.append(np.array(cols).T)
    return J


def _assert_jac(J, Jfds, rtol):
    err = np.min([np.abs(J - Jf).max(axis=0) for Jf in Jfds], axis=0)
    assert np.all(err <= rtol * np.abs(J).max(axis=0)), err / np.abs(J).max(axis=0)


def test_forward_models_match_direct_calls():
    g = grids.gauss_hemisphere(3)
    fit = JointFit(7, _co_params(), _co_model, [
        Bands(_STATES, np.zeros(6)), Susceptibility(_T, np.zeros(len(_T)), grid=g, chi_units="cgs"),
        Magnetization([1000.0, 5000.0, 5000.0], [2.0, 2.0, 10.0], np.zeros(3), grid=g),
        GValues([2, 2, 2])], units="cm^-1")
    calc = fit.calculate(fit.x0())
    t = _CO_TRUTH
    kw = dict(B=t["B"], C=4.5 * t["B"], zeta=t["zeta"], k=(t["k"],) * 3, V=lft.crystal_field(t["Dq"], t["Ds"], t["Dt"]),
              units="cm^-1")
    E = lft.ddn.levels(7, relative=True, **kw)
    np.testing.assert_allclose(calc["bands"], E[np.array(_STATES) - 1].numpy(), rtol=1e-12)
    np.testing.assert_allclose(calc["g"], lft.ddn.g_tensor(7, **kw).principal.numpy(), rtol=1e-12)
    chiT = ddnsus.chi_T(7, _T, grid=g, chi_units="cgs", **kw).powder[0]
    np.testing.assert_allclose(calc["susceptibility"], chiT.numpy(), rtol=1e-11)
    M = ddnmag.magnetization(7, [1000.0, 5000.0], [2.0, 10.0], grid=g, **kw).powder
    np.testing.assert_allclose(calc["magnetization"], [M[0, 0], M[1, 0], M[1, 1]], rtol=1e-11)


def test_fit_units_are_consistent():
    """The same fit expressed in MHz gives the same residuals (bands in MHz)."""
    f = lft.units.energy_factor("cm^-1", "MHz")

    def model_mhz(p):
        return {k: (v * f if k not in ("k",) else v) for k, v in _co_model(p).items()}

    a = JointFit(7, _co_params(), _co_model, [Bands(_STATES, np.full(6, 5000.0), 80.0)], units="cm^-1")
    b = JointFit(7, _co_params(), model_mhz, [Bands(_STATES, np.full(6, 5000.0 * f), 80.0 * f)], units="MHz")
    np.testing.assert_allclose(b.residuals(b.x0()), a.residuals(a.x0()), rtol=1e-9)


def test_joint_jacobian_matches_central_differences():
    g = grids.gauss_hemisphere(2)
    fit = JointFit(7, _co_params(), _co_model, [
        Bands(_STATES, np.zeros(6), 80.0), Susceptibility(_T[::3], np.zeros(5), 0.02, grid=g),
        Magnetization([2000.0, 5000.0], [2.0, 4.0], np.zeros(2), 0.01, grid=g), GValues([2, 2, 2], 0.02)],
        units="cm^-1")
    x = fit.x0()
    free = np.ones(len(x), bool)
    _assert_jac(fit.jacobian(x, free), _fd_best(fit.residuals, x, free), 2e-6)


def test_bands_alone_leave_k_exactly_undetermined():
    fit = JointFit(7, _co_params(), _co_model, [Bands(_STATES, np.zeros(6), 80.0)], units="cm^-1")
    J = fit.jacobian(fit.x0())
    assert np.abs(J[:, list(_CO_TRUTH).index("k")]).max() == 0.0


@pytest.mark.slow
def test_joint_fit_determines_what_single_data_types_cannot():
    """Reproduces telser-tools docs/diff.md: bands leave k undetermined; bands + chi T + g recover all six."""
    g = grids.gauss_hemisphere(3)
    rng = np.random.default_rng(11)
    base = JointFit(7, _co_params(), _co_model, [
        Bands(_STATES, np.zeros(6), 80.0), Susceptibility(_T, np.zeros(len(_T)), 0.02, grid=g),
        GValues([2, 2, 2], 0.02)], units="cm^-1")
    c = base.calculate(base.x0())
    s_chi = 0.01 * np.maximum(c["susceptibility"], 1.0)
    start = np.array(list(_CO_TRUTH.values())) * [1.08, 0.93, 1.3, 0.6, 0.8, 0.95]
    fit = JointFit(7, _co_params(start), _co_model,
                   [Bands(_STATES, c["bands"] + rng.normal(0, 80.0, 6), 80.0),
                    Susceptibility(_T, c["susceptibility"] + rng.normal(0, s_chi), s_chi, grid=g),
                    GValues(c["g"] + rng.normal(0, 0.02, 3), 0.02)], units="cm^-1")
    bands = fit.fit(["bands"])
    assert {"k": 1.0} in [{k: round(v, 6) for k, v in d.items()} for d in bands.null_directions]
    out = fit.fit()
    r = out.result
    assert out.null_directions == []
    truth = np.array(list(_CO_TRUTH.values()))
    assert np.all(np.abs(r.values - truth) < 3.0 * r.stderr)
    zeta = r.names.index("zeta")
    assert r.stderr[zeta] < 0.5 * bands.result.stderr[zeta]


def test_ddnfit_front_end_recovers_parameters():
    """Noiseless d3 bands: DDNFIT recovers B and Dq from a distant start."""
    truth = dict(B=720.0, Dq=1620.0)
    model = lambda p: dict(B=p["B"], C=4.2 * p["B"], V=lft.crystal_field(p["Dq"]))  # noqa: E731
    E = lft.ddn.levels(3, relative=True, units="cm^-1", **model({k: torch.tensor(v, dtype=F64) for k, v in truth.items()}))
    states = [5, 17, 29]
    out = lft.ddnfit.fit(3, [FitParameter("B", 600.0, 400.0, 1100.0), FitParameter("Dq", 1400.0, 800.0, 2500.0)],
                         model, states, E[np.array(states) - 1].numpy(), sigma=50.0, units="cm^-1")
    np.testing.assert_allclose(out.result.values, [720.0, 1620.0], rtol=1e-8)
    assert out.result.success and out.null_directions == []


def test_ddnsus_front_end_recovers_zeta():
    T = np.array([2.0, 5.0, 10.0, 30.0, 100.0, 300.0])
    g = grids.gauss_hemisphere(2)
    model = lambda p: dict(B=820.0, C=3690.0, zeta=p["zeta"], k=(0.85,) * 3,  # noqa: E731
                           V=lft.crystal_field(950.0, 250.0, 60.0))
    data = ddnsus.chi_T(7, T, grid=g, chi_units="cgs", **model({"zeta": torch.tensor(480.0, dtype=F64)}), units="cm^-1").powder[0]
    out = ddnsus.fit(7, [FitParameter("zeta", 380.0, 100.0, 700.0)], model, T, data.numpy(), 0.01, grid=g,
                     units="cm^-1")
    assert out.result.values[0] == pytest.approx(480.0, rel=1e-7)


def test_null_directions_from_rank_deficient_jacobian():
    from torchspin.lft.fit import FitResult
    J = np.array([[1.0, 2.0, 0.0], [2.0, 4.0, 0.0], [0.0, 0.0, 3.0]])
    r = FitResult(["a", "b", "c"], np.zeros(3), np.zeros(3), np.eye(3), np.ones(3, bool), 0.0, 0, J, True, "", 0)
    (d,) = null_directions(r)
    assert d == pytest.approx({"a": 1.0, "b": -0.5})


# ------------------------------------------------------------------------------ EPR spectrum block
def _cu_model(p):
    tilt = p["tilt"]
    eq = [lft.Ligand.isotropic(math.pi / 2 + (-1) ** i * tilt, i * math.pi / 2, p["es"], p["ep"]) for i in range(4)]
    ax = [lft.Ligand.isotropic(0.0, 0.0, 0.8 * p["es"], 0.0), lft.Ligand.isotropic(math.pi, 0.0, 0.8 * p["es"], 0.0)]
    return dict(zeta=830.0, k=(0.8, 0.8, 0.8), V=lft.aom(eq + ax))


_EXP = dict(mwFreq=9.5, Range=[255.0, 360.0], nPoints=400, Harmonic=1)
_OPT = dict(GridSize=[15, 4], GridSymmetry="D2h", Method="matrix")


def _cu_fit(values, spectrum=None, scale="auto"):
    params = [FitParameter("es", values[0], 4000.0, 8000.0), FitParameter("ep", values[1], 200.0, 2000.0),
              FitParameter("tilt", values[2], 0.0, 0.5)]
    blk = Spectrum(Experiment(**_EXP), Options(**_OPT), np.zeros(_EXP["nPoints"]) if spectrum is None else spectrum,
                   1.0, system=dict(lw=[4.0, 0.0]), scale=scale)
    return JointFit(9, params, _cu_model, [blk], units="cm^-1")


@pytest.mark.parametrize("scale", ["auto", 2.0])
def test_spectrum_jacobian_matches_central_differences(scale):
    obs = _cu_fit([6000.0, 1000.0, 0.15], scale=1.0).calculate(np.array([6000.0, 1000.0, 0.15]))["spectrum"]
    assert np.abs(obs).max() > 0
    fit = _cu_fit([6100.0, 950.0, 0.12], spectrum=1.3 * obs, scale=scale)
    x = fit.x0()
    free = np.ones(3, bool)
    J = fit.jacobian(x, free)
    assert np.all(np.abs(J).max(axis=0) > 0)
    _assert_jac(J, _fd_best(fit.residuals, x, free, rels=(1e-5, 1e-6, 1e-7)), 1e-5)


@pytest.mark.slow
def test_spectrum_fit_recovers_ligand_field():
    """Synthetic Cu(II) spectrum (arbitrary amplitude) -> e_pi and tilt recovered (e_sigma fixed: only g enters)."""
    truth = np.array([6000.0, 1000.0, 0.15])
    obs = _cu_fit(truth, scale=1.0).calculate(truth)["spectrum"] * 3.7
    fit = _cu_fit([6000.0, 800.0, 0.08], spectrum=obs)
    fit.parameters[0] = FitParameter("es", 6000.0, 4000.0, 8000.0, fixed=True)
    out = fit.fit()
    np.testing.assert_allclose(out.result.values[1:], truth[1:], rtol=1e-5)
