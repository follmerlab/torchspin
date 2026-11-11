"""torchspin.lft.dtd (Townes-Dailey analysis of a 14N quadrupole tensor), milestone M4.

Self-contained: Telser's DTD.F printout on every fixture set (``tests/data/telser/dtd_fortran.json``,
5-6 printed digits), forward/inverse round trips, closed forms, radians vs degrees, validation,
gradients (central differences, gradcheck in both modes, inverse of the forward Jacobian).
With telser-tools installed: parity with ``telser_tools.dtd`` (corrected mode) at rel 1e-10.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from torchspin.lft import dtd as tdt

F64 = torch.float64
RUNS = json.loads((Path(__file__).parent / "data" / "telser" / "dtd_fortran.json").read_text())["runs"]
SETS = [(r, s) for r in RUNS for s in r["sets"]]


def close(a, b, rtol=1e-10, atol=None):
    a, b = np.asarray(a, float), np.asarray(b, float)
    np.testing.assert_allclose(a, b, rtol=rtol, atol=rtol * max(np.abs(b).max(initial=0.0), 1.0) if atol is None
                               else atol)


def rev_jac(fun, x):
    xt = torch.tensor(np.asarray(x, float), requires_grad=True)
    y = fun(xt).reshape(-1)
    return torch.stack([torch.autograd.grad(y[i], xt, retain_graph=True)[0] for i in range(len(y))]).numpy()


def best_fd(fun, x, J, steps=(1e-4, 1e-5, 1e-6)):
    x = np.asarray(x, float)
    err = np.full(len(x), np.inf)
    for h in steps:
        cols = []
        for j in range(len(x)):
            xp, xm = x.copy(), x.copy()
            xp[j] += h
            xm[j] -= h
            cols.append((fun(xp) - fun(xm)) / (2 * h))
        err = np.minimum(err, np.abs(J - np.array(cols).T).max(0))
    return err.max() / max(np.abs(J).max(), 1e-300)


def _call(mod_fn_angle, mod_fn_alphas, run, s, **kw):
    i = s["input"]
    if run["option"] == "D":
        return mod_fn_angle(run["p0"], i["two_theta"], i["c"], *i["p"], **kw)
    return mod_fn_alphas(run["p0"], i["alpha2"], i["alpha3"], *i["p"], **kw)


def _torch(run, s):
    return _call(tdt.from_angle, tdt.from_alphas, run, s, degrees=True)


# ------------------------------------------------------------------------------ Telser's printout
@pytest.mark.parametrize("rs", SETS, ids=lambda rs: rs[0]["name"])
def test_matches_dtd_fortran_printout(rs):
    """Every set that the corrected mode accepts reproduces dtd.f's printed n, c, a, b, N_T, 2 theta
    to its printed precision; the others raise ValueError (2 theta = 90, c > 2, alpha1^2 <= 0, ...)."""
    run, s = rs
    try:
        got = _torch(run, s)
    except ValueError:
        return
    out = s["output"]
    for k, kt in (("n", "n"), ("c", "c"), ("a", "a"), ("b", "b"), ("nt", "nt"), ("two_theta", "two_theta")):
        if out.get(k) is not None and math.isfinite(out[k]):
            close(getattr(got, kt), out[k], rtol=0, atol=1e-4 * max(1.0, abs(out[k])))


@pytest.mark.parametrize("rs", SETS, ids=lambda rs: rs[0]["name"])
def test_forward_reproduces_the_input_tensor(rs):
    run, s = rs
    try:
        r = _torch(run, s)
    except ValueError:
        return
    P = tdt.efg_from_populations(run["p0"], r.two_theta, r.a, r.b, r.c, degrees=True)
    close(P, s["input"]["p"], rtol=1e-12)


# ------------------------------------------------------------------------------ closed forms and units
def test_hybridization_angle_round_trip_and_sp2():
    tt = torch.tensor([100.0, 120.0, 150.0], dtype=F64)
    n = tdt.hybridization_from_angle(tt, degrees=True)
    close(n[1], 2.0, rtol=1e-14)                                          # 120 degrees: sp2
    close(tdt.angle_from_hybridization(n, degrees=True), tt, rtol=1e-14)
    close(tdt.angle_from_hybridization(n), torch.deg2rad(tt), rtol=1e-14)
    close(tdt.hybridization_from_angle(torch.deg2rad(tt)), n, rtol=1e-14)


def test_radians_equal_degrees():
    a = tdt.from_angle(10.0, 120.0, 1.5, 3.2, -1.8, -1.4, degrees=True)
    b = tdt.from_angle(10.0, math.radians(120.0), 1.5, 3.2, -1.8, -1.4)
    for x, y in zip(a[1:6], b[1:6]):
        close(x, y, rtol=1e-13)
    close(b.two_theta, math.radians(120.0), rtol=0)
    c = tdt.from_alphas(10.0, 0.3, 0.35, 3.2, -1.8, -1.4)
    d = tdt.from_alphas(10.0, 0.3, 0.35, 3.2, -1.8, -1.4, degrees=True)
    close(d.two_theta, math.degrees(float(c.two_theta)), rtol=1e-14)


def test_scale_invariance_in_the_coupling_unit():
    """Populations depend only on P/P0: MHz -> any unit (here cm^-1) leaves them unchanged."""
    f = 1e6 / 2.99792458e10
    a = tdt.from_angle(10.0, 2.1, 1.5, 3.2, -1.8, -1.4)
    b = tdt.from_angle(10.0 * f, 2.1, 1.5, 3.2 * f, -1.8 * f, -1.4 * f, trace_tol=1e-6 * f)
    close(b.populations, a.populations, rtol=1e-13)
    close(tdt.efg_from_populations(10.0 * f, 2.1, 1.0, 1.0, 1.5), f * tdt.efg_from_populations(10.0, 2.1, 1.0, 1.0, 1.5),
          rtol=1e-14)


def test_asymmetry_closed_form_and_traceless_forward():
    close(tdt.asymmetry([3.2, -1.8, -1.4]), (-1.4 + 1.8) / 3.2, rtol=1e-15)
    close(tdt.asymmetry([0.0, 0.0, 0.0]), 0.0)
    P = tdt.efg_from_populations(7.0, 2.0, 1.1, 0.9, 1.6)
    assert abs(float(P.sum())) < 1e-14


def test_batched_inputs():
    tt = torch.tensor([2.0, 2.1, 2.3], dtype=F64)
    r = tdt.from_angle(10.0, tt, 1.5, 3.2, -1.8, -1.4)
    assert r.a.shape == (3,)
    for i in range(3):
        close(r.a[i], tdt.from_angle(10.0, tt[i], 1.5, 3.2, -1.8, -1.4).a, rtol=1e-15)
    with pytest.raises(ValueError):                                       # one bad element rejects the batch
        tdt.from_angle(10.0, torch.tensor([2.0, math.pi / 2]), 1.5, 3.2, -1.8, -1.4)


def test_validation_raises():
    with pytest.raises(ValueError):
        tdt.from_angle(10.0, 90.0, 1.5, 3.2, -1.8, -1.4, degrees=True)
    with pytest.raises(ValueError):
        tdt.from_angle(10.0, 120.0, 2.5, 3.2, -1.8, -1.4, degrees=True)
    with pytest.raises(ValueError):
        tdt.from_angle(10.0, 120.0, 1.5, 3.2, -1.8, -1.0, degrees=True)  # not traceless
    with pytest.raises(ValueError):
        tdt.from_angle(0.0, 120.0, 1.5, 3.2, -1.8, -1.4, degrees=True)
    with pytest.raises(ValueError):
        tdt.from_alphas(10.0, 0.45, 0.3, 3.2, -1.8, -1.4)                 # alpha1^2 <= 0
    with pytest.raises(ValueError):
        tdt.from_alphas(10.0, 0.0, 0.3, 3.2, -1.8, -1.4)


# ------------------------------------------------------------------------------ gradients
def test_gradients_vs_central_differences():
    x = np.array([10.0, 2.0, 1.5, 3.2, -1.8, -1.4])                   # p0, 2 theta (rad), c, P_zz, P_xx, P_yy

    def tf(z):
        r = tdt.from_angle(z[0], z[1], z[2], z[3], z[4], z[5], trace_tol=1.0)
        return torch.stack([r.n, r.a, r.b, r.nt])

    assert best_fd(lambda z: tf(torch.tensor(z)).numpy(), x, rev_jac(tf, x)) < 1e-8
    y = np.array([10.0, 0.3, 0.35, 3.2, -1.8, -1.4])                   # p0, alpha2, alpha3, P

    def ta(z):
        r = tdt.from_alphas(*z.unbind(), trace_tol=1.0)
        return torch.stack([r.two_theta, r.n, r.c, r.a, r.b, r.nt, r.alpha1])

    assert best_fd(lambda z: ta(torch.tensor(z)).numpy(), y, rev_jac(ta, y)) < 1e-8
    w = np.array([8.0, 2.0, 1.1, 0.9, 1.6])                            # p0, 2 theta, a, b, c
    te = lambda z: tdt.efg_from_populations(*z.unbind())
    assert best_fd(lambda z: te(torch.tensor(z)).numpy(), w, rev_jac(te, w)) < 1e-8
    p = np.array([3.2, -1.8, -1.4])
    assert best_fd(lambda z: tdt.asymmetry(torch.tensor(z)).reshape(1).numpy(), p, rev_jac(tdt.asymmetry, p)) < 1e-8


def test_gradcheck_both_modes():
    x = torch.tensor([10.0, 2.0, 1.5, 3.2, -1.8, -1.4], dtype=F64, requires_grad=True)

    def f(z):
        r = tdt.from_angle(*z.unbind(), trace_tol=1.0)
        return torch.stack([r.n, r.a, r.b, r.nt])

    assert torch.autograd.gradcheck(f, (x,), eps=1e-6, atol=1e-7, check_forward_ad=True)
    y = torch.tensor([10.0, 0.3, 0.35, 3.2, -1.8, -1.4], dtype=F64, requires_grad=True)

    def g(z):
        r = tdt.from_alphas(*z.unbind(), trace_tol=1.0)
        return torch.stack([r.two_theta, r.n, r.c, r.a, r.b, r.nt, r.alpha1])

    assert torch.autograd.gradcheck(g, (y,), eps=1e-6, atol=1e-7, check_forward_ad=True)


def test_inverse_is_inverse_of_forward():
    """d(a, b)/d(P_xx, P_yy) at fixed c and P_zz = -(P_xx + P_yy) inverts dP/d(a, b)."""
    p0, tt, a, b, c = 9.0, math.radians(125.0), 1.2, 0.95, 1.7
    xt = torch.tensor([a, b], dtype=F64)
    Jf = torch.func.jacrev(lambda z: tdt.efg_from_populations(p0, tt, z[0], z[1], c)[1:])(xt)
    P = tdt.efg_from_populations(p0, tt, a, b, c)
    Pt = P[1:].clone().requires_grad_()
    r = tdt.from_angle(p0, tt, c, -(Pt[0] + Pt[1]), Pt[0], Pt[1])
    Ji = torch.stack([torch.autograd.grad(v, Pt, retain_graph=True)[0] for v in (r.a, r.b)])
    torch.testing.assert_close(Ji @ Jf, torch.eye(2, dtype=F64), rtol=0, atol=1e-12)
    torch.testing.assert_close(torch.stack([r.a, r.b]).detach(), xt, rtol=1e-12, atol=0)


# ------------------------------------------------------------------------------ parity with telser-tools
@pytest.mark.parametrize("rs", SETS, ids=lambda rs: rs[0]["name"])
def test_parity_with_telser_tools(rs):
    np_td = pytest.importorskip("telser_tools.dtd.townes_dailey")
    run, s = rs
    try:
        ref = _call(np_td.from_angle, np_td.from_alphas, run, s)
    except ValueError:
        with pytest.raises(ValueError):
            _torch(run, s)
        return
    got = _torch(run, s)
    close(got.two_theta, ref.two_theta_deg)
    for k in ("n", "c", "a", "b", "nt"):
        close(getattr(got, k), getattr(ref, k))
    if ref.alpha1 is not None:
        close(got.alpha1, ref.alpha1)
    close(tdt.asymmetry(torch.tensor(s["input"]["p"], dtype=F64)), ref.eta)
    close(tdt.efg_from_populations(run["p0"], ref.two_theta_deg, ref.a, ref.b, ref.c, degrees=True),
          [ref.p_zz, ref.p_xx, ref.p_yy])


def test_gradients_vs_telser_tools_central_differences():
    np_td = pytest.importorskip("telser_tools.dtd.townes_dailey")
    x = np.array([10.0, 120.0, 1.5, 3.2, -1.8, -1.4])

    def npf(z):
        r = np_td.from_angle(*z[:3], *z[3:], trace_tol=1.0)
        return np.array([r.n, r.a, r.b, r.nt])

    def tf(z):
        r = tdt.from_angle(*z.unbind(), degrees=True, trace_tol=1.0)
        return torch.stack([r.n, r.a, r.b, r.nt])

    assert best_fd(npf, x, rev_jac(tf, x)) < 1e-8
