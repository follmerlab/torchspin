"""torchspin.lft.ddnpow: powder EPR (field-swept and FDMR) directly from the complete d^n model.

* forward parity with telser-tools' NumPy DDNPOW (exact mode) on every runnable DDNPOW reference
  input and every cheap option switch (live when telser-tools is importable, else against its
  frozen outputs in data/telser/refs, see ``_telser_refs``);
* a cross-check against :func:`torchspin.pepper` through :func:`torchspin.lft.dn2sys` in the
  effective-spin limit (well-isolated Cu(II) doublet, no hyperfine);
* gradients (forward-mode AD) against central differences, and the Rayleigh-Ritz subspace on
  the graph against the full basis;
* units (cm^-1 vs MHz), the grid copy and the Fortran-only options.

Tolerance of the parity tests.  The d^n levels sit at up to ~10^4-10^5 cm^-1, the EPR
transitions are differences of two such eigenvalues and the lines are 10^-3-10^-2 cm^-1 wide,
so the spectrum is only defined to ~ N eps max|E| / W relative to its peak: NumPy itself moves
by 2e-9 ... 1e-8 when the energy origin is shifted (``test_numpy_itself_is_roundoff_limited``).
The tolerance is max(1e-10, 2 N eps max|E| / W) (N the matrix size, W the narrowest width in
cm^-1).  The parallel-mode Kramers case has an exactly vanishing spectrum (round-off ~1e-24 in
both): it is compared on the scale of its perpendicular-mode spectrum.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.autograd.forward_ad as fwAD

from torchspin import lft
from torchspin.lft import _powder, ddn, ddnpow
from torchspin.lft._engines import igloo
from torchspin.lft.units import energy_factor
from torchspin.tests import _telser_refs as R

F64 = torch.float64
EPS = np.finfo(float).eps
CM_TO_MHZ = energy_factor("cm^-1", "MHz")
INPUTS = json.loads((Path(__file__).parent / "data" / "telser" / "ddnpow_inputs.json").read_text())["cases"]
LARGE = ("d8_ni_hf_pop", "d3_cr_xband_pop_parallel", "d8_orbitals_mixing_hole")

# Parameter sets of the DDNPOW reference inputs (cm^-1), so the gradient tests need no telser-tools.
V_NI = torch.tensor([[5580.0, 0, 0, 0, 92.4], [0, -2920.0, 0, 0, 0], [0, 0, -3300.0, 0, 0], [0, 0, 0, -3140.0, 0],
                     [92.4, 0, 0, 0, 3780.0]], dtype=F64)
V_CU = torch.diag(torch.tensor([9900.0, -2100.0, -5100.0, -5100.0, 2400.0], dtype=F64))
NI = dict(B=900.0, C=3800.0, zeta=550.0, V=V_NI)
CU = dict(zeta=830.0, k=(0.8, 0.8, 0.8), V=V_CU)


def _lowsym_V():
    """A d^1 AOM field without D2h symmetry (hemisphere grid)."""
    ligs = [lft.Ligand.from_degrees(t, p, 0.0, e_sigma=es, e_pi_x=ep, e_pi_y=ep)
            for t, p, es, ep in [(5, 10, 6000, 1200), (175, 200, 5500, 900), (85, 15, 5000, 1000),
                                 (95, 110, 5200, 800), (80, 185, 4800, 1100), (100, 290, 6100, 1000)]]
    return lft.aom(ligs)


# ------------------------------------------------------------------------------ self-contained
def test_grid_engine_matches_powder_grid():
    for exact in (True, False):
        g = _powder.make_grid(5, 2, exact=exact)
        ref = igloo.make_grid(5, 2, exact=exact)
        np.testing.assert_array_equal(g.directions.numpy(), ref.directions)
        np.testing.assert_array_equal(g.weight.numpy(), ref.weight)
    assert float(_powder.exact_grid(7, 1).weight.sum()) == pytest.approx(1.0, abs=1e-14)


def test_units_mhz_vs_cm1():
    B = torch.linspace(280.0, 360.0, 81, dtype=F64)
    kw = dict(igloo=4, derivative=True, temperature=10.0)
    a = ddnpow.simulate(9, B, 9.5, width=torch.tensor([40.0, 40.0, 60.0], dtype=F64) / CM_TO_MHZ, units="cm^-1",
                        **CU, **kw)
    b = ddnpow.simulate(9, B, 9.5, width=(40.0, 40.0, 60.0), units="MHz", zeta=830.0 * CM_TO_MHZ,
                        k=(0.8, 0.8, 0.8), V=V_CU * CM_TO_MHZ, **kw)
    scale = a.spectrum.abs().max()
    assert scale > 0
    assert float((a.spectrum - b.spectrum).abs().max() / scale) < 1e-7
    torch.testing.assert_close(a.axis, B)


def test_effective_spin_limit_matches_pepper():
    """Cu(II), well-isolated ground doublet (next state ~7000 cm^-1 up), no hyperfine: the full d^n
    powder spectrum equals pepper on dn2sys's S = 1/2 system with the doublet's full g.

    ddnpow: Gaussian absorption, isotropic HWHM W = 80 MHz in frequency, no Flygare factor
    (EasySpin convention).  pepper: HStrain = 2W (FWHM in MHz, the same frequency-domain width).
    Both are normalized to unit area.  Expected differences, relative to the peak: (1) quadrature:
    the igloo grid has no interpolation; its error against a converged pepper falls from 0.4
    (igloo 10, W = 40 MHz) to 9e-2 (20) and ~1e-3 (igloo 30, W = 80 MHz, measured 1.1e-3, no
    better at igloo 40); (2) the effective-spin approximation, mu_B B / gap ~ 0.3/7000 ~ 5e-5.
    Tolerance 5e-3.  A wrong g (e.g. k = 0.9 instead of 0.8 in pepper) gives deviations of order 1.
    """
    from torchspin import Experiment, Options, pepper

    W = 80.0
    B = torch.linspace(250.0, 380.0, 261, dtype=F64)
    with torch.no_grad():
        r = ddnpow.simulate(9, B, 9.5, width=torch.full((3,), W / CM_TO_MHZ, dtype=F64), igloo=30,
                            frequency_factor=False, units="cm^-1", **CU)
    sys, gt = lft.dn2sys(9, CU, units="cm^-1", min_gap=5000.0, HStrain=[2 * W] * 3)
    x, y = pepper(sys, Experiment(mwFreq=9.5, Range=[250.0, 380.0], nPoints=261, Harmonic=0),
                  Options(GridSize=[31, 4], GridSymmetry="D2h", Method="matrix"))
    x = np.asarray(x, float).ravel()
    np.testing.assert_allclose(x, B.numpy(), atol=1e-9)
    y1 = r.spectrum.numpy()
    y2 = np.asarray(y, float).ravel()
    y1, y2 = y1 / np.trapezoid(y1, x), y2 / np.trapezoid(y2, x)
    assert np.abs(y1 - y2).max() / y1.max() < 5e-3


def test_fortran_reproduction_is_not_ported():
    B = torch.linspace(300.0, 340.0, 5, dtype=F64)
    for kw in (dict(ddn_model=True), dict(mirror=True), dict(probabilities="fortran"), dict(zero_field="fortran"),
               dict(thresholds=True), dict(mode="fortran")):
        with pytest.raises(NotImplementedError):
            ddnpow.simulate(9, B, 9.5, width=(30.0, 30.0, 30.0), igloo=2, units="cm^-1", **CU, **kw)
    with pytest.raises(TypeError):
        ddnpow.simulate(9, B, 9.5, width=(30.0, 30.0, 30.0), Dq=100.0)
    with pytest.raises(ValueError):
        ddnpow.simulate(9, B, 9.5, width=(30.0, 30.0, 30.0), pairs=[(0, 10)])


# ------------------------------------------------------------------------------ gradients
def _jac_forward(fun, x0: np.ndarray) -> np.ndarray:
    cols = []
    eye = torch.eye(len(x0), dtype=F64)
    for j in range(len(x0)):
        with fwAD.dual_level():
            y = fun(fwAD.make_dual(torch.tensor(x0, dtype=F64), eye[j]))
            cols.append(fwAD.unpack_dual(y).tangent.detach().clone())
    return torch.stack(cols, 1).numpy()


def _fd_jacobian(fun, x, s):
    cols = []
    for j in range(len(x)):
        e = np.zeros_like(x)
        e[j] = s[j]
        with torch.no_grad():
            cols.append(((fun(torch.tensor(x + e)) - fun(torch.tensor(x - e))) / (2 * s[j])).numpy())
    return np.stack(cols, 1)


def _best_fd(fun, x, steps, ad):
    """Central differences at h max(|x_j|, 1) and h/2, plus Richardson; per column the candidate
    closest to AD (the FD reference is limited by the ~1e-8 round-off of the spectrum / step)."""
    cands = []
    for h in steps:
        s = np.maximum(np.abs(x), 1.0) * h
        d1 = _fd_jacobian(fun, x, s)
        d2 = _fd_jacobian(fun, x, s / 2)
        cands += [d1, d2, (4 * d2 - d1) / 3]
    pick = np.stack([np.abs(c - ad).max(0) for c in cands]).argmin(0)
    return np.stack([cands[pick[j]][:, j] for j in range(len(x))], 1)


def _scaled_error(ad, fd, x) -> float:
    s = np.maximum(np.abs(x), 1.0)
    return float(np.abs((ad - fd) * s).max() / np.abs(fd * s).max())


def _setup(base: dict, which, run):
    """x0 and fun(x) for the parameters ``which`` of ``base``; ``t`` is a ligand-field direction
    V = V0 + t dV (dV keeps D2h when V0 has it, so the grid extent does not flip)."""
    V0 = base["V"]
    dV = np.random.default_rng(3).normal(size=(5, 5))
    dV = (dV + dV.T) * 50.0
    if ddnpow.is_d2h(V0):
        m = np.eye(5)
        m[0, 4] = m[4, 0] = 1.0
        dV = dV * m
    dV = torch.tensor(dV)
    vals = {k: np.atleast_1d(np.asarray(0.0 if k == "t" else base[k], float)) for k in which}
    x0 = np.concatenate([vals[k] for k in which])

    def fun(x, **opt):
        kw, a = dict(base), 0
        for k in which:
            n = len(vals[k])
            kw[k] = x[a:a + n] if n > 1 else x[a]
            a += n
        t = kw.pop("t", None)
        if t is not None:
            kw["V"] = V0 + t * dV
        return run(kw, **opt)

    return x0, fun


def _check(base, which, run, tol=1e-5, steps=(1e-4, 1e-5)):
    x0, fun = _setup(base, which, run)
    ad = _jac_forward(fun, x0)
    assert np.isfinite(ad).all() and np.abs(ad).max() > 0
    err = _scaled_error(ad, _best_fd(fun, x0, steps, ad), x0)
    assert err < tol, (which, err)


def _field_run(n, fields, igloo, **fixed):
    def run(kw, **opt):
        mw = kw.pop("mw_freq", 9.5)
        return ddnpow.simulate(n, fields, mw, igloo=igloo, units="cm^-1", **fixed, **kw, **opt).spectrum
    return run


def test_d9_copper_derivative_gradients():
    """Kramers doublet: the field-derivative spectrum vs zeta, k, a D2h ligand-field direction, W, nu."""
    base = dict(CU, k=np.array([0.8, 0.8, 0.8]), width=np.array([40.0, 40.0, 60.0]) / CM_TO_MHZ, mw_freq=9.5, t=0.0)
    _check(base, ["zeta", "k", "t", "width", "mw_freq"],
           _field_run(9, torch.linspace(260.0, 360.0, 201, dtype=F64), 4, derivative=True))


def test_d1_low_symmetry_aom_hemisphere_gradients():
    base = dict(zeta=154.0, V=_lowsym_V(), width=np.full(3, 60.0 / CM_TO_MHZ), t=0.0)
    assert not ddnpow.is_d2h(base["V"])
    _check(base, ["zeta", "t", "width"], _field_run(1, torch.linspace(200.0, 500.0, 151, dtype=F64), 3))


def test_d8_ritz_sweep_population_gradients():
    """d^8 (45 states) at 230 GHz through the Rayleigh-Ritz subspace, Boltzmann populations."""
    fields = torch.linspace(0.0, 12000.0, 61, dtype=F64)
    base = dict(NI, width=np.full(3, 300.0 / CM_TO_MHZ), temperature=5.0, mw_freq=230.0, t=0.0)
    fixed = dict(pairs=[(0, 1), (0, 2), (1, 2)])
    with torch.no_grad():
        assert ddnpow.simulate(8, fields, 230.0, igloo=2, units="cm^-1", **fixed,
                               **{k: v for k, v in base.items() if k not in ("t", "mw_freq")}).subspace_used
    _check(base, ["zeta", "B", "t", "temperature"], _field_run(8, fields, 2, **fixed))


@pytest.mark.parametrize("field,shape,temperature,ig", [(5000.0, "L", 10.0, 6), (0.0, "G", None, 1)])
def test_d8_fdmr_gradients(field, shape, temperature, ig):
    axis = torch.linspace(0.0, 15.0, 301, dtype=F64)

    def run(kw, **opt):
        return ddnpow.fdmr(8, axis, field=field, shape=shape, temperature=temperature, igloo=ig, units="cm^-1",
                           pairs=[(0, 1), (0, 2), (1, 2)], **kw, **opt).spectrum

    base = dict(NI, width=np.full(3, 2000.0 / CM_TO_MHZ), t=0.0)
    _check(base, ["zeta", "B", "t", "width"], run)


def test_ritz_subspace_graph_equals_full_basis():
    """Projecting H onto the fixed (detached) Ritz basis: values and Jacobian vs the full basis."""
    fields = torch.linspace(0.0, 12000.0, 61, dtype=F64)
    base = dict(NI, width=np.full(3, 300.0 / CM_TO_MHZ), temperature=5.0, mw_freq=230.0, t=0.0)
    run = _field_run(8, fields, 2, pairs=[(0, 1), (0, 2), (1, 2)])
    x0, fun = _setup(base, ["zeta", "B", "t", "width", "temperature"], run)
    with torch.no_grad():
        a, b = fun(torch.tensor(x0)), fun(torch.tensor(x0), subspace=False)
    assert float((a - b).abs().max() / b.abs().max()) < 1e-7
    ja = _jac_forward(fun, x0)
    jb = _jac_forward(lambda x: fun(x, subspace=False), x0)
    assert _scaled_error(ja, jb, x0) < 1e-5


# ------------------------------------------------------------------------------ B94: d^7 Rayleigh-Ritz gradients
# Co(II) (d^7, 120 states) in a flattened tetrahedron of four AOM ligands: the reported failing case
# (YangHolonomy co_kramers_epr checkout survey).  Its ground-doublet transition moments respond to
# the ligand field through spin-orbit admixture of excited quartets, i.e. through eigenvector
# components OUTSIDE the Rayleigh-Ritz space.  0.9.29 differentiated the eigenvectors inside the
# fixed (detached) space only.  Measured (torch 2.15, OMP 4), scaled Jacobian error Ritz vs full
# basis: 0.9.29 1.6e-1 (absorption), 6.5e-2 (field derivative), 2.8e-2 (Lorentzian, 4 K); 0.9.30
# 1.7e-9, 3.5e-9, 1.7e-9.  Central differences of this spectrum bottom out at ~1e-6 (levels at
# ~1e4 cm^-1 vs 0.02 cm^-1 lines; d/d e_sigma, h = 35, 3.5, 0.35, 0.035: 8.0e-4, 8.1e-6, 3.1e-6,
# 1.9e-5 on the Ritz path, 8.0e-4, 8.1e-6, 1.2e-6, 2.2e-5 on the full basis), so the full basis is
# the sharp reference and central differences the independent one.
_CO = dict(B=730.0, C=3285.0, zeta=430.0, k=np.array([0.9, 0.9, 0.9]))
_CO_FIELDS = torch.linspace(20.0, 800.0, 157, dtype=F64)


def _co_V(es, tilt_deg=5.0):
    t = float(np.degrees(np.arccos(1.0 / np.sqrt(3.0)))) + tilt_deg
    pos = [(t, 45.0), (t, 225.0), (180.0 - t, 135.0), (180.0 - t, 315.0)]
    return lft.aom([lft.Ligand.isotropic(a, b, es, 900.0, degrees=True) for a, b in pos])


def _co_run(**fixed):
    def run(kw, **opt):
        kw = dict(kw)
        kw["V"] = _co_V(kw.pop("es"))
        return ddnpow.simulate(7, _CO_FIELDS, 9.4, igloo=2, units="cm^-1", pairs=((0, 1),), width=(0.02,) * 3,
                               **fixed, **kw, **opt).spectrum
    return run


_CO_CASES = {"absorption": {}, "field_derivative": dict(derivative=True),
             "lorentzian_4K": dict(shape="lorentzian", temperature=4.0)}


@pytest.mark.parametrize("case", list(_CO_CASES))
def test_d7_ritz_gradients_match_full_basis(case):
    """B94: Jacobian (e_sigma, zeta, Racah B, k) of the Ritz path vs the full basis, d^7."""
    run = _co_run(**_CO_CASES[case])
    x0, fun = _setup(dict(_CO, es=3500.0, V=None), ["es", "zeta", "B", "k"], run)
    with torch.no_grad():
        assert ddnpow.simulate(7, _CO_FIELDS, 9.4, igloo=2, units="cm^-1", pairs=((0, 1),), width=(0.02,) * 3,
                               V=_co_V(3500.0), **_CO).subspace_used
    ja = _jac_forward(fun, x0)
    jb = _jac_forward(lambda x: fun(x, subspace=False), x0)
    assert np.abs(jb).max() > 0
    assert _scaled_error(ja, jb, x0) < 1e-7


def test_d7_ritz_gradients_vs_central_differences():
    """B94: the Ritz-path Jacobian against central differences (Richardson, steps 1e-3/1e-4 relative)."""
    _check(dict(_CO, es=3500.0, V=None), ["es", "zeta", "k"], _co_run(), tol=1e-5, steps=(1e-3, 1e-4))


def test_d7_ritz_values_identical_on_and_off_graph():
    """B94: the graph path returns the no-graph spectrum bit for bit; reverse mode = forward mode."""
    run = _co_run(derivative=True)
    with torch.no_grad():
        y0 = run(dict(_CO, es=torch.tensor(3500.0, dtype=F64)))
    es = torch.tensor(3500.0, dtype=F64, requires_grad=True)
    y = run(dict(_CO, es=es))
    assert torch.equal(y.detach(), y0)
    w = torch.linspace(-1.0, 1.0, len(y0), dtype=F64)
    g_rev = torch.autograd.grad((w * y).sum(), es)[0]
    with fwAD.dual_level():
        yd = run(dict(_CO, es=fwAD.make_dual(torch.tensor(3500.0, dtype=F64), torch.tensor(1.0, dtype=F64))))
        primal, tangent = fwAD.unpack_dual(yd)
        assert torch.equal(primal, y0)
        g_fwd = (w * tangent).sum()
    # two summation orders of a roundoff-limited spectrum (defined to ~1e-9, see the module
    # docstring): measured 2.5e-11 of sum |w| |dy| (torch 2.10/2.15, 1 and 8 threads); 0.9.29's
    # reverse and forward modes differ by 5e-11 per point as well
    assert abs(float(g_rev - g_fwd)) <= 1e-9 * float((w.abs() * tangent.abs()).sum())


# ------------------------------------------------------------------------------ parity with telser-tools
# telser-tools' values come from oracles: live when it is importable, else the frozen copies in
# data/telser/refs/ddnpow.npz (see _telser_refs)
_SWITCH_INPUTS = (("d2_mindiola_hf", dict(igloo=2, npts=60)), ("d1_tet_deriv_lorentz", dict(igloo=3)))
_SWITCHES = (dict(frequency_factor=False), dict(normalized_populations=False), dict(exact_cutoff=False),
             dict(field_derivative=False), dict(grid="fortran"), dict(extent="hemisphere"), dict(subspace=False))


def _case_id(name, over=None, opt=None):
    fmt = lambda d: ",".join(f"{k}={v}" for k, v in (d or {}).items())  # noqa: E731
    return f"{name}|{fmt(over)}|{fmt(opt)}"


_CASES = {_case_id(n): (n, {}, {}) for n in sorted(INPUTS)}
_CASES.update({_case_id(n, over, opt): (n, over, opt) for opt in _SWITCHES for n, over in _SWITCH_INPUTS})
_INP_FIELDS = ("n", "i_min", "i_max", "j_min", "j_max", "width", "shape", "cutoff", "temperature", "h1mode", "igloo",
               "powder", "angle_range", "kind", "h0", "freq_ghz", "deriv")


@R.oracle("ddnpow", cases=list(_CASES))
def _telser_spectrum(case):
    """The parsed input (with the test's overrides; what _ours needs), its x axis (field in G, or energy in
    MHz for kind F), and telser-tools' exact-mode spectrum with the options; for an exactly vanishing
    spectrum also the maximum of the same run with h1mode = 0 (its scale)."""
    import telser_tools.ddnpow as tp
    from telser_tools.ddn.hamiltonian import CODATA2018 as C18
    name, over, opt = _CASES[case]
    inp = tp.parse_input(INPUTS[name])
    for k, v in over.items():
        setattr(inp, k, v)
    p = inp.parameters()
    x = tp.program.energy_axis(inp, CM_TO_MHZ)[1] if inp.kind == "F" else tp.program.field_axis(inp)[1]
    y = tp.simulate(inp, **opt).spectra[0].y
    alt = None
    if np.abs(y).max() < 1e-20:
        i2 = copy.copy(inp)
        i2.h1mode = 0.0
        alt = float(np.abs(tp.simulate(i2, **opt).spectra[0].y).max())
    return dict(C18=dict(mu_b=C18.mu_b, g_e=C18.g_e, k_b=C18.k_b),
                inp={f: getattr(inp, f) for f in _INP_FIELDS},
                params=dict(zeta=p.zeta, V=p.ligand_field, k=list(p.k), alpha=p.alpha,
                            slater=None if p.slater is None else list(p.slater), B=p.B, C=p.C, A=p.A),
                x=x, y=y, h1mode0_max=alt)


def _dn_kwargs(p):
    kw = dict(zeta=p["zeta"], V=None if p["V"] is None else torch.tensor(p["V"]), k=tuple(p["k"]), alpha=p["alpha"])
    kw.update(slater=tuple(p["slater"])) if p["slater"] is not None else kw.update(B=p["B"], C=p["C"], A=p["A"])
    return kw


def _ours(ref, **opt):
    """Our spectrum for a telser-tools DdnpowInput (cm^-1 parameters, telser-tools' CODATA 2018 constants)."""
    inp, c = ref["inp"], ref["C18"]
    kw = _dn_kwargs(ref["params"])
    pairs = [(i - 1, j - 1) for i in range(inp["i_min"], inp["i_max"] + 1)
             for j in range(inp["j_min"], inp["j_max"] + 1)
             if j > i]
    angle_range = None if inp["powder"] or inp["angle_range"] is None else tuple(inp["angle_range"])
    kw.update(pairs=pairs, width=torch.tensor(inp["width"], dtype=F64) / CM_TO_MHZ, shape=inp["shape"],
              cutoff=inp["cutoff"], temperature=inp["temperature"], h1mode=inp["h1mode"], igloo=inp["igloo"],
              units="cm^-1", degrees=True, angle_range=angle_range,
              constants=ddn.Constants(mu_b=c["mu_b"] * 10.0, g_e=c["g_e"], k_b=c["k_b"]), **opt)
    with torch.no_grad():
        if inp["kind"] == "F":
            x = ref["x"] / CM_TO_MHZ                                                # cm^-1 (or THz) -> cm^-1
            kw.pop("subspace", None)
            kw.pop("field_derivative", None)
            return ddnpow.fdmr(inp["n"], torch.tensor(x), field=inp["h0"] / 10.0, **kw).spectrum.numpy()
        y = ddnpow.simulate(inp["n"], ref["x"] / 10.0, inp["freq_ghz"], derivative=inp["deriv"], **kw).spectrum.numpy()
        return y * (0.1 if inp["deriv"] and opt.get("field_derivative", True) else 1.0)   # d/dB per mT -> per G


def _roundoff_tolerance(ref) -> float:
    E = ddn.levels(ref["inp"]["n"], units="cm^-1", **_dn_kwargs(ref["params"])).numpy()
    w_cm = min(w for w in ref["inp"]["width"] if w > 0) / CM_TO_MHZ
    return max(1e-10, 2 * len(E) * EPS * np.abs(E).max() / w_cm)


def _assert_parity(name, over=None, **opt):
    ref = R.ref(_telser_spectrum, _case_id(name, over, opt))
    a = ref["y"]
    y = _ours(ref, **opt)
    scale = np.abs(a).max()
    if scale < 1e-20:
        scale = 1e-2 * ref["h1mode0_max"]
    dev = np.abs(a - y).max() / scale
    tol = _roundoff_tolerance(ref)
    assert dev <= tol, (dev, tol)
    return dev


@pytest.mark.parametrize("name", [n for n in sorted(INPUTS) if n not in LARGE])
def test_parity_golden_inputs(name):
    _assert_parity(name)


@pytest.mark.slow
@pytest.mark.parametrize("name", LARGE)
def test_parity_large_golden_inputs(name):
    """The three largest inputs (d^8, d^3; Rayleigh-Ritz sweep, populations): ~3 s each for both ports."""
    _assert_parity(name)


@pytest.mark.parametrize("over", _SWITCHES)
def test_parity_option_switches(over):
    for name, inp_over in _SWITCH_INPUTS:
        _assert_parity(name, inp_over, **over)


@R.oracle("ddnpow", drift={"*$.shift": None})          # shift is itself a round-off measurement
def _telser_origin_shift():
    """max |y - y'| / max |y| of telser-tools' d1_tet_deriv_lorentz spectrum, y' with H0 + 1000 I."""
    import telser_tools.ddnpow as tp
    from telser_tools.ddnloop import model as M
    inp = tp.parse_input(INPUTS["d1_tet_deriv_lorentz"])
    a = tp.simulate(inp).spectra[0].y
    orig = M.Model.h0
    try:
        M.Model.h0 = property(lambda self: orig.func(self) + 1000.0 * np.eye(self.size))
        b = tp.simulate(inp).spectra[0].y
    finally:
        M.Model.h0 = orig
    return dict(shift=float(np.abs(a - b).max() / np.abs(a).max()))


def test_numpy_itself_is_roundoff_limited():
    """Shifting the energy origin (H0 + 1000 I, the same physics) moves NumPy's own spectrum by the
    same order as the torch/NumPy difference."""
    shift = R.ref(_telser_origin_shift)["shift"]
    dev = _assert_parity("d1_tet_deriv_lorentz")
    assert shift > 1e-9 and dev < 3 * shift


_GRID_ARGS = [(4, 1, None), (6, 2, None), (5, 1, (10.0, 70.0, 0.0, 40.0))]


@R.oracle("ddnpow")
def _telser_grids():
    from telser_tools.ddpowh import grid as tg
    from telser_tools.ddpowh import lineshape as tls
    grids = [{kind: dict(directions=g.directions, weight=g.weight)
              for kind, g in (("exact", tg.exact_grid(*args)), ("fortran", tg.fortran_grid(*args)))}
             for args in _GRID_ARGS]
    return dict(grids=grids, DL2=tls.DL2, area_fraction_G8=tls.area_fraction("G", 8, True))


def test_grid_engine_parity_with_telser_tools():
    ref = R.ref(_telser_grids)
    for args, want in zip(_GRID_ARGS, ref["grids"]):
        for ours, kind in ((igloo.exact_grid(*args), "exact"), (igloo.fortran_grid(*args), "fortran")):
            np.testing.assert_array_equal(ours.directions, want[kind]["directions"])
            np.testing.assert_array_equal(ours.weight, want[kind]["weight"])
    assert igloo.DL2 == ref["DL2"]
    assert igloo.area_fraction("G", 8, True) == ref["area_fraction_G8"]
