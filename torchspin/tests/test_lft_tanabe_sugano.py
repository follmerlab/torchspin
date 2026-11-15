"""torchspin.lft.tanabe_sugano (milestone M4): cubic d^n levels and diagrams (Telser's DTANSUG).

Self-contained checks (one-electron and 10Dq closed forms, the strong-field blocks against the
full determinant spectrum of lft.ddn for every n and both geometries, units, the diagram's
reduced and absolute axes), gradients against central differences (block energies in forward
and reverse mode, the degenerate determinant spectrum, the diagram in C/B, gradcheck with
forward AD), and parity with telser-tools (live when it is importable, else its frozen outputs in
data/telser/refs, see ``_telser_refs``): the copied block tables are bit-identical, and every Dq-
and B-loop run of the DTANSUG reference, all_levels, the determinant spectrum and the diagrams
match.
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

import numpy as np
import pytest
import torch

from torchspin.lft import tanabe_sugano as ts
from torchspin.lft._engines import tanabe_sugano as eng
from torchspin.lft.units import energy_factor
from torchspin.tests import _telser_refs as R

F64 = torch.float64
CM_TO_MHZ = energy_factor("cm^-1", "MHz")
DATA = Path(__file__).parent / "data" / "telser"
RUNS = json.loads((DATA / "tanabe_sugano_runs.json").read_text())


# ------------------------------------------------------------------------------ closed forms
@pytest.mark.parametrize("geo", ["Oh", "Td"])
def test_d1_one_electron_levels(geo):
    """Oh: t2 at -4Dq, e at +6Dq; Td (Dq > 0 physical): e below t2."""
    lv = {x.label: float(x.energy) for x in ts.all_levels(1, 0.0, 0.0, 1500.0, geo, units="cm^-1")}
    t2, e = (-4 * 1500.0, 6 * 1500.0) if geo == "Oh" else (4 * 1500.0, -6 * 1500.0)
    assert lv["2T2"] == pytest.approx(t2, abs=1e-9)
    assert lv["2E"] == pytest.approx(e, abs=1e-9)


@pytest.mark.parametrize("B,C,Dq", [(766.0, 2855.0, 1200.0), (900.0, 3600.0, 2500.0)])
def test_d3_and_d8_first_band_is_10Dq(B, C, Dq):
    """4A2 -> 4T2 (d3) and 3A2 -> 3T2 (d8) are 10Dq for any B, C."""
    for n, a, t in ((3, "4A2", "4T2"), (8, "3A2", "3T2")):
        ea = ts.block_energies(n, a, B, C, Dq, units="cm^-1")[0]
        et = ts.block_energies(n, t, B, C, Dq, units="cm^-1")[0]
        assert float(et - ea) == pytest.approx(10 * Dq, abs=1e-9)


@pytest.mark.parametrize("n", range(1, 10))
@pytest.mark.parametrize("geo", ["Oh", "Td"])
def test_blocks_equal_determinant_spectrum(n, geo):
    """Strong-field blocks (degeneracies expanded) = the C(10, n) determinant spectrum of lft.ddn.

    DTANSUG leaves out the A term and, for d9, the constant -56B + 28C (A = 0 here)."""
    B, C, Dq = 870.0, 3600.0, 1150.0
    lv = ts.all_levels(n, B, C, Dq, geo, units="cm^-1")
    blk = np.sort(np.concatenate([[float(x.energy)] * x.degeneracy for x in lv]))
    if n == 9:
        blk = blk + (-56 * B + 28 * C)
    det = ts.spectrum(n, B, C, Dq, geo, units="cm^-1").numpy()
    assert len(blk) == comb(10, n) == len(det)
    np.testing.assert_allclose(blk, det, rtol=1e-12, atol=1e-8)


def test_racah_A_shift_and_units():
    B, C, Dq, A = 870.0, 3600.0, 1150.0, 50.0
    e0 = ts.spectrum(4, B, C, Dq, units="cm^-1")
    eA = ts.spectrum(4, B, C, Dq, A=A, units="cm^-1")
    torch.testing.assert_close(eA, e0 + 6 * A, rtol=1e-13, atol=1e-8)
    eM = ts.spectrum(4, B * CM_TO_MHZ, C * CM_TO_MHZ, Dq * CM_TO_MHZ, units="MHz")
    torch.testing.assert_close(eM, e0 * CM_TO_MHZ, rtol=1e-12, atol=1e-3)
    bM = ts.block_energies(4, "3T1", B * CM_TO_MHZ, C * CM_TO_MHZ, Dq * CM_TO_MHZ, units="MHz")
    bC = ts.block_energies(4, "3T1", B, C, Dq, units="cm^-1")
    torch.testing.assert_close(bM, bC * CM_TO_MHZ, rtol=1e-12, atol=1e-3)
    with pytest.raises(ValueError):
        ts.block_energies(4, "3T1", B, C, Dq, units="furlongs")


def test_block_solution_and_scans():
    s = ts.solve_block(3, "4T1", 766.0, 2855.0, 1200.0, units="cm^-1")
    np.testing.assert_allclose(s.weights.sum(-1).numpy(), np.ones(2), atol=1e-13)
    torch.testing.assert_close(s.zeroth_order, torch.diagonal(s.matrix))
    sc = ts.scan_dq(6, "1A1", 1058.0, 4040.0, torch.linspace(0, 3000, 7, dtype=F64), units="cm^-1")
    assert sc.energies.shape == (7, 5) and sc.variable == "Dq"
    torch.testing.assert_close(sc.x, sc.Dq)
    sb = ts.scan_b(6, "1A1", 2000.0, torch.linspace(500, 1100, 4, dtype=F64), 4.0, units="cm^-1")
    torch.testing.assert_close(sb.C, 4.0 * sb.B)
    with pytest.raises(ValueError):
        ts.scan_b(6, "1A1", 2000.0, [500.0], 0.0)
    with pytest.raises(KeyError):
        ts.block_energies(3, "3T1", 1.0, 4.0, 1.0)


@pytest.mark.parametrize("n", [3, 4, 6])
def test_diagram_axes(n):
    """E/B at Dq/B, times B, equals all_levels relative to the ground state at Dq = (Dq/B) B."""
    c, B = 4.5, 800.0
    x = torch.linspace(0.0, 3.5, 15, dtype=F64)
    d = ts.tanabe_sugano(n, c, x)
    assert d.energy is None and d.dq is None
    np.testing.assert_allclose(d.e_over_b.min(0).values.numpy(), 0.0, atol=1e-12)
    a = d.at(B, "cm^-1")
    torch.testing.assert_close(a.dq, x * B)
    for k in (0, 7, 14):
        lv = ts.all_levels(n, B, c * B, float(x[k]) * B, units="cm^-1")
        want = sorted(float(v.energy - lv[0].energy) for v in lv)
        np.testing.assert_allclose(np.sort(a.energy[:, k].numpy()), want, atol=1e-8)
    m = a.to("MHz")
    torch.testing.assert_close(m.energy, a.energy * CM_TO_MHZ, rtol=1e-12, atol=1e-6)
    torch.testing.assert_close(ts.tanabe_sugano(n, c, x, B=B * CM_TO_MHZ, units="MHz").energy, m.energy)
    assert len(d.labels) == len(d.degeneracies) == d.e_over_b.shape[0]


def test_diagram_d4_crossover_and_curves():
    d = ts.tanabe_sugano(4, 4.5)
    assert d.ground[0] == "5E" and d.ground[-1] == "3T1"
    (xc,) = d.crossover()
    assert 2.0 < xc < 3.0
    torch.testing.assert_close(d.curve("5E")[0], torch.tensor(0.0, dtype=F64))
    with pytest.raises(ValueError):
        ts.tanabe_sugano(9)


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


def _assert_grad(J, fds, tol=1e-6):
    err = np.min([np.abs(J - f).max(axis=0) for f in fds], axis=0)
    assert np.all(err <= tol * max(np.abs(J).max(), 1.0)), (err, np.abs(J).max())


def _np(fun):
    return lambda x: fun(torch.tensor(x)).detach().numpy()


@pytest.mark.parametrize("n,label,geo", [(3, "4T1", "Oh"), (6, "1A1", "Oh"), (8, "3T1", "Td"), (5, "4T2", "Oh"),
                                         (4, "3T1", "Oh")])
def test_block_energy_gradients(n, label, geo):
    x0 = np.array([830.0, 3500.0, 1400.0])
    f = lambda x: ts.block_energies(n, label, x[0], x[1], x[2], geo, units="cm^-1")  # noqa: E731
    J = torch.func.jacfwd(f)(torch.tensor(x0)).numpy()
    _assert_grad(J, _fd(_np(f), x0))
    x = torch.tensor(x0, requires_grad=True)
    (g,) = torch.autograd.grad(f(x).sum(), x)
    np.testing.assert_allclose(g.numpy(), J.sum(0), rtol=1e-10, atol=1e-12)


def _level_means(e, tol=1e-6):
    out, cur = [], [e[0]]
    for v in e[1:]:
        if v - cur[-1] < tol * max(1.0, abs(v)):
            cur.append(v)
        else:
            out.append(np.mean(cur))
            cur = [v]
    out.append(np.mean(cur))
    return np.array(out)


@pytest.mark.parametrize("n,geo", [(2, "Oh"), (3, "Td"), (6, "Oh")])
def test_degenerate_determinant_spectrum_gradients(n, geo):
    """Every member of a degenerate cubic level gets its level's derivative (= FD of the level mean)."""
    x0 = np.array([830.0, 3500.0, 1400.0])
    f = lambda x: ts.spectrum(n, x[0], x[1], x[2], geo, units="cm^-1")  # noqa: E731
    e0 = _np(f)(x0)
    J = torch.func.jacfwd(f)(torch.tensor(x0)).numpy()
    assert np.isfinite(J).all()
    grp = np.cumsum(np.r_[0, np.diff(e0) > 1e-6 * np.maximum(1.0, np.abs(e0[1:]))])
    assert grp[-1] + 1 < len(e0)
    Jl = np.array([J[grp == g].mean(0) for g in range(grp[-1] + 1)])
    for g in range(grp[-1] + 1):
        np.testing.assert_allclose(J[grp == g], np.broadcast_to(Jl[g], J[grp == g].shape), atol=1e-8)
    _assert_grad(Jl, _fd(lambda x: _level_means(_np(f)(x)), x0))


def test_diagram_gradient_in_c_over_b_and_B():
    x = torch.linspace(0.5, 3.5, 7, dtype=F64)
    f = lambda c: ts.tanabe_sugano(4, c, x).e_over_b.reshape(-1)  # noqa: E731
    J = torch.func.jacfwd(f)(torch.tensor(4.5, dtype=F64)).numpy()
    fds = [(_np(f)(np.array(4.5 + h)) - _np(f)(np.array(4.5 - h))) / (2 * h) for h in (1e-3, 1e-4)]
    assert min(np.abs(J - d).max() for d in fds) < 1e-6 * np.abs(J).max()
    B = torch.tensor(800.0, dtype=F64, requires_grad=True)
    d = ts.tanabe_sugano(4, 4.5, x, B=B, units="cm^-1")
    (g,) = torch.autograd.grad(d.energy.sum(), B)
    assert float(g) == pytest.approx(float(d.e_over_b.sum()), rel=1e-12)


def test_gradcheck_forward_ad():
    def f(B, C, Dq):
        return ts.block_energies(5, "4T1", B, C, Dq, units="cm^-1")

    args = tuple(torch.tensor(v, dtype=F64, requires_grad=True) for v in (830.0, 3500.0, 1400.0))
    assert torch.autograd.gradcheck(f, args, eps=1e-4, atol=1e-6, rtol=1e-6, check_forward_ad=True)


# ------------------------------------------------------------------------------ parity with telser-tools
def _engine_inputs(n):
    rng = np.random.default_rng(n)
    return [(geo, blk.label, *(rng.normal(size=3) * [900.0, 3500.0, 1500.0]).tolist())
            for geo in eng.GEOMETRIES for blk in eng.blocks_for(n)]


@R.oracle("tanabe_sugano", cases=range(1, 10))
def _telser_engine(n):
    """telser-tools' block table and block matrices (scalar arithmetic only: bit-reproducible)."""
    from telser_tools.tanabe_sugano import blocks as tb
    return dict(blocks=[[b.nh, b.first, b.label, list(b.configs)] for b in tb.blocks_for(n)],
                H=[tb.hamiltonian(n, label, B, C, Dq, geo) for geo, label, B, C, Dq in _engine_inputs(n)])


@pytest.mark.parametrize("n", range(1, 10))
def test_engine_copy_is_bit_identical(n):
    ref = R.ref(_telser_engine, n)
    assert [[b.nh, b.first, b.label, list(b.configs)] for b in eng.blocks_for(n)] == ref["blocks"]
    for (geo, label, B, C, Dq), want in zip(_engine_inputs(n), ref["H"], strict=True):
        assert np.array_equal(eng.hamiltonian(n, label, B, C, Dq, geo), want)


def _parity_cases(key):
    """(run index, NFIG, block label) for every run of the DTANSUG reference; the label (used for the
    test id and the torchspin call) is torchspin's copy of the block table, checked against telser-tools'."""
    return [(i, nfig, eng.block_for_nfig(run["n"], nfig).label)
            for i, run in enumerate(RUNS[key]) for nfig in run["nfigs"]]


def _id(key):
    return lambda c: f"d{RUNS[key][c[0]]['n']}-{RUNS[key][c[0]]['geometry']}-{c[2]}"


def _nondegenerate(e, tol):
    gaps = np.abs(e[:, None] - e[None, :]) + np.eye(len(e)) * 1e300
    return gaps.min(axis=1) > tol


def _scan_ok(e):
    """States whose weights the test compares (the others' are basis-dependent inside a degenerate level)."""
    return _nondegenerate(e, 1e-3 * max(np.ptp(e), 1.0))


def _plain_scan(sc):
    weights = np.array(sc.weights, dtype=float)
    for k, e in enumerate(sc.energies):
        weights[k][~_scan_ok(e)] = np.nan                  # not compared; NaN keeps the drift check meaningful
    return dict(energies=sc.energies, zeroth_order=sc.zeroth_order, weights=weights)


def _check_scan(a, b):
    np.testing.assert_allclose(a.energies.numpy(), b["energies"], rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(a.zeroth_order.numpy(), b["zeroth_order"], rtol=1e-12, atol=1e-9)
    for k in range(len(b["energies"])):
        ok = _scan_ok(b["energies"][k])
        np.testing.assert_allclose(a.weights[k].numpy()[ok], b["weights"][k][ok], atol=1e-10)


def _case_key(c):
    return f"{c[0]}-{c[1]}"


@R.oracle("tanabe_sugano", cases=[_case_key(c) for c in _parity_cases("dq_runs")])
def _telser_dq_loop(case):
    import telser_tools.tanabe_sugano as tts
    i, nfig = map(int, case.split("-"))
    run = RUNS["dq_runs"][i]
    n, geo = run["n"], run["geometry"]
    label = tts.blocks.block_for_nfig(n, nfig).label
    grid = tts.fortran_grid(*run["dq"])
    return dict(label=label, grid=grid, scan=_plain_scan(tts.scan_dq(n, label, run["B"], run["C"], grid, geo)),
                H0=tts.hamiltonian(n, label, run["B"], run["C"], grid[0], geo))


@pytest.mark.parametrize("case", _parity_cases("dq_runs"), ids=_id("dq_runs"))
def test_dq_loops_match_telser_tools(case):
    ref = R.ref(_telser_dq_loop, _case_key(case))
    run, label = RUNS["dq_runs"][case[0]], case[2]
    assert ref["label"] == label
    n, geo, grid = run["n"], run["geometry"], ref["grid"]
    _check_scan(ts.scan_dq(n, label, run["B"], run["C"], torch.tensor(grid), geo, units="cm^-1"), ref["scan"])
    np.testing.assert_allclose(ts.block_hamiltonian(n, label, run["B"], run["C"], grid[0], geo, units="cm^-1").numpy(),
                               ref["H0"], rtol=1e-14, atol=1e-10)


@R.oracle("tanabe_sugano", cases=[_case_key(c) for c in _parity_cases("b_runs")])
def _telser_b_loop(case):
    import telser_tools.tanabe_sugano as tts
    i, nfig = map(int, case.split("-"))
    run = RUNS["b_runs"][i]
    n, geo = run["n"], run["geometry"]
    label = tts.blocks.block_for_nfig(n, nfig).label
    b = tts.fortran_grid(*run["b"])
    dq = tts.blocks.cubic_sign(n, geo) * run["dq"]
    return dict(label=label, b=b, dq=dq, scan=_plain_scan(tts.scan_b(n, label, dq, b, run["c_over_b"], geo)))


@pytest.mark.parametrize("case", _parity_cases("b_runs"), ids=_id("b_runs"))
def test_b_loops_match_telser_tools(case):
    ref = R.ref(_telser_b_loop, _case_key(case))
    run, label = RUNS["b_runs"][case[0]], case[2]
    assert ref["label"] == label
    n, geo = run["n"], run["geometry"]
    _check_scan(ts.scan_b(n, label, ref["dq"], torch.tensor(ref["b"]), run["c_over_b"], geo, units="cm^-1"),
                ref["scan"])


def test_parity_cases_were_collected():
    assert len(_parity_cases("dq_runs")) == 141 and len(_parity_cases("b_runs")) == 58


_LEVELS_BCDQ = (870.0, 3600.0, 1150.0)


@R.oracle("tanabe_sugano", cases=[f"{n}-{geo}" for n in range(1, 10) for geo in ("Oh", "Td")])
def _telser_all_levels(case):
    import telser_tools.tanabe_sugano as tts
    from telser_tools.tanabe_sugano import determinants
    n, geo = int(case.split("-")[0]), case.split("-")[1]
    B, C, Dq = _LEVELS_BCDQ
    b = tts.all_levels(n, B, C, Dq, geo)
    return dict(keys=sorted([x.label, x.root, x.degeneracy] for x in b), energies=np.array([x.energy for x in b]),
                spectrum=determinants.spectrum(n, B, C, Dq, geo, A=50.0))


@pytest.mark.parametrize("n", range(1, 10))
@pytest.mark.parametrize("geo", ["Oh", "Td"])
def test_all_levels_and_spectrum_match_telser_tools(n, geo):
    ref = R.ref(_telser_all_levels, f"{n}-{geo}")
    B, C, Dq = _LEVELS_BCDQ
    a = ts.all_levels(n, B, C, Dq, geo, units="cm^-1")
    assert sorted([x.label, x.root, x.degeneracy] for x in a) == ref["keys"]
    np.testing.assert_allclose([float(x.energy) for x in a], ref["energies"], atol=1e-9)
    np.testing.assert_allclose(ts.spectrum(n, B, C, Dq, geo, A=50.0, units="cm^-1").numpy(),
                               ref["spectrum"], rtol=1e-12, atol=1e-9)


_DIAGRAMS = [(3, 4.5, "Oh"), (6, 4.8, "Oh"), (8, 4.5, "Oh"), (5, 4.5, "Td")]


@R.oracle("tanabe_sugano", cases=[f"{n}-{c}-{geo}" for n, c, geo in _DIAGRAMS])
def _telser_diagram(case):
    import telser_tools.tanabe_sugano as tts
    n, c, geo = case.split("-")
    b = tts.tanabe_sugano(int(n), float(c), np.linspace(0, 4, 41), geo)
    return dict(ground=list(b.ground), labels=[[cv.label, cv.root] for cv in b.curves],
                e_over_b=np.array([cv.e_over_b for cv in b.curves]), crossover=b.crossover())


@pytest.mark.parametrize("n,c,geo", _DIAGRAMS)
def test_diagram_matches_telser_tools(n, c, geo):
    ref = R.ref(_telser_diagram, f"{n}-{c}-{geo}")
    x = np.linspace(0, 4, 41)
    a = ts.tanabe_sugano(n, c, torch.tensor(x), geo)
    assert list(a.ground) == ref["ground"]
    assert [list(lab) for lab in a.labels] == ref["labels"]
    np.testing.assert_allclose(a.e_over_b.numpy(), ref["e_over_b"], atol=1e-11)
    np.testing.assert_allclose(a.crossover(), ref["crossover"], rtol=1e-12, atol=1e-12)
