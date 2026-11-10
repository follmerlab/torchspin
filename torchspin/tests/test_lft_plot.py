"""Tests of torchspin.lft.plot: units, returned objects, style isolation, and no overlapping text."""

from __future__ import annotations

import numpy as np
import pytest
import torch

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

from torchspin import lft  # noqa: E402
from torchspin.lft import ddnloop, ddnpow, ddnsus, plot  # noqa: E402
from torchspin.lft import tanabe_sugano as ts  # noqa: E402
from torchspin.lft import units as U  # noqa: E402

F64 = torch.float64
CU = dict(zeta=830.0, k=(0.8, 0.8, 0.8), V=torch.diag(torch.tensor([9900.0, -2100.0, -5100.0, -5100.0, 2400.0],
                                                                       dtype=F64)), units="cm^-1")
CO = dict(B=820.0, C=3690.0, zeta=480.0, k=(0.85, 0.85, 0.85), V=lft.crystal_field(950.0, 250.0, 60.0), units="cm^-1")


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


# ------------------------------------------------------------------------------ overlap checker
def _visible_texts(fig, renderer):
    """(text artist, display bbox) of every drawn, non-empty text in the figure."""
    out = []
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            span = hi - lo
            for which in ("major", "minor"):
                locs = axis.get_majorticklocs() if which == "major" else axis.get_minorticklocs()
                labs = axis.get_majorticklabels() if which == "major" else axis.get_minorticklabels()
                for loc, lab in zip(locs, labs):
                    if lo - 1e-9 * span <= loc <= hi + 1e-9 * span and lab.get_visible() and lab.get_text():
                        out.append(lab)
            if axis.label.get_visible() and axis.label.get_text():
                out.append(axis.label)
        for t in (ax.title, ax._left_title, ax._right_title):
            if t.get_text():
                out.append(t)
        out += [t for t in ax.texts if t.get_visible() and t.get_text()]
        leg = ax.get_legend()
        if leg is not None:
            out += [t for t in leg.get_texts() if t.get_text()]
            if leg.get_title().get_text():
                out.append(leg.get_title())
    out += [t for t in fig.texts if t.get_visible() and t.get_text()]
    seen, res = set(), []
    for t in out:
        if id(t) not in seen:
            seen.add(id(t))
            res.append((t, t.get_window_extent(renderer)))
    return res


def collisions(fig):
    """Pairs of overlapping texts, texts (and legends) touching data lines or markers, and texts
    outside the figure.  Empty list: clean."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = _visible_texts(fig, r)
    bad = []
    fb = fig.bbox
    for t, b in texts:
        if b.x0 < fb.x0 - 1 or b.y0 < fb.y0 - 1 or b.x1 > fb.x1 + 1 or b.y1 > fb.y1 + 1:
            bad.append(("outside figure", t.get_text()))
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            (ta, a), (tb, b) = texts[i], texts[j]
            if a.shrunk(0.98, 0.9).overlaps(b.shrunk(0.98, 0.9)):
                bad.append(("text/text", ta.get_text(), tb.get_text()))
    for ax in fig.axes:
        lines = [ln for ln in ax.lines if ln.get_visible()]
        boxes = [(t.get_text(), b) for t, b in texts]
        leg = ax.get_legend()
        if leg is not None:
            boxes.append(("<legend>", leg.get_window_extent(r)))
        for name, b in boxes:
            if plot._hits(b.padded(-1.5), lines, ax.bbox):
                bad.append(("text/line", name))
    return bad


# ------------------------------------------------------------------------------ fixtures
def _cr_levels():
    return ts.all_levels(3, 918.0, 3850.0, 1740.0, units="cm^-1")


def _cu_zeeman():
    return ddnloop.field_scan(9, torch.linspace(0.0, 450.0, 451, dtype=F64), theta=0.0, pairs=[(0, 1)], **CU)


def _cu_angles(field=310.0):
    th = torch.linspace(0.0, np.pi / 2, 91, dtype=F64)
    return ddnloop.angle_scan(9, field, th, 0.0, pairs=[(0, 1)], **CU)


# ------------------------------------------------------------------------------ units and returned objects
def test_level_diagram_default_units_and_round_trip():
    E = lft.ddn.levels(3, B=918.0, C=3850.0, V=lft.crystal_field(Dq=1740.0), zeta=200.0)       # MHz (default)
    fig, ax = plot.levels(E)
    assert isinstance(fig, mpl.figure.Figure) and isinstance(ax, mpl.axes.Axes)
    assert "cm" in ax.get_ylabel() and "-1" in ax.get_ylabel()
    y_cm = np.array([ln.get_ydata()[0] for ln in ax.lines])
    rel = (E - E[0]).numpy()
    # every bar is at a source energy (each state within 1e-3 cm^-1 of its level), and back to MHz it is the source
    assert len(y_cm) == len(rel)
    np.testing.assert_allclose(np.sort(y_cm) * U.energy_factor("cm^-1", "MHz"), rel, rtol=0, atol=1e-3 * 3e4)
    fig2, ax2 = plot.levels(E, units="MHz")
    assert "MHz" in ax2.get_ylabel()
    y_mhz = np.array([ln.get_ydata()[0] for ln in ax2.lines])
    np.testing.assert_allclose(y_mhz, y_cm * U.energy_factor("cm^-1", "MHz"), rtol=1e-12)
    fig3, ax3 = plot.levels(E, units="K")
    np.testing.assert_allclose(np.array([ln.get_ydata()[0] for ln in ax3.lines]) * U.energy_factor("K", "MHz"), y_mhz,
                               rtol=1e-12)


def test_level_diagram_bars_count_degeneracy():
    fig, ax = plot.levels(_cr_levels(), from_units="cm^-1", labels=None)
    bars = [ln for ln in ax.lines if ln.get_transform() is not ax.transAxes]       # leader lines are in axes coords
    assert len(bars) == 120          # C(10, 3) states, one bar each (no level above max_bars)


@pytest.mark.parametrize("units", ["cm^-1", "MHz", "GHz", "K", "eV"])
def test_tanabe_sugano_units_round_trip(units):
    d = ts.tanabe_sugano(3, B=918.0, units="cm^-1")
    fig, ax = plot.tanabe_sugano(d, units=units)
    curves = [ln for ln in ax.lines if len(ln.get_xdata()) == len(d.dq_over_b)]
    assert len(curves) == len(d.labels)
    f = U.energy_factor("cm^-1", units)
    np.testing.assert_allclose(curves[0].get_xdata(), d.dq.numpy() * f, rtol=1e-12)
    np.testing.assert_allclose(np.stack([c.get_ydata() for c in curves]), d.energy.numpy() * f, rtol=1e-12,
                               atol=1e-12 * f)
    assert U.energy_label(units) in ax.get_xlabel() or (units == "cm^-1" and "cm" in ax.get_xlabel())


def test_tanabe_sugano_reduced_axes_default():
    d = ts.tanabe_sugano(3)
    fig, ax = plot.tanabe_sugano(d)
    assert ax.get_xlabel() == "Dq/B" and ax.get_ylabel() == "E/B"
    curves = [ln for ln in ax.lines if len(ln.get_xdata()) == len(d.dq_over_b)]
    np.testing.assert_allclose(np.stack([c.get_ydata() for c in curves]), d.e_over_b.numpy(), rtol=0, atol=0)


def test_zeeman_units_round_trip_and_resonance():
    r = _cu_zeeman()
    fig, ax = plot.zeeman(r, mw_freq=9.5)
    assert "cm" in ax.get_ylabel() and "mT" in ax.get_xlabel()
    lv = [ln for ln in ax.lines if len(ln.get_xdata()) == 451]
    assert len(lv) == 2                                        # the ground Kramers doublet only
    off = float(r.energies[0, 0])
    np.testing.assert_allclose(lv[0].get_ydata(), (r.energies[:, 0] - off).numpy(), atol=1e-12)
    fig2, ax2 = plot.zeeman(r, units="MHz", field_units="G")
    lv2 = [ln for ln in ax2.lines if len(ln.get_xdata()) == 451]
    np.testing.assert_allclose(lv2[1].get_ydata() * U.energy_factor("MHz", "cm^-1"),
                               (r.energies[:, 1] - off).numpy(), rtol=1e-12, atol=1e-9)   # |E| ~ 1e4 cm^-1 cancels
    np.testing.assert_allclose(lv2[0].get_xdata(), r.fields.numpy() * 10.0)
    # the resonance tick sits where g_obs mu_B B = h nu
    sec = ax.child_axes[0]
    locs = sec.get_xticks()
    assert len(locs) == 1
    g = float(r.g_obs[300, 0])
    hv = 9.5e3 * U.energy_factor("MHz", "cm^-1")
    assert locs[0] == pytest.approx(hv / (g * lft.ddn.TORCHSPIN_CONSTANTS.mu_b), rel=2e-3)


def test_angles_and_roadmap_render():
    fig, ax = plot.angles(_cu_angles(), mw_freq=9.5)
    assert "θ" in ax.get_xlabel() and "cm" in ax.get_ylabel()
    assert len(ax.child_axes[0].get_xticks()) == 1
    th = torch.linspace(0.0, np.pi / 2, 31, dtype=F64)
    scan = ddnloop.scan(9, fields=torch.linspace(270.0, 340.0, 141, dtype=F64), theta=th, phi=0.0, mesh=True,
                        pairs=[(0, 1)], **CU)
    fig, ax = plot.roadmap(scan, 9.5)
    ys = ax.lines[0].get_ydata()
    assert len(ys) == 31 and np.all((ys > 270) & (ys < 340))
    assert ys[0] < ys[-1]                                       # g_par > g_perp: lowest field along z


def test_susceptibility_units_and_panels():
    T = torch.linspace(2.0, 300.0, 40, dtype=F64)
    chiT = ddnsus.chi_T(7, T, grid=lft.grids.gauss_hemisphere(3), **CO)                 # SI m^3 K/mol
    fig, ax = plot.susceptibility(T, chiT)
    assert isinstance(ax, mpl.axes.Axes) and "cm" in ax.get_ylabel()
    np.testing.assert_allclose(ax.lines[0].get_ydata(), chiT.powder[0].numpy() * U.chi_factor("SI", "cgs"))
    fig, ax = plot.susceptibility(T, chiT, chi_units="SI", quantity="chi")
    np.testing.assert_allclose(ax.lines[0].get_ydata(), chiT.powder[0].numpy() / T.numpy())
    fig, ax = plot.susceptibility(T, chiT, quantity="mu_eff")
    np.testing.assert_allclose(ax.lines[0].get_ydata(), ddnsus.mu_eff(chiT.powder[0]).numpy(), rtol=1e-10)
    y = chiT.powder[0].numpy() * U.chi_factor("SI", "cgs")
    fig, axs = plot.susceptibility(T, chiT, data=(T.numpy()[::4], y[::4] * 1.01))
    assert isinstance(axs, list) and len(axs) == 2
    np.testing.assert_allclose(axs[1].lines[-1].get_ydata(), 0.01 * y[::4], rtol=1e-9)


def test_powder_and_magnetization_render():
    B = torch.linspace(250.0, 360.0, 221, dtype=F64)
    p = ddnpow.simulate(9, B, 9.5, width=(0.004, 0.004, 0.006), igloo=6, derivative=True, **CU)
    fig, ax = plot.powder(p, mw_freq=9.5, field_units="G")
    np.testing.assert_allclose(ax.lines[0].get_xdata(), B.numpy() * 10.0)
    y = p.spectrum.numpy()
    np.testing.assert_allclose(ax.lines[0].get_ydata(), y / np.abs(y).max())
    M = lft.ddnmag.magnetization(7, torch.linspace(1.0, 7000.0, 30, dtype=F64), [2.0, 5.0], **CO)
    fig, ax = plot.magnetization(torch.linspace(1.0, 7000.0, 30, dtype=F64), M, [2.0, 5.0])
    np.testing.assert_allclose(ax.lines[0].get_xdata(), np.linspace(1.0, 7000.0, 30) * 1e-3)
    fig, ax = plot.magnetization(torch.linspace(1.0, 7000.0, 30, dtype=F64), M, [2.0, 5.0], x="B/T")
    np.testing.assert_allclose(ax.lines[1].get_xdata(), np.linspace(1.0, 7000.0, 30) * 1e-3 / 5.0)


def test_returns_given_axes():
    fig0, ax0 = plt.subplots()
    fig, ax = plot.levels(_cr_levels(), from_units="cm^-1", ax=ax0)
    assert fig is fig0 and ax is ax0


# ------------------------------------------------------------------------------ style isolation
def test_no_global_rcparams_change_and_font():
    before = dict(mpl.rcParams)
    figs = [plot.levels(_cr_levels(), from_units="cm^-1")[0], plot.tanabe_sugano(ts.tanabe_sugano(6))[0],
            plot.zeeman(_cu_zeeman(), mw_freq=9.5)[0]]
    after = dict(mpl.rcParams)
    assert {k for k in before if before[k] != after[k]} == set()
    for fig in figs:
        fig.canvas.draw()                                   # drawn outside the rc context
        for ax in fig.axes:
            names = {t.get_fontname() for t in ax.get_yticklabels() + [ax.yaxis.label]}
            assert names <= {"Helvetica", "Arial", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"}
            assert ax.yaxis.label.get_fontweight() in ("bold", 700)
            for sp in ax.spines.values():
                assert sp.get_linewidth() == pytest.approx(1.2)
            assert not any(gl.get_visible() for gl in ax.get_xgridlines() + ax.get_ygridlines())


def test_powder_overlay_and_rf_axis_for_ddpowh():
    """lft.ddpowh results: a strain on/off overlay (relative heights kept) and an rf (ENDOR/NMR) axis, clean."""
    from torchspin.lft import ddpowh, telser2torchspin as t2t

    sy = t2t.spin2sys(s=1.5, g=(2.0, 2.0, 2.0), D=0.15, E=0.01)
    f = np.linspace(50.0, 550.0, 126)
    kw = dict(width=150.0, igloo=12, extent="octant")
    plain = ddpowh.epr(sy, f, 9.5, **kw)
    strained = ddpowh.epr(sy, f, 9.5, strain=ddpowh.Strain(ddpowh.strain_weights(sy, ms=[3.0, 0.0, 0.0, 3.0])), **kw)
    fig, ax = plot.powder(strained, mw_freq=9.5, label="strain on mS = ±3/2", overlay=[(plain, "no strain")])
    ymax = np.abs(strained.spectrum.numpy()).max()
    np.testing.assert_allclose(ax.lines[0].get_ydata(), plain.spectrum.numpy() / ymax)
    assert ax.get_legend() is not None and collisions(fig) == []
    plt.close(fig)
    from torchspin import SpinSystem

    nq = ddpowh.nmr(SpinSystem(S=[], Nucs="14N", Q=[-0.6, -0.4, 1.0]), np.linspace(0.0, 2.0, 201), 0.0,
                    nmr_width=0.02, igloo=6)
    fig, ax = plot.powder(nq, kind="rf")
    assert ax.get_xlabel() == "Frequency (MHz)" and collisions(fig) == []
    plt.close(fig)


# ------------------------------------------------------------------------------ overlap checks
def test_spread_keeps_spacing():
    y = np.array([0.5, 0.5, 0.51, 0.2, 0.9, 0.0])
    p = plot._spread(y, 0.05, 0.0, 1.0)
    q = np.sort(p)
    assert np.all(np.diff(q) >= 0.05 - 1e-12) and q[0] >= 0 and q[-1] <= 1.0
    assert np.all(np.argsort(p, kind="stable") == np.argsort(y, kind="stable"))     # order kept


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7, 8])
def test_tanabe_sugano_labels_do_not_overlap(n):
    fig, ax = plot.tanabe_sugano(ts.tanabe_sugano(n))
    assert len(ax.texts) >= 3
    assert collisions(fig) == []


def test_tanabe_sugano_td_and_absolute_no_overlap():
    fig, _ = plot.tanabe_sugano(ts.tanabe_sugano(7, geometry="Td", B=780.0, units="cm^-1"))
    assert collisions(fig) == []


def test_tanabe_sugano_parent_terms():
    fig, ax = plot.tanabe_sugano(ts.tanabe_sugano(3))
    labs = [t.get_text() for t in ax.texts]
    assert any("(F)" in s for s in labs) and any("(G)" in s for s in labs)
    assert sum("T$_{\\mathregular{1g}}$(H)" in s for s in labs) == 2          # 2H -> 2 T1; 2P's T1 is above the plot
    assert not any("(P)" in s and "^{2}" in s for s in labs)


def test_label_heavy_plots_no_overlap():
    E = lft.ddn.levels(7, units="cm^-1", **{k: v for k, v in CO.items() if k not in ("k", "units")})
    figs = [plot.levels(_cr_levels(), from_units="cm^-1")[0],
            plot.levels(E, from_units="cm^-1", labels="energy", emax=10000.0)[0],
            plot.zeeman(_cu_zeeman(), mw_freq=9.5)[0],
            plot.angles(_cu_angles(), mw_freq=9.5)[0]]
    T = np.linspace(2.0, 300.0, 60)
    chiT = ddnsus.chi_T(7, T, grid=lft.grids.gauss_hemisphere(3), chi_units="cgs", **CO).powder[0].numpy()
    rng = np.random.default_rng(0)
    figs.append(plot.susceptibility(T, chiT, model_chi_units="cgs",
                                    data=(T[::3], chiT[::3] + 0.02 * rng.standard_normal(20)))[0])
    for fig in figs:
        assert collisions(fig) == []


def test_level_labels_dropped_rather_than_overlapping():
    E = torch.linspace(0.0, 1.0, 200, dtype=F64)
    fig, ax = plot.levels(E, from_units="cm^-1", labels="energy", tol=1e-6, figsize=(3.0, 3.0))
    assert 0 < len(ax.texts) < 200
    assert collisions(fig) == []


def test_legend_goes_to_free_space():
    x = np.linspace(0, 1, 200)
    fig, ax = plt.subplots()
    ax.plot(x, x, "k-")
    ax.plot(x, 1 - x, "k-")                             # an X: every corner has data near it
    ax.plot([0, 1], [0.5, 0.5], "k-")
    from matplotlib.lines import Line2D
    leg = plot._place_legend(ax, [Line2D([], [], color="k")] * 3, ["one curve label", "two", "three"], owned=False)
    assert leg is not None
    assert collisions(fig) == []


def test_collision_checker_flags_overlaps():
    """Negative control: the checker sees text on text and text on a line."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0.5, 0.5], "k-")
    ax.text(0.5, 0.5, "on the line")
    ax.text(0.1, 0.8, "first")
    ax.text(0.11, 0.8, "second")
    kinds = {b[0] for b in collisions(fig)}
    assert {"text/text", "text/line"} <= kinds


def test_user_axes_get_style():
    fig0, ax0 = plt.subplots()
    plot.zeeman(_cu_zeeman(), ax=ax0)
    assert all(sp.get_linewidth() == pytest.approx(1.2) for sp in ax0.spines.values())
    assert ax0.xaxis.label.get_fontweight() in ("bold", 700)


# ------------------------------------------------------------------------------ Tanabe-Sugano parent terms
_KNOWN_TERMS = {   # free-ion terms (2S+1, L letter) of d2 and d3
    2: sorted([(3, "F"), (3, "P"), (1, "G"), (1, "D"), (1, "S")]),
    3: sorted([(4, "F"), (4, "P"), (2, "H"), (2, "G"), (2, "F"), (2, "D"), (2, "D"), (2, "P")]),
}


def test_branching_rules():
    assert plot.branching(0) == {"A1": 1} and plot.branching(1) == {"T1": 1}
    assert plot.branching(2) == {"E": 1, "T2": 1} and plot.branching(3) == {"A2": 1, "T1": 1, "T2": 1}
    assert plot.branching(4) == {"A1": 1, "E": 1, "T1": 1, "T2": 1}
    assert plot.branching(5) == {"E": 1, "T1": 2, "T2": 1}
    assert plot.branching(6) == {"A1": 1, "A2": 1, "E": 1, "T1": 1, "T2": 2}
    for L in range(9):
        assert sum(k * {"A1": 1, "A2": 1, "E": 2, "T1": 3, "T2": 3}[g] for g, k in plot.branching(L).items()) == 2 * L + 1


@pytest.mark.parametrize("c_over_b", [4.5, 3.8])
def test_free_ion_terms_known(c_over_b):
    for n, ref in _KNOWN_TERMS.items():
        assert sorted((m, "SPDFGHIK"[L]) for _, m, L in plot._free_ion_terms(n, c_over_b)) == ref


@pytest.mark.parametrize("geometry", ["Oh", "Td"])
@pytest.mark.parametrize("c_over_b", [4.5, 3.8])
@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7, 8])
def test_parent_labels_follow_branching_rules(n, c_over_b, geometry):
    """Per (2S+1, Gamma): the multiset of parent letters equals sum over free-ion terms of that spin of
    (number of Gamma in L) x L, and every curve gets a parent."""
    from collections import Counter

    par = plot.parent_terms(n, c_over_b, geometry)
    blocks = ts.blocks_for(n)
    assert set(par) == {(b.label, r) for b in blocks for r in range(b.size)}
    terms = plot._free_ion_terms(n, c_over_b)
    for b in blocks:
        got = Counter(par[(b.label, r)] for r in range(b.size))
        want = Counter()
        for _, m, L in terms:
            if m == b.multiplicity:
                want["SPDFGHIK"[L]] += plot.branching(L).get(b.label[1:], 0)
        assert got == +want, (b.label, got, want)
    if n in (3, 7):                       # the 2P/2H degeneracy: exactly one 2T1 from P, two from H
        c = Counter(par[("2T1", r)] for r in range(ts.get_block(n, "2T1").size))
        assert c["P"] == 1 and c["H"] == 2
