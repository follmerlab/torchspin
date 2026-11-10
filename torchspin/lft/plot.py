"""Plots of ligand-field results: level, Tanabe-Sugano, Zeeman, angle and road-map diagrams,
magnetometry (chi T, chi, mu_eff vs T; M vs B or B/T) with fits, and powder spectra.

Every function returns ``(fig, ax)`` as :func:`torchspin.levelsplot.levelsplot` does (``ax`` is a
list ``[main, residuals]`` when a residual panel is drawn) and takes an optional ``ax`` to draw
into.  matplotlib (torchspin's optional ``plot`` extra) is imported inside the functions.

Units
-----
Ligand-field energy axes default to ``units='cm^-1'`` (:data:`torchspin.lft.units.PLOT_ENERGY_UNITS`)
and accept ``'MHz'``, ``'GHz'``, ``'K'``, ``'eV'`` (any unit of :mod:`torchspin.lft.units`).  The
unit of the data is read from the result object where it carries one (``LoopResult.units``,
``TanabeSuganoDiagram.units``, ``dsoxf.Solution.units``) and is ``from_units`` (default
``'MHz'``, the lft default) otherwise.  Field axes are in mT (``field_units='G'`` or ``'T'`` on
request); magnetization plots default to tesla.  Susceptibility is plotted in ``chi_units``
(``'cgs'``: cm^3/mol, default; ``'SI'``: m^3/mol).

Style
-----
Every plot is drawn inside a local :func:`matplotlib.rc_context` (the global rcParams are never
changed): Helvetica or Arial (sans-serif fallback), bold axis labels and titles, solid dark
spines of 1.2 pt, no grid.  No text is drawn on top of lines, markers or other text: level and
Tanabe-Sugano labels are spread apart with leader lines (labels that cannot be placed without
overlap are left out), resonance positions are tick labels on a secondary axis, and legends go
into a corner checked to be free of data, or outside the axes.

    >>> from torchspin import lft
    >>> from torchspin.lft import plot
    >>> fig, ax = plot.tanabe_sugano(lft.tanabe_sugano.tanabe_sugano(3))
    >>> res = lft.ddnloop.field_scan(9, torch.linspace(0, 450, 451), theta=0.0, units='cm^-1', zeta=830.0,
    ...                              k=(0.8,) * 3, V=V_cu, pairs=[(0, 1)])
    >>> fig, ax = plot.zeeman(res, mw_freq=9.5)
"""

from __future__ import annotations

import functools
import math
import re
from typing import Sequence

import numpy as np
import torch

from . import units as _u

__all__ = ["levels", "tanabe_sugano", "zeeman", "angles", "roadmap", "susceptibility", "magnetization", "powder",
           "fit", "STYLE", "term_label", "parent_terms", "branching"]

#: Line color of levelsplot (MATLAB blue).
_LEVEL_COLOR = (0.0, 0.447, 0.741)
#: Okabe-Ito colors (colorblind safe); black first.
_PALETTE = ("#000000", "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442")
_DARK = "#1a1a1a"

#: rcParams applied inside every plot (``font.family`` is resolved to Helvetica or Arial at call time).
STYLE = {
    "font.size": 10.0,
    "axes.linewidth": 1.2,
    "axes.edgecolor": _DARK,
    "axes.labelcolor": _DARK,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.titlesize": 11.0,
    "axes.labelsize": 10.5,
    "axes.grid": False,
    "axes.unicode_minus": True,
    "xtick.color": _DARK,
    "ytick.color": _DARK,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.minor.width": 0.9,
    "ytick.minor.width": 0.9,
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.3,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "mathtext.default": "regular",
}

_MU_B = 0.00046686447719298217        # cm^-1 / mT (lft.ddn.TORCHSPIN_CONSTANTS.mu_b), for state selection only


# ------------------------------------------------------------------------------ style and figure helpers
def _plt():
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("torchspin.lft.plot needs matplotlib: pip install 'torchspin[plot]'") from exc
    return matplotlib, plt


@functools.lru_cache(maxsize=1)
def _font() -> str:
    """Helvetica, else Arial, else the generic 'sans-serif'.  A concrete name is stored in each text's
    FontProperties, so the font survives a later savefig outside the rc context."""
    from matplotlib import font_manager

    for name in ("Helvetica", "Arial", "Liberation Sans", "Nimbus Sans"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except ValueError:
            continue
    return "sans-serif"


def _styled(func):
    """Run ``func`` inside a local rc_context with :data:`STYLE` (global rcParams untouched)."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        mpl, _ = _plt()
        with mpl.rc_context(dict(STYLE, **{"font.family": [_font(), "DejaVu Sans"]})):
            return func(*args, **kwargs)

    return wrapper


def _style_axes(ax) -> None:
    """Apply the spine / tick / font style to an axes (also one created outside the rc context)."""
    for sp in ax.spines.values():
        sp.set_linewidth(STYLE["axes.linewidth"])
        sp.set_edgecolor(_DARK)
    ax.tick_params(which="major", width=1.2, length=4.0, color=_DARK, labelcolor=_DARK)
    ax.tick_params(which="minor", width=0.9, color=_DARK)
    try:
        ax.tick_params(labelfontfamily=_font())            # matplotlib >= 3.8
    except (ValueError, TypeError, AttributeError):  # pragma: no cover - old matplotlib
        pass
    ax.grid(False)


def _figure(figsize, margins):
    """New figure with one axes at fixed margins (left, right, bottom, top) in inches.

    Fixed margins (no constrained layout) keep the axes size known while labels are placed.
    """
    _, plt = _plt()
    fig = plt.figure(figsize=figsize)
    w, h = figsize
    l, r, b, t = margins
    return fig, fig.add_axes((l / w, b / h, 1 - (l + r) / w, 1 - (b + t) / h))


def _axes(ax, figsize, margins):
    if ax is None:
        fig, ax = _figure(figsize, margins)
        owned = True
    else:
        fig, owned = ax.figure, False
    _style_axes(ax)
    return fig, ax, owned


def _xlabel(ax, s):
    ax.set_xlabel(s, fontweight="bold")


def _ylabel(ax, s):
    ax.set_ylabel(s, fontweight="bold")


def _title(ax, s, **kw):
    if s:
        ax.set_title(s, fontweight="bold", **kw)


def _renderer(fig):
    fig.canvas.draw()
    return fig.canvas.get_renderer()


def _np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


# ------------------------------------------------------------------------------ labels and units
def _energy_axis_label(units: str) -> str:
    u = _u.canonical_energy_unit(units)
    return "cm$^{-1}$" if u == "cm^-1" else _u.energy_label(u)


def _field_axis_label(field_units: str) -> str:
    if field_units not in _u.FIELD_UNITS:
        raise ValueError(f"unknown field unit {field_units!r}; use one of {', '.join(_u.FIELD_UNITS)}")
    return field_units


_TERM = re.compile(r"^(\d+)([AETS-Z])(\d?)([gu]?)$")


def term_label(label: str, parity: str | None = None, parent: str | None = None) -> str:
    """Mathtext for a term / irrep label: ``'4T1'`` -> ^4T_1 (``parity='g'``: ^4T_1g); ``parent``
    is appended in parentheses, e.g. ^4T_1g(P).  Labels of another form are returned unchanged."""
    m = _TERM.match(label.replace(" ", ""))
    if not m:
        return label
    mult, irrep, idx, par = m.groups()
    sub = idx + (par or (parity or ""))
    s = f"$^{{{mult}}}${irrep}" + (f"$_{{\\mathregular{{{sub}}}}}$" if sub else "")
    return s + (f"({parent})" if parent else "")


def _fmt_values(values) -> list[str]:
    """Short labels for a set of energies: as many decimals as the spread needs."""
    v = np.asarray(values, float)
    if v.size == 0:
        return []
    spread = float(np.max(np.abs(v))) if v.size else 1.0
    if spread == 0.0:
        dec = 0
    else:
        dec = int(min(6, max(0, 2 - math.floor(math.log10(spread)))))
    return [f"{x:.{dec}f}" for x in v]


def _spread(y: np.ndarray, s: float, lo: float, hi: float) -> np.ndarray:
    """Positions closest (least squares) to targets ``y`` with spacing >= ``s`` inside [lo, hi].

    Pool-adjacent-violators on z_i = y_(i) - i s (sorted order); needs n s <= hi - lo + s.
    """
    y = np.asarray(y, float)
    n = len(y)
    if n == 0:
        return y
    order = np.argsort(y, kind="stable")
    z = y[order] - s * np.arange(n)
    vals, wts, cnt = [], [], []
    for zi in z:
        vals.append(zi), wts.append(1.0), cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnt.pop()
            v1, w1, c1 = vals.pop(), wts.pop(), cnt.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2)), wts.append(w1 + w2), cnt.append(c1 + c2)
    q = np.repeat(vals, cnt)
    q = np.clip(q, lo, max(lo, hi - s * (n - 1)))
    out = np.empty(n)
    out[order] = q + s * np.arange(n)
    return out


def _place_labels(ax, y_targets, texts, *, x_anchor, x_text, colors=None, fontsize=8.5, priority=None,
                  lo=0.0, hi=1.0, leader_kw=None, outside=False):
    """Labels at axes-x ``x_text`` near axes-y targets, spread so they do not touch, with leader lines
    from (x_anchor, target).  Targets and positions are axes fractions.  If not all fit, the labels
    with the lowest ``priority`` value are kept.  Returns the list of Text artists."""
    y_targets = np.asarray(y_targets, float)
    n = len(y_targets)
    if n == 0:
        return []
    h_px = ax.get_window_extent().height
    if h_px <= 0:
        return []
    s = 1.30 * fontsize / 72.0 * ax.figure.dpi / h_px          # line pitch in axes fraction
    half = 0.55 * fontsize / 72.0 * ax.figure.dpi / h_px
    lo, hi = lo + half, hi - half
    nfit = int(math.floor((hi - lo) / s)) + 1 if hi > lo else 0
    keep = np.arange(n)
    if n > nfit:
        pr = np.arange(n) if priority is None else np.asarray(priority)
        keep = np.sort(np.argsort(pr, kind="stable")[:max(nfit, 0)])
    if len(keep) == 0:
        return []
    pos = _spread(y_targets[keep], s, lo, hi)
    lk = dict(color="0.45", lw=0.6, clip_on=False)
    lk.update(leader_kw or {})
    out = []
    for k, yp in zip(keep, pos):
        c = _DARK if colors is None else colors[k]
        yt = y_targets[k]
        ax.plot([x_anchor, x_text - 0.008, ], [yt, yp], transform=ax.transAxes, **lk)
        out.append(ax.text(x_text, yp, texts[k], transform=ax.transAxes, ha="left", va="center",
                           fontsize=fontsize, color=c, clip_on=False))
    return out


# ------------------------------------------------------------------------------ collision helpers
def _line_paths(line):
    """Display-coordinate polylines (NaN-split) and marker radius (px) of a Line2D."""
    xy = line.get_transform().transform(np.column_stack([_np(line.get_xdata()), _np(line.get_ydata())]))
    ok = np.isfinite(xy).all(1)
    pieces, cur = [], []
    for p, good in zip(xy, ok):
        if good:
            cur.append(p)
        elif cur:
            pieces.append(np.array(cur)), cur.clear()
    if cur:
        pieces.append(np.array(cur))
    ls = line.get_linestyle()
    has_line = ls not in ("None", "none", "", " ") and line.get_linewidth() > 0
    mk = line.get_marker()
    has_marker = mk not in (None, "None", "none", "", " ")
    r = 0.5 * line.get_markersize() * line.figure.dpi / 72.0 if has_marker else 0.0
    return pieces, has_line, r


def _hits(bbox, artists_lines, clip_box=None) -> bool:
    """Does the display bbox touch any of the lines (segments or markers)?"""
    from matplotlib.path import Path

    for line in artists_lines:
        b = bbox
        if line.get_clip_on() and clip_box is not None:
            from matplotlib.transforms import Bbox
            x0, y0 = max(b.x0, clip_box.x0), max(b.y0, clip_box.y0)
            x1, y1 = min(b.x1, clip_box.x1), min(b.y1, clip_box.y1)
            if x0 >= x1 or y0 >= y1:
                continue
            b = Bbox([[x0, y0], [x1, y1]])
        pieces, has_line, r = _line_paths(line)
        for pc in pieces:
            if has_line and len(pc) > 1 and Path(pc).intersects_bbox(b, filled=False):
                return True
            if r > 0 or (has_line and len(pc) == 1):
                inside = ((pc[:, 0] >= b.x0 - r) & (pc[:, 0] <= b.x1 + r) & (pc[:, 1] >= b.y0 - r)
                          & (pc[:, 1] <= b.y1 + r))
                if inside.any():
                    return True
    return False


def _place_legend(ax, handles, labels, *, owned: bool, title=None, ncol=1, extra_axes=()):
    """Legend in the first corner free of data and text, else outside on the right (the axes are
    narrowed to make room when this module created the figure)."""
    fig = ax.figure
    kw = dict(handles=handles, labels=labels, title=title, ncol=ncol, fontsize=STYLE["legend.fontsize"],
              title_fontsize=STYLE["legend.fontsize"], frameon=False, handlelength=1.8, borderaxespad=0.6)
    r = _renderer(fig)
    lines = list(ax.lines)
    others = [t.get_window_extent(r) for t in ax.texts if t.get_visible() and t.get_text()]
    for loc in ("upper right", "upper left", "lower right", "lower left", "center right", "center left",
                "upper center", "lower center"):
        leg = ax.legend(loc=loc, **kw)
        _bold_legend_title(leg)
        bb = leg.get_window_extent(r).padded(3)
        if not _hits(bb, lines, ax.bbox) and not any(bb.overlaps(o) for o in others):
            return leg
        leg.remove()
    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
                    **{k: v for k, v in kw.items() if k != "borderaxespad"})
    _bold_legend_title(leg)
    if owned:
        w = leg.get_window_extent(_renderer(fig)).width / fig.dpi / fig.get_figwidth() + 0.03
        for a in (ax, *extra_axes):
            p = a.get_position()
            a.set_position((p.x0, p.y0, max(0.2, p.width - w), p.height))
    return leg


def _bold_legend_title(leg):
    if leg is not None and leg.get_title() is not None:
        leg.get_title().set_fontweight("bold")
        leg.get_title().set_fontfamily(_font())
    for t in leg.get_texts():
        t.set_fontfamily(_font())


def _prune_ticks(fig, axis, priority):
    """Drop tick labels (lowest priority first) of a (secondary) axis until none overlap."""
    locs = list(axis.get_ticklocs())
    if len(locs) < 2:
        return
    r = _renderer(fig)
    labs = axis.get_ticklabels()
    boxes = [lab.get_window_extent(r).padded(2) for lab in labs]
    order = np.argsort(-np.asarray(priority, float), kind="stable")
    kept = []
    for i in order:
        if all(not boxes[i].overlaps(boxes[j]) for j in kept):
            kept.append(i)
    kept.sort()
    texts = [lab.get_text() for lab in labs]
    axis.set_ticks([locs[i] for i in kept], [texts[i] for i in kept])


# ------------------------------------------------------------------------------ level diagram
def _group(e_cm: np.ndarray, tol: float) -> list[list[int]]:
    groups: list[list[int]] = []
    for i in np.argsort(e_cm, kind="stable"):
        if groups and e_cm[i] - e_cm[groups[-1][-1]] <= tol:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    return groups


@_styled
def levels(energies, *, labels=None, from_units: str | None = None, units: str = _u.PLOT_ENERGY_UNITS,
           relative: bool = True, tol: float = 1e-3, emax=None, parity: str | None = None, max_bars: int = 16,
           title: str | None = None, fontsize: float = 8.5, ax=None, figsize=(3.8, 4.8)):
    """Energy-level diagram: each level a group of short horizontal bars, one bar per state.

    ``energies``: a 1-D tensor / array of state energies (``lft.ddn.levels``, ``lft.dpn.levels``,
    ``DsoxfSystem.levels``) in ``from_units`` (default ``'MHz'``); a :class:`torchspin.lft.dsoxf.Solution`
    (its own units); or a list of :class:`torchspin.lft.tanabe_sugano.CubicLevel` (term labels and
    degeneracies come with them).  States within ``tol`` (cm^-1) form one level.
    ``labels``: None, ``'energy'`` (each level's energy), or one string per level (or per state; the
    first state of a level names it); term labels such as ``'4T2'`` are typeset (^4T_2, with the
    ``parity`` subscript 'g' or 'u' if given).  Labels sit right of the bars, spread apart with
    leader lines; those that cannot be placed without overlap are left out (the lowest levels are kept).
    ``units``: energy axis (default cm^-1).  ``relative``: ground level at 0.  ``emax`` (in ``units``):
    upper limit of the axis.  A level of more than ``max_bars`` states is one bar, its degeneracy in
    the label.
    """
    from .tanabe_sugano import CubicLevel

    units = _u.canonical_energy_unit(units)
    lab_list = None
    if isinstance(energies, (list, tuple)) and energies and isinstance(energies[0], CubicLevel):
        src = _u.canonical_energy_unit(from_units or "MHz")
        e = np.concatenate([np.full(lv.degeneracy, float(_np(lv.energy))) for lv in energies])
        lab_list = [lv.label for lv in energies for _ in range(lv.degeneracy)]
        if labels is None:
            labels = lab_list
    elif hasattr(energies, "energies") and hasattr(energies, "units"):
        src = energies.units
        e = _np(energies.energies).reshape(-1)
    else:
        src = _u.canonical_energy_unit(from_units or "MHz")
        e = _np(energies).reshape(-1)
    if e.ndim != 1 or e.size == 0:
        raise ValueError("levels() needs a 1-D set of state energies")
    e_cm = e * _u.energy_factor(src, "cm^-1")
    groups = _group(e_cm, tol)
    e_plot = e * _u.energy_factor(src, units)
    lev = np.array([e_plot[g].mean() for g in groups])
    deg = [len(g) for g in groups]
    if relative:
        lev = lev - lev[0]
    if isinstance(labels, str) and labels == "energy":
        texts = _fmt_values(lev)
    elif labels is None:
        texts = None
    else:
        labels = list(labels)
        if len(labels) == len(e):
            labels = [labels[g[0]] for g in groups]
        if len(labels) != len(groups):
            raise ValueError(f"labels: need {len(groups)} (one per level) or {len(e)} (one per state), got {len(labels)}")
        texts = [term_label(str(s), parity) for s in labels]
    if texts is not None:
        texts = [t + (f" (×{d})" if d > max_bars else "") for t, d in zip(texts, deg)]

    fig, ax, owned = _axes(ax, figsize, (0.85, 0.15, 0.25, 0.45 if title else 0.2))
    top = float(lev.max()) if emax is None else float(emax)
    bot = float(lev.min())
    span = max(top - bot, 1e-12 if top == bot else 0.0) or 1.0
    ax.set_ylim(bot - 0.05 * span, top + 0.05 * span)
    ax.set_xlim(0.0, 1.0)
    x0, x1 = 0.06, (0.44 if texts is not None else 0.94)
    shown = [i for i in range(len(lev)) if lev[i] <= top + 1e-9 * span]
    for i in shown:
        nb = deg[i] if deg[i] <= max_bars else 1
        w = (x1 - x0) / nb
        gap = 0.18 * w if nb > 1 else 0.0
        for b in range(nb):
            ax.plot([x0 + b * w + gap / 2, x0 + (b + 1) * w - gap / 2], [lev[i], lev[i]], color=_LEVEL_COLOR,
                    lw=1.8, solid_capstyle="butt")
    ax.set_xticks([])
    _ylabel(ax, f"Energy ({_energy_axis_label(units)})")
    _title(ax, title)
    if texts is not None and shown:
        to_ax = ax.transAxes.inverted()
        ys = [to_ax.transform(ax.transData.transform((0.5, lev[i])))[1] for i in shown]
        _place_labels(ax, ys, [texts[i] for i in shown], x_anchor=x1 + 0.02, x_text=0.56, fontsize=fontsize,
                      priority=np.arange(len(shown)))
    return fig, ax


# ------------------------------------------------------------------------------ Tanabe-Sugano
# Characters of O (classes E, 8C3, 3C2, 6C4, 6C2'); Td uses the same labels (T_d ~ O).
_O_CLASSES = ((1, 0.0), (8, 2 * math.pi / 3), (3, math.pi), (6, math.pi / 2), (6, math.pi))
_O_CHARS = {"A1": (1, 1, 1, 1, 1), "A2": (1, 1, 1, -1, -1), "E": (2, -1, 2, 0, 0), "T1": (3, 0, -1, 1, -1),
            "T2": (3, 0, -1, -1, 1)}
_L_LETTERS = "SPDFGHIKLMNOQ"


def branching(L: int) -> dict[str, int]:
    """Multiplicity of each O irrep in the free-ion L (S -> A1, P -> T1, ..., H -> E + 2 T1 + T2)."""
    def chi_L(theta):
        if theta == 0.0:
            return 2 * L + 1
        return math.sin((2 * L + 1) * theta / 2) / math.sin(theta / 2)
    out = {}
    for irrep, ch in _O_CHARS.items():
        k = sum(g * c * chi_L(t) for (g, t), c in zip(_O_CLASSES, ch)) / 24.0
        if round(k):
            out[irrep] = int(round(k))
    return out


def _free_ion_terms(n: int, c_over_b: float) -> list[tuple[float, int, int]]:
    """Free-ion terms (energy above the ground term in units of B, 2S+1, L) of d^n at C/B; L from an
    alpha L^2 shift of every state, S from the term's (2S+1)(2L+1) count."""
    from .ddn import levels as dn_levels

    e0 = _np(dn_levels(n, B=1.0, C=c_over_b, units="cm^-1"))
    e1 = _np(dn_levels(n, B=1.0, C=c_over_b, alpha=1e-5, units="cm^-1"))
    shift = np.rint((e1 - e0) / 1e-5).astype(int)                      # L(L+1); e1 stays in e0's order
    terms = []
    for g in _group(e0, 1e-7):
        for ll in sorted(set(shift[g])):
            L = int(round((-1 + math.sqrt(1 + 4 * ll)) / 2))
            cnt = int(np.sum(shift[g] == ll))
            if L * (L + 1) != ll or cnt % (2 * L + 1):
                raise RuntimeError(f"d{n}: cannot resolve the free-ion terms at C/B = {c_over_b}")
            terms.append((float(e0[g].mean()), cnt // (2 * L + 1), L))
    emin = min(t[0] for t in terms)
    return [(e - emin, m, L) for e, m, L in terms]


def _root_l2(n: int, c_over_b: float, geometry: str, eps: float) -> dict | None:
    """{root: (<L^2>, exact)} at Dq/B = eps (full determinant basis, matched to the roots by energy and
    degeneracy).  ``exact`` is False where roots of several blocks share a level for every Dq (their
    <L^2> is the level average).  None if the match fails at this eps."""
    from . import tanabe_sugano as ts
    from .ddn import hamiltonian
    from .ligand_field import crystal_field

    V = crystal_field(torch.tensor(eps, dtype=torch.float64), geometry=geometry)
    H0 = hamiltonian(n, B=1.0, C=c_over_b, V=V, units="cm^-1")
    L2 = hamiltonian(n, alpha=1.0, units="cm^-1") - hamiltonian(n, units="cm^-1")
    w, U = torch.linalg.eigh(H0)
    w = _np(w)
    l2 = _np(torch.einsum("ai,ab,bi->i", U.conj(), L2, U).real)
    full = [(float(w[g].mean()), len(g), float(l2[g].mean())) for g in _group(w, 1e-10)]
    fmin = full[0][0]
    roots, bmin = {}, math.inf
    for blk in ts.blocks_for(n):
        er = _np(ts.block_energies(n, blk, 1.0, c_over_b, eps, geometry))
        bmin = min(bmin, float(er.min()))
        for r, x in enumerate(er):
            roots[(blk.label, r)] = (float(x), blk.degeneracy)
    out = {}
    for key, (x, deg) in roots.items():
        tol = 1e-9 * max(1.0, abs(x - bmin))
        hit = [f for f in full if abs((f[0] - fmin) - (x - bmin)) < tol]
        same = [k for k, (y, _) in roots.items() if abs(y - x) < tol]       # blocks degenerate for every Dq
        if len(hit) != 1 or hit[0][1] != sum(roots[k][1] for k in same):    # (e.g. 4A1, 4E of d5) share one level
            return None
        out[key] = (hit[0][2], len(same) == 1)
    return out


def parent_terms(n: int, c_over_b: float, geometry: str = "Oh") -> dict:
    """{(block label, root): free-ion parent letter} for every Tanabe-Sugano curve of d^n.

    The roots of a (2S+1)Gamma block that meet at one energy at Dq = 0 come from the free-ion terms
    of that spin at that energy (several when terms are degenerate, e.g. 2P and 2H of d3/d7), each
    contributing Gamma as often as the branching rule says.  Within such a cluster the roots are
    ranked by <L^2> at small Dq/B (full determinant basis) and given the terms in order of L, so the
    labels always follow the branching rules.  Where degenerate terms mix (2P/2H in a cubic field),
    the parent is the dominant one in this ranking, not a pure state.
    """
    from . import tanabe_sugano as ts

    terms = _free_ion_terms(n, c_over_b)
    gmin = min(float(_np(ts.block_energies(n, b, 1.0, c_over_b, 0.0, geometry)).min()) for b in ts.blocks_for(n))
    clusters = []                                   # (block, root cluster at Dq = 0, available L's)
    for blk in ts.blocks_for(n):
        er = _np(ts.block_energies(n, blk, 1.0, c_over_b, 0.0, geometry)) - gmin
        for cl in _group(er, 1e-6 * max(1.0, float(np.abs(er).max()))):
            avail = sorted(L for e, m, L in terms if m == blk.multiplicity and abs(e - er[cl[0]]) < 1e-6 * max(1.0, e)
                           for _ in range(branching(L).get(blk.label[1:], 0)))
            if len(avail) != len(cl):
                raise RuntimeError(f"d{n} {blk.label}: {len(cl)} roots at E/B = {er[cl[0]]:.4f} but the free-ion "
                                   f"terms there give {len(avail)}")
            clusters.append((blk, cl, avail))
    need = [(b.label, r) for b, cl, av in clusters if len(set(av)) > 1 for r in cl]
    l2 = None
    for eps in (1e-3, 2.9e-3, 0.02, 0.05, 0.1):         # d5: the cubic field splits terms only in 2nd order
        l2 = _root_l2(n, c_over_b, geometry, eps)
        if l2 is not None and all(l2[k][1] for k in need):
            break
        l2 = None
    if l2 is None:
        raise RuntimeError(f"d{n}: cannot match block roots to determinant-basis levels")
    out = {}
    for blk, cl, avail in clusters:
        for r, L in zip(sorted(cl, key=lambda r: l2[(blk.label, r)][0]), avail):
            out[(blk.label, r)] = _L_LETTERS[L]
    return out


@_styled
def tanabe_sugano(diagram, *, units: str = _u.PLOT_ENERGY_UNITS, ymax=None, labels: bool = True,
                  parents: bool = True, title: str | None = None, fontsize: float = 8.0, ax=None, figsize=(5.2, 5.4)):
    """Tanabe-Sugano diagram from :func:`torchspin.lft.tanabe_sugano.tanabe_sugano` data.

    Conventional axes E/B vs Dq/B (dimensionless) by default; when the diagram carries Racah B
    (``tanabe_sugano(..., B=...)`` or ``diagram.at(B)``) the axes are E and Dq in ``units``
    (default cm^-1).  Curves are colored by spin multiplicity (legend above the axes); the term
    label of every curve that ends inside the plot is placed in the right margin with a leader line
    (spread apart; Oh labels carry the g subscript; ``parents=True`` adds the free-ion parent term
    where a label occurs more than once, e.g. ^4T_1g(P)).  Ground-state crossovers are dashed
    vertical lines.  ``ymax``: top of the energy axis (default E/B = 70, or 70 B).
    """
    d = diagram
    units = _u.canonical_energy_unit(units)
    x = _np(d.dq_over_b)
    y = _np(d.e_over_b)
    absolute = d.B is not None
    if absolute:
        Bp = float(_np(d.B).reshape(-1)[0]) * _u.energy_factor(d.units, units)
        x, y = x * Bp, y * Bp
        el = _energy_axis_label(units)
        xlab, ylab = f"Dq ({el})", f"E ({el})"
    else:
        Bp = 1.0
        xlab, ylab = "Dq/B", "E/B"
    top = 70.0 * Bp if ymax is None else float(ymax)
    top = min(top, float(np.nanmax(y)) * 1.04) if ymax is None else top
    mults = sorted({int(lab[0][0]) for lab in d.labels}, reverse=True)
    color = {m: _PALETTE[i % len(_PALETTE)] for i, m in enumerate(mults)}
    parity = "g" if d.geometry == "Oh" else None

    fig, ax, owned = _axes(ax, figsize, (0.75, 1.05, 0.6, 0.75))
    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.set_ylim(0.0, top)
    for xc in d.crossover():
        ax.axvline(xc * Bp, color="0.55", lw=0.9, ls=(0, (4, 3)), zorder=1)
    for i, (lab, root) in enumerate(d.labels):
        m = int(lab[0])
        ax.plot(x, y[i], color=color[m], lw=1.2, zorder=2)
    _xlabel(ax, xlab)
    _ylabel(ax, ylab)
    if labels:
        ends = y[:, -1]
        inside = [i for i in range(len(ends)) if ends[i] <= 0.995 * top]
        keys = [d.labels[i] for i in inside]
        dup = {k for k in {lab for lab, _ in keys} if sum(lab == k for lab, _ in d.labels) > 1}
        par = parent_terms(d.n, float(_np(d.c_over_b)), d.geometry) if parents and dup else {}
        texts = [term_label(lab, parity, par.get((lab, r)) if lab in dup else None) for lab, r in keys]
        ys = ends[inside] / top
        _place_labels(ax, ys, texts, x_anchor=1.0, x_text=1.045, fontsize=fontsize,
                      colors=[color[int(d.labels[i][0][0])] for i in inside], priority=ends[inside])
    names = {1: "Singlets", 2: "Doublets", 3: "Triplets", 4: "Quartets", 5: "Quintets", 6: "Sextets"}
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=color[m], lw=1.6) for m in mults]
    head = title if title is not None else (
        f"d$^{{{d.n}}}$ ({d.geometry}),  C/B = {float(_np(d.c_over_b)):.2f}"
        + (f",  B = {_fmt_values([Bp])[0]} {_energy_axis_label(units)}" if absolute else ""))
    leg = ax.legend(handles, [names.get(m, f"2S+1 = {m}") for m in mults], loc="lower left",
                    bbox_to_anchor=(0.0, 1.01), ncol=len(mults), frameon=False, borderaxespad=0.0,
                    title=head, alignment="left", fontsize=STYLE["legend.fontsize"], handlelength=1.8,
                    title_fontsize=10.0, columnspacing=1.2)
    _bold_legend_title(leg)
    return fig, ax


# ------------------------------------------------------------------------------ DDNLOOP diagrams
def _auto_states(e_cm: np.ndarray, fields_mT: np.ndarray) -> list[int]:
    """States whose energy above the ground state stays within 10 mu_B B_max somewhere in the scan."""
    rel = e_cm - e_cm[:, :1]
    win = 10.0 * _MU_B * max(float(np.max(np.abs(fields_mT))), 1.0)
    return [i for i in range(e_cm.shape[1]) if float(np.min(rel[:, i])) <= win]


def _resonances(x, E, states, hv, pair_prob):
    """Crossings of E_j - E_i = hv along x.  Returns [(x*, Ei*, Ej*, weight, (i, j))]."""
    out = []
    for a, i in enumerate(states):
        for j in states[a + 1:]:
            key = (min(i, j), max(i, j))
            if pair_prob is not None and key not in pair_prob:
                continue
            dlt = E[:, key[1]] - E[:, key[0]] - hv
            w = None if pair_prob is None else pair_prob[key]
            for k in range(len(x) - 1):
                d0, d1 = dlt[k], dlt[k + 1]
                if not (np.isfinite(d0) and np.isfinite(d1)) or d0 * d1 > 0 or (d0 == 0 and k > 0) or d0 == d1:
                    continue
                t = d0 / (d0 - d1)
                lerp = lambda v: v[k] + t * (v[k + 1] - v[k])  # noqa: E731
                out.append((lerp(x), lerp(E[:, key[0]]), lerp(E[:, key[1]]),
                            1.0 if w is None else float(lerp(w)), key))
    return out


def _pair_prob(result, shape):
    if not result.pairs or result.level_probabilities is None:
        return None
    lp = _np(result.level_probabilities).reshape(shape + (len(result.pairs),))
    return {p: lp[..., k] for k, p in enumerate(result.pairs)}


def _draw_resonances(ax, res, *, threshold, fmt, axis_label, priority_ticks=True):
    """Vertical transition markers (red = strong, grey = weak, as levelsplot) and top tick labels."""
    if not res:
        sec = ax.secondary_xaxis("top")
        sec.set_xticks([])
        sec.set_xlabel(axis_label + " (none in range)", fontweight="bold")
        _style_axes(sec)
        return sec
    wmax = max(r[3] for r in res) or 1.0
    keep = [r for r in res if r[3] / wmax >= threshold]
    for xs, ei, ej, w, _ in sorted(keep, key=lambda r: r[3]):
        f = w / wmax
        col = f * np.array([0.85, 0.1, 0.1]) + (1 - f) * np.array([0.78, 0.78, 0.78])
        ax.plot([xs, xs], [ei, ej], color=col, lw=1.8, zorder=3, solid_capstyle="butt")
    sec = ax.secondary_xaxis("top")
    _style_axes(sec)
    locs = [r[0] for r in keep]
    order = np.argsort(locs)
    sec.set_xticks([locs[i] for i in order], [fmt(locs[i]) for i in order])
    sec.set_xlabel(axis_label, fontweight="bold")
    _prune_ticks(ax.figure, sec.xaxis, [keep[i][3] for i in order])
    return sec


def _loop_arrays(result, units):
    E = _np(result.energies) * _u.energy_factor(result.units, units)
    e_cm = _np(result.energies) * _u.energy_factor(result.units, "cm^-1")
    return E, e_cm


@_styled
def zeeman(result, *, states=None, units: str = _u.PLOT_ENERGY_UNITS, field_units: str = "mT", mw_freq=None,
           offset="ground", threshold: float = 1e-3, title: str | None = None, ax=None, figsize=(5.0, 4.0)):
    """Zeeman diagram (energies vs field magnitude) from :func:`torchspin.lft.ddnloop.field_scan`.

    ``states``: indices of the states to draw (default: those within 10 mu_B B_max of the ground
    state somewhere in the scan, i.e. the ground manifold an EPR experiment sees).  ``units``:
    energy axis (default cm^-1).  ``field_units``: 'mT' (default), 'G' or 'T'.  ``offset``:
    ``'ground'`` puts the lowest drawn state at the first field point at 0, a number (in ``units``)
    is subtracted, None plots the energies as computed.  ``mw_freq`` (GHz): transitions with
    E_j - E_i = h nu are drawn as vertical markers (red to grey by the level-summed probability
    when the scan has ``pairs``; then only those pairs) and their fields become tick labels on the
    top axis; markers below ``threshold`` x the strongest are omitted.
    """
    units = _u.canonical_energy_unit(units)
    B = _np(result.fields).reshape(-1)
    E, e_cm = _loop_arrays(result, units)
    E, e_cm = E.reshape(len(B), -1), e_cm.reshape(len(B), -1)
    if len(B) < 2 or np.ptp(B) == 0:
        raise ValueError("zeeman() needs a scan over field magnitude (ddnloop.field_scan)")
    if np.any(np.diff(B) < 0):
        o = np.argsort(B, kind="stable")
        B, E, e_cm = B[o], E[o], e_cm[o]
    else:
        o = None
    st = _auto_states(e_cm, B) if states is None else [int(s) for s in states]
    ff = _u.field_factor("mT", field_units)
    xb = B * ff
    off = 0.0 if offset is None else (float(E[0, st].min()) if offset == "ground" else float(offset))
    fig, ax, owned = _axes(ax, figsize, (0.85, 0.2, 0.6, 0.75 if mw_freq is not None else 0.35))
    for i in st:
        ax.plot(xb, E[:, i] - off, color=_LEVEL_COLOR, lw=1.4, zorder=2)
    ax.set_xlim(float(xb[0]), float(xb[-1]))
    _xlabel(ax, f"Magnetic field ({_field_axis_label(field_units)})")
    _ylabel(ax, f"Energy ({_energy_axis_label(units)})")
    _pad_ylim(ax)
    if mw_freq is not None:
        hv = float(mw_freq) * 1e3 * _u.energy_factor("MHz", units)
        pp = _pair_prob(result, tuple(_np(result.fields).shape))
        if pp is not None:
            pp = {k: v.reshape(-1) if o is None else v.reshape(-1)[o] for k, v in pp.items()}
        res = _resonances(xb, E - off, st, hv, pp)
        dec = 0 if field_units == "G" else (3 if field_units == "T" else 1)
        _draw_resonances(ax, res, threshold=threshold, fmt=lambda v: f"{v:.{dec}f}",
                         axis_label=f"Resonance at {float(mw_freq):g} GHz ({field_units})")
    _title(ax, title)
    return fig, ax


def _pad_ylim(ax, frac=0.04):
    lo, hi = ax.get_ylim()
    d = (hi - lo) * frac
    ax.set_ylim(lo - d, hi + d)


def _which_angle(result, angle):
    th, ph = _np(result.theta), _np(result.phi)
    if angle is None:
        vt, vp = np.ptp(th) > 0, np.ptp(ph) > 0
        if vt == vp:
            raise ValueError("pass angle='theta' or 'phi': both or neither vary in this scan")
        angle = "theta" if vt else "phi"
    if angle not in ("theta", "phi"):
        raise ValueError("angle must be 'theta' or 'phi'")
    return angle, (th if angle == "theta" else ph)


_ANGLE_LABEL = {"theta": "θ (deg)", "phi": "φ (deg)"}


@_styled
def angles(result, *, angle: str | None = None, states=None, units: str = _u.PLOT_ENERGY_UNITS, mw_freq=None,
           offset="center", threshold: float = 1e-3, title: str | None = None, ax=None, figsize=(5.0, 4.0)):
    """Angle (rotation) diagram: energies at fixed field vs theta or phi (degrees), from
    :func:`torchspin.lft.ddnloop.angle_scan`.

    ``angle``: 'theta' or 'phi' (default: the one that varies).  ``offset='center'`` subtracts the
    mean energy of the drawn states (a number in ``units``, or None, also work).  ``mw_freq`` (GHz):
    orientations where E_j - E_i = h nu are drawn as vertical markers and their angles become top
    tick labels.  Other keywords as :func:`zeeman`.  For resonance fields vs angle use :func:`roadmap`.
    """
    units = _u.canonical_energy_unit(units)
    angle, a = _which_angle(result, angle)
    E, e_cm = _loop_arrays(result, units)
    nst = E.shape[-1]
    a = np.broadcast_to(a, E.shape[:-1]).reshape(-1)
    E, e_cm = E.reshape(-1, nst), e_cm.reshape(-1, nst)
    fields = np.broadcast_to(_np(result.fields), result.energies.shape[:-1]).reshape(-1)
    if np.ptp(fields) > 0:
        raise ValueError("angles() needs a fixed field; for a field x angle scan use roadmap()")
    o = np.argsort(a, kind="stable")
    a, E, e_cm = np.degrees(a[o]), E[o], e_cm[o]
    st = _auto_states(e_cm, fields) if states is None else [int(s) for s in states]
    off = 0.0 if offset is None else (float(E[:, st].mean()) if offset == "center" else float(offset))
    fig, ax, owned = _axes(ax, figsize, (0.85, 0.2, 0.6, 0.75 if mw_freq is not None else 0.35))
    for i in st:
        ax.plot(a, E[:, i] - off, color=_LEVEL_COLOR, lw=1.4, zorder=2)
    ax.set_xlim(float(a[0]), float(a[-1]))
    _xlabel(ax, _ANGLE_LABEL[angle])
    _ylabel(ax, f"Energy ({_energy_axis_label(units)})")
    _pad_ylim(ax)
    if mw_freq is not None:
        hv = float(mw_freq) * 1e3 * _u.energy_factor("MHz", units)
        pp = _pair_prob(result, tuple(result.energies.shape[:-1]))
        if pp is not None:
            pp = {k: np.broadcast_to(v, result.energies.shape[:-1]).reshape(-1)[o] for k, v in pp.items()}
        res = _resonances(a, E - off, st, hv, pp)
        _draw_resonances(ax, res, threshold=threshold, fmt=lambda v: f"{v:.1f}",
                         axis_label=f"Resonance at {float(mw_freq):g} GHz, {float(fields[0]):g} mT (deg)")
    _title(ax, title)
    return fig, ax


@_styled
def roadmap(result, mw_freq, *, angle: str | None = None, states=None, field_units: str = "mT",
            threshold: float = 1e-3, title: str | None = None, ax=None, figsize=(5.0, 4.0)):
    """Road map (rotation pattern): resonance fields at ``mw_freq`` (GHz) vs theta or phi.

    ``result``: a :func:`torchspin.lft.ddnloop.scan` over angle x field (e.g. ``mesh=True`` with one
    theta or phi value, or broadcast angles (n, 1) against fields (m,)); the field must be the
    last scan axis.  Resonances are found by interpolation along the field; with ``pairs`` in the
    scan only those pairs are used and the line width follows the level-summed probability
    (markers below ``threshold`` x the strongest are dropped).  Field axis in ``field_units``
    (default mT).  One line color per pair (legend when there is more than one).
    """
    angle, a = _which_angle(result, angle)
    shp = tuple(result.energies.shape[:-1])
    nst = result.energies.shape[-1]
    F = np.broadcast_to(_np(result.fields), shp)
    A = np.broadcast_to(a, shp)
    if len(shp) < 2 or shp[-1] < 2:
        raise ValueError("roadmap() needs a scan whose last axis is the field")
    E = (_np(result.energies) * _u.energy_factor(result.units, "cm^-1")).reshape(-1, shp[-1], nst)
    F, A = F.reshape(-1, shp[-1]), A.reshape(-1, shp[-1])
    if np.ptp(F[0]) == 0:
        raise ValueError("roadmap() needs the field to vary along the last scan axis")
    pp = _pair_prob(result, shp)
    if pp is not None:
        pp = {k: v.reshape(-1, shp[-1]) for k, v in pp.items()}
    hv = float(mw_freq) * 1e3 * _u.energy_factor("MHz", "cm^-1")
    st = (_auto_states(E.reshape(-1, nst), F.reshape(-1)) if states is None else [int(s) for s in states])
    ff = _u.field_factor("mT", field_units)
    o = np.argsort(A[:, 0], kind="stable")
    curves: dict = {}
    for row in o:
        res = _resonances(F[row], E[row], st, hv, None if pp is None else {k: v[row] for k, v in pp.items()})
        per: dict = {}
        for xs, _, _, w, key in res:
            per.setdefault(key, []).append((xs, w))
        for key, lst in per.items():
            for m, (xs, w) in enumerate(sorted(lst)):
                curves.setdefault((key, m), []).append((float(np.degrees(A[row, 0])), xs * ff, w))
    fig, ax, owned = _axes(ax, figsize, (0.85, 0.2, 0.6, 0.35))
    wmax = max((w for c in curves.values() for _, _, w in c), default=1.0) or 1.0
    pairs = sorted({k for k, _ in curves})
    pcol = {p: _PALETTE[(i + 1) % len(_PALETTE)] for i, p in enumerate(pairs)}
    for (key, m), pts in curves.items():
        pts = [p for p in pts if p[2] / wmax >= threshold]
        if not pts:
            continue
        xs, ys, ws = (np.array(v) for v in zip(*pts))
        ax.plot(xs, ys, color=pcol[key], lw=1.6, zorder=2)
    angs = np.degrees(A[o, 0])
    ax.set_xlim(float(angs[0]), float(angs[-1]))
    _xlabel(ax, _ANGLE_LABEL[angle])
    _ylabel(ax, f"Resonance field ({field_units})")
    _pad_ylim(ax, 0.08)
    _title(ax, title if title is not None else f"{float(mw_freq):g} GHz")
    if len(pairs) > 1:
        from matplotlib.lines import Line2D
        _place_legend(ax, [Line2D([], [], color=pcol[p], lw=1.6) for p in pairs],
                      [f"{p[0] + 1}↔{p[1] + 1}" for p in pairs], owned=owned, title="States")
    return fig, ax


# ------------------------------------------------------------------------------ magnetometry
_CHI_LABEL = {
    ("chiT", "cgs"): "χT (cm$^{3}$ K mol$^{-1}$)",
    ("chiT", "SI"): "χT (m$^{3}$ K mol$^{-1}$)",
    ("chi", "cgs"): "χ (cm$^{3}$ mol$^{-1}$)",
    ("chi", "SI"): "χ (m$^{3}$ mol$^{-1}$)",
}
_M_LABEL = {"muB": "M (μ$_{\\mathregular{B}}$ per ion)", "emu/mol": "M (emu mol$^{-1}$)", "SI": "M (A m$^{2}$ mol$^{-1}$)"}


def _powder_of(y):
    if hasattr(y, "powder"):
        y = y.powder
    return _np(y)


def _chi_convert(y, T, given: str, from_units: str, quantity: str, to_units: str):
    """chi or chi T (``given``, in ``from_units``) at temperatures T -> ``quantity`` in ``to_units``."""
    from .ddnsus import mu_eff

    if given not in ("chi", "chiT"):
        raise ValueError("given must be 'chi' or 'chiT'")
    chiT = (y * T if given == "chi" else y) * _u.chi_factor(from_units, "SI")
    if quantity == "mu_eff":
        return _np(mu_eff(torch.as_tensor(np.clip(chiT, 0, None)), "SI"))
    if quantity not in ("chi", "chiT"):
        raise ValueError("quantity must be 'chiT', 'chi' or 'mu_eff'")
    out = chiT * _u.chi_factor("SI", to_units)
    return out / T if quantity == "chi" else out


def _resid_axes(fig, ax, owned):
    """Split an owned single axes into (main, residual) panels sharing x."""
    p = ax.get_position()
    hr = 0.26 * p.height
    gap = 0.03
    ax.set_position((p.x0, p.y0 + hr + gap, p.width, p.height - hr - gap))
    axr = fig.add_axes((p.x0, p.y0, p.width, hr), sharex=ax)
    _style_axes(axr)
    ax.tick_params(labelbottom=False)
    return axr


def _resid_ylim(axr, r):
    m = float(np.nanmax(np.abs(r))) if np.size(r) else 1.0
    m = m if m > 0 else 1.0
    axr.set_ylim(-1.35 * m, 1.35 * m)


@_styled
def susceptibility(T, model=None, *, data=None, model_at_data=None, quantity: str = "chiT", given: str = "chiT",
                   model_chi_units: str = "SI", chi_units: str = "cgs", residuals: bool | None = None,
                   model_label: str = "Calc.", data_label: str = "Obs.", title: str | None = None, ax=None,
                   figsize=(4.6, 4.2)):
    """chi T (default), chi or mu_eff vs T, optionally with data, a fitted curve and a residual panel.

    ``T`` (K) and ``model`` (a :class:`~torchspin.lft.ddnmag.MagResult` from ``lft.ddnsus.chi_T`` /
    ``susceptibility`` (one measuring field), or values) are the calculated curve; ``given`` says
    whether ``model`` is 'chiT' (default, as ``ddnsus.chi_T``) or 'chi', in ``model_chi_units``
    (default 'SI', the ddnsus default).  ``quantity``: 'chiT', 'chi' or 'mu_eff' (Bohr magnetons),
    plotted in ``chi_units`` ('cgs': cm^3/mol, default; 'SI': m^3/mol).  ``data``: (T_obs, y_obs)
    already in the plotted quantity and ``chi_units``.  ``residuals`` (default: on with data and
    model): obs - calc in a lower panel, calc at the data from ``model_at_data`` (same units as
    ``model``) or interpolated from the curve.  See :func:`fit` to plot a ``lft.fit`` result directly.
    """
    Tm = _np(T).reshape(-1)
    ym = None
    if model is not None:
        ym = _powder_of(model).reshape(-1)
        if ym.size != Tm.size:
            raise ValueError(f"model has {ym.size} values for {Tm.size} temperatures (one measuring field only)")
        ym = _chi_convert(ym, Tm, given, model_chi_units, quantity, chi_units)
    if residuals is None:
        residuals = data is not None and ym is not None
    axr = None
    want_r = bool(residuals and data is not None and ym is not None)
    if isinstance(ax, (list, tuple)):                       # (main, residual) axes given
        fig, owned = ax[0].figure, False
        ax, axr = ax[0], (ax[1] if want_r and len(ax) > 1 else None)
        _style_axes(ax)
        if axr is not None:
            _style_axes(axr)
    else:
        fig, ax, owned = _axes(ax, figsize, (0.95, 0.2, 0.6, 0.35 if not title else 0.55))
        if want_r and owned:
            axr = _resid_axes(fig, ax, owned)
    handles, labels = [], []
    if ym is not None:
        (ln,) = ax.plot(Tm, ym, color=_PALETTE[1], lw=1.6, zorder=2)
        handles.append(ln), labels.append(model_label)
    if data is not None:
        Td, yd = _np(data[0]).reshape(-1), _np(data[1]).reshape(-1)
        (mk,) = ax.plot(Td, yd, ls="none", marker="o", ms=4.5, mfc="white", mec=_DARK, mew=1.0, zorder=3)
        handles.append(mk), labels.append(data_label)
    qlab = "μ$_{\\mathregular{eff}}$ (μ$_{\\mathregular{B}}$)" if quantity == "mu_eff" else _CHI_LABEL[(quantity, chi_units)]
    _ylabel(ax, qlab)
    _pad_ylim(ax, 0.06)
    lo_t = min([float(np.min(Tm))] + ([float(np.min(_np(data[0])))] if data is not None else []))
    hi_t = max([float(np.max(Tm))] + ([float(np.max(_np(data[0])))] if data is not None else []))
    ax.set_xlim(max(0.0, lo_t - 0.03 * (hi_t - lo_t)), hi_t + 0.03 * (hi_t - lo_t))
    if axr is not None:
        if model_at_data is not None:
            yc = _chi_convert(_powder_of(model_at_data).reshape(-1), Td, given, model_chi_units, quantity, chi_units)
        else:
            o = np.argsort(Tm)
            yc = np.interp(Td, Tm[o], ym[o])
        r = yd - yc
        axr.axhline(0.0, color="0.5", lw=0.8, zorder=1)
        axr.plot(Td, r, ls="none", marker="o", ms=3.5, mfc="white", mec=_DARK, mew=0.9, zorder=3)
        _resid_ylim(axr, r)
        _ylabel(axr, "Obs. − calc.")
        _xlabel(axr, "T (K)")
    else:
        _xlabel(ax, "T (K)")
    _title(ax, title)
    if len(handles) > 1:
        _place_legend(ax, handles, labels, owned=owned, extra_axes=() if axr is None else (axr,))
    return (fig, [ax, axr]) if axr is not None else (fig, ax)


def _m_convert(M, from_units: str, to_units: str):
    from .ddnmag import MAG_CONSTANTS, _m_factor
    return M * (_m_factor(to_units, MAG_CONSTANTS) / _m_factor(from_units, MAG_CONSTANTS))


@_styled
def magnetization(field, M, T=None, *, x: str = "B", field_units: str = "T", from_M_units: str = "muB",
                  M_units: str = "muB", data=None, model_at_data=None, residuals: bool | None = None,
                  title: str | None = None, ax=None, figsize=(4.8, 4.2)):
    """M vs B (``x='B'``) or vs B/T (``x='B/T'``, T/K) for one or more temperatures.

    ``field`` (nB,) in mT (as ``lft.ddnmag``); ``M``: a :class:`~torchspin.lft.ddnmag.MagResult`
    (its powder (nB, nT)) or values (nB,) / (nB, nT) in ``from_M_units`` ('muB' per ion, default;
    'emu/mol'; 'SI'), plotted in ``M_units``.  ``T`` (nT,) K labels the curves (one color each;
    legend in a free corner).  ``field_units``: field axis, default 'T'.  ``data``:
    (field_obs mT, T_obs, M_obs in ``M_units``), markers colored by the matching temperature.
    ``residuals`` (default: on when ``model_at_data`` (M at the data points, ``from_M_units``) is given).
    """
    B = _np(field).reshape(-1)
    Mv = _powder_of(M)
    Mv = Mv.reshape(len(B), -1)
    Mv = _m_convert(Mv, from_M_units, M_units)
    Ts = None if T is None else _np(T).reshape(-1)
    if Ts is not None and len(Ts) != Mv.shape[1]:
        raise ValueError(f"{Mv.shape[1]} curves but {len(Ts)} temperatures")
    if x not in ("B", "B/T"):
        raise ValueError("x must be 'B' or 'B/T'")
    if x == "B/T" and Ts is None:
        raise ValueError("x='B/T' needs T")
    if residuals is None:
        residuals = data is not None and model_at_data is not None
    fig, ax, owned = _axes(ax, figsize, (0.85, 0.2, 0.6, 0.35 if not title else 0.55))
    axr = _resid_axes(fig, ax, owned) if (residuals and owned and data is not None) else None
    ff = _u.field_factor("mT", field_units)
    ncur = Mv.shape[1]
    cols = [_PALETTE[(i + 1) % len(_PALETTE)] for i in range(ncur)] if ncur <= 7 else \
        [tuple(c) for c in _viridis(ncur)]
    handles, labels = [], []

    def xval(b_mT, t):
        return b_mT * 1e-3 / t if x == "B/T" else b_mT * ff

    for c in range(ncur):
        t = None if Ts is None else Ts[c]
        (ln,) = ax.plot(xval(B, t), Mv[:, c], color=cols[c], lw=1.5, zorder=2)
        handles.append(ln)
        labels.append(f"{t:g}" if t is not None else f"{c + 1}")
    if data is not None:
        Bd, Td, Md = (np.broadcast_arrays(*[_np(v).reshape(-1) for v in data]))
        for c in range(ncur):
            sel = np.ones(len(Bd), bool) if Ts is None or ncur == 1 else np.isclose(Td, Ts[c])
            if sel.any():
                ax.plot(xval(Bd[sel], Td[sel]), Md[sel], ls="none", marker="o", ms=4.0, mfc="white", mec=cols[c],
                        mew=1.0, zorder=3)
        if axr is not None:
            r = Md - _m_convert(_np(model_at_data).reshape(-1), from_M_units, M_units)
            axr.axhline(0.0, color="0.5", lw=0.8, zorder=1)
            for c in range(ncur):
                sel = np.ones(len(Bd), bool) if Ts is None or ncur == 1 else np.isclose(Td, Ts[c])
                axr.plot(xval(Bd[sel], Td[sel]), r[sel], ls="none", marker="o", ms=3.2, mfc="white", mec=cols[c],
                         mew=0.9, zorder=3)
            _resid_ylim(axr, r)
            _ylabel(axr, "Obs. − calc.")
    xl = "B/T (T K$^{-1}$)" if x == "B/T" else f"B ({_field_axis_label(field_units)})"
    _xlabel(axr if axr is not None else ax, xl)
    _ylabel(ax, _M_LABEL[M_units])
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0 if float(np.nanmin(Mv)) >= 0 else None)
    _pad_ylim(ax, 0.05)
    _title(ax, title)
    if ncur > 1 or Ts is not None:
        _place_legend(ax, handles, labels, owned=owned, title="T (K)" if Ts is not None else None,
                      extra_axes=() if axr is None else (axr,))
    return (fig, [ax, axr]) if axr is not None else (fig, ax)


def _viridis(n):
    import matplotlib
    return matplotlib.colormaps["viridis"](np.linspace(0.0, 0.9, n))


# ------------------------------------------------------------------------------ powder spectra
@_styled
def powder(result, *, mw_freq=None, field_units: str = "mT", kind: str = "field", from_units: str = "MHz",
           units: str = _u.PLOT_ENERGY_UNITS, normalize: bool = True, data=None, label: str | None = None,
           ylabel: str = "Intensity (arb. u.)", title: str | None = None, ax=None, figsize=(5.0, 3.6)):
    """Powder spectrum from :func:`torchspin.lft.ddnpow.simulate` (``kind='field'``: field axis in
    ``field_units``, default mT) or :func:`~torchspin.lft.ddnpow.fdmr` (``kind='energy'``: axis in
    ``from_units`` -> ``units``, default cm^-1).

    ``mw_freq`` (GHz) is quoted in the legend.  ``normalize`` (default): scale to max |y| = 1 (data too).
    ``data``: (axis, y) in the plotted axis unit, overlaid as a dark line.
    """
    x = _np(result.axis).reshape(-1)
    y = _np(result.spectrum).reshape(-1)
    if kind == "field":
        x = x * _u.field_factor("mT", field_units)
        xl = f"Magnetic field ({_field_axis_label(field_units)})"
    elif kind == "energy":
        x = x * _u.energy_factor(from_units, units)
        xl = f"Energy ({_energy_axis_label(units)})"
    else:
        raise ValueError("kind must be 'field' or 'energy'")
    if normalize:
        y = y / (float(np.max(np.abs(y))) or 1.0)
    fig, ax, owned = _axes(ax, figsize, (0.85, 0.2, 0.6, 0.35 if not title else 0.55))
    handles, labels = [], []
    if data is not None:
        xd, yd = _np(data[0]).reshape(-1), _np(data[1]).reshape(-1)
        if normalize:
            yd = yd / (float(np.max(np.abs(yd))) or 1.0)
        (dl,) = ax.plot(xd, yd, color=_DARK, lw=1.2, zorder=2)
        handles.append(dl), labels.append("Obs.")
    (ln,) = ax.plot(x, y, color=_PALETTE[2] if data is not None else _LEVEL_COLOR, lw=1.4, zorder=3)
    lab = label if label is not None else ("Sim." + (f", {float(mw_freq):g} GHz" if mw_freq is not None else ""))
    handles.append(ln), labels.append(lab)
    if float(np.min(y)) < 0 < float(np.max(y)):
        ax.axhline(0.0, color="0.6", lw=0.7, zorder=1)
    ax.set_xlim(float(x.min()), float(x.max()))
    _pad_ylim(ax, 0.12)
    _xlabel(ax, xl)
    _ylabel(ax, ylabel)
    _title(ax, title)
    if data is not None or mw_freq is not None or label is not None:
        _place_legend(ax, handles, labels, owned=owned)
    return fig, ax


# ------------------------------------------------------------------------------ lft.fit companion
@_styled
def fit(joint, outcome, dataset=None, *, npts: int = 200, quantity: str | None = None, chi_units: str | None = None,
        x: str = "B", field_units: str = "T", title: str | None = None, ax=None):
    """Plot a :class:`torchspin.lft.fit.JointFit` result on one magnetometry data set: the data, the
    best-fit curve (``npts`` points, recomputed from the model at ``outcome.result.values``) and the
    residual panel (obs - calc at the data points, exact).

    ``dataset``: name or index (default: the first Susceptibility or Magnetization block).
    Susceptibility: ``quantity`` (default the block's) and ``chi_units`` (default the block's), see
    :func:`susceptibility`.  Magnetization: ``x`` and ``field_units`` as :func:`magnetization`.
    """
    import dataclasses

    from .fit import Magnetization, Susceptibility

    dsets = joint.datasets
    if dataset is None:
        cand = [d for d in dsets if isinstance(d, (Susceptibility, Magnetization))]
        if not cand:
            raise ValueError("no Susceptibility or Magnetization data set in this fit")
        ds = cand[0]
    elif isinstance(dataset, int):
        ds = dsets[dataset]
    else:
        ds = next((d for d in dsets if d.name == dataset), None)
        if ds is None:
            raise ValueError(f"no data set named {dataset!r}; have {[d.name for d in dsets]}")
    xbest = torch.as_tensor(np.asarray(outcome.result.values, float), dtype=torch.float64)
    with torch.no_grad():
        ctx = joint.context(xbest)
        calc = _np(ds.calculate(ctx))
        if isinstance(ds, Susceptibility):
            Td = _np(ds.T).reshape(-1)
            Tg = np.linspace(float(Td.min()), float(Td.max()), npts)
            dense = dataclasses.replace(ds, T=Tg, values=np.zeros(npts))
            curve = _np(dense.calculate(ctx))
        elif isinstance(ds, Magnetization):
            Bd, Td = np.broadcast_arrays(_np(ds.field).reshape(-1), _np(ds.T).reshape(-1))
            Ts = np.unique(Td)
            Bg = np.linspace(0.0, float(Bd.max()), npts)
            BB, TT = np.meshgrid(Bg, Ts, indexing="ij")
            dense = dataclasses.replace(ds, field=BB.reshape(-1), T=TT.reshape(-1), values=np.zeros(BB.size))
            curve = _np(dense.calculate(ctx)).reshape(npts, len(Ts))
        else:
            raise ValueError(f"plot.fit draws Susceptibility or Magnetization data sets, not {type(ds).__name__}")
    head = title if title is not None else f"{ds.name}: χ²/N = {outcome.chi2[ds.name] / max(outcome.n_points[ds.name], 1):.3g}"
    if isinstance(ds, Susceptibility):
        q = quantity or ds.quantity
        cu = chi_units or ds.chi_units
        yobs = _chi_convert(ds.observed, Td, ds.quantity, ds.chi_units, q, cu)
        return susceptibility.__wrapped__(Tg, curve, data=(Td, yobs), model_at_data=calc, quantity=q, given=ds.quantity,
                                          model_chi_units=ds.chi_units, chi_units=cu, residuals=True, title=head,
                                          model_label="Fit", ax=ax)
    return magnetization.__wrapped__(Bg, curve, Ts, x=x, field_units=field_units, from_M_units=ds.M_units,
                                     M_units=ds.M_units, data=(Bd, Td, ds.observed), model_at_data=calc,
                                     residuals=True, title=head, ax=ax)
