"""Figures, tables and the device audit for a verified workstation benchmark campaign.

    python benchmarks/analysis/make_verified_figures.py --results benchmarks/results/workstation_20260904_verified

Reads ``matlab_workloads.json``, ``cpu_threads{1,8,32}.json`` and (if present)
``gpu.json`` written by ``benchmarks/cluster/run_workstation_verified.sh`` and writes into
``<results>/figures``:

* ``Figure_performance_workstation_verified.{png,pdf,svg}`` -- manuscript Figure 2:
  (A) wall time (median, IQR error bars) for EasySpin, TorchSpin selected CPU thread
  count and TorchSpin CUDA; the CUDA series contains only workloads whose CUDA
  execution was verified (``effective_mode`` ``cuda`` or ``hybrid``; hybrid bars are
  hatched); CPU-only workloads are marked N/A.  (B) EasySpin / TorchSpin speed ratio
  with IQR-propagated error bars, on a log2 axis.
* ``Figure_S_thread_scaling_workstation_verified.{png,pdf,svg}`` -- Figure S5: CPU thread
  scaling heat map (median times, speedup relative to one thread).
* ``raw_replicates.csv``, ``summary.csv``, ``summary_tables.md`` and
  ``device_audit.json`` (requested/effective device, thread counts, CUDA kernels,
  CPU-routed eigendecompositions, CPU/CUDA consistency for every workload).

Style follows the 2026-09-02 manuscript draft (Arial, 600 dpi PNG, editable SVG text).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'workloads'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from workloads import WORKLOAD_DEVICES, STOCHASTIC  # noqa: E402

WORKLOADS = list(WORKLOAD_DEVICES)
SHORT_LABELS = ["pepper\nCu", "pepper\nMn(II)", "pepper\nperturb.", "pepper\nstrain", "pepper\n20 calls",
                "chili\n2 nuclei", "chili\npotential", "saffron\nHYSCORE", "cardamom\ntrajectories"]
LONG_LABELS = ["pepper: Cu + 2 N, matrix", "pepper: Mn(II), matrix", "pepper: perturbative grid",
               "pepper: strain summation", "pepper: 20-call fit pattern", "chili: nitroxide + proton",
               "chili: orienting potential", "saffron: HYSCORE", "cardamom: trajectories"]
THREADS = (1, 8, 32)
COLORS = {"easyspin": "#222222", "cpu": "#FF0000", "cuda": "#0072B2", "unity": "#A0A0A0"}

plt.rcParams.update({
    "font.family": "Arial", "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5, "axes.linewidth": 0.7,
    "lines.solid_capstyle": "round", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.transparent": True,
})


# ----------------------------------------------------------------------------- data
class Series:
    """median / quartiles / replicates per workload from one campaign JSON."""

    def __init__(self, path: Path | None):
        self.path = path
        self.data = json.loads(path.read_text()) if path and path.exists() else None

    @property
    def meta(self):
        return (self.data or {}).get('meta', {})

    def has(self, name):
        return bool(self.data) and name in self.data.get('seconds', {})

    def stat(self, name):
        sec = self.data['seconds'][name]
        reps = self.data.get('replicates', {}).get(name, [sec])
        reps = list(reps) if isinstance(reps, (list, tuple)) else [reps]
        if len(reps) > 1:
            # Recompute all series identically. Earlier TorchSpin JSON files used
            # statistics.quantiles()'s exclusive estimator, whereas the MATLAB
            # series used ordinary 25th/75th percentiles. The difference is
            # conspicuous for n=5 when one short benchmark has a slow replicate.
            q1, med, q3 = np.percentile(reps, [25, 50, 75])
            return float(med), float(q1), float(q3), reps
        st = (self.data or {}).get('stats', {}).get(name)
        if st:
            return st['median'], st['q1'], st['q3'], reps
        return sec, sec, sec, reps

    def audit(self, name):
        return (self.data or {}).get('audit', {}).get(name, {})


def _mode(gpu: Series, name: str) -> str:
    if gpu.data is None:
        return 'not run'
    a = gpu.audit(name)
    if a:
        return a.get('effective_mode', 'unknown')
    # legacy file without audit: trust the implementation table only
    return 'cuda (unverified)' if 'cuda' in WORKLOAD_DEVICES[name] else 'unsupported'


def _ratio_bounds(num, den):
    """EasySpin/TorchSpin ratio of medians with IQR-propagated bounds."""
    m = num[0] / den[0]
    lo = num[1] / den[2] if den[2] > 0 else m
    hi = num[2] / den[1] if den[1] > 0 else m
    return m, lo, hi


def _selected_cpu_indices(cpu: dict) -> np.ndarray:
    """Select the least-threaded configuration indistinguishable by IQR overlap.

    For each workload, first find the configuration with the lowest median wall
    time.  Then select the smallest thread count whose IQR overlaps the IQR of
    that fastest-median configuration.  This avoids presenting negligible timing
    noise as evidence that a larger thread pool is beneficial.
    """
    selected = []
    for name in WORKLOADS:
        stats = [cpu[t].stat(name)[:3] for t in THREADS]
        fastest = int(np.argmin([s[0] for s in stats]))
        fastest_q1, fastest_q3 = stats[fastest][1], stats[fastest][2]
        overlapping = [
            i for i, (_, q1, q3) in enumerate(stats)
            if q1 <= fastest_q3 and q3 >= fastest_q1
        ]
        selected.append(min(overlapping) if overlapping else fastest)
    return np.asarray(selected, dtype=int)


def _save(fig, outdir: Path, stem: str):
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(outdir / f"{stem}.{suffix}", dpi=600 if suffix == "png" else None,
                    bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# ----------------------------------------------------------------------------- figures
def figure_performance(matlab: Series, cpu: dict, gpu: Series, outdir: Path, stem: str):
    n = len(WORKLOADS)
    x = np.arange(n)
    width = 0.245
    es = np.array([matlab.stat(w)[:3] for w in WORKLOADS])                       # (n, 3)
    best_idx = _selected_cpu_indices(cpu)
    best = np.array([cpu[THREADS[i]].stat(w)[:3] for w, i in zip(WORKLOADS, best_idx)])
    best_threads = np.array(THREADS)[best_idx]
    modes = [_mode(gpu, w) for w in WORKLOADS]
    cuda_ok = [m in ('cuda', 'hybrid') for m in modes]
    cuda = np.array([gpu.stat(w)[:3] if ok else [np.nan] * 3 for w, ok in zip(WORKLOADS, cuda_ok)])

    fig, (ax_time, ax_ratio) = plt.subplots(2, 1, figsize=(7.15, 4.85),
                                            gridspec_kw={"height_ratios": [1.25, 1.0], "hspace": 0.19})

    def _err(arr):
        return np.vstack([arr[:, 0] - arr[:, 1], arr[:, 2] - arr[:, 0]])

    ekw = dict(ecolor="#555555", elinewidth=0.7, capsize=1.6, capthick=0.7)
    ax_time.bar(x - width, es[:, 0], width, yerr=_err(es), color=COLORS["easyspin"], label="EasySpin CPU", error_kw=ekw)
    ax_time.bar(x, best[:, 0], width, yerr=_err(best), color=COLORS["cpu"], label="TorchSpin CPU", error_kw=ekw)
    for xi, (m, row, ok) in enumerate(zip(modes, cuda, cuda_ok)):
        if ok:
            ax_time.bar(xi + width, row[0], width, yerr=_err(row[None, :]), color=COLORS["cuda"],
                        hatch="////" if m == 'hybrid' else None, edgecolor="white" if m == 'hybrid' else None,
                        linewidth=0.0, error_kw=ekw)
        else:
            ax_time.text(xi + width, ax_time.get_ylim()[0] if False else es[:, 0].min() * 0.9, "N/A", ha="center",
                         va="bottom", fontsize=7.5, color=COLORS["cuda"], rotation=90)
    ax_time.set_yscale("log")
    ax_time.set_ylabel("Wall time (s)")
    ax_time.set_xticks(x)
    ax_time.set_xticklabels([])
    ax_time.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax_time.yaxis.set_minor_formatter(NullFormatter())
    handles = [Patch(color=COLORS["easyspin"], label="EasySpin CPU"), Patch(color=COLORS["cpu"], label="TorchSpin CPU")]
    if any(m == 'cuda' for m in modes):
        handles.append(Patch(color=COLORS["cuda"], label="TorchSpin CUDA"))
    if any(m == 'hybrid' for m in modes):
        handles.append(Patch(facecolor=COLORS["cuda"], hatch="////", edgecolor="white", label="TorchSpin CUDA (hybrid CPU/CUDA)"))
    ax_time.legend(handles=handles, frameon=False, ncol=len(handles), loc="upper center",
                   bbox_to_anchor=(0.5, 1.16), handlelength=1.6, columnspacing=1.2)
    for xi, runtime, threads in zip(x, best[:, 0], best_threads):
        ax_time.annotate(f"{threads}t", (xi, runtime), xytext=(0, 4), textcoords="offset points",
                         ha="center", va="bottom", fontsize=8.5, color=COLORS["cpu"], fontweight="bold")

    cpu_ratio = np.array([_ratio_bounds(es[i], best[i]) for i in range(n)])
    cuda_ratio = np.array([_ratio_bounds(es[i], cuda[i]) if cuda_ok[i] else [np.nan] * 3 for i in range(n)])
    ax_ratio.axhline(1.0, color=COLORS["unity"], linestyle="--", linewidth=1.0, zorder=1)
    ax_ratio.errorbar(x - 0.08, cpu_ratio[:, 0], yerr=[cpu_ratio[:, 0] - cpu_ratio[:, 1], cpu_ratio[:, 2] - cpu_ratio[:, 0]],
                      fmt="o", ms=6, color=COLORS["cpu"], ecolor="#555555", elinewidth=0.7, capsize=1.6,
                      label="TorchSpin CPU", zorder=3)
    ok = np.array(cuda_ok)
    if ok.any():
        hyb = np.array([m == 'hybrid' for m in modes])
        for sel, lab, mfc in ((ok & ~hyb, "TorchSpin CUDA", COLORS["cuda"]),
                              (ok & hyb, "TorchSpin CUDA (hybrid)", "white")):
            if sel.any():
                ax_ratio.errorbar(x[sel] + 0.08, cuda_ratio[sel, 0],
                                  yerr=[cuda_ratio[sel, 0] - cuda_ratio[sel, 1], cuda_ratio[sel, 2] - cuda_ratio[sel, 0]],
                                  fmt="s", ms=5.5, color=COLORS["cuda"], markerfacecolor=mfc, markeredgewidth=1.2,
                                  ecolor="#555555", elinewidth=0.7, capsize=1.6, label=lab, zorder=3)
    ax_ratio.set_ylabel("Speedup vs EasySpin")
    ax_ratio.set_xticks(x)
    ax_ratio.set_xticklabels(SHORT_LABELS)
    top = np.nanmax(np.concatenate([cpu_ratio[:, 2], cuda_ratio[:, 2][ok] if ok.any() else [1.0]]))
    bottom = np.nanmin(np.concatenate([cpu_ratio[:, 1], cuda_ratio[:, 1][ok] if ok.any() else [1.0]]))
    # log2 axis: a 2x slowdown and a 2x speedup are the same distance from unity
    ax_ratio.set_yscale("log", base=2)
    ax_ratio.set_ylim(min(0.25, bottom * 0.8), max(4.0, top * 1.15))
    ticks = [t for t in (0.25, 0.5, 1, 2, 4, 8, 16) if ax_ratio.get_ylim()[0] <= t <= ax_ratio.get_ylim()[1]]
    ax_ratio.set_yticks(ticks)
    ax_ratio.set_yticklabels([f"{t:g}" for t in ticks])
    ax_ratio.yaxis.set_minor_formatter(NullFormatter())
    ax_ratio.legend(frameon=False, ncol=3, loc="lower right", bbox_to_anchor=(1.0, 1.015), borderaxespad=0.0)
    for ax in (ax_time, ax_ratio):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(which="major", direction="in", length=3.5, width=0.7, bottom=True, left=True, top=False, right=False)
        ax.tick_params(which="minor", direction="in", length=1.8, width=0.55, bottom=True, left=True, top=False, right=False)
    ax_time.text(-0.095, 1.04, "A", transform=ax_time.transAxes, fontsize=15, fontweight="bold")
    ax_ratio.text(-0.095, 1.04, "B", transform=ax_ratio.transAxes, fontsize=15, fontweight="bold")
    fig.subplots_adjust(left=0.12, right=0.985, top=0.91, bottom=0.16)
    _save(fig, outdir, stem)
    return dict(best_threads=best_threads, best=best, es=es, cuda=cuda, modes=modes, cpu_ratio=cpu_ratio, cuda_ratio=cuda_ratio)


def figure_thread_scaling(cpu: dict, outdir: Path, stem: str):
    med = np.array([[cpu[t].stat(w)[0] for t in THREADS] for w in WORKLOADS])
    q1 = np.array([[cpu[t].stat(w)[1] for t in THREADS] for w in WORKLOADS])
    q3 = np.array([[cpu[t].stat(w)[2] for t in THREADS] for w in WORKLOADS])
    n_rep = min(len(cpu[t].stat(w)[3]) for t in THREADS for w in WORKLOADS)
    best_idx = _selected_cpu_indices(cpu)
    speedup = med[:, [0]] / med
    log2 = np.log2(speedup)
    limit = max(float(np.max(np.abs(log2))), 1e-6)
    fig, ax = plt.subplots(figsize=(6.9, 4.8))
    im = ax.imshow(log2, cmap="RdBu", norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit), aspect="auto")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels([f"{t} thread{'s' if t > 1 else ''}" for t in THREADS])
    ax.set_yticks(np.arange(len(WORKLOADS)))
    ax.set_yticklabels(LONG_LABELS)
    ax.set_xlabel(f"TorchSpin CPU configuration (median of {n_rep}; IQR in brackets)")
    ax.set_ylabel("Benchmark workload")
    for row in range(len(WORKLOADS)):
        for col in range(3):
            value = log2[row, col]
            text_color = "white" if abs(value) > 0.58 * limit else "black"
            ax.text(col, row, f"{med[row, col]:.2f} s [{q1[row, col]:.2f}–{q3[row, col]:.2f}]\n{speedup[row, col]:.2f}×",
                    ha="center", va="center", color=text_color, fontsize=7.6,
                    fontweight="bold" if col == best_idx[row] else "normal")
        ax.add_patch(Rectangle((best_idx[row] - 0.495, row - 0.495), 0.99, 0.99, fill=False,
                               edgecolor="#222222", linewidth=1.1, zorder=4))
    for edge in np.arange(-0.5, len(WORKLOADS), 1):
        ax.axhline(edge, color="white", linewidth=1.2)
    for edge in np.arange(-0.5, 3, 1):
        ax.axvline(edge, color="white", linewidth=1.2)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("Speedup relative to one thread (log$_2$ scale)")
    ticks = np.array([-2, -1, 0, 1, 2], dtype=float)
    ticks = ticks[(ticks >= -limit) & (ticks <= limit)]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{2**tick:.2g}×" for tick in ticks])
    ax.tick_params(which="major", direction="in", length=3.5, width=0.7, bottom=True, left=True, top=False, right=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
    cbar.ax.tick_params(which="major", direction="in", length=3.5, width=0.7)
    fig.subplots_adjust(left=0.31, right=0.91, top=0.985, bottom=0.11)
    _save(fig, outdir, stem)


# ----------------------------------------------------------------------------- tables / audit
def write_tables(matlab: Series, cpu: dict, gpu: Series, res: dict, outdir: Path):
    with (outdir / "raw_replicates.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["workload", "implementation", "series", "requested_device", "effective_mode", "torch_threads", "replicate", "seconds"])
        for name in WORKLOADS:
            for k, s in enumerate(matlab.stat(name)[3]):
                w.writerow([name, "EasySpin", "matlab", "cpu", "cpu", matlab.meta.get('maxNumCompThreads'), k + 1, s])
            for t in THREADS:
                for k, s in enumerate(cpu[t].stat(name)[3]):
                    w.writerow([name, "TorchSpin", f"cpu_threads{t}", "cpu", "cpu", t, k + 1, s])
            if gpu.has(name):
                for k, s in enumerate(gpu.stat(name)[3]):
                    w.writerow([name, "TorchSpin", "cuda", "cuda", _mode(gpu, name), gpu.meta.get('torch_num_threads'), k + 1, s])
    rows = []
    with (outdir / "summary.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["workload", "easyspin_median_s", "easyspin_q1", "easyspin_q3",
                    *[f"torchspin_{t}t_median_s" for t in THREADS], *[f"torchspin_{t}t_q1" for t in THREADS], *[f"torchspin_{t}t_q3" for t in THREADS],
                    "best_cpu_threads", "torchspin_best_cpu_median_s", "cuda_effective_mode", "torchspin_cuda_median_s", "torchspin_cuda_q1", "torchspin_cuda_q3",
                    "easyspin_over_best_cpu", "easyspin_over_best_cpu_lo", "easyspin_over_best_cpu_hi",
                    "easyspin_over_cuda", "easyspin_over_cuda_lo", "easyspin_over_cuda_hi"])
        for i, name in enumerate(WORKLOADS):
            es = res['es'][i]; best = res['best'][i]; cu = res['cuda'][i]
            cpus = [cpu[t].stat(name) for t in THREADS]
            row = [name, *es, *[c[0] for c in cpus], *[c[1] for c in cpus], *[c[2] for c in cpus],
                   res['best_threads'][i], best[0], res['modes'][i], *cu, *res['cpu_ratio'][i], *res['cuda_ratio'][i]]
            w.writerow(row); rows.append(row)
    md = ["| Workload | EasySpin (s) | TorchSpin 1 t | 8 t | 32 t | best CPU | CUDA mode | TorchSpin CUDA (s) | ES / best CPU | ES / CUDA |",
          "|---|---:|---:|---:|---:|:--:|:--:|---:|---:|---:|"]

    def f(m, a, b):
        return "n/a" if not np.isfinite(m) else f"{m:.3g} [{a:.3g}–{b:.3g}]"
    for i, name in enumerate(WORKLOADS):
        es = res['es'][i]; cu = res['cuda'][i]
        cpus = [cpu[t].stat(name) for t in THREADS]
        mode = res['modes'][i]
        cr = res['cpu_ratio'][i]; gr = res['cuda_ratio'][i]
        md.append(f"| `{name}` | {f(*es)} | {f(*cpus[0][:3])} | {f(*cpus[1][:3])} | {f(*cpus[2][:3])} | {res['best_threads'][i]} t | {mode} | "
                  f"{f(*cu)} | {f(*cr)} | {f(*gr)} |")
    md.extend([
        "",
        "\"best CPU\" is the smallest tested thread count whose IQR overlaps the IQR of the configuration with the lowest median wall time.",
    ])
    (outdir / "summary_tables.md").write_text("\n".join(md) + "\n")


def write_device_audit(matlab: Series, cpu: dict, gpu: Series, outdir: Path):
    host = dict(cpu_runs={t: {k: cpu[t].meta.get(k) for k in ('torch_num_threads', 'torch_num_interop_threads', 'thread_env', 'native_threadpools',
                                                             'cpu_affinity', 'n_affinity_cpus', 'torchspin_commit', 'torchspin_status', 'warmup', 'repeat', 'started', 'finished')}
                          for t in THREADS},
                cuda_run={k: gpu.meta.get(k) for k in ('gpu', 'cuda_visible_devices', 'torch_num_threads', 'thread_env', 'native_threadpools', 'cpu_affinity',
                                                       'n_affinity_cpus', 'torchspin_commit', 'torchspin_status', 'warmup', 'repeat', 'started', 'finished')} if gpu.data else None,
                matlab_run=matlab.meta,
                software={k: (cpu[1].meta.get(k) or gpu.meta.get(k)) for k in ('host', 'python', 'torch', 'torch_cuda', 'numpy')})
    per = {}
    for name in WORKLOADS:
        a = gpu.audit(name) if gpu.data else {}
        cons = a.get('consistency_vs_cpu')
        per[name] = dict(
            supported_devices=list(WORKLOAD_DEVICES[name]), stochastic=name in STOCHASTIC,
            cpu=dict(effective_mode='cpu', torch_threads=list(THREADS), cuda_kernels_observed=False),
            cuda=dict(requested_device='cuda', effective_mode=_mode(gpu, name),
                      torch_threads=gpu.meta.get('torch_num_threads') if gpu.data else None,
                      cuda_kernels_observed=a.get('cuda_kernels_observed'), cuda_kernel_events=a.get('cuda_kernel_events'),
                      cuda_kernel_seconds=a.get('cuda_kernel_seconds'), profiled_wall_seconds=a.get('profiled_wall_seconds'),
                      cuda_kernel_fraction_of_wall=a.get('cuda_kernel_fraction_of_wall'),
                      memcpy_dtoh_events=a.get('memcpy_dtoh_events'), memcpy_htod_events=a.get('memcpy_htod_events'),
                      eigh_calls=a.get('eigh'), cpu_routed_eigh_dim_gt_32=a.get('cpu_routed_eigh_dim_gt_32'),
                      consistency_vs_cpu=cons, gpu_utilisation_before=a.get('gpu_before'), gpu_utilisation_after=a.get('gpu_after'),
                      note=a.get('note')),
        )
    (outdir / "device_audit.json").write_text(json.dumps(dict(host=host, workloads=per), indent=2, default=str) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=str(Path(__file__).resolve().parents[1] / 'results' / 'workstation_20260904_verified'))
    ap.add_argument('--out', default='')
    ap.add_argument('--stem', default='workstation_verified')
    a = ap.parse_args()
    results = Path(a.results)
    outdir = Path(a.out) if a.out else results / 'figures'
    outdir.mkdir(parents=True, exist_ok=True)
    matlab = Series(results / 'matlab_workloads.json')
    cpu = {t: Series(results / f'cpu_threads{t}.json') for t in THREADS}
    gpu = Series(results / 'gpu.json')
    missing = [p.name for p in [matlab.path, *[cpu[t].path for t in THREADS]] if not p.exists()]
    if missing:
        sys.exit(f'missing result files: {missing}')
    res = figure_performance(matlab, cpu, gpu, outdir, f'Figure_performance_{a.stem}')
    figure_thread_scaling(cpu, outdir, f'Figure_S_thread_scaling_{a.stem}')
    write_tables(matlab, cpu, gpu, res, outdir)
    write_device_audit(matlab, cpu, gpu, outdir)
    print(f'wrote {outdir}:')
    for p in sorted(outdir.iterdir()):
        print(f'  {p.name}')
    print((outdir / 'summary_tables.md').read_text())


if __name__ == '__main__':
    main()
