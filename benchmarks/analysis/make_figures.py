"""Generate manuscript-quality figures and LaTeX tables from comparison.json.

Run after compare_results.py. Reads ``results/comparison.json`` and writes
figures + LaTeX tables to ``results/figures/`` and ``results/tables/``.

Usage::

    python make_figures.py [--results-dir ../results/]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def fig_timing_bars(rows, out_path: Path):
    """Bar chart: wall-time per simulator × backend (MATLAB / CPU / GPU)."""
    if not HAS_MPL:
        return
    sims = sorted(set(r['simulator'] for r in rows))
    n = len(sims)
    matlab_times = []
    cpu_times = []
    gpu_times = []
    labels = []
    for r in rows:
        labels.append(f"{r['simulator']}/{r['case'][:12]}")
        matlab_times.append(r['matlab_best_s'] or np.nan)
        cpu_times.append(r['python_cpu_best_s'] or np.nan)
        gpu_times.append(r['python_gpu_best_s'] or np.nan)

    x = np.arange(len(labels))
    width = 0.27

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.65), 5))
    ax.bar(x - width, [t * 1000 if t else 0 for t in matlab_times],
           width, label='MATLAB EasySpin', color='#888888')
    ax.bar(x,         [t * 1000 if t else 0 for t in cpu_times],
           width, label='torchspin (CPU)', color='#1f77b4')
    ax.bar(x + width, [t * 1000 if t else 0 for t in gpu_times],
           width, label='torchspin (GPU)', color='#2ca02c')
    ax.set_yscale('log')
    ax.set_ylabel('Wall time (ms, log scale)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.set_title('Best-of-5 wall time per benchmark case')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Wrote: {out_path}")


def fig_speedup(rows, out_path: Path):
    """Speedup ratios: torchspin-CPU/MATLAB and torchspin-GPU/CPU."""
    if not HAS_MPL:
        return
    labels, cpu_sp, gpu_sp = [], [], []
    for r in rows:
        if r['cpu_speedup_vs_matlab'] is None and r['gpu_speedup_vs_cpu'] is None:
            continue
        labels.append(f"{r['simulator']}/{r['case'][:12]}")
        cpu_sp.append(r['cpu_speedup_vs_matlab'] or np.nan)
        gpu_sp.append(r['gpu_speedup_vs_cpu'] or np.nan)
    if not labels:
        return
    x = np.arange(len(labels))
    width = 0.4
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.6), 4.5))
    ax.bar(x - width/2, cpu_sp, width, label='torchspin CPU vs MATLAB', color='#1f77b4')
    ax.bar(x + width/2, gpu_sp, width, label='torchspin GPU vs CPU', color='#2ca02c')
    ax.axhline(1.0, color='k', linestyle=':', alpha=0.5)
    ax.set_ylabel('Speedup (x times faster)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.set_title('Speedup of torchspin vs MATLAB / GPU vs CPU')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Wrote: {out_path}")


def fig_accuracy(rows, out_path: Path):
    """Cosine similarity per case (Python CPU vs MATLAB)."""
    if not HAS_MPL:
        return
    labels, cosines = [], []
    for r in rows:
        if r['cosine_cpu_vs_matlab'] is not None:
            labels.append(f"{r['simulator']}/{r['case'][:12]}")
            cosines.append(r['cosine_cpu_vs_matlab'])
    if not labels:
        return
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.6), 4))
    bar_colors = ['#2ca02c' if c >= 0.99 else '#ff7f0e' if c >= 0.9 else '#d62728'
                  for c in cosines]
    ax.bar(x, cosines, color=bar_colors)
    ax.axhline(0.99, color='gray', linestyle=':', alpha=0.5, label='0.99 threshold')
    ax.set_ylim(min(0.0, min(cosines) - 0.05), 1.02)
    ax.set_ylabel('Cosine similarity')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.set_title('torchspin vs MATLAB EasySpin parity (cosine)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Wrote: {out_path}")


def latex_table(rows, out_path: Path):
    """Manuscript-ready LaTeX table."""
    lines = [
        r'\begin{tabular}{l l r r r r r}',
        r'\toprule',
        r'Simulator & Case & MATLAB (ms) & CPU (ms) & GPU (ms) & Speedup CPU/ML & Cosine \\',
        r'\midrule',
    ]
    for r in rows:
        ml = f"{r['matlab_best_s']*1000:.1f}" if r['matlab_best_s'] else '---'
        cpu = f"{r['python_cpu_best_s']*1000:.1f}" if r['python_cpu_best_s'] else '---'
        gpu = f"{r['python_gpu_best_s']*1000:.1f}" if r['python_gpu_best_s'] else '---'
        sp = f"{r['cpu_speedup_vs_matlab']:.2f}" if r['cpu_speedup_vs_matlab'] else '---'
        cos = f"{r['cosine_cpu_vs_matlab']:.4f}" if r['cosine_cpu_vs_matlab'] is not None else '---'
        case_escaped = r['case'].replace('_', r'\_')
        sim_escaped = r['simulator'].replace('_', r'\_')
        lines.append(f"{sim_escaped} & {case_escaped} & {ml} & {cpu} & {gpu} & {sp} & {cos} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    out_path.write_text('\n'.join(lines))
    print(f"  Wrote: {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', default='../results',
                    help='Directory containing comparison.json')
    args = ap.parse_args()

    rd = Path(args.results_dir).resolve()
    cmp_json = rd / 'comparison.json'
    if not cmp_json.exists():
        print(f"comparison.json not found at {cmp_json}; run compare_results.py first")
        return 1
    rows = json.loads(cmp_json.read_text())

    fig_dir = rd / 'figures'
    tab_dir = rd / 'tables'
    fig_dir.mkdir(exist_ok=True)
    tab_dir.mkdir(exist_ok=True)

    print("Generating figures and tables...")
    fig_timing_bars(rows, fig_dir / 'fig_timing.png')
    fig_speedup(rows, fig_dir / 'fig_speedup.png')
    fig_accuracy(rows, fig_dir / 'fig_accuracy.png')
    latex_table(rows, tab_dir / 'benchmark_table.tex')

    print("\nDone. Figures + tables in:")
    print(f"  {fig_dir}")
    print(f"  {tab_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
