"""Generate manuscript figures from the 2026-04-20 benchmark run.

Data is embedded below, transcribed from the 2026-04-20 run log (the post-fix
rerun, after the salt CUDA wiring fix on commit ``68e344e``; the run log itself
is a development record and is not distributed). Running
this script writes 5 PNG figures and a LaTeX parity table into
``benchmarks/results/figures/``.

Usage::

    cd benchmarks/analysis
    python make_manuscript_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------
# Measured data (transcribed from the 2026-04-20 run log, post-fix rerun)
# ----------------------------------------------------------------------

# Section 1: pepper forward simulation
PEPPER_GRID = np.array([11, 19, 31, 51, 76])
PEPPER_CPU_MS = np.array([4.7, 5.3, 7.5, 13.8, 35.3])
PEPPER_GPU_MS = np.array([4.5, 5.6, 7.6, 27.4, 37.6])

# Section 2: differentiable_spectrum forward / forward+backward
DIFF_GRID = np.array([11, 19, 31])
DIFF_CPU_FWD_MS = np.array([292.2, 860.2, 2255.4])
DIFF_CPU_FWDBWD_MS = np.array([680.1, 2014.5, 5427.4])
DIFF_GPU_FWD_MS = np.array([115.3, 323.2, 839.4])
DIFF_GPU_FWDBWD_MS = np.array([294.9, 843.5, 2205.4])

# Section 3: esfit method comparison (2-param powder fit)
# true: gx=2.00, Ax=120.0
ESFIT_METHODS = ['simplex', 'global', 'levmar']
ESFIT_GX = np.array([2.02306, 1.99966, 2.02926])
ESFIT_AX = np.array([127.14, 117.72, 110.00])
ESFIT_RMSD = np.array([2.8060, 0.9743, 3.2231])
ESFIT_TIME_S = np.array([0.8, 23.4, 0.2])
ESFIT_GX_TRUE = 2.00
ESFIT_AX_TRUE = 120.0

# Section 4: 5-parameter stress test (global method)
STRESS_NAMES = [r'$g_x$', r'$g_y$', r'$g_z$', r'$A_x$', 'lwG']
STRESS_TRUE = np.array([2.0000, 2.0500, 2.1000, 80.0000, 0.8000])
STRESS_FIT = np.array([2.00001, 2.04999, 2.10000, 80.03556, 0.80005])
STRESS_ABS_ERR = np.abs(STRESS_TRUE - STRESS_FIT)
STRESS_REL_ERR_PCT = 100.0 * STRESS_ABS_ERR / np.abs(STRESS_TRUE)

# Section 5: differentiable fitting - MSE vs Integral-MSE trajectories
# Sampled at step 0, 50, 100, 150, 199 from the log
DIFF_FIT_STEPS = np.array([0, 50, 100, 150, 199])
DIFF_FIT_MSE_GX = np.array([2.01020, 2.00126, 2.00005, 2.00011, 2.00012])
DIFF_FIT_INT_GX = np.array([2.00980, 2.00004, 1.99143, 1.98355, 1.97631])
DIFF_FIT_TRUE_GX = 2.00000

# MATLAB parity (for the summary table)
PARITY_ROWS = [
    ("pepper",    "CW EPR powder",           "$>0.999$",  r"$\sim$15\% (correction factors)"),
    ("garlic",    "CW solution / fast motion", "1.0000",  "exact"),
    ("chili",     "Slow-motion SLE",         "0.92--0.997", "matches in regime"),
    ("salt",      "ENDOR powder",            "$>0.92$",   "matches in regime"),
    ("endorfrq",  "ENDOR single orientation", r"$\pm 1$ MHz peaks", "exact"),
    ("saffron",   "Pulse EPR (ESEEM/HYSCORE)", "1.0000", "matches"),
    ("curry",     "Magnetometry (χ, M-T)",    "analytical", "exact (SI units)"),
    ("blochsteady", "Bloch steady-state",     "analytical", "exact"),
    ("spidyan",   "Arbitrary pulse (internal)", "self-consistent", "n/a"),
    ("cardamom",  "Trajectory-based (internal)", "self-consistent", "n/a"),
]


# ----------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLOR_CPU = '#3366CC'
COLOR_GPU = '#DC3912'
COLOR_MATLAB = '#109618'


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------

def fig1_pepper_scaling(outdir: Path):
    """pepper forward-sim wall-time, CPU vs GPU, vs GridSize."""
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(PEPPER_GRID, PEPPER_CPU_MS, 'o-', color=COLOR_CPU,
            linewidth=1.6, markersize=6, label='CPU')
    ax.plot(PEPPER_GRID, PEPPER_GPU_MS, 's-', color=COLOR_GPU,
            linewidth=1.6, markersize=6, label='GPU (RTX 4090)')
    ax.set_xlabel('GridSize (SOPHE knots per edge)')
    ax.set_ylabel('Wall time per call (ms)')
    ax.set_title('pepper() single-call latency (S=1/2 + $^{1}$H)')
    ax.legend(loc='upper left', frameon=False)
    ax.set_xticks(PEPPER_GRID)
    fig.tight_layout()
    fig.savefig(outdir / 'fig1_pepper_scaling.png', bbox_inches='tight')
    fig.savefig(outdir / 'fig1_pepper_scaling.pdf', bbox_inches='tight')
    plt.close(fig)


def fig2_diffspec_gpu_speedup(outdir: Path):
    """differentiable_spectrum forward and forward+backward, CPU vs GPU."""
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4), sharey=False)
    x = np.arange(len(DIFF_GRID))
    w = 0.35

    axes[0].bar(x - w/2, DIFF_CPU_FWD_MS, w, color=COLOR_CPU, label='CPU')
    axes[0].bar(x + w/2, DIFF_GPU_FWD_MS, w, color=COLOR_GPU, label='GPU')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'N={n}' for n in DIFF_GRID])
    axes[0].set_ylabel('Forward pass (ms)')
    axes[0].set_title('Forward simulation')
    axes[0].legend(frameon=False)
    for i, (cpu, gpu) in enumerate(zip(DIFF_CPU_FWD_MS, DIFF_GPU_FWD_MS)):
        axes[0].text(i, max(cpu, gpu) * 1.03,
                     f'{cpu/gpu:.1f}\u00d7', ha='center', fontsize=9,
                     fontweight='bold')

    axes[1].bar(x - w/2, DIFF_CPU_FWDBWD_MS, w, color=COLOR_CPU, label='CPU')
    axes[1].bar(x + w/2, DIFF_GPU_FWDBWD_MS, w, color=COLOR_GPU, label='GPU')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'N={n}' for n in DIFF_GRID])
    axes[1].set_ylabel('Forward + backward (ms)')
    axes[1].set_title('Forward + backward (autograd)')
    axes[1].legend(frameon=False)
    for i, (cpu, gpu) in enumerate(zip(DIFF_CPU_FWDBWD_MS, DIFF_GPU_FWDBWD_MS)):
        axes[1].text(i, max(cpu, gpu) * 1.03,
                     f'{cpu/gpu:.1f}\u00d7', ha='center', fontsize=9,
                     fontweight='bold')

    fig.suptitle('differentiable_spectrum() — GPU speedup across grid size',
                 y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / 'fig2_diffspec_gpu_speedup.png', bbox_inches='tight')
    fig.savefig(outdir / 'fig2_diffspec_gpu_speedup.pdf', bbox_inches='tight')
    plt.close(fig)


def fig3_esfit_accuracy_vs_time(outdir: Path):
    """esfit method comparison: accuracy (|gx error|) vs wall time."""
    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    gx_err = np.abs(ESFIT_GX - ESFIT_GX_TRUE)
    ax_err = np.abs(ESFIT_AX - ESFIT_AX_TRUE)

    # plot each method with annotations
    markers = ['o', 's', '^']
    colors = ['#3366CC', '#109618', '#B82E2E']
    for i, (m, method) in enumerate(zip(markers, ESFIT_METHODS)):
        ax.scatter(ESFIT_TIME_S[i], gx_err[i], s=220, c=colors[i], marker=m,
                   edgecolors='black', linewidths=0.8, zorder=3,
                   label=f"{method} (|ΔA|={ax_err[i]:.1f} MHz)")
        ax.annotate(
            f"  {method}",
            (ESFIT_TIME_S[i], gx_err[i]),
            fontsize=10, fontweight='bold',
            xytext=(8, 0), textcoords='offset points',
            va='center',
        )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Wall-clock time (s)')
    ax.set_ylabel(r'$|g_x^{\mathrm{fit}} - g_x^{\mathrm{true}}|$')
    ax.set_title('esfit method comparison (2-param powder fit)\n'
                 'True: $g_x=2.000$, $A_x=120$ MHz; start: $g_x=2.01$, $A_x=110$')
    ax.legend(loc='lower left', frameon=True, fontsize=9)

    # Pareto arrow showing global's dominance
    ax.annotate('',
                xy=(ESFIT_TIME_S[1] * 1.05, gx_err[1] * 0.9),
                xytext=(ESFIT_TIME_S[0] * 0.95, gx_err[0] * 1.1),
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5,
                                linewidth=1))

    fig.tight_layout()
    fig.savefig(outdir / 'fig3_esfit_pareto.png', bbox_inches='tight')
    fig.savefig(outdir / 'fig3_esfit_pareto.pdf', bbox_inches='tight')
    plt.close(fig)


def fig4_stress_recovery(outdir: Path):
    """5-parameter recovery — log absolute error per parameter."""
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    x = np.arange(len(STRESS_NAMES))
    # Make zero-error bars visible (log scale floor)
    eps = 1e-6
    err_plot = np.maximum(STRESS_ABS_ERR, eps)
    colors = ['#3366CC'] * 3 + ['#DC3912', '#FF9900']
    bars = ax.bar(x, err_plot, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(STRESS_NAMES)
    ax.set_ylabel('|fitted − true|  (g-value or MHz or mT)')
    ax.set_title("5-parameter global fit — recovery accuracy\n"
                 "swarm_size=40, max_iter=500, runtime 78.5 s")
    for bar, e in zip(bars, STRESS_ABS_ERR):
        if e < eps:
            label = '= 0'
        elif e < 1e-3:
            label = f'{e:.0e}'
        else:
            label = f'{e:.3f}'
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.15, label,
                ha='center', fontsize=8)
    ax.set_ylim(eps * 0.5, max(STRESS_ABS_ERR) * 5)
    fig.tight_layout()
    fig.savefig(outdir / 'fig4_stress_recovery.png', bbox_inches='tight')
    fig.savefig(outdir / 'fig4_stress_recovery.pdf', bbox_inches='tight')
    plt.close(fig)


def fig5_differentiable_fit_loss(outdir: Path):
    """Differentiable fit trajectory: MSE converges, Integral-MSE diverges."""
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.axhline(DIFF_FIT_TRUE_GX, color='black', linestyle='--',
               linewidth=1.0, alpha=0.6, label=r'true $g_x = 2.000$')
    ax.plot(DIFF_FIT_STEPS, DIFF_FIT_MSE_GX, 'o-', color=COLOR_CPU,
            linewidth=1.6, markersize=6,
            label='raw MSE (final err $1.2\\times10^{-4}$)')
    ax.plot(DIFF_FIT_STEPS, DIFF_FIT_INT_GX, 's-', color=COLOR_GPU,
            linewidth=1.6, markersize=6,
            label='integral-MSE (final err $2.4\\times10^{-2}$)')
    ax.set_xlabel('Adam optimization step')
    ax.set_ylabel(r'Fitted $g_x$')
    ax.set_title('Differentiable fitting — loss choice determines accuracy\n'
                 '(200 Adam steps, lr=2e-4, GPU)')
    ax.legend(loc='center right', frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / 'fig5_diff_fit_loss.png', bbox_inches='tight')
    fig.savefig(outdir / 'fig5_diff_fit_loss.pdf', bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------
# LaTeX parity table
# ----------------------------------------------------------------------

def write_parity_table(outdir: Path):
    """Write a LaTeX booktabs table summarising MATLAB parity."""
    lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\caption{MATLAB EasySpin parity across validated torchspin '
        r'simulators. Cosine similarity is the shape-match metric; '
        r'amplitude refers to absolute-intensity agreement.}',
        r'\label{tab:parity}',
        r'\begin{tabular}{lllc}',
        r'\toprule',
        r'Module & Physics & Cosine & Amplitude \\',
        r'\midrule',
    ]
    for name, physics, cosine, amp in PARITY_ROWS:
        lines.append(f'\\texttt{{{name}}} & {physics} & {cosine} & {amp} \\\\')
    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])
    (outdir / 'table1_parity.tex').write_text('\n'.join(lines) + '\n')


# ----------------------------------------------------------------------

def main():
    here = Path(__file__).resolve().parent
    outdir = here.parent / 'results' / 'figures'
    outdir.mkdir(parents=True, exist_ok=True)

    fig1_pepper_scaling(outdir)
    fig2_diffspec_gpu_speedup(outdir)
    fig3_esfit_accuracy_vs_time(outdir)
    fig4_stress_recovery(outdir)
    fig5_differentiable_fit_loss(outdir)
    write_parity_table(outdir)

    print(f'Wrote figures + LaTeX table to {outdir}')
    for p in sorted(outdir.iterdir()):
        print(f'  {p.name}')


if __name__ == '__main__':
    main()
