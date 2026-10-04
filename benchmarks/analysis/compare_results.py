"""Compare MATLAB / torchspin-CPU / torchspin-GPU benchmark outputs.

Loads:
  results/matlab_timings.mat  + matlab_outputs.mat
  results/<cpu-prefix>.json   + <cpu-prefix>.npz
  results/<gpu-prefix>.json   + <gpu-prefix>.npz   (optional)

Writes:
  results/comparison.json     parity + speedup metrics per case
  results/comparison_table.csv  human-readable table

Computes for each (simulator, case):
  - wall_time:  MATLAB / Python-CPU / Python-GPU best-of-N
  - speedup:    MATLAB / Python-CPU and Python-CPU / Python-GPU
  - cosine:     Python output vs MATLAB output (real part; scale-invariant)
  - scaled_nrmse: residual norm after fitting one global amplitude factor
  - scale_matlab_per_cpu: fitted amplitude conversion factor

Arrays must have identical shapes. The comparison never truncates arrays to
make unlike outputs appear comparable.

Usage:
    python compare_results.py [--results-dir ../results/] \
        [--cpu-prefix python_cpu] [--gpu-prefix python_gpu]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

try:
    from scipy.io import loadmat
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na * nb > 1e-30 else 0.0


def _scaled_errors(py: np.ndarray, ml: np.ndarray) -> tuple[float, float, float]:
    """Return (scale, normalized RMS residual, scaled maximum residual).

    ``scale`` minimizes ``||scale*py - ml||_2``. The two residual metrics are
    normalized by the corresponding norm and maximum magnitude of MATLAB's
    output. This separates shape agreement from documented intensity-convention
    differences.
    """
    py = np.asarray(py, dtype=float).ravel()
    ml = np.asarray(ml, dtype=float).ravel()
    if py.shape != ml.shape:
        raise ValueError(f"shape mismatch: {py.shape} vs {ml.shape}")
    py_power = float(np.dot(py, py))
    scale = float(np.dot(py, ml) / py_power) if py_power > 1e-30 else 0.0
    residual = scale * py - ml
    nrmse = float(np.linalg.norm(residual) / max(np.linalg.norm(ml), 1e-30))
    max_scaled_error = float(
        np.max(np.abs(residual)) / max(np.max(np.abs(ml)), 1e-30)
    )
    return scale, nrmse, max_scaled_error


def load_matlab(results_dir: Path) -> tuple[dict, dict]:
    timings_path = results_dir / 'matlab_timings.mat'
    outputs_path = results_dir / 'matlab_outputs.mat'
    if not timings_path.exists() or not outputs_path.exists():
        return {}, {}
    if not HAS_SCIPY:
        print("scipy not installed; cannot load MATLAB .mat files")
        return {}, {}
    timings = loadmat(str(timings_path), simplify_cells=True)['results']
    outputs = loadmat(str(outputs_path), simplify_cells=True)['outputs']
    return timings, outputs


def load_python(prefix: Path) -> tuple[dict, dict]:
    json_path = prefix.with_suffix('.json')
    npz_path = prefix.with_suffix('.npz')
    if not json_path.exists() or not npz_path.exists():
        return {}, {}
    meta = json.loads(json_path.read_text())
    arrs = dict(np.load(npz_path, allow_pickle=True))
    return meta, arrs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', default='../results',
                    help='Directory containing matlab/python output files')
    ap.add_argument('--cpu-prefix', default='python_cpu',
                    help='Python CPU filename prefix within results-dir')
    ap.add_argument('--gpu-prefix', default='python_gpu',
                    help='Python GPU filename prefix within results-dir')
    ap.add_argument('--output-stem', default='comparison',
                    help='Stem for output JSON and CSV files')
    args = ap.parse_args()

    rd = Path(args.results_dir).resolve()
    print(f"Reading results from: {rd}")

    matlab_timings, matlab_outputs = load_matlab(rd)
    py_cpu_meta, py_cpu_arr = load_python(rd / args.cpu_prefix)
    py_gpu_meta, py_gpu_arr = load_python(rd / args.gpu_prefix)

    print(f"MATLAB cases: {sum(len(v) for v in matlab_timings.values()) if matlab_timings else 0}")
    print(f"Python CPU cases: {sum(len(v) for v in py_cpu_meta.get('results', {}).values())}")
    print(f"Python GPU cases: {sum(len(v) for v in py_gpu_meta.get('results', {}).values())}")

    rows = []
    cpu_results = py_cpu_meta.get('results', {}) if py_cpu_meta else {}
    gpu_results = py_gpu_meta.get('results', {}) if py_gpu_meta else {}

    for sim, sim_cases in cpu_results.items():
        for case_name, cpu_metrics in sim_cases.items():
            row = {
                'simulator': sim,
                'case': case_name,
                'matlab_best_s': None,
                'python_cpu_best_s': cpu_metrics.get('best_s'),
                'python_gpu_best_s': None,
                'cpu_speedup_vs_matlab': None,
                'gpu_speedup_vs_cpu': None,
                'cosine_cpu_vs_matlab': None,
                'cosine_gpu_vs_cpu': None,
                'shape_match_cpu_vs_matlab': None,
                'scale_matlab_per_cpu': None,
                'scaled_nrmse_cpu_vs_matlab': None,
                'scaled_max_err_cpu_vs_matlab': None,
                'cpu_fit_max_abs_err_vs_truth': None,
                'matlab_fit_max_abs_err_vs_truth': None,
                'cpu_vs_matlab_fit_max_abs_err': None,
                'cpu_convergence': cpu_metrics.get('convergence', False),
                'gpu_convergence': None,
                'gpu_error': None,
            }
            # MATLAB timing (handle MATLAB-friendly key prefix 'x' for numeric-leading names)
            matlab_key = case_name
            if sim in matlab_timings and matlab_key not in matlab_timings[sim]:
                matlab_key = 'x' + case_name
            if sim in matlab_timings and matlab_key in matlab_timings[sim]:
                ml = matlab_timings[sim][matlab_key]
                row['matlab_best_s'] = float(ml.get('best_s', np.nan))
                if cpu_metrics.get('best_s') and row['matlab_best_s']:
                    row['cpu_speedup_vs_matlab'] = row['matlab_best_s'] / cpu_metrics['best_s']
            # GPU timing
            if sim in gpu_results and case_name in gpu_results[sim]:
                gm = gpu_results[sim][case_name]
                row['gpu_convergence'] = gm.get('convergence', False)
                row['gpu_error'] = gm.get('error')
                row['python_gpu_best_s'] = gm.get('best_s')
                if cpu_metrics.get('best_s') and gm.get('best_s'):
                    row['gpu_speedup_vs_cpu'] = cpu_metrics['best_s'] / gm['best_s']
            # Numerical comparison
            arr_key = f"{sim}__{case_name}"
            cpu_arr = py_cpu_arr.get(arr_key)
            ml_out_key = f"{sim}__{matlab_key}" if sim in matlab_timings else None
            ml_arr = matlab_outputs.get(ml_out_key) if matlab_outputs else None
            if cpu_arr is not None and ml_arr is not None:
                # MATLAB output is a struct — extract the spectrum/signal
                if isinstance(ml_arr, dict):
                    candidate_keys = ['spc', 'y', 'y1', 'spec']
                    ml_y = None
                    for k in candidate_keys:
                        if k in ml_arr:
                            ml_y = np.asarray(ml_arr[k])
                            break
                    if ml_y is not None:
                        py_y = np.asarray(cpu_arr).ravel()
                        ml_yflat = ml_y.ravel()
                        row['shape_match_cpu_vs_matlab'] = (
                            np.asarray(cpu_arr).shape == np.asarray(ml_y).shape
                        )
                        if row['shape_match_cpu_vs_matlab']:
                            py_real = np.real(py_y)
                            ml_real = np.real(ml_yflat)
                            row['cosine_cpu_vs_matlab'] = _cos(py_real, ml_real)
                            scale, nrmse, max_err = _scaled_errors(py_real, ml_real)
                            row['scale_matlab_per_cpu'] = scale
                            row['scaled_nrmse_cpu_vs_matlab'] = nrmse
                            row['scaled_max_err_cpu_vs_matlab'] = max_err
                if sim == 'esfit' and 'fit_struct' in ml_arr:
                    fit_struct = ml_arr['fit_struct']
                    if isinstance(fit_struct, dict) and 'pfit' in fit_struct:
                        cpu_fit = np.sort(np.asarray(cpu_arr, dtype=float).ravel())
                        ml_fit = np.sort(np.asarray(fit_struct['pfit'], dtype=float).ravel())
                        truth = np.sort(np.asarray(cpu_metrics.get('p_true', []), dtype=float).ravel())
                        if cpu_fit.shape == truth.shape:
                            row['cpu_fit_max_abs_err_vs_truth'] = float(
                                np.max(np.abs(cpu_fit - truth))
                            )
                        if ml_fit.shape == truth.shape:
                            row['matlab_fit_max_abs_err_vs_truth'] = float(
                                np.max(np.abs(ml_fit - truth))
                            )
                        if cpu_fit.shape == ml_fit.shape:
                            row['cpu_vs_matlab_fit_max_abs_err'] = float(
                                np.max(np.abs(cpu_fit - ml_fit))
                            )
            # GPU vs CPU consistency
            gpu_arr = py_gpu_arr.get(arr_key) if py_gpu_arr else None
            if gpu_arr is not None and cpu_arr is not None:
                if np.asarray(gpu_arr).shape == np.asarray(cpu_arr).shape:
                    row['cosine_gpu_vs_cpu'] = _cos(
                        np.real(np.asarray(gpu_arr).ravel()),
                        np.real(np.asarray(cpu_arr).ravel())
                    )
            rows.append(row)

    # Write JSON
    out_json = rd / f'{args.output_stem}.json'
    out_json.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nWrote: {out_json}")

    # Write CSV
    out_csv = rd / f'{args.output_stem}_table.csv'
    if rows:
        with out_csv.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote: {out_csv}")

    # Print summary
    print("\n=== Summary ===")
    print(f"{'simulator/case':<35} {'matlab(ms)':>10} {'cpu(ms)':>10} "
          f"{'gpu(ms)':>10} {'cpu/ml':>7} {'cosine':>8}")
    for r in rows:
        ml_ms = f"{r['matlab_best_s']*1000:.1f}" if r['matlab_best_s'] else '—'
        cpu_ms = f"{r['python_cpu_best_s']*1000:.1f}" if r['python_cpu_best_s'] else '—'
        gpu_ms = f"{r['python_gpu_best_s']*1000:.1f}" if r['python_gpu_best_s'] else '—'
        sp = f"{r['cpu_speedup_vs_matlab']:.2f}x" if r['cpu_speedup_vs_matlab'] else '—'
        cos = f"{r['cosine_cpu_vs_matlab']:.4f}" if r['cosine_cpu_vs_matlab'] is not None else '—'
        print(f"{r['simulator']+'/'+r['case']:<35} {ml_ms:>10} {cpu_ms:>10} "
              f"{gpu_ms:>10} {sp:>7} {cos:>8}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
