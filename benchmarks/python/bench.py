"""Consolidated Python benchmark runner for torchspin.

Runs all benchmark cases (pepper, garlic, chili, saffron, esfit, autograd)
and writes timing + output data to a single JSON / .npz pair.

Usage::

    python bench.py --device cpu --output ../results/python_cpu
    python bench.py --device cuda --output ../results/python_gpu
    python bench.py --simulator pepper --output ../results/cpu_pepper_only

The output JSON has shape::

    {
      "device": "CPU" | "GPU (NVIDIA ...)",
      "torchspin_version": "0.1.0",
      "torch_version": "2.x.y",
      "timestamp": "ISO-8601",
      "results": {
        "<simulator>": {
          "<case_name>": {
            "wall_time_best_s": float,
            "wall_time_median_s": float,
            "wall_time_all_s": [float, ...],
            "n_runs": int,
            "n_warmup": int,
            "convergence": bool,
            "error": str | null,
            "output_shape": list[int]
          }
        }
      }
    }

The companion .npz file holds the actual simulator output arrays for
parity comparison against MATLAB output.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# Make the common/ folder importable, plus the repo root so `torchspin` resolves
# when the script is invoked directly (without `pip install -e .`).
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))                   # benchmarks/
sys.path.insert(0, str(_HERE.parent.parent))            # repo root
from common.spin_systems import (PEPPER_CASES, GARLIC_CASES, CHILI_CASES,
                                  SAFFRON_CASES, ESFIT_CASES)
from timing import best_of_n, device_label

import torch

import torchspin
from torchspin import (SpinSystem, Experiment, Options,
                       pepper, garlic, chili, ChiliOptions,
                       esfit, FitOptions)
from torchspin.saffron import saffron, PulseExperiment, SaffronOptions


# ---------------------------------------------------------------------------
# Benchmark runners (one per simulator)
# ---------------------------------------------------------------------------

def _make_sys(d: dict) -> SpinSystem:
    """Build a SpinSystem from a dict."""
    return SpinSystem(**d)


def _make_exp(d: dict) -> Experiment:
    return Experiment(**d)


def _make_opt(d: dict | None, device: str) -> Options:
    d = dict(d) if d else {}
    d.setdefault('Verbosity', 0)
    d['device'] = device
    return Options(**d)


def bench_pepper(case_name: str, case: dict, device: str) -> tuple[dict, np.ndarray]:
    sys_obj = _make_sys(case['sys'])
    exp_obj = _make_exp(case['exp'])
    opt_obj = _make_opt(case.get('opt'), device)

    def run():
        return pepper(sys_obj, exp_obj, opt_obj)

    best, median, all_t, (B, spc) = best_of_n(run, n_runs=5, n_warmup=1,
                                              sync_gpu=device.startswith('cuda'))
    spc_np = spc.cpu().numpy() if hasattr(spc, 'cpu') else np.asarray(spc)
    return {
        'best_s': best, 'median_s': median, 'all_s': all_t,
        'output_shape': list(spc_np.shape),
        'output_max_abs': float(np.max(np.abs(spc_np))),
    }, spc_np


def bench_garlic(case_name: str, case: dict, device: str) -> tuple[dict, np.ndarray]:
    sys_obj = _make_sys(case['sys'])
    exp_obj = _make_exp(case['exp'])
    # garlic ignores Options.device for its perturb path, but pass it for consistency
    opt_obj = _make_opt({}, device)

    def run():
        return garlic(sys_obj, exp_obj, opt_obj)

    best, median, all_t, (B, spc) = best_of_n(run, n_runs=5, n_warmup=1,
                                              sync_gpu=device.startswith('cuda'))
    spc_np = spc.cpu().numpy() if hasattr(spc, 'cpu') else np.asarray(spc)
    return {
        'best_s': best, 'median_s': median, 'all_s': all_t,
        'output_shape': list(spc_np.shape),
        'output_max_abs': float(np.max(np.abs(spc_np))),
    }, spc_np


def bench_chili(case_name: str, case: dict, device: str) -> tuple[dict, np.ndarray]:
    sys_obj = _make_sys(case['sys'])
    exp_obj = _make_exp(case['exp'])
    opt_dict = dict(case.get('opt', {}))
    opt_obj = ChiliOptions(**opt_dict)

    def run():
        return chili(sys_obj, exp_obj, opt_obj)

    best, median, all_t, (B, spc) = best_of_n(run, n_runs=3, n_warmup=1,
                                              sync_gpu=device.startswith('cuda'))
    spc_np = np.asarray(spc) if not hasattr(spc, 'cpu') else spc.cpu().numpy()
    return {
        'best_s': best, 'median_s': median, 'all_s': all_t,
        'output_shape': list(spc_np.shape),
        'output_max_abs': float(np.max(np.abs(spc_np))),
    }, spc_np


def bench_saffron(case_name: str, case: dict, device: str) -> tuple[dict, np.ndarray]:
    sys_obj = _make_sys(case['sys'])
    exp_obj = PulseExperiment(**case['exp'])
    opt_dict = dict(case.get('opt', {}))
    opt_obj = SaffronOptions(**opt_dict, device=device)

    def run():
        return saffron(sys_obj, exp_obj, opt_obj)

    best, median, all_t, (x, signal, info) = best_of_n(
        run, n_runs=3, n_warmup=1, sync_gpu=device.startswith('cuda'))
    sig_np = np.asarray(signal)
    return {
        'best_s': best, 'median_s': median, 'all_s': all_t,
        'output_shape': list(sig_np.shape),
        'output_max_abs': float(np.max(np.abs(sig_np))),
    }, sig_np


def bench_esfit(case_name: str, case: dict, device: str) -> tuple[dict, np.ndarray]:
    # Build the synthetic data from the "true" system
    sys_true = _make_sys(case['sys_true'])
    exp_obj = _make_exp(case['exp'])
    opt_obj = _make_opt(case.get('opt'), device)
    _, y_true = pepper(sys_true, exp_obj, opt_obj)
    y_true_np = y_true.cpu().numpy() if hasattr(y_true, 'cpu') else np.asarray(y_true)

    # Define the model function for fitting (closes over exp_obj, opt_obj)
    def model_fn(p):
        sys_mod = SpinSystem(S=case['sys_true']['S'],
                             g=[[float(p[0]), float(p[1]), float(p[2])]],
                             lw=case['sys_true']['lw'])
        _, spc = pepper(sys_mod, exp_obj, opt_obj)
        return spc.cpu().numpy() if hasattr(spc, 'cpu') else np.asarray(spc)

    fit_opts = FitOptions(**case['fit_options'])
    p0 = np.asarray(case['p0_init'], dtype=float)
    lb = np.asarray(case['p_lb'], dtype=float)
    ub = np.asarray(case['p_ub'], dtype=float)

    def run():
        return esfit(y_true_np, model_fn, p0=p0, lb=lb, ub=ub, options=fit_opts)

    best, median, all_t, result = best_of_n(run, n_runs=3, n_warmup=1,
                                             sync_gpu=device.startswith('cuda'))
    return {
        'best_s': best, 'median_s': median, 'all_s': all_t,
        'pfit': result.pfit.tolist(),
        'p_true': case['sys_true']['g'][0],
    }, np.asarray(result.pfit)


def bench_autograd(case_name: str, case: dict, device: str) -> tuple[dict, np.ndarray]:
    """torchspin-only benchmark: gradient computation through pepper-like spectrum."""
    from torchspin import differentiable_spectrum
    g = torch.tensor([2.009, 2.006, 2.002], dtype=torch.float64,
                     device=device, requires_grad=True)

    def run():
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        B, spec = differentiable_spectrum(
            g, mwFreq_GHz=9.5, B_range=(330, 350),
            nPoints=case.get('nPoints', 256),
            lw_mT=case.get('lw_mT', 1.0),
            GridSize=case.get('GridSize', 31),
            device=device,
        )
        loss = (spec ** 2).sum()
        if g.grad is not None:
            g.grad = None  # zero before backward
        loss.backward()
        return g.grad.detach().cpu().clone()

    best, median, all_t, grad = best_of_n(
        run, n_runs=3, n_warmup=1, sync_gpu=device.startswith('cuda'))
    grad_np = grad.numpy()
    return {
        'best_s': best, 'median_s': median, 'all_s': all_t,
        'grad': grad_np.tolist(),
        'grad_norm': float(np.linalg.norm(grad_np)),
    }, grad_np


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

_RUNNERS = {
    'pepper': (bench_pepper, PEPPER_CASES),
    'garlic': (bench_garlic, GARLIC_CASES),
    'chili':  (bench_chili,  CHILI_CASES),
    'saffron': (bench_saffron, SAFFRON_CASES),
    'esfit':  (bench_esfit,  ESFIT_CASES),
    'autograd': (bench_autograd, {
        'g_grad_pepper_like': {'description': 'Gradient through differentiable_spectrum',
                                'nPoints': 256, 'GridSize': 31, 'lw_mT': 1.0},
    }),
}


def main() -> int:
    ap = argparse.ArgumentParser(description='torchspin benchmark runner')
    ap.add_argument('--device', default='cpu', help='Device: cpu | cuda | cuda:0')
    ap.add_argument('--output', required=True,
                    help='Output prefix (writes <prefix>.json and <prefix>.npz)')
    ap.add_argument('--simulator', default='all',
                    help='Simulator: all | pepper | garlic | chili | saffron | esfit | autograd')
    args = ap.parse_args()

    out_prefix = Path(args.output)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Validate device
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print(f"WARNING: CUDA requested but not available; falling back to CPU.")
        args.device = 'cpu'

    sims = list(_RUNNERS.keys()) if args.simulator == 'all' else [args.simulator]

    results: dict = {}
    arrays: dict = {}
    for sim in sims:
        runner, cases = _RUNNERS[sim]
        results[sim] = {}
        for case_name, case in cases.items():
            print(f"  [{sim}/{case_name}] running on {args.device}... ", end='', flush=True)
            t0 = time.time()
            try:
                metrics, arr = runner(case_name, case, args.device)
                metrics['convergence'] = True
                metrics['error'] = None
                arrays[f"{sim}__{case_name}"] = arr
                print(f"best={metrics['best_s']*1000:.1f}ms ({time.time()-t0:.1f}s wall)")
            except Exception as e:
                metrics = {
                    'convergence': False,
                    'error': f"{type(e).__name__}: {e}",
                    'traceback': traceback.format_exc(),
                }
                print(f"FAILED: {e}")
            results[sim][case_name] = metrics

    # Write JSON metadata
    out_json = out_prefix.with_suffix('.json')
    out_json.write_text(json.dumps({
        'device': device_label(args.device),
        'torchspin_version': torchspin.__version__,
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'results': results,
    }, indent=2))

    # Write arrays
    out_npz = out_prefix.with_suffix('.npz')
    np.savez_compressed(out_npz, **arrays)

    print(f"\nWrote: {out_json}")
    print(f"Wrote: {out_npz}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
