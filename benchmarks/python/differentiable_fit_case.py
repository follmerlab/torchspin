"""Reproducible differentiable EPR fitting case study.

The case fits three rhombic principal g values and a Gaussian linewidth to a
synthetic first-derivative powder spectrum.  It records parameter recovery,
autograd-versus-central-difference gradient agreement, and the cost of one
four-parameter gradient evaluation.

Usage::

    python benchmarks/python/differentiable_fit_case.py \
        --device cpu --output benchmarks/results/diff_fit_cpu
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Resolve the source checkout when the script is run without an editable install.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torchspin
from torchspin import differentiable_spectrum


DTYPE = torch.float64
TRUE_G = [2.002, 2.006, 2.009]
TRUE_LW_MT = 0.8
INITIAL_G = [2.000, 2.004, 2.011]
INITIAL_LW_MT = 1.1
G_CENTER = [2.005, 2.005, 2.005]
G_SCALE = 0.01
LW_FLOOR_MT = 0.2
LW_SCALE = 0.5
SIMULATION = {
    'mwFreq_GHz': 9.5,
    'B_range': (330.0, 350.0),
    'nPoints': 1024,
    'Harmonic': 1,
    'GridSize': 31,
}


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def _initial_scaled_parameters(device: torch.device) -> torch.Tensor:
    initial_g = torch.tensor(INITIAL_G, dtype=DTYPE, device=device)
    center = torch.tensor(G_CENTER, dtype=DTYPE, device=device)
    q_g = (initial_g - center) / G_SCALE
    q_lw = _inverse_softplus((INITIAL_LW_MT - LW_FLOOR_MT) / LW_SCALE)
    return torch.cat([
        q_g,
        torch.tensor([q_lw], dtype=DTYPE, device=device),
    ])


def _physical_parameters(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    center = torch.tensor(G_CENTER, dtype=DTYPE, device=q.device)
    g = center + G_SCALE * q[:3]
    lw = LW_FLOOR_MT + LW_SCALE * torch.nn.functional.softplus(q[3])
    return g, lw


def _normalized_spectrum(
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g, lw = _physical_parameters(q)
    B, spectrum = differentiable_spectrum(g, lw_mT=lw, **SIMULATION)
    spectrum = spectrum / spectrum.norm().clamp_min(1e-30)
    return B, spectrum, g, lw


def _objective(q: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _, spectrum, _, _ = _normalized_spectrum(q)
    return ((spectrum - target) ** 2).mean()


def _sync(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _timed_best(fn, device: torch.device, repeats: int) -> tuple[float, list[float]]:
    for _ in range(3):
        fn()
    timings = []
    for _ in range(repeats):
        _sync(device)
        start = time.perf_counter()
        fn()
        _sync(device)
        timings.append(time.perf_counter() - start)
    return min(timings), timings


def _git_commit() -> str | None:
    result = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', required=True)
    parser.add_argument('--timing-repeats', type=int, default=20)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available.')

    true_g = torch.tensor(TRUE_G, dtype=DTYPE, device=device)
    with torch.no_grad():
        B, target = differentiable_spectrum(
            true_g, lw_mT=TRUE_LW_MT, device=str(device), **SIMULATION
        )
        target = target / target.norm().clamp_min(1e-30)

    q_initial = _initial_scaled_parameters(device)
    with torch.no_grad():
        _, initial_spectrum, initial_g, initial_lw = _normalized_spectrum(q_initial)
        initial_loss = _objective(q_initial, target).item()

    q_fit = q_initial.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [q_fit],
        lr=0.5,
        max_iter=300,
        tolerance_grad=1e-12,
        tolerance_change=1e-14,
        line_search_fn='strong_wolfe',
    )
    history: list[float] = []

    def closure():
        optimizer.zero_grad()
        loss = _objective(q_fit, target)
        loss.backward()
        history.append(float(loss.detach().cpu()))
        return loss

    _sync(device)
    fit_start = time.perf_counter()
    optimizer.step(closure)
    _sync(device)
    fit_time_s = time.perf_counter() - fit_start

    with torch.no_grad():
        _, final_spectrum, fitted_g, fitted_lw = _normalized_spectrum(q_fit)
        final_loss = _objective(q_fit, target).item()
        cosine = torch.dot(final_spectrum, target) / (
            final_spectrum.norm() * target.norm()
        )

    # Compare one autograd gradient to a four-parameter central difference at
    # the same initial point.
    delta = 1e-5

    def autograd_gradient() -> torch.Tensor:
        q = q_initial.clone().requires_grad_(True)
        loss = _objective(q, target)
        gradient, = torch.autograd.grad(loss, q)
        return gradient.detach()

    def finite_difference_gradient() -> torch.Tensor:
        values = []
        with torch.no_grad():
            for i in range(q_initial.numel()):
                q_plus = q_initial.clone()
                q_minus = q_initial.clone()
                q_plus[i] += delta
                q_minus[i] -= delta
                values.append(
                    (_objective(q_plus, target) - _objective(q_minus, target))
                    / (2.0 * delta)
                )
        return torch.stack(values)

    grad_ag = autograd_gradient()
    grad_fd = finite_difference_gradient()
    grad_cosine = torch.dot(grad_ag, grad_fd) / (
        grad_ag.norm() * grad_fd.norm() + 1e-30
    )
    grad_relative_norm_error = (grad_ag - grad_fd).norm() / (
        grad_fd.norm() + 1e-30
    )

    ag_best_s, ag_all_s = _timed_best(
        autograd_gradient, device, args.timing_repeats
    )
    fd_best_s, fd_all_s = _timed_best(
        finite_difference_gradient, device, args.timing_repeats
    )

    fitted_g_sorted = torch.sort(fitted_g.detach()).values
    true_g_sorted = torch.sort(true_g).values
    max_g_error = torch.max(torch.abs(fitted_g_sorted - true_g_sorted))

    output_prefix = Path(args.output)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    result = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'hostname': platform.node(),
        'platform': platform.platform(),
        'git_commit': _git_commit(),
        'torchspin_version': torchspin.__version__,
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'device': str(device),
        'device_name': (
            torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'
        ),
        'simulation': SIMULATION,
        'true_g': TRUE_G,
        'true_lw_mT': TRUE_LW_MT,
        'initial_g': initial_g.detach().cpu().tolist(),
        'initial_lw_mT': float(initial_lw.detach().cpu()),
        'fitted_g': fitted_g.detach().cpu().tolist(),
        'fitted_g_sorted': fitted_g_sorted.cpu().tolist(),
        'fitted_lw_mT': float(fitted_lw.detach().cpu()),
        'max_abs_g_error': float(max_g_error.cpu()),
        'abs_lw_error_mT': abs(float(fitted_lw.detach().cpu()) - TRUE_LW_MT),
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'final_spectrum_cosine': float(cosine.cpu()),
        'optimizer': 'torch.optim.LBFGS',
        'closure_evaluations': len(history),
        'fit_time_s': fit_time_s,
        'gradient_fd_delta_scaled_coordinates': delta,
        'gradient_cosine_autograd_vs_fd': float(grad_cosine.cpu()),
        'gradient_relative_norm_error': float(grad_relative_norm_error.cpu()),
        'autograd_gradient_best_s': ag_best_s,
        'autograd_gradient_all_s': ag_all_s,
        'finite_difference_gradient_best_s': fd_best_s,
        'finite_difference_gradient_all_s': fd_all_s,
        'fd_over_autograd_gradient_cost': fd_best_s / ag_best_s,
    }
    output_prefix.with_suffix('.json').write_text(json.dumps(result, indent=2))
    np.savez_compressed(
        output_prefix.with_suffix('.npz'),
        B=B.detach().cpu().numpy(),
        target=target.detach().cpu().numpy(),
        initial=initial_spectrum.detach().cpu().numpy(),
        fitted=final_spectrum.detach().cpu().numpy(),
        loss_history=np.asarray(history),
        gradient_autograd=grad_ag.detach().cpu().numpy(),
        gradient_finite_difference=grad_fd.detach().cpu().numpy(),
    )

    print(json.dumps(result, indent=2))
    print(f"Wrote {output_prefix.with_suffix('.json')}")
    print(f"Wrote {output_prefix.with_suffix('.npz')}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
