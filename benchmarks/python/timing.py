"""Timing utilities for the torchspin benchmark suite.

Best-of-N timing with warmup, plus optional GPU synchronization.
"""
from __future__ import annotations

import gc
import time
from typing import Any, Callable

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def best_of_n(
    fn: Callable,
    n_runs: int = 5,
    n_warmup: int = 1,
    sync_gpu: bool = False,
) -> tuple[float, float, list[float], Any]:
    """Time `fn()` and return best-of-N statistics.

    Parameters
    ----------
    fn : callable
        Zero-arg callable to time. Must return the simulation result.
    n_runs : int
        Number of timed runs (default 5).
    n_warmup : int
        Number of untimed warmup runs (default 1) — important for GPU first-time
        kernel compilation and PyTorch graph caches.
    sync_gpu : bool
        If True and torch.cuda is available, synchronize CUDA before/after each
        timed run for accurate GPU timing.

    Returns
    -------
    best_s : float
        Best (minimum) wall-clock time, in seconds.
    median_s : float
        Median wall-clock time across runs.
    all_s : list[float]
        All N timed runs.
    last_result : Any
        The result of the final run (so the caller can verify correctness).
    """
    # Warmup — discard results and timings
    last = None
    for _ in range(n_warmup):
        gc.collect()
        if sync_gpu and HAS_TORCH and torch.cuda.is_available():
            torch.cuda.synchronize()
        last = fn()
        if sync_gpu and HAS_TORCH and torch.cuda.is_available():
            torch.cuda.synchronize()

    # Timed runs
    times: list[float] = []
    for _ in range(n_runs):
        gc.collect()
        if sync_gpu and HAS_TORCH and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        last = fn()
        if sync_gpu and HAS_TORCH and torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times.sort()
    median = times[len(times) // 2] if len(times) % 2 == 1 else 0.5 * (times[len(times) // 2 - 1] + times[len(times) // 2])
    return times[0], median, times, last


def device_label(device: str) -> str:
    """Human-readable device label for output dicts."""
    if device == 'cpu':
        return 'CPU'
    if device.startswith('cuda'):
        if HAS_TORCH and torch.cuda.is_available():
            try:
                idx = int(device.split(':')[1]) if ':' in device else 0
                return f"GPU ({torch.cuda.get_device_name(idx)})"
            except Exception:
                return device
        return device
    return device
