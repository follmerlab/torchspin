"""Stick spectrum builder for torchspin.

Port of EasySpin's ``makespec.m``.

Bins peak positions and amplitudes into a uniformly-spaced spectral array
using scatter-add (GPU-friendly).
"""
from __future__ import annotations

from typing import Optional

import torch

from torchspin._compile import maybe_compile


@maybe_compile
def _bin_peaks(
    x: torch.Tensor,
    spec: torch.Tensor,
    pos: torch.Tensor,
    amp: torch.Tensor,
    n_points: int,
) -> torch.Tensor:
    """Inner kernel: sub-bin scatter-add accumulation."""
    Bmin = x[0]
    Bmax = x[-1]
    frac_idx = (n_points - 1) * (pos - Bmin) / (Bmax - Bmin)
    idx_lo = frac_idx.long()
    idx_hi = idx_lo + 1
    alpha = frac_idx - idx_lo.to(torch.float64)

    mask_lo = (idx_lo >= 0) & (idx_lo < n_points)
    mask_hi = (idx_hi >= 0) & (idx_hi < n_points)

    spec.scatter_add_(0, idx_lo[mask_lo], amp[mask_lo] * (1.0 - alpha[mask_lo]))
    spec.scatter_add_(0, idx_hi[mask_hi], amp[mask_hi] * alpha[mask_hi])
    return spec


def makespec(
    field_range: list[float],
    n_points: int,
    pos: torch.Tensor,
    amp: Optional[torch.Tensor] = None,
    *,
    device: str = 'cpu',
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bin peaks into a stick spectrum.

    Parameters
    ----------
    field_range:
        ``[Bmin, Bmax]`` in mT — the sweep range.
    n_points:
        Number of points in the output spectrum.
    pos:
        Peak positions in mT, shape ``(nPeaks,)``.
    amp:
        Peak amplitudes, shape ``(nPeaks,)``.  Defaults to ones.
    device:
        PyTorch device string.

    Returns
    -------
    x:
        Field axis (mT), shape ``(n_points,)``.
    spec:
        Stick spectrum, shape ``(n_points,)``.
    """
    x = torch.linspace(field_range[0], field_range[1], n_points, device=device,
                       dtype=torch.float64)

    spec = torch.zeros(n_points, dtype=torch.float64, device=device)

    if pos.numel() == 0:
        return x, spec

    pos = pos.to(device=device, dtype=torch.float64)
    if amp is None:
        amp = torch.ones(pos.numel(), dtype=torch.float64, device=device)
    else:
        amp = amp.to(device=device, dtype=torch.float64)

    spec = _bin_peaks(x, spec, pos, amp, n_points)

    return x, spec
