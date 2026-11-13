"""3x3 adjugate for the second-order hyperfine term of the Iwasaki perturbation modules.

The second-order hyperfine invariant A2 (Iwasaki 1974; EasySpin's resfields_perturb.m,
resfreqs_perturb.m, endorfrq_perturb.m) is written det(A) * u' * inv(A) * k.  Only the
product det(A) * inv(A) = adj(A) enters, and the adjugate is a polynomial in the
entries of A: it is defined (and smooth) for a singular A, where it gives the exact
A -> 0 limit of a principal value.  ``resfields_perturb``, ``resfreqs_perturb`` and
``endorfrq_perturb`` use it in place of det(A) * inv(A).
"""
from __future__ import annotations

import torch


def _adjugate3(M: torch.Tensor) -> torch.Tensor:
    """Adjugate (transpose of the cofactor matrix) of a 3x3 matrix, batched over ``...``.

    ``_adjugate3(M) @ M == M @ _adjugate3(M) == det(M) * I`` for any ``M``, singular or
    not; for a non-singular ``M`` it equals ``det(M) * inv(M)`` up to roundoff.  Plain
    products and differences of the entries, so it is differentiable everywhere.
    """
    if M.shape[-2:] != (3, 3):
        raise ValueError(f"_adjugate3 needs (..., 3, 3), got {tuple(M.shape)}")
    a, b, c = M[..., 0, 0], M[..., 0, 1], M[..., 0, 2]
    d, e, f = M[..., 1, 0], M[..., 1, 1], M[..., 1, 2]
    g, h, i = M[..., 2, 0], M[..., 2, 1], M[..., 2, 2]
    rows = (
        torch.stack((e * i - f * h, c * h - b * i, b * f - c * e), dim=-1),
        torch.stack((f * g - d * i, a * i - c * g, c * d - a * f), dim=-1),
        torch.stack((d * h - e * g, b * g - a * h, a * e - b * d), dim=-1),
    )
    return torch.stack(rows, dim=-2)
