"""Rotation utilities for torchspin.

Implements the same z-y'-z'' passive Euler rotation convention as EasySpin's
``erot``.  The rotation matrix ``R`` transforms a vector (or matrix) from
frame 1 to frame 2::

    vec2 = R @ vec1
    mat2 = R @ mat1 @ R.T

The angles (alpha, beta, gamma) rotate the coordinate system sequentially
around z (by alpha), then y' (by beta), then z'' (by gamma).
"""
from __future__ import annotations

import math
import torch


def erot(
    angles_or_alpha,
    beta=None,
    gamma=None,
    device: str = 'cpu',
) -> torch.Tensor:
    """Passive rotation matrix from Euler angles (z-y'-z'' convention).

    Matches EasySpin's ``erot`` function exactly.  The matrix is real-valued.

    Parameters
    ----------
    angles_or_alpha:
        Either a 3-element sequence/tensor ``(alpha, beta, gamma)`` in radians,
        or just ``alpha`` when *beta* and *gamma* are given as separate args.
    beta:
        Second Euler angle in radians (only used when passing 3 separate args).
    gamma:
        Third Euler angle in radians (only used when passing 3 separate args).
    device:
        PyTorch device string.

    Returns
    -------
    torch.Tensor
        3×3 real rotation matrix (float64).

    Notes
    -----
    The explicit decomposition is::

        Rg = Rz(gamma)
        Rb = Ry(beta)
        Ra = Rz(alpha)
        R  = Rg @ Rb @ Ra

    which matches the formula in ``erot.m``::

        R = [ cg*cb*ca-sg*sa,   cg*cb*sa+sg*ca,  -cg*sb;
             -sg*cb*ca-cg*sa,  -sg*cb*sa+cg*ca,   sg*sb;
              sb*ca,            sb*sa,             cb ]
    """
    if beta is not None:
        # Three-argument form: erot(alpha, beta, gamma)
        alpha = float(angles_or_alpha)
        beta  = float(beta)
        gamma = float(gamma) if gamma is not None else 0.0
    elif isinstance(angles_or_alpha, torch.Tensor):
        a = angles_or_alpha.double().cpu().tolist()
        alpha, beta, gamma = a[0], a[1], a[2]
    else:
        alpha = float(angles_or_alpha[0])
        beta  = float(angles_or_alpha[1])
        gamma = float(angles_or_alpha[2])

    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta),  math.sin(beta)   # type: ignore[arg-type]
    cg, sg = math.cos(gamma), math.sin(gamma)  # type: ignore[arg-type]

    R = torch.tensor(
        [
            [ cg*cb*ca - sg*sa,  cg*cb*sa + sg*ca, -cg*sb],
            [-sg*cb*ca - cg*sa, -sg*cb*sa + cg*ca,  sg*sb],
            [ sb*ca,             sb*sa,              cb   ],
        ],
        dtype=torch.float64,
        device=device,
    )
    return R
