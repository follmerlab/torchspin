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
        3×3 real rotation matrix (float64).  If an angle is a tensor that
        requires grad, the matrix is built from torch operations and carries
        the gradient with respect to the angles; otherwise it is computed in
        Python floats (as before) and does not require grad.

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
        parts = (angles_or_alpha, beta, 0.0 if gamma is None else gamma)
    elif isinstance(angles_or_alpha, torch.Tensor):
        parts = None
    else:
        parts = (angles_or_alpha[0], angles_or_alpha[1], angles_or_alpha[2])

    if _on_graph(angles_or_alpha if parts is None else parts):
        return _erot_graph(angles_or_alpha if parts is None else parts, device)

    if parts is None:
        a = angles_or_alpha.detach().double().cpu().tolist()
        alpha, beta, gamma = a[0], a[1], a[2]
    else:
        alpha, beta, gamma = float(parts[0]), float(parts[1]), float(parts[2])

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


def _on_graph(angles) -> bool:
    """True if any angle is a tensor that requires grad (with grad mode on)."""
    if not torch.is_grad_enabled():
        return False
    if isinstance(angles, torch.Tensor):
        return bool(angles.requires_grad)
    return any(isinstance(a, torch.Tensor) and a.requires_grad for a in angles)


def _erot_graph(angles, device) -> torch.Tensor:
    """``erot`` built from torch operations, so the angles keep their gradient.

    Same formula and operation order as the float path in :func:`erot`; used
    only when an angle requires grad, so non-grad results stay bit-identical to
    the float path.  The matrix entries are products of sines and cosines, so
    the gradient is finite everywhere, including the gimbal-lock angles
    beta = 0 and beta = pi.
    """
    if isinstance(angles, torch.Tensor):
        a = angles.to(dtype=torch.float64).reshape(-1)
        alpha, beta, gamma = a[0], a[1], a[2]
    else:
        ref = next(x for x in angles if isinstance(x, torch.Tensor))
        alpha, beta, gamma = (x.to(dtype=torch.float64).reshape(()) if isinstance(x, torch.Tensor)
                              else torch.tensor(float(x), dtype=torch.float64, device=ref.device)
                              for x in angles)
    ca, sa = torch.cos(alpha), torch.sin(alpha)
    cb, sb = torch.cos(beta), torch.sin(beta)
    cg, sg = torch.cos(gamma), torch.sin(gamma)
    R = torch.stack([
        torch.stack([ cg*cb*ca - sg*sa,  cg*cb*sa + sg*ca, -cg*sb]),
        torch.stack([-sg*cb*ca - cg*sa, -sg*cb*sa + cg*ca,  sg*sb]),
        torch.stack([ sb*ca,             sb*sa,              cb   ]),
    ])
    return R.to(device=device)
