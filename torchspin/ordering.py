"""Partially ordered samples (port of EasySpin ``Exp.Ordering`` / ``orifun_M2L.m``).

``Experiment.Ordering`` is either a scalar λ, giving the built-in axial
distribution ``exp(λ·P2(cos β))``, or a callable ``f(alpha, beta, gamma)`` (or
``f(beta)``) of the Euler angles that rotate the sample frame into the
molecular frame.  :func:`orifun_M2L` evaluates the distribution for grid
orientations ``(phi, theta)`` given in the molecular frame, integrating over
the third angle χ with a 12-point trapezoidal rule exactly as EasySpin does.
"""
from __future__ import annotations

import inspect
from typing import Callable, Optional

import numpy as np

from .plegendre import plegendre
from .rotations import erot
from .rotutils import eulang

# NumPy 2.0 removed ``np.trapz``; ``np.trapezoid`` replaces it (NumPy >= 2.0).
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")

__all__ = ['ordering_function', 'orifun_M2L']


def ordering_function(ordering) -> Optional[Callable]:
    """Normalize ``Experiment.Ordering`` to a 3-argument callable (EasySpin p_sampletype)."""
    if ordering is None:
        return None
    if callable(ordering):
        try:
            nargs = len(inspect.signature(ordering).parameters)
        except (TypeError, ValueError):
            nargs = 3
        if nargs == 1:
            return lambda a, b, c: np.asarray(ordering(b), dtype=float) * np.ones(np.shape(c))
        return ordering
    lam = float(ordering)
    return lambda a, b, c: np.exp(lam * np.asarray(plegendre(2, 0, np.cos(b)), dtype=float))


def _R(angles) -> np.ndarray:
    R = erot(list(angles))
    return R.detach().cpu().numpy() if hasattr(R, 'detach') else np.asarray(R)


def orifun_M2L(f: Callable, R_L2S: np.ndarray, phi, theta, chi=None) -> np.ndarray:
    """Orientation weights ``f(alpha,beta,gamma)`` for molecular-frame grid
    orientations; χ integrated (trapezoid, 12 points) when *chi* is None."""
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    R_S2L = np.asarray(R_L2S, dtype=float).T
    w = np.zeros(phi.size)
    if chi is None:
        n_chi = 12
        chis = np.linspace(0, 2 * np.pi, n_chi)
        dchi = chis[1] - chis[0]
        c, s = np.cos(chis), np.sin(chis)
        for i in range(phi.size):
            R_L2M = _R([phi[i], theta[i], 0.0]).T
            xL = np.outer(R_L2M[:, 0], c) + np.outer(R_L2M[:, 1], s)
            yL = -np.outer(R_L2M[:, 0], s) + np.outer(R_L2M[:, 1], c)
            ang = np.zeros((n_chi, 3))
            for k in range(n_chi):
                R = R_L2M.copy()
                R[:, 0] = xL[:, k]
                R[:, 1] = yL[:, k]
                ang[k] = eulang(R @ R_S2L)
            ow = np.asarray(f(ang[:, 0], ang[:, 1], ang[:, 2]), dtype=float) * np.ones(n_chi)
            w[i] = _trapezoid(ow) * dchi
    else:
        chi = np.atleast_1d(np.asarray(chi, dtype=float))
        for i in range(phi.size):
            R_L2M = _R([phi[i], theta[i], chi[i]]).T
            a, b, g = eulang(R_L2M @ R_S2L)
            w[i] = float(np.asarray(f(a, b, g), dtype=float).reshape(-1)[0])
    return w
