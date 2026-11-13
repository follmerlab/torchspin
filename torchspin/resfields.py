"""Resonance field finder for CW EPR (matrix method), one orientation.

:func:`resfields` is a thin wrapper over :func:`torchspin.resfields_batch.resfields_batch`
with a single orientation (M = 1): the same segment model of the level
diagram, Newton/bisection root search, intensities (``Exp.mwMode``,
``Exp.Temperature``, intensity threshold on ``|I|``) and strain widths at each
line's own resonance field.  Every step is a torch operation, so resonance
fields, intensities and widths are differentiable with respect to the
Hamiltonian and the strain parameters (0.9.20; before, the function used a
Python-float Brent search and ``.item()`` and returned no or wrong gradients,
B76).
"""
from __future__ import annotations

from typing import Optional

import torch

from torchspin.experiment import Experiment, Options
from torchspin.spinsystem import SpinSystem


def resfields(
    H0: torch.Tensor,
    mux: torch.Tensor,
    muy: torch.Tensor,
    muz: torch.Tensor,
    phi: float,
    theta: float,
    exp: Experiment,
    opt: Options,
    sys: Optional[SpinSystem] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Find resonance fields and intensities for one orientation.

    Parameters
    ----------
    H0:
        Field-independent Hamiltonian (MHz), shape ``(nStates, nStates)``.
    mux, muy, muz:
        Magnetic-moment operators (MHz/mT), each ``(nStates, nStates)``.
        Convention: H(B) = H0 - (mux*B[0] + muy*B[1] + muz*B[2]).
    phi, theta:
        Orientation angles (radians), floats or 0-d tensors (which may
        require grad).  The static field lies along
        n = (sin(θ)cos(φ), sin(θ)sin(φ), cos(θ)).
    exp:
        Experiment parameters (mwFreq in GHz, Range in mT, mwMode, Temperature).
    opt:
        Options (Threshold, relative to the largest ``|I|``).
    sys:
        Spin system (optional). Required for strain widths and for the
        nuclear-sublevel polarization factor.

    Returns
    -------
    B_res:
        Resonance fields in mT, shape ``(nTransitions,)``, ordered by level
        pair and then by field.  A level pair whose transition loops in the
        field range contributes all of its resonances.
    intensities:
        Transition intensities (a.u.), shape ``(nTransitions,)``.
    widths:
        Gaussian strain FWHM (mT) for each line, shape ``(nTransitions,)``;
        an empty tensor if the system has strains but no line was found.
        ``None`` if ``sys`` is not given or has no strain parameters.

    Raises
    ------
    ValueError
        ``gStrain``/``AStrain`` with a full g matrix (as EasySpin).
    NotImplementedError
        ``Options.QuadraticZeeman`` with field-nonlinear ``Sys.Ham`` terms.
    """
    if sys is not None:
        from torchspin.ham_ezho import check_quadratic_zeeman_unsupported
        check_quadratic_zeeman_unsupported(sys, opt, 'resfields (single orientation)')
    from torchspin.resfields_batch import resfields_batch

    dev = H0.device
    ph = torch.as_tensor(phi, dtype=torch.float64, device=dev).reshape(1)
    th = torch.as_tensor(theta, dtype=torch.float64, device=dev).reshape(1)
    B, I, W = resfields_batch(H0, mux, muy, muz, ph, th, exp, opt, sys=sys)
    widths = W[0]
    if widths is None and sys is not None:
        from torchspin.strainwidth import _has_any_strain
        if _has_any_strain(sys):
            widths = torch.zeros(0, dtype=torch.float64, device=dev)
    return B[0], I[0], widths
