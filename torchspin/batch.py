"""Batch simulation — simulate multiple SpinSystems simultaneously.

Provides ``batch_pepper`` for CW EPR and a generic ``batch_simulate``
dispatcher for any supported simulator (pepper, garlic, chili, saffron).

Key features over calling simulators in a loop:

* **Stacked output**: returns ``(x, spectra)`` with
  ``spectra.shape == (N, nPoints)`` — ready for fitting, plotting, or
  ensemble averaging.
* **Weighted sum**: optional ``return_individual=False`` for
  multi-component spectra.
* **Consistent API**: same interface for all simulators via
  ``batch_simulate()``.

Typical use::

    from torchspin import SpinSystem, Experiment, Options
    from torchspin.batch import batch_pepper

    systems = [
        SpinSystem(S=[0.5], g=[[2.00, 2.05, 2.10]], lw=[0.5]),
        SpinSystem(S=[0.5], g=[[2.01, 2.06, 2.11]], lw=[0.5]),
        SpinSystem(S=[0.5], g=[[1.99, 2.04, 2.09]], lw=[0.5]),
    ]
    exp = Experiment(mwFreq=9.5, Range=[300, 360], Harmonic=0)
    opt = Options(GridSize=19)

    x, spectra = batch_pepper(systems, exp, opt)
    # x: (nPoints,),  spectra: (3, nPoints)

    # Weighted sum for multi-component fitting:
    x, total = batch_pepper(systems, exp, opt,
                            weights=[0.5, 0.3, 0.2],
                            return_individual=False)
    # total: (nPoints,)
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch

from torchspin.experiment import Experiment, Options
from torchspin.spinsystem import SpinSystem


def batch_pepper(
    systems: Sequence[SpinSystem],
    exp: Experiment,
    opt: Optional[Options] = None,
    *,
    weights: Optional[Sequence[float]] = None,
    return_individual: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate CW EPR spectra for multiple SpinSystems in batch.

    Each system is simulated via ``pepper()`` (full pipeline including
    SOPHE projection, strain, and calibrated normalization) and the
    results are stacked into a single tensor.

    Parameters
    ----------
    systems:
        List of SpinSystem objects to simulate.
    exp:
        Shared experimental parameters (mwFreq, Range, nPoints, Harmonic).
    opt:
        Computational options.  Defaults to ``Options()``.
    weights:
        Per-system weights for summing.  Defaults to each system's
        ``weight`` attribute (1.0 if not set).  Only used when
        ``return_individual=False``.
    return_individual:
        If ``True`` (default), return ``(x, spectra)`` where
        ``spectra.shape == (N, nPoints)``.
        If ``False``, return ``(x, weighted_sum)`` where
        ``weighted_sum.shape == (nPoints,)``.

    Returns
    -------
    x:
        Field axis in mT, shape ``(nPoints,)``.
    spectra:
        If ``return_individual=True``: ``(N, nPoints)`` tensor.
        If ``return_individual=False``: ``(nPoints,)`` weighted sum.
    """
    from torchspin.pepper import pepper

    if opt is None:
        opt = Options()

    N = len(systems)
    if N == 0:
        x = torch.linspace(exp.Range[0], exp.Range[1], exp.nPoints,
                           dtype=torch.float64)
        if return_individual:
            return x, torch.zeros(0, exp.nPoints, dtype=torch.float64)
        return x, torch.zeros(exp.nPoints, dtype=torch.float64)

    # Simulate each system
    all_spectra = []
    x_out = None
    for sys in systems:
        x, y = pepper(sys, exp, opt)
        if x_out is None:
            x_out = x
        all_spectra.append(y)

    spectra = torch.stack(all_spectra, dim=0)  # (N, nPoints)

    if return_individual:
        return x_out, spectra

    # Weighted sum
    w_arr = torch.zeros(N, dtype=torch.float64)
    if weights is not None:
        for i, w in enumerate(weights):
            w_arr[i] = w
    else:
        for i, sys in enumerate(systems):
            w_arr[i] = getattr(sys, 'weight', 1.0)
    return x_out, (w_arr.unsqueeze(1) * spectra).sum(dim=0)


def batch_simulate(
    simulator: str,
    systems: Sequence[SpinSystem],
    exp,
    opt=None,
    *,
    weights: Optional[Sequence[float]] = None,
    return_individual: bool = True,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch batch simulation to the appropriate simulator.

    Parameters
    ----------
    simulator:
        One of ``'pepper'``, ``'garlic'``, ``'chili'``, ``'saffron'``.
    systems:
        List of SpinSystems to simulate.
    exp:
        Experiment parameters (type depends on simulator).
    opt:
        Options (type depends on simulator).
    weights:
        Per-system weights for summing.
    return_individual:
        If True, return all individual spectra stacked.
    **kwargs:
        Additional keyword arguments passed to the simulator.

    Returns
    -------
    x, spectra:
        Field/frequency axis and spectra (stacked or summed).
    """
    if simulator == 'pepper':
        return batch_pepper(systems, exp, opt, weights=weights,
                            return_individual=return_individual)
    elif simulator in ('garlic', 'chili', 'saffron'):
        return _batch_sequential(simulator, systems, exp, opt,
                                 weights=weights,
                                 return_individual=return_individual,
                                 **kwargs)
    else:
        raise ValueError(
            f"Unknown simulator '{simulator}'. "
            f"Choose from: 'pepper', 'garlic', 'chili', 'saffron'."
        )


def _batch_sequential(
    name: str,
    systems: Sequence[SpinSystem],
    exp,
    opt,
    *,
    weights: Optional[Sequence[float]] = None,
    return_individual: bool = True,
    **kwargs,
) -> tuple:
    """Run simulations sequentially and stack results.

    Used for all simulators. Each is called independently and the
    results are stacked into a (N, nPoints) tensor.
    """
    import importlib
    mod = importlib.import_module(f'torchspin.{name}')
    sim_fn = getattr(mod, name)

    N = len(systems)
    if N == 0:
        raise ValueError("systems list must not be empty")

    results = []
    for sys in systems:
        if opt is not None:
            out = sim_fn(sys, exp, opt, **kwargs)
        else:
            out = sim_fn(sys, exp, **kwargs)
        x, y = out[0], out[1]
        # Convert to tensor if numpy
        if not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y, dtype=torch.float64)
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float64)
        results.append((x, y))

    x_out = results[0][0]
    all_spectra = torch.stack([r[1] for r in results], dim=0)

    if return_individual:
        return x_out, all_spectra

    w_arr = torch.zeros(N, dtype=torch.float64)
    if weights is not None:
        for i, w in enumerate(weights):
            w_arr[i] = w
    else:
        for i, sys in enumerate(systems):
            w_arr[i] = getattr(sys, 'weight', 1.0)
    return x_out, (w_arr.unsqueeze(1) * all_spectra).sum(dim=0)
