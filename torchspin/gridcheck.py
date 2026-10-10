"""Is the orientation grid fine enough? Answer it by refining and looking.

``Options.GridSize`` controls how many orientations the resonance fields are
computed at before being interpolated to the accumulation grid, and its default
``[19, 4]`` is not always enough: for a 0.54 mT line on a 647 MHz copper
hyperfine coupling the derivative peak at the perpendicular turning point is
still moving by 4 % between ``[91, 4]`` and ``[361, 4]`` — in EasySpin equally.
Nothing in a single spectrum reveals that.

There is no reliable shortcut.  A heuristic on the field step between knots
looks appealing and is wrong: the SOPHE projection integrates analytically over
each grid segment, so a large step per knot is normal and harmless, and such a
heuristic flags dozens of simulations that agree with EasySpin to cosine 0.9999.
What does work is the obvious thing — simulate again on a finer grid and compare
— which is what :func:`grid_convergence` does.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from torchspin.experiment import Experiment, Options

__all__ = ['GridConvergence', 'grid_convergence']


@dataclass
class GridConvergence:
    """Result of comparing a spectrum against a more finely sampled one.

    Attributes
    ----------
    cosine:
        Cosine similarity between the two spectra, 1.0 when converged.
    amplitude_ratio:
        Peak amplitude of the requested grid over that of the refined grid.
        The quantity that moves first, because the derivative extremum at a
        turning point is the most grid-sensitive number in a powder spectrum.
    converged:
        ``cosine >= cosine_tol`` and ``|amplitude_ratio - 1| <= amplitude_tol``.
    grid_size, refined_grid_size:
        What was compared.
    """

    cosine: float
    amplitude_ratio: float
    converged: bool
    grid_size: list
    refined_grid_size: list
    cosine_tol: float
    amplitude_tol: float

    def __str__(self) -> str:
        verdict = 'converged' if self.converged else 'NOT converged'
        return (f'GridSize={self.grid_size} vs {self.refined_grid_size}: {verdict} '
                f'(cosine {self.cosine:.5f}, amplitude ratio {self.amplitude_ratio:.4f})')

    def report(self) -> str:
        """A sentence to print, with what to do about it."""
        if self.converged:
            return (f'GridSize={self.grid_size} looks converged: refining to '
                    f'{self.refined_grid_size} changes the spectrum by cosine '
                    f'{1 - self.cosine:.2g} and the amplitude by '
                    f'{abs(self.amplitude_ratio - 1) * 100:.2g} %.')
        return (f'GridSize={self.grid_size} is NOT converged: refining to '
                f'{self.refined_grid_size} changes the amplitude by '
                f'{abs(self.amplitude_ratio - 1) * 100:.2g} % and the shape by cosine '
                f'{1 - self.cosine:.2g}. Use {self.refined_grid_size} (or finer) for '
                f'quantitative work, and re-check.')


def grid_convergence(sys, exp: Experiment, opt: Optional[Options] = None,
                     refine: int = 4, cosine_tol: float = 0.9999,
                     amplitude_tol: float = 0.01) -> GridConvergence:
    """Simulate twice, once on a finer orientation grid, and compare.

    Parameters
    ----------
    sys, exp, opt:
        Exactly what you would pass to :func:`torchspin.pepper`; ``opt`` supplies
        the ``GridSize`` under test (default ``Options()``).
    refine:
        Factor by which to multiply the coarse knot count for the reference.
        The cost of the check is dominated by that second simulation.
    cosine_tol, amplitude_tol:
        What counts as converged.  The defaults are a tenth of the repository's
        parity tolerances, so that grid error stays well below the level the
        simulators are validated at.

    Returns
    -------
    GridConvergence

    Examples
    --------
    >>> from torchspin import SpinSystem, Experiment, Options, grid_convergence
    >>> c = grid_convergence(sys, exp, Options(GridSize=[19, 4]))   # doctest: +SKIP
    >>> print(c.report())                                           # doctest: +SKIP
    GridSize=[19, 4] is NOT converged: refining to [76, 4] changes the amplitude
    by 23 % and the shape by cosine 0.03. Use [76, 4] (or finer) ...
    """
    from torchspin.pepper import pepper
    if refine < 2:
        raise ValueError('refine must be at least 2, otherwise nothing is refined.')
    opt = opt if opt is not None else Options()
    coarse = [int(v) for v in np.atleast_1d(np.asarray(opt.GridSize)).astype(int)]
    n_coarse = coarse[0]
    n_interp = coarse[1] if len(coarse) > 1 else 1
    fine = [int(n_coarse * refine), int(n_interp)] if len(coarse) > 1 else int(n_coarse * refine)

    from dataclasses import replace as _replace
    _, spc_c = pepper(sys, exp, _replace(opt, GridSize=opt.GridSize))
    _, spc_f = pepper(sys, exp, _replace(opt, GridSize=fine))
    a = np.asarray(spc_c, dtype=float).ravel()
    b = np.asarray(spc_f, dtype=float).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    cos = float(a @ b / denom) if denom > 0 else 1.0
    pk_b = np.abs(b).max()
    ratio = float(np.abs(a).max() / pk_b) if pk_b > 0 else 1.0
    return GridConvergence(
        cosine=cos, amplitude_ratio=ratio,
        converged=bool(cos >= cosine_tol and abs(ratio - 1.0) <= amplitude_tol),
        grid_size=coarse if len(coarse) > 1 else [n_coarse],
        refined_grid_size=fine if isinstance(fine, list) else [fine],
        cosine_tol=cosine_tol, amplitude_tol=amplitude_tol,
    )
