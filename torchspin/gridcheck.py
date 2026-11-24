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
    """Result of :func:`grid_convergence`: a spectrum compared with a refined one.

    Returned by :func:`grid_convergence`; not normally constructed by hand.
    ``str(result)`` gives a one-line summary and :meth:`report` a sentence with
    a recommendation.

    Parameters
    ----------
    cosine : float
        Cosine similarity ``a.b / (|a| |b|)`` between the spectrum on the
        requested grid (``a``) and on the refined grid (``b``), both flattened
        (all rows when pepper returns a 2-D array).  1.0 means identical shape;
        1.0 is also returned when either spectrum is all zero.
    amplitude_ratio : float
        ``max|a| / max|b|``: peak amplitude on the requested grid over that on
        the refined grid (1.0 when the refined spectrum is all zero).  This is
        usually the first quantity to move, because the derivative extremum at
        a turning point is the most grid-sensitive number in a powder
        spectrum.
    converged : bool
        ``cosine >= cosine_tol and abs(amplitude_ratio - 1) <= amplitude_tol``.
    grid_size : list of int
        The ``Options.GridSize`` under test as a list: ``[N]`` for an integer
        ``GridSize`` or ``[N_coarse, N_interp]``.
    refined_grid_size : list of int
        The reference grid: ``[refine * N]`` or ``[refine * N_coarse,
        N_interp]`` (the interpolation factor is kept).
    cosine_tol : float
        Cosine threshold used for ``converged``.
    amplitude_tol : float
        Relative amplitude threshold used for ``converged``.

    See Also
    --------
    grid_convergence : Produces this result.
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
        """Return a sentence describing the result and what to do about it.

        Returns
        -------
        str
            For a converged grid, the size of the change on refinement; for an
            unconverged one, the change and the recommendation to use
            ``refined_grid_size`` (or finer) and re-check.
        """
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
    """Check whether ``Options.GridSize`` is fine enough for a pepper simulation.

    Runs :func:`torchspin.pepper` twice, once with the ``GridSize`` of ``opt``
    and once with the coarse knot count multiplied by ``refine``, and compares
    the two spectra (shape by cosine similarity, height by peak-amplitude
    ratio).  Only ``Options.GridSize`` is changed between the two runs; every
    other option (``GridSymmetry``, ``Method``, ``separate``, ...) is passed
    through unchanged.

    Parameters
    ----------
    sys : SpinSystem or list of SpinSystem
        Spin system(s), exactly as for :func:`torchspin.pepper`.  Tensors must
        not require grad (see Raises).
    exp : Experiment
        Experiment, exactly as for :func:`torchspin.pepper` (field or
        frequency sweep).
    opt : Options, optional
        Options; its ``GridSize`` (int ``N`` or ``[N_coarse, N_interp]``) is the
        grid under test.  Default ``None`` = ``Options()``, i.e. ``GridSize =
        [19, 4]``.
    refine : int, default 4
        Factor applied to the coarse knot count for the reference run:
        ``N -> refine * N`` or ``[N, k] -> [refine * N, k]``.  Must be at least
        2.  The cost of the check is dominated by the refined simulation, whose
        orientation count grows roughly as ``refine**2`` for 2-D grids.
    cosine_tol : float, default 0.9999
        Minimum cosine similarity for ``converged``.
    amplitude_tol : float, default 0.01
        Maximum relative change of the peak amplitude for ``converged``
        (0.01 = 1 %).  Both defaults are a tenth of the repository's parity
        tolerances, so that grid error stays well below the level the
        simulators are validated at.

    Returns
    -------
    GridConvergence
        Cosine similarity, amplitude ratio, the verdict, both grid sizes and
        the tolerances used.

    Raises
    ------
    ValueError
        If ``refine < 2``, and anything :func:`torchspin.pepper` raises for
        these inputs.
    RuntimeError
        If an input tensor requires grad: pepper then returns a spectrum on
        the autograd graph, which cannot be converted to NumPy here.  Pass
        detached values.

    See Also
    --------
    GridConvergence : The result object.
    torchspin.pepper : The simulator being checked.
    torchspin.Options : ``GridSize`` and ``GridSymmetry``.

    Notes
    -----
    A single spectrum does not reveal grid error, and a heuristic on the field
    step between knots is not a reliable substitute (see the module
    docstring); comparing with a refined grid is.  The check is meaningful
    only when pepper uses a powder grid: with ``Experiment.Orientations`` the
    grid size is ignored by pepper and the two spectra are identical.

    Examples
    --------
    A Cu(II) powder spectrum with a narrow line is not converged at the
    default ``GridSize = [19, 4]``:

    >>> from torchspin import SpinSystem, Experiment, Options, grid_convergence
    >>> cu = SpinSystem(S=0.5, g=[2.05, 2.05, 2.25], Nucs='63Cu',
    ...                 A=[50, 50, 500], lw=0.54)
    >>> exp = Experiment(mwFreq=9.5, Range=[260, 360], nPoints=1024)
    >>> c = grid_convergence(cu, exp, Options(GridSize=[19, 4]))
    >>> c.converged, c.refined_grid_size
    (False, [76, 4])
    >>> round(c.amplitude_ratio, 2)
    1.02
    >>> print(c.report())  # doctest: +ELLIPSIS
    GridSize=[19, 4] is NOT converged: refining to [76, 4] changes the amplitude by 1.7 % ...
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
    import torch
    # B109: a diagnostic; parameters that require grad give spectra on the graph
    with torch.no_grad():
        _, spc_c = pepper(sys, exp, _replace(opt, GridSize=opt.GridSize))
        _, spc_f = pepper(sys, exp, _replace(opt, GridSize=fine))
    _np = lambda y: (y.detach().cpu().numpy() if torch.is_tensor(y) else np.asarray(y)).astype(float).ravel()
    a = _np(spc_c)
    b = _np(spc_f)
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
