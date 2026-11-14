"""DDNFIT: least-squares fit of d^n ligand-field parameters to band energies, with the exact Jacobian.

A thin Telser-named front end to :class:`torchspin.lft.fit.JointFit` with one
:class:`~torchspin.lft.fit.Bands` data set.  States are numbered as DDNFIT does: 1-based over
all C(10, n) states in ascending order, energies relative to state 1, in ``units``.  Add
susceptibility, magnetization, g values or an EPR spectrum by using ``JointFit`` directly
(see :func:`fit` for an example).
"""

from __future__ import annotations

from typing import Callable, Sequence

from .fit import Bands, FitParameter, JointFit, JointOutcome

__all__ = ["fit"]


def fit(n: int, parameters: Sequence[FitParameter], model: Callable[[dict], dict], states: Sequence[int],
        energies: Sequence[float], sigma=1.0, units: str = "MHz", **kw) -> JointOutcome:
    """Fit d^n ligand-field parameters to band energies (DDNFIT), with the exact Jacobian.

    Parameters
    ----------
    n : int
        Number of d electrons, 0 to 10.
    parameters : sequence of FitParameter
        Named fit parameters with start values, bounds and fixed flags
        (:class:`torchspin.lft.fit.FitParameter`).
    model : callable
        ``model(p) -> dict``: maps a dict of named scalar tensors to keywords of
        :func:`torchspin.lft.ddn.hamiltonian` (``B``, ``C``, ``zeta``, ``V``,
        ...), in ``units``.
    states : sequence of int
        1-based state numbers as DDNFIT counts them: all C(10, n) states in
        ascending energy, state 1 the ground state.  A degenerate level is
        fitted through its mean.
    energies : sequence of float
        Observed energies relative to state 1, in ``units``.
    sigma : float or sequence of float, default 1.0
        Uncertainties of ``energies`` in ``units`` (residuals are divided by
        them).
    units : str, default 'MHz'
        Energy unit of the model keywords and of the data.
    **kw
        Passed to :meth:`torchspin.lft.fit.JointFit.fit` (and on to
        :func:`scipy.optimize.least_squares`; default trust-region reflective
        with bounds).

    Returns
    -------
    torchspin.lft.fit.JointOutcome
        The fit result (values, standard errors), chi^2 per data set and
        undetermined parameter combinations.

    Raises
    ------
    ValueError
        If a state number is below 1 or parameter names repeat.

    See Also
    --------
    torchspin.lft.fit.JointFit : Joint fits of bands, chi, M, g and spectra.

    Examples
    --------
    Racah B and Dq of a d^3 ion from two band energies (cm^-1), C = 4 B:

    >>> from torchspin import lft
    >>> from torchspin.lft.fit import FitParameter
    >>> out = lft.ddnfit.fit(3, [FitParameter('B', 700, 400, 1100), FitParameter('Dq', 1500, 800, 2500)],
    ...                      lambda p: dict(B=p['B'], C=4.0 * p['B'], V=lft.crystal_field(p['Dq'])),
    ...                      states=[5, 17], energies=[17400.0, 24000.0], sigma=50.0, units='cm^-1')
    >>> out.result.names, out.chi2['bands'] < 1e-6
    (['B', 'Dq'], True)
    """
    return JointFit(n, parameters, model, [Bands(states, energies, sigma)], units=units).fit(**kw)
