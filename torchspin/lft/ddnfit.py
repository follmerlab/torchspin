"""DDNFIT: least-squares fit of d^n ligand-field parameters to band energies, with the exact Jacobian.

A thin Telser-named front end to :class:`torchspin.lft.fit.JointFit` with one
:class:`~torchspin.lft.fit.Bands` data set.  States are numbered as DDNFIT does: 1-based over
all C(10, n) states in ascending order, energies relative to state 1, in ``units``.  Add
susceptibility, magnetization, g values or an EPR spectrum by using ``JointFit`` directly.

    from torchspin import lft
    from torchspin.lft.fit import FitParameter
    out = lft.ddnfit.fit(3, [FitParameter("B", 700, 400, 1100), FitParameter("Dq", 1500, 800, 2500)],
                         lambda p: dict(B=p["B"], C=4.0 * p["B"], V=lft.crystal_field(p["Dq"])),
                         states=[5, 17], energies=[17400.0, 24000.0], sigma=50.0, units="cm^-1")
"""

from __future__ import annotations

from typing import Callable, Sequence

from .fit import Bands, FitParameter, JointFit, JointOutcome

__all__ = ["fit"]


def fit(n: int, parameters: Sequence[FitParameter], model: Callable[[dict], dict], states: Sequence[int],
        energies: Sequence[float], sigma=1.0, units: str = "MHz", **kw) -> JointOutcome:
    """Fit ``model`` (named tensors -> :func:`torchspin.lft.ddn.hamiltonian` keywords) to band energies."""
    return JointFit(n, parameters, model, [Bands(states, energies, sigma)], units=units).fit(**kw)
