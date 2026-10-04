"""Spin ladder decomposition for exchange-coupled two-electron systems.

Port of EasySpin's ``spinladder.m``.

For a system of two electron spins S1 and S2 coupled by isotropic exchange
``H_J = J * S1·S2`` (J in MHz), this module decomposes the full Hilbert
space into sub-spaces of definite total spin S_total = |S1-S2|, ..., S1+S2.

Each sub-space (ladder rung) is returned as a lightweight named tuple
containing the effective spin, the Boltzmann weight at temperature T, and
the energy relative to the singlet ground state.
"""
from __future__ import annotations

import math
from typing import NamedTuple, Optional


__all__ = ['spinladder', 'SpinManifold']


class SpinManifold(NamedTuple):
    """A total-spin manifold for a two-electron exchange system.

    Attributes
    ----------
    S_total : float
        Total spin quantum number.
    multiplicity : int
        2*S_total + 1 (number of m_S states).
    energy_MHz : float
        Energy of this manifold relative to the lowest-energy manifold (MHz).
        ``E = J/2 * S_total*(S_total+1)`` (with energy offset removed).
    weight : float
        Boltzmann population at the requested temperature.  Sums to 1.0
        over all manifolds.  If temperature is None, all manifolds have
        equal weight 1.0 (infinite T).
    g_eff : float
        Effective isotropic g-factor for this manifold (average of input g).
    """
    S_total: float
    multiplicity: int
    energy_MHz: float
    weight: float
    g_eff: float


def spinladder(
    S1: float,
    S2: float,
    J_MHz: float,
    g1: float = 2.0,
    g2: float = 2.0,
    temperature: Optional[float] = None,
) -> list[SpinManifold]:
    """Decompose a two-spin exchange system into total-spin manifolds.

    Port of EasySpin's ``spinladder.m``.

    Parameters
    ----------
    S1 : float
        Spin quantum number of electron 1.
    S2 : float
        Spin quantum number of electron 2.
    J_MHz : float
        Isotropic exchange coupling in MHz.  The Hamiltonian is::

            H_J = J * S1 · S2

        Positive J → antiferromagnetic (singlet ground state for S1=S2=1/2).
        Negative J → ferromagnetic (highest spin ground state).
    g1 : float
        Isotropic g-factor of spin 1.  Default 2.0.
    g2 : float
        Isotropic g-factor of spin 2.  Default 2.0.
    temperature : float or None
        Temperature in Kelvin for Boltzmann weighting.  If ``None``,
        uniform weights (infinite temperature) are used.

    Returns
    -------
    manifolds : list of SpinManifold
        One entry per total-spin manifold, sorted by ascending energy.
        Each manifold has ``.S_total``, ``.multiplicity``, ``.energy_MHz``,
        ``.weight``, and ``.g_eff``.

    Notes
    -----
    Energies use the Landé interval rule::

        E(S_total) = (J/2) * [S_total*(S_total+1) - S1*(S1+1) - S2*(S2+1)]

    The effective g-factor for each manifold is the isotropic average
    ``(g1 + g2) / 2`` (for a symmetric coupled system).  More accurate
    g-values require the full coupled-basis transformation.

    Examples
    --------
    >>> manifolds = spinladder(0.5, 0.5, 100.0)   # J=100 MHz antiferro
    >>> for m in manifolds:
    ...     print(f'S_total={m.S_total}, mult={m.multiplicity}, E={m.energy_MHz:.1f} MHz')
    S_total=0.0, mult=1, E=0.0 MHz
    S_total=1.0, mult=3, E=100.0 MHz

    >>> manifolds = spinladder(1.0, 1.0, -50.0)  # ferromagnetic J<0
    >>> for m in manifolds:
    ...     print(f'S_total={m.S_total}, E={m.energy_MHz:.1f} MHz')
    """
    from torchspin.constants import PLANCK, BOLTZMANN

    S1, S2 = float(S1), float(S2)
    S_min = abs(S1 - S2)
    S_max = S1 + S2

    # Generate all allowed total spins in half-integer steps
    S_values = []
    S = S_min
    while S <= S_max + 1e-9:
        S_values.append(round(S * 2) / 2)  # round to nearest half-integer
        S += 1.0

    # Energies: E(S_tot) = (J/2) * [S_tot*(S_tot+1) - S1*(S1+1) - S2*(S2+1)]
    offset = S1 * (S1 + 1) + S2 * (S2 + 1)
    energies = [(J_MHz / 2.0) * (S * (S + 1) - offset) for S in S_values]

    # Shift so minimum energy = 0
    E_min = min(energies)
    energies = [E - E_min for E in energies]

    multiplicities = [int(round(2 * S + 1)) for S in S_values]

    # Boltzmann weights
    if temperature is None or temperature <= 0:
        weights = [1.0] * len(S_values)
    else:
        # Convert MHz → Joules: E_J = E_MHz * 1e6 * h
        kT = BOLTZMANN * temperature  # J
        h = PLANCK                    # J·s
        boltz = [mult * math.exp(-E * 1e6 * h / kT)
                 for S, E, mult in zip(S_values, energies, multiplicities)]
        Z = sum(boltz)
        weights = [b / Z for b in boltz]

    # Effective g-factor: isotropic average
    g_eff = (g1 + g2) / 2.0

    manifolds = [
        SpinManifold(
            S_total=S,
            multiplicity=mult,
            energy_MHz=E,
            weight=w,
            g_eff=g_eff,
        )
        for S, mult, E, w in zip(S_values, multiplicities, energies, weights)
    ]

    # Sort by energy
    manifolds.sort(key=lambda m: m.energy_MHz)
    return manifolds
