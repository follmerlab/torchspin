"""Graph-safe unit factors for the ligand-field tools (``torchspin.lft``).

The public API of :mod:`torchspin.lft` follows torchspin: energies and couplings in MHz,
fields in mT, frequencies in GHz, angles in radians, molar susceptibility in m^3/mol (SI, as
:func:`torchspin.curry` returns it).  Ligand-field energies are far easier to read in cm^-1,
so every function that returns energies takes ``units=`` and every ligand-field plot defaults
to cm^-1.  The conversions are plain Python floats derived from :mod:`torchspin.constants`,
so multiplying a tensor by one keeps the autograd graph (unlike the NumPy
:func:`torchspin.utils.unitconvert`).

Energy units: ``'MHz'``, ``'GHz'``, ``'Hz'``, ``'cm^-1'`` (aliases ``'cm-1'``, ``'cm⁻¹'``),
``'K'``, ``'eV'``, ``'meV'``, ``'J'``.
Field units: ``'mT'``, ``'T'``, ``'G'``.
Molar susceptibility: ``'SI'`` (m^3/mol) and ``'cgs'`` (cm^3/mol, emu), related by
chi_cgs = chi_SI * 1e6 / (4 pi).
"""

from __future__ import annotations

import math

from torchspin.constants import BOLTZMANN, CLIGHT, EVOLT, PLANCK

__all__ = ["ENERGY_UNITS", "FIELD_UNITS", "CHI_UNITS", "PLOT_ENERGY_UNITS", "energy_factor", "convert_energy",
           "field_factor", "convert_field", "chi_factor", "convert_chi", "energy_label", "canonical_energy_unit"]

# Joules per unit.  CLIGHT is in cm/s, so h*c is J per cm^-1.
_ENERGY_J = {
    "Hz": PLANCK,
    "MHz": PLANCK * 1e6,
    "GHz": PLANCK * 1e9,
    "cm^-1": PLANCK * CLIGHT,
    "K": BOLTZMANN,
    "eV": EVOLT,
    "meV": EVOLT * 1e-3,
    "J": 1.0,
}
_ENERGY_ALIASES = {"cm-1": "cm^-1", "cm⁻¹": "cm^-1", "1/cm": "cm^-1", "wavenumber": "cm^-1"}
_ENERGY_LABELS = {"Hz": "Hz", "MHz": "MHz", "GHz": "GHz", "cm^-1": "cm⁻¹", "K": "K", "eV": "eV",
                  "meV": "meV", "J": "J"}

_FIELD_T = {"T": 1.0, "mT": 1e-3, "G": 1e-4}
_CHI_SI = {"SI": 1.0, "cgs": 4.0 * math.pi * 1e-6}  # m^3/mol per unit

ENERGY_UNITS = tuple(_ENERGY_J)
FIELD_UNITS = tuple(_FIELD_T)
CHI_UNITS = tuple(_CHI_SI)
#: Default unit of every ligand-field plot (level, Tanabe-Sugano, Zeeman and angle diagrams).
PLOT_ENERGY_UNITS = "cm^-1"


def canonical_energy_unit(unit: str) -> str:
    """The canonical spelling of an energy unit (``'cm-1'`` -> ``'cm^-1'``); raises on unknown units."""
    u = _ENERGY_ALIASES.get(unit, unit)
    if u not in _ENERGY_J:
        raise ValueError(f"unknown energy unit {unit!r}; use one of {', '.join(ENERGY_UNITS)}")
    return u


def energy_factor(from_: str, to: str) -> float:
    """Multiply an energy in ``from_`` by this to get it in ``to``."""
    return _ENERGY_J[canonical_energy_unit(from_)] / _ENERGY_J[canonical_energy_unit(to)]


def convert_energy(x, from_: str, to: str):
    """``x`` (float, array or tensor; the graph is kept) converted from ``from_`` to ``to``."""
    f = energy_factor(from_, to)
    return x if f == 1.0 else x * f


def field_factor(from_: str, to: str) -> float:
    """Multiply a magnetic field in ``from_`` by this to get it in ``to``."""
    for u in (from_, to):
        if u not in _FIELD_T:
            raise ValueError(f"unknown field unit {u!r}; use one of {', '.join(FIELD_UNITS)}")
    return _FIELD_T[from_] / _FIELD_T[to]


def convert_field(x, from_: str, to: str):
    f = field_factor(from_, to)
    return x if f == 1.0 else x * f


def chi_factor(from_: str, to: str) -> float:
    """Multiply a molar susceptibility in ``from_`` ('SI' m^3/mol or 'cgs' cm^3/mol) by this to get ``to``."""
    for u in (from_, to):
        if u not in _CHI_SI:
            raise ValueError(f"unknown susceptibility unit {u!r}; use 'SI' (m^3/mol) or 'cgs' (cm^3/mol)")
    return _CHI_SI[from_] / _CHI_SI[to]


def convert_chi(x, from_: str, to: str):
    f = chi_factor(from_, to)
    return x if f == 1.0 else x * f


def energy_label(unit: str) -> str:
    """Axis label text for an energy unit (``'cm^-1'`` -> ``'cm⁻¹'``)."""
    return _ENERGY_LABELS[canonical_energy_unit(unit)]
