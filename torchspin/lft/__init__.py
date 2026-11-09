"""Ligand-field tools: Joshua Telser's programs as differentiable torchspin modules.

Ported from telser-tools (``telser_tools_diff``).  Public names follow Telser's programs
(``lft.ddn``, ``lft.ddnfit``, ``lft.dpn``, ``lft.dsoxf``, ...); see ``docs/lft.md``.

Units follow torchspin at every public boundary (MHz, mT, GHz, radians; molar
susceptibility in m^3/mol), with ``units=`` on request (``'cm^-1'``, ``'K'``, ``'eV'``, ...).
Ligand-field plots default to cm^-1.  The engines work in cm^-1 internally.  See
:mod:`torchspin.lft.units`.

Nothing in this subpackage changes existing torchspin behavior.  Thermal averages use
:func:`torchspin.lft._linalg.thermal_average` (Daleckii-Krein derivatives), not
``torchspin._linalg.eigh``, whose gauge-zeroing backward is right for spectra but drops
f'(lambda) tr(dH A) inside degenerate levels.
"""

from . import units

__all__ = ["units"]
