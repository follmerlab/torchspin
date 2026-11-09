"""Ligand-field tools: Joshua Telser's programs as differentiable torchspin modules.

Ported from telser-tools (``telser_tools_diff``).  Modules carry Telser's program names:

``lft.ddn``           d^n ligand-field levels and doublet g (DDN)
``lft.ligand_field``  AOM, Ballhausen crystal field, orbital energies (also exported here:
                      ``lft.aom``, ``lft.aom_isotropic``, ``lft.crystal_field``, ...)
``lft.bridge``        ``dn2sys``: ligand-field parameters -> SpinSystem with full g
``lft.units``         graph-safe unit factors

Units follow torchspin at every public boundary (MHz, mT, GHz, radians; molar
susceptibility in m^3/mol), with ``units=`` on request (``'cm^-1'``, ``'K'``, ``'eV'``, ...).
Ligand-field plots default to cm^-1.  The engines work in cm^-1 internally.

Thermal averages use :func:`torchspin.lft._linalg.thermal_average` (Daleckii-Krein
derivatives), not ``torchspin._linalg.eigh``, whose gauge-zeroing backward is right for
spectra but drops f'(lambda) tr(dH A) inside degenerate levels.
"""

from . import bridge, ddn, ligand_field, units
from .bridge import dn2sys
from .ligand_field import (AOM_MODES, Ligand, aom, aom_from_angles, aom_from_frames, aom_isotropic, crystal_field,
                           octahedron, orbital_matrix, rotation, tetrahedron)

__all__ = ["AOM_MODES", "Ligand", "aom", "aom_from_angles", "aom_from_frames", "aom_isotropic", "bridge",
           "crystal_field", "ddn", "dn2sys", "ligand_field", "octahedron", "orbital_matrix", "rotation",
           "tetrahedron", "units"]
