"""Ligand-field tools: Joshua Telser's programs as differentiable torchspin modules.

Ported from telser-tools (``telser_tools_diff``).  Modules carry Telser's program names:

``lft.ddn``           d^n ligand-field levels and doublet g (DDN)
``lft.ddnloop``       levels and transitions vs field magnitude, angle or a parameter (DDNLOOP)
``lft.ddnpow``        powder EPR / FDMR directly from the d^n model (DDNPOW)
``lft.ddnfit``        fit d^n parameters to band energies (DDNFIT)
``lft.ddnsus``        chi(T), chi T, mu_eff; and fits to them (DDNSUS)
``lft.ddnmag``        M(B, T); and fits to it (DDNMAG)
``lft.dlsd5``         low-spin d5 g tensor and its inverses (DLSD5)
``lft.dipole``        dipolar hyperfine / electron-electron couplings, STO multipole form (DIPOLE)
``lft.dtd``           14N quadrupole, Townes-Dailey (DTD)
``lft.dpn``           p^n ligand-field levels and doublet g (DPN); its own p-shell AOM/crystal field
``lft.dpnfit``        fit p^n parameters to band energies (DPNFIT)
``lft.dsoxf``         single term + spin-orbit + low-symmetry field, d and f shells (DSOXF)
``lft.tanabe_sugano`` Tanabe-Sugano blocks and diagrams (DTANSUG)
``lft.fit``           joint fits of one model to bands, g, chi, M and EPR spectra (exact Jacobians)
``lft.grids``         field-direction grids for magnetometry (Gauss hemisphere, DDN igloo, ...)
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

from . import bridge, ddn, ddnfit, ddnloop, ddnpow, ddnmag, ddnsus, dipole, dlsd5, dpn, dpnfit, dsoxf, dtd, fit, grids, ligand_field, tanabe_sugano, units
from .bridge import dn2sys
from .ligand_field import (AOM_MODES, Ligand, aom, aom_from_angles, aom_from_frames, aom_isotropic, crystal_field,
                           octahedron, orbital_matrix, rotation, tetrahedron)

__all__ = [
    "aom", "aom_from_angles", "aom_from_frames", "aom_isotropic", "AOM_MODES", "bridge", "crystal_field", "ddn",
    "ddnfit", "ddnmag", "ddnsus", "dipole", "dlsd5", "dn2sys", "dpn", "dpnfit", "dsoxf", "dtd", "fit", "grids", "Ligand", "ligand_field", "octahedron",
    "orbital_matrix", "rotation", "tanabe_sugano", "tetrahedron", "units",
]
