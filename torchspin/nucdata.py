"""Nuclear spin database for torchspin.

Port of EasySpin's ``nucdata.m``. Returns nuclear spin quantum number (I), 
nuclear g-factor (gn), quadrupole moment (qm), and natural abundance for 
stable and selected radioactive isotopes.

Units
-----
* Nuclear g-factor: dimensionless
* Quadrupole moment: barn (10^-28 m^2)
* Natural abundance: fractional (0 to 1, not percent)

Usage
-----
::

    from torchspin.nucdata import nucdata

    # Single nucleus
    I, gn, qm, abund = nucdata('14N')

    # Multiple nuclei (returns arrays)
    I, gn, qm, abund = nucdata('1H,14N,14N')
    # I = [0.5, 1.0, 1.0], gn = [5.585..., 0.403..., 0.403...]

    # Get only some properties
    I = nucdata('63Cu')  # returns scalar
    I, gn = nucdata('63Cu')  # returns tuple

Data Source
-----------
Nuclear magnetic moments from:
  N. Stone, "Table of Nuclear Magnetic Dipole and Electric Quadrupole Moments"
  International Atomic Energy Agency, INDC(NDS)-0658, February 2014

Nuclear quadrupole moments from:
  N. Stone, "Table of Nuclear Quadrupole Moments"
  International Atomic Energy Agency, INDC(NDS)-650, December 2013
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Isotope database singleton (loaded once per session)
# ---------------------------------------------------------------------------
class _IsotopeDatabase:
    """Internal singleton holding isotope data loaded from isotopedata.txt."""
    
    def __init__(self):
        self.symbols: list[str] = []
        self.spins: list[float] = []
        self.gns: list[float] = []
        self.abundances: list[float] = []
        self.qms: list[float] = []
        self.elements: list[str] = []
        self.protons: list[int] = []
        self.nucleons: list[int] = []
        self.radioactive: list[bool] = []
        self._loaded = False
    
    def load(self):
        """Load isotope data from isotopedata.txt (only once)."""
        if self._loaded:
            return
        
        # Find data file
        this_dir = Path(__file__).parent
        data_file = this_dir / 'data' / 'isotopedata.txt'
        
        if not data_file.exists():
            raise FileNotFoundError(
                f"Could not find nuclear isotope database at {data_file}"
            )
        
        # Parse file
        with open(data_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('%'):
                    continue
                
                parts = line.split()
                if len(parts) < 9:
                    continue  # skip malformed lines
                
                protons = int(parts[0])
                nucleons = int(parts[1])
                is_radioactive = (parts[2] == '*')
                element = parts[3]
                # parts[4] is the element name, skip
                spin = float(parts[5])
                gn = float(parts[6])
                abundance = float(parts[7])
                qm_str = parts[8]
                
                # Parse quadrupole moment (may be 'NaN')
                if qm_str.lower() == 'nan':
                    qm = float('nan')
                else:
                    qm = float(qm_str)
                
                # Build isotope symbol (e.g. '14N')
                symbol = f"{nucleons}{element}"
                
                self.protons.append(protons)
                self.nucleons.append(nucleons)
                self.radioactive.append(is_radioactive)
                self.elements.append(element)
                self.symbols.append(symbol)
                self.spins.append(spin)
                self.gns.append(gn)
                self.abundances.append(abundance)
                self.qms.append(qm)
        
        self._loaded = True
    
    def find_isotope(self, symbol: str) -> int:
        """Return index of isotope matching *symbol*, or raise ValueError."""
        try:
            return self.symbols.index(symbol)
        except ValueError:
            # Check if it's an element without mass number
            element = ''.join(c for c in symbol if c.isalpha())
            # Find all isotopes of this element
            matching = [s for s in self.symbols if s.endswith(element)]
            
            if matching:
                raise ValueError(
                    f"Please specify isotope mass number. "
                    f"Available isotopes of {element}: {', '.join(matching)}"
                )
            else:
                raise ValueError(f"Unknown isotope '{symbol}'")


# Global database instance
_DB = _IsotopeDatabase()


# ---------------------------------------------------------------------------
# Helper: parse nucleus string to list
# ---------------------------------------------------------------------------
def _nucstring2list(nuc_string: str) -> list[str]:
    """Parse comma-separated nucleus string to list.
    
    Examples:
        '14N' → ['14N']
        '1H,14N,14N' → ['1H', '14N', '14N']
        '1H, 14N, 14N' → ['1H', '14N', '14N'] (spaces removed)
    
    Does NOT support natural-mixture notation like 'Cu' or '(63,65)Cu'.
    """
    if not nuc_string:
        return []
    
    # Remove whitespace
    nuc_string = nuc_string.replace(' ', '')
    
    # Remove trailing comma if present
    if nuc_string.endswith(','):
        nuc_string = nuc_string[:-1]
    
    # Split on comma
    nucs = nuc_string.split(',')
    
    # Filter empty strings
    return [n for n in nucs if n]


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------
def nucdata(isotopes: Optional[str | list[str]] = None):
    """Return nuclear spin properties for one or more isotopes.
    
    Parameters
    ----------
    isotopes : str or list of str, optional
        Isotope specification. Examples:
        - ``'14N'`` → single nucleus
        - ``'1H,14N,14N'`` → comma-separated list
        - ``['1H', '14N', '14N']`` → Python list
        - ``None`` → returns full isotope database dict (for introspection)
    
    Returns
    -------
    I : float or ndarray
        Nuclear spin quantum number(s)
    gn : float or ndarray
        Nuclear g-factor(s)
    qm : float or ndarray
        Quadrupole moment(s) in barn (10^-28 m^2); NaN if not measured
    abund : float or ndarray
        Natural abundance as fraction (0 to 1)
    
    If ``isotopes`` is a single nucleus, returns a tuple of 4 scalars.
    If ``isotopes`` specifies multiple nuclei, returns a tuple of 4 NumPy arrays.
    
    For convenience, use ``nucspin()``, ``nucgval()``, ``nucqmom()``, or 
    ``nucabund()`` to retrieve only a single property.
    
    Examples
    --------
    >>> from torchspin.nucdata import nucdata, nucspin
    >>> I, gn, qm, abund = nucdata('14N')
    >>> I
    1.0
    >>> gn
    0.40376104
    
    >>> I, gn, qm, abund = nucdata('1H,14N,14N')
    >>> I
    array([0.5, 1. , 1. ])
    >>> gn
    array([5.58569468, 0.40376104, 0.40376104])
    
    >>> # Get only spin using convenience function
    >>> I = nucspin('14N')
    >>> I
    1.0
    """
    # Ensure database is loaded
    _DB.load()
    
    # No arguments → return full database structure (introspection)
    if isotopes is None:
        return {
            'symbols': _DB.symbols,
            'spins': _DB.spins,
            'gns': _DB.gns,
            'qms': _DB.qms,
            'abundances': _DB.abundances,
            'elements': _DB.elements,
            'protons': _DB.protons,
            'nucleons': _DB.nucleons,
            'radioactive': _DB.radioactive,
        }
    
    # Parse input
    if isinstance(isotopes, str):
        nucs = _nucstring2list(isotopes)
    elif isinstance(isotopes, (list, tuple)):
        nucs = list(isotopes)
    else:
        raise TypeError(
            f"isotopes must be str or list, not {type(isotopes).__name__}"
        )
    
    if not nucs:
        # Empty input → return empty arrays
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    # Look up each nucleus
    spins = []
    gns = []
    qms = []
    abundances = []
    
    for nuc in nucs:
        idx = _DB.find_isotope(nuc)
        spins.append(_DB.spins[idx])
        gns.append(_DB.gns[idx])
        qms.append(_DB.qms[idx])
        # Convert abundance from percent to fraction
        abundances.append(_DB.abundances[idx] / 100.0)
    
    # Return scalars for single nucleus, arrays for multiple
    if len(nucs) == 1:
        return spins[0], gns[0], qms[0], abundances[0]
    else:
        return (
            np.array(spins),
            np.array(gns),
            np.array(qms),
            np.array(abundances),
        )


# ---------------------------------------------------------------------------
# Convenience accessors (match EasySpin naming)
# ---------------------------------------------------------------------------
def nucspin(isotopes: str | list[str]):
    """Return nuclear spin quantum number(s) only."""
    result = nucdata(isotopes)
    return result[0] if isinstance(result, tuple) else result


def nucgval(isotopes: str | list[str]):
    """Return nuclear g-factor(s) only."""
    I, gn, *_ = nucdata(isotopes)
    return gn


def nucqmom(isotopes: str | list[str]):
    """Return nuclear quadrupole moment(s) only."""
    I, gn, qm, *_ = nucdata(isotopes)
    return qm


def nucabund(isotopes: str | list[str]):
    """Return natural abundance(s) only (as fraction 0-1)."""
    I, gn, qm, abund = nucdata(isotopes)
    return abund
