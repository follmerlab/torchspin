"""orca2torchspin — Import spin Hamiltonian parameters from ORCA output files.

Port of EasySpin's ``orca2easyspin.m``.  Reads ORCA quantum-chemistry output
and returns a :class:`~torchspin.spinsystem.SpinSystem` (or list thereof for
multi-structure files such as relaxed scans).

Supported file formats (auto-detected):

* Main text output (``.out`` or any text) — all ORCA versions
* Binary property file (``.prop``) — ORCA versions < 5
* Text property file (``_property.txt``) — ORCA versions >= 5

Usage
-----
::

    from torchspin.orca2torchspin import orca2torchspin

    sys, data = orca2torchspin('nitroxide.out')
    sys, data = orca2torchspin('nitroxide.out', hf_cutoff=0.5)  # MHz

Units
-----
* g-tensor: dimensionless (eigenvalues of symmetrized g·gᵀ)
* D-tensor: MHz (converted from cm⁻¹)
* A-tensor: MHz (read directly)
* Q-tensor: MHz (converted from EFG in atomic units)
* Coordinates: Ångström
* Euler angles: radians (z-y'-z'' passive, same as EasySpin)
"""
from __future__ import annotations

import math
import re
import struct
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from scipy.linalg import sqrtm

from torchspin.nucdata import nucdata, nucspin, nucqmom
from torchspin.rotutils import eulang
from torchspin.spinsystem import SpinSystem

__all__ = ['orca2torchspin', 'OrcaData']


# ---------------------------------------------------------------------------
# Physical constants — must match MATLAB EasySpin exactly
# ---------------------------------------------------------------------------
_CLIGHT_CGS = 2.99792458e10     # speed of light in cm/s
_HARTREE = 4.3597447222071e-18  # Hartree energy in J
_BOHR_RADIUS = 5.2917721e-11    # Bohr radius in m
_ECHARGE = 1.60217663e-19       # elementary charge in C
_PLANCK = 6.62607015e-34        # Planck constant in J·s
_BARN = 1e-28                   # barn in m²

# D-tensor conversion: cm⁻¹ → MHz
# 1 cm⁻¹ = c [cm/s] Hz = 2.99792458e10 Hz = 29979.2458 MHz
_CM1_TO_MHZ = _CLIGHT_CGS / 1e6  # = 29979.2458

# EFG conversion: atomic units (Eh/e/a₀²) → V/m²
# MATLAB: efg_SI = efg_au * hartree / echarge / bohrrad^2
_EFG_AU_TO_SI = _HARTREE / _ECHARGE / _BOHR_RADIUS**2

# Coordinates: Bohr → Ångström
_BOHR_TO_ANGSTROM = _BOHR_RADIUS / 1e-10


# ---------------------------------------------------------------------------
# Element symbol ↔ atomic number tables
# ---------------------------------------------------------------------------
_ELEMENTS = [
    'H', 'He',
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt',
    'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf',
    'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og',
]

_ELEMENT_TO_Z = {sym: i + 1 for i, sym in enumerate(_ELEMENTS)}


def _element_to_number(symbol: str) -> int:
    """Convert element symbol to atomic number.  'Cu' → 29."""
    sym = symbol.strip().capitalize()
    # Handle two-letter symbols where second letter might be uppercase
    if len(sym) >= 2:
        sym = sym[0] + sym[1:].lower()
    if sym not in _ELEMENT_TO_Z:
        raise ValueError(f"Unknown element symbol '{symbol}'")
    return _ELEMENT_TO_Z[sym]


def _number_to_element(Z: int) -> str:
    """Convert atomic number to element symbol.  29 → 'Cu'."""
    if Z < 1 or Z > len(_ELEMENTS):
        raise ValueError(f"Atomic number {Z} out of range 1..{len(_ELEMENTS)}")
    return _ELEMENTS[Z - 1]


# ---------------------------------------------------------------------------
# Quadrupole reference isotope table — from MATLAB orca2easyspin_propbin.m
# Most naturally abundant isotope with I > 1/2, by element number (1-indexed).
# 0 means no such isotope exists.
# ---------------------------------------------------------------------------
_QREF_MASS_NO = [
    2, 0, 7, 9, 11, 0, 14, 17, 0, 21,      # 1-10
    23, 25, 27, 0, 0, 33, 35, 0, 39, 43,    # 11-20
    45, 47, 51, 53, 55, 0, 59, 61, 63, 67,  # 21-30
    69, 73, 75, 0, 79, 83, 85, 87, 0, 91,   # 31-40
    93, 95, 0, 101, 0, 105, 0, 0, 115, 0,   # 41-50
    121, 0, 127, 131, 133, 137, 139, 0, 141, 143,  # 51-60
    0, 147, 153, 157, 159, 163, 165, 167, 0, 173,  # 61-70
    175, 177, 181, 0, 187, 189, 193, 0, 197, 201,  # 71-80
    0, 0, 209, 0, 0, 0, 0, 0, 227, 0,       # 81-90
    0, 235, 237, 0, 243, 0, 0, 0, 0, 0,     # 91-100
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,           # 101-110
    0, 0, 0, 0, 0, 0, 0, 0,                 # 111-118
]


# ---------------------------------------------------------------------------
# Reference isotope lookup
# ---------------------------------------------------------------------------
def _reference_isotope(element: str) -> Optional[str]:
    """Return the reference isotope string for HFC (most abundant with I >= 1/2).

    E.g., 'H' → '1H', 'N' → '14N', 'Cu' → '63Cu'.
    Returns None if no isotope with I >= 1/2 exists for this element.
    """
    # Get all isotopes of this element from the database
    db = nucdata(None)
    el_clean = element.strip()

    best_symbol = None
    best_abund = -1.0

    for i, sym in enumerate(db['symbols']):
        # Extract element part from isotope symbol (e.g., '14N' → 'N')
        el_part = ''.join(c for c in sym if c.isalpha())
        if el_part != el_clean:
            continue
        if db['spins'][i] < 0.5:
            continue
        if db['abundances'][i] > best_abund:
            best_abund = db['abundances'][i]
            best_symbol = sym

    return best_symbol


def _reference_q_isotope(element: str) -> Optional[tuple]:
    """Return (isotope_symbol, I, qm) for the most abundant I >= 1 isotope.

    Uses the hardcoded MATLAB table for consistency. Returns None if no
    quadrupole reference isotope exists.
    """
    Z = _element_to_number(element)
    if Z < 1 or Z > len(_QREF_MASS_NO):
        return None
    mass_no = _QREF_MASS_NO[Z - 1]
    if mass_no == 0:
        return None
    iso_symbol = f"{mass_no}{_number_to_element(Z)}"
    try:
        I = float(nucspin(iso_symbol))
        qm = float(nucqmom(iso_symbol))  # in barn
    except (ValueError, KeyError):
        return None
    return (iso_symbol, I, qm)


# ---------------------------------------------------------------------------
# Tensor processing utilities
# ---------------------------------------------------------------------------
def _symmetrize_g(g_raw: np.ndarray) -> np.ndarray:
    """Symmetrize raw g-matrix: g_sym = sqrt(g_raw^T @ g_raw), then average."""
    g2 = g_raw.T @ g_raw
    g_sym = np.real(sqrtm(g2))
    return (g_sym + g_sym.T) / 2


def _ensure_righthanded(V: np.ndarray) -> np.ndarray:
    """Flip first eigenvector if det(V) < 0 to enforce right-handed frame."""
    if np.linalg.det(V) < 0:
        V = V.copy()
        V[:, 0] = -V[:, 0]
    return V


def _diag_and_frame(M: np.ndarray) -> tuple:
    """Diagonalize a real symmetric 3×3 matrix.

    Returns (eigenvalues[3], euler_angles[3]).
    Enforces right-handed eigenvector frame.
    """
    vals, vecs = np.linalg.eigh(M)
    vecs = _ensure_righthanded(vecs)
    angles = eulang(vecs.T)
    return vals, angles


def _efg_to_qtensor(efg_au: np.ndarray, element: str) -> Optional[tuple]:
    """Convert 3×3 EFG matrix (atomic units) → Q principal values (MHz) + Euler angles.

    Returns None if the element has no quadrupole reference isotope.
    """
    qref = _reference_q_isotope(element)
    if qref is None:
        return None

    _iso_symbol, I, qm_barn = qref

    # Convert EFG to SI (V/m²)
    efg_si = efg_au * _EFG_AU_TO_SI

    # Diagonalize
    eq_vals, R = np.linalg.eigh(efg_si)

    # Sort by |eigenvalue| (standard EFG convention)
    idx = np.argsort(np.abs(eq_vals))
    eq_vals = eq_vals[idx]
    R = R[:, idx]
    R = _ensure_righthanded(R)

    # Quadrupole tensor principal values
    eQ = _ECHARGE * qm_barn * _BARN  # nuclear electric quadrupole moment (C·m²)
    e2qQh = _ECHARGE * qm_barn * _BARN * eq_vals[2] / _PLANCK / 1e6  # MHz
    K = e2qQh / (4.0 * I) / (2.0 * I - 1.0)
    eta = (eq_vals[0] - eq_vals[1]) / eq_vals[2] if abs(eq_vals[2]) > 0 else 0.0
    Q_vals = np.array([-(1 - eta), -(1 + eta), 2.0]) * K
    Q_frame = eulang(R.T)
    return Q_vals, Q_frame


def _efg_pv_to_qtensor(efg_pv_si: np.ndarray, element_Z: int) -> np.ndarray:
    """Convert EFG principal values (SI, V/m²) to Q tensor principal values (MHz).

    Used by the binary property parser where EFG is already diagonalized.
    Matches MATLAB's ``efg2Q`` function.
    """
    if element_Z < 1 or element_Z > len(_QREF_MASS_NO):
        return np.zeros(3)
    mass_no = _QREF_MASS_NO[element_Z - 1]
    if mass_no == 0:
        return np.zeros(3)

    iso_symbol = f"{mass_no}{_number_to_element(element_Z)}"
    try:
        eQ = _ECHARGE * float(nucqmom(iso_symbol)) * _BARN
        I = float(nucspin(iso_symbol))
    except (ValueError, KeyError):
        return np.zeros(3)

    # Q principal values in SI (J), then convert to MHz
    Q_si = eQ / (2.0 * I) / (2.0 * I - 1.0) * efg_pv_si
    Q_mhz = Q_si / _PLANCK / 1e6
    return Q_mhz


# ---------------------------------------------------------------------------
# Line-level parsing helpers
# ---------------------------------------------------------------------------
def _readmatrix(lines: list, start_col: int = 0) -> np.ndarray:
    """Read a 3×3 matrix from 3 consecutive text lines."""
    M = np.zeros((3, 3))
    for i in range(3):
        text = lines[i] if start_col == 0 else lines[i][start_col:]
        vals = [float(x) for x in text.split()]
        # Take last 3 values (skip row index if present)
        M[i, :] = vals[-3:]
    return M


def _readmatrix_proptxt(lines: list) -> np.ndarray:
    """Read 3×3 matrix from property.txt format (has row index in col 0)."""
    M = np.zeros((3, 3))
    for i in range(3):
        vals = [float(x) for x in lines[i].split()]
        M[i, :] = vals[1:]  # skip row index
    return M


def _findheader(header: str, lines: list, krange: range) -> Optional[int]:
    """Find line index starting with ``header`` in the given range."""
    for k in krange:
        if lines[k].startswith(header):
            return k
    return None


def _get_orca_version(filepath: Path) -> str:
    """Extract ORCA version string from first 50 lines of main output file."""
    if not filepath.exists():
        return ''
    with open(filepath, 'r', errors='replace') as f:
        for i, line in enumerate(f):
            if i >= 50:
                break
            m = re.search(r'\d+\.\d+\.\d+', line)
            if m:
                return m.group()
    return ''


# ---------------------------------------------------------------------------
# OrcaData — metadata container
# ---------------------------------------------------------------------------
@dataclass
class OrcaData:
    """Metadata from ORCA calculation (raw tensors, coordinates, charges)."""
    graw: Optional[np.ndarray] = None
    g_sym: Optional[np.ndarray] = None
    Draw: Optional[np.ndarray] = None
    hfc: Optional[list] = None
    efg: Optional[list] = None
    xyz: Optional[np.ndarray] = None
    elements: Optional[list] = None
    atom_numbers: Optional[list] = None
    charge: Optional[int] = None
    multiplicity: Optional[int] = None
    mulliken_charge: Optional[np.ndarray] = None
    mulliken_spin: Optional[np.ndarray] = None
    orca_version: str = ''
    input_file: str = ''
    nucs_idx: Optional[list] = None


# ---------------------------------------------------------------------------
# Main output parser
# ---------------------------------------------------------------------------
def _parse_mainoutput(filepath: Path) -> list:
    """Parse main ORCA text output file (.out).

    Returns list of dicts (one per structure), each containing parsed tensors.
    """
    # Try UTF-8 first, then UTF-16 (some .oof files are UTF-16-LE)
    raw = filepath.read_bytes()
    if raw[:2] in (b'\xff\xfe', b'\xfe\xff'):
        text = raw.decode('utf-16', errors='replace')
    else:
        text = raw.decode('utf-8', errors='replace')
    all_lines = text.splitlines()

    # Remove empty / single-char lines (match MATLAB behavior)
    lines = [l for l in all_lines if len(l.strip()) > 1]
    nLines = len(lines)

    # Assert ORCA output
    is_orca = any('O   R   C   A' in lines[i] for i in range(min(5, nLines)))
    if not is_orca:
        raise ValueError("This is not an ORCA output file.")

    # ORCA version
    orca_version = ''
    for l in lines[:50]:
        m = re.search(r'\d+\.\d+\.\d+', l)
        if m:
            orca_version = m.group()
            break

    # Extract input file contents
    input_file = ''
    k = 0
    while k < nLines and not lines[k].startswith('|'):
        k += 1
    start_input = k
    while k < nLines and lines[k].startswith('|'):
        k += 1
    if start_input < k:
        input_lines = lines[start_input:k]
        input_file = '\n'.join(re.sub(r'^\|\s*\d+>\s+', '', l) for l in input_lines)

    # Determine run type (single vs multi-structure)
    run_types = [
        ('* Single Point Calculation *', ''),
        ('* Multiple XYZ Scan Calculation *', 'MULTIPLE XYZ STEP'),
        ('* Parameter Scan Calculation *', 'TRAJECTORY STEP'),
        ('*    Relaxed Surface Scan    *', 'RELAXED SURFACE SCAN STEP'),
    ]

    run_type = -1
    while k < nLines and run_type < 0:
        for i, (header, _step) in enumerate(run_types):
            if header in lines[k]:
                run_type = i
                break
        k += 1

    if run_type < 0:
        raise ValueError("Could not determine run type from ORCA output file.")

    # Find structure boundaries
    _step_marker = run_types[run_type][1]
    if run_type == 0:  # single point
        start_indices = [k]
    else:
        start_indices = [i for i in range(nLines) if _step_marker in lines[i]]

    nStructures = len(start_indices)
    results = []

    for iStruct in range(nStructures):
        k_start = start_indices[iStruct]
        k_end = start_indices[iStruct + 1] if iStruct < nStructures - 1 else nLines
        krange = range(k_start, k_end)

        d = {
            'orca_version': orca_version,
            'input_file': input_file,
        }

        # --- Cartesian coordinates ---
        k = _findheader('CARTESIAN COORDINATES (ANGSTROEM)', lines, krange)
        if k is not None:
            k += 2  # skip header and dash line
            xyz_list = []
            elements = []
            while k < k_end and not lines[k].startswith('-'):
                parts = lines[k].split()
                if len(parts) >= 4:
                    elements.append(parts[0])
                    xyz_list.append([float(parts[1]), float(parts[2]), float(parts[3])])
                k += 1
            d['xyz'] = np.array(xyz_list) if xyz_list else np.empty((0, 3))
            d['elements'] = elements
            d['atom_numbers'] = [_element_to_number(e) for e in elements]
            d['nAtoms'] = len(elements)
        else:
            d['xyz'] = np.empty((0, 3))
            d['elements'] = []
            d['atom_numbers'] = []
            d['nAtoms'] = 0

        nAtoms = d['nAtoms']

        # --- Total charge ---
        d['charge'] = 0
        for ki in krange:
            if re.match(r'^\s*Total Charge', lines[ki]):
                m = re.search(r'(\d+)\s*$', lines[ki])
                if m:
                    d['charge'] = int(m.group(1))
                break

        # --- Multiplicity ---
        d['multiplicity'] = 1
        for ki in krange:
            if re.match(r'^\s*Multiplicity', lines[ki]):
                m = re.search(r'(\d+)\s*$', lines[ki])
                if m:
                    d['multiplicity'] = int(m.group(1))
                break
        d['S'] = (d['multiplicity'] - 1) / 2.0

        # --- Mulliken charges and spin populations ---
        mulliken_titles = [
            'MULLIKEN ATOMIC CHARGES AND SPIN DENSITIES',
            'MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS',
        ]
        mulliken_k = None
        for ki in krange:
            for title in mulliken_titles:
                if lines[ki].strip() == title:
                    mulliken_k = ki
                    break
            if mulliken_k is not None:
                break

        if mulliken_k is not None and nAtoms > 0:
            mk = mulliken_k + 2  # skip header + blank/dash line
            mull_charge = np.zeros(nAtoms)
            mull_spin = np.zeros(nAtoms)
            for iAtom in range(nAtoms):
                if mk + iAtom >= k_end:
                    break
                line = lines[mk + iAtom]
                # Format: "   0 N  :   -0.276959    0.549162" (col 8+)
                vals = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', line[8:])
                if len(vals) >= 2:
                    mull_charge[iAtom] = float(vals[0])
                    mull_spin[iAtom] = float(vals[1])
            d['mulliken_charge'] = mull_charge
            d['mulliken_spin'] = mull_spin
        else:
            d['mulliken_charge'] = np.zeros(0)
            d['mulliken_spin'] = np.zeros(0)

        # --- g-matrix ---
        k = _findheader('ELECTRONIC G-MATRIX', lines, krange)
        if k is not None:
            # Find "The g-matrix" line
            while k < k_end and 'The g-matrix' not in lines[k]:
                k += 1
            g_raw = _readmatrix(lines[k + 1: k + 4])
            g_sym = _symmetrize_g(g_raw)
            g_vals, g_vecs = np.linalg.eigh(g_sym)
            g_vecs = _ensure_righthanded(g_vecs)
            g_frame = eulang(g_vecs.T)
            d['g_raw'] = g_raw
            d['g_sym'] = g_sym
            d['g_vals'] = g_vals
            d['g_frame'] = g_frame
        else:
            d['g_raw'] = None
            d['g_sym'] = None
            d['g_vals'] = None
            d['g_frame'] = None

        # --- Zero-field splitting ---
        k = _findheader('ZERO-FIELD-SPLITTING TENSOR', lines, krange)
        if k is not None:
            # raw-matrix is at k+3..k+5 (after header, blank, "raw-matrix :" line)
            # Find the "raw-matrix" line
            ki = k
            while ki < k_end and 'raw-matrix' not in lines[ki]:
                ki += 1
            D_raw_cm1 = _readmatrix(lines[ki + 1: ki + 4])

            # Diagonalize (recalcVecs = true in MATLAB)
            D_vals_cm1, D_vecs = np.linalg.eigh(D_raw_cm1)
            D_vecs = _ensure_righthanded(D_vecs)

            D_raw_mhz = D_raw_cm1 * _CM1_TO_MHZ
            D_vals_mhz = D_vals_cm1 * _CM1_TO_MHZ
            D_frame = eulang(D_vecs.T)

            d['D_raw'] = D_raw_mhz
            d['D_vals'] = D_vals_mhz
            d['D_frame'] = D_frame
        else:
            d['D_raw'] = None
            d['D_vals'] = None
            d['D_frame'] = None

        # --- Hyperfine and EFG ---
        A_data = [None] * nAtoms     # per-atom A principal values
        AFrame_data = [None] * nAtoms
        Araw_data = [None] * nAtoms
        efg_data = [None] * nAtoms   # per-atom raw EFG matrix (au)
        Q_data = [None] * nAtoms
        QFrame_data = [None] * nAtoms

        k = _findheader('ELECTRIC AND MAGNETIC HYPERFINE STRUCTURE', lines, krange)
        if k is not None:
            current_atom = -1
            current_element = ''
            ki = k
            while ki < k_end:
                line = lines[ki]

                # Detect nucleus block
                if re.match(r'^\s*Nucleus\s+', line):
                    # Extract atom index (0-based in ORCA)
                    m = re.search(r'Nucleus\s+(\d+)', line)
                    if m:
                        current_atom = int(m.group(1))  # 0-based
                        current_element = d['elements'][current_atom] if current_atom < nAtoms else ''

                # Raw HFC matrix
                elif re.match(r'^\s*(Raw HFC matrix|Total HFC matrix)', line):
                    # Check if next line is dashes
                    idx = ki + 1
                    if idx < k_end and '---' in lines[idx][:6]:
                        idx += 1
                    hfc_matrix = _readmatrix(lines[idx: idx + 3])
                    Araw_data[current_atom] = hfc_matrix

                    # Find A(Tot) line
                    aidx = idx + 3
                    while aidx < k_end and ' A(Tot)' not in lines[aidx][:8]:
                        aidx += 1

                    if aidx < k_end:
                        # Parse principal values from A(Tot) line (after initial label)
                        a_text = lines[aidx][12:]  # skip " A(Tot)     "
                        a_vals = [float(x) for x in re.findall(
                            r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', a_text)][:3]
                        A_data[current_atom] = np.array(a_vals)

                        # Orientation matrix: 2 lines down from A(Tot)
                        # "Orientation:" header, then X, Y, Z rows
                        oidx = aidx + 2
                        R = np.zeros((3, 3))
                        for row in range(3):
                            if oidx + row < k_end:
                                ovals = [float(x) for x in re.findall(
                                    r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?',
                                    lines[oidx + row])]
                                if len(ovals) >= 3:
                                    R[row, :] = ovals[:3]
                        AFrame_data[current_atom] = eulang(R.T)

                # Raw EFG matrix
                elif re.match(r'^\s*Raw EFG matrix', line):
                    idx = ki + 1
                    if idx < k_end and '---' in lines[idx][:6]:
                        idx += 1
                    efg_matrix = _readmatrix(lines[idx: idx + 3])
                    efg_data[current_atom] = efg_matrix

                    # Convert to Q tensor
                    result = _efg_to_qtensor(efg_matrix, current_element)
                    if result is not None:
                        Q_data[current_atom] = result[0]
                        QFrame_data[current_atom] = result[1]

                ki += 1

        d['A'] = A_data
        d['AFrame'] = AFrame_data
        d['Araw'] = Araw_data
        d['efg'] = efg_data
        d['Q'] = Q_data
        d['QFrame'] = QFrame_data

        results.append(d)

    return results


# ---------------------------------------------------------------------------
# Binary property parser
# ---------------------------------------------------------------------------
def _parse_propbin(filepath: Path) -> list:
    """Parse ORCA binary .prop file (ORCA < v5).

    Returns list with single dict of parsed data.
    """
    data = filepath.read_bytes()
    if len(data) < 4:
        raise ValueError(f"Binary property file {filepath} is too small.")

    # Detect endianness
    first_id = struct.unpack_from('<i', data, 0)[0]
    if first_id == -1:
        return []  # empty file
    max_id = 53
    if first_id < 0 or first_id > max_id:
        endian = '>'  # big-endian
    else:
        endian = '<'  # little-endian

    offset = 0
    S = None
    xyz = None
    charge = None
    gpv = None
    g_frame = None
    Dpv = None
    D_frame = None
    Apv = {}     # atom_index → [3] principal values
    AFrame = {}  # atom_index → [3] Euler angles
    efg = {}     # atom_index → [3] EFG principal values (SI)
    efgFrame = {}
    atoms = None

    while offset < len(data) - 4:
        prop_id = struct.unpack_from(f'{endian}i', data, offset)[0]
        offset += 4

        if prop_id == -1:
            break
        if prop_id < 0 or prop_id > max_id:
            raise ValueError(f"Unknown property ID {prop_id} in binary file.")

        nRows = struct.unpack_from(f'{endian}i', data, offset)[0]
        offset += 4
        nCols = struct.unpack_from(f'{endian}i', data, offset)[0]
        offset += 4

        nElements = nRows * nCols
        values = np.array(struct.unpack_from(f'{endian}{nElements}d', data, offset))
        offset += nElements * 8

        # MATLAB: data = reshape(data, nColumns, nRows) — fills column-major
        # Python equivalent: Fortran order to match MATLAB's column-major fill
        block = values.reshape(nCols, nRows, order='F')

        # --- Charge (ID 40) ---
        if prop_id == 40:
            charge = int(block[0, 0])

        # --- Spin multiplicity (ID 41) ---
        elif prop_id == 41:
            S = (block[0, 0] - 1) / 2.0

        # --- Coordinates (IDs 17, 42) ---
        elif prop_id in (17, 42):
            xyz = values.reshape((3, -1), order='F').T  # EasySpin: reshape(data,3,[]).' (Bohr)
            xyz = xyz * _BOHR_TO_ANGSTROM

        # --- Atom numbers (IDs 18, 43) ---
        elif prop_id in (18, 43):
            atoms = block.flatten().astype(int)

        # --- g-tensor (IDs 5, 14, 23, 30, 27) ---
        elif prop_id in (5, 14, 23, 30, 27):
            col0 = block[:, 0] if block.ndim == 2 else block
            gpv = col0[:3]
            if np.all(gpv == 0):
                g_frame = np.zeros(3)
            else:
                R = col0[3:12].reshape(3, 3, order='F')   # MATLAB reshape(data(4:12),3,3): column-major
                g_frame = eulang(R.T)

        # --- D-tensor (IDs 4, 13, 22, 29, 36) ---
        elif prop_id in (4, 13, 22, 29, 36):
            col0 = block[:, 0] if block.ndim == 2 else block
            Dpv = col0[2:5] * _CM1_TO_MHZ  # cm⁻¹ → MHz
            R = col0[5:14].reshape(3, 3, order='F')   # column-major, as EasySpin orca2easyspin_propbin
            D_frame = eulang(R.T)

        # --- A-tensors (IDs 6, 15, 24, 31, 38) ---
        elif prop_id in (6, 15, 24, 31, 38):
            nNucs = block.shape[1] if block.ndim == 2 else 1
            for iNuc in range(nNucs):
                col = block[:, iNuc] if block.ndim == 2 else block
                atom_idx = int(col[0])  # 0-based in ORCA
                Apv[atom_idx] = col[2:5]
                M = col[5:14].reshape(3, 3, order='F')   # EasySpin: R = reshape(data(6:14),3,3).'; AFrame = eulang(R.')
                AFrame[atom_idx] = eulang(M)

        # --- EFG tensors (IDs 7, 16, 25, 32, 39) ---
        elif prop_id in (7, 16, 25, 32, 39):
            nNucs = block.shape[1] if block.ndim == 2 else 1
            for iNuc in range(nNucs):
                col = block[:, iNuc] if block.ndim == 2 else block
                atom_idx = int(col[0])
                efg_au = col[1:4]  # atomic units
                efg_si = efg_au * _EFG_AU_TO_SI
                efg[atom_idx] = efg_si
                M = col[4:13].reshape(3, 3, order='F')   # EasySpin: R = reshape(data(5:13),3,3).'; efgFrame = eulang(R.')
                efgFrame[atom_idx] = eulang(M)

    # Assemble result dict
    nAtoms = len(atoms) if atoms is not None else 0
    d = {
        'orca_version': '',
        'input_file': '',
        'S': S if S is not None else 0.5,
        'multiplicity': int(2 * S + 1) if S is not None else 2,
        'charge': charge if charge is not None else 0,
        'xyz': xyz if xyz is not None else np.empty((0, 3)),
        'elements': [_number_to_element(int(a)) for a in atoms] if atoms is not None else [],
        'atom_numbers': list(atoms) if atoms is not None else [],
        'nAtoms': nAtoms,
        'mulliken_charge': np.zeros(0),
        'mulliken_spin': np.zeros(0),
    }

    # g-tensor
    if gpv is not None:
        d['g_vals'] = gpv
        d['g_frame'] = g_frame
        d['g_raw'] = None
        d['g_sym'] = None
    else:
        d['g_vals'] = None
        d['g_frame'] = None
        d['g_raw'] = None
        d['g_sym'] = None

    # D-tensor
    if Dpv is not None:
        d['D_vals'] = Dpv
        d['D_frame'] = D_frame
        d['D_raw'] = None
    else:
        d['D_vals'] = None
        d['D_frame'] = None
        d['D_raw'] = None

    # A-tensor (pad with None for atoms without HFC)
    A_list = [None] * nAtoms
    AFrame_list = [None] * nAtoms
    for idx, vals in Apv.items():
        if idx < nAtoms:
            A_list[idx] = vals
    for idx, vals in AFrame.items():
        if idx < nAtoms:
            AFrame_list[idx] = vals

    # Q-tensor from EFG
    Q_list = [None] * nAtoms
    QFrame_list = [None] * nAtoms
    efg_list = [None] * nAtoms

    if atoms is not None:
        # Build padded arrays matching MATLAB behavior
        nAtoms_A = max(Apv.keys()) + 1 if Apv else 0
        nAtoms_efg = max(efg.keys()) + 1 if efg else 0

        if nAtoms_A > 0:
            Apv_array = np.zeros((max(nAtoms, nAtoms_A), 3))
            AFrame_array = np.zeros((max(nAtoms, nAtoms_A), 3))
            for idx, vals in Apv.items():
                Apv_array[idx] = vals
            for idx, vals in AFrame.items():
                AFrame_array[idx] = vals

        for idx in range(nAtoms):
            if idx in efg:
                efg_list[idx] = efg[idx]
                Q_pv = _efg_pv_to_qtensor(efg[idx], int(atoms[idx]))
                if not np.all(Q_pv == 0):
                    Q_list[idx] = Q_pv
                    QFrame_list[idx] = efgFrame.get(idx, np.zeros(3))

    d['A'] = A_list
    d['AFrame'] = AFrame_list
    d['Araw'] = [None] * nAtoms
    d['efg'] = efg_list
    d['Q'] = Q_list
    d['QFrame'] = QFrame_list

    return [d]


# ---------------------------------------------------------------------------
# Text property parser
# ---------------------------------------------------------------------------
def _parse_proptxt(filepath: Path) -> list:
    """Parse ORCA text property file (_property.txt, ORCA >= v5).

    Returns list of dicts (one per structure).
    """
    with open(filepath, 'r', errors='replace') as f:
        all_lines = f.readlines()
    lines = [l.rstrip('\n').rstrip('\r') for l in all_lines]

    if len(lines) < 3 or '!PROPERTIES!' not in lines[1]:
        raise ValueError("This is not a valid ORCA text property file.")

    # Read geometries
    geom_indices = [i for i, l in enumerate(lines) if '!GEOMETRY!' in l and '!GEOMETRIES!' not in l]
    nStructures = len(geom_indices)

    data_list = []
    for gs in range(nStructures):
        idx0 = geom_indices[gs]
        # nAtoms from next line
        nAtoms_line = lines[idx0 + 1]
        nAtoms = int(nAtoms_line.split(':')[-1].strip()) if ':' in nAtoms_line else int(nAtoms_line.strip())

        d = {
            'orca_version': '',
            'input_file': '',
            'nAtoms': nAtoms,
            'xyz': np.zeros((nAtoms, 3)),
            'elements': [],
            'atom_numbers': [],
            'charge': 0,
            'multiplicity': 1,
            'S': 0.0,
            'mulliken_charge': np.zeros(0),
            'mulliken_spin': np.zeros(0),
            'g_raw': None, 'g_sym': None, 'g_vals': None, 'g_frame': None,
            'D_raw': None, 'D_vals': None, 'D_frame': None,
            'A': [None] * nAtoms,
            'AFrame': [None] * nAtoms,
            'Araw': [None] * nAtoms,
            'efg': [None] * nAtoms,
            'Q': [None] * nAtoms,
            'QFrame': [None] * nAtoms,
        }

        # Parse atom coordinates
        idx = idx0 + 3
        for iAtom in range(nAtoms):
            idx += 1
            if idx >= len(lines):
                break
            line = lines[idx]
            # Format: " 0   N      0.000000    0.000000    0.000000"
            tok = re.match(r'\s*(\d+)\s+([a-zA-Z]+)\s+(.*)', line)
            if tok:
                d['elements'].append(tok.group(2))
                coords = [float(x) for x in tok.group(3).split()][:3]
                d['xyz'][iAtom] = coords

        d['atom_numbers'] = [_element_to_number(e) for e in d['elements']]
        data_list.append(d)

    # Parse sections
    section_indices = [i for i, l in enumerate(lines) if l.startswith('$')]

    for si in section_indices:
        title = lines[si].strip()

        # Determine which structure this section belongs to
        # ORCA uses 1-based "geom. index" — convert to 0-based
        if si + 2 < len(lines):
            m = re.search(r'(\d+)\s*$', lines[si + 2])
            iStruct = int(m.group(1)) - 1 if m else 0  # 1-based → 0-based
        else:
            iStruct = 0
        if iStruct < 0 or iStruct >= nStructures:
            continue

        d = data_list[iStruct]
        nAtoms = d['nAtoms']

        if title == '$ Calculation_Info':
            if si + 4 < len(lines):
                mult_line = lines[si + 4]
                mult_val = mult_line.split(':')[-1].strip() if ':' in mult_line else mult_line.strip()
                d['multiplicity'] = int(float(mult_val))
                d['S'] = (d['multiplicity'] - 1) / 2.0
            if si + 5 < len(lines):
                charge_line = lines[si + 5]
                charge_val = charge_line.split(':')[-1].strip() if ':' in charge_line else charge_line.strip()
                d['charge'] = int(float(charge_val))

        elif title == '$ EPRNMR_GTensor':
            g_raw = _readmatrix_proptxt(lines[si + 8: si + 11])
            g_sym = _symmetrize_g(g_raw)
            g_sym = (g_sym + g_sym.T) / 2
            g_vals, g_vecs = np.linalg.eigh(g_sym)
            g_vecs = _ensure_righthanded(g_vecs)
            d['g_raw'] = g_raw
            d['g_sym'] = g_sym
            d['g_vals'] = g_vals
            d['g_frame'] = eulang(g_vecs.T)

        elif title == '$ EPRNMR_DTensor':
            D_raw_cm1 = _readmatrix_proptxt(lines[si + 8: si + 11])
            D_vals_cm1, D_vecs = np.linalg.eigh(D_raw_cm1)
            D_vecs = _ensure_righthanded(D_vecs)
            d['D_raw'] = D_raw_cm1 * _CM1_TO_MHZ
            d['D_vals'] = D_vals_cm1 * _CM1_TO_MHZ
            d['D_frame'] = eulang(D_vecs.T)

        elif title == '$ EPRNMR_ATensor':
            # Number of stored nuclei
            val_str = re.search(r'\d+\s*$', lines[si + 4])
            nStored = int(val_str.group()) if val_str else 0

            i = si + 7
            for n in range(nStored):
                if i >= len(lines):
                    break
                # MATLAB: regexp(line, '(\d+)\W+(\w+)\W*$', 'tokens')
                m = re.search(r'(\d+)\W+(\w+)\W*$', lines[i])
                if not m:
                    break
                atom_idx = int(m.group(1))  # 0-based

                A_raw = _readmatrix_proptxt(lines[i + 6: i + 9])
                A_vecs = _readmatrix_proptxt(lines[i + 11: i + 14])
                # A vals from line i+16 (skip row index in col 0-8)
                a_text = lines[i + 16][8:] if i + 16 < len(lines) else ''
                a_vals = [float(x) for x in a_text.split()][:3]

                d['Araw'][atom_idx] = A_raw
                d['A'][atom_idx] = np.array(a_vals) if len(a_vals) == 3 else None

                if not np.all(A_vecs == 0):
                    d['AFrame'][atom_idx] = eulang(A_vecs.T)
                else:
                    d['AFrame'][atom_idx] = np.zeros(3)

                i += 18

        elif title == '$ EPRNMR_EFGTensor':
            val_str = re.search(r'\d+\s*$', lines[si + 4])
            nStored = int(val_str.group()) if val_str else 0

            # Check if rho is present
            rho_present = 'true' in lines[si + 7].lower() if si + 7 < len(lines) else False
            step = 19 if rho_present else 18

            i = si + 8
            for n in range(nStored):
                if i >= len(lines):
                    break
                m = re.search(r'(\d+)\W+(\w+)\W*$', lines[i])
                if not m:
                    break
                atom_idx = int(m.group(1))

                Q_raw = _readmatrix_proptxt(lines[i + 6: i + 9])
                Q_vecs = _readmatrix_proptxt(lines[i + 11: i + 14])
                # Q vals from line i+16 (after "V(El+Nuc)  ")
                q_text = lines[i + 16][11:] if i + 16 < len(lines) else ''
                q_vals = [float(x) for x in q_text.split()][:3]

                d['efg'][atom_idx] = Q_raw
                d['Q'][atom_idx] = np.array(q_vals) if len(q_vals) == 3 else None
                d['QFrame'][atom_idx] = eulang(Q_vecs.T)

                i += step

        elif title == '$ EPRNMR_QTensor':
            warnings.warn(
                "EPRNMR_QTensor not supported for property.txt files. "
                "Use the main output file instead.",
                stacklevel=2
            )

    return data_list


# ---------------------------------------------------------------------------
# SpinSystem assembly
# ---------------------------------------------------------------------------
def _build_spinsystem(d: dict, hf_cutoff: float = 0.0) -> tuple:
    """Convert parsed data dict into a SpinSystem + OrcaData pair."""
    nAtoms = d['nAtoms']
    S = d['S']

    # g-tensor
    g_vals = d.get('g_vals')
    g_frame = d.get('g_frame')

    # D-tensor
    D_vals = d.get('D_vals')
    D_frame = d.get('D_frame')

    # Compile nuclear data: only atoms with A or Q data
    nuc_symbols = []
    nuc_indices = []  # 1-based atom indices
    A_rows = []
    AFrame_rows = []
    Q_rows = []
    QFrame_rows = []

    for iAtom in range(nAtoms):
        has_A = d['A'][iAtom] is not None
        has_Q = d['Q'][iAtom] is not None
        if has_A or has_Q:
            element = d['elements'][iAtom]
            ref_iso = _reference_isotope(element)
            if ref_iso is None:
                ref_iso = element  # fallback
            nuc_symbols.append(ref_iso)
            nuc_indices.append(iAtom + 1)  # 1-based

            if has_A:
                A_rows.append(d['A'][iAtom])
                AFrame_rows.append(d['AFrame'][iAtom] if d['AFrame'][iAtom] is not None else np.zeros(3))
            else:
                A_rows.append(np.zeros(3))
                AFrame_rows.append(np.zeros(3))

            if has_Q:
                Q_rows.append(d['Q'][iAtom])
                QFrame_rows.append(d['QFrame'][iAtom] if d['QFrame'][iAtom] is not None else np.zeros(3))
            else:
                Q_rows.append(np.zeros(3))
                QFrame_rows.append(np.zeros(3))

    # Apply hyperfine cutoff
    if hf_cutoff > 0 and A_rows:
        keep = []
        for i, a in enumerate(A_rows):
            if np.max(np.abs(a)) > abs(hf_cutoff):
                keep.append(i)
        nuc_symbols = [nuc_symbols[i] for i in keep]
        nuc_indices = [nuc_indices[i] for i in keep]
        A_rows = [A_rows[i] for i in keep]
        AFrame_rows = [AFrame_rows[i] for i in keep]
        Q_rows = [Q_rows[i] for i in keep]
        QFrame_rows = [QFrame_rows[i] for i in keep]

    # Build SpinSystem kwargs
    kwargs = {'S': [S]}

    if g_vals is not None:
        kwargs['g'] = g_vals.tolist()
        if g_frame is not None:
            kwargs['gFrame'] = g_frame.tolist()

    if D_vals is not None:
        kwargs['D'] = D_vals.tolist()
        if D_frame is not None:
            kwargs['DFrame'] = D_frame.tolist()

    if nuc_symbols:
        kwargs['Nucs'] = nuc_symbols
        kwargs['A'] = np.array(A_rows)
        kwargs['AFrame'] = np.array(AFrame_rows)
        if any(not np.all(q == 0) for q in Q_rows):
            kwargs['Q'] = np.array(Q_rows)
            kwargs['QFrame'] = np.array(QFrame_rows)

    sys = SpinSystem(**kwargs)

    # Build OrcaData
    metadata = OrcaData(
        graw=d.get('g_raw'),
        g_sym=d.get('g_sym'),
        Draw=d.get('D_raw'),
        hfc=d.get('Araw', []),
        efg=d.get('efg', []),
        xyz=d.get('xyz'),
        elements=d.get('elements', []),
        atom_numbers=d.get('atom_numbers', []),
        charge=d.get('charge', 0),
        multiplicity=d.get('multiplicity', 1),
        mulliken_charge=d.get('mulliken_charge', np.zeros(0)),
        mulliken_spin=d.get('mulliken_spin', np.zeros(0)),
        orca_version=d.get('orca_version', ''),
        input_file=d.get('input_file', ''),
        nucs_idx=nuc_indices,
    )

    return sys, metadata


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def orca2torchspin(
    filename: Union[str, Path],
    hf_cutoff: float = 0.0,
) -> tuple["SpinSystem", "OrcaData"] | list[tuple["SpinSystem", "OrcaData"]]:
    """Load ORCA calculation output → torchspin SpinSystem.

    Parameters
    ----------
    filename : str or Path
        Path to ORCA output file (``.out``, ``.prop``, or ``_property.txt``).
    hf_cutoff : float, optional
        Hyperfine coupling cutoff in MHz. Nuclei with ``max(|A|) <= cutoff``
        are excluded.  Default: 0 (include all nuclei with non-zero HFC).

    Returns
    -------
    (SpinSystem, OrcaData) for single-structure files, or
    list of (SpinSystem, OrcaData) for multi-structure files (scans).
    """
    filepath = Path(filename)
    if not filepath.exists():
        raise FileNotFoundError(f"Cannot find ORCA output file: {filepath}")

    # Detect file type
    name = filepath.name
    ext = filepath.suffix.lower()

    if ext == '.prop':
        readmode = 'propbin'
    elif name.endswith('_property.txt'):
        readmode = 'proptxt'
    else:
        readmode = 'mainout'

    # Version guard for buggy ORCA 5.0.0-5.0.3 property files
    if readmode == 'proptxt':
        # Try to find the main output file to check version
        main_path = filepath.parent / filepath.name.replace('_property.txt', '')
        # Try common extensions
        for ext_try in ['', '.out', '.log']:
            candidate = Path(str(main_path) + ext_try)
            if candidate.exists():
                version = _get_orca_version(candidate)
                if version in ('5.0.0', '5.0.1', '5.0.2', '5.0.3'):
                    raise ValueError(
                        f"Cannot read property file for ORCA version {version}. "
                        "Use main output file instead."
                    )
                break

    # Parse
    if readmode == 'mainout':
        parsed_list = _parse_mainoutput(filepath)
    elif readmode == 'propbin':
        parsed_list = _parse_propbin(filepath)
    elif readmode == 'proptxt':
        parsed_list = _parse_proptxt(filepath)
    else:
        raise ValueError(f"Unknown read mode: {readmode}")

    if not parsed_list:
        raise ValueError(f"No data found in ORCA file: {filepath}")

    # Build SpinSystems
    results = []
    for d in parsed_list:
        sys, data = _build_spinsystem(d, hf_cutoff)
        results.append((sys, data))

    if len(results) == 1:
        return results[0]
    return results
