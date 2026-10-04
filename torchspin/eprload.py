"""
Load experimental EPR data from various formats.

This module provides functionality to load EPR spectroscopy data from all major
spectrometer file formats, matching MATLAB EasySpin's eprload.m.

Supported formats:
  Bruker BES3T:        .DTA, .DSC
  Bruker ESP/WinEPR:   .spc, .par
  SpecMan:             .d01, .exp
  Magnettech binary:   .spe
  Magnettech XML:      .xml
  Active Spectrum:     .ESR
  Adani text:          .dat
  Adani JSON:          .json
  CIQTEK:              .epr
  JEOL:                (no standard extension)
  MAGRES:              .plt
  qese/tryscore:       .eco
  Varian E9 ETH:       .spk, .ref
  ESE Weizmann/ETH:    .d00

Based on MATLAB EasySpin's eprload.m and private/eprload_*.m
"""

import json
import struct
import warnings
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any, List
import re
import xml.etree.ElementTree as ET
import base64


def eprload(
    filename: Union[str, Path],
    scaling: str = ''
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load experimental EPR data from file.

    Parameters
    ----------
    filename : str or Path
        Path to the data file. Format is identified by extension:
        - Bruker BES3T: .DTA, .DSC
        - Bruker ESP/WinEPR: .spc, .par
        - SpecMan: .d01, .exp
        - Magnettech: .spe (binary), .xml (XML)
        - Active Spectrum: .ESR
        - Adani: .dat, .json
        - CIQTEK: .epr
        - JEOL: (auto-detected from header)
        - MAGRES: .PLT
        - qese/tryscore: .eco
        - Varian: .spk, .ref
        - ESE: .d00
    scaling : str, optional
        Scaling options for Bruker files:
        - 'n': divide by number of scans
        - 'P': divide by square root of microwave power (mW)
        - 'G': divide by receiver gain
        - 'T': multiply by temperature (K)
        - Can combine: e.g. 'nPG' applies all three

    Returns
    -------
    x : ndarray
        Abscissa (magnetic field, time, frequency, etc.)
    y : ndarray
        Ordinate (spectral intensity)
    params : dict
        Dictionary of parameters from the parameter file
    """
    filepath = Path(filename)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = filepath.suffix.upper()

    # Format dispatch by extension
    if ext in ['.DTA', '.DSC']:
        return _load_bruker_bes3t(filepath, scaling)
    elif ext in ['.SPC', '.PAR']:
        return _load_bruker_esp(filepath, scaling)
    elif ext == '.D01':
        return _load_specman(filepath)
    elif ext == '.SPE':
        return _load_magnettech_binary(filepath)
    elif ext == '.XML':
        return _load_magnettech_xml(filepath)
    elif ext == '.ESR':
        return _load_active_spectrum(filepath)
    elif ext == '.EPR':
        return _load_ciqtek(filepath)
    elif ext == '.DAT':
        return _load_adani_dat(filepath)
    elif ext == '.JSON':
        return _load_adani_json(filepath)
    elif ext == '.ECO':
        return _load_qese_eth(filepath)
    elif ext == '.PLT':
        return _load_magres(filepath)
    elif ext in ['.SPK', '.REF']:
        return _load_varian_e9_eth(filepath)
    elif ext == '.D00':
        return _load_d00_wis_eth(filepath)
    else:
        # Try JEOL auto-detection from header
        try:
            with open(filepath, 'rb') as f:
                header = f.read(16)
            process_type = header.split(b'\x00')[0].decode('ascii', errors='ignore')
            jeol_types = ['spin', 'cAcqu', 'endor', 'pAcqu', 'cidep',
                          'sod', 'iso', 'ani']
            if any(process_type.startswith(t) for t in jeol_types):
                return _load_jeol(filepath)
        except Exception:
            pass
        raise ValueError(
            f"Unsupported file extension '{filepath.suffix}' (normalized to '{ext}').\n"
            f"Supported: .DTA/.DSC, .SPC/.PAR, .D01, .SPE, .XML, .ESR, "
            f".EPR, .DAT, .JSON, .ECO, .PLT, .SPK/.REF, .D00, JEOL formats."
        )


# ---------------------------------------------------------------------------
# Bruker BES3T (.DTA/.DSC)
# ---------------------------------------------------------------------------

def _find_companion(base: Path, candidates: list) -> Path:
    """Locate a Bruker companion file whose extension case may differ.

    Bruker writes DTA/DSC and SPC/PAR in a fixed case by convention, but
    vendor exports -- and checkouts on case-sensitive filesystems -- mix
    them.  Probe every candidate spelling and fall back to the first so the
    caller's missing-file error names the documented convention.
    """
    for ext in candidates:
        p = base.with_suffix(ext)
        if p.exists():
            return p
    return base.with_suffix(candidates[0])


def _load_bruker_bes3t(
    filepath: Path,
    scaling: str = ''
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load Bruker BES3T format (.DTA/.DSC files)."""
    if filepath.suffix.upper() == '.DTA':
        dta_file = filepath
        dsc_file = _find_companion(filepath, ['.DSC', '.dsc'])
    else:
        dsc_file = filepath
        dta_file = _find_companion(filepath, ['.DTA', '.dta'])

    if not dta_file.exists():
        raise FileNotFoundError(f"Data file not found: {dta_file}")
    if not dsc_file.exists():
        raise FileNotFoundError(f"Parameter file not found: {dsc_file}")

    params = _parse_bruker_dsc(dsc_file)

    with open(dta_file, 'rb') as f:
        data_bytes = f.read()

    byte_order = params.get('BSEQ', 'BIG')
    ikkf = params.get('IKKF', 'REAL')
    irfmt = params.get('IRFMT', None)
    xpts = params.get('XPTS', None)

    endian = '>' if byte_order == 'BIG' else '<'

    format_map = {
        'D': ('d', 8), 'F': ('f', 4), 'I': ('i', 4),
        'S': ('h', 2), 'C': ('b', 1),
    }

    irfmt_first = str(irfmt).strip("'\"").split(',')[0].strip().upper() if irfmt else None
    if irfmt_first and irfmt_first in format_map:
        struct_fmt, bytes_per_val = format_map[irfmt_first]
    elif xpts:
        bytes_per_point = len(data_bytes) / xpts
        if bytes_per_point == 8:
            struct_fmt, bytes_per_val = 'd', 8
        elif bytes_per_point == 4:
            struct_fmt, bytes_per_val = 'f', 4
        else:
            struct_fmt, bytes_per_val = 'f', 4
    else:
        struct_fmt, bytes_per_val = 'f', 4

    num_points = len(data_bytes) // bytes_per_val
    fmt = f'{endian}{num_points}{struct_fmt}'

    try:
        data = struct.unpack(fmt, data_bytes)
        y = np.array(data)
    except struct.error:
        num_points = len(data_bytes) // 8
        fmt = f'{endian}{num_points}d'
        data = struct.unpack(fmt, data_bytes)
        y = np.array(data)

    # Several data values per point (IKKF 'CPLX,CPLX', 'REAL,REAL', ...):
    # EasySpin getmatrix returns one dataset per value (cell array)
    ikkf_list = [k.strip().upper() for k in str(ikkf).strip("'\"").split(',') if k.strip()]
    is_cplx = [k == 'CPLX' for k in ikkf_list]
    n_reals = sum(2 if c else 1 for c in is_cplx)
    if len(ikkf_list) > 1:
        n_pts = y.size // n_reals
        dl = y[:n_pts * n_reals].reshape((n_reals, n_pts), order='F')
        datasets, row = [], 0
        for c in is_cplx:
            if c:
                datasets.append(dl[row] + 1j * dl[row + 1]); row += 2
            else:
                datasets.append(dl[row].copy()); row += 1
        y = datasets
    elif ikkf == 'CPLX':
        y = y[::2] + 1j * y[1::2]

    # Dimensions and axes as EasySpin eprload_BrukerBES3T: XPTS/YPTS/ZPTS, linear
    # (IDX: MIN + linspace(0, WID)) or nonlinear axes from companion .XGF/.YGF/.ZGF
    # files (IGD); values stay in the file's units (Bruker: Gauss, ns, ...).
    def _num(v, default):
        try:
            return float(str(v).strip("'\""))
        except (TypeError, ValueError):
            return default
    dims = [int(_num(params.get('XPTS', len(y)), len(y))),
            int(_num(params.get('YPTS', 1), 1) or 1), int(_num(params.get('ZPTS', 1), 1) or 1)]
    n_tot = dims[0] * dims[1] * dims[2]
    y = [d[:n_tot] for d in y] if isinstance(y, list) else y[:n_tot]
    fmt_np = {'D': 'f8', 'F': 'f4', 'I': 'i4', 'S': 'i2'}
    axes = []
    for a, name in enumerate('XYZ'):
        if dims[a] <= 1:
            continue
        atype = str(params.get(name + 'TYP', 'IDX')).strip("'\"").upper()
        ax = None
        if atype == 'IGD':
            comp = _find_companion(dta_file, ['.' + name + 'GF', '.' + name.lower() + 'gf'])
            if comp.exists():
                cfmt = str(params.get(name + 'FMT', 'D')).strip("'\"").upper()
                ax = np.fromfile(comp, dtype=np.dtype(endian + fmt_np.get(cfmt, 'f8'))).astype(float)[:dims[a]]
                if ax.size != dims[a]:
                    raise ValueError(f'Could not read {dims[a]} axis values from companion file {comp}.')
            else:
                warnings.warn(f'Could not read companion file {comp} for nonlinear axis. Assuming linear axis.')
        if ax is None:
            mn = _num(params.get(name + 'MIN', 0.0), 0.0)
            wd = _num(params.get(name + 'WID', 0.0), 0.0)
            if wd == 0:
                mn, wd = 1.0, dims[a] - 1
            ax = mn + np.linspace(0.0, wd, dims[a])
        axes.append(ax)
    shape = [d for d in dims if d > 1] or [dims[0]]
    y = [d.reshape(shape, order='F') for d in y] if isinstance(y, list) else y.reshape(shape, order='F')
    x = axes[0] if len(axes) == 1 else (axes if axes else np.arange(1, dims[0] + 1, dtype=float))
    if scaling:
        y = _apply_scaling(y, params, scaling)
    return x, y, params


# ---------------------------------------------------------------------------
# Bruker ESP/WinEPR (.spc/.par)
# ---------------------------------------------------------------------------

def _load_bruker_esp(
    filepath: Path,
    scaling: str = ''
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load Bruker ESP/WinEPR format (.spc/.par files)."""
    if filepath.suffix.upper() == '.SPC':
        spc_file = filepath
        par_file = _find_companion(filepath, ['.par', '.PAR'])
    else:
        par_file = filepath
        spc_file = _find_companion(filepath, ['.spc', '.SPC'])

    if not spc_file.exists():
        raise FileNotFoundError(f"Data file not found: {spc_file}")
    if not par_file.exists():
        raise FileNotFoundError(f"Parameter file not found: {par_file}")

    params = _parse_bruker_par(par_file)

    # Determine endianness and number format
    # DOS key → little-endian (WinEPR), otherwise big-endian (ESP)
    if 'DOS' in params:
        endian = '<'
        num_fmt = np.float32  # WinEPR: single float
    else:
        endian = '>'
        num_fmt = np.int32  # ESP: 32-bit integer

    with open(spc_file, 'rb') as f:
        raw_bytes = f.read()

    dt = np.dtype(num_fmt).newbyteorder(endian)
    y = np.frombuffer(raw_bytes, dtype=dt).astype(np.float64)

    # Dimensions, complex flag and abscissa as EasySpin eprload_BrukerESP
    def _pnum(key):
        v = params.get(key, None)
        if v is None:
            return None
        try:
            return float(str(v).split()[0])
        except (TypeError, ValueError, IndexError):
            return None
    flags = int(_pnum('JSS') or 0)
    is_complex = bool(flags & (1 << 4))
    two_d = bool(flags & (1 << 12))
    nx, ny = len(y), 1
    if two_d and _pnum('SSX') is not None:
        nx = int(_pnum('SSX')) // (2 if is_complex else 1)
    if two_d and _pnum('SSY') is not None:
        ny = int(_pnum('SSY'))
    if _pnum('ANZ') is not None:
        n_anz = int(_pnum('ANZ'))
        if not two_d:
            nx = n_anz // (2 if is_complex else 1)
        elif nx * ny != n_anz:
            raise ValueError('Two-dimensional data: SSX, SSY and ANZ in .par file are inconsistent.')
    if _pnum('RES') is not None:
        nx = int(_pnum('RES'))
    if _pnum('REY') is not None:
        ny = int(_pnum('REY'))
    if _pnum('XPLS') is not None:
        nx = int(_pnum('XPLS'))
    if not two_d:
        ny = 1   # RES/REY of a WinEPR 2D slice describe the original 2D size
    if is_complex:
        y = y[::2] + 1j * y[1::2]
    y = y[:nx * ny]
    if ny > 1:
        y = y.reshape((nx, ny), order='F')
    jex = str(params.get('JEX', 'field-sweep')).strip()
    jey = str(params.get('JEY', '')).strip()
    HCF, HSW, GST, GSI = _pnum('HCF'), _pnum('HSW'), _pnum('GST'), _pnum('GSI')
    XXLB, XXWI, XYLB, XYWI = _pnum('XXLB'), _pnum('XXWI'), _pnum('XYLB'), _pnum('XYWI')
    take = 0   # 1: GST/GSI, 2: HCF/HSW, 3: XXLB/XXWI(/XYLB/XYWI)
    if jex == 'ENDOR':
        take = 1
    elif None not in (XXLB, XXWI, XYLB, XYWI):
        take = 3
    elif None not in (HCF, HSW, GST, GSI):
        take = 1
    elif None not in (HCF, HSW):
        take = 2
    elif None not in (GST, GSI):
        take = 1
    elif GSI is None and HSW is None:
        HSW, take = 50.0, 2
    elif HCF is None:
        take = 3
    if jex == 'Time-Sweep':
        conv = _pnum('RCT') or 1.0
        x = np.arange(nx) * conv / 1e3
    elif take == 1:
        x = GST + GSI * np.linspace(0.0, 1.0, nx)
    elif take == 2:
        x = HCF + HSW / 2 * np.linspace(-1.0, 1.0, nx)
    elif take == 3 and XXLB is not None and XXWI is not None:
        x = XXLB + np.linspace(0.0, XXWI, nx)
        if XYLB is not None and XYWI is not None and ny > 1:
            x = [x, XYLB + np.linspace(0.0, XYWI, ny)]
    else:
        raise ValueError('Could not determine abscissa range from parameter file!')
    params['_JEY_PowerSweep'] = (jey == 'mw-power-sweep')
    if scaling:
        y = _apply_scaling(y, params, scaling)
    return x, y, params


# ---------------------------------------------------------------------------
# Active Spectrum (.ESR)
# ---------------------------------------------------------------------------

def _load_active_spectrum(
    filepath: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load Active Spectrum ESR format (text, 'FIELD (G)' header)."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find start of data after "FIELD (G)" header line
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('FIELD (G)'):
            start_idx = i + 1
            break
    if start_idx is None:
        raise ValueError(f"Could not find 'FIELD (G)' header in {filepath}")

    data_lines = lines[start_idx:]
    data = []
    for line in data_lines:
        line = line.strip()
        if not line:
            continue
        vals = line.split()
        if len(vals) >= 2:
            data.append([float(vals[0]), float(vals[1])])

    data = np.array(data)
    x = data[:, 0]  # field in Gauss
    y = data[:, 1]
    return x, y, {}


# ---------------------------------------------------------------------------
# Adani text (.dat)
# ---------------------------------------------------------------------------

def _load_adani_dat(
    filepath: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load Adani spectrometer text format (.dat)."""
    with open(filepath, 'r', encoding='latin-1') as f:
        lines = f.readlines()

    # First line should be parameter separator
    sep = '=' * 20
    if not lines[0].strip().startswith(sep[:20]):
        raise ValueError(f"Not an Adani DAT file: {filepath}")

    # Find end of parameter section (second separator line)
    data_start = None
    for i in range(1, len(lines)):
        if lines[i].strip().startswith(sep[:20]):
            data_start = i + 1
            break
    if data_start is None:
        raise ValueError(f"Could not find data section in {filepath}")

    data = []
    for line in lines[data_start:]:
        line = line.strip().replace(',', '.')
        if not line:
            continue
        vals = line.split()
        if len(vals) >= 3:
            data.append([float(vals[0]), float(vals[1]), float(vals[2])])

    data = np.array(data)
    x = data[:, 1]  # field (second column)
    y = data[:, 2]  # signal (third column)
    return x, y, {}


# ---------------------------------------------------------------------------
# Adani JSON (.json)
# ---------------------------------------------------------------------------

def _load_adani_json(
    filepath: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load Adani SPINSCAN JSON format (e-Spinoza software)."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    nSpectra = len(data['Values'])
    nPoints = len(data['Values'][0]['Values'])
    nPhases = len(data['Values'][0]['Values'][0]['Points'])

    phase0 = data.get('Phase', 0)
    s = np.sin(2 * np.pi * np.arange(nPhases) / nPhases + phase0)

    spc = np.zeros((nPoints, nSpectra))
    for iSpectrum in range(nSpectra):
        acqdata = data['Values'][iSpectrum]['Values']
        for iPoint in range(nPoints):
            points = np.array(acqdata[iPoint]['Points'])
            spc[iPoint, iSpectrum] = np.sum(points * s)

    d = data['ExperimentOptions']
    center_field = d['CommonOptions']['CenterMagneticField']
    sweep_width = d['CommonOptions']['SweepWidth']
    B = np.linspace(-1, 1, nPoints) * sweep_width / 2 + center_field

    y = spc if nSpectra > 1 else spc[:, 0]

    if nSpectra > 1:
        axis2 = np.linspace(d['InitialValue2D'], d['FinalValue2D'], nSpectra)
        params = dict(data)
        params['_axis2'] = axis2
        return [B, axis2], y, params

    return B, y, dict(data)


# ---------------------------------------------------------------------------
# CIQTEK (.epr)
# ---------------------------------------------------------------------------

def _load_ciqtek(
    filepath: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load CIQTEK spectrometer JSON format (.epr)."""
    with open(filepath, 'r') as f:
        filecontents = json.load(f)

    lineDataList = filecontents['dataStore']['lineDataList']
    nTraces = len(lineDataList)

    x = np.array([pt[0] for pt in lineDataList[0]['ReData']])
    dataRe = np.zeros((len(x), nTraces))
    dataIm = np.zeros((len(x), nTraces))

    for iTrace in range(nTraces):
        re_data = lineDataList[iTrace]['ReData']
        im_data = lineDataList[iTrace]['ImData']
        dataRe[:, iTrace] = np.array([pt[1] for pt in re_data])
        dataIm[:, iTrace] = np.array([pt[1] for pt in im_data])

    y = dataRe + 1j * dataIm

    if nTraces > 1:
        # Try to find 2D axis from params
        line_params = lineDataList[0].get('params', {})
        y_axis = np.arange(1, nTraces + 1, dtype=float)
        for field_name in ['delay', 'power', 'time2']:
            if field_name in line_params:
                for iTrace in range(nTraces):
                    val_str = lineDataList[iTrace]['params'][field_name]
                    y_axis[iTrace] = float(val_str)
                break
        params = dict(filecontents)
        params['_axis2'] = y_axis
    else:
        y = y[:, 0]
        params = dict(filecontents)

    if '_axis2' in params:
        return [x, params['_axis2']], y, params   # EasySpin: one abscissa per dimension
    return x, y, params


# ---------------------------------------------------------------------------
# MAGRES (.plt)
# ---------------------------------------------------------------------------

def _load_magres(
    filepath: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load MAGRES simulation program output (.plt)."""
    # Find DATA tag to get number of points
    nx = 0
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts and parts[0] == 'DATA':
                nx = int(parts[-1])
                break

    if nx == 0:
        raise ValueError(f"Could not determine number of data points in {filepath}")

    # Read data (skip first 3 header lines)
    with open(filepath, 'r') as f:
        for _ in range(3):
            f.readline()
        data = []
        for line in f:
            for val in line.split():
                data.append(float(val))
                if len(data) >= nx:
                    break
            if len(data) >= nx:
                break

    y = np.array(data[:nx])
    x = np.arange(nx, dtype=float)
    return x, y, {}


# ---------------------------------------------------------------------------
# Magnettech binary (.spe)
# ---------------------------------------------------------------------------

def _load_magnettech_binary(
    filepath: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load Magnettech binary format (MS400 and older, .spe)."""
    file_size = filepath.stat().st_size

    # File overhead = 64 bytes at end; each data point = 2 bytes (int16)
    valid_points = [512, 1024, 2048, 4096]
    nPoints = None
    for n in valid_points:
        if file_size == 2 * n + 64:
            nPoints = n
            break

    if nPoints is None:
        raise ValueError(
            f"File size {file_size} does not match any Magnettech SPE format "
            f"(expected 2*N+64 for N in {valid_points})"
        )

    with open(filepath, 'rb') as f:
        # Read spectral data (int16, little-endian)
        y = np.frombuffer(f.read(nPoints * 2), dtype='<i2').astype(float)

        # Read parameter block at end of file
        f.seek(file_size - 4)
        file_flags = struct.unpack('<B', f.read(1))[0]
        mw_freq_available = bool(file_flags & 1)
        old_format = not bool(file_flags & 2)
        temp_available = bool(file_flags & 4)

        def read_four_byte_single():
            vals = struct.unpack('<2h', f.read(4))
            if old_format:
                return vals[0] + vals[1] / 100.0
            return vals[0] + vals[1] / 1000.0

        # Seek to start of parameter block
        f.seek(2 * nPoints)
        params = {}
        params['B0_Field'] = read_four_byte_single() / 10.0  # G → mT
        params['B0_Scan'] = read_four_byte_single() / 10.0   # G → mT
        params['Modulation'] = read_four_byte_single() / 10000.0  # mT
        params['MW_Attenuation'] = read_four_byte_single()  # dB
        params['ScanTime'] = read_four_byte_single()  # s
        gain_mantissa = read_four_byte_single()
        gain_exponent = read_four_byte_single()
        params['Gain'] = gain_mantissa * 10 ** round(gain_exponent)
        params['Number'] = read_four_byte_single()
        _ = read_four_byte_single()  # reserved
        params['Time_const'] = read_four_byte_single()  # s
        _ = read_four_byte_single()  # reserved
        _ = read_four_byte_single()  # reserved
        params['NumberSamples'] = read_four_byte_single()

        if temp_available:
            params['Temperature'] = struct.unpack('<i', f.read(4))[0]  # °C
        else:
            f.read(4)
            params['Temperature'] = None

        _ = read_four_byte_single()  # reserved
        params['FileFlags'] = struct.unpack('<B', f.read(1))[0]

        if mw_freq_available:
            mwf = struct.unpack('<3B', f.read(3))
            params['mwFreq'] = (mwf[2] + 256 * mwf[1] + 256**2 * mwf[0]) / 1e6  # kHz → GHz
        else:
            params['mwFreq'] = None

    x = params['B0_Field'] + np.linspace(-0.5, 0.5, nPoints) * params['B0_Scan']
    return x, y, params


# ---------------------------------------------------------------------------
# Magnettech XML (.xml)
# ---------------------------------------------------------------------------

def _load_magnettech_xml(
    filepath: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load Magnettech MS5000 XML format."""
    tree = ET.parse(filepath)
    root = tree.getroot()

    # Navigate to ESRXmlFile/Data/Measurement/DataCurves/Curve
    esr_node = root
    if esr_node.tag != 'ESRXmlFile':
        raise ValueError(f"Not a Magnettech XML file: {filepath}")

    data_node = esr_node.find('Data')
    if data_node is None:
        raise ValueError("ESRXmlFile.Data node not found")

    measurement = data_node.find('Measurement')
    if measurement is None:
        raise ValueError("Measurement node not found")

    data_curves = measurement.find('DataCurves')
    if data_curves is None:
        raise ValueError("DataCurves node not found")

    # Extract parameters from Measurement attributes
    params = dict(measurement.attrib)

    # Add Recipe parameters if present
    recipe = measurement.find('Recipe')
    x_axis = 'field'
    if recipe is not None:
        params_node = recipe.find('Parameters')
        if params_node is not None:
            for param in params_node.findall('Param'):
                name = param.get('Name', '')
                text = param.text or ''
                params[name] = text

        recipe_type = recipe.get('Type', 'single')
        type_map = {'single': 'field', 'kinetic': 'time', 'temperature': 'field',
                    'goniometric': 'field', 'modulationSweep': 'field',
                    'powerSweep': 'field', 'xRay': 'field'}
        x_axis = type_map.get(recipe_type, 'field')

        if params.get('BHoldEnabled') == 'True':
            x_axis = 'time'

    # Parse curves — decode Base64 data
    curves = {}
    for curve_elem in data_curves.findall('Curve'):
        name = curve_elem.get('YType', '')
        mode = curve_elem.get('Mode', '')
        if name == 'BField' and mode == 'Raw':
            name = 'BField_Raw'

        text = curve_elem.text
        if text and curve_elem.get('Compression') == 'Base64':
            # Magnettech custom Base64: 9 base64 chars per 8-byte double
            # Replace padding '=' with 'A' (zero bits)
            text = text.replace('=', 'A')
            bytestream = base64.b64decode(text)
            # Remove every 9th byte (padding byte per double)
            arr = bytearray()
            for i in range(len(bytestream)):
                if (i + 1) % 9 != 0:
                    arr.append(bytestream[i])
            data_vals = np.frombuffer(bytes(arr), dtype='<f8')

            x_offset = float(curve_elem.get('XOffset', '0'))
            x_slope = float(curve_elem.get('XSlope', '1'))
            curves[name] = {
                'data': data_vals,
                'x': x_offset + np.arange(len(data_vals)) * x_slope,
            }

    params['Curves'] = curves

    # Build output
    if 'BField' in curves:
        if x_axis == 'field':
            # Interpolate BField onto MW_Absorption time axis
            if 'MW_Absorption' in curves:
                bf_x = curves['BField']['x']
                bf_data = curves['BField']['data']
                mw_x = curves['MW_Absorption']['x']
                mw_data = curves['MW_Absorption']['data']
                x = np.interp(mw_x, bf_x, bf_data)
                y = mw_data
                # Trim to requested field range
                bfrom = float(params.get('Bfrom', x.min()))
                bto = float(params.get('Bto', x.max()))
                mask = (x >= bfrom) & (x <= bto)
                x = x[mask]
                y = y[mask]
            else:
                x = curves['BField']['data']
                y = np.zeros_like(x)
        else:  # time axis
            if 'MW_Absorption' in curves:
                x = curves['MW_Absorption']['x']
                y = curves['MW_Absorption']['data']
            else:
                x = np.array([])
                y = np.array([])
    elif 'Frequency' in curves:
        x = curves['Frequency']['data']
        y = curves.get('ADC_24bit', {}).get('data', np.zeros_like(x))
    else:
        x = np.array([])
        y = np.array([])

    return x, y, params


# ---------------------------------------------------------------------------
# Varian E9 ETH (.spk, .ref)
# ---------------------------------------------------------------------------

def _load_varian_e9_eth(
    filepath: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load Varian E9 ETH-specific format (binary float32)."""
    with open(filepath, 'rb') as f:
        raw = np.fromfile(f, dtype='<f4')

    N = len(raw)
    # Determine actual data size from known file-size cutoffs
    cutoffs = [500, 1000, 2000, 5000, 10000]
    ndata = None
    for K in cutoffs:
        if N > K:
            ndata = K

    if ndata is None:
        raise ValueError(f"Varian E9 file too small: {filepath}")

    y = raw[N - ndata:]
    x = np.arange(len(y), dtype=float)
    return x, y, {}


# ---------------------------------------------------------------------------
# d00 Weizmann/ETH (.d00)
# ---------------------------------------------------------------------------

def _load_d00_wis_eth(
    filepath: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load ESE Weizmann/ETH d00 format (binary complex double)."""
    with open(filepath, 'rb') as f:
        dims = np.fromfile(f, dtype='<i2', count=3)
        # Read interleaved real/imag double data
        raw = np.fromfile(f, dtype='<f8')

    # raw is [real0, imag0, real1, imag1, ...]
    n_complex = len(raw) // 2
    y = raw[0::2][:n_complex] + 1j * raw[1::2][:n_complex]

    # Reshape to dims
    shape = tuple(int(d) for d in dims if d > 0)
    if shape and np.prod(shape) == len(y):
        y = y.reshape(shape)

    x = np.arange(y.shape[0] if y.ndim >= 1 else len(y), dtype=float)
    return x, y, {}


# ---------------------------------------------------------------------------
# qese/tryscore ETH (.eco)
# ---------------------------------------------------------------------------

def _load_qese_eth(
    filepath: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load qese/tryscore ETH format (.eco, text)."""
    with open(filepath, 'r') as f:
        first_line = f.readline().strip().split()
        header = [int(v) for v in first_line]

        if len(header) >= 3:
            dims = header[:2]
            is_complex = bool(header[2])
        elif len(header) == 2:
            dims = header
            is_complex = False
        else:
            dims = [header[0], 1]
            is_complex = False

        n_vals = dims[0] * dims[1] * (2 if is_complex else 1)
        raw = []
        for line in f:
            for val in line.split():
                raw.append(float(val))
                if len(raw) >= n_vals:
                    break
            if len(raw) >= n_vals:
                break

    raw = np.array(raw)

    if is_complex:
        y = raw[0::2] + 1j * raw[1::2]
    else:
        y = raw

    if dims[1] > 1:
        y = y.reshape(dims)

    x = np.arange(dims[0], dtype=float)
    return x, y, {}


# ---------------------------------------------------------------------------
# JEOL (.0, .2d, etc. — auto-detected from binary header)
# ---------------------------------------------------------------------------

def _load_jeol(
    filepath: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load JEOL JES-FA / JES-X3 binary format."""
    with open(filepath, 'rb') as f:
        # [A] HEADER block
        header = {}
        header['processType'] = _read_cstring(f, 16)
        header['filename'] = _read_cstring(f, 64)
        f.read(4)  # reserved
        header['auxNum'] = struct.unpack('<i', f.read(4))[0]
        header['dataSet'] = struct.unpack('<i', f.read(4))[0]
        header['dataLength'] = struct.unpack('<i', f.read(4))[0]
        header['dataKind'] = _read_cstring(f, 15)
        header['dispPart'] = f.read(1).decode('ascii', errors='ignore')
        header['xOffset'] = struct.unpack('<f', f.read(4))[0]
        header['xRange'] = struct.unpack('<f', f.read(4))[0]
        header['xUnit'] = _read_cstring(f, 16)
        header['yMin'] = struct.unpack('<f', f.read(4))[0]
        header['yMax'] = struct.unpack('<f', f.read(4))[0]
        header['yUnit'] = _read_cstring(f, 16)
        header['zPoint'] = struct.unpack('<f', f.read(4))[0]
        header['zUnit'] = _read_cstring(f, 16)
        header['xStart'] = struct.unpack('<f', f.read(4))[0]
        header['xStop'] = struct.unpack('<f', f.read(4))[0]
        header['yStart'] = struct.unpack('<f', f.read(4))[0]
        header['yStop'] = struct.unpack('<f', f.read(4))[0]
        header['zStart'] = struct.unpack('<f', f.read(4))[0]
        header['zStop'] = struct.unpack('<f', f.read(4))[0]
        f.read(88)  # reserved
        header['waveIndex'] = struct.unpack('<i', f.read(4))[0]
        f.read(28)  # reserved
        is_endor = header['processType'] == 'endor'
        f.read(4096)  # data icon bitmap

        # [B] GENERAL block
        general = {}
        f.read(16)  # reserved
        general['system'] = _read_cstring(f, 32)
        general['version'] = _read_cstring(f, 16)
        general['date'] = _read_cstring(f, 32)
        general['title'] = _read_cstring(f, 128)
        general['sample'] = _read_cstring(f, 128)
        general['comment'] = _read_cstring(f, 128)
        f.read(224)  # reserved
        general['previousINFO_filename'] = _read_cstring(f, 64)
        general['previousINFO_dir'] = _read_cstring(f, 512)
        f.read(4)
        general['previousINFO_processType'] = _read_cstring(f, 16)
        general['originalINFO_filename'] = _read_cstring(f, 64)
        general['originalINFO_dir'] = _read_cstring(f, 512)
        f.read(4)
        general['originalINFO_processType'] = _read_cstring(f, 16)

        # [C] SPECTROMETER block
        spec = {}
        spec['sType'] = _read_cstring(f, 12)
        spec['SpectrometerFlag'] = struct.unpack('<i', f.read(4))[0]
        spec['magnet'] = _read_cstring(f, 16)
        spec['water'] = _read_cstring(f, 16)
        spec['magPower'] = _read_cstring(f, 16)
        spec['magCurrent'] = _read_cstring(f, 16)
        spec['centerField'] = _read_cstring(f, 16)
        spec['sweepWidFin'] = _read_cstring(f, 16)
        spec['sweepWidCor'] = _read_cstring(f, 16)
        spec['sweepPulse'] = _read_cstring(f, 16)
        spec['pulseNumber'] = _read_cstring(f, 16)
        spec['sweepFTime'] = _read_cstring(f, 16)
        spec['sweepBTime'] = _read_cstring(f, 16)
        spec['scanTime'] = _read_cstring(f, 16)
        spec['swCont'] = _read_cstring(f, 16)
        spec['sPosition'] = _read_cstring(f, 16)
        spec['ePosition'] = _read_cstring(f, 16)
        spec['pPosition'] = _read_cstring(f, 16)
        spec['cPosition'] = _read_cstring(f, 16)
        spec['lfsBlock'] = _read_cstring(f, 16)
        spec['xPosition'] = _read_cstring(f, 16)
        f.read(112)  # reserved
        spec['modFreq'] = _read_cstring(f, 16)
        spec['modWidFin'] = _read_cstring(f, 16)
        spec['modWinCor'] = _read_cstring(f, 16)
        spec['phase'] = _read_cstring(f, 16)
        # Amp1/Amp2 blocks
        for amp_prefix in ['Amp1_', 'Amp2_']:
            spec[amp_prefix + 'rcvMode'] = _read_cstring(f, 16)
            spec[amp_prefix + 'phasePlus'] = _read_cstring(f, 16)
            spec[amp_prefix + 'amplitudeFin'] = _read_cstring(f, 16)
            spec[amp_prefix + 'amplitudeCor'] = _read_cstring(f, 16)
            spec[amp_prefix + 'timeConstant'] = _read_cstring(f, 16)
            spec[amp_prefix + 'zero'] = _read_cstring(f, 16)
        spec['amplock'] = _read_cstring(f, 16)
        spec['sourceCH2'] = _read_cstring(f, 8)
        f.read(120)  # reserved
        spec['uType'] = _read_cstring(f, 8)
        spec['uFreq'] = _read_cstring(f, 16)
        spec['uFreqUnit'] = _read_cstring(f, 8)
        spec['uPower'] = _read_cstring(f, 16)
        spec['uPowerUnit'] = _read_cstring(f, 8)
        spec['uPhase'] = _read_cstring(f, 16)
        spec['uCoupling'] = _read_cstring(f, 16)
        spec['slavePower'] = _read_cstring(f, 16)
        spec['slavePhase'] = _read_cstring(f, 16)
        spec['uRef'] = _read_cstring(f, 16)
        spec['autoTuneMode'] = _read_cstring(f, 16)
        spec['autoTune'] = _read_cstring(f, 16)
        spec['afc'] = _read_cstring(f, 8)
        spec['afcclk'] = _read_cstring(f, 16)
        spec['mod'] = _read_cstring(f, 8)
        spec['gunPower'] = _read_cstring(f, 8)
        spec['ref'] = _read_cstring(f, 8)
        spec['att_30dB'] = _read_cstring(f, 8)
        spec['afcbalance'] = _read_cstring(f, 8)
        spec['uDetM'] = _read_cstring(f, 8)
        spec['uBalM'] = _read_cstring(f, 8)
        spec['afcphase'] = _read_cstring(f, 8)
        spec['uStab'] = _read_cstring(f, 8)
        spec['shflock'] = _read_cstring(f, 16)
        spec['uFreqCal'] = _read_cstring(f, 16)
        spec['uFreqCalUnits'] = _read_cstring(f, 8)
        spec['RotationStepAngle'] = _read_cstring(f, 10)
        spec['RotationAngle'] = _read_cstring(f, 10)
        spec['RotationZero'] = _read_cstring(f, 10)
        spec['motorLow'] = _read_cstring(f, 40)
        spec['motorUp'] = _read_cstring(f, 40)
        f.read(18)  # reserved
        spec['acqPoint'] = _read_cstring(f, 8)
        spec['dataLength'] = _read_cstring(f, 8)
        spec['acqPulsMode'] = _read_cstring(f, 8)
        spec['enType'] = _read_cstring(f, 16)
        # ENDOR channels
        for ch_prefix in ['ch1_', 'ch2_']:
            spec[ch_prefix + 'eLeftFreq'] = _read_cstring(f, 16)
            spec[ch_prefix + 'eRightFreq'] = _read_cstring(f, 16)
            spec[ch_prefix + 'eCurrentFreq'] = _read_cstring(f, 16)
            spec[ch_prefix + 'eFunit'] = _read_cstring(f, 8)
            spec[ch_prefix + 'ePower'] = _read_cstring(f, 16)
            spec[ch_prefix + 'eDB'] = _read_cstring(f, 12)
            spec[ch_prefix + 'ePunit'] = _read_cstring(f, 8)
            spec[ch_prefix + 'eSweepTime'] = _read_cstring(f, 12)
            spec[ch_prefix + 'eModWidth'] = _read_cstring(f, 8)
        f.read(8)  # reserved
        spec['vtType'] = _read_cstring(f, 16)
        spec['temperature'] = _read_cstring(f, 16)
        spec['tempStart'] = _read_cstring(f, 16)
        spec['tempEnd'] = _read_cstring(f, 16)
        spec['tempStep'] = _read_cstring(f, 16)
        spec['tempUnit'] = _read_cstring(f, 8)
        spec['tempStatus'] = _read_cstring(f, 8)
        spec['tempError'] = _read_cstring(f, 16)
        spec['readyTime'] = _read_cstring(f, 16)
        spec['okTempRange'] = _read_cstring(f, 16)
        spec['tempControl'] = _read_cstring(f, 8)
        spec['tempLock'] = _read_cstring(f, 16)
        spec['endorLock'] = _read_cstring(f, 16)
        spec['selfCheck'] = _read_cstring(f, 8)
        spec['eCenterFreq'] = _read_cstring(f, 16)
        spec['eSweepFreq'] = _read_cstring(f, 16)
        f.read(48)
        spec['MarkerPos1'] = _read_cstring(f, 16)
        spec['MarkerPos2'] = _read_cstring(f, 16)
        spec['controlFA'] = _read_cstring(f, 16)
        f.read(112)

        # [D] GENERATOR block
        generator = {}
        if not is_endor:
            generator['date'] = _read_cstring(f, 32)
            generator['accumulation'] = struct.unpack('<i', f.read(4))[0]
            generator['preAccumulation'] = struct.unpack('<i', f.read(4))[0]
            generator['delayTime'] = struct.unpack('<i', f.read(4))[0]
            generator['intervalTime'] = struct.unpack('<i', f.read(4))[0]
            generator['index'] = struct.unpack('<i', f.read(4))[0]
            generator['repetition'] = struct.unpack('<i', f.read(4))[0]
            f.read(16)  # 4 reserved int32
            generator['sampleTime'] = struct.unpack('<i', f.read(4))[0]
            f.read(16)  # 4 reserved int32
            generator['accumWave'] = _read_cstring(f, 8)
            generator['baselineMode'] = _read_cstring(f, 8)
            generator['sampleMode'] = _read_cstring(f, 8)
            generator['sigTrig'] = _read_cstring(f, 8)
            generator['refFile'] = _read_cstring(f, 64)
            generator['paraFile'] = _read_cstring(f, 64)
            generator['chMode'] = _read_cstring(f, 16)
            f.read(112)
        else:
            generator['date'] = _read_cstring(f, 32)
            generator['accumulation'] = struct.unpack('<i', f.read(4))[0]
            generator['accumWave'] = _read_cstring(f, 8)
            generator['baselineMode'] = _read_cstring(f, 8)
            generator['chMode'] = _read_cstring(f, 16)
            generator['MagnetPosition'] = _read_cstring(f, 16)
            for ch_prefix in ['ch1_', 'ch2_']:
                generator[ch_prefix + 'eLeftFreq'] = _read_cstring(f, 16)
                generator[ch_prefix + 'eRightFreq'] = _read_cstring(f, 16)
                generator[ch_prefix + 'eCurrentFreq'] = _read_cstring(f, 16)
                generator[ch_prefix + 'eFunit'] = _read_cstring(f, 8)
                generator[ch_prefix + 'ePower'] = _read_cstring(f, 16)
                generator[ch_prefix + 'eDB'] = _read_cstring(f, 12)
                generator[ch_prefix + 'ePunit'] = _read_cstring(f, 8)
                generator[ch_prefix + 'eSweepTime'] = _read_cstring(f, 12)
                generator[ch_prefix + 'eModWidth'] = _read_cstring(f, 16)
                f.read(248)
            generator['eCenterFreq'] = _read_cstring(f, 16)
            generator['eSweepFreq'] = _read_cstring(f, 16)
            f.read(352)

        # [E] PROCESS PARA block — skip (many float32/int32 fields)
        # We read it but don't parse every sub-field for simplicity
        process_para = {}
        process_para['res_q'] = struct.unpack('<f', f.read(4))[0]
        process_para['micro_hz'] = struct.unpack('<f', f.read(4))[0]
        # Skip the rest of the process para block (variable size)
        # Read remaining fields as a bulk skip to reach DATA block
        f.read(4 * 5)  # efect_pls, nike_hz, pulser, digitzer, resonator
        f.read(4)  # fftCount int32
        _read_cstring(f, 16)  # lastProcessTyp
        f.read(4)  # std_spinFlag
        f.read(4 * 9)  # std_marleft ... std_spinnum (9 float32)
        f.read(4 * 4)  # reserved
        f.read(4)  # spinFlag
        f.read(4 * 9)  # marleft ... spinnum (9 float32)
        f.read(4 * 4)  # reserved
        # xAreaM (9 fields) + xAreaS (9 fields) + 8*xArea (9 each) + markers
        f.read(4 * 9 * 2)  # xAreaM + xAreaS
        f.read(4 * 9 * 8)  # 8 xArea blocks
        # xLinem1, xLinem2 (4 each) + 8 xLine (4 each)
        f.read(4 * 4 * 2)
        f.read(4 * 4 * 8)
        f.read(4)  # aope
        _read_cstring(f, 16)  # modFilename
        f.read(4)  # yLine
        f.read(4)  # calibFlag
        f.read(128)

        # [F] CALCULATOR block — skip based on processType
        pt = header['processType']
        calc_skip = {
            'yZero': 160, 'xZero': 4, 'yGain': 4, 'xShift': 4, 'reverse': 4,
            'fill': 12, 'window': 80, 'exp': 0, 'log': 0, 'power': 0,
            'fft': 0, 'ifft': 0, 'diff': 32, 'inte': 32,
            'smooth': 16, 'spin': 0, 'fit': 90120,
            'fRtime': 4264, 'tRtime': 4264,
            'add': 0, 'sub': 0, 'mul': 0, 'div': 0,
            'phase': 808, 'mT2MHz': 4, 'hw': 40, 'mem': 2708,
        }
        skip = calc_skip.get(pt, 0)
        if skip > 0:
            f.read(skip)

        # [G] DATA block
        auxNum = header['auxNum']
        dataSet = header['dataSet']
        dataKind = header['dataKind']
        dataLength = header['dataLength']

        ch1 = None
        ch2 = None

        if dataSet == 1:
            ch1 = np.fromfile(f, dtype='<f4', count=dataLength)
            if dataKind == 'complex':
                ch2 = np.fromfile(f, dtype='<f4', count=dataLength)
        elif auxNum == 1 and dataKind == 'complex':
            ch1 = np.zeros((dataSet, dataLength), dtype=np.float32)
            ch2 = np.zeros((dataSet, dataLength), dtype=np.float32)
            for k in range(dataSet):
                f.read(4)  # SEQ_VALUE float32
                f.read(8)  # SEQ_ITEM
                ch1[k] = np.fromfile(f, dtype='<f4', count=dataLength)
                ch2[k] = np.fromfile(f, dtype='<f4', count=dataLength)
        elif auxNum == 12 and dataKind == 'complex':
            ch1 = np.zeros((dataSet, dataLength), dtype=np.float32)
            ch2 = np.zeros((dataSet, dataLength), dtype=np.float32)
            for k in range(dataSet):
                f.read(48)  # 12 SEQ_VALUE float32
                f.read(96)  # SEQ_ITEM
                ch1[k] = np.fromfile(f, dtype='<f4', count=dataLength)
                ch2[k] = np.fromfile(f, dtype='<f4', count=dataLength)

    # Build output
    if dataKind == 'real' and ch1 is not None:
        y = ch1.astype(np.float64)
    elif dataKind == 'complex' and ch1 is not None and ch2 is not None:
        y = ch1.astype(np.float64) + 1j * ch2.astype(np.float64)
    else:
        y = np.array([])

    x = header['xOffset'] + np.linspace(0, header['xRange'], dataLength)

    params = {
        'Header': header,
        'General': general,
        'Spectrometer': spec,
        'Generator': generator,
    }
    return x, y, params


def _read_cstring(f, n: int) -> str:
    """Read n bytes from file, return as null-terminated string."""
    raw = f.read(n)
    # Find null terminator
    null_idx = raw.find(b'\x00')
    if null_idx >= 0:
        raw = raw[:null_idx]
    return raw.decode('ascii', errors='ignore').strip()


# ---------------------------------------------------------------------------
# SpecMan (.d01/.exp)
# ---------------------------------------------------------------------------

def _load_specman(
    filepath: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load SpecMan .d01 binary data with .exp parameter file."""
    d01_file = filepath.with_suffix('.d01') if filepath.suffix.upper() != '.D01' else filepath
    exp_file = filepath.with_suffix('.exp')

    if not d01_file.exists():
        raise FileNotFoundError(f"Data file not found: {d01_file}")

    with open(d01_file, 'rb') as f:
        ndim1 = struct.unpack('<I', f.read(4))[0]
        dformat = struct.unpack('<I', f.read(4))[0]
        dtype_str = '<f4' if dformat == 1 else '<f8'

        streams = []
        ntotal = 0
        for _ in range(ndim1):
            ndim2 = struct.unpack('<i', f.read(4))[0]
            dim = list(struct.unpack('<4i', f.read(16)))
            dim[ndim2:] = [1] * (4 - ndim2)
            total = struct.unpack('<i', f.read(4))[0]
            streams.append({'dim': dim, 'first': ntotal, 'total': total})
            ntotal += total

        tmpdat = np.fromfile(f, dtype=dtype_str, count=ntotal)

    if ndim1 == 0:
        raise ValueError("No data in SpecMan file")

    if ndim1 == 2:
        s1 = streams[0]
        s2 = streams[1]
        re = tmpdat[s1['first']:s1['first'] + s1['total']]
        im = tmpdat[s2['first']:s2['first'] + s2['total']]
        y = re + 1j * im
        y = y.reshape(s1['dim'], order='F')
    elif ndim1 == 1:
        s1 = streams[0]
        y = tmpdat[s1['first']:s1['first'] + s1['total']]
        y = y.reshape(streams[0]['dim'], order='F')
    else:
        # Multiple streams — check if all same dimension
        dim = streams[0]['dim']
        all_same = all(s['dim'] == dim for s in streams)
        is_1d = sum(1 for d in dim if d != 1) == 1

        if all_same and is_1d:
            xdim = [d for d in dim if d != 1][0]
            y = np.zeros((xdim, ndim1))
            for k in range(ndim1):
                s = streams[k]
                y[:, k] = tmpdat[s['first']:s['first'] + s['total']]
        else:
            y = tmpdat

    # Remove trailing singleton dimensions
    y = np.squeeze(y)

    # Parse .exp parameter file
    params = _parse_specman_exp(exp_file)
    x_out = _build_specman_axes(params, y)

    if isinstance(x_out, list):
        x_out = x_out[:max(1, int(np.ndim(y)))]   # one abscissa per data dimension (EasySpin)
    if isinstance(x_out, list) and len(x_out) == 1:
        x_out = x_out[0]

    if isinstance(x_out, np.ndarray):
        x = x_out
    elif isinstance(x_out, list) and len(x_out) > 0:
        x = x_out if len(x_out) > 1 else x_out[0]   # EasySpin: one abscissa per dimension
        params['_axes'] = x_out
    else:
        x = np.arange(y.shape[0] if y.ndim >= 1 else len(y), dtype=float)

    return x, y, params


def _parse_specman_exp(filepath: Path) -> Dict[str, Any]:
    """Parse SpecMan .exp parameter file (INI-like sections)."""
    params = {}
    if not filepath.exists():
        return params

    section = ''
    text_idx = 1
    prg_idx = 1

    with open(filepath, 'r', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            # Check for section header [name]
            if s.startswith('[') and ']' in s:
                section = s[1:s.index(']')].replace('-', '')
                continue

            if section == 'text':
                params[f'text{text_idx}'] = s
                text_idx += 1
            elif section == 'program':
                params[f'prg{prg_idx}'] = s
                prg_idx += 1
            else:
                if '=' in s:
                    key, val = s.split('=', 1)
                    key = key.strip().replace('/', '_').replace('\\', '_').replace(' ', '_')
                    params[f'{section}_{key}'] = val
                else:
                    params[f'{section}_{s}'] = ''

    return params


def _build_specman_axes(params: Dict[str, Any], data: np.ndarray) -> list:
    """Build axis arrays from SpecMan parameters."""
    # Unit prefix conversion
    prefix_map = {
        'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'm': 1e-3,
        'k': 1e3, 'M': 1e6, 'G': 1e9, 'T': 1e12,
    }

    axes = []
    # Parse sweep definitions
    sweep_keys = ['sweep_transient']
    idx = 0
    while True:
        key = f'sweep_sweep{idx}'
        if key not in params:
            break
        sweep_keys.append(key)
        idx += 1

    triggers = 1
    trig_str = params.get('streams_triggers', '1')
    try:
        triggers = int(trig_str)
    except (ValueError, TypeError):
        pass

    axis_label = 'xyz'
    counter = 0

    for i, key in enumerate(sweep_keys):
        if key not in params:
            continue
        parts = params[key].split(',')
        if len(parts) < 3:
            continue

        sweep_type = parts[0].strip()
        try:
            asize = int(parts[1].strip())
        except ValueError:
            asize = 1

        if sweep_type in ('S', 'I', 'A', 'R'):
            asize = 1

        if i == 0:
            asize *= triggers

        if asize <= 1:
            continue

        # Try to find the parameter definition
        var_name = parts[3].strip() if len(parts) > 3 else ''
        param_key = f'params_{var_name.replace(" ", "_")}'

        if sweep_type == 'T':
            param_key = 'params_trans'
            dwell_str = params.get('streams_dwelltime', '1 ns')
            dwell_str = dwell_str.split(',')[0].strip()
            params['params_trans'] = f'0 ns step {dwell_str};'

        arr = None
        if param_key in params:
            pstr = params[param_key]
            if ' step ' in pstr:
                tk1, tk2 = pstr.split(' step ', 1)
                tk2 = tk2.split(';')[0].strip()
                minval, unit = _specman_parse_value(tk1.strip(), prefix_map)
                step, _ = _specman_parse_value(tk2, prefix_map)
                arr = np.arange(asize) * step + minval
            elif ' logto ' in pstr:
                tk1, tk2 = pstr.split(' logto ', 1)
                tk2 = tk2.split(';')[0].strip()
                minval, unit = _specman_parse_value(tk1.strip(), prefix_map)
                maxval, _ = _specman_parse_value(tk2, prefix_map)
                arr = np.logspace(np.log10(minval), np.log10(maxval), asize)
            elif ' to ' in pstr:
                tk1, tk2 = pstr.split(' to ', 1)
                tk2 = tk2.split(';')[0].strip()
                minval, unit = _specman_parse_value(tk1.strip(), prefix_map)
                maxval, _ = _specman_parse_value(tk2, prefix_map)
                arr = np.linspace(minval, maxval, asize)
            else:
                # Comma-separated list
                vals = []
                for item in pstr.split(';')[0].split(','):
                    item = item.strip()
                    if item:
                        v, unit = _specman_parse_value(item, prefix_map)
                        vals.append(v)
                if vals:
                    arr = np.array(vals)

        else:
            unit = 's'
        if arr is None:
            arr = np.arange(asize, dtype=float)
        else:
            # EasySpin eprload_specman: values in units other than G, K and plain s
            # are rescaled to the largest SI prefix below their maximum
            if unit not in ('G', 'K', 's'):
                umax = float(np.max(np.abs(arr)))
                for k in (1e12, 1e9, 1e6, 1e3, 1e-3, 1e-6, 1e-9, 1e-12):
                    if umax > k:
                        arr = arr / k
                        break

        if counter < len(axis_label):
            axes.append(arr)
            counter += 1

    # Fallback: if no axes parsed, use integer index
    if not axes:
        n = data.shape[0] if data.ndim >= 1 else len(data)
        axes.append(np.arange(n, dtype=float))

    return axes


def _specman_parse_value(s: str, prefix_map: dict) -> Tuple[float, str]:
    """Parse a value+unit string like '100 ns' → (1e-7, 's')."""
    m = re.match(r'([+\-]?[0-9]*\.?[0-9]+)\s*(.*)', s.strip())
    if not m:
        return 0.0, ''
    val = float(m.group(1))
    unit = m.group(2).strip()

    if len(unit) > 1 and unit[0] in prefix_map:
        val *= prefix_map[unit[0]]
        unit = unit[1:]

    return val, unit


# ---------------------------------------------------------------------------
# Helper functions for Bruker formats
# ---------------------------------------------------------------------------

def _parse_bruker_dsc(filepath: Path) -> Dict[str, Any]:
    """Parse Bruker .DSC parameter file."""
    params = {}
    with open(filepath, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            match = re.match(r'(\w+)\s+(.+)', line)
            if match:
                key, value = match.groups()
                params[key] = _convert_value(value)
    return params


def _parse_bruker_par(filepath: Path) -> Dict[str, Any]:
    """Parse Bruker .par parameter file."""
    params = {}
    with open(filepath, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                params[parts[0]] = _convert_value(parts[1])
            elif len(parts) == 1:
                params[parts[0]] = ''  # flag-only keys like DOS
    return params


def _convert_value(value_str: str) -> Union[float, int, str]:
    """Convert string value to appropriate type."""
    value_str = value_str.strip().strip("'\"")
    try:
        return int(value_str)
    except ValueError:
        pass
    try:
        return float(value_str)
    except ValueError:
        pass
    return value_str


def _apply_scaling(y: np.ndarray, params: Dict[str, Any], scaling: str) -> np.ndarray:
    """Apply scaling to Bruker spectrum data."""
    y_scaled = y.copy()
    if 'n' in scaling:
        n_scans = params.get('JSD', params.get('NbScansDone', 1))
        if n_scans > 0:
            y_scaled = y_scaled / n_scans
    if 'P' in scaling:
        mw_power = params.get('MWPW', params.get('MP', 1.0))
        if mw_power > 0:
            y_scaled = y_scaled / np.sqrt(mw_power)
    if 'G' in scaling:
        gain = params.get('RCAG', params.get('RRG', 1.0))
        if gain > 0:
            y_scaled = y_scaled / gain
    if 'T' in scaling:
        temp = params.get('STMP', params.get('TE', 298.0))
        y_scaled = y_scaled * temp
    return y_scaled


# ---------------------------------------------------------------------------
# eprload_info — informational display
# ---------------------------------------------------------------------------

def eprload_info(filename: Union[str, Path]) -> None:
    """Display information about an EPR data file."""
    x, y, params = eprload(filename)

    print(f"File: {filename}")
    print(f"Data points: {y.size}")
    if x.size > 0:
        print(f"X-axis: {x.ravel()[0]:.3f} to {x.ravel()[-1]:.3f} ({len(x)} points)")
    yr = np.real(y) if np.iscomplexobj(y) else y
    print(f"Y-axis: min={yr.min():.3e}, max={yr.max():.3e}")
    print(f"\nKey parameters:")

    important_keys = [
        'MWFQ', 'MF', 'MWPW', 'MP', 'RCAG', 'RRG',
        'STMP', 'TE', 'JSD', 'NbScansDone', 'SPTP', 'NSP', 'A1RS', 'RMA',
    ]
    for key in important_keys:
        if key in params:
            print(f"  {key}: {params[key]}")
