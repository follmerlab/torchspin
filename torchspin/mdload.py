"""
mdload — Load and process molecular dynamics trajectories for cardamom.

Reads trajectory (DCD/TRR) and topology (PSF/GRO) files, extracts
spin-label dihedral angles, builds label frame vectors, and removes
global protein rotational diffusion.

Supports R1 (MTSSL, 5 dihedrals) and TOAC (2 dihedrals) spin labels.

Based on EasySpin's mdload.m and associated MD file readers.
"""

import numpy as np
import struct
import os
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union


@dataclass
class MDData:
    """Molecular dynamics trajectory data for cardamom.

    Attributes
    ----------
    nSteps : int
        Total number of time steps.
    dt : float
        Time step in seconds.
    FrameTraj : ndarray, shape (3, 3, nSteps)
        Label frame vectors (x, y, z axes) at each time step.
    FrameTrajwrtProt : ndarray, shape (3, 3, nSteps)
        Frame vectors with protein global rotation removed.
    RProtDiff : ndarray, shape (3, 3, nSteps)
        Protein rotational diffusion matrices.
    dihedrals : ndarray, shape (nDihedrals, nSteps)
        Side chain dihedral angles in radians.
    ProtCAxyz : ndarray, shape (3, nCA, nSteps), optional
        Protein C-alpha coordinates.
    """
    nSteps: int = 0
    dt: float = 0.0
    FrameTraj: np.ndarray = None
    FrameTrajwrtProt: np.ndarray = None
    RProtDiff: np.ndarray = None
    dihedrals: np.ndarray = None
    ProtCAxyz: np.ndarray = None


@dataclass
class MDInfo:
    """Information for loading MD trajectories.

    Attributes
    ----------
    SegName : str
        Segment name in topology file.
    ResName : str
        Residue name of spin label (default: 'CYR1').
    LabelName : str
        Label type: 'R1' (MTSSL) or 'TOAC'.
    AtomNames : dict, optional
        Custom atom names for frame construction.
    """
    SegName: str = ''
    ResName: str = 'CYR1'
    LabelName: str = 'R1'
    AtomNames: Optional[dict] = None


@dataclass
class MDOptions:
    """Options for mdload."""
    Verbosity: int = 0
    keepProtCA: bool = False


# ---------------------------------------------------------------------------
# DCD trajectory reader (md_readdcd.m)
# ---------------------------------------------------------------------------

def _read_dcd(filename, atom_indices=None):
    """Read a DCD trajectory file (NAMD/CHARMM format).

    Parameters
    ----------
    filename : str
        Path to .dcd file.
    atom_indices : list of int, optional
        0-based atom indices to extract (for efficiency).

    Returns
    -------
    xyz : ndarray, shape (nFrames, 3, nAtoms)
        Atomic coordinates in Angstroms.
    nAtoms : int
    nFrames : int
    dt : float
        Time step in seconds.
    """
    with open(filename, 'rb') as f:
        # Read first record length
        rec_len = struct.unpack('<i', f.read(4))[0]

        # Read header
        header = f.read(rec_len)

        nFrames_header = struct.unpack('<i', header[4:8])[0]
        istart = struct.unpack('<i', header[8:12])[0]
        nsavc = struct.unpack('<i', header[12:16])[0]

        # Check for CHARMM extra block (unit cell info)
        charmm_flag = struct.unpack('<i', header[80:84])[0]
        has_extra_block = (charmm_flag & 0x1) != 0

        delta = struct.unpack('<f', header[40:44])[0]
        dt_internal = delta * max(nsavc, 1)
        dt = dt_internal * 48.88821e-15  # AKMA to seconds

        f.read(4)  # End record

        # Title block
        rec_len2 = struct.unpack('<i', f.read(4))[0]
        f.read(rec_len2)
        f.read(4)

        # Number of atoms
        f.read(4)  # Record start
        nAtoms = struct.unpack('<i', f.read(4))[0]
        f.read(4)  # Record end

        # Read frames dynamically (NAMD often writes nFrames=0 in header)
        if atom_indices is not None:
            n_out = len(atom_indices)
        else:
            n_out = nAtoms

        frames_list = []

        while True:
            try:
                # Skip extra block (unit cell) if present
                if has_extra_block:
                    eb_len = struct.unpack('<i', f.read(4))[0]
                    f.read(eb_len)
                    f.read(4)

                frame = np.zeros((3, n_out), dtype=np.float64)
                for dim in range(3):
                    rec_start = struct.unpack('<i', f.read(4))[0]
                    raw = f.read(nAtoms * 4)
                    if len(raw) < nAtoms * 4:
                        raise EOFError
                    coords = np.frombuffer(raw, dtype='<f4')
                    f.read(4)  # Record end

                    if atom_indices is not None:
                        frame[dim, :] = coords[atom_indices]
                    else:
                        frame[dim, :] = coords

                frames_list.append(frame)
            except (struct.error, EOFError, ValueError):
                break

        nFrames = len(frames_list)
        if nFrames > 0:
            xyz = np.stack(frames_list, axis=0)  # (nFrames, 3, nAtoms)
        else:
            xyz = np.zeros((0, 3, n_out))

    return xyz, nAtoms, nFrames, dt


# ---------------------------------------------------------------------------
# PSF topology reader (md_readpsf.m)
# ---------------------------------------------------------------------------

def _read_psf(filename, seg_name, res_name, label_name):
    """Read a PSF topology file (CHARMM/X-PLOR format).

    Parameters
    ----------
    filename : str
    seg_name : str
        Segment name to filter by.
    res_name : str
        Residue name of spin label.
    label_name : str
        'R1' or 'TOAC'.

    Returns
    -------
    data : dict
        Keys: 'nAtoms', 'idx_SpinLabel', 'idx_ProteinCA',
        'atom_names', 'res_names', 'seg_names', plus individual atom indices.
    """
    atom_names = []
    res_names = []
    seg_names = []
    res_ids = []

    with open(filename, 'r') as f:
        in_atoms = False
        n_atoms = 0

        for line in f:
            line = line.strip()
            if '!NATOM' in line:
                n_atoms = int(line.split()[0])
                in_atoms = True
                continue

            if in_atoms:
                if not line or line.startswith('!'):
                    in_atoms = False
                    continue

                parts = line.split()
                if len(parts) >= 5:
                    seg_names.append(parts[1])
                    res_ids.append(int(parts[2]))
                    res_names.append(parts[3])
                    atom_names.append(parts[4])

                if len(atom_names) >= n_atoms:
                    in_atoms = False

    # Find spin label atoms
    seg_mask = np.array([s == seg_name for s in seg_names])
    res_mask = np.array([r == res_name for r in res_names])
    label_mask = seg_mask & res_mask
    idx_spin_label = np.where(label_mask)[0]

    # Find protein CA atoms
    ca_mask = np.array([a == 'CA' for a in atom_names])
    prot_ca_mask = seg_mask & ca_mask & ~res_mask
    idx_prot_ca = np.where(prot_ca_mask)[0]

    # Find specific atoms for frame construction
    data = {
        'nAtoms': len(atom_names),
        'idx_SpinLabel': idx_spin_label,
        'idx_ProteinCA': idx_prot_ca,
        'atom_names': atom_names,
        'res_names': res_names,
        'seg_names': seg_names,
    }

    # Atom name lookup within spin label residue
    label_atom_names = [atom_names[i] for i in idx_spin_label]

    if label_name == 'R1':
        # R1 (MTSSL) frame atoms — CHARMM naming convention
        _find_atom = lambda name: idx_spin_label[label_atom_names.index(name)] if name in label_atom_names else None
        data['idx_ON'] = _find_atom('ON')
        data['idx_NN'] = _find_atom('NN')
        data['idx_C1'] = _find_atom('C1')
        data['idx_C2'] = _find_atom('C2')
        data['idx_C1R'] = _find_atom('C1R')
        data['idx_C2R'] = _find_atom('C2R')
        data['idx_C1L'] = _find_atom('C1L')
        data['idx_S1L'] = _find_atom('S1L')
        data['idx_SG'] = _find_atom('SG')
        data['idx_CB'] = _find_atom('CB')
        data['idx_CA'] = _find_atom('CA')
        data['idx_N'] = _find_atom('N')
    elif label_name == 'TOAC':
        _find_atom = lambda name: idx_spin_label[label_atom_names.index(name)] if name in label_atom_names else None
        # TOAC uses OE/ND (not ON/NN) as default atom names in CHARMM.
        # Use `is not None` explicitly — `x or y` would fall through when x == 0
        # (atom at index 0), silently selecting the wrong atom.
        _on = _find_atom('OE')
        data['idx_ON'] = _on if _on is not None else _find_atom('ON')
        _nn = _find_atom('ND')
        data['idx_NN'] = _nn if _nn is not None else _find_atom('NN')
        data['idx_CGS'] = _find_atom('CG2')
        data['idx_CGR'] = _find_atom('CG1')
        data['idx_CBS'] = _find_atom('CB2')
        data['idx_CBR'] = _find_atom('CB1')
        data['idx_CA'] = _find_atom('CA')
        data['idx_N'] = _find_atom('N')

    return data


# ---------------------------------------------------------------------------
# GRO topology reader (md_readgro.m)
# ---------------------------------------------------------------------------

def _read_gro(filename, res_name, label_name):
    """Read a GRO topology file (Gromos87 format).

    Returns same structure as _read_psf.
    """
    atom_names = []
    res_names = []
    res_ids = []

    with open(filename, 'r') as f:
        title = f.readline()
        n_atoms = int(f.readline().strip())

        for i in range(n_atoms):
            line = f.readline()
            rid = int(line[0:5].strip())
            rname = line[5:10].strip()
            aname = line[10:15].strip()
            res_ids.append(rid)
            res_names.append(rname)
            atom_names.append(aname)

    res_mask = np.array([r == res_name for r in res_names])
    idx_spin_label = np.where(res_mask)[0]

    ca_mask = np.array([a == 'CA' for a in atom_names])
    prot_ca_mask = ca_mask & ~res_mask
    idx_prot_ca = np.where(prot_ca_mask)[0]

    data = {
        'nAtoms': len(atom_names),
        'idx_SpinLabel': idx_spin_label,
        'idx_ProteinCA': idx_prot_ca,
        'atom_names': atom_names,
        'res_names': res_names,
    }

    label_atom_names = [atom_names[i] for i in idx_spin_label]

    if label_name == 'R1':
        _find_atom = lambda name: idx_spin_label[label_atom_names.index(name)] if name in label_atom_names else None
        data['idx_ON'] = _find_atom('ON')
        data['idx_NN'] = _find_atom('NN')
        data['idx_C1'] = _find_atom('C1')
        data['idx_C2'] = _find_atom('C2')
        data['idx_C1R'] = _find_atom('C1R')
        data['idx_C2R'] = _find_atom('C2R')
        data['idx_C1L'] = _find_atom('C1L')
        data['idx_S1L'] = _find_atom('S1L')
        data['idx_SG'] = _find_atom('SG')
        data['idx_CB'] = _find_atom('CB')
        data['idx_CA'] = _find_atom('CA')
        data['idx_N'] = _find_atom('N')
    elif label_name == 'TOAC':
        _find_atom = lambda name: idx_spin_label[label_atom_names.index(name)] if name in label_atom_names else None
        # Use `is not None` check — `x or y` evaluates `0 or y` as `y` (falsy-zero bug)
        _on = _find_atom('OE')
        data['idx_ON'] = _on if _on is not None else _find_atom('ON')
        _nn = _find_atom('ND')
        data['idx_NN'] = _nn if _nn is not None else _find_atom('NN')
        data['idx_CGS'] = _find_atom('CG2')
        data['idx_CGR'] = _find_atom('CG1')
        data['idx_CBS'] = _find_atom('CB2')
        data['idx_CBR'] = _find_atom('CB1')
        data['idx_CA'] = _find_atom('CA')
        data['idx_N'] = _find_atom('N')

    return data


# ---------------------------------------------------------------------------
# TRR trajectory reader (md_readtrr.m)
# ---------------------------------------------------------------------------

def _read_trr(filename, atom_indices=None):
    """Read a TRR trajectory file (GROMACS format).

    Parameters
    ----------
    filename : str
    atom_indices : list of int, optional

    Returns
    -------
    xyz : ndarray, shape (nFrames, 3, nAtoms)
    nAtoms : int
    nFrames : int
    dt : float (seconds)
    """
    frames = []
    dt = 0.0

    with open(filename, 'rb') as f:
        prev_time = None

        while True:
            try:
                # TRR frame header
                magic = struct.unpack('>i', f.read(4))[0]
                if magic != 1993:
                    break

                slen = struct.unpack('>i', f.read(4))[0]
                # Pad to 4-byte boundary
                padded = ((slen + 3) // 4) * 4
                f.read(padded)  # version string

                ir_size = struct.unpack('>i', f.read(4))[0]
                e_size = struct.unpack('>i', f.read(4))[0]
                box_size = struct.unpack('>i', f.read(4))[0]
                vir_size = struct.unpack('>i', f.read(4))[0]
                pres_size = struct.unpack('>i', f.read(4))[0]
                top_size = struct.unpack('>i', f.read(4))[0]
                sym_size = struct.unpack('>i', f.read(4))[0]
                x_size = struct.unpack('>i', f.read(4))[0]
                v_size = struct.unpack('>i', f.read(4))[0]
                f_size = struct.unpack('>i', f.read(4))[0]
                natoms = struct.unpack('>i', f.read(4))[0]
                step = struct.unpack('>i', f.read(4))[0]
                nre = struct.unpack('>i', f.read(4))[0]

                # Determine precision (float or double)
                if x_size > 0:
                    is_double = (x_size // natoms // 3) == 8
                else:
                    is_double = False

                if is_double:
                    fmt = '>d'
                    fsize = 8
                    t = struct.unpack('>d', f.read(8))[0]
                    lam = struct.unpack('>d', f.read(8))[0]
                else:
                    t = struct.unpack('>f', f.read(4))[0]
                    lam = struct.unpack('>f', f.read(4))[0]
                    fmt = '>f'
                    fsize = 4

                if prev_time is not None and dt == 0:
                    dt = (t - prev_time) * 1e-12  # ps to seconds
                prev_time = t

                # Skip box, vir, pres
                f.read(box_size + vir_size + pres_size)

                # Read coordinates
                if x_size > 0:
                    n_vals = natoms * 3
                    dtype = '>f8' if is_double else '>f4'
                    coords = np.frombuffer(f.read(n_vals * fsize), dtype=dtype)
                    coords = coords.reshape(natoms, 3).T  # (3, natoms)

                    if atom_indices is not None:
                        coords = coords[:, atom_indices]

                    # Convert nm to Angstroms
                    frames.append(coords * 10.0)

                # Skip velocities and forces
                f.read(v_size + f_size)

            except (struct.error, ValueError):
                break

    if not frames:
        raise ValueError(f"No frames read from {filename}")

    xyz = np.stack(frames, axis=0)  # (nFrames, 3, nAtoms)
    return xyz, natoms if atom_indices is None else len(atom_indices), len(frames), dt


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _dihedral_angle(p1, p2, p3, p4):
    """Compute dihedral angle from 4 atom positions.

    Parameters
    ----------
    p1, p2, p3, p4 : ndarray, shape (3,) or (3, nSteps)

    Returns
    -------
    angle : float or ndarray
        Dihedral angle in radians.
    """
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    n1 = np.cross(b1, b2, axis=0)
    n2 = np.cross(b2, b3, axis=0)

    n1_norm = np.linalg.norm(n1, axis=0, keepdims=True)
    n2_norm = np.linalg.norm(n2, axis=0, keepdims=True)

    n1 = n1 / np.maximum(n1_norm, 1e-30)
    n2 = n2 / np.maximum(n2_norm, 1e-30)

    b2_unit = b2 / np.maximum(np.linalg.norm(b2, axis=0, keepdims=True), 1e-30)
    m1 = np.cross(n1, b2_unit, axis=0)

    x = np.sum(n1 * n2, axis=0)
    y = np.sum(m1 * n2, axis=0)

    return np.arctan2(y, x)


def _calc_label_frame(xyz, topo_data, label_name):
    """Calculate spin-label frame from atomic coordinates.

    Parameters
    ----------
    xyz : ndarray, shape (nFrames, 3, nAtoms)
    topo_data : dict
    label_name : str

    Returns
    -------
    frames : ndarray, shape (3, 3, nFrames)
        Lab frame for each time step. frames[:, i, :] is axis i.
    """
    nFrames = xyz.shape[0]
    frames = np.zeros((3, 3, nFrames))

    _normalize = lambda v: v / max(np.linalg.norm(v), 1e-30)

    if label_name == 'R1':
        i_ON = topo_data.get('idx_ON')
        i_NN = topo_data.get('idx_NN')
        i_C1 = topo_data.get('idx_C1')
        i_C2 = topo_data.get('idx_C2')

        if any(x is None for x in [i_ON, i_NN, i_C1, i_C2]):
            raise ValueError("Missing R1 frame atoms (ON, NN, C1, C2)")

        for t in range(nFrames):
            # Match MATLAB mdload.m exactly:
            # NNNO = normalize(ON - NN)  (N-O bond vector)
            NNNO = _normalize(xyz[t, :, i_ON] - xyz[t, :, i_NN])
            NNC1 = _normalize(xyz[t, :, i_C1] - xyz[t, :, i_NN])
            NNC2 = _normalize(xyz[t, :, i_C2] - xyz[t, :, i_NN])

            # z-axis: normalize(cross(NNC1, NNNO) + cross(NNNO, NNC2))
            z = _normalize(np.cross(NNC1, NNNO) + np.cross(NNNO, NNC2))

            # x-axis: N-O bond direction
            x = NNNO

            # y-axis: cross(z, x)
            y = np.cross(z, x)

            frames[:, 0, t] = x
            frames[:, 1, t] = y
            frames[:, 2, t] = z

    elif label_name == 'TOAC':
        i_ON = topo_data.get('idx_ON')
        i_NN = topo_data.get('idx_NN')
        i_CGR = topo_data.get('idx_CGR')
        i_CGS = topo_data.get('idx_CGS')

        if any(x is None for x in [i_ON, i_NN]):
            raise ValueError("Missing TOAC frame atoms (ON, NN)")

        for t in range(nFrames):
            NNNO = _normalize(xyz[t, :, i_ON] - xyz[t, :, i_NN])

            if i_CGR is not None and i_CGS is not None:
                NNCGR = _normalize(xyz[t, :, i_CGR] - xyz[t, :, i_NN])
                NNCGS = _normalize(xyz[t, :, i_CGS] - xyz[t, :, i_NN])
                z = _normalize(np.cross(NNCGR, NNNO) + np.cross(NNNO, NNCGS))
            else:
                z = np.array([0, 0, 1.0])

            x = NNNO
            y = np.cross(z, x)

            frames[:, 0, t] = x
            frames[:, 1, t] = y
            frames[:, 2, t] = z

    return frames


def _calc_dihedrals(xyz, topo_data, label_name):
    """Calculate side-chain dihedral angles.

    Parameters
    ----------
    xyz : ndarray, shape (nFrames, 3, nAtoms)
    topo_data : dict
    label_name : str

    Returns
    -------
    dihedrals : ndarray, shape (nDihedrals, nFrames)
    """
    nFrames = xyz.shape[0]

    if label_name == 'R1':
        # chi1-chi5 dihedral angles (CHARMM MTSSL atom names)
        atom_groups = [
            ('idx_N', 'idx_CA', 'idx_CB', 'idx_SG'),       # chi1
            ('idx_CA', 'idx_CB', 'idx_SG', 'idx_S1L'),     # chi2
            ('idx_CB', 'idx_SG', 'idx_S1L', 'idx_C1L'),    # chi3
            ('idx_SG', 'idx_S1L', 'idx_C1L', 'idx_C1R'),   # chi4
            ('idx_S1L', 'idx_C1L', 'idx_C1R', 'idx_C2R'),  # chi5
        ]
    elif label_name == 'TOAC':
        atom_groups = [
            ('idx_CA', 'idx_CBS', 'idx_CGS', 'idx_NN'),    # chi1
            ('idx_CA', 'idx_CBR', 'idx_CGR', 'idx_NN'),    # chi2
        ]
    else:
        return np.zeros((0, nFrames))

    # Filter out groups where any atom is missing
    valid_groups = []
    for group in atom_groups:
        if all(topo_data.get(a) is not None for a in group):
            valid_groups.append(group)

    n_dih = len(valid_groups)
    dihedrals = np.zeros((n_dih, nFrames))

    for i, group in enumerate(valid_groups):
        idx = [topo_data[a] for a in group]
        p1 = xyz[:, :, idx[0]].T  # (3, nFrames)
        p2 = xyz[:, :, idx[1]].T
        p3 = xyz[:, :, idx[2]].T
        p4 = xyz[:, :, idx[3]].T
        dihedrals[i] = _dihedral_angle(p1, p2, p3, p4)

    return dihedrals


def _remove_global_rotation(xyz, idx_prot_ca, nFrames):
    """Remove protein global rotational diffusion using Kabsch alignment.

    Parameters
    ----------
    xyz : ndarray, shape (nFrames, 3, nAtoms)
    idx_prot_ca : ndarray
        Indices of protein C-alpha atoms.

    Returns
    -------
    R_prot : ndarray, shape (3, 3, nFrames)
        Rotation matrices for protein global motion.
    """
    if len(idx_prot_ca) == 0:
        return np.tile(np.eye(3)[:, :, np.newaxis], (1, 1, nFrames))

    R_prot = np.zeros((3, 3, nFrames))
    R_prot[:, :, 0] = np.eye(3)

    ref = xyz[0][:, idx_prot_ca]  # (3, nCA) reference frame

    for t in range(1, nFrames):
        curr = xyz[t][:, idx_prot_ca]  # (3, nCA)

        # Center both sets
        ref_c = ref - ref.mean(axis=1, keepdims=True)
        curr_c = curr - curr.mean(axis=1, keepdims=True)

        # SVD for optimal rotation
        H = curr_c @ ref_c.T
        U, _, Vt = np.linalg.svd(H)

        # Handle reflection
        d = np.linalg.det(Vt.T @ U.T)
        S = np.eye(3)
        if d < 0:
            S[2, 2] = -1

        R_prot[:, :, t] = Vt.T @ S @ U.T

    return R_prot


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def mdload(
    traj_file: Union[str, List[str]],
    top_file: str,
    info: Optional[MDInfo] = None,
    opt: Optional[MDOptions] = None,
) -> MDData:
    """Load MD trajectory and extract spin-label information.

    Parameters
    ----------
    traj_file : str or list of str
        Trajectory file path(s). Supported: .dcd, .trr
    top_file : str
        Topology file path. Supported: .psf, .gro
    info : MDInfo, optional
        Residue/label identification parameters.
    opt : MDOptions, optional
        Loading options.

    Returns
    -------
    md : MDData
        Processed trajectory data.
    """
    if info is None:
        info = MDInfo()
    if opt is None:
        opt = MDOptions()

    # Determine file formats
    top_ext = os.path.splitext(top_file)[1].lower()

    if isinstance(traj_file, str):
        traj_files = [traj_file]
    else:
        traj_files = traj_file

    traj_ext = os.path.splitext(traj_files[0])[1].lower()

    # Read topology
    if top_ext == '.psf':
        topo_data = _read_psf(top_file, info.SegName, info.ResName, info.LabelName)
    elif top_ext == '.gro':
        topo_data = _read_gro(top_file, info.ResName, info.LabelName)
    else:
        raise ValueError(f"Unsupported topology format: {top_ext}")

    # Determine atoms to extract
    idx_spin = topo_data['idx_SpinLabel']
    idx_ca = topo_data['idx_ProteinCA']

    # Combine indices for extraction
    all_idx = np.concatenate([idx_spin, idx_ca])
    all_idx_sorted = np.sort(all_idx)

    # Remap indices after extraction
    idx_map = {old: new for new, old in enumerate(all_idx_sorted)}
    idx_spin_new = np.array([idx_map[i] for i in idx_spin])
    idx_ca_new = np.array([idx_map[i] for i in idx_ca])

    # Remap specific atom indices
    for key in list(topo_data.keys()):
        if key.startswith('idx_') and key not in ['idx_SpinLabel', 'idx_ProteinCA']:
            val = topo_data[key]
            if val is not None and val in idx_map:
                topo_data[key] = idx_map[val]

    # Read trajectories
    all_xyz = []
    total_dt = 0

    for tf in traj_files:
        if traj_ext == '.dcd':
            xyz, _, nf, dt = _read_dcd(tf, atom_indices=list(all_idx_sorted))
        elif traj_ext == '.trr':
            xyz, _, nf, dt = _read_trr(tf, atom_indices=list(all_idx_sorted))
        else:
            raise ValueError(f"Unsupported trajectory format: {traj_ext}")

        all_xyz.append(xyz)
        total_dt = dt

    xyz = np.concatenate(all_xyz, axis=0)
    nFrames = xyz.shape[0]

    # Build label frame
    frames = _calc_label_frame(xyz, topo_data, info.LabelName)

    # Calculate dihedrals
    dihedrals = _calc_dihedrals(xyz, topo_data, info.LabelName)

    # Remove global rotation
    R_prot = _remove_global_rotation(xyz, idx_ca_new, nFrames)

    # Apply rotation removal to frame
    frames_wrt_prot = np.zeros_like(frames)
    for t in range(nFrames):
        frames_wrt_prot[:, :, t] = R_prot[:, :, t] @ frames[:, :, t]

    # Build output
    md = MDData()
    md.nSteps = nFrames
    md.dt = total_dt
    md.FrameTraj = frames
    md.FrameTrajwrtProt = frames_wrt_prot
    md.RProtDiff = R_prot
    md.dihedrals = dihedrals

    if opt.keepProtCA:
        md.ProtCAxyz = xyz[:, :, idx_ca_new].transpose(1, 2, 0)  # (3, nCA, nSteps)

    return md
