"""Spin Hamiltonian symmetry detection.

Determines the point group symmetry of the spin Hamiltonian and the associated
symmetry frame for optimizing powder averaging grids.

Port of EasySpin's ``hamsymm.m``. Two analysis paths exist, selected exactly
as in EasySpin:

* **Geometric analysis** (``hamsymm_geom``): used when every interaction is
  given by principal values + Euler angles and no high-order terms are
  present. Each second-rank tensor is classified as O3 / Dinfh / D2h and the
  symmetries are combined pairwise (g, D, ee, sigma, A, Q, nn order).
* **Eigenvalue analysis** (``hamsymm_eigs``): used when Stevens operator
  terms (``Sys.B``), higher-order Zeeman terms (``Sys.Ham``), crystal-field
  terms (``Sys.CF*``) or any full 3×3 tensors are present. The Hamiltonian is
  diagonalized at twelve probe field directions in each candidate symmetry
  frame and the point group is inferred from eigenvalue coincidences.

torchspin additions (deviations from EasySpin, 0.9.24):

* The strain tensors (HStrain/HStrainL, gStrain, AStrain, DStrain) are symmetry
  inputs too, since the strain widths depend on the field direction (B82);
  strain-free systems get EasySpin's group and frame exactly.
* :func:`grid_symmetry` is the automatic grid symmetry used by the simulators: it is
  :func:`hamsymm` unless a symmetry input requires grad, and then the symmetry of the
  differentiated parameterization (B1, 0.9.26).
* :func:`grid_angles_to_molecular` rotates a symmetry-frame grid into the molecular
  frame (B83).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from torchspin.spinsystem import SpinSystem
from torchspin.rotations import erot

__all__ = ['hamsymm', 'grid_symmetry', 'grid_angles_to_molecular']

# Point groups in EasySpin's ranking order (index = symmetry rank, 1-based).
_EIG_GROUP_NAMES = [
    'Ci', 'C2h', 'D2h', 'C4h', 'D4h', 'S6', 'D3d', 'C6h', 'D6h', 'Th', 'Oh',
    'Dinfh', 'O3',
]


def hamsymm(sys: SpinSystem, debug: bool = False) -> tuple[str, torch.Tensor]:
    """Determine point group symmetry of spin Hamiltonian.

    Parameters
    ----------
    sys:
        Spin system.
    debug:
        Print the decision trace (mirrors EasySpin's ``'debug'`` option).

    Returns
    -------
    pgroup:
        Schönflies point group symbol. One of ``'O3'``, ``'Dinfh'``,
        ``'Oh'``, ``'Th'``, ``'D6h'``, ``'C6h'``, ``'D3d'``, ``'S6'``,
        ``'D4h'``, ``'C4h'``, ``'D2h'``, ``'C2h'``, ``'Ci'``. (The geometric
        path can only yield O3/Dinfh/D2h/C2h/Ci; the eigenvalue path the full
        set.)
    R:
        3×3 rotation matrix **from the molecular frame to the symmetry
        frame**: ``v_sym = R @ v_mol``. Its *rows* are the symmetry-frame axes
        expressed in molecular coordinates. This is the transpose of
        EasySpin's ``RMatrix`` (whose *columns* are the symmetry axes, i.e.
        ``v_mol = RMatrix @ v_sym``); pepper consumes it as
        ``v_mol = R.T @ v_sym``. Both analysis paths use this convention.

    Examples
    --------
    >>> sys = SpinSystem(S=1/2, g=2.0)
    >>> pgroup, R = hamsymm(sys)
    >>> pgroup
    'O3'

    >>> sys = SpinSystem(S=1/2, g=[2.0, 2.0, 2.2])
    >>> pgroup, R = hamsymm(sys)
    >>> pgroup
    'Dinfh'

    >>> sys = SpinSystem(S=1/2, g=[2.0, 2.1, 2.2])
    >>> pgroup, R = hamsymm(sys)
    >>> pgroup
    'D2h'
    """
    higher_zeeman = _higher_zeeman_present(sys)
    crystal_field = _crystal_field_present(sys)
    stevens = _stevens_present(sys)
    # SpinSystem.H0Extra: an arbitrary operator, so only the eigenvalue probe
    # can find the symmetry (never the O3 shortcut or the geometric path).
    h0extra = getattr(sys, 'H0Extra', None) is not None
    high_order = stevens or higher_zeeman or crystal_field or h0extra

    full_tensors = _full_tensors_given(sys)

    # Strain broadening (torchspin deviation from EasySpin, B82): the line
    # widths depend on the field direction through the strain tensors, so
    # the grid symmetry must also be a symmetry of every strain tensor.
    strains = _strain_tensors(sys)

    if debug:
        print('High-order terms present!' if high_order
              else 'No high-order terms present!')

    if not full_tensors and not high_order:
        if debug:
            print('Check whether isotropic...')
        if _is_isotropic(sys) and all(s[0] == 0 for s in strains):
            return 'O3', torch.eye(3, dtype=torch.float64)

    do_qm = high_order or full_tensors
    if debug:
        print('Quantum mechanical symmetry analysis...' if do_qm
              else 'Geometric symmetry analysis...')

    if do_qm:
        pgroup, R_S2M = _hamsymm_eigs(sys, higher_zeeman, debug, strains=strains)
    else:
        pgroup, R_S2M = _hamsymm_geom(sys, debug, strains=strains)

    if getattr(sys, 'tdm', None) is not None:
        pgroup = 'C1'

    # EasySpin returns R_S2M (columns = symmetry axes); torchspin's convention
    # (consumed by pepper) is the mol→sym matrix, i.e. the transpose.
    return pgroup, R_S2M.T.contiguous()


# ---------------------------------------------------------------------------
# Presence checks for the analysis-path decision
# ---------------------------------------------------------------------------

def _is_full(T: Optional[torch.Tensor], count: int) -> bool:
    """True if ``T`` holds ``count`` stacked full 3×3 matrices.

    Requires a 2-D tensor of shape ``(3*count, 3)`` — unlike some
    ``SpinSystem.full*`` properties, a 1-D principal-value triple for a
    single tensor is *not* counted as full.
    """
    if T is None or count <= 0:
        return False
    T = torch.as_tensor(T)
    return T.ndim == 2 and T.shape[0] == 3 * count and T.shape[1] == 3


def _full_tensors_given(sys: SpinSystem) -> bool:
    """EasySpin's ``fullTensorsGiven``: any interaction given as full matrices."""
    nE = sys.nElectrons
    nN = sys.nNuclei
    nEE = nE * (nE - 1) // 2
    nNN = nN * (nN - 1) // 2
    return any([
        _is_full(sys.g, nE),
        # A is (nN, 3*nE) for principal values, (3*nN, 3*nE) for full matrices
        (sys.A is not None and nN > 0 and sys.A.ndim == 2
         and sys.A.shape[0] == 3 * nN),
        _is_full(sys.D, nE),
        _is_full(sys.ee, nEE),
        _is_full(sys.Q, nN),
        _is_full(getattr(sys, 'sigma', None), nN),
        _is_full(getattr(sys, 'nn', None), nNN),
    ])


def _stevens_present(sys: SpinSystem) -> bool:
    """True if any Stevens-operator field ``B_k`` was given (like EasySpin's
    ``~isempty(Sys.B)``, this fires even for all-zero coefficients)."""
    if sys.B is None:
        return False
    return any(Bk is not None for Bk in sys.B)


def _higher_zeeman_present(sys: SpinSystem) -> bool:
    if sys.Ham is None:
        return False
    for val in sys.Ham.values():
        if val is not None and torch.any(torch.as_tensor(val) != 0):
            return True
    return False


def _crystal_field_present(sys: SpinSystem) -> bool:
    for k in range(1, 13):
        cf = getattr(sys, f'CF{k}', None)
        if cf is not None and torch.any(torch.as_tensor(cf) != 0):
            return True
    return False


# ---------------------------------------------------------------------------
# Eigenvalue-based analysis (EasySpin hamsymm_eigs)
# ---------------------------------------------------------------------------

def _candidate_frames(sys: SpinSystem) -> torch.Tensor:
    """Enumerate candidate symmetry frames (EasySpin ``hamsymm_eigs`` prelude).

    Collects the molecular frame plus every tensor frame (gFrame, eeFrame,
    DFrame, AFrame, QFrame, nnFrame, sigmaFrame — Stevens ``BFrame`` is NOT
    included, exactly as in EasySpin), de-duplicates the Euler-angle rows,
    and for each frame forms the three cyclic axis permutations
    ``R@Rz, R@Rx, R@Ry`` with ``R = erot(angles).T``. Returns the unique
    rotation matrices in first-occurrence order, shape ``(nFrames, 3, 3)``;
    each is ``R_S2M`` (columns = candidate symmetry axes in the molecular
    frame).
    """
    rows = [np.zeros(3)]

    def _add(frames):
        if frames is None:
            return
        f = torch.as_tensor(frames, dtype=torch.float64).reshape(-1, 3)
        for r in f:
            rows.append(r.numpy().copy())

    _add(getattr(sys, 'gFrame', None))
    _add(getattr(sys, 'eeFrame', None))
    _add(getattr(sys, 'DFrame', None))
    _add(getattr(sys, 'AFrame', None))   # (nNuclei, 3*nElectrons) → rows of 3
    _add(getattr(sys, 'QFrame', None))
    _add(getattr(sys, 'nnFrame', None))
    _add(getattr(sys, 'sigmaFrame', None))

    # unique rows, first-occurrence order
    pa = []
    for r in rows:
        if not any(np.array_equal(r, p) for p in pa):
            pa.append(r)

    eye = np.eye(3)
    Rz = eye
    Rx = eye[[1, 2, 0], :]
    Ry = eye[[2, 0, 1], :]

    rots = []
    for angles in pa:
        R = erot(angles.tolist()).numpy().T
        for P in (Rz, Rx, Ry):
            rots.append(R @ P)

    # unique 3×3 matrices, first-occurrence order (exact float equality, as
    # MATLAB's unique(...,'rows'))
    uniq = []
    for M in rots:
        if not any(np.array_equal(M, U) for U in uniq):
            uniq.append(M)

    return torch.tensor(np.stack(uniq), dtype=torch.float64)


def _probe_field_vectors() -> torch.Tensor:
    """The twelve probe field vectors (mT) of EasySpin's ``hamsymm_eigs``,
    shape ``(3, 12)``."""
    q = 7.3234    # small theta increment (deg)
    qq = 6.3456   # small phi increment (deg)
    the = np.array([q, q, q, q / 2, q / 2, q, 180 - q, 180 - q, 180 - q,
                    90 - q, q, q]) * math.pi / 180
    phi = (np.array([0, 120, 90, 90, 0, -90, 0, -90, 180, 90, -2 * qq, 180])
           + qq) * math.pi / 180
    field = 350.0
    vecs = field * np.stack([
        np.sin(the) * np.cos(phi),
        np.sin(the) * np.sin(phi),
        np.cos(the),
    ])
    return torch.tensor(vecs, dtype=torch.float64)


def _eqeig(eA: torch.Tensor, eB: torch.Tensor) -> bool:
    """EasySpin ``eqeig``: sorted eigenvalue sets agree to 1e-12 relative."""
    threshold = 1e-12
    eA = torch.sort(eA).values
    eB = torch.sort(eB).values
    return bool(torch.linalg.norm(eA - eB) < threshold * torch.linalg.norm(eA))


def _hamsymm_eigs(
    sys: SpinSystem, higher_zeeman: bool, debug: bool = False, strains=(),
) -> tuple[str, torch.Tensor]:
    """Eigenvalue-based point-group detection (EasySpin ``hamsymm_eigs``).

    ``strains`` (from :func:`_strain_tensors`): anisotropic strain quadratic
    forms ``W``; a symmetry operation must also leave every strain width
    ``z.W.z`` unchanged (torchspin addition, B82; none for strain-free
    systems, which then follow EasySpin exactly).

    Returns ``(pgroup, R_S2M)`` with EasySpin's column convention.
    """
    from torchspin.ham import ham

    Rots = _candidate_frames(sys)
    nFrames = Rots.shape[0]
    FieldVecs = _probe_field_vectors()
    Ws = [W for code, _, W in strains if code > 0]

    if higher_zeeman:
        def _eigs(Bvec: torch.Tensor) -> torch.Tensor:
            B = Bvec.tolist()
            H = ham(sys, B0=B)          # already includes ham_ezho(sys, B0=B)
            return torch.linalg.eigvalsh(H)
    else:
        H0, mux, muy, muz = ham(sys)

        def _eigs(Bvec: torch.Tensor) -> torch.Tensor:
            H = H0 - Bvec[0] * mux - Bvec[1] * muy - Bvec[2] * muz
            return torch.linalg.eigvalsh(H)

    if Ws:
        def eig_at(Bvec: torch.Tensor):
            z = Bvec / torch.linalg.norm(Bvec)
            return _eigs(Bvec), [z @ W @ z for W in Ws]

        def _eqeig_s(a, b) -> bool:
            return _eqeig(a[0], b[0]) and all(
                abs(float(wa - wb)) <= 1e-10 * float(torch.linalg.norm(W))
                for wa, wb, W in zip(a[1], b[1], Ws))
    else:
        eig_at = _eigs
        _eqeig_s = _eqeig

    if debug:
        print('=' * 59)
        print('initial symmetry: C1')

    highest = 0   # C1
    RMatrix = torch.eye(3, dtype=torch.float64)

    for iFrame in range(nFrames):
        if debug:
            print(f'Frame {iFrame + 1} ------------------------')
        R = Rots[iFrame]
        B = R @ FieldVecs  # (3, 12), probe fields in the molecular frame

        eA = eig_at(B[:, 0])
        eB = eig_at(B[:, 1])
        eC = eig_at(B[:, 2])

        C3 = _eqeig_s(eB, eA)   # C3 along z?
        C4 = _eqeig_s(eC, eA)   # C4 along z?
        if debug:
            print('C3z axis found' if C3 else 'No C3z axis found')
            print('C4z axis found' if C4 else 'No C4z axis found')

        if not C3 and not C4:            # Ci, C2h, D2h
            C2z = _eqeig_s(eA, eig_at(B[:, 11]))
            if not C2z:
                pg = 1                   # Ci
            else:
                sigmaxz = _eqeig_s(eA, eig_at(B[:, 10]))
                pg = 3 if sigmaxz else 2  # D2h : C2h
        elif C3 and not C4:              # S6, D3d, Th, C6h, D6h
            sigmaxy = _eqeig_s(eA, eig_at(B[:, 6]))
            if sigmaxy:                  # Th, C6h, D6h
                C2z = _eqeig_s(eC, eig_at(B[:, 5]))
                if C2z:                  # C6h, D6h
                    C2x = _eqeig_s(eC, eig_at(B[:, 7]))
                    pg = 9 if C2x else 8  # D6h : C6h
                else:
                    pg = 10              # Th
            else:                        # S6, D3d
                C2x = _eqeig_s(eC, eig_at(B[:, 7]))
                C2y = False
                if not C2x:
                    C2y = _eqeig_s(eA, eig_at(B[:, 8]))
                pg = 7 if (C2x or C2y) else 6  # D3d : S6
        elif not C3 and C4:              # C4h, D4h, Oh
            C2x = _eqeig_s(eC, eig_at(B[:, 7]))
            if C2x:                      # D4h, Oh
                b = B[:, 9]
                Bs = torch.stack([b[1], b[2], b[0]])
                C3d = _eqeig_s(eig_at(b), eig_at(Bs))
                pg = 11 if C3d else 5    # Oh : D4h
            else:
                pg = 4                   # C4h
        else:                            # C3 and C4: Dinfh, O3
            Cinfx = _eqeig_s(eC, eig_at(B[:, 3]))
            pg = 13 if Cinfx else 12     # O3 : Dinfh

        if debug:
            print(f'Point group {_EIG_GROUP_NAMES[pg - 1]}')

        if pg > highest:
            highest = pg
            RMatrix = R.clone()
            if debug:
                print('Highest symmetry up to now.')

    if highest == 0:
        raise RuntimeError('No point group found! Save input spin system and report bug!')

    group = _EIG_GROUP_NAMES[highest - 1]
    if debug:
        print('=' * 59)
        print(f'Symmetry = {group}')
    return group, RMatrix


# ---------------------------------------------------------------------------
# Geometric analysis (EasySpin hamsymm_geom)
# ---------------------------------------------------------------------------

def _hamsymm_geom(
    sys: SpinSystem, debug: bool = False, strains=(),
) -> tuple[str, torch.Tensor]:
    """Geometric point-group detection. Returns ``(pgroup, R_S2M)``.

    The anisotropic strain tensors (``strains``, from
    :func:`_strain_tensors`) are combined after the Hamiltonian tensors, so
    the frame of a strain-free system and its group are EasySpin's (B82).
    """
    tensor_syms, tensor_axes, tensor_names = _collect_tensor_symmetries(sys)
    for k, (code, R_S2M, _) in enumerate(strains):
        if code > 0:
            tensor_syms.append(code)
            tensor_axes.append(R_S2M)
            tensor_names.append(f'strain{k + 1}')

    if debug:
        print('=' * 59)
        print(f'  {len(tensor_syms)} anisotropic tensor(s)')

    if len(tensor_syms) == 0:
        return 'O3', torch.eye(3, dtype=torch.float64)

    sym_names = ['O3', 'Dinfh', 'D2h', 'C2h', 'Ci']
    total_sym = 0  # O3
    total_rot = torch.eye(3, dtype=torch.float64)
    if debug:
        print('  O3 as starting symmetry')

    for i in range(len(tensor_syms)):
        total_sym, total_rot = _combine_symmetries(
            total_sym, total_rot, tensor_syms[i], tensor_axes[i]
        )
        if debug:
            print(f'   + {sym_names[tensor_syms[i]]} ({tensor_names[i]}) = '
                  f'{sym_names[total_sym]}')

    return sym_names[total_sym], total_rot


def _principal_rows(T: torch.Tensor, count: int) -> torch.Tensor:
    """View a principal-value tensor as ``(count, 3)`` rows.

    Accepts ``(count, 3)``, a single ``(3,)`` triple, or ``(count,)`` scalar
    (isotropic) values — the latter is how ``SpinSystem`` stores e.g.
    ``g=[2, 2]`` for two electrons.
    """
    T = torch.as_tensor(T, dtype=torch.float64)
    if T.ndim == 1:
        if T.shape[0] == 3 and count == 1:
            return T.unsqueeze(0)
        if T.shape[0] == count:
            return T.unsqueeze(1).expand(count, 3)
        if T.shape[0] == 3:
            return T.unsqueeze(0).expand(count, 3)
    return T.reshape(-1, 3)


def _is_isotropic(sys: SpinSystem) -> bool:
    """Check if all interactions are isotropic."""

    def _eq(a, b) -> bool:
        # EasySpin (eqeig / hamsymm): equality to 1e-12 relative — a parameter
        # perturbed by a finite-difference step must break the symmetry
        a = float(torch.as_tensor(a).detach()); b = float(torch.as_tensor(b).detach())
        return abs(a - b) <= 1e-12 * max(abs(a), abs(b), 1e-300)

    def _is_iso_tensor(T: Optional[torch.Tensor]) -> bool:
        """True if tensor is None or isotropic (all principal values equal)."""
        if T is None:
            return True
        if T.ndim == 1 and T.shape[0] != 3:
            return True   # per-spin isotropic scalars
        if T.ndim == 1 and T.shape[0] == 3:
            return _eq(T[0], T[1]) and _eq(T[1], T[2])
        elif T.ndim == 2:
            # Multiple tensors: check each row
            return all(
                _eq(T[i, 0], T[i, 1]) and _eq(T[i, 1], T[i, 2])
                for i in range(T.shape[0])
            )
        return True

    if not _is_iso_tensor(sys.g):
        return False
    if not _is_iso_tensor(sys.D):
        return False

    if sys.A is not None:
        for i in range(sys.nNuclei):
            for e in range(sys.nElectrons):
                A_block = sys.A[i, 3*e:3*e+3]
                if not _eq(A_block[0], A_block[1]) or not _eq(A_block[1], A_block[2]):
                    return False

    if not _is_iso_tensor(sys.ee):
        return False
    if not _is_iso_tensor(sys.Q):
        return False
    if sys.nNuclei > 0:
        if not _is_iso_tensor(getattr(sys, 'sigma', None)):
            return False
        if not _is_iso_tensor(getattr(sys, 'nn', None)):
            return False

    return True


def _collect_tensor_symmetries(
    sys: SpinSystem,
) -> tuple[list[int], list[torch.Tensor], list[str]]:
    """Extract symmetry and orientation of each anisotropic tensor.

    Tensor order follows EasySpin ``hamsymm_geom``: g, D, ee, sigma, A, Q, nn.
    The order matters because the returned frame is that of the first
    tensor in a combination.

    Returns
    -------
    syms: list of int
        Symmetry codes: 0=O3, 1=Dinfh, 2=D2h
    axes: list of Tensor
        3×3 rotation matrices R_S2M (columns = principal axes in the
        molecular frame)
    names: list of str
        Tensor names for debugging
    """
    syms, axes, names = [], [], []
    nE = sys.nElectrons
    nN = sys.nNuclei

    def _push(pv, ea, name):
        sym, ax = _tensor_symmetry(pv, ea)
        if sym > 0:
            syms.append(sym)
            axes.append(ax)
            names.append(name)

    zeros3 = torch.zeros(3, dtype=torch.float64)

    # g-tensors (nElectrons, 3)
    if sys.g is not None:
        g_reshaped = _principal_rows(sys.g, nE)
        gFrame = (sys.gFrame.reshape(-1, 3) if sys.gFrame is not None
                  else torch.zeros(nE, 3, dtype=torch.float64))
        for i in range(nE):
            _push(g_reshaped[i], gFrame[i], f'g{i+1}')

    # D-tensors (zero-field splitting), only for S > 1/2
    if sys.D is not None:
        D_reshaped = _principal_rows(sys.D, nE)
        DFrame = (sys.DFrame.reshape(-1, 3) if sys.DFrame is not None
                  else torch.zeros(nE, 3, dtype=torch.float64))
        for i in range(nE):
            if sys.S[i] > 0.5:
                _push(D_reshaped[i], DFrame[i], f'D{i+1}')

    # ee-tensors (electron-electron)
    if sys.ee is not None:
        n_ee = nE * (nE - 1) // 2
        ee_reshaped = _principal_rows(sys.ee, max(n_ee, 1))
        eeFrame = (sys.eeFrame.reshape(-1, 3) if sys.eeFrame is not None
                   else torch.zeros(max(n_ee, 1), 3, dtype=torch.float64))
        for i in range(min(n_ee, ee_reshaped.shape[0])):
            _push(ee_reshaped[i], eeFrame[i], f'ee{i+1}')

    # sigma (nuclear shielding) tensors
    sigma = getattr(sys, 'sigma', None)
    if sigma is not None and nN > 0:
        sigma_reshaped = sigma.reshape(-1, 3)
        sigmaFrame = getattr(sys, 'sigmaFrame', None)
        sigmaFrame = (sigmaFrame.reshape(-1, 3) if sigmaFrame is not None
                      else torch.zeros(nN, 3, dtype=torch.float64))
        for i in range(min(nN, sigma_reshaped.shape[0])):
            _push(sigma_reshaped[i], sigmaFrame[i], f'sigma{i+1}')

    # A-tensors (hyperfine) - (nNuclei, 3*nElectrons)
    if sys.A is not None:
        AFrame = (sys.AFrame.reshape(nN, -1) if sys.AFrame is not None
                  else torch.zeros(nN, 3 * nE, dtype=torch.float64))
        for n in range(nN):
            for e in range(nE):
                _push(sys.A[n, 3*e:3*e+3], AFrame[n, 3*e:3*e+3], f'A{e+1}{n+1}')

    # Q-tensors (nuclear quadrupole), only for I > 1/2
    if sys.Q is not None:
        Q_reshaped = _principal_rows(sys.Q, nN)
        QFrame = (sys.QFrame.reshape(-1, 3) if sys.QFrame is not None
                  else torch.zeros(nN, 3, dtype=torch.float64))
        for i in range(nN):
            if sys.I[i] > 0.5:
                _push(Q_reshaped[i], QFrame[i], f'Q{i+1}')

    # nn-tensors (nucleus-nucleus)
    nn = getattr(sys, 'nn', None)
    if nn is not None and nN > 1:
        n_nn = nN * (nN - 1) // 2
        nn_reshaped = nn.reshape(-1, 3)
        nnFrame = getattr(sys, 'nnFrame', None)
        nnFrame = (nnFrame.reshape(-1, 3) if nnFrame is not None
                   else torch.zeros(max(n_nn, 1), 3, dtype=torch.float64))
        for i in range(min(n_nn, nn_reshaped.shape[0])):
            _push(nn_reshaped[i], nnFrame[i], f'nn{i+1}')

    return syms, axes, names


def _tensor_symmetry(
    principal_values: torch.Tensor, euler_angles: torch.Tensor
) -> tuple[int, torch.Tensor]:
    """Determine symmetry of a single second-order tensor (EasySpin
    ``tensorsymmetry``).

    Parameters
    ----------
    principal_values: (3,)
        Principal values of the tensor.
    euler_angles: (3,)
        Euler angles (α, β, γ) of the tensor frame relative to the molecular
        frame.

    Returns
    -------
    sym_code: int
        0=O3 (isotropic), 1=Dinfh (axial), 2=D2h (rhombic)
    R_S2M: (3, 3)
        ``erot(angles).T`` with columns cyclically permuted so that the
        unique (symmetry) axis is the third column. Columns are the
        tensor-frame axes expressed in the molecular frame.
    """
    pv = torch.as_tensor(principal_values, dtype=torch.float64)
    pv_sorted = torch.sort(pv).values
    tol = 1e-10 * max(torch.abs(pv_sorted).max().item(), 1e-300)
    n_equal = int(torch.sum(torch.diff(pv_sorted).abs() <= tol).item())

    if n_equal == 2:
        sym_code = 0
        z_axis_idx = 2
    elif n_equal == 1:
        sym_code = 1
        if torch.abs(pv[1] - pv[2]) <= tol:
            z_axis_idx = 0  # x is unique
        elif torch.abs(pv[0] - pv[2]) <= tol:
            z_axis_idx = 1  # y is unique
        else:
            z_axis_idx = 2  # z is unique
    else:
        sym_code = 2
        z_axis_idx = 2

    # EasySpin: symFrame = erot(EulerAngles).'; symFrame = symFrame(:,idx(zAxis+(0:2)))
    R_full = erot(torch.as_tensor(euler_angles, dtype=torch.float64).tolist()).T
    cyclic_perm = [1, 2, 0, 1, 2]
    col_indices = [cyclic_perm[z_axis_idx + i] for i in range(3)]
    R = R_full[:, col_indices].contiguous()

    return sym_code, R


def _combine_symmetries(
    sym1: int,
    R1: torch.Tensor,
    sym2: int,
    R2: torch.Tensor,
) -> tuple[int, torch.Tensor]:
    """Combine two tensor symmetries to get total symmetry (EasySpin
    ``combinesymms``).

    Symmetry codes: 0=O3, 1=Dinfh, 2=D2h, 3=C2h, 4=Ci

    Parameters
    ----------
    sym1, sym2: int
        Symmetry codes of the two tensors.
    R1, R2: (3, 3)
        Rotation matrices R_S2M (columns = principal axes in molecular frame).

    Returns
    -------
    total_sym: int
        Combined symmetry code.
    total_R: (3, 3)
        Combined R_S2M.
    """
    O3, Dinfh, D2h, C2h, Ci = 0, 1, 2, 3, 4

    # Reorder so sym1 >= sym2 (sym1 has lower symmetry)
    if sym1 < sym2:
        sym1, sym2 = sym2, sym1
        R1, R2 = R2, R1

    if sym1 == Ci or sym2 == O3:
        return sym1, R1
    if sym1 == O3:
        return sym2, R2

    delta = 0.2  # deg
    parallel_limit = delta * np.pi / 180
    perp_limit = (90 - delta) * np.pi / 180

    def _angle(a, b):
        return torch.acos(torch.abs(torch.dot(a, b)).clamp(-1, 1)).item()

    cyclic = [1, 2, 0, 1, 2]

    # Case: Dinfh + Dinfh
    if sym1 == Dinfh and sym2 == Dinfh:
        z1 = R1[:, 2]
        z2 = R2[:, 2]
        angle = _angle(z1, z2)
        if angle < parallel_limit:
            return Dinfh, R1
        new_z = torch.linalg.cross(z1, z2)
        new_z = new_z / torch.linalg.norm(new_z)
        R_new = torch.stack([torch.linalg.cross(z1, new_z), z1, new_z], dim=1)
        if angle > perp_limit:
            return D2h, R_new
        return C2h, R_new

    # Case: D2h + Dinfh
    if sym1 == D2h and sym2 == Dinfh:
        z2 = R2[:, 2]
        angles = torch.tensor([_angle(z2, R1[:, i]) for i in range(3)])
        if bool((angles < parallel_limit).any()):
            return D2h, R1
        if bool((angles > perp_limit).any()):
            # z2 in a D2h sigma plane
            new_z_idx = int(angles.argmax().item())
            new_indices = [cyclic[new_z_idx + i] for i in range(3)]
            return C2h, R1[:, new_indices].contiguous()
        return Ci, R1

    # Case: C2h + Dinfh
    if sym1 == C2h and sym2 == Dinfh:
        angle = _angle(R1[:, 2], R2[:, 2])
        if angle < parallel_limit or angle > perp_limit:
            return C2h, R1
        return Ci, R1

    # Case: D2h + D2h
    if sym1 == D2h and sym2 == D2h:
        # allAngles(j, i) = angle between R2[:, j] and R1[:, i]
        all_angles = torch.tensor([
            [_angle(R2[:, j], R1[:, i]) for i in range(3)] for j in range(3)
        ])
        min_angles = all_angles.min(dim=0).values      # per column of R1
        axis_collinear = min_angles < parallel_limit
        if bool(axis_collinear.all()):
            return D2h, R1
        if bool(axis_collinear.any()):
            new_z_idx = int(min_angles.argmin().item())
            new_indices = [cyclic[new_z_idx + i] for i in range(3)]
            return C2h, R1[:, new_indices].contiguous()
        return Ci, R1

    # Case: C2h + D2h
    if sym1 == C2h and sym2 == D2h:
        z1 = R1[:, 2]
        angles = torch.tensor([_angle(z1, R2[:, i]) for i in range(3)])
        if bool((angles < parallel_limit).any()):
            return C2h, R1
        return Ci, R1

    raise ValueError(f'Cannot compute symmetry combination: {sym1} {sym2}')


# ---------------------------------------------------------------------------
# Strain tensors (torchspin addition, B82)
# ---------------------------------------------------------------------------

def _strain_from_principal(pv, euler) -> tuple[int, torch.Tensor, torch.Tensor]:
    """``(sym_code, R_S2M, W_mol)`` of a quadratic form with principal values
    ``pv`` in the frame given by Euler angles ``euler`` (molecular → tensor)."""
    pv = torch.as_tensor(pv, dtype=torch.float64).detach().reshape(3)
    euler = torch.as_tensor(euler, dtype=torch.float64).detach().reshape(3)
    code, R_S2M = _tensor_symmetry(pv, euler)
    R = erot(euler.tolist()).to(torch.float64)          # molecular → tensor frame
    return code, R_S2M, R.T @ torch.diag(pv) @ R


def _strain_from_matrix(W: torch.Tensor) -> tuple[int, torch.Tensor, torch.Tensor]:
    """``(sym_code, R_S2M, W)`` of a symmetric 3×3 quadratic form ``W`` given
    in the molecular frame (the symmetry axes are its eigenvectors)."""
    W = W.detach().to(torch.float64)
    W = 0.5 * (W + W.T)
    if bool(torch.all(W == torch.diag(torch.diagonal(W)))):
        return _strain_from_principal(torch.diagonal(W), torch.zeros(3, dtype=torch.float64))
    ev, V = torch.linalg.eigh(W)                          # ascending
    tol = 1e-10 * max(float(ev.abs().max()), 1e-300)
    d01, d12 = float(ev[1] - ev[0]) <= tol, float(ev[2] - ev[1]) <= tol
    if d01 and d12:
        code, cols = 0, [0, 1, 2]
    elif d01:
        code, cols = 1, [0, 1, 2]                         # unique axis: ev[2]
    elif d12:
        code, cols = 1, [1, 2, 0]                         # unique axis: ev[0]
    else:
        code, cols = 2, [0, 1, 2]
    R_S2M = V[:, cols].contiguous()
    if float(torch.linalg.det(R_S2M)) < 0:
        R_S2M[:, 0] = -R_S2M[:, 0]
    return code, R_S2M, W


def _strain_tensors(sys: SpinSystem) -> list:
    """Orientation dependence of the strain line widths as quadratic forms.

    Every strain width torchspin computes (:mod:`torchspin.strainwidth`) is a
    function of the field direction ``z`` (molecular frame) with the
    symmetry of the quadratic forms ``z.W.z`` returned here; the grid
    symmetry must be a symmetry of each of them (B82).  Returns a list of
    ``(sym_code, R_S2M, W_mol)`` (sym codes as :func:`_tensor_symmetry`):

    * ``HStrain`` / ``HStrainL``: ``diag(HStrain**2)`` in the molecular frame.
    * ``gStrain``: per electron ``G∘G`` (element-wise square, as in the width
      formula) with ``G = R_g2M diag(gStrain/g) R_g2M'``, i.e. the g frame.
    * ``AStrain`` (single electron, first nucleus): ``A∘A`` and ``G∘A`` with
      ``A = R_A2M diag(AStrain) R_A2M'`` (``AFrame``); together with ``G∘G``
      these keep ``(G + c·mI·A)∘(G + c·mI·A)`` invariant for every ``mI``.
    * ``DStrain``: dH/dD (axial) and dH/dE (rhombic) operators in the D frame:
      ``Dinfh`` about the D-frame z axis without E strain, else ``D2h`` in the
      D frame (also with ``DStrainCorr``).
    """
    out = []

    def _nz(t) -> bool:
        return t is not None and bool(torch.any(torch.as_tensor(t).detach() != 0))

    zeros3 = torch.zeros(3, dtype=torch.float64)
    for name in ('HStrain', 'HStrainL'):
        H = getattr(sys, name, None)
        if _nz(H):
            H = torch.as_tensor(H, dtype=torch.float64).detach().reshape(3)
            out.append(_strain_from_principal(H ** 2, zeros3))

    nE = sys.nElectrons
    gS = getattr(sys, 'gStrain', None)
    G = [None] * nE
    if _nz(gS) and sys.g is not None and not _full_tensors_given_g(sys):
        gS = torch.as_tensor(gS, dtype=torch.float64).detach().reshape(-1, 3)
        g_rows = _principal_rows(torch.as_tensor(sys.g).detach(), nE)
        gFrame = getattr(sys, 'gFrame', None)
        gFrame = (torch.zeros(nE, 3, dtype=torch.float64) if gFrame is None
                  else torch.as_tensor(gFrame, dtype=torch.float64).detach().reshape(-1, 3))
        for e in range(min(nE, gS.shape[0])):
            if not bool(torch.any(gS[e] != 0)):
                continue
            M = torch.diag(gS[e] / g_rows[e])
            if bool(torch.any(gFrame[e] != 0)):
                R = erot(gFrame[e].tolist()).to(torch.float64)
                M = R.T @ M @ R
            G[e] = M
            out.append(_strain_from_matrix(M * M))

    AS = getattr(sys, 'AStrain', None)
    if _nz(AS) and sys.nNuclei > 0 and nE == 1:
        A = torch.diag(torch.as_tensor(AS, dtype=torch.float64).detach().reshape(3))
        AFrame = getattr(sys, 'AFrame', None)
        if AFrame is not None:
            a0 = torch.as_tensor(AFrame, dtype=torch.float64).detach().reshape(sys.nNuclei, -1)[0, :3]
            if bool(torch.any(a0 != 0)):
                R = erot(a0.tolist()).to(torch.float64)
                A = R.T @ A @ R
        out.append(_strain_from_matrix(A * A))
        if G[0] is not None:
            out.append(_strain_from_matrix(G[0] * A))

    DS = getattr(sys, 'DStrain', None)
    if _nz(DS):
        DS = torch.as_tensor(DS, dtype=torch.float64).detach()
        DS = DS.reshape(DS.shape[0] if DS.ndim > 1 else 1, -1)
        DFrame = getattr(sys, 'DFrame', None)
        DFrame = (torch.zeros(nE, 3, dtype=torch.float64) if DFrame is None
                  else torch.as_tensor(DFrame, dtype=torch.float64).detach().reshape(-1, 3))
        for e in range(min(nE, DS.shape[0])):
            dD = float(DS[e, 0])
            dE = float(DS[e, 1]) if DS.shape[1] > 1 else 0.0
            if (dD == 0 and dE == 0) or float(sys.S[e]) < 1.0:
                continue
            pv = [1.0, 2.0, 3.0] if dE != 0 else [0.0, 0.0, 1.0]
            out.append(_strain_from_principal(pv, DFrame[e]))
    return out


def _full_tensors_given_g(sys: SpinSystem) -> bool:
    return _is_full(sys.g, sys.nElectrons)


# ---------------------------------------------------------------------------
# Grid symmetry for differentiated parameters (B1)
# ---------------------------------------------------------------------------

#: SpinSystem fields that determine the symmetry: hamsymm's Hamiltonian
#: inputs and the strains (B82).
_SYMMETRY_FIELDS = (
    'g', 'gFrame', 'D', 'DFrame', 'A', 'AFrame', 'ee', 'eeFrame', 'sigma',
    'sigmaFrame', 'Q', 'QFrame', 'nn', 'nnFrame', 'B', 'BFrame', 'Ham',
    'H0Extra', *(f'CF{k}' for k in range(1, 13)),
    'HStrain', 'HStrainL', 'gStrain', 'AStrain', 'DStrain',
)
_FRAME_FIELDS = frozenset(f for f in _SYMMETRY_FIELDS if f.endswith('Frame'))

#: Size of the generic perturbation of a differentiated field, relative to the
#: field's largest entry (at least 1): far above hamsymm's equality tolerances
#: (1e-10 principal values, 1e-12 eigenvalues), far below any anisotropy.
_REL_PERTURBATION = 1e-6
#: Largest Euler-angle change (rad) of a differentiated frame: above the 0.2 deg
#: (3.5e-3 rad) collinearity tolerance of the geometric analysis, or a frame
#: that moves would still be read as aligned.
_ANGLE_PERTURBATION = 1e-2


def _grad_tensors(v, path=()):
    """``(path, tensor)`` for every tensor in a nested field value that requires grad."""
    if torch.is_tensor(v):
        return [(path, v)] if v.requires_grad else []
    if isinstance(v, dict):
        return [x for k, u in v.items() for x in _grad_tensors(u, path + (k,))]
    if isinstance(v, (list, tuple)):
        return [x for i, u in enumerate(v) for x in _grad_tensors(u, path + (i,))]
    return []


def _replaced(v, path, new):
    """Copy of the nested value ``v`` with the item at ``path`` replaced by ``new``."""
    if not path:
        return new
    k, rest = path[0], path[1:]
    if isinstance(v, dict):
        out = dict(v)
        out[k] = _replaced(v[k], rest, new)
        return out
    items = list(v)
    items[k] = _replaced(v[k], rest, new)
    return type(v)(items) if isinstance(v, tuple) else items


def _leaves(tensors):
    """Leaf tensors (autograd ``AccumulateGrad`` variables) behind ``tensors``,
    in a deterministic order (first reached, depth-first)."""
    leaves, seen_leaf, seen_fn = [], set(), set()
    for t in tensors:
        if t.grad_fn is None:
            if id(t) not in seen_leaf:
                seen_leaf.add(id(t)); leaves.append(t)
            continue
        stack = [t.grad_fn]
        while stack:
            fn = stack.pop()
            if fn is None or fn in seen_fn:
                continue
            seen_fn.add(fn)
            var = getattr(fn, 'variable', None)
            if var is not None:
                if id(var) not in seen_leaf:
                    seen_leaf.add(id(var)); leaves.append(var)
                continue
            stack.extend(nf for nf, _ in reversed(fn.next_functions))
    return leaves


def _field_directions(items, gen):
    """Change of each grad-requiring field tensor under a generic perturbation
    of the parameters (the autograd leaves), ``J @ dp``.

    Leaf fields (a tensor the user passed directly) are their own
    parameterization: ``J = I``.  Fields with a graph get ``J @ dp`` from the
    double-backward Jacobian-vector product on the existing graph (the graph is
    retained, nothing is accumulated into ``.grad``).  Raises if the graph does
    not support double backward.
    """
    tensors = [t for _, _, t in items]
    leaves = _leaves(tensors)
    dp = {id(l): torch.rand(l.shape, generator=gen, dtype=torch.float64) + 0.5 for l in leaves}
    out = [None] * len(items)
    graph_idx = []
    for i, t in enumerate(tensors):
        if t.grad_fn is None:
            out[i] = dp[id(t)].to(t.dtype)
        else:
            graph_idx.append(i)
    if graph_idx:
        outs = [tensors[i] for i in graph_idx]
        graph_leaves = leaves
        us = [torch.zeros_like(o, requires_grad=True) for o in outs]
        vjps = torch.autograd.grad(outs, graph_leaves, grad_outputs=us, create_graph=True,
                                   allow_unused=True)
        used = [(g, dp[id(l)].to(g.dtype)) for g, l in zip(vjps, graph_leaves)
                if g is not None and g.requires_grad]
        if used:
            jvps = torch.autograd.grad([g for g, _ in used], us, grad_outputs=[v for _, v in used],
                                       allow_unused=True)
        else:
            jvps = [None] * len(us)
        for i, d in zip(graph_idx, jvps):
            out[i] = torch.zeros_like(tensors[i]).detach() if d is None else d.detach()
    return out


def _perturbed_system(sys, items, directions):
    """Copy of ``sys`` with each grad-requiring field tensor moved along its direction,
    scaled to the generic perturbation size."""
    import copy
    p = copy.copy(sys)
    new_fields = {}
    for (field, path, t), d in zip(items, directions):
        t0 = t.detach()
        d = d.to(t0.dtype)
        dmax = float(d.abs().max()) if d.numel() else 0.0
        if dmax == 0.0:
            new = t0
        elif field in _FRAME_FIELDS:
            new = t0 + (_ANGLE_PERTURBATION / dmax) * d
        else:
            scale = max(float(t0.abs().max()) if t0.numel() else 0.0, 1.0)
            new = t0 + (_REL_PERTURBATION * scale / dmax) * d
        base = new_fields.get(field, getattr(sys, field))
        new_fields[field] = _replaced(base, path, new)
    for field, v in new_fields.items():
        object.__setattr__(p, field, v)
    return p


def _signed_permutation_equivalent(R1: torch.Tensor, R2: torch.Tensor, tol: float = 1e-9) -> bool:
    """True if the rows (axes) of ``R2`` are the rows of ``R1`` up to order and sign."""
    M = (R1.to(torch.float64) @ R2.to(torch.float64).T).abs()
    ones = torch.ones(3, dtype=M.dtype)
    return bool(((M - M.round()).abs() < tol).all()
                and torch.allclose(M.sum(0), ones) and torch.allclose(M.sum(1), ones))


def grid_symmetry(sys: SpinSystem) -> tuple[str, torch.Tensor]:
    """Automatic powder-grid symmetry, valid for the differentiated parameters (B1).

    The grid is chosen outside the autograd graph, so it must hold not only at
    the current values but along every direction the gradient measures.  A
    symmetry that holds by accident at the current values (axial g given as
    three independent numbers, collinear frames) is broken by that perturbation,
    and a grid of that symmetry gives a wrong gradient.

    * No symmetry input (:data:`_SYMMETRY_FIELDS`: tensors, frames, high-order
      terms, strains) requires grad: exactly :func:`hamsymm` (bit-identical
      results, no extra cost).
    * Otherwise the symmetry of the *parameterization*: the autograd leaves
      behind the grad-requiring fields get a deterministic generic perturbation
      ``dp``, each field moves along ``J @ dp`` (double-backward
      Jacobian-vector product on the existing graph; a field that is itself a
      leaf moves along its own ``dp``), scaled to 1e-6 relative (Euler angles:
      1e-2 rad, above the 0.2 deg collinearity tolerance), and :func:`hamsymm`
      runs on that copy.  So a scalar g keeps O3, ``[g_perp, g_perp, g_par]``
      from two leaves keeps Dinfh, and three independent components give D2h.
    * The frame is checked with two independent perturbations: if their groups
      differ, or their frames are not the same axes up to order and sign (the
      symmetry frame moves with the parameters, e.g. differentiated Euler
      angles), the grid is ``'Ci'``, exact for any Hamiltonian.
    * If the graph does not support double backward, every grad-requiring
      field is perturbed independently (its own parameterization), which is
      never less safe.

    Returns ``(pgroup, R)`` with :func:`hamsymm`'s convention.
    """
    items = [(f, path, t) for f in _SYMMETRY_FIELDS
             for path, t in _grad_tensors(getattr(sys, f, None))]
    if not items:
        return hamsymm(sys)
    results = []
    for seed in (20261112, 20261113):
        gen = torch.Generator().manual_seed(seed)
        try:
            directions = _field_directions(items, gen)
        except RuntimeError:
            gen = torch.Generator().manual_seed(seed)
            directions = [torch.rand(t.shape, generator=gen, dtype=torch.float64) + 0.5
                          for _, _, t in items]
        results.append(hamsymm(_perturbed_system(sys, items, directions)))
    (pg1, R1), (pg2, R2) = results
    if pg1 != pg2:
        return 'Ci', torch.eye(3, dtype=torch.float64)
    if pg1 not in ('O3', 'Ci', 'C1') and not _signed_permutation_equivalent(R1, R2):
        return 'Ci', torch.eye(3, dtype=torch.float64)
    return pg1, R1


def grid_angles_to_molecular(phi, theta, R: torch.Tensor):
    """Rotate grid directions ``(phi, theta)`` from the symmetry frame of
    :func:`hamsymm` (``v_sym = R @ v_mol``) into the molecular frame, as
    pepper does.  Returns torch tensors; unchanged when ``R`` is the identity
    (to 1e-10)."""
    phi = torch.as_tensor(phi, dtype=torch.float64)
    theta = torch.as_tensor(theta, dtype=torch.float64)
    R = torch.as_tensor(R, dtype=torch.float64)
    if torch.allclose(R, torch.eye(3, dtype=torch.float64), atol=1e-10):
        return phi, theta
    sin_t = torch.sin(theta)
    v_sym = torch.stack([sin_t * torch.cos(phi), sin_t * torch.sin(phi), torch.cos(theta)], dim=1)
    v_mol = (R.T @ v_sym.T).T
    return torch.atan2(v_mol[:, 1], v_mol[:, 0]), torch.acos(v_mol[:, 2].clamp(-1.0, 1.0))
