"""Density matrix propagation for cardamom.

Implements two propagation methods:

1. **fast** (Sezer et al., JCP 128, 165106, 2008):
   Operates in the m_S = -1/2 subspace only (S = 1/2).
   Without hyperfine: scalar propagator ``U = exp(-i dt omega Gp_zz / 2)``.
   With 14N (I=1): 3x3 matrix exponential from axis-angle decomposition.

2. **ISTOs** (Oganesyan, PCCP 13, 4724, 2011):
   Full Hilbert-space propagation using irreducible spherical tensor
   operators and rank-2 Wigner D-matrices from quaternion trajectories.
"""
from __future__ import annotations

import math

import numpy as np
from torchspin._cardamom_utils import tensor_traj
from torchspin.constants import GFREE


# ---------------------------------------------------------------------------
# Fast method (Sezer 2008)
# ---------------------------------------------------------------------------

def propagate_fast(
    g: np.ndarray,
    RTraj: np.ndarray,
    omega: float,
    dtSpin: float,
    nSteps: int,
    nTraj: int,
    *,
    A: np.ndarray | None = None,
    RLab: np.ndarray | None = None,
    groups: int = 1,
) -> np.ndarray:
    """Propagate density matrix using the fast m_S=-1/2 subspace method.

    With ``groups`` > 1 the trajectories are ``groups`` consecutive blocks of
    ``nTraj // groups`` (one powder orientation each) and the signal is
    averaged per block, returning ``(groups, nSteps)``.

    Parameters
    ----------
    g:
        g-tensor principal values, shape ``(3,)``.
    RTraj:
        Rotation matrix trajectory, shape ``(3, 3, nSteps_spatial, nTraj)``.
        The spatial trajectory (may be longer than nSteps).
    omega:
        Microwave angular frequency in rad/s (= 2*pi*mwFreq_Hz).
    dtSpin:
        Spin propagation time step in seconds.
    nSteps:
        Number of spin propagation steps.
    nTraj:
        Number of trajectories.
    A:
        Optional hyperfine tensor principal values (MHz), shape ``(3,)``.
        If provided, includes nuclear spin dynamics (I=1 for 14N).
    RLab:
        Optional lab-frame rotation matrices, shape ``(3, 3, nSteps, nTraj)``.
        Used to combine local and global dynamics.

    Returns
    -------
    Sprho:
        Density matrix trace signal, shape ``(nSteps,)``.
        This is the trajectory-averaged expectation value sum_k rho_kk(t).
    """
    import torch

    # --- Compute tensor trajectories ---
    g_t = torch.tensor(g, dtype=torch.float64)
    RTraj_t = torch.tensor(RTraj[:, :, :nSteps, :], dtype=torch.float64)

    gTensor = tensor_traj(g_t, RTraj_t).numpy()  # (3,3,nSteps,nTraj)

    includeHF = A is not None
    if includeHF:
        A_t = torch.tensor(A, dtype=torch.float64)
        ATensor = tensor_traj(A_t, RTraj_t).numpy()
        # MHz → rad/s
        ATensor = ATensor * 1e6 * 2.0 * np.pi

    # --- Combine with lab-frame rotation if provided ---
    if RLab is not None:
        gTensor = _rotate_tensor_lab(gTensor, RLab[:, :, :nSteps, :])
        if includeHF:
            ATensor = _rotate_tensor_lab(ATensor, RLab[:, :, :nSteps, :])

    # --- Compute propagators ---
    gIso = np.sum(g) / 3.0
    GpTensor = (gTensor - gIso) / GFREE
    Gp_zz = torch.from_numpy(np.ascontiguousarray(GpTensor[2, 2, :, :]))     # (nSteps, nTraj)
    phase = torch.exp(-1j * dtSpin * 0.5 * omega * Gp_zz)                   # (nSteps, nTraj)

    # --- Propagate density matrix (torch: multi-threaded elementwise ops and
    #     batched 3×3 matmuls; the recursion is sequential in time only) ---
    if includeHF:
        U = _build_hf_propagator(ATensor, phase, dtSpin)                     # (nSteps, nTraj, 3, 3) complex
        rho = torch.zeros(nTraj, 3, 3, dtype=torch.complex128)
        rho[:, 0, 0] = rho[:, 1, 1] = rho[:, 2, 2] = 0.5
        tr = torch.empty(nSteps, nTraj, dtype=torch.complex128)
        tr[0] = 1.5
        for iStep in range(1, nSteps):
            U_prev = U[iStep - 1]
            # rho(t+1) = U @ rho(t) @ U  (not U†, per Sezer fast method)
            rho = U_prev @ rho @ U_prev
            tr[iStep] = rho[:, 0, 0] + rho[:, 1, 1] + rho[:, 2, 2]
        signal = tr.reshape(nSteps, groups, -1).mean(dim=2).T.numpy()        # (groups, nSteps)
    else:
        # rho is scalar: rho(t) = 0.5 * prod_{k<t} U_k^2
        U2 = phase ** 2
        rho = torch.ones(nSteps, nTraj, dtype=torch.complex128) * 0.5
        rho[1:] = 0.5 * torch.cumprod(U2[:-1], dim=0)
        signal = rho.reshape(nSteps, groups, -1).mean(dim=2).T.numpy()

    return signal if groups > 1 else signal[0]


def _build_hf_propagator(ATensor: np.ndarray, phase, dtSpin: float):
    """Build the I=1 hyperfine propagator (Eqs. 35, 37, A1-A2 in Sezer 2008).

    Parameters
    ----------
    ATensor:
        Hyperfine tensor trajectory in rad/s, shape ``(3, 3, nSteps, nTraj)``.
    phase:
        Zeeman phase factor ``exp(-i dt/2 ω Gp_zz)``, torch ``(nSteps, nTraj)``.
    dtSpin:
        Spin propagation time step (s).

    Returns
    -------
    U:
        Propagator, torch complex128 ``(nSteps, nTraj, 3, 3)``.
    """
    import torch
    Az = torch.from_numpy(np.ascontiguousarray(ATensor[:, 2, :, :]))        # (3, nSteps, nTraj): A-tensor z column
    a = torch.sqrt((Az ** 2).sum(dim=0))                                    # Eq. 24
    theta = dtSpin * 0.5 * a
    a_safe = torch.where(a < 1e-30, torch.full_like(a, 1e-30), a)
    nx, ny, nz = Az[0] / a_safe, Az[1] / a_safe, Az[2] / a_safe
    ct = torch.cos(theta) - 1.0
    st = -torch.sin(theta)
    s2 = math.sqrt(0.5)
    nxx, nyy, nzz = nx * nx, ny * ny, nz * nz
    nxy, nzx, nzy = nx * ny, nz * nx, nz * ny
    U = torch.empty(a.shape + (3, 3), dtype=torch.complex128)
    U[..., 0, 0] = torch.complex(1 + ct * (nzz + 0.5 * (nxx + nyy)), st * nz)
    U[..., 0, 1] = torch.complex(s2 * (st * ny + ct * nzx), s2 * (st * nx - ct * nzy))
    U[..., 0, 2] = torch.complex(0.5 * ct * (nxx - nyy), -ct * nxy)
    U[..., 1, 0] = torch.complex(s2 * (-st * ny + ct * nzx), s2 * (st * nx + ct * nzy))
    U[..., 1, 1] = torch.complex(1 + ct * (nxx + nyy), torch.zeros_like(a))
    U[..., 1, 2] = torch.complex(s2 * (st * ny - ct * nzx), s2 * (st * nx + ct * nzy))
    U[..., 2, 0] = torch.complex(0.5 * ct * (nxx - nyy), ct * nxy)
    U[..., 2, 1] = torch.complex(s2 * (-st * ny - ct * nzx), s2 * (st * nx - ct * nzy))
    U[..., 2, 2] = torch.complex(1 + ct * (nzz + 0.5 * (nxx + nyy)), -st * nz)
    # Full propagator: Eq. 35 — U = exp(-i dt/2 ω Gp_zz) * expadotI
    return U * phase[..., None, None]


def _rotate_tensor_lab(T: np.ndarray, RLab: np.ndarray) -> np.ndarray:
    """Rotate tensor trajectory into lab frame: RLab @ T @ RLab^T.

    Parameters
    ----------
    T:
        Tensor trajectory, shape ``(3, 3, nSteps, nTraj)``.
    RLab:
        Lab-frame rotation matrices, shape ``(3, 3, nSteps, nTraj)``.

    Returns
    -------
    T_lab:
        Rotated tensor, shape ``(3, 3, nSteps, nTraj)``.
    """
    # R T Rᵀ for every (step, trajectory): batched matmul in (N, 3, 3) layout on
    # torch (multi-threaded; numpy's small-matrix matmul/einsum take ~45 ms per
    # 200k products, torch ~1 ms) — PERF_PLAN §2.4
    import torch
    Rb = torch.from_numpy(np.ascontiguousarray(np.transpose(RLab, (2, 3, 0, 1))))   # (nSteps, nTraj, 3, 3)
    Tb = torch.from_numpy(np.ascontiguousarray(np.transpose(T, (2, 3, 0, 1))))
    out = (Rb @ Tb @ Rb.transpose(-1, -2)).numpy()
    return np.transpose(out, (2, 3, 0, 1))


# ---------------------------------------------------------------------------
# ISTOs method (Oganesyan 2011)
# ---------------------------------------------------------------------------

def _block_average_d2(D2Traj, block_length):
    """Block-average D2 Wigner matrix trajectory.

    Parameters
    ----------
    D2Traj : ndarray, shape (5, 5, nSteps, nTraj)
    block_length : int
        Number of frames per block.

    Returns
    -------
    D2_avg : ndarray, shape (5, 5, nBlocks, nTraj)
    """
    if block_length <= 1:
        return D2Traj
    nSteps = D2Traj.shape[2]
    nBlocks = nSteps // block_length
    if nBlocks == 0:
        return D2Traj
    D2_trim = D2Traj[:, :, :nBlocks * block_length, :]
    D2_reshaped = D2_trim.reshape(5, 5, nBlocks, block_length, -1)
    return D2_reshaped.mean(axis=3)


def _sliding_window_d2(D2Traj, window_length, stride=1):
    """Extract overlapping windows from D2 trajectory.

    Each window position becomes a virtual trajectory, improving
    statistics from a single long MD trajectory.

    Parameters
    ----------
    D2Traj : ndarray, shape (5, 5, nSteps, nTraj)
    window_length : int
    stride : int

    Returns
    -------
    D2_windowed : ndarray, shape (5, 5, window_length, nWindows * nTraj)
    """
    nSteps = D2Traj.shape[2]
    nTraj = D2Traj.shape[3]
    if window_length >= nSteps:
        return D2Traj[:, :, :window_length, :]
    starts = range(0, nSteps - window_length + 1, stride)
    windows = []
    for s in starts:
        windows.append(D2Traj[:, :, s:s + window_length, :])
    return np.concatenate(windows, axis=3)  # (5, 5, window_length, nWindows*nTraj)


def propagate_istos(
    sys_spins: list[float],
    g: np.ndarray,
    A: np.ndarray | None,
    qTraj: np.ndarray,
    omega: float,
    dtSpin: float,
    nSteps: int,
    nTraj: int,
    CenterField: float,
    *,
    D: np.ndarray | None = None,
    gn: list[float] | None = None,
    nuc_spins: list[float] | None = None,
    qLab: np.ndarray | None = None,
    block_length: int = 1,
    d2_cache: dict | None = None,
    detect_Sz: bool = False,
) -> np.ndarray:
    """Propagate density matrix using the ISTOs method (Oganesyan 2011).

    Full Hilbert-space propagation using irreducible spherical tensor
    operators and rank-2 Wigner D-matrices from quaternion trajectories.

    Parameters
    ----------
    sys_spins:
        Spin quantum numbers, e.g. [0.5] or [0.5, 1.0].
    g:
        g-tensor principal values, shape ``(3,)`` or ``(nElectrons, 3)``.
    A:
        Hyperfine tensor principal values in MHz, shape ``(nNuclei, 3)``
        or ``(3,)`` or ``None``.
    qTraj:
        Quaternion trajectory, shape ``(4, nSteps_spatial, nTraj)``.
    omega:
        Microwave angular frequency (rad/s).
    dtSpin:
        Spin propagation time step (s).
    nSteps:
        Number of spin propagation steps.
    nTraj:
        Number of trajectories.
    CenterField:
        Center magnetic field in mT.
    D:
        ZFS tensor principal values in MHz, shape ``(nElectrons, 3)`` or ``None``.
    gn:
        Nuclear g-values.
    nuc_spins:
        Nuclear spin quantum numbers.
    qLab:
        Lab-frame quaternion trajectory, shape ``(4, nSteps, nTraj)``.

    Returns
    -------
    Sprho:
        Trace of S+ * rho(t), averaged over trajectories, shape ``(nSteps,)``.
    """
    from scipy.linalg import expm
    from torchspin._cardamom_istos import magint, wigD
    from torchspin.spinops import sop

    # --- Compute IST decomposition ---
    g = np.atleast_2d(np.asarray(g, dtype=float))
    A_2d = np.atleast_2d(np.asarray(A, dtype=float)) if A is not None else None

    T, F = magint(
        sys_spins, g, CenterField,
        A=A_2d, D=D, gn=gn, nuc_spins=nuc_spins,
        include_nuc_zeeman=False,
    )

    F0 = F['F0'] * 2 * np.pi  # Hz → rad/s
    F2 = F['F2'] * 2 * np.pi
    T0 = T['T0']
    T2 = T['T2']

    nInt = len(T0)
    nStates = int(np.round(np.prod([2 * s + 1 for s in sys_spins])))

    # --- Build Q0 (zeroth rank, time-independent) ---
    Q0 = np.zeros((nStates, nStates), dtype=complex)
    for k in range(nInt):
        Q0 += np.conj(F0[k]) * T0[k]

    # --- H0: isotropic Zeeman part (for interaction frame) ---
    H0 = np.conj(F0[0]) * T0[0]  # first interaction is electron Zeeman

    # --- Build Q2 (second rank, 5x5 rotational basis operators) ---
    Q2 = np.zeros((5, 5, nStates, nStates), dtype=complex)
    for mp in range(5):
        for m in range(5):
            for iInt in range(nInt):
                Q2[mp, m] += np.conj(F2[iInt, mp]) * T2[iInt][m]

    # --- Compute D2 trajectories from quaternions (with optional caching) ---
    cache_key = id(qTraj)
    if d2_cache is not None and cache_key in d2_cache:
        D2Traj = d2_cache[cache_key]
    else:
        D2Traj = wigD(qTraj[:, :nSteps, :])  # (5, 5, nSteps, nTraj)
        if d2_cache is not None:
            d2_cache[cache_key] = D2Traj

    # --- Block averaging (optional) ---
    if block_length > 1:
        D2Traj = _block_average_d2(D2Traj, block_length)
        nSteps = D2Traj.shape[2]  # update after averaging

    # --- Combine local and global dynamics ---
    if qLab is not None:
        D2Lab = wigD(qLab[:, :nSteps, :])
        # Matrix multiply: D2Traj = D2Lab @ D2Traj per step/traj
        D2Combined = np.zeros_like(D2Traj)
        for iStep in range(nSteps):
            for iTraj in range(nTraj):
                D2Combined[:, :, iStep, iTraj] = (
                    D2Lab[:, :, iStep, iTraj] @ D2Traj[:, :, iStep, iTraj]
                )
        D2Traj = D2Combined

    # --- Build Hamiltonians H(t) = Q0 + sum_{mp,m} D2(m,mp,t) * Q2{mp,m} ---
    H = np.tile(Q0[:, :, np.newaxis, np.newaxis], (1, 1, nSteps, nTraj))
    for mp in range(5):
        for m in range(5):
            # D2Traj[m, mp, :, :] is (nSteps, nTraj)
            # Q2[mp, m] is (nStates, nStates)
            H += D2Traj[m, mp, :, :][np.newaxis, np.newaxis, :, :] * \
                 Q2[mp, m, :, :, np.newaxis, np.newaxis]

    # --- Build propagators ---
    # Interaction frame: U = expm(-i*dt*H0) * expm(+i*dt*H)
    U0 = expm(-1j * dtSpin * H0)

    U = np.zeros((nStates, nStates, nSteps, nTraj), dtype=complex)
    for iStep in range(nSteps):
        for iTraj in range(nTraj):
            U_step = expm(1j * dtSpin * H[:, :, iStep, iTraj])
            U[:, :, iStep, iTraj] = U0 @ U_step

    # --- Initial state: rho(0) = Sx (after pi/2 pulse) ---
    Sx = sop(sys_spins, [1, 1]).numpy()
    rho = np.zeros((nStates, nStates, nSteps, nTraj), dtype=complex)
    rho[:, :, 0, :] = Sx[:, :, np.newaxis]

    # --- Propagate: rho(t+1) = U * rho(t) * U† ---
    for iStep in range(1, nSteps):
        for iTraj in range(nTraj):
            U_prev = U[:, :, iStep - 1, iTraj]
            U_adj = U_prev.conj().T
            rho[:, :, iStep, iTraj] = (
                U_prev @ rho[:, :, iStep - 1, iTraj] @ U_adj
            )

    # --- Average over trajectories ---
    rho_avg = np.mean(rho, axis=3)  # (nStates, nStates, nSteps)

    # --- Apply detection operator ---
    if detect_Sz:
        Det = sop(sys_spins, [1, 3]).numpy()  # Sz
    else:
        Det = sop(sys_spins, [1, 4]).numpy()  # S+
    Sprho = np.zeros(nSteps, dtype=complex)
    for iStep in range(nSteps):
        Sprho[iStep] = np.trace(Det @ rho_avg[:, :, iStep])

    return Sprho
