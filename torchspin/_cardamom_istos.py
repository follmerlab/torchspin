"""IST decomposition and Wigner D-matrices for cardamom ISTOs method.

Implements:
- ``istotensor``: irreducible spherical tensor operators from two vector operators
- ``magint``: IST decomposition of spin Hamiltonian interactions
- ``wigD``: rank-2 Wigner D-matrices from quaternion trajectories

References
----------
[1] Oganesyan, Phys. Chem. Chem. Phys. 13, 4724 (2011)
[2] Mehring, Principles of High Resolution NMR in Solids, Appendix A (1983)
"""
from __future__ import annotations

import numpy as np
from torchspin.constants import BMAGN, NMAGN, PLANCK
from torchspin.spinops import sop


# ---------------------------------------------------------------------------
# istotensor — rank-0, rank-1, rank-2 IST operators from two vector operators
# ---------------------------------------------------------------------------

def istotensor(
    a: list[np.ndarray] | np.ndarray,
    b: list[np.ndarray] | np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Build irreducible spherical tensor operators from two vector operators.

    Port of EasySpin's ``private/istotensor.m``.

    Parameters
    ----------
    a, b:
        Each is either a list/tuple of 3 matrices [ax, ay, az] (spin operators)
        or a length-3 array (e.g. magnetic field components).

    Returns
    -------
    T0:
        Rank-0 tensor (scalar or matrix).
    T1:
        Rank-1 tensor, list of 3 elements [T1(+1), T1(0), T1(-1)].
    T2:
        Rank-2 tensor, list of 5 elements [T2(+2), T2(+1), T2(0), T2(-1), T2(-2)].
    """
    if isinstance(a, (list, tuple)):
        ax, ay, az = a[0], a[1], a[2]
    else:
        a = np.asarray(a)
        ax, ay, az = a[0], a[1], a[2]

    if isinstance(b, (list, tuple)):
        bx, by, bz = b[0], b[1], b[2]
    else:
        b = np.asarray(b)
        bx, by, bz = b[0], b[1], b[2]

    # Rank 0: T0 = -(1/sqrt(3)) * (ax*bx + ay*by + az*bz)
    T0 = -(1 / np.sqrt(3)) * (ax @ bx + ay @ by + az @ bz) \
        if isinstance(ax, np.ndarray) and ax.ndim == 2 \
        else -(1 / np.sqrt(3)) * (ax * bx + ay * by + az * bz)

    # Helper for products (handles both matrix and scalar cases)
    def _prod(x, y):
        if isinstance(x, np.ndarray) and x.ndim == 2:
            return x @ y
        return x * y

    # Rank 1
    T1 = [
        -0.5 * (_prod(ax, bz) - _prod(az, bx) + 1j * (_prod(ay, bz) - _prod(az, by))),  # (+1)
        -(1j / np.sqrt(2)) * (_prod(ay, bx) - _prod(ax, by)),  # (0)
        -0.5 * (_prod(ax, bz) - _prod(az, bx) - 1j * (_prod(ay, bz) - _prod(az, by))),  # (-1)
    ]

    # Rank 2
    T2 = [
        0.5 * (_prod(ax, bx) - _prod(ay, by) + 1j * (_prod(ax, by) + _prod(ay, bx))),  # (+2)
        -0.5 * (_prod(ax, bz) + _prod(az, bx) + 1j * (_prod(ay, bz) + _prod(az, by))),  # (+1)
        np.sqrt(2 / 3) * (_prod(az, bz) - 0.5 * (_prod(ax, bx) + _prod(ay, by))),  # (0)
        0.5 * (_prod(ax, bz) + _prod(az, bx) - 1j * (_prod(ay, bz) + _prod(az, by))),  # (-1)
        0.5 * (_prod(ax, bx) - _prod(ay, by) - 1j * (_prod(ax, by) + _prod(ay, bx))),  # (-2)
    ]

    return T0, T1, T2


# ---------------------------------------------------------------------------
# magint — IST decomposition of spin Hamiltonian
# ---------------------------------------------------------------------------

def magint(
    spins: list[float],
    g: np.ndarray,
    CenterField: float,
    A: np.ndarray | None = None,
    D: np.ndarray | None = None,
    gn: list[float] | None = None,
    nuc_spins: list[float] | None = None,
    include_nuc_zeeman: bool = False,
) -> tuple[dict, dict]:
    """Decompose spin Hamiltonian interactions into IST components.

    Port of EasySpin's ``private/magint.m``.

    Parameters
    ----------
    spins:
        List of spin quantum numbers [S1, ..., I1, ...].
    g:
        g-tensor, shape (nElectrons, 3) or (3,) for single electron.
    CenterField:
        Center magnetic field in mT.
    A:
        Hyperfine tensor principal values in MHz, shape (nNuclei, 3) or (3,).
    D:
        ZFS tensor principal values in MHz, shape (nElectrons, 3).
    gn:
        Nuclear g-values.
    nuc_spins:
        Nuclear spin quantum numbers.
    include_nuc_zeeman:
        Whether to include nuclear Zeeman interaction.

    Returns
    -------
    T:
        dict with 'T0' (list of rank-0 ops), 'T2' (list of 5-element lists of rank-2 ops)
    F:
        dict with 'F0' (array of rank-0 spatial), 'F2' (array of rank-2 spatial, shape (nInt, 5))
    """
    g = np.atleast_2d(np.asarray(g, dtype=float))
    nElSpins = g.shape[0]

    # Count interactions
    nNuclei = 0
    if nuc_spins is not None:
        nNuclei = len(nuc_spins)

    # Build spin operators for all spins
    SpinOps = []
    for iSpin in range(len(spins)):
        Sx = sop(spins, [iSpin + 1, 1]).numpy()
        Sy = sop(spins, [iSpin + 1, 2]).numpy()
        Sz = sop(spins, [iSpin + 1, 3]).numpy()
        SpinOps.append([Sx, Sy, Sz])

    # B0 field vector (z-direction)
    B0 = [0.0, 0.0, CenterField / 1e3]  # mT → T

    T0_list = []
    T2_list = []
    F0_list = []
    F2_list = []

    # --- Electron Zeeman: muB * B * g * S / h ---
    for iEl in range(nElSpins):
        g_mat = np.diag(g[iEl, :])
        # IST operators: istotensor(B0, S)
        T0, T1, T2 = istotensor(B0, SpinOps[iEl])
        T0_list.append(T0)
        T2_list.append(T2)
        # Spatial components: tensor_cart2sph(g * bmagn / planck) in Hz
        g_scaled = g_mat * (BMAGN / PLANCK)  # Hz/T
        F0_scalar = _cart2sph_rank0(g_scaled)
        F2_vec = _tensor_cart2sph_rank2(g_scaled)  # 5-element
        F0_list.append(F0_scalar)
        F2_list.append(F2_vec)

    # --- Hyperfine: S * A * I ---
    if A is not None and nNuclei > 0:
        A = np.atleast_2d(np.asarray(A, dtype=float))
        for iEl in range(nElSpins):
            for iNuc in range(nNuclei):
                A_mat = np.diag(A[iNuc, :]) * 1e6  # MHz → Hz
                S_ops = SpinOps[iEl]
                I_ops = SpinOps[nElSpins + iNuc]
                T0, T1, T2 = istotensor(S_ops, I_ops)
                T0_list.append(T0)
                T2_list.append(T2)
                F0_scalar = _cart2sph_rank0(A_mat)
                F2_vec = _tensor_cart2sph_rank2(A_mat)
                F0_list.append(F0_scalar)
                F2_list.append(F2_vec)

    # --- ZFS: S * D * S ---
    if D is not None:
        D = np.atleast_2d(np.asarray(D, dtype=float))
        for iEl in range(nElSpins):
            if spins[iEl] < 1:
                continue
            D_mat = np.diag(D[iEl, :]) * 1e6  # MHz → Hz
            T0, T1, T2 = istotensor(SpinOps[iEl], SpinOps[iEl])
            T0_list.append(T0)
            T2_list.append(T2)
            F0_scalar = _cart2sph_rank0(D_mat)
            F2_vec = _tensor_cart2sph_rank2(D_mat)
            F0_list.append(F0_scalar)
            F2_list.append(F2_vec)

    # --- Nuclear Zeeman ---
    if include_nuc_zeeman and gn is not None:
        for iNuc in range(nNuclei):
            I_ops = SpinOps[nElSpins + iNuc]
            T0, T1, T2 = istotensor(B0, I_ops)
            T0_list.append(T0)
            T2_list.append(T2)
            gn_scaled = -gn[iNuc] * NMAGN / PLANCK  # Hz/T (scalar → isotropic tensor)
            F0_list.append(_cart2sph_rank0(gn_scaled * np.eye(3)))
            F2_list.append(_tensor_cart2sph_rank2(gn_scaled * np.eye(3)))

    F0 = np.array(F0_list, dtype=complex)
    F2 = np.array(F2_list, dtype=complex) if len(F2_list) > 0 else np.zeros((0, 5), dtype=complex)

    return (
        {'T0': T0_list, 'T2': T2_list},
        {'F0': F0, 'F2': F2},
    )


def _tensor_cart2sph_rank2(T: np.ndarray) -> np.ndarray:
    """Convert 3x3 Cartesian tensor to rank-2 spherical tensor components.

    Uses the Mehring convention (same as EasySpin's tensor_cart2sph.m):
    T2(0) = sqrt(2/3) * (Tzz - 0.5*(Txx + Tyy))

    Returns [T2(+2), T2(+1), T2(0), T2(-1), T2(-2)].
    """
    T = np.asarray(T, dtype=complex)
    Txx, Txy, Txz = T[0, 0], T[0, 1], T[0, 2]
    Tyx, Tyy, Tyz = T[1, 0], T[1, 1], T[1, 2]
    Tzx, Tzy, Tzz = T[2, 0], T[2, 1], T[2, 2]

    T2p2 = 0.5 * ((Txx - Tyy) + 1j * (Txy + Tyx))
    T2p1 = -0.5 * ((Txz + Tzx) + 1j * (Tyz + Tzy))
    T20 = np.sqrt(2 / 3) * (Tzz - 0.5 * (Txx + Tyy))
    T2m1 = 0.5 * ((Txz + Tzx) - 1j * (Tyz + Tzy))
    T2m2 = 0.5 * ((Txx - Tyy) - 1j * (Txy + Tyx))

    return np.array([T2p2, T2p1, T20, T2m1, T2m2])


def _cart2sph_rank0(T: np.ndarray) -> complex:
    """Extract rank-0 (isotropic) component from a 3x3 tensor.

    F0 = -(1/sqrt(3)) * Tr(T)
    """
    T = np.asarray(T, dtype=complex)
    return -(1 / np.sqrt(3)) * np.trace(T)


# ---------------------------------------------------------------------------
# wigD — rank-2 Wigner D-matrices from quaternion trajectories
# ---------------------------------------------------------------------------

def wigD(q: np.ndarray) -> np.ndarray:
    """Compute rank-2 Wigner D-matrices from quaternion trajectories.

    Port of the ``wigD`` subfunction in EasySpin's ``cardamom_propagatedm.m``.

    Parameters
    ----------
    q:
        Quaternion trajectory, shape ``(4, nSteps, nTraj)`` or ``(4, nSteps)``.
        Convention: q = [q0, q1, q2, q3] where q0 is scalar part.

    Returns
    -------
    D2:
        Rank-2 Wigner D-matrix trajectory, shape ``(5, 5, nSteps, nTraj)``.
        Indices correspond to m' = +2, +1, 0, -1, -2 (rows)
        and m = +2, +1, 0, -1, -2 (columns).
    """
    if q.ndim == 2:
        q = q[:, :, np.newaxis]

    # Quaternion components → Cayley-Klein parameters
    A = q[0] - 1j * q[3]
    B = -q[2] - 1j * q[1]
    Ast = q[0] + 1j * q[3]
    Bst = -q[2] + 1j * q[1]

    Z = A * Ast - B * Bst

    ASq = A * A
    BSq = B * B
    AstSq = Ast * Ast
    BstSq = Bst * Bst

    nSteps = q.shape[1]
    nTraj = q.shape[2]
    D2 = np.zeros((5, 5, nSteps, nTraj), dtype=complex)

    sqrt6 = np.sqrt(6)

    D2[0, 0] = ASq * ASq
    D2[0, 1] = 2 * A**3 * B
    D2[0, 2] = sqrt6 * ASq * BSq
    D2[0, 3] = 2 * A * B**3
    D2[0, 4] = BSq * BSq

    D2[1, 0] = -2 * A**3 * Bst
    D2[1, 1] = ASq * (2 * Z - 1)
    D2[1, 2] = sqrt6 * A * B * Z
    D2[1, 3] = BSq * (2 * Z + 1)
    D2[1, 4] = 2 * Ast * B**3

    D2[2, 0] = sqrt6 * ASq * BstSq
    D2[2, 1] = -sqrt6 * A * Bst * Z
    D2[2, 2] = 0.5 * (3 * Z**2 - 1)
    D2[2, 3] = sqrt6 * Ast * B * Z
    D2[2, 4] = sqrt6 * AstSq * BSq

    D2[3, 0] = -2 * A * Bst**3
    D2[3, 1] = BstSq * (2 * Z + 1)
    D2[3, 2] = -sqrt6 * Ast * Bst * Z
    D2[3, 3] = AstSq * (2 * Z - 1)
    D2[3, 4] = 2 * Ast**3 * B

    D2[4, 0] = BstSq * BstSq
    D2[4, 1] = -2 * Ast * Bst**3
    D2[4, 2] = sqrt6 * AstSq * BstSq
    D2[4, 3] = -2 * Ast**3 * Bst
    D2[4, 4] = AstSq * AstSq

    return D2
