"""Density matrix time-domain evolution.

Port of EasySpin's ``evolve.m``.

Evolves a density matrix under a Hamiltonian and detects a signal at each
time step, implementing common pulse EPR incrementation schemes.

Example
-------
>>> import numpy as np
>>> from torchspin.evolve import evolve
>>> # Simple FID: 2-level system
>>> H = np.diag([0.0, 100.0])  # MHz
>>> Sig = np.array([[0.5, 0.5], [0.5, 0.5]])
>>> Det = np.array([[0, 1], [0, 0]])
>>> signal = evolve(Sig, Det, H, n=128, dt=0.01)
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
from scipy.linalg import eig


def evolve(
    Sig: np.ndarray,
    Det: np.ndarray,
    Ham: np.ndarray | list,
    n: int | list | tuple,
    dt: float | list | tuple,
    IncScheme: list | None = None,
    Mix: list | None = None,
) -> np.ndarray:
    """Time-domain evolution of a density matrix.

    Parameters
    ----------
    Sig:
        Initial density matrix (Hermitian), shape ``(N, N)``.
    Det:
        Detection operator, shape ``(N, N)``.
    Ham:
        Hamiltonian in MHz, shape ``(N, N)``.
        For 2D experiments with different Hamiltonians per dimension,
        pass a list ``[Ham_x, Ham_y]``.
    n:
        Number of time points.  Scalar for 1D, ``[nx, ny]`` for 2D.
    dt:
        Time increment in microseconds.  Scalar for 1D, ``[dtx, dty]``
        for 2D.
    IncScheme:
        Incrementation scheme.  Default ``[1]``.

        Supported 1D schemes: ``[1]``, ``[1, 1]``, ``[1, -1]``

        Supported 2D schemes: ``[1, 2]``, ``[1, 1, 2]``, ``[1, -1, 2]``

    Mix:
        List of mixing propagators (square matrices).  Required when
        ``IncScheme`` has more than one element.  Length must equal
        ``len(IncScheme) - 1``.

    Returns
    -------
    Signal:
        Complex time-domain signal.  Shape ``(n,)`` for 1D or
        ``(nx, ny)`` for 2D.
    """
    if IncScheme is None:
        IncScheme = [1]
    IncScheme = list(IncScheme)
    if Mix is None:
        Mix = []
    if not isinstance(Mix, list):
        Mix = [Mix]

    n_periods = len(IncScheme)
    n_dims = max(abs(s) for s in IncScheme)

    # Validate mixing propagators
    if n_periods > 1 and len(Mix) != n_periods - 1:
        raise ValueError(
            f"IncScheme has {n_periods} periods; need {n_periods - 1} "
            f"mixing propagators, got {len(Mix)}"
        )

    # Parse n, dt
    if isinstance(n, (list, tuple)):
        nx, ny = int(n[0]), int(n[1])
    else:
        nx = int(n)
        ny = 1

    if isinstance(dt, (list, tuple)):
        dtx, dty = float(dt[0]), float(dt[1])
    else:
        dtx = float(dt)
        dty = dtx

    Sig = np.asarray(Sig, dtype=np.complex128)
    Det = np.asarray(Det, dtype=np.complex128)
    N = Sig.shape[0]

    # Diagonalize Hamiltonian(s)
    if isinstance(Ham, list):
        # Two Hamiltonians for 2D
        Ham_x = np.asarray(Ham[0], dtype=np.complex128)
        Ham_y = np.asarray(Ham[1], dtype=np.complex128)
        Ex, Vx = _diagonalize(Ham_x)
        Ey, Vy = _diagonalize(Ham_y)

        d = [abs(s) for s in IncScheme]
        Vecs = {1: Vx, 2: Vy}

        # Transform to eigenbasis
        V_first = Vecs[d[0]]
        Density = V_first.conj().T @ Sig @ V_first
        Detector = Vecs[d[-1]].conj().T @ Det @ Vecs[d[-1]]

        for i, mix in enumerate(Mix):
            mix_np = np.asarray(mix, dtype=np.complex128)
            Mix[i] = Vecs[d[i + 1]].conj().T @ mix_np @ Vecs[d[i]]

        diagUX = np.exp(-2j * np.pi * dtx * Ex)
        diagUY = np.exp(-2j * np.pi * dty * Ey)
    else:
        Ham = np.asarray(Ham, dtype=np.complex128)
        E, V = _diagonalize(Ham)

        # Transform to eigenbasis
        Density = V.conj().T @ Sig @ V
        Detector = V.conj().T @ Det @ V
        for i, mix in enumerate(Mix):
            mix_np = np.asarray(mix, dtype=np.complex128)
            Mix[i] = V.conj().T @ mix_np @ V

        diagU = np.exp(-2j * np.pi * dtx * E)
        if n_dims == 2:
            diagUX = diagU
            diagUY = np.exp(-2j * np.pi * dty * E)

    # Vectorized detector for trace: Tr(Det @ rho) = Det_vec . rho_vec
    Det_vec = Detector.T.ravel()

    # Dispatch to scheme implementation
    if IncScheme == [1]:
        Signal = _scheme_1(Density, Det_vec, diagU, nx, N)

    elif IncScheme == [1, 1]:
        Signal = _scheme_1_1(Density, Det_vec, diagU, Mix[0], nx, N)

    elif IncScheme == [1, -1]:
        Signal = _scheme_1_m1(Density, Det_vec, diagU, Mix[0], nx, N)

    elif IncScheme == [1, 2]:
        Signal = _scheme_1_2(Density, Det_vec, diagUX, diagUY, Mix[0], nx, ny, N)

    elif IncScheme == [1, 1, 2]:
        Signal = _scheme_1_1_2(Density, Det_vec, diagUX, diagUY, Mix, nx, ny, N)

    elif IncScheme == [1, -1, 2]:
        Signal = _scheme_1_m1_2(Density, Det_vec, diagUX, diagUY, Mix, nx, ny, N)

    else:
        raise NotImplementedError(
            f"IncScheme {IncScheme} is not yet implemented. "
            f"Supported: [1], [1,1], [1,-1], [1,2], [1,1,2], [1,-1,2]"
        )

    return Signal


# ---------------------------------------------------------------------------
# Scheme implementations
# ---------------------------------------------------------------------------

def _scheme_1(Density, Det_vec, diagU, n, N):
    """Scheme [1]: simple FID / 3p-ESEEM / echo transient."""
    Signal = np.zeros(n, dtype=np.complex128)
    rho = Density.ravel(order='F')  # column-major for Tr trick
    U_ = (diagU[:, None] * diagU[None, :].conj()).ravel(order='F')
    for ix in range(n):
        Signal[ix] = Det_vec @ rho
        rho = U_ * rho
    return Signal


def _scheme_1_1(Density, Det_vec, diagU, Mix1, n, N):
    """Scheme [1, 1]: 2p-ESEEM, CP, RIDME."""
    Signal = np.zeros(n, dtype=np.complex128)
    UU_ = diagU[:, None] * diagU[None, :].T  # U*Mix*U pattern
    M = Mix1.copy()
    for ix in range(n):
        FD = M @ Density @ M.conj().T
        Signal[ix] = Det_vec @ FD.ravel(order='F')
        M = UU_ * M
    return Signal


def _scheme_1_m1(Density, Det_vec, diagU, Mix1, n, N):
    """Scheme [1, -1]: 3p-DEER, 4p-DEER, PEANUT."""
    Signal = np.zeros(n, dtype=np.complex128)
    # Pre-propagate to end of second period
    MixX = np.diag(diagU ** n) @ Mix1
    UtU_ = diagU.conj()[:, None] * diagU[None, :].T
    for ix in range(n):
        FD = MixX @ Density @ MixX.conj().T
        Signal[ix] = Det_vec @ FD.ravel(order='F')
        MixX = UtU_ * MixX
    return Signal


def _scheme_1_2(Density, Det_vec, diagUX, diagUY, Mix1, nx, ny, N):
    """Scheme [1, 2]: HYSCORE, DONUT-HYSCORE."""
    Signal = np.zeros((nx, ny), dtype=np.complex128)
    UX_ = diagUX[:, None] * diagUX[None, :].conj()
    UY_ = (diagUY[:, None] * diagUY[None, :].conj()).ravel(order='F')
    D = Density.copy()
    for ix in range(nx):
        FD = (Mix1 @ D @ Mix1.conj().T).ravel(order='F')
        for iy in range(ny):
            Signal[ix, iy] = Det_vec @ FD
            FD = UY_ * FD
        D = UX_ * D
    return Signal


def _scheme_1_1_2(Density, Det_vec, diagUX, diagUY, Mix, nx, ny, N):
    """Scheme [1, 1, 2]: 2p-ESEEM with echo transient."""
    Signal = np.zeros((nx, ny), dtype=np.complex128)
    UUX_ = diagUX[:, None] * diagUX[None, :].T
    UY_ = (diagUY[:, None] * diagUY[None, :].conj()).ravel(order='F')
    Mix1 = Mix[0].copy()
    Mix2 = Mix[1]
    for ix in range(nx):
        M = Mix2 @ Mix1
        FD = (M @ Density @ M.conj().T).ravel(order='F')
        for iy in range(ny):
            Signal[ix, iy] = Det_vec @ FD
            FD = UY_ * FD
        Mix1 = UUX_ * Mix1
    return Signal


def _scheme_1_m1_2(Density, Det_vec, diagUX, diagUY, Mix, nx, ny, N):
    """Scheme [1, -1, 2]: 3/4p-DEER with echo transient."""
    Signal = np.zeros((nx, ny), dtype=np.complex128)
    UtUX_ = diagUX.conj()[:, None] * diagUX[None, :].T
    UY_ = (diagUY[:, None] * diagUY[None, :].conj()).ravel(order='F')
    MixX = np.diag(diagUX ** nx) @ Mix[0]
    Mix2 = Mix[1]
    for ix in range(nx):
        M = Mix2 @ MixX
        FD = (M @ Density @ M.conj().T).ravel(order='F')
        for iy in range(ny):
            Signal[ix, iy] = Det_vec @ FD
            FD = UY_ * FD
        MixX = UtUX_ * MixX
    return Signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _diagonalize(H):
    """Diagonalize Hermitian matrix. Returns (eigenvalues, eigenvectors)."""
    # Check if already diagonal
    if np.count_nonzero(H) == np.count_nonzero(np.diag(np.diag(H))):
        return np.real(np.diag(H)), np.eye(H.shape[0], dtype=np.complex128)

    E, V = np.linalg.eigh(H)
    return E, V
