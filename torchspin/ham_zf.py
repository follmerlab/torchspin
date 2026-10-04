"""Zero-field splitting (ZFS) Hamiltonian for torchspin.

Port of EasySpin's ``ham_zf.m``.

The quadratic ZFS term is::

    H_ZF = S · D · S = sum_{i,j} D_{ij} * Si * Sj

where D is the 3×3 ZFS tensor in MHz.

High-order ZFS terms use Stevens operators::

    H_ZF = sum_{k,q} B_k^q * O_k^q

where k = 2, 4, 6, ... and q = -k, ..., k.  If D is present, rank-2
Stevens operators are skipped.

All energies are in **MHz** (matching EasySpin's convention).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from torchspin.rotations import erot
from torchspin.spinops import sop
from torchspin.spinsystem import SpinSystem
from torchspin.stev import stev


def ham_zf(
    sys: SpinSystem,
    idxElectrons: Optional[list[int]] = None,
    dtype: torch.dtype = torch.complex128,
    device: str = 'cpu',
) -> torch.Tensor:
    """Zero-field splitting Hamiltonian (MHz).

    Parameters
    ----------
    sys:
        Spin system.  ZFS is specified via ``sys.D`` and ``sys.DFrame``,
        or via Stevens coefficients ``sys.B`` and ``sys.BFrame``.
        If both D and B are ``None``, a zero matrix is returned.
    idxElectrons:
        1-based indices of electrons for which to compute ZFS.
        Defaults to all electrons.
    dtype:
        Output dtype.  Default ``torch.complex128``.
    device:
        PyTorch device string.

    Returns
    -------
    torch.Tensor
        ZFS Hamiltonian matrix (MHz), shape ``(nStates, nStates)``.
    """
    n_states = sys.nStates
    n_electrons = sys.nElectrons
    all_spins = sys.Spins

    if idxElectrons is None:
        idxElectrons = list(range(1, n_electrons + 1))  # 1-based

    H = torch.zeros(n_states, n_states, dtype=dtype, device=device)

    if sys.D is None and sys.B is None:
        return H

    # D/E tensor (quadratic ZFS)
    if sys.D is not None:
        for i in idxElectrons:
            i0 = i - 1  # 0-based

            # Construct 3×3 D matrix (float64)
            if sys.fullD:
                D = sys.D[3 * i0 : 3 * i0 + 3, :].double().to(device=device)  # (3,3)
            else:
                D = torch.diag(sys.D[i0, :]).double().to(device=device)  # (3,3) diagonal

            if not D.any():
                continue

            # Rotate D into molecular frame
            ang = sys.DFrame[i0].tolist()
            if any(a != 0.0 for a in ang):
                R_M2D = erot(ang, device=device)
                R_D2M = R_M2D.T
                D = R_D2M.double() @ D @ R_D2M.T.double()

            # Build S·D·S  = sum_{c1,c2} D[c1,c2] * Sc1_i * Sc2_i
            # c1, c2 in {1=x, 2=y, 3=z}
            for c1 in range(3):
                Sc1 = sop(all_spins, [[i, c1 + 1]], dtype=dtype, device=device)
                for c2 in range(3):
                    Sc2 = sop(all_spins, [[i, c2 + 1]], dtype=dtype, device=device)
                    H = H + D[c1, c2].to(dtype) * (Sc1 @ Sc2)

    # Stevens ZFS coefficients (high-order terms)
    if sys.B is not None:
        has_D = sys.D is not None
        for idx_k, Bk in enumerate(sys.B):
            if Bk is None:
                continue
            # B[0] = k=2, B[1] = k=4, etc.
            k_actual = 2 * (idx_k + 1)
            
            # Skip rank-2 if D is present
            if k_actual == 2 and has_D:
                continue

            # q values: k, k-1, ..., -k
            q_vals = list(range(k_actual, -k_actual - 1, -1))

            for i in idxElectrons:
                i0 = i - 1  # 0-based
                Bk_M = Bk[i0, :]  # coefficients for this electron

                if not Bk_M.any():  # skip if all zero
                    continue

                # Frame rotation (Sys.BFrame): transform the Stevens
                # coefficients from their eigenframe into the molecular
                # frame via the rank-k Wigner D matrix in the Stevens
                # operator basis (EasySpin ham_zf.m):
                #   DB = C_k * D_k.' / C_k;  Bk_M = Bk * DB
                if sys.BFrame is not None:
                    bf = sys.BFrame[idx_k]
                    tilt = None
                    if bf is not None:
                        bf_t = torch.as_tensor(bf, dtype=torch.float64)
                        if bf_t.ndim == 1:
                            tilt = bf_t if i0 == 0 else None
                        else:
                            tilt = bf_t[i0, :]
                    if tilt is not None and bool(tilt.any()):
                        from torchspin.angmom import wignerd, isto2stev
                        a_, b_, g_ = (float(tilt[0]), float(tilt[1]),
                                      float(tilt[2]))
                        Dk = np.asarray(wignerd(k_actual, a_, b_, g_))
                        Ck = isto2stev(k_actual)
                        DB = Ck @ Dk.T @ np.linalg.inv(Ck)
                        DB.real[np.abs(DB.real) < 1e-14] = 0.0
                        DB.imag[np.abs(DB.imag) < 1e-14] = 0.0
                        Bk_row = Bk_M.detach().cpu().numpy().astype(complex)
                        Bk_M = torch.as_tensor((Bk_row @ DB).real,
                                               dtype=torch.float64)

                for iq, q in enumerate(q_vals):
                    coeff = Bk_M[iq]
                    if coeff == 0:
                        continue
                    O_kq = stev(all_spins, k_actual, q, i, dtype=dtype)
                    H = H + coeff.to(dtype) * O_kq

    # Hermitianise (guards against small imaginary noise on diagonal)
    H = (H + H.conj().T) / 2
    return H


def zfsframes(sys: 'SpinSystem'):
    """Extract zero-field splitting parameters and principal frame orientations.

    Port of EasySpin's ``zfsframes.m``.  Analyses the D-tensor of each
    electron spin and returns the conventional D/E ZFS parameters together
    with the Euler angles that rotate the molecular frame to the D principal
    frame.

    Parameters
    ----------
    sys : SpinSystem
        Spin system with D tensors defined.

    Returns
    -------
    D_vals : list of float
        D parameter (MHz) for each electron.  D = Dzz − (Dxx + Dyy) / 2.
    E_vals : list of float
        E parameter (MHz) for each electron.  E = (Dxx − Dyy) / 2.
    Euler_angles : list of ndarray, shape (3,)
        Euler angles ``[alpha, beta, gamma]`` (radians) rotating the molecular
        frame to the D-tensor principal frame for each electron.
    E0_vals : list of float
        Isotropic ZFS offset (MHz) per electron:
        E0 = (Dxx + Dyy + Dzz) / 3  (should be ~0 for traceless D).

    Notes
    -----
    If D is given as diagonal ``(nElectrons, 3)`` the columns are already
    principal values; Euler angles are taken from ``sys.DFrame``.

    If D is given as full 3×3 matrices ``(3*nElectrons, 3)``, the principal
    values are found by diagonalising each D matrix, and the Euler angles
    are extracted from the eigenvector matrix.

    Examples
    --------
    >>> D_vals, E_vals, Euler, E0 = zfsframes(sys)
    >>> print(f'D={D_vals[0]:.1f} MHz, E={E_vals[0]:.1f} MHz')
    """
    import numpy as np
    from torchspin.rotutils import eulang

    n_e = sys.nElectrons
    D_vals, E_vals, Euler_angles, E0_vals = [], [], [], []

    if sys.D is None:
        # No ZFS: D=E=0 for all electrons
        for _ in range(n_e):
            D_vals.append(0.0)
            E_vals.append(0.0)
            Euler_angles.append(np.zeros(3))
            E0_vals.append(0.0)
        return D_vals, E_vals, Euler_angles, E0_vals

    D_t = sys.D.numpy()

    if sys.fullD:
        # Full 3×3 per electron: diagonalise each block
        for i in range(n_e):
            Di = D_t[3*i:3*i+3, :]
            evals, evecs = np.linalg.eigh(Di)
            # Ensure right-handed eigenvector set
            if np.linalg.det(evecs) < 0:
                evecs[:, 0] *= -1
            # Euler angles from eigenvector matrix (rotation mol→D principal frame)
            ang = eulang(evecs.T)
            # Conventional assignment: |Dzz| ≥ |Dyy| ≥ |Dxx|
            # eigh returns eigenvalues in ascending order; Dzz is the one with
            # max absolute value among the three
            abs_e = np.abs(evals)
            idx_zz = np.argmax(abs_e)
            idx_others = [j for j in range(3) if j != idx_zz]
            Dzz = evals[idx_zz]
            Dxx = evals[idx_others[0]]
            Dyy = evals[idx_others[1]]
            D_vals.append(float(Dzz - (Dxx + Dyy) / 2.0))
            E_vals.append(float((Dxx - Dyy) / 2.0))
            E0_vals.append(float((Dxx + Dyy + Dzz) / 3.0))
            Euler_angles.append(ang)
    else:
        # Diagonal D: (nElectrons, 3) = [Dxx, Dyy, Dzz] per electron
        D2 = D_t.reshape(n_e, 3)
        DFrame_t = sys.DFrame.numpy().reshape(n_e, 3)
        for i in range(n_e):
            Dxx, Dyy, Dzz = D2[i]
            D_vals.append(float(Dzz - (Dxx + Dyy) / 2.0))
            E_vals.append(float((Dxx - Dyy) / 2.0))
            E0_vals.append(float((Dxx + Dyy + Dzz) / 3.0))
            Euler_angles.append(DFrame_t[i])

    return D_vals, E_vals, Euler_angles, E0_vals
