"""Electron Zeeman Hamiltonian for torchspin.

Port of EasySpin's ``ham_ez.m``.

The electron Zeeman interaction is::

    H_EZ = -(mux*B[0] + muy*B[1] + muz*B[2])

where the magnetic-dipole-moment operators are::

    mui = -sum_k  sum_{k'} (g_ik' * bmagn / planck) * Sk'   [MHz/mT]

and B is the static field vector in mT.

All energies are in **MHz** (matching EasySpin's convention).
"""
from __future__ import annotations

from typing import Optional

import torch

from torchspin.constants import BMAGN, PLANCK
from torchspin.rotations import erot
from torchspin.spinops import sop
from torchspin.spinsystem import SpinSystem


def ham_ez(
    sys: SpinSystem,
    B0: Optional[list | torch.Tensor] = None,
    eSpins: Optional[list[int]] = None,
    dtype: torch.dtype = torch.complex128,
    device: str = 'cpu',
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Electron Zeeman Hamiltonian (or magnetic-moment operators).

    Parameters
    ----------
    sys:
        Spin system.
    B0:
        Magnetic field vector in mT, shape ``(3,)``.  If given, returns the
        Zeeman Hamiltonian matrix.  If ``None``, returns the three magnetic-
        dipole-moment operator components ``(mux, muy, muz)`` in MHz/mT.
    eSpins:
        1-based indices of electron spins to include.  Defaults to all.
    dtype:
        Output dtype.  Default ``torch.complex128``.
    device:
        PyTorch device string.

    Returns
    -------
    H:
        If *B0* is given — Zeeman Hamiltonian matrix (MHz), shape
        ``(nStates, nStates)``.
    (mux, muy, muz):
        If *B0* is ``None`` — three magnetic-dipole-moment operator matrices
        (MHz/mT), each of shape ``(nStates, nStates)``.
    """
    n_states = sys.nStates
    n_electrons = sys.nElectrons
    all_spins = sys.Spins  # electrons first, then nuclei

    if eSpins is None:
        eSpins = list(range(1, n_electrons + 1))  # 1-based

    # Validate
    for idx in eSpins:
        if idx < 1 or idx > n_electrons:
            raise ValueError(
                f"eSpins index {idx} is out of range for {n_electrons} electrons (1-based)."
            )

    # Prefactors: pre[i,:] = -bmagn/planck * g[i,:]  [Hz/T]  * 1e-9  -> [MHz/mT]
    # Shape: (nElectrons, 3) for diagonal g, or (nElectrons*3, 3) for full g
    # We work in float64, cast to dtype at the end
    pre = -BMAGN / PLANCK * sys.g.double() * 1e-9  # (nElectrons,3) or (3*nElectrons,3)

    muxM = torch.zeros(n_states, n_states, dtype=dtype, device=device)
    muyM = torch.zeros(n_states, n_states, dtype=dtype, device=device)
    muzM = torch.zeros(n_states, n_states, dtype=dtype, device=device)

    for i in eSpins:
        i0 = i - 1  # 0-based

        # Get g matrix (3×3) in principal frame
        if sys.fullg:
            g = pre[3 * i0 : 3 * i0 + 3, :].to(device=device)  # (3,3)
        else:
            g = torch.diag(pre[i0, :]).to(device=device)  # (3,3) diagonal

        # Rotate g into molecular frame
        ang = sys.gFrame[i0].tolist()
        if any(a != 0.0 for a in ang):
            R_M2g = erot(ang, device=device)
            R_g2M = R_M2g.T
            g = R_g2M.double() @ g.double() @ R_g2M.T.double()

        # Build mux, muy, muz by summing over Cartesian components k
        for k in range(3):  # k = x(0), y(1), z(2)
            # sop for spin i, component k+1 (1=x,2=y,3=z)
            Sk = sop(all_spins, [[i, k + 1]], dtype=dtype, device=device)
            muxM = muxM + g[0, k].to(dtype) * Sk
            muyM = muyM + g[1, k].to(dtype) * Sk
            muzM = muzM + g[2, k].to(dtype) * Sk

    if B0 is None:
        return muxM, muyM, muzM

    # Compute Zeeman Hamiltonian
    if isinstance(B0, torch.Tensor):
        B = B0.double().tolist()
    else:
        B = [float(b) for b in B0]

    H = -(muxM * B[0] + muyM * B[1] + muzM * B[2])
    return H
