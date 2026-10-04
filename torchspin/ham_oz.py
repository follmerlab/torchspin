"""Orbital Zeeman interaction Hamiltonian.

Port of EasySpin's ``ham_oz.m``.

The orbital Zeeman interaction is::

    H_OZ = -(mux*B[0] + muy*B[1] + muz*B[2])

where the magnetic-dipole-moment operators are::

    mui = -gL * bmagn / planck * Li   [MHz/mT]

and L is the orbital angular momentum operator for OAM index i.

All energies are in **MHz** (matching EasySpin's convention).
"""
from __future__ import annotations

from typing import Optional

import torch

from torchspin.constants import BMAGN, PLANCK
from torchspin.spinops import sop
from torchspin.spinsystem import SpinSystem


def ham_oz(
    sys: SpinSystem,
    B0: Optional[list | torch.Tensor] = None,
    oam: Optional[list[int]] = None,
    dtype: torch.dtype = torch.complex128,
    device: str = 'cpu',
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Orbital Zeeman Hamiltonian (or magnetic-moment operators).

    Parameters
    ----------
    sys:
        Spin system with orbital angular momenta (``Sys.L``).
    B0:
        Magnetic field vector in mT, shape ``(3,)``.  If given, returns the
        orbital Zeeman Hamiltonian.  If ``None``, returns ``(mux, muy, muz)``.
    oam:
        1-based indices of OAMs to include.  Defaults to all.
    dtype:
        Output dtype.
    device:
        PyTorch device string.

    Returns
    -------
    H:
        If *B0* is given — orbital Zeeman Hamiltonian (MHz).
    (mux, muy, muz):
        If *B0* is ``None`` — orbital magnetic-dipole-moment operators (MHz/mT).
    """
    n_states = sys.nStates

    muxM = torch.zeros(n_states, n_states, dtype=dtype, device=device)
    muyM = torch.zeros(n_states, n_states, dtype=dtype, device=device)
    muzM = torch.zeros(n_states, n_states, dtype=dtype, device=device)

    if sys.nL == 0:
        if B0 is None:
            return muxM, muyM, muzM
        return torch.zeros(n_states, n_states, dtype=dtype, device=device)

    if oam is None:
        oam = list(range(1, sys.nL + 1))  # 1-based

    for idx in oam:
        if idx < 1 or idx > sys.nL:
            raise ValueError(
                f"OAM index {idx} out of range for {sys.nL} OAMs (1-based)."
            )

    all_spins = sys.Spins
    offset = sys.nElectrons + sys.nNuclei  # OAMs start after electrons + nuclei

    for i_1based in oam:
        i_0based = i_1based - 1

        # Skip if L == 0
        if sys.L[i_0based] == 0:
            continue

        # Prefactor: -gL * bmagn / planck in Hz/T, then /1e9 -> MHz/mT
        pre = -sys.gL[i_0based] * BMAGN / PLANCK * 1e-9

        # 1-based spin index in Spins list
        iSpin = offset + i_1based

        # Build Lx, Ly, Lz operators
        for k in range(3):  # 0=x, 1=y, 2=z
            Lk = sop(all_spins, [[iSpin, k + 1]], dtype=dtype, device=device)
            scaled = pre * Lk
            if k == 0:
                muxM = muxM + scaled
            elif k == 1:
                muyM = muyM + scaled
            else:
                muzM = muzM + scaled

    if B0 is None:
        return muxM, muyM, muzM

    # Compute orbital Zeeman Hamiltonian
    if isinstance(B0, torch.Tensor):
        B = B0.double().tolist()
    else:
        B = [float(b) for b in B0]

    H = -(muxM * B[0] + muyM * B[1] + muzM * B[2])
    return H
