"""Spin-orbit coupling Hamiltonian.

Port of EasySpin's ``ham_so.m``.

The spin-orbit coupling Hamiltonian is::

    H_SO = sum_k  soc(k,1) * (S·L) + soc(k,2) * (S·L)^2 + soc(k,3) * (S·L)^3

where S·L = Sx*Lx + Sy*Ly + Sz*Lz, and the sum runs over electron spins
paired with their corresponding orbital angular momenta.

All energies are in **MHz** (matching EasySpin's convention).

Example
-------
>>> from torchspin import SpinSystem
>>> from torchspin.ham_so import ham_so
>>> sys = SpinSystem(S=[1.0], L=[1], soc=[[200.0, 0, 0]])
>>> H = ham_so(sys)
>>> H.shape
torch.Size([9, 9])
"""
from __future__ import annotations

from typing import Optional

import torch

from torchspin.spinops import sop
from torchspin.spinsystem import SpinSystem


def ham_so(
    sys: SpinSystem,
    eSpins: Optional[list[int]] = None,
    dtype: torch.dtype = torch.complex128,
    device: str = 'cpu',
) -> torch.Tensor:
    """Spin-orbit coupling Hamiltonian.

    Parameters
    ----------
    sys:
        Spin system with orbital angular momenta (``Sys.L``) and spin-orbit
        coupling constants (``Sys.soc``).
    eSpins:
        1-based indices of electron spins to include.  Defaults to all.
    dtype:
        Output dtype.
    device:
        PyTorch device string.

    Returns
    -------
    H:
        Spin-orbit coupling Hamiltonian matrix (MHz), shape ``(nStates, nStates)``.
    """
    n_states = sys.nStates
    H = torch.zeros(n_states, n_states, dtype=dtype, device=device)

    # Early exit: no OAMs or no soc parameters
    if sys.nL == 0 or sys.soc is None:
        return H

    if torch.all(sys.soc == 0):
        return H

    n_electrons = sys.nElectrons
    if eSpins is None:
        eSpins = list(range(1, n_electrons + 1))  # 1-based

    for idx in eSpins:
        if idx < 1 or idx > n_electrons:
            raise ValueError(
                f"eSpins index {idx} out of range for {n_electrons} electrons (1-based)."
            )

    all_spins = sys.Spins
    offset = sys.nElectrons + sys.nNuclei  # OAMs start after electrons + nuclei

    for iS_1based in eSpins:
        iS_0based = iS_1based - 1

        # Get soc coefficients for this electron
        soc_row = sys.soc[iS_0based]
        if torch.all(soc_row == 0):
            continue

        # Skip if corresponding OAM is zero
        if iS_0based >= sys.nL or sys.L[iS_0based] == 0:
            continue

        # OAM spin index (1-based) — one OAM per electron, same ordering
        iL_1based = offset + iS_1based

        # Build S·L = Sx*Lx + Sy*Ly + Sz*Lz
        SL = torch.zeros(n_states, n_states, dtype=dtype, device=device)
        for c in range(1, 4):  # 1=x, 2=y, 3=z
            SL = SL + sop(all_spins, [[iS_1based, c], [iL_1based, c]],
                          dtype=dtype, device=device)

        # Add SOC terms: soc(order) * (S·L)^order
        n_orders = soc_row.shape[0]
        SL_power = SL  # (S·L)^1
        for order in range(n_orders):
            coeff = soc_row[order].item()
            if coeff == 0:
                if order < n_orders - 1:
                    SL_power = SL_power @ SL
                continue
            if order == 0:
                H = H + coeff * SL_power
            else:
                H = H + coeff * SL_power
            if order < n_orders - 1:
                SL_power = SL_power @ SL

    return H
