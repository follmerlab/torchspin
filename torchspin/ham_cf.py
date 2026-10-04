"""Crystal-field Hamiltonian for orbital angular momenta.

Port of EasySpin's ``ham_cf.m``.

Constructs the crystal-field Hamiltonian using extended Stevens operator
formalism (ranks k = 1 to 12) for the orbital angular momenta defined in
``Sys.L``.

All energies are in **MHz** (matching EasySpin's convention).

Example
-------
>>> from torchspin import SpinSystem
>>> from torchspin.ham_cf import ham_cf
>>> sys = SpinSystem(S=[0.5], L=[2], CF2=[[0, 0, 100.0, 0, 50.0]])
>>> H = ham_cf(sys)
>>> H.shape
torch.Size([10, 10])
"""
from __future__ import annotations

from typing import Optional

import torch

from torchspin.spinsystem import SpinSystem
from torchspin.stev import stev


def ham_cf(
    sys: SpinSystem,
    idxL: Optional[list[int]] = None,
    dtype: torch.dtype = torch.complex128,
    device: str = 'cpu',
) -> torch.Tensor:
    """Crystal-field Hamiltonian.

    Parameters
    ----------
    sys:
        Spin system with orbital angular momenta (``Sys.L``).
    idxL:
        1-based indices of OAMs to include.  Defaults to all.
    dtype:
        Output dtype.
    device:
        PyTorch device string.

    Returns
    -------
    H:
        Crystal-field Hamiltonian matrix (MHz), shape ``(nStates, nStates)``.
    """
    n_states = sys.nStates
    H = torch.zeros(n_states, n_states, dtype=dtype, device=device)

    if sys.nL == 0:
        return H

    if idxL is None:
        idxL = list(range(1, sys.nL + 1))  # 1-based

    for idx in idxL:
        if idx < 1 or idx > sys.nL:
            raise ValueError(
                f"OAM index {idx} out of range for {sys.nL} OAMs (1-based)."
            )

    all_spins = sys.Spins
    offset = sys.nElectrons + sys.nNuclei  # OAMs start after electrons + nuclei

    for iL_1based in idxL:
        iL_0based = iL_1based - 1

        # Skip if L < 1 (no CF interaction possible)
        if sys.L[iL_0based] < 1:
            continue

        # Loop over all ranks k = 1..12
        for k in range(1, 13):
            CFk = getattr(sys, f'CF{k}', None)
            if CFk is None:
                continue

            # Convert to tensor if needed
            if not isinstance(CFk, torch.Tensor):
                CFk = torch.tensor(CFk, dtype=torch.float64)
            if CFk.ndim == 1:
                CFk = CFk.unsqueeze(0)

            # CFk shape: (nL, 2*k+1), columns correspond to q = k, k-1, ..., -k
            if CFk.shape[1] != 2 * k + 1:
                raise ValueError(
                    f"CF{k} must have {2*k+1} columns per OAM, got {CFk.shape[1]}"
                )

            row = CFk[iL_0based]
            if torch.all(row == 0):
                continue

            # iSpin for stev is 1-based index into Spins list
            iSpin = offset + iL_1based  # 1-based

            for iq in range(2 * k + 1):
                coeff = row[iq].item()
                if coeff == 0:
                    continue
                # q goes from k down to -k
                q = k - iq
                Op = stev(all_spins, k, q, iSpin=iSpin, dtype=dtype)
                H = H + coeff * Op.to(device=device)

    # Hermitianize
    H = (H + H.conj().T) / 2
    return H
