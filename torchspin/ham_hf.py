"""Hyperfine interaction Hamiltonian for torchspin.

Port of EasySpin's ``ham_hf.m``.

The hyperfine interaction between electron spin *e* and nuclear spin *n* is::

    H_HF = S_e · A_{en} · I_n = sum_{c1,c2} A_{c1,c2} * Se_{c1} * In_{c2}

where A is the 3×3 hyperfine tensor in MHz.

All energies are in **MHz** (matching EasySpin's convention).
"""
from __future__ import annotations

from typing import Optional

import torch

from torchspin.rotations import erot
from torchspin.spinops import sop
from torchspin.spinsystem import SpinSystem


def ham_hf(
    sys: SpinSystem,
    elSpins: Optional[list[int]] = None,
    nucSpins: Optional[list[int]] = None,
    dtype: torch.dtype = torch.complex128,
    device: str = 'cpu',
) -> torch.Tensor:
    """Hyperfine interaction Hamiltonian (MHz).

    Parameters
    ----------
    sys:
        Spin system.  Hyperfine tensors are in ``sys.A`` (MHz) and frames
        in ``sys.AFrame``.
    elSpins:
        1-based indices of electron spins to include.  Defaults to all.
    nucSpins:
        1-based indices of nuclear spins to include.  Defaults to all.
    dtype:
        Output dtype.  Default ``torch.complex128``.
    device:
        PyTorch device string.

    Returns
    -------
    torch.Tensor
        Hyperfine Hamiltonian matrix (MHz), shape ``(nStates, nStates)``.
    """
    n_states = sys.nStates
    n_electrons = sys.nElectrons
    n_nuclei = sys.nNuclei
    all_spins = sys.Spins  # electrons first, then nuclei
    nuc_spins_I = sys.I

    H = torch.zeros(n_states, n_states, dtype=dtype, device=device)

    if n_nuclei == 0:
        return H

    if elSpins is None:
        elSpins = list(range(1, n_electrons + 1))
    if nucSpins is None:
        nucSpins = list(range(1, n_nuclei + 1))

    # Validate
    for idx in elSpins:
        if idx < 1 or idx > n_electrons:
            raise ValueError(f"elSpins index {idx} out of range (1..{n_electrons}).")
    for idx in nucSpins:
        if idx < 1 or idx > n_nuclei:
            raise ValueError(f"nucSpins index {idx} out of range (1..{n_nuclei}).")

    # A shape: (nNuclei, 3*nElectrons) for diagonal,
    #          (3*nNuclei, 3*nElectrons) for full
    A_tensor = sys.A  # may be None (but guarded above by n_nuclei>0)
    if A_tensor is None:
        return H

    for eSp in elSpins:
        e0 = eSp - 1  # 0-based
        eidx = slice(3 * e0, 3 * e0 + 3)  # column slice for this electron

        for nSp in nucSpins:
            n0 = nSp - 1  # 0-based

            # Skip zero-spin nuclei
            if nuc_spins_I[n0] == 0:
                continue

            # Construct 3×3 A matrix in principal frame (float64)
            if sys.fullA:
                A = A_tensor[3 * n0 : 3 * n0 + 3, eidx].double().to(device=device)  # (3,3)
            else:
                A = torch.diag(A_tensor[n0, eidx]).double().to(device=device)  # (3,3) diagonal

            if not A.any():
                continue

            # Rotate A into molecular frame
            # AFrame shape: (nNuclei, 3*nElectrons); angles for pair (nSp, eSp):
            ang = sys.AFrame[n0, eidx].tolist()
            if any(a != 0.0 for a in ang):
                R_M2A = erot(ang, device=device)
                R_A2M = R_M2A.T
                A = R_A2M.double() @ A @ R_A2M.T.double()

            # Nuclear spin index in the full Spins list (1-based): nElectrons + nSp
            nSp_full = n_electrons + nSp

            # Build S_e · A · I_n
            for c1 in range(3):
                Sc1 = sop(all_spins, [[eSp, c1 + 1]], dtype=dtype, device=device)
                for c2 in range(3):
                    Ic2 = sop(all_spins, [[nSp_full, c2 + 1]], dtype=dtype, device=device)
                    H = H + A[c1, c2].to(dtype) * (Sc1 @ Ic2)

    # Hermitianise
    H = (H + H.conj().T) / 2
    return H
