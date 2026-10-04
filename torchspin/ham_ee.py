"""Electron-electron interaction Hamiltonian for torchspin.

Port of EasySpin's ``ham_ee.m``.

The bilinear electron-electron interaction for a pair (a, b) is::

    H_ee = S_a · J · S_b = sum_{c1,c2} J_{c1,c2} * Sa_{c1} * Sb_{c2}

where J is the 3×3 coupling tensor in MHz.

An optional biquadratic exchange term can also be included::

    H_ee2 = ee2 * (S_a · S_b)^2

All energies are in **MHz** (matching EasySpin's convention).
"""
from __future__ import annotations

from itertools import combinations
from typing import Optional

import torch

from torchspin.rotations import erot
from torchspin.spinops import sop
from torchspin.spinsystem import SpinSystem


def ham_ee(
    sys: SpinSystem,
    eSpins: Optional[list[int]] = None,
    dtype: torch.dtype = torch.complex128,
    device: str = 'cpu',
) -> torch.Tensor:
    """Electron-electron interaction Hamiltonian (MHz).

    Parameters
    ----------
    sys:
        Spin system.  Coupling tensors are in ``sys.ee`` (MHz) and frames
        in ``sys.eeFrame``.
    eSpins:
        1-based indices of electron spins to include.  At least 2 must be
        given.  Defaults to all electrons.
    dtype:
        Output dtype.  Default ``torch.complex128``.
    device:
        PyTorch device string.

    Returns
    -------
    torch.Tensor
        Electron-electron Hamiltonian matrix (MHz), shape
        ``(nStates, nStates)``.
    """
    n_states = sys.nStates
    n_electrons = sys.nElectrons
    all_spins = sys.Spins

    H = torch.zeros(n_states, n_states, dtype=dtype, device=device)

    if n_electrons < 2:
        return H
    has_ee2 = getattr(sys, 'ee2', None) is not None and bool(sys.ee2.any())
    if (sys.ee is None or not sys.ee.any()) and not has_ee2:
        return H

    if eSpins is None:
        eSpins = list(range(1, n_electrons + 1))

    if len(eSpins) < 2:
        raise ValueError("eSpins must contain at least 2 electron indices.")

    # Build ordered list of all pairs in the full system (for global indexing)
    # Pairs are in lexicographic order matching EasySpin
    all_pairs = list(combinations(range(1, n_electrons + 1), 2))  # 1-based

    # Requested pairs (sorted)
    req_pairs = sorted(combinations(sorted(eSpins), 2))

    for eSp1, eSp2 in req_pairs:
        # Find the global index of this pair
        pair = (min(eSp1, eSp2), max(eSp1, eSp2))
        try:
            iPair = all_pairs.index(pair)
        except ValueError:
            continue

        # Construct 3×3 J matrix (float64)
        if sys.ee is None:
            J = torch.zeros(3, 3, dtype=torch.float64, device=device)
        elif sys.fullee:
            J = sys.ee[3 * iPair : 3 * iPair + 3, :].double().to(device=device)  # (3,3)
        else:
            J = torch.diag(sys.ee[iPair, :]).double().to(device=device)  # (3,3) diagonal

        if not J.any():
            # Check for biquadratic term below even if J is zero
            pass

        # Rotate J into molecular frame
        ang = sys.eeFrame[iPair].tolist()
        if any(a != 0.0 for a in ang):
            R_M2ee = erot(ang, device=device)
            R_ee2M = R_M2ee.T
            J = R_ee2M.double() @ J @ R_ee2M.T.double()

        if J.any():
            # Bilinear term: S_a · J · S_b
            for c1 in range(3):
                Sc1 = sop(all_spins, [[eSp1, c1 + 1]], dtype=dtype, device=device)
                for c2 in range(3):
                    Sc2 = sop(all_spins, [[eSp2, c2 + 1]], dtype=dtype, device=device)
                    H = H + J[c1, c2].to(dtype) * (Sc1 @ Sc2)
        # Isotropic biquadratic exchange +ee2·(S1·S2)² (EasySpin ham_ee.m)
        if has_ee2 and float(sys.ee2[iPair]) != 0.0:
            S1S2 = torch.zeros(n_states, n_states, dtype=dtype, device=device)
            for c in range(3):
                S1S2 = S1S2 + sop(all_spins, [[eSp1, c + 1]], dtype=dtype, device=device) @ \
                    sop(all_spins, [[eSp2, c + 1]], dtype=dtype, device=device)
            H = H + float(sys.ee2[iPair]) * (S1S2 @ S1S2)

    H = (H + H.conj().T) / 2
    return H
