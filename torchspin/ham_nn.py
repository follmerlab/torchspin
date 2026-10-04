"""Nuclear-nuclear spin interaction Hamiltonian for torchspin.

Port of EasySpin's ``ham_nn.m``.

The bilinear nuclear-nuclear coupling for a pair (a, b) is::

    H_NN = I_a · J_{ab} · I_b = sum_{c1,c2} J[c1,c2] * Ia_{c1} * Ib_{c2}

where J is the 3×3 coupling tensor in MHz.

All energies are in **MHz** (matching EasySpin's convention).
"""
from __future__ import annotations

from itertools import combinations
from typing import Optional

import torch

from torchspin.rotations import erot
from torchspin.spinops import sop
from torchspin.spinsystem import SpinSystem


def ham_nn(
    sys: SpinSystem,
    nSpins: Optional[list[int]] = None,
    dtype: torch.dtype = torch.complex128,
    device: str = 'cpu',
) -> torch.Tensor:
    """Nuclear-nuclear coupling Hamiltonian (MHz).

    Parameters
    ----------
    sys:
        Spin system.  Coupling tensors are in ``sys.nn`` (MHz) and frames
        in ``sys.nnFrame``.  If ``sys.nn`` is ``None``, returns zero matrix.
    nSpins:
        1-based indices of nuclear spins to include (at least 2).
        Defaults to all nuclei.
    dtype:
        Output dtype.  Default ``torch.complex128``.
    device:
        PyTorch device string.

    Returns
    -------
    torch.Tensor
        Nuclear-nuclear Hamiltonian (MHz), shape ``(nStates, nStates)``.
    """
    n_states = sys.nStates
    n_electrons = sys.nElectrons
    n_nuclei = sys.nNuclei
    all_spins = sys.Spins

    H = torch.zeros(n_states, n_states, dtype=dtype, device=device)

    if n_nuclei < 2 or sys.nn is None:
        return H

    if nSpins is None:
        nSpins = list(range(1, n_nuclei + 1))

    if len(nSpins) < 2:
        return H

    # Global pair list (1-based nuclear indices, sorted)
    all_nuc_pairs = list(combinations(range(1, n_nuclei + 1), 2))

    # Requested pairs
    req_pairs = sorted(combinations(sorted(nSpins), 2))

    for iNuc_a, iNuc_b in req_pairs:
        pair = (min(iNuc_a, iNuc_b), max(iNuc_a, iNuc_b))
        try:
            iPair = all_nuc_pairs.index(pair)
        except ValueError:
            continue

        # Construct 3×3 J matrix (MHz, float64)
        if sys.fullnn:
            J = sys.nn[3 * iPair : 3 * iPair + 3, :].double().to(device=device)  # (3,3)
        else:
            J = torch.diag(sys.nn[iPair, :]).double().to(device=device)  # (3,3) diagonal

        if not J.any():
            continue

        # Rotate J into molecular frame
        ang = sys.nnFrame[iPair].tolist()
        if any(a != 0.0 for a in ang):
            R_M2nn = erot(ang, device=device)
            R_nn2M = R_M2nn.T
            J = R_nn2M.double() @ J @ R_nn2M.T.double()

        # Full Spins list indices (1-based): n_electrons + iNuc
        nSp_a_full = n_electrons + iNuc_a
        nSp_b_full = n_electrons + iNuc_b

        # Build I_a · J · I_b
        for c1 in range(3):
            Ic1 = sop(all_spins, [[nSp_a_full, c1 + 1]], dtype=dtype, device=device)
            for c2 in range(3):
                Ic2 = sop(all_spins, [[nSp_b_full, c2 + 1]], dtype=dtype, device=device)
                H = H + J[c1, c2].to(dtype) * (Ic1 @ Ic2)

    H = (H + H.conj().T) / 2
    return H
