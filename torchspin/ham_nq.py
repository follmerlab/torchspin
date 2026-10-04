"""Nuclear quadrupole interaction Hamiltonian for torchspin.

Port of EasySpin's ``ham_nq.m``.

The nuclear quadrupole interaction for nucleus *n* is::

    H_NQ = I_n · Q_n · I_n = sum_{c1,c2} Q[c1,c2] * In_{c1} * In_{c2}

where Q is the 3×3 quadrupole coupling tensor in MHz.

Only nuclei with I ≥ 1 contribute (spin-1/2 nuclei have no quadrupole moment).

All energies are in **MHz** (matching EasySpin's convention).
"""
from __future__ import annotations

from typing import Optional

import torch

from torchspin.rotations import erot
from torchspin.spinops import sop
from torchspin.spinsystem import SpinSystem


def ham_nq(
    sys: SpinSystem,
    nSpins: Optional[list[int]] = None,
    dtype: torch.dtype = torch.complex128,
    device: str = 'cpu',
) -> torch.Tensor:
    """Nuclear quadrupole Hamiltonian (MHz).

    Parameters
    ----------
    sys:
        Spin system.  Quadrupole tensors are in ``sys.Q`` (MHz) and frames
        in ``sys.QFrame``.  If ``sys.Q`` is ``None``, returns zero matrix.
    nSpins:
        1-based indices of nuclear spins to include.  Defaults to all.
    dtype:
        Output dtype.  Default ``torch.complex128``.
    device:
        PyTorch device string.

    Returns
    -------
    torch.Tensor
        Nuclear quadrupole Hamiltonian (MHz), shape ``(nStates, nStates)``.
    """
    n_states = sys.nStates
    n_electrons = sys.nElectrons
    n_nuclei = sys.nNuclei
    all_spins = sys.Spins
    nuc_spins_I = sys.I

    H = torch.zeros(n_states, n_states, dtype=dtype, device=device)

    if n_nuclei == 0 or sys.Q is None:
        return H

    if nSpins is None:
        nSpins = list(range(1, n_nuclei + 1))

    for iNuc in nSpins:
        i0 = iNuc - 1  # 0-based

        # Skip spin-1/2 nuclei (no quadrupole interaction)
        if nuc_spins_I[i0] < 1:
            continue

        # Construct 3×3 Q matrix in principal frame (MHz, float64)
        if sys.fullQ:
            Q = sys.Q[3 * i0 : 3 * i0 + 3, :].double().to(device=device)  # (3,3)
        else:
            Q = torch.diag(sys.Q[i0, :]).double().to(device=device)  # (3,3) diagonal

        if not Q.any():
            continue

        # Rotate Q into molecular frame
        ang = sys.QFrame[i0].tolist()
        if any(a != 0.0 for a in ang):
            R_M2Q = erot(ang, device=device)
            R_Q2M = R_M2Q.T
            Q = R_Q2M.double() @ Q @ R_Q2M.T.double()

        # Nuclear spin index in full Spins list (1-based)
        nSp_full = n_electrons + iNuc

        # Build I · Q · I  = sum_{c1,c2} Q[c1,c2] * I_{c1} * I_{c2}
        for c1 in range(3):
            Ic1 = sop(all_spins, [[nSp_full, c1 + 1]], dtype=dtype, device=device)
            for c2 in range(3):
                Ic2 = sop(all_spins, [[nSp_full, c2 + 1]], dtype=dtype, device=device)
                H = H + Q[c1, c2].to(dtype) * (Ic1 @ Ic2)

    # Hermitianise
    H = (H + H.conj().T) / 2
    return H
