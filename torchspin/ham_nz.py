"""Nuclear Zeeman interaction Hamiltonian for torchspin.

Port of EasySpin's ``ham_nz.m``.

The nuclear Zeeman interaction is::

    H_NZ = mux*B[0] + muy*B[1] + muz*B[2]   (positive sign — opposite to electron Zeeman)

where the nuclear magnetic-dipole-moment operators are::

    mui = sum_n  pre_n * sum_k sigma_n_ik * In_k   [MHz/mT]

and the prefactor::

    pre_n = +NMAGN/PLANCK * gn_n * gnscale_n   [Hz/T] * 1e-9  ->  [MHz/mT]

All energies are in **MHz** (matching EasySpin's convention).
"""
from __future__ import annotations

from typing import Optional

import torch

from torchspin.constants import NMAGN, PLANCK
from torchspin.rotations import erot
from torchspin.spinops import sop
from torchspin.spinsystem import SpinSystem


def ham_nz(
    sys: SpinSystem,
    B0: Optional[list | torch.Tensor] = None,
    nSpins: Optional[list[int]] = None,
    dtype: torch.dtype = torch.complex128,
    device: str = 'cpu',
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Nuclear Zeeman Hamiltonian (or magnetic-moment operators).

    Parameters
    ----------
    sys:
        Spin system.
    B0:
        Magnetic field vector in mT, shape ``(3,)``.  If given, returns the
        Zeeman Hamiltonian matrix.  If ``None``, returns ``(mux, muy, muz)``.
    nSpins:
        1-based indices of nuclear spins to include.  Defaults to all.
    dtype:
        Output dtype.  Default ``torch.complex128``.
    device:
        PyTorch device string.

    Returns
    -------
    H:
        If *B0* is given — nuclear Zeeman Hamiltonian (MHz), shape
        ``(nStates, nStates)``.
    (mux, muy, muz):
        If *B0* is ``None`` — three magnetic-dipole-moment operator matrices
        (MHz/mT), each of shape ``(nStates, nStates)``.

    Notes
    -----
    The sign convention is **positive** (opposite to electron Zeeman)::

        H = mux*B[0] + muy*B[1] + muz*B[2]

    Prefactor: ``pre = +NMAGN/PLANCK * gn * gnscale  [Hz/T] * 1e-9  →  [MHz/mT]``
    """
    n_states = sys.nStates
    n_electrons = sys.nElectrons
    n_nuclei = sys.nNuclei
    all_spins = sys.Spins

    muxM = torch.zeros(n_states, n_states, dtype=dtype, device=device)
    muyM = torch.zeros(n_states, n_states, dtype=dtype, device=device)
    muzM = torch.zeros(n_states, n_states, dtype=dtype, device=device)

    if n_nuclei == 0:
        if B0 is None:
            return muxM, muyM, muzM
        return muxM  # zero matrix

    if nSpins is None:
        nSpins = list(range(1, n_nuclei + 1))

    for idx in nSpins:
        if idx < 1 or idx > n_nuclei:
            raise ValueError(f"nSpins index {idx} out of range (1..{n_nuclei}).")

    gn_list = sys.gn  # list of nuclear g-factors from isotope table

    for iNuc in nSpins:
        i0 = iNuc - 1  # 0-based

        # Prefactor: +NMAGN/PLANCK * gn * gnscale  [Hz/T] * 1e-9 -> [MHz/mT]
        pre = NMAGN / PLANCK * gn_list[i0] * sys.gnscale[i0].item() * 1e-9  # MHz/mT (scalar)

        # Chemical shielding tensor sigma (3×3) in principal frame
        if sys.fullsigma:
            sigma = sys.sigma[3 * i0 : 3 * i0 + 3, :].double().to(device=device)  # (3,3)
        else:
            sigma = torch.diag(sys.sigma[i0, :]).double().to(device=device)  # (3,3) diagonal

        # Rotate sigma into molecular frame
        ang = sys.sigmaFrame[i0].tolist()
        if any(a != 0.0 for a in ang):
            R_M2CS = erot(ang, device=device)
            R_CS2M = R_M2CS.T
            sigma = R_CS2M.double() @ sigma @ R_CS2M.T.double()

        # Effective operator matrix: pre * sigma (matches MATLAB EasySpin convention)
        # sigma defaults to eye(3) (no shielding); user-supplied sigma < 1 reduces coupling
        eff = pre * sigma  # (3,3)

        # Nuclear spin index in full Spins list (1-based): n_electrons + iNuc
        nSp_full = n_electrons + iNuc

        for k in range(3):  # k = x(0), y(1), z(2)
            Ik = sop(all_spins, [[nSp_full, k + 1]], dtype=dtype, device=device)
            muxM = muxM + eff[0, k].to(dtype) * Ik
            muyM = muyM + eff[1, k].to(dtype) * Ik
            muzM = muzM + eff[2, k].to(dtype) * Ik

    if B0 is None:
        return muxM, muyM, muzM

    if isinstance(B0, torch.Tensor):
        Bvec = B0.double().tolist()
    else:
        Bvec = [float(b) for b in B0]

    # Positive sign for nuclear Zeeman
    H = muxM * Bvec[0] + muyM * Bvec[1] + muzM * Bvec[2]
    return H
