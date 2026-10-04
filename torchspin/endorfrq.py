"""ENDOR frequency computation for torchspin.

Port of EasySpin's ``endorfrq.m``.

At a fixed EPR resonance field ``B_field`` and orientation ``(phi, theta)``,
computes the frequencies and intensities of nuclear spin transitions
(ENDOR lines).

Algorithm
---------
1. Build muzL (total moment along field) and snmuzL, snmuxL, snmuyL
   (nuclear-only moments along and perpendicular to field).
2. H(B) = H0 - B_field * muzL  →  diagonalize → E[nStates], V[nStates, nStates].
3. All transition pairs (i, j) with i < j and |E_j - E_i| in ``freq_range``.
4. ENDOR intensity (chi-averaged perpendicular nuclear excitation):
      I_{ij} = (|⟨i|snmux_perp|j⟩|² + |⟨i|snmuy_perp|j⟩|²) / 2
5. Filter by intensity threshold relative to max.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from torchspin.ham import ham
from torchspin.ham_nz import ham_nz
from torchspin.spinsystem import SpinSystem


def endorfrq(
    sys: SpinSystem,
    B_field: float,
    phi: float = 0.0,
    theta: float = 0.0,
    freq_range: Optional[tuple[float, float]] = None,
    threshold: float = 1e-4,
    mwFreq: Optional[float] = None,
    # Optional pre-built operators — avoids rebuilding ham() in powder loops.
    # Pass tensors already moved to the target device for GPU acceleration.
    _H0: Optional[torch.Tensor] = None,
    _mux: Optional[torch.Tensor] = None,
    _muy: Optional[torch.Tensor] = None,
    _muz: Optional[torch.Tensor] = None,
    _snmux: Optional[torch.Tensor] = None,
    _snmuy: Optional[torch.Tensor] = None,
    _snmuz: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute ENDOR transition frequencies at a fixed EPR field.

    Parameters
    ----------
    sys:
        Spin system.  Must contain at least one nucleus.
    B_field:
        Fixed EPR field in mT.
    phi, theta:
        Field orientation in radians.
        ``n = (sin(θ)cos(φ), sin(θ)sin(φ), cos(θ))``.
    freq_range:
        ``(f_lo, f_hi)`` in MHz.  Only transitions within this range are
        returned.  ``None`` = no frequency filter.
    threshold:
        Relative intensity cutoff.  Transitions with intensity below
        ``threshold * max_intensity`` are discarded.  Default ``1e-4``.
    mwFreq:
        Microwave frequency in GHz.  Currently unused (reserved for
        orientation-selection weighting).

    Returns
    -------
    freqs:
        Nuclear transition frequencies in MHz, shape ``(nTrans,)``.
    intensities:
        Transition intensities (a.u.), shape ``(nTrans,)``.
    transitions:
        Level-pair indices ``[i, j]`` (0-based), shape ``(nTrans, 2)``.

    Raises
    ------
    ValueError
        If the spin system contains no nuclei.
    """
    if sys.nNuclei == 0:
        raise ValueError("endorfrq: spin system must contain at least one nucleus.")

    # ── Step 1: Build Hamiltonians and moment operators ────────────────────
    # Use pre-built operators when provided (avoids redundant ham() calls in loops).
    if _H0 is not None:
        H0, mux, muy, muz = _H0, _mux, _muy, _muz
        snmux, snmuy, snmuz = _snmux, _snmuy, _snmuz
    else:
        H0, mux, muy, muz = ham(sys, B0=None)
        snmux, snmuy, snmuz = ham_nz(sys, B0=None)

    # ── Step 2: Orientation geometry ──────────────────────────────────────
    sin_th = math.sin(theta)
    cos_th = math.cos(theta)
    sin_ph = math.sin(phi)
    cos_ph = math.cos(phi)

    # Lab-z unit vector (along field): nz = (nx, ny, nz) components
    nz_x = sin_th * cos_ph
    nz_y = sin_th * sin_ph
    nz_z = cos_th

    # Lab-x unit vector (perp to field in xz-plane):
    # nx = (cos(theta)*cos(phi), cos(theta)*sin(phi), -sin(theta))
    nx_x = cos_th * cos_ph
    nx_y = cos_th * sin_ph
    nx_z = -sin_th

    # Lab-y unit vector (perp to field, azimuthal):
    # ny = (-sin(phi), cos(phi), 0)
    ny_x = -sin_ph
    ny_y =  cos_ph
    ny_z =  0.0

    # Total field-direction moment (for Hamiltonian)
    muzL = nz_x * mux + nz_y * muy + nz_z * muz

    # Nuclear moment perpendicular to field
    snmux_perp = nx_x * snmux + nx_y * snmuy + nx_z * snmuz  # lab-x component
    snmuy_perp = ny_x * snmux + ny_y * snmuy + ny_z * snmuz  # lab-y component

    # ── Step 3: Diagonalize H at B_field ──────────────────────────────────
    H_B = H0 - float(B_field) * muzL
    E, V = torch.linalg.eigh(H_B)   # eigenvalues ascending, columns = eigvecs
    nStates = E.shape[0]

    # ── Step 4: All transition pairs (i, j) with j > i — vectorized ──────
    # Build upper-triangular index arrays once.
    pairs_i_all = torch.tensor(
        [i for i in range(nStates) for j in range(i + 1, nStates)],
        dtype=torch.long,
    )
    pairs_j_all = torch.tensor(
        [j for i in range(nStates) for j in range(i + 1, nStates)],
        dtype=torch.long,
    )

    # All transition frequencies (E is ascending from eigh, so E[j]≥E[i])
    freqs_all = (E[pairs_j_all] - E[pairs_i_all]).abs()   # (nPairs,)

    # Frequency range filter — before expensive matrix elements
    if freq_range is not None:
        mask_f = (freqs_all >= freq_range[0]) & (freqs_all <= freq_range[1])
        if not mask_f.any():
            empty_f = torch.zeros(0, dtype=torch.float64)
            empty_i = torch.zeros(0, dtype=torch.float64)
            empty_t = torch.zeros((0, 2), dtype=torch.long)
            return empty_f, empty_i, empty_t
        pairs_i_all = pairs_i_all[mask_f]
        pairs_j_all = pairs_j_all[mask_f]
        freqs_all   = freqs_all[mask_f]

    # Matrix elements for all surviving pairs simultaneously:
    #   me_x[k] = <ψ_i[k] | snmux_perp | ψ_j[k]>
    #   V[:,i_arr] → (dim, nPairs);  snmux_perp @ V[:,j_arr] → (dim, nPairs)
    op_x = snmux_perp @ V[:, pairs_j_all]        # (dim, nPairs)
    me_x = (V[:, pairs_i_all].conj() * op_x).sum(dim=0)   # (nPairs,)
    op_y = snmuy_perp @ V[:, pairs_j_all]        # (dim, nPairs)
    me_y = (V[:, pairs_i_all].conj() * op_y).sum(dim=0)   # (nPairs,)

    intens_t  = ((me_x.abs() ** 2 + me_y.abs() ** 2) / 2.0).real
    freqs_t   = freqs_all
    pairs_i_t = pairs_i_all
    pairs_j_t = pairs_j_all

    if freqs_t.numel() == 0:
        empty_f = torch.zeros(0, dtype=torch.float64)
        empty_i = torch.zeros(0, dtype=torch.float64)
        empty_t = torch.zeros((0, 2), dtype=torch.long)
        return empty_f, empty_i, empty_t

    # ── Step 5: Intensity threshold ────────────────────────────────────────
    if threshold > 0.0 and intens_t.numel() > 0:
        max_i = intens_t.max().item()
        if max_i > 0:
            keep = intens_t >= threshold * max_i
            freqs_t   = freqs_t[keep]
            intens_t  = intens_t[keep]
            pairs_i_t = pairs_i_t[keep]
            pairs_j_t = pairs_j_t[keep]

    if freqs_t.numel() == 0:
        empty_f = torch.zeros(0, dtype=torch.float64)
        empty_i = torch.zeros(0, dtype=torch.float64)
        empty_t = torch.zeros((0, 2), dtype=torch.long)
        return empty_f, empty_i, empty_t

    # Sort by frequency (ascending)
    order = torch.argsort(freqs_t)
    freqs_t   = freqs_t[order]
    intens_t  = intens_t[order]
    transitions = torch.stack([pairs_i_t[order], pairs_j_t[order]], dim=1)

    return freqs_t, intens_t, transitions
