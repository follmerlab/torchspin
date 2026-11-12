"""ENDOR frequency computation for torchspin.

Port of EasySpin's ``endorfrq.m``.

At a fixed EPR resonance field ``B_field`` and orientation ``(phi, theta)``,
computes the frequencies and intensities of nuclear spin transitions
(ENDOR lines).

Algorithm
---------
1. Build H0, mux, muy, muz (``ham``) and the nuclear moments snmux, snmuy,
   snmuz (``ham_nz``).
2. Select level pairs as EasySpin does (endorfrq.m:258-341), independent of
   the orientation: on a coarse grid ``sphgrid(TPSGridSymm, TPSGridSize)``
   (default ``'D2h'``, 4) sum the nuclear-moment rates with the grid weights,
   zero pairs whose frequency never enters ``freq_range`` on that grid, drop
   electronic pairs (levels in different blocks of ``nStates / prod(2S+1)``
   energy-sorted levels) and keep a pair if its rate is
   ``> threshold * max``.  ``Options.Transitions`` replaces the selection.
3. H(B) = H0 - B_field * muzL  →  diagonalize → E[nStates], V[nStates, nStates].
4. Positions ``E_v - E_u`` and intensities (chi-averaged perpendicular
   excitation) ``(|<u|mu_x|v>|^2 + |<u|mu_y|v>|^2) / 2`` of the selected
   pairs, with the nuclear moment (or the total moment with
   ``Options.Enhancement``).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import torch

from torchspin.ham import ham
from torchspin.ham_nz import ham_nz
from torchspin.spinsystem import SpinSystem

if TYPE_CHECKING:
    from torchspin.experiment import Options


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
    opt: Optional["Options"] = None,
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
        ``(f_lo, f_hi)`` in MHz, used by the transition selection only (as
        EasySpin's ``Exp.Range``): a level pair whose frequency is outside the
        range at every orientation of the selection grid is dropped.  The
        selected pairs are returned at ``(phi, theta)`` even where their
        frequency is outside the range (0.9.14; before, the range filtered the
        lines at this orientation).  ``None`` = no range.
    threshold:
        Relative selection threshold (EasySpin ``Opt.Threshold``).  A level
        pair is kept if its nuclear-moment rate summed over the selection grid
        is ``> threshold * max`` (0.9.14; before, intensities at this
        the per-orientation intensities ``>= threshold * max``).  Default
        ``1e-4`` (EasySpin's endorfrq default is 1e-3; ``salt`` passes
        ``Options.Threshold``, 1e-4).
    mwFreq:
        Microwave frequency in GHz.  Currently unused (reserved for
        orientation-selection weighting).
    opt:
        Optional :class:`~torchspin.experiment.Options`.  Reads
        ``Enhancement``, ``Transitions``, ``TPSGridSymm`` and ``TPSGridSize``
        (``threshold`` stays the explicit argument).  With
        ``Enhancement=True`` the intensities use the total magnetic moment
        (electron + nuclear, ``ham()``'s ``mux/muy/muz``) perpendicular to the
        field instead of the nuclear moment, i.e. they include hyperfine
        enhancement (EasySpin ``Opt.Enhancement='on'``); the selection is
        unchanged.  ``Transitions`` (``'all'`` or 1-based level pairs) gives
        the level pairs directly and skips the selection, as EasySpin's
        ``Opt.Transitions``.  ``TPSGridSymm``/``TPSGridSize`` set the
        selection grid (EasySpin's undocumented options of the same names,
        defaults ``'D2h'`` and 4).  ``None`` (default) = all defaults.

    Electronic level pairs are always excluded from the automatic selection,
    as in EasySpin (0.9.14; before, only with ``Enhancement``).  Levels are
    grouped by energy into blocks of ``nStates / prod(2S+1)``, valid when the
    electron Zeeman/zero-field splitting exceeds the nuclear splittings.

    Returns
    -------
    freqs:
        Nuclear transition frequencies in MHz, shape ``(nTrans,)``.
    intensities:
        Transition intensities (a.u.), shape ``(nTrans,)``.
    transitions:
        Level pairs ``[u, v]``, shape ``(nTrans, 2)``: 1-based level numbers
        in ascending energy at ``B_field``, ``u < v``, as EasySpin's
        ``Transitions`` output and ``Options.Transitions`` (0.9.2; before,
        0-based: subtract 1 to index the eigenvalues/eigenvectors).  Rows
        follow ``freqs`` (ascending frequency); EasySpin orders them by
        descending selection rate.

    Raises
    ------
    ValueError
        If the spin system contains no nuclei.
    """
    if sys.nNuclei == 0:
        raise ValueError("endorfrq: spin system must contain at least one nucleus.")
    enhancement = bool(getattr(opt, 'Enhancement', False)) if opt is not None else False

    # ── Step 1: Build Hamiltonians and moment operators ────────────────────
    # Use pre-built operators when provided (avoids redundant ham() calls in loops).
    if _H0 is not None:
        H0, mux, muy, muz = _H0, _mux, _muy, _muz
        snmux, snmuy, snmuz = _snmux, _snmuy, _snmuz
    else:
        H0, mux, muy, muz = ham(sys, B0=None)
        snmux, snmuy, snmuz = ham_nz(sys, B0=None)
    nStates = H0.shape[0]

    # ── Step 2: Level pairs (EasySpin endorfrq.m:258-352) ─────────────────
    # Options.Transitions given: exactly those pairs, no selection.  Otherwise
    # the automatic selection on the coarse TPS grid at B_field, which does not
    # depend on (phi, theta).
    user = getattr(opt, 'Transitions', None) if opt is not None else None
    if user is not None:
        from torchspin.pepper import _user_transitions
        pairs, _ = _user_transitions(user, nStates)
        pairs_i_t = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=H0.device)
        pairs_j_t = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=H0.device)
    else:
        keep = _endor_tps_select(
            (H0, mux, muy, muz, snmux, snmuy, snmuz), float(B_field),
            freq_range=freq_range, threshold=float(threshold),
            n_electron_states=_n_electron_states(sys),
            grid_symm=_tps_option(opt, 'TPSGridSymm'), grid_size=_tps_option(opt, 'TPSGridSize'))
        pairs_i_t, pairs_j_t = torch.nonzero(keep, as_tuple=True)
        pairs_i_t, pairs_j_t = pairs_i_t.to(H0.device), pairs_j_t.to(H0.device)
    if pairs_i_t.numel() == 0:
        return (torch.zeros(0, dtype=torch.float64), torch.zeros(0, dtype=torch.float64),
                torch.zeros((0, 2), dtype=torch.long))

    # ── Step 3: Orientation geometry ──────────────────────────────────────
    sin_th = math.sin(theta)
    cos_th = math.cos(theta)
    sin_ph = math.sin(phi)
    cos_ph = math.cos(phi)

    # Lab-z unit vector (along field), lab-x (perp, in the xz-plane:
    # (cos(theta)cos(phi), cos(theta)sin(phi), -sin(theta))), lab-y
    # (perp, azimuthal: (-sin(phi), cos(phi), 0)).
    nz_x, nz_y, nz_z = sin_th * cos_ph, sin_th * sin_ph, cos_th
    nx_x, nx_y, nx_z = cos_th * cos_ph, cos_th * sin_ph, -sin_th
    ny_x, ny_y = -sin_ph, cos_ph

    # Total field-direction moment (for Hamiltonian)
    muzL = nz_x * mux + nz_y * muy + nz_z * muz
    if enhancement:
        # Total moment perpendicular to the field (hyperfine enhancement)
        op_x = nx_x * mux + nx_y * muy + nx_z * muz
        op_y = ny_x * mux + ny_y * muy
    else:
        # Nuclear moment perpendicular to the field
        op_x = nx_x * snmux + nx_y * snmuy + nx_z * snmuz
        op_y = ny_x * snmux + ny_y * snmuy

    # ── Step 4: Diagonalize H at B_field ──────────────────────────────────
    H_B = H0 - float(B_field) * muzL
    E, V = torch.linalg.eigh(H_B)   # eigenvalues ascending, columns = eigvecs

    # ── Step 5: Positions and intensities of the selected pairs ───────────
    # (EasySpin returns them at every orientation, inside the range or not.)
    freqs_t = (E[pairs_j_t] - E[pairs_i_t]).abs()
    me_x = (V[:, pairs_i_t].conj() * (op_x @ V[:, pairs_j_t])).sum(dim=0)
    me_y = (V[:, pairs_i_t].conj() * (op_y @ V[:, pairs_j_t])).sum(dim=0)
    intens_t = ((me_x.abs() ** 2 + me_y.abs() ** 2) / 2.0).real

    # Sort by frequency (ascending)
    order = torch.argsort(freqs_t.detach())
    freqs_t = freqs_t[order]
    intens_t = intens_t[order]
    # 1-based level numbers, as EasySpin and Options.Transitions
    transitions = torch.stack([pairs_i_t[order], pairs_j_t[order]], dim=1).cpu() + 1

    return freqs_t, intens_t, transitions


# ---------------------------------------------------------------------------
# Automatic transition selection (EasySpin endorfrq.m:258-341), shared by
# endorfrq and every salt matrix path.
# ---------------------------------------------------------------------------

TPS_GRID_SYMM = 'D2h'   # EasySpin Opt.TPSGridSymm default (endorfrq.m:146)
TPS_GRID_SIZE = 4       # EasySpin Opt.TPSGridSize default (endorfrq.m:145)


def _tps_option(opt, name: str):
    """``Options.TPSGridSymm`` / ``TPSGridSize``, EasySpin's defaults without opt."""
    default = TPS_GRID_SYMM if name == 'TPSGridSymm' else TPS_GRID_SIZE
    if opt is None:
        return default
    val = getattr(opt, name, None)
    return default if val is None else val


def _n_electron_states(sys: SpinSystem) -> int:
    """``prod(2S+1)``, EasySpin's ``nSStates`` (endorfrq.m:324)."""
    return int(round(math.prod(2.0 * float(s) + 1.0 for s in sys.S)))


def _tps_rates(ops: tuple, field_mT: float, grid_symm: str = TPS_GRID_SYMM,
               grid_size: int = TPS_GRID_SIZE):
    """Weighted sum of nuclear-moment transition rates over the TPS grid.

    EasySpin endorfrq.m:268-312: on ``sphgrid(grid_symm, grid_size)`` (in the
    molecular frame, independent of the simulation grid), diagonalize
    ``H0 - B n.mu`` and sum ``w (|<u|mu_n,x|v>|^2 + |<u|mu_n,y|v>|^2) / 2``
    with the nuclear moment operators (``ham_nz``) along the lab x and y axes.
    Also returns the minimum and maximum of ``|E_u - E_v|`` over the grid.
    Detached (the selection is discrete); float64 ``(n, n)`` CPU tensors.
    """
    from torchspin._linalg import eigh as _eigh
    from torchspin.sphgrid import sphgrid
    H0, mux, muy, muz, snmux, snmuy, snmuz = (t.detach() for t in ops)
    dev = H0.device
    with torch.no_grad():
        ph, th, w, _ = sphgrid(str(grid_symm), int(grid_size))
        ph = ph.to(dtype=torch.float64, device=dev)
        th = th.to(dtype=torch.float64, device=dev)
        w = w.to(dtype=torch.float64, device=dev)
        st, ct, sp, cp = (torch.sin(th).view(-1, 1, 1), torch.cos(th).view(-1, 1, 1),
                          torch.sin(ph).view(-1, 1, 1), torch.cos(ph).view(-1, 1, 1))
        H = H0.unsqueeze(0) - float(field_mT) * (st * (cp * mux + sp * muy) + ct * muz)
        E, V = _eigh(H)
        Vh = V.conj().transpose(1, 2)
        ax = Vh @ (ct * (cp * snmux + sp * snmuy) - st * snmuz) @ V
        ay = Vh @ (-sp * snmux + cp * snmuy) @ V
        rates = (w.view(-1, 1, 1) * (ax.abs() ** 2 + ay.abs() ** 2) / 2.0).sum(dim=0)
        EE = (E.unsqueeze(2) - E.unsqueeze(1)).abs()
        return (rates.real.to(torch.float64).cpu(), EE.min(dim=0).values.to(torch.float64).cpu(),
                EE.max(dim=0).values.to(torch.float64).cpu())


def _tps_keep(rates: torch.Tensor, minE: torch.Tensor, maxE: torch.Tensor,
              freq_range: Optional[tuple[float, float]], threshold: float,
              n_electron_states: Optional[int]) -> torch.Tensor:
    """Level pairs kept by EasySpin's selection rule (endorfrq.m:314-341).

    Pairs whose frequency never enters ``freq_range`` on the TPS grid get rate
    0; electronic pairs (levels in different blocks of ``n / n_electron_states``
    energy-sorted levels) and the lower triangle are removed; a pair is kept
    if its rate is ``> threshold * max`` (strict, so with ``threshold = 0`` the
    pairs of exactly zero rate are dropped).  ``n_electron_states=None`` keeps
    the electronic pairs (zero-field clusters).  Returns an ``(n, n)`` bool mask
    (upper triangle).
    """
    n = rates.shape[0]
    r = rates.clone()
    if freq_range is not None:
        r[(maxE < float(freq_range[0])) | (minE > float(freq_range[1]))] = 0.0
    cand = torch.ones(n, n, dtype=torch.bool).triu(diagonal=1)
    if n_electron_states is not None and n_electron_states > 1:
        blk = torch.arange(n) // (n // int(n_electron_states))
        cand &= blk.view(-1, 1) == blk.view(1, -1)
    if not bool(cand.any()):
        return cand
    rmax = float(r[cand].max())
    return cand & (r > threshold * rmax)


def _endor_tps_select(ops: tuple, field_mT: float, *, freq_range, threshold: float,
                      n_electron_states: Optional[int], grid_symm: str = TPS_GRID_SYMM,
                      grid_size: int = TPS_GRID_SIZE) -> torch.Tensor:
    """EasySpin endorfrq's automatic transition selection at one field.

    :func:`_tps_rates` then :func:`_tps_keep`.  Returns the ``(n, n)`` bool
    mask (CPU) of the selected 0-based level pairs ``u < v``.  Does not depend
    on any orientation or simulation grid.
    """
    rates, minE, maxE = _tps_rates(ops, field_mT, grid_symm, grid_size)
    return _tps_keep(rates, minE, maxE, freq_range, threshold, n_electron_states)
