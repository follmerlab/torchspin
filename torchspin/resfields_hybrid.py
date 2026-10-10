"""Hybrid resonance fields: exact core plus perturbational ligand nuclei.

EasySpin's ``Opt.Method='hybrid'`` (``resfields.m``) splits a spin system into a
core treated by exact diagonalization — all electron spins plus the nuclei named
in ``Opt.HybridCoreNuclei`` — and the remaining nuclei, which only shift and
split the core lines.  It is the method for Cu(II) with resolved ligand
superhyperfine structure: the exact treatment of Cu + 4 nitrogens needs a
648-dimensional Hilbert space, while the core is 8-dimensional and each nitrogen
contributes a 3x3 problem.

The approximation is a first-order decoupling of the electron and the
perturbational nuclei.  For a core transition |u> -> |v> at its resonance field,
the electron spin operators in the hyperfine term are replaced by their
expectation values in the two core eigenstates,

    H_u = sum_e sum_a <S_a^(e)>_u (A^(e) . I)_a + H_Q + B_res (z_M . H_zeeman)

and likewise for ``v``.  Each nuclear sub-Hamiltonian is then diagonalized
*exactly*, so the nuclear quadrupole interaction, the nuclear Zeeman term and
the mixing of nuclear states are not approximated — only the electron-nuclear
decoupling is.  The two manifolds give different nuclear eigenbases, and the
overlap matrix

    M_rc = |<u_r|v_c>|^2

(the Mims matrix) supplies the line amplitudes, including the forbidden
branches, and reduces to the identity in the strong-decoupling limit.  Energy
shifts become field shifts through the same transition-dependent slope
``dBdE = (d(E_v - E_u)/dB)^-1`` that scales the core intensities and widths.

Sub-lines whose summed amplitude falls below ``Opt.HybridIntThreshold`` times
the strongest are dropped, then the nuclei are combined by an outer sum of
shifts and an outer product of amplitudes.  A set of ``n`` equivalent nuclei
shares one sub-Hamiltonian, so its ``n`` copies are combined once by multiset
enumeration rather than by an n-fold outer product: for four 14N that replaces
9**4 = 6561 combinations by 495.

One deliberate difference from EasySpin: the same threshold is applied again to
the running combination, not only per nucleus, which bounds the growth when
several nuclei each keep several sub-lines.  Its effect is far below the
validation tolerance — every case in
``torchspin/tests/test_pepper_cupc_matlab_validation.py`` matches EasySpin's
hybrid to cosine 0.9998 — and the total amplitude it can remove is at most the
threshold times the strongest line.

The pruning decision depends on the orientations present, so all of them have
to be in one call; ``pepper`` enforces that (see ``_orientation_batch_size``),
because a sub-line index has to mean the same line for every orientation of the
grid or the interpolation between them is meaningless.
"""
from __future__ import annotations

from typing import Optional

import torch

from torchspin._linalg import eigh
from torchspin.constants import NMAGN, PLANCK
from torchspin.isotopologues import multisetlist
from torchspin.rotations import erot
from torchspin.spinops import sop

__all__ = ['hybrid_sublines', 'core_nuclei_split', 'core_spin_operators']

# Amplitudes below this fraction of the strongest sub-line are discarded
# (EasySpin Opt.HybridIntThreshold).
DEFAULT_INT_THRESHOLD = 0.005


def core_nuclei_split(sys, core_nuclei) -> tuple[list[int], list[int]]:
    """Split nucleus indices into (core, perturbational), both 0-based.

    *core_nuclei* follows EasySpin's ``Opt.HybridCoreNuclei``: 1-based indices
    into ``Sys.Nucs``, empty meaning every nucleus is perturbational.  Electron
    spins are always part of the core.
    """
    n_nuc = sys.nNuclei
    if core_nuclei is None:
        idx1 = []
    elif isinstance(core_nuclei, (int, float)):
        idx1 = [int(core_nuclei)]
    else:
        idx1 = [int(v) for v in core_nuclei]
    bad = [v for v in idx1 if v < 1 or v > n_nuc]
    if bad:
        raise ValueError(f'Options.HybridCoreNuclei entries {bad} are out of range; '
                         f'the system has {n_nuc} nuclei (1-based indices).')
    core = sorted(set(v - 1 for v in idx1))
    perturb = [i for i in range(n_nuc) if i not in core]
    return core, perturb


def core_spin_operators(core_sys, dtype=torch.complex128, device='cpu'):
    """``[(Sx, Sy, Sz), ...]`` per electron spin, in the core Hilbert space.

    These give the expectation values ``<u|S|u>`` that decouple the electron
    from the perturbational nuclei (EasySpin ``resfields.m``: ``S(iEl).x =
    sop(CoreSys,[iEl,1])``).  Orientation-independent, so built once per
    simulation.
    """
    spins = core_sys.Spins
    return [tuple(sop(spins, [[e + 1, c]], dtype=dtype, device=device) for c in (1, 2, 3))
            for e in range(core_sys.nElectrons)]


def _nuclear_operators(sys, i_nuc: int, dtype, device):
    """(Ix, Iy, Iz) for one nucleus in its own (2I+1)-dimensional space."""
    I = float(sys.I[i_nuc])
    ops = [sop([I], [[1, c]], dtype=dtype, device=device) for c in (1, 2, 3)]
    return ops


def _tensor_in_molecular_frame(principal, frame_angles, device):
    """Rotate a principal-axis tensor into the molecular frame (as ham_hf/ham_nq)."""
    T = principal.double().to(device=device)
    if any(float(a) != 0.0 for a in frame_angles):
        R_A2M = erot(list(frame_angles), device=device).T
        T = R_A2M.double() @ T @ R_A2M.T.double()
    return T


def _one_nucleus_sublines(sys, i_nuc, S_core, psi_u, psi_v, B_res, dBdE, nvec,
                          dtype, device):
    """Shifts (mT) and amplitudes of one perturbational nucleus.

    Returns ``(shift, amp)``, both ``(nBrackets, (2I+1)**2)``.
    """
    n_el = sys.nElectrons
    Ix, Iy, Iz = _nuclear_operators(sys, i_nuc, dtype, device)
    d = Ix.shape[0]
    I_vec = torch.stack([Ix, Iy, Iz])                       # (3, d, d)

    # Hyperfine: Hhfi[e, a] = sum_b A^(e)_mol[a, b] I_b  (MHz, per unit <S_a>)
    Hhfi = torch.zeros((n_el, 3, d, d), dtype=dtype, device=device)
    A_tensor = sys.A
    for e in range(n_el):
        eidx = slice(3 * e, 3 * e + 3)
        if A_tensor is None:
            continue
        if sys.fullA:
            A_pri = A_tensor[3 * i_nuc:3 * i_nuc + 3, eidx]
        else:
            A_pri = torch.diag(A_tensor[i_nuc, eidx])
        if not A_pri.any():
            continue
        A_mol = _tensor_in_molecular_frame(A_pri, sys.AFrame[i_nuc, eidx].tolist(), device)
        Hhfi[e] = torch.einsum('ab,bij->aij', A_mol.to(dtype), I_vec)

    # Nuclear quadrupole (I >= 1) — independent of the electronic manifold
    H_const = torch.zeros((d, d), dtype=dtype, device=device)
    if float(sys.I[i_nuc]) >= 1 and sys.Q is not None:
        Q_pri = (sys.Q[3 * i_nuc:3 * i_nuc + 3, :] if sys.fullQ
                 else torch.diag(sys.Q[i_nuc, :]))
        if Q_pri.any():
            Q_mol = _tensor_in_molecular_frame(Q_pri, sys.QFrame[i_nuc].tolist(), device)
            H_const = H_const + torch.einsum('ab,aij,bjk->ik', Q_mol.to(dtype), I_vec, I_vec)

    # Nuclear Zeeman, MHz/mT.  ham_nz builds magnetic-moment operators with
    # +NMAGN/PLANCK*gn*gnscale*1e-9; the Hamiltonian term is -mu.B, hence the sign.
    pre = -NMAGN / PLANCK * float(sys.gn[i_nuc]) * float(sys.gnscale[i_nuc]) * 1e-9
    H_zeeman = pre * I_vec                                   # (3, d, d)

    # <S_a^(e)> in the two core eigenstates: (K, 3, n_el)
    def _expect(psi):
        out = torch.empty((psi.shape[0], 3, n_el), dtype=psi.real.dtype, device=device)
        for e in range(n_el):
            for c in range(3):
                op = S_core[e][c].to(dtype=psi.dtype, device=device)
                out[:, c, e] = (psi.conj() * (psi @ op.T)).sum(dim=-1).real
        return out

    Su, Sv = _expect(psi_u), _expect(psi_v)

    # Manifold-independent part, per bracket: H_Q + B_res (z_M . H_zeeman)
    Hc = H_const.unsqueeze(0) + (B_res.view(-1, 1, 1)
                                 * torch.einsum('ka,aij->kij', nvec.to(dtype), H_zeeman))

    def _levels(S_exp):
        H = torch.einsum('kae,eaij->kij', S_exp.to(dtype), Hhfi) + Hc
        H = 0.5 * (H + H.conj().transpose(-1, -2))           # eigh wants exact Hermiticity
        return eigh(H)

    dEu, Vu = _levels(Su)
    dEv, Vv = _levels(Sv)

    # Mims matrix: overlap (r, c) = <u_r|v_c>
    ov = Vu.conj().transpose(-1, -2) @ Vv                    # (K, d, d)
    amp = (ov.real ** 2 + ov.imag ** 2)
    # A shift that raises the transition energy lowers the resonance field.
    delta = dEv.unsqueeze(-2) - dEu.unsqueeze(-1)            # (K, r, c) = dEv[c] - dEu[r]
    shift = -dBdE.view(-1, 1, 1) * delta
    return shift.reshape(shift.shape[0], -1), amp.reshape(amp.shape[0], -1)


def _prune(shift, amp, threshold):
    """Drop sub-lines whose amplitude summed over all brackets is negligible."""
    total = amp.sum(dim=0)
    if total.numel() == 0:
        return shift, amp
    keep = total >= total.max() * threshold
    if not bool(keep.any()):
        keep = total == total.max()
    return shift[:, keep], amp[:, keep]


def _combine_equivalent(shift, amp, n_equiv):
    """Combine *n_equiv* identical copies of one sub-line set by multiset enumeration."""
    if n_equiv <= 1:
        return shift, amp
    n_sub = shift.shape[1]
    kvec, mult = multisetlist(n_equiv, n_sub)
    k = torch.as_tensor(kvec, dtype=shift.dtype, device=shift.device)       # (m, n_sub)
    w = torch.as_tensor(mult, dtype=amp.dtype, device=amp.device)           # (m,)
    shift_eq = shift @ k.T                                                  # (K, m)
    # Product of amp_j ** k_j, in log space: the surviving amplitudes are positive.
    log_amp = amp.clamp_min(torch.finfo(amp.dtype).tiny).log()
    amp_eq = torch.exp(log_amp @ k.to(amp.dtype).T) * w
    return shift_eq, amp_eq


def hybrid_sublines(sys, perturb_idx, S_core, psi_u, psi_v, B_res, dBdE, nvec,
                    int_threshold: float = DEFAULT_INT_THRESHOLD,
                    n_equiv: Optional[list[int]] = None):
    """Nuclear sub-line shifts and amplitudes for every core resonance.

    Parameters
    ----------
    sys:
        The full spin system (hyperfine, quadrupole and nuclear-g data for the
        perturbational nuclei are read from it).
    perturb_idx:
        0-based indices of the nuclei treated perturbationally.
    S_core:
        ``S_core[e]`` is ``(Sx, Sy, Sz)`` for electron *e* in the core space.
    psi_u, psi_v:
        Core eigenvectors of the two levels of each resonance, ``(K, nCore)``.
    B_res:
        Core resonance fields, ``(K,)`` in mT.
    dBdE:
        Frequency-to-field slope of each core transition, ``(K,)`` in mT/MHz.
    nvec:
        Lab z axis in the molecular frame for each resonance, ``(K, 3)``.
    int_threshold:
        Relative amplitude below which a sub-line is dropped.
    n_equiv:
        Multiplicity of each entry of ``sys.Nucs`` (``SpinSystem.n``).  A group
        with multiplicity > 1 is expanded by multiset enumeration.

    Returns
    -------
    (shift, amp):
        Both ``(K, nSub)``; ``shift`` in mT is added to the core resonance
        field and ``amp`` multiplies the core intensity.  ``nSub`` is the
        product of the surviving sub-line counts of all perturbational nuclei.
    """
    K = B_res.shape[0]
    device, dtype = psi_u.device, psi_u.dtype
    rdtype = B_res.dtype
    if not perturb_idx or K == 0:
        return (torch.zeros((K, 1), dtype=rdtype, device=device),
                torch.ones((K, 1), dtype=rdtype, device=device))

    shift_tot = torch.zeros((K, 1), dtype=rdtype, device=device)
    amp_tot = torch.ones((K, 1), dtype=rdtype, device=device)
    for i_nuc in perturb_idx:
        shift, amp = _one_nucleus_sublines(sys, i_nuc, S_core, psi_u, psi_v,
                                           B_res, dBdE, nvec, dtype, device)
        shift, amp = _prune(shift.to(rdtype), amp.to(rdtype), int_threshold)
        mult = 1 if n_equiv is None else int(n_equiv[i_nuc])
        shift, amp = _combine_equivalent(shift, amp, mult)
        if mult > 1:
            shift, amp = _prune(shift, amp, int_threshold)
        # Outer sum of shifts, outer product of amplitudes.
        shift_tot = (shift_tot.unsqueeze(-1) + shift.unsqueeze(-2)).reshape(K, -1)
        amp_tot = (amp_tot.unsqueeze(-1) * amp.unsqueeze(-2)).reshape(K, -1)
        shift_tot, amp_tot = _prune(shift_tot, amp_tot, int_threshold)
    return shift_tot, amp_tot
