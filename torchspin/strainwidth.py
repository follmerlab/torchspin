"""Strain width calculation for orientation-dependent line broadening.

Implements the EasySpin strain algorithm (resfields.m lines 715-1320).
Strain models Gaussian distributions of spin Hamiltonian parameters, resulting
in transition-specific, orientation-dependent Gaussian FWHMs.

References
----------
EasySpin ``resfields.m``, ``resfields_perturb.m``, ``getdstrainops.m``
"""
from __future__ import annotations

from typing import Optional

import torch
from torchspin._linalg import eigh as _eigh


from torchspin.rotations import erot
from torchspin.spinsystem import SpinSystem


def _astrain_active(sys: SpinSystem) -> bool:
    """True when the AStrain width term is used (EasySpin ``useAStrain``)."""
    return (sys.AStrain is not None and sys.nNuclei > 0
            and bool(torch.any(torch.as_tensor(sys.AStrain).detach() != 0)))


def astrain_level_mI(
    sys: SpinSystem,
    H0: torch.Tensor,
    muz: torch.Tensor,
    centerB: float,
) -> Optional[torch.Tensor]:
    """Effective nuclear spin projection per level for the AStrain width (EasySpin model).

    EasySpin ``resfields.m`` (766-788) assigns every level one effective mI of
    the first nucleus, computed once for all orientations: the eigenvectors of
    ``H0 - centerB*muz`` with the field along the *molecular* z axis and
    ``centerB = mean(Exp.Range)``, levels sorted by energy, and
    ``mI_k = <k|Iz|k>`` of nucleus 1.  A transition (u, v) then uses
    ``(mI_u + mI_v)/2`` at every orientation, and the AStrain matrix enters the
    width as ``(G + gAStrainCorr * mI * AStrainMatrix).^2``.  The level index,
    not the state, carries mI, so an orientation where the energy order of the
    levels differs from the molecular-z order gets the mI of another state
    (see KNOWN_LIMITATIONS).

    Parameters
    ----------
    sys:
        Spin system (one electron spin; AStrain belongs to the first nucleus).
    H0, muz:
        Field-independent Hamiltonian and molecular z magnetic moment
        operator (MHz, MHz/mT), each ``(n, n)``.
    centerB:
        Field (mT) at which the levels are labeled; EasySpin uses
        ``mean(Exp.Range)``.

    Returns
    -------
    ``(n,)`` float64 tensor of mI per energy-sorted level, or ``None`` when
    the system has no AStrain (or no nuclei).  Differentiable in H0 and muz
    (degeneracy-safe eigendecomposition).
    """
    if not _astrain_active(sys):
        return None
    from torchspin.spinops import sop
    _, V = _eigh(H0 - float(centerB) * muz)
    Iz = sop(sys.Spins, [[sys.nElectrons + 1, 3]], device=H0.device).to(dtype=V.dtype)
    return torch.real((V.conj() * (Iz @ V)).sum(dim=0)).to(torch.float64)



def compute_strain_widths(
    sys: SpinSystem,
    H0: torch.Tensor,        # field-independent Hamiltonian
    mux: torch.Tensor,       # x magnetic moment operator
    muy: torch.Tensor,       # y magnetic moment operator
    muz: torch.Tensor,       # z magnetic moment operator
    phi: float,              # orientation azimuthal angle (rad)
    theta: float,            # orientation polar angle (rad)
    transitions: torch.Tensor,  # [nTrans, 2] level pairs (u, v) with u < v
    mwFreq: float,           # microwave frequency (GHz)
    B0: float,               # resonance field (mT) for this orientation
    dtype: torch.dtype = torch.complex128,
    Range: Optional[tuple] = None,           # field range (mT); AStrain labels levels at mean(Range)
    mI_levels: Optional[torch.Tensor] = None,  # (n,) AStrain mI per level (overrides Range)
) -> Optional[torch.Tensor]:
    """
    Compute orientation-dependent Gaussian FWHM (mT) for each transition.

    EasySpin's strain widths (``resfields.m`` 1283-1314) for one orientation:
    a thin wrapper over :func:`compute_strain_widths_batch` with M = 1 (since
    0.9.20; before, a separate per-orientation code path with ``.item()`` and
    Python floats that gave no or wrong gradients, B76, and read ``g[0]`` of a
    full g matrix, B5).  Each strain type contributes to the total
    linewidth-squared (MHz^2), converted to mT with
    ``dBdE = 1 / |<v|muzL|v> - <u|muzL|u>|`` at the eigenvectors of
    ``H0 - B0*muzL``.  Differentiable in H0, the moment operators, B0, the
    angles and the strain parameters (DStrain magnitudes are constants).

    Parameters
    ----------
    sys:
        Spin system with optional strain fields (HStrain, gStrain, DStrain, AStrain).
    H0:
        Field-independent Hamiltonian (MHz), shape ``[nStates, nStates]``.
    mux, muy, muz:
        Magnetic moment operators (MHz/mT), shape ``[nStates, nStates]``.
    phi, theta:
        Azimuthal and polar angle of the B0 direction in the molecular frame
        (radians), floats or 0-d tensors.
    transitions:
        Transition level pairs ``[u, v]`` where ``u < v``, shape ``[nTrans, 2]``.
    mwFreq:
        Microwave frequency in GHz (used for gStrain conversion).
    B0:
        Resonance field (mT) at which the eigenvectors are evaluated (float or
        0-d tensor).
    dtype:
        Accepted for backward compatibility; the arithmetic uses H0's dtype.
    Range:
        Sweep range (mT).  AStrain only: EasySpin computes the effective mI
        of every level once, at ``mean(Exp.Range)`` with the field along the
        molecular z axis (:func:`astrain_level_mI`).  Without ``Range`` (and
        without ``mI_levels``) the levels are labeled at ``B0``.
    mI_levels:
        AStrain only: precomputed :func:`astrain_level_mI` (overrides ``Range``).

    Returns
    -------
    widths:
        Gaussian FWHM (mT) for each transition, shape ``[nTrans]``. Returns
        ``None`` if no strain parameters are present.

    Raises
    ------
    ValueError
        gStrain/AStrain with a full g matrix, or AStrain with more than one
        electron (as EasySpin).
    """
    if not _has_any_strain(sys):
        return None
    device = H0.device
    pairs = transitions.to(device=device).reshape(-1, 2).long()
    if pairs.shape[0] == 0:
        return torch.zeros(0, dtype=torch.float64, device=device)
    ph = torch.as_tensor(phi, dtype=torch.float64, device=device).reshape(1)
    th = torch.as_tensor(theta, dtype=torch.float64, device=device).reshape(1)
    st, ct = torch.sin(th), torch.cos(th)
    sp, cp = torch.sin(ph), torch.cos(ph)
    muzL = ((st * cp).view(1, 1, 1) * mux + (st * sp).view(1, 1, 1) * muy
            + ct.view(1, 1, 1) * muz)
    B0_t = torch.as_tensor(B0, dtype=torch.float64, device=device).reshape(1)
    if mI_levels is None and _astrain_active(sys):
        centerB = 0.5 * (float(Range[0]) + float(Range[1])) if Range is not None else float(B0_t.detach())
        mI_levels = astrain_level_mI(sys, H0, muz, centerB)
    return compute_strain_widths_batch(sys, H0, muzL, ph, th, B0_t, [pairs], mwFreq,
                                       mI_levels=mI_levels)[0]


# ---------------------------------------------------------------------------
# Batched strain width computation (vectorized over all orientations)
# ---------------------------------------------------------------------------

def compute_strain_widths_batch(
    sys: SpinSystem,
    H0: torch.Tensor,           # (n, n)  field-independent Hamiltonian
    muzL_batch: torch.Tensor,   # (M, n, n)  per-orientation muzL
    phi_batch: torch.Tensor,    # (M,)  azimuthal angles (rad)
    theta_batch: torch.Tensor,  # (M,)  polar angles (rad)
    B0_batch: torch.Tensor,     # (M,)  representative resonance field per orientation (mT)
    pairs_list: Optional[list], # M items, each (nTrans_m, 2) int tensor or None
    mwFreq: float,              # GHz
    pairs_flat: Optional[torch.Tensor] = None,   # (M, 2): exactly one transition per orientation
    ho_batch: Optional[tuple] = None,            # (Q2, Q3), each (M, n, n) or None: Options.QuadraticZeeman
    mI_levels: Optional[torch.Tensor] = None,    # (n,) AStrain mI per level, from astrain_level_mI
) -> list:
    """Vectorized strain widths for a batch of M orientations.

    Replaces M individual calls to :func:`compute_strain_widths` with one
    batched ``torch.linalg.eigh`` call, giving a large speed-up on
    orientation-dense grids (Ci, C2h) with HStrain / gStrain.

    Handles HStrain, gStrain (single- and multi-electron), gStrain+AStrain and
    DStrain (per electron, tilted D frames, D/E correlation) in vectorised form.

    With ``pairs_flat`` (one transition per orientation, the resfields_batch
    call pattern) the widths are returned as a single ``(M,)`` tensor.

    With ``ho_batch = (Q2, Q3)`` (``Options.QuadraticZeeman``) the Hamiltonian at
    B0 is H0 − B0 muzL + B0² Q2 + B0³ Q3 and dB/dE uses the slope operator
    muzL − 2 B0 Q2 − 3 B0² Q3.

    AStrain needs ``mI_levels``, the effective mI of the first nucleus per
    energy-sorted level from :func:`astrain_level_mI` (EasySpin labels the
    levels once, at ``mean(Exp.Range)`` along the molecular z axis, and uses
    these mI at every orientation; :func:`torchspin.resfields_batch.resfields_batch`
    passes them).  Before 0.9.25 mI was ``<Iz>`` at each orientation's own
    resonance field, which vanishes for the field perpendicular to a large A
    and gave perpendicular lines no AStrain width (B68).

    Returns
    -------
    list of M tensors (nTrans_m,) in mT, or None if orientation had no resonances.

    Raises
    ------
    ValueError
        AStrain without ``mI_levels``; AStrain with more than one electron and
        gStrain (as EasySpin).
    """
    if not _has_any_strain(sys):
        return torch.zeros(pairs_flat.shape[0], dtype=torch.float64, device=H0.device) if pairs_flat is not None else [None] * len(pairs_list)

    device = H0.device
    M = phi_batch.shape[0]
    ph = phi_batch.to(device=device, dtype=torch.float64)
    th = theta_batch.to(device=device, dtype=torch.float64)

    # --- Orientation unit vectors: (M, 3) ---
    sin_th = torch.sin(th);  cos_th = torch.cos(th)
    sin_ph = torch.sin(ph);  cos_ph = torch.cos(ph)
    zLab_M = torch.stack([sin_th * cos_ph, sin_th * sin_ph, cos_th], dim=1)

    # --- Batched eigh at B0 for all M orientations ---
    H_b = H0.unsqueeze(0) - B0_batch[:, None, None] * muzL_batch  # (M, n, n)
    slope_batch = muzL_batch
    if ho_batch is not None:
        Bb = B0_batch.to(torch.float64)[:, None, None]
        Q2, Q3 = ho_batch
        if Q2 is not None:
            H_b = H_b + Bb ** 2 * Q2
            slope_batch = slope_batch - 2.0 * Bb * Q2
        if Q3 is not None:
            H_b = H_b + Bb ** 3 * Q3
            slope_batch = slope_batch - 3.0 * Bb ** 2 * Q3
    _, V_b = _eigh(H_b)  # (M, n, n)

    # --- muzL diagonal in eigenbasis: (M, n) ---
    # diag_k(V^† muzL V) = (V.conj() * (muzL @ V)).sum(dim=-2)
    # (with QuadraticZeeman: the slope operator -dH/dB in place of muzL)
    tmp = slope_batch @ V_b                             # (M, n, n)
    muzL_diag = torch.real((V_b.conj() * tmp).sum(dim=1))  # (M, n)

    # --- Precompute orientation-independent strain matrices ---
    # HStrain: (3,)  →  lw2_H[m] = sum(H^2 * z_m^2)
    lw2_H_vec = torch.zeros(M, dtype=torch.float64, device=device)
    if sys.HStrain is not None:
        H2 = sys.HStrain.to(device=device, dtype=torch.float64) ** 2        # (3,)
        lw2_H_vec = (H2.unsqueeze(0) * zLab_M ** 2).sum(dim=1)   # (M,)

    # gStrain: per-electron gStrainMatrix (3, 3) in molecular frame
    gmats = _gstrain_matrices(sys, mwFreq, device) if sys.gStrain is not None else \
        [torch.zeros(3, 3, dtype=torch.float64, device=device)] * sys.nElectrons
    gStrainMatrix = gmats[0]
    multi_g = sys.gStrain is not None and sys.nElectrons > 1
    if multi_g and sys.AStrain is not None and bool(torch.any(sys.AStrain != 0)):
        raise ValueError("AStrain is not supported in spin systems with more than one electron spin.")

    # lw2_g[m] = z_m^T * (G^2) * z_m   — vectorised over M (single-electron case)
    gStrainMatrix2 = gStrainMatrix ** 2                 # element-wise square
    tmp_g  = zLab_M @ gStrainMatrix2                    # (M, 3)
    lw2_g_vec = (tmp_g * zLab_M).sum(dim=1)             # (M,)

    # Multi-electron g strain: per-electron z_m^T G_e^2 z_m (M,) and the
    # lab-z spin projection diagonals |<k|kSzL_e|k>| in the eigenbasis (M, n)
    gterm_e: list = []
    kSzL_diag_e: list = []
    if multi_g:
        for e in range(sys.nElectrons):
            tmp_e = zLab_M @ (gmats[e] ** 2)
            gterm_e.append((tmp_e * zLab_M).sum(dim=1))                     # (M,)
            Sx, Sy, Sz = _electron_spin_ops(sys, e, device, V_b.dtype)
            zc = zLab_M.to(dtype=V_b.dtype)
            kSzL_b = (zc[:, 0, None, None] * Sx + zc[:, 1, None, None] * Sy
                      + zc[:, 2, None, None] * Sz)                          # (M, n, n)
            kSzL_diag_e.append(torch.real((V_b.conj() * (kSzL_b @ V_b)).sum(dim=1)))  # (M, n)

    # AStrain: build AStrainMatrix (3, 3) in molecular frame; mI per level is
    # orientation-independent (EasySpin resfields.m 766-788, astrain_level_mI)
    AStrainMatrix: Optional[torch.Tensor] = None
    if _astrain_active(sys) and sys.nElectrons == 1:
        if mI_levels is None:
            raise ValueError(
                "compute_strain_widths_batch: AStrain needs mI_levels, the effective mI of the "
                "first nucleus per level (EasySpin's model): pass "
                "mI_levels=astrain_level_mI(sys, H0, muz, centerB) with the molecular muz and "
                "centerB = mean(Exp.Range).")
        mI_levels = mI_levels.to(device=device, dtype=torch.float64).reshape(-1)
        if mI_levels.numel() != H0.shape[0]:
            raise ValueError(f"mI_levels has {mI_levels.numel()} entries; the Hamiltonian has "
                             f"{H0.shape[0]} levels.")
        AStrainMatrix = torch.diag(
            sys.AStrain.to(device=device, dtype=torch.float64)
        )
        if sys.AFrame is not None and torch.any(sys.AFrame[0] != 0):
            R_A2M = erot(sys.AFrame[0].tolist(), device=device).T.to(dtype=torch.float64)
            AStrainMatrix = R_A2M @ AStrainMatrix @ R_A2M.T

    # DStrain: per-electron dH/dD and dH/dE operators (EasySpin getdstrainops;
    # DFrame and DStrainCorr included), diagonals in the eigenbasis: (M, n)
    dstrain_diags: list = []
    if sys.DStrain is not None:
        V_c = V_b.to(torch.complex128)
        for op_pair in _dstrain_operators(sys, device, torch.complex128):
            if op_pair is None:
                continue
            dHdD_op, dHdE_op = op_pair
            dD_diag = torch.real((V_c.conj() * (dHdD_op.unsqueeze(0) @ V_c)).sum(dim=1))
            dE_diag = torch.real((V_c.conj() * (dHdE_op.unsqueeze(0) @ V_c)).sum(dim=1))
            dstrain_diags.append((dD_diag, dE_diag))

    # --- Build per-orientation results (flat over all transitions) ---
    corr = getattr(sys, 'gAStrainCorr', 1.0)
    if pairs_flat is not None:
        counts = None
        pairs_all = pairs_flat.to(device=device)
        m_of = torch.arange(M, device=device)
    else:
        counts = [0 if (p is None or p.numel() == 0) else int(p.shape[0]) for p in pairs_list]
        if sum(counts) == 0:
            return [None] * M
        pairs_all = torch.cat([p.reshape(-1, 2) for p, c in zip(pairs_list, counts) if c > 0], dim=0).to(device=device)
        m_of = torch.repeat_interleave(torch.arange(M, device=device), torch.tensor(counts, device=device))
    u_idx = pairs_all[:, 0].long()
    v_idx = pairs_all[:, 1].long()

    # HStrain (transition-independent)
    lw2 = lw2_H_vec[m_of].clone()

    # gStrain (+AStrain combined)
    if multi_g:
        for e in range(sys.nElectrons):
            w_e = torch.abs(kSzL_diag_e[e][m_of, v_idx] - kSzL_diag_e[e][m_of, u_idx])
            lw2 += w_e * gterm_e[e][m_of]
    elif sys.gStrain is not None and AStrainMatrix is None:
        lw2 += lw2_g_vec[m_of]
    elif AStrainMatrix is not None:
        # SM = G (if gStrain) + corr * mI_avg * A, lw2 += z^T (SM ∘ SM) z per transition
        mI_avg = 0.5 * (mI_levels[u_idx] + mI_levels[v_idx])                            # (K,)
        G = gStrainMatrix if sys.gStrain is not None else torch.zeros_like(AStrainMatrix)
        SM = G.unsqueeze(0) + (corr * mI_avg).view(-1, 1, 1) * AStrainMatrix.unsqueeze(0)  # (K, 3, 3)
        z = zLab_M[m_of]                                                                   # (K, 3)
        lw2 += torch.einsum('ki,kij,kj->k', z, SM ** 2, z)

    # DStrain (sum over electrons)
    for dD_diag, dE_diag in dstrain_diags:
        mD = dD_diag[m_of, v_idx] - dD_diag[m_of, u_idx]
        mE = dE_diag[m_of, v_idx] - dE_diag[m_of, u_idx]
        lw2 += mD ** 2 + mE ** 2

    # dBdE: |<v|muzL|v> - <u|muzL|u>|^{-1}  (already computed)
    muzL_diff = muzL_diag[m_of, v_idx] - muzL_diag[m_of, u_idx]
    dBdE = torch.where(torch.abs(muzL_diff) > 1e-12, 1.0 / torch.abs(muzL_diff), torch.zeros_like(muzL_diff))
    # sqrt with a finite gradient at lw2 = 0 (a zero width, e.g. HStrain = [5 5 0]
    # along z): d sqrt/d lw2 is infinite there and 0 * inf gave NaN gradients
    pos = lw2.detach() > 0
    widths = dBdE * torch.where(pos, torch.sqrt(torch.where(pos, lw2, torch.ones_like(lw2))),
                                torch.zeros_like(lw2))
    if counts is None:
        return widths
    chunks = torch.split(widths, counts)
    return [None if c == 0 else w for w, c in zip(chunks, counts)]


# ---------------------------------------------------------------------------
# Helper functions for each strain type
# ---------------------------------------------------------------------------

def _has_any_strain(sys: SpinSystem) -> bool:
    """Check if any strain parameter is present."""
    return (
        (sys.HStrain is not None and torch.any(sys.HStrain > 0))
        or (sys.gStrain is not None and torch.any(sys.gStrain > 0))
        or (sys.AStrain is not None and torch.any(sys.AStrain > 0))
        or (sys.DStrain is not None and torch.any(sys.DStrain > 0))
    )


_GAUSSIAN_STRAINS = ('HStrain', 'gStrain', 'AStrain', 'DStrain')


def lorentzian_strain_system(sys: SpinSystem) -> tuple[SpinSystem, bool]:
    """Route ``Sys.HStrainL`` through the ``HStrain`` width code.

    ``HStrainL`` has the functional form of ``HStrain`` (FWHM
    ``sqrt(sum_k (W_k n_k)^2)`` in frequency, divided by ``|dE/dB|`` in a field
    sweep); only the line profile differs.  Returns ``(sys, False)`` unchanged
    when ``HStrainL`` is absent or zero, else ``(sys', True)`` where ``sys'`` is
    a copy with ``HStrain = HStrainL`` and ``HStrainL = None``: the caller
    computes the widths as usual and draws every line as a Lorentzian.  The
    tensor is passed through as is, so its autograd graph is kept.

    Raises ``ValueError`` when ``HStrainL`` is combined with a Gaussian strain
    (``HStrain``, ``gStrain``, ``AStrain``, ``DStrain``): that would need a
    Voigt profile per line, which pepper does not implement.
    """
    hl = getattr(sys, 'HStrainL', None)
    if hl is None or not bool(torch.any(torch.as_tensor(hl).detach() != 0)):
        return sys, False
    if bool(torch.any(torch.as_tensor(hl).detach() < 0)):
        raise ValueError("HStrainL values must be non-negative.")
    for k in _GAUSSIAN_STRAINS:
        v = getattr(sys, k, None)
        if v is not None and bool(torch.any(torch.as_tensor(v).detach() != 0)):
            raise ValueError(
                f"SpinSystem.HStrainL (Lorentzian strain) cannot be combined with {k} "
                f"(Gaussian strain): pepper has no per-line Voigt profile. Use one kind "
                f"of strain per system; a global Sys.lw can be added to either.")
    from torchspin.spinsystem import _replace
    return _replace(sys, HStrain=hl, HStrainL=None), True


def _strain_g_values(sys: SpinSystem, e: int, device) -> torch.Tensor:
    """g principal values (g frame) that ``gStrain`` of electron *e* is relative to.

    This is ``sys.g[e]``, except when the linear Zeeman term is given as
    ``Ham110``/``Ham112`` (``SpinSystem`` sets g = 0): then it is the diagonal of
    the g-equivalent matrix of these terms (``ham_ezho.ham_linear_g``) in the
    g frame (``gFrame``).  Before 0.9.13 the division by g = 0 gave an all-zero
    spectrum (B56; EasySpin's pepper does the same).  A g-equivalent matrix that
    is not diagonal in the g frame raises, as gStrain with a full g matrix does.
    """
    g_values = sys.g[e].to(device=device, dtype=torch.float64)
    if bool(torch.any(g_values != 0)):
        return g_values
    from torchspin.ham_ezho import ham_linear_g
    gm = ham_linear_g(sys, e)
    if gm is None:
        return g_values
    gm = gm.to(device=device)
    if bool(torch.any(sys.gFrame[e] != 0)):
        R_g2M = erot(sys.gFrame[e].tolist(), device=device).T.to(dtype=torch.float64)
        gm = R_g2M.T @ gm @ R_g2M
    off = gm - torch.diag(torch.diagonal(gm))
    if float(off.detach().abs().max()) > 1e-9 * float(gm.detach().abs().max()):
        raise ValueError(
            "gStrain with a linear Zeeman term given as Sys.Ham110/Ham112 needs the "
            "g-equivalent matrix of these terms to be diagonal in Sys.gFrame (as for "
            "gStrain with a full g matrix). Give g as Sys.g with gFrame, or set gFrame "
            "to the principal axes of the Ham112 term.")
    return torch.diagonal(gm)


def _gstrain_matrices(sys: SpinSystem, mwFreq: float, device) -> list:
    """Per-electron g-strain matrices in the molecular frame (EasySpin ``gStrainMatrix{iEl}``, MHz).

    ``diag(gStrain(iEl,:)./g(iEl,:))*mwFreq`` rotated from the g frame to the
    molecular frame with ``R_g2M = erot(gFrame).'``.  Electrons without strain
    get a zero matrix.
    """
    mats = []
    for e in range(sys.nElectrons):
        M = torch.zeros(3, 3, dtype=torch.float64, device=device)
        if sys.gStrain is not None and bool(torch.any(sys.gStrain[e] != 0)):
            if sys.fullg:
                raise ValueError("gStrain and AStrain are not supported when full g matrices are given!")
            g_strain = sys.gStrain[e].to(device=device, dtype=torch.float64)
            g_values = _strain_g_values(sys, e, device)
            M = torch.diag((g_strain / g_values) * (mwFreq * 1e3))
            if bool(torch.any(sys.gFrame[e] != 0)):
                R_g2M = erot(sys.gFrame[e].tolist(), device=device).T.to(dtype=torch.float64)
                M = R_g2M @ M @ R_g2M.T
        mats.append(M)
    return mats


def _electron_spin_ops(sys: SpinSystem, e: int, device, dtype):
    """(Sx, Sy, Sz) of electron ``e`` (0-based) in the full spin space."""
    from torchspin.spinops import sop
    Sx = sop(sys.Spins, [[e + 1, 1]], device=device).to(dtype=dtype)
    Sy = sop(sys.Spins, [[e + 1, 2]], device=device).to(dtype=dtype)
    Sz = sop(sys.Spins, [[e + 1, 3]], device=device).to(dtype=dtype)
    return Sx, Sy, Sz


def _dstrain_operators(sys: SpinSystem, device, dtype) -> list:
    """EasySpin ``getdstrainops``: per-electron ``(dH/dD, dH/dE)`` pre-multiplied by the strain FWHMs.

    Handles a tilted D frame (``DFrame``) and a D/E correlation coefficient
    (``DStrainCorr``).  Electrons without D strain (or with S < 1) get ``None``.
    """
    ops = []
    if sys.DStrain is None:
        return [None] * sys.nElectrons
    for e in range(sys.nElectrons):
        dD = float(sys.DStrain[e, 0])
        dE = float(sys.DStrain[e, 1]) if sys.DStrain.shape[1] > 1 else 0.0
        if (dD == 0 and dE == 0) or sys.S[e] < 1.0:
            ops.append(None)
            continue
        Sx, Sy, Sz = _electron_spin_ops(sys, e, device, dtype)
        if sys.DFrame is not None and bool(torch.any(sys.DFrame[e] != 0)):
            R = erot(sys.DFrame[e].tolist(), device=device).to(dtype=dtype)  # molecular -> D frame
            SxD = R[0, 0] * Sx + R[0, 1] * Sy + R[0, 2] * Sz
            SyD = R[1, 0] * Sx + R[1, 1] * Sy + R[1, 2] * Sz
            SzD = R[2, 0] * Sx + R[2, 1] * Sy + R[2, 2] * Sz
            SxD2, SyD2, SzD2 = SxD @ SxD, SyD @ SyD, SzD @ SzD
        else:
            SxD2, SyD2, SzD2 = Sx @ Sx, Sy @ Sy, Sz @ Sz
        dHdD_ = (2 * SzD2 - SxD2 - SyD2) / 3
        dHdE_ = SxD2 - SyD2
        r = float(sys.DStrainCorr[e]) if isinstance(sys.DStrainCorr, (list, tuple)) else float(sys.DStrainCorr)
        if r != 0:
            R12 = r * dD * dE
            cov = torch.tensor([[dD ** 2, R12], [R12, dE ** 2]], dtype=torch.float64)
            L, V = torch.linalg.eigh(cov)          # ascending, like MATLAB eig for symmetric input
            L = torch.sqrt(torch.clamp(L, min=0.0))
            dHdD = float(L[0]) * (float(V[0, 0]) * dHdD_ + float(V[0, 1]) * dHdE_)
            dHdE = float(L[1]) * (float(V[1, 0]) * dHdD_ + float(V[1, 1]) * dHdE_)
        else:
            dHdD = dD * dHdD_
            dHdE = dE * dHdE_
        ops.append((dHdD, dHdE))
    return ops


def compute_strain_widths_freq_batch(
    sys: SpinSystem,
    H0: torch.Tensor,           # (n, n)
    muzL_batch: torch.Tensor,   # (M, n, n)
    phi_batch: torch.Tensor,    # (M,)
    theta_batch: torch.Tensor,  # (M,)
    B0: float,                  # static field (mT), the same for all orientations
    pairs_list: list,           # M items, each (nTrans_m, 2) int tensor or None
    ho_batch: Optional[tuple] = None,   # (Q2, Q3), each (M, n, n) or None: Options.QuadraticZeeman
    clusters: Optional[torch.Tensor] = None,   # (n,) degenerate-cluster label per level (zero field)
) -> list:
    """Frequency-domain strain widths (MHz) for frequency-swept spectra.

    Port of the strain block of EasySpin ``resfreqs_matrix.m``: with
    ``m(op) = <v|op|v> - <u|op|u>`` the squared width is

        HStrain.^2·zL.^2 + Σ_e |m(dHdD_e)|² + |m(dHdE_e)|²
        + |m(dHdAx)|² + |m(dHdAy)|² + |m(dHdAz)|²
        + (m(μzL)·B0·zLᵀ·G·zL)²,   G = diag(gStrain./g) in the molecular frame,

    without the field conversion factor (no dBdE).  Returns a list of M
    tensors ``(nTrans_m,)`` in MHz (None where the orientation has no pairs).

    ``clusters`` (zero field, ``Options.ZeroField``): every diagonal element
    ``<u|op|u>`` is replaced by its mean over the degenerate cluster of ``u``,
    ``tr(P_a op)/n_a``, which does not depend on the eigenbasis inside the
    cluster.
    """
    if not _has_any_strain(sys):
        return [None] * len(pairs_list)
    device = H0.device
    M = phi_batch.shape[0]
    ph = phi_batch.to(device=device, dtype=torch.float64)
    th = theta_batch.to(device=device, dtype=torch.float64)
    zLab_M = torch.stack([torch.sin(th) * torch.cos(ph), torch.sin(th) * torch.sin(ph), torch.cos(th)], dim=1)
    H_b = H0.unsqueeze(0) - float(B0) * muzL_batch
    if ho_batch is not None:
        # QuadraticZeeman: eigenvectors of the full H(B0); the g-strain term below
        # keeps the linear moment (g strain perturbs the linear Zeeman term only)
        if ho_batch[0] is not None:
            H_b = H_b + float(B0) ** 2 * ho_batch[0]
        if ho_batch[1] is not None:
            H_b = H_b + float(B0) ** 3 * ho_batch[1]
    _, V_b = _eigh(H_b)
    V_c = V_b.to(torch.complex128)

    if clusters is None:
        def _cavg(d):
            return d
    else:
        _lab = clusters.to(device=device, dtype=torch.long)
        _oh = torch.zeros(_lab.numel(), int(_lab.max()) + 1, dtype=torch.float64, device=device)
        _oh[torch.arange(_lab.numel(), device=device), _lab] = 1.0

        def _cavg(d):                                             # (M, n) -> cluster means per level
            return ((d @ _oh) / _oh.sum(dim=0))[:, _lab]

    def diag_in_eigbasis(op):
        return _cavg(torch.real((V_c.conj() * (op.to(torch.complex128).unsqueeze(0) @ V_c)).sum(dim=1)))  # (M, n)

    muzL_diag = _cavg(torch.real((V_b.conj() * (muzL_batch @ V_b)).sum(dim=1)))
    lw2_H = torch.zeros(M, dtype=torch.float64, device=device)
    if sys.HStrain is not None:
        lw2_H = ((sys.HStrain.to(device=device, dtype=torch.float64) ** 2).unsqueeze(0) * zLab_M ** 2).sum(dim=1)
    # g strain: dimensionless G = diag(gStrain/g) (rotated); only the first electron, as EasySpin
    zGz = None
    if sys.gStrain is not None and bool(torch.any(sys.gStrain != 0)):
        G = _gstrain_matrices(sys, 1e-3, device)[0]          # mwFreq=1e-3 GHz → factor 1 MHz
        zGz = ((zLab_M @ G) * zLab_M).sum(dim=1)              # (M,)
    # A strain: separate x, y, z terms, first electron / first nucleus
    a_diags: list = []
    if sys.AStrain is not None and sys.nNuclei > 0 and bool(torch.any(sys.AStrain != 0)):
        from torchspin.spinops import sop
        n_e = sys.nElectrons
        for c in (1, 2, 3):
            S_op = sop(sys.Spins, [[1, c]], device=device).to(torch.complex128)
            I_op = sop(sys.Spins, [[n_e + 1, c]], device=device).to(torch.complex128)
            a_diags.append(diag_in_eigbasis(float(sys.AStrain[c - 1]) * (I_op @ S_op)))
    d_diags: list = []
    if sys.DStrain is not None:
        for op_pair in _dstrain_operators(sys, device, torch.complex128):
            if op_pair is None:
                continue
            d_diags.append((diag_in_eigbasis(op_pair[0]), diag_in_eigbasis(op_pair[1])))
    results: list = []
    for m in range(M):
        pairs_m = pairs_list[m]
        if pairs_m is None or pairs_m.numel() == 0:
            results.append(None)
            continue
        u_idx = pairs_m[:, 0].long(); v_idx = pairs_m[:, 1].long()
        lw2 = torch.zeros(pairs_m.shape[0], dtype=torch.float64, device=device) + lw2_H[m]
        for dD, dE in d_diags:
            lw2 += (dD[m, v_idx] - dD[m, u_idx]) ** 2 + (dE[m, v_idx] - dE[m, u_idx]) ** 2
        for ad in a_diags:
            lw2 += (ad[m, v_idx] - ad[m, u_idx]) ** 2
        if zGz is not None:
            lw2 += ((muzL_diag[m, v_idx] - muzL_diag[m, u_idx]) * float(B0) * zGz[m]) ** 2
        results.append(torch.sqrt(torch.clamp(lw2, min=0.0)))
    return results
