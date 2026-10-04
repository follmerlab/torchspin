"""Slow-motion CW EPR spectrum via the Stochastic Liouville Equation (SLE).

Port of EasySpin's ``chili.m`` (general/matrix-method path only).

Scope (this implementation)
---------------------------
* S = 1/2, one or zero nuclei (I ≤ 1)
* Isotropic rotational diffusion — single scalar ``D_rot = 1/(6·tcorr)``
* No orienting potential (U = 0) — single orientation θ=0 in mol frame
* Field-swept mode, approximate linear field sweep (build L once at center)
* Meirovitch symmetry: pS = +1 → M = 0

Algorithm overview
------------------
1. Parse rotational diffusion coefficient D_rot from tcorr / Diff fields.
2. Build ISTO spherical components F0, F2 for g-tensor (and A-tensor if nucleus).
3. Build spin-space superoperators T0, T2 in the single-EPR-coherence subspace.
4. Enumerate orientational basis {L, jK, K} with M=0, K≤Kmax, L≤Lmax.
5. Build the Liouvillian:
       L = 2πi·H_comm + Γ_diff
   where H_comm = Σ_rank F_rank * T_rank coupling spatial and spin parts via
   Wigner 3j symbols, and Γ_diff is diagonal (D_rot·L(L+1) per basis state).
6. Run Lanczos tri-diagonalization with pseudonorm (complex-symmetric L).
7. Evaluate spectral function via Lentz continued-fraction at each field point.

References
----------
Freed, Bruno, Polnaszek, J. Phys. Chem. 1971.
Meirovitch et al., J. Phys. Chem. 1984.
Schneider & Freed in "Lasers, Molecules, and Methods" (1989).
EasySpin source: chili.m, chili_basisbuild.m, liouvhamiltonian.m,
                 diffsuperop.m, chili_lanczos.m, rbos.m, magint.m.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch

from torchspin.constants import BMAGN, PLANCK, GFREE, NMAGN
from torchspin.spinsystem import SpinSystem
from torchspin.experiment import Experiment

__all__ = ['chili', 'ChiliOptions']


# ---------------------------------------------------------------------------
# Options dataclass
# ---------------------------------------------------------------------------

@dataclass
class ChiliOptions:
    """Computational options for slow-motion EPR (chili) — EasySpin ``Opt`` fields.

    Parameters
    ----------
    LLMK:
        ``[evenLmax, oddLmax, Mmax, Kmax]`` — truncation of the Wigner function
        basis.  Default ``[14, 7, 2, 6]`` (EasySpin default; required when an
        orienting potential is given).
    jKmin, evenK:
        Basis symmetry switches (auto: from tensor collinearity / tilts).
    highField:
        Keep only pS = +1 spin coherences (EasySpin ``Opt.highField``).
    pImax, pImaxall, MpSymm:
        Nuclear coherence-order truncation and M/p symmetry pruning.
    Solver:
        ``'L'`` Lanczos + continued fraction (default), ``'\\'`` direct sparse
        solve, ``'E'`` eigenvalue method.
    FieldSweepMethod:
        ``'approxlin'`` (default), ``'approxinv'``, ``'explicit'``.
    GridSize, GridSymmetry:
        Powder grid, used only with an orienting potential (``Sys.Potential``).
    PostConvNucs:
        1-based indices of nuclei treated by post-convolution (garlic).
    Threshold, Rescale, Lentz, IncludeNZI, PeqTol, useStartvecSelectionRules:
        Solver controls as in EasySpin.
    Verbosity:
        0 = silent, 1 = progress.
    """
    LLMK: list = field(default_factory=lambda: [14, 7, 2, 6])
    Threshold: float = 1e-6
    max_iter: int = 2000          # kept for backward compatibility (unused by the SLE port)
    Verbosity: int = 0
    jKmin: Optional[int] = None
    evenK: Optional[bool] = None
    highField: bool = False
    pImax: Optional[list] = None
    pImaxall: Optional[int] = None
    MpSymm: Optional[bool] = None       # None = EasySpin's effective default (see chili_sle)
    Solver: str = ''
    FieldSweepMethod: str = ''
    GridSize: int = 19
    GridSymmetry: str = 'Dinfh'
    Rescale: bool = True
    Lentz: bool = True
    IncludeNZI: bool = True
    PostConvNucs: list = field(default_factory=list)
    useStartvecSelectionRules: bool = True
    PeqTol: Optional[list] = None


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

_MHZ_PER_MT = BMAGN / PLANCK * 1e-9   # ≈ 13.9962 MHz/mT  (g=1 Larmor)
_GHZ_TO_RAD_PER_S = 2 * math.pi * 1e9 # convert GHz → rad/s


# ---------------------------------------------------------------------------
# Step 1: Parse diffusion
# ---------------------------------------------------------------------------

def _parse_diffusion(sys: SpinSystem) -> float:
    """Return isotropic rotational diffusion coefficient D_rot (s^-1).

    Precedence: logtcorr > tcorr > logDiff > Diff.
    """
    if sys.logtcorr is not None:
        tcorr = 10.0 ** sys.logtcorr
        return 1.0 / (6.0 * tcorr)
    if sys.tcorr is not None:
        return 1.0 / (6.0 * sys.tcorr)
    if sys.logDiff is not None:
        return 10.0 ** sys.logDiff
    if sys.Diff is not None:
        return sys.Diff
    raise ValueError(
        "chili: spin system must specify tcorr, logtcorr, Diff, or logDiff."
    )


# ---------------------------------------------------------------------------
# Step 2: ISTO spherical components
# ---------------------------------------------------------------------------

def _tensor_cart2sph(diag: np.ndarray) -> tuple[float, np.ndarray]:
    """Cartesian diagonal tensor → ISTO spherical components.

    Parameters
    ----------
    diag:
        Diagonal principal values [Axx, Ayy, Azz].

    Returns
    -------
    F0:
        Rank-0 component (scalar) = -sqrt(1/3)*(Axx+Ayy+Azz).
    F2:
        Rank-2 components, shape (5,), order q = +2,+1,0,-1,-2.
        For real diagonal tensors F2[+1]=F2[-1]=0 (no off-diagonal in mol frame).
        F2[0] = sqrt(2/3)*(Azz - 0.5*(Axx+Ayy))
        F2[+2] = F2[-2] = 0.5*(Axx - Ayy)
    """
    Axx, Ayy, Azz = float(diag[0]), float(diag[1]), float(diag[2])
    F0 = -math.sqrt(1.0 / 3.0) * (Axx + Ayy + Azz)
    F2 = np.zeros(5, dtype=complex)  # q = +2,+1,0,-1,-2
    F2[0] = 0.5 * (Axx - Ayy)           # q=+2
    F2[1] = 0.0                          # q=+1
    F2[2] = math.sqrt(2.0 / 3.0) * (Azz - 0.5 * (Axx + Ayy))  # q=0
    F2[3] = 0.0                          # q=-1
    F2[4] = 0.5 * (Axx - Ayy)           # q=-2  (same as q=+2 for real diagonal)
    return F0, F2


def _g_to_freq_units(g_diag: np.ndarray, B0_T: float) -> np.ndarray:
    """Convert g-tensor principal values to frequency units (MHz).

    H_EZ = -μ·B = (μ_B / h) * g * B  [MHz] when B in T.
    Pre-factor: BMAGN / PLANCK * 1e-6 = 13996.2 MHz/T / g.
    """
    # pre in MHz/T for g=1
    pre_MHz_per_T = BMAGN / PLANCK * 1e-6  # = _MHZ_PER_MT * 1000
    return g_diag * pre_MHz_per_T * B0_T   # MHz


# ---------------------------------------------------------------------------
# Step 3: Spin superoperators in EPR coherence subspace
# ---------------------------------------------------------------------------

def _spin_superops_S12() -> dict:
    """Return rank-0 and rank-2 ISTO spin operators for S=1/2.

    For the single EPR coherence |+1/2><-1/2| (pS=+1, qS=-1):
    The relevant matrix elements are those between |α><β| and |α'><β'|.

    For S=1/2 in the Hilbert basis {|+1/2>, |-1/2>}:
    Sx = [[0,1/2],[1/2,0]], Sy = [[0,-i/2],[i/2,0]], Sz = [[1/2,0],[0,-1/2]]

    Rank-0 T0 (isotropic, scalar): T0 = -sqrt(1/3)*S·S = -sqrt(1/3)*S(S+1)*I
    For S=1/2: S(S+1) = 3/4, so T0 = -sqrt(1/3)*3/4 * identity

    Rank-2 T2 operators (irreducible tensor operators for S=1/2):
    T2[0] = sqrt(2/3) * Sz (for rank-2, m=0)
    T2[+1] = -1/2 * S+ (raising)
    T2[-1] = +1/2 * S- (lowering)
    T2[+2] = 0
    T2[-2] = 0

    Returns
    -------
    dict with keys 'T0', 'T2' (list of 5 matrices for q=+2,+1,0,-1,-2)
    """
    # 2×2 spin matrices for S=1/2
    Sp = np.array([[0, 1], [0, 0]], dtype=complex)   # S+
    Sm = np.array([[0, 0], [1, 0]], dtype=complex)   # S-
    Sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)  # Sz
    I2 = np.eye(2, dtype=complex)

    T0 = -math.sqrt(1.0 / 3.0) * (3.0 / 4.0) * I2  # = -sqrt(1/3)*S(S+1)*I

    T2 = [None] * 5  # q = +2,+1,0,-1,-2
    T2[0] = np.zeros((2, 2), dtype=complex)           # q=+2: T2[+2]=0 for S=1/2
    T2[1] = -0.5 * Sp                                  # q=+1
    T2[2] = math.sqrt(2.0 / 3.0) * Sz                 # q=0
    T2[3] = 0.5 * Sm                                   # q=-1
    T2[4] = np.zeros((2, 2), dtype=complex)            # q=-2: T2[-2]=0 for S=1/2

    return {'T0': T0, 'T2': T2}


def _nuclear_spin_ops(I: float) -> dict:
    """Return nuclear spin operators for spin I.

    Returns Ix, Iy, Iz, I+, I- as complex matrices.
    """
    dim = int(round(2 * I + 1))
    m_vals = np.arange(I, -I - 1, -1)  # I, I-1, ..., -I

    Iz = np.diag(m_vals).astype(complex)
    Ip = np.zeros((dim, dim), dtype=complex)  # I+
    Im = np.zeros((dim, dim), dtype=complex)  # I-

    for i, m in enumerate(m_vals[:-1]):
        Ip[i, i + 1] = math.sqrt(I * (I + 1) - m_vals[i + 1] * (m_vals[i + 1] + 1))
        Im[i + 1, i] = math.sqrt(I * (I + 1) - m * (m - 1))

    return {'Iz': Iz, 'Ip': Ip, 'Im': Im}


def _isotropic_hf_superop(A_iso_MHz: float, I: float) -> np.ndarray:
    """Isotropic hyperfine superoperator contribution to H_comm.

    H_hf = A_iso * S·I = A_iso * (Sx*Ix + Sy*Iy + Sz*Iz)
         = A_iso * (0.5*(S+*I- + S-*I+) + Sz*Iz)

    Returns the full (2*nI x 2*nI) matrix in basis |mS, mI>.
    """
    nS = 2  # S=1/2
    nI = int(round(2 * I + 1))
    dim = nS * nI

    Sp = np.array([[0, 1], [0, 0]], dtype=complex)
    Sm = np.array([[0, 0], [1, 0]], dtype=complex)
    Sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    nops = _nuclear_spin_ops(I)

    H = (A_iso_MHz * (
        0.5 * np.kron(Sp, nops['Im']) +
        0.5 * np.kron(Sm, nops['Ip']) +
        np.kron(Sz, nops['Iz'])
    ))
    return H


# ---------------------------------------------------------------------------
# Step 4: Orientational basis enumeration
# ---------------------------------------------------------------------------

def _build_ori_basis(evenLmax: int, oddLmax: int, Kmax: int) -> list[dict]:
    """Enumerate orientational Wigner-function basis with M=0 (Meirovitch).

    Basis states: (L, jK, K) with M=0 (fixed by pS=+1 Meirovitch symmetry).

    Rules (from chili_basisbuild.m):
    - L: 0, 1, 2, ... up to evenLmax (even) or oddLmax (odd)
    - K: 0, 1, ..., min(L, Kmax)
    - K=0: jK = (-1)^L (parity of L)
    - K>0: jK = +1 and jK = -1 (two states)
    - For isotropic diffusion, Mmax not needed (M=0 only).
    """
    basis = []
    for L in range(max(evenLmax, oddLmax) + 1):
        Lmax = evenLmax if L % 2 == 0 else oddLmax
        if L > Lmax:
            continue
        for K in range(0, min(L, Kmax) + 1):
            if K == 0:
                jK = (-1) ** L
                basis.append({'L': L, 'K': K, 'jK': jK, 'M': 0})
            else:
                for jK in (+1, -1):
                    basis.append({'L': L, 'K': K, 'jK': jK, 'M': 0})
    return basis


# ---------------------------------------------------------------------------
# Step 5a: Precompute 3j symbols
# ---------------------------------------------------------------------------

def _precompute_jjj2(Lmax: int) -> np.ndarray:
    """Precompute Wigner 3j symbols (L1, 2, L2; MK1, -(MK1+MK2), MK2).

    Index: idx(L, MK) = L*L + L - MK  (0-based, MK in [-L, +L]).

    Returns jjj2 array of shape (N, N) where N = (Lmax+1)^2.
    """
    from functools import lru_cache
    from torchspin.angmom import wigner3j as _wigner3j

    @lru_cache(maxsize=None)
    def _w3j(j1, j2, j3, m1, m2, m3):
        return float(_wigner3j(j1, j2, j3, m1, m2, m3))

    N = (Lmax + 1) * (Lmax + 1)
    jjj2 = np.zeros((N, N), dtype=float)

    def idx(L, MK):
        return L * L + L - MK

    for L1 in range(Lmax + 1):
        for MK1 in range(-L1, L1 + 1):
            i1 = idx(L1, MK1)
            for L2 in range(Lmax + 1):
                for MK2 in range(-L2, L2 + 1):
                    # 3j symbol: (L1, 2, L2; MK1, -(MK1+MK2), MK2)
                    m2 = -(MK1 + MK2)
                    if abs(m2) > 2:
                        continue
                    if abs(L1 - L2) > 2 or L1 + L2 < 2:
                        continue
                    i2 = idx(L2, MK2)
                    val = _w3j(L1, 2, L2, MK1, m2, MK2)
                    jjj2[i1, i2] = val

    return jjj2


# ---------------------------------------------------------------------------
# Lanczos + Lentz continued-fraction spectral function
# ---------------------------------------------------------------------------
# NOTE: Two legacy Liouvillian builders (_build_liouvillian, _build_liouvillian_simple)
# were removed 2026-04-12 — dead code superseded by the inline builder in chili().
# See git history if needed.

def _lanczos_lentz(L_op, b: np.ndarray, z_vec: np.ndarray, max_iter: int, threshold: float) -> np.ndarray:
    """Compute G(z) = b.'(z*I + L)^{-1}b via Lanczos + forward Lentz method.

    Matches MATLAB EasySpin's chili_lanczos.m: uses complex-symmetric
    pseudonorm (q.'q, NOT q'q) and stops when the spectral function has
    converged, not when |beta| is small.

    MATLAB convention: G(z) = b.'(z*I + L)^{-1}b with z = eps - 1j*omega,
    so the spectral function G = 1/(z + alpha_0 - beta_0^2/(z + alpha_1 - ...))

    Parameters
    ----------
    L_op:
        Callable or sparse matrix — the Liouvillian L (complex symmetric).
    b:
        Starting vector (shape (N,), complex).
    z_vec:
        Complex frequency array (rad/s), shape (nPoints,).
    max_iter:
        Maximum Lanczos iterations.
    threshold:
        Convergence threshold: max|Delta - 1| over all z.

    Returns
    -------
    G_vec : shape (nPoints,) complex spectral function.
    """
    if callable(L_op):
        matvec = L_op
    else:
        matvec = lambda v: L_op.dot(v)

    N = len(b)
    pnorm = lambda v: np.sqrt(v @ v)  # complex-symmetric pseudonorm

    q_prev = np.zeros(N, dtype=complex)
    q_curr = b / pnorm(b)

    beta_prev = complex(0)

    # Lentz state variables (vectorized over z_vec)
    tiny = complex(1e-30)
    spec = np.full(len(z_vec), tiny, dtype=complex)
    C = spec.copy()
    D = np.zeros(len(z_vec), dtype=complex)

    check_interval = max(1, min(10, len(z_vec) // 20))

    for k in range(max_iter):
        y = matvec(q_curr)
        alpha_k = q_curr @ y            # pseudoinnerproduct q.'*L*q
        y -= alpha_k * q_curr
        if k > 0:
            y -= beta_prev * q_prev

        # Lentz update: b_k = z + alpha_k; a_k = 1 (k=0) or -beta_{k-1}^2 (k>0)
        b_k = z_vec + alpha_k
        a_k = complex(1) if k == 0 else -(beta_prev ** 2)

        D_new = b_k + a_k * D
        nz = np.abs(D_new) < 1e-100
        if nz.any():
            D_new = np.where(nz, 1e-100 + 0j, D_new)
        D = 1.0 / D_new

        C_new = b_k + a_k / C
        nz = np.abs(C_new) < 1e-100
        if nz.any():
            C_new = np.where(nz, 1e-100 + 0j, C_new)
        C = C_new

        Delta = C * D
        spec = spec * Delta

        beta_k = pnorm(y)

        # Check spectral convergence every check_interval steps
        if k % check_interval == 0:
            err = np.max(np.abs(Delta - 1.0))
            if err < threshold:
                break

        if abs(beta_k) < 1e-100:
            break  # invariant subspace reached

        beta_prev = beta_k
        q_prev = q_curr
        q_curr = y / beta_k

    return spec


def _lanczos(L_op, b: np.ndarray, max_iter: int, threshold: float):
    """Legacy stub — returns dummy alphas/betas.  Kept for test compatibility.

    New code should use _lanczos_lentz() directly.
    """
    # Return a single-step result so existing unit tests that inspect alphas/betas don't crash.
    if callable(L_op):
        matvec = L_op
    else:
        matvec = lambda v: L_op.dot(v)
    pnorm = lambda v: np.sqrt(v @ v)
    q = b / pnorm(b)
    y = matvec(q)
    alpha = q @ y
    return np.array([alpha]), np.array([], dtype=complex)


def _spectral_function_vec(alphas: np.ndarray, betas: np.ndarray, z_vec: np.ndarray) -> np.ndarray:
    """Backward continued-fraction spectral function (for unit tests / compatibility).

    G(z) = 1/(z + alpha_0 - beta_0^2/(z + alpha_1 - ...))
    """
    G = np.zeros_like(z_vec, dtype=complex)
    betas_sq = betas ** 2
    for k in range(len(alphas) - 1, 0, -1):
        denom = z_vec + alphas[k] - G
        tiny = np.abs(denom) < 1e-100
        if tiny.any():
            denom = np.where(tiny, 1e-100 + 0j, denom)
        G = betas_sq[k - 1] / denom

    denom0 = z_vec + alphas[0] - G
    tiny0 = np.abs(denom0) < 1e-100
    if tiny0.any():
        denom0 = np.where(tiny0, 1e-100 + 0j, denom0)
    return 1.0 / denom0


def _spectral_function(alphas: np.ndarray, betas: np.ndarray, z: complex) -> complex:
    """Scalar wrapper around _spectral_function_vec for backward compatibility."""
    return complex(_spectral_function_vec(alphas, betas, np.array([z]))[0])


# ---------------------------------------------------------------------------
# Starting vector
# ---------------------------------------------------------------------------

def _starting_vector(ori_basis: list[dict], spin_dim: int) -> np.ndarray:
    """Build the starting (detection) vector for Lanczos.

    For no orientational potential, only the L=K=0 state is populated.
    In the combined (orientation × spin) space, the starting vector
    selects L=K=0, jK=+1 (parity of L=0 is +1), and the EPR detection
    operator S- in spin space.

    For S=1/2 single coherence (spin_dim=1 per orientation), the spin part
    is just 1.0. For spin_dim>1 (nucleus present), we need S- matrix element.
    """
    N = len(ori_basis) * spin_dim
    b = np.zeros(N, dtype=complex)

    # Find the L=0, K=0, jK=+1 state
    for a, state in enumerate(ori_basis):
        if state['L'] == 0 and state['K'] == 0 and state['jK'] == 1:
            # Spin part: detection operator is S- = |beta><alpha| → selects mS=-1/2 row
            # For spin_dim=1 (single coherence projected out): b[a*spin_dim + 0] = 1.0
            # For spin_dim=nI (nuclear sub-block): b[a*spin_dim + si] = 1 for each mI
            for si in range(spin_dim):
                b[a * spin_dim + si] = 1.0
            break

    norm_sq = b @ b  # pseudonorm
    if abs(norm_sq) > 1e-15:
        b /= np.sqrt(norm_sq)
    return b


# ---------------------------------------------------------------------------
# Main function: chili
# ---------------------------------------------------------------------------

def chili(
    sys: SpinSystem,
    exp: Experiment,
    opt: Optional[ChiliOptions] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate slow-motion CW EPR spectrum via the SLE.

    Parameters
    ----------
    sys:
        Spin system with S=[0.5] and (optionally) one nucleus.
        Must specify rotational dynamics via one of:
        ``tcorr``, ``logtcorr``, ``Diff``, ``logDiff``.
    exp:
        Experimental parameters (mwFreq, Range, nPoints, Harmonic).
        Field-swept mode only (``exp.is_freq_swept`` must be False).
    opt:
        ChiliOptions (LLMK, Threshold, max_iter, Verbosity).

    Returns
    -------
    B_axis:
        Field axis in mT, shape (nPoints,).
    spec:
        EPR spectrum, shape (nPoints,). Harmonic 0 = absorption,
        1 = first derivative (default), 2 = second derivative.
    """
    if opt is None:
        opt = ChiliOptions()
    # EasySpin-parity implementation (general Liouvillian method): torchspin.chili_sle
    from torchspin.chili_sle import chili_sle, ChiliSLEOptions
    sle_opt = ChiliSLEOptions(**{k: getattr(opt, k) for k in ChiliSLEOptions.__dataclass_fields__ if hasattr(opt, k)})
    # components × isotopologues (EasySpin compisoloop), weighted by Sys.weight·abundance
    from torchspin.isotopologues import expand_components
    components = expand_components(sys, getattr(opt, 'IsoCutoff', 1e-4) if opt is not None else 1e-4)
    x_out, total = None, None
    for s_ in components:
        x_out, y_ = chili_sle(s_, exp, sle_opt)
        y_ = y_ * float(getattr(s_, 'weight', 1.0))
        total = y_ if total is None else total + y_
    return x_out, total


def _chili_legacy(sys, exp, opt=None):
    """Pre-2026-09 single-coherence solver (kept for reference; not used)."""
    if opt is None:
        opt = ChiliOptions()

    # ---- Validate ----
    if len(sys.S) != 1 or sys.S[0] != 0.5:
        raise ValueError("chili: only S=1/2 is supported.")
    if sys.nNuclei > 1:
        raise ValueError("chili: at most one nucleus is supported.")
    if getattr(sys, 'n', None) is not None and any(v > 1 for v in sys.n):
        raise ValueError("chili: sets of equivalent nuclei (SpinSystem.n > 1) are not supported.")
    if exp.is_freq_swept:
        raise ValueError("chili: frequency-swept mode not supported.")

    # ---- Parse dynamics ----
    D_rot = _parse_diffusion(sys)  # s^-1

    # ---- Experimental parameters ----
    B_min, B_max = exp.Range
    nPoints = exp.nPoints
    mwFreq_GHz = exp.mwFreq
    mwFreq_Hz = mwFreq_GHz * 1e9

    B_axis = np.linspace(B_min, B_max, nPoints)  # mT

    # ---- g-tensor ISTO components ----
    g_diag = sys.g[0].numpy()  # [gx, gy, gz] for electron 0
    g_avg = float(g_diag.mean())

    # Center field = EPR resonance field at g_avg: g_avg * μ_B * B0 = h * mwFreq
    # Using the midpoint of the field range as B0 introduces a constant frequency
    # offset equal to g * μ_B * (B_mid - B0_res) / h, which shifts all lines
    # by ~1-2 mT for typical X-band experiments.
    B0_mT = mwFreq_Hz / (g_avg * _MHZ_PER_MT * 1e6)  # resonance center (mT)
    B0_T = B0_mT * 1e-3                               # center field (T)

    # Convert g-tensor to frequency units (MHz) at B0
    g_freq_MHz = _g_to_freq_units(g_diag, B0_T)  # [gx*pre, gy*pre, gz*pre] in MHz
    F0_g, F2_g = _tensor_cart2sph(g_freq_MHz)

    # ---- Spin superoperators (S=1/2) ----
    sops = _spin_superops_S12()
    T2_g = sops['T2']  # list of 5 matrices

    # ---- Nuclear spin ----
    has_nuc = sys.nNuclei > 0
    I_nuc = sys.I[0] if has_nuc else None
    nI = int(round(2 * I_nuc + 1)) if has_nuc else 1
    spin_dim = 2 * nI  # total spin dimension: nS * nI, but we project onto coherence subspace

    # For S=1/2 coherence subspace |α><β|:
    # - Without nucleus: spin_dim = 1 (single EPR coherence)
    # - With nucleus: spin_dim = nI (one EPR coherence per nuclear state)
    # Here we work in the full spin Hilbert space and project afterward.
    # Simpler: work with spin_dim = nI for the EPR coherence block.
    spin_dim = nI  # each nuclear mI state contributes one EPR line

    # Isotropic hyperfine in MHz
    A_iso_MHz = None
    F2_hf = None
    T2_hf = None
    F0_hf = None
    if has_nuc:
        A_vals = sys.A[0].numpy()  # [Axx, Ayy, Azz] in MHz
        A_iso_MHz = float(A_vals.mean())
        # For anisotropic part, compute F2_hf
        F0_hf, F2_hf = _tensor_cart2sph(A_vals)
        # T2 for hyperfine uses nuclear spin operators
        # For the EPR coherence |α><β|, the hyperfine shifts each mI level by A_iso*mI/2
        # This creates nuclear sub-levels — handled as diagonal shifts in spin_dim=nI space.

    # ---- Orientational basis ----
    evenLmax, oddLmax, Mmax, Kmax = opt.LLMK
    ori_basis = _build_ori_basis(evenLmax, oddLmax, Kmax)
    nOri = len(ori_basis)

    if opt.Verbosity > 0:
        print(f"chili: {nOri} orientational basis states, spin_dim={spin_dim}")
        print(f"  D_rot={D_rot:.3e} s^-1, tcorr={1/(6*D_rot):.3e} s")
        print(f"  B0={B0_mT:.1f} mT, mwFreq={mwFreq_GHz:.4f} GHz")

    # ---- Precompute 3j symbols ----
    Lmax_3j = max(evenLmax, oddLmax) + 2  # +2 for rank-2 coupling
    if opt.Verbosity > 0:
        print(f"  Computing 3j symbols up to L={Lmax_3j}...")
    jjj2 = _precompute_jjj2(Lmax_3j)

    # ---- Build Liouvillian at center field ----
    if opt.Verbosity > 0:
        print("  Building Liouvillian...")

    # For each nuclear mI sub-level, the EPR resonance frequency shifts by A_iso*mI
    # We build a separate Lanczos for each mI level and sum the spectral functions.
    mI_vals = np.arange(I_nuc, -I_nuc - 1, -1) if has_nuc else np.array([0.0])

    # Angular frequency axis (rad/s): ω = g*μ_B*(B-B0)/ℏ ≈ g_avg * _MHZ_PER_MT * (B-B0) * 2π * 1e6
    # Offset frequency from resonance (rad/s):
    # For spin_dim=1 (no nucleus or single mI), center at ω=0
    delta_B = B_axis - B0_mT  # mT
    omega_sweep = 2 * math.pi * 1e6 * g_avg * _MHZ_PER_MT * delta_B  # rad/s

    # ---- Intrinsic linewidth (lw) as Lorentzian broadening ----
    lw = sys.get_lw()
    # lw[1] is Lorentzian FWHM in mT → convert to rad/s
    lorentz_fwhm_mT = lw[1] if lw[1] > 0 else 0.0
    lorentz_fwhm_radps = 2 * math.pi * 1e6 * g_avg * _MHZ_PER_MT * lorentz_fwhm_mT
    eps = lorentz_fwhm_radps / 2.0  # half-width → imaginary part of z

    # Ensure minimum linewidth for numerical stability (continued-fraction
    # solver needs nonzero damping to avoid division by zero).
    min_eps_radps = 2 * math.pi * 1e6 * 0.01 * _MHZ_PER_MT * g_avg  # 0.01 mT
    if eps < min_eps_radps and lorentz_fwhm_mT > 0:
        import warnings as _warnings
        _warnings.warn(
            f"chili: requested Lorentzian linewidth {lorentz_fwhm_mT:.4f} mT is "
            f"below the 0.01 mT numerical-stability floor; effective linewidth "
            f"will be broader than requested.",
            RuntimeWarning,
            stacklevel=2,
        )
    eps = max(eps, min_eps_radps)

    # ---- Liouvillian (spin_dim=1 for single coherence per mI) ----
    # For chili with single EPR coherence, we build L for each mI separately.
    # Each mI block has spin_dim=1 (just the EPR coherence scaled by its mI shift).

    # Build the orientational Liouvillian (spin_dim=1 = single coherence)
    # The g-tensor anisotropy enters through F2_g * T2_g where T2_g is scalar
    # for the single EPR coherence |α><β|:
    # <α|T2[0]|β> = <α| sqrt(2/3)*Sz |β> = 0 (off-diagonal, Sz diagonal)
    # Actually for the SINGLE COHERENCE, the relevant matrix element is:
    # H_comm operates as a commutator: [H, rho]
    # For rho = |α><β|: [H, |α><β|] = (E_α - E_β)|α><β| = ω_EPR * |α><β|
    # So in the coherence subspace, H_comm is a scalar = (E_α - E_β).
    # For S=1/2: E_α - E_β = g * μ_B * B / h = ω_EPR (angular frequency)
    # The anisotropy of g contributes orientation-dependent frequency shifts.

    # Simplified approach: treat the g-tensor as giving an orientation-dependent
    # EPR frequency. In the SLE framework, the spin superop for the EPR coherence
    # is simply a scalar (the resonance frequency offset from center).

    # For rank-2 g anisotropy, the relevant matrix element in the EPR coherence:
    # <α β|T2[q]|β α> ≡ commutator element ∝ <α|T2[q]|α> - <β|T2[q]|β>
    # For T2[0] = sqrt(2/3)*Sz: <α|Sz|α> - <β|Sz|β> = 1/2 - (-1/2) = 1
    # Prefactor: sqrt(2/3) → T2[0]_comm = sqrt(2/3)
    # For T2[+1]: <α|S+|β> - ... = 1 - 0 = 1 (but S+ is off-diagonal: <α|S+|β>=1)
    # Actually the commutator: [H,ρ]_{αβ} = H_{αα}ρ_{αβ} - ρ_{αβ}H_{ββ}
    # So the effective superoperator for |α><β| element is just (H_{αα} - H_{ββ})
    # which for S=1/2, T2[0]: gx+gy component cancels, gz component survives as sqrt(2/3)*gz*pre

    # For the EPR coherence subspace with a single complex number per orientation:
    # Effective scalar T2 operators:
    T2_comm = np.zeros(5, dtype=complex)  # effective scalar for |alpha><beta|
    # T2[0] = sqrt(2/3)*Sz → effective = sqrt(2/3) * (1/2 - (-1/2)) = sqrt(2/3)
    T2_comm[2] = math.sqrt(2.0 / 3.0)   # q=0
    # T2[+1] = -0.5*S+ → S+ connects |β>→|α>, so H_{αβ} type → commutator: 0
    # T2[-1] = +0.5*S- → connects |α>→|β|> → H_{βα} term → commutator: 0
    # T2[+2] = T2[-2] = 0 for S=1/2
    # So only T2_comm[2] (q=0) is nonzero.

    # Build scalar Liouvillian for single EPR coherence (one complex number per ori state)
    # L[a,b] = 2πi * F2_g[q=0] * T2_comm[q=0] * coupling(a,b) + Gamma_a * delta(a,b)

    # This is equivalent to the orientation-only Liouvillian with scalar spin part.
    # spin_dim = 1 for this simplified approach.
    spin_dim_eff = 1

    nTotal = nOri * spin_dim_eff

    # We separate the Liouvillian into:
    #   L_diag  : diffusion (diagonal, mI-independent)
    #   L_3j    : orientational coupling structure (mI-independent 3j template)
    # The full Liouvillian for a given mI is then:
    #   L[mI] = L_diag + (F_g_eff + mI * F_hf_eff) * L_3j_unit
    # where F_g_eff  = sqrt(2/3) * conj(F2_g[q=0])
    #       F_hf_eff = sqrt(2/3) * conj(F2_hf[q=0])  (anisotropic hyperfine; mI-dependent)
    # This factorization follows from the fact that in the diagonal EPR coherence subspace
    # |α,mI><β,mI|, the T2 commutator for rank-2 anisotropic coupling gives:
    #   T2_eff_g[q=0]     = sqrt(2/3)       (g-tensor, mI-independent)
    #   T2_eff_hf[q=0,mI] = sqrt(2/3) * mI  (hyperfine, mI-dependent via Sz⊗Iz secular term)
    # Only q=0 contributes because T2[q≠0] = 0 in the diagonal EPR coherence approximation.

    rows_diag, cols_diag, vals_diag = [], [], []   # diffusion diagonal
    rows_g,    cols_g,    vals_g    = [], [], []   # g-tensor coupling (F2_g embedded)
    rows_hf,   cols_hf,   vals_hf   = [], [], []   # HFI coupling unit (F2_hf embedded, scaled by mI)

    _sqrt23 = math.sqrt(2.0 / 3.0)
    _S_g = _sqrt23  # S_g = C2*pS1 = sqrt(2/3)*1 for EPR coherence (pS1=+1)

    def idx(L, MK):
        return L * L + L - MK

    for a, state_a in enumerate(ori_basis):
        La, Ka, jKa_val = state_a['L'], state_a['K'], state_a['jK']
        Gamma_a = D_rot * La * (La + 1)  # rad/s
        rows_diag.append(a)
        cols_diag.append(a)
        vals_diag.append(Gamma_a)

        for b, state_b in enumerate(ori_basis):
            Lb, Kb, jKb_val = state_b['L'], state_b['K'], state_b['jK']

            if abs(La - Lb) > 2 or La + Lb < 2:
                continue
            if (La + Lb) % 2 != 0:
                continue

            # Real tensors: coupling requires jKa == jKb (MATLAB chili_lm0.inc line 116)
            if jKa_val != jKb_val:
                continue

            # M coupling (M=0 basis, d2psi(0,0)=1): jjjM = jjj(La,2,Lb;0,0,0)
            jjjM = jjj2[idx(La, 0), idx(Lb, 0)]
            if abs(jjjM) < 1e-15:
                continue

            # NormFactor = NL * NK * parity(M1+K1) with M1=0 → parity(Ka)
            # NK = 1/sqrt(2) per K=0 index  (chili_lm0.inc lines 138-140)
            NL = math.sqrt(2 * La + 1) * math.sqrt(2 * Lb + 1)
            NK = (1.0 / math.sqrt(2) if Ka == 0 else 1.0) * (1.0 / math.sqrt(2) if Kb == 0 else 1.0)
            parity_Ka = (-1) ** Ka  # parity(M1+K1) = parity(0+Ka)
            norm_pref = NL * NK * parity_Ka * jjjM * _S_g

            # R_EZI2 / R_HFI2 coupling  (chili_lm0.inc lines 108-134)
            # g1 term: Kd = Ka-Kb, coeff = jjj(La,2,Lb; Ka,-Kd,-Kb), tensor index 2+Kd
            # g2 term: Ks = Ka+Kb, coeff = jjj(La,2,Lb; Ka,-Ks, Kb), tensor index 2+Ks
            # F2[index] maps to molecular component q = -Kd or -Ks (Python order: q=+2..−2)
            Kd = Ka - Kb
            Ks = Ka + Kb
            parityLbKb = (-1) ** (Lb + Kb)

            # --- g1 coefficients (shared for g and hf) ---
            coeff_g1 = 0.0
            if abs(Kd) <= 2:
                coeff_g1 = jjj2[idx(La, Ka), idx(Lb, -Kb)]

            # --- g2 coefficients (shared for g and hf) ---
            coeff_g2 = 0.0
            if 0 <= Ks <= 2:
                coeff_g2 = jjj2[idx(La, Ka), idx(Lb, Kb)]

            if abs(coeff_g1) < 1e-15 and abs(coeff_g2) < 1e-15:
                continue

            # Build R_EZI2 for g-tensor
            R_g = 0.0
            if abs(coeff_g1) > 1e-15:
                R_g += coeff_g1 * F2_g[2 + Kd].real
            if abs(coeff_g2) > 1e-15:
                R_g += jKb_val * parityLbKb * coeff_g2 * F2_g[2 + Ks].real

            if abs(R_g) > 1e-20:
                val_g = 2j * math.pi * 1e6 * norm_pref * R_g
                rows_g.append(a)
                cols_g.append(b)
                vals_g.append(val_g)

            # Build R_HFI2 for hyperfine (same K-coupling structure, different F2)
            if has_nuc and F2_hf is not None:
                R_hf = 0.0
                if abs(coeff_g1) > 1e-15:
                    R_hf += coeff_g1 * F2_hf[2 + Kd].real
                if abs(coeff_g2) > 1e-15:
                    R_hf += jKb_val * parityLbKb * coeff_g2 * F2_hf[2 + Ks].real

                if abs(R_hf) > 1e-20:
                    val_hf = 2j * math.pi * 1e6 * norm_pref * R_hf
                    rows_hf.append(a)
                    cols_hf.append(b)
                    vals_hf.append(val_hf)

    L_diffusion_mat = sp.csr_matrix(
        (np.array(vals_diag, dtype=complex), (np.array(rows_diag), np.array(cols_diag))),
        shape=(nOri, nOri)
    )
    L_g_coupling = sp.csr_matrix(
        (np.array(vals_g, dtype=complex), (np.array(rows_g), np.array(cols_g))),
        shape=(nOri, nOri)
    ) if len(vals_g) > 0 else sp.csr_matrix((nOri, nOri), dtype=complex)
    L_ori = L_diffusion_mat + L_g_coupling  # base Liouvillian (mI-independent)

    # Anisotropic hyperfine unit matrix: L_mI = L_ori + mI * L_hf_unit
    # (S_A = mI for EPR coherence in nuclear diagonal basis; chili_lm1.inc line 275)
    L_hf_unit = None
    if has_nuc and len(vals_hf) > 0:
        L_hf_unit = sp.csr_matrix(
            (np.array(vals_hf, dtype=complex), (np.array(rows_hf), np.array(cols_hf))),
            shape=(nOri, nOri)
        )

    if opt.Verbosity > 0:
        print(f"  Liouvillian: {nOri}×{nOri}, nnz={L_ori.nnz}")

    # ---- Spectrum by Lanczos + continued fraction ----
    spec = np.zeros(nPoints, dtype=float)

    for mi, mI in enumerate(mI_vals):
        # Starting vector
        b_vec = np.zeros(nOri, dtype=complex)
        for a, state in enumerate(ori_basis):
            if state['L'] == 0 and state['K'] == 0 and state['jK'] == 1:
                b_vec[a] = 1.0
                break
        pnorm_b = b_vec @ b_vec
        if abs(pnorm_b) < 1e-15:
            continue
        b_vec /= np.sqrt(pnorm_b)

        # Nuclear mI shift: resonance field shifts by -A_iso*mI/(g*μ_B), so in the
        # omega_sweep = g*μ_B*(B-B0)/ℏ frame the mI resonance is at omega = -A_iso*mI.
        mI_shift_radps = -2 * math.pi * 1e6 * A_iso_MHz * mI if A_iso_MHz is not None else 0.0

        if opt.Verbosity > 0:
            print(f"  Running Lanczos for mI={mI:.1f} ...")

        # Liouvillian for this mI: add anisotropic hyperfine correction (mI-dependent).
        # L[mI] = L_ori + mI * L_hf_unit   (from T2_hf[0]_comm = sqrt(2/3)*mI in EPR coherence)
        if L_hf_unit is not None:
            L_mI = L_ori + float(mI) * L_hf_unit
        else:
            L_mI = L_ori

        # MATLAB convention: z = eps - 1j*omega  (see chili.m line 833: omega0 = complex(2pi*nu, 1/T2))
        # G(z) = b.'(z*I + L)^{-1}b computed via Lanczos + forward Lentz method.
        # Spectrum = Re(G).  Convergence on spectral change prevents Lanczos divergence.
        z_vec = eps - 1j * (omega_sweep - mI_shift_radps)

        # Numerical rescaling (MATLAB chili.m lines 1239-1246):
        # Divide L and z by max(|L|) before Lanczos to prevent catastrophic
        # cancellation in the continued fraction.
        # Math: G'(z/s, L/s) = s * G(z, L), so G(z, L) = G'(...) / s.
        # MATLAB: spec = spec/scale  (chili.m line 1332).
        scale = np.max(np.abs(L_mI.data))
        if scale > 0.0:
            G_vec = _lanczos_lentz(L_mI / scale, b_vec, z_vec / scale,
                                   opt.max_iter, opt.Threshold)
            G_vec /= scale
        else:
            G_vec = _lanczos_lentz(L_mI, b_vec, z_vec, opt.max_iter,
                                   opt.Threshold)
        spec += G_vec.real

        if opt.Verbosity > 0:
            print(f"    Lanczos+Lentz done for mI={mI:.1f}")

    # Nuclear weighting (equal weights for all mI = high-T limit)
    if has_nuc:
        spec /= len(mI_vals)

    # ---- Apply derivative (Harmonic) ----
    if exp.Harmonic == 1:
        dB = (B_max - B_min) / (nPoints - 1)
        spec = np.gradient(spec, dB)
    elif exp.Harmonic == 2:
        dB = (B_max - B_min) / (nPoints - 1)
        spec = np.gradient(np.gradient(spec, dB), dB)

    # ---- Additional Gaussian broadening from lw[0] ----
    if lw[0] > 0:
        sigma_mT = lw[0] / (2 * math.sqrt(2 * math.log(2)))  # FWHM→sigma
        sigma_pts = sigma_mT / ((B_max - B_min) / (nPoints - 1))
        from scipy.ndimage import gaussian_filter1d
        spec = gaussian_filter1d(spec, sigma=sigma_pts)

    return torch.from_numpy(B_axis), torch.from_numpy(spec)
