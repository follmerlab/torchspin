"""Steady-state Bloch equation solutions under field modulation.

Port of EasySpin's ``blochsteady.m``.

Computes the periodic steady-state solution of the Bloch equations for a
single S=1/2 spin in the presence of CW microwave irradiation and a sinusoidal
field modulation.  The solution is obtained in the frequency domain via a
sparse pentadiagonal linear system, then converted to the time domain by
inverse Fourier transform.

References
----------
Tseitlin, Eaton, Eaton, Appl. Magn. Reson. 44, 1373–1379 (2013).
https://doi.org/10.1007/s00723-013-0494-2
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from torchspin.constants import BMAGN, PLANCK

# ħ = h / (2π)
_HBAR = PLANCK / (2.0 * math.pi)


@dataclass
class BlochOptions:
    """Computational options for :func:`blochsteady`.

    Parameters
    ----------
    kmax:
        Maximum Fourier order. ``None`` = auto-determined from T2 and modulation
        depth (recommended).
    nPoints:
        Number of output time points. ``None`` = auto (2·kmax − 1).
    method:
        Time-domain reconstruction method.
        ``'fft'`` (default) — inverse FFT (faster).
        ``'td'`` — explicit complex-exponential sum (slower, for verification).
    """
    kmax: Optional[int] = None
    nPoints: Optional[int] = None
    method: str = 'fft'


def blochsteady(
    g: float,
    T1: float,
    T2: float,
    DeltaB0: float,
    B1: float,
    modAmp: float,
    modFreq: float,
    opt: Optional[BlochOptions] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the steady-state Bloch magnetization under sinusoidal field modulation.

    Parameters
    ----------
    g:
        Electron g-value (scalar, S=1/2).
    T1:
        Longitudinal relaxation time, µs.
    T2:
        Transverse relaxation time, µs.  Must satisfy T2 ≤ T1.
    DeltaB0:
        Static field offset from resonance, mT.  Positive = above resonance.
    B1:
        Microwave field amplitude (rotating frame), mT.
    modAmp:
        Peak-to-peak modulation field amplitude, mT.
    modFreq:
        Modulation frequency, kHz.
    opt:
        :class:`BlochOptions`.  ``None`` uses defaults.

    Returns
    -------
    t : ndarray, shape (nPoints,)
        Time axis over one modulation period, µs.
    Mx : ndarray, shape (nPoints,)
        Dispersion (in-phase) component of transverse magnetization.
    My : ndarray, shape (nPoints,)
        Absorption (quadrature) component of transverse magnetization.
    Mz : ndarray, shape (nPoints,)
        Longitudinal magnetization.

    Notes
    -----
    The field modulation is Bmod(t) = (modAmp/2) · cos(ωm·t), so modAmp is the
    peak-to-peak amplitude.  The steady-state Mx, My, Mz are periodic with
    period T = 1/modFreq.

    M0 (equilibrium magnetization) is normalized to 1.

    Examples
    --------
    >>> from torchspin.blochsteady import blochsteady
    >>> t, Mx, My, Mz = blochsteady(g=2.0, T1=10.0, T2=1.0,
    ...                              DeltaB0=0.0, B1=0.01,
    ...                              modAmp=0.5, modFreq=100.0)
    """
    if opt is None:
        opt = BlochOptions()

    # ── Input checks ─────────────────────────────────────────────────────────
    if not np.isscalar(g) or g <= 0:
        raise ValueError("blochsteady: g must be a single positive number.")
    if T1 <= 0:
        raise ValueError("blochsteady: T1 must be positive.")
    if T2 <= 0:
        raise ValueError("blochsteady: T2 must be positive.")
    if T2 > T1:
        raise ValueError("blochsteady: T2 cannot exceed T1.")
    if B1 <= 0:
        raise ValueError("blochsteady: B1 must be positive.")
    if modAmp <= 0:
        raise ValueError("blochsteady: modAmp must be positive.")
    if modFreq <= 0:
        raise ValueError("blochsteady: modFreq must be positive.")

    # ── Unit conversions ──────────────────────────────────────────────────────
    T1_s    = T1 * 1e-6          # µs → s
    T2_s    = T2 * 1e-6          # µs → s
    dB0_T   = DeltaB0 * 1e-3     # mT → T
    B1_T    = B1 * 1e-3           # mT → T
    mAmp_T  = modAmp * 1e-3      # mT → T
    mFreq_Hz = modFreq * 1e3     # kHz → Hz
    omegam  = 2.0 * math.pi * mFreq_Hz  # rad/s

    M0 = 1.0
    gamma = BMAGN / _HBAR * g   # rad / (s·T)

    # Derived coupling coefficients (frequency domain)
    b_coef = gamma * dB0_T       # resonance offset, rad/s
    c_coef = gamma * B1_T        # microwave coupling, rad/s
    d_coef = gamma * mAmp_T / 4.0  # modulation coupling (per 3j), rad/s

    # ── Determine kmax ────────────────────────────────────────────────────────
    if opt.kmax is not None:
        kmax = int(opt.kmax)
    else:
        # Estimate 1: based on maximum field offset within modulation cycle
        max_field_off = max(mAmp_T / 2.0 - abs(dB0_T), abs(dB0_T) + mAmp_T / 2.0)
        max_field_off = min(max_field_off, mAmp_T)
        max_freq_off = (gamma / (2.0 * math.pi)) * max_field_off
        kmax_field = math.ceil(max_freq_off / mFreq_Hz * 1.4)

        # Estimate 2: based on T2 envelope threshold 1e-6
        kmax_T2 = math.ceil(-math.log(1e-6) / 2.0 * gamma * mAmp_T * T2_s)

        kmax = max(max(kmax_field, kmax_T2), 20)

    # ── Build frequency-domain auxiliary arrays ───────────────────────────────
    # k = -(kmax+1), ..., kmax+1  (length 2*kmax+3)
    k_full = np.arange(-(kmax + 1), kmax + 2)    # shape (2*kmax+3,)
    a_full = 1j * k_full * omegam                 # i·k·ωm

    tau1_full = 1.0 / (a_full + 1.0 / T1_s)      # shape (2*kmax+3,)
    tau2_full = 1.0 / (a_full + 1.0 / T2_s)

    # First q selection: MATLAB q = (1:2*kmax+1)+1 = 2:2*kmax+2 (1-indexed)
    # → Python indices 1 to 2*kmax+1 (inclusive) = slice(1, 2*kmax+2)
    # These build the nRows = 2*kmax+1 diagonal vectors.
    q   = slice(1, 2 * kmax + 2)   # main
    qm1 = slice(0, 2 * kmax + 1)   # q-1
    qp1 = slice(2, 2 * kmax + 3)   # q+1

    tau2_q   = tau2_full[q]         # shape (2*kmax+1,)
    tau2_qm1 = tau2_full[qm1]
    tau2_qp1 = tau2_full[qp1]
    tau1_q   = tau1_full[q]
    a_q      = a_full[q]

    c2m_v = d_coef**2 * tau2_qm1
    c2p_v = d_coef**2 * tau2_qp1
    c1m_v = b_coef * d_coef * (tau2_q + tau2_qm1)
    c1p_v = b_coef * d_coef * (tau2_q + tau2_qp1)
    c0_v  = (a_q + 1.0 / T2_s
             + b_coef**2 * tau2_q
             + c_coef**2 * tau1_q
             + d_coef**2 * (tau2_qm1 + tau2_qp1))

    # RHS scalar: tau1 at k=0; MATLAB tau1(kmax+2) [1-indexed] = Python tau1_full[kmax+1]
    cL = c_coef * M0 * tau1_full[kmax + 1] / T1_s

    # ── Assemble pentadiagonal sparse matrix ──────────────────────────────────
    # After MATLAB's column-selection trick P(:, 3:end-2), the diagonals become:
    #   c2m_v[2:]   on diagonal -2
    #   c1m_v[1:]   on diagonal -1
    #   c0_v        on diagonal  0
    #   c1p_v[:-1]  on diagonal +1
    #   c2p_v[:-2]  on diagonal +2
    nRows = 2 * kmax + 1
    P = sp.diags(
        [c2m_v[2:], c1m_v[1:], c0_v, c1p_v[:-1], c2p_v[:-2]],
        [-2, -1, 0, 1, 2],
        shape=(nRows, nRows),
        dtype=complex,
        format='csc',
    )

    C0 = np.zeros(nRows, dtype=complex)
    C0[kmax] = cL   # k=0 element sits at index kmax in Y

    # ── Solve sparse linear system for Y (Fourier coefficients of My) ─────────
    Y = spsolve(P, C0)   # shape (nRows,) = (2*kmax+1,)

    # ── Compute Xk, Yk, Zk (second q selection: MATLAB q = 2:2*kmax) ─────────
    # Python indices into Y: 1 to 2*kmax-1; length 2*kmax-1
    Yk    = Y[1: 2 * kmax]         # absorption Fourier coefficients
    Y_prv = Y[0: 2 * kmax - 1]     # Y(q-1)
    Y_nxt = Y[2: 2 * kmax + 1]     # Y(q+1)

    # tau at q+1 (MATLAB) = Python indices 2..2*kmax of tau2_full
    tau2_q2p1 = tau2_full[2: 2 * kmax + 1]   # length 2*kmax-1
    tau1_q2p1 = tau1_full[2: 2 * kmax + 1]

    Xk = tau2_q2p1 * (b_coef * Yk + d_coef * (Y_prv + Y_nxt))

    delta_k0 = np.zeros(2 * kmax - 1, dtype=complex)
    delta_k0[kmax - 1] = 1.0   # k=0 position in the 2nd selection

    Zk = tau1_q2p1 * (-c_coef * Yk + M0 * delta_k0 / T1_s)

    # ── Time-domain reconstruction ─────────────────────────────────────────────
    npts_fft = len(Yk)   # = 2*kmax-1
    npts_out = opt.nPoints if opt.nPoints is not None else npts_fft

    if opt.method == 'fft':
        if npts_out <= npts_fft:
            My_td = npts_fft * np.real(np.fft.ifft(np.fft.ifftshift(Yk)))
            Mx_td = npts_fft * np.real(np.fft.ifft(np.fft.ifftshift(Xk)))
            Mz_td = npts_fft * np.real(np.fft.ifft(np.fft.ifftshift(Zk)))
            if npts_out < npts_fft:
                t_full = np.linspace(0.0, 1.0 / mFreq_Hz, npts_fft, endpoint=False)
                t_req  = np.linspace(0.0, 1.0 / mFreq_Hz, npts_out, endpoint=False)
                My_td = np.interp(t_req, t_full, My_td)
                Mx_td = np.interp(t_req, t_full, Mx_td)
                Mz_td = np.interp(t_req, t_full, Mz_td)
        else:
            # Zero-pad in the middle of the shifted spectrum (MATLAB fill trick)
            def _ft_zeropad(m):
                m_sh = np.fft.ifftshift(m)   # k=0 moved to index 0
                n_zeros = npts_out - len(m)
                padded = np.concatenate([m_sh[:kmax],
                                         np.zeros(n_zeros, dtype=complex),
                                         m_sh[kmax:]])
                return npts_out * np.real(np.fft.ifft(padded))

            My_td = _ft_zeropad(Yk)
            Mx_td = _ft_zeropad(Xk)
            Mz_td = _ft_zeropad(Zk)

    elif opt.method == 'td':
        t_s = np.linspace(0.0, 1.0 / mFreq_Hz, npts_out, endpoint=False)
        k_range = np.arange(-(kmax - 1), kmax)  # k = -(kmax-1)..kmax-1
        phases = np.exp(1j * np.outer(k_range * omegam, t_s))  # (2*kmax-1, npts)
        My_td = np.real(Yk @ phases)
        Mx_td = np.real(Xk @ phases)
        Mz_td = np.real(Zk @ phases)
    else:
        raise ValueError(f"blochsteady: unknown method '{opt.method}'; use 'fft' or 'td'.")

    t_out = np.linspace(0.0, 1.0 / mFreq_Hz, npts_out, endpoint=False) * 1e6  # µs

    return t_out, Mx_td, My_td, Mz_td
