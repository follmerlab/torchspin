"""Excitation profile calculation for arbitrary pulses.

Port of EasySpin's ``exciteprofile.m``.

Computes the excitation profile (Mx, My, Mz vs. frequency offset) for
a given pulse waveform by propagating the S=1/2 Bloch equations.

Example
-------
>>> import numpy as np
>>> from torchspin.exciteprofile import exciteprofile
>>> t = np.linspace(0, 0.1, 500)
>>> IQ = 25 * np.ones_like(t)  # rectangular pulse, 25 MHz amplitude
>>> offsets, Mag = exciteprofile(t, IQ)
"""
from __future__ import annotations

import numpy as np

from torchspin.fdaxis import fdaxis


def exciteprofile(
    t: np.ndarray,
    IQ: np.ndarray,
    offsets: np.ndarray | None = None,
    nOffsets: int = 201,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute excitation profile for an arbitrary pulse.

    Parameters
    ----------
    t:
        Time axis in µs.
    IQ:
        Complex IQ pulse waveform.  ``real(IQ)`` is the I (x) component,
        ``imag(IQ)`` is the Q (y) component, both in MHz.
    offsets:
        Frequency offset axis in MHz.  If ``None``, determined automatically
        from the pulse bandwidth (approximately ±BW centered at the pulse
        center frequency).
    nOffsets:
        Number of offset points when *offsets* is auto-generated (default 201).

    Returns
    -------
    offsets:
        Frequency offset axis in MHz.
    Mag:
        Excitation profile, shape ``(3, nOffsets)``.  Rows are
        Mx/M0, My/M0, Mz/M0.
    """
    t = np.asarray(t, dtype=np.float64).ravel()
    IQ = np.asarray(IQ, dtype=np.complex128).ravel()

    if len(t) != len(IQ):
        raise ValueError(
            f"Time axis ({len(t)}) and IQ ({len(IQ)}) must have equal lengths.")

    nPoints = len(t)
    dt = t[1] - t[0]

    # Auto-determine offset range from pulse bandwidth
    if offsets is None:
        zf = max(1024, 4 * (2 ** int(np.ceil(np.log2(nPoints)))))
        IQft = np.abs(np.fft.fftshift(np.fft.fft(IQ, zf)))
        f = fdaxis(dt, zf)
        indbw = np.where(IQft > 0.5 * np.max(IQft))[0]
        if len(indbw) > 0:
            BW = abs(f[indbw[-1]] - f[indbw[0]])
            center_freq = 0.5 * (f[indbw[-1]] + f[indbw[0]])
        else:
            BW = 1.0 / dt
            center_freq = 0.0
        offsets = np.linspace(-BW, BW, nOffsets) + center_freq
    else:
        offsets = np.asarray(offsets, dtype=np.float64).ravel()

    nOff = len(offsets)

    # Spin-1/2 operators (2x2)
    Sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    Sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    Sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)

    # Equilibrium density matrix: ρ₀ = -Sz
    Density0 = -Sz

    I_signal = np.real(IQ)
    Q_signal = np.imag(IQ)

    Mag = np.zeros((3, nOff))
    eye2 = np.eye(2, dtype=complex)

    for iOff in range(nOff):
        Ham0 = offsets[iOff] * Sz

        if np.min(np.abs(IQ)) == np.max(np.abs(IQ)) and np.max(np.abs(IQ)) > 0:
            # Rectangular pulse: single propagator
            Ham = I_signal[0] * Sx + Q_signal[0] * Sy + Ham0
            tp_eff = dt * (nPoints - 1)
            UPulse = _fast_expm2(-2j * np.pi * tp_eff * Ham, eye2)
        else:
            # General pulse: step-by-step propagation
            UPulse = eye2.copy()
            for it in range(nPoints - 1):
                Ham = I_signal[it] * Sx + Q_signal[it] * Sy + Ham0
                dU = _fast_expm2(-2j * np.pi * dt * Ham, eye2)
                UPulse = dU @ UPulse

        # Propagate density matrix
        Density = UPulse @ Density0 @ UPulse.conj().T

        # Extract observables: Tr(2*S_i * ρ) = -2*Tr(S_i * ρ) for M_i/M_0
        Mag[0, iOff] = -2 * np.real(np.sum(Sx * Density.T))
        Mag[1, iOff] = -2 * np.real(np.sum(Sy * Density.T))
        Mag[2, iOff] = -2 * np.real(np.sum(Sz * Density.T))

    return offsets, Mag


def _fast_expm2(M: np.ndarray, eye2: np.ndarray) -> np.ndarray:
    """Fast matrix exponential for traceless antihermitian 2x2 matrix.

    Uses the cosh/sinh formula: exp(M) = cosh(q)*I + (sinh(q)/q)*M
    where q = sqrt(M[0,0]² - |M[0,1]|²).
    """
    q_sq = M[0, 0] ** 2 - abs(M[0, 1]) ** 2
    q = np.sqrt(q_sq)
    if abs(q) < 1e-10:
        return eye2 + M
    return np.cosh(q) * eye2 + (np.sinh(q) / q) * M
