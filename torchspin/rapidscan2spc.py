"""Conversion of sinusoidal rapid-scan signal to EPR spectrum.

Port of EasySpin's ``rapidscan2spc.m``.

Implements the deconvolution method from:
  Tseitlin, Rinard, Quine, Eaton, Eaton
  "Deconvolution of sinusoidal rapid EPR scans"
  J. Magn. Reson. 208, 279-283, 2011
  https://doi.org/10.1016/j.jmr.2010.11.015

Example
-------
>>> import numpy as np
>>> from torchspin.rapidscan2spc import rapidscan2spc
>>> N = 1024
>>> t = np.linspace(0, 1, N, endpoint=False)
>>> signal = np.exp(1j * 0.1 * np.cos(2 * np.pi * t))  # dummy
>>> dB, spc = rapidscan2spc(signal, rsAmp=10.0, rsFreq=100.0)
"""
from __future__ import annotations

import numpy as np

from torchspin.constants import BMAGN, GFREE, HBAR
from torchspin.fdaxis import fdaxis
from torchspin.utils import unitconvert


def rapidscan2spc(
    M: np.ndarray,
    rsAmp: float,
    rsFreq: float,
    g: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert sinusoidal rapid-scan signal to field-swept EPR spectrum.

    Parameters
    ----------
    M:
        Complex time-domain rapid-scan signal (My + 1j*Mx) over one period
        of the field modulation.  Must be a 1-D complex vector with an even
        number of points.
    rsAmp:
        Peak-to-peak modulation amplitude in mT.
    rsFreq:
        Modulation frequency in kHz.
    g:
        g-value (default: free-electron g).

    Returns
    -------
    dB:
        Field offset axis in mT.
    spc:
        Complex spectrum (real = absorption, imag = dispersion).
    """
    if g is None:
        g = GFREE

    M_raw = np.asarray(M)
    if not np.iscomplexobj(M_raw):
        raise ValueError("Rapid-scan signal must be complex (absorption + 1j*dispersion).")
    M = M_raw.astype(np.complex128).ravel()
    N = len(M)

    if N % 2 != 0:
        raise ValueError("Rapid-scan signal must contain an even number of points.")

    # Construct time vector over one period
    rsFreq_Hz = rsFreq * 1e3  # kHz -> Hz
    T_period = 1.0 / rsFreq_Hz  # s
    t = np.arange(N, dtype=np.float64) / N * T_period  # s

    # Convert modulation amplitude to rad/s
    rsAmp_mT = rsAmp
    A_T = rsAmp_mT / 1e3 / 2.0  # mT -> T, peak (half of peak-to-peak)
    gamma = g * BMAGN / HBAR  # rad/s/T
    A_rad = gamma * A_T  # rad/s

    # Phase transformation
    omega = 2.0 * np.pi * rsFreq_Hz
    ph = A_rad * np.sin(omega * t) / omega  # integral of A*cos(omega*t)
    phf = np.exp(-1j * ph)
    Mph = M * phf

    # Fourier deconvolution of the two half-periods and add up
    half = N // 2
    def fourierdeconv(idx):
        return np.fft.fftshift(np.fft.fft(Mph[idx]) / np.fft.fft(phf[idx]))

    spc = fourierdeconv(slice(0, half)) + fourierdeconv(slice(half, N))

    # Generate field axis and truncate to modulation amplitude range
    nu = fdaxis(t[:half]) / 1e6  # Hz -> MHz
    dB = -unitconvert(nu, 'MHz->mT', g)

    idx = np.abs(dB) <= rsAmp_mT / 2
    dB = dB[idx]
    spc = spc[idx]

    return dB, spc
