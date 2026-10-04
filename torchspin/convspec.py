"""FFT-based lineshape convolution for torchspin.

Port of EasySpin's ``convspec.m``.

Convolves a 1-D spectrum with a Gaussian and/or Lorentzian lineshape.
The derivative is applied in the frequency domain for O(N log N) efficiency.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from torchspin._compile import maybe_compile


@maybe_compile
def _fft_convolve(
    spec_d: torch.Tensor,
    N: int,
    dx: float,
    fwhm_g: float,
    fwhm_l: float,
    deriv: int,
) -> torch.Tensor:
    """Inner kernel: FFT-based lineshape convolution."""
    Npad = 2 * N - 1
    n_freq = Npad // 2 + 1
    f = torch.arange(n_freq, dtype=torch.float64, device=spec_d.device) / (Npad * dx)

    H = torch.ones(n_freq, dtype=torch.complex128, device=spec_d.device)

    if fwhm_g > 0.0:
        sigma_freq = fwhm_g / (2 * math.sqrt(2 * math.log(2)))
        G = torch.exp(-2 * (math.pi ** 2) * (sigma_freq ** 2) * f ** 2)
        H = H * G.to(torch.complex128)

    if fwhm_l > 0.0:
        L = torch.exp(-math.pi * fwhm_l * f)
        H = H * L.to(torch.complex128)

    if deriv == 1:
        H = H * (2j * math.pi * f).to(torch.complex128)
    elif deriv == 2:
        H = H * (-(2 * math.pi * f) ** 2).to(torch.complex128)

    S_fft = torch.fft.rfft(spec_d, n=Npad)
    out_fft = S_fft * H
    out = torch.fft.irfft(out_fft, n=Npad)
    return out[:N].real


def _es_convolve(spec: np.ndarray, dx: float, fwhm: float, deriv: int, alpha: float, phase: float = 0.0) -> np.ndarray:
    """EasySpin ``convspec`` for one line shape (sampled kernel, 1-D).

    The line shape (``alpha``·Gaussian + (1−alpha)·Lorentzian, with derivative
    ``deriv`` and dispersion ``phase``) is *sampled* on the 2N+1 extended grid in
    units of the increment — no analytic renormalisation — exactly as EasySpin
    does, so that widths near or below one increment behave identically.
    """
    from torchspin.lineshape import gaussian as _gauss, lorentzian as _lor
    n = spec.shape[0]
    NN = 2 * n + 1
    mid = int(np.floor(NN / 2 + 0.5)) + 1          # MATLAB round(NN/2)+1 (1-based)
    x = np.arange(1, NN + 1, dtype=float)
    fw = fwhm / dx
    line = np.zeros(NN)
    if alpha != 0:
        ga, gd = _gauss(x, float(mid), fw, deriv, 0.0)
        line += alpha * (ga * math.cos(phase) + gd * math.sin(phase) if phase else ga)
    if alpha != 1:
        la, ld = _lor(x, float(mid), fw, deriv, 0.0)
        line += (1 - alpha) * (la * math.cos(phase) + ld * math.sin(phase) if phase else la)
    line_shift = np.concatenate([line[mid - 1:], line[:mid - 1]])
    decay = np.fft.ifft(line_shift)
    out = np.fft.fft(np.fft.ifft(spec, n=NN) * decay)
    out = NN / dx ** deriv * out
    return np.real(out[:n])


def convspec(
    spec: torch.Tensor,
    dx: float,
    fwhm_g: float = 0.0,
    fwhm_l: float = 0.0,
    deriv: int = 0,
    phase: float = 0.0,
) -> torch.Tensor:
    """Convolve a spectrum with a Gaussian and/or Lorentzian line shape (EasySpin ``convspec``).

    Parameters
    ----------
    spec:
        Input spectrum, shape ``(N,)``.  Should be real.
    dx:
        Abscissa step size (e.g., mT per point).
    fwhm_g:
        Gaussian FWHM in the same units as ``dx``.  Zero = no Gaussian.
    fwhm_l:
        Lorentzian FWHM in the same units as ``dx``.  Zero = no Lorentzian.
    deriv:
        Derivative order: 0 = absorption, 1 = first derivative,
        2 = second derivative (applied through the line shape).
    phase:
        Dispersion phase (rad) of the line shape (EasySpin ``phase``).

    Returns
    -------
    torch.Tensor
        Convolved spectrum, same shape as ``spec``.

    Notes
    -----
    * Port of EasySpin's ``convspec``: the line shape is sampled on the
      extended 2N+1 grid (no analytic renormalisation), so sub-increment
      widths reproduce EasySpin's behaviour bit for bit.  Gaussian and
      Lorentzian parts are applied sequentially (as pepper.m does), which is
      the same as a Voigt convolution; the derivative is carried by the first.
    * With both widths zero and ``deriv > 0`` the spectrum is differentiated
      spectrally (FFT), a torchspin extension.
    """
    N = spec.shape[0]
    if N < 2:
        return spec.clone()
    # Tensor widths: differentiable ones go to the torch implementation, the
    # rest are plain floats for the NumPy kernel below.
    if any(torch.is_tensor(v) and v.requires_grad for v in (fwhm_g, fwhm_l)) or spec.requires_grad:
        if phase != 0.0:
            raise NotImplementedError('convspec: a dispersion phase is not differentiable yet.')
        from torchspin.pepper_autograd import convspec_t
        return convspec_t(spec, dx, fwhm_g, fwhm_l, deriv)
    fwhm_g = float(torch.as_tensor(fwhm_g).detach()) if torch.is_tensor(fwhm_g) else float(fwhm_g)
    fwhm_l = float(torch.as_tensor(fwhm_l).detach()) if torch.is_tensor(fwhm_l) else float(fwhm_l)
    arr = spec.detach().cpu().double().numpy()
    if fwhm_g <= 0.0 and fwhm_l <= 0.0:
        if deriv > 0:
            out = _fft_convolve(spec.double(), N, dx, 0.0, 0.0, deriv)
            return out.to(spec.dtype)
        return spec.clone()
    if fwhm_g > 0.0:
        arr = _es_convolve(arr, dx, fwhm_g, deriv, 1.0, phase)
        deriv_left = 0
    else:
        deriv_left = deriv
    if fwhm_l > 0.0:
        arr = _es_convolve(arr, dx, fwhm_l, deriv_left, 0.0, phase if fwhm_g <= 0.0 else 0.0)
    return torch.tensor(arr, dtype=spec.dtype if spec.dtype.is_floating_point else torch.float64, device=spec.device)
