"""FFT frequency axis with DC at center.

Port of EasySpin's ``fdaxis.m``.

Generates a frequency-domain axis corresponding to an FFT, with the
zero-frequency (DC) component at the center of the array (``fftshift``
convention).

Example
-------
>>> import numpy as np
>>> from torchspin.fdaxis import fdaxis
>>> t = np.linspace(0, 1, 1024, endpoint=False)  # 1 second, 1024 points
>>> f = fdaxis(t)
>>> f[512]  # DC component near center
0.0
"""
from __future__ import annotations

import numpy as np


def fdaxis(dT_or_t, N: int | None = None) -> np.ndarray:
    """Compute FFT frequency axis with DC at center.

    Parameters
    ----------
    dT_or_t:
        Either the sampling period ``dT`` (scalar) or a time-domain vector
        ``t`` from which ``dT`` and ``N`` are extracted.
    N:
        Number of points.  Required if ``dT_or_t`` is a scalar sampling
        period; ignored if a time vector is given.

    Returns
    -------
    freq:
        Frequency axis of length ``N``, centered at DC.
        Units are ``1/[dT]`` (e.g. if ``dT`` is in microseconds, output is
        in MHz).
    """
    dT_or_t = np.asarray(dT_or_t, dtype=np.float64).ravel()

    if dT_or_t.size > 1:
        # Time vector provided
        t = dT_or_t
        dT = t[1] - t[0]
        N = len(t)
    else:
        # Scalar sampling period
        dT = dT_or_t.item()
        if N is None:
            raise ValueError("N (number of points) is required when dT is a scalar")

    f_nyquist = 1.0 / (2.0 * dT)
    unit_axis = (2.0 / N) * (np.arange(N) - N // 2)
    return f_nyquist * unit_axis
