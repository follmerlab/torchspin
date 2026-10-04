"""Stacked spectrum plotting.

Port of EasySpin's ``stackplot.m``.

Displays multiple spectra in a vertically stacked arrangement for easy
comparison.

Example
-------
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from torchspin.stackplot import stackplot
>>> x = np.linspace(300, 400, 1024)
>>> y = np.column_stack([np.sin(x * k) for k in range(1, 6)])
>>> fig, ax = stackplot(x, y)
>>> plt.show()
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np


def stackplot(
    x: np.ndarray,
    y: np.ndarray,
    scale: Union[str, list] = 'maxabs',
    step: Union[float, np.ndarray] = 1.5,
    labels: Optional[list] = None,
    colors: Optional[Union[str, np.ndarray]] = None,
    *,
    ax=None,
):
    """Plot spectra in a vertically stacked arrangement.

    Parameters
    ----------
    x:
        Horizontal axis values, shape ``(N,)``.
    y:
        Data matrix.  If shape ``(N, M)``, plots M slices.  If shape
        ``(M, N)`` (i.e. ``y.shape[0] == len(x)`` is ``False`` but
        ``y.shape[1] == len(x)``), the matrix is auto-transposed.
    scale:
        Scaling mode:

        * ``'maxabs'`` (default): normalize each slice to its max absolute value
        * ``'int'``: normalize by integral
        * ``'dint'``: normalize by double integral
        * ``'none'``: no scaling
        * ``['maxabs', factor]``: apply additional scale factor

    step:
        Vertical spacing.  If scalar, used as a multiplier of the max
        amplitude across slices.  If an array of length ``M``, explicit
        y-positions for each slice.
    labels:
        Labels for each slice (shown on y-axis ticks).
    colors:
        Colormap name (string), single RGB color (3-element), or
        ``(M, 3)`` color array.
    ax:
        Matplotlib axes.  If ``None``, a new figure is created.

    Returns
    -------
    fig:
        Matplotlib figure.
    ax:
        Matplotlib axes.
    lines:
        List of line objects.
    """
    import matplotlib.pyplot as plt
    from torchspin.dataproc import rescaledata

    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64)

    # Auto-transpose if needed
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    elif y.shape[0] != len(x) and y.shape[1] == len(x):
        y = y.T

    if y.shape[0] != len(x):
        raise ValueError(
            f"x has {len(x)} points but y has shape {y.shape}; "
            "cannot determine which dimension corresponds to x"
        )

    n_pts, n_slices = y.shape

    # --- Parse scale ---
    scale_factor = 1.0
    if isinstance(scale, (list, tuple)):
        scale_mode = scale[0]
        scale_factor = float(scale[1])
    else:
        scale_mode = scale

    # --- Scale each slice ---
    y_plot = np.zeros_like(y)
    for k in range(n_slices):
        if scale_mode == 'none':
            y_plot[:, k] = y[:, k]
        else:
            y_scaled, _ = rescaledata(y[:, k], mode=scale_mode)
            y_plot[:, k] = y_scaled
        y_plot[:, k] *= scale_factor

    # --- Compute vertical shifts ---
    max_vals = np.array([np.max(np.abs(y_plot[:, k])) for k in range(n_slices)])

    step_arr = np.asarray(step, dtype=np.float64).ravel()
    if step_arr.size == 1:
        # Scalar step: uniform spacing
        s = step_arr[0]
        shifts = np.arange(n_slices) * s * np.max(max_vals) if np.max(max_vals) > 0 else np.arange(n_slices) * s
    elif step_arr.size == n_slices:
        shifts = step_arr.copy()
        if shifts[-1] < shifts[0]:
            shifts = shifts[::-1]
    else:
        raise ValueError(f"step must be a scalar or length-{n_slices} array")

    # Apply shifts
    for k in range(n_slices):
        y_plot[:, k] += shifts[k]

    # --- Parse colors ---
    if colors is None:
        cmap = plt.get_cmap('tab10')
        color_list = [cmap(i % 10) for i in range(n_slices)]
    elif isinstance(colors, str):
        cmap = plt.get_cmap(colors)
        color_list = [cmap(i / max(n_slices - 1, 1)) for i in range(n_slices)]
    else:
        colors_arr = np.asarray(colors, dtype=np.float64)
        if colors_arr.ndim == 1 and colors_arr.size == 3:
            color_list = [colors_arr] * n_slices
        elif colors_arr.ndim == 2 and colors_arr.shape[0] >= n_slices:
            color_list = [colors_arr[i] for i in range(n_slices)]
        else:
            color_list = [colors_arr[i % len(colors_arr)] for i in range(n_slices)]

    # --- Plot ---
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    lines = []
    for k in range(n_slices):
        line, = ax.plot(x, y_plot[:, k], color=color_list[k])
        lines.append(line)

    # Y-axis labels
    if labels is not None:
        ax.set_yticks(shifts)
        ax.set_yticklabels([str(l) for l in labels])
    else:
        ax.set_yticks(shifts)
        ax.set_yticklabels([str(i + 1) for i in range(n_slices)])

    return fig, ax, lines
