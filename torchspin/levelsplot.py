"""Energy level (Zeeman) diagram plotting.

Port of EasySpin's ``levelsplot.m``.

Plots energy eigenvalues of a spin system as a function of magnetic field,
with optional resonance transitions and stick spectrum.

Example
-------
>>> from torchspin import SpinSystem
>>> from torchspin.levelsplot import levelsplot
>>> sys = SpinSystem(S=[7/2], g=[[2.0, 2.0, 2.0]], D=[[5000, 0]])
>>> fig, ax = levelsplot(sys, 'xy', [0, 6000], mwFreq=95.0)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import math
import numpy as np
import torch

from torchspin.levels import levels
from torchspin.ham import ham
from torchspin.rotutils import vec2ang, ang2vec
from torchspin.utils import unitconvert


@dataclass
class LevelsPlotOptions:
    """Options for :func:`levelsplot`."""
    Units: str = 'GHz'
    nPoints: int = 201
    PlotThreshold: float = 1e-6
    SlopeColor: bool = False
    StickSpectrum: bool = False
    LineWidth: float = 0.5
    TransLineWidth: float = 0.5
    Offset: float = 0.0
    ColorMap: Optional[str] = None


def levelsplot(
    sys,
    ori: Union[str, list, np.ndarray] = 'z',
    B: Union[list, np.ndarray, torch.Tensor] = None,
    mwFreq: Optional[float] = None,
    opt: Optional[LevelsPlotOptions] = None,
    *,
    ax=None,
):
    """Plot energy level diagram.

    Parameters
    ----------
    sys:
        Spin system (:class:`~torchspin.SpinSystem`).
    ori:
        Orientation of the magnetic field in the molecular frame.

        * String: ``'x'``, ``'y'``, ``'z'``, ``'xy'``, ``'xz'``, ``'yz'``, ``'xyz'``
        * 2-element ``[phi, theta]`` (radians)
        * 3-element ``[phi, theta, chi]`` (radians)

    B:
        Magnetic field range in mT.  Either ``[Bmin, Bmax]`` or a full
        vector.  Default: ``[0, 1400]``.
    mwFreq:
        Microwave frequency in GHz.  If given, resonance transitions are
        plotted.
    opt:
        Plot options (see :class:`LevelsPlotOptions`).
    ax:
        Matplotlib axes to plot on.  If ``None``, a new figure is created.

    Returns
    -------
    fig:
        Matplotlib figure.
    ax:
        Matplotlib axes (or list of axes if stick spectrum is enabled).
    """
    import matplotlib.pyplot as plt

    if opt is None:
        opt = LevelsPlotOptions()
    if B is None:
        B = [0, 1400]

    # --- Parse orientation ---
    phi, theta, chi = _parse_orientation(ori)

    # --- Parse field range ---
    B_arr = np.asarray(B, dtype=np.float64).ravel()
    if B_arr.size == 2:
        Bvec = np.linspace(B_arr[0], B_arr[1], opt.nPoints)
    else:
        Bvec = B_arr

    # Field display units
    if np.max(Bvec) >= 2000:
        Bscale = 1e-3
        field_unit = 'T'
    else:
        Bscale = 1.0
        field_unit = 'mT'

    # --- Compute energy levels ---
    B_tensor = torch.tensor(Bvec, dtype=torch.float64)
    _, E_MHz = levels(sys, phi, theta, B_tensor)
    # E_MHz shape: (nStates, nB)
    E_MHz_np = E_MHz.detach().cpu().numpy() - opt.Offset

    # Convert energy units
    E_plot, energy_unit, Escale = _convert_energy(E_MHz_np, opt.Units)

    # --- Create figure ---
    compute_resonances = mwFreq is not None
    if ax is None:
        if opt.StickSpectrum and compute_resonances:
            fig, axes = plt.subplots(2, 1, height_ratios=[3, 1], sharex=True)
            ax_main = axes[0]
            ax_stick = axes[1]
        else:
            fig, ax_main = plt.subplots()
            ax_stick = None
    else:
        ax_main = ax
        fig = ax_main.figure
        ax_stick = None

    nLevels = E_plot.shape[0]

    # --- Plot energy levels ---
    line_color = [0, 0.447, 0.741]

    if opt.SlopeColor:
        cmap_name = opt.ColorMap or 'viridis'
        cmap = plt.get_cmap(cmap_name)
        for i in range(nLevels):
            # Color by slope (derivative)
            slope = np.gradient(E_plot[i, :])
            slope_norm = np.abs(slope)
            if slope_norm.max() > 0:
                slope_norm = slope_norm / slope_norm.max()
            # Plot as colored line segments
            for j in range(len(Bvec) - 1):
                ax_main.plot(
                    Bvec[j:j+2] * Bscale,
                    E_plot[i, j:j+2] * Escale,
                    color=cmap(slope_norm[j]),
                    linewidth=opt.LineWidth,
                )
    else:
        for i in range(nLevels):
            ax_main.plot(
                Bvec * Bscale,
                E_plot[i, :] * Escale,
                color=line_color,
                linewidth=opt.LineWidth,
            )

    # --- Plot resonance transitions ---
    # EasySpin levelsplot.m:244-272.  No blanket try/except: errors in the
    # resonance computation propagate (B78).
    if compute_resonances:
        B_res, intensities, transitions = _resonances(sys, phi, theta, chi, Bvec, mwFreq)

        if len(B_res) > 0:
            tp_max = np.max(np.abs(intensities))
            if tp_max > 0:
                intensities = intensities / tp_max
            abs_int = np.abs(intensities)

            # Sort by intensity (weakest first) so that stronger lines are drawn on top
            idx = np.argsort(abs_int, kind='stable')
            B_res = B_res[idx]
            abs_int = abs_int[idx]
            intensities = intensities[idx]
            transitions = transitions[idx]

            # Energy levels at each resonance field
            zL = np.array(ang2vec(phi, theta), dtype=np.float64).ravel()
            H0, mux, muy, muz = ham(sys, B0=None)
            muzL = zL[0] * mux + zL[1] * muy + zL[2] * muz

            allowed_color = np.array([1, 0, 0])
            forbidden_color = np.array([0.8, 0.8, 0.8])

            for iF in range(len(B_res)):
                if abs_int[iF] < opt.PlotThreshold:
                    continue

                H = H0 - muzL * float(B_res[iF])
                E_i = torch.linalg.eigvalsh(H).detach().cpu().numpy()
                E_i = np.sort(E_i.real) - opt.Offset
                E_i_plot = _convert_energy_values(E_i, opt.Units)

                t = transitions[iF]
                trans_color = abs_int[iF] * allowed_color + (1 - abs_int[iF]) * forbidden_color

                ax_main.plot(
                    [B_res[iF] * Bscale, B_res[iF] * Bscale],
                    [E_i_plot[t[0]] * Escale, E_i_plot[t[1]] * Escale],
                    color=trans_color,
                    linewidth=opt.TransLineWidth,
                    gid='transition',
                )

            # Stick spectrum
            if opt.StickSpectrum and ax_stick is not None:
                ax_stick.axhline(y=0, color='k', linewidth=0.5)
                for iF in range(len(B_res)):
                    ax_stick.plot(
                        [B_res[iF] * Bscale, B_res[iF] * Bscale],
                        [0, intensities[iF]],
                        color=line_color,
                        linewidth=2,
                        gid='line',
                    )
                if np.any(intensities < 0):
                    ax_stick.set_ylim(-1.1, 1.1)
                else:
                    ax_stick.set_ylim(0, 1.1)
                ax_stick.set_ylabel('rel. intensity')
        else:
            ax_main.text(
                0.05, 0.05, 'no EPR resonances in field range',
                transform=ax_main.transAxes, color='red',
                verticalalignment='bottom',
            )

    # --- Labels and formatting ---
    ax_main.set_ylabel(f'energy ({energy_unit})')
    bottom_ax = ax_stick if ax_stick is not None else ax_main
    bottom_ax.set_xlabel(f'magnetic field ({field_unit})')

    # Orientation annotation
    if isinstance(ori, str):
        ori_str = f"'{ori}'"
    else:
        ori_str = ''
    if chi != 0:
        ann = f"{ori_str} (\u03c6,\u03b8,\u03c7) = ({math.degrees(phi):.1f}, {math.degrees(theta):.1f}, {math.degrees(chi):.1f})\u00b0"
    else:
        ann = f"{ori_str} (\u03c6,\u03b8) = ({math.degrees(phi):.1f}, {math.degrees(theta):.1f})\u00b0"
    if mwFreq is not None:
        ann += f"\n{mwFreq} GHz"
    ax_main.text(
        0.02, 0.98, ann, transform=ax_main.transAxes,
        verticalalignment='top', fontsize=8,
    )

    ax_main.set_xlim(Bvec[0] * Bscale, Bvec[-1] * Bscale)

    if ax_stick is not None:
        fig.tight_layout()
        return fig, [ax_main, ax_stick]
    return fig, ax_main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resonances(sys, phi, theta, chi, Bvec, mwFreq):
    """Resonance fields, intensities and level pairs at one orientation.

    EasySpin levelsplot.m:246-249: ``resfields(Sys, Exp, Opt)`` with
    ``Exp.SampleFrame = [-chi -theta -phi]`` and ``Opt.Threshold = 0``,
    ``Opt.Freq2Field = 0``.  The lab axes in the molecular frame are the rows of
    ``erot([phi, theta, chi])``: B0 along ``zL`` (the direction given by
    ``ori``), B1 along ``xL`` (EasySpin's default perpendicular mode).

    Returns ``(B_res (mT), intensities, transitions)``; ``transitions`` holds
    0-based level indices (lower, upper) in ascending-energy order.  The
    intensities are transition rates without the frequency-to-field factor and
    at infinite temperature; levelsplot only uses them relative to the largest.
    """
    from torchspin.resfields_batch import resfields_batch
    from torchspin.experiment import Experiment, Options
    from torchspin.rotations import erot

    H0, mux, muy, muz = ham(sys, B0=None)
    R = erot([phi, theta, chi]).to(dtype=torch.float64, device=H0.device)
    exp = Experiment(mwFreq=float(mwFreq), Range=[float(Bvec[0]), float(Bvec[-1])])
    res_opt = Options()
    res_opt.Threshold = 0.0
    f64 = dict(dtype=torch.float64, device=H0.device)
    with torch.no_grad():
        B_l, I_l, _, P_l = resfields_batch(
            H0, mux, muy, muz,
            torch.tensor([phi], **f64), torch.tensor([theta], **f64),
            exp, res_opt, sys=None, return_pairs=True, R_batch=R.unsqueeze(0),
        )
        if P_l[0] is None or B_l[0].numel() == 0:
            return np.zeros(0), np.zeros(0), np.zeros((0, 2), dtype=int)
        B_res = B_l[0]
        pairs = P_l[0].to(torch.long)
        # Freq2Field = 0: undo resfields_batch's factor dB/dE = 1/|<v|muzL|v> - <u|muzL|u>|
        # (it leaves the factor at 1 for slopes below 1e-5 MHz/mT; so does this).
        muzL = R[2, 0] * mux + R[2, 1] * muy + R[2, 2] * muz
        _, V = torch.linalg.eigh(H0.unsqueeze(0) - B_res.to(H0.dtype).view(-1, 1, 1) * muzL)
        d = (V.conj() * (muzL @ V)).sum(dim=-2).real          # <k|muzL|k>, (nRes, nStates)
        r = torch.arange(B_res.numel(), device=H0.device)
        slope = (d[r, pairs[:, 1]] - d[r, pairs[:, 0]]).abs()
        slope = torch.where(slope > 1e-5, slope, torch.ones_like(slope))
        intensities = I_l[0] * slope
    return (B_res.cpu().numpy().astype(np.float64),
            intensities.cpu().numpy().astype(np.float64),
            pairs.cpu().numpy())


def _parse_orientation(ori):
    """Parse orientation to (phi, theta, chi) in radians."""
    if isinstance(ori, str):
        v = _letter2vec(ori)
        phi_val, theta_val = vec2ang(v)
        return float(phi_val), float(theta_val), 0.0

    ori = np.asarray(ori, dtype=np.float64).ravel()
    if ori.size == 2:
        return float(ori[0]), float(ori[1]), 0.0
    elif ori.size == 3:
        return float(ori[0]), float(ori[1]), float(ori[2])
    else:
        raise ValueError("Orientation must be a string, 2-element, or 3-element array")


def _letter2vec(s: str) -> np.ndarray:
    """Convert letter direction string to unit vector."""
    v = np.zeros(3)
    s = s.strip()
    i = 0
    while i < len(s):
        sign = 1.0
        if s[i] == '-':
            sign = -1.0
            i += 1
        elif s[i] == '+':
            i += 1
        if i >= len(s):
            raise ValueError(f"Invalid direction string: '{s}'")
        c = s[i].lower()
        if c == 'x':
            v[0] += sign
        elif c == 'y':
            v[1] += sign
        elif c == 'z':
            v[2] += sign
        else:
            raise ValueError(f"Unknown direction character '{c}' in '{s}'")
        i += 1
    norm = np.linalg.norm(v)
    if norm < 1e-15:
        raise ValueError(f"Direction string '{s}' results in zero vector")
    return v / norm


def _convert_energy(E_MHz: np.ndarray, units: str):
    """Convert energy from MHz to target units. Returns (E_converted, unit_label, scale)."""
    if units == 'GHz':
        E = E_MHz / 1e3
        Escale = 1.0
        label = 'GHz'
    elif units == 'cm^-1':
        E = E_MHz * unitconvert(1.0, 'MHz->cm^-1')
        if np.max(np.abs(E)) < 1e-2:
            label = '10\u207b\u2074 cm\u207b\u00b9'
            Escale = 1e4
        else:
            label = 'cm\u207b\u00b9'
            Escale = 1.0
    elif units == 'eV':
        E = E_MHz * unitconvert(1.0, 'MHz->eV')
        if np.max(np.abs(E)) < 1:
            label = 'meV'
            Escale = 1e3
        else:
            label = 'eV'
            Escale = 1.0
    else:
        raise ValueError(f"Unsupported unit '{units}'. Use 'GHz', 'cm^-1', or 'eV'.")
    return E, label, Escale


def _convert_energy_values(E_MHz: np.ndarray, units: str) -> np.ndarray:
    """Convert energy values from MHz to target units (no label/scale)."""
    if units == 'GHz':
        return E_MHz / 1e3
    elif units == 'cm^-1':
        return E_MHz * unitconvert(1.0, 'MHz->cm^-1')
    elif units == 'eV':
        return E_MHz * unitconvert(1.0, 'MHz->eV')
    else:
        raise ValueError(f"Unsupported unit '{units}'")
