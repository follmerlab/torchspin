"""
saffron_thyme — Real-pulse time-domain propagation for saffron.

Provides the 'thyme' engine for saffron pulse EPR simulations.
Instead of ideal (delta-function) pulses, thyme uses density matrix
propagation with finite-duration shaped pulses, propagator lookup
tables, phase cycling, and optional relaxation.

This module bridges saffron's orientation averaging with spidyan's
density matrix propagation engine (_thyme). For each orientation in
the powder grid, a full Hamiltonian is constructed, converted to
spidyan-compatible events, and propagated.

References
----------
[1] Pribitzer et al., J. Magn. Reson. (2016) — spidyan algorithm
[2] EasySpin s_thyme.m — propagator lookup table approach

Usage::

    # Called automatically from saffron() when Exp.tp has nonzero entries
    from torchspin.saffron_thyme import saffron_thyme

    x, signal, info = saffron_thyme(sys, exp, opt)
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from torchspin.spinsystem import SpinSystem
from torchspin.saffron import PulseExperiment, SaffronOptions
from torchspin.ham import ham
from torchspin.hamsymm import hamsymm
from torchspin.sphgrid import sphgrid
from torchspin.rotations import erot
from torchspin.constants import PLANCK, BMAGN, GFREE
from torchspin.spidyan import (
    _thyme, _propagation_setup, _sequencer,
    _relaxation_superoperator,
    _predefined_experiments,
    SpidyanOptions, PulseEvent, DelayEvent,
    VaryStructure, RelaxationData,
)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def saffron_thyme(
    sys: SpinSystem,
    exp: PulseExperiment,
    opt: Optional[SaffronOptions] = None,
) -> tuple:
    """Simulate pulse EPR using real-pulse density matrix propagation.

    Uses saffron's orientation averaging grid with spidyan's time-domain
    propagation engine (thyme algorithm).

    Parameters
    ----------
    sys : SpinSystem
        Spin system.
    exp : PulseExperiment
        Experiment specification. Must have nonzero Exp.tp entries.
    opt : SaffronOptions, optional
        Saffron simulation options.

    Returns
    -------
    tuple of (x, signal, info)
        x : ndarray — time axis (µs) or list of axes
        signal : ndarray — simulated signal (time domain or spectrum)
        info : dict — 'td' and 'fd' fields
    """
    if opt is None:
        opt = SaffronOptions()

    # Deep-copy exp so downstream mutations don't leak back to the caller.
    # Completes audit A10 for the thyme dispatch path as well.
    import copy as _copy
    exp = _copy.deepcopy(exp)

    device = opt.device
    dtype = torch.complex128

    # --- Build the spidyan-compatible experiment dict ---
    spidyan_exp = _saffron_exp_to_spidyan(exp)

    # --- Parse into events and vary structure ---
    spidyan_opt = SpidyanOptions(
        Relaxation=_has_relaxation(sys),
    )
    events_template, vary, frame_shift, int_time_step, single_point_det = _sequencer(
        spidyan_exp, spidyan_opt,
    )

    # For saffron, we always want single-point detection (echo amplitude)
    # unless the user explicitly requests time-domain traces
    is_echo_detection = single_point_det

    # --- Orientation grid ---
    crystal_mode = (exp.SampleFrame is not None
                    or exp.CrystalSymmetry is not None
                    or exp.MolFrame is not None)
    if crystal_mode:
        # Single crystal: enumerate symmetry-related sites (EasySpin
        # p_crystalorientations). Each orientation gets weight 4*pi.
        from torchspin.sitetransforms import crystal_orientations
        ang_M2L = crystal_orientations(exp.SampleFrame, exp.CrystalSymmetry,
                                       exp.MolFrame)
        phi_arr = torch.tensor(ang_M2L[:, 0], dtype=torch.float64)
        theta_arr = torch.tensor(ang_M2L[:, 1], dtype=torch.float64)
        chi_arr = torch.tensor(ang_M2L[:, 2], dtype=torch.float64)
        weights_arr = torch.full((ang_M2L.shape[0],), 4.0 * math.pi,
                                 dtype=torch.float64)
    else:
        if opt.GridSymmetry == 'auto':
            try:
                result = hamsymm(sys)
                grid_sym = result[0] if isinstance(result, tuple) else result
            except (RuntimeError, ValueError):
                grid_sym = 'C1'
        else:
            grid_sym = opt.GridSymmetry
        phi_arr, theta_arr, weights_arr, _ = sphgrid(grid_sym, opt.GridSize)
        chi_arr = None
    n_orient = phi_arr.shape[0]

    # --- g-tensor in molecular frame ---
    g = sys.g[0].to(torch.float64).to(device)
    if g.ndim == 1:
        if sys.gFrame is not None:
            gf = sys.gFrame[0].to(torch.float64).to(device)
            Rg = erot(gf, device=device).T
            g_mol = Rg @ torch.diag(g) @ Rg.T
        else:
            g_mol = torch.diag(g)
    else:
        g_mol = g

    # --- Orientation selection setup ---
    orientation_selection = (exp.mwFreq is not None and
                             getattr(exp, 'ExciteWidth', None) is not None)

    # --- Propagation setup (operators, relaxation) ---
    sigma0, det_ops, events_template, relaxation = _propagation_setup(
        sys, events_template, spidyan_opt,
    )

    # ==================================================================
    # Orientation loop
    # ==================================================================
    accumulated_signal = None
    accumulated_time = None
    n_skipped = 0
    total_weight = 0.0

    for i_ori in range(n_orient):
        phi = float(phi_arr[i_ori])
        theta = float(theta_arr[i_ori])
        chi = float(chi_arr[i_ori]) if chi_arr is not None else 0.0
        ori_weight = float(weights_arr[i_ori])

        # Lab frame → molecular frame rotation
        R = erot(torch.tensor([phi, theta, chi], dtype=torch.float64, device=device),
                 device=device)
        zLab_M = R[2]

        # --- Orientation selection ---
        if orientation_selection:
            geff_vec = g_mol @ zLab_M
            geff = geff_vec.norm().item()
            nu_mw = geff * BMAGN * exp.Field / (1e3 * PLANCK * 1e6)
            hstrain = sys.HStrain.to(torch.float64).to(device)
            if hstrain.ndim == 1:
                lw_vec = torch.diag(hstrain) @ zLab_M
            else:
                lw_vec = hstrain @ zLab_M
            lw2 = lw_vec.dot(lw_vec).item() + exp.ExciteWidth ** 2
            orisel_weight = math.exp(-(nu_mw - exp.mwFreq * 1e3) ** 2 / lw2)
            if orisel_weight < getattr(opt, 'OriThreshold', 0.01):
                n_skipped += 1
                continue
        else:
            orisel_weight = 1.0

        # --- Build orientation-specific Hamiltonian ---
        B_vec = [0.0, 0.0, exp.Field]  # mT, along z
        # Rotate field into molecular frame: B_mol = R^T @ B_lab
        B_mol = R.T @ torch.tensor(B_vec, dtype=torch.float64, device=device)
        H = ham(sys, B_mol.tolist())
        Ham0 = H.detach().cpu().numpy().astype(complex)

        # --- Deep-copy events for this orientation (reset propagator cache) ---
        events = _copy_events(events_template)

        # --- Transform relaxation to eigenbasis ---
        relax = None
        if relaxation is not None:
            relax = RelaxationData()
            _, U_vecs = np.linalg.eigh(Ham0)
            R_basis = np.kron(U_vecs.T, U_vecs.conj().T)
            relax.Gamma = R_basis.conj().T @ relaxation.Gamma @ R_basis
            relax.equilibriumState = relaxation.equilibriumState.copy()

        # --- Propagate ---
        time_array, signal_array, _, _, _ = _thyme(
            sigma0.copy(), Ham0, det_ops, events, relax, vary,
        )

        # --- Accumulate weighted signal ---
        weight = ori_weight * orisel_weight
        total_weight += weight

        if signal_array is not None and not (
            isinstance(signal_array, np.ndarray) and signal_array.size == 0
        ):
            # Extract signal for accumulation
            sig = _extract_signal(signal_array, time_array, is_echo_detection)

            if accumulated_signal is None:
                accumulated_signal = weight * sig
                if isinstance(time_array, np.ndarray):
                    accumulated_time = time_array
                elif isinstance(time_array, list) and len(time_array) > 0:
                    # For echo detection, build uniform time axis
                    accumulated_time = np.arange(len(sig)) * (
                        exp.dt if hasattr(exp, 'dt') and exp.dt is not None else 1.0
                    )
            else:
                if sig.shape == accumulated_signal.shape:
                    accumulated_signal += weight * sig
                else:
                    min_len = min(sig.shape[-1], accumulated_signal.shape[-1])
                    accumulated_signal[..., :min_len] += weight * sig[..., :min_len]

    # --- Normalize by total weight ---
    if accumulated_signal is not None and total_weight > 0:
        accumulated_signal /= total_weight

    # --- Format output ---
    if accumulated_signal is None:
        accumulated_signal = np.array([])
        accumulated_time = np.array([])

    # Squeeze single-detector dimension
    if (isinstance(accumulated_signal, np.ndarray) and
            accumulated_signal.ndim >= 2 and accumulated_signal.shape[0] == 1):
        accumulated_signal = accumulated_signal[0]

    # Build info dict (matching saffron output format)
    info = {
        'td': {
            'x': accumulated_time,
            'signal': accumulated_signal,
        },
        'fd': None,
        'nOrientations': n_orient,
        'nSkipped': n_skipped,
    }

    # --- FFT to frequency domain ---
    if not getattr(opt, 'TimeDomain', False) and accumulated_signal.size > 0:
        fd_signal, fd_axis = _fft_thyme(accumulated_signal, accumulated_time, exp)
        info['fd'] = {
            'x': fd_axis,
            'signal': fd_signal,
        }

    return accumulated_time, accumulated_signal, info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_signal(signal_array, time_array, is_echo_detection):
    """Extract signal from _thyme output, handling variable-length traces.

    For echo detection (single-point), extracts the last time point from
    each data point's trace, returning a 1D array of echo amplitudes.
    For full-trace mode, returns the signal array as-is.
    """
    if isinstance(signal_array, np.ndarray):
        return signal_array.copy()

    if isinstance(signal_array, list):
        if is_echo_detection:
            # Extract last point from each data point
            echo_amps = []
            for sig in signal_array:
                if sig is not None and isinstance(sig, np.ndarray) and sig.size > 0:
                    # sig shape: (nDet, nTimePoints) or (nTimePoints,)
                    if sig.ndim == 2:
                        echo_amps.append(sig[0, -1])  # last point, first detector
                    else:
                        echo_amps.append(sig[-1])
                else:
                    echo_amps.append(0.0 + 0j)
            return np.array(echo_amps, dtype=complex)
        else:
            # Try to stack into array
            try:
                return np.array(signal_array, dtype=complex)
            except ValueError:
                # Variable-length — extract last points as fallback
                echo_amps = []
                for sig in signal_array:
                    if sig is not None and isinstance(sig, np.ndarray) and sig.size > 0:
                        if sig.ndim == 2:
                            echo_amps.append(sig[0, -1])
                        else:
                            echo_amps.append(sig[-1])
                    else:
                        echo_amps.append(0.0 + 0j)
                return np.array(echo_amps, dtype=complex)

    return np.array([], dtype=complex)


def _saffron_exp_to_spidyan(exp: PulseExperiment) -> dict:
    """Convert saffron PulseExperiment to spidyan-compatible dict."""
    predefined_names = {
        '2pESEEM', '3pESEEM', '4pESEEM', 'HYSCORE', 'MimsENDOR',
    }

    spidyan_exp = {}

    if isinstance(exp.Sequence, str) and exp.Sequence in predefined_names:
        # Predefined experiment — convert to spidyan format
        spidyan_exp['Sequence'] = exp.Sequence
        spidyan_exp['Field'] = exp.Field

        # Copy required timing parameters
        if exp.tau is not None and exp.tau > 0:
            spidyan_exp['tau'] = exp.tau
        if exp.T is not None and exp.T > 0:
            spidyan_exp['T'] = exp.T
        if exp.t1 is not None and exp.t1 > 0:
            spidyan_exp['t1'] = exp.t1
        if exp.t2 is not None and exp.t2 > 0:
            spidyan_exp['t2'] = exp.t2

        spidyan_exp['dt'] = exp.dt
        spidyan_exp['nPoints'] = exp.nPoints

    else:
        # Custom sequence — build from Flip/Phase/tp
        sequence = []
        flip_angles = exp.Flip
        phases = exp.Phase if exp.Phase is not None else [1.0] * len(flip_angles)
        tp_list = exp.tp if exp.tp is not None else [0.016] * len(flip_angles)
        t_list = exp.t if exp.t is not None else [0.0] * len(flip_angles)

        for i in range(len(flip_angles)):
            # Pulse
            pulse_def = {
                'tp': tp_list[i] if tp_list[i] > 0 else 0.016,
                'Flip': flip_angles[i] * np.pi / 2,  # saffron uses π/2 units
                'Phase': phases[i] * np.pi / 2,
            }
            sequence.append(pulse_def)

            # Delay after pulse (from t array)
            if i < len(t_list):
                sequence.append(t_list[i])

        spidyan_exp['Sequence'] = sequence
        spidyan_exp['Field'] = exp.Field

        # Incrementation dimensions
        if exp.Inc is not None:
            dim_specs = _build_dim_specs(exp.Inc, exp.dt, len(flip_angles))
            for k, v in dim_specs.items():
                spidyan_exp[k] = v

        spidyan_exp['nPoints'] = exp.nPoints

        # Phase cycling
        if hasattr(exp, 'PhaseCycle') and exp.PhaseCycle is not None:
            spidyan_exp['PhaseCycle'] = exp.PhaseCycle

    return spidyan_exp


def _build_dim_specs(Inc, dt, n_pulses):
    """Build Dim1/Dim2 specifications from saffron Inc array."""
    specs = {}
    # Inc values: 0=no change, +1=increment dim1, -1=decrement dim1,
    #             +2=increment dim2, -2=decrement dim2
    for dim in [1, 2]:
        delay_indices = []
        for i, inc in enumerate(Inc):
            if abs(inc) == dim:
                # The delay after pulse i
                delay_idx = i + 1  # 1-based delay index
                delay_indices.append(f'd{delay_idx}')

        if delay_indices:
            dt_val = dt if isinstance(dt, (int, float)) else dt[0]
            specs[f'Dim{dim}'] = [','.join(delay_indices), dt_val]

    return specs


def _has_relaxation(sys):
    """Check if spin system has relaxation parameters."""
    T1 = getattr(sys, 'T1', None)
    T2 = getattr(sys, 'T2', None)
    has_T1 = T1 is not None and np.any(np.array(T1, dtype=float) > 0)
    has_T2 = T2 is not None and np.any(np.array(T2, dtype=float) > 0)
    return has_T1 or has_T2


def _copy_events(events):
    """Deep copy event list, resetting propagator caches."""
    import copy
    new_events = []
    for evt in events:
        new_evt = copy.copy(evt)
        new_evt.Propagation = {}  # Reset cache for new Hamiltonian
        new_events.append(new_evt)
    return new_events


def _fft_thyme(signal, time, exp):
    """FFT time-domain signal to frequency domain.

    Parameters
    ----------
    signal : ndarray
        Time-domain signal, shape (nPoints,) or (nPoints1, nPoints2).
    time : ndarray
        Time axis.
    exp : PulseExperiment

    Returns
    -------
    fd_signal : ndarray
    fd_axis : ndarray (MHz)
    """
    if signal.ndim == 1:
        n = len(signal)
        if n == 0:
            return np.array([]), np.array([])
        dt_val = time[1] - time[0] if len(time) > 1 else 1.0
        # Apodize
        window = np.hamming(n)
        fd = np.fft.fftshift(np.fft.fft(signal * window))
        freq = np.fft.fftshift(np.fft.fftfreq(n, d=dt_val))  # MHz (since dt in µs)
        return fd, freq
    elif signal.ndim == 2:
        n1, n2 = signal.shape
        if isinstance(time, np.ndarray) and time.ndim == 1:
            dt_val = time[1] - time[0] if len(time) > 1 else 1.0
        else:
            dt_val = 1.0
        window1 = np.hamming(n1)
        window2 = np.hamming(n2)
        windowed = signal * window1[:, None] * window2[None, :]
        fd = np.fft.fftshift(np.fft.fft2(windowed))
        freq1 = np.fft.fftshift(np.fft.fftfreq(n1, d=dt_val))
        freq2 = np.fft.fftshift(np.fft.fftfreq(n2, d=dt_val))
        return fd, [freq1, freq2]
    else:
        return signal, time
