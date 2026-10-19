"""
saffron_thyme — Real-pulse time-domain propagation for saffron.

Provides the 'thyme' engine for saffron pulse EPR simulations (EasySpin's
``Opt.SimulationMode = 'thyme'``). Instead of ideal (delta-function) pulses,
thyme propagates the density matrix through finite-duration shaped pulses
using propagator lookup tables, phase cycling and optional relaxation.

This module bridges saffron's orientation averaging with spidyan's density
matrix propagation engine (``_thyme``), and reproduces the five steps
EasySpin's thyme branch performs after propagation (``saffron.m`` 1884-1940):
orientation sum with ``Exp.OriWeights``, detection phase, optional echo
integration (``Exp.DetIntegrate``), down-conversion through
``signalprocessing``, and the ``Exp.DetWindow`` time axis.

References
----------
[1] Pribitzer et al., J. Magn. Reson. (2016) — spidyan algorithm
[2] EasySpin ``saffron.m`` (thyme branch), ``private/s_sequencer.m``,
    ``private/s_thyme.m``, ``signalprocessing.m``

Usage::

    # Called automatically from saffron() for the real-pulse route
    from torchspin.saffron_thyme import saffron_thyme

    x, signal, info = saffron_thyme(sys, exp, opt)
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import numpy as np
import scipy.linalg as la

from torchspin.spinsystem import SpinSystem
from torchspin.saffron import PulseExperiment, SaffronOptions
from torchspin.ham import ham
from torchspin.hamsymm import hamsymm
from torchspin.sphgrid import sphgrid
from torchspin.rotations import erot
from torchspin.spinops import sop
from torchspin.signalprocessing import signalprocessing
from torchspin.spidyan import (
    _thyme, _sequencer,
    _relaxation_superoperator,
    _gamma_to_eigenbasis,
    SpidyanOptions, RelaxationData,
)


# Predefined sequences EasySpin's s_predefinedexperiments.m can build for the
# thyme engine. 4pESEEM and MimsENDOR have no thyme template.
_THYME_PREDEFINED = ('2pESEEM', '3pESEEM', 'HYSCORE')

# Options that only affect the fast (ideal-pulse) algorithm. EasySpin's
# s_sequencer prints that it ignores these in thyme mode.
_FAST_ONLY_OPT = ('ProductRule', 'Expand', 'EndorMethod', 'Window',
                  'ZeroFillFactor')
_FAST_ONLY_EXP = ('ExciteWidth', 'Filter')


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
        Spin system. ``Sys.lw``/``Sys.lwpp`` are ignored, as in EasySpin.
    exp : PulseExperiment
        Experiment specification. ``Exp.mwFreq`` is required. ``Exp.Sequence``
        is either one of ``'2pESEEM'``, ``'3pESEEM'``, ``'HYSCORE'``, a list
        of pulse dicts and delays (EasySpin's general syntax), or saffron's
        ``Flip``/``Phase``/``tp``/``t``/``Inc`` arrays.
    opt : SaffronOptions, optional
        Saffron simulation options. ``GridSize``, ``IntTimeStep`` and
        ``SimFreq`` affect the result.

    Returns
    -------
    tuple of (x, signal, info)
        x : ndarray or list of ndarray — the indirect-dimension axes, with
            the transient time axis appended when detection is not
            single-point (EasySpin ``saffron.m`` 1941-2016).
        signal : ndarray — detected signal, shape ``nPoints`` for
            single-point detection and ``nPoints + (nTime,)`` otherwise.
        info : dict — ``'td'``, ``'fd'``, ``'SinglePointDetection'``,
            ``'nOrientations'``.
    """
    if opt is None:
        opt = SaffronOptions()

    # Deep-copy exp so downstream mutations don't leak back to the caller.
    import copy as _copy
    exp = _copy.deepcopy(exp)

    if exp.mwFreq is None:
        raise ValueError(
            "Exp.mwFreq is required for the real-pulse (thyme) engine "
            "(EasySpin s_sequencer.m: 'Exp.mwFreq is required for the "
            "thyme method').")
    if exp.Temperature is not None:
        raise ValueError("Exp.Temperature is not supported for pulse EPR.")
    if exp.DetOperator is not None:
        warnings.warn(
            "Exp.DetOperator is a spidyan field; saffron always detects S+ "
            "over all electron spins and ignores it. Use spidyan() for other "
            "detection operators.", UserWarning, stacklevel=2)

    # Fast-algorithm-only inputs are ignored here, exactly as EasySpin's
    # s_sequencer reports. Warn rather than silently drop them.
    _opt_defaults = SaffronOptions()
    ignored = [f for f in _FAST_ONLY_EXP if getattr(exp, f, None) is not None]
    ignored += [f for f in _FAST_ONLY_OPT
                if getattr(opt, f, None) != getattr(_opt_defaults, f, None)]
    if ignored:
        warnings.warn(
            "The following fields are specific to saffron's fast algorithm "
            "and are ignored by the real-pulse (thyme) engine: "
            + ', '.join(ignored), UserWarning, stacklevel=2)

    # --- Build the spidyan-compatible experiment dict ---
    spidyan_exp, exp_id = _saffron_exp_to_spidyan(exp)

    # --- Parse into events and vary structure ---
    spidyan_opt = SpidyanOptions(
        Relaxation=_has_relaxation(sys),
        IntTimeStep=opt.IntTimeStep,
        SimFreq=opt.SimFreq,
    )
    events_template, vary, frame_shift, int_time_step, single_point_det = \
        _sequencer(spidyan_exp, spidyan_opt)

    n_points_list = (list(np.atleast_1d(spidyan_exp['nPoints']).astype(int))
                     if 'nPoints' in spidyan_exp else [])
    n_data_points = int(np.prod(n_points_list)) if n_points_list else 1

    # --- Orientation grid ---
    # EasySpin (p_sampletype.m) decides sample type from CrystalSymmetry or
    # MolFrame only: Exp.SampleFrame alone rotates the sample but leaves a
    # disordered powder a powder. This matches pepper()'s crystal_sample rule.
    crystal_mode = (exp.CrystalSymmetry is not None
                    or exp.MolFrame is not None)
    if exp.SampleFrame is not None and not crystal_mode:
        warnings.warn(
            "Exp.SampleFrame was given without Exp.CrystalSymmetry or "
            "Exp.MolFrame, so this is a powder simulation and SampleFrame has "
            "no effect on it (EasySpin p_sampletype.m). For a single crystal, "
            "set Exp.CrystalSymmetry (e.g. 'P1') or Exp.MolFrame.",
            UserWarning, stacklevel=2)
    if crystal_mode:
        # Single crystal: enumerate symmetry-related sites (EasySpin
        # p_crystalorientations). Each orientation gets weight 4*pi.
        from torchspin.sitetransforms import crystal_orientations
        ang_M2L = np.asarray(crystal_orientations(
            exp.SampleFrame, exp.CrystalSymmetry, exp.MolFrame), dtype=float)
        ori_weights = np.full(ang_M2L.shape[0], 4.0 * math.pi)
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
        phi_arr = np.asarray(phi_arr, dtype=float)
        theta_arr = np.asarray(theta_arr, dtype=float)
        # EasySpin powder: Exp.MolFrame = [-chi -theta -phi], and
        # angles_M2L = -fliplr(MolFrame) = [phi theta 0].
        ang_M2L = np.stack(
            [phi_arr, theta_arr, np.zeros_like(phi_arr)], axis=1)
        ori_weights = np.asarray(weights_arr, dtype=float)
    n_orient = ang_M2L.shape[0]

    # --- Spin operators (lab frame) and relaxation ---
    # Everything below works in the lab-frame basis, as EasySpin does: the
    # excitation, detection and initial-state operators are the plain ones
    # (Sx, S+, -Sz) and the orientation enters through the Hamiltonian only.
    # That matters because s_thyme builds its pulse propagators from
    # real(xOp) and 1i*imag(xOp), which is only correct when xOp is Sx.
    spins = list(sys.Spins)
    n_electrons = len(sys.S) if getattr(sys, 'S', None) is not None else 1
    Sx = Sy = Sz = None
    for i_spin in range(n_electrons):
        sx = sop(spins, [i_spin + 1, 1]).detach().cpu().numpy()
        sy = sop(spins, [i_spin + 1, 2]).detach().cpu().numpy()
        sz = sop(spins, [i_spin + 1, 3]).detach().cpu().numpy()
        Sx = sx if Sx is None else Sx + sx
        Sy = sy if Sy is None else Sy + sy
        Sz = sz if Sz is None else Sz + sz
    # Total angular momentum over *all* spins, for the mol -> lab basis change
    Jy = Jz = None
    for i_spin in range(len(spins)):
        jy = sop(spins, [i_spin + 1, 2]).detach().cpu().numpy()
        jz = sop(spins, [i_spin + 1, 3]).detach().cpu().numpy()
        Jy = jy if Jy is None else Jy + jy
        Jz = jz if Jz is None else Jz + jz

    custom_init = getattr(sys, 'initState', None)
    sigma0 = (np.array(custom_init, dtype=complex)
              if custom_init is not None else -Sz.astype(complex))
    det_ops = [Sx + 1j * Sy]
    eq_state = getattr(sys, 'eqState', None)

    relax_template = None
    if _has_relaxation(sys):
        T1 = np.atleast_1d(np.array(getattr(sys, 'T1', 0) or 0, dtype=float))
        T2 = np.atleast_1d(np.array(getattr(sys, 'T2', 0) or 0, dtype=float))
        relax_template = _relaxation_superoperator(spins, T1, T2)

    # ==================================================================
    # Orientation loop (EasySpin saffron.m 1859-1887)
    # ==================================================================
    accumulated = None
    time_axis = None

    events_base = _copy_events(events_template)
    for evt in events_base:
        if evt.type == 'pulse':
            evt.xOp = ((Sx + Sy) if evt.ComplexExcitation
                       else Sx).astype(complex)

    for i_ori in range(n_orient):
        ang = ang_M2L[i_ori]
        # R_M2L: v_lab = R @ v_mol, so the rows of R are the lab axes
        # expressed in molecular coordinates and B_mol is R's third row.
        R = erot(ang.tolist()).detach().cpu().numpy()
        B_mol = (R[2] * float(exp.Field)).tolist()
        Ham_mol = ham(sys, B_mol).detach().cpu().numpy().astype(complex)
        # Change basis from the molecular to the lab frame. EasySpin instead
        # rotates every tensor (saffron.m rotatesystem) and evaluates the
        # Hamiltonian with B along lab z; conjugating with the spin-rotation
        # operator is the same operator in the same basis, and needs no
        # per-interaction tensor handling.
        Ham0 = _to_lab_basis(Ham_mol, ang, Jy, Jz)
        # EasySpin's isotropic g shift into the simulation frame (saffron.m
        # 1838-1846): subtracting gshift from every g principal value removes
        # FrameShift*1e3 MHz from the electron Zeeman term.
        if frame_shift != 0:
            Ham0 = Ham0 - (frame_shift * 1e3) * Sz

        events = _copy_events(events_base)

        relax = None
        if relax_template is not None:
            relax = RelaxationData()
            _, U_vecs = np.linalg.eigh(Ham0)
            relax.Gamma = _gamma_to_eigenbasis(relax_template, U_vecs)
            relax.equilibriumState = (np.array(eq_state, dtype=complex)
                                      if eq_state is not None
                                      else sigma0.copy())

        t_raw, sig_raw, _, _, _ = _thyme(
            sigma0.copy(), Ham0, det_ops, events, relax, vary)

        sig_raw = _as_point_det_time(sig_raw, n_data_points)
        if sig_raw is None:
            continue
        if time_axis is None:
            time_axis = np.asarray(t_raw, dtype=float)
        contribution = sig_raw * ori_weights[i_ori]
        accumulated = (contribution if accumulated is None
                       else accumulated + contribution)

    if accumulated is None:
        empty = np.array([])
        return empty, empty, {'td': empty, 'fd': None,
                              'SinglePointDetection': single_point_det,
                              'nOrientations': n_orient}

    # --- Detection phase (saffron.m 1890-1898) ---
    det_phase = 0.0 if exp.DetPhase is None else float(exp.DetPhase)
    signal = accumulated * np.exp(-1j * det_phase)

    # --- Echo integration (saffron.m 1900-1915) ---
    if not single_point_det and exp.DetIntegrate:
        signal = signal.sum(axis=-1, keepdims=True)
        single_point_det = True

    # --- Down-conversion (saffron.m 1917-1935) ---
    if not single_point_det:
        det_frequency = (float(exp.mwFreq) if exp.DetFrequency is None
                         else float(exp.DetFrequency))
        freq_translation = -(det_frequency - frame_shift)
        n_time = signal.shape[-1]
        processed = np.asarray(
            signalprocessing(time_axis, signal, [freq_translation]),
            dtype=complex)
        # signalprocessing squeezes singleton dimensions, so restore the
        # (nDataPoints, nTime) layout explicitly rather than guessing.
        if processed.size != n_data_points * n_time:
            raise RuntimeError(
                "Down-conversion returned "
                f"{processed.size} points, expected "
                f"{n_data_points * n_time}; the raw signal was probably "
                "returned unprocessed (see the warning above).")
        signal = processed.reshape(n_data_points, n_time)
        # EasySpin: timeAxis = Exp.DetWindow(1):dt:Exp.DetWindow(2), with dt
        # the integration time step stored on the last event.
        dt_det = events_template[-1].TimeStep
        dw = np.atleast_1d(np.asarray(exp.DetWindow, dtype=float))
        n_det_points = int(math.floor((dw[-1] - dw[0]) / dt_det + 1e-9)) + 1
        if n_det_points != n_time:
            # The transient's length comes from the propagation; the window
            # only labels it. EasySpin returns the mismatched axis as is, so
            # say so rather than handing back x and y of different lengths.
            warnings.warn(
                f"Exp.DetWindow spans {n_det_points} steps of "
                f"{dt_det:g} us but the detected transient has {n_time} "
                "points; the window is not a whole number of integration "
                "time steps. The returned time axis is truncated to the "
                "transient.", UserWarning, stacklevel=2)
            n_det_points = n_time
        time_axis = dw[0] + np.arange(n_det_points) * dt_det
    else:
        # One detected point per acquisition: drop the detector and time axes.
        signal = signal[:, 0, 0]

    # --- Output shape and axes (saffron.m 1941-2016) ---
    # With indirect dimensions the acquisition axis unfolds into them; without
    # any, a transient keeps only its time axis and a single detected point
    # stays a one-element array.
    if n_points_list:
        signal = signal.reshape(tuple(n_points_list) + signal.shape[1:])
    elif signal.ndim > 1:
        signal = signal.reshape(signal.shape[1:])
    x_axes = _indirect_axes(exp, exp_id, n_points_list, spidyan_exp)
    if not single_point_det:
        x_axes.append(time_axis)
    if len(x_axes) == 1:
        x_out = x_axes[0]
    elif len(x_axes) == 0:
        x_out = np.array([])
    else:
        x_out = x_axes

    info = {
        'td': signal,
        'fd': None,
        'SinglePointDetection': single_point_det,
        'nOrientations': n_orient,
    }
    # Convenience only (EasySpin's thyme branch returns no spectrum): the
    # Fourier transform of a one-point-per-increment trace.
    if single_point_det and not opt.TimeDomain and signal.size > 0 \
            and signal.ndim == 1:
        fd_signal, fd_axis = _fft_thyme(signal, x_axes[0] if x_axes else None)
        info['fd'] = {'x': fd_axis, 'signal': fd_signal}

    return x_out, signal, info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_lab_basis(ham_mol, angles_M2L, Jy, Jz):
    """Express a molecular-frame Hamiltonian matrix in the lab-frame basis.

    ``D = exp(-i*phi*Jz) exp(-i*theta*Jy) exp(-i*chi*Jz)`` with the total
    angular momentum of every spin satisfies
    ``D' (sum_j R_ij S_j) D = S_i``, so ``D' H D`` turns a Hamiltonian built
    with the field rotated into the molecular frame into the same operator
    expressed in the lab-frame basis -- which is what EasySpin's
    ``rotatesystem`` + ``ham(Sys_L, [0 0 B])`` produces.
    """
    phi, theta, chi = (float(a) for a in angles_M2L)
    if phi == 0.0 and theta == 0.0 and chi == 0.0:
        return ham_mol
    D = la.expm(-1j * phi * Jz) if phi != 0.0 else np.eye(len(Jz))
    if theta != 0.0:
        D = D @ la.expm(-1j * theta * Jy)
    if chi != 0.0:
        D = D @ la.expm(-1j * chi * Jz)
    return D.conj().T @ ham_mol @ D


def _as_point_det_time(sig_raw, n_data_points):
    """Normalize a ``_thyme`` signal to shape ``(nPoints, nDet, nTime)``.

    ``_thyme`` squeezes singleton dimensions for a single acquisition point
    and folds the indirect dimensions into leading axes otherwise; EasySpin's
    ``SignalArray`` is always ``(IndirectDimensions..., nDet, nTime)``.
    """
    if sig_raw is None:
        return None
    if isinstance(sig_raw, list):
        raise NotImplementedError(
            "The detected transients have different lengths at different "
            "acquisition points (EasySpin stores them in a cell array). Use "
            "an increment that is a multiple of Opt.IntTimeStep.")
    sig = np.asarray(sig_raw, dtype=complex)
    if sig.size == 0:
        return None
    if sig.ndim == 1:
        sig = sig.reshape(1, 1, -1)
    elif sig.ndim == 2 and n_data_points == 1:
        sig = sig.reshape(1, *sig.shape)
    else:
        sig = sig.reshape(n_data_points, *sig.shape[-2:])
    return sig


def _indirect_axes(exp, exp_id, n_points_list, spidyan_exp):
    """Axes of the indirect dimensions (EasySpin saffron.m 1945-2006)."""
    if not n_points_list:
        return []
    axes = []
    if exp_id is not None:
        dt = _first(exp.dt)
        tau = float(exp.tau or 0.0)
        T = float(exp.T or 0.0)
        if exp_id == '2pESEEM':
            axes.append(tau + dt * np.arange(n_points_list[0]))
        elif exp_id == '3pESEEM':
            axes.append(tau + T + dt * np.arange(n_points_list[0]))
        elif exp_id == 'HYSCORE':
            for i_dim in range(2):
                t_start = float((exp.t1 if i_dim == 0 else exp.t2) or 0.0)
                axes.append(t_start + dt * np.arange(n_points_list[i_dim]))
        return axes
    for i_dim, n in enumerate(n_points_list):
        spec = spidyan_exp.get(f'Dim{i_dim + 1}')
        if spec is None:
            axes.append(np.arange(1, n + 1, dtype=float))
            continue
        increment = spec[1]
        if np.ndim(increment) == 0:
            axes.append(float(increment) * np.arange(n))
        else:
            axes.append(np.arange(1, n + 1, dtype=float))
    return axes


def _first(value):
    """First element of a scalar-or-list field."""
    if value is None:
        return 0.0
    arr = np.atleast_1d(np.asarray(value, dtype=float))
    return float(arr[0])


def _saffron_exp_to_spidyan(exp: PulseExperiment) -> tuple[dict, Optional[str]]:
    """Convert a saffron ``PulseExperiment`` to a spidyan experiment dict.

    Returns the dict and, for a predefined experiment, its name (used to
    build the indirect-dimension axis the way ``saffron.m`` does).
    """
    spidyan_exp = {'Field': float(exp.Field), 'mwFreq': float(exp.mwFreq)}

    exp_id = None
    # As in saffron's fast path, a string Exp.Sequence names a predefined
    # experiment; any other string (conventionally 'custom') means the
    # sequence comes from Exp.Flip/Exp.tp/Exp.t/Exp.Inc.
    from torchspin.saffron import _EXP_NAMES
    if isinstance(exp.Sequence, str) and exp.Sequence in _EXP_NAMES:
        if exp.Sequence not in _THYME_PREDEFINED:
            raise ValueError(
                f"Exp.Sequence '{exp.Sequence}' has no real-pulse template "
                "(EasySpin's s_predefinedexperiments.m has no case for it). "
                f"The real-pulse engine supports "
                f"{', '.join(_THYME_PREDEFINED)}; give the sequence "
                "explicitly as a list of pulses and delays for anything else.")
        exp_id = exp.Sequence
        spidyan_exp['Sequence'] = exp.Sequence
        if exp.dt is None:
            raise ValueError(
                f"Exp.dt is required to run {exp.Sequence} with the "
                "real-pulse engine.")
        spidyan_exp['dt'] = _first(exp.dt)
        spidyan_exp['tau'] = float(exp.tau or 0.0)
        if exp.Sequence == '3pESEEM':
            spidyan_exp['T'] = float(exp.T or 0.0)
        if exp.Sequence == 'HYSCORE':
            spidyan_exp['t1'] = float(exp.t1 or 0.0)
            spidyan_exp['t2'] = float(exp.t2 or 0.0)
        if exp.nPoints is not None:
            spidyan_exp['nPoints'] = exp.nPoints
    elif isinstance(exp.Sequence, (list, tuple)) and any(
            isinstance(e, dict) for e in exp.Sequence):
        # EasySpin's general syntax: a list of pulse dicts and delays
        spidyan_exp['Sequence'] = [
            dict(e) if isinstance(e, dict) else float(e) for e in exp.Sequence]
        if exp.nPoints is not None:
            spidyan_exp['nPoints'] = exp.nPoints
        for i_dim, spec in enumerate((exp.Dim1, exp.Dim2)):
            if spec is None:
                continue
            spec = list(spec)
            if len(spec) != 2 or not isinstance(spec[0], str):
                raise ValueError(
                    f"Exp.Dim{i_dim + 1} must be ['<events>', increment], "
                    "e.g. ['d1,d2', 0.5]. Several parameter lines per "
                    "dimension are not supported.")
            spidyan_exp[f'Dim{i_dim + 1}'] = spec
    else:
        # saffron's own custom-sequence syntax, translated to real pulses.
        # Flip and Phase are in units of pi/2, as elsewhere in saffron.
        if exp.Flip is None:
            raise ValueError(
                "A custom real-pulse sequence needs either a list-valued "
                "Exp.Sequence or Exp.Flip (flip angles in units of pi/2).")
        flip_angles = list(exp.Flip)
        n_pulses = len(flip_angles)
        phases = list(exp.Phase) if exp.Phase is not None else [1.0] * n_pulses
        tp_list = list(exp.tp) if exp.tp is not None else [0.016] * n_pulses
        t_list = list(exp.t) if exp.t is not None else [0.0] * n_pulses
        if any(float(v) <= 0 for v in tp_list):
            raise ValueError(
                "The real-pulse engine needs a finite length for every pulse; "
                "Exp.tp has a zero or negative entry.")
        sequence = []
        for i in range(n_pulses):
            sequence.append({
                'tp': float(tp_list[i]),
                'Flip': float(flip_angles[i]) * np.pi / 2,
                'Phase': float(phases[i]) * np.pi / 2,
            })
            if i < len(t_list):
                sequence.append(float(t_list[i]))
        spidyan_exp['Sequence'] = sequence
        if exp.Inc is not None:
            spidyan_exp.update(_build_dim_specs(exp.Inc, exp.dt))
        if exp.nPoints is not None:
            spidyan_exp['nPoints'] = exp.nPoints

    if exp.PhaseCycle is not None:
        spidyan_exp['PhaseCycle'] = exp.PhaseCycle

    # Detection fields (EasySpin s_sequencer.m 255-325)
    if exp.DetWindow is not None:
        spidyan_exp['DetWindow'] = exp.DetWindow
    elif exp.DetSequence is not None:
        spidyan_exp['DetSequence'] = exp.DetSequence

    return spidyan_exp, exp_id


def _build_dim_specs(Inc, dt):
    """Build Dim1/Dim2 specifications from saffron's Exp.Inc array."""
    specs = {}
    # Inc values: 0 = no change, +-1 = dimension 1, +-2 = dimension 2.
    # Inc[i] refers to the delay that follows pulse i, i.e. delay d(i+1).
    for dim in (1, 2):
        delay_names = [f'd{i + 1}' for i, inc in enumerate(Inc)
                       if abs(int(inc)) == dim]
        if not delay_names:
            continue
        dt_arr = np.atleast_1d(np.asarray(dt, dtype=float))
        dt_val = float(dt_arr[dim - 1] if dt_arr.size >= dim else dt_arr[0])
        specs[f'Dim{dim}'] = [','.join(delay_names), dt_val]
    return specs


def _has_relaxation(sys):
    """Check whether the spin system has relaxation parameters."""
    T1 = getattr(sys, 'T1', None)
    T2 = getattr(sys, 'T2', None)
    has_T1 = T1 is not None and np.any(np.array(T1, dtype=float) > 0)
    has_T2 = T2 is not None and np.any(np.array(T2, dtype=float) > 0)
    return bool(has_T1 or has_T2)


def _copy_events(events):
    """Shallow-copy the event list, resetting propagator caches."""
    import copy
    new_events = []
    for evt in events:
        new_evt = copy.copy(evt)
        new_evt.Propagation = {}  # Reset cache for a new Hamiltonian
        new_events.append(new_evt)
    return new_events


def _fft_thyme(signal, time):
    """Fourier transform a one-point-per-increment trace (convenience only).

    Not part of EasySpin's thyme branch, which returns the time-domain signal
    alone.
    """
    n = len(signal)
    if n == 0:
        return np.array([]), np.array([])
    time = np.asarray(time, dtype=float) if time is not None else None
    dt_val = (time[1] - time[0]) if (time is not None and len(time) > 1) else 1.0
    window = np.hamming(n)
    fd = np.fft.fftshift(np.fft.fft(signal * window))
    freq = np.fft.fftshift(np.fft.fftfreq(n, d=dt_val))  # MHz (dt in µs)
    return fd, freq
