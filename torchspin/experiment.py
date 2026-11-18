"""Experiment and Options dataclasses for torchspin.

Mirrors EasySpin's ``Exp`` and ``Opt`` structs for CW EPR powder simulation.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union


@dataclass
class Experiment:
    """Experimental parameters for CW EPR simulations (EasySpin ``Exp``).

    One ``Experiment`` describes the sweep, the detection and the sample.  The
    sweep mode follows from which fields are set:

    * **Field sweep** (the default): set ``mwFreq`` (GHz) and ``Range`` or
      ``CenterSweep`` (mT).  The simulators return a field axis in mT.
    * **Frequency sweep**: set ``Field`` (mT) and ``mwRange`` or
      ``mwCenterSweep`` (GHz); ``Range`` is not used.  The simulators return
      a frequency axis in GHz.
    * **Fixed field** (ENDOR): set ``Field`` only.  :func:`~torchspin.salt`
      runs fixed-field ENDOR; :func:`~torchspin.pepper` treats this as
      a frequency sweep with an automatic range.

    Fields are validated on construction (see Raises).  A field that a
    simulator does not read is ignored by it; the "Read by" notes below list
    the CW simulators that use each field.

    Parameters
    ----------
    mwFreq : float, optional
        Microwave frequency in GHz.  Required (``> 0``) for field sweeps;
        not used by frequency sweeps.  Default ``None``.
    Range : list of float, optional
        Field sweep range ``[Bmin, Bmax]`` in mT, ``Bmin < Bmax``.  Default
        ``None``: :func:`~torchspin.pepper` and
        :func:`~torchspin.garlic` then choose the range automatically
        (pepper: S = 1/2 systems only, from the g and A extremes).
    CenterSweep : list of float, optional
        Alternative to ``Range``: ``[center, sweep_width]`` in mT, width
        ``> 0``.  Converted to ``Range = [center - width/2, center + width/2]``
        on construction.  Default ``None``.
    mwCenterSweep : list of float, optional
        Alternative to ``mwRange``: ``[center, sweep_width]`` in GHz, width
        ``> 0``.  Converted to ``mwRange`` on construction.  Default ``None``.
    ModAmp : float, default 0.0
        Peak-to-peak field-modulation amplitude in mT.  ``0`` = no
        modulation: the harmonic is produced by differentiating the line
        shape.  ``> 0``: the absorption spectrum is pseudo-modulated
        (:func:`torchspin.fieldmod`) at ``Harmonic``; pepper and
        garlic raise ``ValueError`` for ``Harmonic = 0`` and for frequency
        sweeps.  ``None`` is read as 0.  Read by pepper, garlic and chili.
    SampleFrame : array_like, shape (3,) or (N, 3), optional
        Euler angles ``[alpha, beta, gamma]`` in radians of the lab-to-sample
        frame transformation, one row per sample orientation.  In pepper this
        describes a single crystal only together with ``MolFrame`` or
        ``CrystalSymmetry``; on its own it leaves a disordered (powder) sample
        unchanged.  Also read by chili.  Default ``None`` (no rotation).
    CrystalSymmetry : int or str, optional
        Crystal symmetry of a single crystal: space-group number (1-230),
        space-group symbol or point-group symbol (e.g. ``'P21/c'``,
        ``'C2h'``).  Determines the symmetry-related sites; ``None`` with
        ``MolFrame`` set is space group P1 (one site).  Setting it makes
        pepper simulate a crystal.  Default ``None``.
    MolFrame : array_like, shape (3,), optional
        Euler angles in radians of the sample-to-molecular frame
        transformation of a single crystal.  Setting it makes pepper simulate
        a crystal.  Default ``None``.
    SampleRotation : tuple, optional
        ``(axis, rho)``: rotate the sample about ``axis`` (``'x'``, ``'y'``,
        ``'z'`` or a lab-frame 3-vector) by ``rho`` radians (a scalar or a
        list of angles, one sample orientation each).  pepper only.  Default
        ``None``.
    nPoints : int, default 1024
        Number of points of the sweep axis, at least 2.
    Harmonic : int, optional
        Detection harmonic: 0 = absorption, 1 = first derivative,
        2 = second derivative.  Default ``None``, resolved on construction to
        1 for a field sweep and to 0 otherwise (frequency sweep or fixed
        field), whether or not ``mwFreq`` is also given.  pepper falls back to
        0 when the system has no line width and no strain.
    mwPhase : float, default 0.0
        Detection phase in radians: 0 = absorption, ``pi/2`` = dispersion.
        Read by pepper, garlic and chili.
    Temperature : float, optional
        Sample temperature in K for Boltzmann populations of the levels.
        ``None`` (default) = infinite temperature (equal populations).
        ``inf`` is the same as ``None``; a value ``<= 0`` raises ``ValueError``.
    lightBeam : str or tuple, optional
        Photoselection by a polarized light beam (pepper only; requires
        ``SpinSystem.tdm``): ``''``/``None`` (default, none),
        ``'perpendicular'`` or ``'parallel'`` (beam along the lab y axis,
        linear polarization perpendicular or parallel to B0),
        ``'unpolarized'``, or ``(k, alpha)`` with the propagation direction
        ``k`` (lab frame: a direction string such as ``'y'`` or ``'-z'``, a
        3-vector or ``[phi, theta]``) and the angle ``alpha`` of the electric
        field in the plane perpendicular to ``k`` (radians, ``nan`` =
        unpolarized).
    lightScatter : float, default 0.0
        Isotropic fraction of the photoselection weights, between 0 and 1:
        weight = ``(1 - lightScatter) * w + lightScatter``.  A value ``>= 1``
        switches photoselection off.  pepper only.
    Ordering : float or callable, optional
        Partial ordering of a disordered sample (pepper only).  A scalar
        ``lambda`` gives the axial distribution ``exp(lambda * P2(cos beta))``;
        a callable ``f(alpha, beta, gamma)`` or ``f(beta)`` (Euler angles in
        radians, sample to molecular frame, array arguments) returns the
        orientational weights.  Requires a Ci grid; cannot be combined with
        ``MolFrame``, ``CrystalSymmetry`` or ``Orientations``.  Default
        ``None`` (isotropic powder).
    Mode : str, default 'perpendicular'
        Microwave excitation geometry: ``'perpendicular'`` (B1 perpendicular
        to B0, standard CW EPR, mainly Delta m_S = +-1 transitions) or
        ``'parallel'`` (B1 parallel to B0, Delta m_S = 0 transitions, e.g.
        integer-spin systems).  Used when ``mwMode`` is ``None``.
    mwMode : str or tuple, optional
        EasySpin's ``Exp.mwMode``; overrides ``Mode`` when set (pepper and
        the resonance-field solvers; garlic reads ``Mode`` only).
        ``'perpendicular'``, ``'parallel'`` or ``(k, mode2)`` with the
        propagation direction ``k`` (``'x'``/``'y'``/``'z'`` with optional
        sign, a 3-vector, ``[phi, theta]`` or one polar angle, radians) and
        ``mode2`` the linear polarization angle (radians) or one of
        ``'unpolarized'``, ``'circular+'``, ``'circular-'``.  Default
        ``None``.
    Field : float, optional
        Static field in mT for frequency sweeps and fixed-field (ENDOR)
        experiments; ``None`` (default) for field sweeps.  Must be ``>= 0``;
        ``Field = 0`` is accepted only by pepper frequency sweeps and salt
        with ``Options(ZeroField=True)``; every other simulator, and pepper
        or salt without that option, raises ``ValueError`` when called.
    mwRange : list of float, optional
        Frequency sweep range ``[nu_lo, nu_hi]`` in GHz, ``nu_lo < nu_hi``.
        Requires ``Field``.  Default ``None``.
    ExciteWidth : float, optional
        Microwave excitation bandwidth (FWHM) in MHz for orientation selection
        in fixed-field experiments (salt ENDOR, orisel, saffron).  ``None``
        (default) = no orientation selection; :func:`~torchspin.orisel`
        requires it (``math.inf`` = no selection).
    Orientations : array_like or torch.Tensor, shape (N, 2), optional
        pepper and orisel only.  Field directions ``[phi, theta]`` in radians
        in the molecular frame: B0 points along ``(sin(theta) cos(phi),
        sin(theta) sin(phi), cos(theta))``, the convention of
        ``levels(sys, phi, theta, B)`` and ``resfields``.  A single
        ``[phi, theta]`` pair is ``N = 1``.  pepper then sums the stick
        spectra of these orientations, each with the powder (B1 averaged
        about B0) transition rates, without grid interpolation or SOPHE
        projection; ``Options.GridSize``/``GridSymmetry`` are ignored.  Cannot
        be combined with ``Ordering``, ``SampleFrame``, ``SampleRotation``,
        ``MolFrame``, ``CrystalSymmetry``, ``Options.Sites`` or
        ``Options.separate='transitions'``/``'sites'``.  orisel returns one
        weight per orientation.  ``None`` (default) = powder average on the
        simulator's grid.  Tensors are kept as given (may require grad);
        other inputs are converted to float arrays.
    OrientationWeights : array_like or torch.Tensor, shape (N,), optional
        pepper only; requires ``Orientations``.  Non-negative, finite weights,
        not all zero, used as given (not renormalized): each is the fraction
        of the full sphere its orientation stands for, so weights summing to
        1 give the amplitude of pepper's powder spectrum (a solid-angle
        quadrature of the sphere divided by 4 pi reproduces the powder).
        ``None`` (default) = equal weights ``1/N``.  May be a tensor that
        requires grad (``pepper_autograd``).

    Raises
    ------
    ValueError
        On construction, if ``Range`` and ``CenterSweep`` (or ``mwRange`` and
        ``mwCenterSweep``) are both given; a ``CenterSweep``/``mwCenterSweep``
        is not ``[center, width]`` with width ``> 0``; ``ModAmp < 0``;
        ``Field < 0`` in a frequency sweep or fixed-field experiment; ``mwRange`` (frequency sweep) or ``Range`` (field
        sweep) is not an increasing pair; a field sweep has no positive
        ``mwFreq``; ``Harmonic`` is not 0, 1 or 2; ``nPoints < 2``;
        ``Temperature <= 0``; or
        ``Orientations``/``OrientationWeights`` have the wrong shape, are not
        finite, or the weights are negative or all zero (or given without
        ``Orientations``).

    See Also
    --------
    Options, SpinSystem, torchspin.pepper.pepper, torchspin.garlic.garlic

    Notes
    -----
    Single-crystal and ordered-sample conventions follow EasySpin; see
    KNOWN_LIMITATIONS.md, "Simulator — user-supplied orientations in pepper
    (0.7.6)" for ``Orientations`` and "Pepper and salt — zero field,
    ``Options.ZeroField`` (0.7.4)" for ``Field = 0``.

    Examples
    --------
    A field sweep at X band; ``Harmonic`` defaults to the first derivative:

    >>> from torchspin import Experiment
    >>> exp = Experiment(mwFreq=9.5, CenterSweep=[340.0, 20.0], nPoints=512)
    >>> exp.Range, exp.Harmonic
    ([330.0, 350.0], 1)

    A frequency sweep at a fixed field; ``Harmonic`` defaults to absorption:

    >>> exp = Experiment(Field=340.0, mwRange=[9.3, 9.7])
    >>> exp.is_freq_swept, exp.Harmonic
    (True, 0)
    """

    mwFreq: Optional[float] = None
    Range: Optional[list] = None
    CenterSweep: Optional[list] = None
    mwCenterSweep: Optional[list] = None
    ModAmp: float = 0.0
    SampleFrame: Optional[list] = None
    CrystalSymmetry: Optional[object] = None   # space group number/symbol or point group (EasySpin Exp.CrystalSymmetry)
    MolFrame: Optional[list] = None            # Euler angles sample->molecule (EasySpin Exp.MolFrame)
    SampleRotation: Optional[tuple] = None     # (axis letter or 3-vector, rho or list of rho) (EasySpin Exp.SampleRotation)
    nPoints: int = 1024
    Harmonic: Optional[int] = None  # detection harmonic; None -> 1 for field sweeps, 0 otherwise
    mwPhase: float = 0.0
    Temperature: Optional[float] = None
    lightBeam: Optional[object] = None   # photoexcitation: '', 'perpendicular', 'parallel', 'unpolarized' or (k, alpha) (EasySpin Exp.lightBeam)
    lightScatter: float = 0.0            # isotropic (scattered-light) contribution to the photoselection weights
    Ordering: Optional[object] = None    # partial ordering: scalar lambda or callable f(alpha,beta,gamma) / f(beta) (EasySpin Exp.Ordering)
    Mode: str = 'perpendicular'
    mwMode: Optional[object] = None      # EasySpin Exp.mwMode: 'perpendicular', 'parallel', or (k, alpha|'unpolarized'|'circular+'|'circular-'); None → Mode
    Field: Optional[float] = None
    mwRange: Optional[list] = None
    ExciteWidth: Optional[float] = None
    Orientations: Optional[object] = None        # (N, 2) [phi, theta] (rad), molecular frame; pepper, orisel
    OrientationWeights: Optional[object] = None  # (N,) weights, fraction of the sphere (default 1/N); pepper only

    @property
    def is_freq_swept(self) -> bool:
        """True when Field and mwRange are both set (frequency-swept mode)."""
        return self.Field is not None and self.mwRange is not None

    @property
    def is_fixed_field(self) -> bool:
        """True when only Field is set (fixed-field mode, e.g. ENDOR)."""
        return (self.Field is not None and self.mwRange is None
                and self.Range is None)

    def __post_init__(self) -> None:
        # CenterSweep / mwCenterSweep → Range / mwRange (EasySpin semantics)
        if self.CenterSweep is not None:
            if self.Range is not None:
                raise ValueError("Give either Experiment.Range or Experiment.CenterSweep, not both.")
            if len(self.CenterSweep) != 2 or self.CenterSweep[1] <= 0:
                raise ValueError("Experiment.CenterSweep must be [center, sweep_width] with width > 0.")
            c, w = float(self.CenterSweep[0]), float(self.CenterSweep[1])
            self.Range = [c - w / 2.0, c + w / 2.0]
        if self.mwCenterSweep is not None:
            if self.mwRange is not None:
                raise ValueError("Give either Experiment.mwRange or Experiment.mwCenterSweep, not both.")
            if len(self.mwCenterSweep) != 2 or self.mwCenterSweep[1] <= 0:
                raise ValueError("Experiment.mwCenterSweep must be [center, sweep_width] with width > 0.")
            c, w = float(self.mwCenterSweep[0]), float(self.mwCenterSweep[1])
            self.mwRange = [c - w / 2.0, c + w / 2.0]
        if self.ModAmp is None:
            self.ModAmp = 0.0
        if self.ModAmp < 0:
            raise ValueError("Experiment.ModAmp must be zero or a positive number (mT).")
        if self.is_freq_swept:
            # Frequency-swept mode validation
            if len(self.mwRange) != 2:
                raise ValueError("Experiment.mwRange must be [nu_lo, nu_hi] in GHz.")
            if self.mwRange[0] >= self.mwRange[1]:
                raise ValueError("Experiment.mwRange[1] must be larger than mwRange[0].")
            # Field = 0 is held back until a simulator sees the Options
            # (Options.ZeroField, see require_positive_field).
            if self.Field < 0:
                raise ValueError("Experiment.Field must be positive (mT).")
        elif self.is_fixed_field:
            # Fixed-field mode validation (ENDOR)
            if self.Field < 0:
                raise ValueError("Experiment.Field must be positive (mT).")
        else:
            # Field-swept mode validation.  Range may be omitted (automatic sweep
            # range, EasySpin garlic/pepper SweepAutoRange); simulation functions
            # that cannot auto-range raise their own error.
            if self.Range is not None and len(self.Range) != 2:
                raise ValueError("Experiment.Range must be [Bmin, Bmax].")
            if self.Range is not None and self.Range[0] >= self.Range[1]:
                raise ValueError("Experiment.Range[1] must be larger than Range[0].")
            if self.mwFreq is None or self.mwFreq <= 0:
                raise ValueError("mwFreq must be positive (GHz) for field-swept mode.")
        if self.Temperature is not None:
            # B102: pepper and salt used to treat T <= 0 as infinite temperature
            # (no Boltzmann factor) while garlic raised
            t = self.Temperature
            t = float(t.detach()) if hasattr(t, 'detach') else float(t)
            if not t > 0:
                raise ValueError("Experiment.Temperature must be positive (K); "
                                 "use None (or inf) for infinite temperature.")
        if self.Harmonic is None:
            # B101: by sweep mode only (a frequency sweep with mwFreq also set
            # used to come back as a first derivative)
            self.Harmonic = 0 if (self.is_freq_swept or self.is_fixed_field) else 1
        self.Harmonic = int(self.Harmonic)
        if self.Harmonic not in (0, 1, 2):
            raise ValueError("Harmonic must be 0, 1, or 2.")
        if self.nPoints < 2:
            raise ValueError("nPoints must be at least 2.")
        if self.Orientations is not None or self.OrientationWeights is not None:
            self._check_orientations()

    def _check_orientations(self) -> None:
        """Validate Exp.Orientations / Exp.OrientationWeights (shapes, finite,
        non-negative weights).  Tensors are kept as given (on the graph); other
        array-likes become float arrays.  Idempotent (dataclasses.replace)."""
        import numpy as _np
        import torch as _torch

        def _arr(v):
            return v if _torch.is_tensor(v) else _np.asarray(v, dtype=float)

        def _np_of(v):
            return v.detach().cpu().numpy().astype(float) if _torch.is_tensor(v) else v

        if self.Orientations is None:
            raise ValueError("Experiment.OrientationWeights requires Experiment.Orientations.")
        ori = _arr(self.Orientations)
        if ori.ndim == 1 and ori.shape[0] == 2:
            ori = ori.reshape(1, 2)
        if ori.ndim != 2 or ori.shape[1] != 2 or ori.shape[0] < 1:
            raise ValueError("Experiment.Orientations must be an (N, 2) array of [phi, theta] "
                             f"in radians; got shape {tuple(ori.shape)}.")
        if not _np.all(_np.isfinite(_np_of(ori))):
            raise ValueError("Experiment.Orientations must be finite.")
        self.Orientations = ori
        if self.OrientationWeights is not None:
            w = _arr(self.OrientationWeights)
            w = w.reshape(-1)
            if w.shape[0] != ori.shape[0]:
                raise ValueError(f"Experiment.OrientationWeights has {w.shape[0]} entries but "
                                 f"Experiment.Orientations has {ori.shape[0]} orientations.")
            wn = _np_of(w)
            if not _np.all(_np.isfinite(wn)) or _np.any(wn < 0) or not _np.any(wn > 0):
                raise ValueError("Experiment.OrientationWeights must be finite, non-negative, "
                                 "and not all zero.")
            self.OrientationWeights = w


def require_positive_field(exp, opt=None) -> bool:
    """Reject ``Experiment.Field = 0`` unless ``opt.ZeroField`` allows it.

    :class:`Experiment` cannot see the Options, so it accepts ``Field = 0`` in
    frequency-swept and fixed-field mode and every simulator that reads
    ``exp.Field`` calls this first: without ``Options(ZeroField=True)`` it raises
    the same ``ValueError`` that Experiment raises for a negative field.  Returns
    True when the experiment is a permitted zero-field one.
    """
    if exp is None or getattr(exp, 'Field', None) is None:
        return False
    if not (exp.is_freq_swept or exp.is_fixed_field):
        return False
    if float(exp.Field) > 0:
        return False
    if float(exp.Field) == 0 and bool(getattr(opt, 'ZeroField', False)):
        return True
    raise ValueError("Experiment.Field must be positive (mT).")


VALID_METHODS = frozenset({
    'matrix', 'exact', 'hybrid', 'perturb',
    'perturb1', 'perturb2', 'perturb3', 'perturb4', 'perturb5',
})

# Perturbation order per method name.  The bare name 'perturb' means different
# orders in different EasySpin simulators: garlic's Breit-Rabi expansion goes to
# fifth order, while pepper's resonance-field solver is second order.  Anything
# pepper cannot deliver is rejected rather than silently run as second order.
GARLIC_PERTURB_ORDER = {
    'perturb': 5, 'perturb1': 1, 'perturb2': 2,
    'perturb3': 3, 'perturb4': 4, 'perturb5': 5,
}
PEPPER_PERTURB_ORDER = {'perturb': 2, 'perturb1': 1, 'perturb2': 2}


@dataclass
class Options:
    """Computational options for the CW EPR and ENDOR simulators (EasySpin ``Opt``).

    Every simulator reads only the fields it needs and ignores the others; the
    description of each field names the functions that use it.  Values are
    validated on construction (see Raises).

    Parameters
    ----------
    Method : str, default 'matrix'
        Resonance solver.  Accepted on construction: ``'matrix'``,
        ``'exact'``, ``'hybrid'``, ``'perturb'``, ``'perturb1'`` to
        ``'perturb5'``; which of them a simulator implements differs:

        * :func:`~torchspin.pepper`: ``'matrix'`` (or ``'exact'``),
          exact diagonalization; ``'perturb'``/``'perturb2'`` second-order
          and ``'perturb1'`` first-order perturbation theory; ``'hybrid'``,
          an exactly diagonalized core (all electrons plus
          ``HybridCoreNuclei``) with the other nuclei treated by perturbation
          theory.  ``'perturb3'`` to ``'perturb5'`` raise ``ValueError``.
        * :func:`~torchspin.garlic`: ``'exact'`` (``'matrix'`` is read
          as ``'exact'``), the Breit-Rabi fixed-point solver; ``'perturb'``
          (fifth order) or ``'perturb1'`` to ``'perturb5'`` (that order);
          ``'hybrid'`` raises ``ValueError``.  The bare name ``'perturb'`` therefore means different orders in
          pepper and garlic, as in EasySpin.
        * :func:`~torchspin.salt` finds resonance fields by
          diagonalization and ENDOR frequencies by first-order perturbation
          theory for one electron without quadrupole couplings, by matrix
          diagonalization otherwise (and always with ``Enhancement=True`` or
          ``SpinSystem.Ham``; an explicit ``'perturb*'`` with
          ``SpinSystem.Ham`` raises).  Both ENDOR paths give
          ``abs(<i|mu_perp|j>)**2`` intensities in (MHz/mT)^2.

        See KNOWN_LIMITATIONS.md, "Pepper — choosing a method, and what each
        costs (2026-10-07)".
    HybridCoreNuclei : int or list of int, optional
        pepper with ``Method='hybrid'`` only.  1-based indices into
        ``SpinSystem.Nucs`` of the nuclei kept in the exactly diagonalized
        core; all electron spins are always in the core.  ``None`` or empty
        (default) makes every nucleus perturbational.  For Cu(II) with
        nitrogen ligands, ``[1]`` keeps the copper exact.
    HybridIntThreshold : float, default 0.005
        pepper with ``Method='hybrid'`` only.  Nuclear sub-lines whose
        amplitude summed over the grid is below this fraction of the
        strongest are dropped.
    AccumMethod : str, optional
        garlic only.  How lines are accumulated into the spectrum:
        ``'binning'`` (nearest-bin stick spectrum, then convolution),
        ``'linear'`` (each line split between its two neighboring bins,
        differentiable in the line positions) or ``'explicit'`` (one
        analytic line shape per line).  ``None`` (default) = ``'explicit'``
        in the fast-motion regime and ``'binning'`` otherwise.  When the
        inputs require grad, ``'binning'`` (given or automatic) is replaced by
        ``'linear'``.  ``Experiment.mwPhase != 0`` needs ``'explicit'``.
    Accuracy : float, default 1e-12
        garlic only.  Relative convergence criterion of the Breit-Rabi
        fixed-point/Newton iterations.
    MaxIterations : int, default 15
        garlic only.  Iteration cap of the resonance-position solvers.
    GridSize : int or list of int, default [19, 4]
        Powder grid size for pepper (and the orientation grids of orisel,
        saffron and chili's slow-motion solver).  An integer ``N``: resonances
        are computed at a grid with ``N`` knots along a quarter meridian (at
        least 2), without interpolation.  ``[N_coarse, N_interp]``:
        resonances are computed on the ``N_coarse`` grid and spline
        interpolated to a fine grid of ``(N_coarse - 1) * N_interp + 1``
        knots per quarter meridian before the spectrum is accumulated
        (``N_coarse >= 2``, ``N_interp >= 1``).  The default gives 73 fine
        knots; increase ``N_coarse`` for strongly anisotropic or narrow-line
        spectra (:func:`~torchspin.grid_convergence` checks
        convergence).
    GridSymmetry : str, default 'auto'
        Symmetry of the powder grid, i.e. which part of the sphere is
        sampled.  ``'auto'`` or ``''``: the symmetry of the spin Hamiltonian
        (including strains) is detected and the matching grid and frame are
        used.  Otherwise one of ``'O3'`` (isotropic, one orientation),
        ``'Dinfh'`` (axial, one meridian), ``'D6h'``, ``'D4h'``, ``'Oh'``,
        ``'D3d'``, ``'Th'``, ``'D2h'`` (one octant), ``'C6h'``, ``'C4h'``,
        ``'S6'``, ``'C2h'``, ``'Ci'`` (a hemisphere) or ``'C1'`` (the full
        sphere).  A given symmetry is used in the molecular frame without
        checking it against the Hamiltonian; ``'Ci'`` is valid for any spin
        Hamiltonian.  See KNOWN_LIMITATIONS.md, "Automatic grid
        symmetry — strains are symmetry inputs (deliberate deviation from
        EasySpin, 0.9.24)".
    Threshold : float, default 1e-4
        Relative intensity cutoff: transitions whose intensity stays below
        ``Threshold`` times the strongest are discarded.  Read by pepper,
        pepper_autograd, salt, chili and the resonance-field solvers, not by
        garlic.  In
        :func:`~torchspin.salt` (all matrix paths) a level pair is kept
        if its nuclear-moment rate summed over the
        ``TPSGridSymm``/``TPSGridSize`` grid exceeds ``Threshold`` times the
        maximum.
    Verbosity : int, default 0
        Progress messages printed to stdout: 0 = silent, 1 = progress,
        2 = detailed (pepper, garlic, salt, chili, cardamom).
    BatchSize : int, optional
        Orientations per batched resonance search (pepper,
        pepper_autograd, salt).  ``None`` (default) = automatic: all
        orientations at once, capped so that the batched Hamiltonians
        (orientations x field knots x dim^2) stay below about 400 MB.  ``1``
        processes orientations one at a time (debugging, low memory).
    Sites : list of int, optional
        pepper single-crystal simulations only: 1-based indices of the
        symmetry-related crystal sites to include.  ``None`` (default) = all
        sites.  Cannot be combined with ``Experiment.Orientations``.
    separate : str, default ''
        Return separate spectra instead of their sum (pepper, garlic).
        ``''`` = one summed spectrum.  ``'components'``: one row per
        component and isotopologue (pepper, garlic).  pepper also accepts
        ``'transitions'`` (one row per transition; not for crystals or
        ``Experiment.Orientations``), ``'orientations'`` (one row per crystal
        sample orientation or per ``Experiment.Orientations`` entry) and
        ``'sites'`` (one row per crystal site).  Rows are stacked along
        axis 0 of the returned spectrum.
    IsoCutoff : float, default 1e-4
        Isotopologues whose abundance is below this fraction of the most
        abundant one are dropped from the isotope-mixture expansion
        (``SpinSystem.Nucs`` entries such as ``'Cu'`` or ``'(63,65)Cu'``).
        Read by pepper, garlic and chili.
    Stretch : float, default 0.25
        garlic only.  Padding of the automatic sweep range (when
        ``Experiment.Range``/``mwRange`` is ``None``), as a fraction of the
        spread of the line positions.
    ModellingAccuracy : float, default 2e-6
        Accuracy of the segment model of the pepper matrix resonance-field
        search, relative to ``mwFreq``.
    RediagLimit : float, default 0.999
        pepper matrix route, spin systems of dimension 16 or more: at a
        resonance, the two transition eigenvectors are interpolated between the
        neighboring field knots (cubic Hermite, with the eigenvector derivatives
        from perturbation theory) when both states overlap with themselves
        across the segment by at least this much, and the field gets a
        Rayleigh-quotient Newton step; otherwise, and whenever a transition
        level is degenerate at a knot, the Hamiltonian is diagonalized at the
        resonance.  ``RediagLimit >= 1`` diagonalizes at every resonance
        (exact, as before 0.9.39).  At the default, spectra differed from exact
        by <= 6.3e-7 of the maximum in the 18 cases checked (S = 1/2 to 7/2,
        dimensions 16 to 216, coupled dimers, strain, 5 K populations) at
        1.0-2.0x the speed; 0.99 was as fast but reached 1e-4 for Cu(II) + 14N.
        Smaller systems and gradient runs always diagonalize.  EasySpin
        ``Opt.RediagLimit`` (default 0.95 there, with linear interpolation).
    device : str, default 'cpu'
        torch device for the large tensors (pepper, pepper_autograd, salt,
        saffron).  A ``'cuda'`` device without CUDA available falls back to
        ``'cpu'`` with a ``RuntimeWarning``.
    QuadraticZeeman : bool, default False
        Include the field-quadratic and field-cubic terms of
        ``SpinSystem.Ham`` (orders lB = 2 and 3) in the resonance search, so
        that the field-dependent Hamiltonian equals ``ham(sys, B)``.
        ``False``: these terms are dropped.  Supported by pepper and
        pepper_autograd with ``Method='matrix'`` (field and frequency sweeps,
        powders and crystals) and by :func:`resfreqs_matrix`; the
        perturbation and hybrid methods, :func:`resfields`, garlic and salt
        raise ``NotImplementedError`` when it is on and such terms are
        present.  Transition moments stay those of the linear moment
        operators.  See KNOWN_LIMITATIONS.md, "Simulator — field-quadratic
        Zeeman terms, ``Options.QuadraticZeeman`` (0.7.1)".
    Enhancement : bool, default False
        :func:`~torchspin.endorfrq` and salt only.  ``False``: the rf
        transition operator is the nuclear magnetic moment.  ``True``: the
        total (electron + nuclear) magnetic moment, which includes hyperfine
        enhancement; salt then always uses its matrix ENDOR path, and
        :func:`~torchspin.endorfrq_perturb` raises.
        Transition selection still uses the nuclear-only rates, and
        electronic (EPR) level pairs are excluded.  See KNOWN_LIMITATIONS.md,
        "ENDOR — hyperfine enhancement (``Options.Enhancement``, 0.7.3)".
    ZeroField : bool, default False
        pepper frequency sweeps and salt fixed-field ENDOR only.  ``True``
        allows ``Experiment.Field = 0`` (zero-field EPR/FDMR, zero-field
        ENDOR): degenerate levels are grouped into clusters (relative
        tolerance 1e-9 of the level spread) and each transition between two
        clusters carries the summed rate of its level pairs, so intensities do
        not depend on the eigenbasis inside a degenerate subspace; the
        isotropic O3 shortcut is replaced by a D2h grid.  See
        KNOWN_LIMITATIONS.md, "Pepper and salt — zero field,
        ``Options.ZeroField`` (0.7.4)".
    Transitions : str or list of tuple of int, optional
        Transition selection for pepper, pepper_autograd and the ENDOR matrix
        paths of endorfrq and salt.  ``None`` (default): automatic
        pre-selection with ``Threshold``.  ``'all'``: every level pair
        ``u < v``, without pre-selection and without the ``Threshold``
        post-selection.  A list of 1-based level pairs ``(u, v)`` (levels
        numbered by ascending energy at the resonance field): exactly those
        transitions, normalized to ``u < v`` with duplicates removed; the
        ``Threshold`` post-selection still applies.  With ``Method='hybrid'``
        the levels are those of the exact core.  Not available with
        ``Method='perturb*'`` or salt's ``ZeroField`` (raises).  See
        KNOWN_LIMITATIONS.md, "Simulator — transition selection,
        ``Options.Transitions`` (0.7.5)".
    PaddingMultiplier : float, default 3.0
        pepper, pepper_autograd and salt.  When the spectrum is nonzero at the
        first or last point, both ends are extended by
        ``round(PaddingMultiplier * max(lw) / increment)`` copies of the edge
        values before the final line-width convolution and cropped afterwards,
        so the convolution does not see a step to zero.  ``0`` disables the
        padding.  Must be a finite number ``>= 0``.  See KNOWN_LIMITATIONS.md,
        "Pepper and salt — range padding, ``Options.PaddingMultiplier``
        (0.9.6)".
    TPSGridSymm : str, default 'D2h'
        endorfrq and salt only.  Symmetry of the coarse molecular-frame grid
        on which the automatic ENDOR transition selection sums the
        nuclear-moment rates (independent of ``GridSize``).
    TPSGridSize : int, default 4
        endorfrq and salt only.  Size of that grid.

    Raises
    ------
    ValueError
        On construction, if ``Method`` or ``AccumMethod`` is not one of the
        accepted names; ``GridSize`` is not an integer ``>= 2`` or a list
        ``[N_coarse >= 2, N_interp >= 1]``; ``PaddingMultiplier`` is not a
        finite number ``>= 0``; or ``GridSymmetry`` is not ``'auto'``, ``''``
        or one of the grid symmetries listed above.

    Warns
    -----
    RuntimeWarning
        If ``device`` names a CUDA device and CUDA is not available
        (``device`` is then set to ``'cpu'``).

    See Also
    --------
    Experiment, SpinSystem, torchspin.pepper.pepper, torchspin.garlic.garlic

    Examples
    --------
    >>> from torchspin import Options
    >>> opt = Options(GridSize=[31, 4], Method='perturb')
    >>> opt.grid_size_coarse, opt.grid_size_interp
    (31, 4)
    >>> Options(GridSize=31).grid_size_interp
    1
    """

    Method: str = 'matrix'
    HybridCoreNuclei: Union[int, List[int], None] = None   # EasySpin Opt.HybridCoreNuclei (1-based)
    HybridIntThreshold: float = 0.005                      # EasySpin Opt.HybridIntThreshold
    AccumMethod: Optional[str] = None
    Accuracy: float = 1e-12
    MaxIterations: int = 15
    GridSize: Union[int, List[int]] = field(default_factory=lambda: [19, 4])
    GridSymmetry: str = 'auto'
    Threshold: float = 1e-4
    Verbosity: int = 0
    BatchSize: Optional[int] = None
    Sites: Optional[list] = None      # crystal sites to include (1-based), EasySpin Opt.Sites
    separate: str = ''                # '', 'components', 'transitions', 'orientations', 'sites' (see docstring)
    IsoCutoff: float = 1e-4           # relative abundance cutoff for isotopologues (EasySpin Opt.IsoCutoff)
    Stretch: float = 0.25             # automatic sweep range padding factor (EasySpin garlic Opt.Stretch)
    ModellingAccuracy: float = 2e-6   # resonance search: segment-model accuracy relative to mwFreq (EasySpin Opt.ModellingAccuracy)
    RediagLimit: float = 0.999        # interpolate resonance eigenvectors above this overlap; >= 1: diagonalize every resonance (EasySpin Opt.RediagLimit)
    device: str = 'cpu'
    QuadraticZeeman: bool = False     # include lB = 2, 3 Sys.Ham terms in the resonance search (default: dropped)
    Enhancement: bool = False         # ENDOR hyperfine enhancement (EasySpin Opt.Enhancement), endorfrq/salt matrix paths
    ZeroField: bool = False           # allow Experiment.Field = 0 (pepper frequency sweeps, salt fixed field)
    Transitions: Union[str, List[Tuple[int, int]], None] = None   # EasySpin Opt.Transitions: None, 'all' or 1-based (u, v) pairs
    PaddingMultiplier: float = 3.0    # EasySpin Opt.PaddingMultiplier: range padding before the final convolution (0: off)
    TPSGridSymm: str = 'D2h'          # ENDOR transition-selection grid symmetry (EasySpin Opt.TPSGridSymm), endorfrq/salt
    TPSGridSize: int = 4              # ENDOR transition-selection grid size (EasySpin Opt.TPSGridSize), endorfrq/salt

    @property
    def grid_size_coarse(self) -> int:
        """Coarse grid knot count (N_coarse from GridSize)."""
        if isinstance(self.GridSize, (list, tuple)):
            return int(self.GridSize[0])
        return int(self.GridSize)

    @property
    def grid_size_interp(self) -> int:
        """Interpolation factor (N_interp from GridSize, 1 = no interpolation)."""
        if isinstance(self.GridSize, (list, tuple)):
            return int(self.GridSize[1]) if len(self.GridSize) > 1 else 1
        return 1

    def __post_init__(self) -> None:
        if self.Method not in VALID_METHODS:
            raise ValueError(
                f"Unknown Options.Method '{self.Method}'. "
                f"Valid: {', '.join(sorted(VALID_METHODS))}."
            )
        if self.AccumMethod not in (None, 'binning', 'linear', 'explicit'):
            raise ValueError(
                f"Options.AccumMethod must be None, 'binning', 'linear' or 'explicit'; got '{self.AccumMethod}'."
            )
        # Validate GridSize — accept int or [N_coarse, N_interp]
        if isinstance(self.GridSize, (list, tuple)):
            if len(self.GridSize) < 1 or len(self.GridSize) > 2:
                raise ValueError(
                    "GridSize must be an int N or a two-element list [N_coarse, N_interp]."
                )
            if self.GridSize[0] < 2:
                raise ValueError("GridSize[0] (coarse knots) must be at least 2.")
            if len(self.GridSize) > 1 and self.GridSize[1] < 1:
                raise ValueError("GridSize[1] (interpolation factor) must be at least 1.")
        else:
            if self.GridSize < 2:
                raise ValueError("GridSize must be at least 2.")
        pm = self.PaddingMultiplier
        if (isinstance(pm, bool) or not isinstance(pm, numbers.Real)
                or not math.isfinite(float(pm)) or float(pm) < 0):
            raise ValueError(
                f"Options.PaddingMultiplier must be a non-negative finite number; got {pm!r}."
            )
        # EasySpin's 14 grid symmetries (private/gridparam.m); '' means 'auto'
        # as in EasySpin (an empty Opt.GridSymmetry)
        from torchspin.sphgrid import _GRIDPARAM
        valid_sym = ('auto', '') + tuple(_GRIDPARAM)
        rl = self.RediagLimit
        if isinstance(rl, bool) or not isinstance(rl, numbers.Real) or not (0.0 <= float(rl)):
            raise ValueError(f"Options.RediagLimit must be a real number >= 0 (>= 1: diagonalize at every "
                             f"resonance); got {rl!r}.")
        if self.GridSymmetry not in valid_sym:
            raise ValueError(
                f"GridSymmetry must be 'auto', '' or one of {tuple(_GRIDPARAM)}; "
                f"got {self.GridSymmetry!r}."
            )
        # Graceful CUDA fallback: warn and revert to CPU if CUDA is requested
        # but not available (e.g. CPU-only PyTorch install).
        if self.device and "cuda" in str(self.device):
            import torch as _torch
            import warnings as _warnings
            if not _torch.cuda.is_available():
                _warnings.warn(
                    f"CUDA device '{self.device}' requested but CUDA is not available; "
                    "falling back to CPU.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.device = 'cpu'


def auto_batch_size(batch_size, dim: int, n_orient: int) -> int:
    """Orientations per batched resonance-search call (``Options.BatchSize``).

    ``None`` = automatic: all orientations at once, capped so that the batched
    Hamiltonians (orientations × ~10 field knots × dim², complex128) stay
    below ~400 MB (PERF_PLAN §2.1 item 2).
    """
    if batch_size:
        return int(batch_size)
    return max(10, min(int(n_orient), int(2.5e7 // (10 * dim * dim))))
