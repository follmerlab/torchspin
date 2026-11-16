"""
Least-squares fitting of EPR spectra.

This module provides functionality for fitting experimental EPR data by
optimizing simulation parameters to minimize residuals. The public entry
point is :func:`esfit`, configured with :class:`FitOptions` and returning a
:class:`FitResult`.
"""

import dataclasses
import time
import warnings

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Callable, Optional, Union, Any
from scipy.optimize import minimize, least_squares, OptimizeResult


__all__ = ['esfit', 'FitOptions', 'FitResult', 'TorchSpinParameterHandler']


@dataclass
class FitOptions:
    """Options for :func:`esfit`: optimizer, scaling, baseline, target, budget.

    All fields have defaults, so ``FitOptions()`` is a complete set of options.
    The dataclass does not validate its fields; an invalid value raises (or is
    ignored, see each entry) when :func:`esfit` uses it.

    Parameters
    ----------
    method : str, default 'simplex'
        Optimization algorithm. Any other value makes :func:`esfit` raise
        ``ValueError``.

        - ``'simplex'``: Nelder-Mead in parameters rescaled to [-1, 1] (local,
          derivative-free; port of EasySpin's ``esfit_simplex``).
        - ``'levmar'``: Levenberg-Marquardt on the residual vector (local;
          forward-difference Jacobian with Broyden updates, port of EasySpin's
          ``esfit_levmar``).
        - ``'trf'``: bounded trust-region reflective least squares on the
          residual vector (local; SciPy ``least_squares(method='trf')``, see
          the ``trf_*`` fields).
        - ``'lbfgsb'``: L-BFGS-B on the RMSD (local; finite-difference
          gradient).
        - ``'powell'``: Powell's method (local, derivative-free).
        - ``'swarm'``: particle swarm only (global, no local polish).
        - ``'global'``: particle swarm followed by a Nelder-Mead polish from
          the swarm's best point; recommended for derivative CW EPR spectra,
          whose loss has many narrow local minima.
        - ``'genetic'``: genetic algorithm (global).
        - ``'montecarlo'``: uniform random sampling followed by simulated
          annealing around the best sample (global).
        - ``'grid'``: systematic grid search between the bounds.

    autoscale : {'lsq', 'maxabs', 'none'}, default 'lsq'
        How the model output is scaled to the data before the residuals are
        formed. ``'lsq'``: least-squares scale (constrained to be
        non-negative, as in EasySpin), solved jointly with the baseline.
        ``'maxabs'``: ``max|data| / max|model|``. ``'none'``: scale 1. Any
        other value is treated as ``'none'``. With ``'lsq'`` the model must
        carry the sign of the signal: fit an inversion-recovery curve with
        ``-exp(-t/T)``, not ``exp(-t/T)``.
    lsq_negative_scale : {'clip', 'abs'}, default 'clip'
        What ``autoscale='lsq'`` does when the unconstrained scale comes out
        negative (model anti-correlated with the data). ``'clip'`` returns the
        constrained least-squares optimum: scale 0 with the baseline refitted
        alone. ``'abs'`` reproduces EasySpin's ``coeffs(1) = abs(coeffs(1))``,
        which mirrors the model but keeps the baseline solved for the opposite
        sign, so the objective is not a least-squares fit of anything.
        :func:`esfit` warns if the final fit has a negative unconstrained
        scale.
    baseline : int, default -1
        Order of the polynomial baseline fitted together with the scale, on
        the abscissa ``linspace(0, 1, n_data)``: ``-1`` no baseline, ``0``
        constant, ``1`` linear, ``2`` quadratic, and so on.
    target : {'fcn', 'int', 'diff', 'auto'}, default 'fcn'
        Transformation of the residual vector ``fit - data`` before the RMSD
        is computed. ``'fcn'`` (EasySpin's default): the residuals as they
        are; with white noise this is the maximum-likelihood fit. ``'int'``:
        their cumulative sum (smooths the loss of derivative spectra with
        many resolved lines, which helps a poor start converge, but weights
        the noise as a random walk, so the fitted parameters scatter more
        than with ``'fcn'``). ``'diff'``: their first difference
        (``numpy.diff`` with a leading 0). ``'auto'``: ``'int'`` when the data
        look like a derivative spectrum (``|mean(data)| < 0.1 * std(data)``),
        else ``'fcn'``; the choice is made from the data alone, not from the
        simulator's ``Harmonic``. For ``'int'`` and ``'diff'`` the reported
        uncertainties account for the noise correlation the transformation
        introduces. A rugged derivative fit can be started with ``'int'`` (or
        ``method='global'``) and finished with ``'fcn'`` from its ``pfit``.
    max_iter : int, default 2000
        Iteration or evaluation budget. Its meaning depends on ``method``:
        SciPy's ``maxiter`` for ``'simplex'``, ``'lbfgsb'`` and ``'powell'``;
        LM iterations for ``'levmar'``; SciPy's ``max_nfev`` for ``'trf'``
        (see Notes); iterations (``'swarm'``) or generations (``'genetic'``);
        for ``'global'`` the swarm phase gets ``max(20, max_iter // 2)``
        iterations and the simplex polish ``max(200, max_iter - evaluations
        used by the swarm)``; for ``'montecarlo'`` the random samples are
        capped at ``max_iter`` and the annealing runs ``max(1, max_iter -
        samples)`` steps; for ``'grid'`` at most ``max_iter`` grid points are
        evaluated.
    tol_fun : float, default 1e-6
        Function tolerance. SciPy's ``fatol`` (``'simplex'``) or ``ftol``
        (``'lbfgsb'``, ``'powell'``, ``'trf'``). For ``'swarm'`` and
        ``'genetic'`` the run stops once the best RMSD (in data units) falls
        below ``tol_fun``.
    tol_x : float, default 1e-6
        Parameter-step tolerance: SciPy's ``xtol`` for ``'powell'`` and
        ``'trf'``, and the relative step threshold of ``'levmar'`` (in the
        [-1, 1]-rescaled parameters). ``'simplex'`` uses a fixed ``xatol`` of
        1e-4 in rescaled parameters instead.
    verbosity : int, default 0
        ``0``: silent. ``1``: a summary of the setup, phase headers, the final
        parameters and fit statistics, and SciPy's ``disp`` output. ``2``:
        also one line per iteration (best RMSD) for ``'swarm'``,
        ``'global'``, ``'genetic'`` and ``'montecarlo'``. Printed to stdout.
    callback : callable or None, default None
        Called once per iteration by ``'swarm'``, ``'global'`` (swarm phase),
        ``'genetic'`` and ``'montecarlo'`` (annealing phase, about 20 times)
        with a dict ``{'method': str, 'iter': int (1-based), 'max_iter': int,
        'best_f': float, 'best_x': numpy.ndarray}``; ``best_x`` holds the
        active parameters. Use it to drive live plots. An exception raised
        in the callback does not stop the fit; the first one per fit is
        reported with a ``RuntimeWarning``. Use ``progress`` for a callback
        that works with every method.
    compute_uncertainties : bool, default True
        Estimate ``pstd``, ``ci95``, ``cov`` and ``corr`` after a successful
        fit (``2 * n_active`` extra model evaluations, see :func:`esfit`
        Notes). ``False`` leaves those :class:`FitResult` fields ``None``.
    grid_points : int, default 10
        ``'grid'``: points per parameter axis (at least 2), from ``lb`` to
        ``ub`` inclusive. The grid has ``grid_points ** n_active`` points;
        only the first ``max_iter`` of them, in ``itertools.product`` order,
        are evaluated.
    mc_samples : int, default 1000
        ``'montecarlo'``: number of uniform random samples between the bounds
        (capped at ``max_iter``).
    mc_temp_start, mc_temp_end : float, default 1.0 and 1e-4
        ``'montecarlo'``: start and end temperature of the annealing phase,
        decreased geometrically. The temperature sets both the acceptance
        probability ``exp(-dRMSD / T)`` and the proposal width ``T * (ub -
        lb)``.
    ga_population : int, default 50
        ``'genetic'``: population size (at least ``ga_elite + 2``).
    ga_mutation : float, default 0.1
        ``'genetic'``: per-parameter mutation probability; a mutation adds
        Gaussian noise with standard deviation ``0.05 * (ub - lb)``.
    ga_crossover : float, default 0.7
        ``'genetic'``: probability of a uniform crossover between the two
        tournament-selected parents.
    ga_elite : int, default 5
        ``'genetic'``: number of best individuals copied unchanged into the
        next generation (at least 1).
    swarm_size : int, default 30
        ``'swarm'`` and ``'global'``: number of particles (at least 4).
    swarm_inertia : float, default 0.7
        Particle-swarm inertia weight.
    swarm_cognitive : float, default 1.4
        Particle-swarm attraction to each particle's own best position.
    swarm_social : float, default 1.4
        Particle-swarm attraction to the swarm's best position.
    lm_lambda : float, default 1e-3
        ``'levmar'``: initial damping, relative to the largest diagonal
        element of ``J.T @ J``.
    lm_delta : float, default 1e-3
        ``'levmar'``: forward-difference step for the Jacobian, in the
        parameters rescaled to [-1, 1] (EasySpin ``FitOpt.delta``).
    lm_gradient_tol : float, default 1e-5
        ``'levmar'``: stop when the infinity norm of the gradient ``J.T @ r``
        falls below this value (EasySpin ``FitOpt.TolFun``).
    lm_step_tol : float, default 1e-8
        Not used. ``'levmar'`` takes its step tolerance from ``tol_x``.
    trf_diff_step : float or None, default None
        ``'trf'``: relative finite-difference step for the Jacobian (SciPy
        ``diff_step``; ``None`` is SciPy's default, about 1.5e-8). Relative to
        the physical parameter values, unlike ``lm_delta``.
    trf_x_scale : {'jac'} or float or numpy.ndarray, default 'jac'
        ``'trf'``: SciPy ``x_scale``. ``'jac'`` scales from the Jacobian
        columns; an array needs one entry per active (non-fixed) parameter.
    trf_gradient_tol : float, default 1e-7
        ``'trf'``: SciPy ``gtol``. ``tol_fun`` and ``tol_x`` are passed as
        ``ftol`` and ``xtol``.
    n_workers : int or 'auto', default 1
        Parallel model evaluations. ``1``: sequential. An int > 1: up to that
        many workers. ``'auto'``: the cores this process may use (affinity
        and SLURM aware), at most 32, and at most one worker per ~0.5 GB of
        half the available memory. Population methods (``'swarm'``,
        ``'global'``, ``'genetic'``, ``'montecarlo'``, ``'grid'``) evaluate
        each population concurrently; ``'levmar'``, ``'trf'``, ``'lbfgsb'``
        and the uncertainty estimate evaluate the finite-difference probes of
        each Jacobian or gradient concurrently (``'trf'`` and ``'lbfgsb'``
        need SciPy >= 1.16 and otherwise warn and run sequentially).
        ``'simplex'`` and ``'powell'`` are always sequential.
    workers : {'auto', 'processes', 'threads'}, default 'auto'
        Backend for ``n_workers > 1``. ``'auto'`` and ``'processes'`` use a
        spawned process pool with the cores split between the workers (each
        worker is a separate Python process with its own torch); the model is
        sent by value with ``cloudpickle`` (needed for models defined in a
        notebook), and a model that cannot be sent falls back to a thread
        pool. ``'threads'`` forces a thread pool, which gains little because
        torchspin's forward models hold the GIL.
    devices : list or None, default None
        GPUs for process workers, e.g. ``['cuda:0', 'cuda:1']`` (CUDA device
        strings, ints or ``torch.device``): each spawned worker is pinned to
        one of them, round-robin, through ``CUDA_VISIBLE_DEVICES``, so a model
        that uses ``device='cuda'`` runs on its own GPU. ``None`` leaves the
        workers' GPU visibility unchanged. Ignored by the thread backend. A
        non-CUDA entry raises ``ValueError`` when the pool is created.
    seed : int or None, default None
        Seed of the NumPy random generator of ``'montecarlo'``, ``'genetic'``,
        ``'swarm'`` and ``'global'``. ``None``: a different realization on
        every run. An int makes these fits reproducible (recommended for
        tutorials, regression tests and published numbers).
    progress : None, 'text', 'bar' or callable, default None
        Real-time progress for every method. ``None``: silent. ``'text'``:
        one flushed line per report (iteration, evaluations, current and best
        RMSD, elapsed time, evaluations per second, ETA when the budget is
        known), plus a resource summary at the start and a totals line at
        the end. ``'bar'``: a ``tqdm`` bar (text if ``tqdm`` is missing).
        A callable is called as ``progress(info)`` with a dict (keys
        ``'method', 'phase', 'iter', 'iter_total', 'n_evaluations',
        'eval_total', 'rmsd', 'best_f', 'best_x', 'elapsed_s',
        'evals_per_s', 'eta_s'``); returning ``False`` stops the fit. An
        exception it raises does not stop the fit; the first one per fit is
        reported with a ``RuntimeWarning``. With ``n_workers > 1`` reports
        arrive per evaluation chunk, not per evaluation.
    progress_every : int or None, default None
        Report at least every this many model evaluations.
    progress_interval : float, default 1.0
        Report at least every this many seconds (wall clock).
    max_time : float or None, default None
        Wall-clock budget in seconds; the fit stops once it is exceeded.
    stop_when : callable or None, default None
        ``stop_when(info) -> bool`` with the same ``info`` dict as
        ``progress``, checked after every evaluation; ``True`` stops the fit.
        An exception it raises propagates out of :func:`esfit`, since a stop
        condition that cannot be evaluated cannot be honored.

    See Also
    --------
    esfit, FitResult

    Notes
    -----
    A stop (``max_time``, ``stop_when``, a ``progress`` callable returning
    ``False``) or a ``KeyboardInterrupt`` returns the best parameters found so
    far, with ``FitResult.success=False`` and ``FitResult.interrupted=True``;
    no uncertainties are computed for it.

    ``method='trf'`` minimizes the same target-transformed residual vector
    whose RMSD the other methods minimize, so the parameter mapping (array or
    dict ``p0``, fixed parameters, bounds), ``autoscale``, ``baseline``,
    ``target``, progress reporting and stopping apply unchanged. Its budget
    ``max_iter`` is SciPy's ``max_nfev``, the number of step evaluations; the
    finite-difference Jacobian probes (about one per active parameter per
    iteration) do not count against it, but they are counted in
    ``FitResult.n_evaluations`` and drive the progress and stop checks. An
    exhausted budget returns ``success=False`` (SciPy's message "The maximum
    number of function evaluations is exceeded.") with ``interrupted=False``.
    ``FitResult.n_iterations`` is SciPy's Jacobian-evaluation count (one per
    trust-region iteration).

    Examples
    --------
    >>> from torchspin import FitOptions
    >>> opt = FitOptions(method='levmar', baseline=0)
    >>> opt.target, opt.max_iter
    ('fcn', 2000)
    """
    method: str = 'simplex'
    autoscale: str = 'lsq'
    lsq_negative_scale: str = 'clip'
    baseline: int = -1
    target: str = 'fcn'
    max_iter: int = 2000
    tol_fun: float = 1e-6
    tol_x: float = 1e-6
    verbosity: int = 0
    callback: Optional[Callable] = None
    compute_uncertainties: bool = True
    # Grid search options
    grid_points: int = 10
    # Monte Carlo / simulated annealing options
    mc_samples: int = 1000
    mc_temp_start: float = 1.0
    mc_temp_end: float = 1e-4
    # Genetic algorithm options
    ga_population: int = 50
    ga_mutation: float = 0.1
    ga_crossover: float = 0.7
    ga_elite: int = 5
    # Particle swarm options
    swarm_size: int = 30
    swarm_inertia: float = 0.7
    swarm_cognitive: float = 1.4
    swarm_social: float = 1.4
    # Levenberg-Marquardt options
    lm_lambda: float = 1e-3
    lm_delta: float = 1e-3        # EasySpin FitOpt.delta (finite-difference step in [-1,1] units)
    lm_gradient_tol: float = 1e-5   # EasySpin FitOpt.Gradient (= TolFun)
    lm_step_tol: float = 1e-8
    # Trust-region reflective least-squares options. max_iter is the solver's
    # max_nfev budget for this method (finite-difference probes are additional).
    # diff_step is relative to the physical parameters, as in scipy.least_squares.
    trf_diff_step: Optional[float] = None
    trf_x_scale: Union[str, float, np.ndarray] = 'jac'
    trf_gradient_tol: float = 1e-7
    # Parallel model evaluations. Population methods (swarm, genetic,
    # montecarlo, grid, global) evaluate each population concurrently; levmar,
    # trf, lbfgsb and the uncertainty estimate evaluate the independent
    # finite-difference probes of each Jacobian or gradient concurrently (trf
    # and lbfgsb need SciPy >= 1.16; older SciPy runs them sequentially). For
    # trf and lbfgsb SciPy evaluates the base point in this process and the
    # probes in the workers, whose thread count differs; torch reductions then
    # differ at ~1e-14, which SciPy's ~1.5e-8 relative step turns into a
    # Jacobian error of ~1e-6 of the signal scale (measured on pepper models),
    # far below what changes a fit. levmar and the uncertainty Jacobian evaluate
    # the base point and all probes in the workers.
    # n_workers=1 is sequential; an int > 1 or 'auto' uses a spawned process
    # pool with the cores split between the workers (workers='auto'|'processes';
    # 'threads' forces a thread pool, which gains little because the forward
    # model holds the GIL). 'auto' = the cores this process may use (affinity /
    # SLURM aware), at most 32, and no more than about 0.5 GB per worker within
    # half the available memory: each worker is a separate Python process with
    # its own torch.
    # Models defined in a notebook cell need cloudpickle (pip install
    # cloudpickle) to reach the workers; otherwise a thread pool is used.
    n_workers: Union[int, str] = 1
    workers: str = 'auto'
    # GPUs for process workers, e.g. ['cuda:0', 'cuda:1']: each spawned worker
    # is pinned to one of them, round-robin, by setting CUDA_VISIBLE_DEVICES
    # before CUDA starts in it, so a model that uses device='cuda' runs on its
    # own GPU. None (default) leaves the workers' GPU visibility unchanged.
    # Ignored by the thread backend (one process sees one set of devices).
    devices: Optional[list] = None
    # Seed for numpy RNGs in population-based optimizers. None (default)
    # = non-deterministic (each run produces a different particle/sample
    # realization). An int (e.g. seed=42) makes the fit bit-reproducible
    # across runs — recommended for tutorials, regression tests, and
    # manuscript figures where the numbers must be citable.
    seed: Optional[int] = None
    # Real-time progress and early stop (all methods).
    #   progress: None (silent) | 'text' (one flushed line per interval/every N
    #     evaluations: iteration, evaluations, current and best RMSD, elapsed,
    #     evaluations/s, ETA where the budget is known) | 'bar' (tqdm bar when tqdm
    #     is importable, else text) | callable(info: dict) — return False to stop.
    #   A resource summary (torch threads, worker processes, CPUs, device, RSS)
    #   is printed once at the start and a totals line at the end.
    #   With n_workers > 1 updates arrive per evaluation chunk, not per evaluation.
    #   max_time: wall-clock budget in seconds; stop_when: predicate(info) -> True
    #   stops.  A stop (or a KeyboardInterrupt / kernel interrupt) returns the
    #   best-so-far FitResult with success=False and interrupted=True.
    #   An exception raised by a progress callable does not stop the fit (the
    #   first one per fit is reported with a warning); an exception raised by
    #   stop_when propagates out of esfit, since a broken stop condition
    #   cannot be honored.
    progress: Union[None, str, Callable] = None
    progress_every: Optional[int] = None
    progress_interval: float = 1.0
    max_time: Optional[float] = None
    stop_when: Optional[Callable] = None


@dataclass
class FitResult:
    """Result of :func:`esfit`.

    "Active" parameters are those that were varied (``lb < ub``); parameters
    fixed with ``vary = 0`` or ``lb == ub`` are not part of ``pfit``. All
    arrays are NumPy arrays on the CPU, in data units unless stated otherwise.

    Parameters
    ----------
    pfit : numpy.ndarray, shape (n_active,)
        Best-fit values of the active parameters, in the order of ``pnames``.
    pnames : list of str, length n_active
        Names of the active parameters: ``'p0'``, ``'p1'``, ... (index into
        the full ``p0`` vector) for array-style fits, ``'Sys.lw[1]'``,
        ``'Sys.g[0,2]'``, ``'Exp.mwFreq'``, ... for dict-style fits.
    p_start : numpy.ndarray, shape (n_active,)
        Start values of the active parameters.
    p_fixed : numpy.ndarray of bool, shape (n_total,)
        Fixed-parameter mask over all parameters, fixed ones included (array
        style: the order of ``p0``; dict style: the order of the fields in
        ``vary``/``lb``, arrays flattened in C order). Note that it is longer
        than ``pfit`` when parameters are fixed.
    fit : numpy.ndarray, shape (n_data,)
        Fitted curve as compared with the data: ``scale * fit_raw +
        baseline``.
    fit_raw : numpy.ndarray, shape (n_data,)
        Model output at ``pfit``, flattened, before scaling and baseline.
    residuals : numpy.ndarray, shape (n_data,)
        ``fit - data``, before the ``FitOptions.target`` transformation.
    scale : float
        Scale factor applied to ``fit_raw`` (1.0 for ``autoscale='none'``).
    baseline : numpy.ndarray, shape (n_data,)
        Fitted polynomial baseline (zeros for ``FitOptions.baseline=-1``).
    rmsd : float
        Root-mean-square of the target-transformed residuals at ``pfit``: the
        quantity the optimizer minimizes.
    ssr : float
        Sum of squares of the target-transformed residuals
        (``n_data * rmsd**2``).
    n_params : int
        Number of active parameters (``n_active``).
    n_data : int
        Number of data points.
    algorithm : str
        ``FitOptions.method`` used.
    n_iterations : int
        Iterations reported by the optimizer: SciPy's ``nit`` for
        ``'simplex'``, ``'lbfgsb'`` and ``'powell'``; Jacobian evaluations for
        ``'trf'``; LM iterations, swarm iterations or generations for
        ``'levmar'``, ``'swarm'``, ``'genetic'``; the sum of both phases for
        ``'global'``; model evaluations for ``'grid'`` and ``'montecarlo'``;
        the tracker's iteration count for an interrupted fit.
    success : bool
        SciPy's convergence flag for ``'simplex'``, ``'lbfgsb'``,
        ``'powell'`` and ``'trf'``; always ``True`` for the other methods
        unless the fit was interrupted (then ``False``).
    message : str
        Optimizer termination message.
    pstd : numpy.ndarray, shape (n_active,), or None
        Standard deviations of the active parameters from the HC1 sandwich
        covariance. ``nan`` for a parameter that has no effect on the
        residuals at the finite-difference step (esfit warns). ``None`` when
        ``FitOptions.compute_uncertainties`` is ``False``, the fit did not
        succeed, or the covariance could not be computed.
    ci95 : numpy.ndarray, shape (n_active, 2), or None
        ``[pfit - 1.96 * pstd, pfit + 1.96 * pstd]`` per parameter (normal
        approximation). ``None`` as for ``pstd``.
    cov : numpy.ndarray, shape (n_active, n_active), or None
        Parameter covariance matrix (``nan`` rows/columns for parameters
        without effect). ``None`` as for ``pstd``.
    corr : numpy.ndarray, shape (n_active, n_active), or None
        Correlation matrix derived from ``cov``. ``None`` as for ``pstd``.
    r_squared : float
        Coefficient of determination, intended as ``1 - SS_res / SS_tot`` of
        the untransformed residuals; see Notes.
    reduced_chi_square : float
        ``sum(residuals**2) / max(n_data - n_active, 1)`` of the
        untransformed residuals: an estimate of the noise variance in data
        units squared, not a chi-square normalized by a known noise level.
    noise_std : float
        ``sqrt(reduced_chi_square)``: estimated noise standard deviation in
        data units.
    n_evaluations : int or None
        Model evaluations of the optimizer, all methods and phases (the
        initial evaluation at ``p0`` included; the final evaluation and the
        uncertainty estimate not).
    elapsed_s : float or None
        Optimizer wall time in seconds (excludes the initial evaluation at
        ``p0`` and the uncertainty estimate).
    interrupted : bool, default False
        ``True`` when the fit was stopped early (``KeyboardInterrupt``,
        ``FitOptions.max_time``, ``stop_when`` or a ``progress`` callable
        returning ``False``); ``pfit`` is then the best point found so far.
    n_failed_evals : int, default 0
        Objective evaluations at which the model raised ``ValueError``,
        ``RuntimeError`` or ``LinAlgError``. Each such evaluation is
        penalized with RMSD = 1e10 so the optimizer moves away; the first
        failure of a fit is reported with a warning. A nonzero count means
        part of the parameter space could not be simulated (a bug in the
        model, or parameter ranges outside its domain).

    See Also
    --------
    esfit, FitOptions

    Notes
    -----
    ``r_squared`` reconstructs the data as ``fit + residuals`` although
    ``residuals = fit - data``, so ``SS_tot`` is taken about ``2 * fit -
    data`` instead of the data. For a close fit the difference is small; for
    a poor fit the value can be far off (even hugely negative). Compute
    ``1 - sum(res.residuals**2) / sum((data - data.mean())**2)`` yourself
    when the exact value matters.
    """
    pfit: np.ndarray
    pnames: list[str]
    p_start: np.ndarray
    p_fixed: np.ndarray
    fit: np.ndarray
    fit_raw: np.ndarray
    residuals: np.ndarray
    scale: float
    baseline: np.ndarray
    rmsd: float
    ssr: float
    n_params: int
    n_data: int
    algorithm: str
    n_iterations: int
    success: bool
    message: str
    pstd: Optional[np.ndarray] = None
    ci95: Optional[np.ndarray] = None
    cov: Optional[np.ndarray] = None
    corr: Optional[np.ndarray] = None
    r_squared: Optional[float] = None
    reduced_chi_square: Optional[float] = None
    noise_std: Optional[float] = None
    n_evaluations: Optional[int] = None      # objective evaluations (all methods, all phases)
    elapsed_s: Optional[float] = None        # optimizer wall time (excludes the initial RMSD evaluation)
    interrupted: bool = False                # True when stopped early (interrupt, max_time, stop_when, callback)
    n_failed_evals: int = 0                  # model evaluations that raised and were penalized with rmsd 1e10


class TorchSpinParameterHandler:
    """Handle dict-style parameters for torchspin simulators.
    
    Supports fitting parameters in SpinSystem and Experiment objects by
    specifying which fields to vary.
    
    Examples
    --------
    >>> from torchspin import SpinSystem, Experiment
    >>> sys0 = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
    >>> exp = Experiment(mwFreq=9.5, Range=[330, 350])
    >>> p0 = {'Sys': sys0, 'Exp': exp}
    >>> vary = {'Sys': {'g': [[0.01, 0.01, 0.01]]}}
    >>> handler = TorchSpinParameterHandler(p0, vary)
    """
    
    def __init__(
        self,
        p0: dict,
        vary: Optional[dict] = None,
        lb: Optional[dict] = None,
        ub: Optional[dict] = None
    ):
        """
        Initialize handler for dict-style parameters.
        
        Parameters
        ----------
        p0 : dict
            Initial parameter values, e.g. {'Sys': SpinSystem(...), 'Exp': Experiment(...)}
        vary : dict, optional
            Variation specification, e.g. {'Sys': {'g': [[0.01, 0.01, 0.01]]}}
        lb, ub : dict, optional
            Explicit bounds (mutually exclusive with vary)
        """
        from torchspin import SpinSystem, Experiment
        
        self.p0_dict = p0
        self.vary_dict = vary
        self.lb_dict = lb
        self.ub_dict = ub
        
        # Parse vary/bounds specification
        if vary is not None and (lb is not None or ub is not None):
            raise ValueError("Cannot specify both 'vary' and 'lb'/'ub'")
        
        # Build parameter map: list of (key, field, indices, value, lb, ub)
        self.param_map = []
        self.pnames = []
        
        if vary is not None:
            self._parse_vary()
        elif lb is not None or ub is not None:
            self._parse_bounds()
        else:
            raise ValueError("Must specify either 'vary' or 'lb'/'ub'")
        
        # Create flat parameter vector
        self.n_params = len(self.param_map)
        self.p0_vec = np.array([entry['value'] for entry in self.param_map])
        self.lb = np.array([entry['lb'] for entry in self.param_map])
        self.ub = np.array([entry['ub'] for entry in self.param_map])
        
        # Check for fixed parameters
        self.fixed_mask = self.lb == self.ub
        self.active_mask = ~self.fixed_mask
        self.n_active = np.sum(self.active_mask)
        
        if self.n_active == 0:
            raise ValueError("No variable parameters - all parameters are fixed (lb == ub)")
        
        self.p0_active = self.p0_vec[self.active_mask]
        self.lb_active = self.lb[self.active_mask]
        self.ub_active = self.ub[self.active_mask]
    
    # Fields that cannot be negative, so that p0 - vary is clipped at zero for
    # them rather than handing the optimizer an unphysical lower bound (EasySpin
    # esfit.m: nonnegFieldNames).  Only reachable in this dict style, where the
    # field name is known; the plain-vector style has no names to go on, which
    # is also true of EasySpin.
    _NONNEGATIVE_FIELDS = frozenset({
        'lw', 'lwpp', 'lwEndor', 'HStrain', 'HStrainL', 'gStrain', 'DStrain', 'AStrain',
        'weight', 'initState', 'tcorr', 'Diff', 'T1', 'T2',
    })

    def _lower_bound(self, field: str, value: float, vary: float) -> float:
        lb = float(value) - float(vary)
        if field in self._NONNEGATIVE_FIELDS and lb < 0.0:
            return 0.0
        return lb

    def _parse_vary(self):
        """Parse vary specification to build parameter map."""
        for key, vary_spec in self.vary_dict.items():
            if key not in self.p0_dict:
                raise ValueError(f"Key '{key}' in vary not found in p0")
            
            obj = self.p0_dict[key]
            
            # Iterate over fields to vary
            for field, vary_val in vary_spec.items():
                if not hasattr(obj, field):
                    raise ValueError(f"Object '{key}' has no field '{field}'")
                
                p0_val = getattr(obj, field)
                
                # Convert torch tensors to numpy
                if isinstance(p0_val, torch.Tensor):
                    p0_val = p0_val.cpu().numpy()
                if isinstance(vary_val, torch.Tensor):
                    vary_val = vary_val.cpu().numpy()
                
                # Handle different value types
                if isinstance(p0_val, (list, np.ndarray)):
                    p0_array = np.asarray(p0_val)
                    vary_array = np.asarray(vary_val)
                    
                    if p0_array.shape != vary_array.shape:
                        raise ValueError(
                            f"{key}.{field}: p0 shape {p0_array.shape} != vary shape {vary_array.shape}"
                        )
                    
                    # Flatten and add each element
                    p0_flat = p0_array.ravel()
                    vary_flat = vary_array.ravel()
                    
                    for i, (p0_elem, vary_elem) in enumerate(zip(p0_flat, vary_flat)):
                        # Compute multi-dimensional index
                        idx = np.unravel_index(i, p0_array.shape)
                        idx_str = ','.join(map(str, idx))
                        
                        self.param_map.append({
                            'key': key,
                            'field': field,
                            'index': idx,
                            'value': float(p0_elem),
                            'lb': self._lower_bound(field, p0_elem, vary_elem),
                            'ub': float(p0_elem + vary_elem)
                        })
                        self.pnames.append(f"{key}.{field}[{idx_str}]")
                
                elif isinstance(p0_val, (int, float, np.number)):
                    # Scalar parameter
                    vary_scalar = float(vary_val)
                    self.param_map.append({
                        'key': key,
                        'field': field,
                        'index': None,
                        'value': float(p0_val),
                        'lb': self._lower_bound(field, p0_val, vary_scalar),
                        'ub': float(p0_val + vary_scalar)
                    })
                    self.pnames.append(f"{key}.{field}")
                
                else:
                    raise TypeError(
                        f"{key}.{field}: unsupported type {type(p0_val)}"
                    )
    
    def _parse_bounds(self):
        """Parse explicit lb/ub bound dictionaries.

        Both ``lb`` and ``ub`` must be given, mirror each other in structure,
        and reference fields present in ``p0`` — same layout as ``vary``,
        e.g. ``lb={'Sys': {'g': [[1.9, 1.9, 1.9]]}}``.
        """
        if self.lb_dict is None or self.ub_dict is None:
            raise ValueError(
                "Dict-style bounds require both 'lb' and 'ub'.")
        if set(self.lb_dict.keys()) != set(self.ub_dict.keys()):
            raise ValueError("'lb' and 'ub' must have the same keys.")

        for key, lb_spec in self.lb_dict.items():
            if key not in self.p0_dict:
                raise ValueError(f"Key '{key}' in lb/ub not found in p0")
            ub_spec = self.ub_dict[key]
            if set(lb_spec.keys()) != set(ub_spec.keys()):
                raise ValueError(
                    f"'{key}': lb and ub must specify the same fields.")
            obj = self.p0_dict[key]

            for field, lb_val in lb_spec.items():
                if not hasattr(obj, field):
                    raise ValueError(f"Object '{key}' has no field '{field}'")
                ub_val = ub_spec[field]
                p0_val = getattr(obj, field)

                if isinstance(p0_val, torch.Tensor):
                    p0_val = p0_val.cpu().numpy()
                if isinstance(lb_val, torch.Tensor):
                    lb_val = lb_val.cpu().numpy()
                if isinstance(ub_val, torch.Tensor):
                    ub_val = ub_val.cpu().numpy()

                if isinstance(p0_val, (list, np.ndarray)):
                    p0_array = np.asarray(p0_val, dtype=float)
                    lb_array = np.asarray(lb_val, dtype=float)
                    ub_array = np.asarray(ub_val, dtype=float)
                    if (p0_array.shape != lb_array.shape
                            or p0_array.shape != ub_array.shape):
                        raise ValueError(
                            f"{key}.{field}: p0 shape {p0_array.shape} != "
                            f"lb shape {lb_array.shape} / ub shape "
                            f"{ub_array.shape}")
                    p0_flat = p0_array.ravel()
                    lb_flat = lb_array.ravel()
                    ub_flat = ub_array.ravel()
                    for i, (p0_e, lb_e, ub_e) in enumerate(
                            zip(p0_flat, lb_flat, ub_flat)):
                        if lb_e > ub_e:
                            raise ValueError(
                                f"{key}.{field}: lb > ub at element {i}")
                        if not (lb_e <= p0_e <= ub_e):
                            raise ValueError(
                                f"{key}.{field}: p0 value {p0_e} outside "
                                f"[{lb_e}, {ub_e}] at element {i}")
                        idx = np.unravel_index(i, p0_array.shape)
                        idx_str = ','.join(map(str, idx))
                        self.param_map.append({
                            'key': key,
                            'field': field,
                            'index': idx,
                            'value': float(p0_e),
                            'lb': float(lb_e),
                            'ub': float(ub_e),
                        })
                        self.pnames.append(f"{key}.{field}[{idx_str}]")

                elif isinstance(p0_val, (int, float, np.number)):
                    lb_s, ub_s = float(lb_val), float(ub_val)
                    if lb_s > ub_s:
                        raise ValueError(f"{key}.{field}: lb > ub")
                    if not (lb_s <= float(p0_val) <= ub_s):
                        raise ValueError(
                            f"{key}.{field}: p0 value {p0_val} outside "
                            f"[{lb_s}, {ub_s}]")
                    self.param_map.append({
                        'key': key,
                        'field': field,
                        'index': None,
                        'value': float(p0_val),
                        'lb': lb_s,
                        'ub': ub_s,
                    })
                    self.pnames.append(f"{key}.{field}")

                else:
                    raise TypeError(
                        f"{key}.{field}: unsupported type {type(p0_val)}")
    
    def to_vector(self, params: Union[np.ndarray, dict]) -> np.ndarray:
        """Convert parameters to active parameter vector."""
        if isinstance(params, np.ndarray):
            return params[self.active_mask]
        
        # Extract from dict
        pvec = np.zeros(self.n_params)
        for i, entry in enumerate(self.param_map):
            obj = params[entry['key']]
            val = getattr(obj, entry['field'])
            
            if entry['index'] is not None:
                # Convert to numpy array to handle tensor/list indexing
                val_array = np.asarray(val)
                pvec[i] = val_array[entry['index']]
            else:
                pvec[i] = val
        
        return pvec[self.active_mask]
    
    def to_dict(self, pvec_active: np.ndarray) -> dict:
        """Convert active parameter vector to dict structure."""
        import copy
        import torch
        from torchspin import SpinSystem, Experiment
        
        # Expand active vector to full vector
        pvec_full = self.p0_vec.copy()
        pvec_full[self.active_mask] = pvec_active
        
        # Deep copy p0 dict
        result = copy.deepcopy(self.p0_dict)
        
        # Update fields
        for i, entry in enumerate(self.param_map):
            obj = result[entry['key']]
            field = entry['field']
            new_val = pvec_full[i]
            
            # Current value in the copy being built (type preserved: tensor,
            # list, float). It must be the copy, not p0: reading p0 for every
            # element made each element of an array field overwrite the ones
            # set before it, so only its last element ever reached the model.
            original_val = getattr(obj, field)

            if entry['index'] is not None:
                # Array parameter - update single element
                if isinstance(original_val, torch.Tensor):
                    # Convert to numpy, modify, convert back to tensor
                    arr = original_val.detach().cpu().numpy().copy()
                    arr[entry['index']] = new_val
                    arr_tensor = torch.tensor(arr, dtype=original_val.dtype, device=original_val.device)
                    setattr(obj, field, arr_tensor)
                else:
                    arr = np.array(original_val, dtype=float)
                    arr[entry['index']] = new_val
                    setattr(obj, field, arr.tolist() if isinstance(original_val, list) else arr)
            else:
                # Scalar parameter
                if isinstance(original_val, torch.Tensor):
                    # Convert scalar to tensor with original dtype/device
                    setattr(obj, field, torch.tensor(float(new_val), dtype=original_val.dtype, device=original_val.device))
                else:
                    setattr(obj, field, float(new_val))
        
        return result
    
    def to_full(self, pvec_active: np.ndarray) -> np.ndarray:
        """Convert active parameter vector to full parameter vector."""
        pvec_full = self.p0_vec.copy()
        pvec_full[self.active_mask] = pvec_active
        return pvec_full
    
    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return bounds for active parameters."""
        return self.lb_active, self.ub_active
    
    def get_names(self) -> list[str]:
        """Return names of active parameters."""
        return [name for i, name in enumerate(self.pnames) if self.active_mask[i]]


class ParameterHandler:
    """Manages parameter conversions and bounds for optimization.
    
    Handles conversion between dict-style and vector-style parameters,
    and enforces parameter bounds during optimization.
    """
    
    def __init__(
        self,
        p0: Union[np.ndarray, dict],
        vary: Optional[Union[np.ndarray, dict]] = None,
        lb: Optional[Union[np.ndarray, dict]] = None,
        ub: Optional[Union[np.ndarray, dict]] = None
    ):
        """
        Initialize parameter handler.
        
        Parameters
        ----------
        p0 : array or dict
            Initial parameter values
        vary : array or dict, optional
            Allowed variation (p0 ± vary). Mutually exclusive with lb/ub.
        lb, ub : array or dict, optional
            Lower and upper bounds. Mutually exclusive with vary.
        """
        self.is_dict_style = isinstance(p0, dict)
        
        if self.is_dict_style:
            # Use TorchSpinParameterHandler for dict-style
            from torchspin.esfit import TorchSpinParameterHandler
            self._handler = TorchSpinParameterHandler(p0, vary, lb, ub)
            # Expose interface
            self.p0 = self._handler.p0_vec
            self.n_params = self._handler.n_params
            self.n_active = self._handler.n_active
            self.lb = self._handler.lb
            self.ub = self._handler.ub
            self.fixed_mask = self._handler.fixed_mask
            self.active_mask = self._handler.active_mask
            self.p0_active = self._handler.p0_active
            self.lb_active = self._handler.lb_active
            self.ub_active = self._handler.ub_active
            self.pnames = self._handler.pnames
            return
        
        # Array style
        self.p0 = np.asarray(p0, dtype=float)
        self.n_params = len(self.p0)
        
        # Parse bounds
        if vary is not None and (lb is not None or ub is not None):
            raise ValueError("Cannot specify both 'vary' and 'lb'/'ub'")
        
        if vary is not None:
            vary = np.asarray(vary, dtype=float)
            if len(vary) != self.n_params:
                raise ValueError(f"vary length {len(vary)} != p0 length {self.n_params}")
            self.lb = self.p0 - vary
            self.ub = self.p0 + vary
        elif lb is not None or ub is not None:
            if lb is None:
                lb = np.full(self.n_params, -np.inf)
            if ub is None:
                ub = np.full(self.n_params, np.inf)
            self.lb = np.asarray(lb, dtype=float)
            self.ub = np.asarray(ub, dtype=float)
            if len(self.lb) != self.n_params or len(self.ub) != self.n_params:
                raise ValueError("Bounds length must match p0 length")
        else:
            raise ValueError(
                "No bounds specified. Provide 'vary' (±variation), or "
                "'lb'/'ub' (explicit bounds). Implicit ±50% defaults "
                "are unreliable for parameters near zero or negative."
            )
        
        # Identify fixed parameters (lb == ub)
        self.fixed_mask = self.lb == self.ub
        self.active_mask = ~self.fixed_mask
        self.n_active = np.sum(self.active_mask)
        
        if self.n_active == 0:
            raise ValueError("No variable parameters - all parameters are fixed (lb == ub)")
        
        # Extract active parameters
        self.p0_active = self.p0[self.active_mask]
        self.lb_active = self.lb[self.active_mask]
        self.ub_active = self.ub[self.active_mask]
        
        # Parameter names
        self.pnames = [f'p{i}' for i in range(self.n_params)]
    
    def to_vector(self, params: Union[np.ndarray, dict]) -> np.ndarray:
        """Convert parameters to active parameter vector."""
        if self.is_dict_style:
            return self._handler.to_vector(params)
        p = np.asarray(params, dtype=float)
        return p[self.active_mask]
    
    def to_full(self, pvec_active: np.ndarray) -> np.ndarray:
        """Convert active parameter vector to full parameter vector."""
        if self.is_dict_style:
            return self._handler.to_full(pvec_active)
        p_full = self.p0.copy()
        p_full[self.active_mask] = pvec_active
        return p_full
    
    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return bounds for active parameters."""
        return self.lb_active, self.ub_active
    
    def get_names(self) -> list[str]:
        """Return names of active parameters."""
        return [name for i, name in enumerate(self.pnames) if self.active_mask[i]]


class ModelWrapper:
    """Wraps model function for consistent interface.
    
    Handles both custom functions (taking parameter vector) and
    torchspin simulators (taking Sys, Exp structures).
    """
    
    def __init__(
        self,
        model_fn: Callable,
        p0: Union[np.ndarray, dict],
        exp_data_len: int
    ):
        """
        Initialize model wrapper.
        
        Parameters
        ----------
        model_fn : callable
            Simulation function
        p0 : array or dict
            Initial parameters (determines calling style)
        exp_data_len : int
            Expected output length for validation
        """
        self.model_fn = model_fn
        self.data_len = exp_data_len
        self.is_dict_style = isinstance(p0, dict)
        
        # Detect if this is a torchspin simulator
        if self.is_dict_style:
            self.is_torchspin = self._detect_torchspin_simulator(model_fn)
        else:
            self.is_torchspin = False
    
    def _detect_torchspin_simulator(self, fn: Callable) -> bool:
        """Check if function is a torchspin simulator (pepper, garlic, etc.)."""
        # Check function name
        if hasattr(fn, '__name__'):
            name = fn.__name__
            if name in ['pepper', 'garlic', 'chili', 'salt']:
                return True
        
        # Check module
        if hasattr(fn, '__module__'):
            if fn.__module__ and 'torchspin' in fn.__module__:
                return True
        
        return False
    
    @staticmethod
    def _garlic_uses_binning(sys, exp) -> bool:
        """Whether garlic's default accumulation for ``sys``/``exp`` is
        'binning': not in the fast-motion regime (any component with a
        positive correlation time) and no mwPhase (which needs 'explicit')."""
        if exp is not None and float(getattr(exp, 'mwPhase', 0.0) or 0.0) != 0.0:
            return False
        for c in (sys if isinstance(sys, (list, tuple)) else [sys]):
            for name, tf in (('logtcorr', lambda v: 10.0 ** v), ('tcorr', lambda v: v)):
                v = getattr(c, name, None)
                if v is not None:
                    v0 = float(torch.as_tensor(v, dtype=torch.float64).detach().reshape(-1)[0])
                    if tf(v0) > 0:
                        return False
                    break
        return True

    def _fd_friendly_options(self, opt, sys=None, exp=None):
        """For garlic, replace the default nearest-bin accumulation by the
        two-bin linear split when the user left ``Opt.AccumMethod`` unset.

        Nearest-bin sticks make the spectrum piecewise-constant in g and A, so
        every finite-difference Jacobian (levmar, trf, uncertainties) is zero
        for them. 'linear' is the form garlic itself uses on the autograd graph.
        Fast-motion and mwPhase runs default to 'explicit' and are left alone.
        """
        if getattr(self.model_fn, '__name__', '') != 'garlic':
            return opt
        if not self._garlic_uses_binning(sys, exp):
            return opt
        from torchspin.experiment import Options
        if opt is None:
            return Options(AccumMethod='linear')
        if getattr(opt, 'AccumMethod', None) is None:
            return dataclasses.replace(opt, AccumMethod='linear')
        return opt

    # After the optimizer has returned, esfit evaluates the final point again for
    # the residuals and the uncertainty estimate evaluates it as the center of its
    # stencil.  The last few outputs are kept, keyed by the exact bytes of the
    # parameter vector, and served once ``serve_memo`` is set (after the
    # optimizer), so those repeats cost nothing and return the very same array.
    # During the optimization every evaluation still reaches the model, so the
    # evaluation counts, budgets and interrupt behavior are unchanged.  Stochastic
    # models (cardamom without a seed) return the earlier draw at a served repeat.
    _MEMO_SIZE = 8
    serve_memo = False

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop('_memo', None)          # not shipped to worker processes
        return state

    def __call__(self, pvec_active: np.ndarray, param_handler: ParameterHandler) -> np.ndarray:
        key = (id(param_handler), np.asarray(pvec_active, dtype=float).tobytes())
        memo = getattr(self, '_memo', ())
        if self.serve_memo:
            for k, out in memo:
                if k == key:
                    return out.copy()
        output = self._evaluate(pvec_active, param_handler)
        # replace the list as a whole: concurrent callers (thread backend) read snapshots
        self._memo = ([(key, output.copy())] + [e for e in memo if e[0] != key])[:self._MEMO_SIZE]
        return output

    def _evaluate(self, pvec_active: np.ndarray, param_handler: ParameterHandler) -> np.ndarray:
        """
        Evaluate model with active parameter vector.
        
        Parameters
        ----------
        pvec_active : np.ndarray
            Active parameter values
        param_handler : ParameterHandler
            Parameter handler for conversion
        
        Returns
        -------
        sim : np.ndarray
            Simulated data vector (1D)
        """
        if self.is_dict_style:
            # Convert to dict and call torchspin simulator
            params_dict = param_handler._handler.to_dict(pvec_active)
            
            if self.is_torchspin:
                # Torchspin simulator: call with (Sys, Exp) or (Sys, Exp, Opt)
                sys = params_dict.get('Sys')
                exp = params_dict.get('Exp')
                opt = params_dict.get('Opt', None)
                opt = self._fd_friendly_options(opt, sys, exp)

                if opt is not None:
                    B, spec = self.model_fn(sys, exp, opt)
                else:
                    B, spec = self.model_fn(sys, exp)
                
                # Extract spectrum
                output = spec
            else:
                # Custom function expecting dict
                output = self.model_fn(params_dict)
        else:
            # Convert to full parameter vector
            p_full = param_handler.to_full(pvec_active)
            
            # Call model function
            output = self.model_fn(p_full)
        
        # Convert output to numpy array
        if isinstance(output, torch.Tensor):
            output = output.detach().cpu().numpy()
        output = np.asarray(output, dtype=float).ravel()
        
        # Validate output length
        if len(output) != self.data_len:
            raise ValueError(
                f"Model output length {len(output)} != data length {self.data_len}"
            )
        
        return output


def _baseline_design(n: int, baseline_order: int) -> np.ndarray:
    """Polynomial baseline columns ``[1, x, x², ...]`` on ``x = linspace(0, 1, n)``."""
    x = np.linspace(0, 1, n)
    return np.column_stack([x**k for k in range(baseline_order + 1)])


def _fit_polynomial_baseline(resid: np.ndarray, baseline_order: int) -> np.ndarray:
    """Least-squares polynomial baseline of order ``baseline_order`` fitted to
    ``resid`` (zero vector for ``baseline_order < 0``)."""
    n = len(resid)
    if baseline_order < 0:
        return np.zeros(n)
    P = _baseline_design(n, baseline_order)
    coeffs, _, _, _ = np.linalg.lstsq(P, resid, rcond=None)
    return P @ coeffs


def _fit_scale_baseline_lsq(
    sim: np.ndarray,
    data: np.ndarray,
    baseline_order: int,
    negative_scale: str = 'clip',
    return_negative: bool = False,
):
    """
    Fit a non-negative scale and a polynomial baseline by least squares.

    Solves ``min ||scale·sim + P·b − data||`` subject to ``scale ≥ 0``, where
    ``P = [1, x, x², ...]``. The scale is forced non-negative, as in EasySpin,
    because a free sign lets the optimizer "fit" by mirroring the model, which
    creates spurious minima on derivative spectra.

    Parameters
    ----------
    sim : np.ndarray
        Simulated data
    data : np.ndarray
        Experimental data
    baseline_order : int
        Polynomial order for baseline (-1 for none)
    negative_scale : {'clip', 'abs'}
        What to do when the unconstrained solve gives a negative scale.
        ``'clip'`` (default) returns the constrained optimum: ``scale = 0``
        with the baseline refitted alone. ``'abs'`` is EasySpin's
        ``coeffs(1) = abs(coeffs(1))``: it mirrors the scale but keeps the
        baseline solved for the opposite sign, which is not a least-squares
        fit of anything.
    return_negative : bool
        Also return whether the unconstrained scale was negative.

    Returns
    -------
    scale : float
        Optimal non-negative scale factor
    baseline : np.ndarray
        Baseline vector
    negative : bool
        Only with ``return_negative=True``.
    """
    if negative_scale not in ('clip', 'abs'):
        raise ValueError(f"lsq_negative_scale must be 'clip' or 'abs', got {negative_scale!r}")
    n = len(sim)

    if baseline_order < 0:
        scale = float((sim @ data) / (sim @ sim + 1e-12))
        baseline = np.zeros(n)
        negative = scale < 0
        if negative:
            scale = abs(scale) if negative_scale == 'abs' else 0.0
    else:
        P = _baseline_design(n, baseline_order)
        D = np.column_stack([sim, P])
        coeffs, _, _, _ = np.linalg.lstsq(D, data, rcond=None)
        scale = float(coeffs[0])
        negative = scale < 0
        if not negative:
            baseline = P @ coeffs[1:]
        elif negative_scale == 'abs':
            # EasySpin parity (esfit.m: coeffs(1) = abs(coeffs(1)))
            scale = abs(scale)
            baseline = P @ coeffs[1:]
        else:
            # One sign-constrained coefficient in a convex problem: the
            # constrained optimum sits on the boundary scale = 0, with the
            # baseline refitted on its own.
            scale = 0.0
            baseline = _fit_polynomial_baseline(data, baseline_order)

    if return_negative:
        return scale, baseline, negative
    return scale, baseline


def _target_transform_adjoint(X: np.ndarray, target: str) -> np.ndarray:
    """``T' X`` along axis 0, where ``T`` is the linear map of
    :func:`_apply_target_transform` (``T r`` = transformed residuals)."""
    if target == 'fcn':
        return X
    if target == 'int':                      # T = lower-triangular ones: reverse cumsum
        return np.cumsum(X[::-1], axis=0)[::-1]
    if target == 'diff':                     # (T r)_i = r_i - r_{i-1}: (T'y)_i = y_i - y_{i+1}
        out = X.copy()
        out[:-1] -= X[1:]
        return out
    raise ValueError(f"Unknown target: '{target}'. Use 'fcn', 'int', or 'diff'")


def _apply_target_transform(residuals: np.ndarray, target: str) -> np.ndarray:
    """Apply target transformation to residuals."""
    if target == 'fcn':
        return residuals
    elif target == 'int':
        return np.cumsum(residuals)
    elif target == 'diff':
        return np.diff(residuals, prepend=0)
    else:
        raise ValueError(f"Unknown target: '{target}'. Use 'fcn', 'int', or 'diff'")


def compute_residuals(
    pvec_active: np.ndarray,
    data: np.ndarray,
    model_wrapper: ModelWrapper,
    param_handler: ParameterHandler,
    options: FitOptions
) -> tuple[np.ndarray, float, dict]:
    """
    Compute residuals and RMSD for given parameters.
    
    Parameters
    ----------
    pvec_active : np.ndarray
        Active parameter values
    data : np.ndarray
        Experimental data
    model_wrapper : ModelWrapper
        Model function wrapper
    param_handler : ParameterHandler
        Parameter handler
    options : FitOptions
        Fitting options
    
    Returns
    -------
    residuals : np.ndarray
        Residuals after scaling/baseline/target transform
    rmsd : float
        Root mean square deviation
    info : dict
        Additional info (sim, scale, baseline, fit, etc.)
    """
    # 1. Evaluate model
    sim = model_wrapper(pvec_active, param_handler)
    
    # 2. Apply autoscaling
    negative_scale = False
    if options.autoscale == 'lsq':
        scale, baseline, negative_scale = _fit_scale_baseline_lsq(
            sim, data, options.baseline,
            negative_scale=options.lsq_negative_scale, return_negative=True)
        fit = scale * sim + baseline
    elif options.autoscale == 'maxabs':
        scale = np.max(np.abs(data)) / (np.max(np.abs(sim)) + 1e-12)
        # polynomial baseline fitted to what the scaled model leaves over
        baseline = _fit_polynomial_baseline(data - scale * sim, options.baseline)
        fit = scale * sim + baseline
    else:  # 'none'
        scale = 1.0
        baseline = _fit_polynomial_baseline(data - sim, options.baseline)
        fit = sim + baseline
    
    # 3. Compute residuals
    residuals_raw = fit - data
    
    # 4. Apply target transformation
    residuals = _apply_target_transform(residuals_raw, options.target)
    
    # 5. Compute RMSD
    rmsd = np.sqrt(np.mean(residuals**2))
    
    info = {
        'sim': sim,
        'fit': fit,
        'scale': scale,
        'baseline': baseline,
        'residuals_raw': residuals_raw,
        'negative_scale': negative_scale,
    }
    
    return residuals, rmsd, info


class _FailedEval(float):
    """The 1e10 penalty RMSD returned for a model evaluation that raised.

    A float (optimizers use it like any objective value) that also carries the
    error text, so the progress tracker in the calling process can count the
    failure and warn, also when the evaluation ran in a worker process.
    """

    def __new__(cls, value: float, error: str = ''):
        obj = super().__new__(cls, value)
        obj.error = error
        return obj

    def __reduce__(self):
        return (_FailedEval, (float(self), self.error))


_FAILED_RMSD = 1e10


def _objective_function(
    pvec_active: np.ndarray,
    data: np.ndarray,
    model_wrapper: ModelWrapper,
    param_handler: ParameterHandler,
    options: FitOptions
) -> float:
    """Objective function for optimization (returns scalar RMSD)."""
    try:
        # Clip parameters to bounds
        lb, ub = param_handler.get_bounds()
        pvec_active_clipped = np.clip(pvec_active, lb, ub)
        
        _, rmsd, _ = compute_residuals(
            pvec_active_clipped, data, model_wrapper, param_handler, options
        )
        return rmsd
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
        # Recoverable errors (out-of-range parameters, convergence failures):
        # penalize so the optimizer moves away. The tracker counts the
        # failure (FitResult.n_failed_evals) and warns once per fit (B80).
        if options.verbosity >= 1:
            print(f"Warning: simulation error at current parameters: {e}")
        return _FailedEval(_FAILED_RMSD, f"{type(e).__name__}: {e}")


def _optimize_simplex(
    objective_fn: Callable,
    p0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    options: FitOptions
) -> OptimizeResult:
    """
    Run Nelder-Mead simplex optimization.

    Matches MATLAB EasySpin's ``esfit_simplex.m`` behavior:
      * Initial simplex edge length = ``delta * (ub - lb)`` with delta=0.1
        in original space, which corresponds to 0.2 in the transformed
        [-1, 1] space (the previous 0.05 was 4x too small and caused
        premature convergence to local minima).
      * For n <= 2 parameters, classical NM coefficients [1, 2, 0.5, 0.5]
        (scipy's ``adaptive=True`` reduces to these for n=2 anyway, so we
        leave it on for n>=3 where Gao-Han adaptive params help).
      * Termination: scipy's NM requires BOTH ``xatol`` and ``fatol`` to
        be satisfied (matches EasySpin's dual-criterion stop), so we map
        ``xatol = TolEdgeLength = 1e-4`` and ``fatol = options.tol_fun``.
    """
    lb, ub = bounds
    n_params = len(p0)

    def transform(x):
        return 2 * (x - lb) / (ub - lb + 1e-12) - 1

    def untransform(x_scaled):
        return lb + (ub - lb) * (x_scaled + 1) / 2

    p0_scaled = transform(p0)

    def obj_scaled(x_scaled):
        x = untransform(x_scaled)
        return objective_fn(x)

    # Initial simplex: vertex i (for i=1..n) perturbs parameter i-1 by
    # delta_t = 0.2 in [-1, 1] space (= EasySpin delta=0.1 * (ub-lb)
    # in original space, since transformed (ub-lb)=2). MATLAB:
    #   v(nParams+1:nParams+1:end) = ... + delta.*(ub-lb).'
    delta_t = 0.2
    initial_simplex = np.zeros((n_params + 1, n_params))
    initial_simplex[0] = p0_scaled
    for i in range(n_params):
        vertex = p0_scaled.copy()
        # Step toward the interior of [-1, 1] so we don't immediately
        # land on the bound when p0 is near ub
        step = delta_t if vertex[i] + delta_t <= 1.0 else -delta_t
        vertex[i] += step
        initial_simplex[i + 1] = vertex

    # iteration counter only — stopping always comes from _StopFit raised inside the
    # objective, so scipy's per-version callback-halting semantics do not matter
    tracker = _tracker_of(objective_fn)
    result = minimize(
        obj_scaled,
        p0_scaled,
        method='Nelder-Mead',
        callback=lambda *a, **k: tracker.iteration(tracker.iter, None),
        options={
            'maxiter': options.max_iter,
            'xatol': 1e-4,
            'fatol': options.tol_fun,
            'disp': options.verbosity > 0,
            'adaptive': True,
            'initial_simplex': initial_simplex,
        }
    )

    result.x = untransform(result.x)
    result.x = np.clip(result.x, lb, ub)

    return result


def _optimize_lbfgsb(
    objective_fn: Callable,
    p0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    options: FitOptions
) -> OptimizeResult:
    """Run L-BFGS-B optimization (supports bounds natively)."""
    from scipy.optimize import Bounds
    
    lb, ub = bounds
    tracker = _tracker_of(objective_fn)
    fun = objective_fn
    opts = {
        'maxiter': options.max_iter,
        'ftol': options.tol_fun,
        'gtol': 1e-5,
        'disp': options.verbosity > 0
    }
    # the n forward-difference probes of each gradient are independent
    mapper = _scipy_map(options, tracker, _scalar, 2 * len(p0))
    if mapper is not None:
        if _scipy_takes_workers('lbfgsb'):
            raw = objective_fn.raw if isinstance(objective_fn, _TrackedObjective) else objective_fn
            fun = _ParentRecorded(raw, tracker, _scalar)
            opts['workers'] = mapper
        else:
            warnings.warn("esfit: parallel L-BFGS-B gradients need SciPy >= 1.16; evaluating them sequentially")
    result = minimize(
        fun,
        p0,
        method='L-BFGS-B',
        bounds=Bounds(lb, ub),
        callback=lambda *a, **k: tracker.iteration(tracker.iter, None),
        options=opts
    )
    return result


def _optimize_powell(
    objective_fn: Callable,
    p0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    options: FitOptions
) -> OptimizeResult:
    """Run Powell optimization (derivative-free)."""
    from scipy.optimize import Bounds
    
    lb, ub = bounds
    tracker = _tracker_of(objective_fn)
    result = minimize(
        objective_fn,
        p0,
        method='Powell',
        bounds=Bounds(lb, ub),
        callback=lambda *a, **k: tracker.iteration(tracker.iter, None),
        options={
            'maxiter': options.max_iter,
            'ftol': options.tol_fun,
            'xtol': options.tol_x,
            'disp': options.verbosity > 0
        }
    )
    return result


def _emit_progress(
    options: FitOptions,
    method: str,
    it: int,
    max_iter: int,
    best_f: float,
    best_x: np.ndarray,
    tracker: Optional['_FitProgress'] = None,
) -> None:
    """Emit per-iteration progress when verbosity>=2 or a callback is set.

    Prints a flushed one-line update (safe to run inside Jupyter cells:
    ``flush=True`` makes the message visible while the cell is still
    executing) and invokes ``options.callback(info)`` if provided.
    """
    if options.verbosity >= 2:
        print(f"  [{method}] iter {it+1:4d}/{max_iter}  best RMSD = {best_f:.5e}",
              flush=True)
    if options.callback is not None:
        try:
            options.callback({
                "method": method,
                "iter": it + 1,
                "max_iter": max_iter,
                "best_f": best_f,
                "best_x": best_x.copy(),
            })
        except Exception as e:
            # A user-supplied callback must not crash the fit, but its failure
            # must not go unnoticed either (B80): warn once per fit.
            msg = (f"esfit: options.callback raised {type(e).__name__}: {e}; "
                   "the fit continues (further callback errors in this fit "
                   "are not reported)")
            if tracker is None:
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
            else:
                tracker.warn_once('callback', msg)


class _StopFit(Exception):
    """Raised from the progress tracker to stop a fit early (max_time, stop_when,
    a progress callback returning False).  An Exception subclass on purpose:
    scipy swallows StopIteration, and nothing between the raise site and esfit
    catches plain Exceptions."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _rss_mb() -> Optional[float]:
    """Resident set size in MB (psutil if available, else getrusage)."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        try:
            import resource, sys as _sys
            r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return r / 1e6 if _sys.platform == 'darwin' else r / 1e3
        except Exception:
            return None


def _population_size(options: 'FitOptions', n_params: int) -> Optional[int]:
    m = options.method
    if m in ('swarm', 'global'):
        return int(options.swarm_size)
    if m == 'genetic':
        return int(options.ga_population)
    if m == 'montecarlo':
        return int(options.mc_samples)
    if m == 'grid':
        return int(min(options.max_iter, max(2, options.grid_points) ** max(1, n_params)))
    return None


class _FitProgress:
    """Per-fit progress tracker: evaluation count, best-so-far, timing, rate-limited
    reporting (text / tqdm bar / callable) and the early-stop checks.  Lives in the
    calling process only (never pickled; worker results are merged in the parent)."""

    def __init__(self, options: 'FitOptions', method: str, n_params: int = 0, n_data: int = 0, enabled: bool = True):
        self.options = options
        self.method = method
        self.phase = method
        self.n_params, self.n_data = int(n_params), int(n_data)
        self.n_evaluations = 0
        self.best_f = float('inf')
        self.best_x: Optional[np.ndarray] = None
        self.last_f = float('nan')
        self.iter = 0
        self.iter_total: Optional[int] = None
        self.eval_total: Optional[int] = None
        self.t0: Optional[float] = None
        self.t_last_emit = float('-inf')
        self.n_last_emit = 0
        prog = options.progress if enabled else None
        self.mode = 'callable' if callable(prog) else (prog if prog in ('text', 'bar') else None)
        self.enabled = self.mode is not None
        self.active = self.enabled or options.max_time is not None or options.stop_when is not None
        self._bar = None
        self.stop_reason: Optional[str] = None
        self.interrupted = False
        self.rss0: Optional[float] = None
        self.elapsed_s: Optional[float] = None
        self.n_failed_evals = 0
        self._warned: set = set()

    def warn_once(self, key: str, message: str) -> None:
        """Issue ``message`` as a RuntimeWarning the first time ``key`` occurs in this fit."""
        if key not in self._warned:
            self._warned.add(key)
            warnings.warn(message, RuntimeWarning, stacklevel=3)

    def _count_failures(self, F) -> None:
        for f in F:
            if isinstance(f, _FailedEval):
                self.n_failed_evals += 1
                self.warn_once(
                    'model', "esfit: the model raised at some parameter values "
                    f"(first error: {f.error}); those evaluations are penalized "
                    f"with rmsd {_FAILED_RMSD:g}. FitResult.n_failed_evals counts them.")

    # -- timing --------------------------------------------------------------
    def elapsed(self) -> float:
        return 0.0 if self.t0 is None else time.perf_counter() - self.t0

    def start(self) -> None:
        self.t0 = time.perf_counter()
        self.t_last_emit = self.t0
        if not self.enabled:
            return
        self.rss0 = _rss_mb()
        if self.mode == 'bar':
            try:
                from tqdm.auto import tqdm
                self._bar = tqdm(total=self.iter_total or self.eval_total, desc=f'esfit[{self.phase}]', leave=True)
            except Exception:
                self._bar = None
                self.mode = 'text'
                print('esfit: progress bar unavailable (no tqdm); using text', flush=True)
        if self.mode in ('text', 'bar'):
            print(self._resource_line(), flush=True)

    def _resource_line(self) -> str:
        import os
        import torch as _torch
        o = self.options
        pop = _population_size(o, self.n_params)
        if pop is None:
            workers = 'serial'
        else:
            nw = _resolve_workers(o.n_workers, pop)
            kind = 'threads' if o.workers == 'threads' else 'procs (spawn)'
            workers = f"{nw} {kind}" if nw > 1 else 'serial'
            workers += f" (n_workers={o.n_workers!r})"
        dev = []
        try:
            dev.append('cuda ' + ('yes' if _torch.cuda.is_available() else 'no'))
            if hasattr(_torch.backends, 'mps'):
                dev.append('mps ' + ('yes' if _torch.backends.mps.is_available() else 'no'))
        except Exception:
            pass
        rss = f" | rss {self.rss0:.0f} MB" if self.rss0 is not None else ''
        return (f"esfit[{self.method}] {self.n_params} params, {self.n_data} points | torch threads "
                f"{_torch.get_num_threads()} | workers {workers} | cpus {os.cpu_count()} | {', '.join(dev)}{rss}")

    # -- recording -----------------------------------------------------------
    def record(self, x, f: float, emit: bool = True) -> None:
        self.n_evaluations += 1
        self._count_failures((f,))
        f = float(f)
        self.last_f = f
        if f < self.best_f:
            self.best_f = f
            self.best_x = np.array(x, dtype=float, copy=True)
        if emit:
            self.maybe_emit()
            self.check()

    def record_batch(self, X, F) -> None:
        if isinstance(F, (list, tuple)):
            self._count_failures(F)
        F = np.asarray(F, dtype=float).ravel()
        if F.size == 0:
            return
        self.n_evaluations += int(F.size)
        i = int(np.argmin(F))
        self.last_f = float(F[i])
        if F[i] < self.best_f:
            self.best_f = float(F[i])
            self.best_x = np.array(np.asarray(X)[i], dtype=float, copy=True)
        self.maybe_emit()
        self.check()

    def iteration(self, it: int, total: Optional[int] = None) -> None:
        self.iter = int(it) + 1
        self.iter_total = None if total is None else int(total)
        if self._bar is not None and self.iter_total is not None:
            self._bar.total = self.iter_total
            self._bar.n = self.iter
            self._bar.refresh()
        self.maybe_emit()
        self.check()

    def set_phase(self, label: str) -> None:
        self.phase = label
        self.iter = 0
        self.iter_total = None
        if self.mode == 'text':
            print(f"  [{label}] phase start at {self.n_evaluations} evaluations, {self.elapsed():.1f} s", flush=True)
        elif self._bar is not None:
            self._bar.set_description(f'esfit[{label}]')

    # -- reporting -----------------------------------------------------------
    def info(self) -> dict:
        el = self.elapsed()
        rate = self.n_evaluations / el if el > 0 else None
        eta = None
        if self.iter_total and self.iter > 0:
            eta = el / self.iter * max(0, self.iter_total - self.iter)
        elif self.eval_total and self.n_evaluations > 0:
            eta = el / self.n_evaluations * max(0, self.eval_total - self.n_evaluations)
        return {'method': self.method, 'phase': self.phase, 'iter': self.iter, 'iter_total': self.iter_total,
                'n_evaluations': self.n_evaluations, 'eval_total': self.eval_total, 'rmsd': self.last_f,
                'best_f': self.best_f, 'best_x': None if self.best_x is None else self.best_x.copy(),
                'elapsed_s': el, 'evals_per_s': rate, 'eta_s': eta}

    def _text_line(self, info: dict) -> str:
        parts = [f"[{info['phase']}]"]
        if info['iter_total']:
            parts.append(f"iter {info['iter']}/{info['iter_total']}")
        elif info['iter']:
            parts.append(f"iter {info['iter']}")
        parts.append(f"eval {info['n_evaluations']}" + (f"/{info['eval_total']}" if info['eval_total'] else ''))
        parts.append(f"rmsd {info['rmsd']:.3e}  best {info['best_f']:.3e}")
        parts.append(f"{info['elapsed_s']:.1f} s")
        if info['evals_per_s']:
            parts.append(f"{info['evals_per_s']:.1f} ev/s")
        if info['eta_s'] is not None:
            parts.append(f"ETA {info['eta_s']:.0f} s")
        return '  ' + '  '.join(parts)

    def maybe_emit(self, force: bool = False) -> None:
        if not self.enabled:
            return
        o = self.options
        now = time.perf_counter()
        due = force
        if o.progress_every and self.n_evaluations - self.n_last_emit >= int(o.progress_every):
            due = True
        if now - self.t_last_emit >= float(o.progress_interval):
            due = True
        if not due:
            return
        self.t_last_emit = now
        self.n_last_emit = self.n_evaluations
        info = self.info()
        if self.mode == 'text':
            print(self._text_line(info), flush=True)
        elif self.mode == 'bar':
            if self._bar is not None:
                if self.iter_total is None:
                    self._bar.total = self.eval_total
                    self._bar.n = self.n_evaluations
                self._bar.set_postfix_str(f"rmsd {info['rmsd']:.3e} best {info['best_f']:.3e}"
                                          + (f" {info['evals_per_s']:.1f} ev/s" if info['evals_per_s'] else ''))
                self._bar.refresh()
        elif self.mode == 'callable':
            try:
                ret = o.progress(info)
            except Exception as e:
                # keep fitting, but do not hide the failure (B80)
                self.warn_once('progress', f"esfit: options.progress raised {type(e).__name__}: {e}; "
                               "the fit continues (further progress errors in this fit are not reported)")
                ret = None
            if ret is False and self.stop_reason is None:
                self.stop_reason = 'progress callback returned False'

    # -- early stop ----------------------------------------------------------
    def check(self) -> None:
        o = self.options
        if self.stop_reason is None:
            if o.max_time is not None and self.t0 is not None and self.elapsed() >= float(o.max_time):
                self.stop_reason = f'max_time {o.max_time} s reached'
            elif o.stop_when is not None:
                # an exception in stop_when propagates (B80): a stop condition
                # that cannot be evaluated must not be silently ignored
                if o.stop_when(self.info()):
                    self.stop_reason = 'stop_when condition met'
        if self.stop_reason is not None:
            self.interrupted = True
            raise _StopFit(self.stop_reason)

    def synthesize_result(self, reason: str) -> OptimizeResult:
        self.interrupted = True
        self.stop_reason = self.stop_reason or reason
        x = None if self.best_x is None else self.best_x.copy()
        return OptimizeResult(x=x, fun=self.best_f, success=False,
                              message=(f"interrupted after {self.n_evaluations} evaluations, {self.elapsed():.1f} s "
                                       f"({self.stop_reason}); returning best so far"),
                              nit=self.iter, nfev=self.n_evaluations)

    def finish(self) -> None:
        self.elapsed_s = self.elapsed()
        if self._bar is not None:
            try:
                self._bar.close()
            except Exception:
                pass
        if not self.enabled or self.mode == 'callable':
            return
        rate = self.n_evaluations / self.elapsed_s if self.elapsed_s > 0 else 0.0
        rss = _rss_mb()
        rss_s = '' if rss is None or self.rss0 is None else f", rss {rss:.0f} MB ({rss - self.rss0:+.0f} MB)"
        if self.interrupted:
            print(f"esfit[{self.method}] interrupted ({self.stop_reason}) after {self.n_evaluations} evals, "
                  f"{self.elapsed_s:.1f} s: returning best so far, rmsd {self.best_f:.3e}{rss_s}", flush=True)
        else:
            print(f"esfit[{self.method}] done: {self.n_evaluations} evals, {self.elapsed_s:.1f} s, {rate:.1f} ev/s, "
                  f"best rmsd {self.best_f:.3e}{rss_s}", flush=True)


class _TrackedObjective:
    """Objective wrapper used on the serial path: records every evaluation in the
    tracker (count, best-so-far, progress, early stop).  Never pickled — the
    population evaluator unwraps ``.raw`` before shipping work to processes."""

    def __init__(self, raw: Callable, tracker: _FitProgress):
        self.raw = raw
        self.tracker = tracker

    def __call__(self, x):
        f = self.raw(x)
        self.tracker.record(x, f)
        return f

    def __reduce__(self):
        raise TypeError('_TrackedObjective is not picklable; pass .raw to worker processes')


def _tracker_of(fn) -> _FitProgress:
    """The tracker behind an objective, or a disabled one for bare callables."""
    t = getattr(fn, 'tracker', None)
    return t if t is not None else _FitProgress(FitOptions(), 'none', enabled=False)


_POOL = None                # cached spawn pool: (n_proc, threads_per_worker, pool, devices)


import contextvars as _contextvars
# FitOptions.devices of the fit in progress (set by esfit, read when a pool is made)
_FIT_DEVICES: _contextvars.ContextVar = _contextvars.ContextVar('esfit_devices', default=None)


def _cuda_index(dev) -> str:
    """'cuda:1' / 1 / torch.device('cuda', 1) -> '1' (a CUDA_VISIBLE_DEVICES entry)."""
    if isinstance(dev, int):
        return str(dev)
    d = torch.device(dev)
    if d.type != 'cuda':
        raise ValueError(f"FitOptions.devices takes CUDA devices, got {dev!r}")
    return str(d.index or 0)


def _pool_init(n_threads: int, device_queue=None):
    import os as _os
    if device_queue is not None:
        # before any CUDA call in this worker, so the runtime only sees this GPU
        _os.environ['CUDA_VISIBLE_DEVICES'] = device_queue.get()
    import torch as _torch
    _torch.set_num_threads(n_threads)


def _get_pool(n_proc: int, per: int):
    """Process pool with the 'spawn' start method (fork is not safe once torch's
    thread pool exists), created once and reused across generations/calls."""
    global _POOL
    import multiprocessing as mp
    devices = _FIT_DEVICES.get()
    devices = tuple(_cuda_index(d) for d in devices) if devices else None
    if _POOL is not None and _POOL[0] == n_proc and _POOL[1] == per and _POOL[3] == devices:
        return _POOL[2]
    if _POOL is not None:
        _POOL[2].terminate()
    # the children inherit the environment at spawn: cap the BLAS/OpenMP thread
    # pools of numpy/scipy/torch per worker (otherwise every worker starts one
    # thread per core and the pool thrashes)
    from torchspin.parallel import native_thread_env
    ctx = mp.get_context('spawn')
    queue = None
    if devices:
        queue = ctx.Queue()
        for k in range(n_proc):
            queue.put(devices[k % len(devices)])
    with native_thread_env(per):
        pool = ctx.Pool(n_proc, initializer=_pool_init, initargs=(per, queue))
    _POOL = (n_proc, per, pool, devices)
    return pool


class _CloudPickled:
    """Carries a non-picklable callable (notebook cell function, closure) to spawned
    workers by value via cloudpickle."""

    def __init__(self, fn):
        import cloudpickle
        self._blob = cloudpickle.dumps(fn)
        self._fn = fn

    def __call__(self, x):
        if self._fn is None:
            import cloudpickle
            self._fn = cloudpickle.loads(self._blob)
        return self._fn(x)

    def __getstate__(self):
        return {'_blob': self._blob, '_fn': None}


def _threads_per_worker(n_workers: int) -> int:
    """Inner threads per outer worker so that workers × threads fits the core budget."""
    import torch as _torch
    from torchspin.parallel import cpu_budget
    total = min(_torch.get_num_threads(), cpu_budget())
    return max(1, total // max(1, n_workers))


def _resolve_workers(n_workers, n: int) -> int:
    """'auto' → the cores this process may use (affinity / SLURM aware), capped by the
    available memory (about 0.5 GB per worker, half the free memory), by 32 and by
    the population size; ints pass through."""
    if isinstance(n_workers, str):
        from torchspin._resources import auto_workers
        return auto_workers(n)
    return int(n_workers or 1)


def _ship(raw: Callable):
    """``raw`` in a form a spawned worker can run, or None if it cannot be sent.

    By value through cloudpickle whenever possible: pickling by reference succeeds
    for functions defined in a notebook's __main__, but the spawned worker cannot
    resolve them, dies before registering the task and the map hangs forever.
    By reference only for functions from importable modules.
    """
    try:
        return _CloudPickled(raw)
    except Exception:
        import pickle
        mod = getattr(raw, '__module__', None)
        if mod not in (None, '__main__', '__mp_main__'):
            try:
                pickle.dumps(raw)
                return raw
            except Exception:
                return None
    return None


def _map_points(raw: Callable, X, n_workers, workers: str = 'auto',
                on_chunk: Optional[Callable] = None, step: Optional[int] = None,
                size_hint: Optional[int] = None) -> list:
    """``[raw(x) for x in X]``, evaluated concurrently, results in input order.

    ``workers='auto'|'processes'`` uses the cached spawn pool (cores split between
    the workers, see :func:`_threads_per_worker`); ``'threads'``, or an objective
    that cannot be sent to a process, uses a thread pool. ``on_chunk(Xc, Fc)`` runs
    in this process after every chunk of ``step`` points (default: all at once).
    ``size_hint`` is the largest batch the caller will send: with ``'auto'`` the
    process pool is sized for it, so batches of different sizes within one fit
    reuse one pool instead of respawning it.
    """
    global _POOL
    X = list(X)
    n = len(X)
    if n <= 1:
        n_workers = 1
    else:
        n_workers = _resolve_workers(n_workers, max(n, int(size_hint or 0)))
    if n_workers <= 1 or n <= 1:
        out = [raw(x) for x in X]
        if on_chunk is not None:
            on_chunk(X, out)
        return out
    if workers in ('auto', 'processes'):
        fn = _ship(raw)
        if fn is not None:
            n_proc = n_workers if size_hint else min(n_workers, n)
            pool = _get_pool(n_proc, _threads_per_worker(n_proc))
            step = n if not step else max(1, int(step))
            out = []
            try:
                for s in range(0, n, step):
                    fc = pool.map(fn, X[s:s + step], chunksize=1)
                    out.extend(fc)
                    if on_chunk is not None:
                        on_chunk(X[s:s + step], fc)
            except (KeyboardInterrupt, _StopFit):
                # do not leave the cached pool grinding the interrupted batch
                try:
                    pool.terminate()
                finally:
                    _POOL = None
                raise
            return out
        if workers == 'processes':
            warnings.warn("esfit: objective is not picklable and cloudpickle is not installed; using a thread pool")
    from concurrent.futures import ThreadPoolExecutor
    # No per-worker thread_budget here. A smaller inner budget makes _linalg skip its
    # serialized single-threaded LAPACK pool, but the native BLAS/OpenMP thread count
    # is process-wide and cannot be capped per Python thread: each worker then calls
    # LAPACK at the full process width, and n_workers x native threads oversubscribe
    # the cores (measured 2.1x slower at 8 threads for the matrix method). The process
    # backend is the one that partitions cores.
    with ThreadPoolExecutor(max_workers=min(n_workers, n)) as ex:
        out = list(ex.map(raw, X))
    if on_chunk is not None:
        on_chunk(X, out)
    return out


def _eval_population(
    objective_fn: Callable,
    X: np.ndarray,
    n_workers=1,
    workers: str = 'auto',
) -> np.ndarray:
    """Evaluate ``objective_fn`` on each row of ``X``.

    ``n_workers > 1`` (or ``'auto'``, see :class:`FitOptions`) evaluates the rows
    concurrently.  ``workers='processes'`` uses a spawned process pool with the
    cores split between the workers — the effective option for torchspin forward
    models, whose Python-level work holds the GIL.  The objective is pickled by
    value through ``cloudpickle`` (so functions defined in a notebook cell and
    closures work); without cloudpickle, only objectives from importable modules
    can be sent by reference and the rest fall back to a thread pool.
    ``workers='threads'`` forces the thread pool.
    """
    n = X.shape[0]
    if _resolve_workers(n_workers, n) <= 1 or n <= 1:
        return np.array([objective_fn(x) for x in X])
    # the progress tracker stays in this process: workers get the raw objective and
    # the parent merges each chunk of results (count, best-so-far, stop checks)
    raw = objective_fn.raw if isinstance(objective_fn, _TrackedObjective) else objective_fn
    tracker = _tracker_of(objective_fn)
    step = None
    if tracker.active and workers in ('auto', 'processes'):
        n_proc = min(_resolve_workers(n_workers, n), n)
        step = max(2 * n_proc, int(tracker.options.progress_every or 0))
    return np.array(_map_points(raw, list(X), n_workers, workers, on_chunk=tracker.record_batch, step=step))


class _ParentRecorded:
    """A residual or objective function SciPy may evaluate in worker processes.

    Calls made directly by the optimizer, in the thread that built the wrapper,
    are recorded in the tracker. Finite-difference probes that SciPy hands to the
    ``workers`` map (see :func:`_scipy_map`) are recorded there instead, so every
    evaluation is counted once. The tracker never leaves this process.
    """

    def __init__(self, raw: Callable, tracker: '_FitProgress', to_f: Callable):
        import threading
        self.raw, self.tracker, self.to_f = raw, tracker, to_f
        self._owner = threading.get_ident()

    def __call__(self, x):
        out = self.raw(x)
        if self.tracker is not None:
            import threading
            if threading.get_ident() == self._owner:
                self.tracker.record(x, self.to_f(out))
        return out

    def __getstate__(self):
        return {'raw': self.raw, 'tracker': None, 'to_f': self.to_f, '_owner': None}


def _scipy_map(options: 'FitOptions', tracker: '_FitProgress', to_f: Callable, size_hint: int):
    """A ``workers`` map for SciPy's finite differences, or None to stay sequential."""
    if _resolve_workers(options.n_workers, 2) <= 1:
        return None

    def mapper(fn, iterable):
        def record(Xc, Fc):
            tracker.record_batch(Xc, [to_f(f) for f in Fc])
        return _map_points(fn, list(iterable), options.n_workers, options.workers, on_chunk=record,
                           size_hint=size_hint)
    return mapper


def _scipy_takes_workers(solver: str) -> bool:
    import inspect
    try:
        if solver == 'trf':
            return 'workers' in inspect.signature(least_squares).parameters
        from scipy.optimize._lbfgsb_py import _minimize_lbfgsb
        return 'workers' in inspect.signature(_minimize_lbfgsb).parameters
    except Exception:
        return False


def _scalar(f) -> float:
    """A scalar objective value; SciPy's wrappers may hand back a 1-element array."""
    if isinstance(f, _FailedEval):
        return f        # keep the failure marker for the tracker
    return float(np.asarray(f, dtype=float).ravel()[0])


def _rms(r) -> float:
    r = np.asarray(r, dtype=float).ravel()
    return float(np.sqrt(np.mean(r * r)))


def _finite_residuals(p, resid_fn):
    r = np.asarray(resid_fn(p), dtype=float).ravel()
    if not np.all(np.isfinite(r)):
        raise ValueError('residuals contain NaN or Inf')
    return r


def _bounded_residuals(pvec, lb, ub, data, model_wrapper, param_handler, options):
    """Residual vector at ``pvec`` clipped to the bounds (picklable for worker pools)."""
    return compute_residuals(np.clip(pvec, lb, ub), data, model_wrapper, param_handler, options)[0]


def _plain_residuals(pvec, data, model_wrapper, param_handler, options):
    """Residual vector at ``pvec`` (picklable for worker pools)."""
    return compute_residuals(pvec, data, model_wrapper, param_handler, options)[0]


def _optimize_grid(
    objective_fn: Callable,
    p0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    options: FitOptions,
) -> OptimizeResult:
    """Systematic grid search over parameter bounds.

    Evaluates the objective on a regular ``grid_points``-point grid per
    dimension, then returns the best point found.  For N parameters the
    grid has ``grid_points^N`` evaluations.  Intended for small N (≤4) or
    low ``grid_points``.
    """
    lb, ub = bounds
    n = len(p0)
    npts = max(2, options.grid_points)

    # Build per-axis grids
    grids = [np.linspace(lb[i], ub[i], npts) for i in range(n)]

    best_x = p0.copy()
    best_f = objective_fn(p0)

    import itertools
    combos = list(itertools.product(*grids))
    if len(combos) > options.max_iter:
        combos = combos[:options.max_iter]
    grid_pts = np.array(combos)
    tracker = _tracker_of(objective_fn)
    tracker.eval_total = len(grid_pts) + 1
    grid_f = _eval_population(objective_fn, grid_pts, options.n_workers, options.workers)
    tracker.eval_total = None
    n_eval = len(grid_pts)

    idx = int(np.argmin(grid_f))
    if grid_f[idx] < best_f:
        best_f = float(grid_f[idx])
        best_x = grid_pts[idx].copy()

    return OptimizeResult(
        x=best_x,
        fun=best_f,
        success=True,
        message=f"Grid search: {n_eval} evaluations, best RMSD={best_f:.4e}",
        nit=n_eval,
        nfev=n_eval,
    )


def _optimize_montecarlo(
    objective_fn: Callable,
    p0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    options: FitOptions,
) -> OptimizeResult:
    """Monte Carlo / simulated annealing optimizer.

    Draws ``mc_samples`` random parameter sets uniformly within bounds,
    then applies a simple simulated annealing refinement around the best
    point found.  Temperature decreases geometrically from ``mc_temp_start``
    to ``mc_temp_end`` over ``max_iter`` SA steps.
    """
    lb, ub = bounds
    rng = np.random.default_rng(options.seed)

    # Phase 1: random sampling (parallelisable)
    n_random = min(options.mc_samples, options.max_iter)
    best_x = p0.copy()
    best_f = objective_fn(p0)

    samples = rng.uniform(lb, ub, size=(n_random, len(p0)))
    tracker = _tracker_of(objective_fn)
    tracker.eval_total = n_random + 1
    sample_f = _eval_population(objective_fn, samples, options.n_workers, options.workers)
    tracker.eval_total = None
    idx = int(np.argmin(sample_f))
    if sample_f[idx] < best_f:
        best_f = float(sample_f[idx])
        best_x = samples[idx].copy()

    n_eval = n_random + 1

    # Phase 2: simulated annealing refinement around best point
    n_sa = max(1, options.max_iter - n_random)
    T_start = options.mc_temp_start
    T_end = options.mc_temp_end
    T_ratio = (T_end / T_start) ** (1.0 / n_sa) if n_sa > 1 else 1.0

    current_x = best_x.copy()
    current_f = best_f
    T = T_start
    span = (ub - lb)

    for step in range(n_sa):
        # Propose random step proportional to temperature
        delta = rng.normal(0, T * span)
        proposal = np.clip(current_x + delta, lb, ub)
        f = objective_fn(proposal)
        n_eval += 1

        # Accept if better, or probabilistically if worse
        dE = f - current_f
        if dE < 0 or (T > 0 and rng.random() < np.exp(-dE / T)):
            current_x = proposal
            current_f = f
            if f < best_f:
                best_f = f
                best_x = proposal.copy()

        T *= T_ratio

        # Emit per-step progress (stride to keep output readable)
        stride = max(1, n_sa // 20)
        if step % stride == 0 or step == n_sa - 1:
            _emit_progress(options, 'montecarlo', step, n_sa, best_f, best_x, tracker)
        tracker.iteration(step, n_sa)

    return OptimizeResult(
        x=best_x,
        fun=best_f,
        success=True,
        message=f"Monte Carlo: {n_eval} evaluations, best RMSD={best_f:.4e}",
        nit=n_eval,
        nfev=n_eval,
    )


def _optimize_genetic(
    objective_fn: Callable,
    p0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    options: FitOptions,
) -> OptimizeResult:
    """Genetic algorithm with tournament selection and Gaussian mutation.

    Maintains a population of ``ga_population`` candidates.  Each generation:

    1. Evaluate all candidates.
    2. Preserve the top ``ga_elite`` individuals unchanged (elitism).
    3. Fill remaining slots via tournament selection (size 2) + crossover.
    4. Apply Gaussian mutation with probability ``ga_mutation``.

    Continues for ``max_iter`` generations or until ``tol_fun`` is met.
    """
    lb, ub = bounds
    n = len(p0)
    pop_size = max(options.ga_elite + 2, options.ga_population)
    elite = max(1, options.ga_elite)
    mut_rate = options.ga_mutation
    cross_rate = options.ga_crossover
    rng = np.random.default_rng(options.seed)

    # Initialize population: elite[0] = p0, rest = random
    pop = np.empty((pop_size, n))
    pop[0] = np.clip(p0, lb, ub)
    pop[1:] = rng.uniform(lb, ub, size=(pop_size - 1, n))

    fitness = _eval_population(objective_fn, pop, options.n_workers, options.workers)
    n_eval = pop_size
    best_f = float('inf')
    best_x = p0.copy()

    span = ub - lb

    for gen in range(options.max_iter):
        # Sort by fitness
        order = np.argsort(fitness)
        pop = pop[order]
        fitness = fitness[order]

        if fitness[0] < best_f:
            best_f = fitness[0]
            best_x = pop[0].copy()

        _emit_progress(options, 'genetic', gen, options.max_iter,
                       best_f, best_x, _tracker_of(objective_fn))
        _tracker_of(objective_fn).iteration(gen, options.max_iter)

        if best_f < options.tol_fun:
            break

        # Build next generation
        new_pop = np.empty_like(pop)
        new_pop[:elite] = pop[:elite]  # elitism

        for i in range(elite, pop_size):
            # Tournament selection (size 2) for two parents
            idx_a = rng.integers(0, pop_size, 2)
            idx_b = rng.integers(0, pop_size, 2)
            pa = pop[idx_a[np.argmin(fitness[idx_a])]]
            pb = pop[idx_b[np.argmin(fitness[idx_b])]]

            # Uniform crossover
            if rng.random() < cross_rate:
                mask = rng.random(n) < 0.5
                child = np.where(mask, pa, pb)
            else:
                child = pa.copy()

            # Gaussian mutation
            for j in range(n):
                if rng.random() < mut_rate:
                    child[j] += rng.normal(0, 0.05 * span[j])
            child = np.clip(child, lb, ub)
            new_pop[i] = child

        # Evaluate new individuals (population eval, possibly parallel)
        new_fit = _eval_population(objective_fn, new_pop[elite:], options.n_workers, options.workers)
        fitness[elite:] = new_fit
        n_eval += pop_size - elite

        pop = new_pop

        if n_eval >= options.max_iter * pop_size:
            break

    return OptimizeResult(
        x=best_x,
        fun=best_f,
        success=True,
        message=f"Genetic: {gen+1} generations, {n_eval} evaluations, best RMSD={best_f:.4e}",
        nit=gen + 1,
        nfev=n_eval,
    )


def _optimize_swarm(
    objective_fn: Callable,
    p0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    options: FitOptions,
) -> OptimizeResult:
    """Particle Swarm Optimization (PSO).

    Port of EasySpin's ``esfit_swarm.m``.

    Maintains a swarm of ``swarm_size`` particles.  Each iteration updates
    velocities via cognitive and social terms, then moves particles.

    Parameters
    ----------
    w  : ``swarm_inertia`` (default 0.7)  — inertia weight
    c1 : ``swarm_cognitive`` (default 1.4) — personal-best attraction
    c2 : ``swarm_social``   (default 1.4) — global-best attraction
    """
    lb, ub = bounds
    n = len(p0)
    swarm_n = max(4, options.swarm_size)
    w = options.swarm_inertia
    c1 = options.swarm_cognitive
    c2 = options.swarm_social
    rng = np.random.default_rng(options.seed)
    span = ub - lb

    # Initialize positions and velocities
    pos = rng.uniform(lb, ub, size=(swarm_n, n))
    pos[0] = np.clip(p0, lb, ub)  # seed with initial guess
    vel = rng.uniform(-span, span, size=(swarm_n, n)) * 0.1

    fitness = _eval_population(objective_fn, pos, options.n_workers, options.workers)
    n_eval = swarm_n

    pbest_pos = pos.copy()
    pbest_fit = fitness.copy()

    gbest_idx = int(np.argmin(pbest_fit))
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fit = float(pbest_fit[gbest_idx])

    for it in range(options.max_iter):
        r1 = rng.uniform(0, 1, (swarm_n, n))
        r2 = rng.uniform(0, 1, (swarm_n, n))

        vel = (w * vel +
               c1 * r1 * (pbest_pos - pos) +
               c2 * r2 * (gbest_pos - pos))

        # Clamp velocity to span to prevent explosion
        vel = np.clip(vel, -span, span)
        pos = np.clip(pos + vel, lb, ub)

        fitness = _eval_population(objective_fn, pos, options.n_workers, options.workers)
        n_eval += swarm_n

        # Update personal bests
        improved = fitness < pbest_fit
        pbest_pos[improved] = pos[improved]
        pbest_fit[improved] = fitness[improved]

        # Update global best
        cur_best_idx = int(np.argmin(pbest_fit))
        if pbest_fit[cur_best_idx] < gbest_fit:
            gbest_fit = float(pbest_fit[cur_best_idx])
            gbest_pos = pbest_pos[cur_best_idx].copy()

        _emit_progress(options, 'swarm', it, options.max_iter,
                       gbest_fit, gbest_pos, _tracker_of(objective_fn))
        _tracker_of(objective_fn).iteration(it, options.max_iter)

        if gbest_fit < options.tol_fun:
            break

    return OptimizeResult(
        x=gbest_pos,
        fun=gbest_fit,
        success=True,
        message=f"Swarm: {it+1} iterations, {n_eval} evaluations, best RMSD={gbest_fit:.4e}",
        nit=it + 1,
        nfev=n_eval,
    )


def _optimize_global(
    objective_fn: Callable,
    p0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    options: FitOptions,
) -> OptimizeResult:
    """Global + local pipeline.

    Phase 1: particle swarm to escape local minima and locate the basin
             of the global optimum (population eval is parallelisable
             via ``options.n_workers``).
    Phase 2: Nelder-Mead simplex polish from the swarm's best point.

    This is the recommended default for derivative-EPR fitting where the
    raw loss landscape has many narrow local minima of width ~ linewidth.
    The global phase finds the right basin; the local phase refines.
    """
    from dataclasses import replace

    # Phase 1: swarm — budget half the iterations and reuse swarm options.
    swarm_iter = max(20, options.max_iter // 2)
    swarm_opts = replace(options, max_iter=swarm_iter)
    if options.verbosity >= 1:
        print(f"  [global] phase 1 — swarm ({swarm_iter} iterations, "
              f"{options.swarm_size} particles)", flush=True)
    tracker = _tracker_of(objective_fn)
    tracker.set_phase('global/swarm')
    swarm_res = _optimize_swarm(objective_fn, p0, bounds, swarm_opts)

    # Phase 2: simplex polish from the swarm's best point.
    polish_iter = max(200, options.max_iter - swarm_res.nfev)
    polish_opts = replace(options, max_iter=polish_iter)
    if options.verbosity >= 1:
        print(f"  [global] phase 2 — simplex polish (best after swarm: "
              f"{swarm_res.fun:.5e})", flush=True)
    tracker.set_phase('global/simplex')
    polish_res = _optimize_simplex(objective_fn, swarm_res.x, bounds, polish_opts)

    final_x = polish_res.x if polish_res.fun < swarm_res.fun else swarm_res.x
    final_f = min(polish_res.fun, swarm_res.fun)
    n_eval = swarm_res.nfev + polish_res.nfev

    return OptimizeResult(
        x=final_x,
        fun=final_f,
        success=True,
        message=(f"Global: swarm {swarm_res.nfev} evals -> {swarm_res.fun:.4e}, "
                 f"polish {polish_res.nfev} evals -> {polish_res.fun:.4e}"),
        nit=swarm_res.nit + polish_res.nit,
        nfev=n_eval,
    )


def _optimize_levmar(
    objective_fn: Callable,
    p0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    options: FitOptions,
    resid_fn: Optional[Callable] = None,
) -> OptimizeResult:
    """Levenberg-Marquardt least squares — port of EasySpin ``esfit_levmar.m``.

    Parameters are rescaled to [-1, 1] (``ScaleParams``), the residual Jacobian
    is estimated by forward differences with step ``lm_delta`` (EasySpin
    ``FitOpt.delta``, 1e-3) and then updated by Broyden rank-1 updates, with one
    coordinate re-estimated by finite differences per iteration when the step
    in that coordinate is small (``gamma = 0.8``).  Damping follows Nielsen's
    update; the loop stops when the gradient (``lm_gradient_tol``, EasySpin
    ``TolFun``) or the step (``tol_x``, EasySpin ``TolStep``) is small.
    ``resid_fn(p)`` must return the residual vector; ``objective_fn`` is only
    used to report the RMSD.
    """
    if resid_fn is None:
        raise ValueError("_optimize_levmar needs resid_fn (residual vector).")
    lb, ub = (np.asarray(b, dtype=float) for b in bounds)
    n = len(p0)
    span = np.where(ub > lb, ub - lb, 1.0)
    transform = lambda p: 2.0 * (p - lb) / span - 1.0
    untransform = lambda x: lb + span * (x / 2.0 + 0.5)
    xlb, xub = transform(lb), transform(ub)
    delta = float(options.lm_delta)
    tol_step = float(options.tol_x)
    tol_grad = float(options.lm_gradient_tol)

    n_eval = [0]
    tracker = _tracker_of(objective_fn)

    def finish(x, f):
        f = np.asarray(f, dtype=float).ravel()
        n_eval[0] += 1
        if not np.all(np.isfinite(f)):
            raise ValueError('residuals contain NaN or Inf')
        F = float(f @ f)
        # F is the sum of squares of the target-transformed residuals: sqrt(F/n) is
        # exactly the RMSD compute_residuals reports
        tracker.record(untransform(x), np.sqrt(F / max(1, f.size)))
        return F, f

    def funeval(x):
        return finish(x, resid_fn(untransform(x)))

    x = np.clip(transform(np.asarray(p0, dtype=float)), xlb, xub)
    # the start point and its n forward-difference probes are independent: with
    # n_workers > 1 they are evaluated concurrently, then recorded in order
    probes = []
    for i in range(n):
        x1 = x.copy(); x1[i] += delta
        probes.append(x1)
    if _resolve_workers(options.n_workers, n + 1) > 1:
        # pool sized for the largest batch of the fit (the 2n-probe uncertainty Jacobian)
        R = _map_points(resid_fn, [untransform(xi) for xi in [x] + probes], options.n_workers, options.workers,
                        size_hint=2 * n)
        F, f = finish(x, R[0])
        fp = [finish(xi, r)[1] for xi, r in zip(probes, R[1:])]
    else:
        F, f = funeval(x)
        fp = [funeval(xi)[1] for xi in probes]
    J = np.zeros((f.size, n))
    for i in range(n):
        J[:, i] = (fp[i] - f) / delta
    _warn_dead_jacobian_columns(J, 'levmar')
    g = J.T @ f
    norm_g = np.linalg.norm(g, ord=np.inf)
    A = J.T @ J
    mu = float(options.lm_lambda) * max(float(np.max(np.diag(A))), 1e-300)
    nu = 2.0
    j = 0
    it = 0
    message = 'max_iter reached'
    for it in range(1, int(options.max_iter) + 1):
        tracker.iteration(it - 1, None)
        if norm_g <= tol_grad:
            message = 'gradient below threshold'
            break
        # LM step: (A + mu I) h = -g, increasing mu until positive definite
        while True:
            try:
                R = np.linalg.cholesky(A + mu * np.eye(n))
                if 1.0 / np.linalg.cond(R) < 1e-15:
                    raise np.linalg.LinAlgError
                break
            except np.linalg.LinAlgError:
                mu *= 10.0
        h = np.linalg.solve(R.T, np.linalg.solve(R, -g))
        norm_h = np.linalg.norm(h)
        if norm_h <= tol_step * (tol_step + np.linalg.norm(x)):
            message = 'parameter step below threshold'
            break
        xnew = np.clip(x + h, xlb, xub)
        Fnew, fnew = funeval(xnew)
        # refresh one Jacobian column by finite differences when its step was small
        j = (j % n) + 1
        if abs(h[j - 1]) < 0.8 * norm_h:
            xu = x.copy(); xu[j - 1] += delta
            _, fu = funeval(xu)
            hu = xu - x
            J = J + np.outer((fu - f - J @ hu) / (hu @ hu), hu)
        J = J + np.outer((fnew - f - J @ h) / (h @ h), h)
        rho = (F - Fnew) / (0.5 * (h @ (mu * h - g)))
        if rho > 0:
            x, F, f = xnew, Fnew, fnew
            mu *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
            nu = 2.0
        else:
            mu *= nu
            nu *= 2.0
        g = J.T @ f
        norm_g = np.linalg.norm(g, ord=np.inf)
        A = J.T @ J
        if not np.isfinite(norm_g) or not np.isfinite(A).all():
            message = 'non-finite gradient'
            break
    p_best = untransform(x)
    return OptimizeResult(x=p_best, fun=float(objective_fn(p_best)), nfev=n_eval[0],
                          nit=it, success=True, message=message)


def _optimize_trf(
    objective_fn: Callable,
    p0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    options: FitOptions,
    resid_fn: Optional[Callable] = None,
) -> OptimizeResult:
    """Bounded trust-region reflective least squares (SciPy ``least_squares``).

    Minimizes the target-transformed residual vector returned by ``resid_fn``
    within ``bounds``, so esfit's autoscale/baseline/target handling is
    retained.  Every residual evaluation, including SciPy's finite-difference
    Jacobian probes, is recorded in the progress tracker: progress reporting,
    ``max_time``/``stop_when`` and interrupts behave as for the other methods
    (a stop raises ``_StopFit`` out of SciPy and esfit returns the best point
    so far).  ``options.max_iter`` is SciPy's ``max_nfev`` (step evaluations;
    Jacobian probes are not counted against it).  SciPy's success flag is kept,
    so an exhausted budget reports ``success=False``.  ``objective_fn`` only
    supplies the tracker.
    """
    if resid_fn is None:
        raise ValueError("_optimize_trf needs resid_fn (residual vector).")
    lb, ub = (np.asarray(b, dtype=float) for b in bounds)
    tracker = _tracker_of(objective_fn)

    import functools
    # sqrt(mean(r^2)) is exactly the RMSD compute_residuals reports
    residual = _ParentRecorded(functools.partial(_finite_residuals, resid_fn=resid_fn), tracker, _rms)
    kw = {}
    mapper = _scipy_map(options, tracker, _rms, 2 * len(p0))
    if mapper is not None:
        if _scipy_takes_workers('trf'):
            kw['workers'] = mapper
        else:
            warnings.warn("esfit: parallel trf Jacobians need SciPy >= 1.16; evaluating them sequentially")
    x0 = np.clip(np.asarray(p0, dtype=float), lb, ub)
    fit = least_squares(residual, x0, bounds=(lb, ub), method='trf',
                        x_scale=options.trf_x_scale, diff_step=options.trf_diff_step,
                        max_nfev=int(options.max_iter), ftol=options.tol_fun, xtol=options.tol_x,
                        gtol=options.trf_gradient_tol, **kw)
    # SciPy reports no iteration count for 'trf'; each trust-region iteration
    # evaluates the Jacobian once, so njev is the closest measure.
    n_iter = fit.njev if getattr(fit, 'njev', None) is not None else fit.nfev
    if getattr(fit, 'jac', None) is not None:
        _warn_dead_jacobian_columns(fit.jac, 'trf')
    return OptimizeResult(x=fit.x, fun=float(np.sqrt(np.mean(fit.fun ** 2))), success=bool(fit.success),
                          message=str(fit.message), nfev=int(fit.nfev), nit=int(n_iter))


_DEAD_COLUMN_HINT = (
    "A piecewise-constant model has no finite-difference derivative: garlic's "
    "default Options.AccumMethod='binning' places each line in its nearest field "
    "bin, so small shifts of g or A do not change the spectrum. Use "
    "AccumMethod='linear' or 'explicit' in the model.")


def _dead_jacobian_columns(J: np.ndarray, rel_tol: float = 1e-12) -> np.ndarray:
    """Boolean mask of Jacobian columns that are zero, or negligible relative to
    the largest column (the parameter has no effect at the finite-difference step)."""
    J = np.asarray(J, dtype=float)
    if J.ndim != 2 or J.shape[1] == 0:
        return np.zeros(0, dtype=bool)
    norms = np.linalg.norm(J, axis=0)
    ref = float(np.max(norms)) if np.all(np.isfinite(norms)) else 0.0
    return ~(norms > rel_tol * ref) if ref > 0 else np.ones(J.shape[1], dtype=bool)


def _warn_dead_jacobian_columns(J: np.ndarray, where: str, names=None) -> np.ndarray:
    """Warn if any Jacobian column is dead (see _dead_jacobian_columns); return the mask."""
    dead = _dead_jacobian_columns(J)
    if dead.any():
        idx = np.flatnonzero(dead)
        label = ', '.join(str(names[i]) if names is not None else f'#{i}' for i in idx)
        warnings.warn(
            f"esfit ({where}): the residuals do not change with parameter(s) {label} at "
            f"the finite-difference step, so they are invisible to the "
            f"{'optimizer' if where != 'uncertainties' else 'error estimate'}. "
            + _DEAD_COLUMN_HINT, UserWarning, stacklevel=3)
    return dead


def _estimate_uncertainties(
    pfit: np.ndarray,
    data: np.ndarray,
    model_wrapper: ModelWrapper,
    param_handler: ParameterHandler,
    options: FitOptions
) -> dict:
    """
    Estimate parameter uncertainties via Jacobian + covariance.
    
    Returns
    -------
    uncertainties : dict
        'pstd': standard deviations
        'ci95': 95% confidence intervals (n_params x 2)
        'cov': covariance matrix
        'corr': correlation matrix
    """
    # Residual function: a partial of a module-level function on a copy of the
    # options without the in-process callables, so it pickles for the worker pool
    import functools
    from dataclasses import replace as _replace
    residuals_fn = functools.partial(
        _plain_residuals, data=data, model_wrapper=model_wrapper, param_handler=param_handler,
        options=_replace(options, progress=None, stop_when=None, callback=None))

    # Compute Jacobian via finite differences (probes concurrent with n_workers > 1)
    J_all = _jacobian_finite_diff(residuals_fn, pfit, n_workers=options.n_workers, workers=options.workers)
    # A parameter the residuals do not depend on would get sigma = 0 from pinv,
    # which reads as perfectly determined. Report it as undetermined (nan)
    # instead and leave it out of the covariance of the others.
    dead = _warn_dead_jacobian_columns(J_all, 'uncertainties', param_handler.get_names())
    live = ~dead
    if not live.any():
        nan = np.full(len(pfit), np.nan)
        return {'pstd': nan, 'ci95': np.column_stack([nan, nan]),
                'cov': np.full((len(pfit),) * 2, np.nan),
                'corr': np.full((len(pfit),) * 2, np.nan)}
    J = J_all[:, live]

    # Residuals at the optimum, before the target transformation T: the noise
    # is (close to) independent there, so the sandwich's middle term is
    # J' T diag(r_raw^2) T' J (for 'fcn', T = I: the usual HC1). Using the
    # transformed residuals instead would treat the random walk of 'int' as
    # independent noise and understate the uncertainties (B131).
    residuals = compute_residuals(pfit, data, model_wrapper, param_handler, options)[2]['residuals_raw']
    n_data = len(residuals)
    n_params = len(pfit)

    def _embed(m_live):
        m = np.full((n_params, n_params), np.nan)
        m[np.ix_(live, live)] = m_live
        return m

    # Compute covariance matrix (HC1 estimator)
    try:
        JtJ_inv = np.linalg.pinv(J.T @ J)
        # HC1 sandwich estimator: avoid building O(n²) diagonal matrix
        # (M.T @ diag(r²) @ M) == (M.T * r²) @ M, M = T' J
        W   = residuals ** 2                          # (n_data,)
        M   = _target_transform_adjoint(J, options.target)
        JtWJ = (M.T * W) @ M                          # (n_params, n_params)
        cov  = JtJ_inv @ JtWJ @ JtJ_inv
        cov *= n_data / max(n_data - n_params, 1)     # degrees-of-freedom correction

        # Standard deviations — clamp to avoid sqrt(negative) from float noise
        diag_cov = np.maximum(np.diag(cov), 0.0)

        # Correlation matrix — use same clamped diagonal to avoid NaN
        D_inv = np.diag(1.0 / (np.sqrt(diag_cov) + 1e-12))
        corr = _embed(D_inv @ cov @ D_inv)
        cov = _embed(cov)
        pstd = np.full(n_params, np.nan)
        pstd[live] = np.sqrt(diag_cov)

        # 95% confidence intervals (1.96 = norm.ppf(0.975))
        ci95 = np.column_stack([pfit - 1.96 * pstd, pfit + 1.96 * pstd])
        
        return {
            'pstd': pstd,
            'ci95': ci95,
            'cov': cov,
            'corr': corr
        }
    except np.linalg.LinAlgError as e:
        if options.verbosity > 0:
            print(f"Warning: Could not compute uncertainties: {e}")
        return {
            'pstd': None,
            'ci95': None,
            'cov': None,
            'corr': None
        }


def _jacobian_finite_diff(
    fn: Callable,
    x: np.ndarray,
    eps: float = 1e-7,
    n_workers=1,
    workers: str = 'auto',
) -> np.ndarray:
    """Compute Jacobian via central finite differences.

    The 2n probes are independent; with ``n_workers > 1`` they are evaluated
    concurrently (see :func:`_map_points`).
    """
    n_params = len(x)
    steps, plus, minus = [], [], []
    for i in range(n_params):
        x_plus = x.copy()
        x_minus = x.copy()
        step = eps * max(abs(x[i]), 1.0)
        x_plus[i] += step
        x_minus[i] -= step
        steps.append(step); plus.append(x_plus); minus.append(x_minus)
    if _resolve_workers(n_workers, 2 * n_params) > 1:
        R = _map_points(fn, plus + minus, n_workers, workers, size_hint=2 * n_params)
        fp, fm = R[:n_params], R[n_params:]
    else:
        fp, fm = [], []
        for i in range(n_params):
            fp.append(fn(plus[i])); fm.append(fn(minus[i]))
    n_data = len(np.asarray(fp[0]).ravel())
    J = np.zeros((n_data, n_params))
    for i in range(n_params):
        J[:, i] = (np.asarray(fp[i]) - np.asarray(fm[i])) / (2 * steps[i])
    return J


def _compute_fit_diagnostics(
    residuals: np.ndarray,
    fit: np.ndarray,
    n_params: int
) -> dict:
    """
    Compute additional fit quality metrics.
    
    Parameters
    ----------
    residuals : np.ndarray
        Fit residuals
    fit : np.ndarray
        Fitted values
    n_params : int
        Number of fitted parameters
    
    Returns
    -------
    diagnostics : dict
        R²: coefficient of determination
        reduced_chi_square: χ²/(n_data - n_params)
        noise_std: estimated noise level
    """
    data = fit + residuals  # Reconstruct data
    n_data = len(residuals)
    
    # R²
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((data - np.mean(data))**2)
    r_squared = 1 - ss_res / (ss_tot + 1e-12)
    
    # Reduced chi-square: ss_res / dof
    # Without an independent noise estimate, this equals the estimated noise
    # variance.  It is NOT a true chi-square test (which requires known sigma).
    # Values >> 1 indicate a poor fit; values near 0 indicate overfitting.
    dof = max(n_data - n_params, 1)
    reduced_chi_square = ss_res / dof
    noise_std = np.sqrt(reduced_chi_square)

    return {
        'r_squared': r_squared,
        'reduced_chi_square': reduced_chi_square,
        'noise_std': noise_std
    }


def esfit(
    data: Union[np.ndarray, torch.Tensor],
    model_fn: Callable,
    p0: Union[np.ndarray, dict],
    vary: Optional[Union[np.ndarray, dict]] = None,
    lb: Optional[Union[np.ndarray, dict]] = None,
    ub: Optional[Union[np.ndarray, dict]] = None,
    options: Optional[FitOptions] = None
) -> FitResult:
    """Fit a model to data by least squares (port of EasySpin's ``esfit``).

    The model output is scaled and given a polynomial baseline
    (``FitOptions.autoscale``, ``FitOptions.baseline``), the residuals
    ``fit - data`` are transformed (``FitOptions.target``), and their RMSD is
    minimized over the varied parameters within their bounds.

    Parameters
    ----------
    data : array_like or torch.Tensor
        Experimental data. Flattened to 1-D, so any shape works as long as
        the model returns the same number of values (``n_data``).
    model_fn : callable
        The model. Its calling convention depends on ``p0``:

        - Array ``p0``: ``model_fn(p)`` with ``p`` the full parameter vector,
          a ``numpy.ndarray`` of shape ``(n_params,)`` (fixed entries
          included).
        - Dict ``p0`` and a torchspin simulator (``model_fn.__name__`` is
          ``pepper``, ``garlic``, ``chili`` or ``salt``, or the function is
          defined in a ``torchspin`` module): ``model_fn(p['Sys'],
          p['Exp'])``, or ``model_fn(p['Sys'], p['Exp'], p['Opt'])`` when
          ``p0`` has an ``'Opt'`` entry; the second return value (the
          spectrum) is the model output. For :func:`~torchspin.garlic`
          an unset ``Options.AccumMethod`` is switched to ``'linear'``
          (except in the fast-motion regime or with ``Exp.mwPhase``), because
          the default nearest-bin accumulation is piecewise constant in g
          and A.
        - Dict ``p0`` and any other callable: ``model_fn(p)`` with ``p`` a
          deep copy of ``p0`` whose varied fields hold the trial values.

        The output may be any array-like or tensor (detached and flattened);
        it must have ``n_data`` elements.
    p0 : array_like, shape (n_params,), or dict
        Start values. Array style: one float per parameter. Dict style: a
        mapping from names to objects, e.g. ``{'Sys': SpinSystem(...),
        'Exp': Experiment(...), 'Opt': Options(...)}``; the fields named in
        ``vary`` (or ``lb``/``ub``) are read from these objects and must be
        scalars (int or float) or lists/arrays/tensors, which are varied
        element-wise in the shape the field has on the object. For a
        :class:`~torchspin.spinsystem.SpinSystem` that is the normalized
        shape, e.g. ``g`` is ``(nElectrons, 3)``, ``A`` is ``(nNuclei,
        3*nElectrons)`` and ``lw`` is ``[GaussianFWHM, LorentzianFWHM]``.
    vary : array_like, shape (n_params,), or dict, optional
        Allowed variation: the bounds are ``p0 - vary`` and ``p0 + vary``,
        and ``vary = 0`` fixes a parameter. Dict style mirrors ``p0``, e.g.
        ``{'Sys': {'lw': [0.0, 0.3]}}``, each value in the shape of its field.
        In the dict style the lower bound of fields that cannot be negative
        (``lw``, ``lwpp``, ``lwEndor``, ``HStrain``, ``HStrainL``,
        ``gStrain``, ``AStrain``, ``DStrain``, ``weight``, ``initState``,
        ``tcorr``, ``Diff``, ``T1``, ``T2``) is clipped at zero, as EasySpin's
        ``esfit`` does. A plain vector carries no field names and is used as
        given, so ``vary`` larger than ``p0`` there means a negative lower
        bound. Give either ``vary`` or ``lb``/``ub``.
    lb, ub : array_like, shape (n_params,), or dict, optional
        Explicit lower and upper bounds, used exactly as given; ``lb == ub``
        fixes a parameter. Dict style: both are required, with the same keys
        and fields as each other, the layout of ``vary``, and ``p0`` within
        them. Array style: a missing ``lb`` or ``ub`` is ``-inf`` or
        ``+inf``, which only ``'lbfgsb'``, ``'powell'`` and ``'trf'`` handle;
        the other methods need finite bounds (with an infinite bound they
        return NaN parameters, raise, or, for ``'grid'``, return ``p0``).
    options : FitOptions, optional
        Algorithm, scaling, baseline, target, budget, parallelism and
        progress. Default ``FitOptions()``.

    Returns
    -------
    FitResult
        Best-fit active parameters (``pfit``, ``pnames``), the fitted curve
        and residuals (each of shape ``(n_data,)``), ``scale``, ``baseline``,
        ``rmsd``, uncertainties (``pstd``, ``ci95``, ``cov``, ``corr``) and
        diagnostics. See :class:`FitResult`.

    Raises
    ------
    ValueError
        If both ``vary`` and ``lb``/``ub`` or neither are given; if an array
        ``vary``, ``lb`` or ``ub`` does not have one entry per parameter; if
        every parameter is fixed; in the dict style, if a key is not in
        ``p0``, an object has no such field, a shape does not match the
        field, only one of ``lb``/``ub`` is given, they name different keys
        or fields, ``lb > ub``, or ``p0`` lies outside ``[lb, ub]``; if
        ``FitOptions.method`` is unknown (checked after the initial model
        evaluation). An unknown ``FitOptions.target`` or
        ``lsq_negative_scale``, or a model output with the wrong number of
        values, raises ``ValueError`` only at the final evaluation, after the
        optimizer has run (see Notes).
    TypeError
        In the dict style, if a varied field is neither a number nor a
        list/array/tensor.

    Warns
    -----
    UserWarning
        If ``autoscale='lsq'`` finds a negative scale at the fitted
        parameters; if a parameter has no effect on the residuals at the
        finite-difference step (its ``pstd`` is then ``nan``); if parallel
        ``'trf'``/``'lbfgsb'`` probes are requested with SciPy < 1.16.

    See Also
    --------
    FitOptions, FitResult
    torchspin.pepper, torchspin.garlic : Typical models.

    Notes
    -----
    Before each evaluation the parameters are clipped to their bounds. During
    the optimization a model that raises ``ValueError``, ``RuntimeError`` or
    ``numpy.linalg.LinAlgError`` scores an RMSD of 1e10 instead of stopping
    the fit. The first such failure of a fit is reported with a warning, and
    ``FitResult.n_failed_evals`` counts them all, so a configuration error
    does not go unnoticed even though the fit runs on.

    The local optimizers (``'levmar'``, ``'trf'``, ``'lbfgsb'``) and the
    uncertainty estimate use finite differences, so the model must be smooth
    in its parameters. A custom model that wraps garlic should pass
    ``Options(AccumMethod='linear')`` or ``'explicit'``. See docs/pitfalls.md,
    "garlic's ``AccumMethod='binning'`` in fits" and "Model sign with
    ``autoscale='lsq'``".

    Uncertainties: the Jacobian of the target-transformed residuals with
    respect to the active parameters is taken by central differences (step
    ``1e-7 * max(|p|, 1)``), the covariance is the HC1 sandwich estimator
    ``(J'J)^-1 J' diag(r^2) J (J'J)^-1 * n_data / (n_data - n_active)``, and
    ``ci95`` is ``pfit +/- 1.96 * pstd``. ``r`` are the untransformed
    residuals; for ``target='int'`` or ``'diff'`` the middle term is
    ``J' T diag(r^2) T' J`` with ``T`` the transformation, which accounts for
    the correlation it gives the noise.

    Examples
    --------
    Fit the position and width of a Gaussian line; the amplitude is
    absorbed by the least-squares scale:

    >>> import numpy as np
    >>> from torchspin import esfit, FitOptions
    >>> x = np.linspace(-5, 5, 201)
    >>> def model(p):
    ...     return np.exp(-0.5 * ((x - p[0]) / p[1]) ** 2)
    >>> data = 3.0 * model([0.5, 1.2])
    >>> res = esfit(data, model, p0=[0.0, 1.0], vary=[2.0, 0.5],
    ...             options=FitOptions(method='levmar'))
    >>> np.round(res.pfit, 3).tolist(), round(res.scale, 3)
    ([0.5, 1.2], 3.0)

    Explicit bounds; ``lb == ub`` fixes the width, so only the position is
    fitted:

    >>> res = esfit(data, model, p0=[0.0, 1.2], lb=[-1.0, 1.2], ub=[1.0, 1.2],
    ...             options=FitOptions(method='levmar'))
    >>> res.pnames, res.p_fixed.tolist(), round(float(res.pfit[0]), 3)
    (['p0'], [False, True], 0.5)

    Dict style with a torchspin simulator: fit the Lorentzian line width of
    a garlic spectrum (``lw = [GaussianFWHM, LorentzianFWHM]`` in mT; the
    Gaussian width is fixed by ``vary = 0``):

    >>> from torchspin import SpinSystem, Experiment, garlic
    >>> exp = Experiment(mwFreq=9.5, Range=[335, 343], nPoints=256)
    >>> _, data = garlic(SpinSystem(S=0.5, g=2.003, lw=[0.0, 0.5]), exp)
    >>> p0 = {'Sys': SpinSystem(S=0.5, g=2.003, lw=[0.0, 0.4]), 'Exp': exp}
    >>> res = esfit(data, garlic, p0, vary={'Sys': {'lw': [0.0, 0.3]}},
    ...             options=FitOptions(method='levmar'))
    >>> res.pnames, round(float(res.pfit[0]), 2)
    (['Sys.lw[1]'], 0.5)
    """
    # 1. Validate inputs
    data = np.asarray(data).ravel()
    if options is None:
        options = FitOptions()

    # Resolve target='auto' (opt-in since B131; the default is 'fcn', as in
    # EasySpin 6): 'int' for derivative-like data (mean << std, typical of
    # field-modulated CW EPR), whose integrated loss is smoother for local
    # optimizers from a poor start.
    if options.target == 'auto':
        data_mean = float(np.abs(np.mean(data)))
        data_std = float(np.std(data))
        is_derivative_like = data_std > 0 and data_mean < 0.1 * data_std
        from dataclasses import replace
        options = replace(options, target='int' if is_derivative_like else 'fcn')
        if options.verbosity > 0:
            print(f"  Target (auto-detected): {options.target} "
                  f"(|mean|/std = {data_mean / max(data_std, 1e-30):.3g})")

    if options.verbosity > 0:
        print("Starting esfit...")
        print(f"  Algorithm: {options.method}")
        print(f"  Autoscale: {options.autoscale}")
        print(f"  Baseline:  {options.baseline}")
        print(f"  Target:    {options.target}")
        print(f"  Data points: {len(data)}")
    
    # 2. Parse parameters and bounds
    param_handler = ParameterHandler(p0, vary=vary, lb=lb, ub=ub)
    pvec_0 = param_handler.to_vector(p0)
    lb_vec, ub_vec = param_handler.get_bounds()
    pnames = param_handler.get_names()
    
    if options.verbosity > 0:
        print(f"  Parameters: {param_handler.n_active} variable, {param_handler.n_params - param_handler.n_active} fixed")
    
    # 3. Wrap model function
    model_wrapper = ModelWrapper(model_fn, p0, len(data))
    
    # 4. Define objective function (a partial of a worker-safe options copy, so it
    #    pickles for the process pool; progress/stop callables stay in this process)
    import functools
    from dataclasses import replace as _replace
    worker_options = _replace(options, progress=None, stop_when=None, callback=None)
    obj_fn = functools.partial(_objective_function, data=data, model_wrapper=model_wrapper,
                               param_handler=param_handler, options=worker_options)
    tracker = _FitProgress(options, options.method, n_params=param_handler.n_active, n_data=len(data))
    tracked = _TrackedObjective(obj_fn, tracker)
    
    # 5. Evaluate initial RMSD (recorded, so an early interrupt still returns p0)
    rmsd_0 = obj_fn(pvec_0)
    tracker.record(pvec_0, rmsd_0, emit=False)     # before the guarded block: no reporting/stop yet
    if options.verbosity > 0:
        print(f"  Initial RMSD: {rmsd_0:.6e}")
    
    # 6. Run optimization
    if options.verbosity > 0:
        print("\nOptimizing...")
    if options.method not in ('simplex', 'lbfgsb', 'powell', 'grid', 'montecarlo', 'genetic', 'swarm', 'global', 'levmar', 'trf'):
        raise ValueError(f"Unknown method: '{options.method}'")

    # FitOptions.devices for the worker pools of this fit only; reset on every exit path,
    # so a later pool (another fit, ml.simulate_batch) is not pinned to this fit's GPUs
    _dev_token = _FIT_DEVICES.set(list(options.devices) if options.devices else None)
    try:
        tracker.start()
        try:
            if options.method == 'simplex':
                opt_result = _optimize_simplex(tracked, pvec_0, (lb_vec, ub_vec), options)
            elif options.method == 'lbfgsb':
                opt_result = _optimize_lbfgsb(tracked, pvec_0, (lb_vec, ub_vec), options)
            elif options.method == 'powell':
                opt_result = _optimize_powell(tracked, pvec_0, (lb_vec, ub_vec), options)
            elif options.method == 'grid':
                opt_result = _optimize_grid(tracked, pvec_0, (lb_vec, ub_vec), options)
            elif options.method == 'montecarlo':
                opt_result = _optimize_montecarlo(tracked, pvec_0, (lb_vec, ub_vec), options)
            elif options.method == 'genetic':
                opt_result = _optimize_genetic(tracked, pvec_0, (lb_vec, ub_vec), options)
            elif options.method == 'swarm':
                opt_result = _optimize_swarm(tracked, pvec_0, (lb_vec, ub_vec), options)
            elif options.method == 'global':
                opt_result = _optimize_global(tracked, pvec_0, (lb_vec, ub_vec), options)
            else:
                # a partial of a module-level function, so it pickles for the worker pool
                resid_fn = functools.partial(_bounded_residuals, lb=lb_vec, ub=ub_vec, data=data,
                                             model_wrapper=model_wrapper, param_handler=param_handler,
                                             options=worker_options)
                local_solver = _optimize_trf if options.method == 'trf' else _optimize_levmar
                opt_result = local_solver(tracked, pvec_0, (lb_vec, ub_vec), options, resid_fn=resid_fn)
        except KeyboardInterrupt:
            opt_result = tracker.synthesize_result('KeyboardInterrupt')
        except _StopFit as e:
            opt_result = tracker.synthesize_result(e.reason)
        finally:
            tracker.finish()
    
        pfit_vec = opt_result.x
    
        # 7. Compute final fit and residuals (repeats of points the optimizer
        # evaluated are served from ModelWrapper's memo from here on)
        model_wrapper.serve_memo = True
        residuals, rmsd, info = compute_residuals(
            pfit_vec, data, model_wrapper, param_handler, options
        )
        if info.get('negative_scale'):
            warnings.warn(
                "esfit: lsq autoscale found a negative scale at the fitted parameters "
                "(the model is anti-correlated with the data), so the scale was "
                f"{'clipped to 0' if options.lsq_negative_scale == 'clip' else 'replaced by its absolute value'}. "
                "Check the sign of the model: with autoscale='lsq' it must carry the "
                "sign of the signal.", UserWarning, stacklevel=2)
    
        if options.verbosity > 0:
            print(f"\nFinal RMSD: {rmsd:.6e}")
            if rmsd_0 > 0:
                print(f"Improvement: {(rmsd_0 - rmsd)/rmsd_0 * 100:.1f}%")
            print(f"Success: {opt_result.success}")
            print(f"Message: {opt_result.message}")
    
        # 8. Compute uncertainties if requested
        if options.compute_uncertainties and opt_result.success:
            if options.verbosity > 0:
                print("\nComputing uncertainties...")
            uncertainties = _estimate_uncertainties(
                pfit_vec, data, model_wrapper, param_handler, options
            )
        else:
            uncertainties = {
                'pstd': None,
                'ci95': None,
                'cov': None,
                'corr': None
            }
    
        # 9. Compute fit diagnostics
        diagnostics = _compute_fit_diagnostics(info['residuals_raw'], info['fit'], len(pfit_vec))
    
        # 10. Construct result object
        result = FitResult(
            pfit=pfit_vec,
            pnames=pnames,
            p_start=pvec_0,
            p_fixed=param_handler.fixed_mask,
            fit=info['fit'],
            fit_raw=info['sim'],
            residuals=info['residuals_raw'],
            scale=info['scale'],
            baseline=info['baseline'],
            rmsd=rmsd,
            ssr=np.sum(residuals**2),
            n_params=len(pfit_vec),
            n_data=len(data),
            algorithm=options.method,
            n_iterations=opt_result.nit if hasattr(opt_result, 'nit') else opt_result.nfev,
            success=opt_result.success,
            message=opt_result.message,
            pstd=uncertainties['pstd'],
            ci95=uncertainties['ci95'],
            cov=uncertainties['cov'],
            corr=uncertainties['corr'],
            r_squared=diagnostics['r_squared'],
            reduced_chi_square=diagnostics['reduced_chi_square'],
            noise_std=diagnostics['noise_std'],
            n_evaluations=tracker.n_evaluations,
            elapsed_s=tracker.elapsed_s,
            interrupted=tracker.interrupted,
            n_failed_evals=tracker.n_failed_evals,
        )
    
        if options.verbosity > 0:
            print("\nFitted parameters:")
            for i, (name, val, std) in enumerate(zip(pnames, pfit_vec, uncertainties['pstd'] if uncertainties['pstd'] is not None else [None]*len(pfit_vec))):
                if std is not None:
                    print(f"  {name}: {val:.6e} ± {std:.6e}")
                else:
                    print(f"  {name}: {val:.6e}")
            print(f"\nFit quality:")
            print(f"  RMSD:      {result.rmsd:.6e}")
            print(f"  R²:        {result.r_squared:.6f}")
            print(f"  Red. χ²:   {result.reduced_chi_square:.6f}")
            print(f"  Noise std: {result.noise_std:.6e}")
            print()
    
        return result
    finally:
        _FIT_DEVICES.reset(_dev_token)
