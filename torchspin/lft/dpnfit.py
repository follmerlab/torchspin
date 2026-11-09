"""DPNFIT: fit p^n ligand-field parameters to observed level (band) energies, with an exact Jacobian.

Telser's DPNFIT model (DPNFITCAL.F SCALC) in torch: the parameter vector X maps to the p^n
Hamiltonian of :mod:`torchspin.lft.dpn`; an observation assigned to state number IE
(1-based, **all** states counted in ascending energy, every spin and every component) is
compared with D(IE) - D(1).  The residuals are differentiable in X, the Jacobian is exact
(forward-mode AD, ``torch.func.jacfwd``), and :meth:`DpnFit.fit` runs
``scipy.optimize.least_squares`` (trust-region reflective, with bounds) on it.

X vector (NX = 1 for p^1/p^5, 3 otherwise; missing trailing slots read as 0, as in SCALC):

    X[0]             zeta (|zeta| is used; d|zeta|/dzeta = +1 at 0)
    X[1], X[2]       F2 (Slater F^2, as in DPN), Trees alpha             (not for p^1, p^5)
    lf_type 'X'      Ds, Dr                                              (crystal field)
    lf_type 'A'      nlam values per ligand: sigma, pi-s (y), pi-c (x)   (AOM)
    lf_type 'P'      E(px), E(py), E(pz)                                 (orbital energies)

with DPNFIT's fixed constraints: AOM pi factors (``pi_factors`` per ligand: e_pi_s = f_ss
e_sigma, e_pi_c = f_cs e_sigma, then e_pi_c = f_sc e_pi_s, each only if nonzero), ligand
equivalences (``isymm``, 1-based, with ``blfact``), and option P orbital equivalences
(``orbital_isymm``).  There are no k slots: k enters only with a field, as fixed input.

Units
-----
Every energy slot of X, the observed energies, and the energies returned are in ``units``
(default ``'MHz'``; ``'cm^-1'`` for Telser-style input).  A weight w multiplies the squared
residual: residual = sqrt(w) (calc - obs); with w = 1/sigma^2 (sigma in ``units``) the
residuals are dimensionless.  The field magnitude ``B0`` is in mT; angles (ligand and field
direction) in radians, or degrees with ``degrees=True``.

Modes
-----
``mode="exact"`` (default): the direct p^n calculation, torchspin constants, the geometric
ligand field with every element, and the full Zeeman operator.  ``mode="dpn"`` reproduces the
compiled program (:func:`torchspin.lft.dpn.dpn_hamiltonian` with ``program='dpnfit'``): DPN's
constants, the hole route for p^4/p^5 with -V and -zeta, the mirror-image AOM for
p^1/p^2/p^4/p^5, the dropped <+-1|V|0> AOM elements (DPN-01), and the Zeeman defect (DPNFIT-01).
Without a field and without tilted AOM ligands the two agree.

LOWSPN
------
With ``lowspn=True`` the ground state must carry the spin ``spngrnd`` (the S of its dominant
free-ion term, ties to the later term in DPN's order).  Where it does not, every weighted
residual is replaced by sqrt(w) * 1e5 cm^-1 (``lowspn_policy="penalty"``, default; DPNFIT
itself returns a stale objective there, finding DPNFIT-02) with derivative 0, or the residuals
are computed anyway (``"ignore"``).

Uncertainties
-------------
:class:`FitResult` carries the exact Jacobian at the solution, (J^T J)^-1, residual-scaled
standard errors, and the parameter combinations the data do not determine (``null``; e.g.
e_sigma - e_pi of one ligand only).  ``measurements`` repeats the statistics with each band
counted once: DPNFIT inputs list every component of a degenerate level as a row with the same
energy, which overstates the data by the degeneracy (rows with the same energy and weight on
consecutive state numbers form one measurement; rows with a vanishing Jacobian, such as the
ground level at 0, carry no information).

Not ported: the DPNFIT input reader and report, STEPIT itself, and the assignment monitor and
level-crossing tracker of the NumPy program.

    >>> from torchspin.lft import dpnfit
    >>> m = dpnfit.DpnFitModel(2, lf_type="X", units="cm^-1")
    >>> prob = dpnfit.DpnFit(m, states=[6, 11], energies=[10000.0, 25000.0], x0=[500.0, 40000.0, 0.0, 0.0, 0.0],
    ...                      free=[True, True, False, False, False])
    >>> res = prob.fit()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, NamedTuple, Sequence

import numpy as np
import torch
from scipy.optimize import least_squares
from torch.func import jacfwd

from . import units as _u
from ._linalg import eigvalsh_gauge
from .dpn import (DPN_CONSTANTS, TORCHSPIN_CONSTANTS, Constants, aom_from_angles, crystal_field, dpn_hamiltonian,
                  field_vector, hamiltonian, orbital_energies, pn_operators)

__all__ = ["AOM_MODES", "LOWSPN_PENALTY_CM", "ORBITALS", "DpnFit", "DpnFitModel", "Evaluation", "FitResult",
           "MeasurementStats", "null_directions", "spin_clamp"]

_F64 = torch.float64

AOM_MODES = ("esigma", "epi_s", "epi_c", "edelta_s", "edelta_c")
ORBITALS = ("E_px", "E_py", "E_pz")
#: penalty per weighted residual (cm^-1) when LOWSPN rejects the ground spin
LOWSPN_PENALTY_CM = 1.0e5


def spin_clamp(np_: int, s: float) -> float:
    """DPNFIT.F ll. 217-226: the ground-state spin as the program stores it (NP = n or 6 - n)."""
    if np_ == 1:
        return 0.5
    if np_ == 2:
        return 0.0 if s <= 0.0 else 1.0
    return 0.5 if s <= 0.5 else 1.5


# ------------------------------------------------------------------------------ model
@dataclass
class DpnFitModel:
    """Fixed (non-fitted) input of DPNFIT: configuration, ligand-field model, field, units.

    ``angles``: per AOM ligand (theta, phi, psi); ``isymm``: 1-based ligand equivalence
    (ligand j copies ligand isymm[j] - 1 scaled by ``blfact[j]``; isymm[j] = j + 1 is
    independent); ``pi_factors``: per ligand (f_ss, f_cs, f_sc), see the module docstring;
    ``orbital_isymm``: option P equivalences (E(p_j) = E(p_i) if equal, i < j).
    ``B0`` field magnitude in mT along (``theta``, ``phi``); ``k`` the orbital reduction.
    ``constants`` default to torchspin's (``'exact'``) or DPN's (``'dpn'``).
    """

    n: int
    lf_type: str = "X"
    nlam: int = 1
    angles: Sequence[Sequence[float]] = field(default_factory=list)
    isymm: Sequence[int] = field(default_factory=list)
    blfact: Sequence[float] = field(default_factory=list)
    pi_factors: Sequence[Sequence[float]] = field(default_factory=list)
    orbital_isymm: Sequence[int] = (1, 2, 3)
    B0: float = 0.0
    theta: float = 0.0
    phi: float = 0.0
    k: Sequence[float] = (1.0, 1.0, 1.0)
    mode: str = "exact"
    degrees: bool = False
    units: str = "MHz"
    constants: Constants | None = None

    def __post_init__(self):
        if self.mode not in ("dpn", "exact"):
            raise ValueError("mode must be 'dpn' or 'exact'")
        if not 1 <= self.n <= 5:
            raise ValueError("DPNFIT handles p1 to p5")
        if self.lf_type.upper() not in ("X", "A", "P"):
            raise ValueError("lf_type must be 'X' (Ds, Dr), 'A' (AOM) or 'P' (orbital energies)")
        self.units = _u.canonical_energy_unit(self.units)
        if self.constants is None:
            self.constants = DPN_CONSTANTS if self.mode == "dpn" else TORCHSPIN_CONSTANTS
        ang = np.asarray(self.angles, float).reshape(-1, 3)
        self._angles = torch.as_tensor(np.radians(ang) if self.degrees else ang, dtype=_F64)

    @property
    def np_(self) -> int:
        """DPNFIT's NP: n for n <= 3, else 6 - n."""
        return self.n if self.n <= 3 else 6 - self.n

    @property
    def nlig(self) -> int:
        return len(self._angles)

    @property
    def offset(self) -> int:
        """Index (0-based) of the first ligand-field slot (DPNFIT's NX)."""
        return 1 if self.np_ == 1 else 3

    @property
    def n_states(self) -> int:
        return pn_operators(self.n).size

    def parameter_names(self) -> list[str]:
        names = ["zeta"] + ([] if self.np_ == 1 else ["F2", "alpha"])
        t = self.lf_type.upper()
        if t == "A":
            names += [f"{AOM_MODES[m]}{i + 1}" for i in range(self.nlig) for m in range(self.nlam)]
        elif t == "P":
            names += list(ORBITALS)
        else:
            names += ["Ds", "Dr"]
        return names

    @property
    def n_parameters(self) -> int:
        return len(self.parameter_names())

    def _x(self, x) -> torch.Tensor:
        x = x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x, float), dtype=_F64)
        need = self.n_parameters
        if x.shape[-1] < need:
            x = torch.cat([x, x.new_zeros(need - x.shape[-1])])
        return x

    # ------------------------------------------------------------------ ligand field
    def bonding(self, x) -> torch.Tensor:
        """(nlig, 3) (e_sigma, e_pi_s(y), e_pi_c(x)) after pi factors and equivalences, in ``units``."""
        x = self._x(x)
        o, nl, nlam = self.offset, self.nlig, self.nlam
        zero = x.new_zeros(())
        e = [[x[o + nlam * i + m] if m < nlam else zero for m in range(5)] for i in range(nl)]
        isymm = list(self.isymm) if len(self.isymm) else [0] * nl
        if nlam >= 3 and len(self.pi_factors):
            for i in range(nl):
                if isymm[i] < i + 1:
                    continue
                pss, pcs, psc = (float(v) for v in self.pi_factors[i])
                if pss != 0.0:
                    e[i][1] = e[i][0] * pss
                if pcs != 0.0:
                    e[i][2] = e[i][0] * pcs
                if psc != 0.0:
                    e[i][2] = e[i][1] * psc
        for i in range(nl - 1):
            for j in range(i + 1, nl):
                if isymm[j] != i + 1:
                    continue
                f = float(self.blfact[j]) if len(self.blfact) else 1.0
                e[j][:nlam] = [v * f for v in e[i][:nlam]]
        if not nl:
            return x.new_zeros(0, 3)
        return torch.stack([torch.stack(row[:3]) for row in e])

    def orbital_energies(self, x) -> torch.Tensor:
        """Option P: (E(px), E(py), E(pz)) after the equivalences, in ``units``."""
        x = self._x(x)
        o = self.offset
        e = [x[o], x[o + 1], x[o + 2]]
        s = list(self.orbital_isymm)
        for i in range(2):
            for j in range(i + 1, 3):
                if s[j] == s[i]:
                    e[j] = e[i]
        return torch.stack(e)

    def ligand_field(self, x) -> torch.Tensor:
        """Real-orbital ligand-field matrix (px, py, pz), geometric convention, in ``units``."""
        x = self._x(x)
        t = self.lf_type.upper()
        if t == "A":
            if not self.nlig:
                return x.new_zeros(3, 3)
            b = self.bonding(x)
            e = torch.stack([b[:, 0], b[:, 2], b[:, 1]], -1)        # (sigma, pi_x = pi-c, pi_y = pi-s)
            a = self._angles
            return aom_from_angles(a[:, 0], a[:, 1], a[:, 2], e)
        if t == "P":
            return orbital_energies(self.orbital_energies(x))
        o = self.offset
        return crystal_field(x[o], x[o + 1])

    def field(self) -> torch.Tensor | None:
        """Field vector in mT, None at zero field."""
        if self.B0 <= 0.0:
            return None
        return field_vector(self.B0, self.theta, self.phi, degrees=self.degrees)

    def hamiltonian(self, x) -> torch.Tensor:
        """Complex Hermitian H(X) in ``units`` (size C(6, n) in both modes)."""
        x = self._x(x)
        zero = x.new_zeros(())
        F2, alpha = (zero, zero) if self.np_ == 1 else (x[1], x[2])
        V = self.ligand_field(x)
        k = torch.as_tensor(np.asarray(self.k, float), dtype=_F64)
        f = self.field()
        if self.mode == "exact":
            zeta = x[0] * torch.where(x[0].detach() < 0, -1.0, 1.0).to(x.dtype)
            return hamiltonian(self.n, F2=F2, zeta=zeta, alpha=alpha, V=V, k=k, field=f, units=self.units,
                               constants=self.constants)
        return dpn_hamiltonian(self.n, F2=F2, zeta=x[0], alpha=alpha, V=V, k=k, field=f, lf_type=self.lf_type,
                               program="dpnfit", units=self.units, constants=self.constants)


# ------------------------------------------------------------------------------ results
class Evaluation(NamedTuple):
    """SCALC at one X: every energy relative to the lowest (``units``), per-observation values."""

    x: np.ndarray
    energies: np.ndarray
    calc: np.ndarray
    residuals: np.ndarray
    objective: float                 # sum w (calc - obs)^2 (DPNFIT's FOBJ, before any LOWSPN penalty)
    spins: np.ndarray | None         # S of the dominant term per state (None for p1, p5)
    dominant: list[str]              # label of the dominant term per state
    ground_spin_ok: bool


@dataclass
class MeasurementStats:
    """Uncertainties with repeated components of one band counted once (see the module docstring)."""

    groups: list[list[int]]
    informative: list[bool]
    n_measurements: int
    objective: float
    cov_unscaled: np.ndarray
    stderr: np.ndarray               # residual-scaled, all parameters (NaN for fixed or undetermined)
    stderr_unscaled: np.ndarray      # sqrt(diag (J^T J)^-1): right for weights 1/sigma^2

    @property
    def dof(self) -> int:
        return self.n_measurements - int(np.sum(np.isfinite(self.stderr_unscaled)))


@dataclass
class FitResult:
    """Result of :meth:`DpnFit.fit`; arrays over all parameters (fixed included) unless noted."""

    names: list[str]
    x: np.ndarray
    free: np.ndarray
    objective: float                 # sum r^2 at the solution
    initial_objective: float
    residuals: np.ndarray
    n_data: int                      # informative rows (w > 0, state != 1)
    jac: np.ndarray                  # (n_residuals, n_free), exact
    cov_unscaled: np.ndarray         # (J^T J)^-1, free x free
    nfev: int
    njev: int
    success: bool
    message: str
    at_bound: np.ndarray             # (n_free,)
    null: list[dict[str, float]]
    measurements: MeasurementStats | None = None
    final: Evaluation | None = None

    @property
    def n_free(self) -> int:
        return int(self.free.sum())

    @property
    def dof(self) -> int:
        return self.n_data - self.n_free

    def _full(self, v: np.ndarray) -> np.ndarray:
        out = np.full(len(self.names), np.nan)
        out[self.free] = v
        return out

    @property
    def cov(self) -> np.ndarray:
        """s^2 (J^T J)^-1 with s^2 = objective / dof (free x free)."""
        s2 = self.objective / self.dof if self.dof > 0 else math.nan
        return s2 * self.cov_unscaled

    @property
    def stderr(self) -> np.ndarray:
        return self._full(np.sqrt(np.clip(np.diag(self.cov), 0, None)))

    @property
    def stderr_unscaled(self) -> np.ndarray:
        return self._full(np.sqrt(np.clip(np.diag(self.cov_unscaled), 0, None)))

    @property
    def correlation(self) -> np.ndarray:
        c = self.cov_unscaled
        d = np.sqrt(np.clip(np.diag(c), 1e-300, None))
        return c / np.outer(d, d)


def null_directions(jac: np.ndarray, names: Sequence[str], rtol: float = 1e-7) -> list[dict[str, float]]:
    """Combinations of the Jacobian's columns that the data do not determine.

    Columns are normalized first; a right singular vector with singular value below ``rtol``
    times the largest is returned as {name: coefficient} (unit length, parameter units).
    """
    jac = np.asarray(jac, float)
    if jac.size == 0:
        return []
    norms = np.linalg.norm(jac, axis=0)
    out = []
    dead = norms <= 1e-300
    for j in np.flatnonzero(dead):
        out.append({names[j]: 1.0})
    live = np.flatnonzero(~dead)
    if live.size:
        _, s, vt = np.linalg.svd(jac[:, live] / norms[live], full_matrices=True)
        s = np.concatenate([s, np.zeros(live.size - s.size)])          # fewer rows than columns: rank deficient
        for k in np.flatnonzero(s < rtol * s.max()):
            v = vt[k] / norms[live]
            v = v / np.linalg.norm(v)
            v = v * np.sign(v[np.argmax(np.abs(v))])
            out.append({names[live[i]]: float(c) for i, c in enumerate(v) if abs(c) > 1e-6})
    return out


def _jacfwd_columns(fn: Callable[[torch.Tensor], torch.Tensor], x: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """d fn / d x[columns] by one vectorized forward-mode pass."""
    x0 = torch.as_tensor(np.asarray(x, float), dtype=_F64)
    cols = torch.as_tensor(np.asarray(columns, int))
    if cols.numel() == 0:
        with torch.no_grad():
            return np.zeros((fn(x0).numel(), 0))
    sel = torch.zeros(len(x0), len(cols), dtype=_F64)
    sel[cols, torch.arange(len(cols))] = 1.0
    base = x0 - sel @ x0[cols]
    return jacfwd(lambda xf: fn(base + sel @ xf))(x0[cols]).detach().numpy()


# ------------------------------------------------------------------------------ problem
class DpnFit:
    """A DPNFIT problem: model, observations, start vector and fit configuration.

    ``states`` 1-based state numbers (DPNFIT's IE: all states in ascending energy, state 1 the
    ground state); ``energies`` observed D(IE) - D(1) in ``model.units``; ``weights`` (default
    1).  Negative energies and weights are set to 0, as DPNFIT does.  ``x0`` the start vector
    (``model.parameter_names()`` order; extra entries are carried along unused); ``free``
    (default all) marks the fitted entries; ``lower``/``upper`` bounds (default unbounded);
    ``step`` typical changes (scipy ``x_scale``; default max(|x0|, 1)).
    """

    def __init__(self, model: DpnFitModel, states: Sequence[int], energies: Sequence[float],
                 weights: Sequence[float] | None = None, *, x0: Sequence[float], free: Sequence[bool] | None = None,
                 lower: Sequence[float] | None = None, upper: Sequence[float] | None = None,
                 step: Sequence[float] | None = None, spngrnd: float = 0.5, lowspn: bool = False,
                 lowspn_policy: str = "penalty"):
        self.model = model
        self.states = np.asarray(states, int)
        nst = model.n_states
        bad = [int(s) for s in self.states if not 1 <= s <= nst]
        if bad:
            raise ValueError(f"state numbers {bad} outside 1..{nst} for p{model.n} (DPNFIT does not check; "
                             "finding DPNFIT-04)")
        self.exp = np.maximum(np.asarray(energies, float), 0.0)
        self.w = np.ones(len(self.states)) if weights is None else np.maximum(np.asarray(weights, float), 0.0)
        if not len(self.exp) == len(self.w) == len(self.states):
            raise ValueError("states, energies and weights must have the same length")
        self.x0 = np.asarray(x0, float)
        nx = len(self.x0)
        self.free = np.ones(nx, bool) if free is None else np.asarray(free, bool)
        self.lower = np.full(nx, -np.inf) if lower is None else np.asarray(lower, float)
        self.upper = np.full(nx, np.inf) if upper is None else np.asarray(upper, float)
        self.step = np.maximum(np.abs(self.x0), 1.0) if step is None else np.abs(np.asarray(step, float))
        self.spngrnd = spin_clamp(model.np_, spngrnd)
        self.lowspn = bool(lowspn) and model.np_ != 1
        if lowspn_policy not in ("penalty", "ignore"):
            raise ValueError("lowspn_policy must be 'penalty' or 'ignore'")
        self.lowspn_policy = lowspn_policy
        self._idx = torch.as_tensor(self.states - 1)
        self._sw = torch.as_tensor(np.sqrt(self.w), dtype=_F64)
        self._exp = torch.as_tensor(self.exp, dtype=_F64)
        self._penalty = LOWSPN_PENALTY_CM * _u.energy_factor("cm^-1", model.units)

    @property
    def n_states(self) -> int:
        return self.model.n_states

    def parameter_names(self) -> list[str]:
        names = self.model.parameter_names()
        nv = len(self.x0)
        return (names + [f"X{i + 1}" for i in range(len(names), nv)])[:nv]

    def n_effective(self) -> int:
        """Rows that carry information: weight > 0 and not state 1 (always 0)."""
        return int(np.sum((self.w > 0) & (self.states != 1)))

    # ------------------------------------------------------------------ evaluation
    def energies(self, x) -> torch.Tensor:
        """All energies relative to the lowest, ascending, in ``units`` (differentiable in x)."""
        e = eigvalsh_gauge(self.model.hamiltonian(x))
        return e - e[..., :1]

    def _spins(self, x) -> tuple[np.ndarray | None, list[str]]:
        with torch.no_grad():
            h = self.model.hamiltonian(torch.as_tensor(np.asarray(x, float), dtype=_F64))
            _, v = torch.linalg.eigh(h)
        npn = self.model.np_ if self.model.mode == "dpn" else self.model.n
        ops = pn_operators(npn)
        P = ops.term_projectors.to(v.dtype)
        tw = torch.einsum("ia,tij,ja->at", v.conj(), P, v).real.numpy()
        terms = ops.terms
        best = len(terms) - 1 - np.argmax(tw[:, ::-1], axis=1)           # ties -> later term (DPNFITCAL l. 378)
        dominant = [terms[b].label for b in best]
        spins = None if self.model.np_ == 1 else np.array([terms[b].S for b in best])
        return spins, dominant

    def _ground_ok(self, x) -> bool:
        if not self.lowspn:
            return True
        spins, _ = self._spins(x)
        return bool(spins[0] == self.spngrnd)

    def residual_function(self, x) -> Callable[[torch.Tensor], torch.Tensor]:
        """The torch residual function of the full X, with the LOWSPN branch decided at ``x``."""
        penalty = self.lowspn and self.lowspn_policy == "penalty" and not self._ground_ok(x)
        fixed = self._sw * self._penalty

        def residuals(xt: torch.Tensor) -> torch.Tensor:
            r = self._sw * (self.energies(xt)[self._idx] - self._exp)
            return fixed + 0.0 * r if penalty else r

        return residuals

    def residuals(self, x) -> torch.Tensor:
        """sqrt(w) (calc - obs) at X (tensor; differentiable unless the LOWSPN penalty is active)."""
        x_np = x.detach().numpy() if isinstance(x, torch.Tensor) else np.asarray(x, float)
        xt = x if isinstance(x, torch.Tensor) else torch.as_tensor(x_np, dtype=_F64)
        return self.residual_function(x_np)(xt)

    def jacobian(self, x, free: Sequence[bool] | None = None) -> np.ndarray:
        """Exact d residuals / d X[free] (n_residuals, n_free) by forward-mode AD."""
        x = np.asarray(x, float)
        cols = np.flatnonzero(np.ones(len(x), bool) if free is None else np.asarray(free, bool))
        return _jacfwd_columns(self.residual_function(x), x, cols)

    def evaluate(self, x) -> Evaluation:
        """Energies, spin labels, residuals and objective at X (NumPy)."""
        x = np.asarray(x, float)
        with torch.no_grad():
            rel = self.energies(torch.as_tensor(x, dtype=_F64)).numpy()
            r = self.residuals(x).numpy()
        calc = rel[self.states - 1]
        rr = np.sqrt(self.w) * (calc - self.exp)
        spins, dominant = self._spins(x)
        ok = True if not self.lowspn else bool(spins[0] == self.spngrnd)
        return Evaluation(x, rel, calc, r, float(rr @ rr), spins, dominant, ok)

    # ------------------------------------------------------------------ fit
    def fit(self, *, ftol: float = 1e-12, xtol: float = 1e-12, gtol: float = 1e-12, max_nfev: int | None = None,
            verbose: int = 0) -> FitResult:
        """Least-squares fit of the free entries of X (scipy ``least_squares``, 'trf', exact Jacobian).

        Starts from ``x0`` clamped strictly inside the bounds.  A trust-region iterate that
        stops just inside a bound is moved onto it when that lowers the objective, and the
        other parameters are then re-optimized with it held there.
        """
        names = self.parameter_names()
        free = self.free.copy()
        full0 = self.x0.copy()
        lo, hi = self.lower[free], self.upper[free]

        def full(xf, mask=free, base=None):
            v = (full0 if base is None else base).copy()
            v[mask] = xf
            return v

        def fun(xf):
            with torch.no_grad():
                return self.residuals(full(xf)).numpy()

        def jac(xf):
            return self.jacobian(full(xf), free)

        r0 = fun(full0[free])
        f0 = float(r0 @ r0)
        nd = self.n_effective()
        if not free.any():
            return FitResult(names, full0, free, f0, f0, r0, nd, np.zeros((r0.size, 0)), np.zeros((0, 0)), 1, 0, True,
                             "all parameters fixed", np.zeros(0, bool), [], None, self.evaluate(full0))
        x0 = full0[free]
        span = np.where(np.isfinite(hi - lo), hi - lo, np.abs(x0) + 1.0)
        eps = 1e-10 * np.maximum(span, 1e-300)
        x0 = np.minimum(np.maximum(x0, lo + eps), hi - eps)
        scale = self.step[free]
        opts = dict(method="trf", x_scale=scale, ftol=ftol, xtol=xtol, gtol=gtol, max_nfev=max_nfev, verbose=verbose)
        res = least_squares(fun, x0, jac=jac, bounds=(lo, hi), **opts)
        nfev, njev = int(res.nfev), int(res.njev or 0)
        xf, r = _snap_to_bounds(fun, res.x, lo, hi, span)
        if r is None:
            r = fun(xf)
        else:
            on = (xf == lo) | (xf == hi)
            if (~on).any():
                fmask = free.copy()
                fmask[np.flatnonzero(free)[on]] = False
                base = full(xf)
                z0 = np.minimum(np.maximum(xf[~on], lo[~on] + eps[~on]), hi[~on] - eps[~on])
                r2 = least_squares(lambda z: fun(full(z, fmask, base)[free]), z0,
                                   jac=lambda z: self.jacobian(full(z, fmask, base), fmask),
                                   bounds=(lo[~on], hi[~on]), **{**opts, "x_scale": scale[~on]})
                nfev += int(r2.nfev)
                njev += int(r2.njev or 0)
                if 2.0 * r2.cost < float(r @ r):
                    xf = xf.copy()
                    xf[~on] = r2.x
                    r = fun(xf)
        xfull = full(xf)
        jmat = self.jacobian(xfull, free)
        fnames = [nm for nm, f in zip(names, free) if f]
        null = null_directions(jmat, fnames)
        cov_u = _inverse(jmat, bool(null))
        at_bound = (np.abs(xf - lo) <= 1e-8 * span) | (np.abs(hi - xf) <= 1e-8 * span)
        out = FitResult(names, xfull, free, float(r @ r), f0, r, nd, jmat, cov_u, nfev, njev, bool(res.success),
                        str(res.message), at_bound, null)
        out.measurements = self.measurement_stats(out)
        out.final = self.evaluate(xfull)
        return out

    # ------------------------------------------------------------------ measurement-level statistics
    def measurement_groups(self) -> list[list[int]]:
        """Row indices grouped into measurements (same energy and weight, consecutive states)."""
        groups: list[list[int]] = []
        for k in range(len(self.states)):
            if groups:
                last = groups[-1][-1]
                if (self.exp[k] == self.exp[last] and self.w[k] == self.w[last]
                        and self.states[k] == self.states[last] + 1):
                    groups[-1].append(k)
                    continue
            groups.append([k])
        return groups

    def measurement_stats(self, result: FitResult, rtol: float = 1e-6) -> MeasurementStats:
        """Uncertainties with each measurement counted once (one representative row per group)."""
        groups = self.measurement_groups()
        jac, r, free = np.asarray(result.jac, float), np.asarray(result.residuals, float), result.free
        scale = float(np.max(np.abs(jac))) if jac.size else 0.0
        rows, res, info = [], [], []
        for g in groups:
            rep = max(g, key=lambda k: float(np.linalg.norm(jac[k])) if jac.size else 0.0)
            ok = self.w[rep] > 0 and jac.size > 0 and float(np.linalg.norm(jac[rep])) > rtol * max(scale, 1e-300)
            info.append(bool(ok))
            if ok:
                rows.append(jac[rep])
                res.append(r[rep])
        nan = np.full(len(result.names), np.nan)
        rd = np.array(res)
        if not rows or not free.any():
            return MeasurementStats(groups, info, len(rows), float(rd @ rd) if rows else 0.0, np.zeros((0, 0)),
                                    nan.copy(), nan.copy())
        jd = np.array(rows)
        cu = _inverse(jd, bool(result.null))
        dof = len(rows) - int(free.sum())
        s2 = float(rd @ rd) / dof if dof > 0 else np.nan
        se, ses = nan.copy(), nan.copy()
        se[free] = np.sqrt(np.clip(np.diag(cu) * s2, 0, None))
        ses[free] = np.sqrt(np.clip(np.diag(cu), 0, None))
        return MeasurementStats(groups, info, len(rows), float(rd @ rd), cu, se, ses)


def _inverse(jac: np.ndarray, singular: bool) -> np.ndarray:
    """(J^T J)^-1, NaN when the data leave a direction undetermined."""
    n = jac.shape[1]
    if singular:
        return np.full((n, n), np.nan)
    try:
        return np.linalg.inv(jac.T @ jac)
    except np.linalg.LinAlgError:
        return np.full((n, n), np.nan)


def _snap_to_bounds(fun, x, lo, hi, span, rel: float = 1e-3):
    """Put parameters within ``rel * span`` of a bound onto it if that lowers the objective.

    Returns (x, residuals), residuals None if x is unchanged.
    """
    near = [(j, lo[j]) for j in range(len(x)) if np.isfinite(lo[j]) and x[j] - lo[j] <= rel * span[j]]
    near += [(j, hi[j]) for j in range(len(x)) if np.isfinite(hi[j]) and hi[j] - x[j] <= rel * span[j]]
    if not near:
        return x, None
    best_r = fun(x)
    best = (float(best_r @ best_r), x, None)
    for trial in [[jb] for jb in near] + ([near] if len(near) > 1 else []):
        xt = x.copy()
        for j, b in trial:
            xt[j] = b
        rt = fun(xt)
        ft = float(rt @ rt)
        if ft < best[0]:
            best = (ft, xt, rt)
    return best[1], best[2]
