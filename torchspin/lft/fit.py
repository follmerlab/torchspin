"""Joint least-squares fits of ONE d^n ligand-field model to several kinds of data, with exact Jacobians.

Telser's programs fit one data type each (DDNFIT: band energies; DDNSUS: chi(T); DDNMAG:
M(B, T)), and EPR g values come from DDN/DDNPOW without fitting.  Here a single parameter
vector goes through one differentiable Hamiltonian into every observable, the residuals of all
data sets (each divided by its own sigma) are minimized together, and the Jacobian is exact.
Different data constrain different parameter combinations (bands: B, Dq, low-symmetry
splittings; susceptibility: zeta, k, zero-field splitting; g: the ground-doublet admixture;
an EPR spectrum: g through the lineshape), so a direction one data type leaves undetermined
(:attr:`JointOutcome.null_directions`) can be fixed by another.

    from torchspin import lft
    from torchspin.lft.fit import FitParameter, JointFit, Bands, Susceptibility, GValues

    params = [FitParameter("B", 900, 500, 1200), FitParameter("Dq", 1200, 500, 2000), ...]
    def model(p):                      # named scalar tensors -> lft.ddn keywords
        return dict(B=p["B"], C=4.2 * p["B"], zeta=p["zeta"], k=p["k"].expand(3),
                    V=lft.crystal_field(p["Dq"], p["Ds"], p["Dt"]))
    fit = JointFit(7, params, model, [Bands([5, 13], [8100, 17500], sigma=50),
                                      Susceptibility(T, chiT, quantity="chiT", chi_units="cgs", sigma=0.02),
                                      GValues([2.1, 2.3, 4.9], sigma=0.01)], units="cm^-1")
    out = fit.fit()
    print(out.summary())

Units: the model's energy keywords, and ``Bands`` energies, are in the fit's ``units`` (default
'MHz'; 'cm^-1' is natural for ligand-field work).  Fields in mT, temperatures in K,
susceptibility in each block's ``chi_units``, magnetization in ``M_units``.

Jacobians: forward-mode AD (``torch.func.jacfwd``) through :mod:`torchspin.lft` for bands, g,
chi and M (cost about one evaluation plus a small amount per parameter).  ``Spectrum`` blocks go
through :func:`torchspin.pepper_autograd.pepper_autograd`, which does not support forward mode;
their Jacobian is the exact chain rule d(spectrum)/d(g) . d(g)/d(x), with d(spectrum)/d(g) for the
six independent g components from a double-backward Jacobian-vector product through pepper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field as _dc_field
from typing import Callable, Sequence

import numpy as np
import torch
from torch.func import jacfwd

from . import units as _u
from ._linalg import eigvalsh_gauge
from ._operators import dn_operators
from .ddn import _g_from, _hamiltonian_cm, zeeman_operators
from .ddnmag import MAG_CONSTANTS, MagConstants, _chi_to_si, _m_factor, moment_core
from .grids import OrientationGrid, gauss_hemisphere

__all__ = ["Bands", "Context", "Dataset", "FitParameter", "FitResult", "GValues", "JointFit", "JointOutcome",
           "Magnetization", "Spectrum", "Susceptibility", "null_directions"]

_F64 = torch.float64


def _arr(x) -> np.ndarray:
    return np.atleast_1d(np.asarray(x, float))


# ------------------------------------------------------------------------------ parameters and results
@dataclass(frozen=True)
class FitParameter:
    """A named fit parameter with start value, bounds, and a fixed flag."""

    name: str
    value: float
    lower: float = -math.inf
    upper: float = math.inf
    fixed: bool = False

    def __post_init__(self):
        if not self.lower <= self.value <= self.upper:
            raise ValueError(f"{self.name}: start value {self.value} outside [{self.lower}, {self.upper}]")


@dataclass
class FitResult:
    """Best-fit values; standard errors and correlations from the exact Jacobian at the solution.

    ``stderr`` = sqrt(diag((J^T J)^+)) for the free parameters (residuals already divided by
    sigma, so these are 1-sigma errors if the sigmas are right; multiply by sqrt(chi2/dof)
    otherwise).  Fixed parameters have stderr 0; undetermined ones NaN.
    """

    names: list[str]
    values: np.ndarray
    stderr: np.ndarray
    correlation: np.ndarray
    free: np.ndarray
    chi2: float
    dof: int
    jacobian: np.ndarray
    success: bool
    message: str
    nfev: int

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.names, self.values.tolist()))

    def table(self) -> str:
        lines = [f"{'parameter':<14}{'value':>14}{'stderr':>12}", "-" * 40]
        for nm, v, e, f in zip(self.names, self.values, self.stderr, self.free):
            lines.append(f"{nm:<14}{v:>14.6g}{(f'{e:.3g}' if f else 'fixed'):>12}")
        lines.append(f"chi^2 = {self.chi2:.6g}, dof = {self.dof}, {'converged' if self.success else 'NOT converged'}")
        return "\n".join(lines)


def null_directions(result: FitResult, rtol: float = 1e-6, min_coef: float = 0.05) -> list[dict[str, float]]:
    """Parameter combinations the data do not determine (free parameters only).

    Right singular vectors of the column-scaled Jacobian with singular value below
    ``rtol`` x the largest; coefficients are for unit-scaled parameters (|x_j| or 1).
    """
    J = result.jacobian
    if J.size == 0:
        return []
    names = [n for n, f in zip(result.names, result.free) if f]
    scale = np.maximum(np.abs(result.values[result.free]), 1.0)
    _, s, vt = np.linalg.svd(J * scale, full_matrices=True)
    s_full = np.concatenate([s, np.zeros(vt.shape[0] - len(s))])
    out = []
    for sv, v in zip(s_full, vt):
        if sv <= rtol * max(s_full.max(), 1e-300):
            v = v / v[np.argmax(np.abs(v))]
            out.append({n: float(c) for n, c in zip(names, v) if abs(c) >= min_coef})
    return out


# ------------------------------------------------------------------------------ evaluation context
class Context:
    """Quantities shared by the data sets at one parameter vector (computed once, on demand).

    ``H0`` and ``energies`` are in cm^-1 (internal); ``f`` converts the fit's units to cm^-1.
    """

    def __init__(self, ops, kwargs: dict, units: str, constants: MagConstants):
        kw = dict(kwargs)
        self.k = kw.pop("k", (1.0, 1.0, 1.0))
        if "field" in kw:
            raise ValueError("the model must not return 'field'")
        self.ops, self.constants, self.units = ops, constants, units
        self.f = _u.energy_factor(units, "cm^-1")
        self.H0 = _hamiltonian_cm(ops, self.f, kw.pop("B", 0.0), kw.pop("C", 0.0), kw.pop("A", 0.0),
                                  kw.pop("zeta", 0.0), kw.pop("V", None), self.k, kw.pop("alpha", 0.0),
                                  kw.pop("slater", None), None, constants.ddn)
        if kw:
            raise TypeError(f"model returned unexpected keywords: {sorted(kw)}")
        self._mu = self._E = self._g = None

    @property
    def mu(self) -> torch.Tensor:
        if self._mu is None:
            self._mu = zeeman_operators(self.ops, self.k, self.constants.ddn)
        return self._mu

    @property
    def energies(self) -> torch.Tensor:
        if self._E is None:
            self._E = eigvalsh_gauge(self.H0)
        return self._E

    @property
    def g(self):
        """(ggT, principal, g, gap_cm) of the ground doublet."""
        if self._g is None:
            self._g = _g_from(self.H0, self.mu)
        return self._g


# ------------------------------------------------------------------------------ data sets
class Dataset:
    """A block of observations with its forward model; residuals are (calc - obs) / sigma."""

    name: str = ""
    forward_ad: bool = True        # False: Jacobian supplied by :meth:`jacobian_block`

    def calculate(self, ctx: Context) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def observed(self) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def sigma(self) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def residuals(self, ctx: Context) -> torch.Tensor:
        obs = torch.as_tensor(np.array(self.observed), dtype=_F64)
        return (self.calculate(ctx) - obs) / torch.as_tensor(np.array(self.sigma), dtype=_F64)


def _bcast(s, shape):
    return np.broadcast_to(_arr(s), shape).copy()


@dataclass
class Bands(Dataset):
    """Band energies (in the fit's ``units``) of states numbered as DDNFIT does: 1-based over ALL
    states in ascending order, relative to state 1.  A degenerate level is fitted through its mean."""

    states: Sequence[int]
    energies: Sequence[float]
    sigma_: float | Sequence[float] = 1.0
    name: str = "bands"

    def __post_init__(self):
        self._idx = torch.as_tensor(np.asarray(self.states, int) - 1)
        if torch.any(self._idx < 0):
            raise ValueError("state numbers are 1-based")

    @property
    def observed(self):
        return _arr(self.energies)

    @property
    def sigma(self):
        return _bcast(self.sigma_, self.observed.shape)

    def calculate(self, ctx):
        e = ctx.energies
        return (e - e[0])[self._idx] / ctx.f


@dataclass
class GValues(Dataset):
    """Principal g values of the ground doublet (effective S = 1/2), compared sorted ascending."""

    g: Sequence[float]
    sigma_: float | Sequence[float] = 0.01
    name: str = "g"

    @property
    def observed(self):
        return np.sort(_arr(self.g))

    @property
    def sigma(self):
        return _bcast(self.sigma_, (3,))[np.argsort(_arr(self.g))]

    def calculate(self, ctx):
        return ctx.g[1]


@dataclass
class Susceptibility(Dataset):
    """Powder chi = M/H (``quantity='chi'``) or chi T (``'chiT'``) at measuring ``field`` (mT), in ``chi_units``."""

    T: Sequence[float]
    values: Sequence[float]
    sigma_: float | Sequence[float] = 0.01
    field: float = 100.0
    quantity: str = "chiT"
    chi_units: str = "cgs"
    grid: OrientationGrid = _dc_field(default_factory=lambda: gauss_hemisphere(6))
    chi0: float = 0.0
    name: str = "susceptibility"

    def __post_init__(self):
        if self.quantity not in ("chi", "chiT"):
            raise ValueError("quantity must be 'chi' or 'chiT'")
        if self.field <= 0:
            raise ValueError("chi = M/H needs field > 0")

    @property
    def observed(self):
        return _arr(self.values)

    @property
    def sigma(self):
        return _bcast(self.sigma_, self.observed.shape)

    def calculate(self, ctx):
        c = ctx.constants
        T = torch.as_tensor(_arr(self.T), dtype=_F64)
        m = moment_core(ctx.H0, ctx.mu, [self.field], T, self.grid.vectors, c)            # (ndir, 1, nT) muB/ion
        m = torch.einsum("d,dt->t", torch.as_tensor(self.grid.weights, dtype=_F64), m[:, 0])
        chi = m * (c.N_A * c.mu_b_SI) * c.mu0 / (self.field * 1e-3) / _chi_to_si(self.chi_units, c) + self.chi0
        return chi * T if self.quantity == "chiT" else chi


@dataclass
class Magnetization(Dataset):
    """Powder M at points (field_i in mT, T_i in K), in ``M_units`` ('muB' per ion, 'emu/mol', 'SI')."""

    field: Sequence[float]
    T: Sequence[float]
    values: Sequence[float]
    sigma_: float | Sequence[float] = 0.01
    M_units: str = "muB"
    grid: OrientationGrid = _dc_field(default_factory=lambda: gauss_hemisphere(6))
    name: str = "magnetization"

    def __post_init__(self):
        B, T = np.broadcast_arrays(_arr(self.field), _arr(self.T))
        self._B, self._T = np.unique(B), np.unique(T)
        self._ib = torch.as_tensor(np.searchsorted(self._B, B))
        self._it = torch.as_tensor(np.searchsorted(self._T, T))

    @property
    def observed(self):
        return _arr(self.values)

    @property
    def sigma(self):
        return _bcast(self.sigma_, self.observed.shape)

    def calculate(self, ctx):
        m = moment_core(ctx.H0, ctx.mu, self._B, self._T, self.grid.vectors, ctx.constants)
        m = torch.einsum("d,dbt->bt", torch.as_tensor(self.grid.weights, dtype=_F64), m)[self._ib, self._it]
        return m * _m_factor(self.M_units, ctx.constants)


_SYM = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]


@dataclass
class Spectrum(Dataset):
    """A CW EPR powder spectrum simulated by ``pepper_autograd`` from the ground-doublet g.

    ``experiment``, ``options``: torchspin :class:`Experiment`, :class:`Options` (matrix method).
    ``system``: extra :class:`SpinSystem` keywords (``Nucs``, ``A`` in MHz, ``lw``, ...); ``S``
    and ``g`` are set from the model.  ``scale='auto'`` fits the amplitude analytically
    (least-squares projection; differentiated exactly), or give a fixed float.  Only g carries
    the model parameters into the spectrum.
    """

    experiment: object
    options: object
    spectrum: Sequence[float]
    sigma_: float | Sequence[float] = 1.0
    system: dict = _dc_field(default_factory=dict)
    scale: float | str = "auto"
    name: str = "spectrum"
    forward_ad = False

    @property
    def observed(self):
        return _arr(self.spectrum)

    @property
    def sigma(self):
        return _bcast(self.sigma_, self.observed.shape)

    def _sim(self, g: torch.Tensor) -> torch.Tensor:
        from torchspin.pepper_autograd import pepper_autograd
        from torchspin.spinsystem import SpinSystem

        if "S" in self.system or "g" in self.system:
            raise ValueError("Spectrum.system must not set S or g")
        return pepper_autograd(SpinSystem(S=0.5, g=g, **self.system), self.experiment, self.options)[1].reshape(-1)

    def _scaled(self, c: torch.Tensor):
        o = torch.as_tensor(self.observed, dtype=_F64)
        if self.scale == "auto":
            return (c @ o) / (c @ c), o
        return torch.as_tensor(float(self.scale), dtype=_F64), o

    def calculate(self, ctx):
        c = self._sim(ctx.g[2])
        s, _ = self._scaled(c)
        return s * c

    def jacobian_block(self, g: torch.Tensor, dg_dx: torch.Tensor) -> torch.Tensor:
        """d residuals / d x = sum_k d r / d g_k . d g_k / d x over the 6 symmetric components.

        ``g`` (3, 3) detached; ``dg_dx`` (3, 3, nfree).
        """
        gv = g.detach().clone().requires_grad_()
        c = self._sim(gv)
        u = torch.zeros_like(c, requires_grad=True)
        (v,) = torch.autograd.grad(c, gv, grad_outputs=u, create_graph=True)
        cols, coef = [], []
        for i, j in _SYM:
            d = torch.zeros(3, 3, dtype=_F64)
            d[i, j] = d[j, i] = 1.0
            (jc,) = torch.autograd.grad(v, u, grad_outputs=d, retain_graph=True)
            cols.append(jc.detach())
            coef.append(dg_dx[i, j])                    # g symmetric: the (i, j) and (j, i) entries move together
        Jc = torch.stack(cols, -1)                       # (npts, 6) = d c / d g_k
        c = c.detach()
        s, o = self._scaled(c)
        if self.scale == "auto":
            cc = c @ c
            ds = (Jc.T @ o - 2.0 * s * (Jc.T @ c)) / cc  # (6,)
            Jr = s * Jc + c[:, None] * ds[None, :]
        else:
            Jr = s * Jc
        Jr = Jr / torch.as_tensor(self.sigma, dtype=_F64)[:, None]
        return Jr @ torch.stack(coef, 0)                 # (npts, nfree)


# ------------------------------------------------------------------------------ the fit
@dataclass
class JointOutcome:
    """A joint fit: the shared :class:`FitResult`, the chi^2 of every data set, and undetermined directions."""

    result: FitResult
    chi2: dict[str, float]
    n_points: dict[str, int]
    null_directions: list[dict[str, float]]

    def summary(self) -> str:
        lines = [self.result.table(), "data set            points      chi^2   chi^2/point"]
        for k, c in self.chi2.items():
            n = self.n_points[k]
            lines.append(f"{k:<18}{n:>8}{c:>12.4g}{c / max(n, 1):>13.4g}")
        if self.null_directions:
            lines.append("undetermined parameter combinations: " + "; ".join(
                " + ".join(f"{v:.3g}*{k}" for k, v in d.items()) for d in self.null_directions))
        return "\n".join(lines)


class JointFit:
    """Fit one d^n model to several data sets.

    ``parameters``: :class:`FitParameter` list.  ``model(p)``: maps the dict of named scalar
    tensors to :func:`torchspin.lft.ddn.hamiltonian` keywords (``B, C, A, zeta, V, k, alpha,
    slater``) in ``units``.  ``constants``: :class:`torchspin.lft.ddnmag.MagConstants`.
    """

    def __init__(self, n: int, parameters: Sequence[FitParameter], model: Callable[[dict], dict],
                 datasets: Sequence[Dataset], units: str = "MHz", constants: MagConstants = MAG_CONSTANTS,
                 device="cpu"):
        self.n = int(n)
        self.parameters = list(parameters)
        self.names = [p.name for p in self.parameters]
        if len(set(self.names)) != len(self.names):
            raise ValueError("parameter names must be unique")
        self.model = model
        self.datasets = list(datasets)
        if len({d.name for d in self.datasets}) != len(self.datasets):
            for i, d in enumerate(self.datasets):
                d.name = f"{d.name}{i + 1}"
        self.units = _u.canonical_energy_unit(units)
        self.constants = constants
        self.ops = dn_operators(self.n, device)

    # ------------------------------------------------------------------ evaluation
    def x0(self) -> np.ndarray:
        return np.array([p.value for p in self.parameters], float)

    def free_mask(self) -> np.ndarray:
        return np.array([not p.fixed for p in self.parameters])

    def context(self, x: torch.Tensor) -> Context:
        return Context(self.ops, self.model({nm: x[i] for i, nm in enumerate(self.names)}), self.units,
                       self.constants)

    def residuals_torch(self, x: torch.Tensor, datasets=None) -> torch.Tensor:
        ctx = self.context(x)
        ds = self.datasets if datasets is None else datasets
        return torch.cat([d.residuals(ctx) for d in ds]) if ds else torch.zeros(0, dtype=_F64)

    def residuals(self, x) -> np.ndarray:
        with torch.no_grad():
            return self.residuals_torch(torch.as_tensor(np.asarray(x, float), dtype=_F64)).numpy()

    def calculate(self, x) -> dict[str, np.ndarray]:
        """Calculated values of every data set at the full parameter vector x."""
        with torch.no_grad():
            ctx = self.context(torch.as_tensor(np.asarray(x, float), dtype=_F64))
            return {d.name: d.calculate(ctx).numpy() for d in self.datasets}

    def jacobian(self, x, free=None) -> np.ndarray:
        """Exact d residuals / d x[free]: forward-mode AD, chain rule through pepper for Spectrum blocks."""
        x0 = torch.as_tensor(np.asarray(x, float), dtype=_F64)
        cols = torch.as_tensor(np.flatnonzero(self.free_mask() if free is None else free))
        nf = len(cols)
        sel = torch.zeros(len(x0), nf, dtype=_F64)
        sel[cols, torch.arange(nf)] = 1.0
        base = x0 - sel @ x0[cols]
        full = lambda xf: base + sel @ xf  # noqa: E731
        fwd = [d for d in self.datasets if d.forward_ad]
        rev = [d for d in self.datasets if not d.forward_ad]
        blocks = {}
        if fwd and nf:
            Jf = jacfwd(lambda xf: self.residuals_torch(full(xf), fwd))(x0[cols])
            i = 0
            for d in fwd:
                k = len(d.observed)
                blocks[d.name] = Jf[i:i + k]
                i += k
        if rev and nf:
            g_of = lambda xf: self.context(full(xf)).g[2]  # noqa: E731
            dg = jacfwd(g_of)(x0[cols])                                     # (3, 3, nf)
            with torch.no_grad():
                g = g_of(x0[cols])
            for d in rev:
                blocks[d.name] = d.jacobian_block(g, dg)
        if not nf:
            return np.zeros((sum(len(d.observed) for d in self.datasets), 0))
        return torch.cat([blocks[d.name] for d in self.datasets]).detach().numpy()

    # ------------------------------------------------------------------ fit
    def fit(self, datasets: Sequence[str] | None = None, **kw) -> JointOutcome:
        """Fit all data sets (or only those named); ``kw`` go to :func:`scipy.optimize.least_squares`."""
        from scipy.optimize import least_squares

        sub = self if datasets is None else self.subset(datasets)
        x_full = sub.x0()
        free = sub.free_mask()
        lo = np.array([p.lower for p in sub.parameters])[free]
        hi = np.array([p.upper for p in sub.parameters])[free]

        def expand(xf):
            x = x_full.copy()
            x[free] = xf
            return x

        opts = dict(method="trf", x_scale="jac", xtol=1e-12, ftol=1e-12, gtol=1e-12)
        opts.update(kw)
        if free.any():
            sol = least_squares(lambda xf: sub.residuals(expand(xf)), x_full[free],
                                jac=lambda xf: sub.jacobian(expand(xf), free), bounds=(lo, hi), **opts)
            xbest, success, message, nfev = expand(sol.x), bool(sol.success), str(sol.message), int(sol.nfev)
        else:
            xbest, success, message, nfev = x_full, True, "no free parameters", 0
        res = sub.residuals(xbest)
        J = sub.jacobian(xbest, free) if free.any() else np.zeros((len(res), 0))
        cov = np.linalg.pinv(J.T @ J) if J.size else np.zeros((0, 0))
        err = np.zeros(len(xbest))
        with np.errstate(invalid="ignore"):
            d = np.sqrt(np.clip(np.diag(cov), 0, None)) if J.size else np.zeros(0)
            err[free] = d
            corr = cov / np.outer(d, d) if J.size else cov
        result = FitResult(sub.names, xbest, err, corr, free, float(res @ res), len(res) - int(free.sum()), J,
                           success, message, nfev)
        nulls = null_directions(result)
        for nd in nulls:                                            # undetermined: stderr is meaningless
            for nm in nd:
                result.stderr[sub.names.index(nm)] = np.nan
        chi2, npts, i = {}, {}, 0
        for ds in sub.datasets:
            k = len(ds.observed)
            chi2[ds.name] = float(res[i:i + k] @ res[i:i + k])
            npts[ds.name] = k
            i += k
        return JointOutcome(result, chi2, npts, nulls)

    def subset(self, names: Sequence[str]) -> "JointFit":
        """The same model fitted to the named data sets only (for identifiability comparisons)."""
        missing = set(names) - {d.name for d in self.datasets}
        if missing:
            raise ValueError(f"unknown data sets {sorted(missing)}")
        out = JointFit.__new__(JointFit)
        out.__dict__.update(self.__dict__)
        out.datasets = [d for d in self.datasets if d.name in names]
        return out

    def with_start(self, x) -> "JointFit":
        """Copy with the parameter start values replaced by x (full vector)."""
        out = JointFit.__new__(JointFit)
        out.__dict__.update(self.__dict__)
        out.parameters = [FitParameter(p.name, float(v), p.lower, p.upper, p.fixed) for p, v in zip(self.parameters, x)]
        return out
