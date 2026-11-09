"""DSOXF: one free-ion term (2S+1)L of d^n or f^n in a crystal field, differentiable.

    H = H_CF + s |lambda| L.S + rho SS + mu_B B . (s_L k L + g_e S)

in DSOXF's |m_L m_S> basis (m_L outer, both descending; Telser's DSOXF via telser-tools), for
the d and f shells, high and low spin (low-spin d4/d5 as the fictitious L~ = 1 term), Oh/Td,
cubic/axial/rhombic and tetragonal/trigonal fields, with both ``conventions``
(``'corrected'``, the default, or ``'dsoxf'`` = the compiled program; see
:mod:`torchspin.lft._engines.dsoxf`).  SS = -[(L.S)^2 + L.S/2 - L(L+1)S(S+1)/3] is the
Abragam-Bleaney spin-spin term (zero for S < 1).

For a fixed configuration and discrete setting, H_CF is linear in the crystal-field parameters
(``CF_PARAMETERS``: dq, ds, dt4 (= Delta4 = -Dt), v2, v4 for d; b4, b6, b20 ... b66 for f, as
DSOXF reads them).  The matrices are probed once from the NumPy model
(:class:`torchspin.lft._engines.dsoxf.DsoxfModel`) and checked for linearity, so the Stevens
operators, DSOXF's scalings (B2X, B4X, F_kq), the symmetry selection and the whole sign table
keep one definition.  ``lam`` is a magnitude: lambda_eff = s |lambda| with the sign s that the
program chooses (d|lambda|/dlambda := +1 at 0).

Units
-----
Energy-valued inputs (``lam``, ``rho`` and every crystal-field parameter) and energy-valued
outputs (H, energies, doublet energies) are in ``units`` (default ``'MHz'``; ``'cm^-1'``,
``'GHz'``, ``'K'``, ``'eV'``).  The magnetic field is in mT, temperatures in K.  ``k``, g values
and occupancies are dimensionless; transition moments |<i|mu.b1|j>|^2 are in mu_B^2.  Level
tolerances (``tol``) are in cm^-1 whatever ``units`` is.  Internally everything is cm^-1.

    >>> from torchspin.lft import dsoxf
    >>> s = dsoxf.DsoxfSystem("d", 9, symmetry="axial", units="cm^-1", lam=830.0, dq=1200.0, ds=2000.0)
    >>> sol = s.solve(field=(0.0, 0.0, 300.0), ds=torch.tensor(2000.0, requires_grad=True))   # mT
    >>> sol.energies, sol.spin_expectation(), sol.orbital_occupancies(), s.solve().kramers_g()

Gauge: per-state quantities (<S>, occupancies, |<i|mu|j>|^2) depend on the arbitrary basis
inside a degenerate level, exactly as in DSOXF, and have no derivative there; their sums over
the level (``level_matrix(E) @ x``, :func:`level_sum`, ``Solution.level_transition_moments``)
are gauge-invariant and differentiable.  Kramers-doublet g tensors use the doublet projector.

Not ported: DSOXF's legacy EPR probabilities ((sum_a |<i|S_a|j>|)^2, a sum of moduli that
depends on eigenvector phases and is not smooth; telser-tools finding DSOXF-05), the listing
and spin-magnitude cutoffs, the input-file reader and the printed report.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple, Sequence

import numpy as np
import torch

from . import units as _u
from ._engines.dsoxf import D_ORBITALS, F_CUBIC_ORBITALS, F_GENERAL_ORBITALS, DsoxfModel, orbital_vectors
from ._linalg import eigh_gauge, eigvalsh_gauge, sqrtm_psd
from .ddn import TORCHSPIN_CONSTANTS, Constants

__all__ = ["CF_PARAMETERS", "D_ORBITALS", "F_CUBIC_ORBITALS", "F_GENERAL_ORBITALS", "LEVEL_TOL", "DoubletG",
           "DsoxfOperators", "DsoxfSystem", "Solution", "doublet_g", "dsoxf_operators", "hamiltonian", "level_matrix",
           "level_sum", "levels", "ml_occupancies", "orbital_occupancies", "solve", "spin_expectation",
           "transition_moments", "zeeman_operators"]

_F64, _C128 = torch.float64, torch.complex128
CF_PARAMETERS = ("dq", "ds", "dt4", "v2", "v4", "b4", "b6", "b20", "b40", "b60", "b44", "b64", "b43", "b63", "b66",
                 "b22", "b42", "b62")
_DISCRETE = ("shell", "code", "geometry", "symmetry", "axis", "conventions")
#: DSOXF's level criterion for Kramers doublets and level sums (cm^-1, chained).
LEVEL_TOL = 1e-6


def _t(x, device=None) -> torch.Tensor:
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=_F64, device=device)


# ------------------------------------------------------------------------------ operators
@dataclass(frozen=True, eq=False)
class DsoxfOperators:
    """Constants of one DSOXF configuration and discrete setting (unit-free; energies cm^-1).

    ``cf`` (18, n, n) float64 in ``CF_PARAMETERS`` order: H_CF = sum_p value_p cf[p];
    ``LS``, ``SS`` (n, n) float64 (SS: the rho operator, zero for S < 1); ``Lv``, ``Sv`` (3, n, n)
    complex128; ``so_sign`` = the s of lambda_eff = s |lambda|; ``zeeman_sign`` = s_L
    (-1 for the corrected L~ = 1 terms); ``L``, ``S`` quantum numbers; ``nl``, ``ns`` the orbital
    and spin dimensions; ``term`` e.g. ``'2D'``.
    """

    shell: str
    code: int
    term: str
    L: int
    S: float
    nl: int
    ns: int
    cf: torch.Tensor
    LS: torch.Tensor
    SS: torch.Tensor
    Lv: torch.Tensor
    Sv: torch.Tensor
    so_sign: float
    zeeman_sign: float

    @property
    def size(self) -> int:
        return self.nl * self.ns

    @property
    def device(self) -> torch.device:
        return self.LS.device


@lru_cache(maxsize=None)
def _operators(key: tuple, device: str) -> DsoxfOperators:
    model = DsoxfModel(**dict(zip(_DISCRETE, key)))
    z = model.with_(lam=0.0, rho=0.0, **{p: 0.0 for p in CF_PARAMETERS})
    b = model.basis
    if np.any(z.crystal_field != 0):
        raise AssertionError("crystal field has a constant part")
    cf = np.array([z.with_(**{p: 1.0}).crystal_field for p in CF_PARAMETERS])
    rng = np.random.default_rng(0)
    for _ in range(2):
        x = rng.normal(size=len(CF_PARAMETERS)) * 500.0
        want = z.with_(**dict(zip(CF_PARAMETERS, x.tolist()))).crystal_field
        if np.abs(want - np.tensordot(x, cf, 1)).max() > 1e-9 * max(1.0, np.abs(want).max()):
            raise AssertionError("DSOXF crystal field is not linear in its parameters")
    so_sign = z.with_(lam=1.0).spin_orbit
    if z.with_(lam=-2.0).spin_orbit != 2.0 * so_sign:
        raise AssertionError("spin-orbit coefficient is not s |lambda|")
    f64 = lambda a: torch.as_tensor(np.array(a), dtype=_F64, device=device)  # noqa: E731
    c128 = lambda a: torch.as_tensor(np.array(a), dtype=_C128, device=device)  # noqa: E731
    c = model.config
    return DsoxfOperators(shell=c.shell, code=c.code, term=c.term, L=b.L, S=b.S, nl=b.nl, ns=b.ns, cf=f64(cf),
                          LS=f64(b.LS), SS=f64(z.spin_spin), Lv=c128(b.Lv), Sv=c128(b.Sv), so_sign=float(so_sign),
                          zeeman_sign=float(z.orbital_zeeman_sign))


def dsoxf_operators(shell: str = "d", code: int = 1, *, geometry: str = "Oh", symmetry: str = "cubic",
                    axis: str = "tetragonal", conventions: str = "corrected",
                    device: str | torch.device = "cpu") -> DsoxfOperators:
    """Cached constants of one configuration and discrete setting.

    ``shell`` 'd' or 'f'; ``code`` DSOXF's configuration number (d: 1 ... 9 high spin, -4 ... -7 low
    spin; f: 1 ... 13); ``geometry`` 'Oh'/'Td'; ``symmetry`` 'cubic'/'axial'/'rhombic' and ``axis``
    'tetragonal'/'trigonal' select which crystal-field parameters act, as DSOXF reads them;
    ``conventions`` 'corrected' (default) or 'dsoxf'.
    """
    model = DsoxfModel(shell=shell, code=int(code), geometry=geometry, symmetry=symmetry, axis=axis,
                       conventions=conventions)
    return _operators(tuple(getattr(model, a) for a in _DISCRETE), str(torch.device(device)))


def zeeman_operators(ops: DsoxfOperators, k=(1.0, 1.0, 1.0),
                     constants: Constants = TORCHSPIN_CONSTANTS) -> torch.Tensor:
    """(..., 3, n, n): mu_i / mu_B = s_L k_i L_i + g_e S_i (k dimensionless, (..., 3))."""
    k = _t(k, ops.device) * ops.zeeman_sign
    return k[..., :, None, None].to(_C128) * ops.Lv + constants.g_e * ops.Sv


def _abs1(x: torch.Tensor) -> torch.Tensor:
    """|x| with derivative +1 at 0."""
    return x * torch.where(x.detach() < 0, -1.0, 1.0).to(x.dtype)


def _hamiltonian_cm(ops: DsoxfOperators, f: float, lam, rho, k, field, constants: Constants, cf: dict) -> torch.Tensor:
    """H in cm^-1; energy inputs are multiplied by f (their unit -> cm^-1)."""
    bad = set(cf) - set(CF_PARAMETERS)
    if bad:
        raise TypeError(f"unknown crystal-field parameters {sorted(bad)}; use {', '.join(CF_PARAMETERS)}")
    dev = ops.device
    h = torch.zeros(ops.size, ops.size, dtype=_F64, device=dev)
    if cf:
        names = [p for p in CF_PARAMETERS if p in cf]
        vals = torch.stack(torch.broadcast_tensors(*[_t(cf[p], dev) for p in names]), -1)
        h = h + f * torch.einsum("...p,pij->...ij", vals, ops.cf[[CF_PARAMETERS.index(p) for p in names]])
    lam_eff = ops.so_sign * _abs1(_t(lam, dev))
    h = h + f * (lam_eff[..., None, None] * ops.LS + _t(rho, dev)[..., None, None] * ops.SS)
    h = h.to(_C128)
    if field is not None:
        mu = zeeman_operators(ops, k, constants)
        h = h + constants.mu_b * torch.einsum("...i,...inm->...nm", _t(field, dev).to(_C128), mu)
    return h


def hamiltonian(ops: DsoxfOperators, *, lam=0.0, rho=0.0, k=(1.0, 1.0, 1.0), field=None, units: str = "MHz",
                constants: Constants = TORCHSPIN_CONSTANTS, **cf) -> torch.Tensor:
    """Complex Hermitian H (..., n, n) in ``units``.

    ``lam`` (the magnitude |lambda|), ``rho`` and ``cf`` (any of ``CF_PARAMETERS``, default 0) in
    ``units``; ``k`` (..., 3) dimensionless; ``field`` (..., 3) in mT.  Every value may be a tensor
    with leading batch dimensions and gradients.
    """
    f = _u.energy_factor(units, "cm^-1")
    return _hamiltonian_cm(ops, f, lam, rho, k, field, constants, cf) / f


def levels(ops: DsoxfOperators, *, relative: bool = False, units: str = "MHz", **kw) -> torch.Tensor:
    """Ascending energies (..., n) in ``units``; ``relative=True`` puts the ground state at 0.

    Keywords as :func:`hamiltonian`.  A degenerate eigenvalue gets its level's mean derivative.
    """
    E = eigvalsh_gauge(hamiltonian(ops, units=units, **kw))
    return E - E[..., :1] if relative else E


# ------------------------------------------------------------------------------ derived quantities
def _expect(V: torch.Tensor, op: torch.Tensor) -> torch.Tensor:
    """Re <i|op_k|i> for op (..., k, n, n) -> (..., n, k)."""
    return torch.einsum("...ai,...kab,...bi->...ik", V.conj(), op.to(V.dtype), V).real


def spin_expectation(V: torch.Tensor, ops: DsoxfOperators) -> torch.Tensor:
    """(..., n, 3): <S_x>, <S_y>, <S_z> of every eigenvector (columns of V)."""
    return _expect(V, ops.Sv)


def ml_occupancies(V: torch.Tensor, ops: DsoxfOperators) -> torch.Tensor:
    """(..., n, 2L+1): sum over m_S of |c(m_L, m_S)|^2, m_L = L ... -L."""
    p = (V.conj() * V).real
    return p.reshape(*p.shape[:-2], ops.nl, ops.ns, p.shape[-1]).sum(-2).transpose(-2, -1)


def orbital_occupancies(V: torch.Tensor, ops: DsoxfOperators, kind: str = "real") -> torch.Tensor:
    """(..., n, 2L+1) real-orbital weights; each row sums to 1.

    L = 2: columns ``D_ORBITALS`` (``kind`` ignored).  L = 3: ``kind`` 'general'
    (``F_GENERAL_ORBITALS``) or 'cubic' (``F_CUBIC_ORBITALS``).
    """
    _, u = orbital_vectors(ops.L, kind)
    u = torch.as_tensor(u, dtype=_C128, device=V.device)
    c = V.reshape(*V.shape[:-2], ops.nl, ops.ns, V.shape[-1])
    proj = torch.einsum("mo,...msi->...osi", u.conj(), c)
    return (proj.conj() * proj).real.sum(-2).transpose(-2, -1)


def transition_moments(V: torch.Tensor, mu: torch.Tensor, b1) -> torch.Tensor:
    """(..., n, n) |<i|mu . b1|j>|^2 in mu_B^2, b1 (..., 3) the microwave-field direction."""
    op = torch.einsum("...k,...kab->...ab", _t(b1, V.device).to(_C128), mu)
    m = V.conj().transpose(-2, -1) @ op @ V
    return (m.conj() * m).real


def level_matrix(E: torch.Tensor, tol: float = LEVEL_TOL, units: str = "MHz") -> torch.Tensor:
    """(..., n, n) float: 1 where states i, j are in the same level (chained |E_i+1 - E_i| <= tol).

    ``E`` ascending in ``units``; ``tol`` in cm^-1.  Decided on detached energies;
    ``level_matrix(E) @ x`` sums a per-state quantity over each level (gauge-invariant), and
    ``D @ M @ D`` sums a pair quantity over both levels.
    """
    e = E.detach() * _u.energy_factor(units, "cm^-1")
    step = torch.cat([torch.zeros_like(e[..., :1], dtype=torch.long), (e[..., 1:] - e[..., :-1] > tol).long()], -1)
    lab = step.cumsum(-1)
    return (lab.unsqueeze(-1) == lab.unsqueeze(-2)).to(E.dtype)


def level_sum(E: torch.Tensor, x: torch.Tensor, tol: float = LEVEL_TOL, units: str = "MHz") -> torch.Tensor:
    """Per-state x (..., n) or (..., n, m) summed over each state's degenerate level (same shape)."""
    D = level_matrix(E, tol, units)
    return (D @ x.unsqueeze(-1)).squeeze(-1) if x.dim() == E.dim() else D @ x


class DoubletG(NamedTuple):
    """g of one doublet: ``energy`` (mean, in the solution's units), ``principal`` (ascending |g|),
    ``ggT`` = g g^T, ``g`` = (g g^T)^(1/2) (symmetric, exact gradients at axial g)."""

    energy: torch.Tensor
    principal: torch.Tensor
    ggT: torch.Tensor
    g: torch.Tensor


def doublet_g(V: torch.Tensor, mu: torch.Tensor, idx: Sequence[int], E: torch.Tensor | None = None) -> DoubletG:
    """g of the doublet spanned by eigenvectors ``idx`` (columns of V):
    (g g^T)_ij = 2 tr(P mu_i P mu_j), P the doublet projector (gauge-invariant).  Use at zero field."""
    Vd = V[..., :, list(idx)]
    P = Vd @ Vd.conj().transpose(-2, -1)
    pm = P.unsqueeze(-3) @ mu
    gg = 2.0 * torch.einsum("...inm,...jmn->...ij", pm, pm).real
    gg = 0.5 * (gg + gg.transpose(-2, -1))
    principal = torch.sqrt(torch.clamp(eigvalsh_gauge(gg), min=0.0))
    e = E[..., list(idx)].mean(-1) if E is not None else torch.zeros((), dtype=_F64, device=V.device)
    return DoubletG(e, principal, gg, sqrtm_psd(gg))


# ------------------------------------------------------------------------------ solution
class Solution:
    """Eigenstates of one DSOXF Hamiltonian (the torch counterpart of DSOXF's solution).

    ``energies`` (n,) ascending in ``units``; ``vectors`` (n, n) eigenvector columns in the
    |m_L m_S> basis; ``mu`` (3, n, n) the magnetic moment operator in mu_B; ``field`` in mT or None.
    """

    def __init__(self, ops: DsoxfOperators, field, energies: torch.Tensor, vectors: torch.Tensor, mu: torch.Tensor,
                 units: str, constants: Constants):
        self.ops, self.field, self.energies, self.vectors, self.mu = ops, field, energies, vectors, mu
        self.units, self.constants = _u.canonical_energy_unit(units), constants
        self._f = _u.energy_factor(units, "cm^-1")

    @property
    def relative(self) -> torch.Tensor:
        """Energies relative to the ground state, in ``units``."""
        return self.energies - self.energies[..., :1]

    def levels(self, tol: float = 1e-5) -> list[tuple[torch.Tensor, list[int]]]:
        """[(energy, [state indices])], DSOXF's chained criterion (``tol`` in cm^-1) on detached energies."""
        e = self.energies.detach() * self._f
        out: list[tuple[torch.Tensor, list[int]]] = []
        for i in range(len(e)):
            if out and float(e[i] - e[out[-1][1][-1]]) <= tol:
                out[-1][1].append(i)
            else:
                out.append((self.energies[i], [i]))
        return out

    def level_sum(self, x: torch.Tensor, tol: float = LEVEL_TOL) -> torch.Tensor:
        return level_sum(self.energies, x, tol, self.units)

    def spin_expectation(self) -> torch.Tensor:
        """(n, 3) <S> per state."""
        return spin_expectation(self.vectors, self.ops)

    def _unit_field(self) -> torch.Tensor:
        f = self.field
        if f is None or not torch.any(_t(f) != 0):
            return torch.tensor([0.0, 0.0, 1.0], dtype=_F64, device=self.ops.device)
        f = _t(f, self.ops.device)
        return f / torch.linalg.vector_norm(f)

    def spin_along_field(self) -> torch.Tensor:
        """<S . h>, h the unit field vector (DSOXF's '<S para>'); z at zero field."""
        return self.spin_expectation() @ self._unit_field()

    def ml_occupancies(self) -> torch.Tensor:
        return ml_occupancies(self.vectors, self.ops)

    def orbital_occupancies(self, kind: str = "real") -> torch.Tensor:
        return orbital_occupancies(self.vectors, self.ops, kind)

    def g_tensor(self, states: Sequence[int] = (0, 1)) -> DoubletG:
        """g of the doublet spanned by ``states`` (use at zero field)."""
        return doublet_g(self.vectors, self.mu, states, self.energies)

    def kramers_g(self, tol: float = LEVEL_TOL) -> list[DoubletG]:
        """g of every doubly degenerate level (zero field); ``tol`` in cm^-1."""
        return [self.g_tensor(idx) for _, idx in self.levels(tol) if len(idx) == 2]

    def field_g(self, i: int, j: int) -> torch.Tensor:
        """DSOXF's g(obs) = (E_j - E_i) / (mu_B |B|)."""
        f = self.field
        if f is None or not torch.any(_t(f) != 0):
            raise ValueError("field_g needs a nonzero field")
        dE = (self.energies[j] - self.energies[i]) * self._f
        return dE / (self.constants.mu_b * torch.linalg.vector_norm(_t(f, self.ops.device)))

    def populations(self, temperature) -> torch.Tensor:
        """exp(-(E_i - E_0) / k_B T), temperature in K (DSOXF's 'relative population', not normalized)."""
        return torch.exp(-self.relative * self._f / (self.constants.k_b * _t(temperature, self.ops.device)))

    def transition_moments(self, b1=None) -> torch.Tensor:
        """(n, n) |<i|mu . b1|j>|^2 in mu_B^2; ``b1`` defaults to the static-field direction (parallel mode)."""
        return transition_moments(self.vectors, self.mu, self._unit_field() if b1 is None else b1)

    def level_transition_moments(self, b1=None, tol: float = LEVEL_TOL) -> torch.Tensor:
        """|<I|mu . b1|J>|^2 summed over the members of both levels (gauge-invariant); ``tol`` in cm^-1."""
        D = level_matrix(self.energies, tol, self.units)
        return D @ self.transition_moments(b1) @ D


def solve(ops: DsoxfOperators, *, field=None, k=(1.0, 1.0, 1.0), units: str = "MHz",
          constants: Constants = TORCHSPIN_CONSTANTS, **kw) -> Solution:
    """Diagonalize H (keywords as :func:`hamiltonian`); eigenvalues/vectors with degenerate-safe derivatives."""
    E, V = eigh_gauge(hamiltonian(ops, field=field, k=k, units=units, constants=constants, **kw))
    return Solution(ops, field, E, V, zeeman_operators(ops, k, constants), units, constants)


class DsoxfSystem:
    """One DSOXF configuration with stored parameter values (all in ``units``) and cached operators.

    >>> s = DsoxfSystem("f", 1, symmetry="axial", axis="trigonal", units="cm^-1", lam=600.0, b4=2000.0, b6=800.0)
    >>> s.levels(b20=torch.tensor(300.0, requires_grad=True))      # keyword overrides per call

    Parameters: ``lam``, ``rho``, ``k`` and any of ``CF_PARAMETERS``.  ``units`` is fixed for the
    system: stored values, overrides and returned energies are all in it.  Field in mT.
    """

    def __init__(self, shell: str = "d", code: int = 1, *, geometry: str = "Oh", symmetry: str = "cubic",
                 axis: str = "tetragonal", conventions: str = "corrected", units: str = "MHz",
                 constants: Constants = TORCHSPIN_CONSTANTS, device: str | torch.device = "cpu", **params):
        self.ops = dsoxf_operators(shell, code, geometry=geometry, symmetry=symmetry, axis=axis,
                                   conventions=conventions, device=device)
        self.units = _u.canonical_energy_unit(units)
        self.constants = constants
        self.defaults = self._check(dict(lam=0.0, rho=0.0, k=(1.0, 1.0, 1.0)) | params)

    @staticmethod
    def _check(p: dict) -> dict:
        bad = set(p) - {"lam", "rho", "k", *CF_PARAMETERS}
        if bad:
            raise TypeError(f"unknown DSOXF parameters {sorted(bad)}")
        return p

    @classmethod
    def from_model(cls, model, *, units: str = "MHz", constants: Constants = TORCHSPIN_CONSTANTS,
                   device: str | torch.device = "cpu") -> "DsoxfSystem":
        """From a DSOXF model object with Telser's cm^-1 values (``torchspin.lft._engines.dsoxf.DsoxfModel``
        or telser-tools' ``DsoxfModel``); the energies are converted to ``units``."""
        f = _u.energy_factor("cm^-1", units)
        p = {a: float(getattr(model, a)) * f for a in ("lam", "rho", *CF_PARAMETERS)}
        return cls(model.shell, model.code, geometry=model.geometry, symmetry=model.symmetry, axis=model.axis,
                   conventions=model.conventions, units=units, constants=constants, device=device,
                   k=tuple(model.k), **p)

    @property
    def size(self) -> int:
        return self.ops.size

    def parameters(self, **kw) -> dict:
        return self.defaults | self._check(kw)

    def _kw(self, kw: dict) -> dict:
        return dict(units=self.units, constants=self.constants, **self.parameters(**kw))

    def hamiltonian(self, field=None, **kw) -> torch.Tensor:
        return hamiltonian(self.ops, field=field, **self._kw(kw))

    def levels(self, field=None, relative: bool = False, **kw) -> torch.Tensor:
        return levels(self.ops, field=field, relative=relative, **self._kw(kw))

    def solve(self, field=None, **kw) -> Solution:
        return solve(self.ops, field=field, **self._kw(kw))

    def zeeman_operators(self, k=None) -> torch.Tensor:
        return zeeman_operators(self.ops, self.defaults["k"] if k is None else k, self.constants)
