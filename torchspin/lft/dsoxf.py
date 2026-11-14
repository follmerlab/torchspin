"""DSOXF: one free-ion term (2S+1)L of d^n or f^n in a crystal field, differentiable.

The Hamiltonian of Telser's DSOXF (via telser-tools), in its ``|m_L m_S>`` basis (``m_L``
outer, both descending)::

    H = H_CF + s abs(lambda) L.S + rho SS + mu_B B . (s_L k L + g_e S)

for the d and f shells, high and low spin (low-spin d4/d5 as the fictitious L~ = 1 term), Oh/Td,
cubic/axial/rhombic and tetragonal/trigonal fields, with both ``conventions`` (``'corrected'``,
the default, or ``'dsoxf'`` = the compiled program).  ``SS = -[(L.S)^2 + L.S/2 -
L(L+1)S(S+1)/3]`` is the Abragam-Bleaney spin-spin term (zero for S < 1).

For a fixed configuration and discrete setting, H_CF is linear in the crystal-field parameters
(``CF_PARAMETERS``: ``dq``, ``ds``, ``dt4`` (= Delta4 = -Dt), ``v2``, ``v4`` for d; ``b4``,
``b6``, ``b20`` ... ``b66`` for f, as DSOXF reads them).  The matrices are probed once from the
NumPy model (``torchspin.lft._engines.dsoxf.DsoxfModel``) and checked for linearity, so the
Stevens operators, DSOXF's scalings (B2X, B4X, F_kq), the symmetry selection and the whole sign
table keep one definition.  ``lam`` is a magnitude: ``lambda_eff = s abs(lambda)`` with the
sign ``s`` that the program chooses (``d abs(lambda)/d lambda := +1`` at 0).

Units
-----
Energy-valued inputs (``lam``, ``rho`` and every crystal-field parameter) and energy-valued
outputs (H, energies, doublet energies) are in ``units`` (default ``'MHz'``; ``'cm^-1'``,
``'GHz'``, ``'K'``, ``'eV'``, ...; see :mod:`torchspin.lft.units`).  The magnetic field is in
mT (DSOXF itself takes gauss), temperatures in K.  ``k``, g values and occupancies are
dimensionless; transition moments ``abs(<i|mu.b1|j>)**2`` are in mu_B^2.  Level tolerances
(``tol``) are in cm^-1 whatever ``units`` is.  Internally everything is cm^-1.

Gauge
-----
Per-state quantities (<S>, occupancies, ``abs(<i|mu|j>)**2``) depend on the arbitrary basis
inside a degenerate level, exactly as in DSOXF, and have no derivative there; their sums over
the level (``level_matrix(E) @ x``, :func:`level_sum`, :meth:`Solution.level_transition_moments`)
are gauge-invariant and differentiable.  Kramers-doublet g tensors use the doublet projector.

Not ported: DSOXF's legacy EPR probabilities (``(sum_a abs(<i|S_a|j>))**2``, a sum of moduli
that depends on eigenvector phases and is not smooth; telser-tools finding DSOXF-05), the
listing and spin-magnitude cutoffs, the input-file reader and the printed report.

Examples
--------
Cu(II), d9, tetragonally elongated (telser-tools' DSOXF quick start): the ground Kramers
doublet has g_perp = 2.069 and g_par = 2.421.

>>> import torch
>>> from torchspin.lft import dsoxf
>>> s = dsoxf.DsoxfSystem("d", 9, symmetry="axial", units="cm^-1", lam=830.0, dq=1200.0,
...                       ds=2000.0, dt4=-500.0, k=(0.8, 0.8, 0.8))
>>> sol = s.solve()
>>> [round(e, 2) for e in sol.relative[::2].tolist()]
[0.0, 10392.24, 12010.6, 15304.75, 16176.33]
>>> [round(g, 4) for g in sol.kramers_g()[0].principal.tolist()]
[2.0686, 2.0686, 2.4212]

Derivatives flow from every continuous parameter, here Ds at a field of 300 mT along z:

>>> ds = torch.tensor(2000.0, dtype=torch.float64, requires_grad=True)
>>> sol = s.solve(field=(0.0, 0.0, 300.0), ds=ds)
>>> (dE,) = torch.autograd.grad(sol.energies[2], ds)
>>> sol.energies.shape, sol.spin_expectation().shape, sol.orbital_occupancies().shape
(torch.Size([10]), torch.Size([10, 3]), torch.Size([10, 5]))
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
    """Constant operators of one DSOXF configuration and discrete setting (energies in cm^-1).

    Built and cached by :func:`dsoxf_operators`; ``N = nl * ns`` is the size of the
    ``|m_L m_S>`` basis.

    Attributes
    ----------
    shell : str
        ``'d'`` or ``'f'``.
    code : int
        DSOXF's configuration number (see :func:`dsoxf_operators`).
    term : str
        Free-ion term, e.g. ``'2D'``; ``'3P~'``/``'2P~'`` for the low-spin L~ = 1 terms.
    L : int
        Orbital quantum number of the term.
    S : float
        Spin quantum number of the term.
    nl, ns : int
        Orbital (2L+1) and spin (2S+1) dimensions.
    cf : torch.Tensor, shape (18, N, N), float64
        Crystal-field operators in ``CF_PARAMETERS`` order: ``H_CF = sum_p value_p cf[p]``
        (parameters inactive in this setting have zero matrices).
    LS, SS : torch.Tensor, shape (N, N), float64
        L.S and the spin-spin (rho) operator (zero for S < 1).
    Lv, Sv : torch.Tensor, shape (3, N, N), complex128
        Orbital and spin angular-momentum vectors.
    so_sign : float
        The sign s of ``lambda_eff = s abs(lambda)``.
    zeeman_sign : float
        s_L, the sign of the orbital Zeeman term (-1 for the corrected L~ = 1 terms).
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
        """Basis size ``n = nl * ns``."""
        return self.nl * self.ns

    @property
    def device(self) -> torch.device:
        """Device of the operator tensors."""
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
    """Cached constant operators of one DSOXF configuration and discrete setting.

    Parameters
    ----------
    shell : {'d', 'f'}, default 'd'
        Electron shell.
    code : int, default 1
        DSOXF's configuration number.  d: 1 ... 9 (high spin, d^n), -4 ... -7 (low spin d4 ... d7;
        -4 and -5 are the fictitious L~ = 1 terms); f: 1 ... 13.
    geometry : {'Oh', 'Td'}, default 'Oh'
        'Td' reverses the sign of every crystal-field parameter.
    symmetry : {'cubic', 'axial', 'rhombic'}, default 'cubic'
        Which crystal-field parameters act, as DSOXF reads them.  d shell: cubic ``dq``; axial
        adds ``ds``, ``dt4``; rhombic adds ``v2``, ``v4``.  f shell: cubic ``b4``, ``b6``; axial
        adds ``b20``, ``b40``, ``b60`` and ``b44``, ``b64`` (tetragonal) or ``b43``, ``b63``, ``b66``
        (trigonal); rhombic adds ``b22``, ``b42``, ``b62``.  Inactive parameters are ignored.
    axis : {'tetragonal', 'trigonal'}, default 'tetragonal'
        Quantization axis: z along C4 or along C3.
    conventions : {'corrected', 'dsoxf'}, default 'corrected'
        'corrected' fixes the signs of the L~ = 1 spin-orbit and orbital-Zeeman terms, the F-term
        rank-2 terms and low-spin d7 (telser-tools findings DSOXF-01 to -03); 'dsoxf' reproduces
        the compiled program.
    device : str or torch.device, default 'cpu'
        Device of the operator tensors.

    Returns
    -------
    DsoxfOperators
        Cached per (setting, device); unit-free (cm^-1).

    Raises
    ------
    ValueError
        For an unknown ``shell``/``code``, ``geometry``, ``symmetry``, ``axis`` or ``conventions``.

    Examples
    --------
    >>> from torchspin.lft import dsoxf
    >>> ops = dsoxf.dsoxf_operators("d", 7, symmetry="axial")
    >>> ops.term, ops.size, tuple(ops.cf.shape)
    ('4F', 28, (18, 28, 28))
    """
    model = DsoxfModel(shell=shell, code=int(code), geometry=geometry, symmetry=symmetry, axis=axis,
                       conventions=conventions)
    return _operators(tuple(getattr(model, a) for a in _DISCRETE), str(torch.device(device)))


def zeeman_operators(ops: DsoxfOperators, k=(1.0, 1.0, 1.0),
                     constants: Constants = TORCHSPIN_CONSTANTS) -> torch.Tensor:
    """Magnetic-moment operators ``mu / mu_B = s_L k_i L_i + g_e S_i``.

    Parameters
    ----------
    ops : DsoxfOperators
        Configuration operators.
    k : array_like or torch.Tensor, shape (..., 3), default (1, 1, 1)
        Orbital reduction factors (dimensionless) along x, y, z.
    constants : torchspin.lft.ddn.Constants, optional
        Physical constants (``g_e``); default :data:`torchspin.lft.ddn.TORCHSPIN_CONSTANTS`.

    Returns
    -------
    torch.Tensor, shape (..., 3, N, N), complex128
        The x, y, z components in units of mu_B.
    """
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
    """DSOXF Hamiltonian as a complex Hermitian matrix.

    Parameters
    ----------
    ops : DsoxfOperators
        Configuration operators (:func:`dsoxf_operators`).
    lam : float or torch.Tensor, default 0.0
        Magnitude ``abs(lambda)`` of the spin-orbit constant, in ``units``; the sign is chosen by
        the configuration (``ops.so_sign``).
    rho : float or torch.Tensor, default 0.0
        Spin-spin parameter in ``units`` (acts only for S >= 1).
    k : array_like or torch.Tensor, shape (..., 3), default (1, 1, 1)
        Orbital reduction factors (Zeeman term only).
    field : array_like or torch.Tensor, shape (..., 3), optional
        Magnetic field vector in mT; ``None`` (default) = zero field.
    units : str, default 'MHz'
        Energy unit of the energy inputs and of the result.
    constants : torchspin.lft.ddn.Constants, optional
        Physical constants, default :data:`torchspin.lft.ddn.TORCHSPIN_CONSTANTS`.
    **cf : float or torch.Tensor
        Crystal-field parameters, any of ``CF_PARAMETERS`` (default 0), in ``units``; which act
        depends on the setting (:func:`dsoxf_operators`).

    Every value may be a tensor with leading batch dimensions and gradients.

    Returns
    -------
    torch.Tensor, shape (..., N, N), complex128
        H in ``units``.

    Raises
    ------
    TypeError
        For a keyword that is not one of ``CF_PARAMETERS``.
    ValueError
        For an unknown ``units``.
    """
    f = _u.energy_factor(units, "cm^-1")
    return _hamiltonian_cm(ops, f, lam, rho, k, field, constants, cf) / f


def levels(ops: DsoxfOperators, *, relative: bool = False, units: str = "MHz", **kw) -> torch.Tensor:
    """Eigenvalues of the DSOXF Hamiltonian.

    Parameters
    ----------
    ops : DsoxfOperators
        Configuration operators.
    relative : bool, default False
        Subtract the lowest eigenvalue (ground state at 0).
    units : str, default 'MHz'
        Energy unit of the inputs and of the result.
    **kw
        ``lam``, ``rho``, ``k``, ``field`` (mT), ``constants`` and crystal-field parameters, as
        :func:`hamiltonian`.

    Returns
    -------
    torch.Tensor, shape (..., N), float64
        Ascending energies in ``units``.  A degenerate eigenvalue gets its level's mean
        derivative.
    """
    E = eigvalsh_gauge(hamiltonian(ops, units=units, **kw))
    return E - E[..., :1] if relative else E


# ------------------------------------------------------------------------------ derived quantities
def _expect(V: torch.Tensor, op: torch.Tensor) -> torch.Tensor:
    """Re <i|op_k|i> for op (..., k, n, n) -> (..., n, k)."""
    return torch.einsum("...ai,...kab,...bi->...ik", V.conj(), op.to(V.dtype), V).real


def spin_expectation(V: torch.Tensor, ops: DsoxfOperators) -> torch.Tensor:
    """Spin expectation values of eigenvectors.

    Parameters
    ----------
    V : torch.Tensor, shape (..., N, N)
        Eigenvectors as columns, in the ``|m_L m_S>`` basis.
    ops : DsoxfOperators
        Configuration operators.

    Returns
    -------
    torch.Tensor, shape (..., N, 3)
        ``<S_x>, <S_y>, <S_z>`` of every eigenvector (gauge dependent inside degenerate levels).
    """
    return _expect(V, ops.Sv)


def ml_occupancies(V: torch.Tensor, ops: DsoxfOperators) -> torch.Tensor:
    """Weights of the ``m_L`` components of eigenvectors.

    Parameters
    ----------
    V : torch.Tensor, shape (..., N, N)
        Eigenvectors as columns.
    ops : DsoxfOperators
        Configuration operators.

    Returns
    -------
    torch.Tensor, shape (..., N, 2L+1)
        ``sum over m_S of abs(c(m_L, m_S))**2``, columns ``m_L = L ... -L``.
    """
    p = (V.conj() * V).real
    return p.reshape(*p.shape[:-2], ops.nl, ops.ns, p.shape[-1]).sum(-2).transpose(-2, -1)


def orbital_occupancies(V: torch.Tensor, ops: DsoxfOperators, kind: str = "real") -> torch.Tensor:
    """Weights of the real orbitals in eigenvectors; each row sums to 1.

    Parameters
    ----------
    V : torch.Tensor, shape (..., N, N)
        Eigenvectors as columns.
    ops : DsoxfOperators
        Configuration operators.
    kind : {'real', 'general', 'cubic'}, default 'real'
        L = 2: ignored, columns ``D_ORBITALS``.  L = 3: ``'general'`` (``F_GENERAL_ORBITALS``)
        or ``'cubic'`` (``F_CUBIC_ORBITALS``).

    Returns
    -------
    torch.Tensor, shape (..., N, 2L+1)
        Real-orbital weights per eigenvector.
    """
    _, u = orbital_vectors(ops.L, kind)
    u = torch.as_tensor(u, dtype=_C128, device=V.device)
    c = V.reshape(*V.shape[:-2], ops.nl, ops.ns, V.shape[-1])
    proj = torch.einsum("mo,...msi->...osi", u.conj(), c)
    return (proj.conj() * proj).real.sum(-2).transpose(-2, -1)


def transition_moments(V: torch.Tensor, mu: torch.Tensor, b1) -> torch.Tensor:
    """Squared transition moments ``abs(<i|mu . b1|j>)**2``.

    Parameters
    ----------
    V : torch.Tensor, shape (..., N, N)
        Eigenvectors as columns.
    mu : torch.Tensor, shape (..., 3, N, N)
        Moment operators in mu_B (:func:`zeeman_operators`).
    b1 : array_like or torch.Tensor, shape (..., 3)
        Microwave-field direction (not normalized here).

    Returns
    -------
    torch.Tensor, shape (..., N, N)
        In mu_B^2 (gauge dependent inside degenerate levels; see :func:`level_matrix`).
    """
    op = torch.einsum("...k,...kab->...ab", _t(b1, V.device).to(_C128), mu)
    m = V.conj().transpose(-2, -1) @ op @ V
    return (m.conj() * m).real


def level_matrix(E: torch.Tensor, tol: float = LEVEL_TOL, units: str = "MHz") -> torch.Tensor:
    """Indicator matrix of degenerate levels.

    Parameters
    ----------
    E : torch.Tensor, shape (..., N)
        Ascending energies in ``units``.
    tol : float, default ``LEVEL_TOL`` (1e-6)
        Level tolerance in cm^-1: states are chained into one level while
        ``abs(E[i+1] - E[i]) <= tol``.
    units : str, default 'MHz'
        Unit of ``E``.

    Returns
    -------
    torch.Tensor, shape (..., N, N)
        1 where states i and j belong to the same level, else 0 (decided on detached energies).
        ``level_matrix(E) @ x`` sums a per-state quantity over each level (gauge-invariant), and
        ``D @ M @ D`` sums a pair quantity over both levels.
    """
    e = E.detach() * _u.energy_factor(units, "cm^-1")
    step = torch.cat([torch.zeros_like(e[..., :1], dtype=torch.long), (e[..., 1:] - e[..., :-1] > tol).long()], -1)
    lab = step.cumsum(-1)
    return (lab.unsqueeze(-1) == lab.unsqueeze(-2)).to(E.dtype)


def level_sum(E: torch.Tensor, x: torch.Tensor, tol: float = LEVEL_TOL, units: str = "MHz") -> torch.Tensor:
    """Sum a per-state quantity over each state's degenerate level.

    Parameters
    ----------
    E : torch.Tensor, shape (..., N)
        Ascending energies in ``units``.
    x : torch.Tensor, shape (..., N) or (..., N, m)
        Per-state quantity.
    tol : float, default ``LEVEL_TOL``
        Level tolerance in cm^-1.
    units : str, default 'MHz'
        Unit of ``E``.

    Returns
    -------
    torch.Tensor
        Same shape as ``x``; entry i is the sum over the level containing state i.
    """
    D = level_matrix(E, tol, units)
    return (D @ x.unsqueeze(-1)).squeeze(-1) if x.dim() == E.dim() else D @ x


class DoubletG(NamedTuple):
    """g tensor of one Kramers doublet.

    Attributes
    ----------
    energy : torch.Tensor
        Mean energy of the doublet, in the solution's units (0 if no energies were given).
    principal : torch.Tensor, shape (..., 3)
        Principal values ``abs(g)``, ascending.
    ggT : torch.Tensor, shape (..., 3, 3)
        ``g g^T`` (gauge-invariant).
    g : torch.Tensor, shape (..., 3, 3)
        ``(g g^T)^(1/2)``, symmetric, with exact gradients at axial g.
    """

    energy: torch.Tensor
    principal: torch.Tensor
    ggT: torch.Tensor
    g: torch.Tensor


def doublet_g(V: torch.Tensor, mu: torch.Tensor, idx: Sequence[int], E: torch.Tensor | None = None) -> DoubletG:
    """g tensor of the doublet spanned by two eigenvectors.

    ``(g g^T)_ij = 2 tr(P mu_i P mu_j)`` with ``P`` the doublet projector, so the result is
    gauge-invariant.  Meaningful at zero field.

    Parameters
    ----------
    V : torch.Tensor, shape (..., N, N)
        Eigenvectors as columns.
    mu : torch.Tensor, shape (..., 3, N, N)
        Moment operators in mu_B.
    idx : sequence of int
        The two column indices of the doublet.
    E : torch.Tensor, shape (..., N), optional
        Energies, for the ``energy`` field.

    Returns
    -------
    DoubletG
    """
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
    """Eigenstates of one DSOXF Hamiltonian (returned by :func:`solve` and :meth:`DsoxfSystem.solve`).

    Attributes
    ----------
    ops : DsoxfOperators
        Configuration operators.
    energies : torch.Tensor, shape (N,)
        Ascending eigenvalues in ``units``.
    vectors : torch.Tensor, shape (N, N), complex128
        Eigenvector columns in the ``|m_L m_S>`` basis.
    mu : torch.Tensor, shape (3, N, N), complex128
        Magnetic-moment operator in mu_B.
    field : array_like or None
        Field vector in mT, or None.
    units : str
        Energy unit of ``energies``.
    constants : torchspin.lft.ddn.Constants
        Physical constants used.
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
        """Group states into degenerate levels.

        Parameters
        ----------
        tol : float, default 1e-5
            DSOXF's chained criterion in cm^-1, applied to detached energies.

        Returns
        -------
        list of (torch.Tensor, list of int)
            ``(energy of the first member, state indices)`` per level, ascending.
        """
        e = self.energies.detach() * self._f
        out: list[tuple[torch.Tensor, list[int]]] = []
        for i in range(len(e)):
            if out and float(e[i] - e[out[-1][1][-1]]) <= tol:
                out[-1][1].append(i)
            else:
                out.append((self.energies[i], [i]))
        return out

    def level_sum(self, x: torch.Tensor, tol: float = LEVEL_TOL) -> torch.Tensor:
        """Per-state ``x`` summed over each degenerate level (:func:`level_sum`; ``tol`` in cm^-1)."""
        return level_sum(self.energies, x, tol, self.units)

    def spin_expectation(self) -> torch.Tensor:
        """``<S>`` per state, shape (N, 3)."""
        return spin_expectation(self.vectors, self.ops)

    def _unit_field(self) -> torch.Tensor:
        f = self.field
        if f is None or not torch.any(_t(f) != 0):
            return torch.tensor([0.0, 0.0, 1.0], dtype=_F64, device=self.ops.device)
        f = _t(f, self.ops.device)
        return f / torch.linalg.vector_norm(f)

    def spin_along_field(self) -> torch.Tensor:
        """``<S . h>`` per state, ``h`` the unit field vector (DSOXF's "<S para>"); ``h = z`` at zero field."""
        return self.spin_expectation() @ self._unit_field()

    def ml_occupancies(self) -> torch.Tensor:
        """``m_L`` weights per state, shape (N, 2L+1) (:func:`ml_occupancies`)."""
        return ml_occupancies(self.vectors, self.ops)

    def orbital_occupancies(self, kind: str = "real") -> torch.Tensor:
        """Real-orbital weights per state, shape (N, 2L+1) (:func:`orbital_occupancies`)."""
        return orbital_occupancies(self.vectors, self.ops, kind)

    def g_tensor(self, states: Sequence[int] = (0, 1)) -> DoubletG:
        """g of the doublet spanned by ``states`` (default the lowest two); use at zero field.  Returns :class:`DoubletG`."""
        return doublet_g(self.vectors, self.mu, states, self.energies)

    def kramers_g(self, tol: float = LEVEL_TOL) -> list[DoubletG]:
        """g of every doubly degenerate level (zero field).

        Parameters
        ----------
        tol : float, default ``LEVEL_TOL``
            Level tolerance in cm^-1.

        Returns
        -------
        list of DoubletG
            One entry per two-fold level, ascending in energy.
        """
        return [self.g_tensor(idx) for _, idx in self.levels(tol) if len(idx) == 2]

    def field_g(self, i: int, j: int) -> torch.Tensor:
        """DSOXF's g(obs) of the pair of states ``i``, ``j``.

        ``g(obs) = (E_j - E_i) / (mu_B abs(B))`` with the energies in cm^-1 and ``mu_B`` in cm^-1/mT.

        Raises
        ------
        ValueError
            If the solution has no field or a zero field.
        """
        f = self.field
        if f is None or not torch.any(_t(f) != 0):
            raise ValueError("field_g needs a nonzero field")
        dE = (self.energies[j] - self.energies[i]) * self._f
        return dE / (self.constants.mu_b * torch.linalg.vector_norm(_t(f, self.ops.device)))

    def populations(self, temperature) -> torch.Tensor:
        """Boltzmann factors ``exp(-(E_i - E_0) / k_B T)`` per state.

        ``temperature`` in K (float or tensor).  DSOXF's "relative population": not normalized.
        """
        return torch.exp(-self.relative * self._f / (self.constants.k_b * _t(temperature, self.ops.device)))

    def transition_moments(self, b1=None) -> torch.Tensor:
        """Squared transition moments ``abs(<i|mu . b1|j>)**2`` in mu_B^2, shape (N, N).

        ``b1`` is the microwave-field direction; default the static-field direction (parallel mode;
        z at zero field).
        """
        return transition_moments(self.vectors, self.mu, self._unit_field() if b1 is None else b1)

    def level_transition_moments(self, b1=None, tol: float = LEVEL_TOL) -> torch.Tensor:
        """Transition moments summed over the members of both levels (gauge-invariant).

        ``abs(<I|mu . b1|J>)**2`` in mu_B^2, shape (N, N); ``b1`` as :meth:`transition_moments`,
        ``tol`` in cm^-1.
        """
        D = level_matrix(self.energies, tol, self.units)
        return D @ self.transition_moments(b1) @ D


def solve(ops: DsoxfOperators, *, field=None, k=(1.0, 1.0, 1.0), units: str = "MHz",
          constants: Constants = TORCHSPIN_CONSTANTS, **kw) -> Solution:
    """Diagonalize the DSOXF Hamiltonian.

    Parameters
    ----------
    ops : DsoxfOperators
        Configuration operators.
    field : array_like or torch.Tensor, shape (3,), optional
        Field vector in mT.
    k : array_like, shape (3,), default (1, 1, 1)
        Orbital reduction factors.
    units : str, default 'MHz'
        Energy unit of the inputs and of the solution.
    constants : torchspin.lft.ddn.Constants, optional
        Physical constants.
    **kw
        ``lam``, ``rho`` and crystal-field parameters, as :func:`hamiltonian`.

    Returns
    -------
    Solution
        Eigenvalues and eigenvectors with degenerate-safe derivatives.
    """
    E, V = eigh_gauge(hamiltonian(ops, field=field, k=k, units=units, constants=constants, **kw))
    return Solution(ops, field, E, V, zeeman_operators(ops, k, constants), units, constants)


class DsoxfSystem:
    """One DSOXF configuration with stored parameter values and cached operators.

    Parameters
    ----------
    shell, code, geometry, symmetry, axis, conventions
        The configuration and discrete setting, as :func:`dsoxf_operators`.
    units : str, default 'MHz'
        Energy unit of the stored parameters, of per-call overrides and of every returned energy;
        fixed for the system.
    constants : torchspin.lft.ddn.Constants, optional
        Physical constants, default :data:`torchspin.lft.ddn.TORCHSPIN_CONSTANTS`.
    device : str or torch.device, default 'cpu'
        Device of the operators.
    **params
        Stored defaults: ``lam`` (``abs(lambda)``, default 0), ``rho`` (default 0), ``k``
        (default ``(1, 1, 1)``) and any of ``CF_PARAMETERS`` (default 0), energies in ``units``.

    Raises
    ------
    TypeError
        For an unknown parameter name (also for per-call overrides).

    Examples
    --------
    >>> import torch
    >>> from torchspin.lft import dsoxf
    >>> s = dsoxf.DsoxfSystem("f", 1, symmetry="axial", axis="trigonal", units="cm^-1",
    ...                       lam=600.0, b4=2000.0, b6=800.0)
    >>> s.ops.term, s.size
    ('2F', 14)
    >>> E = s.levels(b20=torch.tensor(300.0), relative=True)   # keyword overrides per call
    >>> E.shape
    torch.Size([14])
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
        """Build from a DSOXF model object holding Telser's cm^-1 values.

        ``model`` is ``torchspin.lft._engines.dsoxf.DsoxfModel`` or telser-tools' ``DsoxfModel``;
        its energies are converted from cm^-1 to ``units``.
        """
        f = _u.energy_factor("cm^-1", units)
        p = {a: float(getattr(model, a)) * f for a in ("lam", "rho", *CF_PARAMETERS)}
        return cls(model.shell, model.code, geometry=model.geometry, symmetry=model.symmetry, axis=model.axis,
                   conventions=model.conventions, units=units, constants=constants, device=device,
                   k=tuple(model.k), **p)

    @property
    def size(self) -> int:
        """Basis size ``(2L+1)(2S+1)``."""
        return self.ops.size

    def parameters(self, **kw) -> dict:
        """Stored parameters updated with the keyword overrides ``kw`` (raises ``TypeError`` for unknown names)."""
        return self.defaults | self._check(kw)

    def _kw(self, kw: dict) -> dict:
        return dict(units=self.units, constants=self.constants, **self.parameters(**kw))

    def hamiltonian(self, field=None, **kw) -> torch.Tensor:
        """Hamiltonian (n, n) in ``self.units``; ``field`` in mT, keywords override the stored parameters (:func:`hamiltonian`)."""
        return hamiltonian(self.ops, field=field, **self._kw(kw))

    def levels(self, field=None, relative: bool = False, **kw) -> torch.Tensor:
        """Ascending energies (n,) in ``self.units``; ``field`` in mT, keywords override the stored parameters (:func:`levels`)."""
        return levels(self.ops, field=field, relative=relative, **self._kw(kw))

    def solve(self, field=None, **kw) -> Solution:
        """Diagonalize at ``field`` (mT) with keyword overrides; returns a :class:`Solution` (:func:`solve`)."""
        return solve(self.ops, field=field, **self._kw(kw))

    def zeeman_operators(self, k=None) -> torch.Tensor:
        """Moment operators (3, n, n) in mu_B for ``k`` (default: the stored ``k``)."""
        return zeeman_operators(self.ops, self.defaults["k"] if k is None else k, self.constants)
