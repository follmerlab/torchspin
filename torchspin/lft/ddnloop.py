"""DDNLOOP: d^n levels and EPR transitions over field, angle and parameter scans.

Torch port of the numerical core of Josh Telser's DDNLOOP (via telser-tools,
``telser_tools_diff.ddnloop``), built on :mod:`torchspin.lft.ddn`.  For every scan point
(field magnitude B, polar angle theta, azimuth phi) the complete d^n Hamiltonian with the field
B (sin t cos p, sin t sin p, cos t) is diagonalized (batched over the scan), and DDNLOOP's
quantities are returned as tensors (data for Zeeman and angle diagrams; plots are separate):

* level energies;
* Boltzmann factors exp(-(E_i - E_0)/kT), normalized (default) or as DDNLOOP prints them;
* for the requested pairs (i, j): transition energies E_j - E_i, probabilities (the
  B1-averaged full magnetic dipole ``abs(<i|B1.(k L + g_e S)|j>)**2``, in mu_B^2), the same summed
  over the degenerate members of each level, population factors p_i - p_j, DDNLOOP's microwave
  weights and g_obs = (E_j - E_i)/(mu_B B).

Every parameter (B, C, A, F^k, zeta, k, alpha, the ligand field V or whatever builds it, such as
AOM bond vectors), the temperature and the scan variables themselves may carry gradients.

Units
-----
Energy-valued inputs (``B``, ``C``, ``A``, ``zeta``, ``alpha``, ``slater``, ``V``) and the
returned energies and transition energies are in ``units`` (default ``'MHz'``; ``'cm^-1'``,
``'GHz'``, ``'K'``, ``'eV'``).  Fields in mT, angles in radians (``degrees=True`` for Telser-style
input), temperature in K, microwave frequency in GHz.  Probabilities are in mu_B^2, g_obs and
populations are dimensionless.

    >>> import torch
    >>> from torchspin import lft
    >>> V = lft.crystal_field(Dq=1500.0, Ds=300.0, Dt=50.0)
    >>> res = lft.ddnloop.field_scan(1, torch.linspace(0, 1000, 101), theta=0.3, V=V, zeta=154.0,
    ...                              units='cm^-1', pairs=[(0, 1)], temperature=4.2, mw_freq=9.5)
    >>> res.energies.shape, res.g_obs.shape
    (torch.Size([101, 10]), torch.Size([101, 1]))

Gauge.  The per-pair probability ``abs(<i|O|j>)**2`` (what DDNLOOP prints) depends on the arbitrary
basis inside a degenerate level and has no derivative there: at B > 0 the Zeeman term lifts
the Kramers degeneracies, but levels the field leaves degenerate (S = 0 orbital multiplets at
special orientations, every level at B = 0) make the per-pair value basis dependent.
``level_probabilities`` sums ``abs(<a|O|b>)**2`` over a in level(i), b in level(j) (= tr(P_I O P_J O^+),
gauge-invariant and differentiable everywhere off accidental crossings); it equals the
per-pair value wherever both levels are non-degenerate.  Levels are decided on detached
energies with ``torchspin.lft._linalg.degeneracy_mask``.

Not ported (DDN-reproduction conventions of telser-tools' ``Options``): EPRTRANS
(spin-only, eigenvector-phase dependent probabilities, finding DDNLOOP-01), DDN's hole problem,
constants and absolute energy scale (``ddn_model``), the doubled Boltzmann factor of the
population factor (DDNLOOP-03), the 1e-10 output thresholds, the input reader, the listing
and the plot.  DDN's mirror-image ligand field (DDNLOOP-02) is available as
:func:`mirror_image` to apply to V by hand.
"""

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple, Sequence

import numpy as np
import torch

from . import units as _u
from ._engines.dn import U_REAL
from ._linalg import degeneracy_mask, eigh_gauge
from ._operators import DnOperators, dn_operators
from .ddn import TORCHSPIN_CONSTANTS, Constants, hamiltonian, levels, zeeman_operators

__all__ = ["LoopResult", "MW_WIDTH_FRACTION", "angle_scan", "directions", "exact_probabilities", "field_scan",
           "level_probabilities", "mirror_image", "pair_elements", "pair_probability_matrix", "parameter_scan",
           "scan"]

_F64, _C128 = torch.float64, torch.complex128
#: DDNLOOP's microwave weight is P W^2 / (W^2 + (nu - nu_mw)^2) with W = 2 % of nu_mw.
MW_WIDTH_FRACTION = 2.0e-2
_PARAMS = ("B", "C", "A", "zeta", "V", "k", "alpha", "slater")


def _t(x, device="cpu") -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x if x.dtype in (_F64, _C128) else x.to(_F64)
    return torch.as_tensor(np.asarray(x, float), dtype=_F64, device=device)


def _ops(n, device) -> DnOperators:
    return n if isinstance(n, DnOperators) else dn_operators(n, device)


# ------------------------------------------------------------------------------ functional core
def directions(theta, phi, *, degrees: bool = False) -> torch.Tensor:
    """Unit field directions ``(sin t cos p, sin t sin p, cos t)``.

    Parameters
    ----------
    theta, phi : float or array_like or torch.Tensor
        Polar and azimuthal angles, broadcast together; radians unless
        ``degrees``.
    degrees : bool, default False
        Angles are in degrees.

    Returns
    -------
    torch.Tensor, shape broadcast(theta, phi) + (3,), float64
        Unit vectors in the molecular frame.

    Examples
    --------
    >>> from torchspin import lft
    >>> lft.ddnloop.directions(90.0, 0.0, degrees=True).round(decimals=6).tolist()
    [1.0, 0.0, 0.0]
    """
    t, p = torch.broadcast_tensors(_t(theta), _t(phi))
    if degrees:
        t, p = torch.deg2rad(t), torch.deg2rad(p)
    st, ct, sp, cp = torch.sin(t), torch.cos(t), torch.sin(p), torch.cos(p)
    return torch.stack([st * cp, st * sp, ct], -1)


@lru_cache(maxsize=None)
def _mirror_matrix() -> torch.Tensor:
    return torch.as_tensor((U_REAL.conj().T @ U_REAL.conj()).real, dtype=_F64)


def mirror_image(V) -> torch.Tensor:
    """DDN's internal real-orbital field (DDNLOOP-02), the mirror image y -> -y of ``V``.

    DDN works with the complex conjugate of V in the ``|m>`` basis, which is the
    field of the ligand set reflected through the xz plane (the xy and yz rows
    and columns change sign).  Apply it to ``V`` by hand to reproduce DDN's
    numbers for a general field direction.  Linear, so gradients pass.

    Parameters
    ----------
    V : array_like or torch.Tensor, shape (..., 5, 5)
        Real-orbital ligand field (x2-y2, xy, xz, yz, z2), any energy unit.

    Returns
    -------
    torch.Tensor, shape (..., 5, 5), float64
        The mirrored field, same unit.
    """
    V = _t(V)
    M = _mirror_matrix().to(V.device)
    return M @ V @ M.T


def pair_elements(vectors: torch.Tensor, mu: torch.Tensor, pi: Sequence[int], pj: Sequence[int]) -> torch.Tensor:
    """Matrix elements ``<i|mu_k|j>`` for selected state pairs.

    Parameters
    ----------
    vectors : torch.Tensor, shape (..., N, N)
        Eigenvectors as columns.
    mu : torch.Tensor, shape (3, N, N)
        Operators (e.g. :func:`torchspin.lft.ddn.zeeman_operators`).
    pi, pj : sequence of int
        0-based state indices of the pairs (same length P).

    Returns
    -------
    torch.Tensor, shape (..., 3, P), complex
        The matrix elements, in the unit of ``mu`` (mu_B for the moment).
    """
    vi = vectors[..., :, list(pi)]
    vj = vectors[..., :, list(pj)]
    return torch.einsum("...ap,kab,...bp->...kp", vi.conj(), mu, vj)


def _mode(perp, par, h1mode: float):
    if h1mode == 0:
        return perp
    if h1mode < 0:
        return par
    return perp + h1mode * par


def exact_probabilities(elements: torch.Tensor, dirs: torch.Tensor, h1mode: float = 0.0) -> torch.Tensor:
    """B1-averaged transition probabilities from matrix elements.

    Parameters
    ----------
    elements : torch.Tensor, shape (..., 3, P)
        ``<i|mu_k|j>`` (:func:`pair_elements`), in mu_B.
    dirs : array_like or torch.Tensor, shape (..., 3)
        Unit directions n of the static field.
    h1mode : float, default 0.0
        Excitation mode: ``0`` perpendicular, ``(abs(M)**2 - abs(n.M)**2)/2``
        (the mean over B1 normal to B0); ``< 0`` parallel ``abs(n.M)**2``;
        ``> 0`` perpendicular + ``h1mode`` times parallel.

    Returns
    -------
    torch.Tensor, shape (..., P), float64
        Probabilities in mu_B^2.
    """
    n = _t(dirs).to(elements.dtype)
    nm = torch.einsum("...k,...kp->...p", n, elements)
    par = (nm.conj() * nm).real
    tot = (elements.conj() * elements).real.sum(-2)
    return _mode(0.5 * (tot - par), par, h1mode)


def pair_probability_matrix(vectors: torch.Tensor, mu: torch.Tensor, dirs: torch.Tensor,
                            h1mode: float = 0.0) -> torch.Tensor:
    """Transition probability of every state pair (the per-pair DDNLOOP quantity).

    Parameters
    ----------
    vectors : torch.Tensor, shape (..., N, N)
        Eigenvectors as columns.
    mu : torch.Tensor, shape (3, N, N)
        Moment operators in mu_B.
    dirs : array_like or torch.Tensor, shape (..., 3)
        Unit directions of the static field.
    h1mode : float, default 0.0
        Excitation mode, as :func:`exact_probabilities`.

    Returns
    -------
    torch.Tensor, shape (..., N, N), float64
        Probabilities in mu_B^2 (basis dependent inside degenerate levels).
    """
    m = vectors.conj().transpose(-2, -1).unsqueeze(-3) @ mu @ vectors.unsqueeze(-3)       # (..., 3, N, N)
    n = _t(dirs).to(m.dtype)
    nm = torch.einsum("...k,...kij->...ij", n, m)
    par = (nm.conj() * nm).real
    tot = (m.conj() * m).real.sum(-3)
    return _mode(0.5 * (tot - par), par, h1mode)


def level_probabilities(energies: torch.Tensor, pmat: torch.Tensor, pi: Sequence[int],
                        pj: Sequence[int]) -> torch.Tensor:
    """Transition probabilities summed over the degenerate members of each level.

    ``sum_{a in level(i), b in level(j)} pmat[a, b]``: gauge invariant and
    differentiable off accidental crossings; equal to ``pmat[i, j]`` where both
    levels are nondegenerate.  Levels are decided on detached energies.

    Parameters
    ----------
    energies : torch.Tensor, shape (..., N)
        Ascending energies (any unit).
    pmat : torch.Tensor, shape (..., N, N)
        Per-pair probabilities (:func:`pair_probability_matrix`).
    pi, pj : sequence of int
        0-based state indices of the pairs.

    Returns
    -------
    torch.Tensor, shape (..., P), float64
        Summed probabilities in the unit of ``pmat``.
    """
    D = degeneracy_mask(energies.detach()).to(pmat.dtype)
    S = D @ pmat @ D
    return S[..., list(pi), list(pj)]


# ------------------------------------------------------------------------------ scans
class LoopResult(NamedTuple):
    """DDNLOOP's quantities over a scan of shape ``S`` (returned by :func:`scan`).

    Entries are ``None`` when not requested: without ``temperature`` the
    populations and population factors, without ``mw_freq`` the microwave
    weights, without ``pairs`` every pair quantity.  DDNLOOP lists g_obs only
    with a microwave frequency; here it is always returned with ``pairs``.

    Attributes
    ----------
    theta, phi : torch.Tensor, shape S
        Field direction angles in radians.
    fields : torch.Tensor, shape S
        Field magnitudes in mT.
    directions : torch.Tensor, shape S + (3,)
        Unit field directions.
    energies : torch.Tensor, shape S + (N,)
        Ascending energies in ``units``.
    vectors : torch.Tensor, shape S + (N, N)
        Eigenvectors as columns.
    populations : torch.Tensor or None, shape S + (N,)
        Boltzmann factors (normalized to sum 1, or DDNLOOP's
        ``exp(-(E_i - E_0)/kT)``).
    pairs : tuple of (int, int)
        0-based pairs (i, j), i < j; P = ``len(pairs)``.
    transition_energies : torch.Tensor or None, shape S + (P,)
        ``E_j - E_i`` in ``units``.
    probabilities : torch.Tensor or None, shape S + (P,)
        Per-pair probabilities in mu_B^2 (basis dependent inside degenerate
        levels).
    level_probabilities : torch.Tensor or None, shape S + (P,)
        Probabilities summed over degenerate members (gauge invariant).
    population_factors : torch.Tensor or None, shape S + (P,)
        ``p_i - p_j``.
    mw_weights : torch.Tensor or None, shape S + (P,)
        DDNLOOP's microwave weights ``P W^2 / (W^2 + (nu - nu_mw)^2)`` with
        ``W = MW_WIDTH_FRACTION * nu_mw`` (2 %), from the per-pair probability.
    g_obs : torch.Tensor or None, shape S + (P,)
        ``(E_j - E_i) / (mu_B B)``, dimensionless; NaN at B = 0.
    units : str
        Canonical energy unit of ``energies`` and ``transition_energies``.
    relative : torch.Tensor, shape S + (N,)
        Property: energies with the ground state of every scan point at 0.
    """

    theta: torch.Tensor                 # S, radians
    phi: torch.Tensor                   # S, radians
    fields: torch.Tensor                # S, mT
    directions: torch.Tensor            # S + (3,)
    energies: torch.Tensor              # S + (N,), ascending
    vectors: torch.Tensor               # S + (N, N), eigenvector columns
    populations: torch.Tensor | None    # S + (N,)
    pairs: tuple[tuple[int, int], ...]
    transition_energies: torch.Tensor | None   # S + (P,), E_j - E_i
    probabilities: torch.Tensor | None         # per pair (basis dependent inside degenerate levels)
    level_probabilities: torch.Tensor | None   # summed over degenerate members (gauge-invariant)
    population_factors: torch.Tensor | None    # p_i - p_j
    mw_weights: torch.Tensor | None            # P W^2/(W^2 + (nu - nu_mw)^2), W = 2 % of nu_mw
    g_obs: torch.Tensor | None                 # (E_j - E_i)/(mu_B B); NaN at B = 0
    units: str

    @property
    def relative(self) -> torch.Tensor:
        """Energies with the ground state of every scan point at 0."""
        return self.energies - self.energies[..., :1]


def _pairs(pairs, nstates: int) -> tuple[tuple[int, int], ...]:
    if pairs is None:
        return ()
    if isinstance(pairs, str):
        if pairs != "all":
            raise ValueError("pairs must be a sequence of (i, j) or 'all'")
        return tuple((i, j) for i in range(nstates) for j in range(i + 1, nstates))
    out = tuple((int(i), int(j)) for i, j in pairs)
    for i, j in out:
        if not 0 <= i < j < nstates:
            raise ValueError(f"pair {(i, j)}: need 0 <= i < j < {nstates} (0-based state indices)")
    return out


def scan(n, *, fields, theta=0.0, phi=0.0, mesh: bool = False, degrees: bool = False, pairs=None,
         temperature=None, normalized_populations: bool = True, h1mode: float = 0.0, mw_freq=None,
         units: str = "MHz", constants: Constants = TORCHSPIN_CONSTANTS, device="cpu", **params) -> LoopResult:
    """Levels and transitions of d^n over a scan of field magnitude and direction.

    For every scan point the full d^n Hamiltonian with the field
    ``fields * (sin t cos p, sin t sin p, cos t)`` is diagonalized (batched).

    Parameters
    ----------
    n : int or DnOperators
        Number of d electrons (0 to 10) or cached operators.
    fields : float or array_like or torch.Tensor
        Field magnitudes in mT.
    theta, phi : float or array_like or torch.Tensor, default 0.0
        Field direction, radians (degrees with ``degrees=True``).
    mesh : bool, default False
        False: ``fields``, ``theta`` and ``phi`` are broadcast together to the
        scan shape S.  True: they must be 1-D (or scalars) and are meshed as
        DDNLOOP loops them, S = (n_theta, n_phi, n_field) (theta outermost;
        ``reshape(-1)`` gives DDNLOOP's listing order).
    degrees : bool, default False
        Angles are in degrees.
    pairs : sequence of (int, int) or 'all', optional
        0-based state pairs (i, j), i < j, for the transition quantities.
        Default ``None``: none.
    temperature : float or torch.Tensor, optional
        Temperature in K for Boltzmann populations.  Default ``None``: none.
    normalized_populations : bool, default True
        Normalize the Boltzmann factors to sum 1; False gives DDNLOOP's printed
        ``exp(-(E_i - E_0)/kT)``.
    h1mode : float, default 0.0
        Excitation mode: 0 perpendicular, < 0 parallel, > 0 perpendicular +
        ``h1mode`` times parallel.
    mw_freq : float or torch.Tensor, optional
        Microwave frequency in GHz for DDNLOOP's microwave weights.
    units : str, default 'MHz'
        Energy unit of the Hamiltonian keywords and of the returned energies.
    constants : torchspin.lft.ddn.Constants, default TORCHSPIN_CONSTANTS
        mu_B, g_e, k_B.
    device : str or torch.device, default 'cpu'
        Device.
    **params
        Hamiltonian keywords of :func:`torchspin.lft.ddn.hamiltonian`: ``B``,
        ``C``, ``A``, ``zeta``, ``V``, ``k``, ``alpha``, ``slater``.

    Returns
    -------
    LoopResult

    Raises
    ------
    TypeError
        For a keyword that is not a Hamiltonian parameter.
    ValueError
        If ``mesh=True`` with multi-dimensional inputs, or for invalid
        ``pairs`` (not ``'all'``, or not ``0 <= i < j < N``).

    Examples
    --------
    >>> import torch
    >>> from torchspin import lft
    >>> res = lft.ddnloop.scan(1, fields=torch.linspace(0, 500, 6), theta=torch.tensor([0.0, 1.0]),
    ...                        mesh=True, V=lft.crystal_field(Dq=1500.0, Ds=300.0), zeta=154.0,
    ...                        units='cm^-1', pairs='all')
    >>> res.energies.shape, res.probabilities.shape
    (torch.Size([2, 1, 6, 10]), torch.Size([2, 1, 6, 45]))
    """
    unknown = set(params) - set(_PARAMS)
    if unknown:
        raise TypeError(f"unexpected keywords: {sorted(unknown)}")
    ops = _ops(n, device)
    dev = ops.device
    th, ph, hh = _t(theta, dev), _t(phi, dev), _t(fields, dev)
    if mesh:
        if max(th.ndim, ph.ndim, hh.ndim) > 1:
            raise ValueError("mesh=True needs 1-D theta, phi and fields")
        th, ph, hh = torch.meshgrid(th.reshape(-1), ph.reshape(-1), hh.reshape(-1), indexing="ij")
    else:
        th, ph, hh = torch.broadcast_tensors(th, ph, hh)
    if degrees:
        th, ph = torch.deg2rad(th), torch.deg2rad(ph)
    dirs = directions(th, ph)
    f = _u.energy_factor(units, "cm^-1")
    H = hamiltonian(ops, field=hh[..., None] * dirs, units=units, constants=constants, **params)
    E, vec = eigh_gauge(H)
    E_cm = E * f
    pops = None
    if temperature is not None:
        bkt = constants.k_b * _t(temperature, dev)
        pops = torch.exp(-(E_cm - E_cm[..., :1]) / bkt)
        if normalized_populations:
            pops = pops / pops.sum(-1, keepdim=True)
    pr = _pairs(pairs, ops.size)
    if not pr:
        return LoopResult(th, ph, hh, dirs, E, vec, pops, (), None, None, None, None, None, None,
                          _u.canonical_energy_unit(units))
    pi = [i for i, _ in pr]
    pj = [j for _, j in pr]
    mu = zeeman_operators(ops, params.get("k", (1.0, 1.0, 1.0)), constants)
    pmat = pair_probability_matrix(vec, mu, dirs, h1mode)
    prob = pmat[..., pi, pj]
    lprob = level_probabilities(E, pmat, pi, pj)
    tr = E[..., pj] - E[..., pi]
    tr_cm = tr * f
    popf = None if pops is None else pops[..., pi] - pops[..., pj]
    mw = None
    if mw_freq is not None:
        freq = _t(mw_freq, dev) * 1.0e3                              # MHz
        wmw = MW_WIDTH_FRACTION * freq
        tmhz = tr_cm * _u.energy_factor("cm^-1", "MHz")
        mw = prob * wmw * wmw / (wmw * wmw + (tmhz - freq) ** 2)
    pos = (hh > 0)[..., None]
    safe = torch.where(hh > 0, hh, torch.ones_like(hh))
    g = torch.where(pos, tr_cm / (constants.mu_b * safe[..., None]), torch.full_like(tr_cm, float("nan")))
    return LoopResult(th, ph, hh, dirs, E, vec, pops, pr, tr, prob, lprob, popf, mw, g,
                      _u.canonical_energy_unit(units))


def field_scan(n, fields, theta=0.0, phi=0.0, **kw) -> LoopResult:
    """Zeeman-diagram data: :func:`scan` over field magnitudes at a fixed direction.

    Parameters
    ----------
    n : int or DnOperators
        Number of d electrons or cached operators.
    fields : array_like or torch.Tensor
        Field magnitudes in mT.
    theta, phi : float or torch.Tensor, default 0.0
        Field direction (radians unless ``degrees=True``).
    **kw
        Further keywords of :func:`scan`.

    Returns
    -------
    LoopResult

    Examples
    --------
    >>> import torch
    >>> from torchspin import lft
    >>> res = lft.ddnloop.field_scan(1, torch.linspace(0, 1000, 11), theta=0.3,
    ...                              V=lft.crystal_field(Dq=1500.0, Ds=300.0, Dt=50.0),
    ...                              zeta=154.0, units='cm^-1', pairs=[(0, 1)])
    >>> res.transition_energies.shape, bool(res.g_obs[0, 0].isnan())
    (torch.Size([11, 1]), True)
    """
    return scan(n, fields=fields, theta=theta, phi=phi, **kw)


def angle_scan(n, field, theta, phi=0.0, *, mesh: bool = True, **kw) -> LoopResult:
    """Angle-diagram data: :func:`scan` at a fixed field magnitude over directions.

    Parameters
    ----------
    n : int or DnOperators
        Number of d electrons or cached operators.
    field : float or torch.Tensor
        Field magnitude in mT.
    theta, phi : array_like or torch.Tensor
        Directions (radians unless ``degrees=True``); ``phi`` defaults to 0.
    mesh : bool, default True
        True: scan shape (n_theta, n_phi) from 1-D ``theta`` and ``phi``.
        False: theta and phi are broadcast (e.g. a path through orientation
        space).
    **kw
        Further keywords of :func:`scan`.

    Returns
    -------
    LoopResult

    Examples
    --------
    >>> import torch
    >>> from torchspin import lft
    >>> res = lft.ddnloop.angle_scan(1, 350.0, torch.linspace(0, 90, 7), [0.0, 45.0], degrees=True,
    ...                              V=lft.crystal_field(Dq=1500.0, Ds=300.0), zeta=154.0, units='cm^-1')
    >>> res.energies.shape
    torch.Size([7, 2, 10])
    """
    th, ph = _t(theta), _t(phi)
    if mesh:
        th, ph = th.reshape(-1, 1), ph.reshape(1, -1)
    return scan(n, fields=field, theta=th, phi=ph, mesh=False, **kw)


def parameter_scan(n, name: str, values, *, field=None, relative: bool = False, units: str = "MHz",
                   **params) -> torch.Tensor:
    """Energy levels for a series of values of one Hamiltonian parameter (correlation diagram).

    Parameters
    ----------
    n : int or DnOperators
        Number of d electrons or cached operators.
    name : str
        Parameter to scan: ``'B'``, ``'C'``, ``'A'``, ``'zeta'`` or
        ``'alpha'`` (values shape (m,)), ``'k'`` ((m, 3)) or ``'V'``
        ((m, 5, 5)).
    values : array_like or torch.Tensor
        The m values, in ``units`` for energies.
    field : array_like or torch.Tensor, shape (3,), optional
        Field vector in mT, the same for every value.  Default ``None``.
    relative : bool, default False
        Put each ground state at 0.
    units : str, default 'MHz'
        Energy unit of the inputs and the result.
    **params
        The other (fixed) Hamiltonian keywords of
        :func:`torchspin.lft.ddn.hamiltonian`.

    Returns
    -------
    torch.Tensor, shape (m, N), float64
        Ascending levels in ``units`` for each value.

    Raises
    ------
    ValueError
        If ``name`` cannot be scanned or ``values`` has the wrong number of
        dimensions.
    TypeError
        If ``name`` is also passed as a keyword.

    Examples
    --------
    >>> from torchspin import lft
    >>> E = lft.ddnloop.parameter_scan(2, 'B', [600.0, 800.0, 1000.0], C=3200.0, units='cm^-1')
    >>> E.shape
    torch.Size([3, 45])
    """
    if name not in ("B", "C", "A", "zeta", "alpha", "k", "V"):
        raise ValueError(f"cannot scan {name!r}; use one of B, C, A, zeta, alpha, k, V")
    if name in params:
        raise TypeError(f"{name!r} is the scanned parameter; do not also pass it as a keyword")
    vals = _t(values)
    want = {"k": 2, "V": 3}.get(name, 1)
    if vals.ndim != want:
        raise ValueError(f"values for {name!r} must have {want} dimension(s), got shape {tuple(vals.shape)}")
    params[name] = vals
    return levels(n, field=field, relative=relative, units=units, **params)
