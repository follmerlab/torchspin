"""DDNMAG: differentiable molar magnetization M(B, T) of a d^n ion from the full ligand-field model.

    m(n, B, T) = <-n.(k L + g_e S)>_T  (Bohr magnetons per ion along the field direction n)

with H = H_LF(d^n) + mu_B abs(B) n.(k L + g_e S), the exact thermal average over every state of the
d^n configuration (Hellmann-Feynman), and powder averages as weighted sums over an
:class:`torchspin.lft.grids.OrientationGrid`.  This is Telser's DDNMAG with ``method="exact"``;
DDN's finite-difference schemes (Hendrickson, McGarvey, ...) approximate the same quantity and
are not ported.

The thermal average uses ``torchspin.lft._linalg.thermal_average``, whose derivative
(Daleckii-Krein) is exact through the degeneracies M always has (Kramers pairs, cubic levels,
field-induced crossings), so every ligand-field parameter can carry gradients.

Units
-----
``field`` in mT, ``T`` in K.  Energy-valued ligand-field parameters in ``units`` (default 'MHz';
'cm^-1' etc. on request; see :mod:`torchspin.lft.ddn`).  ``M_units``: ``'muB'`` (Bohr magnetons
per ion, as :func:`torchspin.curry` returns), ``'emu/mol'`` (DDNMAG's unit; 1 N_A mu_B = 5585
emu/mol) or ``'SI'`` (A m^2/mol).  Temperature-independent paramagnetism ``chi0`` is in
``chi_units`` (``'SI'`` m^3/mol, default, or ``'cgs'`` cm^3/mol).
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from torchspin.constants import AVOGADRO, BMAGN, MU0

from . import units as _u
from ._linalg import thermal_average
from ._operators import dn_operators
from .ddn import TORCHSPIN_CONSTANTS, _hamiltonian_cm, zeeman_operators
from .grids import OrientationGrid, as_grid

__all__ = ["MagConstants", "MAG_CONSTANTS", "MagResult", "fit", "magnetization", "moment_core", "mirror_y"]

_F64, _C128 = torch.float64, torch.complex128


class MagConstants(NamedTuple):
    """Physical constants of the magnetometry functions.

    Attributes
    ----------
    mu_b : float
        Bohr magneton over h c, in cm^-1 per mT.
    g_e : float
        Free-electron g value.
    k_b : float
        Boltzmann constant over h c, in cm^-1 per K.
    N_A : float
        Avogadro constant, 1/mol.
    mu_b_SI : float
        Bohr magneton in J/T.
    mu0 : float
        Vacuum permeability in H/m.
    ddn : torchspin.lft.ddn.Constants
        Property: the ``(mu_b, g_e, k_b)`` subset as :class:`torchspin.lft.ddn.Constants`.
    """

    mu_b: float
    g_e: float
    k_b: float
    N_A: float
    mu_b_SI: float
    mu0: float

    @property
    def ddn(self):
        from .ddn import Constants
        return Constants(self.mu_b, self.g_e, self.k_b)


#: From ``torchspin.constants`` (CODATA 2022 mu_B and mu_0).
MAG_CONSTANTS = MagConstants(mu_b=TORCHSPIN_CONSTANTS.mu_b, g_e=TORCHSPIN_CONSTANTS.g_e, k_b=TORCHSPIN_CONSTANTS.k_b,
                             N_A=AVOGADRO, mu_b_SI=BMAGN, mu0=MU0)

_M_UNITS = ("muB", "emu/mol", "SI")


def _m_factor(M_units: str, c: MagConstants) -> float:
    """Bohr magnetons per ion -> M_units."""
    if M_units == "muB":
        return 1.0
    if M_units == "SI":
        return c.N_A * c.mu_b_SI                       # A m^2 / mol
    if M_units == "emu/mol":
        return c.N_A * c.mu_b_SI * 1e3                 # 1 A m^2 = 1 J/T = 1e3 erg/G
    raise ValueError(f"M_units must be one of {_M_UNITS}")


def _chi_to_si(chi_units: str, c: MagConstants) -> float:
    if chi_units == "SI":
        return 1.0
    if chi_units == "cgs":
        return 10.0 * c.mu0                             # 4 pi 1e-6 with exact mu_0
    raise ValueError("chi_units must be 'SI' (m^3/mol) or 'cgs' (cm^3/mol)")


class MagResult(NamedTuple):
    """Magnetization (or susceptibility) per field direction and powder averaged.

    Attributes
    ----------
    per_direction : torch.Tensor, shape (..., ndir, nB, nT)
        Values for each grid direction, field and temperature.
    powder : torch.Tensor, shape (..., nB, nT)
        Grid-weighted average over the directions.
    """

    per_direction: torch.Tensor
    powder: torch.Tensor


_MIRROR = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0], dtype=_F64)


def mirror_y(V: torch.Tensor) -> torch.Tensor:
    """Ligand field of the mirror image (y -> -y), ``P V P`` with ``P = diag(1, -1, 1, -1, 1)``.

    This is what DDNSUS/DDNMAG effectively compute (their internal field is
    ``conj(V_m)``); pass ``ddn_mirror=True`` to :func:`magnetization` to
    reproduce DDN's numbers for a general field direction.

    Parameters
    ----------
    V : torch.Tensor, shape (..., 5, 5)
        Real-orbital ligand field (x2-y2, xy, xz, yz, z2), any energy unit.

    Returns
    -------
    torch.Tensor, shape (..., 5, 5)
        The mirrored field, same unit.
    """
    p = _MIRROR.to(V.device)
    return V * p[:, None] * p[None, :]


def moment_core(H0: torch.Tensor, mu: torch.Tensor, field, T, directions, constants: MagConstants = MAG_CONSTANTS,
                chunk: int | None = None) -> torch.Tensor:
    """Thermal average of the moment along the field, in Bohr magnetons per ion.

    Low-level kernel of :func:`magnetization`: ``<-n.mu>_T`` for
    ``H = H0 + mu_B B n.mu``, exact over every state.  The moment is set to 0
    at B = 0 (time reversal).

    Parameters
    ----------
    H0 : torch.Tensor, shape (..., nd, nd)
        Zero-field Hamiltonian in cm^-1.
    mu : torch.Tensor, shape (..., 3, nd, nd)
        Moment operators ``k L + g_e S`` in mu_B.
    field : array_like or torch.Tensor, shape (nB,)
        Field magnitudes in mT.
    T : array_like or torch.Tensor, shape (nT,)
        Temperatures in K, all > 0.
    directions : array_like, shape (ndir, 3)
        Unit field directions.
    constants : MagConstants, default MAG_CONSTANTS
        mu_B and k_B.
    chunk : int, optional
        Directions diagonalized at once (memory); default automatic.

    Returns
    -------
    torch.Tensor, shape (..., ndir, nB, nT), float64
        Moment in mu_B per ion.

    Raises
    ------
    ValueError
        If a temperature is not positive.
    """
    dev = H0.device
    B = torch.as_tensor(field, dtype=_F64, device=dev).reshape(-1)
    T = torch.as_tensor(T, dtype=_F64, device=dev).reshape(-1)
    if torch.any(T <= 0):
        raise ValueError("temperatures must be > 0 K")
    dirs = torch.as_tensor(directions, dtype=_F64, device=dev).reshape(-1, 3)
    beta = 1.0 / (constants.k_b * T)
    nd = H0.shape[-1]
    if chunk is None:
        chunk = max(1, int(2.0e6 // (nd * nd * max(1, len(B)))))
    parts = []
    for s in range(0, len(dirs), chunk):
        zn = torch.einsum("ci,...inm->...cnm", dirs[s:s + chunk].to(_C128), mu)
        hs = H0.unsqueeze(-3).unsqueeze(-3) + (constants.mu_b * B).to(_C128)[:, None, None] * zn.unsqueeze(-3)
        parts.append(thermal_average(hs, -zn.unsqueeze(-3).unsqueeze(-3), beta)[..., 0])
    m = torch.cat(parts, dim=-3)
    return torch.where((B == 0)[:, None], torch.zeros_like(m), m)      # time reversal: M(0) = 0


def _pieces(n, units, k, ddn_mirror, constants: MagConstants, device, params):
    ops = dn_operators(n, device) if not hasattr(n, "lf_real") else n
    f = _u.energy_factor(units, "cm^-1")
    V = params.pop("V", None)
    if V is not None and ddn_mirror:
        V = mirror_y(torch.as_tensor(V, dtype=_F64, device=ops.device))
    H0 = _hamiltonian_cm(ops, f, params.pop("B", 0.0), params.pop("C", 0.0), params.pop("A", 0.0),
                         params.pop("zeta", 0.0), V, k, params.pop("alpha", 0.0), params.pop("slater", None), None,
                         constants.ddn)
    if params:
        raise TypeError(f"unexpected keywords: {sorted(params)}")
    return H0, zeeman_operators(ops, k, constants.ddn)


def magnetization(n, field, T, *, grid: OrientationGrid | None = None, M_units: str = "muB", chi0=0.0,
                  chi_units: str = "SI", units: str = "MHz", k=(1.0, 1.0, 1.0), ddn_mirror: bool = False,
                  constants: MagConstants = MAG_CONSTANTS, chunk: int | None = None, device="cpu",
                  **params) -> MagResult:
    """Molar magnetization of a d^n ion along the field, per direction and powder averaged.

    Parameters
    ----------
    n : int or DnOperators
        Number of d electrons (0 to 10) or cached operators.
    field : array_like or torch.Tensor, shape (nB,)
        Field magnitudes in mT.
    T : array_like or torch.Tensor, shape (nT,)
        Temperatures in K, all > 0.
    grid : OrientationGrid or array_like, optional
        Field directions and weights (:mod:`torchspin.lft.grids`), or direction
        vectors (normalized, equal weights).  Default ``None``: the field
        along z only.
    M_units : {'muB', 'emu/mol', 'SI'}, default 'muB'
        Unit of the result: Bohr magnetons per ion, emu/mol (DDNMAG's unit,
        ``1 N_A mu_B = 5585 emu/mol``) or A m^2/mol.
    chi0 : float or torch.Tensor, default 0.0
        Temperature-independent paramagnetism in ``chi_units``; adds
        ``chi0 * H``.
    chi_units : {'SI', 'cgs'}, default 'SI'
        Unit of ``chi0``: m^3/mol or cm^3/mol.
    units : str, default 'MHz'
        Energy unit of the ligand-field keywords.
    k : array_like or torch.Tensor, shape (3,), default (1.0, 1.0, 1.0)
        Orbital reduction factors.
    ddn_mirror : bool, default False
        Apply :func:`mirror_y` to ``V`` (DDN's convention).
    constants : MagConstants, default MAG_CONSTANTS
        Physical constants.
    chunk : int, optional
        Directions diagonalized at once (memory).
    device : str or torch.device, default 'cpu'
        Device.
    **params
        Ligand-field keywords of :func:`torchspin.lft.ddn.hamiltonian` (``B``,
        ``C``, ``A``, ``zeta``, ``V``, ``alpha``, ``slater``) in ``units``;
        may be batched and carry gradients.

    Returns
    -------
    MagResult
        ``per_direction`` (..., ndir, nB, nT) and ``powder`` (..., nB, nT) in
        ``M_units``.

    Raises
    ------
    ValueError
        For a non-positive temperature, unknown ``M_units`` or (with a nonzero
        ``chi0``) unknown ``chi_units``.
    TypeError
        For an unknown keyword.

    Examples
    --------
    d^1 in a strong octahedral field (cm^-1), field along z:

    >>> from torchspin import lft
    >>> m = lft.ddnmag.magnetization(1, [1000.0, 5000.0], [2.0, 300.0],
    ...                              V=lft.crystal_field(Dq=2000.0), zeta=500.0, units='cm^-1')
    >>> m.per_direction.shape, m.powder.shape
    (torch.Size([1, 2, 2]), torch.Size([2, 2]))
    """
    g = as_grid(grid)
    H0, mu = _pieces(n, units, k, ddn_mirror, constants, device, params)
    m = moment_core(H0, mu, field, T, g.vectors, constants, chunk)
    Bt = torch.as_tensor(field, dtype=_F64, device=m.device).reshape(-1)
    fac = _m_factor(M_units, constants)
    per = fac * m
    chi0 = torch.as_tensor(chi0, dtype=_F64, device=m.device)
    if chi0.ndim or float(chi0) != 0.0:
        # M_tip (A m^2/mol) = chi0_SI * H = chi0_SI * B / mu_0; then to M_units
        tip = chi0 * _chi_to_si(chi_units, constants) * (Bt * 1e-3) / constants.mu0
        per = per + (tip * fac / (constants.N_A * constants.mu_b_SI))[..., None, :, None]
    w = torch.as_tensor(g.weights, dtype=_F64, device=m.device)
    return MagResult(per, torch.einsum("d,...dbt->...bt", w, per))


def fit(n: int, parameters, model, field, T, values, sigma=0.01, *, M_units: str = "muB", grid=None,
        units: str = "MHz", **kw):
    """DDNMAG fit: ligand-field parameters to powder M(field_i, T_i), with the exact Jacobian.

    Thin front end to :class:`torchspin.lft.fit.JointFit` with one
    :class:`~torchspin.lft.fit.Magnetization` data set.

    Parameters
    ----------
    n : int
        Number of d electrons.
    parameters : sequence of FitParameter
        Fit parameters (:class:`torchspin.lft.fit.FitParameter`).
    model : callable
        Maps a dict of named scalar tensors to keywords of
        :func:`torchspin.lft.ddn.hamiltonian` (in ``units``).
    field, T, values : array_like
        Data points: fields in mT, temperatures in K, magnetizations in
        ``M_units`` (broadcast together).
    sigma : float or array_like, default 0.01
        Uncertainties of ``values`` (``M_units``).
    M_units : {'muB', 'emu/mol', 'SI'}, default 'muB'
        Unit of ``values``.
    grid : OrientationGrid, optional
        Powder grid; default the exact Gauss hemisphere, n = 6.
    units : str, default 'MHz'
        Energy unit of the model's keywords.
    **kw
        Passed to :meth:`torchspin.lft.fit.JointFit.fit` (and on to
        :func:`scipy.optimize.least_squares`).

    Returns
    -------
    torchspin.lft.fit.JointOutcome
    """
    from .fit import JointFit, Magnetization
    from .grids import gauss_hemisphere

    block = Magnetization(field, T, values, sigma, M_units=M_units, grid=gauss_hemisphere(6) if grid is None else grid)
    return JointFit(n, parameters, model, [block], units=units).fit(**kw)
