"""DDNSUS: differentiable molar susceptibility chi(T), chi T and mu_eff of a d^n ion.

Telser's definition: chi = M / H at a finite measuring field (DDNSUS; ``method="exact"``),
with M from :func:`torchspin.lft.ddnmag.magnetization`.  In the linear regime (mu_B B << k_B T
and below any field-induced crossing) this equals the differential susceptibility dM/dH that
:func:`torchspin.curry` returns; at high field / low temperature the two differ, and M/H is
what a SQUID reports.

Units: ``T`` in K, measuring ``field`` in mT (default 100 mT = DDN's 1000 G), chi in
``chi_units`` = ``'SI'`` (m^3/mol, as curry, default) or ``'cgs'`` (cm^3/mol), chi T in
chi_units x K.  Ligand-field parameters in ``units`` as in :mod:`torchspin.lft.ddn`.
"""

from __future__ import annotations

import math

import torch

from .ddnmag import MAG_CONSTANTS, MagConstants, MagResult, _chi_to_si, magnetization

__all__ = ["susceptibility", "chi_T", "mu_eff", "curie_constant", "fit"]

_F64 = torch.float64


def susceptibility(n, T, *, field=100.0, grid=None, chi_units: str = "SI", chi0=0.0, units: str = "MHz",
                   k=(1.0, 1.0, 1.0), ddn_mirror: bool = False, constants: MagConstants = MAG_CONSTANTS,
                   chunk: int | None = None, device="cpu", **params) -> MagResult:
    """Molar susceptibility chi = M/H of a d^n ion, per direction and powder averaged.

    Telser's DDNSUS definition at a finite measuring field (module docstring).

    Parameters
    ----------
    n : int or DnOperators
        Number of d electrons (0 to 10) or cached operators.
    T : array_like or torch.Tensor, shape (nT,)
        Temperatures in K, all > 0.
    field : float or array_like, default 100.0
        Measuring field(s) in mT (100 mT = DDN's 1000 G), all > 0.
    grid : OrientationGrid or array_like, optional
        Field directions (:mod:`torchspin.lft.grids`); default the field along z.
    chi_units : {'SI', 'cgs'}, default 'SI'
        m^3/mol (as :func:`torchspin.curry`) or cm^3/mol.
    chi0 : float or torch.Tensor, default 0.0
        Temperature-independent paramagnetism in ``chi_units``, added.
    units : str, default 'MHz'
        Energy unit of the ligand-field keywords.
    k : array_like, shape (3,), default (1.0, 1.0, 1.0)
        Orbital reduction factors.
    ddn_mirror : bool, default False
        Use DDN's mirror-image field (:func:`torchspin.lft.ddnmag.mirror_y`).
    constants : MagConstants, default MAG_CONSTANTS
        Physical constants.
    chunk : int, optional
        Directions diagonalized at once.
    device : str or torch.device, default 'cpu'
        Device.
    **params
        Ligand-field keywords of :func:`torchspin.lft.ddn.hamiltonian`.

    Returns
    -------
    MagResult
        ``per_direction`` (..., ndir, nB, nT) and ``powder`` (..., nB, nT) in
        ``chi_units``.

    Raises
    ------
    ValueError
        If a field or temperature is not positive or ``chi_units`` is unknown.

    Examples
    --------
    >>> from torchspin import lft
    >>> chi = lft.ddnsus.susceptibility(1, [2.0, 300.0], grid=lft.grids.gauss_hemisphere(2),
    ...                                 V=lft.crystal_field(Dq=2000.0), zeta=500.0,
    ...                                 units='cm^-1', chi_units='cgs')
    >>> chi.powder.shape
    torch.Size([1, 2])
    """
    Bt = torch.as_tensor(field, dtype=_F64).reshape(-1)
    if torch.any(Bt <= 0):
        raise ValueError("chi = M/H needs field > 0")
    m = magnetization(n, Bt, T, grid=grid, M_units="SI", units=units, k=k, ddn_mirror=ddn_mirror,
                      constants=constants, chunk=chunk, device=device, **params)
    scale = constants.mu0 / (Bt.to(m.powder.device) * 1e-3) / _chi_to_si(chi_units, constants)   # A m^2/mol -> chi
    chi0 = torch.as_tensor(chi0, dtype=_F64, device=m.powder.device)
    return MagResult(m.per_direction * scale[:, None] + chi0, m.powder * scale[:, None] + chi0)


def chi_T(n, T, **kw) -> MagResult:
    """chi T, per direction and powder averaged, in ``chi_units`` x K.

    Parameters
    ----------
    n : int or DnOperators
        Number of d electrons or cached operators.
    T : array_like or torch.Tensor, shape (nT,)
        Temperatures in K.
    **kw
        Keywords of :func:`susceptibility`.

    Returns
    -------
    MagResult
    """
    chi = susceptibility(n, T, **kw)
    Tt = torch.as_tensor(T, dtype=_F64, device=chi.powder.device).reshape(-1)
    return MagResult(chi.per_direction * Tt, chi.powder * Tt)


def curie_constant(chi_units: str = "SI", constants: MagConstants = MAG_CONSTANTS) -> float:
    """Curie constant ``N_A mu_0 mu_B^2 / (3 k_B)``.

    ``chi T = curie_constant * g^2 S(S+1)`` for a free spin.

    Parameters
    ----------
    chi_units : {'SI', 'cgs'}, default 'SI'
        Unit of chi.
    constants : MagConstants, default MAG_CONSTANTS
        Physical constants.

    Returns
    -------
    float
        In ``chi_units`` x K: about 1.5714e-6 m^3 K/mol (SI) or 0.12505
        cm^3 K/mol (cgs).

    Examples
    --------
    >>> from torchspin import lft
    >>> round(lft.ddnsus.curie_constant('cgs'), 5)
    0.12505
    """
    k_B = constants.k_b * 1.0 / (constants.mu_b / (constants.mu_b_SI * 1e-3))   # J/K from cm^-1/K
    c_si = constants.N_A * constants.mu0 * constants.mu_b_SI ** 2 / (3.0 * k_B)
    return c_si / _chi_to_si(chi_units, constants)


def mu_eff(chiT, chi_units: str = "SI", constants: MagConstants = MAG_CONSTANTS):
    """Effective moment in Bohr magnetons from chi T, ``sqrt(chiT / curie_constant)``.

    In cgs this is ``2.828 sqrt(chi T)``.

    Parameters
    ----------
    chiT : float or torch.Tensor
        chi T in ``chi_units`` x K.
    chi_units : {'SI', 'cgs'}, default 'SI'
        Unit of ``chiT``.
    constants : MagConstants, default MAG_CONSTANTS
        Physical constants.

    Returns
    -------
    float or torch.Tensor
        mu_eff in mu_B (a tensor for tensor input).

    Examples
    --------
    >>> from torchspin import lft
    >>> round(lft.ddnsus.mu_eff(0.375, 'cgs'), 3)
    1.732
    """
    c = curie_constant(chi_units, constants)
    if isinstance(chiT, torch.Tensor):
        return torch.sqrt(chiT / c)
    return math.sqrt(chiT / c)


def fit(n: int, parameters, model, T, values, sigma=0.01, *, quantity: str = "chiT", field=100.0,
        chi_units: str = "cgs", grid=None, chi0: float = 0.0, units: str = "MHz", **kw):
    """DDNSUS fit: ligand-field parameters to powder chi(T) or chi T, with the exact Jacobian.

    Thin front end to :class:`torchspin.lft.fit.JointFit` with one
    :class:`~torchspin.lft.fit.Susceptibility` data set.

    Parameters
    ----------
    n : int
        Number of d electrons.
    parameters : sequence of FitParameter
        Fit parameters (:class:`torchspin.lft.fit.FitParameter`).
    model : callable
        Maps a dict of named scalar tensors to keywords of
        :func:`torchspin.lft.ddn.hamiltonian` (in ``units``).
    T, values : array_like
        Temperatures in K and the measured chi or chi T.
    sigma : float or array_like, default 0.01
        Uncertainties of ``values``.
    quantity : {'chiT', 'chi'}, default 'chiT'
        What ``values`` are.
    field : float, default 100.0
        Measuring field in mT.
    chi_units : {'SI', 'cgs'}, default 'cgs'
        Unit of ``values`` (note: cgs here, SI in :func:`susceptibility`).
    grid : OrientationGrid, optional
        Powder grid; default the exact Gauss hemisphere, n = 6.
    chi0 : float, default 0.0
        Fixed temperature-independent paramagnetism in ``chi_units``.
    units : str, default 'MHz'
        Energy unit of the model's keywords.
    **kw
        Passed to :meth:`torchspin.lft.fit.JointFit.fit`.

    Returns
    -------
    torchspin.lft.fit.JointOutcome
    """
    from .fit import JointFit, Susceptibility
    from .grids import gauss_hemisphere

    block = Susceptibility(T, values, sigma, field=field, quantity=quantity, chi_units=chi_units,
                           grid=gauss_hemisphere(6) if grid is None else grid, chi0=chi0)
    return JointFit(n, parameters, model, [block], units=units).fit(**kw)
