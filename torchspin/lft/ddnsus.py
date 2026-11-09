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
    """chi = M/H, per direction (..., ndir, nB, nT) and powder (..., nB, nT); ``chi0`` (TIP) in chi_units."""
    Bt = torch.as_tensor(field, dtype=_F64).reshape(-1)
    if torch.any(Bt <= 0):
        raise ValueError("chi = M/H needs field > 0")
    m = magnetization(n, Bt, T, grid=grid, M_units="SI", units=units, k=k, ddn_mirror=ddn_mirror,
                      constants=constants, chunk=chunk, device=device, **params)
    scale = constants.mu0 / (Bt.to(m.powder.device) * 1e-3) / _chi_to_si(chi_units, constants)   # A m^2/mol -> chi
    chi0 = torch.as_tensor(chi0, dtype=_F64, device=m.powder.device)
    return MagResult(m.per_direction * scale[:, None] + chi0, m.powder * scale[:, None] + chi0)


def chi_T(n, T, **kw) -> MagResult:
    """chi T (chi_units x K); keywords as :func:`susceptibility`."""
    chi = susceptibility(n, T, **kw)
    Tt = torch.as_tensor(T, dtype=_F64, device=chi.powder.device).reshape(-1)
    return MagResult(chi.per_direction * Tt, chi.powder * Tt)


def curie_constant(chi_units: str = "SI", constants: MagConstants = MAG_CONSTANTS) -> float:
    """N_A mu_0 mu_B^2 / (3 k_B) in chi_units x K (SI: 1.5714e-6 m^3 K/mol; cgs: 0.12505 cm^3 K/mol).

    chi T = curie_constant * g^2 S(S+1) for a free spin.
    """
    k_B = constants.k_b * 1.0 / (constants.mu_b / (constants.mu_b_SI * 1e-3))   # J/K from cm^-1/K
    c_si = constants.N_A * constants.mu0 * constants.mu_b_SI ** 2 / (3.0 * k_B)
    return c_si / _chi_to_si(chi_units, constants)


def mu_eff(chiT, chi_units: str = "SI", constants: MagConstants = MAG_CONSTANTS):
    """Effective moment (Bohr magnetons) from chi T: sqrt(chi T / curie_constant) (2.828 sqrt(chi T) in cgs)."""
    c = curie_constant(chi_units, constants)
    if isinstance(chiT, torch.Tensor):
        return torch.sqrt(chiT / c)
    return math.sqrt(chiT / c)


def fit(n: int, parameters, model, T, values, sigma=0.01, *, quantity: str = "chiT", field=100.0,
        chi_units: str = "cgs", grid=None, chi0: float = 0.0, units: str = "MHz", **kw):
    """DDNSUS fit: ligand-field parameters to powder chi(T) or chi T, with the exact Jacobian.

    Thin front end to :class:`torchspin.lft.fit.JointFit` with one
    :class:`~torchspin.lft.fit.Susceptibility` block (default grid: exact Gauss hemisphere, n = 6).
    """
    from .fit import JointFit, Susceptibility
    from .grids import gauss_hemisphere

    block = Susceptibility(T, values, sigma, field=field, quantity=quantity, chi_units=chi_units,
                           grid=gauss_hemisphere(6) if grid is None else grid, chi0=chi0)
    return JointFit(n, parameters, model, [block], units=units).fit(**kw)
