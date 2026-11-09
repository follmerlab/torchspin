"""DDN: differentiable d^n ligand-field Hamiltonian, levels, and doublet g tensor.

    H = A M_A + B M_B + C M_C + alpha L^2 + sum_ab V[a, b] E_real[a, b] + zeta sum_i l_i.s_i
        + mu_B sum_i B_i (k_i L_i + g_e S_i)

in the full basis of C(10, n) Slater determinants (Telser's DDN, via telser-tools).  Every
parameter may be a tensor carrying gradients, with leading batch dimensions.

Units
-----
Energy-valued inputs (``B``, ``C``, ``A``, ``zeta``, ``alpha``, ``slater``, the ligand field
``V``) and energy-valued outputs are in ``units`` (default ``'MHz'``, torchspin's unit; use
``units='cm^-1'`` to work in wavenumbers, or ``'GHz'``, ``'K'``, ``'eV'``).  The magnetic field
is in mT.  The orbital reduction ``k`` is dimensionless.  Internally everything is cm^-1.

    >>> from torchspin import lft
    >>> V = lft.crystal_field(Dq=1200.0)                       # cm^-1
    >>> E = lft.ddn.levels(3, B=800.0, C=3200.0, V=V, zeta=200.0, units='cm^-1')
    >>> g = lft.ddn.g_tensor(9, zeta=830.0, V=lft.aom(lft.octahedron(6000.0, 1000.0)),
    ...                      k=(0.8, 0.8, 0.8), units='cm^-1')

Degenerate levels are handled by :mod:`torchspin.lft._linalg`: the gradient of a degenerate
eigenvalue is that of its level's mean, and the doublet g is differentiated through the
projector onto the doublet, which is exact at the Kramers degeneracy.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from torchspin.constants import BMAGN, BOLTZMANN, CLIGHT, GFREE, PLANCK

from . import units as _u
from ._linalg import eigvalsh_gauge, ground_manifold, sqrtm_psd
from ._operators import DnOperators, dn_operators

__all__ = ["Constants", "TORCHSPIN_CONSTANTS", "DnSystem", "GTensor", "g_tensor", "hamiltonian", "levels",
           "zeeman_operators"]

_F64, _C128 = torch.float64, torch.complex128


class Constants(NamedTuple):
    """mu_B/(h c) in cm^-1 per mT, free-electron g, k_B/(h c) in cm^-1 per K."""

    mu_b: float
    g_e: float
    k_b: float


#: Derived from :mod:`torchspin.constants`.
TORCHSPIN_CONSTANTS = Constants(mu_b=BMAGN * 1e-3 / (PLANCK * CLIGHT), g_e=GFREE, k_b=BOLTZMANN / (PLANCK * CLIGHT))


def _t(x, device) -> torch.Tensor:
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=_F64, device=device)


def _scal(x, device) -> torch.Tensor:
    return _t(x, device)[..., None, None]


def _ops(n_or_ops, device) -> DnOperators:
    return n_or_ops if isinstance(n_or_ops, DnOperators) else dn_operators(n_or_ops, device)


def zeeman_operators(n, k=(1.0, 1.0, 1.0), constants: Constants = TORCHSPIN_CONSTANTS, device="cpu") -> torch.Tensor:
    """mu_i / mu_B = k_i L_i + g_e S_i, shape (..., 3, nd, nd); k is (..., 3)."""
    ops = _ops(n, device)
    k = _t(k, ops.device)
    return k[..., :, None, None].to(_C128) * ops.L + constants.g_e * ops.S


def _hamiltonian_cm(ops: DnOperators, f: float, B, C, A, zeta, V, k, alpha, slater, field,
                    constants: Constants) -> torch.Tensor:
    """H in cm^-1; energy inputs are multiplied by f (their unit -> cm^-1)."""
    dev = ops.device
    if slater is not None:
        f0, f2, f4 = (f * _scal(x, dev) for x in slater)
        h = f0 * ops.coulomb[0] + f2 * ops.coulomb[1] + f4 * ops.coulomb[2]
    else:
        h = f * (_scal(A, dev) * ops.racah[0] + _scal(B, dev) * ops.racah[1] + _scal(C, dev) * ops.racah[2])
    h = h + f * (_scal(alpha, dev) * ops.L2 + _scal(zeta, dev) * ops.so)
    h = h.to(_C128)
    if V is not None:
        h = h + f * torch.einsum("...ab,abij->...ij", _t(V, dev).to(_C128), ops.lf_real)
    if field is not None:
        mu = zeeman_operators(ops, k, constants)
        h = h + constants.mu_b * torch.einsum("...i,...inm->...nm", _t(field, dev).to(_C128), mu)
    return h


def hamiltonian(n, *, B=0.0, C=0.0, A=0.0, zeta=0.0, V=None, k=(1.0, 1.0, 1.0), alpha=0.0, slater=None,
                field=None, units: str = "MHz", constants: Constants = TORCHSPIN_CONSTANTS,
                device="cpu") -> torch.Tensor:
    """Complex Hermitian H (..., nd, nd) of d^n, in ``units``.

    ``n`` is the electron count (or a cached operator set).  ``slater = (F0, F2, F4)`` replaces
    Racah ``A, B, C``.  ``V`` is the real 5x5 ligand field (..., 5, 5) (see
    :mod:`torchspin.lft.ligand_field`).  ``field`` (..., 3) in mT.
    """
    ops = _ops(n, device)
    f = _u.energy_factor(units, "cm^-1")
    return _hamiltonian_cm(ops, f, B, C, A, zeta, V, k, alpha, slater, field, constants) / f


def levels(n, *, relative: bool = False, units: str = "MHz", **kw) -> torch.Tensor:
    """Ascending energies (..., nd) in ``units``; ``relative=True`` puts the ground state at 0.

    Keywords as :func:`hamiltonian`.  The gradient of a degenerate eigenvalue is that of its
    level's mean (exact for symmetry-protected degeneracies, a symmetric subgradient at an
    accidental crossing).
    """
    E = eigvalsh_gauge(hamiltonian(n, units=units, **kw))
    return E - E[..., :1] if relative else E


class GTensor(NamedTuple):
    """Effective-spin-1/2 g of the lowest Kramers doublet.

    ``ggT`` (..., 3, 3) = g g^T; ``principal`` (..., 3) ascending; ``g`` = (g g^T)^(1/2), the
    symmetric molecular-frame g (exact gradients at axial g); ``gap`` from the doublet to the
    next state, in the units asked for (gradients scale as 1/gap: check it).
    """

    ggT: torch.Tensor
    principal: torch.Tensor
    g: torch.Tensor
    gap: torch.Tensor


def _g_from(H: torch.Tensor, mu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    man = ground_manifold(H, 2)
    pm = man.projector.unsqueeze(-3) @ mu
    ggT = 2.0 * torch.einsum("...inm,...jmn->...ij", pm, pm).real
    ggT = 0.5 * (ggT + ggT.transpose(-2, -1))
    principal = torch.sqrt(torch.clamp(eigvalsh_gauge(ggT), min=0.0))
    return ggT, principal, sqrtm_psd(ggT), man.gap


def g_tensor(n, *, k=(1.0, 1.0, 1.0), units: str = "MHz", constants: Constants = TORCHSPIN_CONSTANTS,
             device="cpu", **kw) -> GTensor:
    """Zero-field g of the lowest doublet: (g g^T)_ij = 2 tr(P mu_i P mu_j), P its projector.

    Keywords as :func:`hamiltonian` (without ``field``); only ``gap`` carries ``units``.
    Meaningful for odd n (Kramers doublets) or any isolated ground doublet.
    """
    if kw.get("field") is not None:
        raise ValueError("g_tensor is the zero-field doublet g; do not pass field")
    ops = _ops(n, device)
    f = _u.energy_factor(units, "cm^-1")
    H = _hamiltonian_cm(ops, f, kw.pop("B", 0.0), kw.pop("C", 0.0), kw.pop("A", 0.0), kw.pop("zeta", 0.0),
                        kw.pop("V", None), k, kw.pop("alpha", 0.0), kw.pop("slater", None), None, constants)
    if kw:
        raise TypeError(f"unexpected keywords: {sorted(kw)}")
    ggT, principal, g, gap = _g_from(H, zeeman_operators(ops, k, constants))
    return GTensor(ggT, principal, g, gap / f)


class DnSystem:
    """A d^n ion with cached operators: ``DnSystem(n).levels(...)``, ``.g_tensor(...)``, ``.hamiltonian(...)``.

    ``units`` set here is the default for every call (override per call).
    """

    def __init__(self, n: int, *, units: str = "MHz", constants: Constants = TORCHSPIN_CONSTANTS, device="cpu"):
        self.n = int(n)
        self.units = _u.canonical_energy_unit(units)
        self.constants = constants
        self.ops = dn_operators(self.n, device)

    @property
    def size(self) -> int:
        return self.ops.size

    def _kw(self, kw: dict) -> dict:
        kw.setdefault("units", self.units)
        kw.setdefault("constants", self.constants)
        return kw

    def hamiltonian(self, **kw) -> torch.Tensor:
        return hamiltonian(self.ops, **self._kw(kw))

    def levels(self, **kw) -> torch.Tensor:
        return levels(self.ops, **self._kw(kw))

    def g_tensor(self, **kw) -> GTensor:
        return g_tensor(self.ops, **self._kw(kw))

    def zeeman_operators(self, k=(1.0, 1.0, 1.0)) -> torch.Tensor:
        return zeeman_operators(self.ops, k, self.constants)
