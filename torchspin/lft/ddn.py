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

Degenerate levels are handled by ``torchspin.lft._linalg``: the gradient of a degenerate
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
    """Physical constants used by the d^n Hamiltonian.

    Attributes
    ----------
    mu_b : float
        Bohr magneton over h c, in cm^-1 per mT.
    g_e : float
        Free-electron g value (dimensionless).
    k_b : float
        Boltzmann constant over h c, in cm^-1 per K.
    """

    mu_b: float
    g_e: float
    k_b: float


#: Derived from ``torchspin.constants``.
TORCHSPIN_CONSTANTS = Constants(mu_b=BMAGN * 1e-3 / (PLANCK * CLIGHT), g_e=GFREE, k_b=BOLTZMANN / (PLANCK * CLIGHT))


def _t(x, device) -> torch.Tensor:
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=_F64, device=device)


def _scal(x, device) -> torch.Tensor:
    return _t(x, device)[..., None, None]


def _ops(n_or_ops, device) -> DnOperators:
    return n_or_ops if isinstance(n_or_ops, DnOperators) else dn_operators(n_or_ops, device)


def zeeman_operators(n, k=(1.0, 1.0, 1.0), constants: Constants = TORCHSPIN_CONSTANTS, device="cpu") -> torch.Tensor:
    """Magnetic moment operators in Bohr magnetons, ``k_i L_i + g_e S_i``.

    Parameters
    ----------
    n : int or DnOperators
        Number of d electrons (0 to 10) or a cached operator set.
    k : array_like or torch.Tensor, shape (..., 3), default (1.0, 1.0, 1.0)
        Orbital reduction factors along x, y, z (dimensionless).
    constants : torchspin.lft.ddn.Constants, default TORCHSPIN_CONSTANTS
        Supplies ``g_e``.
    device : str or torch.device, default 'cpu'
        Device of the operators (ignored when ``n`` is a ``DnOperators``).

    Returns
    -------
    torch.Tensor, shape (..., 3, nd, nd), complex128
        ``mu_x, mu_y, mu_z`` over mu_B in the Slater-determinant basis,
        ``nd = C(10, n)``.

    Raises
    ------
    ValueError
        If ``n`` is not an integer from 0 to 10.

    Examples
    --------
    >>> from torchspin import lft
    >>> lft.ddn.zeeman_operators(1).shape
    torch.Size([3, 10, 10])
    """
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
    """Complex Hermitian d^n Hamiltonian in the full determinant basis.

    Builds ``A M_A + B M_B + C M_C + alpha L^2 + sum_ab V[a, b] E_real[a, b] +
    zeta sum_i l_i.s_i + mu_B field . (k L + g_e S)`` (module docstring).  Every
    parameter may be a tensor with leading batch dimensions and gradients.

    Parameters
    ----------
    n : int or DnOperators
        Number of d electrons (0 to 10) or a cached operator set.
    B, C, A : float or torch.Tensor, default 0.0
        Racah parameters, in ``units``.  ``A`` only shifts all levels.
    zeta : float or torch.Tensor, default 0.0
        One-electron spin-orbit coupling constant, in ``units``.
    V : array_like or torch.Tensor, shape (..., 5, 5), optional
        Real symmetric ligand-field matrix in the real-orbital basis
        (x2-y2, xy, xz, yz, z2), in ``units`` (see
        :mod:`torchspin.lft.ligand_field`).  Default ``None``: no ligand field.
    k : array_like or torch.Tensor, shape (..., 3), default (1.0, 1.0, 1.0)
        Orbital reduction factors (dimensionless); enter only the Zeeman term.
    alpha : float or torch.Tensor, default 0.0
        Trees correction (coefficient of L^2), in ``units``.
    slater : tuple of 3, optional
        ``(F0, F2, F4)`` Slater integrals in ``units``; replaces ``A, B, C``
        when given.
    field : array_like or torch.Tensor, shape (..., 3), optional
        Magnetic field vector in mT.  Default ``None``: zero field.
    units : str, default 'MHz'
        Energy unit of the inputs and of the result: ``'MHz'``, ``'GHz'``,
        ``'Hz'``, ``'cm^-1'``, ``'K'``, ``'eV'``, ``'meV'`` or ``'J'``.
    constants : torchspin.lft.ddn.Constants, default TORCHSPIN_CONSTANTS
        mu_B and g_e of the Zeeman term.
    device : str or torch.device, default 'cpu'
        Device of the operators.

    Returns
    -------
    torch.Tensor, shape (..., nd, nd), complex128
        Hamiltonian in ``units``, ``nd = C(10, n)``.

    Raises
    ------
    ValueError
        If ``n`` is not an integer from 0 to 10 or ``units`` is unknown.

    Examples
    --------
    >>> from torchspin import lft
    >>> H = lft.ddn.hamiltonian(2, B=800.0, C=3200.0, units='cm^-1')
    >>> H.shape, H.dtype
    (torch.Size([45, 45]), torch.complex128)
    """
    ops = _ops(n, device)
    f = _u.energy_factor(units, "cm^-1")
    return _hamiltonian_cm(ops, f, B, C, A, zeta, V, k, alpha, slater, field, constants) / f


def levels(n, *, relative: bool = False, units: str = "MHz", **kw) -> torch.Tensor:
    """Ascending energy levels of d^n.

    Parameters
    ----------
    n : int or DnOperators
        Number of d electrons (0 to 10) or a cached operator set.
    relative : bool, default False
        If True, subtract the lowest level (ground state at 0).
    units : str, default 'MHz'
        Energy unit of the inputs and the result (as :func:`hamiltonian`).
    **kw
        Hamiltonian keywords of :func:`hamiltonian` (``B``, ``C``, ``A``,
        ``zeta``, ``V``, ``k``, ``alpha``, ``slater``, ``field``,
        ``constants``, ``device``).

    Returns
    -------
    torch.Tensor, shape (..., nd), float64
        Eigenvalues in ``units``, ascending, ``nd = C(10, n)``.

    Notes
    -----
    The gradient of a degenerate eigenvalue is that of its level's mean
    (exact for symmetry-protected degeneracies, a symmetric subgradient at an
    accidental crossing).

    Examples
    --------
    d^3 in an octahedral field (cm^-1); the ground 4A2 level is fourfold:

    >>> from torchspin import lft
    >>> E = lft.ddn.levels(3, B=800.0, C=3200.0, V=lft.crystal_field(Dq=1500.0),
    ...                    units='cm^-1', relative=True)
    >>> E.shape, [round(float(x), 6) for x in E[:4]]
    (torch.Size([120]), [0.0, 0.0, 0.0, 0.0])
    """
    E = eigvalsh_gauge(hamiltonian(n, units=units, **kw))
    return E - E[..., :1] if relative else E


class GTensor(NamedTuple):
    """Effective spin-1/2 g tensor of the lowest doublet (returned by :func:`g_tensor`).

    Attributes
    ----------
    ggT : torch.Tensor, shape (..., 3, 3)
        ``g g^T`` in the molecular frame.
    principal : torch.Tensor, shape (..., 3)
        Principal g values (square roots of the eigenvalues of ``ggT``),
        ascending.
    g : torch.Tensor, shape (..., 3, 3)
        ``(g g^T)^(1/2)``, the symmetric molecular-frame g (exact gradients at
        axial g).
    gap : torch.Tensor, shape (...)
        Energy from the doublet to the next state, in the ``units`` of the call.
        Gradients of g scale as 1/gap: check that it is large.
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
    """Zero-field g tensor of the lowest doublet of d^n.

    Computes ``(g g^T)_ij = 2 tr(P mu_i P mu_j)`` with ``P`` the projector onto
    the two lowest states and ``mu = k L + g_e S``.

    Parameters
    ----------
    n : int or DnOperators
        Number of d electrons (0 to 10) or a cached operator set.
    k : array_like or torch.Tensor, shape (..., 3), default (1.0, 1.0, 1.0)
        Orbital reduction factors.
    units : str, default 'MHz'
        Energy unit of the Hamiltonian keywords and of ``gap``.
    constants : torchspin.lft.ddn.Constants, default TORCHSPIN_CONSTANTS
        g_e of the moment operator.
    device : str or torch.device, default 'cpu'
        Device of the operators.
    **kw
        Hamiltonian keywords of :func:`hamiltonian` except ``field`` (``B``,
        ``C``, ``A``, ``zeta``, ``V``, ``alpha``, ``slater``).

    Returns
    -------
    GTensor
        ``ggT``, ``principal``, ``g`` (dimensionless) and ``gap`` (``units``).

    Raises
    ------
    ValueError
        If ``field`` is given (this is the zero-field g).
    TypeError
        For an unknown keyword.

    Notes
    -----
    Meaningful for odd n (Kramers doublets) or any isolated ground doublet.

    Examples
    --------
    d^9 (Cu(II)) in an elongated octahedron, energies in cm^-1:

    >>> import torch
    >>> from torchspin import lft
    >>> V = torch.diag(torch.tensor([9900., -2100., -5100., -5100., 2400.], dtype=torch.float64))
    >>> res = lft.ddn.g_tensor(9, zeta=830.0, V=V, k=(0.8, 0.8, 0.8), units='cm^-1')
    >>> [round(float(x), 3) for x in res.principal]
    [2.071, 2.071, 2.42]
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
    """A d^n ion with cached operators.

    ``DnSystem(n).levels(...)``, ``.g_tensor(...)``, ``.hamiltonian(...)`` and
    ``.zeeman_operators(k)`` call the module functions with the cached
    operators; ``units`` and ``constants`` set here are the defaults of every
    call (override per call).

    Parameters
    ----------
    n : int
        Number of d electrons, 0 to 10.
    units : str, default 'MHz'
        Default energy unit.
    constants : torchspin.lft.ddn.Constants, default TORCHSPIN_CONSTANTS
        Default constants.
    device : str or torch.device, default 'cpu'
        Device of the operators.

    Attributes
    ----------
    size : int
        Basis size ``C(10, n)``.

    Examples
    --------
    >>> from torchspin import lft
    >>> ion = lft.ddn.DnSystem(8, units='cm^-1')
    >>> ion.size, ion.levels(B=900.0, C=3600.0).shape
    (45, torch.Size([45]))
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
