"""DPN: differentiable complete-basis p^n ligand-field Hamiltonian, levels, and doublet g tensor.

    H = A n(n-1)/2 + F2 (c2 + n(n-1)/25) + alpha L^2 + sum_ab V[a, b] E_real[a, b]
        + zeta sum_i l_i.s_i + mu_B B . (k L + g_e S)

in the full basis of C(6, n) Slater determinants (Telser's DPN, via telser-tools), n = 0..6
(no hole formalism: p^n and p^(6-n) are computed directly).  Every parameter may be a tensor
carrying gradients, with leading batch dimensions.

``F2`` is the Slater integral F^2 (DPN's "F2", not the Condon-Shortley F_2 = F^2/25).  The
diagonal is measured from the configuration's e-e barycenter: ``A`` = F^0 - 2 F^2/25 (the
average repulsion of one p-p pair) defaults to 0, which reproduces DPN's absolute energies.
For p^2: 3P = -3/25 F2, 1D = +3/25 F2, 1S = +12/25 F2.

Ligand fields are real symmetric 3x3 matrices in the real-orbital basis ``(px, py, pz)``
(DPN's order, geometric phase convention): :func:`crystal_field` (Ds, Dr),
:func:`orbital_energies`, the AOM (:func:`aom_from_angles`, :func:`aom_from_frames`,
:func:`aom_isotropic`, :func:`aom`), and :func:`to_m_basis`.  These are p-shell constructors;
``torchspin.lft.aom`` and ``torchspin.lft.crystal_field`` are the d-shell ones.

Units
-----
Energy-valued inputs (``F2``, ``zeta``, ``alpha``, ``A``, the ligand field ``V`` / ``V_m``)
and energy-valued outputs are in ``units`` (default ``'MHz'``; ``'cm^-1'``, ``'GHz'``, ``'K'``,
``'eV'`` via :mod:`torchspin.lft.units`).  The magnetic field is in mT.  The orbital reduction
``k`` is dimensionless.  AOM bonding parameters pass straight through to the matrix (it is in
their unit).  Angles in radians (``degrees=True`` for Telser-style input).  Internally cm^-1.

    >>> from torchspin.lft import dpn
    >>> E = dpn.levels(2, F2=46000.0, zeta=28.0, units='cm^-1', relative=True)   # C I 2p2
    >>> V = dpn.aom_isotropic([[0.0, 0.0, 1.0]], 1000.0)                            # one axial sigma donor
    >>> g = dpn.g_tensor(1, zeta=300.0, V=V, units='cm^-1')

DPN's own matrix
----------------
:func:`dpn_hamiltonian` reproduces what the compiled DPN / DPNFIT diagonalize: the hole route
for n > 3 (with -V and -|zeta|), the mirror-image AOM for p^1/p^2, the dropped <+-1|V|0>
elements of the AOM (finding DPN-01), DPN's vs DPNFIT's hole signs, and DPNFIT's Zeeman defect
(DPNFIT-01).  It defaults to DPN's own constants (:data:`DPN_CONSTANTS`).

Not ported: the DPN input reader, report and term-basis printout, and DPN's EPR / optical
transition probabilities (phase dependent).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np
import torch

from . import units as _u
from ._engines import pn as _pn
from ._linalg import eigvalsh_gauge
from .ddn import TORCHSPIN_CONSTANTS, Constants, GTensor, _g_from
from .ligand_field import Ligand, rotation

__all__ = ["DPN_CONSTANTS", "TORCHSPIN_CONSTANTS", "Constants", "GTensor", "PnOperators", "PnSystem", "aom",
           "aom_from_angles", "aom_from_frames", "aom_isotropic", "crystal_field", "dpn_hamiltonian", "field_vector",
           "g_tensor", "hamiltonian", "internal_m_matrix", "levels", "orbital_energies", "pn_operators",
           "to_m_basis", "zeeman_operators"]

_F64, _C128 = torch.float64, torch.complex128

#: DPN's own constants (DPN.F ll. 69-71, DPNIO.F l. 41; the same as DDN's), mu_B in cm^-1 per mT.
DPN_CONSTANTS = Constants(mu_b=4.6686437e-4, g_e=2.002319304386, k_b=6.9503877e-1)


def _t(x, device=None) -> torch.Tensor:
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=_F64, device=device)


def _scal(x, device) -> torch.Tensor:
    return _t(x, device)[..., None, None]


# ------------------------------------------------------------------------------ operators
@dataclass(frozen=True, eq=False)
class PnOperators:
    """Torch constants of p^n (nd = C(6, n) determinants).

    ``pairs``          n(n-1)/2 (coefficient of A on the diagonal)
    ``coulomb_bary``   (nd, nd) float64: F^2 coefficient measured from the barycenter, c2 + n(n-1)/25
    ``exc_m``          (3, 3, nd, nd) float64: E[m, m'] (|m = -1, 0, 1>)
    ``lf_real``        (3, 3, nd, nd) complex128: E_real[a, b] (px, py, pz)
    ``so``, ``L2``, ``S2`` (nd, nd) float64; ``L``, ``S`` (3, nd, nd) complex128
    ``term_projectors`` (nterms, nd, nd) float64, DPN term order
    ``defect_terms``   (nterms,) bool: terms with L = 0 or S = 0 (DPNFIT's Zeeman defect)
    """

    n: int
    pairs: int
    coulomb_bary: torch.Tensor
    exc_m: torch.Tensor
    lf_real: torch.Tensor
    so: torch.Tensor
    L: torch.Tensor
    S: torch.Tensor
    L2: torch.Tensor
    S2: torch.Tensor
    term_projectors: torch.Tensor
    defect_terms: torch.Tensor

    @property
    def size(self) -> int:
        return self.so.shape[-1]

    @property
    def device(self) -> torch.device:
        return self.so.device

    @property
    def terms(self) -> list[_pn.Term]:
        return _pn.term_projectors(self.n)[0]


@lru_cache(maxsize=None)
def _pn_operators(n: int, device: str) -> PnOperators:
    ops = _pn.operators(n)
    u = _pn.U_REAL
    lf_real = np.einsum("ma,nb,mnij->abij", u, u.conj(), ops.excitation)
    terms, projs = _pn.term_projectors(n)
    pairs = ops.n_pairs
    bary = ops.coulomb[1] + 2.0 * pairs / 25.0 * np.eye(ops.size)
    f64 = lambda a: torch.as_tensor(np.array(a), dtype=_F64, device=device)  # noqa: E731
    c128 = lambda a: torch.as_tensor(np.array(a), dtype=_C128, device=device)  # noqa: E731
    defect = torch.tensor([t.L == 0 or t.multiplicity == 1 for t in terms], device=device)
    return PnOperators(n=n, pairs=pairs, coulomb_bary=f64(bary), exc_m=f64(ops.excitation), lf_real=c128(lf_real),
                       so=f64(ops.so), L=c128(ops.L), S=c128(ops.S), L2=f64(ops.L2), S2=f64(ops.S2),
                       term_projectors=f64(projs), defect_terms=defect)


def pn_operators(n: int, device: str | torch.device = "cpu") -> PnOperators:
    """Cached torch operators of p^n (n = 0..6) on ``device``."""
    if int(n) != n or not 0 <= n <= 6:
        raise ValueError("n must be an integer 0..6")
    return _pn_operators(int(n), str(torch.device(device)))


def _ops(n_or_ops, device) -> PnOperators:
    return n_or_ops if isinstance(n_or_ops, PnOperators) else pn_operators(n_or_ops, device)


# ------------------------------------------------------------------------------ Hamiltonian
def zeeman_operators(n, k=(1.0, 1.0, 1.0), constants: Constants = TORCHSPIN_CONSTANTS, device="cpu") -> torch.Tensor:
    """mu_i / mu_B = k_i L_i + g_e S_i, shape (..., 3, nd, nd); k is (..., 3), dimensionless."""
    ops = _ops(n, device)
    k = _t(k, ops.device)
    return k[..., :, None, None].to(_C128) * ops.L + constants.g_e * ops.S


def field_vector(B0, theta=0.0, phi=0.0, degrees: bool = False) -> torch.Tensor:
    """Field vector (..., 3) of magnitude ``B0`` (any field unit) along polar angle theta, azimuth phi."""
    B0, theta, phi = torch.broadcast_tensors(_t(B0), _t(theta), _t(phi))
    if degrees:
        theta, phi = torch.deg2rad(theta), torch.deg2rad(phi)
    st = torch.sin(theta)
    return B0[..., None] * torch.stack([st * torch.cos(phi), st * torch.sin(phi), torch.cos(theta)], -1)


def _hamiltonian_cm(ops: PnOperators, f: float, F2, zeta, A, alpha, V, V_m, k, field,
                    constants: Constants) -> torch.Tensor:
    """H in cm^-1; energy inputs are multiplied by f (their unit -> cm^-1)."""
    dev = ops.device
    eye = torch.eye(ops.size, dtype=_F64, device=dev)
    h = f * (_scal(A, dev) * (ops.pairs * eye) + _scal(F2, dev) * ops.coulomb_bary + _scal(alpha, dev) * ops.L2
             + _scal(zeta, dev) * ops.so)
    h = h.to(_C128)
    if V is not None:
        h = h + f * torch.einsum("...ab,abij->...ij", _t(V, dev).to(_C128), ops.lf_real)
    if V_m is not None:
        h = h + f * torch.einsum("...ab,abij->...ij", _t(V_m, dev).to(_C128), ops.exc_m.to(_C128))
    if field is not None:
        mu = zeeman_operators(ops, k, constants)
        h = h + constants.mu_b * torch.einsum("...i,...inm->...nm", _t(field, dev).to(_C128), mu)
    return h


def hamiltonian(n, *, F2=0.0, zeta=0.0, A=0.0, alpha=0.0, V=None, V_m=None, k=(1.0, 1.0, 1.0), field=None,
                units: str = "MHz", constants: Constants = TORCHSPIN_CONSTANTS, device="cpu") -> torch.Tensor:
    """Complex Hermitian H (..., nd, nd) of p^n, in ``units``.

    ``n`` is the electron count (or a cached :class:`PnOperators`).  ``V``: real-orbital ligand
    field (..., 3, 3) in (px, py, pz).  ``V_m``: a field given directly in the |m = -1, 0, 1>
    basis (complex; need not be time-even).  ``field`` (..., 3) in mT.
    """
    ops = _ops(n, device)
    f = _u.energy_factor(units, "cm^-1")
    return _hamiltonian_cm(ops, f, F2, zeta, A, alpha, V, V_m, k, field, constants) / f


def levels(n, *, relative: bool = False, units: str = "MHz", **kw) -> torch.Tensor:
    """Ascending energies (..., nd) in ``units``; ``relative=True`` puts the ground state at 0.

    Keywords as :func:`hamiltonian`.  The gradient of a degenerate eigenvalue is that of its
    level's mean (exact for the Kramers and orbital degeneracies of p^n).
    """
    E = eigvalsh_gauge(hamiltonian(n, units=units, **kw))
    return E - E[..., :1] if relative else E


def g_tensor(n, *, k=(1.0, 1.0, 1.0), units: str = "MHz", constants: Constants = TORCHSPIN_CONSTANTS,
             device="cpu", **kw) -> GTensor:
    """Zero-field effective-spin-1/2 g of the lowest doublet: (g g^T)_ij = 2 tr(P mu_i P mu_j).

    Keywords as :func:`hamiltonian` (without ``field``); only ``gap`` carries ``units``.
    Meaningful for odd n (Kramers doublets) or any isolated ground doublet (check ``gap``:
    for p^3 the 4S ground quartet splits only by second-order effects).
    """
    if kw.get("field") is not None:
        raise ValueError("g_tensor is the zero-field doublet g; do not pass field")
    kw.pop("field", None)
    ops = _ops(n, device)
    f = _u.energy_factor(units, "cm^-1")
    H = _hamiltonian_cm(ops, f, kw.pop("F2", 0.0), kw.pop("zeta", 0.0), kw.pop("A", 0.0), kw.pop("alpha", 0.0),
                        kw.pop("V", None), kw.pop("V_m", None), k, None, constants)
    if kw:
        raise TypeError(f"unexpected keywords: {sorted(kw)}")
    ggT, principal, g, gap = _g_from(H, zeeman_operators(ops, k, constants))
    return GTensor(ggT, principal, g, gap / f)


# ------------------------------------------------------------------------------ ligand fields (p shell)
def _u_real(device=None) -> torch.Tensor:
    return torch.as_tensor(_pn.U_REAL, dtype=_C128, device=device)


def to_m_basis(V) -> torch.Tensor:
    """Real-orbital (px, py, pz) matrix (..., 3, 3) -> |m = -1, 0, 1> matrix U V U^+ (complex)."""
    V = _t(V)
    u = _u_real(V.device)
    return u @ V.to(_C128) @ u.conj().T


def crystal_field(Ds=0.0, Dr=0.0) -> torch.Tensor:
    """Ballhausen-style axial (Ds) and rhombic (Dr) p-shell field (DPNIO.F ll. 235-237).

    In the |m> basis <+-1|V|+-1> = Ds, <0|V|0> = -2 Ds, <1|V|-1> = Dr; real orbitals:
    diag(Ds - Dr, Ds + Dr, -2 Ds).  A cubic field does not split p orbitals (no Dq).
    """
    ds, dr = torch.broadcast_tensors(_t(Ds), _t(Dr))
    return torch.diag_embed(torch.stack([ds - dr, ds + dr, -2.0 * ds], -1))


def orbital_energies(e) -> torch.Tensor:
    """Diagonal real-orbital matrix from (..., 3) energies (E(px), E(py), E(pz)) (DPN option P)."""
    return torch.diag_embed(_t(e))


def aom_from_frames(R: torch.Tensor, e) -> torch.Tensor:
    """V = sum_L e_sigma z'z'^T + e_pi_x x'x'^T + e_pi_y y'y'^T.

    ``R`` (..., nlig, 3, 3) ligand frames (columns x', y', z'; z' from metal to ligand);
    ``e`` (..., nlig, 3) bonding parameters in the order (e_sigma, e_pi_x, e_pi_y).
    """
    e = _t(e, R.device)
    cols = R[..., :, [2, 0, 1]]                                         # modes (sigma, pi_x, pi_y)
    return torch.einsum("...lm,...lam,...lbm->...ab", e, cols, cols)


def aom_from_angles(theta, phi, psi, e, degrees: bool = False) -> torch.Tensor:
    """AOM from per-ligand angles (..., nlig) and (e_sigma, e_pi_x, e_pi_y) (..., nlig, 3).

    Frame R = Rz(phi) Ry(theta) Rz(psi) (:func:`torchspin.lft.ligand_field.rotation`).
    DPN's "pi-c (x)" and "pi-s (y)" are ``e_pi_x`` and ``e_pi_y``.
    """
    return aom_from_frames(rotation(theta, phi, psi, degrees), _t(e))


def aom_isotropic(n, e_sigma, e_pi=0.0) -> torch.Tensor:
    """AOM from bond vectors n (..., nlig, 3) (normalized here): sum e_sigma n n^T + e_pi (I - n n^T).

    Smooth in n everywhere (no Euler angles); e_sigma, e_pi broadcast against (..., nlig).
    """
    n = _t(n)
    n = n / torch.linalg.vector_norm(n, dim=-1, keepdim=True)
    nn = n.unsqueeze(-1) * n.unsqueeze(-2)
    es, ep = (torch.broadcast_to(_t(x, n.device), n.shape[:-1])[..., None, None] for x in (e_sigma, e_pi))
    eye = torch.eye(3, dtype=n.dtype, device=n.device)
    return (es * nn + ep * (eye - nn)).sum(-3)


def aom(ligands: Iterable[Ligand], device=None) -> torch.Tensor:
    """p-shell AOM from :class:`torchspin.lft.Ligand` s (angles in radians; e_sigma, e_pi_x, e_pi_y).

    p orbitals have no delta bonding: a ligand with nonzero ``e_delta_*`` raises ValueError
    (DPNFIT accepts NLAM = 5 and silently corrupts the field, finding DPNFIT-03).
    """
    ligs = list(ligands)
    if not ligs:
        return torch.zeros(3, 3, dtype=_F64, device=device)
    for lig in ligs:
        if any(float(torch.as_tensor(getattr(lig, f)).abs().max()) != 0.0 for f in ("e_delta_x2y2", "e_delta_xy")):
            raise ValueError("p orbitals have no delta bonding; set e_delta_x2y2 = e_delta_xy = 0")
    col = lambda f: torch.stack([_t(getattr(l, f), device) for l in ligs])  # noqa: E731
    e = torch.stack([col(f) for f in ("e_sigma", "e_pi_x", "e_pi_y")], -1)
    return aom_from_angles(col("theta"), col("phi"), col("psi"), e)


# ------------------------------------------------------------------------------ DPN's own matrix
def internal_m_matrix(V, lf_type: str = "X", drop_o21: bool = True, np_: int = 1) -> torch.Tensor:
    """The electron-picture |m> matrix DPN/DPNFIT enter for a (geometric) real-orbital field.

    ``lf_type`` 'A' (AOM): for p^1/p^2 (``np_`` = 1, 2) the complex conjugate (mirror image
    y -> -y; p^3 is geometric), and with ``drop_o21`` <1|V|0> = <0|V|-1> = 0 (finding DPN-01).
    'X', 'P': the geometric matrix.
    """
    vm = to_m_basis(V)
    if lf_type.upper() != "A":
        return vm
    if np_ in (1, 2):
        vm = vm.conj()
    if drop_o21:
        mask = torch.ones(3, 3, dtype=_F64, device=vm.device)
        for a, b in ((2, 1), (1, 0)):
            mask[a, b] = mask[b, a] = 0.0
        vm = vm * mask
    return vm


def _defect_operator(ops: PnOperators, k, constants: Constants) -> torch.Tensor:
    """mu_z / mu_B restricted to the terms with L = 0 or S = 0 (what DPNFIT's SCALC omits)."""
    P = ops.term_projectors[ops.defect_terms].to(_C128)
    k = _t(k, ops.device)
    mu_z = k[..., 2, None, None].to(_C128) * ops.L[2] + constants.g_e * ops.S[2]
    return torch.einsum("tab,...bc,tcd->...ad", P, mu_z, P)


def _signed_abs(x: torch.Tensor) -> torch.Tensor:
    """|x| with d|x|/dx = +1 at x = 0 (zeta >= 0 is the feasible side)."""
    return x * torch.where(x.detach() < 0, -1.0, 1.0).to(x.dtype)


def dpn_hamiltonian(n: int, *, F2=0.0, zeta=0.0, V=None, k=(1.0, 1.0, 1.0), alpha=0.0, field=None,
                    lf_type: str = "X", program: str = "dpn", drop_o21: bool = True,
                    zeeman_defect: bool | None = None, units: str = "MHz", constants: Constants = DPN_CONSTANTS,
                    device="cpu") -> torch.Tensor:
    """The matrix DPN (``program='dpn'``) or DPNFIT (``'dpnfit'``) diagonalizes, in ``units``.

    For n > 3 this is the p^(6-n) hole problem with -V and -|zeta| (DPN flips only the real
    part of the |m> field, DPNFIT all of it), so its size is C(6, 6 - n).  ``zeta`` enters as
    |zeta|.  ``lf_type`` as :func:`internal_m_matrix` ('X' crystal field, 'A' AOM, 'P' orbital
    energies).  ``zeeman_defect`` (default: True for DPNFIT) removes the diagonal B_z Zeeman
    elements of terms with L = 0 or S = 0 (DPNFIT-01).  ``field`` (..., 3) in mT.  Constants
    default to DPN's own (:data:`DPN_CONSTANTS`).
    """
    if program not in ("dpn", "dpnfit"):
        raise ValueError("program must be 'dpn' or 'dpnfit'")
    npn = n if n <= 3 else 6 - n
    ops = pn_operators(npn, device)
    f = _u.energy_factor(units, "cm^-1")
    z = _signed_abs(_t(zeta, ops.device))
    vm = None
    if V is not None:
        vm = internal_m_matrix(V, lf_type, drop_o21, npn)
    if n > 3:
        z = -z
        if vm is not None:
            vm = (-vm.real + 1j * vm.imag) if program == "dpn" else -vm
    h = _hamiltonian_cm(ops, f, F2, z, 0.0, alpha, None, vm, k, field, constants)
    if zeeman_defect is None:
        zeeman_defect = program == "dpnfit"
    if zeeman_defect and field is not None:
        fz = _t(field, ops.device)[..., 2]
        h = h - constants.mu_b * fz[..., None, None].to(_C128) * _defect_operator(ops, k, constants)
    return h / f


# ------------------------------------------------------------------------------ system
class PnSystem:
    """A p^n ion with cached operators: ``PnSystem(n).levels(...)``, ``.g_tensor(...)``, ``.hamiltonian(...)``.

    ``units`` set here is the default for every call (override per call).
    """

    def __init__(self, n: int, *, units: str = "MHz", constants: Constants = TORCHSPIN_CONSTANTS, device="cpu"):
        self.n = int(n)
        self.units = _u.canonical_energy_unit(units)
        self.constants = constants
        self.ops = pn_operators(self.n, device)

    @property
    def size(self) -> int:
        return self.ops.size

    @property
    def terms(self) -> list[_pn.Term]:
        return self.ops.terms

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
