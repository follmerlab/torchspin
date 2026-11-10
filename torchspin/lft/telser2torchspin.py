"""Converters from Joshua Telser's spin-Hamiltonian conventions to torchspin objects.

Telser's spin-Hamiltonian programs (DDZHQ, DDPOWH, DJDZE, DDPOWJ; ported to Python in telser-tools as
``telser_tools.{spin, ddzhq, ddpowh, dimer, djdze, ddpowj}``) compute physics that torchspin already has
(``ham``, ``levels``, ``resfields``, ``pepper``, ``salt``, two-electron systems with exchange).  This module
maps Telser's parameters onto a :class:`torchspin.SpinSystem` (and an :class:`torchspin.Experiment` where
an experiment is described), so his inputs run through torchspin.  It does not import telser-tools:
parameter objects are read by attribute name (duck typing), and every function also takes plain keywords.

Units
-----
Inputs follow Telser:

* zero-field and exchange energies (D, E, B4q, Telser's a/F/b, J, j, Ddip, Edip, d, JQ) in ``units``,
  default ``'cm^-1'`` (``'MHz'``, ``'GHz'``, ``'K'``, ``'eV'`` accepted, via :mod:`torchspin.lft.units`);
* hyperfine A, U and quadrupole Q in MHz (as Telser);
* fields in gauss (``field_units='G'``, or ``'mT'``/``'T'``), frequencies in GHz, line widths in MHz (HWHM);
* angles in degrees (``degrees=True``, Telser's), radians with ``degrees=False``.

The outputs are torchspin's: MHz, mT, GHz, radians.  The SpinSystem's couplings are always in MHz.

Map (checked numerically against telser-tools in ``tests/test_lft_telser2torchspin.py``)
---------------------------------------------------------------------------------------
==========================================  ====================================================================
Telser                                       torchspin
==========================================  ====================================================================
g = (gx, gy, gz), g frame = molecular frame  ``g = [gx, gy, gz]``
(D/3) O20 + E O22                            ``D = [-D/3 + E, -D/3 - E, 2D/3]`` (MHz)
B40 O40 + B42 O42 + B44 O44 (Stevens)        ``B = [None, [[B44, 0, B42, 0, B40, 0, 0, 0, 0]]]`` (q = 4 ... -4)
a, F, b (cubic, axial, rhombic 4th order)    B40 = a/120 + F/180, B42 = b/3, B44 = a/24 (:func:`abf_to_stevens`)
u: mu_B sum_k u_k B_k C_k (3rd-order Zeeman)  ``Ham = {'132': ..., '134': ...}`` (:func:`third_order_zeeman_ham`)
A, Q principal values + Euler (deg)          ``A``, ``AFrame`` / ``Q``, ``QFrame`` in rad (same angles), or full
                                             tensors R diag R^T with ``full_tensors=True``
g_N of Telser's table                        ``gnscale = g_N / nucgval(isotope)``
dimer: + J S1.S2 (J > 0 antiferromagnetic)    ``ee`` = J 1 + Ddip + [d]x (full 3x3, MHz); same sign as EasySpin
dimer: j[(S1.S2)^2 - X1 X2/3]                ``ee2 = j``, levels shifted by -j X1 X2/3 (:func:`dimer_energy_offset`)
dimer: 24 JQ T2(S1).T2(S2) (S1, S2 >= 1)      ``ee2 += 24 JQ``, ``ee += 12 JQ``, shift -8 JQ X1 X2
dimer: site g2, D2 Euler angles               ``gFrame``/``DFrame`` (or full tensors)
EPR half width W_k (MHz, HWHM, frequency)    ``HStrain = 2 W`` (FWHM, same sqrt(sum W_k^2 n_k^2) rule)
ENDOR line half width w (MHz)                ``salt(..., lw_mhz=2 w)``
H1 mode 0 / < 0                              ``Experiment.Mode = 'perpendicular' / 'parallel'``
populations at T                             ``Experiment.Temperature = T`` (normalized Boltzmann, Telser exact mode)
==========================================  ====================================================================

Telser's exchange convention is EasySpin's (H = +J S1.S2).  Other conventions are converted with
``J_convention`` (:func:`telser_J`): H = -2J' S1.S2 (Heisenberg-Dirac-van Vleck, Bleaney-Bowers) is
``J_convention='-2J'``.

Transition probabilities: Telser's (telser-tools exact mode) B1-averaged P is normalized so that S = 1/2,
g = 2 gives 1.  torchspin's rate (|mu|^2 - |n.mu|^2)/2 is mu_B^2 P (with the nuclear moment, which Telser
leaves out by default); ``resfields`` intensities are mu_B^2 P/(2I + 1)/|dE/dB|.

Not mapped (see ``docs/lft_gap_analysis.md``): the second-order Zeeman w (torchspin's ``levels``,
``resfields`` and ``pepper`` use only the part of H linear in B), the I.S^3 hyperfine U, NMR/NQR powder
spectra (S = 0; the nuclear SpinSystem itself is built), the DDPOWH/DDPOWJ strain models, Lorentzian
frequency-domain widths, zero-field FDMR/ENDOR and mixed H1 modes.  With ``strict=True`` (default) these
raise ``NotImplementedError``; with ``strict=False`` they are dropped with a warning and listed in
``TelserConversion.extras['dropped']``.  Telser's Flygare frequency factor has no counterpart either; compare
with telser-tools' ``frequency_factor=False``.

Constants: torchspin uses CODATA 2022 mu_B, telser-tools CODATA 2018 (1.4e-9 apart).  ``constants=`` takes
a telser-tools ``Constants``-like object (``mu_b_mhz`` and ``mu_n_mhz`` in MHz/G, ``mu_b_cm`` in cm^-1/G);
g, u and g_N are then rescaled so that torchspin's Hamiltonian equals Telser's at the same field (and
cm^-1 -> MHz uses its ``mu_b_mhz/mu_b_cm``).  This is for parity tests: the g values in the SpinSystem
are then effective values, not the entered ones.

Gradients: g, D, E, B4q, A, Q principal values, J, j, Ddip, Edip, d, JQ and the dipolar Euler angles may be
tensors and keep their graph.  The other Euler angles keep it only with ``full_tensors=True`` (``gFrame``/``AFrame``/``DFrame`` are read
with ``.tolist()`` inside torchspin, mol2g bug B3).  The third-order Zeeman ``Ham`` coefficients are
detached inside ``ham_ezho``.
"""

from __future__ import annotations

import math
import re
import warnings
from functools import lru_cache
from typing import Any, NamedTuple

import torch

from torchspin.constants import BMAGN, NMAGN, PLANCK
from torchspin.experiment import Experiment
from torchspin.nucdata import nucgval, nucspin
from torchspin.spinsystem import SpinSystem

from . import units as _u

__all__ = ["TelserConversion", "MU_B_MHZ_PER_MT", "MU_N_MHZ_PER_MT", "abf_to_stevens", "stevens_to_abf",
           "zfs_principal", "euler_matrix", "rotate_tensor", "telser_J", "antisymmetric_tensor", "exchange_tensor",
           "third_order_zeeman_ham", "isotope_for", "dimer_energy_offset", "spin2sys", "dimer2sys",
           "epr_experiment", "ordering", "ddzhq2sys", "ddpowh2sys", "djdze2sys", "ddpowj2sys",
           "QuadrupoleEcho", "quadrupole_echo", "spin_projections"]

_F64 = torch.float64

#: mu_B/h and mu_N/h in MHz per mT, from :mod:`torchspin.constants` (what torchspin's Hamiltonian uses).
MU_B_MHZ_PER_MT = BMAGN / PLANCK * 1e-9
MU_N_MHZ_PER_MT = NMAGN / PLANCK * 1e-9

#: representative isotopes by nuclear spin, for nuclei given only as (I, g_N) (g_N is set through gnscale)
_ISOTOPE_BY_SPIN = {0.5: "1H", 1.0: "14N", 1.5: "7Li", 2.0: "36Cl", 2.5: "17O", 3.0: "10B", 3.5: "59Co",
                    4.0: "40K", 4.5: "93Nb", 5.0: "138La", 6.0: "50V"}

_J_CONVENTIONS = {"+J": 1.0, "J": 1.0, "-J": -1.0, "-2J": -2.0, "+2J": 2.0, "2J": 2.0}


class TelserConversion(NamedTuple):
    """Result of a program converter: the SpinSystem, an Experiment (or None) and extra information.

    ``extras`` keys (as present): ``'dropped'`` (features not mapped, strict=False), ``'notes'``,
    ``'energy_offset'`` (MHz; Telser's levels = torchspin's + offset), ``'cycles'`` (DDZHQ/DJDZE field
    points), ``'spectra'`` (DDPOWH per-spectrum experiments and ENDOR settings), ``'h1mode'``.
    """

    sys: SpinSystem
    exp: Experiment | None
    extras: dict


# ----------------------------------------------------------------------------- small conversions
def _t(x) -> torch.Tensor:
    return x.to(_F64) if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=_F64)


def _vec3(x) -> torch.Tensor:
    v = _t(x if x is not None else (0.0, 0.0, 0.0))
    if v.ndim == 0:
        v = v.expand(3)
    if v.shape != (3,):
        raise ValueError(f"expected 3 values, got shape {tuple(v.shape)}")
    return v


def _nonzero(x) -> bool:
    return x is not None and bool(torch.any(_t(x) != 0))


def abf_to_stevens(a=0.0, F=0.0, b=0.0):
    """(B40, B42, B44) from Telser's a (cubic, = 120 B4), F (axial, = 180 B40) and b (rhombic, = 3 B42).

    The cubic term is (a/120)(O40 + 5 O44), so B40 = a/120 + F/180, B42 = b/3, B44 = a/24
    (``telser_tools.spin.SpinParameters.from_abf``; ddzhqea.f ll. 161-175).  Same units in and out.
    """
    return a / 120.0 + F / 180.0, b / 3.0, a / 24.0


def stevens_to_abf(B40=0.0, B42=0.0, B44=0.0):
    """Inverse of :func:`abf_to_stevens`: (a, F, b) = (24 B44, 180 (B40 - B44/5), 3 B42)."""
    return 24.0 * B44, 180.0 * (B40 - B44 / 5.0), 3.0 * B42


def zfs_principal(D, E) -> torch.Tensor:
    """(-D/3 + E, -D/3 - E, 2D/3): the principal values of (D/3) O20 + E O22 as S.D.S (same units)."""
    D, E = _t(D), _t(E)
    return torch.stack([-D / 3 + E, -D / 3 - E, 2 * D / 3])


def euler_matrix(alpha, beta, gamma, *, degrees: bool = False) -> torch.Tensor:
    """McGavin z-y'-z'' rotation R = Rz(alpha) Ry(beta) Rz(gamma) of Telser's DEULER.F (graph-safe).

    The columns of R are the rotated axes in the molecular frame; a tensor with principal values t is
    R diag(t) R^T in the molecular frame.  EasySpin/torchspin frames (``AFrame``, ``gFrame``, ...) with the
    same angles give the same tensor.
    """
    a, b, g = (_t(x) for x in (alpha, beta, gamma))
    if degrees:
        a, b, g = (x * (math.pi / 180.0) for x in (a, b, g))
    ca, sa, cb, sb, cg, sg = torch.cos(a), torch.sin(a), torch.cos(b), torch.sin(b), torch.cos(g), torch.sin(g)
    return torch.stack([
        torch.stack([ca * cb * cg - sa * sg, -ca * cb * sg - sa * cg, ca * sb]),
        torch.stack([sa * cb * cg + ca * sg, -sa * cb * sg + ca * cg, sa * sb]),
        torch.stack([-sb * cg, sb * sg, cb]),
    ])


def rotate_tensor(principal, angles=(0.0, 0.0, 0.0), *, degrees: bool = False) -> torch.Tensor:
    """R diag(principal) R^T with R = :func:`euler_matrix` (3x3, graph-safe)."""
    r = euler_matrix(*_vec3(angles), degrees=degrees)
    return r @ torch.diag(_vec3(principal)) @ r.T


def telser_J(J, convention: str = "+J"):
    """J in Telser's (and EasySpin's) convention H = +J S1.S2, from J given in ``convention``.

    ``'+J'`` (H = +J S1.S2), ``'-J'`` (H = -J S1.S2), ``'-2J'`` (H = -2J S1.S2, Heisenberg-Dirac-van Vleck,
    Bleaney-Bowers), ``'+2J'``.  For example, J' = -10 cm^-1 in -2J' S1.S2 (ferromagnetic) is +20 cm^-1 here.
    """
    try:
        return _J_CONVENTIONS[convention.replace(" ", "")] * J
    except KeyError:
        raise ValueError(f"unknown exchange convention {convention!r}; use one of {', '.join(_J_CONVENTIONS)}") from None


def antisymmetric_tensor(d) -> torch.Tensor:
    """The 3x3 tensor T with S1.T.S2 = d.(S1 x S2) (antisymmetric, Dzyaloshinskii-Moriya)."""
    dx, dy, dz = _vec3(d)
    z = torch.zeros((), dtype=_F64)
    return torch.stack([torch.stack([z, dz, -dy]), torch.stack([-dz, z, dx]), torch.stack([dy, -dx, z])])


def exchange_tensor(J=0.0, Ddip=0.0, Edip=0.0, dip_euler=(0.0, 0.0, 0.0), d=(0.0, 0.0, 0.0), *,
                    degrees: bool = True) -> torch.Tensor:
    """Full 3x3 coupling S1.T.S2 = J S1.S2 + S1.Ddip.S2 + d.(S1 x S2) (Telser's DJDZE/DDPOWJ terms; same units).

    Ddip = R diag(-Ddip/3 + Edip, -Ddip/3 - Edip, 2Ddip/3) R^T with R from ``dip_euler``.
    """
    t = _t(J) * torch.eye(3, dtype=_F64)
    if _nonzero(Ddip) or _nonzero(Edip):
        t = t + rotate_tensor(zfs_principal(Ddip, Edip), dip_euler, degrees=degrees)
    if _nonzero(d):
        t = t + antisymmetric_tensor(d)
    return t


def _energy_to_mhz(units: str, constants) -> float:
    if _u.canonical_energy_unit(units) == "cm^-1" and constants is not None:
        return 1.0 / _constants_mhz_to_cm(constants)
    return _u.energy_factor(units, "MHz")


def _constants_mhz_to_cm(c) -> float:
    m = getattr(c, "mhz_to_cm", None)
    return float(m) if m is not None else float(c.mu_b_cm) / float(c.mu_b_mhz)


def _g_scale(constants) -> float:
    return 1.0 if constants is None else float(constants.mu_b_mhz) * 10.0 / MU_B_MHZ_PER_MT


def _gn_scale(constants) -> float:
    return 1.0 if constants is None else float(constants.mu_n_mhz) * 10.0 / MU_N_MHZ_PER_MT


def _field_to_mt(field_units: str) -> float:
    return _u.field_factor(field_units, "mT")


def _drop(name: str, strict: bool, dropped: list, why: str) -> None:
    msg = f"{name}: {why} (see docs/lft_gap_analysis.md)"
    if strict:
        raise NotImplementedError(msg + "; pass strict=False to drop it")
    warnings.warn("dropped " + msg, stacklevel=3)
    dropped.append(name)


# ----------------------------------------------------------------------------- third-order Zeeman (u)
def _cubic_third_order(s: float) -> list[torch.Tensor]:
    """C_k = S_k^3 - (3X - 1)/5 S_k, X = S(S+1) (Telser's u operators), |m = s ... -s> basis."""
    m = s - torch.arange(int(round(2 * s)) + 1, dtype=_F64)
    n = len(m)
    sp = torch.zeros((n, n), dtype=torch.complex128)
    for k in range(1, n):
        sp[k - 1, k] = math.sqrt(s * (s + 1) - float(m[k]) * (float(m[k]) + 1))
    sm = sp.T.conj()
    ops = [(sp + sm) / 2, (sp - sm) / 2j, torch.diag(m).to(torch.complex128)]
    x = s * (s + 1)
    return [o @ o @ o - (3 * x - 1) / 5 * o for o in ops]


@lru_cache(maxsize=32)
def _u_coefficients(s: float) -> tuple[tuple[tuple[float, ...], tuple[float, ...]], ...]:
    """For k = x, y, z: the ('132', '134') Ham rows whose linear-in-B part of torchspin's H is B_k C_k (MHz/mT).

    Found by projection through :func:`torchspin.ham.ham` (the (H0, mu) path that ``levels``,
    ``resfields`` and ``pepper`` use), so the map follows whatever sign that path applies (see bug B11 in
    the gap analysis: ``ham(sys, B0)`` adds the Ham1xx term with the opposite sign).
    """
    from torchspin.ham import ham

    base = SpinSystem(S=[s], g=[0.0, 0.0, 0.0])
    mu0 = torch.stack(list(ham(base)[1:]))                                    # (3, n, n)
    cols, keys = [], [("132", 5), ("134", 9)]
    for key, width in keys:
        for c in range(width):
            row = [0.0] * width
            row[c] = 1.0
            sy = SpinSystem(S=[s], g=[0.0, 0.0, 0.0], Ham={key: [row]})
            dmu = torch.stack(list(ham(sy)[1:])) - mu0                       # H gets -B.dmu
            cols.append((-dmu).reshape(-1))
    a = torch.stack(cols, dim=1)                                              # (3 n^2, 14)
    a_ri = torch.cat([a.real, a.imag], dim=0)
    ck = _cubic_third_order(s)
    out = []
    for k in range(3):
        target = torch.zeros(3, *ck[0].shape, dtype=torch.complex128)
        target[k] = ck[k]
        tv = target.reshape(-1)
        t_ri = torch.cat([tv.real, tv.imag])
        x = torch.linalg.lstsq(a_ri, t_ri.unsqueeze(1), driver="gelsd").solution.squeeze(1)
        res = float((a_ri @ x - t_ri).abs().max())
        if res > 1e-9 * max(1.0, float(t_ri.abs().max())):
            raise RuntimeError(f"third-order Zeeman term of S = {s} is not representable in Ham 13l (residual {res:.3g})")
        out.append((tuple(float(v) for v in x[:5]), tuple(float(v) for v in x[5:])))
    return tuple(out)


def third_order_zeeman_ham(s, u, *, mu_b: float = MU_B_MHZ_PER_MT, electron: int = 0,
                           n_electrons: int = 1) -> dict:
    """``SpinSystem.Ham`` entries for Telser's third-order Zeeman term mu_B sum_k u_k B_k C_k.

    C_k = S_k^3 - (3X - 1)/5 S_k (X = S(S+1)); ``u`` is dimensionless (one value, or (ux, uy, uz)), S >= 3/2.
    ``mu_b`` is mu_B/h in MHz/mT.  For a multi-electron system, ``electron`` (0-based) selects the row and
    ``n_electrons`` sets the number of rows (other rows zero).  The term is linear in B, with spin rank 3
    and coupling ranks l = 2 and 4 (``'132'``, ``'134'``; for ux = uy = uz only l = 4, the cubic invariant).
    The coefficients are detached (``ham_ezho`` reads them with ``.detach()``).
    """
    s = float(s)
    if s < 1.5:
        raise ValueError("the third-order Zeeman term needs S >= 3/2")
    uv = [float(v) for v in _vec3(u).detach()]
    coef = _u_coefficients(s)
    r2 = [mu_b * sum(uv[k] * coef[k][0][c] for k in range(3)) for c in range(5)]
    r4 = [mu_b * sum(uv[k] * coef[k][1][c] for k in range(3)) for c in range(9)]
    rows2 = [[0.0] * 5 for _ in range(n_electrons)]
    rows4 = [[0.0] * 9 for _ in range(n_electrons)]
    rows2[electron], rows4[electron] = r2, r4
    return {"132": rows2, "134": rows4}


def _merge_ham(a: dict | None, b: dict) -> dict:
    if a is None:
        return b
    out = dict(a)
    for k, v in b.items():
        out[k] = [[x + y for x, y in zip(r1, r2)] for r1, r2 in zip(out[k], v)] if k in out else v
    return out


# ----------------------------------------------------------------------------- nuclei
def isotope_for(spin: float, label: str | None = None) -> str:
    """A torchspin isotope name for a nucleus of spin I given by Telser's label (``'14N'``, ``'2H (D)'``, ...).

    The label is used when torchspin knows it with the same spin; otherwise a representative isotope with
    spin I is returned (the nuclear g value is then set through ``gnscale``).
    """
    if label:
        mt = re.match(r"\s*(\d+)\s*([A-Z][a-z]?)", label)
        if mt:
            name = f"{mt.group(1)}{mt.group(2)}"
            try:
                if abs(float(nucspin(name)) - float(spin)) < 1e-9:
                    return name
            except (KeyError, ValueError, IndexError):
                pass
    key = round(2 * float(spin)) / 2
    if key not in _ISOTOPE_BY_SPIN:
        raise ValueError(f"no stand-in isotope with nuclear spin {spin}")
    return _ISOTOPE_BY_SPIN[key]


# ----------------------------------------------------------------------------- one electron spin
def _get(obj, name, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _pick(kw_value, obj, name, default):
    return kw_value if kw_value is not None else _get(obj, name, default)


def spin2sys(params: Any = None, *, s=None, g=None, w=None, u=None, D=None, E=None, B40=None, B42=None, B44=None,
             a=None, F=None, b=None, nucleus: Any = None, I=None, g_n=None, A=None, A_euler=None, U=None, Q=None,
             Q_euler=None, isotope: str | None = None, units: str = "cm^-1", degrees: bool = True, constants=None,
             full_tensors: bool = False, strict: bool = True, dropped: list | None = None, **sys_kwargs) -> SpinSystem:
    """SpinSystem of one electron spin S (and at most one nucleus) in Telser's DDZHQ/DDPOWH parameterization.

    ``params`` is a telser-tools ``spin.SpinParameters`` (or any object/dict with ``s, g, w, u, D, E, B40, B42,
    B44, nucleus``; the nucleus with ``spin, g_n, A, A_euler, U, Q, Q_euler, label``).  Keywords override it
    and may be tensors.  ``a, F, b`` (Telser's 4th-order form) are added to B40, B42, B44.

    Units: D, E, B4q, a, F, b in ``units`` (default cm^-1); A, U, Q in MHz; angles in degrees unless
    ``degrees=False``.  ``isotope``: torchspin name of the nucleus (default from the label, see
    :func:`isotope_for`).  ``full_tensors=True`` passes A and Q as full 3x3 tensors (Euler angles stay on
    the autograd graph).  ``sys_kwargs`` go to :class:`SpinSystem` (``HStrain``, ``lw``, ...).

    S = 0 gives a nuclear spin system (``S=[]``: quadrupole and nuclear Zeeman), DDPOWH's NMR/NQR case.

    Telser's terms with no torchspin counterpart, w (second-order Zeeman) and U (I.S^3 hyperfine), raise
    ``NotImplementedError`` (``strict=False``: dropped with a warning, names appended to ``dropped``).
    """
    dropped = [] if dropped is None else dropped
    s = float(_pick(s, params, "s", 0.5))
    f = _energy_to_mhz(units, constants)
    gs = _g_scale(constants)
    if s >= 0.5:
        g = _vec3(_pick(g, params, "g", (2.0, 2.0, 2.0)))
        kw: dict = dict(S=[s], g=(g * gs).unsqueeze(0))
    else:                                                   # S = 0: a nuclear spin system (NMR/NQR)
        kw = dict(S=[])
    if s >= 1:
        D_, E_ = _pick(D, params, "D", 0.0), _pick(E, params, "E", 0.0)
        if _nonzero(D_) or _nonzero(E_):
            kw["D"] = (zfs_principal(D_, E_) * f).unsqueeze(0)
        w_ = _pick(w, params, "w", 0.0)
        if _nonzero(w_):
            _drop("w (second-order Zeeman)", strict, dropped,
                  "torchspin's levels/resfields/pepper use only the part of H linear in B")
    if s >= 1.5:
        u_ = _pick(u, params, "u", (0.0, 0.0, 0.0))
        if _nonzero(u_):
            kw["Ham"] = third_order_zeeman_ham(s, u_, mu_b=MU_B_MHZ_PER_MT * gs)
    if s >= 2:
        b4 = [_t(_pick(x, params, n, 0.0)) for x, n in ((B40, "B40"), (B42, "B42"), (B44, "B44"))]
        if any(v is not None for v in (a, F, b)):
            extra = abf_to_stevens(_t(a if a is not None else 0.0), _t(F if F is not None else 0.0),
                                   _t(b if b is not None else 0.0))
            b4 = [x + y for x, y in zip(b4, extra)]
        if any(_nonzero(v) for v in b4):
            z = torch.zeros((), dtype=_F64)
            row = torch.stack([b4[2], z, b4[1], z, b4[0], z, z, z, z]) * f
            kw["B"] = [None, row.unsqueeze(0)]
    nuc = _pick(nucleus, params, "nucleus", None)
    xi = float(_pick(I, nuc, "spin", 0.0)) if (nuc is not None or I is not None) else 0.0
    if s < 0.5 and xi < 0.5:
        raise ValueError("S = 0 needs a nucleus with I >= 1/2")
    if xi >= 0.5:
        iso = isotope or isotope_for(xi, _get(nuc, "label", None))
        gn = _t(_pick(g_n, nuc, "g_n", 0.0))
        kw["Nucs"] = [iso]
        kw["gnscale"] = (gn / float(nucgval(iso)) * _gn_scale(constants)).reshape(1)
        A_ = _vec3(_pick(A, nuc, "A", (0.0, 0.0, 0.0)))
        Ae = _vec3(_pick(A_euler, nuc, "A_euler", (0.0, 0.0, 0.0)))
        deg = math.pi / 180.0 if degrees else 1.0
        if s < 0.5:
            pass
        elif full_tensors:
            kw["A"] = rotate_tensor(A_, Ae * deg)
        else:
            kw["A"] = A_.unsqueeze(0)
            kw["AFrame"] = (Ae * deg).unsqueeze(0)
        if xi >= 1:
            Q_ = _vec3(_pick(Q, nuc, "Q", (0.0, 0.0, 0.0)))
            if _nonzero(Q_):
                Qe = _vec3(_pick(Q_euler, nuc, "Q_euler", (0.0, 0.0, 0.0)))
                if full_tensors:
                    kw["Q"] = rotate_tensor(Q_, Qe * deg)
                else:
                    kw["Q"] = Q_.unsqueeze(0)
                    kw["QFrame"] = (Qe * deg).unsqueeze(0)
        U_ = _pick(U, nuc, "U", 0.0)
        if s >= 1.5 and _nonzero(U_):
            _drop("U (I.S^3 hyperfine)", strict, dropped, "torchspin has no hyperfine term cubic in S")
    for key in sys_kwargs:
        if key in kw and key != "Ham":
            raise TypeError(f"spin2sys sets {key!r} itself")
    if "Ham" in sys_kwargs:
        kw["Ham"] = _merge_ham(kw.get("Ham"), sys_kwargs.pop("Ham"))
    return SpinSystem(**kw, **sys_kwargs)


# ----------------------------------------------------------------------------- dimers
def dimer_energy_offset(s1: float, s2: float, j=0.0, JQ=0.0, *, units: str = "cm^-1", constants=None):
    """Telser's dimer levels minus torchspin's (MHz): -(j + 24 JQ) X1 X2/3 (JQ only for S1, S2 >= 1).

    Telser's biquadratic term is j[(S1.S2)^2 - X1 X2/3] (X = S(S+1)); torchspin's ``ee2`` has no constant.
    The isotropic quadrupole term 24 JQ T2(S1).T2(S2) = 24 JQ [(S1.S2)^2 + S1.S2/2 - X1 X2/3].
    """
    x1, x2 = s1 * (s1 + 1), s2 * (s2 + 1)
    jq = JQ if (s1 >= 1 and s2 >= 1) else 0.0
    return -(j + 24.0 * jq) * x1 * x2 / 3.0 * _energy_to_mhz(units, constants)


def _site_tensor(principal, euler, full: bool, deg: float):
    if full:
        return rotate_tensor(principal, _vec3(euler) * deg), None
    return _vec3(principal), _vec3(euler) * deg


def dimer2sys(params: Any = None, *, site1: Any = None, site2: Any = None, J=None, j=None, Ddip=None, Edip=None,
              dip_euler=None, d=None, JQ=None, J_convention: str = "+J", units: str = "cm^-1", degrees: bool = True,
              constants=None, full_tensors: bool = False, **sys_kwargs) -> SpinSystem:
    """SpinSystem of Telser's exchange-coupled dimer (DJDZE, DDPOWJ): two electron spins S1, S2.

    ``params``: a telser-tools ``dimer.DimerParameters`` (``site1``, ``site2``, ``coupling``) or anything with
    those attributes.  A site has ``s, g, g_euler, D, E, D_euler, B40, B42, B44, mu``; the coupling ``J, j,
    Ddip, Edip, dip_euler, d, JQ``.  ``site1``/``site2`` (objects or dicts) and the coupling keywords
    override it; values may be tensors.

    H = sum_i [mu_B B.g_i.S_i + S_i.D_i.S_i + B4q_i O4q + mu_i mu_B sum_k B_k C_k(S_i)] + J S1.S2
    + S1.Ddip.S2 + d.(S1 x S2) + j[(S1.S2)^2 - X1X2/3] + 24 JQ T2(S1).T2(S2), all tensors in the site-1 frame.
    ``J`` is in ``J_convention`` (default Telser's/EasySpin's +J; ``'-2J'`` for H = -2J S1.S2).  The
    coupling becomes the full 3x3 ``ee`` (MHz) and ``ee2``; levels differ from Telser's by the constant
    :func:`dimer_energy_offset`.  A site with S = 0 is left out (a one-electron SpinSystem).
    """
    cp = _get(params, "coupling", None)
    sites_in = [site1 if site1 is not None else _get(params, "site1", None),
                site2 if site2 is not None else _get(params, "site2", None)]
    f = _energy_to_mhz(units, constants)
    gs = _g_scale(constants)
    deg = math.pi / 180.0 if degrees else 1.0
    spins = [float(_get(si, "s", 0.0)) for si in sites_in]
    keep = [k for k in range(2) if spins[k] >= 0.5]
    if not keep:
        raise ValueError("both sites have S = 0")
    ne = len(keep)
    g_rows, gf_rows, d_rows, df_rows, b4_rows, ham = [], [], [], [], [], None
    any_d, any_b4 = False, False
    for e, k in enumerate(keep):
        si, sk = sites_in[k], spins[k]
        gt, gfr = _site_tensor(_vec3(_get(si, "g", (2.0, 2.0, 2.0))) * gs, _get(si, "g_euler", (0.0, 0.0, 0.0)),
                               full_tensors, deg)
        g_rows.append(gt)
        gf_rows.append(gfr)
        D_, E_ = _get(si, "D", 0.0), _get(si, "E", 0.0)
        if sk >= 1 and (_nonzero(D_) or _nonzero(E_)):
            any_d = True
            dt, dfr = _site_tensor(zfs_principal(D_, E_) * f, _get(si, "D_euler", (0.0, 0.0, 0.0)), full_tensors, deg)
        else:
            dt, dfr = (torch.zeros(3, 3, dtype=_F64), None) if full_tensors else (torch.zeros(3, dtype=_F64),
                                                                                 torch.zeros(3, dtype=_F64))
        d_rows.append(dt)
        df_rows.append(dfr)
        z = torch.zeros((), dtype=_F64)
        b40, b42, b44 = (_t(_get(si, n, 0.0)) for n in ("B40", "B42", "B44"))
        if sk >= 2 and any(_nonzero(v) for v in (b40, b42, b44)):
            if _nonzero(_get(si, "D_euler", (0.0, 0.0, 0.0))):
                raise NotImplementedError("4th-order terms are only defined in the molecular frame (as in Telser's "
                                          "programs, which do not read them for a rotated site)")
            any_b4 = True
            b4_rows.append(torch.stack([b44, z, b42, z, b40, z, z, z, z]) * f)
        else:
            b4_rows.append(torch.zeros(9, dtype=_F64))
        mu = _get(si, "mu", 0.0)
        if sk >= 1.5 and _nonzero(mu):
            ham = _merge_ham(ham, third_order_zeeman_ham(sk, _t(mu).expand(3), mu_b=MU_B_MHZ_PER_MT * gs,
                                                          electron=e, n_electrons=ne))
    kw: dict = dict(S=[spins[k] for k in keep])
    if full_tensors:
        kw["g"] = torch.cat(g_rows, dim=0)
    else:
        kw["g"] = torch.stack(g_rows)
        kw["gFrame"] = torch.stack(gf_rows)
    if any_d:
        if full_tensors:
            kw["D"] = torch.cat(d_rows, dim=0)
        else:
            kw["D"] = torch.stack(d_rows)
            kw["DFrame"] = torch.stack(df_rows)
    if any_b4:
        kw["B"] = [None, torch.stack(b4_rows)]
    if ham is not None:
        kw["Ham"] = ham
    if ne == 2:
        Jv = telser_J(_t(_pick(J, cp, "J", 0.0)), J_convention)
        jv = _t(_pick(j, cp, "j", 0.0))
        JQv = _t(_pick(JQ, cp, "JQ", 0.0))
        both1 = spins[0] >= 1 and spins[1] >= 1
        ee = exchange_tensor(Jv, _pick(Ddip, cp, "Ddip", 0.0), _pick(Edip, cp, "Edip", 0.0),
                             _pick(dip_euler, cp, "dip_euler", (0.0, 0.0, 0.0)), _pick(d, cp, "d", (0.0, 0.0, 0.0)),
                             degrees=degrees)
        ee2 = jv
        if both1:
            ee = ee + 12.0 * JQv * torch.eye(3, dtype=_F64)
            ee2 = ee2 + 24.0 * JQv
        kw["ee"] = ee * f
        kw["ee2"] = (ee2 * f).reshape(1)
    for key in sys_kwargs:
        if key in kw and key != "Ham":
            raise TypeError(f"dimer2sys sets {key!r} itself")
    if "Ham" in sys_kwargs:
        kw["Ham"] = _merge_ham(kw.get("Ham"), sys_kwargs.pop("Ham"))
    return SpinSystem(**kw, **sys_kwargs)


# ----------------------------------------------------------------------------- experiments
#: stand-in for a zero field in fixed-field Experiments (strict=False), mT
_ZERO_FIELD_MT = 1e-6

_MIXED_NOTE = ("mixed H1 mode r: Telser's P = P_perp + r P_par; exp is perpendicular, the parallel part is "
               "the same Experiment with Mode='parallel'")


def _mode(h1mode: float, strict: bool, dropped: list) -> str:
    if h1mode < 0:
        return "parallel"
    if h1mode > 0:
        _drop(f"mixed H1 mode {h1mode}", strict, dropped,
              "torchspin simulates one mode at a time: add pepper(..., Mode='parallel') x ratio to the "
              "perpendicular spectrum")
    return "perpendicular"


def epr_experiment(freq_ghz: float, field_range, npts: int, *, field_units: str = "G", deriv: int = 0,
                   temperature: float | None = None, h1mode: float = 0.0, strict: bool = True,
                   dropped: list | None = None) -> Experiment:
    """Experiment of a Telser field sweep: ``npts`` intervals (npts + 1 points, as DDPOWH/DDPOWJ write them).

    ``field_range`` (Bmin, Bmax) in ``field_units`` (Telser: G or T); ``deriv`` 0/1 -> ``Harmonic``.
    The exact-mode derivative of telser-tools (d/dB) is pepper's ``Harmonic=1``.
    """
    dropped = [] if dropped is None else dropped
    fm = _field_to_mt(field_units)
    ex = dict(mwFreq=float(freq_ghz), Range=[float(field_range[0]) * fm, float(field_range[1]) * fm],
              nPoints=int(npts) + 1, Harmonic=int(deriv), Mode=_mode(h1mode, strict, dropped))
    if temperature is not None:
        ex["Temperature"] = float(temperature)
    return Experiment(**ex)


def ordering(lam_x: float, lam_z: float):
    """``Experiment.Ordering`` callable for DDPOWH's partial orientation ORFACT = exp(lam_x cos^2 phi + lam_z cos^2 theta).

    (theta, phi) is the field direction in the molecular frame.  With the field along the lab z axis, the
    molecular frame at EasySpin Euler angles (alpha, beta, gamma) sees it at theta = beta, phi = pi - gamma,
    so the weight is exp(lam_x cos^2 gamma + lam_z cos^2 beta) (independent of alpha, the rotation about B0).
    """
    import numpy as np

    lx, lz = float(lam_x), float(lam_z)
    return lambda a, b, c: np.exp(lx * np.cos(c) ** 2 + lz * np.cos(b) ** 2) * np.ones(np.shape(a))


# ----------------------------------------------------------------------------- program inputs
def _fourth_order_from_input(inp) -> tuple[float, float, float]:
    """(B40, B42, B44) of a DDZHQ/DDPOWH input in exact mode (Bnm form: the entered B44 is the Stevens B44)."""
    if float(inp.s) < 2:
        return 0.0, 0.0, 0.0
    x1, x2, x3 = inp.fourth_order
    if getattr(inp, "fourth_order_form", "a") == "B":       # entered B44, B40, B42
        return x2, x3, x1
    return abf_to_stevens(x1, x2, x3)


class _Nucleus(NamedTuple):
    spin: float
    g_n: float
    A: tuple
    A_euler: tuple
    U: float
    Q: tuple
    Q_euler: tuple
    label: str


def _single_spin_params(inp) -> dict:
    s = float(inp.s)
    b40, b42, b44 = _fourth_order_from_input(inp)
    nuc = None
    xi = float(getattr(inp, "nucleus_spin", 0.0))
    if xi >= 0.5:
        rot = bool(getattr(inp, "rotate", False))
        nuc = _Nucleus(xi, float(inp.g_n), tuple(inp.A), tuple(inp.A_euler) if rot else (0.0, 0.0, 0.0),
                       float(inp.U) if s >= 1.5 else 0.0, tuple(inp.Q) if xi >= 1 else (0.0, 0.0, 0.0),
                       tuple(inp.Q_euler) if rot else (0.0, 0.0, 0.0), str(getattr(inp, "nucleus_label", "")))
    return dict(s=s, g=tuple(inp.g), w=float(inp.w) if s >= 1 else 0.0,
                u=tuple(inp.u) if s > 1 else (0.0, 0.0, 0.0), D=float(inp.D) if s >= 1 else 0.0,
                E=float(inp.E) if s >= 1 else 0.0, B40=b40, B42=b42, B44=b44, nucleus=nuc)


def _direction(theta: float, phi: float) -> torch.Tensor:
    return torch.tensor([math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)],
                        dtype=_F64)


def ddzhq2sys(inp, *, constants=None, full_tensors: bool = False, strict: bool = True, **sys_kwargs) -> TelserConversion:
    """A DDZHQ input (telser-tools ``ddzhq.DdzhqInput``, read by attribute) as SpinSystem + field points.

    ``extras['cycles']``: one dict per computed cycle, in input order: ``field_mT``, ``theta``/``phi``
    (radians, the field direction in the molecular frame), ``direction`` (unit 3-vector), ``g_eff`` and
    ``zero_field``.  A g-value entry is converted to the field nu/(mu_B g) (with ``constants`` if given).
    ``exp`` is an Experiment at the input's frequency (``Mode`` from the H1 mode; ``Temperature`` if
    populations are on), for :func:`torchspin.resfields`.  Exact-mode semantics of telser-tools: the
    Bnm-form B44 is the Stevens coefficient (finding DDZHQ-06), zero field is a real zero field.
    """
    dropped: list = []
    sy = spin2sys(_single_spin_params(inp), units="cm^-1", constants=constants, full_tensors=full_tensors,
                  strict=strict, dropped=dropped, **sys_kwargs)
    mub = MU_B_MHZ_PER_MT if constants is None else float(constants.mu_b_mhz) * 10.0
    cycles, last = [], None
    for cy in inp.cycles:
        if not getattr(cy, "computed", True):
            continue
        zero = bool(getattr(cy, "zero_field", cy.value <= 0 if getattr(cy, "entered", "H") == "H" else False))
        if zero:
            cycles.append(dict(field_mT=0.0, theta=0.0, phi=0.0, direction=_direction(0.0, 0.0), g_eff=None,
                               zero_field=True))
            continue
        if cy.new_field or last is None:
            if cy.entered == "H":
                b_mt = float(cy.value) / 10.0
                geff = float(inp.freq_ghz) * 1e3 / (mub * b_mt)
            else:
                geff = float(cy.value)
                b_mt = float(inp.freq_ghz) * 1e3 / (mub * geff)
            last = (b_mt, geff)
        b_mt, geff = last
        th, ph = math.radians(cy.theta), math.radians(cy.phi)
        cycles.append(dict(field_mT=b_mt, theta=th, phi=ph, direction=_direction(th, ph), g_eff=geff,
                           zero_field=False))
    exp_kw = dict(mwFreq=float(inp.freq_ghz), Mode="parallel" if float(inp.h1mode) < 0 else "perpendicular")
    if getattr(inp, "populations", False):
        exp_kw["Temperature"] = float(inp.temperature)
    notes = [_MIXED_NOTE] if float(inp.h1mode) > 0 else []
    return TelserConversion(sy, Experiment(**exp_kw), dict(cycles=cycles, dropped=dropped, notes=notes,
                                                           h1mode=float(inp.h1mode)))


def _ddpowh_strain_check(inp, strict: bool, dropped: list) -> None:
    if getattr(inp, "strain", False):
        ms = [float(x) for x in getattr(inp, "ms_strain", ())]
        mi = [float(x) for x in getattr(inp, "mi_strain", ())]
        b = float(getattr(inp, "strain_b", 0.0))
        if any(ms) or any(mi) or b != 0.0:
            _drop("DDPOWH strain (mS/mI level widths, (H/Hset)^b)", strict, dropped,
                  "torchspin's HStrain/gStrain/AStrain/DStrain are different models")


def ddpowh2sys(inp, *, constants=None, full_tensors: bool = False, strict: bool = True,
               **sys_kwargs) -> TelserConversion:
    """A DDPOWH input (telser-tools ``ddpowh.DdpowhInput``) as SpinSystem + Experiment for pepper/salt.

    * Field-swept EPR (kind ``'E'``): ``exp`` for :func:`torchspin.pepper` (Range in mT, npts + 1 points,
      Harmonic = derivative flag, Temperature with populations).
    * FDMR (``'F'``): one frequency-swept Experiment per spectrum (``Field`` in mT, ``mwRange`` in GHz),
      in ``extras['spectra']``; ``exp`` is the first.
    * ENDOR (``'N'``): one fixed-field Experiment per spectrum (``Field``, ``mwFreq``, ``ExciteWidth=1e-3``)
      and the :func:`torchspin.salt` keywords ``freq_range`` (MHz, absolute; for a centered axis the Larmor
      frequency is added) and ``lw_mhz`` = 2 w (Telser's NMR half width w), in ``extras['spectra']``.
      Telser selects orientations with the EPR line itself (``HStrain`` = 2 W here and ExciteWidth ~ 0 in
      salt, as telser-tools' docs map it).

    The EPR half widths W (MHz, HWHM) become ``HStrain = 2 W``.  Telser's Lorentzian frequency-domain line,
    the Flygare nu factors (compare with telser-tools' ``frequency_factor=False``), the DDPOWH strain model,
    ORFACT (use :func:`ordering`), the <g^2> normalization (a constant) and NMR/NQR spectra (S = 0) are not mapped.
    """
    dropped: list = []
    kind = getattr(inp, "kind", "E")
    if float(inp.s) < 0.5:
        _drop("NMR/NQR powder spectrum (S = 0)", strict, dropped, "torchspin has no powder NMR/NQR simulator; "
              "spin2sys gives the nuclear SpinSystem for levels")
    if any(float(x) for x in getattr(inp, "sigma_khz", (0.0, 0.0, 0.0))):
        _drop("S = 0 chemical shielding", strict, dropped, "not mapped to SpinSystem.sigma")
    if getattr(inp, "epr_shape", "G") in ("L", "l") and any(float(x) for x in inp.epr_width):
        _drop("Lorentzian EPR line (frequency domain)", strict, dropped, "HStrain is Gaussian")
    _ddpowh_strain_check(inp, strict, dropped)
    if not getattr(inp, "powder", True) and (float(getattr(inp, "lam_x", 0.0)) or float(getattr(inp, "lam_z", 0.0))):
        dropped.append("ORFACT (pass Experiment(Ordering=telser2torchspin.ordering(lam_x, lam_z)))")
    hs = [2.0 * float(x) for x in inp.epr_width]
    extra = dict(sys_kwargs)
    if any(hs):
        extra.setdefault("HStrain", hs)
    sy = spin2sys(_single_spin_params(inp), units="cm^-1", constants=constants, full_tensors=full_tensors,
                  strict=strict, dropped=dropped, **extra)
    temp = float(inp.temperature) if getattr(inp, "populations", False) else None
    mub = MU_B_MHZ_PER_MT if constants is None else float(constants.mu_b_mhz) * 10.0
    mun = MU_N_MHZ_PER_MT if constants is None else float(constants.mu_n_mhz) * 10.0
    spectra = []
    if kind == "E":
        exp = epr_experiment(inp.freq_ghz, inp.field_range, inp.npts, field_units=inp.field_units,
                             deriv=inp.epr_deriv, temperature=temp, h1mode=float(inp.h1mode), strict=strict,
                             dropped=dropped)
        return TelserConversion(sy, exp, dict(dropped=dropped, spectra=[dict(exp=exp)], h1mode=float(inp.h1mode)))
    if float(inp.s) < 0.5:
        return TelserConversion(sy, None, dict(dropped=dropped, spectra=[], h1mode=float(inp.h1mode)))
    mode = _mode(float(inp.h1mode), strict, dropped)
    for sp in inp.spectra:
        if sp.entered == "G":
            b_mt = float(inp.freq_ghz) * 1e3 / (mub * float(sp.value))
        else:
            b_mt = float(sp.value) / 10.0
        if b_mt <= 0:
            _drop(f"zero-field {'FDMR' if kind == 'F' else 'ENDOR'} spectrum", strict, dropped,
                  f"torchspin's fixed-field experiments need Field > 0; {_ZERO_FIELD_MT} mT is used instead")
            b_mt = _ZERO_FIELD_MT
        if kind == "F":
            lo, hi = inp.energy_range
            fac = 29.9792458 if getattr(inp, "energy_units", "C") in ("C", "c") else 1e3   # cm^-1 or THz -> GHz
            if constants is not None and getattr(inp, "energy_units", "C") in ("C", "c"):
                fac = 1e-3 / _constants_mhz_to_cm(constants)
            ex = dict(Field=b_mt, mwRange=[lo * fac, hi * fac], nPoints=int(inp.npts) + 1,
                      Harmonic=int(inp.epr_deriv), Mode=mode)
            if temp is not None:
                ex["Temperature"] = temp
            spectra.append(dict(exp=Experiment(**ex), field_mT=b_mt))
        else:
            nu_n = abs(float(inp.g_n)) * mun * b_mt
            lo, hi = inp.rf_range
            if getattr(inp, "center", False):
                lo, hi = lo + nu_n, hi + nu_n
            ex = dict(Field=b_mt, mwFreq=float(inp.freq_ghz), ExciteWidth=1e-3, Mode=mode)
            if temp is not None:
                ex["Temperature"] = temp
            spectra.append(dict(exp=Experiment(**ex), field_mT=b_mt, nu_n_mhz=nu_n,
                                salt=dict(freq_range=(lo, hi), n_points=int(inp.npts) + 1,
                                          lw_mhz=2.0 * float(inp.nmr_width[0]))))
            if len({float(x) for x in inp.nmr_width}) > 1:
                dropped.append("anisotropic NMR width (salt takes one lw_mhz; the x value is used)")
    return TelserConversion(sy, spectra[0]["exp"] if spectra else None,
                            dict(dropped=dropped, spectra=spectra, h1mode=float(inp.h1mode)))


def _dimer_params(inp, separate_d_euler: bool):
    sites = []
    for k, si in enumerate(inp.sites):
        rot = k == 1 and bool(inp.rotate2)
        if separate_d_euler:
            ge = tuple(inp.g_euler2) if rot else (0.0, 0.0, 0.0)
            de = tuple(inp.D_euler2) if rot else (0.0, 0.0, 0.0)
        else:
            ge = de = tuple(inp.euler2) if rot else (0.0, 0.0, 0.0)
        s = float(si.s)
        b4 = abf_to_stevens(si.a, si.F, si.b) if s >= 2 else (0.0, 0.0, 0.0)
        sites.append(dict(s=s, g=tuple(si.g) if s >= 0.5 else (0.0, 0.0, 0.0), g_euler=ge, D=float(si.D),
                          E=float(si.E), D_euler=de, B40=b4[0], B42=b4[1], B44=b4[2],
                          mu=float(si.mu) if s >= 1.5 else 0.0))
    coupling = dict(J=float(inp.J), j=float(inp.j), Ddip=float(inp.Ddip), Edip=float(inp.Edip),
                    dip_euler=tuple(inp.euler_dip) if inp.rotate_dip else (0.0, 0.0, 0.0), d=tuple(inp.JA),
                    JQ=float(inp.JQ))
    return dict(site1=sites[0], site2=sites[1], coupling=coupling)


def _offset_for(p: dict, constants) -> float:
    s1, s2 = p["site1"]["s"], p["site2"]["s"]
    if s1 < 0.5 or s2 < 0.5:
        return 0.0
    return dimer_energy_offset(s1, s2, p["coupling"]["j"], p["coupling"]["JQ"], constants=constants)


def djdze2sys(inp, *, constants=None, full_tensors: bool = False, **sys_kwargs) -> TelserConversion:
    """A DJDZE input (telser-tools ``djdze.DjdzeInput``) as a two-electron SpinSystem + field points.

    Site 2's g and D are rotated by the same Euler angles (DJDZE).  ``extras['energy_offset']`` (MHz) is
    added to torchspin's levels to get Telser's; ``extras['cycles']`` lists the field points (``field_mT``,
    ``theta``, ``phi``, ``direction``) as DJDZE visits them (a zero field keeps the previous orientation).
    """
    p = _dimer_params(inp, separate_d_euler=False)
    sy = dimer2sys(p, constants=constants, full_tensors=full_tensors, **sys_kwargs)
    mub = MU_B_MHZ_PER_MT if constants is None else float(constants.mu_b_mhz) * 10.0
    cycles, h0, th, ph = [], 0.0, 0.0, 0.0
    for cy in inp.cycles:                          # djdze.f paths: 201 first field, 208 field, 210 g, 225 orientation
        if cy.path == "201":
            h0 = max(float(cy.value), 0.0)
        elif cy.path == "208":
            h0 = float(cy.value)
        elif cy.path == "210" and float(cy.value) > 0:
            h0 = float(inp.freq_ghz) * 1e3 / (mub * float(cy.value)) * 10.0
        if cy.theta is not None:
            th, ph = math.radians(cy.theta), math.radians(cy.phi)
        cycles.append(dict(field_mT=h0 / 10.0, theta=th, phi=ph, direction=_direction(th, ph),
                           zero_field=h0 == 0))
    exp = None
    if getattr(inp, "epr_setup", False) and float(inp.freq_ghz) > 0:
        exp_kw = dict(mwFreq=float(inp.freq_ghz), Mode="parallel" if float(inp.h1mode) < 0 else "perpendicular")
        if inp.populations:
            exp_kw["Temperature"] = float(inp.temperature)
        exp = Experiment(**exp_kw)
    notes = [_MIXED_NOTE] if float(getattr(inp, "h1mode", 0.0)) > 0 else []
    return TelserConversion(sy, exp, dict(cycles=cycles, energy_offset=_offset_for(p, constants), notes=notes,
                                          h1mode=float(getattr(inp, "h1mode", 0.0)), dropped=[]))


def ddpowj2sys(inp, *, constants=None, full_tensors: bool = False, strict: bool = True,
               **sys_kwargs) -> TelserConversion:
    """A DDPOWJ input (telser-tools ``ddpowj.DdpowjInput``) as a two-electron SpinSystem + pepper Experiment.

    Site 2's g and D have separate Euler angles (DDPOWJ v5.0).  W (MHz, HWHM) -> ``HStrain = 2 W``.  Use
    ``Options(Threshold=0)`` in pepper: its default transition pruning drops weak singlet-triplet lines.
    The DDPOWJ strain model (WMS1 x WMS2 basis weights, (H/Hset)^b) and Lorentzian lines are not mapped.
    """
    dropped: list = []
    if getattr(inp, "epr_shape", "G") in ("L", "l"):
        _drop("Lorentzian EPR line (frequency domain)", strict, dropped, "HStrain is Gaussian")
    if getattr(inp, "strain", False):
        _drop("DDPOWJ strain (WMS1 x WMS2 weights, (H/Hset)^b)", strict, dropped,
              "torchspin's strain models are different")
    p = _dimer_params(inp, separate_d_euler=True)
    hs = [2.0 * float(x) for x in inp.epr_width]
    extra = dict(sys_kwargs)
    if any(hs):
        extra.setdefault("HStrain", hs)
    sy = dimer2sys(p, constants=constants, full_tensors=full_tensors, **extra)
    temp = float(inp.temperature) if inp.populations else None
    exp = epr_experiment(inp.freq_ghz, inp.field_range, inp.npts, field_units=inp.field_units, deriv=inp.epr_deriv,
                         temperature=temp, h1mode=float(inp.h1mode), strict=strict, dropped=dropped)
    return TelserConversion(sy, exp, dict(dropped=dropped, energy_offset=_offset_for(p, constants),
                                          h1mode=float(inp.h1mode)))


# ----------------------------------------------------------------------------- small gaps filled here
class QuadrupoleEcho(NamedTuple):
    """DDZHQ's quadrupole echo: P_par = 3/2 Q_zz, eta = (Q_xx - Q_yy)/Q_zz, e^2qQ/h = 2I(2I - 1) Q_zz (MHz)."""

    p_par: torch.Tensor
    eta: torch.Tensor
    e2qQ: torch.Tensor


def quadrupole_echo(Q, I: float) -> QuadrupoleEcho | None:
    """P_par, eta and e^2qQ/h (MHz) of traceless principal values ``Q`` (MHz) for nuclear spin ``I`` >= 1.

    The axes are relabeled so that |Q_zz| >= |Q_yy| >= |Q_xx| (0 <= eta <= 1), the standard definition that
    telser-tools' exact mode prints (the Fortran's echo has eta <= 0, finding DDZHQ-08).  ``[e2qQ, eta]`` is
    EasySpin's/torchspin's two-value ``SpinSystem.Q`` form.  ``None`` if Q = 0.  Graph-safe in Q.
    """
    if float(I) < 1:
        raise ValueError("a quadrupole coupling needs I >= 1")
    q = _vec3(Q)
    order = torch.argsort(q.detach().abs(), stable=True)
    qxx, qyy, qzz = q[order[0]], q[order[1]], q[order[2]]
    if float(qzz.detach()) == 0.0:
        return None
    xi = float(I)
    return QuadrupoleEcho(1.5 * qzz, (qxx - qyy) / qzz, 2.0 * xi * (2.0 * xi - 1.0) * qzz)


def spin_projections(sys: SpinSystem, field_mT: float, direction) -> torch.Tensor:
    """<k|S_i|k>.n for every eigenstate k (ascending energy) and electron i: shape (nStates, nElectrons).

    DJDZE's "spin magnitudes" in telser-tools' exact form (the projection of each site's spin on the field
    direction n; the Fortran's sum_k <S_k> n_k^2 is finding DJDZE-07).  ``field_mT`` along the unit vector
    ``direction`` (molecular frame), with torchspin's Hamiltonian H0 - B.mu.  Within a degenerate level only
    the sum over the level is basis-free.
    """
    from torchspin.ham import ham
    from torchspin.spinops import sop

    n = _vec3(direction)
    H0, mx, my, mz = ham(sys)
    e, v = torch.linalg.eigh(H0 - float(field_mT) * (n[0] * mx + n[1] * my + n[2] * mz))
    spins = sys.Spins
    out = []
    for i in range(sys.nElectrons):
        sn = sum(n[c] * sop(spins, [i + 1, c + 1]) for c in range(3))
        out.append(torch.einsum("ak,ab,bk->k", v.conj(), sn.to(v.dtype), v).real)
    return torch.stack(out, dim=-1)
