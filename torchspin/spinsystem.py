"""SpinSystem dataclass for torchspin.

Mirrors EasySpin's ``Sys`` struct.  A ``SpinSystem`` collects all spin quantum
numbers, interaction tensors, and Euler frame angles needed to build the full
spin Hamiltonian.

Minimal usage::

    import torch
    from torchspin import SpinSystem

    # S=1/2 radical with g anisotropy
    sys = SpinSystem(S=[0.5], g=[[2.005, 2.004, 2.002]])

    # S=1/2 with one 14N nucleus and isotropic hyperfine
    sys = SpinSystem(
        S=[0.5],
        g=[[2.006, 2.006, 2.002]],
        Nucs=['14N'],
        A=[[10.0, 10.0, 50.0]],   # MHz
    )

Units
-----
* All fields involving energies / splittings are in **MHz** (matching EasySpin).
* Magnetic fields passed to Hamiltonian functions are in **mT**.
* Euler angles are in **radians**.
"""
from __future__ import annotations

import dataclasses
import math
import re
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from torchspin.nucdata import nucspin, nucgval, nucabund, nucdata
from torchspin.isotopologues import is_isotope_mixture


def _quadrupole_spin(nuc: str) -> float:
    """Nuclear spin for the Q shorthand conversion; for isotope mixtures the
    most abundant isotope with I >= 1 (0 if none)."""
    if not is_isotope_mixture(nuc):
        return float(_nuclear_spin(nuc))
    from torchspin.isotopologues import _parse_nucleus
    element, mass, abund = _parse_nucleus(nuc, None) if '(' not in nuc else (None, None, None)
    if element is None:   # custom mixture: use the listed isotopes
        inner = nuc[nuc.index('(') + 1:nuc.index(')')]
        element = nuc[nuc.index(')') + 1:]
        mass = [int(x) for x in inner.replace(' ', '').split(',') if x]
        abund = [1.0] * len(mass)
    best, best_ab = 0.0, -1.0
    for m, ab in zip(mass, abund):
        I = float(_nuclear_spin(f'{m}{element}'))
        if I >= 1 and ab > best_ab:
            best, best_ab = I, ab
    return best


def _nuclear_spin(isotope: str) -> float:
    """Return the nuclear spin quantum number for *isotope*."""
    return nucspin(isotope)


def _nuclear_gn(isotope: str) -> float:
    """Return the nuclear g-factor for *isotope*."""
    return nucgval(isotope)


# ---------------------------------------------------------------------------
# SpinSystem
# ---------------------------------------------------------------------------

@dataclass
class SpinSystem:
    """Spin system: spins, interactions and broadenings (EasySpin ``Sys``).

    A ``SpinSystem`` holds everything the Hamiltonian builders and simulators
    need: electron spins ``S``, nuclei ``Nucs`` (and optional orbital angular
    momenta ``L``), interaction tensors with their frames, line widths and
    strains, and simulator-specific fields (dynamics, relaxation, component
    weight).  Only ``S`` is required.

    Inputs may be scalars, nested lists, NumPy arrays or torch tensors; the
    constructor normalizes the tensor fields to ``torch.float64`` tensors of
    a canonical shape (listed below as "stored as"), converting EasySpin
    shorthands such as axial ``[g_perp, g_par]`` or ``[D, E]``.  Tensors are
    converted without copying or detaching, so a field that requires grad
    stays on the autograd graph (``pepper_autograd``, ``esfit``).  Assigning
    a field after construction (``sys.g = ...``) normalizes it the same way.

    Conventions: energies, couplings and strains in MHz; magnetic fields and
    field-domain line widths in mT; Euler angles in radians (``[alpha, beta,
    gamma]``, rotation from the molecular frame to the tensor's principal
    frame, :func:`torchspin.erot` convention); ``nElectrons = len(S)``,
    ``nNuclei = len(Nucs)``, ``nPairs = nElectrons*(nElectrons-1)/2`` (pairs
    ordered (1,2), (1,3), ..., (2,3), ...).  "Full tensors" are 3x3 blocks
    stacked along axis 0.

    Parameters
    ----------
    S : float or list of float
        Electron spin quantum numbers, one per electron spin, positive
        multiples of 1/2, e.g. ``0.5`` or ``[0.5, 1.0]``.  Stored as a list of
        float.
    g : float or array_like or torch.Tensor, optional
        Electron g tensors (dimensionless).  Accepted: a scalar (isotropic,
        every electron); ``[g1, g2, ...]`` or ``(nElectrons, 1)`` (one isotropic
        value per electron); ``[gx, gy, gz]`` (principal values, every
        electron); ``(nElectrons, 2)`` axial ``[g_perp, g_par]`` rows (for one
        electron also the flat ``[g_perp, g_par]``); ``(nElectrons, 3)``
        principal values; ``(3*nElectrons, 3)`` full tensors.  Stored as
        ``(nElectrons, 3)`` or ``(3*nElectrons, 3)``.  Default ``None``:
        isotropic free-electron g (``torchspin.GFREE``) for every electron, or
        zero when ``Ham`` contains a linear Zeeman term (``'110'``/``'112'``).
    gFrame : array_like, shape (nElectrons, 3), optional
        Euler angles (rad) of the g principal frames; ``[a, b, c]`` for one
        electron spin.  Default ``None`` or empty: zero.
    D : float or array_like or torch.Tensor, optional
        Zero-field splitting (MHz), needs S >= 1.  Accepted: a scalar ``D``
        (E = 0, every electron); for one electron ``[D]``, ``[D, E]`` or
        ``[Dxx, Dyy, Dzz]``; ``(nElectrons, 1)`` rows of D; ``(nElectrons, 2)``
        rows of ``[D, E]``; ``(nElectrons, 3)`` principal values;
        ``(3*nElectrons, 3)`` full tensors; for four or more electrons a flat
        list of one D per electron.  ``[D, E]`` becomes the principal values
        ``[-D/3 + E, -D/3 - E, 2D/3]``.  For two or three electrons a flat
        list of 2 or 3 values is ambiguous and raises.  Stored as
        ``(nElectrons, 3)`` or ``(3*nElectrons, 3)``.  Default ``None``: no
        zero-field splitting.
    D_ : array_like, shape (nElectrons, 2), optional
        Alternative to ``D``: ``[D, E/D]`` per electron (MHz, dimensionless),
        or the same values as a flat list.  Converted to ``D``.
    DFrame : array_like, shape (nElectrons, 3), optional
        Euler angles (rad) of the D principal frames, as ``gFrame``.
    B : list of array_like, optional
        Even-rank Stevens zero-field coefficients (MHz): ``B[0]`` is rank 2,
        ``B[1]`` rank 4, ``B[2]`` rank 6, each of shape ``(nElectrons, 2k+1)``
        with columns ``q = k, k-1, ..., -k``; ``None`` entries are skipped.
        The rank-2 entry is ignored when ``D`` is given (``validate`` rejects
        nonzero ``D`` together with nonzero ``B[0]``).  Default ``None``.
    BFrame : list of array_like, optional
        Euler angles (rad) per entry of ``B``: ``(nElectrons, 3)``, or
        ``(3,)`` for the first electron.  Default ``None`` (no rotation).
    Nucs : str or list of str, default []
        Nuclei: isotopes (``'1H'``, ``'14N'``, ``'63Cu'``), elements in
        natural abundance (``'Cu'``) or custom mixtures (``'(63,65)Cu'``, with
        ``Abund``).  A comma-separated string ``'14N,1H'`` is split.  Elements
        and mixtures are expanded into isotopologues by pepper, garlic and
        chili (module ``torchspin.isotopologues``, ``Options.IsoCutoff``).
    n : int or list of int, optional
        Number of equivalent nuclei per ``Nucs`` entry (same isotope, same
        isotropic coupling), positive integers.  Supported by garlic and by
        pepper with ``Method='perturb*'``/``'hybrid'``; pepper's matrix
        method and chili raise for ``n > 1``.  Default ``None``: 1 each.
    A : float or array_like or torch.Tensor, optional
        Hyperfine couplings (MHz).  Accepted: a scalar (isotropic, every
        nucleus and electron); for one electron ``[A1, A2, ...]`` (one
        isotropic value per nucleus) or ``[Ax, Ay, Az]`` (every nucleus;
        with three nuclei a flat triple is one value per nucleus);
        ``(nNuclei, nElectrons)`` isotropic, ``(nNuclei, 2*nElectrons)`` axial
        ``[A_perp, A_par]`` and ``(nNuclei, 3*nElectrons)`` principal values
        (one 3-block per electron); ``(3*nNuclei, 3*nElectrons)`` full
        tensors.  A flat list is one nucleus (e.g. ``[A_e1, A_e2]`` for two
        electrons).  Stored as ``(nNuclei, 3*nElectrons)`` or
        ``(3*nNuclei, 3*nElectrons)``.  Default ``None``: zero (with nuclei).
        Giving ``A``, ``AFrame``, ``Q`` or ``QFrame`` without ``Nucs`` raises.
    A_ : array_like, optional
        Alternative to ``A`` in spherical form per nucleus and electron:
        ``[aiso]``, ``[aiso, T]`` or ``[aiso, T, rho]`` (MHz), one row per
        nucleus; principal values ``aiso + T*[-1, -1, 2] + rho*[-1, 1, 0]``.
        A scalar is ``aiso`` of one nucleus.
    AFrame : array_like, shape (nNuclei, 3*nElectrons), optional
        Euler angles (rad) of the hyperfine principal frames, one 3-block per
        electron; for one electron ``[a, b, c]`` applies to every nucleus.
        Default ``None`` or empty: zero.
    ee : float or array_like or torch.Tensor, optional
        Electron-electron exchange/dipolar couplings (MHz),
        ``H = S1 . J . S2`` per pair.  Accepted: a scalar (isotropic, every
        pair); one isotropic value per pair (``[J]`` for one pair,
        ``(nPairs, 1)``); ``[Jx, Jy, Jz]`` (every pair; for three pairs a
        flat triple is read this way); ``(nPairs, 3)`` principal values;
        ``(3*nPairs, 3)`` full tensors.  Default ``None``: zero.
    ee2 : float or array_like, optional
        Isotropic biquadratic exchange (MHz), one value per pair, adds
        ``ee2 * (S1 . S2)**2``.  Default ``None``: zero.
    eeFrame : array_like, shape (nPairs, 3), optional
        Euler angles (rad) of the ``ee`` principal frames; ``[a, b, c]``
        applies to every pair.  Default ``None``: zero.
    gnscale : float or array_like, shape (nNuclei,), optional
        Scale factor of each nuclear g factor (dimensionless); a scalar
        applies to every nucleus.  Default ``None``: 1.
    sigma : float or array_like, optional
        Chemical-shielding factor of the nuclear Zeeman term, which is built
        as ``gn * muN * B . sigma . I``, so ``1`` means no shielding (EasySpin
        convention, not ``1 - sigma``).  A scalar, one value per nucleus,
        ``[sx, sy, sz]`` (every nucleus; with three nuclei a flat triple is
        one value per nucleus), ``(nNuclei, 3)`` principal values or
        ``(3*nNuclei, 3)`` full tensors.  Default ``None``: ones.
    sigmaFrame : array_like, shape (nNuclei, 3), optional
        Euler angles (rad) of the ``sigma`` principal frames.  Default zero.
    Q : float or array_like or torch.Tensor, optional
        Nuclear quadrupole couplings (MHz), for nuclei with I >= 1 (zero
        otherwise).  Accepted: one ``e2qQ/h`` per nucleus (eta = 0);
        ``(nNuclei, 2)`` rows of ``[e2qQ/h, eta]``; ``(nNuclei, 3)`` principal
        values; ``(3*nNuclei, 3)`` full tensors.  ``[e2qQ/h, eta]`` becomes
        ``e2qQ/h / (4I(2I-1)) * [-1 + eta, -1 - eta, 2]``.  Default ``None``:
        no quadrupole interaction.
    QFrame : array_like, shape (nNuclei, 3), optional
        Euler angles (rad) of the Q principal frames.  Default zero.
    nn : float or array_like, optional
        Nucleus-nucleus couplings (MHz), per nuclear pair: a scalar, one
        isotropic value per pair, ``[Jx, Jy, Jz]``, ``(nNucPairs, 3)``
        principal values or ``(3*nNucPairs, 3)`` full tensors.  Default
        ``None``: zero.
    nnFrame : array_like, shape (nNucPairs, 3), optional
        Euler angles (rad) of the ``nn`` principal frames.  Default zero.
    lw : float or array_like, optional
        Line width ``[GaussianFWHM, LorentzianFWHM]``, in mT for field sweeps
        and in MHz for frequency sweeps.  A scalar or ``[x]`` is ``[x, 0]``;
        ``[[G, L]]`` is ``[G, L]``; other sizes raise.  A 2-element list or
        tensor is kept as given (tensors keep their graph).  Default ``None``:
        ``[0.0, 0.0]`` (no broadening) unless ``lwpp`` is given.
    lwpp : float or array_like, optional
        Alternative to ``lw``: peak-to-peak widths of the first-derivative
        lines ``[Gaussian, Lorentzian]``, same units and shapes.  Converted by
        :meth:`get_lw` with the factors ``sqrt(2 ln 2)`` (Gaussian) and
        ``sqrt(3)`` (Lorentzian).  Giving both ``lw`` and ``lwpp`` raises.
    HStrain : float or array_like, optional
        Gaussian frequency-domain strain FWHM (MHz) along the molecular x, y,
        z axes.  One value is ``[x, x, x]``, two are ``[x, x, y]``, three
        are kept (any shape), empty is no strain; other counts raise.
        Stored as shape ``(3,)``.  Default ``None``.
    gStrain : float or array_like, optional
        g strain FWHM (dimensionless, g units) along the g principal axes,
        one row per electron: ``(nElectrons, 1|2|3)`` with ``[x]`` -> ``[x, x,
        x]`` and ``[x, y]`` -> ``[x, x, y]``; a flat list is one electron;
        a scalar or ``[x, y, z]`` applies to every electron.  Not with full
        g tensors; for several electrons not in frequency sweeps.  Stored as
        ``(nElectrons, 3)``.  Default ``None``.
    AStrain : float or array_like, optional
        Hyperfine strain FWHM (MHz) of the first nucleus along its AFrame
        axes, shapes as ``HStrain``.  One electron spin only (raises
        otherwise).  See KNOWN_LIMITATIONS.md, "AStrain — EasySpin's
        effective-mI model, followed deliberately (0.9.25)".  Default
        ``None``.
    DStrain : float or array_like, optional
        D and E strain FWHM (MHz), one row ``[FWHM_D, FWHM_E]`` per electron
        (1 to 3 columns; one column is ``[FWHM_D, 0]``, a third column is
        dropped).  One row or a scalar applies to every electron; for four or
        more electrons a flat list is one FWHM_D per electron.  Empty is no
        strain.  Stored as ``(nElectrons, 2)``.  Default ``None``.
    gAStrainCorr : float, default 1.0
        Correlation of g and A strain: ``+1`` correlated, ``-1``
        anticorrelated.
    DStrainCorr : float or list of float, default 0.0
        Correlation coefficient between the D and E strains, in ``[-1, 1]``,
        one per electron (a scalar applies to every electron).  Stored as a
        list.
    weight : float, default 1.0
        Relative weight of this component when a list of spin systems is
        simulated (pepper, garlic, chili, saffron); multiplied by the
        isotopologue abundances.
    Abund : list, optional
        Abundances for the custom isotope mixtures in ``Nucs``: one array (or
        ``None`` for a non-mixture) per nucleus, or a flat array for a
        single nucleus.  Default ``None``.
    initState : tuple or str or array_like, optional
        Non-equilibrium initial state instead of Boltzmann populations
        (pepper matrix method, spidyan, saffron): ``(pops, basis)`` or
        ``(rho, basis)`` with ``basis`` one of ``'zerofield'``, ``'eigen'``,
        ``'xyz'``, ``'coupled'``, ``'uncoupled'``; a density matrix in the
        uncoupled product basis; ``'T0'`` or ``'singlet'``.  See
        ``torchspin.initstate``.  Default ``None``.
    tdm : str or array_like, optional
        Transition dipole moment direction in the molecular frame for
        photoselection (``Experiment.lightBeam``, pepper): a string such as
        ``'x'``, ``'xz'``, ``'-y'``, a 3-vector or ``[phi, theta]`` (rad).
        Default ``None``.
    tcorr : float, optional
        Rotational correlation time in s (chili, cardamom, and garlic's
        fast-motion regime).  chili's slow-motion solver also accepts 2 or 3
        values (axial or rhombic).  Default ``None``.
    logtcorr : float, optional
        ``log10(tcorr / s)``, alternative to ``tcorr``.  Default ``None``.
    Diff : float or array_like, optional
        Rotational diffusion rate in s^-1, alternative to ``tcorr``
        (``tcorr = 1/(6 Diff)``); chili, cardamom and garlic (fast-motion
        regime, isotropic only).  Default ``None``.
    logDiff : float or array_like, optional
        ``log10(Diff / s^-1)``.  Default ``None``.
    T1, T2 : float, optional
        Longitudinal and transverse relaxation times in microseconds (pulse
        simulators saffron, spidyan).  Default ``None`` (no relaxation).
    lwEndor : float, optional
        ENDOR line width FWHM in MHz (saffron; required for Mims ENDOR).
        Default ``None``.
    Potential : array_like, optional
        Orienting potential: rows ``[L, M, K, lambda]`` (chili, cardamom), or
        a 3-D grid (cardamom).  Default ``None``.
    DiffGlobal : float or array_like, optional
        Global rotational diffusion rate in s^-1 (cardamom).  Default
        ``None``.
    TransRates, TransProb : array_like, shape (nStates, nStates), optional
        Jump-model transition rate matrix (s^-1) or transition probability
        matrix (cardamom).  Default ``None``.
    Orientations : array_like, optional
        Euler angles (rad) of each jump-model state, ``(nStates, 3)`` or
        ``(3, nStates)`` (cardamom).  Default ``None``.
    L : float or list of float, optional
        Orbital angular momentum quantum numbers, one per OAM; OAM ``i``
        couples to electron ``i`` through ``soc``.  Default ``None``.
    gL : float or list of float, optional
        Orbital g factors, one per OAM.  Default ``None``: 1.0 each.
    soc : array_like, shape (nElectrons, nOrders), optional
        Spin-orbit coupling (MHz); row ``i`` couples electron ``i`` and OAM
        ``i`` as ``soc[i,0] (S.L) + soc[i,1] (S.L)**2 + ...``.  A flat list
        is one row.  Default ``None``.
    CF1, ..., CF12 : array_like, shape (nL, 2k+1), optional
        Crystal-field Stevens coefficients of rank k (MHz) per OAM, columns
        ``q = k, ..., -k``; rank must not exceed ``2L`` (``validate``).
        Default ``None``.
    Ham : dict, optional
        Higher-order Zeeman terms (McGavin-Tennant-Weil), keys ``'{lB}{lS}{l}'``
        (orders in field and spin and coupling rank, e.g. ``'112'``; EasySpin's
        ``'Ham112'`` is accepted and stored as ``'112'``), values
        ``(nElectrons, 2l+1)`` coefficient arrays.  A linear term
        (``'110'``/``'112'``) replaces ``g`` (giving both raises).  Field
        orders 2 and 3 enter the resonance search only with
        ``Options(QuadraticZeeman=True)``; perturbation methods raise.  See
        KNOWN_LIMITATIONS.md, "Higher-order Zeeman terms (``Sys.Ham*``):
        deliberate deviation from EasySpin (0.6.7, 0.9.8, 0.9.13)".  Default
        ``None``.
    H0Extra : list of tuple, optional
        Extra field-independent Hamiltonian terms ``[(coeff, op), ...]``,
        added as ``sum(coeff * op)`` (MHz) by :func:`torchspin.ham`.
        ``coeff`` is a real scalar (float or 0-d tensor, may require grad);
        ``op`` an ``(nStates, nStates)`` Hermitian matrix in the full basis in
        :func:`torchspin.sop` order (electrons, nuclei, OAMs).  Not an
        EasySpin field.  Not with isotope mixtures or ``n > 1``; solvers that
        do not use ``ham()`` (perturbation theory, hybrid, garlic, chili,
        fastmotion, cardamom, saffron's fast algorithm, nucspinrmv/keep/add)
        raise.  See KNOWN_LIMITATIONS.md, "Simulator — extra
        field-independent operators (``Sys.H0Extra``) (0.7.2)".  Default
        ``None``.
    HStrainL : float or array_like, shape (3,), optional
        Lorentzian counterpart of ``HStrain`` (MHz FWHM along the molecular
        axes, a scalar is expanded, non-negative): the width of a line is
        ``sqrt(sum_k (HStrainL_k n_k)**2)`` for a field along ``n``, drawn as
        a Lorentzian.  pepper and pepper_autograd only; cannot be combined
        with the Gaussian strains.  Not an EasySpin field.  See
        KNOWN_LIMITATIONS.md, "Simulator — Lorentzian strain,
        ``SpinSystem.HStrainL`` (0.7.7)".  Default ``None``.

    Attributes
    ----------
    nElectrons, nNuclei, nL : int
        Number of electron spins, nuclei and orbital angular momenta.
    I, gn : list of float
        Nuclear spins and nuclear g factors from ``Nucs``.
    Spins : list of float
        All spin quantum numbers: electrons, nuclei, OAMs.
    nStates : int
        Hilbert-space dimension ``prod(2s + 1)``.
    fullg, fullD, fullA, fullee, fullsigma, fullQ, fullnn : bool
        Whether the field is stored as full 3x3 tensors.

    Raises
    ------
    ValueError
        On construction, for a tensor field whose shape matches none of the
        accepted forms (``g``, ``D``, ``D_``, ``A``, ``A_``, ``AFrame``,
        ``gFrame``, ``DFrame``, ``ee``, ``ee2``, ``eeFrame``, ``Q``,
        ``QFrame``, ``lw``, ``lwpp``, ``HStrain``, ``AStrain``, ``gStrain``,
        ``DStrain``, ``DStrainCorr``, ``n``), for alternative fields given together (``D`` and
        ``D_``, ``A`` and ``A_``, ``lw`` and ``lwpp``, ``g`` and a linear
        ``Ham`` term), for ``A``/``Q`` data without ``Nucs``, for negative
        ``HStrainL``, and for an invalid ``H0Extra`` term.
    TypeError
        If ``H0Extra`` is not a list of ``(coeff, op)`` pairs.

    See Also
    --------
    Experiment, Options, torchspin.ham, torchspin.pepper, torchspin.garlic

    Notes
    -----
    The constructor checks shapes only.  :meth:`validate` performs further
    physical checks (spin values, S >= 1 for D, I >= 1 for Q, signs of
    widths, CF ranks); the simulators do not call it.  Which fields a
    simulator reads, ignores or rejects is listed in that simulator's
    docstring.  See KNOWN_LIMITATIONS.md, "``SpinSystem`` input shapes follow
    EasySpin (0.9.15)" and "``SpinSystem`` field assignment normalizes
    (0.8.7)".

    Examples
    --------
    A nitroxide; axial ``A`` given as ``[A_perp, A_par]``:

    >>> from torchspin import SpinSystem
    >>> sys = SpinSystem(S=0.5, g=[2.0088, 2.0061, 2.0027],
    ...                  Nucs='14N', A=[[20.0, 95.0]], lw=0.5)
    >>> sys.g.shape, sys.A.tolist(), sys.lw
    (torch.Size([1, 3]), [[20.0, 20.0, 95.0]], [0.5, 0.0])
    >>> sys.nStates
    6

    A triplet with ``[D, E]`` in MHz, stored as principal values:

    >>> trip = SpinSystem(S=1, D=[300.0, 30.0])
    >>> [round(v, 1) for v in trip.D[0].tolist()]
    [-70.0, -130.0, 200.0]

    Assigning a field normalizes it:

    >>> trip.HStrain = 10.0
    >>> trip.HStrain.tolist()
    [10.0, 10.0, 10.0]
    """

    S: list[float]
    g: Optional[list | torch.Tensor] = None
    gFrame: Optional[list | torch.Tensor] = None
    D: Optional[list | torch.Tensor] = None
    D_: Optional[list | torch.Tensor] = None   # zero-field splitting as [D, E/D] per electron (EasySpin Sys.D_)
    DFrame: Optional[list | torch.Tensor] = None
    B: Optional[list[Optional[torch.Tensor]]] = None
    BFrame: Optional[list[Optional[torch.Tensor]]] = None
    Nucs: list[str] = field(default_factory=list)
    n: Optional[list[int]] = None  # Number of equivalent nuclei per entry in Nucs (EasySpin Sys.n); default 1
    A: Optional[list | torch.Tensor] = None
    A_: Optional[list | torch.Tensor] = None  # spherical hyperfine form per nucleus/electron: [aiso], [aiso T] or [aiso T rho] (EasySpin Sys.A_)
    AFrame: Optional[list | torch.Tensor] = None
    ee: Optional[list | torch.Tensor] = None
    ee2: Optional[list | torch.Tensor] = None  # isotropic biquadratic exchange per electron pair, +ee2·(S1·S2)² (MHz)
    eeFrame: Optional[list | torch.Tensor] = None
    # Nuclear interaction fields
    gnscale: Optional[list | torch.Tensor] = None
    sigma: Optional[list | torch.Tensor] = None
    sigmaFrame: Optional[list | torch.Tensor] = None
    Q: Optional[list | torch.Tensor] = None
    QFrame: Optional[list | torch.Tensor] = None
    nn: Optional[list | torch.Tensor] = None
    nnFrame: Optional[list | torch.Tensor] = None
    # Linewidth for spectrum simulation
    lw: Optional[list[float]] = None
    lwpp: Optional[list[float]] = None  # Peak-to-peak linewidth (alternative to lw)
    # Strain parameters for anisotropic broadening
    HStrain: Optional[list | torch.Tensor] = None  # [3] MHz, molecular frame
    gStrain: Optional[list | torch.Tensor] = None  # [nElectrons, 3]
    AStrain: Optional[list | torch.Tensor] = None  # [3] MHz (first nucleus only)
    DStrain: Optional[list | torch.Tensor] = None  # MHz, [FWHM_D, FWHM_E] (one row per electron)
    gAStrainCorr: float = 1.0  # +1 (correlated) or -1 (anticorrelated)
    DStrainCorr: float | list[float] = 0.0  # Correlation coefficient between D and E (per electron)
    # Multi-component spectra (mirrors MATLAB Sys.weight)
    weight: float = 1.0  # Relative intensity weight for multi-component simulations
    Abund: Optional[list] = None  # isotope abundances for custom mixtures in Nucs, e.g. '(63,65)Cu' (EasySpin Sys.Abund)
    initState: Optional[object] = None  # non-equilibrium state: (pops_or_rho, basis), density matrix, 'T0', 'singlet' (EasySpin Sys.initState)
    tdm: Optional[object] = None        # transition dipole moment direction in the molecular frame: 'x', [phi theta] or 3-vector (EasySpin Sys.tdm)
    # Slow-motion dynamics (chili)
    tcorr: Optional[float] = None    # Isotropic rotational correlation time (s)
    logtcorr: Optional[float] = None # log10(tcorr) — alternative to tcorr
    Diff: Optional[float] = None     # Isotropic rotational diffusion coefficient (s^-1)
    logDiff: Optional[float] = None  # log10(Diff) — alternative to Diff
    # Pulse EPR (saffron)
    T1: Optional[float] = None       # Longitudinal relaxation time (µs)
    T2: Optional[float] = None       # Transverse relaxation time (µs)
    lwEndor: Optional[float] = None  # ENDOR linewidth FWHM (MHz)
    # Trajectory-based simulation (cardamom)
    Potential: Optional[list | torch.Tensor | np.ndarray] = None  # Orienting potential: [L,M,K,λ] rows or 3D grid
    DiffGlobal: Optional[float | list] = None  # Global rotational diffusion rate (s⁻¹)
    TransRates: Optional[list | torch.Tensor | np.ndarray] = None  # Transition rate matrix (nStates×nStates)
    TransProb: Optional[list | torch.Tensor | np.ndarray] = None   # Transition probability matrix (nStates×nStates)
    Orientations: Optional[list | torch.Tensor | np.ndarray] = None  # Euler angles per state (nStates×3 or 3×nStates)
    # Orbital angular momentum (OAM)
    L: Optional[list[float]] = None       # OAM quantum numbers, one per OAM
    gL: Optional[list[float]] = None      # Orbital g-factors (default 1.0 per OAM)
    soc: Optional[list | torch.Tensor] = None  # Spin-orbit coupling [nElectrons, nOrders] MHz
    # Crystal-field parameters (Stevens operator coefficients, MHz)
    CF1: Optional[list | torch.Tensor] = None   # rank 1, shape (nL, 3)
    CF2: Optional[list | torch.Tensor] = None   # rank 2, shape (nL, 5)
    CF3: Optional[list | torch.Tensor] = None   # rank 3, shape (nL, 7)
    CF4: Optional[list | torch.Tensor] = None   # rank 4, shape (nL, 9)
    CF5: Optional[list | torch.Tensor] = None   # rank 5, shape (nL, 11)
    CF6: Optional[list | torch.Tensor] = None   # rank 6, shape (nL, 13)
    CF7: Optional[list | torch.Tensor] = None   # rank 7, shape (nL, 15)
    CF8: Optional[list | torch.Tensor] = None   # rank 8, shape (nL, 17)
    CF9: Optional[list | torch.Tensor] = None   # rank 9, shape (nL, 19)
    CF10: Optional[list | torch.Tensor] = None  # rank 10, shape (nL, 21)
    CF11: Optional[list | torch.Tensor] = None  # rank 11, shape (nL, 23)
    CF12: Optional[list | torch.Tensor] = None  # rank 12, shape (nL, 25)
    # Higher-order Zeeman (McGavin-Tennant-Weil formalism)
    # Keys: '{lB}{lS}{l}' strings (e.g., '110', '112', '222')
    # Values: (nElectrons, 2*l+1) arrays of coefficients
    Ham: Optional[dict] = None
    # Generic extra field-independent terms: H0 += sum_k coeff_k * op_k (MHz).
    # coeff: real scalar (float or 0-d tensor, may require grad); op: (n, n)
    # Hermitian matrix in the full spin-system basis (torchspin order, as sop
    # builds it). Default None: no extra term.
    H0Extra: Optional[list] = None
    # Lorentzian frequency-domain strain width, [3] MHz FWHM, molecular frame
    # (pepper only; not with the Gaussian strains). Default None.
    HStrainL: Optional[list | torch.Tensor] = None

    def __post_init__(self) -> None:
        from torchspin.constants import GFREE

        # Normalize S
        if isinstance(self.S, (int, float)):
            self.S = [float(self.S)]
        else:
            self.S = [float(s) for s in self.S]

        # Normalize Nucs
        if isinstance(self.Nucs, str):
            if self.Nucs:
                # Split comma-separated nuclei, strip whitespace
                # split on commas outside parentheses ('(63,65)Cu,N' → ['(63,65)Cu', 'N'])
                self.Nucs = [nuc.strip() for nuc in re.split(r',(?![^()]*\))', self.Nucs) if nuc.strip()]
            else:
                self.Nucs = []

        # Normalize L (orbital angular momenta)
        if self.L is not None:
            if isinstance(self.L, (int, float)):
                self.L = [float(self.L)]
            else:
                self.L = [float(l) for l in self.L]
            # Default gL to 1.0 per OAM
            if self.gL is None:
                self.gL = [1.0] * len(self.L)
            elif isinstance(self.gL, (int, float)):
                self.gL = [float(self.gL)] * len(self.L)
            else:
                self.gL = [float(gl) for gl in self.gL]
            # Normalize soc to tensor
            if self.soc is not None:
                self.soc = _to_tensor(self.soc, torch.float64)
                if self.soc.ndim == 1:
                    self.soc = self.soc.unsqueeze(0)

        # --- Ham (higher-order Zeeman) ---
        if self.Ham is not None:
            ham_dict = {}
            for key, val in self.Ham.items():
                # B103: 'Ham112' (EasySpin field name) and '112' are the same
                # term; the Hamiltonian code reads the digits only
                k = str(key)
                k = k[3:] if k.startswith('Ham') else k
                if len(k) < 3 or not k.isdigit():
                    raise ValueError(f"SpinSystem.Ham: key {key!r} is not of the form "
                                     "'{lB}{lS}{l}' (e.g. '112' or 'Ham112').")
                if k in ham_dict:
                    raise ValueError(f"SpinSystem.Ham: term {k!r} given twice.")
                ham_dict[k] = _to_tensor(val, torch.float64)
                if ham_dict[k].ndim == 1:
                    ham_dict[k] = ham_dict[k].unsqueeze(0)
            self.Ham = ham_dict

        n_e = self.nElectrons
        n_n = self.nNuclei
        from itertools import combinations
        n_pairs = sum(1 for _ in combinations(range(n_e), 2))

        # --- g tensor ---
        # EasySpin: Sys.g may not be combined with Ham110/Ham112 (the linear
        # Zeeman term is then given entirely by the Ham parameters, g = 0).
        _ham_linear = self.Ham is not None and any(
            k in ('110', '112', 'Ham110', 'Ham112') for k in self.Ham.keys()
        )
        if self.g is None:
            if _ham_linear:
                self.g = torch.zeros(n_e, 3, dtype=torch.float64)
            else:
                # Default: isotropic gfree for each electron
                self.g = torch.tensor(
                    [[GFREE, GFREE, GFREE]] * n_e, dtype=torch.float64
                )
        else:
            self.g = _to_tensor(self.g, torch.float64)
            if _ham_linear and bool(torch.any(self.g != 0)):
                raise ValueError(
                    "Cannot use g and Ham112 or Ham110 simultaneously. Remove one of them."
                )
            # Normalize g to shape (nElectrons, 3) or (3*nElectrons, 3)
            if self.g.ndim == 0:
                # Scalar: isotropic g for all electrons
                self.g = self.g.expand(n_e, 3).clone()
            elif self.g.ndim == 1:
                if self.g.shape[0] == n_e:
                    # [g1, g2, ...]: one isotropic g per electron (EasySpin column form)
                    self.g = self.g.unsqueeze(1).expand(n_e, 3).clone()
                elif self.g.shape[0] == 3:
                    # [gx, gy, gz] for one electron (replicated for all)
                    self.g = self.g.unsqueeze(0).expand(n_e, 3).clone()
                elif self.g.shape[0] == 2 and n_e == 1:
                    # [g_perp, g_par]: a 1x2 row is EasySpin's (nElectrons, 2)
                    # axial form for one electron (validatespinsys.m:141-143)
                    self.g = self.g[[0, 0, 1]].unsqueeze(0)
            elif self.g.ndim == 2 and self.g.numel() == n_e:
                # (nElectrons, 1) or (1, nElectrons): one isotropic g per electron
                # (numel(Sys.g) == nElectrons, validatespinsys.m:138-140)
                self.g = self.g.reshape(n_e, 1).expand(n_e, 3).clone()
            elif self.g.ndim == 2 and self.g.shape == (n_e, 2):
                # axial [g_perp, g_par] per electron -> [g_perp g_perp g_par]
                # (validatespinsys.m:141-143)
                self.g = self.g[:, [0, 0, 1]]
            # EasySpin accepts (3*nElectrons, 3), numel == nElectrons,
            # (nElectrons, 2) and (nElectrons, 3); anything else is 'Sys.g has
            # wrong size.' (validatespinsys.m:135-149)
            if not (self.g.ndim == 2 and tuple(self.g.shape) in ((n_e, 3), (3 * n_e, 3))):
                raise ValueError(
                    f'g has wrong size {tuple(self.g.shape)} for {n_e} electron spin(s): expected a '
                    f'scalar, one value per electron spin, [gx, gy, gz], (nElectrons, 1), '
                    f'(nElectrons, 2) axial, (nElectrons, 3) or full tensors (3*nElectrons, 3).')

        # --- gFrame ---
        self.gFrame = _euler_rows(self.gFrame, n_e, 'gFrame')

        # --- D_ = [D, E/D] per electron (EasySpin Sys.D_) → D = [D, E] ---
        if self.D_ is not None:
            if self.D is not None:
                raise ValueError('Zero-field splitting given in both D and D_. Please remove one of them.')
            D_ = _to_tensor(self.D_, torch.float64)
            if D_.numel() == 0 or D_.numel() % 2:
                # EasySpin needs one [D, E/D] row per electron: 'Sys.D_ has
                # wrong size.' (validatespinsys.m:220-230)
                raise ValueError(
                    f'D_ has wrong size {tuple(D_.shape)}: expected one [D, E/D] row per electron '
                    f'spin ({n_e}, 2).')
            D_ = D_.reshape(-1, 2)
            if D_.shape[0] != n_e:
                raise ValueError('D_ has wrong size: expected one [D, E/D] row per electron.')
            self.D = torch.stack([D_[:, 0], D_[:, 0] * D_[:, 1]], dim=1)
            self.D_ = D_
        # --- D tensor ---
        if self.D is not None:
            self.D = _to_tensor(self.D, torch.float64)
            # Normalize D: scalar -> (n_e, 3), vector of length 3 -> (n_e, 3)
            if self.D.ndim == 0:
                # Scalar D (EasySpin: D with E = 0) -> [-1 -1 2]/3 * D for all electrons
                self.D = (self.D * torch.tensor([-1.0, -1.0, 2.0], dtype=torch.float64) / 3).unsqueeze(0).expand(n_e, 3).clone()
            elif self.D.ndim == 1 and n_e > 1 and self.D.shape[0] == n_e and n_e not in (2, 3):
                # [D1, D2, ...]: one scalar D (E = 0) per electron (EasySpin column form)
                self.D = (self.D.unsqueeze(1) * torch.tensor([-1.0, -1.0, 2.0], dtype=torch.float64) / 3).clone()
            elif self.D.ndim == 2 and self.D.shape == (n_e, 1):
                # (n_e, 1) column: one scalar D (E = 0) per electron
                self.D = (self.D * torch.tensor([-1.0, -1.0, 2.0], dtype=torch.float64) / 3).clone()
            elif self.D.ndim == 1:
                if self.D.shape[0] == 1:
                    # [D]: one D value (E = 0), as a scalar D. EasySpin
                    # (validatespinsys.m:208-210) reads numel(Sys.D) == nElectrons
                    # as one D per electron, so a single value is valid only for
                    # one electron spin; for several it is 'Sys.D has wrong size.'
                    if n_e != 1:
                        raise ValueError(
                            f'D has wrong size: a one-element D is one D value for one electron spin, '
                            f'but the system has {n_e}. Give a scalar D (same D, E = 0 for every '
                            f'electron) or one row per electron spin.')
                    self.D = (self.D * torch.tensor([-1.0, -1.0, 2.0], dtype=torch.float64) / 3).unsqueeze(0).clone()
                elif self.D.shape[0] in (2, 3) and n_e > 1:
                    # B61: a flat [a, b] or [a, b, c] for several electron spins
                    # is ambiguous.  EasySpin (validatespinsys.m:208-210) reads
                    # numel(Sys.D) == nElectrons as one D per electron, while
                    # torchspin up to 0.9.14 read it as one shared [D, E] /
                    # [Dxx, Dyy, Dzz]; for nElectrons != numel it is EasySpin's
                    # 'Sys.D has wrong size.'.  Neither reading is adopted: raise
                    # and ask for one row per electron (deliberate deviation).
                    k = self.D.shape[0]
                    raise ValueError(
                        f'D is ambiguous for {n_e} electron spins: a flat list of {k} values could be '
                        f'one {"[D, E]" if k == 2 else "[Dxx, Dyy, Dzz]"} shared by every electron '
                        f'(torchspin up to 0.9.14) or one D value per electron (EasySpin). Give one row '
                        f'per electron spin, e.g. D=[[D1, E1], [D2, E2]], D=[[Dxx, Dyy, Dzz], ...], '
                        f'D=[[D1], [D2]], or D_; a scalar D applies the same D (E = 0) to every '
                        f'electron.')
                elif self.D.shape[0] == 2:
                    # EasySpin [D, E] shorthand (one electron): convert to [Dxx, Dyy, Dzz]
                    D_val, E_val = self.D[0], self.D[1]
                    Dxx = -D_val / 3 + E_val
                    Dyy = -D_val / 3 - E_val
                    Dzz = 2 * D_val / 3
                    principal = torch.stack([Dxx, Dyy, Dzz])
                    self.D = principal.unsqueeze(0).clone()
                elif self.D.shape[0] == 3:
                    # [Dxx, Dyy, Dzz] for one electron
                    self.D = self.D.unsqueeze(0).clone()
            elif self.D.ndim == 2 and self.D.shape[1] == 2:
                # (n_e, 2) EasySpin [D, E] per electron: convert each row
                D_vals = self.D[:, 0]
                E_vals = self.D[:, 1]
                Dxx = -D_vals / 3 + E_vals
                Dyy = -D_vals / 3 - E_vals
                Dzz = 2 * D_vals / 3
                self.D = torch.stack([Dxx, Dyy, Dzz], dim=1).clone()
            # Otherwise assume correct 2D shape
            if not (self.D.ndim == 2 and tuple(self.D.shape) in ((n_e, 3), (3 * n_e, 3))):
                raise ValueError(
                    f'D has wrong size {tuple(self.D.shape)} for {n_e} electron spin(s): expected a '
                    f'scalar, [D, E], [Dxx, Dyy, Dzz], one row per electron spin ((nElectrons, 1), '
                    f'(nElectrons, 2) or (nElectrons, 3)) or full tensors (3*nElectrons, 3).')

        # --- DFrame ---
        self.DFrame = _euler_rows(self.DFrame, n_e, 'DFrame')

        # --- B (Stevens ZFS coefficients) ---
        # B is a list indexed by rank offset: B[0] = k=2, B[1] = k=4, etc.
        # Each B[i] is (nElectrons, 2*k+1) for the corresponding k
        if self.B is not None:
            # Convert each element to tensor if present
            for i, Bk in enumerate(self.B):
                if Bk is not None:
                    self.B[i] = _to_tensor(Bk, torch.float64)

        # --- BFrame ---
        if self.BFrame is not None:
            for i, BFk in enumerate(self.BFrame):
                if BFk is not None:
                    self.BFrame[i] = _to_tensor(BFk, torch.float64)

        # --- A, AFrame, Q, QFrame without nuclei ---
        # EasySpin (validatespinsys.m:444-452) rejects these when Nucs is empty;
        # torchspin ignored them.  QFrame defaults to a zero row even without
        # nuclei (and is passed on by rebuilds), so for it only nonzero angles
        # count as given.
        if n_n == 0:
            def _given(v, nonzero=False):
                if v is None:
                    return False
                t = _to_tensor(v, torch.float64)
                return t.numel() > 0 and (not nonzero or bool(torch.any(t != 0)))
            if (_given(self.A) or _given(self.AFrame) or _given(self.Q)
                    or _given(self.QFrame, nonzero=True)):
                raise ValueError(
                    'The system contains A and/or Q fields, but no nucleus is specified! '
                    'Give the nuclei in Nucs.')

        # --- n (equivalent nuclei, EasySpin Sys.n) ---
        # Each entry of Nucs may stand for a group of n identical, equivalent
        # nuclei (same isotope, same isotropic coupling).  Only garlic uses
        # this (via equivcouple); pepper/chili reject n>1 like EasySpin.
        if self.n is None:
            self.n = [1] * n_n
        else:
            if isinstance(self.n, torch.Tensor):
                self.n = self.n.flatten().tolist()
            elif isinstance(self.n, (int, float)):
                self.n = [self.n]
            self.n = [int(round(float(v))) for v in self.n]
            if len(self.n) != n_n:
                raise ValueError(
                    f"SpinSystem.n has {len(self.n)} entries, expected one per nucleus ({n_n})."
                )
            if any(v < 1 for v in self.n):
                raise ValueError("SpinSystem.n entries must be positive integers.")

        # --- A_ (spherical hyperfine form, EasySpin validatespinsys) → A principal values
        #     A = aiso + T·[-1 -1 2] + rho·[-1 +1 0], per (nucleus, electron) block ---
        if n_n > 0 and self.A_ is not None:
            if self.A is not None:
                raise ValueError('Hyperfine data given in both A and A_. Please remove one of them.')
            A_ = _to_tensor(self.A_, torch.float64)
            if A_.ndim == 0:
                # scalar aiso: EasySpin's [1 nNuclei] row for one nucleus
                # (validatespinsys.m:540-543)
                A_ = A_.reshape(1, 1)
            if A_.ndim == 1:
                A_ = A_.reshape(n_n, -1) if (A_.numel() == n_n and n_n > 1) else A_.reshape(1, -1)
            if A_.shape[0] != n_n:
                raise ValueError(f'A_ must have one row per nucleus ({n_n}).')
            per_e = A_.shape[1] // n_e
            if A_.shape[1] != per_e * n_e or per_e not in (1, 2, 3):
                raise ValueError('A_ needs 1, 2 or 3 values per nucleus and electron: [aiso], [aiso T] or [aiso T rho].')
            cols = []
            for e in range(n_e):
                blk = A_[:, e * per_e:(e + 1) * per_e]
                aiso = blk[:, 0]
                T = blk[:, 1] if per_e >= 2 else torch.zeros(n_n, dtype=torch.float64)
                rho = blk[:, 2] if per_e == 3 else torch.zeros(n_n, dtype=torch.float64)
                cols.append(torch.stack([aiso - T - rho, aiso - T + rho, aiso + 2 * T], dim=1))
            self.A = torch.cat(cols, dim=1)
            self.A_ = A_
        # --- A tensor ---
        if n_n > 0 and self.A is not None:
            self.A = _to_tensor(self.A, torch.float64)
            self.A = _normalize_A(self.A, n_n, n_e)
        elif n_n > 0 and self.A is None:
            # zero hyperfine by default
            self.A = torch.zeros(n_n, 3 * n_e, dtype=torch.float64)

        # --- AFrame ---
        # EasySpin (validatespinsys.m:624-628): empty -> zeros, otherwise exactly
        # (nNuclei, 3*nElectrons).  A 1-D vector is a MATLAB 1xk row, i.e. one
        # nucleus; [alpha, beta, gamma] with one electron spin is applied to
        # every nucleus (torchspin shorthand).
        if self.AFrame is None:
            self.AFrame = torch.zeros(n_n, 3 * n_e, dtype=torch.float64)
        elif n_n > 0:
            self.AFrame = _to_tensor(self.AFrame, torch.float64)
            if self.AFrame.numel() == 0:
                self.AFrame = torch.zeros(n_n, 3 * n_e, dtype=torch.float64)
            elif self.AFrame.ndim == 1:
                if self.AFrame.shape[0] == 3 and n_e == 1:
                    # Euler angles for one nucleus, one electron
                    self.AFrame = self.AFrame.unsqueeze(0).expand(n_n, 3).clone()
                else:
                    self.AFrame = self.AFrame.unsqueeze(0)
            if tuple(self.AFrame.shape) != (n_n, 3 * n_e):
                raise ValueError(
                    f'Spin system field AFrame has wrong size {tuple(self.AFrame.shape)} for the given '
                    f'spins: expected ({n_n}, {3 * n_e}) Euler angles (nNuclei, 3*nElectrons), or '
                    f'[alpha, beta, gamma] for every nucleus with one electron spin.')

        # --- ee2 (biquadratic exchange, one value per electron pair) ---
        if n_pairs > 0:
            if self.ee2 is None:
                self.ee2 = torch.zeros(n_pairs, dtype=torch.float64)
            else:
                self.ee2 = _to_tensor(self.ee2, torch.float64).reshape(-1)
                if self.ee2.numel() != n_pairs:
                    raise ValueError(f'ee2 must have one value per electron pair ({n_pairs}).')
        # --- ee tensor ---
        if n_pairs > 0 and self.ee is not None:
            self.ee = _to_tensor(self.ee, torch.float64)
            # Normalize ee: scalar -> (n_pairs, 3), vector of length 3 -> (n_pairs, 3)
            if self.ee.ndim == 0:
                # Scalar ee: replicate for all pairs
                self.ee = self.ee.expand(n_pairs, 3).clone()
            elif self.ee.ndim == 1:
                if self.ee.shape[0] == 3 and n_pairs != 3:
                    # [Jxx, Jyy, Jzz] for one pair
                    self.ee = self.ee.unsqueeze(0).expand(n_pairs, 3).clone()
                elif self.ee.shape[0] == n_pairs:
                    # One isotropic J per pair (EasySpin column form)
                    self.ee = self.ee.unsqueeze(1).expand(n_pairs, 3).clone()
                elif self.ee.shape[0] == 3:
                    # Ambiguous n_pairs == 3 with 3 values: treat as
                    # principal values for every pair (legacy behavior)
                    self.ee = self.ee.unsqueeze(0).expand(n_pairs, 3).clone()
            elif self.ee.ndim == 2 and self.ee.numel() == n_pairs:
                # (nPairs, 1) or (1, nPairs): one isotropic coupling per pair
                # (numel(Sys.ee) == nElPairs, validatespinsys.m:329-331)
                self.ee = self.ee.reshape(n_pairs, 1).expand(n_pairs, 3).clone()
            # EasySpin then requires (3*nPairs, 3) full or (nPairs, 3) principal
            # values (validatespinsys.m:333-338)
            if not (self.ee.ndim == 2 and tuple(self.ee.shape) in ((n_pairs, 3), (3 * n_pairs, 3))):
                raise ValueError(
                    f'Spin system field ee has wrong size {tuple(self.ee.shape)} for {n_pairs} electron '
                    f'pair(s): expected a scalar, one value per pair, [Jx, Jy, Jz], (nPairs, 3) '
                    f'principal values or (3*nPairs, 3) full tensors.')
        elif n_pairs > 0 and self.ee is None:
            self.ee = torch.zeros(n_pairs, 3, dtype=torch.float64)

        # --- eeFrame ---
        if self.eeFrame is None:
            self.eeFrame = torch.zeros(max(n_pairs, 1), 3, dtype=torch.float64)
        else:
            self.eeFrame = _to_tensor(self.eeFrame, torch.float64)
            # Normalize eeFrame: 1D [α, β, γ] for one pair -> (n_pairs, 3)
            if self.eeFrame.ndim == 1 and self.eeFrame.shape[0] == 3:
                self.eeFrame = self.eeFrame.unsqueeze(0).expand(n_pairs, 3).clone()
            # EasySpin: exactly (nPairs, 3) (validatespinsys.m:414-416)
            if n_pairs > 0 and tuple(self.eeFrame.shape) != (n_pairs, 3):
                raise ValueError(
                    f'Spin system field eeFrame has wrong size {tuple(self.eeFrame.shape)} for the given '
                    f'spins: expected ({n_pairs}, 3) Euler angles, one row per electron pair.')

        # --- gnscale ---
        if self.gnscale is None:
            self.gnscale = torch.ones(max(n_n, 1), dtype=torch.float64)
        else:
            self.gnscale = _to_tensor(self.gnscale, torch.float64).reshape(-1)
            if self.gnscale.numel() == 1 and n_n > 1:
                self.gnscale = self.gnscale.expand(n_n).clone()

        # --- sigma (chemical shielding) ---
        # Convention matches MATLAB EasySpin: sigma defaults to ones(nNuclei, 3),
        # and ham_nz uses pre * sigma * Ik directly (no (I-sigma) transformation).
        def _per_nucleus_rows(T: torch.Tensor, n_rows: int, isotropic_scalar: bool) -> torch.Tensor:
            """Normalize 0-D / 1-D per-nucleus (or per-pair) input to (n_rows, 3)."""
            if T.ndim == 0:
                return T.expand(n_rows, 3).clone()
            if T.ndim == 1:
                if n_rows > 1 and T.shape[0] == n_rows and (n_rows != 3 or isotropic_scalar):
                    # one isotropic value per row (EasySpin column form)
                    return T.unsqueeze(1).expand(n_rows, 3).clone()
                if T.shape[0] == 3:
                    return T.unsqueeze(0).expand(n_rows, 3).clone()
                if T.shape[0] == 3 * n_rows:
                    return T.reshape(n_rows, 3).clone()
            if T.ndim == 2 and T.shape == (n_rows, 1):
                return T.expand(n_rows, 3).clone()
            return T

        if self.sigma is None:
            self.sigma = torch.ones(max(n_n, 1), 3, dtype=torch.float64)
        else:
            self.sigma = _per_nucleus_rows(_to_tensor(self.sigma, torch.float64), max(n_n, 1), True)

        # --- sigmaFrame ---
        if self.sigmaFrame is None:
            self.sigmaFrame = torch.zeros(max(n_n, 1), 3, dtype=torch.float64)
        else:
            self.sigmaFrame = _per_nucleus_rows(_to_tensor(self.sigmaFrame, torch.float64), max(n_n, 1), False)

        # --- Q tensor ---
        # EasySpin forms (validatespinsys): one value per nucleus (e2qQ/h, eta=0),
        # [e2qQ/h eta] per nucleus, principal values [Qx Qy Qz] per nucleus, or a
        # full (3·nNuclei, 3) tensor.  The first two are converted to principal
        # values e2qQ/h/(4I(2I-1))·[-1+eta, -1-eta, 2] (zero for I < 1).
        if self.Q is not None:
            self.Q = _to_tensor(self.Q, torch.float64)
            Q = self.Q
            if Q.ndim == 0:
                Q = Q.reshape(1, 1)
            elif Q.ndim == 1:
                if Q.shape[0] == n_n:
                    Q = Q.reshape(n_n, 1)
                elif Q.shape[0] == 2 * n_n:
                    Q = Q.reshape(n_n, 2)
                elif Q.shape[0] == 3 * n_n:
                    Q = Q.reshape(n_n, 3)
            if Q.ndim == 2 and Q.shape[0] == n_n and Q.shape[1] in (1, 2):
                rows = []
                for i in range(n_n):
                    nuc = self.Nucs[i]
                    I_q = _quadrupole_spin(nuc)
                    # torch ops only, so a Q that requires grad stays on the graph
                    e2qQh = Q[i, 0]
                    eta = Q[i, 1] if Q.shape[1] == 2 else torch.zeros_like(e2qQh)
                    if I_q < 1:
                        rows.append(torch.zeros(3, dtype=torch.float64))
                    else:
                        rows.append(e2qQh / (4 * I_q * (2 * I_q - 1))
                                    * torch.stack([-1 + eta, -1 - eta, 2 * torch.ones_like(eta)]))
                Q = torch.stack(rows)
            # After the shorthand conversion EasySpin requires (nNuclei, 3)
            # principal values or (3*nNuclei, 3) full tensors (validatespinsys.m:
            # 645-667, 'Spin system field Q has wrong size for the given spins.')
            if n_n > 0 and not (Q.ndim == 2 and tuple(Q.shape) in ((n_n, 3), (3 * n_n, 3))):
                raise ValueError(
                    f'Spin system field Q has wrong size {tuple(self.Q.shape)} for {n_n} nucleus/nuclei: '
                    f'expected one e2qQ/h per nucleus, [e2qQ/h, eta] per nucleus (nNuclei, 2), '
                    f'principal values (nNuclei, 3) or full tensors (3*nNuclei, 3).')
            self.Q = Q

        # --- QFrame ---
        if self.QFrame is None:
            self.QFrame = torch.zeros(max(n_n, 1), 3, dtype=torch.float64)
        else:
            self.QFrame = _per_nucleus_rows(_to_tensor(self.QFrame, torch.float64), max(n_n, 1), False)
            # EasySpin: exactly (nNuclei, 3) (validatespinsys.m:686-690)
            if n_n > 0 and tuple(self.QFrame.shape) != (n_n, 3):
                raise ValueError(
                    f'Spin system field QFrame has wrong size {tuple(self.QFrame.shape)} for the given '
                    f'spins: expected ({n_n}, 3) Euler angles, one row per nucleus.')

        # --- nn tensor (nuclear-nuclear coupling) ---
        from itertools import combinations as _comb
        n_nuc_pairs = sum(1 for _ in _comb(range(n_n), 2))
        if self.nn is not None:
            self.nn = _to_tensor(self.nn, torch.float64)
            if n_nuc_pairs > 0:
                if self.nn.ndim == 0:
                    self.nn = self.nn.expand(n_nuc_pairs, 3).clone()
                elif self.nn.ndim == 1:
                    if self.nn.shape[0] == n_nuc_pairs and n_nuc_pairs != 3:
                        # one isotropic coupling per pair (EasySpin column form)
                        self.nn = self.nn.unsqueeze(1).expand(n_nuc_pairs, 3).clone()
                    elif self.nn.shape[0] == 3:
                        self.nn = self.nn.unsqueeze(0).expand(n_nuc_pairs, 3).clone()
                elif self.nn.ndim == 2 and self.nn.shape == (n_nuc_pairs, 1):
                    self.nn = self.nn.expand(n_nuc_pairs, 3).clone()

        # --- nnFrame ---
        if self.nnFrame is None:
            self.nnFrame = torch.zeros(max(n_nuc_pairs, 1), 3, dtype=torch.float64)
        else:
            self.nnFrame = _per_nucleus_rows(_to_tensor(self.nnFrame, torch.float64), max(n_nuc_pairs, 1), False)

        # --- lw and lwpp ---
        # Handle linewidth: lw (FWHM) and lwpp (peak-to-peak) are mutually exclusive
        # Keep both fields to track which one the user specified (important for fitting)
        # Conversion (get_lw): Gaussian lw = lwpp * sqrt(2 ln 2), Lorentzian lw = lwpp * sqrt(3)
        if self.lw is not None and self.lwpp is not None:
            raise ValueError("Specify either lw or lwpp, not both.")
        
        # Normalize to [Gaussian, Lorentzian] format
        if self.lwpp is not None:
            self.lwpp = _normalize_lw(self.lwpp, 'lwpp')
        if self.lw is not None:
            self.lw = _normalize_lw(self.lw, 'lw')
        
        # Set default if neither specified
        if self.lw is None and self.lwpp is None:
            self.lw = [0.0, 0.0]  # Default: no broadening (EasySpin Sys.lw = 0)

        # --- Strain parameters ---
        # HStrain: [3] MHz in molecular frame
        if self.HStrain is not None:
            self.HStrain = _three_widths(self.HStrain, 'HStrain')
        if self.HStrainL is not None:
            self.HStrainL = _to_tensor(self.HStrainL, torch.float64)
            if self.HStrainL.ndim == 0:
                self.HStrainL = self.HStrainL.expand(3).clone()
            elif self.HStrainL.shape[0] != 3 or self.HStrainL.ndim != 1:
                raise ValueError("HStrainL must have shape [3].")
            if bool(torch.any(self.HStrainL.detach() < 0)):
                raise ValueError("HStrainL values must be non-negative.")

        # gStrain: [nElectrons, 3] unitless
        # EasySpin (validatespinsys.m:810-827): empty -> no strain; one row per
        # electron spin with 1, 2 or 3 columns, [x] -> [x x x], [x y] -> [x x y].
        # A 1-D vector is a MATLAB 1xk row (one electron spin).  torchspin
        # shorthands kept: a scalar or [x, y, z] for every electron spin.
        if self.gStrain is not None:
            self.gStrain = _to_tensor(self.gStrain, torch.float64)
            if self.gStrain.numel() == 0:
                self.gStrain = None
            elif self.gStrain.ndim == 0:
                # Scalar: isotropic for all electrons
                self.gStrain = self.gStrain.expand(n_e, 3).clone()
            elif self.gStrain.ndim == 1 and self.gStrain.shape[0] == 3:
                # [3] -> expand to all electrons
                self.gStrain = self.gStrain.unsqueeze(0).expand(n_e, 3).clone()
            else:
                if self.gStrain.ndim == 1:
                    self.gStrain = self.gStrain.unsqueeze(0)
                if self.gStrain.ndim != 2 or self.gStrain.shape[0] != n_e:
                    raise ValueError(
                        f'gStrain must have {n_e} rows, one per electron spin (got shape '
                        f'{tuple(self.gStrain.shape)}).')
                cols = self.gStrain.shape[1]
                if cols == 1:
                    self.gStrain = self.gStrain[:, [0, 0, 0]]
                elif cols == 2:
                    self.gStrain = self.gStrain[:, [0, 0, 1]]
                elif cols != 3:
                    raise ValueError(
                        f'gStrain must have 1, 2, or 3 columns (got shape {tuple(self.gStrain.shape)}).')

        # AStrain: [3] MHz (first nucleus only)
        if self.AStrain is not None:
            self.AStrain = _three_widths(self.AStrain, 'AStrain')

        # DStrain: normalized to (nElectrons, 2) = [FWHM_D, FWHM_E] per electron
        # (EasySpin Sys.DStrain: one row per electron spin, 1-3 columns).
        if self.DStrain is not None:
            self.DStrain = _to_tensor(self.DStrain, torch.float64)
            if self.DStrain.numel() == 0:
                # empty: no D strain, as EasySpin (validatespinsys.m:837-839)
                self.DStrain = None
        if self.DStrain is not None:
            if self.DStrain.ndim == 0:
                self.DStrain = self.DStrain.reshape(1, 1)
            elif self.DStrain.ndim == 1:
                if n_e > 1 and self.DStrain.shape[0] == n_e and n_e > 3:
                    self.DStrain = self.DStrain.unsqueeze(1)          # one FWHM_D per electron
                else:
                    self.DStrain = self.DStrain.unsqueeze(0)          # one row, all electrons
            if self.DStrain.shape[1] > 3 or self.DStrain.shape[1] < 1:
                raise ValueError("DStrain must have 1, 2 or 3 columns ([FWHM_D, FWHM_E, ...]).")
            if self.DStrain.shape[0] == 1 and n_e > 1:
                self.DStrain = self.DStrain.expand(n_e, self.DStrain.shape[1]).clone()
            if self.DStrain.shape[0] != n_e:
                raise ValueError(f"DStrain must have {n_e} rows, one per electron spin.")
            if self.DStrain.shape[1] == 1:
                self.DStrain = torch.cat([self.DStrain, torch.zeros(n_e, 1, dtype=torch.float64)], dim=1)
            else:
                self.DStrain = self.DStrain[:, :2].clone()
        # DStrainCorr: one D/E correlation coefficient per electron
        if isinstance(self.DStrainCorr, torch.Tensor):
            self.DStrainCorr = self.DStrainCorr.flatten().tolist()
        if isinstance(self.DStrainCorr, (int, float)):
            self.DStrainCorr = [float(self.DStrainCorr)] * n_e
        else:
            self.DStrainCorr = [float(v) for v in self.DStrainCorr]
            if len(self.DStrainCorr) == 1 and n_e > 1:
                self.DStrainCorr = self.DStrainCorr * n_e
            if len(self.DStrainCorr) != n_e:
                raise ValueError(f"DStrainCorr must have {n_e} entries, one per electron spin.")

        # --- H0Extra (generic extra field-independent operators) ---
        if self.H0Extra is not None:
            self.H0Extra = _normalize_h0extra(self)

        # From here on, assigning a field re-runs the normalization (__setattr__).
        object.__setattr__(self, '_normalized', True)

    def __setattr__(self, name: str, value) -> None:
        """Assigning a field normalizes it as the constructor does.

        ``sys.HStrain = [1, 2, 3]`` after construction rebuilds the system with
        the new value (``SpinSystem(**{**fields, name: value})``) and adopts the
        result, so the field holds the same tensor it would have if it had been
        given to the constructor, and a value the constructor rejects raises
        here.  Tensors are not copied, so an assigned tensor that requires grad
        stays on the autograd graph.  Assigning one of an alternative pair
        (``D``/``D_``, ``A``/``A_``, ``lw``/``lwpp``, ``tcorr``/``logtcorr``,
        ``Diff``/``logDiff``) clears the other.  During construction, and for
        attributes that are not constructor fields, this is plain assignment.
        To bypass the normalization use ``object.__setattr__(sys, name, value)``.
        """
        if name not in _INIT_FIELDS or not self.__dict__.get('_normalized', False):
            object.__setattr__(self, name, value)
            return
        kw = _init_kwargs(self)
        for other in _ALTERNATIVE_FIELDS.get(name, ()):
            kw[other] = None
        kw[name] = value
        rebuilt = type(self)(**kw)
        self.__dict__.update(rebuilt.__dict__)

    # ------------------------------------------------------------------
    # Read-only derived properties
    # ------------------------------------------------------------------

    @property
    def nElectrons(self) -> int:
        """Number of electron spins, ``len(S)``."""
        return len(self.S)

    @property
    def nNuclei(self) -> int:
        """Number of nuclei, ``len(Nucs)`` (an element or mixture counts once)."""
        return len(self.Nucs)

    @property
    def I(self) -> list[float]:
        """Nuclear spin quantum numbers (derived from Nucs)."""
        return [_nuclear_spin(iso) for iso in self.Nucs]

    @property
    def gn(self) -> list[float]:
        """Nuclear g-factors (derived from Nucs)."""
        return [_nuclear_gn(iso) for iso in self.Nucs]

    @property
    def nL(self) -> int:
        """Number of orbital angular momenta."""
        return len(self.L) if self.L is not None else 0

    @property
    def Spins(self) -> list[float]:
        """All spin quantum numbers: electrons first, then nuclei, then OAMs."""
        oam = [float(l) for l in self.L] if self.L is not None else []
        return self.S + self.I + oam

    @property
    def nStates(self) -> int:
        """Total Hilbert space dimension."""
        return math.prod(int(round(2 * s + 1)) for s in self.Spins)

    @property
    def fullg(self) -> bool:
        """True if g is given as full 3×3 matrices."""
        if self.g.ndim < 2:
            return False
        return self.g.shape[0] == 3 * self.nElectrons

    @property
    def fullD(self) -> bool:
        """True if D is given as full 3×3 matrices."""
        if self.D is None:
            return False
        if self.D.ndim < 2:
            return False
        return self.D.shape[0] == 3 * self.nElectrons

    @property
    def fullA(self) -> bool:
        """True if A is given as full 3×3 matrices per nucleus per electron."""
        if self.A is None or self.nNuclei == 0:
            return False
        return self.A.shape[0] == 3 * self.nNuclei

    @property
    def fullee(self) -> bool:
        """True if ee is given as full 3×3 matrices."""
        if self.ee is None:
            return False
        from itertools import combinations
        n_pairs = sum(1 for _ in combinations(range(self.nElectrons), 2))
        return n_pairs > 0 and self.ee.shape[0] == 3 * n_pairs

    @property
    def fullsigma(self) -> bool:
        """True if sigma is given as full 3×3 matrices."""
        return self.sigma.shape[0] == 3 * max(self.nNuclei, 1)

    @property
    def fullQ(self) -> bool:
        """True if Q is given as full 3×3 matrices."""
        if self.Q is None:
            return False
        return self.Q.shape[0] == 3 * self.nNuclei

    @property
    def fullnn(self) -> bool:
        """True if nn is given as full 3×3 matrices."""
        if self.nn is None:
            return False
        from itertools import combinations
        n_pairs = sum(1 for _ in combinations(range(self.nNuclei), 2))
        return n_pairs > 0 and self.nn.shape[0] == 3 * n_pairs

    def get_lw(self) -> list[float]:
        """Return the line width as ``[GaussianFWHM, LorentzianFWHM]``.

        ``lw`` as stored if it is set; otherwise ``lwpp`` converted from
        peak-to-peak to FWHM (as EasySpin's validatespinsys.m):

        * Gaussian: ``lw = lwpp * sqrt(2 ln 2)``, about ``1.177 * lwpp``;
        * Lorentzian: ``lw = lwpp * sqrt(3)``, about ``1.732 * lwpp``.

        Returns
        -------
        list
            ``[GaussianFWHM, LorentzianFWHM]`` in the units of ``lw``/``lwpp``
            (mT for field sweeps, MHz for frequency sweeps).  The entries are
            floats or 0-d tensors (tensors keep their autograd graph).

        Examples
        --------
        >>> from torchspin import SpinSystem
        >>> [round(v, 3) for v in SpinSystem(S=0.5, lwpp=[1.0, 1.0]).get_lw()]
        [1.177, 1.732]
        """
        import math
        
        if self.lw is not None:
            return self.lw
        elif self.lwpp is not None:
            # Convert lwpp to lw using correct factors for each lineshape
            # Gaussian: sqrt(2*log(2)) for derivative peak-to-peak to FWHM
            # Lorentzian: sqrt(3) for derivative peak-to-peak to FWHM
            gauss_factor = math.sqrt(2.0 * math.log(2.0))  # ≈ 1.177
            lorentz_factor = math.sqrt(3.0)  # ≈ 1.732
            return [self.lwpp[0] * gauss_factor, self.lwpp[1] * lorentz_factor]
        else:
            # Should not happen due to __post_init__, but handle gracefully
            return [0.0, 0.0]

    def validate(self) -> None:
        """Check the spin system for physical consistency.

        Checks the quantum numbers (S positive multiples of 1/2, L
        non-negative), tensor and frame shapes, S >= 1 for a nonzero ``D``,
        I >= 1 for a nonzero ``Q``, known isotopes, non-negative line widths
        and strains, ``DStrainCorr`` in ``[-1, 1]``, ``gAStrainCorr`` = +-1,
        positive ``tcorr``/``Diff``/``T1``/``T2``, exclusive pairs
        (``tcorr``/``logtcorr``, ``Diff``/``logDiff``, nonzero ``D`` and
        ``B[0]``), symmetric full ``Q`` blocks and crystal-field ranks
        ``k <= 2L``.  Partial port of EasySpin's ``validatespinsys.m``.

        The constructor checks only shapes; the simulators do not call this
        method, so call it explicitly to catch physically invalid input.

        Raises
        ------
        ValueError
            On the first problem found.

        Examples
        --------
        >>> from torchspin import SpinSystem
        >>> SpinSystem(S=1.0, D=[300.0, 30.0]).validate()
        >>> SpinSystem(S=0.5, D=300.0).validate()
        Traceback (most recent call last):
            ...
        ValueError: D tensor is nonzero for electron 0 with S=0.5, but ZFS requires S >= 1.
        """
        from itertools import combinations

        n_e = self.nElectrons
        n_n = self.nNuclei
        n_pairs = sum(1 for _ in combinations(range(n_e), 2))
        n_nuc_pairs = sum(1 for _ in combinations(range(n_n), 2))

        # --- Electron spins ---
        if len(self.S) == 0:
            raise ValueError("At least one electron spin is required (S list is empty).")
        for s in self.S:
            if s <= 0:
                raise ValueError(f"Electron spin must be positive, got S={s}.")
            if (2 * s) % 1 != 0:
                raise ValueError(
                    f"Electron spin must be a positive multiple of 1/2, got S={s}."
                )

        # --- g tensor ---
        if self.g.shape[0] not in (n_e, 3 * n_e):
            raise ValueError(
                f"g tensor has {self.g.shape[0]} rows, expected {n_e} or {3*n_e} "
                f"for {n_e} electron(s)."
            )
        if self.g.ndim == 2 and self.g.shape[1] != 3:
            raise ValueError(
                f"g tensor has {self.g.shape[1]} columns, expected 3."
            )

        # --- gFrame ---
        if self.gFrame.shape != (n_e, 3):
            raise ValueError(
                f"gFrame shape {tuple(self.gFrame.shape)} does not match "
                f"expected ({n_e}, 3)."
            )

        # --- D tensor ---
        if self.D is not None:
            if self.D.shape[0] not in (n_e, 3 * n_e):
                raise ValueError(
                    f"D tensor has {self.D.shape[0]} rows, expected {n_e} or "
                    f"{3*n_e} for {n_e} electron(s)."
                )
            # ZFS requires S >= 1
            for i, s in enumerate(self.S):
                if s < 1:
                    # Check if D is actually nonzero for this electron
                    if not self.fullD:
                        row = self.D[i]
                        if torch.any(row != 0):
                            raise ValueError(
                                f"D tensor is nonzero for electron {i} with S={s}, "
                                f"but ZFS requires S >= 1."
                            )

        # --- DFrame ---
        if self.DFrame.shape != (n_e, 3):
            raise ValueError(
                f"DFrame shape {tuple(self.DFrame.shape)} does not match "
                f"expected ({n_e}, 3)."
            )

        # --- A tensor ---
        if n_n > 0 and self.A is not None:
            expected_cols = 3 * n_e
            if self.A.shape[0] not in (n_n, 3 * n_n):
                raise ValueError(
                    f"A tensor has {self.A.shape[0]} rows, expected {n_n} or "
                    f"{3*n_n} for {n_n} nucleus/nuclei."
                )
            if self.A.ndim == 2 and self.A.shape[1] != expected_cols:
                raise ValueError(
                    f"A tensor has {self.A.shape[1]} columns, expected "
                    f"{expected_cols} (3 × {n_e} electrons)."
                )

        # --- AFrame ---
        if n_n > 0 and self.AFrame is not None:
            if self.AFrame.shape[0] not in (n_n, 3 * n_n):
                raise ValueError(
                    f"AFrame has {self.AFrame.shape[0]} rows, expected {n_n}."
                )

        # --- Nuclei ---
        for nuc in self.Nucs:
            if is_isotope_mixture(nuc):
                # 'Cu' / '(63,65)Cu': expanded by torchspin.isotopologues before simulation
                continue
            try:
                nucdata(nuc)
            except (ValueError, KeyError):
                raise ValueError(
                    f"Unknown nucleus '{nuc}'. Use isotope notation like '14N', '1H'."
                )
        if n_n > 0 and self.A is None:
            raise ValueError(
                "Nuclei are specified (Nucs) but no hyperfine coupling (A) is given."
            )

        # --- Q tensor ---
        if self.Q is not None:
            if self.Q.shape[0] not in (n_n, 3 * n_n):
                raise ValueError(
                    f"Q tensor has {self.Q.shape[0]} rows, expected {n_n} or "
                    f"{3*n_n} for {n_n} nucleus/nuclei."
                )
            # Quadrupole requires I >= 1
            for i, nuc in enumerate(self.Nucs):
                if is_isotope_mixture(nuc):
                    continue
                I_val = _nuclear_spin(nuc)
                if I_val < 1 and not self.fullQ:
                    row = self.Q[i]
                    if torch.any(row != 0):
                        raise ValueError(
                            f"Q tensor is nonzero for nucleus '{nuc}' (I={I_val}), "
                            f"but quadrupole interaction requires I >= 1."
                        )

        # --- ee coupling ---
        if n_e > 1:
            if self.ee is not None:
                if self.ee.shape[0] not in (n_pairs, 3 * n_pairs):
                    raise ValueError(
                        f"ee tensor has {self.ee.shape[0]} rows, expected {n_pairs} "
                        f"or {3*n_pairs} for {n_pairs} electron pair(s)."
                    )

        # --- nn coupling ---
        if self.nn is not None:
            if self.nn.shape[0] not in (n_nuc_pairs, 3 * n_nuc_pairs):
                raise ValueError(
                    f"nn tensor has {self.nn.shape[0]} rows, expected {n_nuc_pairs} "
                    f"or {3*n_nuc_pairs} for {n_nuc_pairs} nuclear pair(s)."
                )

        # --- Strain parameters ---
        if self.HStrain is not None and self.HStrain.shape[0] != 3:
            raise ValueError("HStrain must have 3 elements.")
        if self.gStrain is not None:
            if self.gStrain.shape != (n_e, 3):
                raise ValueError(
                    f"gStrain shape {tuple(self.gStrain.shape)} does not match "
                    f"expected ({n_e}, 3)."
                )
        if self.AStrain is not None and self.AStrain.shape[0] != 3:
            raise ValueError("AStrain must have 3 elements.")
        if self.DStrain is not None and tuple(self.DStrain.shape) != (self.nElectrons, 2):
            raise ValueError("DStrain must have shape (nElectrons, 2) = [FWHM_D, FWHM_E] per electron.")

        # --- Linewidth ---
        if self.lw is not None and self.lwpp is not None:
            raise ValueError("Specify either lw or lwpp, not both.")
        if self.lw is not None:
            if any(v < 0 for v in self.lw):
                raise ValueError("Linewidth (lw) values must be non-negative.")
        if self.lwpp is not None:
            if any(v < 0 for v in self.lwpp):
                raise ValueError("Linewidth (lwpp) values must be non-negative.")

        # --- OAM ---
        if self.L is not None:
            for l_val in self.L:
                if l_val < 0 or (2 * l_val) % 1 != 0:
                    raise ValueError(
                        f"OAM quantum number must be a non-negative multiple of "
                        f"1/2, got L={l_val}."
                    )
            if self.gL is not None and len(self.gL) != len(self.L):
                raise ValueError(
                    f"gL has {len(self.gL)} elements but L has {len(self.L)}."
                )

        # --- Dynamics ---
        if self.tcorr is not None and self.tcorr <= 0:
            raise ValueError("tcorr must be positive.")
        if self.Diff is not None and self.Diff <= 0:
            raise ValueError("Diff must be positive.")
        if self.T1 is not None and self.T1 <= 0:
            raise ValueError("T1 must be positive.")
        if self.T2 is not None and self.T2 <= 0:
            raise ValueError("T2 must be positive.")

        # --- Mutual exclusivity: dynamics parameters ---
        if self.tcorr is not None and self.logtcorr is not None:
            raise ValueError(
                "Specify either tcorr or logtcorr, not both."
            )
        if self.Diff is not None and self.logDiff is not None:
            raise ValueError(
                "Specify either Diff or logDiff, not both."
            )

        # --- Mutual exclusivity: D vs B2 (Stevens rank-2) ---
        if self.D is not None and self.B is not None:
            has_D = torch.any(self.D != 0).item()
            has_B2 = (len(self.B) > 0 and self.B[0] is not None
                      and torch.any(self.B[0] != 0).item())
            if has_D and has_B2:
                raise ValueError(
                    "Cannot use both D and B2 (Stevens rank-2 coefficients) "
                    "simultaneously. Use one or the other."
                )

        # --- Strain correlation bounds ---
        if any(r < -1 or r > 1 for r in (self.DStrainCorr if isinstance(self.DStrainCorr, (list, tuple)) else [self.DStrainCorr])):
            raise ValueError(
                f"DStrainCorr must be between -1 and +1, got {self.DStrainCorr}."
            )
        if self.gAStrainCorr not in (1.0, -1.0):
            raise ValueError(
                f"gAStrainCorr must be +1 (correlated) or -1 (anticorrelated), "
                f"got {self.gAStrainCorr}."
            )

        # --- Strain non-negativity ---
        if self.HStrain is not None and torch.any(self.HStrain < 0):
            raise ValueError("HStrain values must be non-negative.")
        if self.HStrainL is not None and torch.any(self.HStrainL < 0):
            raise ValueError("HStrainL values must be non-negative.")
        if self.gStrain is not None and torch.any(self.gStrain < 0):
            raise ValueError("gStrain values must be non-negative.")
        if self.AStrain is not None and torch.any(self.AStrain < 0):
            raise ValueError("AStrain values must be non-negative.")
        if self.DStrain is not None and torch.any(self.DStrain < 0):
            raise ValueError("DStrain values must be non-negative.")

        # --- Q matrix symmetry (when full 3x3 blocks) ---
        if self.Q is not None and self.fullQ:
            for i in range(n_n):
                Q_block = self.Q[3 * i:3 * i + 3, :]
                if not torch.allclose(Q_block, Q_block.T, atol=1e-10):
                    raise ValueError(
                        f"Q tensor for nucleus {i} ('{self.Nucs[i]}') is not "
                        f"symmetric. Full Q matrices must be symmetric."
                    )

        # --- Crystal field coherence with L ---
        for k in range(1, 13):
            cf_field = getattr(self, f'CF{k}', None)
            if cf_field is not None:
                if self.L is None:
                    raise ValueError(
                        f"CF{k} requires orbital angular momentum L to be specified."
                    )
                for j, l_val in enumerate(self.L):
                    if k > 2 * l_val:
                        raise ValueError(
                            f"CF{k} rank {k} exceeds 2*L={2*l_val} for "
                            f"orbital {j}."
                        )


# ---------------------------------------------------------------------------
# Nuclear spin manipulation functions
# ---------------------------------------------------------------------------

def _nucleus_indices(sys: 'SpinSystem', idx) -> list[int]:
    """Normalize nucleus index/indices to a sorted list of 0-based ints."""
    n = sys.nNuclei
    if isinstance(idx, (int, np.integer)):
        idx = [int(idx)]
    else:
        idx = [int(i) for i in idx]
    for i in idx:
        if i < 0 or i >= n:
            raise ValueError(f"Nucleus index {i} out of range (0-based, nNuclei={n}).")
    return sorted(set(idx))


def nucspinkeep(sys: 'SpinSystem', idx) -> 'SpinSystem':
    """Return a new SpinSystem with only the specified nuclei kept.

    Port of EasySpin's ``nucspinkeep``: every field of *sys* is carried over,
    and the per-nucleus fields are reduced to the kept nuclei (see
    :func:`nucspinrmv`).

    Parameters
    ----------
    sys : SpinSystem
        Input spin system.
    idx : int or sequence of int
        0-based nucleus index/indices to keep.

    Returns
    -------
    SpinSystem
        New spin system with only the specified nuclei.

    Examples
    --------
    >>> sys = SpinSystem(S=0.5, Nucs=['14N', '1H', '1H'], A=[40.0, 10.0, 5.0])
    >>> sys2 = nucspinkeep(sys, [0, 2])   # keep nuclei 0 and 2
    >>> sys2.Nucs
    ['14N', '1H']
    """
    _reject_h0extra(sys, 'nucspinkeep')
    keep = _nucleus_indices(sys, idx)
    return _filter_nuclei(sys, keep, 'nucspinkeep')


def nucspinrmv(sys: 'SpinSystem', idx) -> 'SpinSystem':
    """Return a new SpinSystem with the specified nuclei removed.

    Port of EasySpin's ``nucspinrmv`` (``NewSys = Sys``, then the per-nucleus
    fields are filtered): every field of *sys* (electron interactions such as
    ``ee``, ``ee2``, ``Ham``, ``L``/``soc``/``CF*``, linewidths, strains,
    dynamics, ...) is carried over unchanged, tensors without copying, so the
    result stays on the autograd graph of *sys*.  The per-nucleus fields
    ``Nucs``, ``n``, ``A``/``A_``, ``AFrame``, ``gnscale``, ``sigma``,
    ``sigmaFrame``, ``Q``, ``QFrame`` and ``Abund`` are reduced to the
    remaining nuclei (full 3x3 tensor forms included).

    Raises ``ValueError`` if nonzero nucleus-nucleus couplings ``nn`` would
    remain (as EasySpin), or if ``initState`` is given in the full spin space
    and the nuclear dimension changes.  ``AStrain`` belongs to the first
    nucleus: it is dropped, with a ``UserWarning``, when that nucleus is
    removed.

    Parameters
    ----------
    sys : SpinSystem
        Input spin system.
    idx : int or sequence of int
        0-based nucleus index/indices to remove.

    Returns
    -------
    SpinSystem
        New spin system without the specified nuclei.

    Examples
    --------
    >>> sys = SpinSystem(S=0.5, Nucs=['14N', '1H'], A=[40.0, 10.0])
    >>> sys2 = nucspinrmv(sys, 1)   # remove nucleus 1
    >>> sys2.Nucs
    ['14N']
    """
    _reject_h0extra(sys, 'nucspinrmv')
    rmv = _nucleus_indices(sys, idx)
    keep = [i for i in range(sys.nNuclei) if i not in rmv]
    return _filter_nuclei(sys, keep, 'nucspinrmv')


def nucspinadd(sys: 'SpinSystem', nuc: str, A=None, Q=None,
               AFrame=None, QFrame=None, gnscale=1.0) -> 'SpinSystem':
    """Return a new SpinSystem with one nucleus appended.

    Port of EasySpin's ``nucspinadd``: every field of *sys* is carried over
    and the per-nucleus fields get one more entry.  If either the existing
    or the new hyperfine (quadrupole) tensor is a full 3x3 matrix, both are
    written as full matrices in the molecular frame, as EasySpin does.
    Unlike EasySpin, systems with several electron spins are accepted (a
    3-element ``A`` is used for every electron).

    Parameters
    ----------
    sys : SpinSystem
        Input spin system.
    nuc : str
        Isotope string, e.g. ``'1H'`` or ``'14N'``.
    A : array_like or None
        Hyperfine coupling (MHz): a scalar (isotropic), ``[Aperp, Apar]``,
        ``[Ax, Ay, Az]`` (used for all electrons), ``(3*nElectrons,)``
        principal values per electron, or a full ``(3, 3*nElectrons)`` matrix.
        ``None`` → zero.  A tensor that requires grad stays on the graph.
    Q : array_like or None
        Nuclear quadrupole coupling (MHz): ``e2qQ/h``, ``[e2qQ/h, eta]``,
        principal values ``[Qx, Qy, Qz]`` or a full 3x3 matrix.  Only for I ≥ 1.
    AFrame : array_like or None
        Euler angles for A-tensor frame.  Shape ``(3,)`` or ``(3*nElectrons,)``.
    QFrame : array_like or None
        Euler angles for Q-tensor frame.  Shape ``(3,)``.
    gnscale : float
        Nuclear g-factor scale factor (dimensionless).  Default 1.0.

    Returns
    -------
    SpinSystem
        New spin system with the nucleus appended.

    Examples
    --------
    >>> sys = SpinSystem(S=0.5, Nucs=['14N'], A=[20.0, 20.0, 95.0])
    >>> sys2 = nucspinadd(sys, '1H', A=[5.0, 5.0, 10.0])
    >>> sys2.Nucs
    ['14N', '1H']
    """
    _reject_h0extra(sys, 'nucspinadd')
    if _nonzero(sys.nn):
        raise ValueError("nucspinadd does not work if Sys.nn is given (nonzero "
                         "nucleus-nucleus couplings), as in EasySpin.")
    f64 = torch.float64
    n_e = sys.nElectrons
    n_n = sys.nNuclei
    kw = _init_kwargs(sys)
    kw['Nucs'] = list(sys.Nucs) + [nuc]
    kw['n'] = list(sys.n) + [1]

    # --- hyperfine ---
    A_new, A_new_full = _new_hyperfine(A, n_e)
    AF_new = _new_frame(AFrame, 3 * n_e, 'AFrame')
    if n_n == 0 or sys.A is None:
        new_A, new_AF = A_new, AF_new
    else:
        A0, AF0 = sys.A, sys.AFrame             # sys.A is derived from A_ if that was given
        if sys.fullA or A_new_full:
            if not sys.fullA:
                A0, AF0 = _full_from_principal(A0, AF0, n_e), torch.zeros_like(AF0)
            if not A_new_full:
                A_new, AF_new = _full_from_principal(A_new, AF_new, n_e), torch.zeros_like(AF_new)
        new_A = torch.cat([A0, A_new.to(A0.dtype)], dim=0)
        new_AF = torch.cat([AF0, AF_new], dim=0)
    kw.update(A=new_A, A_=None, AFrame=new_AF)

    # --- quadrupole ---
    QF0 = sys.QFrame
    QF_new = _new_frame(QFrame, 3, 'QFrame')
    if Q is not None or sys.Q is not None:
        Q_new, Q_new_full = _new_quadrupole(Q, nuc)
        if n_n == 0:
            new_Q = Q_new
        else:
            Q0 = sys.Q if sys.Q is not None else torch.zeros(n_n, 3, dtype=f64)
            full0 = sys.Q is not None and sys.fullQ
            if full0 or Q_new_full:
                if not full0:
                    Q0, QF0 = _full_from_principal(Q0, QF0, 1), torch.zeros_like(QF0)
                if not Q_new_full:
                    Q_new, QF_new = _full_from_principal(Q_new, QF_new, 1), torch.zeros_like(QF_new)
            new_Q = torch.cat([Q0, Q_new.to(Q0.dtype)], dim=0)
        kw['Q'] = new_Q
    kw['QFrame'] = QF_new if n_n == 0 else torch.cat([QF0, QF_new], dim=0)

    # --- nuclear Zeeman scaling, chemical shielding ---
    gs_new = _to_tensor(gnscale, f64).reshape(1)
    kw['gnscale'] = gs_new if n_n == 0 else torch.cat([sys.gnscale, gs_new])
    if n_n > 0:
        sig_new = (torch.eye(3, dtype=f64) if sys.fullsigma
                   else torch.ones(1, 3, dtype=f64))
        kw['sigma'] = torch.cat([sys.sigma, sig_new], dim=0)
        kw['sigmaFrame'] = torch.cat([sys.sigmaFrame, torch.zeros(1, 3, dtype=f64)], dim=0)

    kw['nn'] = kw['nnFrame'] = None
    if sys.Abund is not None:
        kw['Abund'] = _abund_per_nucleus(sys.Abund, n_n) + [None]
    _check_init_state(sys, list(sys.Nucs) + [nuc], 'nucspinadd')
    return SpinSystem(**kw)


# ---------------------------------------------------------------------------
# Rebuilding a SpinSystem, per-nucleus field helpers
# ---------------------------------------------------------------------------

_INIT_FIELDS = frozenset(f.name for f in dataclasses.fields(SpinSystem) if f.init)

# Fields that give the same quantity in two forms: assigning one clears the other.
_ALTERNATIVE_FIELDS = {
    'D': ('D_',), 'D_': ('D',),
    'A': ('A_',), 'A_': ('A',),
    'lw': ('lwpp',), 'lwpp': ('lw',),
    'tcorr': ('logtcorr',), 'logtcorr': ('tcorr',),
    'Diff': ('logDiff',), 'logDiff': ('Diff',),
}


def _init_kwargs(sys: 'SpinSystem') -> dict:
    """Constructor keyword arguments that rebuild *sys*.

    ``SpinSystem(**_init_kwargs(sys))`` reproduces every field of *sys*: all
    dataclass init fields are passed as stored.  Tensors are passed as they
    are (no copy, no detach), so the rebuilt system stays on the autograd
    graph of *sys*; lists and dicts are copied one level so that the new
    ``__post_init__`` (which normalizes some of them in place, e.g. ``B``)
    cannot modify *sys*.

    Fields that ``__post_init__`` derives from an alternative input and keeps
    next to it are passed through their source only, since giving both raises:
    ``D`` is rebuilt from ``D_`` and ``A`` from ``A_`` when those are set.
    (``lw``/``lwpp`` never both hold a value; ``tcorr``/``logtcorr`` and
    ``Diff``/``logDiff`` are not converted and are passed as they are.)
    """
    kw = {}
    for f in dataclasses.fields(sys):
        if not f.init:
            continue
        v = getattr(sys, f.name)
        if isinstance(v, list):
            v = list(v)
        elif isinstance(v, dict):
            v = dict(v)
        kw[f.name] = v
    if kw.get('D_') is not None:
        kw['D'] = None
    if kw.get('A_') is not None and sys.nNuclei > 0:
        kw['A'] = None
    return kw


def _replace(sys: 'SpinSystem', **changes) -> 'SpinSystem':
    """``dataclasses.replace`` for a :class:`SpinSystem` that accepts ``D_``/``A_``.

    ``dataclasses.replace`` passes every stored field, so a system built with
    ``D_`` (or ``A_``) also passes the ``D`` (``A``) that ``__post_init__``
    derived from it, and the constructor raises ("given in both D and D_").
    This rebuilds from :func:`_init_kwargs` instead, which passes such pairs
    through their source only, and a changed field clears its alternative
    (``_ALTERNATIVE_FIELDS``), as attribute assignment does.  Tensors are not
    copied, so the new system stays on the autograd graph of *sys*.
    """
    kw = _init_kwargs(sys)
    for name in changes:
        for other in _ALTERNATIVE_FIELDS.get(name, ()):
            kw[other] = None
    kw.update(changes)
    return type(sys)(**kw)


def _nonzero(T) -> bool:
    return T is not None and bool(torch.any(torch.as_tensor(T) != 0))


def _nuc_rows(T: torch.Tensor, keep: list[int], n_n: int, name: str) -> torch.Tensor:
    """Rows of a per-nucleus field for the nuclei *keep*: one row per nucleus,
    or a 3-row block per nucleus (full 3x3 tensor form).  Indexing keeps the
    autograd graph."""
    if T.shape[0] == n_n:
        idx = list(keep)
    elif T.shape[0] == 3 * n_n:
        idx = [3 * i + k for i in keep for k in range(3)]
    else:
        raise ValueError(f"SpinSystem.{name} has {T.shape[0]} rows; expected {n_n} "
                         f"(one per nucleus) or {3 * n_n} (full 3x3 tensors).")
    return T[idx]


def _abund_per_nucleus(abund, n_n: int) -> list:
    """``Sys.Abund`` as one entry per nucleus (isotopologues.py conventions)."""
    if n_n == 1 and not (isinstance(abund, (list, tuple)) and len(abund) > 0
                         and isinstance(abund[0], (list, tuple, np.ndarray, torch.Tensor))):
        return [abund]
    if len(abund) != n_n:
        raise ValueError('Abund must have one entry (array or None) per nucleus.')
    return list(abund)


def _check_init_state(sys: 'SpinSystem', new_nucs: list, where: str) -> None:
    """Raise if ``sys.initState`` is given in the full spin space and the
    nuclear dimension changes; electron-space forms (strings, ``'xyz'``,
    states of the electron spins only) are valid for any nuclei."""
    st = getattr(sys, 'initState', None)
    if st is None or isinstance(st, str):
        return
    n_nuc_old = math.prod(int(round(2 * I + 1)) for I in sys.I)
    n_nuc_new = math.prod(int(round(2 * _nuclear_spin(nc) + 1)) for nc in new_nucs)
    if n_nuc_old == n_nuc_new:
        return
    if isinstance(st, (tuple, list)) and len(st) == 2 and isinstance(st[1], str):
        state, basis = st
        if basis == 'xyz':
            return
    else:
        state = st
    shape = tuple(state.shape) if hasattr(state, 'shape') else np.shape(state)
    is_vec = len(shape) <= 1 or 1 in shape
    dim = int(np.prod(shape)) if is_vec else int(shape[0])
    n_es = math.prod(int(round(2 * s + 1)) for s in sys.S)
    if dim == n_es:
        return
    raise ValueError(
        f"{where}: SpinSystem.initState is given in the full spin space ({dim} states), "
        f"which does not fit the new nuclei ({n_es * n_nuc_new} states). Give it for the "
        f"electron spins only ({n_es} states) or set it for the new spin system.")


def _new_hyperfine(A, n_e: int) -> tuple[torch.Tensor, bool]:
    """Hyperfine input of nucspinadd → ((1, 3n_e) principal values, False) or
    ((3, 3n_e) full matrix, True)."""
    f64 = torch.float64
    if A is None:
        return torch.zeros(1, 3 * n_e, dtype=f64), False
    T = _to_tensor(A, f64)
    if T.ndim == 2 and tuple(T.shape) == (3, 3 * n_e):
        return T, True
    T = T.reshape(-1)
    if T.numel() == 1:
        T = T.expand(3)
    elif T.numel() == 2:
        T = T[[0, 0, 1]]
    if T.numel() == 3:
        return T.repeat(n_e).reshape(1, 3 * n_e), False
    if T.numel() == 3 * n_e:
        return T.reshape(1, 3 * n_e), False
    if T.numel() == 9 * n_e:
        return T.reshape(3, 3 * n_e), True
    raise ValueError(f"nucspinadd: A must have 1, 2, 3, {3 * n_e} or {9 * n_e} elements.")


def _new_quadrupole(Q, nuc: str) -> tuple[torch.Tensor, bool]:
    """Quadrupole input of nucspinadd → ((1, 3) principal values, False) or
    ((3, 3) full matrix, True); ``e2qQ/h`` and ``[e2qQ/h, eta]`` converted as
    in ``SpinSystem.__post_init__``."""
    f64 = torch.float64
    if Q is None:
        return torch.zeros(1, 3, dtype=f64), False
    T = _to_tensor(Q, f64)
    if T.numel() == 9:
        return T.reshape(3, 3), True
    T = T.reshape(-1)
    if T.numel() in (1, 2):
        I_q = _quadrupole_spin(nuc)
        if I_q < 1:
            return torch.zeros(1, 3, dtype=f64), False
        e2qQh = T[0]
        eta = T[1] if T.numel() == 2 else torch.zeros_like(e2qQh)
        return (e2qQh / (4 * I_q * (2 * I_q - 1))
                * torch.stack([-1 + eta, -1 - eta, 2 * torch.ones_like(eta)])).reshape(1, 3), False
    if T.numel() == 3:
        return T.reshape(1, 3), False
    raise ValueError("nucspinadd: Q must have 1, 2, 3 or 9 elements.")


def _new_frame(F, width: int, name: str) -> torch.Tensor:
    """Euler angles of the new nucleus as a (1, width) row (zeros if None)."""
    f64 = torch.float64
    if F is None:
        return torch.zeros(1, width, dtype=f64)
    T = _to_tensor(F, f64).reshape(-1)
    if T.numel() == 3 and width > 3:
        T = T.repeat(width // 3)
    if T.numel() != width:
        raise ValueError(f"nucspinadd: {name} must have 3 or {width} elements.")
    return T.reshape(1, width)


def _full_from_principal(T: torch.Tensor, frames: torch.Tensor, n_blocks: int) -> torch.Tensor:
    """Principal values (m, 3*n_blocks) with Euler angles (m, 3*n_blocks) →
    full molecular-frame matrices (3m, 3*n_blocks), rotated as ham_hf/ham_nq do."""
    from torchspin.rotations import erot
    rows = []
    for i in range(T.shape[0]):
        blocks = []
        for b in range(n_blocks):
            M = torch.diag(T[i, 3 * b:3 * b + 3])
            ang = frames[i, 3 * b:3 * b + 3].tolist()
            if any(a != 0.0 for a in ang):
                R_A2M = erot(ang).T.to(M.dtype)
                M = R_A2M @ M @ R_A2M.T
            blocks.append(M)
        rows.append(torch.cat(blocks, dim=1))
    return torch.cat(rows, dim=0)


def _filter_nuclei(sys: 'SpinSystem', keep: list[int], where: str = 'nucspinrmv') -> 'SpinSystem':
    """New SpinSystem with every field of *sys*, the per-nucleus fields reduced
    to the nuclei at indices *keep* (EasySpin nucspinrmv.m)."""
    n_n = sys.nNuclei
    if len(keep) >= 2 and _nonzero(sys.nn):
        # With fewer than two nuclei left there is no pair for nn to couple.
        raise ValueError(f"{where} does not work if Sys.nn is given (nonzero "
                         f"nucleus-nucleus couplings), as in EasySpin.")
    kw = _init_kwargs(sys)
    kw['Nucs'] = [sys.Nucs[i] for i in keep]
    per_nucleus = ('AFrame', 'gnscale', 'sigma', 'sigmaFrame', 'Q', 'QFrame')
    if keep:
        kw['n'] = [sys.n[i] for i in keep]
        if sys.A_ is not None:
            kw['A_'] = _nuc_rows(sys.A_, keep, n_n, 'A_')
        elif sys.A is not None:
            kw['A'] = _nuc_rows(sys.A, keep, n_n, 'A')
        for name in per_nucleus:
            v = getattr(sys, name)
            if v is not None:
                kw[name] = _nuc_rows(v, keep, n_n, name)
    else:
        for name in ('n', 'A', 'A_') + per_nucleus:
            kw[name] = None
    kw['nn'] = kw['nnFrame'] = None
    if sys.Abund is not None:
        ab = [_abund_per_nucleus(sys.Abund, n_n)[i] for i in keep]
        kw['Abund'] = ab if any(a is not None for a in ab) else None
    if sys.AStrain is not None and (not keep or keep[0] != 0):
        warnings.warn(
            f"{where}: SpinSystem.AStrain applies to the first nucleus only, and that "
            f"nucleus ({sys.Nucs[0]}) is removed; AStrain is dropped.",
            UserWarning, stacklevel=3)
        kw['AStrain'] = None
    _check_init_state(sys, [sys.Nucs[i] for i in keep], where)
    return SpinSystem(**kw)


def isotopologues(sys: 'SpinSystem') -> list[tuple[float, 'SpinSystem']]:
    """Generate all isotopologues of a spin system with natural-abundance weights.

    For each nucleus in *sys*, queries ``nucabund`` to find the natural
    abundance of its isotope.  Returns a list of ``(weight, SpinSystem)``
    tuples where *weight* is the product of abundances for all nuclei.

    Parameters
    ----------
    sys : SpinSystem
        Input spin system (typically with specific isotopes, e.g. ``'14N'``).

    Returns
    -------
    list of (float, SpinSystem)
        Each entry is ``(weight, sys_variant)`` where *weight* is the natural
        abundance fraction and *sys_variant* is a new SpinSystem identical to
        *sys* but with *weight* set.  The sum of weights ≤ 1.0.

    Notes
    -----
    Only a single variant is returned per isotope (the specified isotope in
    *sys.Nucs* with its natural abundance).  To generate variants for *all*
    naturally occurring isotopes of each element, use a more specialized
    function.

    Examples
    --------
    >>> sys = SpinSystem(S=0.5, Nucs=['14N', '1H'], A=[40.0, 10.0])
    >>> variants = isotopologues(sys)
    >>> for w, s in variants:
    ...     print(f"weight={w:.4f}, Nucs={s.Nucs}")
    weight=0.9962, Nucs=['14N', '1H']
    """
    if sys.nNuclei == 0:
        return [(1.0, sys)]

    # Get abundance for each nucleus
    abundances = [float(nucabund(nuc)) for nuc in sys.Nucs]

    # Total weight = product of all abundances
    weight = 1.0
    for a in abundances:
        weight *= a

    if weight <= 0:
        return []

    # Return single variant (the exact isotope combination specified)
    import copy as _copy
    new_sys = _copy.copy(sys)
    object.__setattr__(new_sys, 'weight', weight) if hasattr(new_sys, '__dataclass_fields__') else setattr(new_sys, 'weight', weight)
    return [(weight, new_sys)]


# ---------------------------------------------------------------------------
# spinvec
# ---------------------------------------------------------------------------

def spinvec(sys: 'SpinSystem') -> list:
    """Return all spin quantum numbers (electrons + nuclei) as a flat list.

    Port of EasySpin's ``spinvec.m``.

    Parameters
    ----------
    sys : SpinSystem

    Returns
    -------
    list of float
        Spin quantum numbers [S1, S2, ..., I1, I2, ...] in SpinSystem order.

    Examples
    --------
    >>> from torchspin import SpinSystem
    >>> spinvec(SpinSystem(S=[0.5], Nucs=['14N']))
    [0.5, 1.0]
    """
    return list(sys.Spins.tolist()) if hasattr(sys.Spins, 'tolist') else list(sys.Spins)


# ---------------------------------------------------------------------------
# H0Extra
# ---------------------------------------------------------------------------

_H0EXTRA_HERM_TOL = 1e-10


def _normalize_h0extra(sys: 'SpinSystem') -> Optional[list]:
    """Validate and normalize ``sys.H0Extra`` to a list of
    ``(coeff, op)`` = (0-d float64 tensor, (n, n) complex128 tensor) pairs.

    Idempotent (``dataclasses.replace`` re-runs ``__post_init__``) and graph
    preserving: tensors are converted with ``.to``/``.reshape``, never
    detached, so a coefficient or operator that requires grad stays on the
    autograd graph.  An empty list normalizes to ``None``.
    """
    terms = sys.H0Extra
    if isinstance(terms, tuple) and len(terms) == 2 and not isinstance(terms[0], (tuple, list)):
        raise TypeError("SpinSystem.H0Extra must be a list of (coeff, op) pairs, "
                        "e.g. H0Extra=[(coeff, op)].")
    if not isinstance(terms, (list, tuple)):
        raise TypeError("SpinSystem.H0Extra must be a list of (coeff, op) pairs.")
    if len(terms) == 0:
        return None
    mixtures = [nc for nc in sys.Nucs if is_isotope_mixture(nc)]
    if mixtures:
        raise ValueError(
            f"SpinSystem.H0Extra cannot be combined with isotope mixtures in Nucs "
            f"({mixtures}): the isotopologue expansion changes the nuclei, so one "
            f"operator cannot fit every isotopologue. Give explicit isotopes "
            f"(e.g. '63Cu') and build one component per isotopologue.")
    if any(v > 1 for v in (sys.n or [])):
        raise ValueError("SpinSystem.H0Extra cannot be combined with sets of equivalent "
                         "nuclei (SpinSystem.n > 1): the operator is defined on the full "
                         "Hilbert space; list each nucleus separately in Nucs.")
    dim = sys.nStates
    out = []
    for k, term in enumerate(terms):
        if not isinstance(term, (list, tuple)) or len(term) != 2:
            raise TypeError(f"SpinSystem.H0Extra[{k}] must be a (coeff, op) pair.")
        coeff, op = term
        c = coeff if torch.is_tensor(coeff) else torch.as_tensor(np.asarray(coeff))
        if c.is_complex():
            raise ValueError(f"SpinSystem.H0Extra[{k}]: coeff must be real (a Hermitian "
                             f"term needs a real coefficient), got a complex value.")
        if c.numel() != 1:
            raise ValueError(f"SpinSystem.H0Extra[{k}]: coeff must be a scalar, got "
                             f"shape {tuple(c.shape)}.")
        c = c.to(torch.float64).reshape(())
        O = op if torch.is_tensor(op) else torch.as_tensor(np.asarray(op))
        if O.ndim != 2 or tuple(O.shape) != (dim, dim):
            raise ValueError(
                f"SpinSystem.H0Extra[{k}]: op has shape {tuple(O.shape)}, expected "
                f"({dim}, {dim}) for spins {sys.Spins} (full spin-system basis).")
        O = O.to(torch.complex128)
        Od = O.detach()
        scale = max(1.0, float(Od.abs().max()))
        if float((Od - Od.conj().T).abs().max()) > _H0EXTRA_HERM_TOL * scale:
            raise ValueError(f"SpinSystem.H0Extra[{k}]: op is not Hermitian "
                             f"(tolerance {_H0EXTRA_HERM_TOL:g} relative).")
        out.append((c, O))
    return out


def _reject_h0extra(sys, where: str) -> None:
    """Raise if *sys* carries ``H0Extra`` and *where* cannot include it."""
    if getattr(sys, 'H0Extra', None) is not None:
        raise ValueError(
            f"{where} builds the spin Hamiltonian from the interaction tensors "
            f"without torchspin.ham() and cannot include SpinSystem.H0Extra. Use a "
            f"solver that diagonalizes the full Hamiltonian (e.g. pepper with "
            f"Options(Method='matrix')), or remove H0Extra.")


def _ham_linear_orders(sys) -> list:
    """Keys of the nonzero ``Sys.Ham*`` terms of order lB <= 1 in the field.

    These are the terms that ``ham()`` puts into H0 - B·μ and that every
    solver diagonalizing it includes.  Terms of order lB >= 2 enter only with
    ``Options.QuadraticZeeman`` and are dropped by default everywhere.
    """
    ham = getattr(sys, 'Ham', None)
    if ham is None:
        return []
    out = []
    for key, val in ham.items():
        k = str(key)
        k = k[3:] if k.startswith('Ham') else k
        if (val is not None and bool(torch.any(torch.as_tensor(val) != 0))
                and int(k[0]) <= 1):
            out.append(str(key))
    return out


def _reject_ham(sys, where: str, alternative: str) -> None:
    """Raise if *sys* carries ``Sys.Ham*`` terms that *where* (a
    perturbation-theory solver) silently drops.

    These solvers build the electron Zeeman term from ``Sys.g`` only.  With g
    given as ``Ham110``/``Ham112``, ``Sys.g`` is zero, so they returned NaN
    frequencies or an all-zero spectrum; other terms of order lB <= 1 were
    ignored.  EasySpin's resfreqs_perturb rejects every ``Sys.Ham*``
    (resfreqs_perturb.m:75-77).  Terms of order lB >= 2 are dropped by every
    solver unless ``Options.QuadraticZeeman`` is on, which the perturbation
    solvers reject separately (``check_quadratic_zeeman_unsupported``).
    """
    keys = _ham_linear_orders(sys)
    if keys:
        raise ValueError(
            f"{where} does not support Sys.Ham* parameters (here {', '.join(keys)}): "
            f"perturbation theory builds the Zeeman term from Sys.g only. Give the "
            f"linear Zeeman term as Sys.g, or use {alternative}.")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _normalize_A(A: torch.Tensor, n_n: int, n_e: int) -> torch.Tensor:
    """Normalize a hyperfine array to ``(nNuclei, 3*nElectrons)`` principal
    values or ``(3*nNuclei, 3*nElectrons)`` full tensors.

    Follows EasySpin's validatespinsys.m:591-617 in its order, reading a 1-D
    vector as a MATLAB 1xk row: full tensors; ``[1, nNuclei]`` one isotropic
    value per nucleus (one electron spin only); ``[nNuclei, nElectrons]``
    isotropic per electron; ``[nNuclei, 2*nElectrons]`` axial
    ``[A_perp, A_par]`` per electron; ``[nNuclei, 3*nElectrons]`` principal
    values.  torchspin shorthands kept: a scalar (isotropic, every nucleus and
    electron) and, for one electron spin, ``[Ax, Ay, Az]`` for every nucleus.
    Any other shape raises ``ValueError`` (EasySpin: 'Size of Sys.A ... is
    inconsistent with number of nuclei and electrons').
    """
    if A.ndim == 0:
        # torchspin shorthand: isotropic for all nuclei and electrons
        return A.expand(n_n, 3 * n_e).clone()
    if A.ndim == 1:
        if n_e == 1 and A.shape[0] == 3 and n_n not in (1, 3):
            # torchspin shorthand: [Ax, Ay, Az] for every nucleus
            return A.unsqueeze(0).expand(n_n, 3).clone()
        A = A.unsqueeze(0).clone()  # MATLAB 1xk row
    if A.ndim == 2:
        r, c = A.shape
        if (r, c) in ((3 * n_n, 3 * n_e), (n_n, 3 * n_e)):
            return A                # full tensors (591-593) / principal values (612-613)
        if (r, c) == (1, n_n):
            # one isotropic value per nucleus, one electron spin only (594-601)
            if n_e == 1:
                return A.reshape(n_n, 1).expand(n_n, 3).clone()
            raise ValueError(
                f'Size of A {tuple(A.shape)} is inconsistent with number of electrons: one value '
                f'per nucleus is only for one electron spin, the system has {n_e}. Give '
                f'({n_n}, {n_e}) isotropic values, one per nucleus and electron.')
        if (r, c) == (n_n, n_e):
            # isotropic per nucleus and electron: kron(A, [1 1 1]) (602-604)
            return A.repeat_interleave(3, dim=1)
        if (r, c) == (n_n, 2 * n_e):
            # axial [A_perp, A_par] per electron -> [A_perp A_perp A_par] (605-611)
            idx = [2 * k + i for k in range(n_e) for i in (0, 0, 1)]
            return A[:, idx]
    raise ValueError(
        f'Size of A {tuple(A.shape)} is inconsistent with number of nuclei ({n_n}) and electrons '
        f'({n_e}): expected a scalar, one value per nucleus (one electron spin), or '
        f'(nNuclei, nElectrons) isotropic, (nNuclei, 2*nElectrons) axial, (nNuclei, 3*nElectrons) '
        f'principal values or (3*nNuclei, 3*nElectrons) full tensors.')


def _normalize_lw(v, name: str):
    """Normalize a line width ``lw`` / ``lwpp`` to ``[Gaussian, Lorentzian]``.

    As EasySpin's validatespinsys.m (774-777, 790-793): a scalar is
    ``[x, 0]``, otherwise the value must be a 1x2 row ``[G, L]`` (a nested
    ``[[G, L]]`` is the same row).  A length-2 list, tuple, array or tensor is
    kept as given (tensors keep their autograd graph).  Any other size raises
    ``ValueError`` ('System.lw has wrong size.').
    """
    if isinstance(v, (int, float)):
        return [float(v), 0.0]
    if isinstance(v, (torch.Tensor, np.ndarray)):
        shape = tuple(v.shape)
    else:
        try:
            shape = np.shape(v)
        except Exception:
            shape = (len(v),)
    if len(shape) == 2 and shape[0] == 1:
        v = v[0]          # 1xk row
        shape = shape[1:]
    if len(shape) == 0:
        return [float(v), 0.0]
    if len(shape) != 1 or shape[0] not in (1, 2):
        raise ValueError(
            f'{name} has wrong size {tuple(shape)}: expected a scalar or [GaussianFWHM, LorentzianFWHM].')
    if shape[0] == 1:
        return [float(v[0]), 0.0]
    return v


def _three_widths(x, name: str) -> Optional[torch.Tensor]:
    """Normalize ``HStrain`` / ``AStrain`` to three widths ``[x, y, z]``.

    As EasySpin's validatespinsys.m (877-899), which counts elements and
    ignores the array's shape: empty is no strain (``None``), one value is
    ``[x x x]``, two are ``[x x y]``, three are kept, any other number raises
    ``ValueError`` ('Sys.HStrain must have 3 elements!').
    """
    t = _to_tensor(x, torch.float64)
    if t.numel() == 0:
        return None
    if t.ndim != 1:
        t = t.reshape(-1)
    if t.numel() == 1:
        return t[[0, 0, 0]]
    if t.numel() == 2:
        return t[[0, 0, 1]]
    if t.numel() == 3:
        return t
    raise ValueError(
        f'{name} must have 3 elements (or 1, [x] -> [x x x], or 2, [x y] -> [x x y]); got {t.numel()}.')


def _euler_rows(x, n_rows: int, name: str) -> torch.Tensor:
    """Normalize Euler angles of ``n_rows`` tensors (gFrame, DFrame) to ``(n_rows, 3)``.

    As EasySpin's validatespinsys.m (189-193 for gFrame, 240-244 for DFrame):
    ``None`` or an empty array is all zeros, and anything else must have
    exactly ``(n_rows, 3)`` angles.  A 1-D ``[a, b, c]`` is the 1x3 row of a
    MATLAB vector, so it is one electron spin's angles.  Any other shape
    raises ``ValueError`` ('Spin system field ... has wrong size').
    """
    if x is None:
        return torch.zeros(n_rows, 3, dtype=torch.float64)
    t = _to_tensor(x, torch.float64)
    if t.numel() == 0:
        return torch.zeros(n_rows, 3, dtype=torch.float64)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    if tuple(t.shape) != (n_rows, 3):
        raise ValueError(
            f'Spin system field {name} has wrong size {tuple(t.shape)} for the given spins: '
            f'expected ({n_rows}, 3) Euler angles, one row [alpha, beta, gamma] per tensor.')
    return t


def _to_tensor(x, dtype: torch.dtype) -> torch.Tensor:
    """Convert *x* to a ``torch.Tensor`` with the given dtype.

    Tensors are converted with ``.to`` (no copy, autograd graph preserved), so
    parameters that ``requires_grad`` stay differentiable inside the spin system
    (``pepper_autograd``, ``differentiable_spectrum``).  Lists or tuples whose
    elements are tensors (e.g. ``g=[g_tensor]``, ``A=[[ax, ay, az]]`` with
    tensor entries) are stacked instead of copied through ``torch.tensor``,
    which would detach them (and fails for multi-element tensors).
    """
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    if isinstance(x, (list, tuple)) and _contains_tensor(x):
        return torch.stack([_to_tensor(e, dtype) for e in x]) if len(x) else torch.zeros(0, dtype=dtype)
    return torch.tensor(x, dtype=dtype)


def _contains_tensor(x) -> bool:
    if isinstance(x, torch.Tensor):
        return True
    if isinstance(x, (list, tuple)):
        return any(_contains_tensor(e) for e in x)
    return False
