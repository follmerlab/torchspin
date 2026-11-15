"""DDPOWJ: field-swept powder EPR of an exchange-coupled dimer, with Telser's product level strain.

Torch port of Josh Telser's DDPOWJ in its exact mode (via telser-tools, ``telser_tools_diff.ddpowj``).  A
dimer is a two-electron torchspin :class:`~torchspin.SpinSystem` (``ee``, ``ee2``; build it from Telser's
parameters with :func:`torchspin.lft.telser2torchspin.dimer2sys` or :func:`~torchspin.lft.telser2torchspin.ddpowj2sys`),
so the spectrum is :func:`torchspin.lft.ddpowh.epr` with DDPOWJ's defaults:

* every level pair i < j (``pairs=None``; DDPOWJ's level ranges are 0-based (i, j) lists here, as
  :func:`level_range_pairs` builds them);
* the strain weights of a basis state ``|m1, m2>`` are the **product** WMS1(m1) WMS2(m2) of the two sites'
  factors (:func:`strain_weights`), not DDPOWH's sum.

Open question (DDPOWJ-10, waiting on J. Telser): an S = 1/2 site has no m_S strain factors (WMS = 0), so in
the exact model every product weight of a dimer with an S = 1/2 partner is 0, and only (B/B_set)^b W(n)
remains.  The Fortran instead sets WMS = 1 for S = 1/2 sites, which makes every weight 1 and doubles every
width (1 + (1 + 1)/2 = 2).  ``s12_weight`` keeps both readings: 0.0 (default, exact) or 1.0 (Fortran).  Until
Telser answers, treat strain fits of dimers with an S = 1/2 partner as provisional.

Units, intensities, gradients and the other keywords: :mod:`torchspin.lft.ddpowh`.
"""

from __future__ import annotations

import torch

from torchspin.spinsystem import SpinSystem

from .ddpowh import TORCHSPIN_CONSTANTS, PowderResult, SpinConstants, Strain, _t
from . import ddpowh as _h
from .ddpowh import simulate

__all__ = ["SpinConstants", "TORCHSPIN_CONSTANTS", "Strain", "PowderResult", "strain_weights", "level_range_pairs",
           "epr", "simulate"]

_F64 = torch.float64


def strain_weights(sys: SpinSystem, ms1=None, ms2=None, *, s12_weight: float = 0.0, partner_weight=None) -> torch.Tensor:
    """DDPOWJ's product strain weights ``WMS1(m1) WMS2(m2)`` of a dimer.

    Parameters
    ----------
    sys : SpinSystem
        The dimer: one or two electron spins and no nuclei.  A one-electron
        system is the dimer of an S = 0 site with a spin S.
    ms1 : array_like or torch.Tensor, shape (2S1+1,), optional
        Site 1 factors for ``m = S1, ..., -S1`` (Telser's order),
        dimensionless.  ``None``: zeros.
    ms2 : array_like or torch.Tensor, shape (2S2+1,), optional
        Site 2 factors.  For a one-electron ``sys``, used if ``ms1`` is
        ``None``.
    s12_weight : float or torch.Tensor, default 0.0
        Weight of every state of a site with S < 1, which has no factors in
        Telser's input (given factors are ignored): 0.0 is the exact model,
        1.0 the Fortran (finding DDPOWJ-10; see the module docstring).
    partner_weight : float or torch.Tensor, optional
        One-electron ``sys`` only: the single weight of the S = 0 site that
        multiplies the electron's factors; default ``s12_weight``.

    Returns
    -------
    torch.Tensor, shape (nStates,)
        Weights for every basis state ``|m1, m2>`` (both m descending), to
        be passed as ``Strain(weights)``.  Tensors keep their graph.

    Raises
    ------
    ValueError
        If ``sys`` has nuclei or not one or two electron spins, or a factor
        vector has the wrong length.

    See Also
    --------
    torchspin.lft.ddpowh.strain_weights : DDPOWH's additive weights.

    Examples
    --------
    An S = 1 site coupled to an S = 1/2 site, Fortran reading of the S = 1/2
    weights:

    >>> from torchspin import SpinSystem
    >>> from torchspin.lft import ddpowj
    >>> dimer = SpinSystem(S=[1, 0.5], ee=3000.0)
    >>> ddpowj.strain_weights(dimer, ms1=[1.0, 0.0, 1.0], s12_weight=1.0).tolist()
    [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    """
    spins = [float(s) for s in sys.Spins][: int(sys.nElectrons)]
    if int(sys.nNuclei) != 0:
        raise ValueError("DDPOWJ dimers have no nuclei")
    if len(spins) not in (1, 2):
        raise ValueError("a DDPOWJ dimer has one or two electron spins")

    def site(vec, s):
        n = int(round(2 * s)) + 1
        if s < 1:
            return _t(s12_weight).reshape(()).expand(n)
        if vec is None:
            return torch.zeros(n, dtype=_F64)
        v = _t(vec)
        if v.shape != (n,):
            raise ValueError(f"S = {s}: need {n} strain factors (m = S ... -S), got {tuple(v.shape)}")
        return v

    if len(spins) == 1:
        pw = _t(s12_weight if partner_weight is None else partner_weight)
        return site(ms1 if ms1 is not None else ms2, spins[0]) * pw
    w1, w2 = site(ms1, spins[0]), site(ms2, spins[1])
    return (w1[:, None] * w2[None, :]).reshape(-1)


def level_range_pairs(level_range, n_levels: int) -> tuple[tuple[int, int], ...]:
    """DDPOWJ's level-range pair selection.

    Parameters
    ----------
    level_range : sequence of 4 int
        ``(a, b, c, d)``, 1-based level numbers (ascending energy at each
        orientation and field): i runs over a to b, j over c to d, and only
        ``i < j`` is kept.
    n_levels : int
        Number of levels; pairs beyond it are dropped.

    Returns
    -------
    tuple of tuple of int
        0-based pairs ``(i, j)``.  (The Fortran's ``i >= j`` pairs are a
        telser-tools reproduction option, not ported.)

    Examples
    --------
    >>> from torchspin.lft import ddpowj
    >>> ddpowj.level_range_pairs((1, 2, 3, 4), 6)
    ((0, 2), (0, 3), (1, 2), (1, 3))
    """
    a, b, c, d = (int(x) for x in level_range)
    out = []
    for i in range(a, b + 1):
        for j in range(c, d + 1):
            if i < j and 1 <= i <= n_levels and 1 <= j <= n_levels:
                out.append((i - 1, j - 1))
    return tuple(out)


def epr(sys: SpinSystem, fields, mw_freq, *, width, pairs=None, strain: Strain | None = None, **kwargs) -> PowderResult:
    """Field-swept powder EPR of a dimer: :func:`torchspin.lft.ddpowh.epr` with DDPOWJ's defaults.

    Parameters
    ----------
    sys : SpinSystem
        The dimer (two electron spins coupled by ``ee``/``ee2``), e.g. from
        :func:`torchspin.lft.telser2torchspin.dimer2sys`.
    fields : array_like or torch.Tensor, shape (npts,)
        Field points in mT.
    mw_freq : float or torch.Tensor
        Microwave frequency in GHz.
    width : float or array_like, shape (3,)
        HWHM ``(Wx, Wy, Wz)`` in ``units`` (default MHz).
    pairs : sequence of (int, int), optional
        0-based level pairs with ``i < j`` (:func:`level_range_pairs`);
        ``None`` (default): every pair.
    strain : Strain, optional
        Level strain with :func:`strain_weights` (product form).
    **kwargs
        Every other keyword of :func:`torchspin.lft.ddpowh.epr`.

    Returns
    -------
    PowderResult
        ``axis`` = ``fields`` (mT), ``spectrum`` of shape ``(npts,)``.

    Raises
    ------
    ValueError
        If a pair has ``i >= j``, and as :func:`torchspin.lft.ddpowh.epr`.

    Examples
    --------
    Two S = 1/2 spins with strong exchange (a triplet at g = 2):

    >>> import torch
    >>> from torchspin import SpinSystem
    >>> from torchspin.lft import ddpowj
    >>> dimer = SpinSystem(S=[0.5, 0.5], ee=30000.0)
    >>> res = ddpowj.epr(dimer, torch.linspace(300, 380, 81), 9.5, width=50.0, igloo=6)
    >>> res.spectrum.shape
    torch.Size([81])
    """
    if pairs is not None:
        bad = [(i, j) for i, j in pairs if not i < j]
        if bad:
            raise ValueError(f"pairs need i < j (0-based); got {bad[:3]}")
    return _h.epr(sys, fields, mw_freq, width=width, pairs=pairs, strain=strain, **kwargs)
