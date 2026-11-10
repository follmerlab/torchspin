"""DDPOWJ: field-swept powder EPR of an exchange-coupled dimer, with Telser's product level strain.

Torch port of Joshua Telser's DDPOWJ in its exact mode (via telser-tools, ``telser_tools_diff.ddpowj``).  A
dimer is a two-electron torchspin :class:`~torchspin.SpinSystem` (``ee``, ``ee2``; build it from Telser's
parameters with :func:`torchspin.lft.telser2torchspin.dimer2sys` or :func:`~torchspin.lft.telser2torchspin.ddpowj2sys`),
so the spectrum is :func:`torchspin.lft.ddpowh.epr` with DDPOWJ's defaults:

* every level pair i < j (``pairs=None``; DDPOWJ's level ranges are 0-based (i, j) lists here, as
  :func:`level_range_pairs` builds them);
* the strain weights of a basis state |m1, m2> are the **product** WMS1(m1) WMS2(m2) of the two sites'
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
    """DDPOWJ's basis weights WMS1(m1) WMS2(m2) for every |m1, m2> state of a dimer ``sys``.

    ``ms1`` (2S1+1,), ``ms2`` (2S2+1,): each site's factors for m = S ... -S (Telser's order).  Sites with
    S < 1 have no factors in Telser's input; their weights are ``s12_weight`` (0.0: exact; 1.0: the Fortran,
    finding DDPOWJ-10; see the module docstring).  A one-electron ``sys`` (the dimer of an S = 0 site with
    a spin S) multiplies the electron's factors by the S = 0 site's single weight ``partner_weight``
    (default ``s12_weight``).  Tensors keep their graph.
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
    """DDPOWJ's pair selection: (a, b, c, d) 1-based level numbers, i in a ... b, j in c ... d, i < j (0-based).

    Levels are numbered by ascending energy at each (orientation, field).  Pairs outside ``n_levels`` are
    dropped.  (The Fortran's i >= j pairs are a reproduction option of telser-tools, not ported.)
    """
    a, b, c, d = (int(x) for x in level_range)
    out = []
    for i in range(a, b + 1):
        for j in range(c, d + 1):
            if i < j and 1 <= i <= n_levels and 1 <= j <= n_levels:
                out.append((i - 1, j - 1))
    return tuple(out)


def epr(sys: SpinSystem, fields, mw_freq, *, width, pairs=None, strain: Strain | None = None, **kwargs) -> PowderResult:
    """Field-swept powder EPR of the dimer ``sys``: :func:`torchspin.lft.ddpowh.epr` with DDPOWJ's defaults.

    ``pairs`` None: every pair i < j.  ``strain``: a :class:`Strain` with :func:`strain_weights` (product
    form).  Other keywords: :func:`torchspin.lft.ddpowh.epr`.
    """
    if pairs is not None:
        bad = [(i, j) for i, j in pairs if not i < j]
        if bad:
            raise ValueError(f"pairs need i < j (0-based); got {bad[:3]}")
    return _h.epr(sys, fields, mw_freq, width=width, pairs=pairs, strain=strain, **kwargs)
