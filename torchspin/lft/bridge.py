"""Bridges from ligand-field models to torchspin spin systems.

:func:`dn2sys` turns d^n ligand-field parameters into a :class:`torchspin.SpinSystem` with
S = 1/2 and the **full 3x3 molecular-frame g matrix** of the ground Kramers doublet.  The full
matrix (not principal values plus ``gFrame``) keeps the autograd graph intact, so
ligand-field parameters -> g -> :func:`torchspin.pepper` (or ``pepper_autograd``) is
differentiable end to end without touching the spectrum code.
"""

from __future__ import annotations

from torchspin.spinsystem import SpinSystem

from . import ddn as _ddn

__all__ = ["dn2sys"]


def dn2sys(n: int, params: dict, *, units: str = "MHz", min_gap=None, **sys_kwargs) -> tuple[SpinSystem, _ddn.GTensor]:
    """(SpinSystem, GTensor) for the ground doublet of d^n.

    ``params``: keywords of :func:`torchspin.lft.ddn.g_tensor` (``B``, ``C``, ``zeta``, ``V``,
    ``k``, ...) in ``units``.  They are passed as a dict because ``A`` means the Racah A there
    and the hyperfine tensor in :class:`SpinSystem`.

    ``sys_kwargs`` go to :class:`SpinSystem` (``Nucs``, ``A``, ``lw``, ``HStrain``, ...), in
    torchspin's units.  ``S`` and ``g`` are set here.

    ``min_gap`` (in ``units``): raise if the doublet is closer than this to the next state,
    where an effective spin 1/2 is not a good description (and gradients grow as 1/gap).
    """
    for key in ("S", "g"):
        if key in sys_kwargs:
            raise TypeError(f"dn2sys sets {key!r} itself")
    gt = _ddn.g_tensor(n, units=units, **params)
    if min_gap is not None and bool((gt.gap < min_gap).any()):
        raise ValueError(f"ground doublet is {float(gt.gap.min()):.4g} {units} from the next state, "
                         f"below min_gap = {min_gap}")
    if gt.g.ndim != 2:
        raise ValueError("dn2sys builds one SpinSystem; pass unbatched parameters")
    return SpinSystem(S=0.5, g=gt.g, **sys_kwargs), gt
