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
    """Spin-1/2 SpinSystem of the ground Kramers doublet of a d^n ligand-field model.

    Parameters
    ----------
    n : int
        Number of d electrons (an odd n gives a Kramers doublet).
    params : dict
        Keywords of :func:`torchspin.lft.ddn.g_tensor` (``B``, ``C``,
        ``zeta``, ``V``, ``k``, ...), energies in ``units``.  They are passed
        as a dict because ``A`` is the Racah A there and the hyperfine
        tensor in :class:`~torchspin.SpinSystem`.  Tensors that require grad
        stay on the graph.
    units : str, default 'MHz'
        Energy unit of ``params`` and ``min_gap`` (``'cm^-1'``, ``'MHz'``,
        ... of :mod:`torchspin.lft.units`).
    min_gap : float, optional
        Raise if the doublet is closer than this (in ``units``) to the next
        state, where an effective spin 1/2 is not a good description and
        gradients grow as 1/gap.  ``None`` (default): no check.
    **sys_kwargs
        Passed to :class:`~torchspin.SpinSystem` (``Nucs``, ``A``, ``lw``,
        ``HStrain``, ...) in torchspin's units (MHz, mT).  ``S`` and ``g``
        are set here.

    Returns
    -------
    sys : SpinSystem
        ``S = 1/2`` with the full 3x3 molecular-frame g matrix
        (``sys.g`` of shape (3, 3); not principal values plus ``gFrame``, so
        the graph stays intact).
    gt : torchspin.lft.ddn.GTensor
        The g tensor (``g``, ``ggT``, ``principal`` ascending, ``gap`` in
        ``units``).

    Raises
    ------
    TypeError
        If ``sys_kwargs`` contains ``S`` or ``g``.
    ValueError
        If the gap is below ``min_gap``, or the parameters are batched (more
        than one g tensor).

    See Also
    --------
    torchspin.lft.ddn.g_tensor, torchspin.pepper

    Examples
    --------
    Tetragonally elongated Cu(II), d^9, parameters in cm^-1:

    >>> from torchspin import lft
    >>> V = lft.crystal_field(1200.0, 2000.0, 300.0)       # Dq, Ds, Dt
    >>> sys, gt = lft.dn2sys(9, dict(V=V, zeta=830.0, k=(0.8, 0.8, 0.8)), units='cm^-1', lw=1.0)
    >>> sys.g.shape, [round(v, 3) for v in gt.principal.tolist()]
    (torch.Size([3, 3]), [2.063, 2.063, 2.423])
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
