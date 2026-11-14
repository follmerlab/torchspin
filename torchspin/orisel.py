"""Orientation selectivity weights for a fixed-field EPR experiment.

Port of EasySpin's ``orisel.m``.

At a fixed static field ``Exp.Field`` and microwave frequency ``Exp.mwFreq``,
``orisel`` computes how strongly each orientation of a powder is excited by a
microwave pulse (or a narrow spectrometer bandwidth) of width
``Exp.ExciteWidth``.  The weights are used to select orientations in ENDOR and
pulse EPR powder simulations.

The weight model is EasySpin's (orisel.m:110-153): the full spin Hamiltonian is
diagonalized at the field, and every level pair ``(u, v)`` contributes its
transition rate times a Gaussian in frequency centered at ``mwFreq``, whose
width combines ``ExciteWidth`` and the ``HStrain`` projected on the field
direction.  The computation is a differentiable torch graph (``g``, ``A``,
``D``, ``HStrain``, ...), shared with :func:`torchspin.salt.salt`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch

from torchspin._linalg import eigh as _eigh

__all__ = ['orisel', 'OriselOptions']

_GRID_SYMMETRIES = ('O3', 'Dinfh', 'D6h', 'D4h', 'Oh', 'D3d', 'Th', 'D2h',
                    'C4h', 'C6h', 'C2h', 'S6', 'Ci', 'C1')


@dataclass
class OriselOptions:
    """Options for :func:`orisel` (EasySpin ``Opt`` of ``orisel``).

    Attributes
    ----------
    GridSize : int
        Number of SOPHE grid knots along a quarter meridian.  Default 46, as in
        EasySpin (orisel.m:66, the same as salt and nucfrq2d).  Not used when
        ``Exp.Orientations`` is given.
    GridSymmetry : str
        Grid symmetry (``'Dinfh'``, ``'D2h'``, ``'C2h'``, ``'Ci'``, ``'C1'``,
        ...).  ``'auto'`` or ``''`` (default ``'auto'``): determined by
        :func:`~torchspin.hamsymm.hamsymm`, and the grid is rotated into the
        symmetry frame it returns (EasySpin ``GridFrame``, orisel.m:79-93).  An
        explicit symmetry uses the molecular frame.
    AveragedIntensity : bool
        ``True`` (default): the transition rate of a level pair is averaged over
        the microwave field direction in the plane perpendicular to the static
        field, ``(|<u|mu_xL|v>|^2 + |<u|mu_yL|v>|^2)/2``.  ``False``: only the
        lab x direction, ``|<u|mu_xL|v>|^2`` (EasySpin orisel.m:145-149).

    The pre-0.9.0 options ``ExcitBandwidth`` and ``Threshold`` were removed; the
    excitation width is ``Experiment.ExciteWidth`` (FWHM, MHz).
    """
    GridSize: int = 46
    GridSymmetry: str = 'auto'
    AveragedIntensity: bool = True
    # Removed in 0.9.0; present only to give a helpful error.
    ExcitBandwidth: Optional[float] = field(default=None, repr=False)
    Threshold: Optional[float] = field(default=None, repr=False)

    def __post_init__(self):
        if self.ExcitBandwidth is not None:
            raise ValueError(
                "OriselOptions.ExcitBandwidth was removed in torchspin 0.9.0: orisel now "
                "follows EasySpin and reads the excitation width from "
                "Experiment.ExciteWidth (FWHM, MHz).")
        if self.Threshold is not None:
            raise ValueError(
                "OriselOptions.Threshold was removed in torchspin 0.9.0: the weights are no "
                "longer normalized (EasySpin orisel); threshold them yourself, e.g. "
                "w[w < thr * w.max()] = 0.")
        if isinstance(self.GridSize, bool) or not isinstance(self.GridSize, int) \
                or self.GridSize < 2:
            raise ValueError(f"OriselOptions.GridSize must be an integer >= 2; got "
                             f"{self.GridSize!r}.")
        if self.GridSymmetry not in ('auto', '') + _GRID_SYMMETRIES:
            raise ValueError(f"OriselOptions.GridSymmetry must be 'auto' or one of "
                             f"{_GRID_SYMMETRIES}; got {self.GridSymmetry!r}.")


def _orisel_weights(H0, mux, muy, muz, phi, theta, Field, mwFreq_MHz, ExciteWidth,
                    HStrain=None, averaged_intensity=True):
    """Per-transition orientation-selection weights (EasySpin orisel.m:120-154).

    Differentiable torch implementation shared by :func:`orisel` and
    :func:`torchspin.salt.salt`.

    Parameters
    ----------
    H0, mux, muy, muz : Tensor
        Field-independent Hamiltonian (MHz) and magnetic-moment operators
        (MHz/mT), as returned by ``ham(sys)``.
    phi, theta : Tensor, shape (M,)
        Static-field directions (rad) in the molecular frame.
    Field : float or Tensor
        Static field (mT).
    mwFreq_MHz : float or Tensor
        Microwave frequency (MHz).
    ExciteWidth : float or Tensor
        Excitation FWHM (MHz), finite.
    HStrain : Tensor, shape (3,), optional
        Frequency-domain strain FWHM (MHz) along the molecular axes.
    averaged_intensity : bool
        Transition rate averaged over the B1 direction (EasySpin
        ``Opt.AveragedIntensity``).

    Returns
    -------
    W : Tensor, shape (M, P)
        ``rate_uv * exp(-2 xi_uv^2)`` for the P = N(N-1)/2 level pairs u < v,
        in the order of EasySpin's ``nchoosek(1:N, 2)``;
        ``xi = (|E_v - E_u| - mwFreq) / Gamma``,
        ``Gamma^2 = (sum_i n_i^2 HStrain_i^2 + ExciteWidth^2) / (2 ln 2)``
        with ``n`` the field direction.
    E : Tensor, shape (M, N)
        Ascending energy levels (MHz) at each orientation.
    u_idx, v_idx : Tensor, shape (P,)
        0-based level indices of each pair.
    """
    device = H0.device
    dt64 = torch.float64
    n_orient = phi.shape[0]
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    sin_p = torch.sin(phi)
    cos_p = torch.cos(phi)
    # rows of erot(phi, theta, 0): lab axes in the molecular frame
    zx, zy, zz = sin_t * cos_p, sin_t * sin_p, cos_t
    xx, xy, xz = cos_t * cos_p, cos_t * sin_p, -sin_t
    yx, yy = -sin_p, cos_p  # yLab_z = 0

    muzL = (zx.view(-1, 1, 1) * mux + zy.view(-1, 1, 1) * muy
            + zz.view(-1, 1, 1) * muz)
    muxL = (xx.view(-1, 1, 1) * mux + xy.view(-1, 1, 1) * muy
            + xz.view(-1, 1, 1) * muz)

    H_B = H0.unsqueeze(0) - Field * muzL
    E_b, V_b = _eigh(H_B)          # (M, dim), (M, dim, dim)
    dim = H0.shape[0]
    iu = torch.triu_indices(dim, dim, offset=1, device=device)
    u_idx, v_idx = iu[0], iu[1]
    freqs = (E_b[:, v_idx] - E_b[:, u_idx]).abs()   # (M, P) MHz

    Vh = V_b.conj().transpose(1, 2)
    mxe = torch.bmm(Vh, torch.bmm(muxL, V_b))
    if averaged_intensity:
        muyL = yx.view(-1, 1, 1) * mux + yy.view(-1, 1, 1) * muy
        mye = torch.bmm(Vh, torch.bmm(muyL, V_b))
        rate = (mxe.abs() ** 2 + mye.abs() ** 2) / 2.0
    else:
        rate = mxe.abs() ** 2
    rate_uv = rate[:, u_idx, v_idx]                  # (M, P)

    if torch.is_tensor(ExciteWidth):
        lw2 = (ExciteWidth.to(dtype=dt64, device=device) ** 2).expand(n_orient)
    else:
        lw2 = torch.full((n_orient,), float(ExciteWidth) ** 2, dtype=dt64, device=device)
    if HStrain is not None:
        hs = torch.as_tensor(HStrain).to(dtype=dt64, device=device)
        h_os = torch.stack([zx, zy, zz], dim=1)
        lw2 = lw2 + ((h_os ** 2) * (hs ** 2).unsqueeze(0)).sum(dim=1)
    gamma2 = lw2 / (2.0 * math.log(2.0))             # GammaWidth^2
    xi2 = (freqs - mwFreq_MHz) ** 2 / gamma2.unsqueeze(1)
    W = rate_uv * torch.exp(-2.0 * xi2)
    return W, E_b, u_idx, v_idx


def _vec2ang(v):
    """EasySpin vec2ang: v (3, M) -> phi, theta (rad)."""
    x, y, z = v[0], v[1], v[2]
    phi = torch.atan2(y, x)
    theta = math.pi / 2 - torch.atan2(z, torch.sqrt(x ** 2 + y ** 2))
    return phi, theta


def _scalar(x) -> float:
    return float(x.detach()) if torch.is_tensor(x) else float(x)


def _required(exp, name, what):
    val = getattr(exp, name, None)
    if val is None:
        hint = ''
        if name == 'Field' and getattr(exp, 'Range', None) is not None:
            hint = (" orisel no longer sweeps Experiment.Range (torchspin 0.9.0): it "
                    "follows EasySpin and evaluates the selection at one field. Use e.g. "
                    "Experiment(mwFreq=9.5, Field=340, ExciteWidth=100).")
        if name == 'ExciteWidth':
            hint = (" (OriselOptions.ExcitBandwidth was removed in torchspin 0.9.0; "
                    "use math.inf for no selection.)")
        raise ValueError(f"orisel: Experiment.{name} ({what}) is missing.{hint}")
    return val


def orisel(sys, exp, opt: Optional[OriselOptions] = None, *, per_transition: bool = False):
    """Orientation selectivity weights at a fixed field (EasySpin ``orisel``).

    For each orientation of the static field in the molecular frame, the
    spin Hamiltonian is diagonalized at ``exp.Field``; every pair of levels
    ``u < v`` contributes

    .. math::

        w_{uv} = R_{uv}\\, \\exp(-2\\xi_{uv}^2), \\qquad
        \\xi_{uv} = \\frac{|E_v - E_u| - \\nu_{\\text{mw}}}{\\Gamma}, \\qquad
        \\Gamma = \\frac{\\sqrt{\\sum_i n_i^2 \\,\\text{HStrain}_i^2
                 + \\text{ExciteWidth}^2}}{\\sqrt{2\\ln 2}},

    with :math:`R_{uv}` the transition rate
    :math:`(|\\langle u|\\mu_{xL}|v\\rangle|^2 + |\\langle u|\\mu_{yL}|v\\rangle|^2)/2`
    (``AveragedIntensity``) and :math:`n` the field direction (orisel.m:120-154).
    The weights are neither normalized nor multiplied by grid (solid-angle)
    weights, as in EasySpin.  Any number of electron and nuclear spins is
    supported (full diagonalization).  The result is differentiable with respect
    to tensor-valued spin-system parameters (``g``, ``A``, ``D``, ``HStrain``, ...).

    Parameters
    ----------
    sys : SpinSystem
        Spin system (``HStrain`` in MHz, molecular frame; ``lw`` is not used).
    exp : Experiment
        Required: ``Field`` (mT, > 0), ``mwFreq`` (GHz) and ``ExciteWidth``
        (excitation FWHM, MHz; ``math.inf`` gives weight 1 for every
        orientation and transition, as in EasySpin).  Optional:
        ``Orientations``, an ``(N, 2)`` array of ``[phi, theta]`` (rad) in the
        molecular frame; when given, these orientations are used instead of a
        grid.
    opt : OriselOptions, optional
        ``GridSize`` (default 46), ``GridSymmetry`` (default from
        :func:`~torchspin.hamsymm.hamsymm`, grid in its symmetry frame) and
        ``AveragedIntensity`` (default True).
    per_transition : bool, keyword-only
        ``False`` (default): EasySpin's one-output form, weights summed over
        transitions.  ``True``: EasySpin's two-output form
        ``[Weights, Transitions] = orisel(...)``.

    Returns
    -------
    With ``per_transition=False``: ``(weights, phi, theta)``

    weights : Tensor, shape (nOri,)
        Selectivity per orientation (sum over all level pairs).
    phi, theta : Tensor, shape (nOri,)
        Orientations (rad, molecular frame) the weights belong to: the grid
        rotated into the molecular frame (EasySpin ``vec2ang(GridFrame*vecs)``)
        or ``exp.Orientations``.

    With ``per_transition=True``: ``(Weights, Transitions, phi, theta)``

    Weights : Tensor, shape (nOri, nTrans)
        Weight of every level pair at every orientation.
    Transitions : Tensor (int64), shape (nTrans, 2)
        Level pairs ``(u, v)``, **1-based** with ``u < v`` and levels numbered
        in ascending energy, in the order of EasySpin's ``nchoosek(1:N, 2)``
        (the same numbering as ``Options.Transitions``).  All
        ``N(N-1)/2`` pairs are listed, including forbidden ones.

    Examples
    --------
    >>> from torchspin import SpinSystem
    >>> from torchspin.experiment import Experiment
    >>> from torchspin.orisel import orisel
    >>> sys = SpinSystem(S=0.5, g=[2.0, 2.1, 2.2], HStrain=[50, 50, 50])
    >>> exp = Experiment(mwFreq=9.5, Field=320, ExciteWidth=100)
    >>> w, phi, theta = orisel(sys, exp)
    >>> W, trans, phi, theta = orisel(sys, exp, per_transition=True)
    >>> trans.tolist()
    [[1, 2]]
    """
    if opt is None:
        opt = OriselOptions()
    elif not isinstance(opt, OriselOptions):
        # duck-typed options (e.g. the Options used for salt), as in EasySpin
        # where the same Opt struct can be passed to orisel and salt
        opt = OriselOptions(
            GridSize=getattr(opt, 'GridSize', 46),
            GridSymmetry=getattr(opt, 'GridSymmetry', 'auto'),
            AveragedIntensity=getattr(opt, 'AveragedIntensity', True))

    from torchspin.ham import ham
    from torchspin.hamsymm import grid_symmetry
    from torchspin.sphgrid import sphgrid

    Field = _required(exp, 'Field', 'mT')
    mwFreq = _required(exp, 'mwFreq', 'GHz')
    ExciteWidth = _required(exp, 'ExciteWidth', 'FWHM, MHz')
    if _scalar(Field) <= 0:
        raise ValueError("orisel: Experiment.Field must be positive (mT).")
    ew = _scalar(ExciteWidth)
    if math.isnan(ew) or ew < 0:
        raise ValueError("orisel: Experiment.ExciteWidth must be non-negative (FWHM, MHz).")
    HStrain = getattr(sys, 'HStrain', None)
    hs_d = (torch.zeros(3, dtype=torch.float64) if HStrain is None
            else torch.as_tensor(HStrain).detach().to(torch.float64).reshape(-1))
    if bool(torch.any(hs_d + ew <= 0)):
        # orisel.m:58-60
        raise ValueError("orisel: Sys.HStrain or Experiment.ExciteWidth must be positive.")

    # ── Orientations ────────────────────────────────────────────────────────
    if getattr(exp, 'Orientations', None) is not None:
        ori = torch.as_tensor(exp.Orientations, dtype=torch.float64)
        if ori.ndim == 1 and ori.shape[0] == 2:
            ori = ori.reshape(1, 2)
        if ori.ndim != 2 or ori.shape[1] != 2:
            raise ValueError("orisel: Experiment.Orientations must be an (N, 2) array of "
                             f"[phi, theta] (rad); got shape {tuple(ori.shape)}.")
        phi, theta = ori[:, 0], ori[:, 1]
    else:
        sym = opt.GridSymmetry
        if sym in ('auto', ''):
            sym, R = grid_symmetry(sys)
            grid_frame = R.T          # EasySpin GridFrame: v_mol = GridFrame @ v_sym
        else:
            grid_frame = torch.eye(3, dtype=torch.float64)
        _, _, _, vecs = sphgrid(sym, opt.GridSize)
        if vecs.shape[0] != 3:
            vecs = vecs.T
        phi, theta = _vec2ang(grid_frame.to(vecs.dtype) @ vecs)

    # ── Weights ─────────────────────────────────────────────────────────────
    H0, mux, muy, muz = ham(sys, B0=None)
    N = H0.shape[0]
    iu = torch.triu_indices(N, N, offset=1)
    transitions = (iu.T + 1).contiguous()           # 1-based (u, v)

    if math.isinf(ew):
        W = torch.ones(phi.shape[0], transitions.shape[0], dtype=torch.float64)
    else:
        W, _, _, _ = _orisel_weights(
            H0, mux, muy, muz, phi, theta, Field, mwFreq * 1e3, ExciteWidth,
            HStrain=HStrain, averaged_intensity=bool(opt.AveragedIntensity))

    if per_transition:
        return W, transitions, phi, theta
    return W.sum(dim=1), phi, theta
