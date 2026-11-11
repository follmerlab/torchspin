"""Hermitian eigendecomposition with derivatives that are correct at degenerate levels.

The d^n Hamiltonians are full of exact degeneracies: every Kramers doublet at zero field,
orbital degeneracies in cubic and axial fields, and accidental crossings.  The stock
``torch.linalg.eigh`` backward contains 1/(lambda_j - lambda_i) and returns inf/NaN or,
worse, finite but wrong gradients there.  Three primitives cover what the ligand-field
programs need; each is exact for its class of output and has both a backward (reverse
mode) and a jvp (forward mode, used for least-squares Jacobians: few parameters, many
residuals).

:func:`eigh_gauge`
    Eigenvalues and eigenvectors for outputs that are invariant under unitary rotations
    inside a degenerate level (sums over a level, projectors).  The within-level
    1/(lambda_j - lambda_i) terms carry only gauge information and are dropped (as in
    torchspin's ``_EighDegenerateSafe``), and the eigenvalue derivative is averaged over
    each degenerate level.  The averaging makes the derivative independent of the arbitrary
    basis LAPACK picks inside the level: it is the exact derivative of the level's mean
    energy, which is the exact derivative of each member when the degeneracy is
    symmetry-protected (Kramers, cubic), and a symmetric subgradient at an accidental
    crossing (where no derivative exists).

:func:`thermal_average`
    <A>_T = tr(exp(-beta H) A) / Z for several operators and temperatures.  This is NOT a
    gauge-invariant function of (lambda, V) in the sense above: dropping the within-level
    terms loses f'(lambda) tr_level(dH A), which is generally nonzero (see the test
    ``test_gauge_zeroing_is_wrong_for_thermal_averages``).  The derivative uses the
    Daleckii-Krein formula instead,

        d<A> = (1/Z) sum_ij f[1](lambda_i, lambda_j) dH'_ij (A'_ji - <A> delta_ji) + <dA>,
        f[1](a, b) = (f(a) - f(b)) / (a - b),   f[1](a, a) = f'(a),   f(x) = exp(-beta x),

    (primes: eigenbasis), which is smooth through every degeneracy.  This is what makes
    differentiable magnetometry (M, chi) correct.

:func:`ground_manifold`
    The projector onto the lowest ``m`` states, differentiated through the gaps between the
    manifold and the rest only (exact at the Kramers degeneracy).

:func:`sqrtm_psd` is a Newton-Schulz matrix square root, because an eigh-based square root of
g g^T gives finite but WRONG gradients at axial g.

All are ``float64``/``complex128`` throughout.  The custom functions do not support double
backward (not needed for Jacobians).
"""

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import torch
from torch.autograd.function import once_differentiable

__all__ = ["DEGENERACY_RTOL", "DEGENERACY_ATOL", "THERMAL_CUTOFF", "Manifold", "degeneracy_mask", "eigh_gauge", "eigvalsh_gauge",
           "ground_manifold", "sqrtm_psd", "thermal_average"]

DEGENERACY_RTOL = 1e-11   # relative to max |lambda| of each matrix
DEGENERACY_ATOL = 1e-9    # cm-1 floor


def _herm(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + x.conj().transpose(-2, -1))


def _h(x: torch.Tensor) -> torch.Tensor:
    return x.conj().transpose(-2, -1)


def degeneracy_mask(L: torch.Tensor, rtol: float = DEGENERACY_RTOL, atol: float = DEGENERACY_ATOL) -> torch.Tensor:
    """Boolean (..., n, n): True where |lambda_i - lambda_j| is within the degeneracy tolerance."""
    scale = L.abs().amax(dim=-1, keepdim=True).unsqueeze(-1)
    d = (L.unsqueeze(-1) - L.unsqueeze(-2)).abs()
    return d <= rtol * scale + atol


def _level_mean(x: torch.Tensor, deg: torch.Tensor) -> torch.Tensor:
    w = deg.to(x.dtype)
    return (w @ x.unsqueeze(-1)).squeeze(-1) / w.sum(-1)


def _inv_gaps(L: torch.Tensor, deg: torch.Tensor) -> torch.Tensor:
    """F_ij = 1/(lambda_j - lambda_i) off the degenerate blocks, 0 inside them."""
    E = L.unsqueeze(-2) - L.unsqueeze(-1)
    return torch.where(deg, torch.zeros_like(E), 1.0 / torch.where(deg, torch.ones_like(E), E))


# ------------------------------------------------------------------------------ eigh_gauge
class _EighGauge(torch.autograd.Function):
    generate_vmap_rule = True

    tol = (DEGENERACY_RTOL, DEGENERACY_ATOL)    # bound per subclass (see _eigh_gauge_fn): torch.func
                                                # transforms need tensor-only inputs

    @staticmethod
    def forward(H):
        L, V = torch.linalg.eigh(H)
        return L, V                 # a plain tuple: torch.func cannot unflatten eigh's namedtuple

    @classmethod
    def setup_context(cls, ctx, inputs, output):
        L, V = output
        ctx.save_for_backward(L, V)
        ctx.save_for_forward(L, V)
        ctx.tol = cls.tol

    @staticmethod
    @once_differentiable
    def backward(ctx, gL, gV):
        L, V = ctx.saved_tensors
        deg = degeneracy_mask(L, *ctx.tol)
        Vh = _h(V)
        inner = torch.zeros_like(V)
        if gL is not None:
            inner = inner + torch.diag_embed(_level_mean(gL, deg).to(V.dtype))
        if gV is not None:
            inner = inner + _inv_gaps(L, deg).to(V.dtype) * (Vh @ gV)
        return _herm(V @ inner @ Vh)

    @staticmethod
    def jvp(ctx, dH):
        L, V = ctx.saved_tensors
        deg = degeneracy_mask(L, *ctx.tol)
        dHp = _h(V) @ _herm(dH) @ V
        dL = _level_mean(torch.diagonal(dHp, dim1=-2, dim2=-1).real, deg)
        # dV_j = sum_i V_i dH'_ij / (lambda_j - lambda_i), within-level terms dropped
        dV = V @ (_inv_gaps(L, deg).to(V.dtype) * dHp)
        return dL, dV


def eigh_gauge(H: torch.Tensor, rtol: float = DEGENERACY_RTOL, atol: float = DEGENERACY_ATOL):
    """(lambda, V) of Hermitian H (..., n, n) with the degenerate-safe derivative described above."""
    return _eigh_gauge_fn(float(rtol), float(atol)).apply(H)


@lru_cache(maxsize=None)
def _eigh_gauge_fn(rtol: float, atol: float):
    if (rtol, atol) == _EighGauge.tol:
        return _EighGauge
    return type("_EighGaugeTol", (_EighGauge,), {"tol": (rtol, atol)})


def eigvalsh_gauge(H: torch.Tensor, rtol: float = DEGENERACY_RTOL, atol: float = DEGENERACY_ATOL) -> torch.Tensor:
    """Eigenvalues only; the derivative of a degenerate eigenvalue is that of its level's mean."""
    return eigh_gauge(H, rtol, atol)[0]


# ------------------------------------------------------------------------------ thermal averages
THERMAL_CUTOFF = 50.0   # states above this many k_B T (highest T) are treated as unpopulated:
                        # their mutual terms enter the derivative with weight < exp(-50) = 2e-22


def _n_populated(L: torch.Tensor, beta: torch.Tensor) -> int:
    """Number of lowest states (over the whole batch) within THERMAL_CUTOFF k_B T of the ground state."""
    n = L.shape[-1]
    try:
        x = (L - L[..., :1]) * beta.min()
        return max(1, int((x < THERMAL_CUTOFF).sum(-1).max()))
    except RuntimeError:          # inside a vmap over L itself: no data-dependent shapes
        return n


def _divided_difference(L: torch.Tensor, beta: torch.Tensor, rows: int | None = None) -> torch.Tensor:
    """f[1](lambda_i, lambda_j) for f = exp(-beta (x - lambda_0)), shape (..., nT, rows, n).

    Written as f(lo) * expm1(-beta d)/d with lo = min(lambda_i, lambda_j), d = |lambda_i - lambda_j|
    so that it neither overflows nor loses precision as d -> 0 (limit -beta f(lo)).  Only
    the first ``rows`` rows are formed (f[1] is symmetric; rows of unpopulated states vanish).
    """
    L0 = L - L[..., :1]
    r = L0.shape[-1] if rows is None else rows
    li, lj = L0[..., :r].unsqueeze(-1), L0.unsqueeze(-2)
    lo = torch.minimum(li, lj).unsqueeze(-3)                      # (..., 1, r, n)
    d = (li - lj).abs().unsqueeze(-3)
    b = beta.view(*([1] * (L.dim() - 1)), -1, 1, 1)               # (..., nT, 1, 1)
    flo = torch.exp(-b * lo)
    x = b * d
    small = x < 1e-12
    safe_d = torch.where(small, torch.ones_like(d), d)
    return torch.where(small, -b * flo, flo * torch.expm1(-x) / safe_d)


def _populations(L: torch.Tensor, beta: torch.Tensor):
    L0 = L - L[..., :1]
    f = torch.exp(-beta.view(*([1] * (L.dim() - 1)), -1, 1) * L0.unsqueeze(-2))   # (..., nT, n)
    Z = f.sum(-1, keepdim=True)
    return f / Z, Z


def _rotate(A: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """A' = V^H A V for operators A (..., k, n, n) and V (..., n, n)."""
    return _h(V).unsqueeze(-3) @ A @ V.unsqueeze(-3)


class _ThermalAverage(torch.autograd.Function):
    # Saves (L, V, A) only: the rotated operators are recomputed in the derivative, which keeps
    # memory at one eigenvector matrix per Hamiltonian (A is usually shared across fields).
    generate_vmap_rule = True

    @staticmethod
    def forward(H, A, beta):
        L, V = torch.linalg.eigh(H)
        diag = torch.diagonal(_rotate(A, V), dim1=-2, dim2=-1).real              # (..., k, n)
        p, Z = _populations(L, beta)
        out = p @ diag.transpose(-2, -1)                                         # (..., nT, k)
        return out, L, V

    @staticmethod
    def setup_context(ctx, inputs, output):
        H, A, beta = inputs
        out, L, V = output
        ctx.mark_non_differentiable(L, V)
        ctx.save_for_backward(L, V, A, beta, out)
        ctx.save_for_forward(L, V, A, beta, out)
        ctx.a_shape = A.shape

    @staticmethod
    @once_differentiable
    def backward(ctx, g, _gL, _gV):
        L, V, A, beta, out = ctx.saved_tensors
        Vh = _h(V)
        n = L.shape[-1]
        p, Z = _populations(L, beta)
        gH = gA = None
        if ctx.needs_input_grad[0]:
            Ap = _rotate(A, V)
            nl = _n_populated(L, beta)
            f1 = _divided_difference(L, beta, nl) / Z.unsqueeze(-1)              # (..., nT, nl, n)
            eye = torch.eye(n, dtype=Ap.dtype, device=Ap.device)[:nl]
            # B_t = sum_k g_tk (A'_k - <A_k>_t I);  dL/dH' = sum_t f1_t o B_t (Hermitian; the
            # block between unpopulated states vanishes, the rest follows by symmetry)
            Bt = torch.einsum("...tk,...kij->...tij", g.to(Ap.dtype), Ap[..., :nl, :]) \
                - (g * out).sum(-1)[..., None, None].to(Ap.dtype) * eye
            rows = (f1.to(Ap.dtype) * Bt).sum(-3)                                 # (..., nl, n)
            inner = torch.zeros(*rows.shape[:-2], n, n, dtype=Ap.dtype, device=Ap.device)
            inner[..., :nl, :] = rows
            inner[..., nl:, :nl] = _h(rows[..., :, nl:])
            gH = _herm(V @ inner @ Vh)
        if ctx.needs_input_grad[1]:
            # d<A_k>_t/dA_k = rho_t: gradient V diag(sum_t g_tk p_t) V^H
            w = torch.einsum("...tk,...tn->...kn", g, p).to(V.dtype)            # (..., k, n)
            gA = _reduce_to(V.unsqueeze(-3) @ (w.unsqueeze(-1) * Vh.unsqueeze(-3)), ctx.a_shape)
        return gH, gA, None

    @staticmethod
    def jvp(ctx, dH, dA, _dbeta):
        L, V, A, beta, out = ctx.saved_tensors
        p, Z = _populations(L, beta)
        dout = torch.zeros_like(out)
        if dH is not None:
            n = L.shape[-1]
            nl = _n_populated(L, beta)
            Vl = V[..., :, :nl]
            Ap = _h(Vl).unsqueeze(-3) @ A @ V.unsqueeze(-3)                       # rows < nl: (..., k, nl, n)
            dHp = _h(Vl) @ _herm(dH) @ V                                          # (..., nl, n)
            f1 = _divided_difference(L, beta, nl) / Z.unsqueeze(-1)              # (..., nT, nl, n)
            # sum over all (i, j) = rows i < nl, with the (populated, unpopulated) pairs doubled:
            # Re(dH'_ij A'_ji) is symmetric in (i, j) and f1 is symmetric
            c = torch.ones(n, dtype=L.dtype, device=L.device)
            c[nl:] = 2.0
            M = (dHp.unsqueeze(-3) * Ap.conj()).real * c                          # Re dH'_ij A'_ji, (..., k, nl, n)
            t1 = torch.einsum("...tij,...kij->...tk", f1, M)
            t2 = torch.einsum("...tii,...ii->...t", f1[..., :, :nl], dHp[..., :, :nl].real)
            dout = dout + t1 - out * t2.unsqueeze(-1)
        if dA is not None:
            dd = torch.diagonal(_rotate(dA, V), dim1=-2, dim2=-1).real           # (..., k, n)
            dout = dout + p @ dd.transpose(-2, -1)
        return dout, None, None


def _reduce_to(x: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    while x.dim() > len(shape):
        x = x.sum(0)
    for i, s in enumerate(shape):
        if s == 1 and x.shape[i] != 1:
            x = x.sum(i, keepdim=True)
    return x


def thermal_average(H: torch.Tensor, A: torch.Tensor, beta) -> torch.Tensor:
    """Boltzmann averages <A_k>_t = Re tr(exp(-beta_t H) A_k) / Z_t.

    H: Hermitian (..., n, n).  A: Hermitian operators (..., k, n, n), broadcastable against H
    (pass ``A[None]`` for one operator).  beta: 1/(k_B T) in the units of H, scalar or (nT,).
    Returns (..., nT, k).  Differentiable in H and A (reverse and forward mode), exact
    through degeneracies.
    """
    rdt = H.real.dtype if H.is_complex() else H.dtype
    beta = torch.as_tensor(beta, dtype=rdt, device=H.device).reshape(-1)
    if torch.any(beta <= 0):
        raise ValueError("beta must be > 0")
    if H.is_complex() and not A.is_complex():
        A = A.to(H.dtype)
    return _ThermalAverage.apply(H, A, beta)[0]


# ------------------------------------------------------------------------------ ground manifold
def _cross_inv_gaps(L: torch.Tensor, m: int) -> torch.Tensor:
    """K_ij = K_ji = 1/(lambda_in - lambda_out) = -1/|lambda_i - lambda_j| between the manifold
    (lowest m) and the rest, 0 inside either block."""
    n = L.shape[-1]
    inside = torch.arange(n, device=L.device) < m
    cross = inside.unsqueeze(-1) ^ inside.unsqueeze(-2)
    E = (L.unsqueeze(-1) - L.unsqueeze(-2)).abs()
    return torch.where(cross, -1.0 / torch.where(cross, E, torch.ones_like(E)), torch.zeros_like(E))


class _GroundProjector(torch.autograd.Function):
    # dP = V (K o dH') V^H with dH' = V^H dH V and K the symmetric cross-gap kernel above:
    # the manifold-internal rotations and splittings do not enter.  K is real symmetric, so
    # the map is self-adjoint and the backward is the same formula.
    generate_vmap_rule = True

    @staticmethod
    def forward(H, m):
        L, V = torch.linalg.eigh(H)
        Vm = V[..., :m]
        return Vm @ _h(Vm), L, V

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, L, V = output
        ctx.mark_non_differentiable(V)
        ctx.save_for_backward(L, V)
        ctx.save_for_forward(L, V)
        ctx.m = inputs[1]

    @staticmethod
    @once_differentiable
    def backward(ctx, gP, gL, _gV):
        L, V = ctx.saved_tensors
        Vh = _h(V)
        inner = torch.zeros_like(V)
        if gP is not None:
            inner = inner + _cross_inv_gaps(L, ctx.m).to(V.dtype) * (Vh @ _herm(gP) @ V)
        if gL is not None:
            inner = inner + torch.diag_embed(gL.to(V.dtype))
        return _herm(V @ inner @ Vh), None

    @staticmethod
    def jvp(ctx, dH, _m):
        L, V = ctx.saved_tensors
        dHp = _h(V) @ _herm(dH) @ V
        dP = V @ (_cross_inv_gaps(L, ctx.m).to(V.dtype) * dHp) @ _h(V)
        dL = torch.diagonal(dHp, dim1=-2, dim2=-1).real
        return dP, dL, None


class Manifold(NamedTuple):
    """Projector onto the lowest ``m`` states, all eigenvalues, and the gap to state m (cm-1)."""

    projector: torch.Tensor
    energies: torch.Tensor
    gap: torch.Tensor


def ground_manifold(H: torch.Tensor, m: int = 2) -> Manifold:
    """P = sum of |i><i| over the lowest ``m`` eigenstates of H (..., n, n), differentiable.

    P is a smooth function of H as long as the manifold is separated from the rest
    (``gap`` > 0); its derivative scales as 1/gap.  Rotations and splittings INSIDE the
    manifold (Kramers partner choice, a small Zeeman splitting) do not enter, so the
    derivative is exact at the Kramers degeneracy where ``torch.linalg.eigh`` returns NaN.
    Use only symmetric functions of a degenerate level's ``energies``.
    """
    n = H.shape[-1]
    if not 0 < m < n:
        raise ValueError(f"manifold size must be 1..{n - 1}")
    P, L, _ = _GroundProjector.apply(H, int(m))
    gap = (L[..., m] - L[..., m - 1]).detach()
    return Manifold(P, L, gap)


# ------------------------------------------------------------------------------ matrix square root
def sqrtm_psd(A: torch.Tensor, iters: int = 30) -> torch.Tensor:
    """Square root of symmetric positive-definite A (..., k, k) by coupled Newton-Schulz.

    Differentiable by plain autograd through the iteration (exact at repeated eigenvalues,
    unlike an eigh-based square root).  A is scaled by its Frobenius norm, which puts every
    eigenvalue in (0, 1] so that the iteration converges for any SPD input; it converges
    quadratically, and the fixed ``iters`` (no data-dependent stopping) keeps it usable
    under ``torch.func`` transforms.  For g g^T (condition number < ~2) 30 iterations reach
    machine precision with a wide margin.
    """
    k = A.shape[-1]
    eye = torch.eye(k, dtype=A.dtype, device=A.device).expand_as(A)
    norm = torch.linalg.matrix_norm(A).unsqueeze(-1).unsqueeze(-1)
    Y, Z = A / norm, eye
    for _ in range(iters):
        T = 0.5 * (3.0 * eye - Z @ Y)
        Y, Z = Y @ T, T @ Z
    return Y * torch.sqrt(norm)
