"""Multi-core helpers: batched Hermitian eigendecompositions and chunked map-reduce.

``torch.linalg.eigh`` on a batch of small matrices loops serially over the batch on
the CPU (one LAPACK call per matrix), so a 32-core node is no faster than one core.
Splitting the batch over a thread pool of single-threaded LAPACK calls scales almost
linearly (64-core workstation, 2400 × 72×72 complex: 1.5 s → 0.066 s with 32 workers).  On CUDA,
cuSOLVER's batched Jacobi solver only covers matrices up to 32×32; larger batches are
diagonalised one launch at a time and are 5–10× slower than the CPU pool, so they are
routed to the CPU and copied back.  Results are identical to the serial path (each
matrix is still handled by the same LAPACK routine).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import torch

_MIN_BATCH = 32            # below this the pool overhead is not worth it
_CUSOLVER_BATCHED_MAX = 32  # syevjBatched limit
_executors: dict = {}


def _executor(w: int) -> ThreadPoolExecutor:
    ex = _executors.get(w)
    if ex is None:
        ex = _executors[w] = ThreadPoolExecutor(w, thread_name_prefix='torchspin-eigh')
    return ex


def _pool_size(nb: int) -> int:
    return max(1, min(torch.get_num_threads(), nb // 8))


def _pooled(fn, H: torch.Tensor):
    nb = H.shape[:-2].numel()
    n = H.shape[-1]
    w = _pool_size(nb)
    if w <= 1 or nb < _MIN_BATCH:
        return fn(H)
    chunks = torch.chunk(H.reshape(nb, n, n), w)
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        res = list(_executor(w).map(fn, chunks))
    finally:
        torch.set_num_threads(prev)
    return res


class _EighDegenerateSafe(torch.autograd.Function):
    """``torch.linalg.eigh`` with a degeneracy-safe backward.

    The eigenvector gradient of a Hermitian eigendecomposition contains the
    factors 1/(λ_j − λ_i).  At exactly degenerate levels (axial or isotropic
    tensors, Kramers pairs at symmetric orientations) they are undefined and
    torch returns finite but wrong parameter gradients for directions that lift
    the degeneracy (checked: 9 % error in d spec/dA_x for an axial Cu(II) system
    at A_x = A_y, 1e-8 once A_x ≠ A_y).  Spectra are invariant under unitary
    rotations inside a degenerate subspace (the transitions share a position and
    only the summed intensity enters), so the within-subspace terms carry no
    information: they are set to zero, which is the correct gradient of any
    gauge-invariant function of (λ, V).
    """

    @staticmethod
    def forward(ctx, H):
        L, V = torch.linalg.eigh(H)
        ctx.save_for_backward(L, V)
        return L, V

    @staticmethod
    def backward(ctx, gL, gV):
        L, V = ctx.saved_tensors
        Vh = V.conj().transpose(-2, -1)
        scale = L.abs().amax(dim=-1, keepdim=True).unsqueeze(-1)
        E = L.unsqueeze(-2) - L.unsqueeze(-1)                     # E_ij = λ_j − λ_i
        F = torch.where(E.abs() > 1e-9 * (scale + 1e-300), 1.0 / torch.where(E == 0, torch.ones_like(E), E),
                        torch.zeros_like(E))
        inner = torch.zeros_like(V)
        if gL is not None:
            inner = inner + torch.diag_embed(gL.to(V.dtype))
        if gV is not None:
            inner = inner + F.to(V.dtype) * (Vh @ gV)
        gH = V @ inner @ Vh
        return 0.5 * (gH + gH.conj().transpose(-2, -1))


def eigh(H: torch.Tensor):
    """Batched ``torch.linalg.eigh`` (see module docstring); on the autograd graph
    the backward is the degeneracy-safe variant (:class:`_EighDegenerateSafe`)."""
    if H.requires_grad:
        return _EighDegenerateSafe.apply(H)
    if H.ndim < 3:
        return torch.linalg.eigh(H)
    nb = H.shape[:-2].numel()
    if H.is_cuda:
        if H.shape[-1] <= _CUSOLVER_BATCHED_MAX or nb < 4:
            return torch.linalg.eigh(H)
        E, V = eigh(H.cpu())
        return E.to(H.device), V.to(H.device)
    res = _pooled(torch.linalg.eigh, H)
    if isinstance(res, tuple):
        return res
    E = torch.cat([r[0] for r in res]).reshape(*H.shape[:-1])
    V = torch.cat([r[1] for r in res]).reshape(H.shape)
    return E, V


def eigvalsh(H: torch.Tensor):
    """Batched ``torch.linalg.eigvalsh`` (see module docstring)."""
    if H.ndim < 3 or H.requires_grad:
        return torch.linalg.eigvalsh(H)
    nb = H.shape[:-2].numel()
    if H.is_cuda:
        if H.shape[-1] <= _CUSOLVER_BATCHED_MAX or nb < 4:
            return torch.linalg.eigvalsh(H)
        return eigvalsh(H.cpu()).to(H.device)
    res = _pooled(torch.linalg.eigvalsh, H)
    if isinstance(res, torch.Tensor):
        return res
    return torch.cat(res).reshape(*H.shape[:-1])


def pool_map(fn, items: list):
    """Apply ``fn`` to every item in a thread pool of single-threaded torch calls.

    For chunked CPU work whose kernels do not parallelise well (scatter/bincount
    accumulations): each worker runs with one intra-op thread so the chunks run
    concurrently instead of contending.  Falls back to a plain map for one item
    or one thread.
    """
    w = min(torch.get_num_threads(), len(items))
    if w <= 1:
        return [fn(it) for it in items]
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return list(_executor(w).map(fn, items))
    finally:
        torch.set_num_threads(prev)
