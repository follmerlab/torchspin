"""Multi-core helpers: batched Hermitian eigendecompositions and chunked map-reduce.

Native threads (B128): torch's BLAS (OpenBLAS on many builds) has its own thread
pool that ``torch.set_num_threads`` does not control.  Every LAPACK call on a small
matrix could then start BLAS threads of its own, inside the eigh pool as well as
outside it; with all cores busy (a laptop with other work, or every core requested)
that oversubscription made a batch of 2000 72x72 eigendecompositions 30x slower
(4.0 s vs 0.13 s at 14 threads).  BLAS is therefore capped at one thread while the
pool runs, and around unpooled eigendecompositions of matrices smaller than
``_BLAS_THREADED_MIN`` (BLAS threads help only from about 128x128 on).  Matrices
below ``_MIN_POOL_DIM`` are not pooled: for them the pool costs more than it saves.
Neither changes results: every matrix is still handled by the same LAPACK routine,
and a single-threaded BLAS gives the 1-thread result at every thread count.

``torch.linalg.eigh`` on a batch of small matrices loops serially over the batch on
the CPU (one LAPACK call per matrix), so a 32-core node is no faster than one core.
Splitting the batch over a thread pool of single-threaded LAPACK calls scales almost
linearly (64-core workstation, 2400 × 72×72 complex: 1.5 s → 0.066 s with 32 workers).  On CUDA,
cuSOLVER's batched Jacobi solver only covers matrices up to 32×32; larger batches are
diagonalized one launch at a time and are 5–10× slower than the CPU pool, so they are
routed to the CPU and copied back.  Results are identical to the serial path (each
matrix is still handled by the same LAPACK routine).

LAPACK's divide-and-conquer driver (``zheevd``) occasionally fails to converge on
highly degenerate Hermitian input — a spin system with several identical nuclei
produces matrices with hundreds of exactly equal eigenvalue gaps.  The failure is
a property of the driver, not of the input (the matrix is Hermitian and finite),
so :func:`eigh` and :func:`eigvalsh` retry such matrices individually through
other routines instead of letting one bad matrix abort a whole orientation batch.
"""
from __future__ import annotations

import contextlib
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from torchspin.parallel import inner_threads

_MIN_BATCH = 32            # below this the pool overhead is not worth it
_MIN_POOL_DIM = 10         # smaller matrices: one batched call (6x6 x 5000: 11 ms vs 15 ms pooled)
_BLAS_THREADED_MIN = 128   # unpooled matrices smaller than this run with single-threaded BLAS
_CUSOLVER_BATCHED_MAX = 32  # syevjBatched limit
# Workspace the batched CUDA solver reserves per matrix. Measured on torch 2.14 / CUDA 13:
# about 1.1-1.5 MB per matrix at every size from 6x6 to 32x32, so an unchunked batch of
# 32768 6x6 matrices asked for 37 GB. Use the upper end, with headroom.
_CUDA_WS_PER_MATRIX = 2 * 2**20
_CUDA_MIN_CHUNK = 256
_executors: dict = {}

# torch.set_num_threads is process-global, so the save/set/restore in _pooled and
# pool_map must not interleave with another thread doing the same (esfit evaluates
# objective functions in a thread pool).  Concurrent callers queue instead; they
# would contend for the same cores anyway.
_threads_lock = threading.Lock()

_LinAlgError = torch._C._LinAlgError

# Relative size of the diagonal perturbation used as the last-resort retry.  Well
# below double-precision resolution of the eigenvalues, but enough to break the
# exact ties that stall the divide-and-conquer driver.
_JITTER = 1e-14


_blas_ctl: list = []


def _blas_controller():
    """Cached ``threadpoolctl.ThreadpoolController`` (None if unavailable)."""
    if not _blas_ctl:
        try:
            from threadpoolctl import ThreadpoolController
            _blas_ctl.append(ThreadpoolController())
        except Exception:          # threadpoolctl missing or unable to inspect the process
            _blas_ctl.append(None)
    return _blas_ctl[0]


_blas_lock = threading.Lock()
_blas_state = {'depth': 0, 'limiter': None}


@contextlib.contextmanager
def serial_blas():
    """Context manager: BLAS libraries of the process limited to one thread.

    The limit is process-wide, so it is reference counted: the first thread to
    enter sets it, the last one to leave restores the previous setting, and
    concurrent callers (esfit's thread backend) neither wait for each other nor
    restore it under one another.  A no-op when threadpoolctl is unavailable."""
    ctl = _blas_controller()
    if ctl is None:
        yield
        return
    with _blas_lock:
        if _blas_state['depth'] == 0:
            _blas_state['limiter'] = ctl.limit(limits=1, user_api='blas')
        _blas_state['depth'] += 1
    try:
        yield
    finally:
        with _blas_lock:
            _blas_state['depth'] -= 1
            if _blas_state['depth'] == 0:
                _blas_state['limiter'].restore_original_limits()
                _blas_state['limiter'] = None


def _executor(w: int) -> ThreadPoolExecutor:
    ex = _executors.get(w)
    if ex is None:
        ex = _executors[w] = ThreadPoolExecutor(w, thread_name_prefix='torchspin-eigh')
    return ex


def _pool_size(nb: int) -> int:
    # inner_threads() is torch.get_num_threads(), or less inside a
    # torchspin.parallel.thread_budget (e.g. in an esfit worker)
    return max(1, min(inner_threads(), nb // 8))


def _one(fn, np_fn, A: torch.Tensor, allow_numpy: bool = True):
    """Diagonalize a single Hermitian matrix, working around driver failures.

    Tried in order: the torch routine on its own (a matrix can succeed alone that
    failed inside a batch), the NumPy routine (a different LAPACK build, and
    ``zheevr``/``zheev`` rather than the divide-and-conquer driver), and finally a
    tiny deterministic diagonal perturbation that lifts the exact degeneracies.
    Spectra are invariant under unitary rotations inside a degenerate subspace, so
    none of these change the result beyond round-off.

    ``allow_numpy`` is cleared when ``A`` carries gradients, because the NumPy
    detour would silently drop them.
    """
    try:
        return fn(A)
    except _LinAlgError:
        pass
    if allow_numpy:
        try:
            out = np_fn(A.detach().cpu().numpy())
        except np.linalg.LinAlgError:
            pass
        else:
            if isinstance(out, tuple):
                return (torch.as_tensor(out[0], dtype=A.real.dtype, device=A.device),
                        torch.as_tensor(out[1], dtype=A.dtype, device=A.device))
            return torch.as_tensor(out, dtype=A.real.dtype, device=A.device)
    scale = float(A.abs().amax().detach()) or 1.0
    ramp = torch.linspace(-1.0, 1.0, A.shape[-1], dtype=A.real.dtype, device=A.device)
    return fn(A + torch.diag(ramp * (_JITTER * scale)).to(A.dtype))


def _safe(fn, np_fn):
    """Wrap a batched torch routine so a single bad matrix does not abort the batch."""
    def run(H: torch.Tensor):
        try:
            return fn(H)
        except _LinAlgError:
            pass
        allow_numpy = not H.requires_grad
        if H.ndim < 3:
            return _one(fn, np_fn, H, allow_numpy)
        flat = H.reshape(-1, H.shape[-1], H.shape[-1])
        outs = [_one(fn, np_fn, flat[i], allow_numpy) for i in range(flat.shape[0])]
        if isinstance(outs[0], tuple):
            return (torch.stack([o[0] for o in outs]).reshape(*H.shape[:-1]),
                    torch.stack([o[1] for o in outs]).reshape(H.shape))
        return torch.stack(outs).reshape(*H.shape[:-1])
    return run


_safe_eigh = _safe(torch.linalg.eigh, np.linalg.eigh)
_safe_eigvalsh = _safe(torch.linalg.eigvalsh, np.linalg.eigvalsh)


def _pooled(fn, H: torch.Tensor):
    nb = H.shape[:-2].numel()
    n = H.shape[-1]
    w = _pool_size(nb)
    if w <= 1 or nb < _MIN_BATCH or n < _MIN_POOL_DIM:
        if n < _BLAS_THREADED_MIN:
            with serial_blas():
                return fn(H)
        return fn(H)
    chunks = torch.chunk(H.reshape(nb, n, n), w)
    with _threads_lock:
        prev = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with serial_blas():
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
        L, V = _safe_eigh(H)
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


def _cuda_chunk(nb: int, device) -> int:
    """Matrices per batched CUDA eigensolver call that fit in half the free memory."""
    try:
        free, _ = torch.cuda.mem_get_info(device)
    except Exception:
        free = 4 * 2**30
    return max(_CUDA_MIN_CHUNK, min(nb, int(0.5 * free) // _CUDA_WS_PER_MATRIX))


def _cuda_chunked(fn, H: torch.Tensor):
    """Run a batched CUDA eigensolver in memory-bounded chunks (see _CUDA_WS_PER_MATRIX)."""
    n = H.shape[-1]
    nb = H.shape[:-2].numel()
    chunk = _cuda_chunk(nb, H.device)
    if chunk >= nb:
        return fn(H)
    flat = H.reshape(nb, n, n)
    parts = [fn(flat[i:i + chunk]) for i in range(0, nb, chunk)]
    if isinstance(parts[0], tuple):
        return (torch.cat([p[0] for p in parts]).reshape(*H.shape[:-1]),
                torch.cat([p[1] for p in parts]).reshape(H.shape))
    return torch.cat(parts).reshape(*H.shape[:-1])


def eigh(H: torch.Tensor):
    """Batched ``torch.linalg.eigh`` (see module docstring); on the autograd graph
    the backward is the degeneracy-safe variant (:class:`_EighDegenerateSafe`)."""
    if H.requires_grad:
        return _EighDegenerateSafe.apply(H)
    if H.ndim < 3:
        return _safe_eigh(H)
    nb = H.shape[:-2].numel()
    if H.is_cuda:
        if H.shape[-1] <= _CUSOLVER_BATCHED_MAX or nb < 4:
            return _cuda_chunked(_safe_eigh, H)
        E, V = eigh(H.cpu())
        return E.to(H.device), V.to(H.device)
    res = _pooled(_safe_eigh, H)
    if isinstance(res, tuple):
        return res
    E = torch.cat([r[0] for r in res]).reshape(*H.shape[:-1])
    V = torch.cat([r[1] for r in res]).reshape(H.shape)
    return E, V


def eigvalsh(H: torch.Tensor):
    """Batched ``torch.linalg.eigvalsh`` (see module docstring)."""
    if H.ndim < 3 or H.requires_grad:
        return _safe_eigvalsh(H)
    nb = H.shape[:-2].numel()
    if H.is_cuda:
        if H.shape[-1] <= _CUSOLVER_BATCHED_MAX or nb < 4:
            return _cuda_chunked(_safe_eigvalsh, H)
        return eigvalsh(H.cpu()).to(H.device)
    res = _pooled(_safe_eigvalsh, H)
    if isinstance(res, torch.Tensor):
        return res
    return torch.cat(res).reshape(*H.shape[:-1])


@contextlib.contextmanager
def single_threaded():
    """Run the enclosed code with one intra-op torch thread and single-threaded BLAS.

    For stretches of many small or mid-sized tensor operations (pepper's slot
    interpolation and projection), where every parallel region costs a thread
    wake-up that exceeds the work it splits (B128).  Holds ``_threads_lock`` like
    :func:`pool_map`, so pools started inside run serially."""
    with _threads_lock:
        prev = torch.get_num_threads()
        if prev > 1:
            torch.set_num_threads(1)
        try:
            with serial_blas():
                yield
        finally:
            if prev > 1:
                torch.set_num_threads(prev)


def pool_map(fn, items: list):
    """Apply ``fn`` to every item in a thread pool of single-threaded torch calls.

    For chunked CPU work whose kernels do not parallelise well (scatter/bincount
    accumulations): each worker runs with one intra-op thread so the chunks run
    concurrently instead of contending.  Falls back to a plain map for one item
    or one thread.
    """
    w = min(inner_threads(), len(items))
    if w <= 1:
        return [fn(it) for it in items]
    with _threads_lock:
        prev = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with serial_blas():
                return list(_executor(w).map(fn, items))
        finally:
            torch.set_num_threads(prev)
