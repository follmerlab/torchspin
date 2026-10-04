"""torchspin._linalg: the thread-pooled batched eigh/eigvalsh must equal the serial
torch call (same LAPACK routine per matrix) for every shape it dispatches on."""
import pytest
import torch

from torchspin import _linalg


@pytest.mark.parametrize('shape', [(40, 6, 6), (5, 9, 36, 36), (70, 72, 72), (3, 8, 8), (12, 12)])
def test_pooled_eigh_matches_serial(shape):
    torch.manual_seed(0)
    A = torch.randn(*shape, dtype=torch.complex128)
    H = A + A.conj().transpose(-1, -2)
    E0, V0 = torch.linalg.eigh(H)
    prev = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        E1, V1 = _linalg.eigh(H)
        w1 = _linalg.eigvalsh(H)
    finally:
        torch.set_num_threads(prev)
    assert E1.shape == E0.shape and V1.shape == V0.shape
    assert torch.allclose(E1, E0, atol=1e-12, rtol=1e-12)
    assert torch.allclose(w1, torch.linalg.eigvalsh(H), atol=1e-12, rtol=1e-12)
    # eigenvectors agree up to a phase per column
    ph = (V0.conj() * V1).sum(dim=-2)
    assert torch.allclose(ph.abs(), torch.ones_like(ph.abs()), atol=1e-10)
    assert torch.get_num_threads() == prev


def test_pooled_eigh_keeps_autograd_path():
    A = torch.randn(64, 4, 4, dtype=torch.float64, requires_grad=True)
    H = A + A.transpose(-1, -2)
    E, V = _linalg.eigh(H)
    E.sum().backward()
    assert A.grad is not None and torch.isfinite(A.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_cuda_large_matrices_routed_to_cpu():
    A = torch.randn(40, 48, 48, dtype=torch.complex128, device='cuda')
    H = A + A.conj().transpose(-1, -2)
    E, V = _linalg.eigh(H)
    assert E.is_cuda and V.is_cuda
    E0 = torch.linalg.eigvalsh(H)
    assert torch.allclose(E, E0, atol=1e-9, rtol=1e-9)
