"""Closed-form cubic roots in the resonance search (perf train P2a, A3).

``resfields_batch._cubic_real_roots`` replaces the companion-matrix ``eigvals``
call.  It must report the same set of real roots (same count per cubic, so no
resonance appears or disappears) and the same values to roundoff, including
near double and triple roots where the closed form hands rows back to the
companion path."""
import numpy as np
import pytest
import torch

from torchspin.resfields_batch import _companion_roots, _cubic_real_roots


def _normalized(C3, C2, C1, C0):
    sc = torch.stack([C3.abs(), C2.abs(), C1.abs(), C0.abs()], 1).amax(1)
    C = [c / sc for c in (C3, C2, C1, C0)]
    keep = C[0].abs() > 1e-10                       # the search's cubic branch
    return [c[keep] for c in C]


def _cases(K=20_000, seed=0):
    g = torch.Generator().manual_seed(seed)
    rnd = lambda: torch.randn(K, generator=g, dtype=torch.float64)
    fa, fb, da, db = rnd(), rnd(), rnd(), rnd()
    yield 'hermite', (2 * (fa - fb) + da + db, 3 * (fb - fa) - 2 * da - db, da, fa)
    r = torch.rand(K, generator=g, dtype=torch.float64)
    s = torch.rand(K, generator=g, dtype=torch.float64)
    eps = (10.0 ** -torch.randint(4, 16, (K,), generator=g).double()) * torch.sign(rnd())
    one = torch.ones(K, dtype=torch.float64)
    yield 'near_double', (one, -(2 * r + s), r * r + 2 * r * s, -r * r * s + eps)
    yield 'near_triple', (one, -3 * r, 3 * r * r, -r ** 3 + eps)
    yield 'exact_double', (one, -(2 * r + s), r * r + 2 * r * s, -r * r * s)


def _sorted(R):
    return torch.sort(torch.nan_to_num(R, nan=np.inf), dim=1).values


@pytest.mark.parametrize('name,coeffs', list(_cases()), ids=lambda x: x if isinstance(x, str) else '')
def test_same_real_roots_as_companion(name, coeffs):
    C = _normalized(*coeffs)
    ref, new = _sorted(_companion_roots(*C)), _sorted(_cubic_real_roots(*C))
    assert torch.equal(torch.isinf(ref), torch.isinf(new))          # same roots exist
    fin = torch.isfinite(ref)
    tol = 1e-11 if name == 'hermite' else 1e-7                    # companion error ~ sqrt(eps) at multiple roots
    assert (ref[fin] - new[fin]).abs().max() <= tol


def test_three_real_and_one_real_known_roots():
    roots = torch.tensor([[0.1, 0.4, 0.9], [-2.0, 0.25, 3.0]], dtype=torch.float64)
    C3 = torch.ones(2, dtype=torch.float64)
    C2 = -roots.sum(1)
    C1 = roots[:, 0] * roots[:, 1] + roots[:, 0] * roots[:, 2] + roots[:, 1] * roots[:, 2]
    C0 = -roots.prod(1)
    R = _sorted(_cubic_real_roots(C3, C2, C1, C0))
    assert torch.allclose(R, roots, atol=1e-14, rtol=0)
    # t^3 + t + 1: one real root, a complex pair
    one = torch.ones(1, dtype=torch.float64)
    R = _cubic_real_roots(one, 0 * one, one, one)
    assert torch.isnan(R).sum() == 2
    t = R[~torch.isnan(R)]
    assert abs(float(t ** 3 + t + 1)) < 1e-15


def test_empty_batch():
    e = torch.zeros(0, dtype=torch.float64)
    assert _cubic_real_roots(e, e, e, e).shape == (0, 3)


@pytest.mark.parametrize('name,coeffs', list(_cases(K=2000, seed=1)), ids=lambda x: x if isinstance(x, str) else '')
def test_gradients_match_companion(name, coeffs):
    """The implicit root derivative equals eigvals' derivative; no NaN leaks from
    complex or masked roots into the coefficient gradients."""
    C = [c.clone().requires_grad_(True) for c in _normalized(*coeffs)]
    w = torch.linspace(0.5, 1.5, 3, dtype=torch.float64)

    def grads(fn):
        R = fn(*C)
        R = torch.where(torch.isnan(R), torch.zeros_like(R), R)
        ordered = torch.sort(R.detach(), dim=1).indices
        loss = (torch.gather(R, 1, ordered) * w).sum()
        return torch.autograd.grad(loss, C), R.detach()

    try:
        g_new, R_new = grads(_cubic_real_roots)
    except torch.linalg.LinAlgError:
        # Rows at a multiple root go through the companion eigvals (as every row did up to
        # 0.9.38).  The root has no derivative there; with the LAPACK builds on Linux and
        # Windows eigvals' backward pass says so (singular solve), on macOS it returns finite
        # numbers.  An error is fine, a silent NaN is not; only the built double roots get here.
        assert name == 'exact_double'
        return
    for b in g_new:
        assert torch.isfinite(b).all()
    if name == 'hermite':                       # well-separated roots: same derivative to roundoff
        g_ref, R_ref = grads(_companion_roots)
        for a, b in zip(g_ref, g_new):
            assert torch.allclose(a, b, rtol=1e-8, atol=1e-8 * a.abs().max().item())
