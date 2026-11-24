"""pepper_autograd._interp_matrix (perf train P1, B1): for G3 positions on the
triangular closed-phi grids the cached matrix is _d2h_interp_matrix itself; it
must equal the matrix built column by column from the numpy interpolation (the
0.9.38 construction) exactly."""
import sys

import numpy as np
import pytest

import torchspin.pepper_autograd  # noqa: F401
from torchspin.sphgrid import sphgrid

_pa = sys.modules['torchspin.pepper_autograd']


def _by_columns(sym, n_c_knots, n_f_knots, mode, linear):
    n_c = sphgrid(sym, n_c_knots)[0].numel()
    n_f = sphgrid(sym, n_f_knots)[0].numel()
    M = np.zeros((n_f, n_c))
    eye = np.eye(n_c)
    for j in range(n_c):
        M[:, j] = _pa._numpy_interp(eye[:, j], sym, n_c_knots, n_f_knots, mode, linear)
    return M


@pytest.mark.parametrize('sym,nc,nf', [('D2h', 19, 73), ('D2h', 7, 25), ('D4h', 10, 37), ('Oh', 10, 28),
                                       ('D3d', 10, 37), ('Th', 6, 16)])
@pytest.mark.parametrize('mode,linear', [('pos', False), ('val', True), ('pos', True)])
def test_interp_matrix_equals_column_build(sym, nc, nf, mode, linear):
    _pa._interp_matrix.cache_clear()
    got = _pa._interp_matrix(sym, nc, nf, mode, linear)
    assert np.array_equal(got, _by_columns(sym, nc, nf, mode, linear))


@pytest.mark.parametrize('sym,nc,nf', [('Ci', 7, 25), ('Ci', 10, 37), ('C2h', 9, 33), ('C1', 7, 19),
                                       ('C4h', 8, 29), ('S6', 7, 25), ('C6h', 7, 19)])
@pytest.mark.parametrize('mode,linear', [('pos', False), ('val', True), ('pos', True)])
def test_open_phi_separable_operators(sym, nc, nf, mode, linear):
    """Open-phi grids are interpolated with separable operators (perf train P2a,
    B1b) instead of the dense column-built matrix: same values to roundoff, same
    NaN pattern for slots with missing knots, gradients flow."""
    import torch
    rng = np.random.default_rng(nc)
    n_c = sphgrid(sym, nc)[0].numel()
    V = torch.as_tensor(rng.standard_normal((3, n_c)) + 300.0)
    ref = V.numpy() @ _by_columns(sym, nc, nf, mode, linear).T
    got = _pa._interp_rows_t(V, sym, nc, nf, mode, linear)
    assert np.abs(got.numpy() - ref).max() <= 1e-13 * np.abs(ref).max()
    Vn = V.clone(); Vn[1, 3] = float('nan')
    gotn = _pa._interp_rows_t(Vn, sym, nc, nf, mode, True).numpy()
    refn = np.stack([_pa._numpy_interp(v, sym, nc, nf, mode, True) for v in Vn.numpy()])
    assert np.array_equal(np.isnan(gotn), np.isnan(refn))
    ok = ~np.isnan(refn)
    assert np.abs(gotn[ok] - refn[ok]).max() <= 1e-13 * np.abs(refn[ok]).max()
    Vg = V.clone().requires_grad_(True)
    _pa._interp_rows_t(Vg, sym, nc, nf, mode, linear).sum().backward()
    assert torch.isfinite(Vg.grad).all()
