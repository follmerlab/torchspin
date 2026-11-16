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
