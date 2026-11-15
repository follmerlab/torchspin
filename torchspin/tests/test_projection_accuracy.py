"""Physical accuracy of the projected powder spectrum and its gradients against converged
references (B85, 0.9.35; scratch comparison of the hat kernel, the centered box and 0.9.34).

Converged reference: the same system on a 32x finer field axis (8x for the 4096-point
isotropic case), GridSize [181 1] without interpolation (the O3 template for the isotropic
case), compared at the coarse points x_k (point values of the converged spectrum).
Bounds are the measured values x 1.05; 0.9.34's values are in the docstrings (all fail).
"""
import numpy as np
import pytest
import torch

from torchspin import Experiment, Options, SpinSystem, pepper
from torchspin.pepper_autograd import pepper_autograd

F64 = torch.float64
G0 = [1.98, 2.05, 2.12]


def _exp(n, H=1):
    return Experiment(mwFreq=9.5, Range=[310.0, 350.0], nPoints=n, Harmonic=H)


def _w(x):
    return np.cos((x - 310) / 3.1) * np.exp(-((x - 330) / 12) ** 2)


def _converged(lw=1.0, gy=G0[1], F=32, n=512):
    x, y = pepper(SpinSystem(S=0.5, g=[G0[0], gy, G0[2]], lw=[lw, 0.0]), _exp(F * (n - 1) + 1),
                  Options(GridSize=[181, 1]))
    return x.numpy()[::F], y.numpy()[::F]


def test_rhombic_narrow_line_vs_converged():
    """Rhombic S = 1/2, 512 points over 40 mT, lw 0.3 mT, first harmonic, GridSize [91 1].
    Measured: max error 3.73e-2 of the maximum, 1 - cos 2.90e-4 (0.9.34: 0.321, 2.20e-2;
    centered box, dev/b85-box: 3.12e-2, 2.48e-4)."""
    xr, r = _converged(lw=0.3)
    x, y = pepper(SpinSystem(S=0.5, g=G0, lw=0.3), _exp(512), Options(GridSize=[91, 1]))
    y = y.numpy()
    assert np.allclose(x.numpy(), xr, atol=1e-9)
    m = slice(2, -2)
    err = np.abs(y[m] - r[m]).max() / np.abs(r[m]).max()
    cs = 1 - y[m] @ r[m] / np.linalg.norm(y[m]) / np.linalg.norm(r[m])
    print(f'max error {err:.3e}, 1 - cos {cs:.3e}')
    assert err < 3.91e-2 and cs < 3.04e-4


@pytest.fixture(scope='module')
def coarse_grads():
    lw = torch.tensor(1.0, dtype=F64, requires_grad=True)
    g = torch.tensor(G0, dtype=F64, requires_grad=True)
    x, y = pepper_autograd(SpinSystem(S=0.5, g=g, lw=torch.stack([lw, torch.zeros((), dtype=F64)])),
                           _exp(512), Options(GridSize=[91, 1]))
    J = (torch.from_numpy(_w(x.numpy())) * y).sum()
    gl, gg = torch.autograd.grad(J, (lw, g))
    return float(gl), float(gg[1])


def _J(**kw):
    x, y = _converged(**kw)
    return float((_w(x) * y).sum())


def test_lw_gradient_vs_converged(coarse_grads):
    """d/d lw of sum_k w(x_k) y_k (lw 1 mT, 512 points, first harmonic, GridSize [91 1]) vs the
    converged value -0.616864 (central differences of the converged spectrum, steps 1e-3 and
    3e-4 agree to 2e-8).  Measured relative error 1.10e-3 (0.9.34: 5.2e-2; box: 9.8e-4)."""
    h = 1e-3
    ref = (_J(lw=1 + h) - _J(lw=1 - h)) / (2 * h)
    err = abs(coarse_grads[0] / ref - 1)
    print(f'd/dlw {coarse_grads[0]:.6f}, converged {ref:.6f}, rel {err:.2e}')
    assert err < 1.15e-3


def test_g_gradient_vs_converged(coarse_grads):
    """d/d g_y of the same sum vs the converged value -14833.88 (steps 1e-5 and 3e-6 agree to
    1.6e-8).  Measured relative error 3.3e-5 (0.9.34: 3.3e-3; centered box: 2.0e-3)."""
    h = 1e-5
    ref = (_J(gy=G0[1] + h) - _J(gy=G0[1] - h)) / (2 * h)
    err = abs(coarse_grads[1] / ref - 1)
    print(f'd/dg_y {coarse_grads[1]:.4f}, converged {ref:.4f}, rel {err:.2e}')
    assert err < 3.5e-5


def test_isotropic_direction_gradient_vs_converged_o3():
    """g = [s s s] from three leaves (auto D2h grid), 4096 points over 60 mT, lw 1 mT, first
    harmonic: d/ds of sum_k sin(x_k/2.3) y_k / 8 vs the converged O3 value (template on an 8x
    finer axis, sampled at x_k) -13801.2.  Measured -14237.0, 3.16e-2 (a point mass split
    linearly: chord slope, first order in dx); 0.9.34: +1001.8 (the truncated sub-bin deposit
    had no position dependence)."""
    n, F = 4096, 8
    L = [torch.tensor(2.0, dtype=F64, requires_grad=True) for _ in range(3)]
    e = Experiment(mwFreq=9.5, Range=[310, 370], nPoints=n, Harmonic=1)
    x, y = pepper_autograd(SpinSystem(S=0.5, g=torch.stack(L), lw=1.0), e, Options())
    wt = torch.from_numpy(np.sin(x.numpy() / 2.3) * (512 / n))
    g3 = float(sum(torch.autograd.grad((wt * y).sum(), L)))
    s = torch.tensor(2.0, dtype=F64, requires_grad=True)
    ef = Experiment(mwFreq=9.5, Range=[310, 370], nPoints=F * (n - 1) + 1, Harmonic=1)
    xf, yf = pepper_autograd(SpinSystem(S=0.5, g=s, lw=1.0), ef, Options())
    (go,) = torch.autograd.grad((wt * yf[::F]).sum(), s)
    err = abs(g3 / float(go) - 1)
    print(f'three leaves {g3:.2f}, converged O3 {float(go):.2f}, rel {err:.2e}')
    assert err < 3.32e-2
