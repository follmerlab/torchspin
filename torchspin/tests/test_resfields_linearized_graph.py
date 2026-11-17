"""B14: on the autograd graph, ``resfields_batch`` diagonalizes every candidate field once
without the graph (pooled, like the forward-only path) and attaches the graph through
first-order perturbation theory: energies by Hellmann-Feynman, the two transition
eigenvectors by the sum over states with ``_linalg``'s degeneracy rule, and the resonance
field by the implicit derivative dB/dθ = -(∂f/∂θ)/(∂f/∂B).  Before, every knot and candidate
``eigh`` was on the graph and the backward pass replayed them all (Cu + 2 14N, [10 4]:
backward 11.8 s -> 2.3 s, peak memory 8.3 -> 5.5 GB).

Values are those of the detached diagonalization: resonance fields with and without grad
are identical bit for bit; intensities differ by <= 1e-18 relative and spectra by ~1e-16, as
they did before this change (intensity and line-shape stages); derivatives are checked
against central differences."""
import pytest
import torch

import torchspin._linalg as _linalg
from torchspin import Experiment, Options, SpinSystem, pepper
from torchspin.ham import ham
from torchspin.pepper_autograd import pepper_autograd
from torchspin.resfields import resfields

DT = torch.float64


def T(*a, grad=False):
    return torch.tensor(a, dtype=DT, requires_grad=grad)


SPECTRA = {
    'cu_tilted_A': (lambda p: SpinSystem(S=[0.5], g=[p['g']], Nucs='63Cu', A=[[40.0, 50.0, 550.0]],
                                         AFrame=[[0.0, 0.4, 0.0]], lw=[1.0, 0.0]),
                    dict(g=[2.05, 2.06, 2.25]), dict(mwFreq=9.5, Range=[250, 360]), 'C2h'),
    'triplet_4K': (lambda p: SpinSystem(S=[1.0], g=[[2.0, 2.0, 2.0]], D=[p['D']], lw=[1.5, 0.0]),
                   dict(D=[900.0, 150.0]), dict(mwFreq=9.5, Range=[250, 430], Temperature=4.0), 'D2h'),
    'nitroxide_strain': (lambda p: SpinSystem(S=[0.5], g=[p['g']], Nucs='14N', A=[[20.0, 20.0, 85.0]],
                                              HStrain=[10.0, 12.0, 30.0], lw=[0.2, 0.0]),
                         dict(g=[2.008, 2.006, 2.003]), dict(mwFreq=9.5, Range=[330, 350]), 'D2h'),
}


@pytest.mark.parametrize('name', list(SPECTRA))
def test_spectrum_with_grad_identical_to_without(name):
    build, p0, ekw, sym = SPECTRA[name]
    exp = Experiment(nPoints=512, Harmonic=1, **ekw)
    opt = Options(Verbosity=0, GridSize=[19, 4], GridSymmetry=sym)
    leaves = {k: T(*v, grad=True) for k, v in p0.items()}
    _, y_graph = pepper_autograd(build(leaves), exp, opt)
    _, y_plain = pepper_autograd(build({k: T(*v) for k, v in p0.items()}), exp, opt)
    assert y_graph.requires_grad and not y_plain.requires_grad
    assert float((y_graph.detach() - y_plain).abs().max()) <= 1e-14 * float(y_plain.abs().max())


@pytest.mark.parametrize('name', list(SPECTRA))
def test_lines_with_grad_identical_to_without(name):
    build, p0, ekw, _ = SPECTRA[name]
    exp = Experiment(**ekw)
    for phi, theta in ((0.4, 0.9), (1.2, 0.3), (0.0, 1.5)):
        on = _lines(build({k: T(*v, grad=True) for k, v in p0.items()}), phi, theta, exp)
        off = _lines(build({k: T(*v) for k, v in p0.items()}), phi, theta, exp)
        assert on[0].requires_grad and on[0].numel() > 0
        assert torch.equal(on[0].detach(), off[0])                       # fields: bit for bit
        for a, b in zip(on[1:], off[1:]):
            if a is not None:
                assert float((a.detach() - b).abs().max()) <= 1e-15 * float(b.abs().max())


def test_no_eigendecomposition_on_the_graph(monkeypatch):
    """The gradient run must not put an eigh on the graph (its backward is O(n^3) per
    candidate and kept every knot alive)."""
    def _fail(*a, **k):
        raise AssertionError('eigh on the autograd graph')
    monkeypatch.setattr(_linalg._EighDegenerateSafe, 'apply', _fail)
    g = T(2.04, 2.08, 2.25, grad=True)
    A = T(30.0, 30.0, 617.3, grad=True)
    sys = SpinSystem(S=[0.5], g=[g], Nucs='63Cu', A=A.view(1, 3), lw=[1.0, 0.0])
    _, y = pepper_autograd(sys, Experiment(mwFreq=9.5, Range=[260, 360], nPoints=256, Harmonic=1),
                           Options(Verbosity=0, GridSize=[10, 4]))
    gg, gA = torch.autograd.grad((y * torch.linspace(-1, 1, 256, dtype=DT)).sum(), [g, A])
    assert torch.isfinite(gg).all() and torch.isfinite(gA).all() and gg.abs().max() > 0


def _lines(sd, phi, theta, exp):
    H0, mux, muy, muz = ham(sd)
    return resfields(H0, mux, muy, muz, phi, theta, exp, Options(Verbosity=0, Threshold=0.0), sys=sd)


def _fd4(f, x0, h):
    return (-f(x0 + 2 * h) + 8 * f(x0 + h) - 8 * f(x0 - h) + f(x0 - 2 * h)) / (12 * h)


# name -> (builder(x) -> SpinSystem for a scalar parameter x, x0, h, Experiment kwargs)
LINES = {
    # Boltzmann at 4 K: every level energy carries the graph (D)
    'triplet_4K_D': (lambda x: SpinSystem(S=[1.0], g=[[2.0, 2.0, 2.0]], D=torch.stack([x, x * 0 + 150.0]).view(1, 2)),
                     900.0, 1e-2, dict(mwFreq=9.5, Range=[150, 500], Temperature=4.0)),
    # complex eigenvectors (tilted A), four sublevels per manifold
    'cu_tilted_gz': (lambda x: SpinSystem(S=[0.5], g=torch.stack([x * 0 + 2.05, x * 0 + 2.06, x]).view(1, 3),
                                          Nucs='63Cu', A=[[40.0, 50.0, 550.0]], AFrame=[[0.0, 0.4, 0.0]]),
                     2.25, 1e-5, dict(mwFreq=9.5, Range=[200, 400])),
    # nearly degenerate sublevels: two equivalent 14N
    'cu_2n_Az': (lambda x: SpinSystem(S=[0.5], g=[[2.05, 2.06, 2.25]], Nucs='63Cu,14N,14N',
                                      A=torch.stack([torch.stack([x * 0 + 40.0, x * 0 + 40.0, x]),
                                                     T(40.0, 40.0, 35.0), T(40.0, 40.0, 35.0)])),
                 550.0, 1e-2, dict(mwFreq=9.5, Range=[230, 380])),
}


@pytest.mark.parametrize('name', list(LINES))
def test_line_positions_and_intensities_vs_central_differences(name):
    build, x0, h, ekw = LINES[name]
    exp = Experiment(**ekw)
    phi, theta = 0.4, 0.9
    x = torch.tensor(x0, dtype=DT, requires_grad=True)
    B, I, _ = _lines(build(x), phi, theta, exp)
    assert B.numel() > 0
    w = torch.linspace(0.5, 1.5, B.numel(), dtype=DT)
    dB, = torch.autograd.grad((w * B).sum(), x, retain_graph=True)
    dI, = torch.autograd.grad((w * I).sum(), x)

    def f(xv, which):
        with torch.no_grad():
            Bv, Iv, _ = _lines(build(torch.tensor(xv, dtype=DT)), phi, theta, exp)
        assert Bv.numel() == B.numel()
        return float((w * (Bv if which == 'B' else Iv)).sum())
    fdB = _fd4(lambda v: f(v, 'B'), x0, h)
    fdI = _fd4(lambda v: f(v, 'I'), x0, h)
    assert abs(float(dB) - fdB) <= 1e-6 * max(abs(fdB), 1e-12), (float(dB), fdB)
    assert abs(float(dI) - fdI) <= 1e-5 * max(abs(fdI), float((w * I.detach()).abs().sum()) * 1e-6), (float(dI), fdI)


def test_orientation_derivative_vs_central_differences():
    """theta itself on the graph (muzL depends on it)."""
    sd = SpinSystem(S=[0.5], g=[[2.04, 2.08, 2.25]], Nucs='63Cu', A=[[30.0, 30.0, 617.3]])
    exp = Experiment(mwFreq=9.5, Range=[250, 380])
    th = torch.tensor(0.9, dtype=DT, requires_grad=True)
    B, I, _ = _lines(sd, 0.4, th, exp)
    w = torch.linspace(0.5, 1.5, B.numel(), dtype=DT)
    dB, = torch.autograd.grad((w * B).sum(), th, retain_graph=True)
    dI, = torch.autograd.grad((w * I).sum(), th)

    def f(t, which):
        Bv, Iv, _ = _lines(sd, 0.4, t, exp)
        return float((w * (Bv if which == 'B' else Iv)).sum())
    fdB = _fd4(lambda t: f(t, 'B'), 0.9, 1e-4)
    fdI = _fd4(lambda t: f(t, 'I'), 0.9, 1e-4)
    assert abs(float(dB) - fdB) <= 1e-6 * abs(fdB)
    assert abs(float(dI) - fdI) <= 1e-5 * abs(fdI)


def test_spectrum_gradient_vs_central_differences_of_pepper_cu_2n():
    """Larger Hilbert space (dim 72) with near-degenerate nuclear sublevels."""
    exp = Experiment(mwFreq=9.5, Range=[260, 360], nPoints=512, Harmonic=1)
    opt = Options(Verbosity=0, GridSize=[7, 4], GridSymmetry='D2h')
    w = torch.linspace(-1, 1, 512, dtype=DT) ** 3
    A = T(40.0, 40.0, 550.0, grad=True)

    def build(a):
        return SpinSystem(S=[0.5], g=[[2.05, 2.06, 2.25]], Nucs='63Cu,14N',
                          A=torch.stack([a, T(40.0, 40.0, 35.0)]), lw=[1.0, 0.0])
    _, y = pepper_autograd(build(A), exp, opt)
    g, = torch.autograd.grad((w * y).sum(), A)
    for i, h in ((0, 1e-3), (2, 1e-3)):
        e = torch.zeros(3, dtype=DT); e[i] = h
        fd = (float((w * pepper(build(T(40.0, 40.0, 550.0) + e), exp, opt)[1]).sum())
              - float((w * pepper(build(T(40.0, 40.0, 550.0) - e), exp, opt)[1]).sum())) / (2 * h)
        assert abs(float(g[i]) - fd) <= 1e-5 * abs(fd), (i, float(g[i]), fd)
