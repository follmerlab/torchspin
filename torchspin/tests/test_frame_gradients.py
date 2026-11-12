"""Euler frame angles on the autograd graph (B3, 0.9.17).

Before 0.9.17 every tensor frame (``gFrame``, ``AFrame``, ``DFrame``,
``QFrame``, ``eeFrame``, ``nnFrame``, ``sigmaFrame``, ``BFrame``) was read with
``.tolist()`` or ``float()`` before ``erot``, so a frame tensor with
``requires_grad=True`` got no gradient and no warning.  ``erot`` now builds the
matrix from torch operations when an angle requires grad, and every call site
passes the tensor; the detached values only decide whether to rotate.

Gradients are checked against central differences of the same function
(fixed grid symmetry for powders: the automatic grid is a discrete decision and
may change under the finite-difference step at symmetric points).  Values with
a frame on the graph equal those of the float path (``test_frame_on_graph_gives_float_path_values``).
"""
import math

import numpy as np
import pytest
import torch

from torchspin import Experiment, Options, SpinSystem, ham, levels, pepper, salt
from torchspin.pepper_autograd import pepper_autograd
from torchspin.rotations import erot

F64 = torch.float64


def _t(x, rg=True):
    return torch.tensor(x, dtype=F64, requires_grad=rg)


def _grad_vs_cd(make, a0, h, tol, label, seed=0, stencil=2):
    """Autograd gradient of sum(w * make(a)) against central differences (all
    three angles; ``stencil`` 2 or 4 points); ``tol`` bounds the relative norm
    error."""
    a = _t(a0)
    out = make(a)
    if torch.is_complex(out):
        out = torch.view_as_real(out)
    gen = torch.Generator().manual_seed(seed)
    w = torch.rand(out.shape, dtype=F64, generator=gen)
    (g,) = torch.autograd.grad((w * out).sum(), [a], allow_unused=True)
    assert g is not None, f'{label}: no gradient (detached)'

    def f(x):
        y = make(x)
        return (w * (torch.view_as_real(y) if torch.is_complex(y) else y)).sum()
    cd = torch.zeros(len(a0), dtype=F64)
    with torch.no_grad():
        for k in range(len(a0)):
            e = torch.zeros(len(a0), dtype=F64)
            e[k] = h
            x = _t(a0, False)
            if stencil == 2:
                cd[k] = (f(x + e) - f(x - e)) / (2 * h)
            else:
                cd[k] = (8 * (f(x + e) - f(x - e)) - (f(x + 2 * e) - f(x - 2 * e))) / (12 * h)
    assert cd.norm() > 0, f'{label}: frame has no effect, test is vacuous'
    rel = float((g - cd).norm() / cd.norm())
    assert rel < tol, f'{label}: autograd {g.tolist()} vs central differences {cd.tolist()} (rel {rel:.1e})'


# ---------------------------------------------------------------------------
# erot
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('ang', [[0.3, 0.7, -1.1], [0.2, 0.0, 0.4], [0.5, math.pi, 0.1], [0.0, 0.0, 0.0]])
def test_erot_graph_path_matches_float_path(ang):
    R0 = erot(ang)
    assert not R0.requires_grad
    R1 = erot(_t(ang))
    assert R1.requires_grad and R1.dtype == F64 and R1.shape == (3, 3)
    torch.testing.assert_close(R1.detach(), R0, rtol=0, atol=4e-16)
    # three-argument form with mixed tensor / float angles
    R2 = erot(ang[0], _t(ang[1]), ang[2])
    assert R2.requires_grad
    torch.testing.assert_close(R2.detach(), R0, rtol=0, atol=4e-16)


def test_erot_without_grad_is_the_float_path():
    """A tensor that does not require grad (or grad mode off) takes the float
    path: bit-identical to the list input, as before 0.9.17."""
    ang = [0.31, 1.2, -2.3]
    R = erot(ang)
    assert torch.equal(erot(torch.tensor(ang, dtype=F64)), R)
    with torch.no_grad():
        assert torch.equal(erot(_t(ang)), R)


@pytest.mark.parametrize('ang', [[0.3, 0.7, -1.1], [0.4, 0.0, 0.9], [0.4, math.pi, 0.9]])
def test_erot_gradient(ang):
    """Finite at gimbal lock (beta = 0, pi): the entries are products of sines
    and cosines."""
    _grad_vs_cd(lambda a: erot(a), ang, 1e-6, 1e-9, f'erot {ang}')


# ---------------------------------------------------------------------------
# ham(): every tensor frame
# ---------------------------------------------------------------------------

B0 = torch.tensor([120.0, -80.0, 310.0], dtype=F64)
HAM_CASES = {
    'gFrame': dict(S=0.5, g=[2.0, 2.1, 2.3], Nucs='1H', A=[5.0, 7.0, 12.0]),
    'AFrame': dict(S=0.5, g=[2.0, 2.1, 2.3], Nucs='63Cu', A=[40.0, 60.0, 550.0]),
    'DFrame': dict(S=1, g=[2.0, 2.0, 2.0], D=[300.0, 60.0]),
    'QFrame': dict(S=0.5, g=2.0, Nucs='14N', A=[20.0, 30.0, 90.0], Q=[1.5, 0.3]),
    'eeFrame': dict(S=[0.5, 0.5], g=[[2.0, 2.0, 2.0], [2.05, 2.05, 2.05]], ee=[100.0, 150.0, -250.0]),
    'nnFrame': dict(S=0.5, g=2.0, Nucs='1H,1H', A=[[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]], nn=[1.0, 2.0, -3.0]),
    'sigmaFrame': dict(S=0.5, g=2.0, Nucs='1H', A=[5.0, 5.0, 5.0], sigma=[0.9, 1.1, 1.4]),
}


@pytest.mark.parametrize('frame', list(HAM_CASES))
@pytest.mark.parametrize('ang', [[0.3, 0.5, -0.4], [0.3, 0.0, -0.4], [0.0, 0.0, 0.0]], ids=['tilt', 'beta0', 'zero'])
def test_ham_frame_gradient(frame, ang):
    kw = HAM_CASES[frame]
    _grad_vs_cd(lambda a: ham(SpinSystem(**kw, **{frame: a}), B0), ang, 1e-3, 1e-8, f'ham {frame}', stencil=4)


@pytest.mark.parametrize('ang', [[0.2, 0.4, 0.1], [0.2, 0.0, 0.1]], ids=['tilt', 'beta0'])
def test_ham_BFrame_gradient(ang):
    """Stevens BFrame: the Wigner rotation is built in torch on the graph."""
    kw = dict(S=2, g=2.0, B=[None, [[0.0, 0.0, 0.0, 3.0, 10.0, 0.0, -2.0, 0.0, 4.0]]])

    def make(a):
        return ham(SpinSystem(**kw, BFrame=[None, a.reshape(1, 3)]), B0)
    _grad_vs_cd(make, ang, 1e-3, 1e-8, 'ham BFrame', stencil=4)
    with torch.no_grad():
        H_np = make(_t(ang, False))
    H_t = make(_t(ang))
    torch.testing.assert_close(H_t.detach(), H_np, rtol=0, atol=1e-11)


def test_ham_B_coefficients_keep_gradient_with_BFrame():
    """With a nonzero BFrame the B coefficients were detached as well."""
    b0 = [0.0, 0.0, 0.0, 3.0, 10.0, 0.0, -2.0, 0.0, 4.0]
    bf = [None, [[0.2, 0.4, 0.1]]]

    def make(b):
        return ham(SpinSystem(S=2, g=2.0, B=[None, b.reshape(1, 9)], BFrame=bf), B0)
    a = _t(b0)
    (g,) = torch.autograd.grad(make(a).real.sum() + make(a).imag.sum(), [a], allow_unused=True)
    assert g is not None and float(g.abs().max()) > 0


def test_levels_frame_gradient():
    kw = HAM_CASES['AFrame']
    B = torch.linspace(50.0, 400.0, 6, dtype=F64)
    _grad_vs_cd(lambda a: levels(SpinSystem(**kw, AFrame=a), 0.4, 1.1, B)[1], [0.3, 0.5, -0.4], 1e-3, 1e-8,
                'levels AFrame', stencil=4)
    _grad_vs_cd(lambda a: levels(SpinSystem(**kw, gFrame=a), 0.4, 1.1, B)[1], [0.3, 0.5, -0.4], 1e-3, 1e-8,
                'levels gFrame', stencil=4)


# ---------------------------------------------------------------------------
# strain widths: frames with gStrain / AStrain / DStrain
# ---------------------------------------------------------------------------

STRAIN_CASES = {
    'gFrame+gStrain': ('gFrame', dict(S=0.5, g=[2.04, 2.06, 2.25], Nucs='63Cu', A=[40.0, 60.0, 550.0],
                                      gStrain=[0.01, 0.02, 0.03])),
    'AFrame+AStrain': ('AFrame', dict(S=0.5, g=[2.04, 2.06, 2.25], Nucs='63Cu', A=[40.0, 60.0, 550.0],
                                      AStrain=[10.0, 20.0, 40.0])),
    'gFrame+AFrame strains': ('AFrame', dict(S=0.5, g=[2.04, 2.06, 2.25], gFrame=[0.1, 0.2, 0.3], Nucs='63Cu',
                                             A=[40.0, 60.0, 550.0], gStrain=[0.01, 0.02, 0.03],
                                             AStrain=[10.0, 20.0, 40.0])),
    'DFrame+DStrain': ('DFrame', dict(S=1, g=[2.0, 2.0, 2.0], D=[300.0, 60.0], DStrain=[30.0, 10.0])),
}


@pytest.mark.parametrize('case', list(STRAIN_CASES))
def test_strain_width_frame_gradient(case):
    """The batched strain widths (pepper's and pepper_autograd's path) are
    differentiable in the frame, also next to the strains."""
    from torchspin.strainwidth import compute_strain_widths_batch
    frame, kw = STRAIN_CASES[case]
    pair = [0, 3] if kw['S'] == 0.5 else [0, 1]

    def batch(a):
        s = SpinSystem(**kw, **{frame: a})
        H0, mux, muy, muz = ham(s)
        phi = torch.tensor([0.4, 1.3], dtype=F64)
        th = torch.tensor([1.1, 0.6], dtype=F64)
        st = torch.sin(th)
        z = torch.stack([st * torch.cos(phi), st * torch.sin(phi), torch.cos(th)], dim=1).to(mux.dtype)
        muzL = z[:, 0, None, None] * mux + z[:, 1, None, None] * muy + z[:, 2, None, None] * muz
        return compute_strain_widths_batch(s, H0, muzL, phi, th, torch.tensor([320.0, 330.0], dtype=F64),
                                           None, 9.5, pairs_flat=torch.tensor([pair, pair]))
    _grad_vs_cd(batch, [0.3, 0.5, -0.4], 1e-3, 1e-8, f'compute_strain_widths_batch {case}', stencil=4)


# ---------------------------------------------------------------------------
# pepper_autograd / pepper
# ---------------------------------------------------------------------------

CU = dict(S=0.5, g=[2.04, 2.06, 2.25], Nucs='63Cu', A=[40.0, 60.0, 550.0], lw=[0.6, 0.0])
EXP_CU = dict(mwFreq=9.5, Range=[250.0, 360.0], nPoints=256, Harmonic=1)
PEPPER_CASES = {
    'gFrame': ('gFrame', CU, EXP_CU),
    'AFrame': ('AFrame', CU, EXP_CU),
    'gFrame+gStrain': ('gFrame', dict(CU, gStrain=[0.01, 0.02, 0.03]), EXP_CU),
    'AFrame+AStrain': ('AFrame', dict(CU, AStrain=[10.0, 20.0, 40.0]), EXP_CU),
    'DFrame': ('DFrame', dict(S=1, g=[2.0, 2.02, 2.04], D=[300.0, 60.0], lw=[2.0, 0.0]),
               dict(mwFreq=9.5, Range=[200.0, 480.0], nPoints=256)),
    'DFrame+DStrain': ('DFrame', dict(S=1, g=[2.0, 2.02, 2.04], D=[300.0, 60.0], DStrain=[30.0, 10.0],
                                      lw=[2.0, 0.0]),
                       dict(mwFreq=9.5, Range=[200.0, 480.0], nPoints=256)),
    'QFrame': ('QFrame', dict(S=0.5, g=[2.0, 2.05, 2.2], Nucs='14N', A=[20.0, 30.0, 90.0], Q=[15.0, 0.3],
                              lw=[0.5, 0.0]),
               dict(mwFreq=9.5, Range=[290.0, 350.0], nPoints=256)),
    'eeFrame': ('eeFrame', dict(S=[0.5, 0.5], g=[[2.0, 2.0, 2.0], [2.05, 2.05, 2.05]],
                                ee=[100.0, 100.0, -200.0], lw=[1.0, 0.0]),
                dict(mwFreq=9.5, Range=[320.0, 350.0], nPoints=256)),
}


@pytest.mark.parametrize('case', list(PEPPER_CASES))
def test_pepper_autograd_powder_frame_gradient(case):
    frame, kw, ek = PEPPER_CASES[case]
    opt = Options(GridSize=[7, 4], GridSymmetry='Ci')
    _grad_vs_cd(lambda a: pepper_autograd(SpinSystem(**kw, **{frame: a}), Experiment(**ek), opt)[1],
                [0.3, 0.5, -0.4], 1e-6, 1e-5, f'pepper_autograd {case}')


@pytest.mark.parametrize('frame', ['gFrame', 'AFrame'])
def test_pepper_autograd_gimbal_lock(frame):
    """beta = 0 and beta = pi: finite gradients equal to central differences."""
    opt = Options(GridSize=[7, 4], GridSymmetry='Ci')
    for ang in ([0.3, 0.0, -0.4], [0.3, math.pi, -0.4]):
        _grad_vs_cd(lambda a: pepper_autograd(SpinSystem(**CU, **{frame: a}), Experiment(**EXP_CU), opt)[1],
                    ang, 1e-6, 1e-5, f'pepper_autograd {frame} {ang}')


def test_pepper_autograd_perturb_frame_gradient():
    opt = Options(GridSize=[7, 4], GridSymmetry='Ci', Method='perturb')
    for frame in ('gFrame', 'AFrame'):
        _grad_vs_cd(lambda a: pepper_autograd(SpinSystem(**CU, **{frame: a}), Experiment(**EXP_CU), opt)[1],
                    [0.3, 0.5, -0.4], 1e-6, 1e-5, f'pepper_autograd perturb {frame}')


def test_pepper_autograd_orientations_frame_gradient():
    ek = dict(EXP_CU, Orientations=[[0.3, 0.4], [1.2, 1.0], [2.0, 2.2]])
    for frame in ('gFrame', 'AFrame'):
        _grad_vs_cd(lambda a: pepper_autograd(SpinSystem(**CU, **{frame: a}), Experiment(**ek), Options())[1],
                    [0.3, 0.5, -0.4], 1e-6, 1e-5, f'pepper_autograd Orientations {frame}')


def test_pepper_routes_frame_grad_to_differentiable_path():
    """A system whose only grad tensor is a frame goes to pepper_autograd; the
    automatic grid (a frame changes it) is taken from the detached values."""
    ang = _t([0.0, 0.3, 0.0])
    exp, opt = Experiment(**EXP_CU), Options(GridSize=[11, 4])
    _, y = pepper(SpinSystem(**CU, AFrame=ang), exp, opt)
    assert torch.is_tensor(y) and y.requires_grad
    (g,) = torch.autograd.grad((y * torch.linspace(0, 1, y.numel(), dtype=F64)).sum(), [ang], allow_unused=True)
    assert g is not None and torch.isfinite(g).all() and float(g.abs().max()) > 0
    _, y_ref = pepper(SpinSystem(**CU, AFrame=[0.0, 0.3, 0.0]), exp, opt)
    y_ref = torch.as_tensor(y_ref)
    assert float((y.detach() - y_ref).abs().max()) < 1e-9 * float(y_ref.abs().max())


def test_repro_from_bug_report():
    """The B3 repro: gradient was None."""
    ang = _t([0.0, 0.3, 0.0])
    s = SpinSystem(S=[0.5], g=[[2.04, 2.06, 2.25]], Nucs=['63Cu'], A=[[40., 60., 550.]], AFrame=ang, lw=[0.6, 0.])
    y = pepper_autograd(s, Experiment(mwFreq=9.5, Range=[250., 360.], nPoints=1024, Harmonic=1),
                        Options(GridSize=[31, 4], GridSymmetry='C2h'))[1]
    (g,) = torch.autograd.grad((y * torch.linspace(0, 1, y.numel(), dtype=F64)).sum(), [ang], allow_unused=True)
    assert g is not None and float(g.abs().max()) > 0


@pytest.mark.parametrize('name, val', [('SampleFrame', [[0.1, 0.2, 0.3]]), ('MolFrame', [0.1, 0.2, 0.3])])
def test_pepper_experiment_frame_with_grad_raises(name, val):
    """Crystal frames (Exp.SampleFrame, Exp.MolFrame) are not differentiable:
    loud error, not a detached spectrum."""
    ek = dict(EXP_CU, CrystalSymmetry='P1', **{name: _t(val)})
    if name == 'MolFrame':
        ek['SampleFrame'] = [[0.0, 0.0, 0.0]]
    with pytest.raises(NotImplementedError):
        pepper(SpinSystem(**CU), Experiment(**ek), Options(GridSize=[7, 4]))


def test_hamsymm_with_grad_frame_on_eigenvalue_path():
    """hamsymm's eigenvalue path called .numpy() on the frame (RuntimeError for
    a tensor that requires grad); the point group is a detached decision."""
    from torchspin.hamsymm import hamsymm
    kw = dict(S=0.5, g=[[2.04, 0.01, 0.0], [0.01, 2.06, 0.0], [0.0, 0.0, 2.25]], Nucs='63Cu',
              A=[40.0, 60.0, 550.0])
    pg0, R0 = hamsymm(SpinSystem(**kw, AFrame=[0.0, 0.3, 0.0]))
    pg1, R1 = hamsymm(SpinSystem(**kw, AFrame=_t([0.0, 0.3, 0.0])))
    assert pg1 == pg0 and torch.equal(R1, R0) and not R1.requires_grad


# ---------------------------------------------------------------------------
# salt, saffron
# ---------------------------------------------------------------------------

SALT_CASES = {
    'AFrame perturb': ('AFrame', dict(S=0.5, g=[2.0, 2.05, 2.2], Nucs='1H', A=[2.0, 3.0, 9.0])),
    'gFrame perturb': ('gFrame', dict(S=0.5, g=[2.0, 2.05, 2.2], Nucs='1H', A=[2.0, 3.0, 9.0])),
    'QFrame matrix': ('QFrame', dict(S=0.5, g=[2.0, 2.05, 2.2], Nucs='14N', A=[4.0, 5.0, 9.0], Q=[1.5, 0.3])),
}


@pytest.mark.parametrize('case', list(SALT_CASES))
def test_salt_fixed_field_frame_gradient(case):
    frame, kw = SALT_CASES[case]
    exp = Experiment(Field=330.0, mwFreq=9.4, ExciteWidth=200.0)
    opt = Options(GridSize=[7, 4], GridSymmetry='Ci')
    _grad_vs_cd(lambda a: salt(SpinSystem(**kw, **{frame: a}), exp, opt, freq_range=(0.5, 25.0),
                               n_points=256, lw_mhz=0.3)[1],
                [0.3, 0.5, -0.4], 1e-5, 1e-5, f'salt {case}')


@pytest.mark.parametrize('frame', ['gFrame', 'AFrame', 'DFrame', 'QFrame', 'eeFrame'])
def test_salt_field_swept_with_grad_frame_raises(frame):
    kw = dict(S=0.5, g=[2.0, 2.05, 2.2], Nucs='14N', A=[4.0, 5.0, 9.0], Q=[1.5, 0.3])
    if frame == 'DFrame':
        kw = dict(kw, S=1, D=[300.0, 30.0])
    if frame == 'eeFrame':
        kw = dict(S=[0.5, 0.5], g=[[2.0] * 3, [2.05] * 3], ee=[10.0, 10.0, -20.0], Nucs='1H',
                  A=[[2.0, 3.0, 9.0, 0.0, 0.0, 0.0]])
    exp = Experiment(mwFreq=9.4, Range=[300.0, 360.0])
    with pytest.raises(NotImplementedError, match='field-swept'):
        salt(SpinSystem(**kw, **{frame: _t([0.3, 0.5, -0.4])}), exp, Options(GridSize=[5, 4]),
             freq_range=(0.5, 25.0), n_points=128)


@pytest.mark.parametrize('frame', ['gFrame', 'AFrame', 'QFrame'])
def test_saffron_frame_gradient(frame):
    """A frame alone switches saffron to its differentiable (time-domain) accumulation."""
    from torchspin.saffron import PulseExperiment, SaffronOptions, saffron
    exp = PulseExperiment(Sequence='2pESEEM', Field=350, dt=0.01, nPoints=64, tau=0.1)
    opt = SaffronOptions(GridSize=7, GridSymmetry='Ci')
    kw = dict(S=0.5, g=[2.01, 2.03, 2.06], Nucs='14N', A=[3.2, 2.7, 6.1], Q=[-0.6, -0.4, 1.0])
    a = _t([0.3, 0.5, -0.4])
    y = saffron(SpinSystem(**kw, **{frame: a}), exp, opt)[1]
    assert torch.is_tensor(y) and y.requires_grad
    _grad_vs_cd(lambda a: torch.as_tensor(saffron(SpinSystem(**kw, **{frame: a}),
                                                  exp, SaffronOptions(GridSize=7, GridSymmetry='Ci',
                                                                      TimeDomain=True))[1]).real,
                [0.3, 0.5, -0.4], 1e-6, 1e-6, f'saffron {frame}')


# ---------------------------------------------------------------------------
# values: a frame on the graph gives the float path's values
# ---------------------------------------------------------------------------

def _np(y):
    return (y.detach() if torch.is_tensor(y) else torch.as_tensor(y)).numpy()


def _frame_outputs(mk, graph_paths):
    """Outputs of the configurations above with tilted frames; ``mk`` turns an
    angle list into the frame value.  ``graph_paths`` runs the functions a grad
    frame is routed to (``pepper_autograd``, ``saffron`` time domain) in place
    of ``pepper`` and ``saffron``'s default, so both arms of a comparison take
    the same code path."""
    from torchspin.saffron import PulseExperiment, SaffronOptions, saffron
    from torchspin.strainwidth import compute_strain_widths
    out = {}
    a = [0.3, 0.5, -0.4]
    for frame, kw in HAM_CASES.items():
        out[f'ham_{frame}'] = _np(torch.view_as_real(ham(SpinSystem(**kw, **{frame: mk(a)}), B0)))
    H = ham(SpinSystem(S=2, g=2.0, B=[None, [[0.0, 0.0, 0.0, 3.0, 10.0, 0.0, -2.0, 0.0, 4.0]]],
                       BFrame=[None, mk([[0.2, 0.4, 0.1]])]), B0)
    out['ham_BFrame'] = _np(torch.view_as_real(H))
    sim = pepper_autograd if graph_paths else pepper
    for case, (frame, kw, ek) in PEPPER_CASES.items():
        for method in ('matrix', 'perturb'):
            if method == 'perturb' and case not in ('gFrame', 'AFrame', 'gFrame+gStrain', 'AFrame+AStrain'):
                continue
            _, y = sim(SpinSystem(**kw, **{frame: mk(a)}), Experiment(**ek), Options(GridSize=[7, 4], Method=method))
            out[f'pepper_{method}_{case}'] = _np(y)
    for case, (frame, kw) in STRAIN_CASES.items():
        s = SpinSystem(**kw, **{frame: mk(a)})
        H0, mux, muy, muz = ham(s)
        trans = torch.tensor([[0, 3]] if kw['S'] == 0.5 else [[0, 1]])
        out[f'width_{case}'] = _np(compute_strain_widths(sys=s, H0=H0, mux=mux, muy=muy, muz=muz, phi=0.4,
                                                         theta=1.1, transitions=trans, mwFreq=9.5, B0=320.0))
    for case, (frame, kw) in SALT_CASES.items():
        _, y = salt(SpinSystem(**kw, **{frame: mk(a)}), Experiment(Field=330.0, mwFreq=9.4, ExciteWidth=200.0),
                    Options(GridSize=[7, 4]), freq_range=(0.5, 25.0), n_points=256, lw_mhz=0.3)
        out[f'salt_{case}'] = _np(y)
    exp = PulseExperiment(Sequence='2pESEEM', Field=350, dt=0.01, nPoints=64, tau=0.1)
    kw = dict(S=0.5, g=[2.01, 2.03, 2.06], Nucs='14N', A=[3.2, 2.7, 6.1], Q=[-0.6, -0.4, 1.0])
    for frame in ('gFrame', 'AFrame', 'QFrame'):
        _, y, _ = saffron(SpinSystem(**kw, **{frame: mk(a)}), exp,
                          SaffronOptions(GridSize=7, TimeDomain=True) if graph_paths else SaffronOptions(GridSize=7))
        out[f'saffron_{frame}'] = _np(y)
    return out


def test_frame_on_graph_gives_float_path_values():
    """The same 30 configurations with frames as Python lists (float ``erot``)
    and as tensors that require grad (torch ``erot``, outputs detached), each
    through the function a grad frame is routed to.

    Bit-identical except: ``ham_BFrame`` (the torch Wigner D sums in another
    order; 23 of 50 entries differ, at most 7 ulp of the entry and under 1 ulp
    of max|H|, bound 4 ulp of max|H|) and ``pepper_autograd`` with a grad
    ``gFrame`` (rel 1.4e-13, bound 1e-11 of the maximum; not ``erot``: a grad
    ``g`` with no frame differs from a float ``g`` by 2e-14 there too).

    That this PR leaves non-grad results unchanged was checked against main
    0.9.0 when it was written: 466 arrays (150 random systems with every frame
    type, ``ham``, ``H0``, ``muz``, 16 ``pepper`` spectra with strains) and
    these 30 outputs were bit-identical.  They are not stored, because later
    fixes change some of these outputs on purpose."""
    ref = _frame_outputs(lambda v: v, graph_paths=True)
    out = _frame_outputs(_t, graph_paths=True)
    assert sorted(out) == sorted(ref) and len(ref) == 30
    for k in sorted(ref):
        scale = np.abs(ref[k]).max()
        assert scale > 0, k
        if k == 'ham_BFrame':
            np.testing.assert_allclose(out[k], ref[k], rtol=0, atol=4 * np.spacing(scale), err_msg=k)
        elif k in ('pepper_matrix_gFrame', 'pepper_matrix_gFrame+gStrain', 'pepper_perturb_gFrame+gStrain',
                   'pepper_matrix_DFrame+DStrain'):
            np.testing.assert_allclose(out[k], ref[k], rtol=0, atol=1e-11 * scale, err_msg=k)
        else:
            assert np.array_equal(out[k], ref[k]), f'{k}: differs by {np.abs(out[k] - ref[k]).max() / scale:.1e} (rel)'


def test_tensor_frame_without_grad_gives_float_path_values():
    """Frames as tensors without grad take the float path at every call site
    (``pepper``, ``saffron`` default included): bit-identical to lists."""
    ref = _frame_outputs(lambda v: v, graph_paths=False)
    out = _frame_outputs(lambda v: _t(v, False), graph_paths=False)
    assert sorted(out) == sorted(ref) and len(ref) == 30
    for k in sorted(ref):
        assert np.array_equal(out[k], ref[k]), f'{k}: differs from the list frame'
