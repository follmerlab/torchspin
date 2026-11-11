"""Tests for torchspin.orisel: orientation selectivity weights (EasySpin orisel).

Since 0.9.0 (bug B26) orisel follows EasySpin's model (orisel.m:110-153):
full diagonalization at Exp.Field, rate x Gaussian in frequency with width from
Exp.ExciteWidth and HStrain, no grid weights, no normalization.
References: tests/data/ref_orisel.mat (generate_orisel_refs.m) and EasySpin's own
regression data orisel_rhombic.mat / orisel_onenucleus.mat.
"""
import math
import os

import numpy as np
import pytest
import torch

from torchspin.experiment import Experiment, Options
from torchspin.orisel import OriselOptions, orisel
from torchspin.spinsystem import SpinSystem

loadmat = pytest.importorskip('scipy.io').loadmat

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'data')
DEG = math.pi / 180

# Spin systems of generate_orisel_refs.m
SYS = {
    'rh': dict(S=0.5, g=[2.0, 2.05, 2.1]),
    'cu': dict(S=0.5, g=[2.05, 2.05, 2.25], Nucs='63Cu', A=[50, 50, 500]),
    'g6': dict(S=0.5, g=[1.9, 2.0, 6.0]),
    's1': dict(S=1, g=2.0, D=[1500, 300], HStrain=[60, 80, 100]),
    's32': dict(S=1.5, g=2.0, D=[3000, 600], HStrain=[50, 50, 50]),
    'two': dict(S=[0.5, 0.5], g=[[2.0, 2.0, 2.1], [2.05, 2.05, 2.0]], ee=[[100, 100, -200]]),
    'rhs': dict(S=0.5, g=[2.0, 2.05, 2.1], HStrain=[30, 40, 60]),
    'rot': dict(S=0.5, g=[2.0, 2.1, 2.2], gFrame=[[10 * DEG, 30 * DEG, 0]], HStrain=[30, 40, 60]),
    'cuf': dict(S=0.5, g=[2.05, 2.05, 2.25], Nucs='63Cu', A=[50, 50, 500],
                AFrame=[[0, 20 * DEG, 0]]),
    'ax': dict(S=0.5, g=[2.0, 2.0, 2.2]),
    'iso': dict(S=0.5, g=2.0),
}
CASES = {
    # Exp.Orientations (400-point Fibonacci sphere): the weight model alone
    'rh_gz': 'rh', 'rh_gy': 'rh', 'rh_gx': 'rh',          # rhombic S=1/2 at 3 fields
    'cu_294': 'cu', 'cu_perp': 'cu',                     # Cu(II), 63Cu
    'g6_g6': 'g6', 'g6_g4': 'g6',                        # g = [1.9 2.0 6.0] at g = 6, 4
    's1_hstrain': 's1', 's1_noavg': 's1',                # S=1 with D, HStrain; AveragedIntensity
    's32_geff4': 's32',                                  # S=3/2 with D
    'two_electrons': 'two',                              # two coupled electron spins
    'inf_excitewidth': 'rh',                             # ExciteWidth = Inf -> ones
    # default grids (hamsymm symmetry and GridFrame, sphgrid, vec2ang)
    'grid_rh_default': 'rhs', 'grid_rotated_d2h': 'rot', 'grid_cu_aframe': 'cuf',
    'grid_axial': 'ax', 'grid_explicit_ci': 'rot', 'grid_isotropic': 'iso',
}


@pytest.fixture(scope='module')
def refs():
    path = os.path.join(DATA_DIR, 'ref_orisel.mat')
    if not os.path.exists(path):
        pytest.skip(f'Reference file not found: {path}')
    return loadmat(path, simplify_cells=True)


def _run_case(r, sys_key):
    E, O = r['Exp'], r['Opt']
    kw = dict(Field=float(E['Field']), mwFreq=float(E['mwFreq']),
              ExciteWidth=float(E['ExciteWidth']))
    if 'Orientations' in E:
        kw['Orientations'] = np.asarray(E['Orientations'], float).reshape(2, -1).T
    ok = {}
    if isinstance(O, dict):
        if 'GridSize' in O:
            ok['GridSize'] = int(O['GridSize'])
        if 'GridSymmetry' in O:
            ok['GridSymmetry'] = str(O['GridSymmetry'])
        if 'AveragedIntensity' in O:
            ok['AveragedIntensity'] = bool(O['AveragedIntensity'])
    return orisel(SpinSystem(**SYS[sys_key]), Experiment(**kw), OriselOptions(**ok),
                  per_transition=True)


# ---------------------------------------------------------------------------
# EasySpin references
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('name', list(CASES))
def test_matches_easyspin(refs, name):
    """[Weights, Transitions] and the orientations match EasySpin orisel (rel 1e-10)."""
    r = refs[name]
    W, trans, phi, theta = _run_case(r, CASES[name])
    W_ref = np.asarray(r['W'], float).reshape(W.shape[0], -1)
    assert W.shape == W_ref.shape
    scale = np.abs(W_ref).max()
    assert scale > 0
    assert np.abs(W.numpy() - W_ref).max() <= 1e-10 * scale
    assert np.array_equal(trans.numpy(), np.atleast_2d(r['Trans']))
    # one-output form: sum over level pairs
    w_ref = np.atleast_1d(np.asarray(r['w'], float))
    assert np.abs(W.sum(dim=1).numpy() - w_ref).max() <= 1e-10 * np.abs(w_ref).max()
    # orientations: Exp.Orientations as given, or EasySpin's grid in GridFrame
    ori = np.asarray(r['ori'], float).reshape(2, -1)
    assert np.abs(theta.numpy() - ori[1]).max() < 1e-12
    dphi = np.angle(np.exp(1j * (phi.numpy() - ori[0])))
    assert np.abs(dphi * np.sin(ori[1])).max() < 1e-12


def test_default_grid_size_and_symmetry(refs):
    """Default options: GridSize 46 on the hamsymm grid, 1081 D2h knots as in EasySpin."""
    w, phi, theta = orisel(SpinSystem(**SYS['rhs']),
                           Experiment(mwFreq=9.5, Field=float(refs['grid_rh_default']['Exp']['Field']),
                                      ExciteWidth=100))
    assert w.shape == (1081,) == phi.shape == theta.shape
    assert OriselOptions().GridSize == 46


def test_selection_follows_field(refs):
    """The bug B26 symptom: the selected orientations move with Exp.Field.  For g = [1.9 2.0 6.0]
    at the g = 6 field the weight sits along z, at g = 2 in the xy plane (EasySpin)."""
    sy = SpinSystem(**SYS['g6'])
    ori = np.asarray(refs['g6_g6']['Exp']['Orientations'], float).reshape(2, -1).T
    mw = 9.5
    from torchspin.constants import BMAGN, PLANCK

    def mean_cos2(g):
        Bres = PLANCK * mw * 1e9 / (g * BMAGN) * 1e3
        w, _, th = orisel(sy, Experiment(mwFreq=mw, Field=Bres, ExciteWidth=100, Orientations=ori))
        return float((w * torch.cos(th) ** 2).sum() / w.sum())
    assert mean_cos2(6.0) > 0.95
    assert mean_cos2(1.95) < 0.05


@pytest.mark.parametrize('fname,kw', [
    ('orisel_rhombic.mat', {}),
    ('orisel_onenucleus.mat', dict(Nucs='63Cu', A=[200, 200, 400])),
])
def test_easyspin_regression_data(fname, kw):
    """EasySpin's own regression tests orisel_rhombic.m / orisel_onenucleus.m (GridSize 91,
    compared there as w/max(w) to 1e-5 abs)."""
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        pytest.skip(f'Reference file not found: {path}')
    ref = np.asarray(loadmat(path, simplify_cells=True)['data']['w'], float)
    sy = SpinSystem(S=0.5, g=[2, 2.1, 2.2], HStrain=[100, 100, 100], **kw)
    w, _, _ = orisel(sy, Experiment(mwFreq=9, Field=310, ExciteWidth=500),
                     OriselOptions(GridSize=91))
    w = w.numpy()
    assert w.shape == ref.shape
    assert np.abs(w / w.max() - ref / ref.max()).max() < 1e-5


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def _sys_exp(**ekw):
    sy = SpinSystem(S=0.5, g=[2.0, 2.1, 2.2], HStrain=[50, 50, 50])
    kw = dict(mwFreq=9.5, Field=320.0, ExciteWidth=100.0)
    kw.update(ekw)
    return sy, Experiment(**kw)


class TestOriselAPI:
    def test_one_output_form(self):
        sy, ex = _sys_exp()
        w, phi, theta = orisel(sy, ex, OriselOptions(GridSize=10))
        assert torch.is_tensor(w) and w.ndim == 1
        assert w.shape == phi.shape == theta.shape
        assert bool(torch.all(w >= 0))

    def test_per_transition_form(self):
        sy = SpinSystem(S=1, g=2.0, D=[1500, 300])
        ex = Experiment(mwFreq=9.5, Field=300, ExciteWidth=200)
        W, trans, phi, theta = orisel(sy, ex, OriselOptions(GridSize=8), per_transition=True)
        w, _, _ = orisel(sy, ex, OriselOptions(GridSize=8))
        assert W.shape == (phi.shape[0], 3)
        assert trans.tolist() == [[1, 2], [1, 3], [2, 3]]
        assert trans.dtype == torch.int64
        assert torch.equal(W.sum(dim=1), w)

    def test_not_normalized(self):
        """No normalization to max 1 (EasySpin): S=1/2 weights are rate-scaled."""
        sy, ex = _sys_exp()
        w, _, _ = orisel(sy, ex, OriselOptions(GridSize=10))
        assert float(w.max()) > 10.0

    def test_narrow_excitation_more_selective(self):
        sy = SpinSystem(S=0.5, g=[2.0, 2.1, 2.2])
        fr = []
        for ew in (5000.0, 20.0):
            w, _, _ = orisel(sy, Experiment(mwFreq=9.5, Field=320.0, ExciteWidth=ew),
                             OriselOptions(GridSize=15))
            fr.append(float((w > 0.5 * w.max()).double().mean()))
        assert fr[1] < fr[0]

    def test_infinite_excitewidth_gives_ones(self):
        sy, ex = _sys_exp(ExciteWidth=math.inf)
        W, trans, _, _ = orisel(sy, ex, OriselOptions(GridSize=5), per_transition=True)
        assert torch.equal(W, torch.ones_like(W))

    def test_orientations_input(self):
        sy, _ = _sys_exp()
        ori = [[0.0, 0.0], [0.3, 1.0], [1.2, math.pi / 2]]
        w, phi, theta = orisel(sy, Experiment(mwFreq=9.5, Field=320, ExciteWidth=100,
                                              Orientations=ori))
        assert w.shape == (3,)
        assert torch.allclose(phi, torch.tensor([0.0, 0.3, 1.2], dtype=torch.float64))
        assert torch.allclose(theta, torch.tensor([0.0, 1.0, math.pi / 2], dtype=torch.float64))

    def test_duck_typed_options(self):
        """The Options used for salt can be passed, as in EasySpin (GridSize, GridSymmetry)."""
        sy, ex = _sys_exp()
        w1, _, _ = orisel(sy, ex, Options(GridSize=12, GridSymmetry='Ci'))
        w2, _, _ = orisel(sy, ex, OriselOptions(GridSize=12, GridSymmetry='Ci'))
        assert torch.equal(w1, w2)

    def test_h0extra_supported(self):
        """orisel is ham-based since 0.9.0: a zero-field splitting given as H0Extra gives the
        weights of the native D."""
        from torchspin import sop
        D, E = 1500.0, 300.0
        native = SpinSystem(S=1, g=2.0, D=[D, E])
        Sx, Sy, Sz = (sop(native.Spins, [[1, c]]) for c in (1, 2, 3))
        op = D * (Sz @ Sz - 2.0 / 3.0 * torch.eye(3, dtype=Sz.dtype)) + E * (Sx @ Sx - Sy @ Sy)
        extra = SpinSystem(S=1, g=2.0, H0Extra=[(1.0, op)])
        ex = Experiment(mwFreq=9.5, Field=300, ExciteWidth=200)
        opt = OriselOptions(GridSize=8, GridSymmetry='D2h')
        w1, _, _ = orisel(native, ex, opt)
        w2, _, _ = orisel(extra, ex, opt)
        assert torch.allclose(w1, w2, rtol=1e-10, atol=1e-12 * float(w1.max()))


class TestOriselErrors:
    """Missing inputs, removed pre-0.9.0 API, and (B15) error propagation."""

    def test_missing_field_with_old_range_usage(self):
        sy, _ = _sys_exp()
        with pytest.raises(ValueError, match=r'Experiment\.Field.*no longer sweeps Experiment\.Range'):
            orisel(sy, Experiment(mwFreq=9.5, Range=[300, 360]))

    @pytest.mark.parametrize('drop', ['Field', 'mwFreq', 'ExciteWidth'])
    def test_missing_required_field(self, drop):
        sy, ex = _sys_exp()
        setattr(ex, drop, None)
        with pytest.raises(ValueError, match=rf'Experiment\.{drop} .*is missing'):
            orisel(sy, ex)

    @pytest.mark.parametrize('name', ['ExcitBandwidth', 'Threshold'])
    def test_removed_options(self, name):
        with pytest.raises(ValueError, match=f'{name} was removed'):
            OriselOptions(**{name: 1.0})

    def test_zero_width(self):
        """orisel.m:58-60: HStrain + ExciteWidth must be positive."""
        sy = SpinSystem(S=0.5, g=[2.0, 2.1, 2.2])
        with pytest.raises(ValueError, match='must be positive'):
            orisel(sy, Experiment(mwFreq=9.5, Field=320, ExciteWidth=0.0))
        sy = SpinSystem(S=0.5, g=[2.0, 2.1, 2.2], HStrain=[10, 10, 10])
        w, _, _ = orisel(sy, Experiment(mwFreq=9.5, Field=320, ExciteWidth=0.0),
                         OriselOptions(GridSize=5))
        assert float(w.max()) > 0

    def test_bad_inputs(self):
        sy, ex = _sys_exp()
        with pytest.raises(ValueError, match='GridSize'):
            OriselOptions(GridSize=1)
        with pytest.raises(ValueError, match='GridSymmetry'):
            OriselOptions(GridSymmetry='D3')
        ex.Field = 0.0
        with pytest.raises(ValueError, match='Field must be positive'):
            orisel(sy, ex)

    def test_solver_error_propagates(self, monkeypatch):
        """B15 (0.8.6): an exception in the diagonalization reaches the caller instead of
        becoming zero selectivity."""
        import importlib
        om = importlib.import_module('torchspin.orisel')

        def boom(*args, **kwargs):
            raise RuntimeError('solver failed')
        monkeypatch.setattr(om, '_eigh', boom)
        sy, ex = _sys_exp()
        with pytest.raises(RuntimeError, match='solver failed'):
            orisel(sy, ex, OriselOptions(GridSize=5))

    def test_no_resonance_gives_zero_weights(self):
        """A field far from any resonance is a legitimate empty result: all weights 0."""
        sy, _ = _sys_exp()
        w, _, _ = orisel(sy, Experiment(mwFreq=9.5, Field=15.0, ExciteWidth=100),
                         OriselOptions(GridSize=10))
        assert bool(torch.all(w == 0.0))

    def test_two_electrons_supported(self):
        """Before 0.9.0 two electron spins raised NotImplementedError (perturbation theory)."""
        sy = SpinSystem(S=[0.5, 0.5], g=[[2.0, 2.0, 2.0], [2.1, 2.1, 2.1]], ee=[300.0])
        w, _, _ = orisel(sy, Experiment(mwFreq=9.5, Field=330, ExciteWidth=300),
                         OriselOptions(GridSize=6))
        assert float(w.max()) > 0


# ---------------------------------------------------------------------------
# Gradients (reverse mode vs central differences)
# ---------------------------------------------------------------------------

def _central_diff(f, x, h):
    g = torch.zeros_like(x)
    for i in range(x.numel()):
        e = torch.zeros_like(x)
        e.view(-1)[i] = h
        g.view(-1)[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g


_ORI = torch.tensor([[0.1 + 0.37 * k, math.acos(1 - 2 * (k + 0.5) / 40)] for k in range(40)],
                    dtype=torch.float64)
_WTS = torch.linspace(0.5, 1.5, 40, dtype=torch.float64)


def _loss(**sys_kw):
    sy = SpinSystem(**sys_kw)
    W, _, _, _ = orisel(sy, Experiment(mwFreq=9.5, Field=330.0, ExciteWidth=150.0,
                                       Orientations=_ORI), per_transition=True)
    return (W.sum(dim=1) * _WTS).sum()


@pytest.mark.parametrize('param,x0,h,build', [
    ('g', [2.0, 2.05, 2.1], 1e-6,
     lambda x: dict(S=0.5, g=x.unsqueeze(0), HStrain=[30, 40, 60])),
    ('A', [50.0, 60.0, 400.0], 1e-3,
     lambda x: dict(S=0.5, g=[2.05, 2.05, 2.15], Nucs='63Cu', A=x.unsqueeze(0))),
    ('D', [1200.0, 250.0], 1e-2,
     lambda x: dict(S=1, g=2.0, D=x.unsqueeze(0), HStrain=[50, 50, 50])),
    ('HStrain', [30.0, 40.0, 60.0], 1e-3,
     lambda x: dict(S=0.5, g=[2.0, 2.05, 2.1], HStrain=x)),
])
def test_gradient_vs_central_differences(param, x0, h, build):
    x = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
    loss = _loss(**build(x))
    g_ad, = torch.autograd.grad(loss, x)
    with torch.no_grad():
        g_fd = _central_diff(lambda y: _loss(**build(y)), x.detach(), h)
    assert torch.all(torch.isfinite(g_ad))
    assert g_fd.abs().max() > 0
    assert torch.allclose(g_ad, g_fd, rtol=1e-5, atol=1e-6 * float(g_fd.abs().max())), \
        f'{param}: autograd {g_ad.tolist()} vs FD {g_fd.tolist()}'


def test_gradient_excitewidth():
    """Exp.ExciteWidth as a tensor stays on the graph (autograd vs central differences)."""
    sy = SpinSystem(S=0.5, g=[2.0, 2.05, 2.1], HStrain=[30, 40, 60])

    def f(ew):
        w, _, _ = orisel(sy, Experiment(mwFreq=9.5, Field=330.0, ExciteWidth=ew,
                                        Orientations=_ORI))
        return (w * _WTS).sum()
    x = torch.tensor(150.0, dtype=torch.float64, requires_grad=True)
    g_ad, = torch.autograd.grad(f(x), x)
    h = 1e-3
    with torch.no_grad():
        g_fd = (f(x.detach() + h) - f(x.detach() - h)) / (2 * h)
    assert float(g_fd) != 0.0
    assert float(g_ad) == pytest.approx(float(g_fd), rel=1e-6)
