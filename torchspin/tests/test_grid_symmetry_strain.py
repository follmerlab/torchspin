"""B82: the automatic grid symmetry must also be a symmetry of the strain tensors.

The strain line widths depend on the field direction (HStrain in the molecular
frame, gStrain in the g frame, AStrain in the A frame, DStrain in the D frame).
hamsymm used to look at the Hamiltonian only, so an axial g with a rhombic strain
got a Dinfh grid: one meridian, which samples the strain along a single azimuth
(79 % of the maximum off for HStrain = [200 20 20] MHz, while D2h [31,4] and
[91,4] agree to 0.9 %).  EasySpin's hamsymm has no strain term either; this is a
documented torchspin deviation (KNOWN_LIMITATIONS.md).
"""
import numpy as np
import pytest
import torch

from torchspin import SpinSystem, Experiment, Options, pepper
from torchspin.hamsymm import hamsymm

EXP_G = Experiment(mwFreq=9.5, Range=[300., 345.], nPoints=1024, Harmonic=1)
EXP_CU = Experiment(mwFreq=9.5, Range=[260., 345.], nPoints=1024, Harmonic=1)
EXP_D = Experiment(mwFreq=9.5, Range=[0., 700.], nPoints=1024, Harmonic=1)


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max() / np.abs(b).max())


# axial Hamiltonian + rhombic strain in the molecular frame: D2h in the molecular
# frame, so the default spectrum must be the forced-D2h spectrum exactly
RHOMBIC_STRAIN = {
    'HStrain': (dict(S=0.5, g=[2.05, 2.05, 2.15], lw=0.3, HStrain=[200., 20., 60.]), EXP_G),
    'gStrain': (dict(S=0.5, g=[2.05, 2.05, 2.15], lw=0.3, gStrain=[0.02, 0.002, 0.006]), EXP_G),
    'AStrain': (dict(S=0.5, g=[2.05, 2.05, 2.25], Nucs='63Cu', A=[40., 40., 550.],
                     AStrain=[60., 5., 25.], lw=1.0), EXP_CU),
    'DStrain': (dict(S=1, g=2.0, D=[1500., 0.], DStrain=[50., 150.], lw=0.5), EXP_D),
}


@pytest.mark.parametrize('name', list(RHOMBIC_STRAIN))
def test_rhombic_strain_uses_d2h(name):
    kw, exp = RHOMBIC_STRAIN[name]
    sys = SpinSystem(**kw)
    pg, R = hamsymm(sys)
    assert pg == 'D2h'
    assert torch.allclose(R.abs(), torch.eye(3, dtype=R.dtype))
    y_auto = pepper(sys, exp, Options(GridSize=[31, 4]))[1].numpy()
    y_d2h = pepper(sys, exp, Options(GridSize=[31, 4], GridSymmetry='D2h'))[1].numpy()
    np.testing.assert_array_equal(y_auto, y_d2h)


@pytest.mark.parametrize('kw', [
    dict(HStrain=[200., 20., 20.]),        # axial about x, perpendicular to g's z axis
    dict(gStrain=[0.02, 0.002, 0.002]),
])
def test_b82_case_matches_converged_reference(kw):
    """The reported case: auto vs D2h [61,4] within the [31,4] discretization
    (~1 %); the old Dinfh grid was 79 % / 76 % off."""
    sys = SpinSystem(S=0.5, g=[2.05, 2.05, 2.15], lw=0.3, **kw)
    assert hamsymm(sys)[0] == 'D2h'
    y_auto = pepper(sys, EXP_G, Options(GridSize=[31, 4]))[1].numpy()
    y_ref = pepper(sys, EXP_G, Options(GridSize=[61, 4], GridSymmetry='D2h'))[1].numpy()
    y_d2h = pepper(sys, EXP_G, Options(GridSize=[31, 4], GridSymmetry='D2h'))[1].numpy()
    assert _rel(y_auto, y_ref) < 3e-2
    assert _rel(y_auto, y_ref) < 2 * _rel(y_d2h, y_ref) + 1e-3


@pytest.mark.parametrize('kw, expected', [
    # strains that keep the Hamiltonian's symmetry
    (dict(S=0.5, g=[2.0, 2.0, 2.2], HStrain=[20., 20., 20.]), 'Dinfh'),
    (dict(S=0.5, g=[2.0, 2.0, 2.2], HStrain=[20., 20., 50.]), 'Dinfh'),
    (dict(S=0.5, g=[2.0, 2.0, 2.2], gStrain=[0.01, 0.01, 0.03]), 'Dinfh'),
    (dict(S=1, g=2.0, D=[300., 0.], DStrain=[30., 0.]), 'Dinfh'),
    (dict(S=1, g=2.0, D=[300., 50.], DStrain=[30., 10.]), 'D2h'),
    (dict(S=0.5, g=[2.0, 2.0, 2.0], HStrain=[10., 10., 10.]), 'O3'),
    # strains that lower it
    (dict(S=0.5, g=[2.0, 2.0, 2.0], HStrain=[10., 10., 30.]), 'Dinfh'),
    (dict(S=0.5, g=[2.0, 2.0, 2.0], HStrain=[10., 20., 30.]), 'D2h'),
    (dict(S=0.5, g=[2.0, 2.0, 2.2], gFrame=[0., 0.6, 0.], HStrain=[10., 20., 30.]), 'C2h'),
    (dict(S=0.5, g=[2.0, 2.0, 2.2], gFrame=[0.3, 0.6, 0.2], HStrain=[10., 20., 30.]), 'Ci'),
    (dict(S=1, g=2.0, D=[300., 0.], DStrain=[30., 10.]), 'D2h'),
    (dict(S=1, g=2.0, D=[300., 0.], DFrame=[0., 0.5, 0.], DStrain=[30., 10.], HStrain=[5., 5., 9.]), 'C2h'),
    (dict(S=0.5, g=[2.0, 2.0, 2.2], Nucs='1H', A=[2., 2., 8.], AStrain=[1., 2., 3.]), 'D2h'),
    (dict(S=[0.5, 0.5], g=[[2.0, 2.0, 2.2], [2.0, 2.0, 2.2]], ee=10., gStrain=[[0.01, 0.01, 0.03], [0.01, 0.02, 0.03]]), 'D2h'),
    (dict(S=0.5, g=[2.0, 2.0, 2.2], HStrainL=[10., 20., 30.]), 'D2h'),
])
def test_hamsymm_strain_groups(kw, expected):
    assert hamsymm(SpinSystem(**kw))[0] == expected


def test_eigenvalue_path_includes_strain():
    """High-order terms take the eigenvalue probe: the strain widths are probed too."""
    cubic = dict(S=[3.5], g=[[2, 2, 2]], B=[None, [[50., 0, 0, 0, 10., 0, 0, 0, 0]], None])
    assert hamsymm(SpinSystem(**cubic))[0] == 'Oh'
    assert hamsymm(SpinSystem(**cubic, HStrain=[10., 10., 30.]))[0] == 'D4h'
    assert hamsymm(SpinSystem(**cubic, HStrain=[10., 20., 30.]))[0] == 'D2h'
    full_g = dict(S=0.5, g=[[2.0, 0, 0], [0, 2.0, 0], [0, 0, 2.2]])
    assert hamsymm(SpinSystem(**full_g))[0] == 'Dinfh'
    assert hamsymm(SpinSystem(**full_g, HStrain=[10., 20., 30.]))[0] == 'D2h'


def test_strain_free_systems_unchanged():
    """No strain (or zero strain): hamsymm's group and frame are those of the
    strain-free system (values recorded on 0.9.18)."""
    cases = [
        (dict(S=0.5, g=[2.0, 2.0, 2.2]), 'Dinfh'),
        (dict(S=0.5, g=[2.0, 2.0, 2.2], gFrame=[0.1, 0.7, 0.2], Nucs='14N', A=[[5., 5., 30.]],
              AFrame=[[0.1, 0.7, 0.2]], Q=[[-0.5, -0.5, 1.0]]), 'C2h'),     # Q in the molecular frame
        (dict(S=0.5, g=[2.0, 2.05, 2.2], Nucs='1H', A=[[3., 4., 8.]], AFrame=[[0.4, 0., 0.]]), 'C2h'),
        (dict(S=1, g=2.0, D=[300., 50.], DFrame=[0.2, 0.3, 0.4]), 'D2h'),
        (dict(S=[3.5], g=[[2, 2, 2]], B=[None, [[50., 0, 0, 0, 10., 0, 0, 0, 0]], None]), 'Oh'),
        (dict(S=0.5, g=[2.2, 2.0, 2.0]), 'Dinfh'),
        (dict(S=0.5, g=[2.0, 2.0, 2.0], Nucs='1H', A=[[2., 2., 8.]], AFrame=[[0., 0.7, 0.]]), 'Dinfh'),
        (dict(S=0.5, g=[[2.0, 0.01, 0], [0.01, 2.0, 0], [0, 0, 2.2]]), 'C2h'),
        (dict(S=[0.5, 0.5], g=[[2.0, 2.0, 2.2], [2.0, 2.1, 2.2]], ee=[10., 10., 20.], eeFrame=[0, 0.3, 0]), 'C2h'),
    ]
    for kw, expected in cases:
        pg, R = hamsymm(SpinSystem(**kw))
        assert pg == expected
        for zero in (dict(HStrain=[0., 0., 0.]), dict(gStrain=[0., 0., 0.])):
            pg0, R0 = hamsymm(SpinSystem(**kw, **zero))
            assert pg0 == pg and torch.equal(R0, R)


# ---------------------------------------------------------------------------
# EasySpin reference (tests/data/ref_hamsymm_strain.mat, generate_hamsymm_strain_refs.m)
# ---------------------------------------------------------------------------

def _es_cases():
    import os
    loadmat = pytest.importorskip('scipy.io').loadmat
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'data', 'ref_hamsymm_strain.mat')
    if not os.path.exists(path):
        pytest.skip(f'Reference data not found: {path}')
    refs = loadmat(path, simplify_cells=True)
    return {c['name']: c for c in refs['cases']}


@pytest.mark.parametrize('name, kw', [
    ('hstrain', dict(HStrain=[200., 20., 20.])),
    ('gstrain', dict(gStrain=[0.02, 0.002, 0.002])),
])
def test_b82_vs_easyspin(name, kw):
    """EasySpin picks Dinfh (78.7 % / 76.0 % off its own D2h spectrum); torchspin's
    default spectrum equals EasySpin's forced-D2h one."""
    c = _es_cases()[name]
    assert c['group'] == 'Dinfh'
    es_auto, es_d2h, es_d2h_91 = (np.asarray(c[k], float) for k in ('spc_auto', 'spc_d2h', 'spc_d2h_91'))
    assert _rel(es_auto, es_d2h) > 0.5                 # EasySpin's own B82 defect
    assert _rel(es_d2h, es_d2h_91) < 1e-2             # D2h [31 4] is converged
    sys = SpinSystem(S=0.5, g=[2.05, 2.05, 2.15], lw=0.3, **kw)
    exp = Experiment(mwFreq=9.5, Range=[300., 345.], nPoints=1024, Harmonic=1)
    y = pepper(sys, exp, Options(GridSize=[31, 4]))[1].numpy()
    cos = float(np.dot(y, es_d2h_91) / (np.linalg.norm(y) * np.linalg.norm(es_d2h_91)))
    assert cos > 0.999
    assert abs(np.abs(y).max() / np.abs(es_d2h_91).max() - 1) < 0.03
