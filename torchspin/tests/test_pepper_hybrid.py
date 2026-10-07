"""Hybrid resonance fields: the properties that must hold regardless of EasySpin.

The EasySpin comparison lives in ``test_pepper_cupc_matlab_validation.py``.  What
is checked here is internal consistency, which is what tells a correct
implementation of an approximate method from a plausible-looking wrong one:

* it reduces to exact diagonalization as the perturbational coupling vanishes,
* it improves monotonically as nuclei move into the exact core,
* it conserves the total absorption intensity,
* a set of equivalent nuclei gives the same answer as writing them out, and
* the sub-line bookkeeping does not depend on how the grid is batched.

The test system is the reported one: axial Cu(II) with A_par about 647 MHz,
where second-order perturbation theory is not usable.
"""

import numpy as np
import pytest
import torch

from torchspin import Experiment, Options, SpinSystem, pepper
from torchspin.resfields_hybrid import core_nuclei_split

A_CU = [15.3311, 15.3311, 646.629]
G_CU = [2.04894, 2.04894, 2.181]
GRID = [91, 4]


def _exp(harmonic=1):
    return Experiment(mwFreq=9.347144, Range=[233.8, 433.8], nPoints=2667, Harmonic=harmonic)


def _sys(n_nitrogen, a_n=45.0, n=None, nucs=None):
    nucs = nucs or (['63Cu'] + ['14N'] * n_nitrogen)
    return SpinSystem(S=[0.5], g=[G_CU], Nucs=nucs,
                      A=[A_CU] + [[a_n, a_n, a_n]] * n_nitrogen, n=n,
                      lw=[0.542209, 0.0])


def _cos(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _hybrid_opt(core=(1,), **kw):
    return Options(Method='hybrid', HybridCoreNuclei=list(core), GridSize=GRID,
                   Verbosity=0, **kw)


def _spc(sys, opt, harmonic=1):
    return pepper(sys, _exp(harmonic), opt)[1].numpy()


@pytest.mark.parametrize('a_n, floor', [(45.0, 0.97), (20.0, 0.995), (5.0, 0.999), (1.0, 0.9998)])
def test_approaches_exact_as_coupling_vanishes(a_n, floor):
    """The decoupling is first order in the perturbational hyperfine coupling, so
    hybrid has to converge on the matrix result as that coupling shrinks."""
    sys = _sys(1, a_n=a_n)
    exact = _spc(sys, Options(GridSize=GRID, Verbosity=0))
    hybrid = _spc(sys, _hybrid_opt())
    assert _cos(hybrid, exact) > floor


def test_perturbation_theory_does_not_converge_the_same_way():
    """Context for the test above: second-order perturbation theory is wrong about
    the 647 MHz copper coupling itself, so shrinking the nitrogen coupling does
    not fix it, and hybrid must be the better approximation at every size."""
    worst_pt = 1.0
    for a_n in (45.0, 1.0):
        sys = _sys(1, a_n=a_n)
        exact = _spc(sys, Options(GridSize=GRID, Verbosity=0))
        cos_pt = _cos(_spc(sys, Options(Method='perturb', GridSize=GRID, Verbosity=0)), exact)
        cos_hy = _cos(_spc(sys, _hybrid_opt()), exact)
        assert cos_hy > cos_pt, f'A_N={a_n}: hybrid {cos_hy:.4f} should beat perturb {cos_pt:.4f}'
        worst_pt = min(worst_pt, cos_pt)
    assert worst_pt < 0.99, 'perturbation theory is expected to stay inaccurate here'


def test_enlarging_the_exact_core_improves_accuracy():
    sys = _sys(2)
    exact = _spc(sys, Options(GridSize=GRID, Verbosity=0))
    cos_cu = _cos(_spc(sys, _hybrid_opt(core=(1,))), exact)
    cos_cu_n = _cos(_spc(sys, _hybrid_opt(core=(1, 2))), exact)
    assert cos_cu_n > cos_cu


def test_total_intensity_is_independent_of_the_nuclei():
    """Adding nuclei redistributes intensity, it does not create or destroy it, so
    the integrated absorption spectrum must not move."""
    dx = (433.8 - 233.8) / 2666
    reference = _spc(_sys(0), Options(GridSize=GRID, Verbosity=0), harmonic=0).sum() * dx
    for n_nitrogen in (1, 2, 3):
        total = _spc(_sys(n_nitrogen), _hybrid_opt(), harmonic=0).sum() * dx
        assert total == pytest.approx(reference, rel=2e-3), f'{n_nitrogen} nitrogens'


def test_equivalent_nuclei_match_explicit_nuclei():
    explicit = _spc(_sys(4), _hybrid_opt())
    collapsed = _spc(_sys(1, n=[1, 4], nucs=['63Cu', '14N']), _hybrid_opt())
    assert _cos(collapsed, explicit) > 0.9999
    assert np.abs(collapsed).max() / np.abs(explicit).max() == pytest.approx(1.0, rel=2e-3)


def test_sub_line_bookkeeping_is_independent_of_batching():
    """A sub-line index has to mean the same line for every orientation, so a
    BatchSize that would split the grid must not silently change the answer."""
    sys = _sys(2)
    full = _spc(sys, _hybrid_opt())
    assert np.all(np.isfinite(full))
    with pytest.raises(ValueError, match='one batch'):
        _spc(sys, _hybrid_opt(BatchSize=4))


def test_component_without_nuclei_is_simulated_exactly():
    """Method='hybrid' must work on a list of components where only some carry
    nuclei — a metal centre plus a radical impurity is the common case."""
    metal = _sys(2)
    radical = SpinSystem(S=[0.5], g=[[2.003, 2.003, 2.003]], lw=[0.3, 0.0], weight=0.05)
    B, both = pepper([metal, radical], _exp(), _hybrid_opt())
    assert np.all(np.isfinite(both.numpy()))
    # The radical alone is unaffected by HybridCoreNuclei, which names a nucleus
    # it does not have.
    alone_hybrid = _spc(radical, _hybrid_opt())
    alone_matrix = _spc(radical, Options(GridSize=GRID, Verbosity=0))
    assert _cos(alone_hybrid, alone_matrix) > 1 - 1e-12


def test_core_nucleus_cannot_be_a_set_of_equivalent_nuclei():
    """Only the perturbational nuclei can carry a multiplicity: a core nucleus is
    diagonalized exactly, so its copies each need their own states."""
    sys = _sys(1, n=[1, 4], nucs=['63Cu', '14N'])
    with pytest.raises(ValueError, match='cannot be a set of equivalent'):
        _spc(sys, _hybrid_opt(core=(1, 2)))
    # Naming only the copper is fine: the nitrogens stay perturbational.
    assert np.all(np.isfinite(_spc(sys, _hybrid_opt(core=(1,)))))


def test_matrix_method_still_rejects_equivalent_nuclei():
    sys = _sys(1, n=[1, 4], nucs=['63Cu', '14N'])
    with pytest.raises(ValueError, match='equivalent nuclei'):
        pepper(sys, _exp(), Options(GridSize=[7, 4], Verbosity=0))


def test_unimplemented_perturbation_orders_are_rejected():
    """They used to be accepted and silently run as second order."""
    sys = _sys(0)
    for method in ('perturb3', 'perturb4', 'perturb5'):
        with pytest.raises(ValueError, match='does not implement'):
            pepper(sys, _exp(), Options(Method=method, GridSize=[7, 4], Verbosity=0))
    # The ones pepper does implement still work.
    for method in ('perturb', 'perturb1', 'perturb2'):
        spc = _spc(sys, Options(Method=method, GridSize=[7, 4], Verbosity=0))
        assert np.all(np.isfinite(spc))


def test_requested_perturbation_order_reaches_the_solver():
    """The order used to be dropped on the way to resfields_perturb, so every
    perturb* name ran second order.  First and second order must now differ, and
    'perturb' must mean second order as it does in EasySpin's pepper."""
    sys = _sys(1)
    opt = dict(GridSize=GRID, Verbosity=0)
    first = _spc(sys, Options(Method='perturb1', **opt))
    second = _spc(sys, Options(Method='perturb2', **opt))
    bare = _spc(sys, Options(Method='perturb', **opt))
    assert not np.allclose(first, second), 'first and second order gave identical spectra'
    np.testing.assert_allclose(bare, second, rtol=0, atol=0)
    # Second order has to be the better approximation to exact diagonalization.
    exact = _spc(sys, Options(GridSize=GRID, Verbosity=0))
    assert _cos(second, exact) > _cos(first, exact)


def test_core_nuclei_split_is_one_based_and_validated():
    sys = _sys(2)
    assert core_nuclei_split(sys, [1]) == ([0], [1, 2])
    assert core_nuclei_split(sys, 1) == ([0], [1, 2])
    assert core_nuclei_split(sys, [1, 3]) == ([0, 2], [1])
    assert core_nuclei_split(sys, None) == ([], [0, 1, 2])
    for bad in ([0], [4], [1, 9]):
        with pytest.raises(ValueError, match='out of range'):
            core_nuclei_split(sys, bad)


def test_hybrid_is_not_silently_differentiable():
    """pepper_autograd would otherwise run the matrix branch, which is a different
    forward model, not merely a missing gradient."""
    from torchspin.pepper_autograd import pepper_autograd
    sys = _sys(1)
    with pytest.raises(NotImplementedError, match='hybrid'):
        pepper_autograd(sys, _exp(), _hybrid_opt())
