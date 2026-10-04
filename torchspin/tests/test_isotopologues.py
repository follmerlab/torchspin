"""Ports of EasySpin's isotopologues_* tests (natural-abundance and custom isotope
mixtures, equivalent nuclei, A/Q scaling, frames, weights)."""
import math

import numpy as np
import pytest

from torchspin.isotopologues import isotopologues, multisetlist
from torchspin.nucdata import nucabund, nucgval, nucqmom
from torchspin.spinsystem import SpinSystem


def weights(iso):
    return np.array([s.weight for s in iso])


def test_abund_cu():  # isotopologues_abund_cu
    ab63, ab65 = float(nucabund('63Cu')), float(nucabund('65Cu'))
    assert np.allclose(weights(isotopologues('Cu')), [ab63, ab65], atol=1e-10)
    w0 = [ab63 ** 2, ab63 * ab65, ab63 * ab65, ab65 ** 2]
    for nucs in ('Cu,Cu', 'Cu,Cu,63Cu', 'Cu,63Cu,Cu', '63Cu,Cu,Cu'):
        assert np.allclose(weights(isotopologues(nucs)), w0, atol=1e-10), nucs
    assert np.allclose(weights(isotopologues('(63,65)Cu', 1, [0.5, 0.5])), [0.5, 0.5])
    assert np.allclose(weights(isotopologues('(63,65)Cu,1H', [1, 1], [[0.5, 0.5], None])), [0.5, 0.5])
    assert np.allclose(weights(isotopologues('(63,65)Cu,(1,2)H', [1, 1], [[0.5, 0.5], [0.9, 0.1]])),
                       [0.45, 0.05, 0.45, 0.05])


def test_c60():  # isotopologues_c60
    n = 60
    iso = isotopologues('C', n, None, 0.0)
    assert len(iso) == n + 1
    assert iso[0].Nucs == [] and not iso[0].n
    assert [s.n[0] for s in iso[1:]] == list(range(1, n + 1))
    ab13, ab12 = float(nucabund('13C')), float(nucabund('12C'))
    w = weights(iso)
    assert math.isclose(w[0], ab12 ** n, abs_tol=1e-7)
    assert math.isclose(w[1], ab12 ** (n - 1) * ab13 * 60, abs_tol=1e-7)
    assert math.isclose(w[2], ab12 ** (n - 2) * ab13 ** 2 * 60 * 59 / 2, abs_tol=1e-7)


def test_equiv():  # isotopologues_equiv
    iso = isotopologues('(63,65)Cu,1H', [2, 1], [[0.5, 0.5], None])
    assert np.allclose(weights(iso), [0.25, 0.5, 0.25])


def test_equiv12():  # isotopologues_equiv12
    n = 12
    iso = isotopologues('Cl', n, None, 0.0)
    assert len(iso) == 13
    a1, a2 = float(nucabund('35Cl')), float(nucabund('37Cl'))
    n2 = np.arange(n + 1); n1 = n - n2
    w0 = a1 ** n1 * a2 ** n2 * np.array([math.comb(n, int(k)) for k in n2])
    assert np.allclose(weights(iso), w0, rtol=1e-10)


def test_multiset_multiplicities_sum():
    kvec, mult = multisetlist(4, 3)
    assert kvec.shape == (15, 3) and mult.sum() == 3 ** 4


def test_multinuc():  # isotopologues_multinuc
    a11, a12 = 0.9, 0.1
    a21, a22 = 0.8, 0.2
    sys = SpinSystem(S=[0.5], Nucs='(1,2)H,(12,13)C', n=[2, 2], A=[[1, 1, 1], [1, 1, 1]],
                     Abund=[[a11, a12], [a21, a22]])
    ab0 = [a11**2*a21**2, a11**2*a21*a22*2, a11**2*a22**2, 2*a11*a12*a21**2,
           4*a11*a12*a21*a22, 2*a11*a12*a22**2, a12**2*a21**2, 2*a12**2*a21*a22, a12**2*a22**2]
    assert np.allclose(weights(isotopologues(sys, rel_threshold=0.0)), ab0, atol=1e-7)


def test_sys_aframe():  # isotopologues_sys_aframe
    sys = SpinSystem(S=[0.5], Nucs='C', n=[3], A=[[10, 10, 20]], AFrame=[[1, 2, 2]])
    iso = isotopologues(sys, rel_threshold=0.0)
    assert len(iso) == 4
    assert iso[0].Nucs == []
    for s in iso[1:]:
        assert s.AFrame.shape == (1, 3) and np.allclose(s.AFrame.numpy(), [[1, 2, 2]])


def test_sys_hfformat_and_scale():  # isotopologues_sys_hfformat / hfscale
    ratio = float(nucgval('2H')) / float(nucgval('1H'))
    iso = isotopologues(SpinSystem(S=[0.5], Nucs='H', A=[[1, 2, 3]]))
    assert np.allclose(iso[0].A.numpy(), [[1, 2, 3]]) and np.allclose(iso[1].A.numpy(), np.array([[1, 2, 3]]) * ratio)
    full = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    iso = isotopologues(SpinSystem(S=[0.5], Nucs='H', A=full))
    assert np.allclose(iso[0].A.numpy(), full) and np.allclose(iso[1].A.numpy(), full * ratio)
    iso = isotopologues(SpinSystem(S=[0.5], Nucs='H', A=10.0))
    assert np.allclose(iso[0].A.numpy(), 10.0) and np.allclose(iso[1].A.numpy(), 10.0 * ratio)


def test_sys_nonmagnetic():  # isotopologues_sys_nonmagnetic
    iso = isotopologues(SpinSystem(S=[0.5], Nucs='(12,13)C', A=5.0, Abund=[0.9, 0.1]))
    assert iso[0].Nucs == [] and iso[1].Nucs == ['13C']


def test_sys_qformat():  # isotopologues_sys_qformat (principal values and full tensor)
    ratio = float(nucqmom('65Cu')) / float(nucqmom('63Cu'))
    for Q in ([[1, 2, 4]], np.array([[3, 1, 2], [4, 5, 6], [7, 4, 3]], dtype=float)):
        iso = isotopologues(SpinSystem(S=[0.5], Nucs='Cu', A=[[50, 50, 400]], Q=Q))
        assert np.allclose(iso[0].Q.numpy(), np.asarray(Q, dtype=float), rtol=1e-10)
        assert np.allclose(iso[1].Q.numpy(), np.asarray(Q, dtype=float) * ratio, rtol=1e-10)


def test_sys_twoel():  # isotopologues_sys_twoel
    sys = SpinSystem(S=[0.5, 1], g=[[2, 2, 2], [2.2, 2.2, 2.2]], Nucs='H,H',
                     A=[[10, 10, 20, 0, 0, 0], [0, 0, 0, 8, 8, 3]])
    for s in isotopologues(sys):
        assert tuple(s.A.shape) == (2, 6)


def test_sys_weight():  # isotopologues_sys_weight
    w = 0.8
    iso = isotopologues(SpinSystem(S=[0.5], g=2.0, Nucs='(1,2)H', A=10.0, Abund=[w, 1 - w], weight=0.1))
    assert math.isclose(iso[0].weight, w * 0.1) and math.isclose(iso[1].weight, (1 - w) * 0.1)


def test_sys_nonuc():  # isotopologues_sys_nonuc
    iso = isotopologues(SpinSystem(S=[0.5], g=2.0))
    assert len(iso) == 1 and iso[0].weight == 1
    iso = isotopologues('', 3, None)
    assert len(iso) == 1 and iso[0].weight == 1


def test_sys_qframe():  # isotopologues_sys_qframe
    sys = SpinSystem(S=[0.5], Nucs='Cu', n=[3], A=[[10, 10, 20]], Q=[[10, 10, -20]], QFrame=[[1, 2, 2]])
    iso = isotopologues(sys)
    assert [s.QFrame.numel() for s in iso] == [3, 6, 6, 3]


def test_sys_hfsphericalformat():  # isotopologues_sys_hfsphericalformat (A_ = [aiso T rho])
    ratio = float(nucgval('2H')) / float(nucgval('1H'))
    for A_ in ([1.0], [1.0, 2.0], [1.0, 2.0, 3.0]):
        iso = isotopologues(SpinSystem(S=[0.5], Nucs='H', A_=[A_]))
        assert np.allclose(iso[1].A.numpy(), iso[0].A.numpy() * ratio)
    s = SpinSystem(S=[0.5], Nucs='1H', A_=[[5.0, 2.0, 0.3]])
    assert np.allclose(s.A.numpy(), [[5 - 2 - 0.3, 5 - 2 + 0.3, 5 + 4]])
