"""SpinSystem.HStrainL: Lorentzian strain in pepper (opt-in, not an EasySpin field).

EasySpin's strains are Gaussian, so there is no direct EasySpin reference.  The
references are:

1. the isotropic case, which must equal pepper's ``Sys.lw(2)`` Lorentzian
   (itself validated against EasySpin) exactly;
2. single orientations (``Exp.Orientations``, crystal mode, frequency sweep),
   which must be one analytic bin-integrated Lorentzian at the resonance with
   FWHM ``sqrt(sum_k (W_k n_k)^2) / (g_eff mu_B / h)``;
3. a brute-force powder sum (fine octant grid, analytic widths) for rhombic
   S = 1/2, with the same tolerance the Gaussian ``HStrain`` meets;
4. gradients of ``pepper_autograd`` against central differences.
"""
import dataclasses
import math

import numpy as np
import pytest
import torch

from torchspin import SpinSystem, Experiment, Options, pepper, nucspinrmv
from torchspin.constants import BMAGN, PLANCK
from torchspin.pepper_autograd import pepper_autograd
from torchspin.strainwidth import lorentzian_strain_system

MHZ_PER_MT = BMAGN / PLANCK * 1e-9          # MHz per mT for g = 1
G_RHOMBIC = [1.95, 2.05, 2.25]
W_RHOMBIC = [40.0, 80.0, 150.0]              # MHz FWHM along x, y, z


def _rel(a, b):
    return float((a - b).abs().max() / b.abs().max())


def _lorentz_bins(x, x0, fwhm):
    """Bin-integrated unit-area Lorentzian on the (uniform) axis x."""
    dx = x[1] - x[0]
    hw = fwhm / 2.0
    return (np.arctan((x + dx / 2 - x0) / hw) - np.arctan((x - dx / 2 - x0) / hw)) / np.pi


def _unit(phi, theta):
    return np.array([math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)])


# ---------------------------------------------------------------------------
# SpinSystem field
# ---------------------------------------------------------------------------

def test_field_normalization_and_validation():
    s = SpinSystem(S=[0.5], g=[2, 2, 2], HStrainL=50.0)
    assert torch.equal(s.HStrainL, torch.tensor([50.0, 50.0, 50.0], dtype=torch.float64))
    assert SpinSystem(S=[0.5], g=[2, 2, 2]).HStrainL is None
    with pytest.raises(ValueError, match="shape"):
        SpinSystem(S=[0.5], HStrainL=[1.0, 2.0])
    with pytest.raises(ValueError, match="non-negative"):
        SpinSystem(S=[0.5], HStrainL=[1.0, -2.0, 3.0])


def test_field_survives_copies():
    s = SpinSystem(S=[0.5], g=G_RHOMBIC, Nucs='1H', A=[5, 5, 5], HStrainL=W_RHOMBIC)
    assert torch.equal(dataclasses.replace(s, weight=0.5).HStrainL, s.HStrainL)
    assert torch.equal(nucspinrmv(s, [0]).HStrainL, s.HStrainL)


def test_tensor_keeps_graph():
    W = torch.tensor(W_RHOMBIC, dtype=torch.float64, requires_grad=True)
    s2, lor = lorentzian_strain_system(SpinSystem(S=[0.5], g=G_RHOMBIC, HStrainL=W))
    assert lor and s2.HStrainL is None and s2.HStrain.requires_grad


@pytest.mark.parametrize("other", ["HStrain", "gStrain", "AStrain", "DStrain"])
def test_combination_with_gaussian_strain_raises(other):
    kw = dict(S=[1.0], g=G_RHOMBIC, D=[300.0, 30.0], Nucs='1H', A=[5, 5, 5], HStrainL=W_RHOMBIC)
    kw[other] = {'HStrain': [10, 10, 10], 'gStrain': [0.01, 0.01, 0.01],
                 'AStrain': [1, 1, 1], 'DStrain': [10.0, 1.0]}[other]
    exp = Experiment(mwFreq=9.5, Range=[200, 450], nPoints=256)
    with pytest.raises(ValueError, match="HStrainL"):
        pepper(SpinSystem(**kw), exp)


# ---------------------------------------------------------------------------
# Default unchanged
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("extra", [dict(HStrain=W_RHOMBIC), dict(lw=[0.5, 0.3])])
def test_zero_hstrainl_is_bit_identical_to_none(extra):
    exp = Experiment(mwFreq=9.5, Range=[290, 360], nPoints=512, Harmonic=1)
    _, a = pepper(SpinSystem(S=[0.5], g=G_RHOMBIC, **extra), exp)
    _, b = pepper(SpinSystem(S=[0.5], g=G_RHOMBIC, HStrainL=[0.0, 0.0, 0.0], **extra), exp)
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# Isotropic: identical to Sys.lw(2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("harmonic", [0, 1, 2])
def test_isotropic_field_sweep_equals_lw_lorentzian(harmonic):
    g, W = 2.0, 60.0
    exp = Experiment(mwFreq=9.5, Range=[320, 360], nPoints=2001, Harmonic=harmonic)
    _, a = pepper(SpinSystem(S=[0.5], g=[g, g, g], HStrainL=W), exp)
    _, b = pepper(SpinSystem(S=[0.5], g=[g, g, g], lw=[0.0, W / (g * MHZ_PER_MT)]), exp)
    assert _rel(a, b) < 1e-12


def test_isotropic_frequency_sweep_equals_lw_lorentzian():
    exp = Experiment(Field=340.0, mwRange=[9.3, 9.75], nPoints=2001, Harmonic=0)
    _, a = pepper(SpinSystem(S=[0.5], g=[2, 2, 2], HStrainL=60.0), exp)
    _, b = pepper(SpinSystem(S=[0.5], g=[2, 2, 2], lw=[0.0, 60.0]), exp)   # Sys.lw in MHz here
    assert _rel(a, b) < 1e-12


def test_isotropic_mwphase_with_residual_lorentzian():
    """Lorentzian strain (dispersion admixture in the line) convolved with a
    residual Sys.lw(2) absorption Lorentzian = one Lorentzian of the summed
    width with the admixture.  The residual width is applied by FFT
    convolution, which truncates the slowly decaying (1/x) dispersion tails:
    1.3 % in a 40 mT window, 0.22 % in this 200 mT one (absorption 0.16 % and
    0.006 %), so the difference is the window, not the phase handling."""
    g, W = 2.0, 60.0
    exp = Experiment(mwFreq=9.5, Range=[240, 440], nPoints=4001, Harmonic=0, mwPhase=0.4)
    _, a = pepper(SpinSystem(S=[0.5], g=[g, g, g], HStrainL=W, lw=[0.0, 0.3]), exp)
    _, b = pepper(SpinSystem(S=[0.5], g=[g, g, g], lw=[0.0, W / (g * MHZ_PER_MT) + 0.3]), exp)
    assert _rel(a, b) < 6e-3


# ---------------------------------------------------------------------------
# Single orientations: one analytic Lorentzian with the anisotropic width
# ---------------------------------------------------------------------------

PHI, THETA = 0.4, 0.7


def _single_line_reference(x, field_sweep=True, B=330.0, nu=9.5):
    n = _unit(PHI, THETA)
    geff = np.linalg.norm(np.asarray(G_RHOMBIC) * n)
    w_mhz = np.sqrt(((np.asarray(W_RHOMBIC) * n) ** 2).sum())
    if field_sweep:
        return _lorentz_bins(x, nu * 1e3 / (geff * MHZ_PER_MT), w_mhz / (geff * MHZ_PER_MT))
    return _lorentz_bins(x, geff * MHZ_PER_MT * B / 1e3, w_mhz / 1e3)


def _shape_dev(y, ref):
    y = y.numpy()
    s = y.sum() / ref.sum()
    return np.abs(y / s - ref).max() / ref.max()


def test_orientation_list_single_line():
    exp = Experiment(mwFreq=9.5, Range=[290, 360], nPoints=1401, Harmonic=0, Orientations=[[PHI, THETA]])
    x, y = pepper(SpinSystem(S=[0.5], g=G_RHOMBIC, HStrainL=W_RHOMBIC), exp)
    assert _shape_dev(y, _single_line_reference(x.numpy())) < 1e-12


def test_crystal_single_line_along_z():
    exp = Experiment(mwFreq=9.5, Range=[290, 360], nPoints=1401, Harmonic=0, MolFrame=[0, 0, 0],
                     CrystalSymmetry='P1', SampleFrame=[[0, 0, 0]])
    x, y = pepper(SpinSystem(S=[0.5], g=G_RHOMBIC, HStrainL=W_RHOMBIC), exp)
    gz, wz = G_RHOMBIC[2], W_RHOMBIC[2]
    ref = _lorentz_bins(x.numpy(), 9.5e3 / (gz * MHZ_PER_MT), wz / (gz * MHZ_PER_MT))
    assert _shape_dev(y, ref) < 1e-12


def test_frequency_sweep_single_line():
    exp = Experiment(Field=330.0, mwRange=[8.6, 10.4], nPoints=1801, Harmonic=0, Orientations=[[PHI, THETA]])
    f, y = pepper(SpinSystem(S=[0.5], g=G_RHOMBIC, HStrainL=W_RHOMBIC), exp)
    assert _shape_dev(y, _single_line_reference(f.numpy(), field_sweep=False)) < 1e-12


def test_area_matches_gaussian_strain_up_to_lorentzian_tails():
    """Same intensity bookkeeping as HStrain: inside the window the Lorentzian
    loses exactly its tail area, (atan(b) - atan(a))/pi."""
    exp = Experiment(mwFreq=9.5, Range=[290, 360], nPoints=1401, Harmonic=0, Orientations=[[PHI, THETA]])
    x, yl = pepper(SpinSystem(S=[0.5], g=G_RHOMBIC, HStrainL=W_RHOMBIC), exp)
    _, yg = pepper(SpinSystem(S=[0.5], g=G_RHOMBIC, HStrain=W_RHOMBIC), exp)
    ref = _single_line_reference(x.numpy())
    assert abs(float(yl.sum() / yg.sum()) - ref.sum()) < 1e-9


# ---------------------------------------------------------------------------
# Powder: brute-force reference, all accumulation paths
# ---------------------------------------------------------------------------

def _powder_reference(x, lorentzian, n_t=300, n_p=300):
    """Octant midpoint grid, S = 1/2 rhombic g: positions h nu/(g_eff mu_B),
    rate ~ (perpendicular g^2)/g_eff (field sweep), analytic widths."""
    from scipy.special import erf
    g = np.asarray(G_RHOMBIC); W = np.asarray(W_RHOMBIC)
    th = (np.arange(n_t) + 0.5) * (np.pi / 2 / n_t)
    ph = (np.arange(n_p) + 0.5) * (np.pi / 2 / n_p)
    T, P = np.meshgrid(th, ph, indexing='ij')
    T, P = T.ravel(), P.ravel()
    n = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], 1)
    geff = np.linalg.norm(n * g, axis=1)
    B0 = 9.5e3 / (geff * MHZ_PER_MT)
    gx, gy, gz = g
    nx, ny, nz = n.T
    perp = (gx**2 * gy**2 * nz**2 + gy**2 * gz**2 * nx**2 + gz**2 * gx**2 * ny**2) / geff**2 + geff**2
    amp = np.sin(T) * perp / geff
    fw = np.sqrt(((W * n) ** 2).sum(1)) / (geff * MHZ_PER_MT)
    dx = x[1] - x[0]
    out = np.zeros_like(x)
    for s in range(0, B0.size, 5000):
        b, w, a = B0[s:s + 5000, None], fw[s:s + 5000, None], amp[s:s + 5000, None]
        if lorentzian:
            out += (a * (np.arctan((x + dx / 2 - b) / (w / 2)) - np.arctan((x - dx / 2 - b) / (w / 2)))).sum(0) / np.pi
        else:
            sg = w / math.sqrt(8 * math.log(2)) * math.sqrt(2)
            out += (a * (erf((x + dx / 2 - b) / sg) - erf((x - dx / 2 - b) / sg))).sum(0) / 2
    return out


@pytest.fixture(scope="module")
def powder_refs():
    x = np.linspace(290, 360, 1401)
    return x, _powder_reference(x, True), _powder_reference(x, False)


@pytest.mark.parametrize("symmetry", [None, 'Ci', 'C1'])
def test_powder_matches_brute_force_like_gaussian_strain(powder_refs, symmetry):
    """D2h (summation over facets), Ci (interpolated per-orientation fallback)
    and C1 (per-orientation) paths.  The Gaussian HStrain meets the same
    reference to 0.7 % (grid smoothing; measured), the Lorentzian to 0.6 %."""
    x, ref_l, ref_g = powder_refs
    exp = Experiment(mwFreq=9.5, Range=[290, 360], nPoints=1401, Harmonic=0)
    opt = Options(GridSize=[31, 4]) if symmetry is None else Options(GridSize=[31, 4], GridSymmetry=symmetry)
    _, yl = pepper(SpinSystem(S=[0.5], g=G_RHOMBIC, HStrainL=W_RHOMBIC), exp, opt)
    _, yg = pepper(SpinSystem(S=[0.5], g=G_RHOMBIC, HStrain=W_RHOMBIC), exp, opt)
    dev_l = _shape_dev(yl, ref_l)
    dev_g = _shape_dev(yg, ref_g)
    assert dev_l < 1e-2
    assert dev_l < 1.5 * dev_g
    # same absolute scale as the Gaussian strain (identical intensity bookkeeping)
    s_l = float(yl.numpy() @ ref_l / (ref_l @ ref_l))
    s_g = float(yg.numpy() @ ref_g / (ref_g @ ref_g))
    assert abs(s_l / s_g - 1) < 2e-3


def test_hybrid_matches_matrix_for_small_hyperfine():
    kw = dict(S=[0.5], g=G_RHOMBIC, Nucs='63Cu', A=[10, 10, 60], HStrainL=W_RHOMBIC)
    exp = Experiment(mwFreq=9.5, Range=[250, 360], nPoints=1101, Harmonic=1)
    _, ym = pepper(SpinSystem(**kw), exp)
    _, yh = pepper(SpinSystem(**kw), exp, Options(Method='hybrid'))
    assert _rel(yh, ym) < 1.5e-2      # measured 7e-3; HStrain (Gaussian): 8e-3


# ---------------------------------------------------------------------------
# Gradients (pepper_autograd; no forward mode in pepper, bug B7)
# ---------------------------------------------------------------------------

GRAD_CASES = [
    ('powder', dict(), Options(GridSize=[19, 4], GridSymmetry='D2h')),
    ('orientations', dict(Orientations=[[0.4, 0.7], [1.2, 1.3], [0.1, 0.2]],
                          OrientationWeights=[0.3, 0.5, 0.2]), Options()),
]


@pytest.mark.parametrize("label,exp_kw,opt", GRAD_CASES, ids=[c[0] for c in GRAD_CASES])
def test_gradients_match_central_differences(label, exp_kw, opt):
    exp = Experiment(mwFreq=9.5, Range=[290, 360], nPoints=701, Harmonic=1, **exp_kw)
    W0 = torch.tensor(W_RHOMBIC, dtype=torch.float64)
    g0 = torch.tensor(G_RHOMBIC, dtype=torch.float64)
    r = torch.randn(701, dtype=torch.float64, generator=torch.Generator().manual_seed(0))

    def f(W, g):
        return pepper_autograd(SpinSystem(S=[0.5], g=g, HStrainL=W), exp, opt)[1]

    W = W0.clone().requires_grad_()
    g = g0.clone().requires_grad_()
    y = f(W, g)
    # forward model equals pepper's
    _, yp = pepper(SpinSystem(S=[0.5], g=g0, HStrainL=W0), exp, opt)
    assert _rel(y.detach(), yp) < 1e-12
    (y @ r).backward()
    for base, grad, h, which in ((W0, W.grad, 1e-3, 0), (g0, g.grad, 1e-6, 1)):
        fd = []
        for i in range(3):
            e = torch.zeros(3, dtype=torch.float64)
            e[i] = h
            args_p = (base + e, g0) if which == 0 else (W0, base + e)
            args_m = (base - e, g0) if which == 0 else (W0, base - e)
            fd.append(float((f(*args_p) - f(*args_m)) @ r) / (2 * h))
        fd = torch.tensor(fd, dtype=torch.float64)
        assert float((grad - fd).abs().max() / fd.abs().max()) < 1e-6


def test_pepper_routes_grad_inputs_to_autograd():
    exp = Experiment(mwFreq=9.5, Range=[290, 360], nPoints=301, Harmonic=1)
    W = torch.tensor(W_RHOMBIC, dtype=torch.float64, requires_grad=True)
    _, y = pepper(SpinSystem(S=[0.5], g=G_RHOMBIC, HStrainL=W), exp, Options(GridSize=[13, 4]))
    assert y.requires_grad
    y.sum().backward()
    assert W.grad is not None and torch.isfinite(W.grad).all()


# ---------------------------------------------------------------------------
# Fitting front ends know the field
# ---------------------------------------------------------------------------

def test_esfit_and_ml_know_the_field():
    from torchspin.esfit import TorchSpinParameterHandler
    from torchspin import ml
    assert 'HStrainL' in TorchSpinParameterHandler._NONNEGATIVE_FIELDS
    assert 'HStrainL' in ml._SYS_FIELDS
    kw = ml._template_kwargs(SpinSystem(S=[0.5], g=G_RHOMBIC, HStrainL=W_RHOMBIC))
    assert torch.equal(torch.as_tensor(kw['HStrainL']), torch.tensor(W_RHOMBIC, dtype=torch.float64))
