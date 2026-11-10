"""torchspin.lft.ddpowh / lft.ddpowj: Telser's field-point powder programs on a SpinSystem, with level strain.

1. Self-contained: strain-weight layout (sum and product forms, DDPOWJ-10's S = 1/2 weight), pair selection,
   exact identities of the strain model, level selectivity, agreement with ``pepper`` when strain is off
   (tolerances from the grid calibration in docs/lft_gap_analysis.md), NQR/NMR closed forms, the w and U
   operators, and gradients (autograd vs central differences; forward mode with dual numbers equals reverse
   mode).  ``torch.func.jacfwd`` does not apply: its vmap cannot run the data-dependent line windows.
2. Parity with telser-tools (skipped when it is absent): every golden DDPOWH (70) and DDPOWJ (33) input of
   telser-tools tests/reference, converted with ``telser2torchspin.ddpowh2sys/ddpowj2sys(target='lft')``,
   matches telser-tools' exact mode to 1e-10 of the spectrum maximum (measured <= 4.5e-11), with the
   single-option switches; and the same inputs with strain off agree with ``pepper`` as the converter's
   pepper path does.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.autograd.forward_ad as fwAD

from torchspin import Experiment, Options, SpinSystem, pepper
from torchspin.lft import ddpowh, ddpowj
from torchspin.lft import telser2torchspin as t2t

F64 = torch.float64
GOLD = json.loads((Path(__file__).parent / "data" / "telser" / "ddpow_golden_inputs.json").read_text())

pytestmark = pytest.mark.filterwarnings("ignore:Sparse CSR tensor support:UserWarning")


def _norm_dev(ref, y):
    ref = ref.detach().numpy() if isinstance(ref, torch.Tensor) else np.asarray(ref)
    y = y.detach().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
    a, b = ref / np.abs(ref).max(), y / np.abs(y).max()
    return float(np.abs(a - b).max()), float(np.corrcoef(a, b)[0, 1])


def _central(f, p, h):
    cols = []
    for k in range(p.numel()):
        e = torch.zeros_like(p)
        e[k] = h[k] if isinstance(h, (list, tuple)) else h
        cols.append((f(p + e) - f(p - e)) / (2 * e[k]))
    return torch.stack(cols, -1)


def _check_grad(f, p, h, rtol=1e-6):
    """Reverse-mode Jacobian vs central differences (relative to the largest entry) and vs forward mode."""
    J = torch.autograd.functional.jacobian(f, p)
    with torch.no_grad():
        Jc = _central(f, p.detach().clone(), h)
    scale = J.abs().max()
    assert scale > 0
    assert float((J - Jc).abs().max() / scale) < rtol, float((J - Jc).abs().max() / scale)
    cols = []
    for k in range(p.numel()):
        with fwAD.dual_level():
            t = torch.zeros_like(p)
            t[k] = 1.0
            cols.append(fwAD.unpack_dual(f(fwAD.make_dual(p.detach(), t))).tangent)
    Jf = torch.stack(cols, -1)
    assert float((Jf - J).abs().max() / scale) < 1e-12
    return J


# ============================================================================== 1. self-contained
def test_strain_weights_sum_form_layout():
    sy = SpinSystem(S=1.5, g=[2.0, 2.0, 2.0], Nucs="1H", A=[10.0, 10.0, 10.0])
    w = ddpowh.strain_weights(sy, ms=[1.0, 2.0, 3.0, 4.0], mi=[0.1, 0.2])
    np.testing.assert_allclose(w.numpy(), [1.1, 1.2, 2.1, 2.2, 3.1, 3.2, 4.1, 4.2], rtol=1e-15)
    s12 = SpinSystem(S=0.5, g=[2.0, 2.0, 2.0])
    np.testing.assert_allclose(ddpowh.strain_weights(s12, mi=[0.3]).numpy(), [0.3, 0.3])   # Telser's WMI(1), no I
    with pytest.raises(ValueError):
        ddpowh.strain_weights(sy, ms=[1.0, 2.0])
    with pytest.raises(ValueError):
        ddpowh.strain_weights(s12, mi=[0.3, 0.4])


def test_pair_selection():
    assert ddpowh.manifold_pairs((1, 1, 2, 2), 2) == ((0, 2), (0, 3), (1, 2), (1, 3))
    assert ddpowh.manifold_pairs((1, 2, 1, 2), 1) == ((0, 1),)
    assert ddpowh.all_pairs(3) == ((0, 1), (0, 2), (1, 2))
    assert ddpowj.level_range_pairs((1, 2, 2, 4), 4) == ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3))
    assert ddpowj.level_range_pairs((1, 1, 2, 9), 4) == ((0, 1), (0, 2), (0, 3))


def test_ddpowj_strain_weights_product_and_s12_weight():
    ni = SpinSystem(S=[1.0, 1.0], g=[[2.0] * 3] * 2, ee=[1000.0])
    w = ddpowj.strain_weights(ni, [1.0, 2.0, 3.0], [0.5, 0.0, 2.0])
    np.testing.assert_allclose(w.numpy(), np.outer([1.0, 2.0, 3.0], [0.5, 0.0, 2.0]).ravel())
    cu = SpinSystem(S=[0.5, 0.5], g=[[2.0] * 3] * 2, ee=[1000.0])
    np.testing.assert_array_equal(ddpowj.strain_weights(cu).numpy(), np.zeros(4))           # exact: WMS = 0
    np.testing.assert_array_equal(ddpowj.strain_weights(cu, s12_weight=1.0).numpy(), np.ones(4))   # Fortran
    one = SpinSystem(S=1.0, g=[2.0] * 3)                                                    # S1 = 1 with an S = 0 site
    np.testing.assert_allclose(ddpowj.strain_weights(one, [1.0, 2.0, 3.0], s12_weight=1.0).numpy(), [1.0, 2.0, 3.0])


FIELDS = np.linspace(280.0, 380.0, 201)


def _rhombic():
    return SpinSystem(S=0.5, g=[1.95, 2.05, 2.25])


def test_uniform_strain_weight_is_a_width_factor():
    """w_b = c for every basis state: every level has sqrt(sum |c_b|^2 c) = sqrt(c), so W -> W (1 + sqrt(c))."""
    sy = _rhombic()
    c = 0.44
    st = ddpowh.Strain(torch.full((2,), c, dtype=F64))
    a = ddpowh.epr(sy, FIELDS, 9.5, width=30.0, strain=st, igloo=20, extent="octant").spectrum
    b = ddpowh.epr(sy, FIELDS, 9.5, width=30.0 * (1 + math.sqrt(c)), igloo=20, extent="octant").spectrum
    np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-12, atol=1e-14 * float(b.abs().max()))
    z = ddpowh.epr(sy, FIELDS, 9.5, width=30.0, strain=ddpowh.Strain(torch.zeros(2, dtype=F64), b=0.0, field=1.0),
                   igloo=20, extent="octant").spectrum
    np.testing.assert_allclose(z.numpy(), ddpowh.epr(sy, FIELDS, 9.5, width=30.0, igloo=20, extent="octant")
                               .spectrum.numpy(), rtol=1e-13)


def test_level_strain_is_level_selective():
    """S = 3/2, D >> h nu: at X band only the m_S = +-1/2 Kramers doublet absorbs.  Strain on the +-3/2 states
    barely changes the spectrum (their admixture into the doublet is ~(g mu_B B/2D)^2), strain on +-1/2
    broadens it by about 1 + sqrt(w)."""
    sy = SpinSystem(S=1.5, g=[2.0] * 3, D=[-1e5, -1e5, 2e5])                  # D = 3e5 MHz = 10 cm^-1
    fields = np.linspace(100.0, 400.0, 301)
    kw = dict(width=80.0, igloo=20, extent="octant", pairs=((0, 1),))
    ref = ddpowh.epr(sy, fields, 9.5, **kw).spectrum
    outer = ddpowh.epr(sy, fields, 9.5, strain=ddpowh.Strain(ddpowh.strain_weights(sy, ms=[1.0, 0, 0, 1.0])),
                       **kw).spectrum
    inner = ddpowh.epr(sy, fields, 9.5, strain=ddpowh.Strain(ddpowh.strain_weights(sy, ms=[0, 1.0, 1.0, 0])),
                       **kw).spectrum
    # admixture of +-3/2 into the doublet <= (g mu_B B / 2D)^2 ~ 3.5e-4 at 400 mT: the width factor moves by at most
    # sqrt(3.5e-4)/2 ~ 1 % (measured 0.3 % of the peak)
    assert _norm_dev(ref.numpy(), outer.numpy())[0] < 1e-2
    assert float(inner.max() / ref.max()) < 0.75


def test_ddpowj_s12_partner_strain_is_exactly_off_or_a_doubling():
    """Cu(II)2 triplet (S1 = S2 = 1/2): exact weights WMS1 WMS2 = 0 leave W(n); the Fortran's WMS = 1 gives
    1 + (1 + 1)/2 = 2 (DDPOWJ-10)."""
    sy = SpinSystem(S=[0.5, 0.5], g=[[2.05, 2.05, 2.2]] * 2, ee=[3000.0, 3000.0, 2400.0])
    f = np.linspace(260.0, 380.0, 121)
    kw = dict(width=60.0, igloo=12, extent="octant")
    base = ddpowj.epr(sy, f, 9.5, **kw).spectrum
    ex = ddpowj.epr(sy, f, 9.5, strain=ddpowj.Strain(ddpowj.strain_weights(sy), b=0.0), **kw).spectrum
    np.testing.assert_allclose(ex.numpy(), base.numpy(), rtol=1e-13, atol=1e-15 * float(base.abs().max()))
    ft = ddpowj.epr(sy, f, 9.5, strain=ddpowj.Strain(ddpowj.strain_weights(sy, s12_weight=1.0)), **kw).spectrum
    dbl = ddpowj.epr(sy, f, 9.5, width=120.0, igloo=12, extent="octant").spectrum
    np.testing.assert_allclose(ft.numpy(), dbl.numpy(), rtol=1e-12, atol=1e-14 * float(dbl.abs().max()))


def test_field_exponent_scales_width_with_field():
    """(B/B_set)^b at one orientation: an isotropic S = 1/2 line at B_res with b = 1, B_set = B_res is the
    unstrained line; with B_set = B_res/2 it is the line of twice the width (at the resonance field)."""
    sy = SpinSystem(S=0.5, g=[2.0] * 3)
    f = np.linspace(330.0, 349.0, 381)
    one = dict(angle_range=(0.0, 0.0, 0.0, 0.0), igloo=1)
    b_res = 9.5e3 / (ddpowh.TORCHSPIN_CONSTANTS.mu_b * 2.0)
    zero = torch.zeros(2, dtype=F64)
    a = ddpowh.epr(sy, f, 9.5, width=20.0, strain=ddpowh.Strain(zero, 1.0, b_res), **one).spectrum
    b = ddpowh.epr(sy, f, 9.5, width=20.0, **one).spectrum
    k = int(np.argmin(np.abs(f - b_res)))
    assert float(a[k]) == pytest.approx(float(b[k]), rel=1e-4)
    a2 = ddpowh.epr(sy, f, 9.5, width=20.0, strain=ddpowh.Strain(zero, 1.0, b_res / 2), **one).spectrum
    assert float(a2[k]) == pytest.approx(float(b[k]) / 2, rel=1e-3)                     # 1/W peak height


# pepper comparisons: strain off, no Flygare factor.  The systems, grids, field steps and tolerances are those of
# the calibrated converter tests (test_lft_telser2torchspin: gen_rhombic_abs, gen_cu_abs, s1_D, cu2_triplet), built
# here with the converters' keyword form so that no telser-tools input is needed.
def _pepper_case(case, **kw):
    if case == "rhombic":
        return t2t.spin2sys(s=0.5, g=(1.95, 2.05, 2.25), **kw), np.linspace(290.0, 360.0, 351), 20.0, 60
    if case == "cu":
        sy = t2t.spin2sys(s=0.5, g=(2.05, 2.05, 2.25), nucleus=dict(spin=1.5, g_n=1.4848971, A=(60.0, 60.0, 550.0),
                                                                     label="63Cu"), **kw)
        return sy, np.linspace(260.0, 350.0, 151), 40.0, 60
    if case == "s1_D":
        return t2t.spin2sys(s=1.0, g=(2.0, 2.0, 2.0), D=0.05, E=0.01, **kw), np.linspace(200.0, 460.0, 301), 30.0, 60
    site = dict(s=0.5, g=(2.05, 2.06, 2.25))
    return (t2t.dimer2sys(site1=site, site2=site, J=-5.0, Ddip=0.06, Edip=0.005, **kw), np.linspace(240.0, 430.0, 401),
            60.0, 40)


@pytest.mark.parametrize("case,tol", [("rhombic", 3e-3), ("cu", 1e-2), ("s1_D", 1.5e-2), ("cu2", 3e-3)])
def test_matches_pepper_without_strain(case, tol):
    sy, f, w, igl = _pepper_case(case)
    y = ddpowh.epr(sy, f, 9.5, width=w, igloo=igl, extent="octant", frequency_factor=False).spectrum
    _, yp = pepper(_pepper_case(case, HStrain=[2 * w] * 3)[0], Experiment(mwFreq=9.5, Range=[f[0], f[-1]], nPoints=len(f), Harmonic=0),
                   Options(GridSize=91, Threshold=0.0))
    dev, corr = _norm_dev(y, yp)
    assert dev < tol and corr > 0.999, (dev, corr)


def test_nqr_lines_at_closed_form_frequencies():
    """14N at zero field: I.Q.I with traceless principal values Q_k has levels -Q_k, so the lines are |Q_i - Q_j|."""
    Q = [-0.6, -0.4, 1.0]
    sy = SpinSystem(S=[], Nucs="14N", Q=Q)
    rf = np.linspace(0.0, 2.0, 2001)
    y = ddpowh.nmr(sy, rf, 0.0, nmr_width=0.01, igloo=10).spectrum.numpy()
    expect = sorted({abs(Q[i] - Q[j]) for i in range(3) for j in range(i + 1, 3)})
    peaks = rf[1:-1][(y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]) & (y[1:-1] > 0.05 * y.max())]
    np.testing.assert_allclose(sorted(peaks), expect, atol=1.1e-3)


def test_nmr_shielding_shifts_the_larmor_line():
    """1H at 340 mT: sigma = 5 kHz along every axis moves the line from nu_N to nu_N - 5 kHz exactly."""
    sy = SpinSystem(S=[], Nucs="1H")
    b = 340.0
    nu = float(ddpowh._nuclear_larmor(ddpowh._Model(sy, zeeman2=None, h0_extra=None, moment="full",
                                                    constants=ddpowh.TORCHSPIN_CONSTANTS, device="cpu"), b))
    rf = nu + np.linspace(-0.05, 0.05, 1001)
    kw = dict(nmr_width=0.002, igloo=4)
    y0 = ddpowh.nmr(sy, rf, b, **kw).spectrum.numpy()
    y1 = ddpowh.nmr(sy, rf, b, shielding_khz=(5.0, 5.0, 5.0), **kw).spectrum.numpy()
    assert rf[np.argmax(y0)] == pytest.approx(nu, abs=1e-4)
    assert rf[np.argmax(y1)] == pytest.approx(nu - 5e-3, abs=1e-4)


def test_second_order_zeeman_operator_closed_form():
    """B^2 n.Z2.n = w (mu_B B n.S)^2: for B || z, the m-diagonal w (mu_B B m)^2 (w in 1/cm^-1, result MHz)."""
    sy = SpinSystem(S=1.0, g=[2.0] * 3)
    w = 0.3                                                          # 1/cm^-1
    Z2 = t2t.second_order_zeeman_operator(sy, w)
    mub = t2t.MU_B_MHZ_PER_MT
    cm = ddpowh.TORCHSPIN_CONSTANTS.mhz_per_cm
    b = 500.0
    diag = (b * b * Z2[2, 2]).diagonal().real.numpy()
    expect = [w * (mub / cm * b * m) ** 2 * cm for m in (1.0, 0.0, -1.0)]
    np.testing.assert_allclose(diag, expect, rtol=1e-13)


def test_u_term_operator_is_hermitian_and_cubic_invariant():
    sy = SpinSystem(S=1.5, g=[2.0] * 3, Nucs="1H")
    U = t2t.u_term_operator(sy, 2.0)
    assert torch.allclose(U, U.conj().T)
    assert float(U.abs().max()) > 0
    with pytest.raises(ValueError):
        t2t.u_term_operator(SpinSystem(S=1.0, g=[2.0] * 3, Nucs="1H"), 2.0)


# ------------------------------------------------------------------------------ gradients
def test_gradients_s1_strain_populations_derivative():
    """S = 1: g, D, E, widths, strain weights, b, B_set, mw frequency, temperature; derivative spectrum."""
    f = np.linspace(260.0, 420.0, 81)

    def run(p):
        sy = SpinSystem(S=1.0, g=p[0:3], D=torch.stack([-p[3] / 3 + p[4], -p[3] / 3 - p[4], 2 * p[3] / 3]))
        st = ddpowh.Strain(ddpowh.strain_weights(sy, ms=p[5:8]), p[8], p[9])
        return ddpowh.epr(sy, f, p[10], width=p[11:14], strain=st, temperature=p[14], derivative=True,
                          igloo=6, extent="octant").spectrum

    p = torch.tensor([2.0, 2.02, 2.05, 900.0, 120.0, 0.3, 0.1, 0.5, 0.7, 330.0, 9.5, 60.0, 70.0, 80.0, 4.0],
                     dtype=F64)
    _check_grad(run, p, [1e-6] * 3 + [1e-3, 1e-3] + [1e-6] * 4 + [1e-4, 1e-7] + [1e-4] * 3 + [1e-5], rtol=2e-6)


def test_gradients_hyperfine_euler_free_tensor_and_lorentzian():
    """Cu(II) with a full A tensor (rotation angle on the graph), Lorentzian lines, hemisphere grid."""
    f = np.linspace(270.0, 350.0, 81)

    def run(p):
        c, s = torch.cos(p[3]), torch.sin(p[3])
        R = torch.stack([torch.stack([c, -s, 0 * c]), torch.stack([s, c, 0 * c]),
                         torch.tensor([0.0, 0.0, 1.0], dtype=F64)])
        A = R @ torch.diag(p[4:7]) @ R.T
        sy = SpinSystem(S=0.5, g=p[0:3], Nucs="63Cu", A=A)
        return ddpowh.epr(sy, f, 9.5, width=50.0, shape="lorentzian", cutoff=16, igloo=5).spectrum

    p = torch.tensor([2.04, 2.06, 2.25, 0.3, 50.0, 70.0, 480.0], dtype=F64)
    _check_grad(run, p, [1e-6, 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4], rtol=2e-6)


def test_gradients_w_and_u_terms():
    """S = 3/2: Telser's second-order Zeeman w and the U (I.S^3) hyperfine through zeeman2 / h0_extra."""
    f = np.linspace(100.0, 400.0, 61)
    base = SpinSystem(S=1.5, g=[2.0] * 3, D=[-1e5, -1e5, 2e5], Nucs="1H", A=[30.0, 30.0, 30.0])

    def run(p):
        return ddpowh.epr(base, f, 9.5, width=150.0, igloo=4, extent="octant",
                          zeeman2=t2t.second_order_zeeman_operator(base, p[0]),
                          h0_extra=t2t.u_term_operator(base, p[1])).spectrum

    _check_grad(run, torch.tensor([0.05, 20.0], dtype=F64), [1e-4, 1e-2])          # weak responses: larger steps


def test_gradients_fdmr_zero_field_and_endor():
    sy0 = lambda D: SpinSystem(S=1.0, g=[2.0] * 3, D=torch.stack([-D / 3, -D / 3, 2 * D / 3]))   # noqa: E731

    def fd(p):
        return ddpowh.fdmr(sy0(p[0]), np.linspace(0.0, 2000.0, 101), 0.0, width=p[1], igloo=4,
                           extent="octant").spectrum

    _check_grad(fd, torch.tensor([900.0, 80.0], dtype=F64), [1e-3, 1e-4])

    def en(p):
        sy = SpinSystem(S=0.5, g=[2.0, 2.0, 2.1], Nucs="14N", A=p[0:3], Q=torch.stack([-p[3] / 2, -p[3] / 2, p[3]]))
        return ddpowh.endor(sy, np.linspace(0.0, 30.0, 121), 330.0, 9.4, width=40.0, nmr_width=0.4,
                            igloo=6, extent="octant").spectrum

    _check_grad(en, torch.tensor([10.0, 12.0, 25.0, 1.0], dtype=F64), [1e-5] * 4, rtol=3e-6)


def test_gradients_nqr_and_shielding():
    def run(p):
        sy = SpinSystem(S=[], Nucs="14N", Q=torch.stack([p[0], p[1], -p[0] - p[1]]))
        return ddpowh.nmr(sy, np.linspace(0.5, 2.5, 101), 300.0, nmr_width=0.05, shielding_khz=p[2:5],
                          igloo=4).spectrum

    _check_grad(run, torch.tensor([-0.6, -0.4, 1.0, 2.0, 3.0], dtype=F64), [1e-6] * 2 + [1e-4] * 3, rtol=2e-6)


def test_gradients_dimer_exchange_and_strain():
    f = np.linspace(250.0, 400.0, 61)

    def run(p):
        sy = SpinSystem(S=[1.0, 0.5], g=[[2.0, 2.0, 2.05], [2.05, 2.05, 2.2]],
                        D=torch.stack([torch.stack([-p[1] / 3, -p[1] / 3, 2 * p[1] / 3]), torch.zeros(3, dtype=F64)]),
                        ee=p[0].reshape(1))
        st = ddpowj.Strain(ddpowj.strain_weights(sy, p[2:5], None, s12_weight=p[5]))
        return ddpowj.epr(sy, f, 9.5, width=70.0, strain=st, igloo=5, extent="octant").spectrum

    _check_grad(run, torch.tensor([4000.0, 3000.0, 0.2, 0.1, 0.4, 0.8], dtype=F64), [1e-2, 1e-2] + [1e-6] * 4,
                rtol=3e-6)


# ============================================================================== 2. telser-tools parity
def _tt(name):
    return pytest.importorskip(f"telser_tools.{name}")


@pytest.fixture(scope="module")
def C():
    return pytest.importorskip("telser_tools.spin").CODATA2018


def _scale_for(sp):
    """lft's field derivatives are per mT, Telser's per G."""
    return 0.1 if sp["func"] == "epr" and sp["kwargs"].get("derivative") else 1.0


def _assert_parity(ref_spectra, conv, out, rtol=1e-10):
    assert len(ref_spectra) == len(out)
    for a, b, sp in zip(ref_spectra, out, conv.extras["spectra"]):
        np.testing.assert_allclose(np.asarray(sp["x"]), np.asarray(a.x), rtol=1e-14, atol=1e-12)
        y = b.spectrum.detach().numpy() * _scale_for(sp)
        sc = np.abs(a.y).max()
        if sc < 1e-20:                                   # an exactly vanishing spectrum (parallel mode, iso g)
            assert np.abs(y).max() < 1e-25
            continue
        assert np.abs(a.y - y).max() <= rtol * sc, np.abs(a.y - y).max() / sc


_SLOW = {"s52_aFb_rot"}


@pytest.mark.parametrize("name", [pytest.param(n, marks=pytest.mark.slow) if n in _SLOW else n
                                  for n in GOLD["ddpowh"]])
def test_ddpowh_golden_inputs_match_telser_tools(name, C):
    nd = _tt("ddpowh")
    inp = nd.read_input(GOLD["ddpowh"][name])
    conv = t2t.ddpowh2sys(inp, constants=C, target="lft")
    assert conv.extras["dropped"] == []
    with torch.no_grad():
        out = ddpowh.simulate(conv)
    _assert_parity(nd.simulate(inp).spectra, conv, out)


@pytest.mark.parametrize("name", list(GOLD["ddpowj"]))
def test_ddpowj_golden_inputs_match_telser_tools(name, C):
    nj = _tt("ddpowj")
    inp = nj.read_input(GOLD["ddpowj"][name])
    conv = t2t.ddpowj2sys(inp, constants=C, target="lft")
    with torch.no_grad():
        out = ddpowj.simulate(conv)
    _assert_parity(nj.simulate(inp).spectra, conv, out)


@pytest.mark.parametrize("telser,lft", [
    (dict(frequency_factor=False), dict(frequency_factor=False)),
    (dict(normalized_populations=False), dict(normalized_populations=False)),
    (dict(exact_cutoff=False), dict(exact_cutoff=False)),
    (dict(field_derivative=False), dict(field_derivative=False)),
    (dict(grid="fortran"), dict(grid="fortran")),
])
def test_ddpowh_option_switches(telser, lft, C):
    nd = _tt("ddpowh")
    for name in ("s32_pop_deriv", "rhombic_s12_lorentz_deriv", "s32_strain"):
        inp = nd.read_input(GOLD["ddpowh"][name])
        conv = t2t.ddpowh2sys(inp, constants=C, target="lft")
        with torch.no_grad():
            out = ddpowh.simulate(conv, **lft)
        ref = nd.simulate(inp, **telser).spectra
        if lft.get("field_derivative") is False and inp.epr_deriv:   # d/dnu: no per-gauss factor
            out = tuple(r._replace(spectrum=r.spectrum * 10.0) for r in out)
        _assert_parity(ref, conv, out)


def test_ddpowh_endor_option_switches(C):
    nd = _tt("ddpowh")
    for name, telser, lft in (("endor_flag_LG", dict(endor_shape_flag="epr"), dict(intensity_shape="epr")),
                              ("endor_s32_pop_strain", dict(endor_field_strain=False), dict(field_strain=False))):
        inp = nd.read_input(GOLD["ddpowh"][name])
        conv = t2t.ddpowh2sys(inp, constants=C, target="lft")
        with torch.no_grad():
            out = ddpowh.simulate(conv, **lft)
        _assert_parity(nd.simulate(inp, **telser).spectra, conv, out)


@pytest.mark.parametrize("sigma", [(5.0, 5.0, 5.0), (2.0, 4.0, 9.0)])
def test_nmr_shielding_matches_telser_tools(sigma, C):
    """The golden shielding input (nmr_h1_shield) shifts the line out of its +-10 kHz window in both programs;
    with shieldings of a few kHz the anisotropic shielded powder line is inside it (measured 9e-12)."""
    nd = _tt("ddpowh")
    inp = dataclasses.replace(nd.read_input(GOLD["ddpowh"]["nmr_h1_shield"]), sigma_khz=sigma)
    conv = t2t.ddpowh2sys(inp, constants=C, target="lft")
    assert conv.extras["spectra"][0]["func"] == "nmr"
    with torch.no_grad():
        out = ddpowh.simulate(conv)
    ref = nd.simulate(inp).spectra
    assert np.abs(ref[0].y).max() > 0
    _assert_parity(ref, conv, out)


@pytest.mark.parametrize("name", ["cu2_strain", "s1_s12_strain"])
def test_ddpowj_fortran_s12_strain_weight(name, C):
    """DDPOWJ-10: s12_strain_weight=1.0 reproduces telser-tools' option of the same name (the Fortran's WMS = 1)."""
    nj = _tt("ddpowj")
    inp = nj.read_input(GOLD["ddpowj"][name])
    conv = t2t.ddpowj2sys(inp, constants=C, target="lft", s12_strain_weight=1.0)
    with torch.no_grad():
        out = ddpowj.simulate(conv)
    ref = nj.simulate(inp, s12_strain_weight=1.0).spectra
    _assert_parity(ref, conv, out)
    if inp.strain:
        y0 = nj.simulate(inp).spectra[0].y
        assert np.abs(ref[0].y - y0).max() > 1e-3 * np.abs(y0).max()     # the option matters here


def test_converter_lft_target_maps_what_pepper_cannot(C):
    """Every input the pepper path refuses (strain, Lorentzian, mixed H1, S = 0) converts with target='lft'."""
    nd = _tt("ddpowh")
    for name in ("rhombic_s12_lorentz", "s32_strain", "mode_1_nrho8", "nqr_n14"):
        inp = nd.read_input(GOLD["ddpowh"][name])
        with pytest.raises(NotImplementedError):
            t2t.ddpowh2sys(inp)
        conv = t2t.ddpowh2sys(inp, target="lft")
        assert conv.extras["target"] == "lft" and conv.extras["dropped"] == []
    with pytest.raises(ValueError):
        t2t.ddpowh2sys(nd.read_input(GOLD["ddpowh"]["rhombic_s12"]), target="easyspin")
    with pytest.raises(ValueError):
        ddpowh.simulate(t2t.ddpowh2sys(nd.read_input(GOLD["ddpowh"]["rhombic_s12"])))


def test_default_constants_codata_2022(C):
    """Without constants= the lft run uses torchspin's CODATA 2022 mu_B: 1e-9-level shifts, not 1e-12."""
    nd = _tt("ddpowh")
    inp = nd.read_input(GOLD["ddpowh"]["cu_axial"])
    ref = nd.simulate(inp).spectra[0].y
    with torch.no_grad():
        y = ddpowh.simulate(t2t.ddpowh2sys(inp, target="lft"))[0].spectrum.numpy()
    d = np.abs(ref - y).max() / np.abs(ref).max()
    assert d < 1e-6


# pepper agreement of the lft engine on Telser inputs with strain off, the same cases and tolerances as the
# converter's pepper path (test_lft_telser2torchspin.test_ddpowh_field_sweeps_match_pepper / ddpowj)
def _w(w):
    return dict(epr_width=(w, w, w))


@pytest.mark.parametrize("prog,name,replace,tol", [
    ("h", "rhombic_s12", dict(igloo=60, epr_deriv=1), 4e-2),
    ("h", "s1_D", dict(igloo=60), 1.5e-2),
    ("j", "cu2_triplet", dict(igloo=40, **_w(60.0)), 3e-3),
    ("j", "cu2_pop_4K", dict(igloo=40, **_w(60.0)), 1e-2),
    ("j", "cu2_halffield", dict(igloo=40), 5e-3),
])
def test_lft_on_telser_inputs_matches_pepper(prog, name, replace, tol):
    mod = _tt("ddpowh" if prog == "h" else "ddpowj")
    inp = dataclasses.replace(mod.read_input(GOLD[f"ddpow{prog}"][name]), **replace)
    f = t2t.ddpowh2sys if prog == "h" else t2t.ddpowj2sys
    lm = ddpowh if prog == "h" else ddpowj
    with torch.no_grad():
        y = lm.simulate(f(inp, target="lft"), frequency_factor=False)[0].spectrum.numpy()
    cv = f(inp)
    _, yp = pepper(cv.sys, cv.exp, Options(GridSize=91, Threshold=0.0))
    dev, corr = _norm_dev(y, yp.numpy())
    assert dev < tol and corr > 0.999, (dev, corr)
