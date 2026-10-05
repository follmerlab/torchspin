"""
Transition-level validation of strain widths against EasySpin ``resfields``.

Reference: tests/data/strain_ref_widths.mat (generate_strain_widths_refs.m):
for five multi-electron / correlated-D-strain systems and six single-crystal
field directions, EasySpin's resonance fields, level pairs and strain widths
(mT).  torchspin's :func:`compute_strain_widths` is evaluated at exactly the
same field direction, level pair and resonance field.

Tolerance: EasySpin (resfields.m ~1147) builds U and V by *linear
interpolation* of the eigenvectors at the two knots of the field segment that
brackets the resonance (re-diagonalizing only for unstable states), whereas
torchspin diagonalizes at the resonance field itself.  That interpolation
limits the agreement to ~1e-4..1e-3 relative (largest for strongly mixed
states, e.g. two S=1/2 with ee = 1000 MHz at oblique directions).  Observed
maximum over the five systems x six directions: 1.3e-3; asserted: 3e-3.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io import loadmat

from torchspin.spinsystem import SpinSystem
from torchspin.ham import ham
from torchspin.rotations import erot
from torchspin.strainwidth import compute_strain_widths, compute_strain_widths_batch

REF_FILE = Path(__file__).parent.parent.parent / "tests" / "data" / "strain_ref_widths.mat"
MWFREQ = 9.5  # GHz


@pytest.fixture(scope="module")
def cases():
    if not REF_FILE.exists():
        pytest.skip(f"Reference data not found: {REF_FILE}")
    refs = loadmat(str(REF_FILE), squeeze_me=True, struct_as_record=False)
    return {str(c.name): c for c in np.atleast_1d(refs['cases'])}


def _get(o, name, default=None):
    return getattr(o, name) if name in getattr(o, '_fieldnames', []) else default


def sys_from_mat(S) -> SpinSystem:
    Sv = np.atleast_1d(np.asarray(S.S, dtype=float)).tolist()
    n_e = len(Sv)
    kw = {'S': Sv, 'lw': [0.0, 0.0]}
    g = np.asarray(S.g, dtype=float)
    if g.ndim == 1 and g.size == n_e and n_e > 1:
        g = np.repeat(g.reshape(n_e, 1), 3, axis=1)
    kw['g'] = g.tolist() if g.ndim else float(g)
    for name in ('gFrame', 'gStrain', 'DFrame', 'DStrain'):
        v = _get(S, name)
        if v is not None:
            kw[name] = np.asarray(v, dtype=float).tolist()
    D = _get(S, 'D')
    if D is not None:
        D = np.asarray(D, dtype=float)
        kw['D'] = [D.tolist()] if (D.ndim == 1 and n_e == 1) else D.tolist()
    r = _get(S, 'DStrainCorr')
    if r is not None:
        kw['DStrainCorr'] = np.atleast_1d(np.asarray(r, dtype=float)).tolist()
    ee = _get(S, 'ee')
    if ee is not None:
        kw['ee'] = [float(ee)]
    return SpinSystem(**kw)


def _angles(n):
    n = np.asarray(n, dtype=float); n = n / np.linalg.norm(n)
    theta = math.acos(max(-1.0, min(1.0, n[2])))
    phi = math.atan2(n[1], n[0])
    return phi, theta


def _resonance_mismatch(H0, mux, muy, muz, n, B0, u, v):
    """|E_v - E_u - h*mwFreq| (MHz) at field B0 along n."""
    H = H0 - B0 * (n[0] * mux + n[1] * muy + n[2] * muz)
    E = torch.linalg.eigvalsh(H)
    return abs(float(E[v] - E[u]) - MWFREQ * 1e3)


def _field_direction(c, o, H0, mux, muy, muz, pos, trans):
    """Field direction in the molecular frame for reference orientation ``o``.

    The reference stores R.'*z for R = erot(SampleFrame); the alternative
    convention R*z is tried too, and the one satisfying the resonance
    condition is used.  EasySpin's positions come from a cubic-spline model
    of the energy levels, so the condition holds only to ~1 MHz.
    """
    R = erot(np.asarray(c.Exp.SampleFrame, dtype=float).tolist()).numpy()
    candidates = [
        np.asarray(o.dir, dtype=float).reshape(3),      # intended direction (SampleFrame = [0 -theta -phi])
        np.asarray(o.nB_M, dtype=float).reshape(3),     # R.' * z
        R @ np.array([0.0, 0.0, 1.0]),                  # R * z
    ]
    u0, v0 = int(trans[0, 0]) - 1, int(trans[0, 1]) - 1
    mism = [_resonance_mismatch(H0, mux, muy, muz, n, float(pos[0]), u0, v0) for n in candidates]
    k = int(np.argmin(mism))
    assert mism[k] < 1.0, f"field-direction convention could not be established (mismatches {mism} MHz)"
    return candidates[k]


def _iter_orientations(c):
    for o in np.atleast_1d(c.oris):
        pos = np.atleast_1d(np.asarray(o.Pos, dtype=float))
        wid = np.atleast_1d(np.asarray(o.Wid, dtype=float))
        trans = np.atleast_2d(np.asarray(o.Trans, dtype=int))
        yield o, pos, wid, trans


@pytest.mark.parametrize("name", [
    'two_spinhalf_gstrain_both', 'two_spinhalf_gstrain_second',
    'triplet_dstrain_tilted', 'two_triplets_dstrain_corr', 'triplet_dstrain_corr',
])
def test_strain_widths_vs_easyspin(cases, name):
    c = cases[name]
    sys = sys_from_mat(c.Sys)
    H0, mux, muy, muz = ham(sys)
    n_checked = 0
    for o, pos, wid, trans in _iter_orientations(c):
        # Field direction in the molecular frame; the reference stores R.'*z for
        # R = erot(SampleFrame). Check the resonance condition and fall back to
        # R*z if the convention differs.
        n = _field_direction(c, o, H0, mux, muy, muz, pos, trans)
        phi, theta = _angles(n)
        for k in range(len(pos)):
            if not np.isfinite(pos[k]):
                continue
            u, v = int(trans[k, 0]) - 1, int(trans[k, 1]) - 1
            w = compute_strain_widths(
                sys=sys, H0=H0, mux=mux, muy=muy, muz=muz, phi=phi, theta=theta,
                transitions=torch.tensor([[u, v]]), mwFreq=MWFREQ, B0=float(pos[k]),
            )
            assert w is not None
            assert abs(float(w[0]) - wid[k]) <= 3e-3 * max(0.1, abs(wid[k])), \
                f"{name} dir={o.dir} trans={u+1},{v+1}: torchspin {float(w[0]):.8f} vs EasySpin {wid[k]:.8f} mT"
            n_checked += 1
    assert n_checked > 0


@pytest.mark.parametrize("name", ['two_spinhalf_gstrain_both', 'two_triplets_dstrain_corr'])
def test_batched_widths_match_single(cases, name):
    """compute_strain_widths_batch must reproduce the per-orientation function."""
    c = cases[name]
    sys = sys_from_mat(c.Sys)
    H0, mux, muy, muz = ham(sys)
    phis, thetas, B0s, pairs = [], [], [], []
    singles = []
    for o, pos, wid, trans in _iter_orientations(c):
        n = _field_direction(c, o, H0, mux, muy, muz, pos, trans)
        phi, theta = _angles(n)
        # one representative B0 per orientation (batch API); compare against
        # the single-orientation function evaluated at that same B0
        k = 0
        B0 = float(pos[k])
        pr = torch.tensor([[int(trans[j, 0]) - 1, int(trans[j, 1]) - 1] for j in range(len(pos))])
        phis.append(phi); thetas.append(theta); B0s.append(B0); pairs.append(pr)
        singles.append(compute_strain_widths(sys=sys, H0=H0, mux=mux, muy=muy, muz=muz,
                                             phi=phi, theta=theta, transitions=pr,
                                             mwFreq=MWFREQ, B0=B0))
    phis_t = torch.tensor(phis); thetas_t = torch.tensor(thetas); B0_t = torch.tensor(B0s)
    st, ct = torch.sin(thetas_t), torch.cos(thetas_t)
    sp, cp = torch.sin(phis_t), torch.cos(phis_t)
    muzL_b = (st * cp)[:, None, None] * mux + (st * sp)[:, None, None] * muy + ct[:, None, None] * muz
    batch = compute_strain_widths_batch(sys, H0, muzL_b, phis_t, thetas_t, B0_t, pairs, MWFREQ)
    for wb, ws in zip(batch, singles):
        torch.testing.assert_close(wb, ws, rtol=1e-6, atol=1e-9)
