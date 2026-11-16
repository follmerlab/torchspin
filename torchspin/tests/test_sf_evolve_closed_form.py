"""saffron_peaks.sf_evolve (perf train P2b, B2): the vectorized time-domain
evolution must reproduce the 0.9.38 per-time-point loops (copied below as the
reference) for every incrementation scheme, to roundoff, values and gradients,
on random Hermitian-derived inputs."""
import math

import numpy as np
import pytest
import torch

from torchspin.saffron_peaks import sf_evolve


def _ref_sf_evolve(inc_scheme_id: int,
              n_points: list[int],
              dt: list[float],
              free_l: list[int],
              free_r: list[int],
              Ea: torch.Tensor,
              Eb: torch.Tensor,
              G: torch.Tensor,
              D: torch.Tensor,
              *mixing_mats: torch.Tensor) -> torch.Tensor:
    """Direct time-domain signal accumulation.

    This is autograd-compatible (no scatter_add_ in-place ops).

    Parameters
    ----------
    inc_scheme_id : int
        Incrementation scheme identifier.
    n_points : list of int
        Number of points per dimension.
    dt : list of float
        Time step per dimension (µs).
    free_l, free_r : list of int
        Left/right manifold indices (1=alpha, 2=beta).
    Ea, Eb : Tensor
        Eigenvalues for alpha/beta manifolds.
    G, D : Tensor
        Start coherence and detection matrices.
    *mixing_mats : Tensor
        Mixing matrices.

    Returns
    -------
    Tensor
        Complex time-domain signal, shape ``(nPoints,)`` or ``(nPoints1, nPoints2)``.
    """
    E = [Ea, Eb]
    dtype = torch.complex128
    device = Ea.device

    if inc_scheme_id == 1:
        # [1]: 3-pulse ESEEM — direct port of sf_evolve case 1
        nP = n_points[0] if isinstance(n_points, (list, tuple)) else n_points
        E1l = E[free_l[0] - 1]
        E1r = E[free_r[0] - 1]

        # Density and detector (vectorized form)
        density = G.reshape(-1)  # (N^2,)
        detector = D.T.reshape(1, -1)  # (1, N^2) — note .T to match MATLAB Detector=reshape(D.',1,NN)

        # Evolution operator per step: UUt[j,k] = exp(-2πi*E_l[j]*dt) * exp(-2πi*E_r[k]*dt)'
        UUt = (torch.exp(-2j * math.pi * E1l * dt[0]).unsqueeze(1) *
               torch.exp(+2j * math.pi * E1r * dt[0]).unsqueeze(0))
        UUt = UUt.reshape(-1)

        signal = torch.zeros(nP, dtype=dtype, device=device)
        fd = density.clone()
        for k in range(nP):
            signal[k] = (detector @ fd).squeeze()
            fd = UUt * fd
        return signal

    elif inc_scheme_id == 2:
        # [1 1]: 2-pulse ESEEM — direct port of sf_evolve case 2
        nP = n_points[0] if isinstance(n_points, (list, tuple)) else n_points
        T1l, T1r = mixing_mats[0], mixing_mats[1]
        E1l = E[free_l[0] - 1]
        E1r = E[free_r[0] - 1]
        E2l = E[free_l[1] - 1]
        E2r = E[free_r[1] - 1]

        detector = D.T.reshape(1, -1)

        # Evolution: UUleft[i,j] = exp(-2πi*E2l[i]*dt) * exp(-2πi*E1l[j]*dt)
        UUleft = (torch.exp(-2j * math.pi * E2l * dt[0]).unsqueeze(1) *
                  torch.exp(-2j * math.pi * E1l * dt[0]).unsqueeze(0))
        UUright = (torch.exp(+2j * math.pi * E1r * dt[0]).unsqueeze(1) *
                   torch.exp(+2j * math.pi * E2r * dt[0]).unsqueeze(0))

        signal = torch.zeros(nP, dtype=dtype, device=device)
        T1l_ev = T1l.clone()
        T1r_ev = T1r.clone()
        for k in range(nP):
            fd = T1l_ev @ G @ T1r_ev
            signal[k] = (detector @ fd.reshape(-1)).squeeze()
            T1l_ev = UUleft * T1l_ev
            T1r_ev = UUright * T1r_ev
        return signal

    elif inc_scheme_id == 11:
        # [1 2]: HYSCORE — 2D, direct port of sf_evolve case 11
        nP1, nP2 = n_points
        T1l, T1r = mixing_mats[0], mixing_mats[1]
        E1l = E[free_l[0] - 1]
        E1r = E[free_r[0] - 1]
        E2l = E[free_l[1] - 1]
        E2r = E[free_r[1] - 1]
        N = len(Ea)

        detector = D.T.reshape(1, -1)

        UUt1 = (torch.exp(-2j * math.pi * E1l * dt[0]).unsqueeze(1) *
                torch.exp(+2j * math.pi * E1r * dt[0]).unsqueeze(0))
        UUt2 = (torch.exp(-2j * math.pi * E2l * dt[1]).unsqueeze(1) *
                torch.exp(+2j * math.pi * E2r * dt[1]).unsqueeze(0))
        UUt2_flat = UUt2.reshape(-1)

        signal = torch.zeros(nP1, nP2, dtype=dtype, device=device)
        G_ev = G.clone()
        for k1 in range(nP1):
            fd = (T1l @ G_ev @ T1r).reshape(-1)
            for k2 in range(nP2):
                signal[k1, k2] = (detector @ fd).squeeze()
                fd = UUt2_flat * fd
            G_ev = UUt1 * G_ev
        return signal

    elif inc_scheme_id == 3:
        # [1 -1]: 2 free evolutions, 1D — same structure as scheme 2
        # but with sign flip on second dimension
        # nu = -E2l[i] + E1l[j] - E1r[k] + E2r[l]
        nP = n_points[0] if isinstance(n_points, (list, tuple)) else n_points
        T1l, T1r = mixing_mats[0], mixing_mats[1]
        E1l = E[free_l[0] - 1]
        E1r = E[free_r[0] - 1]
        E2l = E[free_l[1] - 1]
        E2r = E[free_r[1] - 1]

        detector = D.T.reshape(1, -1)

        # For [1,-1]: second dimension is decremented, so evolution sign flips
        UUleft = (torch.exp(+2j * math.pi * E2l * dt[0]).unsqueeze(1) *
                  torch.exp(-2j * math.pi * E1l * dt[0]).unsqueeze(0))
        UUright = (torch.exp(+2j * math.pi * E1r * dt[0]).unsqueeze(1) *
                   torch.exp(-2j * math.pi * E2r * dt[0]).unsqueeze(0))

        signal = torch.zeros(nP, dtype=dtype, device=device)
        T1l_ev = T1l.clone()
        T1r_ev = T1r.clone()
        for k in range(nP):
            fd = T1l_ev @ G @ T1r_ev
            signal[k] = (detector @ fd.reshape(-1)).squeeze()
            T1l_ev = UUleft * T1l_ev
            T1r_ev = UUright * T1r_ev
        return signal

    elif inc_scheme_id in (12, 14):
        # [1 2 1] or [1 1 2]: 3 free evolutions, 2D
        # General 2D with 2 mixing matrix pairs
        nP1, nP2 = n_points
        T1l, T1r, T2l, T2r = mixing_mats[0], mixing_mats[1], mixing_mats[2], mixing_mats[3]
        El = [E[fl - 1] for fl in free_l]
        Er = [E[fr - 1] for fr in free_r]
        N = len(Ea)

        detector = D.T.reshape(1, -1)

        if inc_scheme_id == 12:
            # [1 2 1]: dim1 uses E1+E3, dim2 uses E2
            UUt1_l = (torch.exp(-2j * math.pi * El[2] * dt[0]).unsqueeze(1) *
                      torch.exp(-2j * math.pi * El[0] * dt[0]).unsqueeze(0))
            UUt1_r = (torch.exp(+2j * math.pi * Er[0] * dt[0]).unsqueeze(1) *
                      torch.exp(+2j * math.pi * Er[2] * dt[0]).unsqueeze(0))
            UUt2 = (torch.exp(-2j * math.pi * El[1] * dt[1]).unsqueeze(1) *
                    torch.exp(+2j * math.pi * Er[1] * dt[1]).unsqueeze(0))
        else:
            # [1 1 2]: dim1 uses E1+E2, dim2 uses E3
            UUt1_l = (torch.exp(-2j * math.pi * El[1] * dt[0]).unsqueeze(1) *
                      torch.exp(-2j * math.pi * El[0] * dt[0]).unsqueeze(0))
            UUt1_r = (torch.exp(+2j * math.pi * Er[0] * dt[0]).unsqueeze(1) *
                      torch.exp(+2j * math.pi * Er[1] * dt[0]).unsqueeze(0))
            UUt2 = (torch.exp(-2j * math.pi * El[2] * dt[1]).unsqueeze(1) *
                    torch.exp(+2j * math.pi * Er[2] * dt[1]).unsqueeze(0))

        UUt2_flat = UUt2.reshape(-1)

        signal = torch.zeros(nP1, nP2, dtype=dtype, device=device)
        T1l_ev = T1l.clone()
        T1r_ev = T1r.clone()
        for k1 in range(nP1):
            fd = (T2l @ T1l_ev @ G @ T1r_ev @ T2r).reshape(-1)
            for k2 in range(nP2):
                signal[k1, k2] = (detector @ fd).squeeze()
                fd = UUt2_flat * fd
            T1l_ev = UUt1_l * T1l_ev
            T1r_ev = UUt1_r * T1r_ev
        return signal

    elif inc_scheme_id in (15, 17):
        # [1 2 2 1] or [1 1 2 2]: 4 free evolutions, 2D
        nP1, nP2 = n_points
        T1l, T1r, T2l, T2r, T3l, T3r = (mixing_mats[0], mixing_mats[1],
                                           mixing_mats[2], mixing_mats[3],
                                           mixing_mats[4], mixing_mats[5])
        El = [E[fl - 1] for fl in free_l]
        Er = [E[fr - 1] for fr in free_r]
        N = len(Ea)

        detector = D.T.reshape(1, -1)

        if inc_scheme_id == 15:
            # [1 2 2 1]: dim1 = E1+E4, dim2 = E2+E3
            UUt1_l = (torch.exp(-2j * math.pi * El[3] * dt[0]).unsqueeze(1) *
                      torch.exp(-2j * math.pi * El[0] * dt[0]).unsqueeze(0))
            UUt1_r = (torch.exp(+2j * math.pi * Er[0] * dt[0]).unsqueeze(1) *
                      torch.exp(+2j * math.pi * Er[3] * dt[0]).unsqueeze(0))
            UUt2_l = (torch.exp(-2j * math.pi * El[2] * dt[1]).unsqueeze(1) *
                      torch.exp(-2j * math.pi * El[1] * dt[1]).unsqueeze(0))
            UUt2_r = (torch.exp(+2j * math.pi * Er[1] * dt[1]).unsqueeze(1) *
                      torch.exp(+2j * math.pi * Er[2] * dt[1]).unsqueeze(0))
        else:
            # [1 1 2 2]: dim1 = E1+E2, dim2 = E3+E4
            UUt1_l = (torch.exp(-2j * math.pi * El[1] * dt[0]).unsqueeze(1) *
                      torch.exp(-2j * math.pi * El[0] * dt[0]).unsqueeze(0))
            UUt1_r = (torch.exp(+2j * math.pi * Er[0] * dt[0]).unsqueeze(1) *
                      torch.exp(+2j * math.pi * Er[1] * dt[0]).unsqueeze(0))
            UUt2_l = (torch.exp(-2j * math.pi * El[3] * dt[1]).unsqueeze(1) *
                      torch.exp(-2j * math.pi * El[2] * dt[1]).unsqueeze(0))
            UUt2_r = (torch.exp(+2j * math.pi * Er[2] * dt[1]).unsqueeze(1) *
                      torch.exp(+2j * math.pi * Er[3] * dt[1]).unsqueeze(0))

        signal = torch.zeros(nP1, nP2, dtype=dtype, device=device)
        T1l_ev = T1l.clone()
        T1r_ev = T1r.clone()
        for k1 in range(nP1):
            inner_l = T2l @ T1l_ev
            inner_r = T1r_ev @ T2r
            T2l_ev = T2l.clone() if k1 == 0 else T2l_ev  # reset dim2 each dim1 step
            T2r_ev = T2r.clone() if k1 == 0 else T2r_ev
            # Actually for 2D: dim2 mixing resets each dim1 step
            fd_base = T3l @ inner_l @ G @ inner_r @ T3r
            fd = fd_base.reshape(-1)
            T2l_inner = torch.eye(N, dtype=dtype, device=Ea.device)
            T2r_inner = torch.eye(N, dtype=dtype, device=Ea.device)
            for k2 in range(nP2):
                fd2 = (T2l_inner @ fd_base @ T2r_inner).reshape(-1)
                signal[k1, k2] = (detector @ fd2).squeeze()
                T2l_inner = UUt2_l * T2l_inner
                T2r_inner = UUt2_r * T2r_inner
            T1l_ev = UUt1_l * T1l_ev
            T1r_ev = UUt1_r * T1r_ev
        return signal

    else:
        raise NotImplementedError(
            f"sf_evolve: IncSchemeID={inc_scheme_id} not yet implemented."
        )


def _inputs(seed, N, n_mix):
    g = torch.Generator().manual_seed(seed)
    c = lambda *s: torch.complex(torch.randn(*s, generator=g, dtype=torch.float64),
                                 torch.randn(*s, generator=g, dtype=torch.float64))
    Ea = torch.randn(N, generator=g, dtype=torch.float64) * 3
    Eb = torch.randn(N, generator=g, dtype=torch.float64) * 3
    return Ea, Eb, c(N, N), c(N, N), [c(N, N) for _ in range(n_mix)]


CASES = [  # scheme, n_points, dt, free_l, free_r, n_mix
    (1, [37], [0.013], [1], [2], 0),
    (2, [41], [0.011], [1, 2], [2, 1], 2),
    (3, [29], [0.017], [2, 1], [1, 2], 2),
    (11, [13, 17], [0.02, 0.03], [1, 2], [2, 1], 2),
    (12, [11, 9], [0.02, 0.025], [1, 2, 1], [2, 1, 2], 4),
    (14, [9, 12], [0.015, 0.02], [2, 1, 2], [1, 2, 1], 4),
    (15, [8, 10], [0.02, 0.03], [1, 2, 2, 1], [2, 1, 1, 2], 6),
    (17, [10, 7], [0.025, 0.02], [1, 1, 2, 2], [2, 2, 1, 1], 6),
]


@pytest.mark.parametrize('scheme,npts,dt,fl,fr,nm', CASES, ids=[str(c[0]) for c in CASES])
@pytest.mark.parametrize('N', [2, 4])
def test_sf_evolve_matches_loops(scheme, npts, dt, fl, fr, nm, N):
    Ea, Eb, G, D, mix = _inputs(scheme * 10 + N, N, nm)
    ref = _ref_sf_evolve(scheme, npts, dt, fl, fr, Ea, Eb, G, D, *mix)
    got = sf_evolve(scheme, npts, dt, fl, fr, Ea, Eb, G, D, *mix)
    assert got.shape == ref.shape
    assert (got - ref).abs().max() <= 1e-11 * ref.abs().max()


@pytest.mark.parametrize('scheme,npts,dt,fl,fr,nm', CASES[:4], ids=[str(c[0]) for c in CASES[:4]])
def test_sf_evolve_gradients_match_loops(scheme, npts, dt, fl, fr, nm):
    Ea, Eb, G, D, mix = _inputs(scheme, 3, nm)
    grads = []
    for fn in (_ref_sf_evolve, sf_evolve):
        a = Ea.clone().requires_grad_(True); b = Eb.clone().requires_grad_(True)
        fn(scheme, npts, dt, fl, fr, a, b, G, D, *mix).abs().pow(2).sum().backward()
        grads.append(torch.cat([a.grad, b.grad]))
    assert (grads[0] - grads[1]).abs().max() <= 1e-10 * grads[0].abs().max()
