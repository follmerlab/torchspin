"""saffron time-domain evolution (sf_evolve) against the frequency-domain peaks (sf_peaks,
the port of EasySpin's sf_peaks.c), for every 2D incrementation scheme (B130).

With every level energy a multiple of both bin widths 1/(n1 dt1) and 1/(n2 dt2), the
frequency binning is exact and the time signal is the inverse DFT of the peak buffer:
    td[k1, k2] = Σ_peaks amp · exp(-2πi (ν1 k1 dt1 + ν2 k2 dt2)) = n1 n2 · ifft2(buffer).
Up to 0.9.38 schemes 12 [1 2 1], 15 [1 2 2 1] and 17 [1 1 2 2] failed this at O(1) (the
evolutions were applied in the wrong order); EasySpin has no time-domain route for them."""
import numpy as np
import pytest
import torch

from torchspin.saffron_peaks import sf_evolve, sf_peaks

N1, N2, DT1, DT2 = 16, 12, 0.01, 0.02
STEP = 25.0     # MHz: a multiple of 1/(N1 DT1) = 6.25 and 1/(N2 DT2) = 4.1667

SCHEMES = {  # id: (free_l, free_r, number of mixing matrices)
    11: ([1, 2], [2, 1], 2),
    12: ([1, 2, 1], [2, 1, 2], 4),
    14: ([1, 1, 2], [2, 2, 1], 4),
    15: ([1, 2, 2, 1], [2, 1, 1, 2], 6),
    17: ([1, 1, 2, 2], [2, 2, 1, 1], 6),
}


def _unitary(N, g):
    a = torch.randn(N, N, generator=g, dtype=torch.float64) + 1j * torch.randn(N, N, generator=g, dtype=torch.float64)
    return torch.linalg.qr(a)[0]


def _inputs(scheme, N, seed):
    g = torch.Generator().manual_seed(1000 * scheme + 10 * N + seed)
    Ea = torch.randint(-5, 6, (N,), generator=g).double() * STEP
    Eb = torch.randint(-5, 6, (N,), generator=g).double() * STEP
    return Ea, Eb, _unitary(N, g), _unitary(N, g), [_unitary(N, g) for _ in range(SCHEMES[scheme][2])]


@pytest.mark.parametrize('scheme', list(SCHEMES))
@pytest.mark.parametrize('N, seed', [(2, 0), (3, 1), (4, 2), (6, 3)])
def test_time_domain_is_the_inverse_dft_of_the_peaks(scheme, N, seed):
    fl, fr, _ = SCHEMES[scheme]
    Ea, Eb, G, D, mix = _inputs(scheme, N, seed)
    td = sf_evolve(scheme, [N1, N2], [DT1, DT2], fl, fr, Ea, Eb, G, D, *mix).numpy()
    bre = torch.zeros(N1, N2, dtype=torch.float64); bim = torch.zeros(N1, N2, dtype=torch.float64)
    sf_peaks(scheme, bre, bim, [DT1, DT2], fl, fr, Ea, Eb, G, D, *mix)
    ref = np.fft.ifft2((bre + 1j * bim).numpy()) * N1 * N2
    assert np.abs(td - ref).max() <= 1e-12 * np.abs(ref).max()


@pytest.mark.parametrize('scheme', [12, 15, 17])
def test_time_domain_gradients_match_central_differences(scheme):
    fl, fr, _ = SCHEMES[scheme]
    g = torch.Generator().manual_seed(scheme)
    N = 3
    Ea = torch.randn(N, generator=g, dtype=torch.float64) * 3
    Eb = torch.randn(N, generator=g, dtype=torch.float64) * 3
    G, D = _unitary(N, g), _unitary(N, g)
    mix = [_unitary(N, g) for _ in range(SCHEMES[scheme][2])]
    f = lambda a, b: sf_evolve(scheme, [7, 6], [0.02, 0.03], fl, fr, a, b, G, D, *mix).abs().pow(2).sum()
    a = Ea.clone().requires_grad_(True); b = Eb.clone().requires_grad_(True)
    ga, gb = torch.autograd.grad(f(a, b), (a, b))
    h = 1e-6
    for i in range(N):
        e = torch.zeros(N, dtype=torch.float64); e[i] = h
        cd_a = (f(Ea + e, Eb) - f(Ea - e, Eb)) / (2 * h)
        cd_b = (f(Ea, Eb + e) - f(Ea, Eb - e)) / (2 * h)
        assert float(ga[i]) == pytest.approx(float(cd_a), rel=1e-6, abs=1e-8)
        assert float(gb[i]) == pytest.approx(float(cd_b), rel=1e-6, abs=1e-8)
