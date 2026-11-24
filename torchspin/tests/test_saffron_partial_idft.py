"""B12: saffron's product rule takes the first nPoints of the inverse DFT of each zero-filled
peak buffer from its occupied bins only.  Must equal the FFT route to roundoff, sparse or
dense (FFT fallback), 1D and 2D, and end to end."""
import numpy as np
import pytest
import torch

from torchspin.saffron import PulseExperiment, SaffronOptions, _partial_idft, saffron
from torchspin.spinsystem import SpinSystem


def _buffers(shape, n_occ, seed):
    g = torch.Generator().manual_seed(seed)
    re = torch.zeros(shape, dtype=torch.float64); im = torch.zeros(shape, dtype=torch.float64)
    idx = [torch.randint(0, s, (n_occ,), generator=g) for s in shape]
    re[tuple(idx)] = torch.randn(n_occ, generator=g, dtype=torch.float64)
    im[tuple(idx)] = torch.randn(n_occ, generator=g, dtype=torch.float64)
    return re, im


@pytest.mark.parametrize('shape, n_out, n_occ', [((512,), (128,), 10), ((1024,), (256,), 40),
                                                 ((64,), (16,), 40),            # dense: FFT fallback
                                                 ((512, 512), (128, 128), 30), ((256, 512), (64, 100), 12),
                                                 ((64, 64), (16, 16), 200)])     # dense: FFT fallback
@pytest.mark.parametrize('seed', [0, 1])
def test_partial_idft_equals_fft(shape, n_out, n_occ, seed):
    re, im = _buffers(shape, n_occ, seed)
    buf = torch.complex(re, im)
    ref = (torch.fft.ifftn(buf) * buf.numel())[tuple(slice(0, n) for n in n_out)]
    got = _partial_idft(re, im, n_out)
    assert got.shape == ref.shape
    assert float((got - ref).abs().max()) <= 1e-12 * float(ref.abs().max())


N14H = SpinSystem(S=[0.5], g=[2.0], Nucs='14N,1H', A=[[0.3, 0.3, 1.8], [-2.0, -2.0, 4.0]], Q=[[1.0], [0.0]])
H2 = SpinSystem(S=[0.5], g=[2.0], Nucs='1H,1H', A=[[-2.0, -2.0, 4.0], [1.0, 1.0, -2.0]])


@pytest.mark.parametrize('sys_, ekw', [
    (N14H, dict(Sequence='HYSCORE', Field=340, dt=0.024, nPoints=64, tau=0.136)),
    (H2, dict(Sequence='HYSCORE', Field=340, dt=0.05, nPoints=48, tau=0.2)),
    (N14H, dict(Sequence='3pESEEM', Field=340, dt=0.012, nPoints=256, tau=0.136)),
    (H2, dict(Sequence='2pESEEM', Field=340, dt=0.012, nPoints=256, tau=0.1)),
], ids=['hyscore_14N1H', 'hyscore_2H', 'eseem3p_14N1H', 'eseem2p_2H'])
def test_product_rule_end_to_end_matches_fft(sys_, ekw, monkeypatch):
    import sys
    sf = sys.modules['torchspin.saffron']
    e = PulseExperiment(**ekw)
    o = SaffronOptions(ProductRule=True, GridSize=11)
    new = np.asarray(saffron(sys_, e, o)[2]['td'])

    def via_fft(re, im, n_out):
        buf = torch.complex(re, im)
        return (torch.fft.ifftn(buf) * buf.numel())[tuple(slice(0, n) for n in n_out)]
    monkeypatch.setattr(sf, '_partial_idft', via_fft)
    ref = np.asarray(saffron(sys_, e, o)[2]['td'])
    assert np.abs(new - ref).max() <= 1e-12 * np.abs(ref).max()
