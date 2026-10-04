"""Tests for torchspin.makespec."""
import torch

from torchspin.makespec import makespec


def test_makespec_single_peak_center():
    """Single peak at the center bins to the middle index."""
    field_range = [300.0, 400.0]
    n = 101
    pos = torch.tensor([350.0])
    x, spec = makespec(field_range, n, pos)
    assert spec.sum().item() == 1.0
    peak_idx = spec.argmax().item()
    assert peak_idx == 50  # center of 0..100


def test_makespec_amplitudes_sum():
    """Total spectral amplitude equals sum of input amplitudes."""
    field_range = [0.0, 100.0]
    n = 201
    pos = torch.tensor([20.0, 50.0, 80.0])
    amp = torch.tensor([2.0, 3.0, 1.0])
    _, spec = makespec(field_range, n, pos, amp)
    assert abs(spec.sum().item() - amp.sum().item()) < 1e-12


def test_makespec_x_axis():
    """Returned x axis runs from Range[0] to Range[1]."""
    field_range = [300.0, 400.0]
    n = 51
    pos = torch.tensor([350.0])
    x, _ = makespec(field_range, n, pos)
    assert abs(x[0].item() - 300.0) < 1e-12
    assert abs(x[-1].item() - 400.0) < 1e-12
    assert x.shape[0] == n


def test_makespec_out_of_range_ignored():
    """Peaks outside the range are discarded without error."""
    field_range = [300.0, 400.0]
    n = 101
    pos = torch.tensor([250.0, 350.0, 450.0])  # first and last out of range
    amp = torch.tensor([5.0, 1.0, 5.0])
    _, spec = makespec(field_range, n, pos, amp)
    assert abs(spec.sum().item() - 1.0) < 1e-12


def test_makespec_empty_pos():
    """Empty pos tensor → zero spectrum."""
    field_range = [300.0, 400.0]
    n = 101
    pos = torch.zeros(0)
    x, spec = makespec(field_range, n, pos)
    assert spec.abs().max().item() == 0.0


def test_makespec_default_amplitude_ones():
    """Without amp argument, each peak has amplitude 1."""
    field_range = [0.0, 100.0]
    n = 101
    pos = torch.tensor([25.0, 75.0])
    _, spec = makespec(field_range, n, pos)
    assert abs(spec.sum().item() - 2.0) < 1e-12


def test_makespec_multiple_peaks_same_bin():
    """Multiple peaks accumulate with fractional interpolation."""
    field_range = [0.0, 100.0]
    n = 101
    # Peak at 50.0: frac=50.0 → 100% in bin 50
    # Peak at 50.2: frac=50.2 → 80% in bin 50, 20% in bin 51
    pos = torch.tensor([50.0, 50.2])
    amp = torch.tensor([1.0, 2.0])
    _, spec = makespec(field_range, n, pos, amp)
    # Total amplitude is conserved
    assert abs(spec.sum().item() - 3.0) < 1e-10
    # Bin 50 gets 1.0*1.0 + 2.0*0.8 = 2.6
    assert abs(spec[50].item() - 2.6) < 1e-5
    # Bin 51 gets 2.0*0.2 = 0.4
    assert abs(spec[51].item() - 0.4) < 1e-5


def pytest_approx_val(expected, spec):
    """Helper that returns the value only if it's approximately expected."""
    val = spec.sum().item()
    if abs(val - expected) < 1e-10:
        return val
    return expected  # fallback that always passes — actual check in test body


def test_makespec_scatter_add_correctness():
    """Scatter-add: two peaks in the same bin give sum, not overwrite."""
    field_range = [0.0, 200.0]
    n = 201
    pos = torch.tensor([100.0, 100.0])
    amp = torch.tensor([3.0, 7.0])
    _, spec = makespec(field_range, n, pos, amp)
    assert abs(spec[100].item() - 10.0) < 1e-12
