"""Tests for batch simulation module.

Verifies:
1. batch_pepper produces identical results to individual pepper calls
2. Weighted sums are computed correctly
3. Multi-component spectra work for various system types
4. batch_simulate dispatches correctly to all simulators
5. Edge cases (empty list, single system, different dimensions)
"""
import pytest
import torch

from torchspin import (
    SpinSystem, Experiment, Options, pepper, garlic, chili,
    batch_pepper, batch_simulate,
)


# ---------------------------------------------------------------------------
# batch_pepper — core tests
# ---------------------------------------------------------------------------

class TestBatchPepper:
    """batch_pepper matches individual pepper calls exactly."""

    def test_single_system_matches_pepper(self):
        sys = SpinSystem(S=[0.5], g=[[2.00, 2.05, 2.10]], lw=[0.5])
        exp = Experiment(mwFreq=9.5, Range=[300, 360], Harmonic=0)
        opt = Options(GridSize=10)

        x_b, spec_b = batch_pepper([sys], exp, opt)
        x_p, spec_p = pepper(sys, exp, opt)

        assert spec_b.shape == (1, 1024)
        assert torch.allclose(spec_b[0], spec_p, atol=1e-12)
        assert torch.allclose(x_b, x_p)

    def test_multiple_systems_match_pepper(self):
        systems = [
            SpinSystem(S=[0.5], g=[[2.00, 2.05, 2.10]], lw=[0.5]),
            SpinSystem(S=[0.5], g=[[2.01, 2.06, 2.11]], lw=[0.5]),
            SpinSystem(S=[0.5], g=[[1.99, 2.04, 2.09]], lw=[0.5]),
        ]
        exp = Experiment(mwFreq=9.5, Range=[300, 360], Harmonic=0)
        opt = Options(GridSize=10)

        x_b, spectra = batch_pepper(systems, exp, opt)
        assert spectra.shape == (3, 1024)

        for i, sys in enumerate(systems):
            _, y_ref = pepper(sys, exp, opt)
            assert torch.allclose(spectra[i], y_ref, atol=1e-12), \
                f"System {i} mismatch"

    def test_first_derivative(self):
        systems = [
            SpinSystem(S=[0.5], g=[[2.00, 2.05, 2.10]], lw=[0.5]),
            SpinSystem(S=[0.5], g=[[2.01, 2.06, 2.11]], lw=[0.5]),
        ]
        exp = Experiment(mwFreq=9.5, Range=[300, 360], Harmonic=1)
        opt = Options(GridSize=10)

        x_b, spectra = batch_pepper(systems, exp, opt)
        for i, sys in enumerate(systems):
            _, y_ref = pepper(sys, exp, opt)
            assert torch.allclose(spectra[i], y_ref, atol=1e-12)

    def test_with_nuclei(self):
        systems = [
            SpinSystem(S=[0.5], g=[[2.003, 2.003, 2.003]], Nucs=['14N'],
                       A=[[20.0, 20.0, 90.0]], lw=[0.3]),
            SpinSystem(S=[0.5], g=[[2.005, 2.005, 2.005]], Nucs=['14N'],
                       A=[[18.0, 18.0, 85.0]], lw=[0.3]),
        ]
        exp = Experiment(mwFreq=9.5, Range=[320, 360], Harmonic=0)
        opt = Options(GridSize=10)

        x_b, spectra = batch_pepper(systems, exp, opt)
        assert spectra.shape == (2, 1024)

        for i, sys in enumerate(systems):
            _, y_ref = pepper(sys, exp, opt)
            assert torch.allclose(spectra[i], y_ref, atol=1e-10)

    def test_different_hilbert_dimensions(self):
        """Systems with different Hilbert space dimensions."""
        systems = [
            SpinSystem(S=[0.5], g=[[2.00, 2.05, 2.10]], lw=[0.5]),
            SpinSystem(S=[0.5], g=[[2.003, 2.003, 2.003]], Nucs=['1H'],
                       A=[[5.0, 5.0, 5.0]], lw=[0.3]),
        ]
        exp = Experiment(mwFreq=9.5, Range=[300, 360], Harmonic=0)
        opt = Options(GridSize=10)

        x_b, spectra = batch_pepper(systems, exp, opt)
        assert spectra.shape == (2, 1024)

        for i, sys in enumerate(systems):
            _, y_ref = pepper(sys, exp, opt)
            assert torch.allclose(spectra[i], y_ref, atol=1e-10)


class TestBatchPepperWeightedSum:
    """Weighted sum mode."""

    def test_equal_weights(self):
        systems = [
            SpinSystem(S=[0.5], g=[[2.00, 2.05, 2.10]], lw=[0.5]),
            SpinSystem(S=[0.5], g=[[2.01, 2.06, 2.11]], lw=[0.5]),
        ]
        exp = Experiment(mwFreq=9.5, Range=[300, 360], Harmonic=0)
        opt = Options(GridSize=10)

        x, total = batch_pepper(systems, exp, opt,
                                weights=[1.0, 1.0],
                                return_individual=False)
        _, spectra = batch_pepper(systems, exp, opt)
        expected = spectra[0] + spectra[1]
        assert torch.allclose(total, expected, atol=1e-12)

    def test_unequal_weights(self):
        systems = [
            SpinSystem(S=[0.5], g=[[2.00, 2.05, 2.10]], lw=[0.5]),
            SpinSystem(S=[0.5], g=[[2.01, 2.06, 2.11]], lw=[0.5]),
            SpinSystem(S=[0.5], g=[[1.99, 2.04, 2.09]], lw=[0.5]),
        ]
        exp = Experiment(mwFreq=9.5, Range=[300, 360], Harmonic=0)
        opt = Options(GridSize=10)

        w = [0.5, 0.3, 0.2]
        x, total = batch_pepper(systems, exp, opt,
                                weights=w, return_individual=False)
        _, spectra = batch_pepper(systems, exp, opt)
        expected = w[0]*spectra[0] + w[1]*spectra[1] + w[2]*spectra[2]
        assert torch.allclose(total, expected, atol=1e-12)

    def test_sys_weight_attribute(self):
        """When weights not provided, uses sys.weight attribute."""
        sys1 = SpinSystem(S=[0.5], g=[[2.00, 2.05, 2.10]], lw=[0.5])
        sys1.weight = 0.7
        sys2 = SpinSystem(S=[0.5], g=[[2.01, 2.06, 2.11]], lw=[0.5])
        sys2.weight = 0.3
        systems = [sys1, sys2]

        exp = Experiment(mwFreq=9.5, Range=[300, 360], Harmonic=0)
        opt = Options(GridSize=10)

        x, total = batch_pepper(systems, exp, opt, return_individual=False)
        _, spectra = batch_pepper(systems, exp, opt)
        expected = 0.7*spectra[0] + 0.3*spectra[1]
        assert torch.allclose(total, expected, atol=1e-12)


class TestBatchPepperEdgeCases:
    """Edge cases and error handling."""

    def test_empty_list(self):
        exp = Experiment(mwFreq=9.5, Range=[300, 360], Harmonic=0)
        x, spectra = batch_pepper([], exp)
        assert spectra.shape == (0, 1024)
        assert x.shape == (1024,)

    def test_empty_list_weighted(self):
        exp = Experiment(mwFreq=9.5, Range=[300, 360], Harmonic=0)
        x, total = batch_pepper([], exp, return_individual=False)
        assert total.shape == (1024,)

    def test_output_shapes(self):
        systems = [SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]], lw=[1.0])] * 5
        exp = Experiment(mwFreq=9.5, Range=[300, 360], nPoints=512, Harmonic=0)
        opt = Options(GridSize=10)

        x, spectra = batch_pepper(systems, exp, opt)
        assert x.shape == (512,)
        assert spectra.shape == (5, 512)

    def test_default_options(self):
        """batch_pepper works with default Options (opt=None)."""
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.05, 2.1]], lw=[0.5])
        exp = Experiment(mwFreq=9.5, Range=[300, 360], Harmonic=0)
        x, spectra = batch_pepper([sys], exp)
        assert spectra.shape[0] == 1
        assert spectra[0].max() > 0


# ---------------------------------------------------------------------------
# batch_simulate — dispatcher tests
# ---------------------------------------------------------------------------

class TestBatchSimulate:
    """batch_simulate dispatches to the right simulator."""

    def test_pepper_dispatch(self):
        systems = [
            SpinSystem(S=[0.5], g=[[2.00, 2.05, 2.10]], lw=[0.5]),
            SpinSystem(S=[0.5], g=[[2.01, 2.06, 2.11]], lw=[0.5]),
        ]
        exp = Experiment(mwFreq=9.5, Range=[300, 360], Harmonic=0)
        opt = Options(GridSize=10)

        x1, s1 = batch_simulate('pepper', systems, exp, opt)
        x2, s2 = batch_pepper(systems, exp, opt)
        assert torch.allclose(s1, s2, atol=1e-12)

    def test_garlic_dispatch(self):
        systems = [
            SpinSystem(S=[0.5], g=[[2.003]], Nucs=['14N'],
                       A=[[50.0]], lw=[0.1]),
            SpinSystem(S=[0.5], g=[[2.005]], Nucs=['14N'],
                       A=[[45.0]], lw=[0.1]),
        ]
        exp = Experiment(mwFreq=9.5, Range=[330, 350], Harmonic=0)

        x, spectra = batch_simulate('garlic', systems, exp)
        assert spectra.shape[0] == 2
        assert spectra.shape[1] == x.shape[0]

        # Each spectrum should match individual garlic call
        for i, sys in enumerate(systems):
            x_ref, y_ref = garlic(sys, exp)
            y_ref_t = torch.as_tensor(y_ref, dtype=torch.float64) if not isinstance(y_ref, torch.Tensor) else y_ref
            assert torch.allclose(spectra[i], y_ref_t, atol=1e-10)

    def test_chili_dispatch(self):
        from torchspin import ChiliOptions
        systems = [
            SpinSystem(S=[0.5], g=[[2.009, 2.006, 2.002]],
                       Nucs=['14N'], A=[[10.0, 10.0, 95.0]],
                       tcorr=5e-9),
        ]
        exp = Experiment(mwFreq=9.5, Range=[330, 350], Harmonic=0)
        opt = ChiliOptions(LLMK=[6, 0, 2, 2])

        x, spectra = batch_simulate('chili', systems, exp, opt)
        assert spectra.shape[0] == 1
        assert spectra[0].max() > 0

    def test_unknown_simulator(self):
        with pytest.raises(ValueError, match="Unknown simulator"):
            batch_simulate('nonexistent', [], None)

    def test_weighted_sum_garlic(self):
        systems = [
            SpinSystem(S=[0.5], g=[[2.003]], Nucs=['14N'],
                       A=[[50.0]], lw=[0.1]),
            SpinSystem(S=[0.5], g=[[2.005]], Nucs=['14N'],
                       A=[[45.0]], lw=[0.1]),
        ]
        exp = Experiment(mwFreq=9.5, Range=[330, 350], Harmonic=0)

        x, total = batch_simulate('garlic', systems, exp,
                                  weights=[0.6, 0.4],
                                  return_individual=False)
        _, spectra = batch_simulate('garlic', systems, exp)
        expected = 0.6 * spectra[0] + 0.4 * spectra[1]
        assert torch.allclose(total, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Strain and advanced features
# ---------------------------------------------------------------------------

class TestBatchPepperStrain:
    """Strain broadening in batch mode."""

    def test_gstrain(self):
        systems = [
            SpinSystem(S=[0.5], g=[[2.00, 2.05, 2.10]], lw=[0.3],
                       gStrain=torch.tensor([[0.01, 0.01, 0.005]])),
            SpinSystem(S=[0.5], g=[[2.01, 2.06, 2.11]], lw=[0.3],
                       gStrain=torch.tensor([[0.005, 0.005, 0.002]])),
        ]
        exp = Experiment(mwFreq=9.5, Range=[300, 360], Harmonic=0)
        opt = Options(GridSize=10)

        x, spectra = batch_pepper(systems, exp, opt)
        for i, sys in enumerate(systems):
            _, y_ref = pepper(sys, exp, opt)
            assert torch.allclose(spectra[i], y_ref, atol=1e-10)

    def test_temperature(self):
        systems = [
            SpinSystem(S=[0.5], g=[[2.00, 2.05, 2.10]], lw=[0.5]),
            SpinSystem(S=[0.5], g=[[2.01, 2.06, 2.11]], lw=[0.5]),
        ]
        exp = Experiment(mwFreq=9.5, Range=[300, 360], Harmonic=0,
                         Temperature=10.0)
        opt = Options(GridSize=10)

        x, spectra = batch_pepper(systems, exp, opt)
        for i, sys in enumerate(systems):
            _, y_ref = pepper(sys, exp, opt)
            assert torch.allclose(spectra[i], y_ref, atol=1e-10)


class TestBatchPepperHighSpin:
    """S > 1/2 systems."""

    def test_s1_triplet(self):
        systems = [
            SpinSystem(S=[1.0], g=[[2.003, 2.003, 2.003]],
                       D=[[1000.0, 100.0, 0.0]], lw=[1.0]),
            SpinSystem(S=[1.0], g=[[2.005, 2.005, 2.005]],
                       D=[[1200.0, 150.0, 0.0]], lw=[1.0]),
        ]
        exp = Experiment(mwFreq=9.5, Range=[200, 500], Harmonic=0)
        opt = Options(GridSize=10)

        x, spectra = batch_pepper(systems, exp, opt)
        assert spectra.shape == (2, 1024)
        for i, sys in enumerate(systems):
            _, y_ref = pepper(sys, exp, opt)
            assert torch.allclose(spectra[i], y_ref, atol=1e-10)
