# Solid-state (powder) spectra

[`pepper`](../api/pepper.rst) simulates CW EPR spectra of solids: powders and
frozen solutions (averaged over all orientations), or single crystals.  It
diagonalizes the full spin Hamiltonian, so it handles any number of electron
and nuclear spins, zero-field splittings, and large hyperfine couplings.

## A first spectrum

A Cu(II) complex at X-band, with g and copper hyperfine strain broadening:

```pycon
>>> from torchspin import SpinSystem, Experiment, pepper
>>> cu = SpinSystem(S=0.5, g=[2.05, 2.05, 2.25], Nucs='63Cu',
...                 A=[50, 50, 520], HStrain=[60, 60, 90])
>>> exp = Experiment(mwFreq=9.5, Range=[260, 360], nPoints=1024)
>>> B, spc = pepper(cu, exp)
>>> B.shape, spc.shape
(torch.Size([1024]), torch.Size([1024]))
>>> float(B[0]), float(B[-1])
(260.0, 360.0)
```

`B` is the field axis in mT and `spc` the first-derivative spectrum
(`Harmonic=1` is the default), both torch tensors; call `.numpy()` to plot
them with matplotlib.

## Line widths

- `lw = [GaussianFWHM, LorentzianFWHM]` (mT) convolves the whole spectrum.
- `HStrain` (MHz, one value per principal axis) gives every transition an
  orientation-dependent Gaussian width; `gStrain`, `AStrain` and `DStrain`
  model distributions of g, A and D.

## Common variations

| To get | Do this |
|---|---|
| absorption instead of the derivative | `Experiment(..., Harmonic=0)` |
| a frequency-swept spectrum | `Experiment(Field=340, mwRange=[9.0, 10.0])` |
| thermal populations (high-spin systems at low temperature) | `Experiment(..., Temperature=5)` |
| a mixture of species | `pepper([sys_a, sys_b], exp)`, weighted by each `SpinSystem.weight` |
| a single crystal | `Experiment(..., CrystalSymmetry=..., SampleFrame=[a, b, c])` (or `MolFrame`; see [Experiment](../api/experiment.rst)) |
| natural-abundance isotopes | `Nucs='Cu'`: every isotopologue is simulated and weighted |
| a faster, approximate simulation of many ligand nuclei | `Options(Method='perturb')` or `'hybrid'` |

```pycon
>>> sys_b = SpinSystem(S=0.5, g=2.003, lw=[1.0, 0.0], weight=0.3)
>>> B, spc = pepper([cu, sys_b], exp)
>>> spc.shape
torch.Size([1024])
```

Second-order perturbation theory (`'perturb'`) is inaccurate when a hyperfine
coupling is a sizable fraction of the microwave frequency; the
[pitfalls page](../pitfalls.md) has the numbers.  The default `'matrix'` is
exact.

## Check the orientation grid

The default grid is not always fine enough for narrow lines on large
anisotropic couplings.  [`grid_convergence`](grid.md) tells you.

See also: the [pepper API page](../api/pepper.rst) for every argument, the
[quickstart tutorial](../tutorials/01_torchspin_quickstart.ipynb) and the
[pepper playground](../tutorials/08_pepper_playground.ipynb).
