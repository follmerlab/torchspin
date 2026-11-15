# torchspin

**Simulate and fit EPR spectra in Python.** torchspin is a PyTorch
reimplementation of [EasySpin](https://easyspin.org)'s simulators: powder and
single-crystal CW EPR (`pepper`), solution and slow-motion spectra (`garlic`,
`chili`), ENDOR (`salt`), pulse EPR (`saffron`), magnetometry (`curry`) and
least-squares fitting (`esfit`), with EasySpin's parameter names, units and
conventions.  Because it is built on PyTorch, spectra are differentiable with
respect to the spin Hamiltonian parameters and can run on a GPU.

## Start here

- **Install**: `pip install torchspin` (details and extras in
  [Getting started](getting_started.md#installation)).
- **Simulate a first spectrum**: [Solid-state (powder) spectra](guide/powder.md),
  or the [quickstart tutorial](../examples/notebooks/01_torchspin_quickstart.ipynb).
- **Fit a measured spectrum**: [Fitting spectra](guide/fitting.md).
- **Use gradients or a GPU**: [Gradients and differentiable simulation](guide/gradients.md).

```python
from torchspin import SpinSystem, Experiment, pepper

cu = SpinSystem(S=0.5, g=[2.05, 2.05, 2.25], Nucs='63Cu', A=[50, 50, 520], HStrain=[60, 60, 90])
B, spc = pepper(cu, Experiment(mwFreq=9.5, Range=[260, 360]))   # field in mT, derivative spectrum
```

## What is here

- [User guide](guide/index.md): short task pages (powder, solution and slow-motion, ENDOR and
  pulse, fitting, gradients, orientation grids, ligand-field tools) with runnable examples.
- [Tutorials](tutorials/index.md): executed notebooks, from the quickstart to real-world fitting.
- [API reference](api/index.rst): every public function and class with the parameters it accepts.
- [Troubleshooting and reference](reference.md): pitfalls, known limitations per simulator, and the
  change log.

## Tutorials

| Notebook | What it shows |
|---|---|
| [01 quickstart](../examples/notebooks/01_torchspin_quickstart.ipynb) | Spin systems, Hamiltonians, powder (`pepper`) and solution (`garlic`) spectra, level diagrams |
| [02 GPU and autograd](../examples/notebooks/02_gpu_and_autograd.ipynb) | `device='cuda'`, `differentiable_spectrum`, gradients with respect to spin-Hamiltonian parameters, gradient-based fitting |
| [03 spectral fitting](../examples/notebooks/03_spectral_fitting.ipynb) | `esfit` on synthetic powder and solution spectra with `method='global'` |
| [04 benchmarks](../examples/notebooks/04_benchmarks.ipynb) | MATLAB/EasySpin vs torchspin timings (threads, GPU, worker processes) from the 2026-09 campaign, plus live timing |
| [05 real-world fitting](../examples/notebooks/05_fitting_real_world.ipynb) | Bad starting guesses, loss-landscape slices, simplex traps vs global search, 5-parameter stress test |
| [06 blue copper](../examples/notebooks/06_blue_copper_fitting.ipynb) | Rhombic Cu(II) case study: simplex, Levenberg–Marquardt and global head to head |
| [07 EasySpin parity](../examples/notebooks/07_easyspin_parity.ipynb) | Overlays against stored EasySpin outputs (pepper, garlic, chili, saffron HYSCORE, cardamom) with cosine similarities |
| [08 pepper playground](../examples/notebooks/08_pepper_playground.ipynb) | Simulate, add noise, fit with every `esfit` method with live progress; compare recovery, residual and time |
| [09 fit GUI](../examples/notebooks/09_fit_gui.ipynb) | Interactive panel (`torchspin.fitgui`): sliders, fix boxes, Start/Stop, live fit/residual/RMSD trace |

## Running on a workstation

- CPU: torchspin uses all cores through thread pools inside the simulators (`torch.set_num_threads(n)`
  caps it). Population-based fits add `FitOptions(n_workers='auto')` (worker processes for the allocated cores, at most 32,
  about 0.5 GB of memory each and never more than half the free memory;
  `pip install "torchspin[fit]"` for models defined in notebooks).
- GPU: `Options(device='cuda')`; worthwhile for autograd and batched fitting, not for single simulations.
- Thread and process budgets: `torchspin.parallel` (`cpu_budget()`, `thread_budget(n)`, `native_thread_env(n)`)
  for code that runs torchspin inside its own worker pools.
- Progress and early stop: `FitOptions(progress='text', max_time=…)`; a kernel interrupt returns the best
  fit so far.

## Benchmarks and tests

- [Benchmark report](../benchmarks/results/BENCHMARK_REPORT.md): MATLAB/EasySpin vs torchspin on nine
  workloads, before and after the 2026-09 optimizations.  The
  [verified matched-host campaign](https://github.com/follmerlab/torchspin/blob/main/benchmarks/results/workstation_20260904_verified/BENCHMARK_VERIFICATION.md)
  (five replicates per point, with a per-workload device audit) supersedes it wherever the two disagree.
- `python -m pytest` runs the test suite (about 5800 tests, including comparisons against stored
  EasySpin/MATLAB references in `tests/data`; no MATLAB installation needed).

```{toctree}
:hidden:
:maxdepth: 2

getting_started
guide/index
tutorials/index
api/index
reference
```
