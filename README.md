# torchspin

**A differentiable PyTorch framework for EPR/ESR spin-Hamiltonian and spectrum simulation, developed against [EasySpin](https://easyspin.org).**

torchspin reimplements the core physics engine of EasySpin (MATLAB) in
Python/PyTorch, with end-to-end autograd support, optional GPU execution,
and a Python-native API.

TorchSpin is a reimplementation, not a drop-in replacement for every EasySpin
feature. See [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) and
[benchmarks/results/BENCHMARK_REPORT.md](benchmarks/results/BENCHMARK_REPORT.md)
before using it for publication-critical calculations.

- **2659 passing tests, 0 failures**, including 30 MATLAB-validation modules that compare
  against stored EasySpin outputs (cosine ≥ 0.999 for `pepper`, `garlic`,
  `chili`, `salt`/`endorfrq`, `saffron`, `curry`, `spidyan`; fitted `esfit`
  parameters within 3×10⁻⁴ in g of EasySpin's)
- **Competitive with EasySpin on a multi-core node, not uniformly faster**: on
  the audited matched-host campaign (nine workloads, five replicates, medians
  with IQRs) torchspin's best CPU configuration is faster on six — slow-motion
  `chili` 7.3× and 1.5×, `pepper` Mn(II) 2.8×, trajectory `cardamom` 2.0×,
  a 20-call `pepper` fit loop 1.9×, perturbative `pepper` 1.3× — and slower on
  three: `pepper` Cu/2N matrix 0.34×, `pepper` strain summation 0.82×,
  `saffron` HYSCORE 0.88×.  Details and the device audit:
  [`benchmarks/results/workstation_20260904_verified/BENCHMARK_VERIFICATION.md`](benchmarks/results/workstation_20260904_verified/BENCHMARK_VERIFICATION.md)
- **Differentiable** spectra (autograd through the eigendecomposition) and a
  GPU path whose payoff is batched/gradient fitting rather than raw throughput
- **Fitting**: `esfit` with local (simplex, Powell, L-BFGS-B, Levenberg–Marquardt,
  bounded trust-region reflective `trf`),
  global (swarm, genetic, Monte Carlo, grid) and `global` (swarm + simplex
  polish) methods; population methods run in a process pool
  (`FitOptions(n_workers='auto')`; `pip install cloudpickle` for models defined in notebooks);
  `progress='text'` reports RMSD/evaluations/ETA live and a kernel interrupt returns the best fit so far

---

## Installation

```bash
pip install torchspin                 # core
pip install "torchspin[plot]"         # + matplotlib for the plotting helpers
pip install "torchspin[gui]"          # + the interactive fitting panel (torchspin.fitgui)
```

CPU-only PyTorch (avoids the large CUDA wheels):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torchspin
```

Requires Python 3.10+, PyTorch 2.0+, NumPy 1.23+, SciPy 1.10+.

From source, for development or to run the test suite — the MATLAB reference
data lives in the repository, not in the PyPI package, so the cross-validation
tests only run from a checkout:

```bash
git clone https://github.com/follmerlab/torchspin.git
cd torchspin
pip install -e ".[dev]"
pytest -q                             # full suite, ~25 min
pytest -q -m "not slow" -k "not matlab_validation"   # fast pass, ~4 min
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the test layout and what CI runs.

---

## Quick start

### CW powder spectrum (pepper)

```python
from torchspin import SpinSystem, Experiment, Options, pepper

sys = SpinSystem(
    S=[0.5],
    g=[[2.009, 2.006, 2.002]],
    Nucs=['14N'],
    A=[[10.0, 10.0, 95.0]],   # MHz
    lw=[1.0, 0.0],            # Gaussian FWHM (mT)
)
exp = Experiment(mwFreq=9.5, Range=[330, 350], nPoints=1024, Harmonic=1)
B, spec = pepper(sys, exp, Options(GridSize=50))
```

### Solution EPR (garlic)

```python
from torchspin import SpinSystem, Experiment, garlic

sys = SpinSystem(S=[0.5], g=2.003, Nucs='1H', A=10.0, lw=[0.2, 0.0])
exp = Experiment(mwFreq=9.5, Range=[336, 342], nPoints=512, Harmonic=1)
B, spec = garlic(sys, exp)
```

### Fitting (esfit)

```python
from torchspin import esfit, FitOptions

def model(p):
    sys_mod = SpinSystem(S=[0.5], g=[[p[0], p[1], p[2]]], lw=[1.0, 0.0])
    _, spc = pepper(sys_mod, exp, Options(GridSize=31, Verbosity=0))
    return spc.numpy()

result = esfit(
    data_measured, model,
    p0=np.array([2.0, 2.1, 2.2]),
    lb=np.array([1.95, 2.05, 2.15]),
    ub=np.array([2.05, 2.15, 2.25]),
    options=FitOptions(method='simplex'),
)
print('Best-fit g:', result.pfit)
```

### Differentiable spectrum

```python
import torch
from torchspin import differentiable_spectrum

g = torch.tensor([2.0, 2.1, 2.2], dtype=torch.float64, requires_grad=True)
B, spec = differentiable_spectrum(g, mwFreq_GHz=9.5, B_range=(300, 380),
                                   nPoints=512, lw_mT=1.0)
loss = ((spec - target_spec) ** 2).sum()
loss.backward()
print('dL/dg:', g.grad)
```

### Interactive notebooks

An index of all tutorials and reference documents is in [docs/index.md](docs/index.md).

Example notebooks live in `examples/notebooks/` and cover quickstart usage,
GPU/autograd workflows, spectral fitting, benchmark reproduction, and two
end-to-end fitting case studies.

- `01_torchspin_quickstart.ipynb` - core API tour: spin systems, Hamiltonians, powder and solution spectra.
- `02_gpu_and_autograd.ipynb` - GPU execution and differentiable spectrum workflows.
- `03_spectral_fitting.ipynb` - parameter recovery with `esfit` on simulated spectra.
- `04_benchmarks.ipynb` - MATLAB/EasySpin vs torchspin timings from the 2026-09 campaign (CPU threads, GPU, worker processes) plus live timing.
- `05_fitting_real_world.ipynb` - realistic multi-parameter fitting example.
- `06_blue_copper_fitting.ipynb` - end-to-end blue copper Cu(II) fitting case study.
- `07_easyspin_parity.ipynb` - overlays of torchspin against stored EasySpin references (pepper, garlic, chili, saffron, cardamom) with cosine similarities.
- `08_pepper_playground.ipynb` - simulate a `pepper` spectrum, add noise, then fit it with every `esfit` method and compare recovery, residual and time.
- `09_fit_gui.ipynb` - interactive fitting panel (`torchspin.fitgui`, `pip install "torchspin[gui]"`): sliders, fix boxes, Start/Stop, live fit and RMSD trace.

Recommended starting point: open `01_torchspin_quickstart.ipynb`, then move to
`02_gpu_and_autograd.ipynb` or `03_spectral_fitting.ipynb` depending on whether
you care more about differentiable simulation or fitting.

---

## Features

| Category | Modules |
|----------|---------|
| CW EPR | `pepper` (powder), `garlic` (solution), `chili` (slow-motion), `salt` (ENDOR) |
| Pulse EPR | `saffron` (ESEEM/HYSCORE), `spidyan` (arbitrary sequences), `saffron_thyme` (real pulses) |
| Trajectory | `cardamom` (MD/diffusion/jump), `mdload`, `mdhmm` |
| Magnetometry | `curry` (susceptibility, magnetization) |
| Fitting | `esfit` (10 methods: simplex, L-BFGS-B, Powell, Levenberg–Marquardt, trust-region reflective `trf`, grid, Monte Carlo, genetic, swarm, and `global` = swarm + simplex polish); `torchspin.fitgui` interactive panel |
| Autograd | `differentiable_spectrum` (`pepper`'s own forward path via `pepper_autograd`); `garlic`, `salt`, `saffron` and `curry` are differentiable too |
| ML | `torchspin.ml` — parameter packing (`ParamSpec`), batched simulation, `SpectrumDataset`, differentiable losses |
| Data I/O | `eprload` (14 vendor formats), `eprsave` (BES3T), `orca2torchspin` |
| Batch/GPU | `batch_pepper`, `batch_simulate`; `Options.device='cuda'` |

---

## Conventions

- **Energy units:** MHz throughout
- **Field units:** mT
- **Euler angles:** radians, z-y'-z'' **passive** rotation
- **Basis ordering:** m = S, S−1, …, −S (descending)
- **Default dtype:** `torch.complex128`

---

## Validation status

Every simulator with an EasySpin counterpart is validated against stored MATLAB
outputs (`tests/data/*.mat`, generated by the scripts in `tests/`); the
validation modules are `torchspin/tests/test_*_matlab_validation.py`; what
changed when is in `CHANGELOG.md`.
As of 2026-10-04 (2659 passed, 0 failed, 4 skipped, 3 xfailed, on the clean-room host):

- `pepper` (powders, crystals, strains, frequency sweeps, non-equilibrium
  populations, photoselection, ordering, isotopologues), `garlic`, `salt`,
  `endorfrq`, `saffron`, `spidyan`, `curry`/`blochsteady`: cosine ≥ 0.999
  (most 0.9999–1.0000), absolute intensities included where EasySpin defines them;
- `chili`: 60-case suite, 59 at cosine ≥ 0.999;
- `esfit`: fitted parameters within 3×10⁻⁴ (g) and 0.03 mT (linewidth) of
  EasySpin's on the shared cross-validation cases;
- `cardamom`: stochastic, cosine > 0.75 at 100 trajectories by design.

Performance against MATLAB/EasySpin (same nine workloads, one workstation):
`benchmarks/results/workstation_20260904_verified/BENCHMARK_VERIFICATION.md`
(audited and authoritative) with the earlier rounds in
`benchmarks/results/BENCHMARK_REPORT.md`; remaining gaps and open items:
`KNOWN_LIMITATIONS.md`.

---

## Citation

If you use torchspin in published work, please cite both:

- **EasySpin** (the MATLAB toolbox this port is based on):
  Stoll, S. & Schweiger, A. *J. Magn. Reson.* **178** (2006) 42–55.
- **TorchSpin** (the PyTorch reimplementation):
  See `CITATION.cff` in the repository root.

---

## Related

- [EasySpin](https://easyspin.org) — the MATLAB reference implementation
- [PyTorch](https://pytorch.org) — autograd engine

---

## License

MIT. See `LICENSE.md`.
