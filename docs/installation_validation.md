# Installation validation record

This page records which routes in [installation.md](installation.md) were run, where, and with what
result. Update it when the routes or the platforms change.

## Environment

- Date: 2026-11-14
- Machine: macOS 15.5 (24F74), Apple silicon (arm64), 14 cores, no NVIDIA GPU
- Python: 3.12.15 (Anaconda, `python3 -m venv`), plus 3.10.22, 3.11.18 and 3.13.18 (uv-managed)
- pip: 25.0.1 in fresh `venv`s (bundled), 26.3 for building and tooling
- uv: 0.13.8 (3d97a6aad 2026-11-12, aarch64-apple-darwin), installed into a scratch venv
- build: 1.6.1 (setuptools backend in an isolated build environment)
- torchspin: wheel and sdist built from commit 807b78b (0.9.31 plus the `notebooks` extra) with
  `python -m build`, from a `git archive` export. Installing that wheel stands in for installing from PyPI.
- PyTorch at the time: 2.15.0 on PyPI (`Requires-Python >=3.11`). The PyTorch indexes served
  `cpu`, `cu118`, `cu126`, `cu128`, `cu129`, `cu130`, `rocm6.4` and `xpu` (HTTP 200).
- torchspin on PyPI at the time: 0.8.0 (the repository was at 0.9.31, later 0.9.32).

Every environment was new and empty, and pip/uv caches were redirected to a scratch directory. Each
route ended with a smoke test: `import torchspin`, then a 256-point `pepper` powder spectrum
(`S=1/2`, rhombic g, `GridSize=19`), with the shape and finiteness checked and the torchspin/PyTorch
versions and `torch.version.cuda` printed.

## Validated routes (macOS arm64)

| # | Route | Commands (abridged) | Result |
|---|---|---|---|
| 1 | pip, default PyTorch | `pip install <wheel>` | OK. torch 2.15.0 from PyPI, `cuda=None` |
| 2 | pip, CPU index first | `pip install torch --index-url .../whl/cpu`, then `pip install <wheel>` | OK. torch kept |
| 3 | pip keeps an existing PyTorch | `pip install torch==2.5.1 --index-url .../whl/cpu`, then `pip install <wheel>` and `pip install -U <wheel>` | OK. pip printed `Requirement already satisfied: torch<3,>=2.0 ... (2.5.1)` both times. torch stayed 2.5.1 |
| 3b | pip `--upgrade-strategy eager` (dry run) | `pip install -U --upgrade-strategy eager --dry-run <wheel>` | Would replace torch 2.5.1 with 2.15.0 from PyPI, as documented |
| 3c | pip with a constraints file | `pip freeze \| grep ^torch== > c.txt`, then `pip install -U --upgrade-strategy eager -c c.txt <wheel>` (dry run) | torch held at 2.5.1. `pip install -c c.txt "torch>=2.6"` fails with `ResolutionImpossible`, as intended |
| 4 | uv pip keeps an existing PyTorch | `uv pip install torch==2.5.1 --index-url .../whl/cpu`, then `uv pip install <wheel>` | OK. torch stayed 2.5.1 |
| 4b | `uv pip install --upgrade` (dry run) | `uv pip install --upgrade --dry-run <wheel>` | **Would replace torch 2.5.1 with 2.15.0.** `--upgrade` upgrades every package. Documented |
| 4c | `uv pip install --upgrade-package torchspin` (dry run) | as named | `Would make no changes`. torch kept |
| 4d | uv pip `--upgrade` with a constraints file (dry run) | `uv pip install --upgrade -c c.txt <wheel>` | torch held at 2.5.1 |
| 5 | uv pip `--torch-backend=cpu` | `uv venv`, then `uv pip install <wheel> --torch-backend=cpu` | OK. torch 2.15.0 from `download.pytorch.org/whl/cpu` (verbose log) |
| 5b | uv pip `--torch-backend=auto` | as above with `auto` | OK. No GPU found, so the CPU index was used |
| 5c | Python 3.10 and 3.13 | route 5 on uv-managed 3.10.22 and 3.13.18 | OK. 3.10 got torch 2.14.1 (newest with 3.10 wheels), 3.13 got 2.15.0 |
| 6 | uv project, default | `uv init --bare`, then `uv add <wheel>` and `uv run python smoke.py` | OK. `uv.lock` pins torch to `https://pypi.org/simple`. For Linux the lock contains 45 `nvidia-*`/`cuda-toolkit` package entries |
| 6b | uv project with an explicit CPU index | `uv add torch --index pytorch-cpu=https://download.pytorch.org/whl/cpu` | OK. `uv.lock` pins torch to the CPU index with `+cpu` wheels for `manylinux_2_28_x86_64` and `win_amd64`, and 0 NVIDIA packages. `uv add --index` does not write `explicit = true` (documented) |
| 7 | real PyPI | `pip install torchspin` | OK. Installs **0.8.0**. Its `Requires-Dist` matches the repository except for the new extra. `pip install "torchspin[notebooks]"` prints `WARNING: torchspin 0.8.0 does not provide the extra 'notebooks'` (documented) |
| 7b | GitHub main | `pip install "torchspin @ git+https://github.com/follmerlab/torchspin"` | OK. Installed 0.9.32 (commit 723e73e) |
| 8 | editable, pip (CONTRIBUTING route) | `pip install torch --index-url .../whl/cpu`, then `pip install -e ".[dev,plot]"` | OK. Imported from the checkout. `pytest -m "not slow" -k "not matlab_validation" torchspin/tests/test_pepper*.py`: 346 passed, 3 skipped, 4 xfailed |
| 9 | editable, uv | `uv venv`, then `uv pip install -e ".[dev,plot]" --torch-backend=cpu` | OK |
| 10 | MPS | `pepper(..., Options(device='mps'))` | **Fails**: `TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64`. `device='cpu'` OK. Documented as unsupported |
| 11 | `notebooks` extra | `pip install torch --index-url .../whl/cpu`, then `pip install "torchspin[notebooks] @ file://<wheel>"` | OK. torch kept. Installed jupyter 1.1.1, jupyterlab 4.6.6, notebook 7.6.5, ipykernel 7.4.0, ipywidgets 8.1.9, ipympl 0.10.0, matplotlib 3.11.2, cloudpickle 3.1.2, nbconvert 7.17.2 |
| 12 | proposed CI job, routes part | the `routes` job steps of `.github/workflows/install-docs.yml`, run locally | OK on macOS. torch kept, CPU builds, and the dry-run report has no CUDA packages |

The constraints-file and `--upgrade` checks (3b, 3c, 4b, 4c, 4d) were dry runs. The commands in
3 and 4 were real installs.

### Notebooks executed with the `notebooks` extra (route 11)

`jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=<T>` (T = 600 s for 01,
1200 s for 02 to 09, 3600 s for the 03 retry), run from
`examples/notebooks/` in a clean export of the repository, with the wheel-installed torchspin (not the
checkout). The machine was shared with other heavy jobs (load average up to about 140 on 14 cores).

| Notebook | Result | Wall time |
|---|---|---|
| 01 quickstart | OK | 64 s |
| 02 GPU and autograd | OK (CPU only; GPU parts skipped) | 64 s |
| 03 spectral fitting | First run: cell timeout at T = 1200 s in the global-fit cell (the shipped output records 953.8 s for that cell). Retry with T = 3600 s: OK. The slowness came from machine load, not from the installation | 1570 s (timeout), 3813 s (retry) |
| 04 benchmarks | OK | 43 s |
| 05 real-world fitting | OK | 767 s |
| 06 blue copper | OK | 792 s |
| 07 EasySpin parity | OK (reads `tests/data/` from the checkout) | 15 s |
| 08 pepper playground | OK (`n_workers='auto'`: 12 spawned worker processes, notebook-cell model shipped by cloudpickle, no thread-pool fallback warning) | 642 s |
| 09 fit GUI | OK headless (`%matplotlib widget`, `torchspin.fitgui` panel built, a 740-evaluation `FitSession` fit completed) | 30 s |

## PyTorch version floor (metadata says `torch>=2.0,<3`)

Smoke test with older PyTorch versions installed first, then the wheel:

| PyTorch | NumPy | SciPy | Python | Result |
|---|---|---|---|---|
| 2.0.1 | 2.4.6 | 1.17.1 | 3.11 | Fails: `A module that was compiled using NumPy 1.x cannot be run in NumPy 2` / `RuntimeError: Numpy is not available` |
| 2.0.1 | 1.26.4 | 1.17.1 | 3.11 | Fails in `pepper`: `TypeError: any() received an invalid combination of arguments - got (dim=tuple, )` (`torchspin/resfields_batch.py:224`, tuple `dim` needs PyTorch 2.2) |
| 2.1.2 | 1.26.4 | 1.13.1 | 3.11 | Same `any(dim=tuple)` failure |
| 2.2.2 | 2.5.4 | 1.18.1 | 3.12 | Fails: `RuntimeError: Numpy is not available` |
| 2.2.2 | 1.26.4 | 1.17.1 | 3.12 | OK (`pip install torch==2.2.2 <wheel> "numpy<2"` in one command) |
| 2.3.1 | 2.5.4 | 1.18.1 | 3.12 | OK |
| 2.4.1 | 2.5.4 | 1.18.1 | 3.12 | OK |
| 2.5.1 | 2.5.4 | 1.18.1 | 3.12 | OK |

The working floor is PyTorch 2.2, or 2.3 with NumPy 2. installation.md and the README state that floor.
The metadata still says 2.0 (see the open question below). On Python 3.11 under uv-managed CPython,
pip and uv chose the `macosx_11_0_x86_64` wheel for PyTorch 2.0.1 and 2.1.2. Those releases tag their
arm64 wheels `cp311-none-...`. The `any(dim=tuple)` failure is an API error and does not depend on the
architecture.

## Repository-specific uv configuration (prototype, not added)

This configuration was tried in a scratch copy of the repository (appended to `pyproject.toml`):

```toml
[dependency-groups]          # PEP 735: not part of the published wheel metadata
cpu = ["torch>=2.0,<3"]
cu130 = ["torch>=2.0,<3"]

[tool.uv]
conflicts = [[{ group = "cpu" }, { group = "cu130" }]]

[tool.uv.sources]
torch = [
    { index = "pytorch-cpu", group = "cpu" },
    { index = "pytorch-cu130", group = "cu130", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true
```

Results: `uv lock` resolved 217 packages, with `+cpu` and `+cu130` wheels for Linux and Windows.
`uv sync --group cpu --extra plot` followed by `uv run --group cpu` passed the smoke test.
`uv sync --group cpu --group cu130` fails with the conflict error, as intended. `uv build` produced a
wheel whose `Requires-Dist`/`Provides-Extra` lines are identical to the wheel built without the
configuration. `pip install -e ".[dev]"` from that copy and `pip install` of that wheel both passed the
smoke test.

It was not added, for these reasons. `uv pip install -e ".[dev,plot]" --torch-backend=cpu` (route 9)
gives contributors the same result with no configuration. A plain `uv sync` would still take PyTorch
from PyPI. The named CUDA index (`cu130`) would need maintenance as PyTorch moves on. And the
repository does not commit a `uv.lock`. Extras (rather than dependency groups) were rejected because
they would be published: `pip install "torchspin[cpu]"` would look like a CPU install but would fetch
PyTorch from PyPI.

## Untested

- **Linux** (x86_64, aarch64): no install was run. Evidence is limited to PyPI metadata
  (torch 2.15.0 requires `cuda-toolkit[...]==13.2.2; platform_system == "Linux"` and
  `nvidia-*-cu13` packages, and the `manylinux_2_28_x86_64` wheel is 539 MB) and uv's universal lock
  (routes 6 and 6b). `pip install --dry-run --platform manylinux_2_28_x86_64` on macOS cannot check
  this, because pip evaluates environment markers for the running machine, so it reported no CUDA
  packages.
- **Windows**: no install was run. The claim that PyPI's wheel is CPU-only rests on the metadata (CUDA
  dependencies are Linux-only, and `win_amd64` is 127 MB) and on uv's lock.
- **CUDA on any platform**: no NVIDIA GPU was available. The `--index-url .../whl/cu130` route,
  `--torch-backend=cu130` / `auto` with a GPU, and `Options(device='cuda')` are untested here (the GPU
  consistency tests in `torchspin/tests/test_gpu_consistency.py` cover the latter on CUDA workstations).
- **AMD ROCm, Intel XPU**: untested and not claimed.
- **Intel Macs**: untested. The PyTorch 2.2.2 + NumPy 1.26 combination was validated on arm64 only.
- **conda / mamba** environments with conda-installed PyTorch: untested.
- **JupyterLab interactive use** of notebook 09 (`%matplotlib widget`, live panel): only headless
  `nbconvert` execution was checked.

GitHub Actions (`python-tests.yml`) runs the test suite on Linux, Windows and macOS after installing CPU
PyTorch and `pip install -e ".[dev]"`. It does not check the documented routes. The proposed
`.github/workflows/install-docs.yml` does, for CPU builds on all three operating systems, and it checks
that only the Linux PyPI resolution pulls CUDA packages.

## Open questions

1. Raise the `torch` floor in `pyproject.toml` from `>=2.0` to `>=2.2` (or `>=2.3`, which also rules out
   the NumPy 2 incompatibility)? This changes published metadata, so it is a release decision. It was
   not changed here.
2. The `notebooks` extra only reaches users with the next PyPI release. PyPI is at 0.8.0.
