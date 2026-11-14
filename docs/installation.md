# Installing torchspin

torchspin is a pure-Python package on PyPI. Its dependencies are PyTorch, NumPy and SciPy, and the only
real decision during installation is **which PyTorch build you get**: CPU-only, or CUDA for an NVIDIA
GPU. torchspin cannot make that choice for you (see
[What the package metadata can and cannot enforce](#what-the-package-metadata-can-and-cannot-enforce)),
so this page shows how to make it with pip and with uv.

What was actually tested, on which system, is recorded in
[installation_validation.md](installation_validation.md). In short: every command on this page was run
on macOS (Apple silicon). The Linux, Windows and CUDA commands are untested, apart from checking how uv
resolves them for those platforms.

## Quick reference

| You want | Commands |
|---|---|
| CPU only, any OS (pip) | `pip install torch --index-url https://download.pytorch.org/whl/cpu` then `pip install torchspin` |
| CPU only, any OS (uv) | `uv pip install torchspin --torch-backend=cpu` |
| NVIDIA GPU, Linux | `pip install torchspin` (PyPI's Linux PyTorch wheel is a CUDA build), or choose the CUDA version: see [NVIDIA GPU](#nvidia-gpu-linux-and-windows) |
| NVIDIA GPU, Windows | `pip install torch --index-url https://download.pytorch.org/whl/cu130` (pick your `cuXXX`) then `pip install torchspin` |
| GPU auto-detect (uv) | `uv pip install torchspin --torch-backend=auto` |
| macOS | `pip install torchspin` (torchspin runs on the CPU there; see [macOS](#macos)) |
| Keep the PyTorch you already have | `pip install torchspin` keeps it if it is compatible; see [Keeping an existing PyTorch](#keeping-an-existing-pytorch) |
| Tutorials | add the `notebooks` extra: `pip install "torchspin[notebooks]"`; see [Notebooks](#notebooks) |

## Requirements

- Python 3.10 or newer. Current PyTorch releases need Python 3.11 or newer, so on Python 3.10 pip and uv
  install the newest PyTorch that still supports 3.10 (2.14.1 in November 2026).
- PyTorch 2.2 or newer, and PyTorch 2.3 or newer together with NumPy 2. The package metadata declares
  `torch>=2.0,<3`, but PyTorch 2.0 and 2.1 fail in `pepper`, and PyTorch 2.2 cannot use NumPy 2 (see the
  validation record). A fresh install always gets a recent PyTorch. This matters only if you keep an
  older one (with PyTorch 2.2, add `"numpy<2"` to the install command).
- NumPy 1.23 or newer, SciPy 1.10 or newer (installed automatically).

The version on PyPI can lag the repository. Check `pip index versions torchspin` or
<https://pypi.org/project/torchspin/>. To install the current `main` branch instead:

```bash
pip install "torchspin @ git+https://github.com/follmerlab/torchspin"
```

Use a virtual environment for everything below. With pip:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

With uv, `uv venv` creates `.venv` and every later `uv pip` command in that directory uses it.

## Choosing the PyTorch build

What a plain `pip install torchspin` (or `uv pip install torchspin`) gets you depends on the platform,
because PyTorch publishes different wheels on PyPI for each platform:

| Platform | PyTorch wheel from PyPI | Notes |
|---|---|---|
| Linux x86_64 and aarch64 | CUDA build. It pulls in the CUDA runtime as separate packages (NVIDIA libraries, several GB) | It still runs on machines without a GPU. Use the CPU index to avoid the download |
| Windows x86_64 | CPU-only build | For CUDA, install PyTorch from the PyTorch index first |
| macOS, Apple silicon | The only macOS build (CPU and MPS) | torchspin does not run on MPS, see [macOS](#macos) |
| macOS, Intel | None for current PyTorch. The last x86_64 macOS wheels are PyTorch 2.2.2 | Untested, see [macOS](#macos) |

To get a different build, install PyTorch **first** from the PyTorch package index. Then install
torchspin, which keeps that PyTorch (see [Keeping an existing PyTorch](#keeping-an-existing-pytorch)).
The PyTorch indexes have the form `https://download.pytorch.org/whl/<backend>`, where `<backend>` is
`cpu`, a CUDA version (`cu126`, `cu128`, `cu129`, `cu130`, ...), a ROCm version or `xpu`. The
selector at <https://pytorch.org/get-started/locally/> shows which ones the current release supports.

## pip

### CPU only

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torchspin
```

On Linux this avoids the CUDA download. On Windows and macOS PyPI's PyTorch wheel is already a CPU
build, so the first line is optional there. On Linux and Windows the CPU index serves `+cpu` builds
(for example `2.15.0+cpu`); on macOS it serves the same wheel as PyPI.

Use **`--index-url` for the PyTorch line only**, as a separate command. Avoid
`pip install torchspin --extra-index-url https://download.pytorch.org/whl/cpu`:

- pip gives its indexes no priority. It collects candidates from all of them and installs the "best"
  version it finds ([pip docs: Finding Packages](https://pip.pypa.io/en/stable/cli/pip_install/#finding-packages)).
  If PyPI has a newer PyTorch than the CPU index, pip takes PyPI's, which on Linux is the CUDA build.
- Mixing indexes for every package opens the door to dependency confusion, which pip's documentation warns
  about under `--extra-index-url`
  ([pip install examples](https://pip.pypa.io/en/stable/cli/pip_install/#examples)).
- `pip install torchspin --index-url https://download.pytorch.org/whl/cpu` fails, because torchspin is not
  on the PyTorch index.

### NVIDIA GPU (Linux and Windows)

Linux: `pip install torchspin` installs PyPI's PyTorch, which is a CUDA build. To pin a CUDA version
that matches your driver, or on Windows (where PyPI's wheel is CPU-only):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130   # pick the cuXXX for your driver
pip install torchspin
```

Check that PyTorch sees the GPU, then use it in torchspin with `Options(device='cuda')`:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

If you request `device='cuda'` when CUDA is unavailable, torchspin warns (`RuntimeWarning`) and runs on
the CPU. AMD (ROCm) and Intel (XPU) PyTorch builds have not been tried with torchspin.

### Keeping an existing PyTorch

pip's default upgrade strategy is `only-if-needed`: dependencies that already satisfy the requirement are
left alone ([pip docs: `--upgrade-strategy`](https://pip.pypa.io/en/stable/cli/pip_install/#cmdoption-upgrade-strategy)).
So if a compatible PyTorch is already installed, `pip install torchspin` and `pip install --upgrade torchspin`
keep it, whichever index it came from. This was validated with a PyTorch 2.5.1 CPU build installed first.

PyTorch is replaced when:

- you ask for it (`pip install --upgrade torchspin torch`, or `--upgrade-strategy eager`), or
- the installed version does not satisfy `torch>=2.0,<3`.

In both cases the replacement comes from the indexes in that command, normally PyPI, which may be a
different build. On Linux, a CPU build gets replaced by a CUDA one. On Windows, a CUDA build gets
replaced by a CPU one. To make pip refuse instead of replacing, pin the installed version with a
constraints file:

```bash
pip freeze | grep -i "^torch==" > torch-constraint.txt     # e.g. torch==2.5.1+cpu
pip install --upgrade -c torch-constraint.txt torchspin
```

(On Windows PowerShell: `pip freeze | Select-String "^torch==" > torch-constraint.txt`.)

With conda, install PyTorch, NumPy and SciPy with conda, then run `pip install torchspin` inside the same
environment. This has not been tested here. If pip proposes to replace conda's PyTorch, cancel and use
`pip install --no-deps torchspin` instead.

## uv

uv has two interfaces, and they choose PyTorch in different ways.

### `uv pip` (drop-in for pip, works in any virtual environment)

```bash
uv venv
uv pip install torchspin --torch-backend=cpu     # CPU-only PyTorch
uv pip install torchspin --torch-backend=cu130   # a specific CUDA version
uv pip install torchspin --torch-backend=auto    # detect the GPU, else CPU
```

`--torch-backend` makes uv fetch PyTorch packages from the matching PyTorch index. With `auto`, uv
looks for an NVIDIA driver, an AMD GPU or an Intel GPU, and falls back to the CPU index if it finds none
([uv docs: PyTorch, automatic backend selection](https://docs.astral.sh/uv/guides/integration/pytorch/#automatic-backend-selection)).
The option was checked with uv 0.13.8. It exists for `uv pip install`, `uv pip compile` and
`uv pip sync` only, not for `uv add` or `uv sync`. Without it, `uv pip` behaves like pip:
`uv pip install torch --index-url https://download.pytorch.org/whl/cpu` followed by
`uv pip install torchspin` also works.

Keeping an existing PyTorch: `uv pip install torchspin` leaves a compatible installed PyTorch alone.
However, **`uv pip install --upgrade torchspin` upgrades every package, including PyTorch** (from PyPI,
unless you pass `--torch-backend` or an index). Upgrade only torchspin with:

```bash
uv pip install --upgrade-package torchspin torchspin
```

or pass the constraints file from the pip section with `-c torch-constraint.txt`. Both were validated.

### uv projects (`uv add torchspin` in your own project)

`uv add torchspin` resolves PyTorch from PyPI, with the per-platform builds in the table above.
Nothing in torchspin's published metadata can change that. To choose the build, configure **your**
project's `pyproject.toml`
([uv docs: Using uv with PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/)):

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true            # use this index only for packages pinned to it below

[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]
```

`uv add torch --index pytorch-cpu=https://download.pytorch.org/whl/cpu` writes the same configuration
without `explicit = true`, so add that line yourself. Then `uv add torchspin`, `uv sync` and `uv run`
use CPU PyTorch on every platform. Validated: with this configuration the lockfile pins `+cpu` wheels for
Linux and Windows and contains no NVIDIA packages; without it, the lockfile contains 45 NVIDIA/CUDA
packages for Linux. For CUDA on Linux and Windows with CPU on macOS, or for CPU/CUDA chosen by an extra,
use the `marker`- and `extra`-based examples in the uv guide.

### Working in the torchspin repository

These settings live in `[tool.uv.*]` tables, which only uv reads
([uv docs: "Sources are only respected by uv"](https://docs.astral.sh/uv/concepts/projects/dependencies/#dependency-sources)).
They take effect where uv resolves that `pyproject.toml` itself: `uv sync`, `uv run`, `uv lock`, and
`uv pip install` run on the checkout. They are not part of the published wheel, so they never affect
`pip install torchspin` or `uv add torchspin` in someone else's project. torchspin's repository has no
uv configuration at present, so in a clone use `--torch-backend`:

```bash
git clone https://github.com/follmerlab/torchspin.git
cd torchspin
uv venv
uv pip install -e ".[dev,plot]" --torch-backend=cpu   # or auto, or cu130
```

The pip equivalent is in [CONTRIBUTING.md](../CONTRIBUTING.md). Install CPU PyTorch first as above if
you are on Linux and do not need CUDA.

## Platform notes

### macOS

- **Apple silicon:** `pip install torchspin` is all you need. **torchspin does not run on the Apple GPU
  (MPS).** It computes in float64, which MPS does not support, so `Options(device='mps')` fails with
  `TypeError: Cannot convert a MPS Tensor to float64 dtype ...`. Use the CPU (the default).
- **Intel Macs:** PyTorch no longer publishes macOS x86_64 wheels. The last were PyTorch 2.2.2 (Python
  3.12 at most), so pip falls back to 2.2.2. That version needs NumPy < 2, so ask for it in the same
  command and let pip pick a matching SciPy: `pip install torchspin "numpy<2"`. The PyTorch 2.2.2 +
  NumPy 1.26 combination was validated on Apple silicon. Untested on an Intel Mac.
- In zsh (the default macOS shell) quote anything with square brackets: `pip install "torchspin[notebooks]"`.

### Linux

- PyPI's PyTorch wheel bundles CUDA for both x86_64 and aarch64. On CPU-only machines, CI runners and
  clusters, install from the CPU index first (or use `uv pip install ... --torch-backend=cpu`) to avoid
  several GB of NVIDIA packages.
- Current PyTorch wheels are `manylinux_2_28`, so they need glibc 2.28 or newer.
- On clusters that provide PyTorch as a module or in a shared environment, keep that PyTorch:
  install torchspin into the same environment with `pip install torchspin` (see
  [Keeping an existing PyTorch](#keeping-an-existing-pytorch)).

### Windows

- PyPI's PyTorch wheel is CPU-only. For an NVIDIA GPU, install PyTorch from
  `https://download.pytorch.org/whl/cuXXX` first.
- Activate a virtual environment with `.venv\Scripts\activate`. In PowerShell, quote extras the same way
  as above.

## Optional extras

| Extra | Adds | For |
|---|---|---|
| `plot` | matplotlib | the plotting helpers (`levelsplot`, `stackplot`, ...) |
| `fit` | cloudpickle | process-pool `esfit` with models defined in a notebook or closure |
| `hpc` | threadpoolctl, cloudpickle | BLAS thread caps and process pools |
| `gui` | ipywidgets, matplotlib, ipympl, IPython, cloudpickle | the interactive fitting panel `torchspin.fitgui` |
| `notebooks` | Jupyter (Notebook and JupyterLab), ipykernel, matplotlib, ipywidgets, ipympl, cloudpickle | the tutorials in `examples/notebooks/` |
| `test`, `dev` | pytest, build, twine, ... | development |

Extras combine: `pip install "torchspin[plot,hpc]"`. The PyTorch build is chosen as described above and
is the same with or without extras.

## Notebooks

The tutorials live in [`examples/notebooks/`](../examples/notebooks/) and are not part of the installed
package. To run them:

```bash
git clone https://github.com/follmerlab/torchspin.git
cd torchspin
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional, see above
pip install "torchspin[notebooks]"        # or: pip install -e ".[notebooks]" for the checkout's version
jupyter lab examples/notebooks
```

The `notebooks` extra is derived from what notebooks 01 to 09 import or need beyond the core
dependencies: matplotlib (01 to 08), ipywidgets and ipympl (09 uses `%matplotlib widget` and
`torchspin.fitgui`), and cloudpickle (08 and 09 run `esfit` in worker processes with models defined in
notebook cells). It also installs Jupyter itself.

- Notebooks 04 (benchmarks) and 07 (EasySpin parity) read files from the repository (`benchmarks/`,
  `tests/data/`), so run them from a clone.
- Notebook 02 has GPU cells. Without CUDA they fall back to the CPU or are skipped.
- If you already use Jupyter from another environment (JupyterHub, VS Code), install
  `"torchspin[gui]" ipykernel` in the torchspin environment instead and register it as a kernel:
  `python -m ipykernel install --user --name torchspin`.
- The `notebooks` extra is new after 0.9.31. A release that predates it prints
  `WARNING: torchspin ... does not provide the extra 'notebooks'` and installs only the core package. In
  that case use `pip install "torchspin[gui]" jupyter`.

## Check the installation

```python
import torch, torchspin
from torchspin import SpinSystem, Experiment, Options, pepper

sys_ = SpinSystem(S=[0.5], g=[[2.0, 2.1, 2.2]], lw=[1.0, 0.0])
exp = Experiment(mwFreq=9.5, Range=[300.0, 350.0], nPoints=256, Harmonic=1)
B, spc = pepper(sys_, exp, Options(GridSize=19, Verbosity=0))
print(torchspin.__version__, torch.__version__, torch.version.cuda, spc.shape)
```

`torch.version.cuda` is `None` for CPU builds and the CUDA version (for example `13.0`) for CUDA builds.

## What the package metadata can and cannot enforce

A wheel on PyPI states its dependencies as `Requires-Dist` entries: a package name, version bounds,
optional environment markers (Python version, OS, machine) and extras. For torchspin that is
`torch>=2.0,<3`, `numpy>=1.23,<3` and `scipy>=1.10`.

- **It can** require a PyTorch version range, and could vary it by OS, machine or Python version
  through markers.
- **It cannot** choose CPU vs CUDA, or a CUDA version. To the installer, `torch 2.15.0` and
  `torch 2.15.0+cpu` are the same project, and the build is decided by which index the installer
  searches. The metadata has no field for a package index
  ([core metadata specification](https://packaging.python.org/en/latest/specifications/core-metadata/)).
- **It cannot** point at the PyTorch index through a direct URL either. Dependency specifiers do allow
  `name @ URL`
  ([dependency specifiers](https://packaging.python.org/en/latest/specifications/dependency-specifiers/)),
  but "PyPI and other standards-conformant package indices **do not** accept packages that declare
  dependencies using direct URLs"
  ([setuptools: Direct URL dependencies](https://setuptools.pypa.io/en/latest/userguide/dependency_management.html#direct-url-dependencies)).
  PyPI rejects such an upload with "Can't have direct dependency"
  ([Warehouse source](https://github.com/pypi/warehouse/blob/main/warehouse/forklift/metadata.py)).
  This applies to extras too, so an extra like `cpu = ["torch @ https://download.pytorch.org/whl/cpu"]`
  cannot be published. It would not work anyway, because that URL is an index, not a wheel file.
- **An extra named `cpu` or `cuda`** that just lists `torch` would publish, but it would mislead:
  `pip install "torchspin[cpu]"` would still fetch PyTorch from PyPI. That is why torchspin has no such
  extras.
- **uv's `[tool.uv.sources]` and `[[tool.uv.index]]`** are not metadata. Only uv reads them, and only in
  the project that contains them (see [uv projects](#uv-projects-uv-add-torchspin-in-your-own-project)).
- The proposed "wheel variants" standard would let an installer choose a GPU build automatically
  ([PEP 817](https://peps.python.org/pep-0817/), Draft, and
  [PEP 825](https://peps.python.org/pep-0825/), Provisional, as of November 2026). It is not usable
  with mainstream pip today.
