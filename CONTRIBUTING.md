# Contributing to torchspin

Thanks for your interest in torchspin. This is a small open-source project
maintained by the Follmer Lab. Contributions of any size are welcome —
bug reports, documentation fixes, new tests, and pull requests.

## Quick start (development install)

```bash
git clone https://github.com/follmerlab/torchspin.git
cd torchspin
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,plot]"
```

The optional `dev` group pulls in `pytest`, `pytest-cov`, `build`, and
`twine`. The `plot` group pulls in `matplotlib` for plotting helpers and
the benchmark figures.

## Running tests

```bash
# Everything, including the MATLAB cross-validation suite (the reference
# .mat files are in the repository, so no MATLAB installation is needed)
pytest -q

# Fast pass: skip the slow tests and the cross-validation suite
pytest -q -m "not slow" -k "not matlab_validation"

# Also run the minutes-long exact-diagonalization comparisons, which are
# skipped by default (one of them takes about 10 minutes)
TORCHSPIN_RUN_EXPENSIVE=1 pytest -q -k cupc

# Only the GPU consistency suite (workstations with CUDA)
pytest torchspin/tests/test_gpu_consistency.py -v

# Only MATLAB cross-validation
pytest -q -k matlab_validation
```

A bare `pytest` picks up `testpaths = ["torchspin/tests"]` from
`pyproject.toml`.  The top-level `tests/` directory holds reference data and
the MATLAB scripts that generated it, not test code.  The cross-validation
tests are selected by module name (`-k matlab_validation`); the
`requires_matlab_data` marker is registered but not currently applied to any
test, so filtering on it deselects nothing.

CI (`.github/workflows/python-tests.yml`) runs a fast cross-platform matrix
(Linux/macOS/Windows x Python 3.10-3.13, cross-validation deselected) plus one
`full-suite` job on Linux that runs everything with `TORCHSPIN_STRICT_REFS=1`,
so a missing reference file fails the build instead of silently skipping.
That job has no marker filter, so a test that takes minutes costs minutes on
every build: the two exact-diagonalization CuPc comparisons are therefore
skipped unless `TORCHSPIN_RUN_EXPENSIVE=1` is set, rather than marked `slow`.

The full passing baseline is **2718 passed, 0 failed, 16 skipped, 3 documented
xfails** on a CUDA-less machine (14 of the skips need a GPU, 2 are the
expensive comparisons above).  See
[KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md).

## Adding a MATLAB cross-validation test

torchspin's scientific accuracy claims are anchored in MATLAB EasySpin
reference data. To add a new validation test:

1. Write a MATLAB script in `tests/` (e.g. `mysim_ref.m`) that runs
   EasySpin and saves the output to `tests/data/mysim_ref.mat`.
2. In MATLAB: `matlab -batch "addpath('easyspin'); run('tests/mysim_ref.m')"`.
3. Add a Python test that loads the `.mat` file and compares to torchspin
   output via cosine similarity (and ideally peak-position + amplitude).
   See `torchspin/tests/test_pepper_matlab_validation.py` for the pattern.
4. Update `tests/data/DATA_PROVENANCE.md` to mark the new file as MATLAB-
   generated.

## Reporting bugs

Open an issue at <https://github.com/follmerlab/torchspin/issues> with:
- A minimal reproducer (`SpinSystem` + `Experiment` + the call you made)
- The output you got vs what you expected
- Your platform: OS, Python version, PyTorch version, CUDA availability
- Output of `python -c "import torchspin; print(torchspin.__version__)"`

## Pull requests

- Keep PRs focused on one issue or feature
- Run `pytest torchspin/tests/ -q` and ensure no new failures before opening
- If you add a public function, add a docstring with at least one example
- If you change scientific output, add or update a MATLAB-validation test

## Conventions

- Energy units: MHz throughout
- Field units: mT
- Euler angles: radians, z-y'-z'' passive rotation
- Default dtype: `torch.complex128`
- Naming: match MATLAB EasySpin where possible (`SpinSystem`, `Experiment`,
  `Options`, `pepper`, `garlic`, `chili`, `esfit`, `saffron`)

## License

By contributing you agree your contributions are licensed under the MIT
License (see `LICENSE.md`).
