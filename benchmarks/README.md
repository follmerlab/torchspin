# torchspin Benchmark Suite

Reproducible benchmarks comparing **MATLAB EasySpin**, **torchspin on CPU**, and
**torchspin on NVIDIA GPU** for the manuscript.

## Layout

```
benchmarks/
├── README.md                 (this file)
├── common/
│   └── spin_systems.py       Test cases — Python definitions used by both runners
├── matlab/
│   ├── bench_pepper.m        Powder CW EPR benchmarks
│   ├── bench_garlic.m        Solution EPR benchmarks
│   ├── bench_chili.m         Slow-motion benchmarks
│   ├── bench_saffron.m       Pulse EPR benchmarks
│   ├── bench_esfit.m         Fitting benchmarks
│   └── run_all.m             Runs all MATLAB benchmarks; writes timings.mat
├── python/
│   ├── bench.py              Matched simulator benchmarks
│   └── differentiable_fit_case.py
│                              Four-parameter CPU/GPU fit and gradient check
├── analysis/
│   ├── compare_results.py    Loads MATLAB + Python outputs, computes parity metrics
│   ├── make_figures.py       Figures and LaTeX tables from comparison.json
│   ├── make_manuscript_figures.py
│   │                          Figures from the embedded 2026-04-20 run
│   ├── make_verified_figures.py
│   │                          Figures, tables and device audit for a verified campaign
│   └── workstation_report.py Markdown summary of a campaign directory
└── results/                  Raw outputs and dated run logs
```

## Reproduction protocol

1. **Set up Python environment:**
   ```bash
   pip install -e ".[plot]"
   ```

2. **Set up MATLAB environment:**
   - MATLAB R2023a or later
   - An EasySpin checkout on the MATLAB path. Record its exact repository
     commit; do not infer a numbered release from a development checkout.

3. **Run MATLAB benchmarks:**
   ```bash
   matlab -batch "addpath('easyspin'); cd benchmarks/matlab; run_all"
   ```
   Output: `benchmarks/results/matlab_timings.mat`, `benchmarks/results/matlab_outputs.mat`

4. **Run Python CPU benchmarks:**
   ```bash
   python benchmarks/python/bench.py --device cpu \
     --output benchmarks/results/python_cpu
   ```

5. **Run Python GPU benchmarks** (requires CUDA-enabled PyTorch):
   ```bash
   python benchmarks/python/bench.py --device cuda \
     --output benchmarks/results/python_gpu
   ```

6. **Compare results:**
   ```bash
   python benchmarks/analysis/compare_results.py \
     --results-dir benchmarks/results \
     --cpu-prefix python_cpu \
     --gpu-prefix python_gpu \
     --output-stem parity
   ```

7. **Run the isolated differentiable case on each same-host device:**
   ```bash
   python benchmarks/python/differentiable_fit_case.py \
     --device cpu --output benchmarks/results/diff_fit_cpu
   python benchmarks/python/differentiable_fit_case.py \
     --device cuda:0 --output benchmarks/results/diff_fit_gpu
   ```

## What gets measured

For each test case:
| Metric | Description |
|--------|-------------|
| `wall_time_s` | Best of 5 runs, with 1 warmup |
| `cosine_similarity` | torchspin output vs MATLAB output |
| `scaled_nrmse` | Residual after fitting exactly one global amplitude |
| `convergence` | Did the simulation finish without error? |

## Test case selection

The benchmark suite covers cases representative of typical research workflows:

| Module | Test cases |
|--------|------------|
| pepper | nitroxide X-band, Cu(II) S=1/2 rhombic, organic radical gStrain, S=1 triplet ZFS |
| garlic | nitroxide fast-motion, 14N triplet, multi-nucleus pattern |
| chili | nitroxide slow-motion (tcorr=1ns and tcorr=10ns), aniso-g rotation |
| saffron | 2pESEEM (1H), 3pESEEM, HYSCORE (1H+14N) |
| esfit | nitroxide g-fit (3 params), multi-component fit (6 params) |
| autograd | g/A gradient computation (no MATLAB equivalent) |

## Audit-driven caveats (read before publishing)

1. **pepper absolute intensity**: torchspin uses 4 empirical correction factors
   (8.48 / 6.29 / 4.27 / 3.806). Shape cosine >0.999 is reliable; absolute
   intensity is qualified to ~15%.
2. **cardamom**: requires running `tests/generate_cardamom_matlab_refs.m` first.
3. **chili**: cosine 0.85–0.997 depending on LLMK truncation regime.
4. **saffron**: cosine 1.0000 across 2pESEEM, 3pESEEM, HYSCORE
   (verified 2026-04-19 after pf-double-application fix).
5. **GPU vs CPU**: the full mixed benchmark is not a GPU speedup figure.
   Small and sequential routines can be slower on CUDA. Use isolated,
   same-host, workflow-level measurements such as the differentiable fit.
6. **Array shapes**: the comparison rejects unequal shapes. Do not truncate or
   interpolate outputs silently.

## Reported numbers and their provenance

Every reported number, with the conditions it was measured under, is in
`results/BENCHMARK_REPORT.md` and — for the matched-host campaign — in
`results/workstation_20260904_verified/BENCHMARK_VERIFICATION.md`.  Read those
before reusing any number.  `make_manuscript_figures.py` regenerates the
2026-04-20 figure set from data embedded in the script;
`make_verified_figures.py` regenerates the matched-host figures from the
campaign JSON files.

## Verified matched-host campaign (2026-09-04)

`benchmarks/cluster/run_workstation_verified.sh` reruns the nine matched EasySpin/TorchSpin
workloads with the protocol described in
`benchmarks/results/workstation_20260904_verified/BENCHMARK_VERIFICATION.md`: every process
pinned to the same 32-core set, separate processes for 1/8/32 CPU threads with torch
and the native BLAS/OpenMP pools set together (recorded with threadpoolctl), CUDA runs
capped at 32 host threads, one warm-up plus five replicates everywhere (every replicate
saved, medians and IQR reported), and a per-workload device audit: CPU-only workloads
(`chili`, `cardamom`) are never timed as CUDA, and CUDA runs are verified with a
`torch.profiler` trace (kernels observed, eigendecompositions routed back to the CPU
pool) and a CPU/CUDA consistency check.  `benchmarks/analysis/make_verified_figures.py`
regenerates manuscript Figure 2 / Figure S5, `device_audit.json` and the CSV tables
from the campaign JSON files.
