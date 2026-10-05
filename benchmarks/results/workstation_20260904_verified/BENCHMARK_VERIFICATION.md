# Verified matched-host benchmark — reference workstation, 2026-09-04

Audit, correction and rerun of the nine matched EasySpin/TorchSpin performance
workloads used for manuscript Figure 2 / Table 2 and Figure S5.  The scientific
parameters and grids are unchanged from `benchmarks/workloads/workloads.py` and
`benchmarks/workloads/matlab_workloads.m` (2026-09-02 campaign,
`benchmarks/results/workstation_20260902_after2/`, which is left untouched).  torchspin's
simulation code is the released v0.2.2 (commit 420cbc9); only the benchmark harness
changed (list at the end).

## 1. What the audit found in the 2026-09-02 campaign

| Finding | Consequence in `workstation_20260902_after2/gpu.json` | Correction |
|---|---|---|
| `chili` (scipy.sparse SLE solver) and `cardamom` (NumPy trajectory propagation) have no device argument; their `device` factory parameter was ignored | three "CUDA" timings were CPU runs | `WORKLOAD_DEVICES` marks them CPU-only; the runner records `effective_mode='unsupported'` and does not time them on CUDA |
| `saffron_hyscore_512` built `SaffronOptions(GridSize=91)` without `device` | the "CUDA" HYSCORE timing was a CPU run | `SaffronOptions(GridSize=91, device=device)` |
| `torchspin/_linalg.eigh` routes CUDA batches of matrices larger than 32×32 (cuSOLVER `syevjBatched` limit) to the CPU thread pool; the Cu/2N system is 72×72 and the Mn(II) system 36×36 | both matrix-method "CUDA" timings were CPU eigendecompositions plus GPU bookkeeping | instrumented run counts every `torch.linalg.eigh`/`eigvalsh` call by device and dimension; such runs are labeled `hybrid` |
| pepper's grid interpolation, triangle projection and strain accumulation are host-side NumPy/SciPy regardless of `device` | CUDA kernels are a small fraction of the wall time even for the small-matrix workloads | profiler trace records the CUDA-kernel fraction of wall time; `hybrid` when < 50 % |
| only `torch.set_num_threads` was set; OpenBLAS/OpenMP pools of NumPy/SciPy/torch stayed at their defaults (64) | thread-count series did not control the native pools | OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS exported before import and recorded with threadpoolctl |
| single timed call (`--repeat 1` CPU, best of 2 CUDA, one MATLAB call including first-call JIT) | no dispersion; EasySpin times included JIT/warm-up | one untimed warm-up, five timed replicates, median and IQR, every replicate saved |
| `cardamom` has no public RNG seed (`DiffusionPar.seed` exists but `CardamomPar` does not expose it) | stochastic replicates | documented: independent ensembles of fixed size 200 trajectories × 1000 steps |

## 2. Protocol

* **Host**: reference workstation — AMD Ryzen Threadripper PRO 5995WX (64 cores / 128 threads, 1 socket,
  1 NUMA node), 503 GB RAM, 2 × NVIDIA GeForce RTX 4090 (24 GB), driver 595.71.05,
  CUDA 13.2 runtime reported by the driver, Ubuntu 22.04.5, kernel 5.15.0-177.
* **Software**: Python 3.12.13, PyTorch 2.5.1+cu121 (CUDA 12.1, cuDNN 9.1), NumPy 2.4.3,
  SciPy 1.17.1, threadpoolctl 3.6.0; MATLAB R2026a (26.1.0.3203278); EasySpin
  development checkout `~/code/EPR/EasySpin` at commit
  98abb0dc394742a5488650256a5d55a0d375cf1b (2026-06-15, clean tree); torchspin
  commit 420cbc9 (v0.2.2, branch main) with the harness modifications listed below
  uncommitted in the working tree (recorded in `env.txt`).
* **CPU affinity**: every process (pytest, torchspin CPU, torchspin CUDA, MATLAB) ran under
  `taskset -c 0-31` — 32 distinct physical cores (SMT siblings 64–95 unused).  MATLAB
  reported `maxNumCompThreads = 32`, `feature('numcores') = 32`, affinity 0-31.
* **torchspin CPU**: separate processes for 1, 8 and 32 threads.  In each,
  `torch.set_num_threads(N)` and `OMP_NUM_THREADS = MKL_NUM_THREADS = OPENBLAS_NUM_THREADS =
  NUMEXPR_NUM_THREADS = N` before NumPy/torch import; threadpoolctl confirmed OpenBLAS
  (NumPy and SciPy) and libgomp (torch) pools at N in every run (`meta.native_threadpools`).
  `torch.get_num_interop_threads()` stayed at its default 64 (not used by these workloads).
* **torchspin CUDA**: one process, `CUDA_VISIBLE_DEVICES=0`, host side fixed at 32 threads
  (torch and native pools), so a hybrid calculation cannot use more CPU than the
  largest CPU comparison.  `torch.cuda.synchronize()` immediately before and after every
  timed call.  Per workload, before the timed replicates: one profiled call
  (`torch.profiler`, CPU+CUDA activities) with `torch.linalg.eigh/eigvalsh` wrapped to
  count calls by device and matrix dimension, and one CPU call for the consistency
  check (`max |Δ| / max |ref|` and NRMSE over all output arrays).
* **EasySpin**: same 32-core set, MATLAB's own threading, `BENCH_WARMUP=1`, `BENCH_NREP=5`.
* **Replicates**: one untimed warm-up then five timed replicates for every supported
  workload/mode; medians with interquartile ranges are reported. Quartiles are the
  linearly interpolated 25th and 75th percentiles, calculated identically for the
  EasySpin and TorchSpin series. Every replicate is in `raw_replicates.csv` and the
  JSON files.
* **Job isolation**: stages ran sequentially.  The first CPU/MATLAB pass (06:09–06:21)
  overlapped one to two single-core Python processes of another user (≈110 % CPU each,
  plus their job on GPU 1); those files are kept in `pass1_with_foreign_load/`.  The CUDA
  stage (07:11–07:13) and the second CPU/MATLAB pass (07:14–07:26), which are the
  reported results, ran with no foreign process above 5 % CPU and both GPUs at 0 %
  utilization before and after each stage.  (The raw `system_state.log` recorded by the
  harness is not distributed: it lists other users' processes.  Rerunning
  `benchmarks/cluster/run_workstation_verified.sh` regenerates it locally.)  Pass 1 and pass 2 agree
  within a few per cent for every workload.  Three idle Jupyter kernels of the
  benchmark user (5–7 % CPU each) were present throughout.

## 3. Results (median [IQR] of 5, seconds)

| Workload | EasySpin (s) | TorchSpin 1 t | 8 t | 32 t | best CPU | CUDA mode | TorchSpin CUDA (s) | ES / best CPU | ES / CUDA |
|---|---:|---:|---:|---:|:--:|:--:|---:|---:|---:|
| `pepper_cu_2n_matrix` | 0.292 [0.229–0.351] | 5.26 [5.25–5.29] | 1.14 [1.14–1.15] | 0.847 [0.828–0.874] | 32 t | hybrid | 0.674 [0.666–0.678] | 0.344 [0.262–0.424] | 0.433 [0.337–0.527] |
| `pepper_mn_S52_matrix` | 3.03 [3.03–3.14] | 4.98 [4.96–5] | 1.41 [1.4–1.41] | 1.07 [1.07–1.08] | 32 t | hybrid | 0.954 [0.954–0.962] | 2.83 [2.79–2.95] | 3.18 [3.14–3.29] |
| `pepper_perturb_dense_grid` | 0.0818 [0.072–0.0825] | 0.0616 [0.0613–0.0619] | 0.0676 [0.0661–0.0679] | 0.0755 [0.0755–0.0757] | 1 t | hybrid | 0.162 [0.162–0.162] | 1.33 [1.16–1.35] | 0.506 [0.445–0.511] |
| `pepper_strain_summation` | 0.814 [0.808–0.816] | 3.41 [3.41–3.41] | 0.994 [0.986–1] | 1.17 [1.14–1.17] | 8 t | hybrid | 1.23 [1.22–1.24] | 0.82 [0.807–0.828] | 0.661 [0.65–0.669] |
| `pepper_fit_loop_20` | 2.5 [2.5–2.5] | 1.65 [1.64–1.65] | 1.29 [1.29–1.31] | 5.28 [5.18–5.36] | 8 t | hybrid | 1.77 [1.77–1.77] | 1.93 [1.91–1.93] | 1.41 [1.41–1.41] |
| `chili_nitroxide_2nuc` | 1.15 [1.14–1.16] | 0.157 [0.157–0.157] | 0.158 [0.158–0.159] | 0.157 [0.157–0.158] | 1 t | unsupported | n/a | 7.32 [7.23–7.37] | n/a |
| `chili_powder_potential` | 4.53 [4.53–4.53] | 3.08 [3.07–3.08] | 3.08 [3.08–3.09] | 3.09 [3.09–3.1] | 1 t | unsupported | n/a | 1.47 [1.47–1.47] | n/a |
| `saffron_hyscore_512` | 0.66 [0.536–0.667] | 4 [4–4.02] | 1.06 [1.05–1.06] | 0.752 [0.751–0.753] | 32 t | hybrid | 0.228 [0.228–0.228] | 0.877 [0.713–0.888] | 2.89 [2.35–2.92] |
| `cardamom_diffusion_200x1000` | 20.4 [20.4–20.4] | 19.3 [19.3–19.3] | 9.97 [9.91–10.1] | 10.4 [10.2–10.5] | 8 t | unsupported | n/a | 2.04 [2.02–2.06] | n/a |

"best CPU" is the smallest tested thread count whose IQR overlaps the IQR of the
configuration with the lowest median wall time.  This prevents negligible timing noise
from being presented as evidence that a larger thread pool is beneficial.  EasySpin /
TorchSpin ratios use medians; the bracketed bounds propagate the IQRs (EasySpin q1 /
TorchSpin q3 and vice versa).  Machine-readable: `summary.csv`, `raw_replicates.csv`.

**Headline changes relative to the 2026-09-02 tables**

* With the first-call JIT excluded by the warm-up, EasySpin's Cu/2N matrix powder takes
  0.29 s instead of the 2.05 s single cold call recorded before; strain summation
  (0.81 s) and HYSCORE (0.66 s) also become faster than TorchSpin's best CPU
  configuration.  Under this protocol TorchSpin's best CPU configuration is faster
  than EasySpin in six of nine workloads (ratios 1.33–7.3) and slower in three (Cu
  matrix 0.34, strain 0.82, HYSCORE 0.88).  The manuscript statement "as fast as or
  faster than EasySpin in all nine workloads" does not hold under the replicated
  warm protocol.
* With the native BLAS/OpenMP pools capped together with torch, the 32-thread run of the
  20-call fit loop is 5.3 s (8 threads: 1.29 s): the small per-call problems
  oversubscribe the 32-core set.  The best-CPU column is unaffected (8 threads).
* `chili_nitroxide_2nuc` is 0.157 s at any thread count (0.42–0.63 s in the earlier
  campaign, where the SciPy OpenBLAS pool was uncapped at 64 threads).

## 4. Device audit (`device_audit.json`)

| Workload | implementation | effective mode on `--device cuda` | CUDA kernel events / kernel time as % of wall | device→host copies | eigendecompositions in the profiled call | max Δ / max ref | NRMSE |
|---|---|---|---|---|---|---|---|
| `pepper_cu_2n_matrix` | cpu, cuda | hybrid | 806 / 3.4 % | 114 | 144 on CPU (dim 72), 0 on CUDA | 5.6e-14 | 6.5e-15 |
| `pepper_mn_S52_matrix` | cpu, cuda | hybrid | 766 / 3.8 % | 614 | 138 on CPU (dim 36), 0 on CUDA | 6.8e-14 | 1.2e-14 |
| `pepper_perturb_dense_grid` | cpu, cuda | hybrid | 269 / 0.2 % | 8386 | none (perturbative) | 5.4e-14 | 3.4e-14 |
| `pepper_strain_summation` | cpu, cuda | hybrid | 2521 / 3.7 % | 7610 | 4 on CUDA (dim 6) | 1.0e-12 | 1.7e-13 |
| `pepper_fit_loop_20` | cpu, cuda | hybrid | 20560 / 9.1 % | 30660 | 60 on CUDA (dim 6) | 3.3e-12 | 2.2e-12 |
| `chili_nitroxide_2nuc` | cpu only | unsupported | – | – | – | – | – |
| `chili_powder_potential` | cpu only | unsupported | – | – | – | – | – |
| `saffron_hyscore_512` | cpu, cuda | hybrid | 14083 / 16.6 % | 460 | 182 on CUDA (dim 6) | 2.4e-15 | 2.2e-15 |
| `cardamom_diffusion_200x1000` | cpu only | unsupported | – | – | – | – | – |
* **CUDA kernels were observed for all six CUDA-capable workloads**, but in none of them
  do kernels account for more than 17 % of the wall time, so all six are reported as
  *hybrid CPU/CUDA* and drawn hatched in Figure 2.  No workload qualifies as pure CUDA
  execution.
* **CPU fallbacks**: in `pepper_cu_2n_matrix` (72×72) and `pepper_mn_S52_matrix` (36×36)
  every batched eigendecomposition was executed by the CPU thread pool
  (`torchspin/_linalg.eigh`: matrices above the 32×32 cuSOLVER batched limit are moved
  to the host); zero `eigh` calls ran on CUDA.  Their CUDA-mode timings are CPU
  diagonalization (32 threads) plus GPU-resident Hamiltonian construction and
  resonance search, which is why they are within 10–20 % of the 32-thread CPU times.
  The 6×6 nitroxide systems (strain, fit loop) diagonalize on CUDA but interpolate,
  project and accumulate on the host (thousands of device→host copies per call);
  the perturbative workload runs 0.2 % of its wall time in kernels.
* **Unsupported on CUDA**: `chili_nitroxide_2nuc`, `chili_powder_potential`,
  `cardamom_diffusion_200x1000` — CPU-only implementations; omitted from the CUDA
  series (N/A in Figure 2).
* **Consistency**: CPU and CUDA outputs agree to ≤ 3.3 × 10⁻¹² of the maximum signal
  (NRMSE ≤ 2.2 × 10⁻¹²) for every CUDA-capable workload; all GPU timings were accepted.
  `torchspin/tests/test_gpu_consistency.py`: 8 passed, 1 skipped (`gpu_consistency.txt`).
* **Stochastic workload**: `cardamom` draws fresh trajectories on every call (no seed
  through `CardamomPar`); the five replicates are independent ensembles of identical
  size (200 × 1000).  Its spread (IQR 0.1 s on 20 s) is dominated by run-to-run
  timing, not ensemble differences.

## 5. Figures

* `figures/Figure_performance_workstation_verified.{png,pdf,svg}` — Figure 2.  (A) wall time,
  median with IQR error bars: EasySpin, TorchSpin selected CPU thread count (annotated),
  TorchSpin CUDA mode (hatched = hybrid CPU/CUDA; N/A for the CPU-only workloads).
  (B) EasySpin / TorchSpin ratio on a log₂ axis with IQR-propagated bounds.
* `figures/Figure_S_thread_scaling_workstation_verified.{png,pdf,svg}` — Figure S5, CPU thread
  scaling: median [IQR] and speedup relative to one thread; selected configuration boxed.
* Regenerate with `python benchmarks/analysis/make_verified_figures.py`.

## 6. Remaining caveats

* One host, one EasySpin checkout, one MATLAB release; absolute times are host-specific.
* The 32-core affinity set is shared by all comparisons, but MATLAB's internal threading
  is not otherwise controlled (it used all 32 cores it was given); torchspin's thread
  series is the only controlled scaling experiment.
* CUDA measurements used GPU 0 only; the two GPUs are identical.
* Torch's inter-op pool (64) was not capped; these workloads do not use it.
* The CUDA-mode timings are hybrid executions and should not be read as GPU
  performance of the simulators; the GPU claim of the manuscript remains restricted to
  the differentiable workflow (Figure 4), which was not part of this campaign.
* MATLAB replicates of the Cu matrix and HYSCORE workloads show wider IQRs
  (0.23–0.35 s, 0.54–0.67 s) than the others; the medians are reported.
* EasySpin warned that the spectrum extended beyond the fixed 260–360 mT sweep during
  the Mn(II) workload.  The timing was retained because both implementations evaluated
  the same fixed field window and the test measures computational performance rather
  than spectral completeness; this limitation should be disclosed with the workload.

## 7. Files

`env.txt`, `campaign.log`, `gpu_consistency.txt`,
`cpu_threads{1,8,32}.json`, `gpu.json` (+ `gpu.json.<workload>.cuda_profile.txt`
profiler tables), `matlab_workloads.json`, `pass1_with_foreign_load/` (first CPU/MATLAB
pass), `figures/` (figures, `summary.csv`, `raw_replicates.csv`, `summary_tables.md`,
`device_audit.json`).

## 8. Harness changes (no simulator code changed)

1. `benchmarks/workloads/workloads.py` — `SaffronOptions(..., device=device)` for the
   HYSCORE workload; `WORKLOAD_DEVICES`/`STOCHASTIC`/`supports_device()` tables marking
   `chili` and `cardamom` CPU-only and `cardamom` stochastic; comments on the ignored
   `device` argument.
2. `benchmarks/workloads/run_workloads.py` — rewritten runner: `--warmup`, replicated
   timing with all replicates and median/quartile statistics saved (`seconds` keeps the
   median for older analysis scripts); thread environment variables set before import
   and recorded with threadpoolctl, CPU affinity and git state in `meta`; unsupported
   devices skipped and recorded; CUDA verification (profiler trace, kernel fraction,
   memcpy counts, eigh device/dimension counter, CPU/CUDA consistency, GPU utilization
   snapshots) and `effective_mode` classification per workload; `--no-verify`;
   `--list` shows supported devices.  The `--procs` and `--profile` modes are kept.
3. `benchmarks/workloads/matlab_workloads.m` — `BENCH_WARMUP`/`BENCH_NREP`/`BENCH_OUT`
   environment control, every replicate saved, `meta` with MATLAB version,
   `maxNumCompThreads`, `numcores`, EasySpin path and CPU affinity.
4. `benchmarks/cluster/run_workstation_verified.sh` — new staged campaign script (env, cpu,
   matlab, gpu), `taskset` pinning, thread environment per stage, system-state
   snapshots before/after each stage.
5. `benchmarks/analysis/make_verified_figures.py` — new figure/table/audit generator;
   near-tied CPU configurations are represented by the smallest thread count whose IQR
   overlaps that of the fastest-median configuration.
6. `benchmarks/README.md` — section describing the verified campaign.
