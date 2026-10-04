# Differentiable torchspin — status and roadmap (2026-09-08)

Principle: one forward path per simulator; differentiability is added by porting the
stages that cut the autograd graph to torch, with forward equality to the existing
implementation (≈1e-10) and gradients validated against central finite differences of the
regular simulator. Grad tensors entering an unsupported stage raise instead of detaching.

## Status

| simulator | differentiable | notes / tests |
|---|---|---|
| `pepper` (`pepper_autograd`, `differentiable_spectrum`) | matrix + perturbation routes, strains (HStrain/gStrain/AStrain; DStrain magnitudes constant), `ModAmp`, Temperature | forward = `pepper` to 1e-10; `test_pepper_autograd.py`. Not yet: `mwPhase`, ordering, photoselection, crystals, frequency sweeps |
| `salt` | fixed-field ENDOR (perturbative and matrix paths) | `test_salt_autograd.py`; field-swept mode raises for grad tensors |
| `garlic` | all modes | grad tensors switch nearest-bin sticks to `AccumMethod='linear'` (sub-bin quantisation difference); `test_garlic_autograd.py` |
| `saffron` | ESEEM/HYSCORE (ideal pulses) | grad tensors → tensor outputs and `TimeDomain=True`; Mims-ENDOR raises; `test_saffron_autograd.py` |
| `curry` | yes (pure torch) | `test_curry_autograd.py` |
| `cardamom` | no (NumPy propagation); `CardamomPar.seed` for reproducible ensembles | `test_cardamom_seed.py` |
| `chili` | no (scipy.sparse Liouvillian, Lanczos) | design below |

Shared pieces: `torchspin/pepper_autograd.py` (torch projection `projecttriangles_t`/
`projectzones_t`, interpolation matrices, `convspec_t`, `gaussian_bins_t`),
`dataproc.fieldmod_t`, `_linalg.eigh` degeneracy-safe backward, `SpinSystem` keeps
requires-grad tensors. ML plumbing: `torchspin/ml.py` (`ParamSpec`, `simulate_batch`,
`SpectrumDataset`, losses; `test_ml.py`).

Lessons for gradient tests: compare finite differences on the *same* accumulation path
(garlic linear/explicit, saffron TimeDomain); at exactly symmetric parameter points fix
`GridSymmetry` (automatic grid selection changes under the step); shrink the step to
separate kinks of piecewise-linear projections from errors.

## Open items (in suggested order)

1. **chili** (1–2 weeks). `chili_sle.py`: keep index/structure computations in NumPy,
   compute the T0/T1/T2 coefficient arrays in torch (`magint`, `rbos`), assemble the
   Liouvillian as `torch.sparse_coo` (dense for n ≲ 3000), solve per field point with
   batched dense `torch.linalg.solve` for the gradient path (adjoint for free) and keep
   the Lanczos path for the no-grad forward; validate (i) vs (ii) to the Lanczos
   threshold, then gradients vs FD; replace `gaussian_filter1d` with `convspec_t`.
2. **cardamom** torch propagation (`propagate_fast` via `torch.linalg.matrix_exp`,
   torch quaternion trajectories with fixed noise → reparameterised gradients).
3. **mwPhase**: Dawson function in torch (dispersion line shapes) for pepper/garlic.
4. **pepper**: ordering/photoselection weights, crystals, frequency sweeps
   (`resfreqs_batch` has the same structure as `resfields_batch`).
5. ~~CUDA~~ verified on the reference workstation 2026-09-08: all `test_*_autograd.py` and `test_ml.py` pass on the RTX 4090; salt and saffron CPU-vs-CUDA forward and gradients agree to 1e-13 (compare gradients with a non-conserved loss — the spectrum sum is conserved for salt).
6. Notebooks 02/03/05/08 still describe the old broadband model in prose (API unchanged).
7. `differentiable_spectrum(method='broadband'|'analytical')` is deprecated; remove in 0.4.

## Benchmarks

Verified matched-host campaign (2026-09-04): `benchmarks/results/workstation_20260904_verified/`
(`BENCHMARK_VERIFICATION.md`, `figures/device_audit.json`); regenerate figures with
`benchmarks/analysis/make_verified_figures.py`; rerun with
`benchmarks/cluster/run_workstation_verified.sh` only on an idle host.
