# esfit: fitting

`esfit` fits any torchspin simulator, or any user model, to data. It offers these optimizers:

- local: `levmar` (a port of EasySpin's `esfit_levmar.m`), `simplex`, `lbfgsb`, `powell` and `trf`;
- global: `global` (swarm plus simplex polish), `swarm`, `genetic`, `montecarlo` and `grid`.

It has an automatic target choice (`target='auto'`), progress reporting, early stopping, and worker
pools.

**How well it is validated.** On 6 EasySpin fits started from offset values, torchspin reaches
EasySpin's residual or better. The other evidence is self-consistency: agreement with SciPy's
`least_squares`, and recovery of synthetic parameters.

Test files are in `torchspin/tests/`.

## Validation

**Against EasySpin:**

- `test_esfit_matlab_validation.py::test_esfit_vs_matlab`: torchspin's RMS must be at most 1.05×
  EasySpin's, and g, lwpp and A are compared.
- `test_esfit.py::TestMATLABCrossValidation::test_fit_g_and_hyperfine_from_matlab_spectrum`.
- `test_esfit.py::test_simplex_initial_edge_matches_easyspin`.

**Consistency:**

- `test_esfit_trf.py::test_trf_matches_direct_scipy_with_scale_baseline_and_bounds`: rtol 1e-7.
- `test_esfit_parallel_fd.py::test_parallel_probes_match_sequential_threads`.
- Synthetic recovery throughout `test_esfit.py`.

## Limitations

- When torchspin's RMS is below 0.95× EasySpin's, `test_esfit_vs_matlab` returns early and does not
  compare parameters.
- The uncertainty outputs (`pstd`, `cov`, `ci95`) are checked for presence and shape only, not against
  a known covariance.
- Fitting pitfalls (autoscale sign, satellite minima, threads) are in
  [pitfalls.md](../pitfalls.md#fitting).
