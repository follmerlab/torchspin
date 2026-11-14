# pepper: rigid-limit CW EPR

`pepper` simulates continuous-wave EPR of powders, frozen solutions and single crystals in the rigid
limit, for field or frequency sweeps. `pepper_autograd` and `differentiable_spectrum` compute the
same spectrum on the autograd graph, so you can take gradients with respect to g, A, D, line widths,
strains, frames and temperature.

**How well it is validated.**

- About 250 stored EasySpin cases agree at cosine ≥ 0.999 and amplitude ±2 %.
- The absolute intensity matches a closed-form powder integral to 1e-4.
- Where torchspin deliberately differs from EasySpin, it is checked against converged grids instead.
- The differentiable path reproduces `pepper` to 1e-9, and its gradients match central differences.

Test files are in `torchspin/tests/`. The evidence categories are explained on the
[summary page](../physics_validation.md#how-torchspin-validates).

## Model

The static spin Hamiltonian is

$$
\hat H(\mathbf B) = \mu_B \mathbf B\cdot\mathbf g\cdot\hat{\mathbf S}
 + \hat{\mathbf S}\cdot\mathbf D\cdot\hat{\mathbf S}
 + \sum_k \hat{\mathbf S}\cdot\mathbf A_k\cdot\hat{\mathbf I}_k
 - \mu_N g_{n,k}\,\mathbf B\cdot\hat{\mathbf I}_k
 + \sum_k \hat{\mathbf I}_k\cdot\mathbf Q_k\cdot\hat{\mathbf I}_k + \dots
$$

It can also include multi-electron `ee`, higher-order Zeeman terms (`Sys.Ham*`) and `H0Extra`. The
field-swept powder spectrum is

$$
S(B) = \int_{S^2}\frac{d\Omega}{4\pi}\,\sum_{(u,v)} P_{uv}\,
 |\langle u|\hat\mu_{\perp}|v\rangle|^2\,
 \Big|\frac{\partial (E_v-E_u)}{\partial B}\Big|^{-1}_{B=B_{uv}(\Omega)}
 \, f\big(B-B_{uv}(\Omega);\,\Gamma_{uv}(\Omega)\big)
$$

- $|\partial\Delta E/\partial B|^{-1}$ is the Aasa–Vänngård factor.
- $P_{uv}$ is the population difference.
- $f$ is the line shape.
- $\Gamma_{uv}$ is the line width, including strains:
  $\Gamma^2=\Gamma_{\rm lw}^2+\sum_i (\partial B_{uv}/\partial p_i)^2\sigma_{p_i}^2$.

## Method

**Resonance fields** (`Options.Method`):

- `'matrix'` (default): exact diagonalization, with coarse field bracketing and Newton/bisection
  refinement (`resfields_batch.py`).
- `'perturb'`: second-order perturbation theory, Iwasaki 1974 (`resfields_perturb.py`).
- `'hybrid'`: an exact core plus first-order perturbational ligand nuclei (`resfields_hybrid.py`).
  This is the method for Cu(II) with ligand superhyperfine structure.

**Powder average:**

1. `hamsymm` chooses the grid symmetry.
2. G3 spline interpolation refines the grid.
3. SOPHE triangle/zone projection builds the spectrum without strain. Per-facet Gaussian summation is
   used for anisotropic widths.
4. A final convolution and a field derivative, or pseudo-modulation, complete the spectrum.

Which method to use, and what each costs, is in
[§ Pepper — choosing a method](../../KNOWN_LIMITATIONS.md#pepper--choosing-a-method-and-what-each-costs-2026-10-07).

**Differentiable path.**

- `differentiable_spectrum` builds a `SpinSystem` and calls `pepper_autograd`.
- `pepper_autograd` runs the matrix path. Every stage after the resonance search is written in torch.
- Resonance-field gradients come from the converged Newton step, by the implicit-function theorem:
  $\partial B_{uv}/\partial p=-(\partial\Delta E/\partial p)/(\partial\Delta E/\partial B)$.
- Isotropic (O3) systems use exact line templates.

## Validation

**Against stored EasySpin output:**

- `test_pepper_matlab_validation_ext2.py::test_pepper_ext2_vs_matlab`: 101 cases ported from about
  30 EasySpin pepper tests, at cosine ≥ 0.999 and amplitude ±2 %.
- `test_pepper_crystal_matlab_validation.py`: 21 crystal cases.
- `test_pepper_cupc_matlab_validation.py`: 34 Cu(II) cases with matrix, perturb and hybrid. Two of
  them are opt-in through `TORCHSPIN_RUN_EXPENSIVE=1`.
- Feature suites: `test_pepper_features_matlab_validation.py` (10 cases),
  `test_pepper_features2_matlab_validation.py` (23 cases) and
  `test_pepper_round3_matlab_validation.py` (26 cases).
- Strain:
  - `test_pepper_dstrain_matlab_validation.py`.
  - `test_strain_widths_matlab_validation.py`: per-transition widths, relative 3e-3.
  - `test_strain_multielectron_matlab_validation.py`.
  - `test_astrain_mi_model.py::test_widths_vs_easyspin` and `::test_spectra_vs_easyspin`.
- Convolution and interpolation:
  - `test_pepper_narrowlw_matlab_validation.py`: cosine 1 − 1e-7, amplitude 1e-5.
  - `test_pepper_dinfh_interp_matlab_validation.py` and
    `test_closedphi_interp_matlab_validation.py`: residual < 5e-6.
- Options:
  - `test_zero_field.py::test_pepper_zero_field_vs_easyspin`.
  - `test_quadratic_zeeman.py::test_pepper_quadratic_zeeman_vs_easyspin`.
  - `test_pepper_transitions_matlab_validation.py`.
- Differentiable path: `test_pepper_autograd_o3.py::test_o3_vs_easyspin`, cosine > 0.99999.

**Analytical checks:**

- `test_pepper_matlab_validation_ext.py::test_isotropic_powder_integral_formula`:
  $\int S\,dB = 8\pi^2\,\mathrm{rate}\cdot dB/dE$ to 1e-4, for the matrix and perturbation methods.
- `test_projecttriangles_edge_bins.py::test_fuzz_vs_exact_reference`: exact rational per-bin
  integrals to < 1e-12.
- `test_quadratic_zeeman.py::test_closed_form_S1_isotropic_BS_squared`.
- `test_zero_hyperfine_limit.py::test_zero_principal_value_is_continuous_limit`.
- `test_hstrain_lorentzian.py::test_orientation_list_single_line`.
- `test_freq_astrain_frame.py::test_unit_widths_equal_eigenvalue_derivatives_generic_frame`: widths
  equal eigenvalue central differences to 1e-6 MHz.

**Converged grids.** torchspin's own output at `[181 1]` without interpolation is the reference
(`ref_missing_knots.npz`):

- `test_pepper_nanlinear_matlab_validation.py::test_nanlinear_vs_converged`.
- `test_missing_knots.py::test_vs_converged_reference`.
- `test_grid_symmetry_strain.py::test_b82_case_matches_converged_reference`.
- `test_c1_lower_rows.py::test_cu_spectrum_vs_no_interpolation`.
- `test_pepper_dstrain_matlab_validation.py::test_multidstrain_converged_reference`.

**Gradients:**

- `test_pepper_autograd.py::test_gradients_vs_finite_differences_of_pepper_cu`: Jacobian columns
  against central differences of `pepper`, relative norm 1e-5 to 1e-6.
- `test_frame_gradients.py`: frame angles, relative 1e-5.
- `test_grid_symmetry_grad.py`, `test_modamp_grad.py` and `test_resfields_single_autograd.py`.
- `test_projecttriangles_edge_bins.py::test_gradcheck_nondegenerate`.

**Consistency:**

- `test_pepper_autograd.py::test_forward_equals_pepper`: < 1e-9.
- `test_pepper_autograd_o3.py::test_o3_autograd_equals_pepper`.
- `test_pepper_matlab_validation_ext.py::test_rhombic_invariance_across_grid_symmetries`.
- `test_pepper_hybrid.py::test_approaches_exact_as_coupling_vanishes`.
- `test_pepper_transitions.py::test_subset_sums_and_central_transition`.
- `test_pepper_thread_safety.py`.
- `test_gpu_consistency.py`. It is skipped without CUDA, so it does not run in CI.

## Limitations

**Approximations.**

- Rigid limit, with thermal or user-set populations.
- Transitions are pre-selected by an intensity threshold, which drops weak transitions in both codes.
- Strain widths are linear in each strain parameter.
- AStrain uses EasySpin's effective-mI labels, fixed per level index
  ([§ AStrain](../../KNOWN_LIMITATIONS.md#astrain--easyspins-effective-mi-model-followed-deliberately-0925)).
- The line width is linearized in field. Telser's field-point line shape differs from it by
  1.6–5.6 % for strongly curved levels ([lft_gap_analysis.md](../lft_gap_analysis.md)).
- Grid discretization: the default `GridSize=[19,4]` is not converged for narrow lines on a large
  hyperfine coupling. Check with `torchspin.grid_convergence`.
- Each bin's integral is reported at its lower edge, half a bin low, as in EasySpin. The pending B85
  change centers the bins.
- In the differentiable path, the projection is piecewise linear. Gradients are exact almost
  everywhere but have kinks where a vertex crosses a bin edge.
- With differentiated frame angles the grid becomes Ci
  ([§ Automatic grid symmetry — differentiated parameters](../../KNOWN_LIMITATIONS.md#automatic-grid-symmetry--differentiated-parameters-0926)).
- The deprecated `differentiable_spectrum(method='broadband'|'analytical')` models omit interpolation
  and projection, and carry orientation ripple. `test_autograd.py` checks them only against finite
  differences (relative < 1 %) and against pepper (cosine 0.95–0.99).

**Unsupported.**

- `Sys.n > 1` on the matrix path.
- In `pepper_autograd`, these raise: `mwPhase`, ordering, photoselection, crystals, frequency sweeps
  and `initState`.
- Field-cubic `Sys.Ham` terms; see
  [§ Higher-order Zeeman terms](../../KNOWN_LIMITATIONS.md#higher-order-zeeman-terms-sysham-deliberate-deviation-from-easyspin-067-098-0913).

**Deliberate deviations from EasySpin.** Each deviation keeps the EasySpin comparison as a strict
xfail, with an objective check in its place:

- [C1 lower hemisphere (B74)](../../KNOWN_LIMITATIONS.md#c1-grid-interpolation-of-the-lower-hemisphere-deliberate-deviation-from-easyspin-fixed-0921)
- [strains as grid-symmetry inputs (B82)](../../KNOWN_LIMITATIONS.md#automatic-grid-symmetry--strains-are-symmetry-inputs-deliberate-deviation-from-easyspin-0924)
- [range-cut transitions re-searched (B88/B75)](../../KNOWN_LIMITATIONS.md#field-sweeps-whose-range-cuts-transitions-fixed-0928-was--powder-grid-interpolation-when-resonances-are-missing-094)
- [frequency-sweep AStrain along AFrame (B90)](../../KNOWN_LIMITATIONS.md#frequency-sweeps--astrain-along-the-aframe-axes-deliberate-deviation-from-easyspin-0931)
- [zero hyperfine principal values (B65/B79)](../../KNOWN_LIMITATIONS.md#perturbation-theory--zero-hyperfine-principal-values-deliberate-deviation-from-easyspin-0929)

**Open defect.** `test_astrain_mi_model.py::test_spectra_vs_easyspin_tilted_aframe` is a strict
xfail. For a ⁶³Cu system with a tilted AFrame, pepper reaches cosine 0.873 against EasySpin and
`pepper_autograd` reaches 0.962, so the two torchspin routes also disagree with each other.
