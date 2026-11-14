# Physics and validation map

This page links each major torchspin simulator to five things: its governing equations, its
numerical method, its approximations, the options it does not support, and the evidence that it is
correct. It describes **torchspin 0.9.31** (`main` at 80c782d). The validation scope of the
manuscript (arXiv:2610.18924) is a different thing; see
[Current implementation vs the manuscript's measured release](#current-implementation-vs-the-manuscripts-measured-release).
The investigation of chili's basis-symmetry and amplitude rules is in
[chili_empirical_rules.md](chili_empirical_rules.md).

All test files are in `torchspin/tests/`. All stored EasySpin references are in `tests/data/`. Their
provenance is in `tests/data/DATA_PROVENANCE.md`.

## How to read the evidence

Every validation claim below names a test file, and where it matters, a test function. Each test is
classified by **what it compares against**, not by its file name. For example,
`test_matlab_validation.py` tests only `ham()` matrices, and every EasySpin comparison in
`test_pepper_nanlinear_matlab_validation.py` is a strict xfail.

| Tag | Category | What agreement establishes |
|---|---|---|
| **R** | Reference comparison against stored EasySpin output (`tests/data/*.mat`, made by the `generate_*.m` scripts) | torchspin computes what EasySpin computes. It is **not independent of model errors shared with EasySpin** (see below). |
| **A** | Analytical check against a closed form, an exact limit, or a hand-built calculation | The physics is right in that limit, independently of EasySpin. |
| **G** | Gradient check: autograd against central differences, or `torch.autograd.gradcheck` | The derivative is the derivative of the forward model. It says nothing about whether the forward model is right. |
| **C** | Converged-grid or converged-basis check: torchspin against torchspin on a much finer grid, without interpolation, or with a larger basis | Discretization error is bounded, independently of EasySpin's discretization. |
| **S** | Self-consistency: two torchspin routes agree (CPU/GPU, `pepper`/`pepper_autograd`, solver A/B), or an invariance holds (sample tilt equals tensor tilt, subsets add up) | The routes are consistent. A defect they share passes. |

**EasySpin is a physics reference, not a specification.** An R test can pass on a defect that both
codes share. Round 8 of the bug log found three:

- **B84.** The field-sweep gStrain width uses the element-wise square of the rotated strain matrix,
  which is not rotation-covariant. EasySpin's `resfields.m` has the same defect.
- **B85.** The triangle projection reports each bin's integral at the bin's lower edge, so every
  projected spectrum sits half a bin low. EasySpin's `projecttriangles.c` does the same.
- **B90.** The frequency-sweep AStrain width ignored `AFrame`. EasySpin's `resfreqs_matrix.m`
  rotates S with the transpose.

B74 (C1 lower-hemisphere interpolation), B75 (triangles crossing the range ends), and B82 (strains
ignored by `hamsymm`) are also shared with EasySpin.

Each of these was found and verified with A, C or G evidence: eigenvalue central differences,
converged grids, and rotation covariance. torchspin then deviates from EasySpin on purpose and
documents the deviation. The EasySpin comparisons for the affected cases are kept as **strict
xfails**, so they record the deviation and cannot silently start passing. When an R test and an
A/C test disagree, the A/C test decides.

**Round-8 status.** B94 (`lft.ddnpow` Ritz gradients, 0.9.30), B90, and B89 (frequency-sweep AStrain
frame and multi-electron strain, 0.9.31) are merged. These are pending and **not on `main`**:

- B85: centered (hat) bin projection. It shifts all projected spectra by Δ/2, and about 168 EasySpin
  comparisons become strict xfails.
- B86: salt O3 gradient.
- B93: Bessel J_n.
- B84: gStrain covariance.
- B97: DCD unit-cell block in `mdload`.
- B80: broad `except` clauses.

The sections below describe `main`. Each section notes where a pending item will change it.

## Map at a glance

| Simulator | Model | Numerical method | Main approximations | Evidence (strongest first) |
|---|---|---|---|---|
| `pepper` | Rigid-limit CW EPR, static spin Hamiltonian, orientation average | Resonance-field search (matrix / perturb / hybrid), grid symmetry, G3 interpolation, SOPHE triangle projection, convolution | Rigid limit; Gaussian strain widths; AStrain effective-mI model; grid discretization; half-bin projection (B85 pending) | R (≈300 cases), A (closed-form intensity), C (converged grids), G, S |
| `pepper_autograd` / `differentiable_spectrum` | Same as pepper | pepper's forward path on the autograd graph; implicit-function gradients through the field search | As pepper; piecewise-linear projection, so gradients have kinks at bin crossings | S (forward = pepper to 1e-11), G (central differences of `pepper`), R (O3 path) |
| `garlic` | Isotropic solution and fast-motion CW EPR | Breit–Rabi fixed-point line positions; Kivelson–Freed Lorentzian widths | Isotropic averaging; Redfield fast-motion widths | R (18 + 11 cases, positions to 1e-6 mT), G |
| `chili` | Slow-motion stochastic Liouville equation (SLE) | Wigner-function × spin-transition basis; sparse Liouvillian; Lanczos continued fraction (or direct / eigen solver) | Basis truncation (LLMK); Brownian rotational diffusion; approximate field sweep | R (60 + 4 + 1 cases); S. No C, no G |
| `salt` | Fixed-field powder ENDOR | EPR resonance search, `endorfrq`, triangle projection | First-order perturbation for simple systems; Gaussian ENDOR width only | R (8-case suite + narrow-width, padding, enhancement, selection suites), G, S |
| `saffron` | Ideal-pulse ESEEM / HYSCORE / Mims ENDOR | Eigen-decomposition per orientation; binned frequency domain or exact time domain; product rule | Ideal (δ) pulses; binning (default) | R (strict relative L2 ≤ 1e-6..1e-11), S, G |
| `saffron_thyme` | Real-pulse saffron | Piecewise-constant propagation over a powder | Fixed integration step | R (28 cases at 1e-9..1e-12), A, S |
| `spidyan` | Arbitrary pulse sequences, Liouville–von Neumann with relaxation | Stepwise propagators in a simulation frame | Phenomenological T1/T2 relaxation; time step | R (10 cases at 1e-8..1e-9), A (relaxation superoperator), S |
| `cardamom` | Trajectory-based CW EPR | Stochastic or MD orientation trajectories; density-matrix propagation; FFT of the FID | Finite ensemble; time step; no `SzFilter` in ISTOs | R (3 stochastic cases at cosine > 0.97), S (seeds) |
| `curry` / `blochsteady` | Thermodynamic magnetization and susceptibility; Bloch equations under field modulation | Boltzmann averages over a powder, χ by forward difference; frequency-domain pentadiagonal solve | χ by finite difference δB (as EasySpin) | R (4 + 1 cases), A (Curie law), G |
| `esfit` | Least-squares fitting | Nine optimizers; `levmar` ported from `esfit_levmar.m` | — | R (6 EasySpin fits), S |
| `lft.*` | d^n/p^n ligand-field theory; Telser's programs | Full determinant basis, dense eigh with degenerate-safe derivatives, igloo powder average | Igloo grid; Rayleigh–Ritz subspace (ddnpow) | A, G, live parity with telser-tools (**skipped in CI**), Fortran printout (DTD, part of DLSD5) |

## pepper (rigid-limit CW EPR)

**Governing equations.** The static spin Hamiltonian is

$$
\hat H(\mathbf B) = \mu_B \mathbf B\cdot\mathbf g\cdot\hat{\mathbf S}
 + \hat{\mathbf S}\cdot\mathbf D\cdot\hat{\mathbf S}
 + \sum_k \hat{\mathbf S}\cdot\mathbf A_k\cdot\hat{\mathbf I}_k
 - \mu_N g_{n,k}\,\mathbf B\cdot\hat{\mathbf I}_k
 + \sum_k \hat{\mathbf I}_k\cdot\mathbf Q_k\cdot\hat{\mathbf I}_k + \dots
$$

It can also include multi-electron `ee`, higher-order Zeeman `Sys.Ham*`, and `H0Extra`. The
field-swept powder spectrum is

$$
S(B) = \int_{S^2}\frac{d\Omega}{4\pi}\,\sum_{(u,v)} P_{uv}\,
 |\langle u|\hat\mu_{\perp}|v\rangle|^2\,
 \Big|\frac{\partial (E_v-E_u)}{\partial B}\Big|^{-1}_{B=B_{uv}(\Omega)}
 \, f\big(B-B_{uv}(\Omega);\,\Gamma_{uv}(\Omega)\big)
$$

The factor $|\partial\Delta E/\partial B|^{-1}$ is the Aasa–Vänngård factor. $P_{uv}$ is the
population difference, $f$ the line shape, and $\Gamma_{uv}$ the strain width:

$$
\Gamma^2=\Gamma_{\rm lw}^2+\sum_i (\partial B_{uv}/\partial p_i)^2\sigma_{p_i}^2
$$

**Method.**

- `Options.Method='matrix'` (default) finds resonance fields by coarse field bracketing and
  Newton/bisection (`resfields_batch.py`).
- `'perturb'` uses second-order Iwasaki perturbation theory (`resfields_perturb.py`).
- `'hybrid'` uses an exact core with first-order perturbational ligand nuclei
  (`resfields_hybrid.py`).
- `hamsymm` picks the grid symmetry, and G3 spline interpolation refines the grid.
- SOPHE triangle/zone projection gives spectra without strain. Per-facet Gaussian summation gives
  anisotropic widths. A final convolution and a field derivative or pseudo-modulation complete the
  spectrum.

**Approximations.**

- Rigid limit, with thermal or user populations.
- Transition pre-selection by intensity threshold. This drops weak transitions in both codes; see
  [§ Pepper — choosing a method](../KNOWN_LIMITATIONS.md#pepper--choosing-a-method-and-what-each-costs-2026-10-07).
- Strain widths are linear in each strain parameter.
- AStrain uses EasySpin's effective-mI labels, which are fixed per level index
  ([§ AStrain](../KNOWN_LIMITATIONS.md#astrain--easyspins-effective-mi-model-followed-deliberately-0925)).
- The width is linearized in field (Telser's field-point line differs by 1.6–5.6 % for strongly curved
  levels; [lft_gap_analysis.md](lft_gap_analysis.md)).
- Grid discretization: the default `GridSize=[19,4]` is not converged for narrow lines on a large
  hyperfine coupling.
- The half-bin projection offset (B85, pending).

**Unsupported.**

- `n > 1` on the matrix path.
- In `pepper_autograd`, `mwPhase`, ordering, photoselection, crystals, frequency sweeps and
  `initState` raise.
- See also [§ Higher-order Zeeman terms](../KNOWN_LIMITATIONS.md#higher-order-zeeman-terms-sysham-deliberate-deviation-from-easyspin-067-098-0913).

**Evidence.**

- **R**
  - `test_pepper_matlab_validation_ext2.py::test_pepper_ext2_vs_matlab`: 101 cases ported from about
    30 EasySpin pepper tests, cosine ≥ 0.999 and amplitude ±2 %.
  - `test_pepper_crystal_matlab_validation.py`: 21 crystal cases.
  - `test_pepper_cupc_matlab_validation.py`: 34 matrix/perturb/hybrid Cu(II) cases. Two are opt-in
    via `TORCHSPIN_RUN_EXPENSIVE=1`.
  - `test_pepper_features_matlab_validation.py` (10 cases) and
    `test_pepper_features2_matlab_validation.py` (23 cases).
  - `test_pepper_round3_matlab_validation.py`: 26 cases.
  - `test_pepper_dstrain_matlab_validation.py`, `test_strain_widths_matlab_validation.py` (per-transition
    widths, rel. 3e-3) and `test_strain_multielectron_matlab_validation.py`.
  - `test_pepper_narrowlw_matlab_validation.py`: cosine 1 − 1e-7, amplitude 1e-5.
  - `test_pepper_dinfh_interp_matlab_validation.py` and `test_closedphi_interp_matlab_validation.py`:
    residual < 5e-6.
  - `test_zero_field.py::test_pepper_zero_field_vs_easyspin`.
  - `test_quadratic_zeeman.py::test_pepper_quadratic_zeeman_vs_easyspin`.
- **A**
  - `test_pepper_matlab_validation_ext.py::test_isotropic_powder_integral_formula`:
    $\int S\,dB = 8\pi^2\,\mathrm{rate}\cdot dB/dE$ to 1e-4.
  - `test_projecttriangles_edge_bins.py::test_fuzz_vs_exact_reference`: exact rational per-bin
    integrals < 1e-12.
  - `test_quadratic_zeeman.py::test_closed_form_S1_isotropic_BS_squared`.
  - `test_zero_hyperfine_limit.py::test_zero_principal_value_is_continuous_limit`.
  - `test_hstrain_lorentzian.py::test_orientation_list_single_line`.
  - `test_freq_astrain_frame.py::test_unit_widths_equal_eigenvalue_derivatives_generic_frame`:
    widths equal eigenvalue central differences to 1e-6 MHz.
- **C**
  - `test_pepper_nanlinear_matlab_validation.py::test_nanlinear_vs_converged`.
  - `test_missing_knots.py::test_vs_converged_reference`.
  - `test_grid_symmetry_strain.py::test_b82_case_matches_converged_reference`.
  - `test_c1_lower_rows.py::test_cu_spectrum_vs_no_interpolation`.
  - `test_pepper_dstrain_matlab_validation.py::test_multidstrain_converged_reference`.
  - The converged references in `ref_missing_knots.npz` are torchspin's own output at `[181 1]`
    without interpolation.
- **S**
  - `test_pepper_matlab_validation_ext.py::test_rhombic_invariance_across_grid_symmetries`.
  - `test_pepper_hybrid.py::test_approaches_exact_as_coupling_vanishes`.
  - `test_pepper_transitions.py::test_subset_sums_and_central_transition`.
  - `test_pepper_thread_safety.py`.
  - `test_gpu_consistency.py` (skipped without CUDA, so it never runs in CI).

**Deviations from EasySpin, with reasons.** All are documented in KNOWN_LIMITATIONS, and each
EasySpin comparison is a strict xfail with an objective check in its place:

- [C1 lower hemisphere (B74)](../KNOWN_LIMITATIONS.md#c1-grid-interpolation-of-the-lower-hemisphere-deliberate-deviation-from-easyspin-fixed-0921)
- [strains as grid-symmetry inputs (B82)](../KNOWN_LIMITATIONS.md#automatic-grid-symmetry--strains-are-symmetry-inputs-deliberate-deviation-from-easyspin-0924)
- [range-cut transitions re-searched (B88/B75)](../KNOWN_LIMITATIONS.md#field-sweeps-whose-range-cuts-transitions-fixed-0928-was--powder-grid-interpolation-when-resonances-are-missing-094)
- [frequency-sweep AStrain along AFrame (B90)](../KNOWN_LIMITATIONS.md#frequency-sweeps--astrain-along-the-aframe-axes-deliberate-deviation-from-easyspin-0931)
- [zero hyperfine principal values (B65/B79)](../KNOWN_LIMITATIONS.md#perturbation-theory--zero-hyperfine-principal-values-deliberate-deviation-from-easyspin-0929)

There is one open defect, not a deviation:
`test_astrain_mi_model.py::test_spectra_vs_easyspin_tilted_aframe` is a strict xfail. On a tilted-AFrame
⁶³Cu system, pepper gives cosine 0.873 and `pepper_autograd` gives 0.962, so the two torchspin routes
also disagree with each other.

## pepper_autograd and differentiable_spectrum

**Model and method.** The model is pepper's, for the matrix path. `differentiable_spectrum` builds a
`SpinSystem` and calls `pepper_autograd`. Every stage after the resonance search is torch code.
Resonance-field gradients come from the converged Newton step, by the implicit-function theorem:

$$
\frac{\partial B_{uv}}{\partial p}=-\frac{\partial\Delta E/\partial p}{\partial\Delta E/\partial B}
$$

Isotropic (O3) systems use exact line templates.

**Approximations.** The projection is piecewise linear, so the spectrum has kinks where a vertex
crosses a bin edge, and gradients are exact almost everywhere. With differentiated frame angles the
grid becomes Ci
([§ Automatic grid symmetry — differentiated parameters](../KNOWN_LIMITATIONS.md#automatic-grid-symmetry--differentiated-parameters-0926)).
The deprecated `method='broadband'` and `'analytical'` models omit interpolation and projection, and
carry orientation ripple.

**Evidence.**

- **S**
  - `test_pepper_autograd.py::test_forward_equals_pepper`: < 1e-9.
  - `test_pepper_autograd_o3.py::test_o3_autograd_equals_pepper`.
- **G**
  - `test_pepper_autograd.py::test_gradients_vs_finite_differences_of_pepper_cu`: Jacobian columns
    against central differences of `pepper`, relative norm 1e-5 to 1e-6.
  - `test_frame_gradients.py`: frames, relative 1e-5.
  - `test_grid_symmetry_grad.py`.
  - `test_modamp_grad.py`.
  - `test_resfields_single_autograd.py`.
  - `test_projecttriangles_edge_bins.py::test_gradcheck_nondegenerate`.
- **R**: `test_pepper_autograd_o3.py::test_o3_vs_easyspin`, cosine > 0.99999.
- **Deprecated models**: `test_autograd.py` only checks them against finite differences (rel < 1 %)
  and against pepper (cosine > 0.95–0.99).

## garlic (isotropic and fast-motion solution EPR)

**Governing equations.** For each nuclear group, line positions come from the exact Breit–Rabi
levels of $\hat H=g\mu_B B\hat S_z+a\,\hat{\mathbf S}\cdot\hat{\mathbf I}-g_n\mu_N B\hat I_z$ with
S = 1/2:

$$
E_{\pm}(m)=-\frac{a}{4}-g_n\mu_N B\,m\pm\frac{a(2I+1)}{4}\sqrt{1+\frac{4m\,x}{2I+1}+x^2},
\qquad
x=\frac{(g\mu_B+g_n\mu_N)B}{a\,(I+\tfrac12)}
$$

The resonance condition $E_+(m+\tfrac12)-E_-(m-\tfrac12)=h\nu$ is solved by the fixed-point
iteration of Weil (1971), or by its perturbation series in `perturb1`–`perturb5`. In the fast-motion regime, each line is a Lorentzian
with the Kivelson–Freed width:

$$
\Gamma(m_I)=A+B\,m_I+C\,m_I^2
$$

$A$, $B$ and $C$ are spectral densities $j(0)$ and $j(\omega_0)$ of the g and A anisotropy at
$\tau_c$ (`fastmotion.py`).

**Approximations.** The fast-motion widths are second-order Redfield theory, valid when
$\tau_c\,\Delta\omega\ll1$. Line shapes are Lorentzian/Gaussian/Voigt.

**Unsupported.** Automatic sweep range, natural-abundance isotope mixtures, multi-component input via
`compisoloop`, `Opt.separate`, and equivalent nuclei in fast motion
([§ Garlic](../KNOWN_LIMITATIONS.md#garlic--what-is-and-is-not-ported)).

**Evidence.**

- **R**
  - `test_garlic_exact_matlab_validation.py`: 18 cases. Resonance positions to 1e-6 mT, cosine
    > 0.9999, amplitude 1e-3. `test_perturb_methods` covers the perturbation orders, and
    `test_perturb_scale_matches_easyspin` the scale.
  - `test_garlic_features_matlab_validation.py`: 11 cases, cosine > 0.999, amplitude 2 %.
  - `test_garlic_fastmotion_matlab.py`: cosine ≥ 0.999.
  - `test_garlic_matlab_validation.py`: older cases at a looser cosine of 0.96–0.99, with no stated
    reason.
- **G**: `test_garlic_autograd.py::test_fast_motion_gradients`, relative 1e-6.
- **Missing**: there is no analytic check of the Kivelson coefficients. `test_garlic_fastmotion.py`
  checks trends only, and its `test_curie_law_consistency` asserts a width ratio in [3, 20].

## chili (slow-motion SLE)

**Governing equations.** The SLE for the orientation-dependent spin density $\rho(\Omega,t)$ is

$$
\frac{\partial\rho}{\partial t}=-i\,[\hat H(\Omega),\rho]-\hat\Gamma_\Omega\,\rho,
\qquad
\hat\Gamma_\Omega=\hat{\mathbf L}\cdot\mathbf R\cdot\hat{\mathbf L}
\;(+\ \text{potential terms})
$$

The CW spectrum is

$$
I(\omega)=\mathrm{Re}\,\langle v|\,(\hat{\mathcal L}-i\omega)^{-1}|v\rangle,
\qquad
\hat{\mathcal L}=i\hat H^\times+\hat\Gamma
$$

The starting vector is $|v\rangle\propto\sqrt{P_{\rm eq}(\Omega)}\otimes\hat S_x$.

**Method.**

- The basis is EasySpin's general-method LjKKM Wigner functions times the spin transition basis
  $|p_S q_S p_I q_I\rangle$, truncated by `LLMK` and pruned by `jKmin`, `evenK`, `pSmin`
  (`highField`), `pImax` and `MpSymm`.
- The Liouvillian is a sparse matrix that is complex symmetric.
- The solver is a pseudo-norm Lanczos continued fraction (default), a direct sparse solve
  (`Solver='\\'`), or eigen-decomposition.
- The field sweep uses `approxlin` by default.
- A powder grid is used only with an orienting potential (MOMD).

**Approximations.**

- Brownian rotational diffusion.
- Basis truncation. The user chooses LLMK, as in EasySpin; there is no adaptive basis.
- Approximate field sweep.
- Post-convolution nuclei.
- Lanczos convergence is judged by the change of the continued fraction (`Threshold`).

**Unsupported.** EasySpin's compiled `fast` builder (`chili_lm.c`), `Sys.Exchange`, `Exp.Ordering`,
`Opt.useLMKbasis`, the automatic sweep range, and gradients (chili has no torch path)
([§ Chili](../KNOWN_LIMITATIONS.md#chili--slow-motion-sle-rewritten-2026-09-01)).

**Empirical rules.** When `MpSymm=None`, systems in EasySpin's `fast` scope use the M–pS–pI symmetry
basis at half amplitude (`chili_sle.py:1038-1046`, `:1187`). The rule turns out to be an exact
identity, not an empirical fit: it reproduces the converged plain-general solution. The ~6 % "general
vs fast" difference it was introduced to fix is a premature stop of the Lanczos continued fraction.
The basis, the scope, one suspected amplitude defect (`highField=True`), and the proposed tests are in
[chili_empirical_rules.md](chili_empirical_rules.md).

**Evidence.**

- **R**
  - `test_chili_ext_matlab_validation.py::test_chili_vs_matlab`: 60 cases mirroring EasySpin's
    `chili_*` tests, cosine > 0.999 and amplitude ±2 %. The one loosened case is `simple_tc1e-07`,
    at 0.997 / 8 %; measured now 0.99798 / 1.066.
  - `test_chili_default_basis.py::test_default_basis_matches_easyspin_default_across_tcorr`: cosine
    > 0.9999.
  - `test_chili_matlab_validation.py`: 4 older cases at cosine 0.85–0.90. The stated reasons are
    stale: "adaptive basis selection" (EasySpin used the same fixed LLMK) and a legacy solver.
- **S**: `test_chili_liouv_vectorised.py`, vectorized against loop Liouvillian, 1e-12.
- **Missing**
  - No converged-basis (C) test and no Lanczos-vs-direct-solver test.
  - Most unit tests in `test_chili.py` exercise legacy helpers (`_lanczos`, `_build_ori_basis`) that
    `chili()` no longer calls.

## salt and ENDOR

**Governing equations.** At a fixed field $B_0$, the ENDOR spectrum is

$$
S(\nu)=\int\frac{d\Omega}{4\pi}\,w_{\rm EPR}(\Omega)\sum_{(i,j)}
|\langle i|\hat\mu_{\perp}^{\rm rf}|j\rangle|^2\,f(\nu-\nu_{ij}(\Omega))
$$

The weight $w_{\rm EPR}$ is the EPR excitation weight (`ExciteWidth`). The nuclear transitions
$\nu_{ij}$ within each electron manifold come from `endorfrq`, optionally with hyperfine enhancement.

**Method.** The resonance search uses `resfields_batch`. The ENDOR lines come from first-order
perturbation theory (one electron, no Q, no `Ham`; EasySpin `perturb1`) or else from diagonalization.
The powder average uses G3 interpolation and triangle projection.

**Approximations.**

- First-order path: 0.3–1 % area error against EasySpin's matrix default for |A| ≤ 16 MHz.
- The absolute scale is EasySpin's × Δx/(8π²), a documented convention.
- The field-swept mode is a torchspin extension with no EasySpin counterpart.

**Unsupported.** `Sys.n > 1`, `Sys.L`, a Lorentzian ENDOR width, and `Exp.Harmonic`
([§ ENDOR](../KNOWN_LIMITATIONS.md#endor-salt)).

**Evidence.**

- **R**
  - `test_salt_matlab_validation.py`: 8-case suite. Cosine > 0.999, orientation-selected cases
    > 0.99, S = 3/2 > 0.98, and `[20,5]` grid > 0.9999.
  - `test_salt_first_order_matlab_validation.py::test_first_order_path_matches_easyspin_perturb1`:
    cosine > 0.999999, sum ratio 1e-6.
  - `test_salt_narrowlw_matlab_validation.py`: 1e-9 of the peak.
  - `test_endorfrq_perturb_matlab_validation.py::test_elementwise_vs_easyspin`: rtol 1e-12.
  - `test_endor_enhancement.py::test_endorfrq_vs_easyspin`.
  - `test_endor_tps_selection.py`.
  - `test_range_padding_matlab_validation.py`.
- **A**
  - `test_endor_enhancement.py::test_enhancement_factor_isotropic_1H`.
  - `test_endorfrq_perturb.py::test_order_and_intensity_closed_form`.
- **G**: `test_salt_autograd.py` (fixed field) and `test_salt_narrowlw_matlab_validation.py`.
- **S**: `test_salt_fieldswept_scale.py` and `test_salt_ham.py`. These are the only evidence for the
  field-swept mode.
- **Weak**: `test_endorfrq_matlab_validation.py` compares torchspin's field-swept mode with EasySpin's
  fixed-field references at cosine > 0.92, and line frequencies within ±1 MHz.

## saffron and saffron_thyme (pulse EPR)

**Governing equations.** With ideal pulses, the echo signal of an S = 1/2 system in electron manifolds
α and β is a sum over nuclear-frequency pathways. For two-pulse ESEEM:

$$
V(\tau)=\mathrm{Re}\sum_{ijkl}M_{ij}M_{jk}^{\dagger}M_{kl}M_{li}^{\dagger}\,
 e^{-2\pi i(\nu^\alpha_{ij}-\nu^\beta_{kl})\tau}
$$

Here $M=\langle\alpha|\beta\rangle$ is the manifold overlap matrix, averaged over orientations.
`saffron_thyme` propagates the density matrix through real pulses:

$$
\rho(t+\Delta t)=U\rho U^\dagger,\qquad U=e^{-i\hat H(t)\Delta t}
$$

It then detects $\langle\hat S_+\rangle$.

**Method.** The default is binned frequency-domain accumulation followed by an inverse FFT, as in
EasySpin. `TimeDomain=True` gives exact time-domain sums. `ProductRule` treats one nucleus at a time.

**Approximations.** Ideal pulses; binning; and the product rule, which neglects nucleus–nucleus
correlation beyond the product.

**Unsupported.** Gradients for Mims ENDOR. Thyme mode has no 4pESEEM or Mims template, as in
EasySpin. A transient `DetSequence` is not cross-validated, because EasySpin itself errors
([§ Pulse EPR](../KNOWN_LIMITATIONS.md#pulse-epr--saffron-spidyan-and-saffron_thyme-are-matlab-validated)).

**Evidence.**

- **R**, by relative L2 error, not cosine:
  - `test_saffron_matlab_validation.py`: about 20 cases, < 1e-6.
  - `test_saffron_phase_matlab_validation.py`: raw < 1e-11, mean-centered < 1e-9.
  - `test_saffron_thyme_matlab_validation.py`: < 1e-10 / 1e-9.
  - `test_saffron_thyme_shaped_matlab_validation.py`: EasySpin's own 10 shaped-pulse cases, < 1e-9.
- **A**
  - `test_saffron_thyme.py::test_on_resonance_fid_is_a_constant_phasor`.
  - `test_saffron_thyme.py::test_t2_decay_rate_matches_the_relaxation_time`.
- **S**
  - `test_saffron_thyme.py::test_single_crystal_equals_spidyan_times_grid_weight`.
  - `test_saffron.py::test_2p_td_vs_fd`: 5e-3.
  - `test_saffron.py::test_product_rule_vs_direct`: cosine > 0.9.
- **G**: `test_saffron_autograd.py`, relative 1e-5. Gradients use `TimeDomain=True`, but the MATLAB
  references validate the binned path. The gradient path is tied to EasySpin only through the 5e-3
  S check.
- **Placeholder tests**: four `test_saffron_thyme_*` functions in
  `test_saffron_matlab_validation.py` only load the reference file and never call torchspin.

## spidyan (arbitrary pulse sequences)

**Governing equations.** The Liouville–von Neumann equation with relaxation is

$$
\frac{d\rho}{dt}=-i[\hat H(t),\rho]-\hat\Gamma(\rho-\rho_{\rm eq})
$$

$\hat\Gamma$ holds phenomenological T1 and T2 rates. The equation is integrated in a simulation frame
(`SimFreq`) with piecewise-constant propagators.

**Evidence.**

- **R**: `test_spidyan_matlab_validation.py` (9 cases) and `test_spidyan_tp_matlab_validation.py`,
  relative L2 < 1e-8. The test `test_spidyan_relaxation_complex_detection` checks S+ detection under
  relaxation.
- **A**: `test_spidyan.py::test_T2_on_diagonal`.
- **S**: `test_spidyan.py::test_propagator_unitary`.
- **Missing**: there is no analytic Rabi-oscillation or free-precession check. The Larmor-frequency
  check in CHANGELOG 0.6.0 was a diagnosis, not a test.

## cardamom (trajectory-based CW EPR)

**Governing equations.** Orientation trajectories $\Omega(t)$ come from rotational Brownian
diffusion, jumps, or MD. Each trajectory evolves its own density matrix:

$$
\rho_{n+1}=e^{-i\hat H(\Omega_n)\Delta t}\rho_n e^{+i\hat H(\Omega_n)\Delta t}
$$

The FID is $\langle\hat S_+\rangle(t)$, averaged over the ensemble. The spectrum is the imaginary
part of its FFT, mapped to field. The ISTOs variant propagates in an irreducible spherical tensor
operator basis.

**Approximations.** Finite ensemble and time step. ISTOs does not apply `SzFilter`. The potential
torque in `stochtraj_diffusion` uses Euler-angle finite differences, which is exact only for axial
L=2 potentials ([§ Cardamom](../KNOWN_LIMITATIONS.md#cardamom--validated-against-easyspin-2026-09-02)).

**Evidence.**

- **R**: `test_cardamom_matlab_validation.py`, 3 cases at cosine > 0.97 and amplitude ±10 %. The
  MATLAB side is seeded (`rng(42)`). The torchspin side is unseeded in this test, so each run is a new
  realization.
- **S**: `test_cardamom_seeding.py::test_seeded_spectrum_independent_of_thread_count`.
- **Weak**
  - `TestCardamomMATLABRef` in `test_cardamom.py` only loads Python-generated files.
  - `test_fast_motion` (motional narrowing) is xfail.
  - `test_tier3e.py::test_istos_vs_fast_qualitative` compares nothing.

## curry and blochsteady

**Governing equations.** For curry:

$$
\langle\mu_z\rangle=\frac{\sum_k e^{-\beta E_k}\langle k|\hat\mu_z|k\rangle}{\sum_k e^{-\beta E_k}},
\qquad
\chi\approx\frac{\langle\mu_z\rangle(B+\delta B)-\langle\mu_z\rangle(B)}{\delta B}
$$

The forward difference uses $\delta B$ = 1 mT, as in EasySpin, and the result is averaged over a
powder.

blochsteady solves the periodic steady state of the Bloch equations under field modulation, in the
frequency domain (Tseitlin et al. 2013).

**Evidence.**

- **R**: `test_curry_bloch_matlab_validation.py`. Four curry cases at 0.5–1 % of the maximum, and one
  blochsteady case at cosine > 0.999.
- **A**: `test_curry.py::test_chi_mol_curie_law_S_half` (Curie law to 5 %) and
  `test_chi_T_matches_textbook_cgs_value`.
- **G**: `test_curry_autograd.py`.
- **S**: `test_blochsteady.py::test_fft_and_td_agree`.
- **Missing**
  - The Brillouin helpers in `test_curry.py` are never asserted against.
  - `test_blochsteady.py::test_larger_kmax_changes_spectrum` asserts only finiteness.

## esfit

**Method.** Nine optimizers (`levmar`, `simplex`, `global`, `swarm`, ...), and `target='auto'`.

**Evidence.**

- **R**: `test_esfit_matlab_validation.py::test_esfit_vs_matlab`, 6 EasySpin fits. torchspin's RMS
  must be ≤ 1.05× EasySpin's. **If torchspin's RMS is below 0.95× EasySpin's, the test returns before
  comparing parameters.**
- **S**: `test_esfit_trf.py` (against scipy `least_squares`, rtol 1e-7) and `test_esfit.py`
  (synthetic recovery, plus `test_simplex_initial_edge_matches_easyspin`).
- **Missing**: no check of the uncertainty outputs (`pstd`, `cov`, `ci95`) against a known covariance.

## lft (ligand-field tools)

**Governing equations.** The d^n (or p^n) Hamiltonian in the full $\binom{10}{n}$ determinant basis
is

$$
\hat H=\hat H_{ee}(B,C;\alpha)+\zeta\sum_i\hat{\mathbf l}_i\cdot\hat{\mathbf s}_i+\hat V_{\rm LF}
 +\mu_B\mathbf B\cdot(k\hat{\mathbf L}+g_e\hat{\mathbf S})
$$

$\hat V_{\rm LF}$ comes from AOM ($e_\sigma$, $e_\pi$) or from Ballhausen crystal-field parameters.
`ddnpow`, `ddpowh` and `ddpowj` give powder spectra on Telser's igloo grid with exact field-point
line shapes.

**Method.** Dense eigh. Degenerate levels get level-mean derivatives (`eigh_gauge`), thermal averages
use Daleckii–Krein, and $g=(gg^T)^{1/2}$ uses a Newton–Schulz square root. `ddnpow` uses a
Rayleigh–Ritz subspace, with exact gradients since B94 (0.9.30). Details: [lft.md](lft.md).

**Evidence.**

- **A**
  - `test_lft_ddn.py::test_d1_octahedral_splitting_is_10Dq`.
  - `test_lft_ddn.py::test_d2_free_ion_3P_minus_3F_is_15B`.
  - `test_lft_magnetometry.py::test_d3_van_vleck_tip_is_8NmuB2_over_10Dq`.
  - `test_lft_magnetometry.py::test_spin_only_limit_equals_curry`.
  - `test_lft_ddpowh.py::test_nqr_lines_at_closed_form_frequencies`.
- **G**
  - `test_lft_core.py::test_ground_projector_gradcheck_at_kramers_degeneracy`.
  - `test_lft_ddnpow.py::test_d7_ritz_gradients_match_full_basis`: < 1e-7.
- **Live parity with telser-tools**, at rel 1e-10 on every golden input:
  - `test_lft_ddpowh.py::test_ddpowh_golden_inputs_match_telser_tools`: 70 DDPOWH and 33 DDPOWJ inputs.
  - `test_lft_telser2torchspin.py::test_ddzhq_levels_and_probabilities`.

  These tests `importorskip` telser-tools. CI does not install it, so **they never run in CI**.
- **Fortran printout**: only `test_lft_dtd.py::test_matches_dtd_fortran_printout` and part of DLSD5
  (`test_lft_dlsd5.py::test_tetragonal_branches_reproduce_dlsd5_fortran_k`).
- **Note on the reference**: telser-tools "exact mode" is itself a corrected port of the Fortran. The
  parity therefore tests against corrected physics, not against Telser's printed output.

## Coverage gaps

The gaps were identified from the evidence above before any test was proposed. Each entry says what
is untested, why it matters, and what test would close it. The tests are proposals only; none were
written.

1. **chili: no check of the Lanczos result against a direct solve.**
   - Untested: whether the continued fraction converged. On the plain general basis it stops early.
     The τc = 100 ns nitroxide stops at step 40 on a near-breakdown (|β| = 2.3e-3, median 0.25). The
     result has cosine 0.9375 against its own direct solution (chili_empirical_rules.md, Exp. 1/4).
   - Why it matters: this silently affects `MpSymm=False`. Systems outside the automatic rule's scope
     may also be affected.
   - Proposed test: for each stored chili case and a set of slow-motion bases, compare Lanczos and
     `Solver='\\'` at 1e-6.
2. **chili: `MpSymm=True` with director tilt.**
   - Untested, and silently wrong: an explicit `MpSymm=True` with an orienting potential on a powder
     grid. The M–pS–pI pruning is exact only without director tilt. Measured cosine 0.726 against the
     plain basis at `GridSize=5` (Exp. 5).
   - EasySpin's general method prunes in the same way, without a tilt check. Its `fast` basis builder
     does check (`~DirTilt`).
   - The automatic rule is not affected, because it excludes potentials.
   - Proposed test: `MpSymm=True` with `GridSize > 1` either raises or is ignored at tilted
     orientations, and equals the plain basis.
3. **chili: automatic rule with `highField=True`.**
   - Untested: no stored case. The rule halves the amplitude (measured ratio 0.5000 against the plain
     direct solution), although nothing is pruned.
   - Proposed test: auto vs `MpSymm=False, Solver='\\'` amplitude ratio 1 to 1e-6, plus one EasySpin
     reference with `highField=true` and default `LiouvMethod`.
4. **chili: no converged-basis tests.**
   - Untested: LLMK convergence. Every R case runs at EasySpin's LLMK, so a truncation error shared
     with EasySpin passes.
   - Proposed test: spectra at LLMK and at (L+4, L+4, M+2, K+2) for the slow-motion cases, with the
     bound measured.
5. **lft parity tests are skipped in CI.**
   - Untested in CI: telser-tools is not installed there, so the main reference evidence for ddn,
     ddnloop, ddnpow, ddpowh/ddpowj, dpn, dsoxf and tanabe_sugano never runs.
   - Proposed test: freeze telser-tools' exact-mode outputs for the golden inputs as fixtures in
     `torchspin/tests/data/telser/`, and compare against them without the package.
   - Also: use the stored DLSD5 trigonal and DIPOLE Fortran outputs, which are present but unread.
6. **Shared-model defects need objective checks per option.**
   - Covariance and converged references exist for B74, B82, B84 (pending), B85 (pending) and B90.
   - There is no systematic rotation-covariance test (rotate tensors, frames and sample together;
     the spectrum must not change) for every strain and frame option.
   - B96 (AStrain mI labels not rotation-covariant, 7–20 %) is a candidate in this class.
   - Proposed test: a parametrized covariance test over {gStrain, AStrain, DStrain, HStrain} × {field,
     frequency} sweeps, with a converged-grid reference.
7. **saffron gradient path not tied to EasySpin.**
   - Gradients run on `TimeDomain=True`, but the EasySpin references validate the binned path. The
     link is one 5e-3 time-vs-frequency check (`test_saffron.py::test_2p_td_vs_fd`).
   - Proposed test: EasySpin references generated with `Opt.TimeDomain=1` for 2p/3p ESEEM and HYSCORE,
     compared at relative L2 1e-10, plus the analytic Mims modulation depth
     $k=(\omega_I B/\omega_\alpha\omega_\beta)^2$ for S = I = 1/2.
8. **garlic fast-motion widths: no analytic check.**
   - Untested: the A, B, C coefficients in `fastmotion.py`.
   - Proposed test: coefficients against the closed-form Kivelson expressions (Atherton pp. 331–332)
     for a nitroxide at fixed τc, rtol 1e-10.
9. **cardamom: no statistical acceptance bound.**
   - The R test is unseeded at cosine 0.97. Motional narrowing is xfail. ISTOs vs fast is not compared.
   - Proposed test: seeded ensembles with N replicas, and a bound from the replica spread. Also check
     the fast-motion limit against garlic's fast-motion spectrum.
10. **esfit uncertainty outputs.**
   - Untested: `pstd`, `cov` and `ci95` against a model whose covariance is known (a linear model
     with Gaussian noise).
   - Also: make `test_esfit_vs_matlab` compare parameters even when torchspin's RMS is lower.
11. **Closed forms missing for simple physics.**
    - `levels`: Breit–Rabi energies for S = I = 1/2 at 1e-9.
    - `spidyan`: Rabi nutation of S = 1/2 under a rectangular pulse, and the free-precession frequency.
    - `curry`: the Brillouin M(B, T) for S = 5/2, and χ convergence as δB → 0.
    - `blochsteady`: the slow-passage Lorentzian limit, and convergence in kmax.
12. **Loose or stale thresholds.**
    - `test_chili_matlab_validation.py`: 0.85–0.90, with stale reasons.
    - `test_garlic_matlab_validation.py`: 0.96–0.98, no reason given.
    - `test_strain_validation.py`: 10 % RMS, and its docstring mentions an 8/π factor that no longer
      exists.
    - `test_endorfrq_matlab_validation.py`: compares different modes.
    - `test_pepper_beta.py`: two conditional asserts that can pass vacuously.
    - Proposed: re-measure each case and tighten it to its measured value, or record the reason.
13. **CI scope.**
    - The full-suite job does not set `TORCHSPIN_STRICT_REFS`, although its comment says a missing
      reference is a hard failure.
    - CUDA tests and the two expensive cupc cases never run in CI.
    - Proposed: set the variable in `python-tests.yml`, and run a periodic CUDA job.
14. **After B85 (pending).**
    - About 168 EasySpin comparisons become strict xfails. That removes positive R evidence for the
      projection-based paths (pepper, salt).
    - Proposed: before release, check that every xfailed comparison is paired with a passing check,
      either against EasySpin mapped to centered bins or against a converged reference. The B85
      branch adds `test_projection_accuracy.py` for part of this.

## Current implementation vs the manuscript's measured release

**Measured release.** The submitted manuscript (arXiv:2610.18924; source snapshot
`torchspin_submitted_main.tex`, 2026-10-12) says: *"The reference-comparison and performance analyses
used TorchSpin v0.2.2."*

The performance audit
([BENCHMARK_VERIFICATION.md](../benchmarks/results/workstation_20260904_verified/BENCHMARK_VERIFICATION.md))
records torchspin commit 420cbc9 (v0.2.2, 2026-09-03). That commit is not in this repository: the
pre-0.3.0 history was squashed (CHANGELOG footnote), and the first tag is `v0.3.0`.

`BENCHMARK_REPORT.md` Table 1 and Figures 1–5 come from an earlier run (2026-04-20, v0.1.0 era,
EasySpin 6.0).

Everything else on this page describes 0.9.31. The manuscript's numbers are historical and are not
edited here.

| Manuscript claim (v0.2.2) | Current status (0.9.31) |
|---|---|
| garlic Fremy's salt NRMSE < 1e-15 | Still exact: `test_garlic_exact_matlab_validation.py`, positions 1e-6 mT. |
| chili nitroxide (τc 3 ns), NRMSE 7.23e-5. The EasySpin reference is from the *fast* builder; torchspin used "a general Liouvillian pathway". | Numbers stand. The description is incomplete: since 2026-09-02, which includes v0.2.2, torchspin's default for this system is the general method **with** the M–pS–pI symmetry basis at half amplitude (the automatic rule). The rule is exact; see chili_empirical_rules.md. |
| pepper `pepper_rhombiclw`, NRMSE 9.09e-5 | Measured with the half-bin projection convention that EasySpin shares. **Pending B85** moves every projected spectrum by Δ/2. After it, the agreement with EasySpin is a documented deviation, and correctness rests on converged references. |
| salt `salt_invariance_axial1`, cosine 1.00, global amplitude factor 3.65 | The scale difference is the documented convention (EasySpin × Δx/8π²). Since v0.2.2 salt changed its narrow-width convolution rule (0.8.11), edge padding (0.9.6), hyperfine enhancement (0.7.3) and transition selection (0.9.14). The manuscript case has not been re-measured. Pending B85/B86 change it again. |
| cardamom, NRMSE 0.0201 | Same model. The finite-ensemble residual is expected. Seeded reproducibility was added later (`CardamomPar.seed`). |
| saffron `saffron_2psimple`, NRMSE 0.0101 | Measured **before** the 0.5.0 saffron fixes: the tau phase factor in the 2p/3p/4p ESEEM and HYSCORE routes, and the S = 1/2 manifold assignment. The test had a 10× smaller quadrupole coupling and the wrong grid (20 vs 31). The current suite holds the case at relative L2 ≤ 1e-6. Whether the manuscript's Fig. S2 trace was affected has not been re-measured. |
| spidyan e–¹H, NRMSE 6.39e-5 | This is the residual that 0.6.0 traced to a **test transcription error**: g = 2.00232 instead of gfree, a 3.5e-7 offset accumulating as phase. The manuscript states g = 2.00232. The current agreement is 1e-14..1e-12; no library code changed. |
| curry Gd–Cu μ_eff, NRMSE 1.48e-7 | Unchanged model (`test_curry_bloch_matlab_validation.py`). |
| Performance (Fig. 2, Table 2, Fig. S5): nine workloads, v0.2.2 | Not re-measured. pepper's projection, interpolation and strain paths and the chili default have changed since. README's speedups (chili 7.3×, cardamom 2.0×, Cu(II) 0.34×) are these v0.2.2 numbers. |
| `BENCHMARK_REPORT.md` Table 1 (2026-04-20): pepper amplitude "~15 % (correction factors)"; chili cosine 0.92–0.997; spidyan and cardamom "self-consistent / internal" | Superseded. The empirical pepper factors were removed on 2026-09-01. chili was rewritten on 2026-09-01. spidyan and cardamom are now EasySpin-validated. The report says it is a layered record and defers to the verified campaign. |

Smaller statements in the repository docs that no longer match the tests:

- README says agreement is "cosine ≥ 0.999" for salt and curry. The orientation-selected salt cases
  pass at > 0.99 (measured 0.993–0.998), S = 3/2 at > 0.98, and curry at 0.5–1 % of the maximum.
- `docs/index.md` says "about 2800 tests". The suite has about 5800.
