# salt: powder ENDOR

`salt` simulates fixed-field, rf-swept powder ENDOR. It includes optional orientation selection by
the EPR excitation (`ExciteWidth`), hyperfine enhancement, and transition selection. torchspin also
offers a field-swept mode, which EasySpin does not have. The fixed-field path is differentiable.

**How well it is validated.**

- An 8-case suite mirroring EasySpin's salt tests agrees at cosine > 0.999. Orientation-selected
  cases reach > 0.99, and S = 3/2 > 0.98.
- The first-order path matches EasySpin `perturb1` to cosine > 0.999999.
- Gradients match central differences.
- The field-swept mode has consistency checks only.

Test files are in `torchspin/tests/`.

## Model

At a fixed field $B_0$, the ENDOR spectrum is

$$
S(\nu)=\int\frac{d\Omega}{4\pi}\,w_{\rm EPR}(\Omega)\sum_{(i,j)}
|\langle i|\hat\mu_{\perp}^{\rm rf}|j\rangle|^2\,f(\nu-\nu_{ij}(\Omega))
$$

$w_{\rm EPR}$ is the EPR excitation weight. The nuclear transitions $\nu_{ij}$ within each electron
manifold come from `endorfrq`, optionally with hyperfine enhancement.

## Method

1. The EPR resonance search uses `resfields_batch`.
2. The ENDOR lines come from first-order perturbation theory when there is one electron, no Q and no
   `Ham` (EasySpin's `perturb1`). Otherwise they come from diagonalization.
3. The powder average uses G3 interpolation and triangle projection, as in pepper.

## Validation

**Against stored EasySpin output:**

- `test_salt_matlab_validation.py`: the 8-case suite. The `[20,5]` grid case reaches > 0.9999.
- `test_salt_first_order_matlab_validation.py::test_first_order_path_matches_easyspin_perturb1`:
  cosine > 0.999999, sum ratio within 1e-6.
- `test_salt_narrowlw_matlab_validation.py`: 1e-9 of the peak.
- `test_endorfrq_perturb_matlab_validation.py::test_elementwise_vs_easyspin`: rtol 1e-12.
- `test_endor_enhancement.py::test_endorfrq_vs_easyspin`.
- `test_endor_tps_selection.py`.
- `test_range_padding_matlab_validation.py`.

**Analytical checks:**

- `test_endor_enhancement.py::test_enhancement_factor_isotropic_1H`.
- `test_endorfrq_perturb.py::test_order_and_intensity_closed_form`.

**Gradients:** `test_salt_autograd.py`, for the fixed-field mode, and the gradient tests in
`test_salt_narrowlw_matlab_validation.py`.

**Consistency:** `test_salt_fieldswept_scale.py` and `test_salt_ham.py`. These are the only evidence
for the field-swept mode.

## Limitations

**Approximations.**

- The first-order path has a 0.3–1 % area error against EasySpin's matrix default, for |A| ≤ 16 MHz.
- The absolute scale is EasySpin's output × Δx/(8π²). This is a documented convention.
- Projected spectra use centered bins (0.9.35, B85), not EasySpin's lower-edge bins; see the
  [pepper page](pepper.md).

**Weak check.** `test_endorfrq_matlab_validation.py` compares the field-swept mode against EasySpin's
fixed-field references, at cosine > 0.92.

**Unsupported** ([§ ENDOR](../../KNOWN_LIMITATIONS.md#endor-salt)):

- `Sys.n > 1`;
- `Sys.L`;
- a Lorentzian ENDOR width;
- `Exp.Harmonic`;
- gradients in the field-swept mode.
