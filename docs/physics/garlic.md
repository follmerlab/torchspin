# garlic: isotropic and fast-motion solution EPR

`garlic` simulates CW EPR in solution:

- isotropic spectra with exact line positions (`tcorr` = 0);
- fast-motion spectra, where each hyperfine line gets its own Lorentzian width from the rotational
  correlation time.

It supports sets of equivalent nuclei, frequency sweeps, field modulation, Boltzmann polarization
and full g/A matrices. It is differentiable in all modes.

**How well it is validated.** Line positions agree with EasySpin to 1e-6 mT, and spectra to cosine
> 0.9999 with amplitude within 1e-3. Gradients match central differences to 1e-6. The fast-motion
width formula itself has no analytic check.

Test files are in `torchspin/tests/`.

## Model

For each nuclear group, line positions come from the exact Breit–Rabi levels of
$\hat H=g\mu_B B\hat S_z+a\,\hat{\mathbf S}\cdot\hat{\mathbf I}-g_n\mu_N B\hat I_z$ with S = 1/2:

$$
E_{\pm}(m)=-\frac{a}{4}-g_n\mu_N B\,m\pm\frac{a(2I+1)}{4}\sqrt{1+\frac{4m\,x}{2I+1}+x^2},
\qquad
x=\frac{(g\mu_B+g_n\mu_N)B}{a\,(I+\tfrac12)}
$$

In the fast-motion regime, each line is a Lorentzian with the Kivelson–Freed width
$\Gamma(m_I)=A+B\,m_I+C\,m_I^2$. The coefficients A, B and C combine the spectral densities $j(0)$
and $j(\omega_0)$ of the g and A anisotropy at $\tau_c$ (`fastmotion.py`).

## Method

- The resonance condition $E_+(m+\tfrac12)-E_-(m-\tfrac12)=h\nu$ is solved by Weil's fixed-point
  iteration (1971), the default `Options.Method='exact'`.
- `'perturb1'`–`'perturb5'` use the perturbation series instead.
- Lines are accumulated as sticks and convolved. For gradient tensors a two-bin linear split is used.

## Validation

**Against stored EasySpin output:**

- `test_garlic_exact_matlab_validation.py`: 18 cases. Resonance positions agree to 1e-6 mT, cosine
  > 0.9999, and amplitude to 1e-3. `test_perturb_methods` covers the perturbation orders, and
  `test_perturb_scale_matches_easyspin` the scale.
- `test_garlic_features_matlab_validation.py`: 11 cases covering isotopologues, multiple components,
  automatic range and equivalent nuclei, at cosine > 0.999 and amplitude 2 %.
- `test_garlic_fastmotion_matlab.py`: fast motion, cosine ≥ 0.999.
- `test_garlic_matlab_validation.py`: older cases at cosine 0.96–0.99, with no reason stated for the
  looser thresholds.

**Gradients:** `test_garlic_autograd.py::test_fast_motion_gradients`, relative 1e-6.

## Limitations

**Approximations.**

- The fast-motion widths are second-order (Redfield) theory, valid for $\tau_c\,\Delta\omega\ll1$.
  For slower motion, use `chili`.
- No analytic check of the A, B and C coefficients exists. `test_garlic_fastmotion.py` checks trends
  only.

**Unsupported**
([§ Garlic](../../KNOWN_LIMITATIONS.md#garlic--what-is-and-is-not-ported)):

- automatic sweep range;
- natural-abundance isotope mixtures and multi-component input through `compisoloop`;
- `Opt.separate`;
- equivalent nuclei in the fast-motion regime, as in EasySpin.
