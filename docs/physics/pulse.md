# Pulse EPR: saffron, saffron_thyme, spidyan

Three routes cover pulse EPR:

- **`saffron`** simulates predefined pulse experiments with ideal pulses: 2-, 3- and 4-pulse ESEEM,
  HYSCORE and Mims ENDOR. It also runs custom sequences, on powders and crystals.
- **`saffron_thyme`** runs the same experiments with real (finite, shaped, chirped) pulses.
- **`spidyan`** propagates arbitrary pulse sequences with relaxation, phase cycling and indirect
  dimensions.

`saffron` is differentiable for ESEEM and HYSCORE.

**How well it is validated.** All three agree with EasySpin to relative L2 errors of 1e-6 to 1e-12,
on both the raw and the mean-centered signal. A cosine is not used, because it is dominated by the
DC offset and cannot see the modulation. The main caveat is the gradient path: it uses
`TimeDomain=True`, but the EasySpin references validate the default binned path.

Test files are in `torchspin/tests/`.

## Model

**saffron, ideal pulses.** The echo of an S = 1/2 system with electron manifolds α and β is a sum
over nuclear-frequency pathways. For two-pulse ESEEM:

$$
V(\tau)=\mathrm{Re}\sum_{ijkl}M_{ij}M_{jk}^{\dagger}M_{kl}M_{li}^{\dagger}\,
 e^{-2\pi i(\nu^\alpha_{ij}-\nu^\beta_{kl})\tau}
$$

$M=\langle\alpha|\beta\rangle$ is the manifold overlap matrix. The signal is averaged over
orientations.

**saffron_thyme, real pulses.** The density matrix is propagated as
$\rho(t+\Delta t)=U\rho U^\dagger$ with $U=e^{-i\hat H(t)\Delta t}$, and $\langle\hat S_+\rangle$ is
detected.

**spidyan.** The Liouville–von Neumann equation with relaxation,

$$
\frac{d\rho}{dt}=-i[\hat H(t),\rho]-\hat\Gamma(\rho-\rho_{\rm eq})
$$

$\hat\Gamma$ holds phenomenological T1 and T2 rates. The equation is integrated with
piecewise-constant propagators in a simulation frame (`SimFreq`).

## Method

- **saffron:**
  - The default is binned frequency-domain accumulation followed by an inverse FFT, as in EasySpin.
  - `TimeDomain=True` gives exact time-domain sums.
  - `ProductRule` treats one nucleus at a time.
- **saffron_thyme and spidyan:** stepwise propagation with time step `IntTimeStep`. Pulses come from
  `pulse()` and `rfmixer`.

## Validation

**Against stored EasySpin output**, by relative L2 error:

- `test_saffron_matlab_validation.py`: about 20 cases, < 1e-6.
- `test_saffron_phase_matlab_validation.py`: raw < 1e-11, mean-centered < 1e-9.
- `test_saffron_thyme_matlab_validation.py`: 18 rectangular-pulse cases, < 1e-10 / 1e-9.
- `test_saffron_thyme_shaped_matlab_validation.py`: EasySpin's own 10 shaped-pulse cases, < 1e-9.
- `test_spidyan_matlab_validation.py` (9 cases) and `test_spidyan_tp_matlab_validation.py`: < 1e-8.
  `test_spidyan_relaxation_complex_detection` checks S+ detection under relaxation.

**Analytical checks:**

- `test_saffron_thyme.py::test_on_resonance_fid_is_a_constant_phasor`.
- `test_saffron_thyme.py::test_t2_decay_rate_matches_the_relaxation_time`.
- `test_spidyan.py::test_T2_on_diagonal`.

**Consistency:**

- `test_saffron_thyme.py::test_single_crystal_equals_spidyan_times_grid_weight`.
- `test_saffron.py::test_2p_td_vs_fd`: 5e-3.
- `test_saffron.py::test_product_rule_vs_direct`: cosine > 0.9.
- `test_spidyan.py::test_propagator_unitary`.

**Gradients:** `test_saffron_autograd.py`, relative 1e-5.

## Limitations

**Approximations.**

- `saffron`: ideal pulses; binning, by default; the product rule.
- `saffron_thyme` and `spidyan`: the references pin `IntTimeStep`, so what is validated is that both
  codes do the same calculation, not that the step is converged. See
  [pitfalls.md](../pitfalls.md#integration-step).

**Gaps.**

- The gradient path (`TimeDomain=True`) is tied to EasySpin only through the 5e-3 time-vs-frequency
  check.
- There is no analytic ESEEM or Rabi check.
- Four `test_saffron_thyme_*` functions in `test_saffron_matlab_validation.py` only load their
  reference files and never call torchspin.

**Unsupported**
([§ Pulse EPR](../../KNOWN_LIMITATIONS.md#pulse-epr--saffron-spidyan-and-saffron_thyme-are-matlab-validated)):

- gradients for Mims ENDOR;
- thyme templates for 4pESEEM and Mims ENDOR (EasySpin has none either);
- a transient `DetSequence`, which is not cross-validated because EasySpin itself errors.
