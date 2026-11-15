# cardamom: trajectory-based CW EPR

`cardamom` simulates CW EPR of spin labels from orientation trajectories. The trajectories can be
stochastic rotational diffusion, jump models, or molecular dynamics. Use it when the motion is not
simple Brownian diffusion, or when you have an MD trajectory. Pass `CardamomPar.seed` for
reproducible results.

**How well it is validated.** Three EasySpin references agree at cosine > 0.97 (measured
0.9992–0.9998 for diffusion) and amplitude ±10 %. The comparison is statistical: the MATLAB and NumPy
random streams cannot be matched. Seeded runs are reproducible and independent of the thread count.

Test files are in `torchspin/tests/`.

## Model

Each trajectory $\Omega(t)$ evolves its own density matrix:

$$
\rho_{n+1}=e^{-i\hat H(\Omega_n)\Delta t}\rho_n e^{+i\hat H(\Omega_n)\Delta t}
$$

The FID is $\langle\hat S_+\rangle(t)$, averaged over the ensemble. The spectrum is the imaginary
part of its FFT, mapped to field. The ISTOs variant propagates in an irreducible spherical tensor
operator basis.

## Validation

**Against stored EasySpin output:** `test_cardamom_matlab_validation.py` has three cases: diffusion
(fast), jump (fast) and diffusion (ISTOs). The MATLAB side is seeded (`rng(42)`). The torchspin side
of this test is unseeded, so each run is a new realization.

**Consistency:** `test_cardamom_seeding.py::test_seeded_spectrum_independent_of_thread_count` and
`test_cardamom_seed.py`.

## Limitations

**Approximations**
([§ Cardamom](../../KNOWN_LIMITATIONS.md#cardamom--validated-against-easyspin-2026-09-02)):

- finite ensemble and time step;
- the ISTOs method does not apply `SzFilter`;
- the potential torque in `stochtraj_diffusion` uses Euler-angle finite differences, which are exact
  only for axial L = 2 potentials.

**Weak or missing checks:**

- `TestCardamomMATLABRef` in `test_cardamom.py` only loads Python-generated files.
- `test_cardamom.py::test_fast_motion` (motional narrowing) is xfail.
- `test_tier3e.py::test_istos_vs_fast_qualitative` compares nothing.
- The MD-driven modes have smoke tests only.
