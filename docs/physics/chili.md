# chili: slow-motion EPR

`chili` simulates CW EPR in the slow-motion regime, where rotational diffusion only partly averages
the magnetic anisotropy. It solves the stochastic Liouville equation (SLE). It is a
function-by-function port of EasySpin's general Liouvillian method. It supports:

- any number of electron and nuclear spins;
- orienting potentials with a powder (MOMD) grid;
- field and frequency sweeps;
- post-convolution nuclei;
- multi-component input.

**How well it is validated.**

- 60 cases mirroring EasySpin's `chili_*` tests agree at cosine > 0.999 and amplitude ±2 %.
- Every intermediate was checked against EasySpin's diagnostics dump: the basis is identical, and the
  Hamiltonian and diffusion operator agree to machine precision.
- chili is not differentiable.
- No test checks convergence in the basis size or of the Lanczos solver; see Limitations.

Test files are in `torchspin/tests/`.

## Model

The SLE for the orientation-dependent spin density $\rho(\Omega,t)$ is

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
\hat{\mathcal L}=i\hat H^\times+\hat\Gamma,
\qquad
|v\rangle\propto\sqrt{P_{\rm eq}(\Omega)}\otimes\hat S_x
$$

## Method

**Basis.** EasySpin's LjKKM Wigner functions times the spin transition basis
$|p_S q_S p_I q_I\rangle$. It is truncated by `LLMK`, which you choose as in EasySpin; there is no
adaptive basis. It is pruned by `jKmin`, `evenK`, `highField`, `pImax` and `MpSymm`.

**Solver.** The default is a pseudo-norm Lanczos continued fraction on the complex-symmetric
Liouvillian. The alternatives are a direct sparse solve (`Solver='\\'`) and eigen-decomposition
(`'E'`).

**Field sweep.** `approxlin` by default, as in EasySpin.

**Powder grid.** Used only with an orienting potential.

**Default basis symmetry.** With `MpSymm=None`, a system in the scope of EasySpin's `fast` builder
(one S = 1/2, ≤ 2 nuclei, no potential, Mmax > 0 and Kmax > 0) uses the M–pS–pI symmetry basis at
half amplitude. This reproduces EasySpin's default output. It is also exactly equal to the converged
plain-basis solution, at about 6× smaller cost. See
[chili_empirical_rules.md](../chili_empirical_rules.md).

## Validation

**Against stored EasySpin output:**

- `test_chili_ext_matlab_validation.py::test_chili_vs_matlab`: 60 cases, cosine > 0.999, amplitude
  ±2 %. The one loosened case, `simple_tc1e-07` (0.997 / 8 %), measures 0.99798 / 1.066. The residual
  lies between EasySpin's `fast` output and the converged general model; see
  [chili_empirical_rules.md](../chili_empirical_rules.md), Exp. 7.
- `test_chili_default_basis.py::test_default_basis_matches_easyspin_default_across_tcorr`: cosine
  > 0.9999 at four correlation times.
- `test_chili_matlab_validation.py`: four older cases at cosine 0.85–0.90. Their stated reasons
  (adaptive basis, legacy solver) are stale.

**Consistency:** `test_chili_liouv_vectorised.py`, vectorized against loop Liouvillian, 1e-12.

## Limitations

**Approximations.**

- Brownian rotational diffusion.
- Basis truncation (`LLMK`).
- Approximate field sweep.
- Post-convolution nuclei.
- Lanczos convergence is judged by the change of the continued fraction (`Threshold`).

**Untested.**

- There is no converged-basis test.
- There is no Lanczos-vs-direct-solver test.
- Most unit tests in `test_chili.py` exercise legacy helpers that `chili()` no longer calls.

**Being fixed (B114–B116, branch `fix/chili-mpsymm`).** Three defects found in this investigation;
details in [chili_empirical_rules.md](../chili_empirical_rules.md):

- the plain-basis Lanczos can stop early;
- the automatic rule halves the amplitude with `highField=True`;
- an explicit `MpSymm=True` is wrong under director tilt.

**Unsupported**
([§ Chili](../../KNOWN_LIMITATIONS.md#chili--slow-motion-sle-rewritten-2026-09-01)):

- EasySpin's compiled `fast` builder;
- `Sys.Exchange`;
- `Exp.Ordering`;
- `Opt.useLMKbasis`;
- the automatic sweep range;
- gradients.
