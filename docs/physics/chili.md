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
- The symmetry-pruned basis is checked against the plain basis, and the Lanczos against the direct
  solver; four MATLAB checks confirm both against EasySpin's converged output.
- chili is not differentiable.
- No test checks convergence in the basis size; see Limitations.

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
Liouvillian. A step that passes the convergence test is accepted only if the spectrum then stays put
over the next interval; a Lanczos that does not converge warns and falls back to the direct solver.
The alternatives are a direct sparse solve (`Solver='\\'`) and eigen-decomposition (`'E'`).

**Field sweep.** `approxlin` by default, as in EasySpin.

**Powder grid.** Used only with an orienting potential.

**Basis symmetry.** Without director tilt, the Liouvillian conserves pS + pI − M. With
`MpSymm=None` (the default), chili then keeps only the +1 block, which is about 6× smaller, and adds
the −1 block exactly as its complex-conjugate mirror. The result equals the plain-basis spectrum for
every `MpSymm` setting, and for `highField=True` too. Under director tilt the plain basis is used, and
an explicit `MpSymm=True` raises. See [chili_empirical_rules.md](../chili_empirical_rules.md).

## Validation

**Against stored EasySpin output:**

- `test_chili_ext_matlab_validation.py::test_chili_vs_matlab`: 60 cases, cosine > 0.999, amplitude
  ±2 %, with no loosened case. `simple_tc1e-07` is a strict xfail at 0.99798 / 1.066: its reference,
  EasySpin's default output, stopped its continued fraction early. Against EasySpin's converged
  output torchspin agrees at cosine 0.9999966
  (`test_chili_lanczos.py::test_simple_tc1e7_matches_converged_easyspin_fast`).
- `test_chili_default_basis.py::test_default_basis_matches_easyspin_default_across_tcorr`: cosine
  > 0.9999 at four correlation times, for the default and for the plain basis.
- `test_chili_lanczos.py`: the default and the plain-basis Lanczos against EasySpin's general method
  solved directly and its fast output at a tight threshold (MATLAB checks a and c).
- `test_chili_mpsymm.py::test_highfield_matches_easyspin`: `highField=True` amplitude (MATLAB check
  b).
- `test_chili_matlab_validation.py`: four older cases at cosine 0.85–0.90. Their stated reasons
  (adaptive basis, legacy solver) are stale.

**Consistency:**

- `test_chili_liouv_vectorised.py`: vectorized against loop Liouvillian, 1e-12.
- `test_chili_mpsymm.py`: pruned basis with the mirror term against the plain basis solved directly
  (5e-16 for a nitroxide, 2e-8 for S = 1 with D = 500 MHz); `MpSymm=True` raises under director tilt.
- `test_chili_lanczos.py`: Lanczos against the direct solver; full-length tridiagonalization;
  fallback to the direct solver with a warning.

## Limitations

**Approximations.**

- Brownian rotational diffusion.
- Basis truncation (`LLMK`).
- Approximate field sweep.
- Post-convolution nuclei.
- Lanczos convergence is judged by the change of the continued fraction (`Threshold`), confirmed
  over one more interval.

**Untested.**

- There is no converged-basis test.
- Most unit tests in `test_chili.py` exercise legacy helpers that `chili()` no longer calls.

**Fixed in 0.9.38 (B114–B116).** Details in
[chili_empirical_rules.md](../chili_empirical_rules.md):

- the plain-basis Lanczos could stop early (silently);
- the old automatic rule halved the amplitude with `highField=True`;
- an explicit `MpSymm=True` was wrong under director tilt (it now raises).

**Unsupported**
([§ Chili](../../KNOWN_LIMITATIONS.md#chili--slow-motion-sle-rewritten-2026-09-01)):

- EasySpin's compiled `fast` builder;
- `Sys.Exchange`;
- `Exp.Ordering`;
- `Opt.useLMKbasis`;
- the automatic sweep range;
- gradients.
