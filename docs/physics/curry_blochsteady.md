# curry and blochsteady: magnetometry and the Bloch equations

**`curry`** computes magnetization and magnetic susceptibility of powders: χ, χT, μ_eff and M(B, T).
It is pure torch and differentiable.

**`blochsteady`** solves the periodic steady state of the Bloch equations for one S = 1/2 spin under
CW irradiation and sinusoidal field modulation.

**How well they are validated.** Four curry cases agree with EasySpin to 0.5–1 % of the maximum. The
Curie law and the textbook χT of S = 1/2 hold. Gradients match finite differences. blochsteady has
one EasySpin case, at cosine > 0.999.

Test files are in `torchspin/tests/`.

## Model

**curry.**

$$
\langle\mu_z\rangle=\frac{\sum_k e^{-\beta E_k}\langle k|\hat\mu_z|k\rangle}{\sum_k e^{-\beta E_k}},
\qquad
\chi\approx\frac{\langle\mu_z\rangle(B+\delta B)-\langle\mu_z\rangle(B)}{\delta B}
$$

χ is a forward difference with $\delta B$ = 1 mT, as in EasySpin, averaged over a powder grid.

**blochsteady.** A frequency-domain pentadiagonal solve (Tseitlin, Eaton and Eaton 2013), followed by
an inverse FFT.

## Validation

**Against stored EasySpin output:** `test_curry_bloch_matlab_validation.py`.

- `test_curry_GdCu_mueff` and three more curry cases, at 0.5–1 % of the maximum.
- `test_blochsteady_simple`, at cosine > 0.999.

**Analytical checks:**

- `test_curry.py::test_chi_mol_curie_law_S_half`: Curie law, within 5 %.
- `test_curry.py::test_chi_T_matches_textbook_cgs_value`: χT = 0.375 cm³ K/mol.

**Gradients:** `test_curry_autograd.py::test_curry_gradients_vs_finite_differences`.

**Consistency:** `test_blochsteady.py::test_fft_and_td_agree`.

## Limitations

- χ is a finite difference in field. Shrink `delta_B` if χ curves on a 1 mT scale.
- `test_curry.py` defines Brillouin helpers but never asserts against them. Anisotropy and
  zero-field-splitting tests are qualitative.
- `test_blochsteady.py::test_larger_kmax_changes_spectrum` asserts only finiteness, so convergence in
  `kmax` is untested.
