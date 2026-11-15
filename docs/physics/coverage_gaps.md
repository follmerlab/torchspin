# Coverage gaps (developer notes)

This page is for maintainers. It lists what the test suite does not cover, as of 0.9.31 (`main`
80c782d). Items closed by 0.9.38 are marked **Closed** and kept for the record. The user-facing evidence is on the [summary page](../physics_validation.md) and the
per-simulator pages.

The gaps were identified from the evidence on those pages before any test was proposed. Each entry says what
is untested, why it matters, and what test would close it. The tests are proposals only; none were
written.

1. **chili: no check of the Lanczos result against a direct solve.** **Closed in 0.9.38** (B116,
   `test_chili_lanczos.py`).
   - Untested: whether the continued fraction converged. On the plain general basis it stops early.
     The τc = 100 ns nitroxide stops at step 40 on a near-breakdown (|β| = 2.3e-3, median 0.25). The
     result has cosine 0.9375 against its own direct solution ([chili_empirical_rules.md](../chili_empirical_rules.md), Exp. 1/4).
   - Why it matters: this silently affects `MpSymm=False`. Systems outside the automatic rule's scope
     may also be affected.
   - Proposed test: for each stored chili case and a set of slow-motion bases, compare Lanczos and
     `Solver='\\'` at 1e-6.
2. **chili: `MpSymm=True` with director tilt.** **Closed in 0.9.38** (B115: raises;
   `test_chili_mpsymm.py`).
   - Untested, and silently wrong: an explicit `MpSymm=True` with an orienting potential on a powder
     grid. The M–pS–pI pruning is exact only without director tilt. Measured cosine 0.726 against the
     plain basis at `GridSize=5` (Exp. 5).
   - EasySpin's general method prunes in the same way, without a tilt check. Its `fast` basis builder
     does check (`~DirTilt`).
   - The automatic rule is not affected, because it excludes potentials.
   - Proposed test: `MpSymm=True` with `GridSize > 1` either raises or is ignored at tilted
     orientations, and equals the plain basis.
3. **chili: automatic rule with `highField=True`.** **Closed in 0.9.38** (B114;
   `test_chili_mpsymm.py`, with an EasySpin reference).
   - Untested: no stored case. The rule halves the amplitude (measured ratio 0.5000 against the plain
     direct solution), although nothing is pruned.
   - Proposed test: auto vs `MpSymm=False, Solver='\\'` amplitude ratio 1 to 1e-6, plus one EasySpin
     reference with `highField=true` and default `LiouvMethod`.
4. **chili: no converged-basis tests.**
   - Untested: LLMK convergence. Every R case runs at EasySpin's LLMK, so a truncation error shared
     with EasySpin passes.
   - Proposed test: spectra at LLMK and at (L+4, L+4, M+2, K+2) for the slow-motion cases, with the
     bound measured.
5. **lft parity tests are skipped in CI.** **Closed in 0.9.38**: frozen telser-tools references
   (`torchspin/tests/_telser_refs.py`); magnetometry parity stays live-only. The DLSD5 trigonal and
   DIPOLE Fortran outputs below are still unread.
   - Untested in CI: telser-tools is not installed there, so the main reference evidence for ddn,
     ddnloop, ddnpow, ddpowh/ddpowj, dpn, dsoxf and tanabe_sugano never runs.
   - Proposed test: freeze telser-tools' exact-mode outputs for the golden inputs as fixtures in
     `torchspin/tests/data/telser/`, and compare against them without the package.
   - Also: use the stored DLSD5 trigonal and DIPOLE Fortran outputs, which are present but unread.
6. **Shared-model defects need objective checks per option.**
   - Covariance and converged references exist for B74, B82, B84 (0.9.33), B85 (0.9.35) and B90.
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
    - CUDA tests and the two expensive cupc cases never run in CI. (The full-suite job does set
      `TORCHSPIN_STRICT_REFS`, so a missing reference file fails it.)
    - Proposed: a periodic CUDA job.
14. **After B85 (landed in 0.9.35).**
    - About 168 EasySpin comparisons become strict xfails. That removes positive R evidence for the
      projection-based paths (pepper, salt).
    - Proposed: before release, check that every xfailed comparison is paired with a passing check,
      either against EasySpin mapped to centered bins or against a converged reference. The B85
      branch adds `test_projection_accuracy.py` for part of this.
