# Known Limitations

torchspin was developed and audited rigorously, but a few simulators
have documented caveats users should understand before relying on them for
publication-quality work. This document is the public record of where the
package is fully MATLAB-validated, where there are quantified caveats, and
where validation is still pending.

If you find behavior that contradicts this document, please open an issue.

---

## Trust matrix at a glance

| Module | Shape parity vs MATLAB | Absolute intensity | Caveats |
|---|---|---|---|
| `pepper` (CW powder, crystals, frequency sweeps) | ≥0.999 on 102+12+11 general cases plus 22 crystal, 4 parallel-mode, 10 feature (isotopologues, ModAmp, dispersion, separate) and 26 feature-2 cases (non-equilibrium populations, ordering, photoselection, frequency sweeps), plus 34 Cu(II)+ligand cases covering `matrix`/`perturb`/`hybrid` on the same systems | matches EasySpin from first principles (isotropic closed form to 1e-4, amplitude ratio 1.00±0.02) | See § Pepper |
| `garlic` (solution) | line positions exact vs EasySpin Breit–Rabi (<1e-6 mT, 18-case suite); spectra cosine >0.9999; components/isotopologues/auto-range 11-case suite identical | matches EasySpin's TransitionRate·dBdE convention to 1e-6 | See § Garlic |
| `chili` (slow-motion) | 59/60 MATLAB cases at cosine ≥0.999 (general Liouvillian port; intermediates exact) | matches EasySpin's scaling chain | See § Chili |
| `salt` (ENDOR powder) | 8-case EasySpin-mirroring suite passes (fixed-field mode) | matches in regime | See § ENDOR |
| `saffron` (pulse EPR) | 26 MATLAB cases (2p/3p/4p ESEEM incl. 14N + quadrupole, HYSCORE, MimsENDOR, S>1/2, custom sequences, single crystals, two components): cosine ≥0.98, most ≥0.999 | not a current parity target | HYSCORE performance |
| `spidyan` (arbitrary pulse) | 8 MATLAB cases (incl. `tp` sweeps), cosine > 0.99 | n/a | See § Pulse |
| `saffron_thyme` (real-pulse) | internal-consistency tests only (incl. crystal orientations) | n/a | See § Pulse |
| `curry`, `blochsteady` | 5 MATLAB cases within 0.5–1 % | n/a | None |
| `hamsymm` | 22 MATLAB cases exact (groups and frames) | n/a | See § Pepper (tilted axial frames) |
| `cardamom` (trajectory) | 3 EasySpin 6 references (diffusion/fast, jump/fast, diffusion/ISTOs): cosine 0.9998 / ≥0.97 / 0.9992, amplitude 1.00 | matches EasySpin's scale | See § Cardamom |
| `esfit` (fitting) | 6 EasySpin esfit references (levmar/simplex from offset starts): torchspin reaches EasySpin's residual or better; `levmar` is a port of `esfit_levmar.m` | n/a | EasySpin's own levmar stalls at symmetric starts in 2 of 6 cases |
| `orca2torchspin` | 33 ORCA files identical to `orca2easyspin` (tensors compared in the molecular frame) | n/a | atoms without data are not listed (EasySpin lists zero rows) |
| `autograd` (differentiable) | `differentiable_spectrum` = `pepper_autograd`: forward equals `pepper` to 1e-11; Jacobians vs finite differences of `pepper` for g, A, D, lw, Temperature (see § Autograd) | same as `pepper` | strains, ModAmp, mwPhase, ordering, crystals, frequency sweeps not differentiable |
| Hamiltonians, utilities, constants | exact — 92 cases mirroring EasySpin's `ham_*` test suite (rtol 1e-10) plus `ee2`/`D_` cases | exact | None |
| `eprload` | identical to EasySpin on all 75 readable files in `tests/eprfiles` (Bruker, JEOL, specman, Magnettech, Adani, CIQTEK, ETH formats) | n/a | See § eprload (native units) |
| `isotopologues` | EasySpin's 15 `isotopologues_*` tests ported (incl. the `A_` spherical form); pepper/garlic/chili mixtures validated in MATLAB | exact | None |

---

## Pepper — choosing a method, and what each costs (2026-10-07)

Added after a Cu(II) phthalocyanine fit (CuPc diluted in ZnPc: S=1/2, g =
[2.049 2.181], A(Cu) = [15.3 646.6] MHz, four effectively isotropic 14N at
45 MHz, lw 0.54 mT, X-band) ran into every limit of the resonance solvers at
once.  All timings below are the same machine, `GridSize=[91,4]`, 2667 points,
with EasySpin on MATLAB R2024b for comparison; the generator is
`tests/data/generate_pepper_cupc_refs.m` and the comparison
`torchspin/tests/test_pepper_cupc_matlab_validation.py` (34 cases).

**Accuracy.** Against the exact `matrix` result for the same system:

| Nuclei | `hybrid` | `perturb` (2nd order) |
|---|---|---|
| Cu + 1×14N | 0.9805 | 0.896 |
| Cu + 2×14N | 0.9740 | 0.918 |

These are cosine similarities, and **EasySpin's own hybrid gives 0.982 and 0.974
on the same comparison** — the deviation is the first-order electron–nuclear
decoupling that defines the method, not a port defect.  It shrinks as the
perturbational coupling shrinks (0.9805 → 0.9999 as A(N) goes 45 → 0.1 MHz) and
as the exact core grows (0.974 → 0.984 when the first nitrogen joins the core).
Second-order perturbation theory does **not** converge the same way: with
A∥(Cu) ≈ 650 MHz it is wrong about the copper itself, so it plateaus near 0.95
however small the nitrogen coupling is.

Pick `matrix` when it is affordable, `hybrid` when a large central hyperfine
coupling is surrounded by small ligand couplings, and `perturb` only when all
couplings are small compared with the microwave quantum.

**Cost.** Exact diagonalization grows with the Hilbert space, and the
natural-abundance isotopologue expansion multiplies whatever that costs
(Cu + 4N natural abundance is 10 separate simulations):

| System | method | torchspin | EasySpin |
|---|---|---|---|
| 63Cu + 2×14N (72 states) | `matrix` | 5.9 s | 16.3 s |
| 63Cu + 2×14N | `hybrid` | 0.06 s | 0.14 s |
| 63Cu + 4×14N (648 states) | `matrix` | hours | ~10 min |
| 63Cu + 4×14N | `hybrid` | 0.27 s | 0.28 s |
| 63Cu + 4×14N | `perturb` | 0.06 s | 0.04 s |
| Cu + 4×N, natural abundance | `hybrid` | 2.0 s | 5.6 s |
| Cu + 4×N, natural abundance | `perturb` | 0.41 s | 0.71 s |
| Cu + N with `n=[1,4]` | `hybrid` | 0.30 s | not supported |

Two things follow.  `matrix` on Cu + 4×14N is not a fit-loop operation in either
code — that is what `hybrid` is for.  And if the isotope matters less than the
run time, name the isotopes (`'63Cu'`, `'14N'`) or raise `Opt.IsoCutoff`:
natural-abundance copper alone doubles the work for a 0.3 % spectral change.

**Sets of equivalent nuclei.** `pepper` accepts `SpinSystem.n > 1` on the
`hybrid` and `perturb` paths, where nuclei are treated one at a time and
combined combinatorially, so a multiplicity is a pure saving — four equivalent
14N written as `n=[1,4]` cost 0.30 s instead of 2.0 s and agree with four
explicit nitrogens to cosine 0.99999999.  The `matrix` path still rejects
`n > 1`, because there the nuclei multiply the Hilbert space and cannot be
collapsed; EasySpin's `pepper` rejects `n > 1` for every method.  Equivalence
means *identical couplings and identical orientations*: four nitrogens with
different `AFrame` tilts are not a set of equivalent nuclei, whatever their
principal values.

**Default `GridSize=[19,4]` is unconverged for narrow lines on a large
hyperfine.** For this system the default grid leaves the powder average
dominated by ripple, and the two codes place that ripple differently: the same
Cu + 4×N spectrum agrees with EasySpin to cosine 0.9998 at `[91,4]` but only
0.96–0.97 at `[19,4]`.  Neither result is right — both are unconverged.  There
is no automatic warning; converge the grid yourself whenever the linewidth is
small compared with the field spread between neighboring grid points.  The two
`cupc_grid19_4N_*` cases are kept in the validation suite as the record of this.

## Pepper — absolute intensity (fixed 2026-09-01)

Until 2026-09-01 `pepper()` multiplied the spectrum by one of four empirical
factors (8.48, 6.29, 4.27, 3.806) chosen by accumulation path. These are gone.
The intensity chain now follows EasySpin exactly: resonance-field intensities
carry the Aasa–Vänngård factor dBdE = 1/|⟨v|μ_z|v⟩−⟨u|μ_z|u⟩| and the
1/∏(2I+1) share per hyperfine line, orientation weights sum to 4π, the χ
integral contributes 2π, and stick/line accumulations are converted to a
spectral density by 1/ΔB. Verified against EasySpin's own checks: the
isotropic-powder closed form ∫spec dB = 8π²·TransitionRate·dBdE (matrix and
perturbation methods, 1e-4), nuclei-independent double integrals (1e-6),
per-line intensities from `resfields` (7 digits), and amplitude ratios of
1.00 ± 0.02 against every MATLAB pepper reference. The only path still off is
the `Ci`/`C1` scattered-interpolation path (≈5 % low), tracked in
the round-2 entry in `CHANGELOG.md`.

## Pepper — tilted axial symmetry frames (fixed 2026-09-01)

`hamsymm` reproduces EasySpin's point-group and symmetry-frame detection
exactly. Until 2026-09-01 pepper interpolated the rotated grid in the
molecular frame, which broke every **Dinfh** system whose unique axis is not
the molecular z (tilted `gFrame`/`DFrame`, x- or y-axial tensors: cosine 0.34
or an all-zero spectrum). Interpolation and zone weights now use the
symmetry-frame grid; these cases match MATLAB to cosine 0.99996–1.0
(`test_pepper_matlab_validation_ext.py`).

## Pepper — strain broadening (validated 2026-09-01)

Strain widths are validated per transition and orientation against EasySpin
`resfields` (≤1.3e-3 relative; the residual is EasySpin's linear eigenvector
interpolation between field knots). Powder spectra with H/g/A/D strain,
including several electron spins, tilted D frames and correlated D/E strain,
match EasySpin to cosine ≥0.999 and amplitude within 3 % on 24 MATLAB cases
(`test_strain_*_matlab_validation.py`, `test_pepper_dstrain_matlab_validation.py`).
pepper follows EasySpin's summation scheme for anisotropic widths (per-facet
Gaussians on the interpolated grid with Lambda smoothing) and its level-pair
transition bookkeeping.

## Chili — slow-motion SLE (rewritten 2026-09-01)

`chili()` is now a function-by-function port of EasySpin's *general*
Liouvillian method (`torchspin/chili_sle.py`): LjKKM basis with the same
pruning switches (`jKmin`, `evenK`, `highField`, `pImax`, `MpSymm`), ISTO
interaction coefficients, rotational-basis operators, the Liouville
Hamiltonian, the diffusion superoperator with orienting potentials of any
(L, M, K), the equilibrium starting vector, the Lanczos/Lentz continued
fraction (and direct / eigenvalue solvers), EasySpin's field-sweep methods
and output scaling, post-convolution nuclei and multi-component input.
Every intermediate was checked against EasySpin's `Opt.Diagnostics` dump
(basis identical, H and Γ to machine precision). On 60 MATLAB cases
mirroring EasySpin's chili tests, 59 match to cosine ≥0.999 and amplitude
within 2 %; the one at 0.998 is a case whose reference used EasySpin's
compiled `fast` builder, which differs from EasySpin's own general method by
0.2 % in the slow-motion limit.

Default basis symmetry (fixed 2026-09-02): EasySpin auto-selects its compiled
`fast` builder for one S = 1/2 electron with ≤ 2 nuclei and no potential.  In
MATLAB that builder equals the general method *with* the M–pS–pI (Meirovitch)
symmetry basis at half the amplitude whenever the basis is rhombic (Mmax > 0
and Kmax > 0), and equals the plain general method when the basis is axial
(Mmax = 0 or Kmax = 0); the plain general basis differs from it in the
slow-motion limit (cosine 0.937 at τc = 100 ns for the nitroxide example,
1.000 with the symmetry basis).  torchspin applies that empirical rule when
`ChiliOptions.MpSymm` is left at `None`, so its default output reproduces
EasySpin's default output on every stored case; an explicit `MpSymm=True/False`
behaves like EasySpin's general method with that option.  The `fast` builder
itself (`chili_lm.c`) is not ported, so systems outside the stored cases
(e.g. unusual LLMK choices) should be checked against EasySpin.

Not ported: the `fast` C builder (`chili_lm.c`), `Sys.Exchange`,
`Exp.Ordering`, `Opt.useLMKbasis`, and the automatic sweep range. The basis
size (`Opt.LLMK`) must be chosen by the user, exactly as in EasySpin (there is
no adaptive basis selection in EasySpin either).

## ENDOR (`salt`)

Since 2026-09-01 `salt()` implements EasySpin's fixed-field, rf-swept powder
ENDOR (`Experiment.Field` + RF `Range`, optional `ExciteWidth` orientation
selection) and is validated on an 8-case suite mirroring EasySpin's salt
tests (`test_salt_matlab_validation.py`). Since 2026-09-03 the powder
average uses EasySpin's scheme (grid interpolation with `GridSize=[N, Ninterp]`
followed by triangle/zone projection) instead of stick binning, so the
reference `Opt.GridSize = [20 5]` case reproduces MATLAB to cosine 0.999998.
The three orientation-selected (`ExciteWidth`) cases remain at cosine
0.993–0.998; the residual there is in the orientation weights, not the grid.
Since 2026-09-06 the fixed-field path is differentiable (torch interpolation /
projection / convolution shared with `pepper_autograd`; gradients vs finite
differences 1e-6–1e-8 for g, A, Q, lw, `test_salt_autograd.py`); the field-swept
(multi-field) mode is not and raises for grad tensors.

## Garlic — what is and is not ported

Line positions use EasySpin's algorithm exactly (per-nuclear-group
Breit–Rabi fixed-point solver by default, or `Options.Method='perturbN'`),
including sets of equivalent nuclei (`SpinSystem.n`), frequency sweeps
(`Field` + `mwRange`), `CenterSweep`, field modulation (`ModAmp`), Boltzmann
polarization and full g/A matrices. Not ported: automatic sweep ranging
(`Experiment.Range` or `CenterSweep` is required), natural-abundance isotope
mixtures and multi-component input (`compisoloop`), and `Opt.separate`.
Equivalent nuclei are not allowed in the fast-motion regime (same as
EasySpin). When no broadening is given, a requested derivative harmonic
falls back to the absorption stick spectrum (EasySpin's auto-harmonic).

## Pulse EPR — saffron and spidyan are MATLAB-validated; saffron_thyme is not

`saffron()` is cross-validated on 25 MATLAB reference cases: predefined
2p/3p/4p ESEEM, HYSCORE and MimsENDOR, S>1/2 systems, custom sequences,
`ProductRule` (with and without `TimeDomain`), single crystals
(`SampleFrame`/`CrystalSymmetry`/`MolFrame`, all 230 space groups via
`sitetransforms`) and two-component input. Cosine ≥0.98 throughout (crystal
HYSCORE 0.989, two components >0.98, everything else ≥0.999). HYSCORE
remains much slower than EasySpin.

The former `saffron_2psimple` xfail (14N + quadrupole) is resolved: the test
had used a quadrupole coupling ten times smaller than EasySpin's own test, and
`SpinSystem.Q` now converts EasySpin's `[e2qQ/h eta]` shorthand (cosine 0.99995).

`spidyan()` is cross-validated on 7 MATLAB cases (cosine > 0.99) after the
simulation-frame port (ZeemanFreq→g translation, `SimFreq`, pulse carriers
via `pulse()`/`rfmixer`, phase cycling, `DetFreq` down-conversion, `Dim`
sweeps of pulse parameters including the pulse length `tp`; 8 MATLAB cases).

`saffron_thyme()` (real-pulse engine) has internal-consistency tests only,
including crystal orientations; it has **no direct MATLAB cross-validation**.

---

## Cardamom — validated against EasySpin (2026-09-02)

`tests/generate_cardamom_matlab_refs.m` (EasySpin 6 parameter names) was run with
`rng(42)`; `test_cardamom_matlab_validation.py` compares three cases (nitroxide
diffusion/fast, two-site jump/fast, diffusion/ISTOs) and reaches cosine 0.9998,
≥0.97 and 0.9992 with amplitude ratio 1.00 — the residual differences are the
random-number streams, which cannot be matched across MATLAB and NumPy.

Three EasySpin conventions were adopted to get there: field-swept cardamom
spectra are the imaginary FFT of the time-weighted FID and ignore
`Exp.Harmonic`; the spectrum scale is 1/nOrients of the weighted orientation sum
(EasySpin weights and then averages again); the jump model takes its dynamics
from `Sys.TransRates`/`Sys.Orientations` and needs no correlation time.

Remaining caveats audited in v0.1.0:
- The `SzFilter` secular projection used in MATLAB cardamom ISTOs is
  not yet applied in torchspin; for hyperfine systems this can cause subtle
  pseudo-secular differences.
- The `stochtraj_diffusion` Wigner-potential torque is computed via Euler-
  angle finite differences, which is exact only for axial L=2,M=0,K=0
  potentials. Generic potentials produce a known approximation.

---

## Pepper — round 3 (2026-09-02)

Added and MATLAB-validated (`test_pepper_round3_matlab_validation.py`, 32 cases):
`Experiment.mwMode` (tilted linear, unpolarised and circular excitation for
powders and crystals on the matrix, frequency-sweep and perturbation paths),
photoselection with perturbation theory, `Options.separate='transitions'`,
automatic field and frequency sweep ranges, ordering in frequency sweeps.

Two EasySpin-side approximations surfaced and are documented rather than
copied: (1) EasySpin's matrix path interpolates eigenvectors across the field
segment; for 63Cu+14N its intensities are 3–4 % off the exact value and its
spectrum disagrees with its own perturb2 (cosine 0.951), while torchspin's exact
intensities agree with EasySpin perturb2 at 0.988 (`septrans_CuN_matrix` xfail).
(2) For S=5/2 in parallel mode torchspin and EasySpin agree exactly on the coarse
grid; with the default [19 4] interpolation EasySpin sits farther from its own
converged spectrum (0.9947) than torchspin (0.9983); the test threshold is 0.995.

Internals aligned with EasySpin in the process: global transition pre-/post-
selection (`Opt.Threshold`) instead of per-orientation thresholds, a
Hermite-cubic tangent test that finds paired resonances of looping transitions,
linear (L1) / local-cubic (L3) interpolation of intensities and widths, and
`hamsymm` equality at 1e-12 relative so finite-difference steps break symmetry.

---

## Pepper — features validated in round 2 (2026-09-01)

All of the following are MATLAB-validated (`test_pepper_crystal_*`,
`test_pepper_parallel_*`, `test_pepper_features_*`, `test_pepper_features2_*`):
single crystals (`SampleFrame`/`CrystalSymmetry`/`MolFrame`/`SampleRotation`,
`Options.Sites`, `separate='orientations'|'sites'`), parallel-mode detection,
natural-abundance and custom isotopologues (`'Cu'`, `'(63,65)Cu'` + `Abund`,
`Options.IsoCutoff`), non-equilibrium populations (`SpinSystem.initState` in the
zero-field/eigen/xyz/coupled/uncoupled bases, density matrices, `'T0'`,
`'singlet'`), partial ordering (`Experiment.Ordering`, λ or callable),
photoselection (`lightBeam`, `lightScatter`, `tdm`), field modulation
(`ModAmp`), dispersion (`mwPhase`), `separate='components'`, and frequency
sweeps with strain/interpolation/projection through the same code path as
field sweeps. Not ported: `separate='transitions'`, unpolarised/circular
excitation, ordering in frequency sweeps, photoselection with perturbation
theory, automatic sweep ranging in pepper (garlic auto-ranges).

---

## eprload — EasySpin oracle and native units (behavior change, 2026-09-01)

`eprload` is validated against EasySpin's output for every readable file in
`tests/eprfiles` (75 files). To achieve parity the Bruker readers now return
the abscissa in the file's native units (Gauss for BES3T/ESP, MHz for ENDOR,
seconds for time sweeps) instead of converting to mT, multi-dimensional data
are returned as arrays with one abscissa per dimension (a list), several data
values per point (`IKKF 'CPLX,CPLX'`) give a list of datasets, and JEOL /
specman axes follow EasySpin's conventions. Callers that relied on mT must
divide by 10.

---

## Stochastic trajectories — reproducibility

`DiffusionPar` and `JumpPar` now expose a `seed: Optional[int]` field that
threads through all RNG calls. Setting `seed=42` produces bit-reproducible
trajectories across runs. Without a seed, output is non-deterministic.

This was added in v0.1.0 to support reproducible benchmarks; users porting
from earlier development snapshots should pass `seed=` explicitly.

---

## Autograd — differentiable pepper

Since 2026-09-05 `differentiable_spectrum()` builds a `SpinSystem` from the
parameter tensors and evaluates it with `pepper_autograd()`, which is `pepper`'s
own forward path (resonance-field search with transition tracking, EasySpin
grid interpolation, SOPHE triangle/zone projection, EasySpin convolution and
harmonic) written on the autograd graph. Forward output equals `pepper()` to
1e-11 relative for every grid symmetry, `GridSize` (int or `[N, Ninterp]`) and
harmonic; gradients through the resonance-field search come from the converged
Newton step (implicit-function theorem), no custom backward.

Gradient validation against central finite differences of `pepper()`
(`test_pepper_autograd.py`): A, D, Gaussian and Lorentzian lw and Temperature
agree to ~1e-8 relative norm error; g gradients agree to the finite-difference
limit set by the piecewise-linear projection (a vertex crossing a field bin is
a kink): relative norm error 6e-2 / 8e-3 / 1.5e-4 / 2e-8 for steps 1e-4 /
1e-5 / 1e-6 / 1e-7 in g, identically for `pepper` and `pepper_autograd`.
Optimizers see an almost-everywhere exact gradient of a piecewise-smooth model.

The earlier stand-alone models remain as `method='broadband'` and
`method='analytical'` (deprecated). They sum discrete orientations without
interpolation or projection and carry orientation-grid ripple at GridSizes
where pepper is converged: for a Cu(II) hyperfine system at GridSize 31 the
cosine to EasySpin is 0.985–0.999 (absorption) and 0.73–0.88 (first
derivative), against 0.9998–1.0000 / 0.999 for `pepper`; the ripple only falls
below 1 % of the signal around GridSize 91. Do not train models on their output.

Strains are differentiable (HStrain, gStrain, AStrain; DStrain magnitudes
are constants). Not yet differentiable (`pepper_autograd` raises
`NotImplementedError`, `pepper()` itself is unchanged): `Exp.mwPhase`,
partial ordering, photoselection, crystals, frequency sweeps, `Sys.initState`.
`salt` (fixed-field mode) and `garlic` (all modes) are
differentiable through the same approach (`test_salt_autograd.py`,
`test_garlic_autograd.py`); `garlic` switches its nearest-bin stick spectrum to
the two-bin linear split for grad tensors (sub-bin quantisation difference).
`saffron` is differentiable for ESEEM/HYSCORE with grad tensors (returns tensors
and uses `TimeDomain=True` accumulation; Mims-ENDOR raises). `chili` has no
differentiable path yet (needs a torch SLE solve); `curry` is pure torch and
differentiable.

## Platform and dependency support

`pyproject.toml` declares Python >= 3.10, `torch>=2.0,<3`, `numpy>=1.23,<3`,
`scipy>=1.10`.  Two combinations are exercised directly:

| | development host | clean-room host |
|---|---|---|
| OS | Ubuntu 22.04 (x86-64) | Ubuntu 22.04 (x86-64) |
| Python | 3.12 | 3.12 |
| NumPy / SciPy | 2.4 / 1.17 | 2.5 / 1.18 |
| PyTorch | 2.5 (CUDA 12.1) | 2.14 (CUDA 13.0) |
| GPU | 2× RTX 4090 | 2× RTX 3090 |

CI additionally runs the suite on Linux, macOS and Windows across Python
3.10-3.13 on every push, which is the full range the package claims support
for (`requires-python = ">=3.10"`).

The clean-room host installs the package into an empty environment from the
declared dependencies alone and runs the full suite; it is what the release is
gated on.  NumPy 1.x is still supported (the `np.trapezoid` / `np.trapz`
fallbacks) but is not routinely exercised.

Platform-specific breakage is real and worth testing for rather than assuming
away: the first CI run the project ever completed found that `eprload` could
not pair Bruker `.SPC`/`.PAR` files on a case-sensitive filesystem, which no
amount of testing on macOS would have surfaced.

CUDA is used opportunistically.  Batched eigendecompositions of matrices
larger than 32×32 are routed to the CPU thread pool (the cuSOLVER
`syevjBatched` limit), and `pepper`'s grid interpolation, triangle projection
and strain accumulation are host-side regardless of `device`; see the device
audit in
`benchmarks/results/workstation_20260904_verified/BENCHMARK_VERIFICATION.md`.
`chili` and `cardamom` have no CUDA path at all.

## Test counts (v0.3.0, 2026-10-04, clean-room host)

| Outcome | Count |
|---|---|
| Passed | 2659 |
| Failed | 0 |
| Skipped | 4 |
| xfailed (documented known issues) | 3 |

The documented xfails are:
1. `test_pepper_round3_matlab_validation.py::[septrans_CuN_matrix]` — EasySpin's
   matrix-path eigenvector interpolation (see § Pepper — round 3)
2. `test_potential_biases_distribution` — stochastic Wigner-potential biasing
   requires longer equilibration than the test budget
3. `test_cardamom.py::test_fast_motion` — stochastic fast-motion narrowing
   needs more trajectories than the test budget

The 4 skips are environment-dependent, not unimplemented behavior:

* 3 in `test_fitgui.py` — `ipywidgets` is absent (it ships in the `gui`
  extra, not `test`); install `torchspin[gui]` to run them.
* 1 in `test_gpu_consistency.py` — it exercises the CPU fallback taken when
  CUDA is *missing*, so it skips on a machine that has a GPU.

Counts come from the clean-room host described above, with `pip install
-e ".[test]"` and both GPUs visible.  On a CPU-only machine the CUDA-variant
tests skip instead of running, so the skip count rises and the pass count
falls correspondingly; no test fails.

---

## How this document is maintained

This file is updated as caveats are addressed or new ones are discovered.
Each release notes which limitations changed in `CHANGELOG.md`.

This document is the distilled public record of an internal audit (~50 pages
of findings and synthesis notes) carried out during the port.  The raw audit
notes are not distributed; everything they established that bears on using
torchspin is stated here, and the evidence is the test suite plus the stored
EasySpin references in `tests/data/`.
