# Physics and validation

This page summarizes, for each torchspin simulator, what it computes and how well that is checked.
Each simulator has its own page with the equations, the method, the full list of tests, and the
limitations. This page describes torchspin 0.9.31.

| Simulator | Computes | Method | Validated against | Strongest evidence | Main caveat |
|---|---|---|---|---|---|
| [pepper](physics/pepper.md), `pepper_autograd`, `differentiable_spectrum` | Rigid-limit CW EPR of powders and crystals, field or frequency sweeps; gradients | Exact, perturbative or hybrid resonance fields; interpolated powder grid; triangle projection | EasySpin (about 250 cases), closed-form intensity, converged grids, central differences | Cosine ≥ 0.999 and amplitude ±2 % on 101 ported EasySpin tests; powder integral to 1e-4; autograd = pepper to 1e-9 | The default grid is not converged for narrow lines on a large hyperfine; check with `grid_convergence` ([§ choosing a method](../KNOWN_LIMITATIONS.md#pepper--choosing-a-method-and-what-each-costs-2026-10-07)) |
| [garlic](physics/garlic.md) | Isotropic and fast-motion solution CW EPR | Exact Breit–Rabi line positions; Kivelson–Freed widths | EasySpin, central differences | Line positions to 1e-6 mT, spectra cosine > 0.9999 | Fast motion only; use chili for slow motion ([§ Garlic](../KNOWN_LIMITATIONS.md#garlic--what-is-and-is-not-ported)) |
| [chili](physics/chili.md) | Slow-motion CW EPR | Stochastic Liouville equation; Lanczos continued fraction | EasySpin (60 cases), EasySpin's internal matrices | Cosine > 0.999, amplitude ±2 %; basis and operators identical to EasySpin | Not differentiable; choose `LLMK` yourself; solver fixes in progress ([chili rules](chili_empirical_rules.md)) |
| [salt](physics/salt.md) | Powder ENDOR at fixed field | EPR resonance search, ENDOR transitions, triangle projection | EasySpin, closed forms, central differences | Cosine > 0.999 on an 8-case suite; first-order path = EasySpin `perturb1` to cosine > 0.999999 | Absolute scale is EasySpin × Δx/(8π²); field-swept mode is a torchspin extension ([§ ENDOR](../KNOWN_LIMITATIONS.md#endor-salt)) |
| [saffron, saffron_thyme, spidyan](physics/pulse.md) | Pulse EPR: ESEEM, HYSCORE, Mims ENDOR, real and shaped pulses, arbitrary sequences | Ideal-pulse pathway sums; stepwise propagation | EasySpin, closed forms | Relative L2 error 1e-6 to 1e-12 on raw and mean-centered signals | saffron gradients use the time-domain path, which EasySpin validates only indirectly ([§ Pulse EPR](../KNOWN_LIMITATIONS.md#pulse-epr--saffron-spidyan-and-saffron_thyme-are-matlab-validated)) |
| [cardamom](physics/cardamom.md) | CW EPR from stochastic or MD orientation trajectories | Density-matrix propagation along trajectories, FFT of the FID | EasySpin (3 cases, statistical) | Cosine 0.9992–0.9998 for diffusion | Finite-ensemble noise; test tolerance only cosine > 0.97 ([§ Cardamom](../KNOWN_LIMITATIONS.md#cardamom--validated-against-easyspin-2026-09-02)) |
| [curry, blochsteady](physics/curry_blochsteady.md) | Magnetization, susceptibility; Bloch steady state | Boltzmann averages over a powder; frequency-domain Bloch solve | EasySpin, Curie law | 0.5–1 % of the maximum on 4 curry cases | χ is a 1 mT finite difference, as in EasySpin |
| [esfit](physics/esfit.md) | Fits of any simulator to data | Ten local and global optimizers | EasySpin fits, SciPy | Reaches EasySpin's residual or better on 6 fits | Uncertainty outputs are not checked against a known covariance |
| [lft](physics/ligand_field.md) | Ligand-field models: levels, g, magnetometry, powder spectra | Full dⁿ determinant basis; degenerate-safe derivatives | Closed forms, telser-tools, part of the Fortran output | 10Dq, 15B, Curie, TIP; exact gradients through degeneracies | telser-tools parity runs only where telser-tools is installed, not in CI ([lft.md](lft.md)) |

## How torchspin validates

Each test belongs to one of five kinds of evidence. Every claim on these pages cites the test file
(in `torchspin/tests/`) that supports it.

| Kind | What it compares against | What agreement shows |
|---|---|---|
| EasySpin reference | Stored EasySpin output in `tests/data/` (provenance: `tests/data/DATA_PROVENANCE.md`) | torchspin computes what EasySpin computes |
| Analytical | A closed form, an exact limit, or a hand-built calculation | The physics is right in that limit, independently of EasySpin |
| Converged reference | torchspin itself on a much finer grid, without interpolation, or with a larger basis | Discretization error is bounded, independently of EasySpin |
| Gradient | Central differences, or `torch.autograd.gradcheck` | The gradient is the derivative of the forward model |
| Consistency | Two torchspin routes, or an invariance (CPU/GPU, autograd/forward, sample tilt = tensor tilt) | The routes agree |

**EasySpin is the physics reference, not the specification.** Agreement with EasySpin cannot reveal
an error that both codes share. Several shared errors were found with analytical, converged-grid or
gradient checks instead:

- a strain width that was not rotation-covariant;
- projected spectra sitting half a bin low;
- a frequency-sweep AStrain width that ignored the tensor frame;
- a mirrored lower hemisphere in C1 interpolation;
- strains ignored by the grid-symmetry choice.

In such cases torchspin follows the objective check and documents the deviation. The EasySpin
comparison is kept as a strict xfail, so it records the difference and cannot start passing
unnoticed. When an EasySpin comparison and a converged or analytical check disagree, the second one
decides.

**Gradients are tested against the forward model, not against EasySpin.** The differentiable paths
are first checked to reproduce the forward simulator to round-off. Their gradients are then compared
with central differences of that simulator.

## Further reading

- [KNOWN_LIMITATIONS.md](../KNOWN_LIMITATIONS.md): what is and is not ported, option by option.
- [chili_empirical_rules.md](chili_empirical_rules.md): why chili's default basis rule is exact, and
  the three solver defects being fixed.
- For maintainers:
  - [coverage gaps](physics/coverage_gaps.md): what the tests do not yet cover.
  - [manuscript claims](physics/manuscript_claims.md): results measured on the manuscript's release
    (v0.2.2) compared with the current code.
