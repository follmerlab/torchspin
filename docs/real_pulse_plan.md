# Plan: make the real-pulse / `saffron_thyme` route EasySpin-equivalent

Status: **proposed, not started.** Written 2026-10-19, after the 0.5.0 pulse-EPR
audit. Everything below is scoped against EasySpin on MATLAB R2024b, reading
`saffron.m` (thyme branch, ~line 1829 onward), `private/s_sequencer.m` and
`signalprocessing.m`.

## Why this route is the weak point

After 0.5.0 the ideal-pulse routes (predefined 2p/3p/4pESEEM, HYSCORE,
Mims-ENDOR, custom sequences, direct and frequency-binned) and `spidyan` all
agree with EasySpin at 1e-16..1e-12 under strict relative-error tests. The
real-pulse route does not, and is not covered by any MATLAB comparison:

- `saffron()` with nonzero `Exp.tp` returns per-increment **transients**
  (`nPoints x nDet x nTime`) where EasySpin returns one detected point per
  increment. No reduction of those transients (sum, mean, centre point, sum of
  magnitudes) recovers EasySpin's trace: the closest is ~3% off *after* fitting
  an arbitrary scale factor, and the scale itself is unexplained.
- `saffron_thyme` has **no** MATLAB cross-validation — only internal
  consistency tests, which cannot catch a shared-mode error.
- Being unvalidated is the real risk here, not a known wrong number. The 0.5.0
  audit found eight defects in routes that *were* nominally validated, because
  the metric was too weak. This route has no metric at all.

## Root cause: the detection stage is missing, not the propagation

EasySpin's thyme branch does five things after propagating each orientation
(`saffron.m` 1884-1940). torchspin does the first and skips the rest:

| Step | EasySpin | torchspin |
|---|---|---|
| Orientation sum | `Signal += rawSignals{i}*Exp.OriWeights(i)` | present |
| Detection phase | `Signal *= exp(-1i*Exp.DetPhase)` | **missing** |
| Echo integration | if `Exp.DetIntegrate`, sum over the transient → one point per increment, then set `SinglePointDetection = true` | **missing** |
| Down-conversion | `DetFrequency` defaults to `mwFreq`; `FreqTranslation = -(DetFrequency - FrameShift)`; then `signalprocessing(...)` | **missing** |
| Time axis | `Exp.DetWindow(1):dt:Exp.DetWindow(2)` | **missing** |

That explains the output-shape difference directly: EasySpin collapses to one
point per increment whenever detection is single-point or `DetIntegrate` is set,
and otherwise returns *processed* (down-converted) transients over `DetWindow`.
torchspin returns raw, un-down-converted transients in both cases, so neither
shape nor scale can match — and no post-hoc reduction can reconstruct the
down-conversion.

`signalprocessing.m` is the same function whose complex-trace factor of two was
fix #8 in 0.5.0. Reusing the already-corrected
`_complex_trace_needs_doubling` helper from `torchspin/spidyan.py` is the
natural move.

## The API gap

`PulseExperiment` has **no** `Det*` fields at all, and `SaffronOptions` has no
`SimulationMode`:

- Missing on `PulseExperiment`: `DetWindow`, `DetSequence`, `DetPhase`,
  `DetFrequency`, `DetIntegrate`, `DetOperator`.
- Missing on `SaffronOptions`: `SimulationMode` (`'fast'` | `'thyme'`).

Two consequences:

1. **Engine selection diverges.** EasySpin chooses the engine explicitly with
   `Opt.SimulationMode`; torchspin infers it implicitly from `Exp.tp` being
   nonzero. So EasySpin can run a *predefined* sequence through thyme
   (`s_sequencer` calls `s_predefinedexperiments`), which torchspin cannot
   express, while torchspin silently accepts predefined + `Exp.tp`, which
   EasySpin rejects outright ("You cannot use predefined sequences with real
   pulses").
2. **`saffron.py:765` is a lie.** Multi-electron systems raise
   `NotImplementedError` telling the user to "use `Opt.SimulationMode='thyme'`",
   an option that does not exist. Cheap to fix and worth doing first.

## Detection semantics to port (`s_sequencer.m` 211-325)

`Opt.SinglePointDetection` starts `false` and becomes `true` when either:

- `Exp.DetWindow` is scalar, or `DetWindow(1) == DetWindow(2)`; a zero-length
  event is appended; or
- `sum(Exp.DetSequence) == 1` and the single detected event is a zero-length
  delay.

Otherwise detection is a transient of length `diff(Exp.DetWindow)`. Note
`DetWindow` is applied by *shifting the last sequence element* by
`DetWindow(1)`, not by slicing after the fact — getting this wrong moves the
echo rather than the window.

## Proposed order of work

Each stage ends with a MATLAB-validated strict test, so the route is never left
"probably right".

**Stage 0 — truth in errors (hours).** Fix the `saffron.py:765` message to name
an option that exists. Make predefined + `Exp.tp` either raise as EasySpin does
or warn explicitly that it is an unvalidated torchspin extension. No physics
change; removes two ways to be misled.

**Stage 1 — detection API (1 day).** Add the `Det*` fields to
`PulseExperiment` and `SimulationMode` to `SaffronOptions`, with EasySpin's
defaults (`DetFrequency` falling back to `mwFreq`, `DetPhase` 0). Keep the
existing `Exp.tp` dispatch working so nothing breaks, but let
`SimulationMode='thyme'` select the engine explicitly. API only — no behaviour
change yet.

**Stage 2 — port the detection stage (2-3 days).** Implement, in order:
detection phase; `SinglePointDetection` resolution per `s_sequencer`;
`DetIntegrate`; down-conversion through a shared port of `signalprocessing`
(reusing the corrected complex-trace factor of two); and the `DetWindow` time
axis. This is where the output contract becomes EasySpin's.

**Stage 3 — the first real oracle (1-2 days).** Generator
`tests/data/generate_saffron_thyme_refs.m` plus
`test_saffron_thyme_matlab_validation.py`, asserting **relative L2 error**, not
cosine, following `test_saffron_phase_matlab_validation.py`. Start with
rectangular pulses, since they need no shape support: a 2-pulse echo and a
3-pulse sequence, single-point and windowed detection, with and without
`DetIntegrate`, powder and explicit P1 crystal. Generate the references with
`Opt.SimulationMode='thyme'` explicitly, and state `GridSize` in every case —
two of the three harness faults found in the 0.5.0 audit were an unstated
`Opt.TimeDomain` and an unmatched `GridSize`.

**Stage 4 — shaped pulses (1 week, optional).** `Type` (`'quartersin/linear'`
etc.), `trise`, `Frequency` chirps. `torchspin/pulse.py` already builds shaped
pulses for `spidyan`, so this is mostly plumbing shapes through
`_saffron_exp_to_spidyan` rather than new physics. Only needed to run EasySpin's
own `saffron_thyme_*` test cases, which all use chirps. Defer if Stage 3 closes
the practical gap.

**Stage 5 — ideal-vs-real convergence (hours).** As `tp -> 0`, the real-pulse
result must approach the ideal-pulse result for the same sequence. This needs no
MATLAB reference and would catch a whole class of normalization error. Cheap;
worth having regardless of how far Stages 1-4 get.

## Verification rules, learned the hard way

1. **Assert relative L2 error, never a cosine.** A cosine is scale-invariant and
   dominated by the DC offset; it missed a factor of 8, a factor of 2 and ~100%
   modulation errors in 0.5.0.
2. **State every option that affects the result in both codes** —
   `SimulationMode`, `GridSize`, `IntTimeStep`, `DetWindow`, `DetIntegrate`.
   Three of the 0.5.0 harness faults were unstated options, not wrong code.
3. **Generate references fresh and keep the generator in the repo.** One
   committed fixture is already known stale by 1.4e-6
   (`saffron_orientation_selective.mat`).
4. **Use `GFREE`, not a rounded 2.00232,** when the generator uses EasySpin's
   `gfree`. A 3.5e-7 error in g accumulates into a visible phase drift and was
   briefly mistaken for a simulator defect.
5. **Check that cross-validation tests actually ran**, not just that CI was
   green: the matrix jobs deselect `-k "not matlab_validation"`, so these only
   execute in the full-suite job, and a skipped comparison looks identical to a
   passing one.

## Out of scope

Relaxation during shaped pulses beyond what `spidyan` already supports,
`Opt.LookupTable` performance work, and resonator compensation in the saffron
path. None blocks EasySpin equivalence for the cases above.
