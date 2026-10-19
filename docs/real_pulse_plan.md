# Plan: make the real-pulse / `saffron_thyme` route EasySpin-equivalent

Status: **All stages done, 2026-10-19.** Stage 4 needed no new code; Stage 5's
premise turned out to be false in EasySpin too and was replaced by three
reference-free relations that do hold. Written
2026-10-19, after the 0.5.0 pulse-EPR audit. Everything below is scoped against
EasySpin on MATLAB R2024b, reading `saffron.m` (thyme branch, ~line 1829
onward), `private/s_sequencer.m` and `signalprocessing.m`.

See **Outcome** at the end for what the execution found, including the three
places where this plan's diagnosis was wrong.

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
`SimulationMode='thyme'` select the engine explicitly. API only — no behavior
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

---

## Outcome (2026-10-19)

The route is now EasySpin-equivalent and MATLAB-validated on 18 cases with
rectangular pulses, at 3e-13..3e-12 relative L2 on both the raw and the
mean-centered signal: `tests/data/generate_saffron_thyme_refs.m` →
`torchspin/tests/test_saffron_thyme_matlab_validation.py`. Two things in the
plan above were wrong, and both mattered.

### "The detection stage is missing, not the propagation" was wrong

The detection stage was indeed missing, exactly as tabulated. But the
propagation was broken too, in three independent ways, any one of which makes
the output meaningless:

1. **No pulse carrier.** `_saffron_exp_to_spidyan` never passed `Exp.mwFreq`
   to the sequencer, so `FrameShift` was 0, the pulse IQ was a DC envelope,
   and it was applied in the lab frame against a ~9.5 GHz electron Zeeman
   term. Nothing was excited. There was also no simulation-frame g shift
   (`saffron.m` 1838-1846).
2. **Mixed frames.** The Hamiltonian was built in the molecular frame with the
   field rotated into it (`ham(sys, B_mol)`), while `-Sz`, `Sx` and `S+` stayed
   lab-frame operators. Any orientation with a nonzero phi angle was wrong;
   the error vanishes at phi = 0, which is why a single crystal at
   `SampleFrame = [0 beta 0]` looked fine.
3. **Renormalized orientation sum.** The accumulated signal was divided by the
   total grid weight. EasySpin does not: `Exp.OriWeights` sums to `4*pi` for a
   powder and each crystal site carries `4*pi`, so a powder was ~12.6x too
   small and a two-site crystal stack came out as a mean instead of a sum.

Rotating the *operators* into the molecular frame instead is not an option:
`s_thyme` builds its pulse propagators from `real(xOp)` and `1i*imag(xOp)`,
which presumes `xOp = Sx`. What works is to conjugate the Hamiltonian with the
spin-rotation operator `exp(-i*phi*Jz) exp(-i*theta*Jy) exp(-i*chi*Jz)` taken
over *all* spins, which reproduces EasySpin's `rotatesystem` + `ham(Sys_L,
[0 0 B])` to 5e-13 with no per-interaction tensor handling.

The `Opt.IntTimeStep` default also did not match: EasySpin divides the Nyquist
time by 50 and rounds to two significant digits; torchspin divided by 10 and
did not round.

### "The API gap is the Det* fields" was incomplete

`Opt.SimulationMode` and the `Exp.Det*` fields were missing as described, and
the multi-electron error at `saffron.py:765` did point at an option that did
not exist. Two further gaps:

- **`Exp.Sequence` could not express a real-pulse sequence at all.** EasySpin's
  thyme route takes a list of pulse dicts and delays; `PulseExperiment` had only
  a predefined name or saffron's `Flip`/`Phase`/`tp`/`t`/`Inc` arrays, so no
  EasySpin thyme input could be transcribed. `Exp.Sequence` now accepts the
  list form, with `Exp.Dim1`/`Exp.Dim2` and `Exp.PhaseCycle`.
- **Predefined + `Exp.tp` was not an unvalidated extension; it was silently
  ignored.** `saffron()` never reached the thyme dispatch for a predefined
  sequence, so `Exp.tp` was dropped and an *ideal-pulse* result was returned.
  EasySpin rejects the combination in fast mode (`saffron.m` 351) and supports
  it in thyme mode via `s_predefinedexperiments.m`; both now hold. 4pESEEM and
  MimsENDOR have no thyme template in EasySpin either and raise.

### A defect found on the way, outside this plan's scope

Relaxation in Liouville space propagated `conj(rho)`. `spidyan`'s commutator
superoperator was written as EasySpin writes it, `kron(I,H) - kron(H.',I)`,
which is correct for MATLAB's column-major `vec()`; numpy flattens row-major,
where the two Kronecker factors swap. The existing `spidyan` relaxation
reference detects `Sz`, whose trace is real, and was compared on its real part
only, so it could not see this. `generate_spidyan_refs.m` gains an `S+`
relaxation case compared on its imaginary part as well.

### Stage 4 needed no new code

All ten of EasySpin's own `saffron_thyme_*` cases now agree at 3e-13..8e-12
(`tests/data/generate_saffron_thyme_shaped_refs.m` →
`test_saffron_thyme_shaped_matlab_validation.py`), and not one line of pulse
shaping had to be written. `torchspin/pulse.py` already built `quartersin/
linear` chirps with `trise` and a swept `Frequency` for `spidyan`, and a pulse
dict in a list-valued `Exp.Sequence` passes through to it unchanged. The plan's
guess that Stage 4 was "mostly plumbing shapes through
`_saffron_exp_to_spidyan`" overestimated it: the plumbing already existed, and
what had kept these cases from running was the propagation and detection fixed
in Stage 2. The one case that did need the new API is
`saffron_thyme_indirectdimensions`, whose `Exp.Dim2 = {'p1.Flip' pi/3}`
regenerates the chirp at each point of the second dimension.

### Stage 5's premise is false, in EasySpin too

"As `tp -> 0`, the real-pulse result must approach the ideal-pulse result for
the same sequence" does not hold. At matched settings torchspin reproduces
EasySpin's *own* ideal-vs-real mismatch to 1e-7 in scale across pulse lengths
from 20 ns down to 2.5 ns, so there is nothing here for torchspin to fix.
Three reasons, none a defect:

1. With single-point detection EasySpin does not down-convert (`saffron.m`
   1917 runs `signalprocessing` only when detection is *not* single-point),
   so the detected point keeps the simulation-frame carrier phase, which
   advances ~50 cycles per 20 ns increment.
2. A single detected point contains coherence pathways the ideal route
   excludes analytically. Neither a two-step cycle on either pulse nor echo
   integration over a window removes them for this observable.
3. As `tp -> 0` at fixed flip angle, B1 = Flip/(2*pi*tp) grows without bound
   and eventually exceeds the carrier frequency, so the rotating-wave picture
   breaks down and a rectangular real pulse does not approach an ideal one.

Moving the simulation frame onto the carrier to kill (1) is not neutral: it
removes the carrier from the pulse as well, so the linearly polarized field
becomes a rotating-frame field of twice the strength.

Three reference-free relations cover what Stage 5 was reaching for and do
hold, in `test_saffron_thyme.py`:

- a single-site P1 crystal equals `4*pi` times the independently validated
  `spidyan` result (1e-12), which is the cheapest possible guard on the
  orientation weight, the detection stage and the down-conversion at once;
- an isotropic powder equals a single crystal;
- a nominal `Pulse.Flip` of `pi/2` delivers full transverse magnetization and
  `pi` delivers none, which pins the pulse amplitude against a factor of two.

### Stage 3 notes, kept for reference

- `Opt.IntTimeStep` is pinned at 1e-4 µs in the references, which EasySpin
  warns is coarse (its own default here is 4.3e-6 µs). What is validated is
  that the two codes do the same calculation; `eseem2p_fine` repeats one case
  at 2e-5 µs and agrees equally well.
- A *transient* `Exp.DetSequence` cannot be cross-validated: `saffron.m` line
  1937 reads `Exp.DetWindow(1)` unconditionally once detection is not
  single-point, so EasySpin itself errors with `Unrecognized field name
  "DetWindow"`. The `Exp.DetSequence` case in the suite uses a zero-length
  final delay, which is the single-point path.
- `Exp.DetOperator` is ignored by EasySpin's saffron (only `spidyan` transfers
  it to `Opt`); torchspin warns and always detects `S+`.
- EasySpin's own `saffron_thyme_*` test cases are all covered by the shaped
  suite; its generator pins `Opt.IntTimeStep` at 2e-5 us for the four chirp
  cases and 1e-4 us for the two without chirps, both well below the relevant
  Nyquist times, with one case repeated at 5e-6 us.
