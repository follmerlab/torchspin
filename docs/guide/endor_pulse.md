# ENDOR and pulse EPR

## ENDOR with salt

[`salt`](../api/generated/torchspin.salt.rst) simulates a powder ENDOR
spectrum at a fixed magnetic field (`Experiment.Field`, mT), averaged over
orientations.

```pycon
>>> from torchspin import SpinSystem, Experiment, salt
>>> sys = SpinSystem(S=0.5, g=[2.0, 2.0, 2.1], Nucs='1H', A=[2, 2, 8])
>>> exp = Experiment(mwFreq=9.5, Field=339.3, ExciteWidth=100)
>>> nu, spc = salt(sys, exp, freq_range=(5, 25), n_points=512, lw_mhz=0.2)
>>> float(nu[0]), float(nu[-1])                # frequency axis in MHz
(5.0, 25.0)
```

With `mwFreq` and `ExciteWidth` (MHz) set, each orientation is weighted by
how close its EPR transition at `Field` is to the microwave frequency
(orientation selection, including `HStrain`); without them every orientation
contributes.  `lw_mhz` is the Gaussian ENDOR line width (MHz).

How salt computes the spectrum:

- Resonance fields always come from matrix diagonalization.
- ENDOR frequencies come from first-order perturbation theory for one
  electron spin without quadrupole couplings, and from matrix
  diagonalization otherwise.  The matrix path is also used with
  `Options(Enhancement=True)` (hyperfine enhancement: the rf field couples to
  the total magnetic moment) and with a custom `SpinSystem.Ham`.
- Both paths give the same intensity scale, |⟨i|μ⊥|j⟩|² in (MHz/mT)².

### Zero field

`Experiment(Field=0)` is rejected unless you pass
`Options(ZeroField=True)`, which salt (zero-field ENDOR) and pepper
frequency sweeps (zero-field EPR) accept.  Degenerate levels are then grouped
into clusters (relative tolerance 1e-9 of the level spread), and each
transition between two clusters carries the summed rate of its level pairs,
so intensities do not depend on the arbitrary basis inside a degenerate (for
example Kramers) pair.

## Pulse EPR with saffron

[`saffron`](../api/generated/torchspin.saffron.rst) simulates pulse EPR
experiments with ideal pulses: two-, three- and four-pulse ESEEM, HYSCORE,
Mims ENDOR, and user-defined sequences.  The experiment is a
[`PulseExperiment`](../api/generated/torchspin.PulseExperiment.rst):

```pycon
>>> from torchspin import PulseExperiment, saffron
>>> pexp = PulseExperiment(Field=350.0, Sequence='3pESEEM', tau=0.12,
...                        dt=0.012, nPoints=256)
>>> t, signal, info = saffron(SpinSystem(S=0.5, g=2.0, Nucs='1H', A=[2, 2, 8]), pexp)
>>> t.shape, signal.shape
((256,), (256,))
```

Times are in µs.  For real (shaped, finite) pulses and explicit time
evolution, see [`spidyan`](../api/generated/torchspin.spidyan.rst) and the
other pages under [Pulse EPR](../api/pulse.rst).
