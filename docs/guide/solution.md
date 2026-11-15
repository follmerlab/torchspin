# Solution and slow-motion spectra

Radicals in solution tumble, which averages the anisotropic parts of g and
the hyperfine couplings.  Which simulator to use depends on how fast:

| Regime | Simulator | What it models |
|---|---|---|
| isotropic limit | [`garlic`](../api/garlic.rst) | sharp lines at the isotropic g and A, broadened by `lw` |
| fast motion (`tcorr` ≲ 1 ns) | `garlic` with `Sys.tcorr` | a separate Lorentzian width for every hyperfine line (Kivelson/Freed) |
| slow motion (`tcorr` ≳ 1 ns) | [`chili`](../api/generated/torchspin.chili.rst) | the full stochastic Liouville equation |

## Isotropic and fast-motion spectra with garlic

```pycon
>>> from torchspin import SpinSystem, Experiment, Options, garlic, chili
>>> nitroxide = SpinSystem(S=0.5, g=[2.0088, 2.0061, 2.0027],
...                        Nucs='14N', A=[16, 16, 86], lw=[0.0, 0.1])
>>> exp = Experiment(mwFreq=9.4, Range=[326, 344], nPoints=2048)
>>> B, spc = garlic(nitroxide, exp)            # isotropic limit
>>> nitroxide.tcorr = 1e-10                    # 100 ps rotational correlation time
>>> B, spc_fast = garlic(nitroxide, exp)       # fast motion
>>> spc.shape, spc_fast.shape
(torch.Size([2048]), torch.Size([2048]))
```

garlic handles one electron spin S = 1/2 with any number of nuclei,
including sets of equivalent nuclei (`n`), which keeps large radicals cheap.
The sweep range may be omitted; it is then chosen from the line positions.

### How garlic places the lines

Each `Nucs` entry (or set of `n` equivalent nuclei) contributes its list of
resonance shifts, and the lists are combined additively.  The shifts come
from the exact Breit–Rabi solution (`Options(Method='exact')`, the default)
or from perturbation theory in the hyperfine coupling (`'perturb1'` to
`'perturb5'`; `'perturb'` is fifth order).  Perturbation theory converges
slowly when a coupling is a sizable fraction of the microwave frequency.  For
isotropic ⁹¹Zr with A = 780 MHz at 9.65 GHz, the relative L2 error of each
order against `'exact'`:

```pycon
>>> import torch
>>> zr = SpinSystem(S=0.5, g=2.0, Nucs='91Zr', A=780, lw=[1.2, 1.8])
>>> zexp = Experiment(mwFreq=9.65, nPoints=4096)
>>> _, ref = garlic(zr, zexp, Options(Method='exact'))
>>> errors = []
>>> for order in (2, 3, 4, 5):
...     _, s = garlic(zr, zexp, Options(Method=f'perturb{order}'))
...     errors.append(round(float(torch.linalg.norm(s - ref) / torch.linalg.norm(ref)), 2))
>>> errors
[0.49, 0.12, 0.05, 0.03]
```

Keep the default `'exact'` unless you know the couplings are small.

### Fitting with garlic

By default garlic adds each line to its nearest field bin
(`Options.AccumMethod='binning'`, as in EasySpin).  That is piecewise
constant in the line positions, so it has no useful derivative with respect
to g or A.  [`esfit`](fitting.md) switches garlic to `'linear'` (a two-bin
split) by itself; in your own model functions, and for gradient-based
optimizers, pass `Options(AccumMethod='linear')` or `'explicit'`.  On the
autograd graph garlic switches to `'linear'` itself.

## Slow motion with chili

When the correlation time approaches the inverse of the spectral anisotropy
(about a nanosecond for a nitroxide at X-band), the fast-motion widths are no
longer valid.  chili solves the stochastic Liouville equation:

```pycon
>>> nitroxide.tcorr = 10e-9                    # 10 ns
>>> B, spc_slow = chili(nitroxide, exp)
>>> spc_slow.shape
torch.Size([2048])
```

Rotational diffusion can also be given as `Diff` or `logDiff`, an ordering
potential as `Potential`; chili's own numerical options are in
[`ChiliOptions`](../api/generated/torchspin.ChiliOptions.rst).
[Known limitations](../known_limitations.md) lists what of EasySpin's chili
is ported.
