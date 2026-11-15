# Spin systems, experiments and options

Every simulation takes three objects, named after EasySpin's `Sys`, `Exp` and
`Opt` structures:

- a [`SpinSystem`](../api/spinsystem.rst): spins, interactions, line widths;
- an [`Experiment`](../api/experiment.rst): what is measured (microwave
  frequency, field range, temperature, ...);
- optionally [`Options`](../api/options.rst): how it is computed (method,
  orientation grid, device).

```pycon
>>> from torchspin import SpinSystem, Experiment, Options
>>> nitroxide = SpinSystem(S=0.5, g=[2.0088, 2.0061, 2.0027],
...                        Nucs='14N', A=[16, 16, 86], lw=[0.0, 0.3])
>>> exp = Experiment(mwFreq=9.5, Range=[330, 345], nPoints=1024)
>>> opt = Options(GridSize=[19, 4])
```

## Input forms

Tensor fields accept the shorthands EasySpin users know and store the result
as a `torch.float64` tensor of a fixed shape.  The common ones, for one
electron spin:

| Field | You can write | Stored as |
|---|---|---|
| `g` | `2.0` (isotropic); `[g_perp, g_par]` (axial); `[gx, gy, gz]`; a 3×3 matrix | principal values `(1, 3)`, or the full `(3, 3)` matrix |
| `D` (MHz, needs S ≥ 1) | `D`; `[D, E]`; `[Dx, Dy, Dz]`; a 3×3 matrix | principal values, e.g. `[D, E]` → `[-D/3 + E, -D/3 - E, 2D/3]` |
| `A` (MHz) | a scalar (isotropic, every nucleus); `[Ax, Ay, Az]`; one row per nucleus; full 3×3 blocks | `(nNuclei, 3)` |
| `A_` (MHz) | `[aiso, T]` or `[aiso, T, rho]` per nucleus | converted to `A`: `aiso + T*[-1, -1, 2] + rho*[-1, 1, 0]` |
| `Q` (MHz) | `[e²Qq/h, eta]` per nucleus, or principal values | principal values |
| `ee` (MHz, two or more electrons) | a scalar (isotropic exchange); `[Jx, Jy, Jz]`; one row per pair | `(nPairs, 3)` |
| `Nucs` | `'14N'`, `'14N,1H'`, `['63Cu', '14N']`, an element in natural abundance (`'Cu'`), a mixture (`'(63,65)Cu'` with `Abund`) | list of str |
| `n` | equivalent nuclei per `Nucs` entry | list of int |
| `lw`, `lwpp` | `fwhm` (Gaussian) or `[GaussianFWHM, LorentzianFWHM]`; mT for field sweeps | `[G, L]` |

Frames (`gFrame`, `AFrame`, `DFrame`, ...) are Euler angles in radians,
`[alpha, beta, gamma]`, from the molecular frame to the tensor's principal
frame.  Assigning a field later (`sys.g = ...`) converts it the same way.

```pycon
>>> SpinSystem(S=0.5, g=[2.05, 2.25]).g          # axial shorthand
tensor([[2.0500, 2.0500, 2.2500]], dtype=torch.float64)
>>> SpinSystem(S=1, D=[300, 30]).D               # [D, E] in MHz
tensor([[ -70., -130.,  200.]], dtype=torch.float64)
>>> SpinSystem(S=0.5, Nucs='1H', A_=[10, 2]).A   # [aiso, T]
tensor([[ 8.,  8., 14.]], dtype=torch.float64)
>>> SpinSystem(S=0.5, lw=0.5).lw                 # Gaussian only
[0.5, 0.0]
```

Every field, its accepted forms and which simulator reads it are listed on the
[SpinSystem](../api/spinsystem.rst) page.

## Experiments

Field-swept CW EPR needs `mwFreq` (GHz) and `Range` (mT, or `CenterSweep`).
Frequency-swept spectra take a fixed `Field` (mT) and `mwRange` (GHz).
`Harmonic` selects absorption (0), first (1, default) or second (2)
derivative; `Temperature` (K) turns on Boltzmann populations.  ENDOR and other
fixed-field experiments set `Field` only.

```pycon
>>> Experiment(mwFreq=9.5, CenterSweep=[340, 20]).Range
[330.0, 350.0]
```

## Options

Defaults reproduce EasySpin's.  The settings you are most likely to change:

- `Method`: `'matrix'` (exact diagonalization, pepper's default) or
  perturbation theory (`'perturb'`, ...); see the pages for each simulator.
- `GridSize`: orientation grid for powders; see
  [Is my orientation grid fine enough?](grid.md).
- `device`: `'cpu'` or `'cuda'`.
