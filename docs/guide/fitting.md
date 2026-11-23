# Fitting spectra

[`esfit`](../api/esfit.rst) adjusts simulation parameters until the simulated
spectrum matches your data, by least squares.  Give it the data, a model (a
torchspin simulator or any function), start values and how far each
parameter may move.

## Fit a simulator directly

Pass the simulator and a dictionary of start objects.  Here the Lorentzian
line width of a solution spectrum is fitted; `vary` gives the allowed change
of each field in the field's own shape (`lw = [Gaussian, Lorentzian]`, and
`0` fixes the Gaussian part):

```pycon
>>> from torchspin import SpinSystem, Experiment, garlic, esfit, FitOptions
>>> exp = Experiment(mwFreq=9.5, Range=[335, 343], nPoints=256)
>>> _, data = garlic(SpinSystem(S=0.5, g=2.003, lw=[0.0, 0.5]), exp)
>>> p0 = {'Sys': SpinSystem(S=0.5, g=2.003, lw=[0.0, 0.4]), 'Exp': exp}
>>> res = esfit(data, garlic, p0, vary={'Sys': {'lw': [0.0, 0.3]}},
...             options=FitOptions(method='levmar'))
>>> res.pnames, round(float(res.pfit[0]), 2)
(['Sys.lw[1]'], 0.5)
```

The result, a [`FitResult`](../api/esfit.rst), holds the fitted parameters
(`pfit`, `pnames`), the fitted curve (`fit`), `residuals`, `rmsd`, and the
uncertainties `pstd`, `ci95`, `cov` and `corr`.

## Fit your own model

Any function of a parameter vector works.  The amplitude never needs to be a
parameter: the model is scaled to the data by least squares
(`FitOptions.autoscale='lsq'`), and a polynomial baseline can be added
(`baseline=0` constant, `1` linear, ...).

```pycon
>>> import numpy as np
>>> x = np.linspace(-5, 5, 201)
>>> def model(p):
...     return np.exp(-0.5 * ((x - p[0]) / p[1]) ** 2)
>>> data = 3.0 * model([0.5, 1.2])
>>> res = esfit(data, model, p0=[0.0, 1.0], vary=[2.0, 0.5],
...             options=FitOptions(method='levmar'))
>>> np.round(res.pfit, 3).tolist(), round(res.scale, 3)
([0.5, 1.2], 3.0)
```

Instead of `vary` you can give explicit bounds `lb`, `ub`; `lb == ub` fixes a
parameter.

## Choosing a method

| `FitOptions(method=...)` | Kind | Use when |
|---|---|---|
| `'simplex'` (default) | local, derivative-free (Nelder–Mead) | robust first choice from a good start |
| `'levmar'` | local, Levenberg–Marquardt | smooth models, fast convergence near the minimum |
| `'trf'` | local, trust-region least squares (SciPy) | bounded problems, many data points |
| `'lbfgsb'`, `'powell'` | local | alternatives when the above stall |
| `'global'` | particle swarm, then simplex polish | rugged loss landscapes, poor start values (derivative spectra) |
| `'swarm'`, `'genetic'`, `'montecarlo'`, `'grid'` | global | exploring the parameter space |

Derivative (field-modulated) spectra have many local minima.
`FitOptions(target='int')` (or `'auto'`, which picks `'int'` for
derivative-like data) compares the integrated residuals, which smooths the
landscape and helps a poor start converge.  It also weights the noise as a
random walk, so its parameters scatter more than those of the default
`target='fcn'` (the maximum-likelihood fit for white noise, and EasySpin's
default).  For the final parameters and uncertainties, finish with `'fcn'`
started from the `'int'` result.  The tutorials
[real-world fitting](../tutorials/05_fitting_real_world.ipynb) and
[blue copper](../tutorials/06_blue_copper_fitting.ipynb) compare the methods
on hard cases.

Fits with pepper's `matrix` method of S = 1/2 systems with a large metal hyperfine coupling and
ligand nuclei (for example Cu(II) with 14N) can stall above the noise level: the simulated spectrum
changes in small steps with the parameters there, which no local optimizer handles well.  Compare the
final RMSD with the noise in your data, and try several nearby starts.  See
[known limitations](../known_limitations.md) (matrix method, spectra with several nuclei).

## Uncertainties

After a successful fit, esfit takes the Jacobian of the (target-transformed)
residuals with respect to the varied parameters by central differences and
computes the HC1 sandwich covariance,
`(J'J)^-1 J' diag(r^2) J (J'J)^-1 · n_data / (n_data - n_active)`, with `r`
the residuals before the target transformation `T`.  For `'int'` and
`'diff'` the middle term is `J' T diag(r^2) T' J`, which accounts for the
correlation the transformation gives the noise.
`pstd` are the square roots of its diagonal, `ci95` is `pfit ± 1.96 pstd`,
and `corr` shows which parameters trade off against each other.  A parameter
that does not change the residuals gets `nan` and a warning.
`FitOptions(compute_uncertainties=False)` skips this (it costs
2 × n_active extra simulations).

## Budgets and stopping

- `max_iter` bounds the work.  For `'trf'` it is SciPy's `max_nfev`, the
  number of step evaluations; the finite-difference Jacobian probes (about
  one per varied parameter per iteration) do not count against it but do
  count in `FitResult.n_evaluations`.  An exhausted budget returns
  `success=False`.
- `max_time` (seconds), `stop_when`, or interrupting the kernel stop the fit
  and return the best parameters so far, with `interrupted=True`.
- `progress='text'` prints progress; `n_workers='auto'` runs population-based
  methods in parallel processes.

## Before you fit

- The local optimizers and the uncertainties use finite differences, so the
  model must be smooth in its parameters: in a custom model that wraps garlic,
  pass `Options(AccumMethod='linear')`.
- With `autoscale='lsq'` the scale cannot be negative, so the model must have
  the sign of the data.
- More in [Pitfalls](../pitfalls.md).  For an interactive panel with sliders,
  see the [fit GUI tutorial](../tutorials/09_fit_gui.ipynb).
