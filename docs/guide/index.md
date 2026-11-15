# User guide

Short, task-oriented pages: what each part of torchspin is for and how to use
it, with examples you can paste into Python.  Every example on these pages is
run against the code (`python docs/scripts/check_guide_examples.py`).  For the
full list of parameters of a function or class, follow the links to the
[API reference](../api/index.rst).

| Page | Use it to |
|---|---|
| [Spin systems, experiments and options](spin_systems.md) | describe a sample, a measurement and the numerical settings |
| [Solid-state (powder) spectra](powder.md) | simulate frozen-solution and powder CW EPR with `pepper` |
| [Is my orientation grid fine enough?](grid.md) | check a powder spectrum for grid convergence |
| [Solution and slow-motion spectra](solution.md) | isotropic and fast-motion spectra with `garlic`, slow motion with `chili` |
| [ENDOR and pulse EPR](endor_pulse.md) | ENDOR spectra with `salt`, pulse experiments with `saffron` |
| [Fitting spectra](fitting.md) | fit simulation parameters to measured data with `esfit` |
| [Gradients and differentiable simulation](gradients.md) | get derivatives of spectra, use a GPU |
| [Ligand-field tools](ligand_field.md) | go from a ligand-field model to g values, magnetometry and spectra |

Units throughout: energies and couplings in MHz, magnetic fields in mT,
microwave frequencies in GHz, angles in radians, temperatures in K.

When a result looks wrong, see [Troubleshooting and reference](../reference.md):
pitfalls, known limitations and the change log.

```{toctree}
:hidden:

spin_systems
powder
grid
solution
endor_pulse
fitting
gradients
ligand_field
```
