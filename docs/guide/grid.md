# Is my orientation grid fine enough?

A powder spectrum is an average over orientations.  pepper computes resonance
fields at the knots of an orientation grid (`Options.GridSize`, default `[19, 4]`: 19
orientations from the z axis to the xy plane, refined four-fold by
interpolation) and integrates between them.  For broad lines that is plenty.  For a narrow line on a large
anisotropic coupling, the turning points of the powder pattern can still move
when the grid is refined, and nothing in a single spectrum shows it.

[`grid_convergence`](../api/grid_convergence.rst) answers the question
directly: it simulates the spectrum again on a finer grid (four times finer by
default) and compares the two.

```pycon
>>> from torchspin import SpinSystem, Experiment, Options, grid_convergence
>>> cu = SpinSystem(S=0.5, g=[2.06, 2.06, 2.27], Nucs='63Cu',
...                 A=[60, 60, 647], lw=0.54)
>>> exp = Experiment(mwFreq=9.5, Range=[250, 350], nPoints=2048)
>>> check = grid_convergence(cu, exp)
>>> check.converged
False
>>> print(check)
GridSize=[19, 4] vs [76, 4]: NOT converged (cosine 0.98957, amplitude ratio 0.9980)
```

The default grid is not enough here.  Use the grid it suggests, and check
again:

```pycon
>>> check = grid_convergence(cu, exp, Options(GridSize=[91, 4]))
>>> check.converged
True
```

`check.report()` gives the same result as a sentence with a recommendation.
The criteria are a cosine similarity of at least 0.9999 and a peak amplitude
within 1 % (`cosine_tol`, `amplitude_tol`); the amplitude usually moves
first, because the derivative extremum at a turning point is the most
grid-sensitive number in a powder spectrum.

## Why not a rule of thumb?

A heuristic on the field step between neighboring grid knots looks appealing
but does not work: pepper integrates analytically over each grid segment, so a
large field step per knot is normal and harmless, and such a rule flags many
simulations that are already converged.  Refining and comparing is the
reliable test, at the cost of one extra simulation.  Do it once per spin
system and field range, not inside a fit.
