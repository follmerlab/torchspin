# Ligand-field tools

`torchspin.lft` brings Joshua Telser's ligand-field programs (DDN, DDNFIT,
DDNSUS, DDNMAG, DDNPOW, DPN, DSOXF, Tanabe–Sugano, DLSD5, ...) into torchspin,
ported from [telser-tools](https://github.com/follmerlab/telser-tools).  Use
them to go from a ligand-field model (Racah parameters, crystal field or
angular-overlap parameters, spin–orbit coupling) to energy levels, g values,
magnetic susceptibility and EPR spectra.  Every parameter can carry
gradients, so one model can be fitted to band energies, magnetometry, g
values and spectra together.

## Units

Every function takes `units=`, which sets the unit of the energy inputs (B,
C, ζ, Dq, e_σ, ...) and of the energy outputs; `units="cm^-1"` is the usual
choice.  Without it, energies are in MHz like the rest of torchspin.  Angles
are in radians unless you pass `degrees=True`.

## Levels and g values

Energy levels of d³ in an octahedral field, relative to the ground state:

```pycon
>>> from torchspin import lft
>>> V = lft.crystal_field(Dq=1700.0)                       # cm^-1
>>> E = lft.ddn.levels(3, B=800.0, C=3200.0, V=V, zeta=200.0,
...                    units="cm^-1", relative=True)
>>> E.shape                                                # all C(10, 3) states
torch.Size([120])
```

The g tensor of the ground doublet of Cu(II) (d⁹) in an elongated octahedron,
built from bond directions and angular-overlap parameters:

```pycon
>>> lig = [lft.Ligand.isotropic(theta, phi, 6000.0, 1000.0, degrees=True)
...        for theta, phi in [(90, 0), (90, 90), (90, 180), (90, 270)]]
>>> lig += [lft.Ligand.isotropic(theta, 0, 4800.0, 0.0, degrees=True) for theta in (0, 180)]
>>> g = lft.ddn.g_tensor(9, V=lft.aom(lig), zeta=830.0, k=(0.8, 0.8, 0.8), units="cm^-1")
>>> [round(x, 3) for x in g.principal.tolist()]
[2.071, 2.071, 2.362]
```

## From a ligand field to an EPR spectrum

`lft.dn2sys` turns the ligand-field model into a `SpinSystem` (S = 1/2 with
the full g matrix of the ground doublet) that any torchspin simulator
accepts.  Ligand-field parameters go in as a dictionary, because `A` means
the Racah parameter to the ligand-field code and the hyperfine tensor to
`SpinSystem`:

```pycon
>>> from torchspin import Experiment, pepper
>>> sys, gt = lft.dn2sys(9, dict(V=lft.aom(lig), zeta=830.0, k=(0.8,) * 3),
...                      units="cm^-1", Nucs="63Cu", A=[60.0, 60.0, 500.0],
...                      lw=[1.0, 0.0], min_gap=1000.0)
>>> B, spc = pepper(sys, Experiment(mwFreq=9.5, Range=[255, 360]))
>>> spc.shape
torch.Size([1024])
```

`min_gap` raises an error if the ground doublet is too close to the next
state for an effective spin 1/2.

## Magnetometry and fits

```pycon
>>> from torchspin.lft import ddnsus, grids
>>> kw = dict(B=820.0, C=3690.0, zeta=480.0, k=(0.85,) * 3,
...           V=lft.crystal_field(950.0, 250.0, 60.0), units="cm^-1")
>>> chiT = ddnsus.chi_T(7, T=[2, 10, 50, 300], field=100.0, chi_units="cgs",
...                     grid=grids.gauss_hemisphere(6), **kw).powder   # cm^3 K/mol
>>> chiT.shape
torch.Size([1, 4])
```

`lft.ddnmag` computes magnetization, `lft.fit.JointFit` fits one model to
several data types at once with exact Jacobians, and `lft.plot` draws level
diagrams, Tanabe–Sugano diagrams, Zeeman plots and fits.

## Where to go next

- [Ligand-field tools: technical reference](../lft.md): every module, joint
  fits, plots, Telser's spin-Hamiltonian programs, level strain, validation
  and what is not ported.
- [torchspin.lft API](../api/lft.rst): every function and its parameters.
