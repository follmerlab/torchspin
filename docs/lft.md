# `torchspin.lft`: ligand-field tools (Telser's programs, differentiable)

`torchspin.lft` brings Joshua Telser's ligand-field and EPR programs into torchspin. Each one is
ported from [telser-tools](https://github.com/follmerlab/telser-tools) (`telser_tools_diff`), which
reproduces the original Fortran. Every parameter can carry gradients, so a ligand-field model can be
fitted directly to band energies, magnetometry, g values and EPR spectra, together, with exact
Jacobians.

Modules carry Telser's program names. Shared building blocks have descriptive names.

| Module | Telser program | What it computes |
|---|---|---|
| `lft.ddn` | DDN | dⁿ levels, Hamiltonian, ground-doublet g, in the full C(10, n) determinant basis |
| `lft.ddnloop` | DDNLOOP | levels and transitions vs field magnitude, field direction or one parameter |
| `lft.ddnpow` | DDNPOW | powder EPR (field-swept) and FDMR (frequency domain) directly from the dⁿ model |
| `lft.ddnfit` | DDNFIT | fit dⁿ parameters to band energies |
| `lft.ddnsus` | DDNSUS | χ = M/H, χT, μ_eff, and fits to them |
| `lft.ddnmag` | DDNMAG | M(B, T), and fits to it |
| `lft.dpn`, `lft.dpnfit` | DPN, DPNFIT | the same for pⁿ (with its own p-shell AOM and crystal field) |
| `lft.dsoxf` | DSOXF | one free-ion term + spin–orbit + low-symmetry field, d and f shells |
| `lft.tanabe_sugano` | DTANSUG | Tanabe–Sugano blocks and diagram data |
| `lft.dlsd5` | DLSD5 | low-spin d⁵ g tensor, and every inverse g → ligand-field solution |
| `lft.dipole` | DIPOLE | dipolar hyperfine and electron–electron couplings (Slater-orbital multipole form) |
| `lft.dtd` | DTD | ¹⁴N quadrupole couplings, Townes–Dailey |
| `lft.ligand_field` | — | AOM (`lft.aom`, `lft.aom_isotropic`, `lft.Ligand`, `lft.octahedron`, …), Ballhausen crystal field (`lft.crystal_field`), orbital energies |
| `lft.fit` | — | `JointFit`: one model, several data types (`Bands`, `GValues`, `Susceptibility`, `Magnetization`, `Spectrum`) |
| `lft.bridge` | — | `lft.dn2sys`: ligand-field parameters → `SpinSystem` with the full g matrix → `pepper` |
| `lft.plot` | — | level, Tanabe–Sugano, Zeeman/angle/road-map, χT, M, powder and fit plots (cm⁻¹ by default) |
| `lft.telser2torchspin` | DDZHQ, DDPOWH, DJDZE, DDPOWJ | Telser spin-Hamiltonian inputs → `SpinSystem` + `Experiment` (these programs are not re-ported; torchspin already has their physics) |
| `lft.grids` | — | field-direction grids for magnetometry (exact Gauss hemisphere, DDN's igloo, axes) |
| `lft.units` | — | unit factors that keep the autograd graph |

## Units

- **torchspin's units at every public function.** Energies in MHz, fields in mT, frequencies in GHz,
  angles in radians, temperatures in K. Molar susceptibility is in m³/mol and magnetization in Bohr
  magnetons per ion, as `curry` returns them. Constants come from `torchspin.constants`.
- **One `units=` per call** sets the unit of the energy-valued inputs (B, C, ζ, Dq, e_σ, the
  ligand-field matrix, widths) *and* of the energy-valued outputs (levels, gaps, FDMR axes).
  `units='cm^-1'` therefore works throughout; `'GHz'`, `'K'`, `'eV'` too. Internally everything is
  cm⁻¹.
- **Ligand-field plots default to cm⁻¹** (`units=` changes it).
- `chi_units='SI'` (m³/mol) or `'cgs'` (cm³/mol, the magnetochemistry unit; 1 m³/mol =
  10⁶/(4π) cm³/mol). `M_units='muB'`, `'emu/mol'` or `'SI'`.
- Telser-style input: `degrees=True` on angle arguments, `lft.Ligand.from_degrees(...)`.
- The ligand-field constructors pass units through: `lft.crystal_field(Dq=1500.0)` is in whatever
  unit 1500 is in, and the next call's `units=` says which.

```python
import torch
from torchspin import lft

V = lft.crystal_field(Dq=1700.0)                                        # cm^-1
E = lft.ddn.levels(3, B=800.0, C=3200.0, V=V, zeta=200.0, units="cm^-1", relative=True)

# Cu(II), elongated octahedron from bond directions and AOM parameters
lig = [lft.Ligand.isotropic(t, p, 6000.0, 1000.0, degrees=True)
       for t, p in [(90, 0), (90, 90), (90, 180), (90, 270)]]
lig += [lft.Ligand.isotropic(t, 0, 4800.0, 0.0, degrees=True) for t in (0, 180)]
g = lft.ddn.g_tensor(9, V=lft.aom(lig), zeta=830.0, k=(0.8, 0.8, 0.8), units="cm^-1")
g.principal, g.gap                                                       # gap in cm^-1
```

## From a ligand field to an EPR spectrum

`lft.dn2sys` builds a `SpinSystem` (S = 1/2) whose g is the full 3×3 molecular-frame g of the
ground Kramers doublet. Passing the full matrix rather than principal values plus `gFrame` keeps the
autograd graph intact, so ligand-field parameters → g → `pepper` is differentiable end to end.

```python
from torchspin import Experiment, Options
from torchspin.pepper_autograd import pepper_autograd

e_pi = torch.tensor(1000.0, dtype=torch.float64, requires_grad=True)
sys, gt = lft.dn2sys(9, dict(V=lft.aom(...), zeta=830.0, k=(0.8,) * 3), units="cm^-1",
                     Nucs="63Cu", A=[60.0, 60.0, 500.0], lw=[1.0, 0.0], min_gap=1000.0)
B, spec = pepper_autograd(sys, Experiment(mwFreq=9.5, Range=[255, 360]), Options(Method="matrix"))
```

Ligand-field parameters go in as a dict because `A` means Racah A to DDN and the hyperfine tensor to
`SpinSystem`. `min_gap` raises if the doublet is too close to the next state for an effective spin ½.

## Magnetometry

```python
from torchspin.lft import ddnsus, ddnmag, grids

kw = dict(B=820.0, C=3690.0, zeta=480.0, k=(0.85,) * 3, V=lft.crystal_field(950.0, 250.0, 60.0),
          units="cm^-1")
chiT = ddnsus.chi_T(7, T=[2, 10, 50, 300], field=100.0, chi_units="cgs",
                    grid=grids.gauss_hemisphere(6), **kw).powder           # cm^3 K/mol
M = ddnmag.magnetization(7, [1000.0, 5000.0], [2.0], **kw).powder       # muB per ion
```

χ is Telser's M/H at the measuring field (what a SQUID reports). `curry` returns the differential
dM/dH; the two agree in the linear regime. The thermal averages run over every state of the dⁿ
configuration with exact derivatives through the degeneracies M always has (see "Degenerate
levels" below). With orbital terms switched off (k = 0, ζ = 0), `ddnsus` equals `curry` on the
equivalent `SpinSystem` to 1e-6; `torchspin/tests/test_lft_magnetometry.py` checks this.

## Joint fits

```python
from torchspin.lft.fit import FitParameter, JointFit, Bands, Susceptibility, GValues, Spectrum

params = [FitParameter("B", 900, 500, 1200), FitParameter("Dq", 1000, 500, 1600),
          FitParameter("Ds", 200, -800, 800), FitParameter("Dt", 50, -300, 300),
          FitParameter("zeta", 450, 100, 700), FitParameter("k", 0.9, 0.5, 1.0)]

def model(p):                         # named scalar tensors -> lft.ddn keywords (in the fit's units)
    return dict(B=p["B"], C=4.5 * p["B"], zeta=p["zeta"], k=p["k"].expand(3),
                V=lft.crystal_field(p["Dq"], p["Ds"], p["Dt"]))

fit = JointFit(7, params, model, [Bands(states, energies, 80.0),
                                  Susceptibility(T, chiT, 0.02, chi_units="cgs"),
                                  GValues([2.1, 2.3, 4.9], 0.02)], units="cm^-1")
out = fit.fit()
print(out.summary())                  # values, stderr, chi^2 per data set, undetermined combinations
fit.fit(["bands"]).null_directions    # what bands alone cannot determine (here: k, exactly)
```

- Bands are numbered as DDNFIT does: 1-based over all states in ascending order, relative to
  state 1. A degenerate level is fitted through its mean.
- Jacobians are exact: forward-mode AD for bands, g, χ and M. For `Spectrum`, which goes through
  `pepper_autograd` (no forward mode yet), the exact chain rule d(spectrum)/dg · dg/dx is used, with
  a double-backward product through pepper for the six g components. The spectrum amplitude is
  fitted analytically unless `scale=` is given.
- `out.null_directions` lists parameter combinations the data do not determine.
- `lft.ddnfit.fit`, `lft.ddnsus.fit`, `lft.ddnmag.fit` are single-data-type front ends with
  Telser's names. `lft.dpnfit.DpnFit` reproduces DPNFIT, including its parameter-vector rules.

## Plots

`lft.plot` (needs the `plot` extra, matplotlib; imported only when a plot is drawn). Every function
returns `(fig, ax)` (or `(fig, [ax, ax_resid])` with a residual panel) and takes an optional `ax`.
Energy axes default to **cm⁻¹**; pass `units=` for MHz, GHz, K or eV.

| Function | Plots |
|---|---|
| `plot.levels(energies or Solution or levels)` | level diagram, one bar per state, term labels |
| `plot.tanabe_sugano(diagram)` | E/B vs Dq/B (absolute E, Dq when the diagram has B), curves colored by spin, term labels with free-ion parents where needed |
| `plot.zeeman(loop, mw_freq=)` | levels vs field (mT; `field_units='G'`), transitions at the microwave frequency, resonance fields as top ticks |
| `plot.angles(loop, mw_freq=)`, `plot.roadmap(scan, mw_freq)` | levels vs angle; resonance field vs angle |
| `plot.susceptibility(T, model, data=, quantity='chiT'/'chi'/'mu_eff')` | χT, χ or μ_eff vs T, optional data and residual panel |
| `plot.magnetization(field, M, T, x='B'/'B/T')` | M vs B (T by default) or B/T |
| `plot.powder(result, mw_freq=)` | DDNPOW spectrum |
| `plot.fit(joint, outcome)` | data, best-fit curve on a dense grid and exact residuals for a `JointFit` |

Styling (Helvetica/Arial, bold axis labels, 1.2 pt spines, no grid, no overlapping text) is applied
inside a local `rc_context`; the global matplotlib settings are never changed. Labels that cannot
be placed without overlapping are spread with leader lines or dropped.

```python
from torchspin.lft import plot, tanabe_sugano
fig, ax = plot.tanabe_sugano(tanabe_sugano.tanabe_sugano(3, c_over_b=4.5))
```

## Telser's spin-Hamiltonian programs

DDZHQ, DDPOWH, DJDZE and DDPOWJ duplicate physics torchspin already has (`ham`, `levels`,
`resfields`, `pepper`, `salt`, `endorfrq`, multi-electron systems), so they are not ported again.
`lft.telser2torchspin` converts their inputs instead:

```python
from torchspin.lft import telser2torchspin as t2t

conv = t2t.ddpowh2sys(inp)            # a telser-tools DdpowhInput (read by attribute) -> SpinSystem + Experiment
B, spec = pepper(conv.sys, conv.exp)
sys = t2t.spin2sys(s=1.5, g=(2.0, 2.0, 2.0), D=1.2, E=0.1, a=0.01, nucleus="14N", A=(20, 20, 30),
                   units="cm^-1", degrees=True)
dim = t2t.dimer2sys(site1=..., site2=..., J=12.0, Ddip=0.02)   # Telser's +J S1.S2 (EasySpin's sign)
```

- **Inputs** are Telser's: energies in cm⁻¹ by default (`units=`), A, U and Q in MHz, fields in G,
  angles in degrees. Outputs are torchspin's.
- **Mappings:** Telser's cubic a/F/b go to Stevens B4q, and the third-order Zeeman term u to
  `SpinSystem.Ham`. Exchange is Telser's +J S₁·S₂, which is EasySpin's sign; `telser_J` converts
  other conventions. Biquadratic exchange is exact up to a constant (`dimer_energy_offset`).
  `full_tensors=True` passes rotated tensors instead of Euler frames, so angles stay on the
  autograd graph.
- **Features with no torchspin counterpart** raise `NotImplementedError` with `strict=True` (the
  default). With `strict=False` they are dropped, with a warning, and listed in `extras['dropped']`.
  Small gaps are filled in the module: `quadrupole_echo`, `spin_projections`, `ordering`.
- **Parity:** levels for every DDZHQ and DJDZE reference input match to 1e-12 (with telser-tools'
  constants), resonance fields to 1e-7, ENDOR frequencies to 1e-8. Powder spectra match `pepper`/`salt`
  within tolerances calibrated by refining both programs' grids: 0.3–8 %, with the largest for
  looping transitions.
- [lft_gap_analysis.md](lft_gap_analysis.md) lists every Telser spin-family feature as supported,
  converted (with the difference), or a gap, with proposals for the larger gaps: second-order Zeeman
  w, the U (I·S³) term, DDPOWH/DDPOWJ level-dependent strain, zero-field FDMR/ENDOR, and S = 0 powder
  NMR/NQR.

## Degenerate levels

Ligand-field Hamiltonians are full of exact degeneracies: Kramers doublets, cubic levels, axial g.
The stock `torch.linalg.eigh` backward divides by λⱼ − λᵢ and gives NaN or wrong gradients there.
`lft._linalg` provides what the programs need, each with a backward and a forward-mode rule:

- `eigh_gauge`: the derivative of a degenerate eigenvalue is that of its level's mean. That is
  exact for symmetry-protected degeneracies and a symmetric subgradient at an accidental crossing.
- `thermal_average`: ⟨A⟩_T with the Daleckii–Krein derivative, smooth through every degeneracy.
  Thermal averages must **not** use a gauge-zeroing eigh backward (such as
  `torchspin._linalg.eigh`'s), which drops f′(λ) tr(dH A) inside degenerate levels.
- `ground_manifold`: the projector onto the ground doublet, differentiated through the gaps to the
  rest only (exact at the Kramers degeneracy).
- `sqrtm_psd`: a Newton–Schulz square root, so g = (g gᵀ)^½ has exact gradients at axial g.

## Validation

`torchspin/tests/test_lft_*.py` (about 1300 tests, a few minutes with `OMP_NUM_THREADS=1`):

- **Parity with telser-tools** on every reference input of every program: rel 1e-10 (DDNPOW 1e-8,
  the round-off limit of the NumPy reference itself). These tests skip when telser-tools is not
  installed; torchspin never imports it.
- **Copied NumPy matrix builders** (`lft/_engines/`) are bit-identical to the telser-tools originals.
- **Gradients** match central differences (best of several steps) at rel 1e-6 (DDNPOW 1e-5),
  including the degenerate cases; `gradcheck` with forward AD passes.
- **Physics:** 10Dq; the 15B term separation; Landé intervals; Curie law; Brillouin saturation;
  Van Vleck TIP = 8Nμ_B²/10Dq for d³; g = g_e without spin–orbit coupling; DSOXF d¹/d⁹ = DDN;
  Tanabe–Sugano blocks = the DDN determinant spectrum.
- **Against torchspin:** `ddnsus` with k = 0, ζ = 0 equals `curry`; a DDNPOW Cu(II) spectrum
  equals `pepper` via `dn2sys` to 1e-3 of the peak (limited by the igloo grid); the gradient of a
  pepper spectrum with respect to e_π matches central differences.
- **Fits:** the synthetic Co(II) identifiability result reproduces (bands alone leave k exactly
  undetermined; bands + χT + g determine all six parameters); e_π and the tilt angle are recovered
  from a synthetic Cu(II) spectrum.

## Not ported

Fortran-reproduction modes with no physics content (EISPACK-phase-dependent transition
probabilities, Fortran grid and threshold quirks, DDN's finite-difference magnetometry methods,
input readers and report writers) stay in telser-tools. Where a function would need them it raises
`NotImplementedError` with a pointer.
