# Telser's spin-Hamiltonian programs vs torchspin: gap analysis (`torchspin.lft` M5)

Telser's spin family is the shared spin core and four programs, ported to Python in telser-tools:

- DDZHQ: levels and EPR/NMR transitions at one field.
- DDPOWH: powder EPR, FDMR and ENDOR.
- DJDZE: levels and EPR transitions of an exchange-coupled dimer.
- DDPOWJ: powder EPR of a dimer.

These programs mostly duplicate physics that torchspin already has (`ham`, `levels`, `resfields`,
`pepper`, `salt`, `endorfrq`, multi-electron `SpinSystem`s with `ee`/`ee2`). M5 therefore does not port
them again. It adds converters from Telser's conventions to torchspin objects
(`torchspin/lft/telser2torchspin.py`), parity tests against telser-tools
(`torchspin/tests/test_lft_telser2torchspin.py`), and this table.

Status legend:

- **as is**: torchspin has it with the same meaning; the converter only changes units and layout.
- **converter**: mapped by `telser2torchspin`, with the stated difference or condition.
- **gap**: no torchspin counterpart. Small, clear gaps are filled in `telser2torchspin` (marked
  *filled*); larger ones are proposals, with what a new function would need.
- **not needed**: a Fortran artifact, an I/O feature, or a numerical shortcut that has no physics to
  carry over.
- **filled (lft)**: not in torchspin's core (`pepper`, `resfields`, ...), but available in `lft.ddpowh` /
  `lft.ddpowj`, the field-point ports of DDPOWH/DDPOWJ on a `SpinSystem`. The converters'
  `target='lft'` maps it. The core change, where one is proposed, is still open.

The "exact mode" of telser-tools (the corrected physics, `mode="exact"`) is the reference throughout.
The Fortran quirks it documents (findings DDZHQ-01 … DDPOWJ-12) are not ported and not mapped; see
`telser-tools/docs/FINDINGS.md`.

## How the parity was measured

| quantity | inputs | agreement |
|---|---|---|
| DDZHQ levels at every field cycle | all 31 golden inputs (w and U set to 0 in both programs for the 5 inputs that use them) | rel 1e-12 with telser-tools' constants passed through `constants=` (observed ≤ 3e-15); 5e-9 with torchspin's CODATA 2022 μ_B |
| DDZHQ B₁-averaged probabilities, all pairs | same | 1e-9 of the largest, summed over degenerate levels |
| DJDZE levels (with the biquadratic constant) and probabilities | all 50 golden inputs | rel 1e-12 / 1e-9 |
| Resonance fields, intensities (`resfields`) | S = ½ + ¹⁴N, ⁶³Cu (rotated A, Q); S = 3/2 (D ≫ hν, u, ¹H); S = 5/2 (B4q, u) | fields rel 1e-7 (resfields' convergence), intensities rel 1e-6 |
| ENDOR frequencies and intensities (`endorfrq`) | 3 DDZHQ inputs | 1e-8 |
| DJDZE spin magnitudes (`spin_projections`) | every DJDZE input that prints them | 1e-9, summed over degenerate levels |
| DDPOWH/DDPOWJ powder spectra (`pepper`, `salt`) | 17 comparisons | max abs. difference after normalizing both spectra to their maximum; tolerances below |
| DDPOWH/DDPOWJ powder spectra (`lft.ddpowh`, `lft.ddpowj`, `target='lft'`) | all 70 DDPOWH and 33 DDPOWJ golden inputs (EPR, FDMR, ENDOR, NMR/NQR; strain, Lorentzian, w, U, mixed modes), plus option switches | 1e-10 of the maximum, unnormalized (observed ≤ 4.5e-11) |

The powder tolerances do not test the converter, which is exact (the levels and probabilities above
show that). They measure how differently the two programs integrate the same physics. Each tolerance
was calibrated by refining both grids:

- **Orientation grid.** Telser averages over the igloo grid with solid-angle weights. Pepper
  interpolates and projects. For narrow lines Telser's average converges slowly: S = 1, D = 0.2 cm⁻¹,
  W = 30 MHz is 44 %, 16 % and 8 % from pepper at igloo 20, 40 and 80. The tests therefore use igloo
  ≥ 40 and line widths of at least ~5 field steps, at which telser-tools is self-converged (igloo N
  vs 2N) to ≤ 0.3 %.
- **Field step.** Telser evaluates the frequency-domain line at each field point. Pepper puts stick
  intensities into bins and convolves. The difference scales as (field step/FWHM)². Cu(II) with 6 G
  steps differs by 2.6 % at a 14 G FWHM and by 0.5 % at 28 G.
- **Line shape in field.** Telser's field-swept line is the frequency line mapped through the exact
  ν_ij(B). Pepper linearizes ν_ij(B) at the resonance field (HStrain/(dν/dB)). The two differ where
  the levels curve within a line width: 1.6 % for S = 1, D = 0.3 cm⁻¹ at Q band with W = 200 MHz, and
  2.5 % in the derivative of a rhombic S = ½ pattern.
- **Looping transitions.** With D ≈ hν (S = 1, D = 0.2–0.3 cm⁻¹ at X band) some transitions turn
  around in field (dν/dB = 0). Pepper's field width W/|dν/dB| diverges there; Telser's field-point line
  does not. The difference is 4.5–5.6 % with both grids refined. The same mapping with D ≪ hν agrees
  to 0.6 %. We attribute the difference to pepper's linearization; it was not traced further.
- **Moment.** Telser's EPR moment is g·S. torchspin's is −dH/dB, which also contains the third-order
  Zeeman u term and the nuclear moment. With u = 0.05 the powder pattern differs by a few %; see the
  `u` row.

The tolerances are 1.5–2× the calibrated difference:

- 0.3 % for rhombic S = ½ and the Cu(II)₂ triplet;
- 1 % for Cu(II), S = 1 with D ≪ hν, and the Cu(II)₂ triplet with populations;
- 2.5–4 % for derivatives, S = 1 at 2 K, and a general d vector;
- 7–8 % for S = 3/2 with u and for S = 1 with D ≈ hν.

## Spin core: Hamiltonian (`telser_tools.spin`, DDZHQ and DDPOWH)

| Telser feature | status | torchspin / converter | notes |
|---|---|---|---|
| g = (gx, gy, gz), g frame = molecular frame | as is | `SpinSystem.g` | |
| (D/3)O₂⁰ + E O₂² (cm⁻¹) | as is | `D = [−D/3+E, −D/3−E, 2D/3]` (MHz) | |
| Stevens B40 O₄⁰ + B42 O₄² + B44 O₄⁴ | as is | `B = [None, [[B44,0,B42,0,B40,0,0,0,0]]]` | Telser's O_k^q equal `stev` (telser-tools test) |
| Telser's a, F, b (cubic, axial, rhombic 4th order) | converter | `abf_to_stevens`: B40 = a/120 + F/180, B42 = b/3, B44 = a/24 | checked: S = 5/2 Γ₇ at −2a, Γ₈ at +a |
| "Bnm" input form with the Fortran's 5·B44 (DDZHQ-06) | not needed | exact mode reads B44 as the Stevens coefficient | |
| third-order Zeeman μ_B Σ u_k B_k C_k, C_k = S_k³ − (3X−1)/5 S_k | converter | `third_order_zeeman_ham` → `SpinSystem.Ham['132','134']` | Projected numerically through `ham()`'s (H0, μ) path, exact to 1e-15. Two differences: (1) the Ham coefficients are detached inside `ham_ezho` (no gradient w.r.t. u); (2) torchspin's transition moment includes u (−dH/dB), Telser's EPRTRANS does not (relation tested). Bug B11 (fixed in 0.6.7). |
| second-order Zeeman w(μ_B B·S)² | **gap**; filled (lft) | `lft.ddpowh(..., zeeman2=second_order_zeeman_operator(sys, w))` | `ham_ezho` has lB = 2 terms, but `levels`, `resfields` and `pepper` use only H0 − B·μ, so the term would be silently dropped. The converter raises on the pepper path (`strict=False`: drops with a warning). `lft.ddpowh` evaluates H(B) = H0 − B n·μ + B² n·Z2·n at every field point (parity with DDPOWH s52_pop_w_u at 1e-13). **Core proposal (item C1):** a field-quadratic operator carried with (H0, μ); `resfields` would need bracketing plus Newton, as in telser-tools `spin.resonance_fields`. |
| A principal values + McGavin Euler angles | as is | `A`, `AFrame` (same angles, radians), or full R diag Rᵀ (`full_tensors=True`) | Frames leave the autograd graph (B3); full tensors keep it |
| U Σ I_k C_k(S) (I·S³ hyperfine, S ≥ 3/2) | **gap**; filled (lft) | `lft.ddpowh(..., h0_extra=u_term_operator(sys, U))` | No SpinSystem slot. `lft.ddpowh` adds the operator to H0, keeping its graph (parity with DDPOWH s72_I12_U at 1e-13). **Core proposal (item C2):** a generic extra-operator term (H0 += Σ coeff·op) in `ham`. |
| Q principal values + Euler angles (traceless, MHz) | as is | `Q`, `QFrame` | |
| quadrupole echo of DDZHQ (P∥ = 3/2 Q_zz, η, e²qQ/h) | gap, *filled* | `quadrupole_echo(Q, I)` | Standard ordering \|Q_zz\| ≥ \|Q_yy\| ≥ \|Q_xx\| (the Fortran's η ≤ 0 is finding DDZHQ-08); `[e2qQ, eta]` is SpinSystem's two-value Q form (round trip tested) |
| g_N from Telser's (corrected) nuclide table | converter | `Nucs = [isotope]`, `gnscale = g_N/nucgval` | Unknown labels get a stand-in isotope with the same I (`isotope_for`). The tables differ for some metals (B10). DDPOWH inputs carry the original table's g_N; the DDPOWH converters apply the corrections of findings DNTABLE-01/-02 (¹⁴⁹Sm sign, ²H digits), as telser-tools' DDPOWH exact mode does (its DDZHQ reader keeps the original table). |
| −g_N μ_N B·I | as is | nuclear Zeeman (same sign) | |
| field direction (θ, φ) in the g frame | as is | `levels(sys, phi, theta, B)`, `resfields(..., phi, theta)` | |
| S up to 9/2, I up to 9/2 | as is | | |
| constants (CODATA 2018 / Telser's BEMHZ = 1.3996) | converter | `constants=` rescales g, u, g_N and cm⁻¹ → MHz | For parity tests. Without it, torchspin's CODATA 2022 μ_B is used (1.4e-9 apart). |
| S = 0 (nuclear spin only) | converter | `SpinSystem(S=[])` with Q and nuclear Zeeman | levels via `ham`; spectra: see DDPOWH NMR/NQR |

## Spin core: transitions

| Telser feature | status | torchspin | notes |
|---|---|---|---|
| B₁-averaged perpendicular P = (\|⟨i\|e₁·M\|j⟩\|² + \|⟨i\|e₂·M\|j⟩\|²)/2 | as is | rate (Σ\|μ_k\|² − \|n·μ\|²)/2 = μ_B² P | Telser leaves out the nuclear moment by default (≈1e-3 for ¹H) and the u term (see above) |
| parallel mode (H1 mode < 0) | as is | `Experiment(Mode='parallel')` | pepper vs DDPOWH tested |
| mixed mode P⊥ + r P∥ (H1 mode r > 0) | converter (relation); filled (lft) | two runs, perpendicular + r × parallel; `lft.ddpowh(h1mode=r)` | Converter refuses the mixed mode itself on the pepper path (strict); the relation is tested. `lft.ddpowh` takes Telser's H1 mode directly. |
| zero-field isotropic average (1/3)Σ\|M_k\|² | gap (trivial) | from `ham` operators | Used in the tests; not worth a function |
| EPRTRANS/NMRTRANS (moduli, ρ loop, H1MODE = gav; DDZHQ-01…03, DDPOWH-01) | not needed | — | Fortran defects |
| NMR probability (operator I, no g_N) | as is | `endorfrq` intensity = (g_N μ_N)² P | tested |
| hyperfine enhancement option (I − (μ_B/g_N μ_N) g·S) | gap | — | `endorfrq` has no enhancement option (Telser and EasySpin default to none). `lft.ddpowh.endor` follows Telser's exact mode (operator I, no enhancement). Core proposal: item C3. |
| population weights, 100 MHz microwave Lorentzian, listing rules (P ≥ 0.1, mS manifolds, K numbering) | not needed | levels/vectors from `ham` | DDZHQ/DJDZE report formatting; recomputed in a few lines when needed |
| resonance fields | as is | `resfields` | intensity = μ_B² P/(2I+1)/\|dE/dB\| (tested) |

## DDPOWH: powder EPR, FDMR, ENDOR

| Telser feature | status | torchspin | notes |
|---|---|---|---|
| field-swept EPR (TRTYP E), G or T | converter | `pepper`, `epr_experiment` (Range in mT, npts + 1 points) | field axes equal to 1e-12 (tested) |
| Gaussian frequency-domain half width W(n) = √Σ W_k² n_k² (HWHM, MHz) | as is | `HStrain = 2W` | |
| Lorentzian frequency-domain width | **gap**; filled (lft) | `lft.ddpowh(shape='lorentzian')` | `HStrain` is Gaussian; `lw` Lorentzian is in field units. For FDMR an isotropic W maps to `lw = [0, 2W]` (frequency sweep). The converter raises on the pepper path. **Core proposal (item C6):** a Lorentzian component of `HStrain` (EasySpin has none either). |
| line cutoff at ICUTOFF half widths (Fortran: √ICUTOFF, DDPOWH-05) | not needed; filled (lft) | pepper convolves fully; `lft.ddpowh(cutoff=, exact_cutoff=)` | A cut Lorentzian loses area: (2/π)·atan(ICUTOFF) is kept. `exact_cutoff=False` gives the Fortran's √ICUTOFF. ENDOR has separate EPR and NMR cutoffs (`nmr_cutoff=`). |
| exact derivative d/dB (Fortran: d/dν, DDPOWH-02) | as is | `Harmonic = 1` | tested |
| Flygare factor ν (Gaussian) or ν³/ν₀² (Lorentzian) | converter (relation); filled (lft) | none in EasySpin; `lft.ddpowh(frequency_factor=True)` (default) | Compare pepper with telser-tools `frequency_factor=False`. Tested: y_ff = hν_mw·y to O(W/ν). For a THz FDMR axis Telser divides ν³ (cm⁻¹) by the axis in THz²; the converter folds that constant into `scale`. |
| strain: W (H/Hset)^b (1 + ½[√Σ\|c_bi\|²w_b + √Σ\|c_bj\|²w_b]), w_b = WMS(mS) + WMI(mI) | **filled (lft)** | `lft.ddpowh.epr(..., strain=Strain(strain_weights(sys, ms=, mi=), b, B_set))` | Per-basis-state weights, so any SpinSystem works (DDPOWH sum form; DDPOWJ product form below). Also in the ENDOR selection (`field_strain=`). Exact identities tested (uniform weights = width × (1 + √c); level selectivity at D ≫ hν); gradients in the weights, b, B_set. pepper has no width that depends on level composition; a pepper-side version was not built (the field-point port also gives exact line shapes where pepper linearizes). |
| Boltzmann populations (normalized, exact mode; Fortran unnormalized: DDPOWH-04) | as is | `Experiment.Temperature` | tested (S = 1, 2 K, Q band) |
| FDMR (TRTYP F), cm⁻¹ or THz | converter; zero field filled (lft) | `pepper` frequency sweep (`Field`, `mwRange` in GHz); `lft.ddpowh.fdmr(sys, axis, field=0.0)` | Zero-field FDMR: torchspin needs Field > 0, so the pepper path raises (`strict=False` uses 1e-6 mT). `lft.ddpowh.fdmr` takes a true zero field (core proposal: item C4). |
| ENDOR (TRTYP N), H or g observation, centered axis | converter; exact (lft) | `salt(Field, ExciteWidth≈0, HStrain = 2W; lw_mhz = 2w)`; `lft.ddpowh.endor` | Telser's ENDOR intensity also has the ν_NMR (Flygare) factor, NMR-pair populations p_i − p_j and an anisotropic NMR width (salt takes one lw; the x value is used). salt labels bins by their lower edge (half-bin shift). Tested at 0.6 %. `lft.ddpowh.endor` has all of these (parity 1e-10, including zero field). |
| ENDOR selection Σ P_epr f(ν_ij − ν_mw) over all EPR pairs | as is | salt's orientation selection | same model |
| NMR/NQR (S = 0), shielding σ in kHz | **filled (lft)** | `lft.ddpowh.nmr(SpinSystem(S=[], Nucs=..., Q=...), rf, field, shielding_khz=)` | Grid × eigh × \|⟨i\|I\|j⟩\|² × line on `SpinSystem(S=[])`, Telser's DDPOWH NMR mode. σ_k (kHz) shifts the Larmor frequency for the field along k: the nuclear Zeeman moment along k is scaled by 1 − σ_k/\|ν_N\|. NQR closed form, shielded line position and parity (NQR, NMR, shielding: ≤ 1e-11) tested. `SpinSystem.sigma` is not used. |
| igloo grid, octant/hemisphere (NROT) (Fortran φ windows: DDPOWH-03) | not needed | pepper's grid and symmetry (`hamsymm`) | Set `GridSymmetry` explicitly when differentiating (B1) |
| angle ranges θ/φ (non-powder) | **gap**; filled (lft) | `lft.ddpowh(angle_range=(θmin, θmax, φmin, φmax))` | pepper has crystal orientations but no restricted powder grid. **Core proposal (item C7):** user-supplied orientation list + weights in pepper. |
| partial orientation ORFACT = exp(λx cos²φ + λz cos²θ) | converter; filled (lft) | `Experiment.Ordering = ordering(lam_x, lam_z)`; `lft.ddpowh(ordering=(lam_x, lam_z))` | Field at θ = β, φ = π − γ in EasySpin's Euler angles. Weight function tested; DDPOWH applies ORFACT only on a restricted (non-powder) grid, which `lft.ddpowh` reproduces (parity: partial_orient, endor_s1_partial). |
| ⟨g₁²⟩ normalization | not needed | | constant divisor |
| mS manifold ranges (pair selection) | **gap**; filled (lft) | `lft.ddpowh(pairs=manifold_pairs((a, b, c, d), 2I + 1))` | pepper has no transition selection (EasySpin's `Opt.Transitions` is not ported; core proposal: item C5). Default ranges select all pairs. |
| H1 mode, NRHO | converter | `Mode` | NRHO only matters for EPRTRANS |
| WinEPR/Bruker output, plot files | not needed | | |

## Dimer: DJDZE and DDPOWJ (`telser_tools.dimer`)

| Telser feature | status | torchspin / converter | notes |
|---|---|---|---|
| isotropic exchange **+J S₁·S₂** (J > 0 antiferromagnetic) | as is | `ee = J·1` (MHz) | Same sign as EasySpin. `telser_J(J, '-2J')` converts H = −2J′S₁·S₂ (J = −2J′), and also '−J', '+2J'. Tested: triplet J above the singlet. |
| biquadratic j[(S₁·S₂)² − X₁X₂/3] | converter | `ee2 = j`, levels − j X₁X₂/3 (`dimer_energy_offset`) | Exact up to the constant; Fortran's imaginary cross term (DJDZE-02) not mapped |
| quadrupolar 24 JQ T₂(S₁)·T₂(S₂) (exact form) | converter | `ee2 += 24 JQ`, `ee += 12 JQ`, offset −8 JQ X₁X₂ | identity T₂·T₂ = (S₁·S₂)² + ½S₁·S₂ − X₁X₂/3; the Fortran's non-invariant form (DJDZE-03) not mapped |
| dipolar Ddip, Edip + Euler angles | converter | full 3×3 `ee` (graph-safe rotation) | |
| antisymmetric exchange d·(S₁ × S₂) ("JA") | converter | antisymmetric part of `ee` | closed form tested |
| site g, D principal values; site-2 Euler angles (shared in DJDZE, separate in DDPOWJ) | as is | `gFrame`/`DFrame` or full tensors | |
| site third-order Zeeman μ (S ≥ 3/2) | converter | `Ham` row per electron | as u above |
| site 4th order (molecular frame only) | as is | `B` rows per electron | rotated site with B4q raises, as in Telser |
| S = 0 site | converter | one-electron SpinSystem | |
| spin magnitudes ⟨S_i⟩·n per level (Fortran Σ⟨S_k⟩n_k²: DJDZE-07) | gap, *filled* | `spin_projections(sys, B, n)` | tested against all DJDZE inputs that print them |
| NSMAX, population cutoff, listing, g_obs, microwave weighting | not needed | | report |
| DDPOWJ powder spectrum | converter; exact (lft) | `pepper` with `Options(Threshold=0)`; `lft.ddpowj.epr` | default pruning drops weak singlet–triplet lines (6 %) |
| DDPOWJ strain: basis weights WMS₁(m₁)·WMS₂(m₂) (product), (H/Hset)^b | **filled (lft)** | `lft.ddpowj.epr(..., strain=Strain(ddpowj.strain_weights(sys, ms1, ms2, s12_weight=0.0), b, B_set))` | Telser's product form. **Open question DDPOWJ-10 (waiting on J. Telser):** an S = ½ partner has no WMS factors (WMS = 0), so every product weight is 0 and only (H/Hset)^b remains; the Fortran sets WMS = 1, which doubles every width. `s12_weight` (converter: `s12_strain_weight`) keeps both: 0.0 (default, exact) and 1.0 (Fortran); both match telser-tools' option of the same name. Treat strain fits of dimers with an S = ½ partner as provisional. |
| DDPOWJ level ranges (all pairs i < j by default) | gap; filled (lft) | `lft.ddpowj.level_range_pairs((a, b, c, d), n)` | as DDPOWH |
| DDPOWJ Lorentzian, cutoff, Flygare, populations, octant (DDPOWJ-06 … -11) | as DDPOWH; filled (lft) | `lft.ddpowj.epr` | Octant with general d (DDPOWJ-11): pepper picks the symmetry itself; `lft.ddpowj` takes Telser's NROT from the input. |

## Proposals, by size

1. **Small, done in M5:** `quadrupole_echo`, `spin_projections`, the J-convention helper, the
   third-order Zeeman `Ham` map, the ORFACT ordering function.
2. **Large, done (lft only):** Telser's level-strain width (DDPOWH, DDPOWJ). `lft.ddpowh` / `lft.ddpowj`
   are field-point ports of `telser_tools_diff.ddpowh`/`ddpowj` on a torchspin `SpinSystem`. They match
   telser-tools at 1e-10 on every golden input and are differentiable (reverse and forward mode). The same
   ports also cover the Lorentzian width, m_S-manifold pair selection, the Flygare factor and cutoff,
   restricted angle ranges, mixed H1 modes, zero-field FDMR/ENDOR, and w and U for these programs. They
   give exact field-swept line shapes where pepper's linearization shows (strongly curved levels, broad
   lines).
3. **Medium, done (lft only):** the powder NMR/NQR function for S = 0 (`lft.ddpowh.nmr`).
4. **Core changes, opt-in, one branch each off main (item C; API proposals first):**
   - the field-quadratic w term in `levels`/`resfields`/`pepper` (C1);
   - an extra-operator hook in `ham` for U (I·S³) (C2);
   - a hyperfine-enhancement option in `endorfrq` (C3);
   - Field = 0 in frequency sweeps (C4);
   - a transition (level-range) selection in `pepper`, like EasySpin's `Opt.Transitions` (C5);
   - a Lorentzian `HStrain` (C6);
   - restricted orientation grids in `pepper` (C7).

## torchspin defects found (logged as new entries in the mol2g bug log)

- **B11 (fixed in torchspin 0.6.7).** `ham(sys)` added the linear higher-order Zeeman terms
  (`Ham['1..']`) to μ with the sign opposite to `ham(sys, B0)` (it added G1 to μ, and H = H0 − B·μ).
  `levels`, `resfields` and `pepper` use the (H0, μ) path. 0.6.7 changed that path to −G1, so both
  paths now agree, and `test_ham_full_field_path_agrees_with_moment_path_for_ham1xx` checks it. The
  converter calibrates its `Ham` rows on the (H0, μ) path at run time, so it followed the fix
  without code changes.
