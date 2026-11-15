# torchspin.lft: ligand-field tools

`torchspin.lft` ports Josh Telser's ligand-field programs (DDN, DDNPOW, DDPOWH, DPN, DSOXF,
Tanabe–Sugano and others) as differentiable PyTorch code. It fits one ligand-field model jointly to
band energies, magnetometry, g values and EPR spectra. The user guide is [lft.md](../lft.md).

**How well it is validated.**

- Closed forms hold: 10Dq, 15B, Landé intervals, the Curie law, and Van Vleck TIP.
- Gradients are exact through degeneracies.
- Every golden input of Telser's programs matches telser-tools to 1e-10.
- That parity runs in CI against frozen telser-tools references, and live where telser-tools is
  installed.

Test files are in `torchspin/tests/`.

## Model

The d^n (or p^n) Hamiltonian in the full $\binom{10}{n}$ determinant basis is

$$
\hat H=\hat H_{ee}(B,C;\alpha)+\zeta\sum_i\hat{\mathbf l}_i\cdot\hat{\mathbf s}_i+\hat V_{\rm LF}
 +\mu_B\mathbf B\cdot(k\hat{\mathbf L}+g_e\hat{\mathbf S})
$$

$\hat V_{\rm LF}$ comes from the angular overlap model ($e_\sigma$, $e_\pi$) or from Ballhausen
crystal-field parameters. `ddnpow`, `ddpowh` and `ddpowj` give powder spectra on Telser's igloo grid,
with exact field-point line shapes.

## Method

- Dense eigh.
- Degenerate levels get level-mean derivatives (`eigh_gauge`).
- Thermal averages use the Daleckii–Krein derivative.
- $g=(gg^T)^{1/2}$ uses a Newton–Schulz square root.
- `ddnpow` uses a Rayleigh–Ritz subspace, with exact gradients since B94 (0.9.30).

## Validation

**Analytical checks:**

- `test_lft_ddn.py::test_d1_octahedral_splitting_is_10Dq`.
- `test_lft_ddn.py::test_d2_free_ion_3P_minus_3F_is_15B`.
- `test_lft_magnetometry.py::test_d3_van_vleck_tip_is_8NmuB2_over_10Dq`.
- `test_lft_magnetometry.py::test_spin_only_limit_equals_curry`.
- `test_lft_ddpowh.py::test_nqr_lines_at_closed_form_frequencies`.

**Gradients:**

- `test_lft_core.py::test_ground_projector_gradcheck_at_kramers_degeneracy`.
- `test_lft_ddnpow.py::test_d7_ritz_gradients_match_full_basis`: < 1e-7.

**Reference programs:**

- Live parity with telser-tools, rel 1e-10 on every golden input:
  - `test_lft_ddpowh.py::test_ddpowh_golden_inputs_match_telser_tools`: 70 DDPOWH and 33 DDPOWJ
    inputs.
  - `test_lft_telser2torchspin.py::test_ddzhq_levels_and_probabilities`.

  Without telser-tools they compare against frozen references
  (`torchspin/tests/data/telser/refs/`); `test_lft_telser_refs.py` checks that the frozen copies
  still match live telser-tools. Only the magnetometry parity is live-only.
- Stored Fortran printout is compared only by `test_lft_dtd.py::test_matches_dtd_fortran_printout`
  and part of DLSD5 (`test_lft_dlsd5.py::test_tetragonal_branches_reproduce_dlsd5_fortran_k`).
- telser-tools "exact mode" is itself a corrected port. The parity therefore tests against corrected
  physics, not against Telser's printed output.

**Consistency:** a DDNPOW Cu(II) spectrum equals `pepper` via `dn2sys` to 1e-3 of the peak.

## Limitations

**Not ported** ([lft.md](../lft.md#not-ported)): the Fortran-reproduction modes, such as
EISPACK-phase-dependent probabilities and DDN's finite-difference magnetometry.

**Igloo grids converge slowly.** Use igloo ≥ 40 ([lft_gap_analysis.md](../lft_gap_analysis.md)).

**Self-consistency only:** the joint fits (`JointFit`, `ddnfit`, `ddnsus`, `ddnmag`) are checked only
by recovering synthetic parameters.
