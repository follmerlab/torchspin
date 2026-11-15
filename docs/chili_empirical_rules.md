# chili's M–pS–pI symmetry basis: derivation and validation

`chili` computes slow-motion spectra in a basis that is pruned by an exact symmetry, the M–pS–pI
(Meirovitch) selection rule, whenever that symmetry holds. This page derives the rule, shows how
torchspin recovers the full spectrum from the pruned basis, and records the evidence: the torchspin
tests, four MATLAB checks against EasySpin, and the investigation that replaced an earlier empirical
rule.

## What torchspin does (since 0.9.38)

- **When the basis is pruned.** `ChiliOptions.MpSymm=None` (the default) prunes whenever there is no
  director tilt: any spin system, with no orienting potential or one with M = 0 terms only, and
  `GridSize=1` or an untilted `SampleFrame`. Under director tilt, or with M ≠ 0 potential terms, the
  plain basis is used.
- **The spectrum does not depend on `MpSymm`.** The pruned basis keeps the pS + pI − M = +1 block.
  torchspin adds the −1 block's response exactly, as the complex-conjugate mirror of the +1 block
  (see [Basis of the rule](#basis-of-the-rule)). `MpSymm=None`, `True` and `False` therefore return
  the plain-basis spectrum, and pruning shrinks the basis about 6×.
- **`highField=True`.** There is no −1 half (pSmin = +1), so nothing is mirrored and the factor is 1.
- **An explicit `MpSymm=True` where the symmetry does not hold** (director tilt, or M ≠ 0 potential
  terms) raises `ValueError`.
- **Lanczos stopping.** A step that passes the continued-fraction test is accepted only if the
  spectrum then moves by less than 100 × `Threshold` over the next interval. A full-length
  tridiagonalization, or an exact breakdown, counts as converged. If the Lanczos does not converge,
  chili warns once (`RuntimeWarning`) and uses the direct solver.

**Deviations from EasySpin, by design** (also in
[KNOWN_LIMITATIONS § Chili](../KNOWN_LIMITATIONS.md#chili--slow-motion-sle-rewritten-2026-09-01)):

- With `MpSymm=True`, EasySpin's general method drops the −1 half and returns twice the plain-basis
  amplitude. torchspin returns the plain-basis spectrum.
- EasySpin's general method prunes under director tilt too, and the result is wrong. torchspin raises
  instead.
- EasySpin's Lanczos accepts the first passing check step. On a near-breakdown that can be far from
  converged, and a non-converged run returns `ones(nPoints)`. torchspin checks one more interval and
  falls back to the direct solver.

## Before 0.9.38: the empirical rule

From 2026-09-02 (CHANGELOG, "Fixed — chili default basis", so also in v0.2.2, the manuscript's
release) to 0.9.37, `MpSymm=None` applied an empirical compatibility rule. The symmetry basis was
used at half amplitude only for one S = 1/2 electron with at most two nuclei, no orienting potential
and a rhombic basis (Mmax > 0 and Kmax > 0). Every other system used the plain basis. The rule was
fitted by matching EasySpin's `fast` builder against its general method.

The investigation below found the following:

1. **The rule follows from an exact symmetry; it is not an empirical fit.** Without director tilt the
   automatic result equalled the converged plain-basis solution to 1e-8 of the maximum.
2. **The plain general method was not a different model.** Its ~6 % difference from the `fast`
   builder (cosine 0.937 at τc = 100 ns, 2048 points) was the Lanczos continued fraction stopping
   early.
3. **The rhombic/axial split had no physical meaning.** It only marked where, in the stored cases,
   the plain-basis Lanczos happened to stop early.
4. **The rule halved `highField=True` spectra** (B114).
5. **An explicit `MpSymm=True` was wrong under director tilt** (B115; cosine 0.726 in Exp. 5b), as
   EasySpin's general method still is.
6. **The plain-basis Lanczos could stop early, silently** (B116).

0.9.38 fixes all three defects (B114–B116).

## Basis of the rule

**Exact block structure.** Without director tilt, the lab-frame Liouvillian conserves pS + pI − M:

- each Hamiltonian term $D^{2}_{m k}(\Omega)^{*}\,F_{k}\,\hat T_{2,m}$ changes the coherence order
  $p=p_S+p_I$ and the Wigner index M by the same $m$;
- rotational diffusion, and a potential with M = 0 in the director frame, do not change M.

This is the selection rule of Meirovitch, Igner, Igner, Moro and Freed, *J. Chem. Phys.* 77, 3915
(1982), Eq. (A47). EasySpin cites it in `chili.m` and `chili_lm*.inc`.

The starting vector is

$$
|v\rangle\propto|L{=}M{=}K{=}0\rangle\otimes\hat S_x,
\qquad
\hat S_x=\tfrac12(\hat S_++\hat S_-)
$$

It therefore has two halves of equal norm: pS = +1 (pS + pI − M = +1) and pS = −1 (pS + pI − M = −1).
They lie in decoupled blocks of $\hat{\mathcal L}$. Write $I_\pm$ for the response of each block to
its own normalized starting vector. Then

$$
I_{\rm plain}(\omega)=\langle v|(\hat{\mathcal L}-i\omega)^{-1}|v\rangle=\tfrac12 I_{+}(\omega)+\tfrac12 I_{-}(\omega)
$$

`MpSymm` keeps only the +1 block and renormalizes the starting vector, so $I_{\rm MpSymm}=I_{+}$.

The pS = −1 block is the negative-frequency coherence. At $\omega\approx+\omega_0$ its contribution
is 2e-10 to 5e-8 of the maximum (Exp. 5a). Hence

$$
I_{\rm plain}=\tfrac12\,I_{\rm MpSymm}\,\big(1+O(10^{-8})\big)
$$

This is the half amplitude of the old rule. The pruning also shrinks the basis about 6× (5616 → 969
functions for the default LLMK and one ¹⁴N).

**The exact mirror term.** The 1e-8 residual above matters for large non-secular terms: it grows to
4.9e-6 for ⁶³Cu with A_z = 600 MHz and to 2.2e-3 for S = 1 with D = 2 GHz at X-band. torchspin
therefore does not use the factor ½. The −1 block is the complex-conjugate mirror of the +1 block, so

$$
I_{\rm plain}(z)=\tfrac12\big[I_{+}(z)+\overline{I_{+}(\bar z)}\big]
$$

with $z$ the complex frequency. The Lanczos needs no second recursion: the same tridiagonalization
is evaluated at the mirrored frequencies. Against the plain basis solved directly, the result agrees
to 5e-16 for a nitroxide, 6e-16 for ⁶³Cu, 5e-11 for S = 1 with D = 100 MHz and 2e-8 with
D = 500 MHz. The last residual is not yet explained; direct and Lanczos solves give the same value.

**EasySpin's `fast` builder computes $I_{\rm plain}$.**

- `chili_startingvector.m` puts unit weight on every |pS| = 1, pI = 0, L = K = M = 0, jK = 1
  function. That is the same $\hat S_x$ content, including the pS = −1 half.
- `Opt.MpSymm` defaults to `false` (`chili.m:672-675`), so `chili_basisbuild.m` and `chili_lm1.inc`
  prune nothing.

"fast = general + MpSymm at half amplitude" is therefore the identity above. It is not a property
peculiar to the C code.

**Why EasySpin's general method appeared to differ.** Exp. 3–4 show three things:

- The Liouvillian is exactly complex symmetric in every basis: ‖L − Lᵀ‖ = 0 (Exp. 3).
- On the plain basis (default LLMK, nitroxide, τc = 100 ns), the pseudo-norm Lanczos reaches a
  near-breakdown at step 40: |β₄₀| = 2.3e-3, against a median of 0.25. The continued-fraction change
  passes the 1e-6 test there, so it stops.
- Continued without the early stop, the same recursion converges to the direct solution by step 200
  (Exp. 4). The symmetry basis (210 steps) and `highField=True` (80 steps) converge normally.

Up to 0.9.37, torchspin's port reproduced EasySpin's general method with this behavior, including its
stopping rule.

## MATLAB checks (EasySpin, 2026-11-14)

Four small jobs (`matlab -singleCompThread -batch`, one at a time) with the generator
`tests/data/generate_chili_mpsymm_refs.m`. The outputs are stored as
`tests/data/ref_chili_mpsymm_{a,b,c,d}.mat`. Settings: the nitroxide below, τc = 100 ns, default
LLMK, 256 points.

- **(a) fast vs general.** EasySpin's `fast` output and its general-method Lanczos agree (cosine
  0.999999). Both are off the general method solved directly (cosine 0.964724, amplitude ratio
  1.0129). The general Lanczos against general direct gives 0.964772, the number torchspin 0.9.37
  gave for its plain basis. **EasySpin's default output for this system is a continued fraction that
  stopped early.** The stored 2048-point example that the old rule was fitted to happened to
  converge.
- **(b) `highField`.** fast vs general direct: cosine 1.000000, ratio 1.00000. The factor is 1, which
  confirms B114.
- **(c) Tight threshold.** fast with `Opt.Threshold` 1e-10 or 1e-12 matches general direct (cosine
  0.999998, ratio 1.00013). torchspin's default (pruned basis with the mirror term) and its plain
  direct solve both match EasySpin's general direct solve at cosine 1.000000, ratio 1.00000.
- **(d) `simple_tc1e-07`.** For the one case of `test_chili_ext_matlab_validation.py` that had been
  loosened (0.997 / 8 %), EasySpin's default fast output is 0.9979 / 1.069 from its own output at
  `Threshold` 1e-10. torchspin matches the converged fast output at cosine 0.9999966. The residual
  is EasySpin's early stop, not a torchspin error, so the comparison with EasySpin's default output
  is now a strict xfail at the default 0.999 / 2 %.

## Tests

- `test_chili_mpsymm.py`: the pruned basis equals the plain basis over several spin systems and
  options; the mirror term is needed and is exact; `highField` amplitude (also against EasySpin, b);
  `MpSymm=True` raises under director tilt or M ≠ 0 potential terms; the automatic basis is plain
  under tilt.
- `test_chili_lanczos.py`: plain-basis Lanczos equals the direct solve; the default equals
  EasySpin's converged output (a, c); a full-length tridiagonalization counts as converged; a
  non-converged Lanczos falls back to the direct solver with a warning; `simple_tc1e-07` matches
  EasySpin's converged fast output (d).
- `test_chili_default_basis.py`: the default matches EasySpin's default at four correlation times
  (cosine > 0.9999), and the plain basis now does too. Before the fix this test asserted
  0.93 < cosine < 0.95 for the plain basis, which encoded the early stop.

Default outputs on the stored EasySpin cases changed by at most 0.06 % in amplitude
(`simplepotential_grid19`). No tolerance was loosened.

## Investigation (torchspin 0.9.31, before the fix)

All runs used torchspin `main` 80c782d (0.9.31) with torch 2.15.0, NumPy 2.5.4 and SciPy 1.18.1, on
CPU:

```
PYTHONPATH=<worktree> OMP_NUM_THREADS=2..4 python expN_*.py <worktree>
```

The scripts are in the session scratch directory (`docs-physics/exp*.py`). Each one is a few lines
of `chili` calls with `ChiliOptions(MpSymm=..., Solver=...)`. The MATLAB checks came later (see above).

The nitroxide used throughout:

- g = [2.0089 2.0058 2.0021], ¹⁴N with A = [16 16 100] MHz, lw = [0, 0.05] mT;
- 9.5 GHz, 328–352 mT, Harmonic 1;
- default LLMK [14 7 2 6], unless stated otherwise.

"Direct" means `Solver='\\'`.

**Exp. 1** (2048 points; reference: stored EasySpin default output `example_nitroxide_tcorr.mat`):

| τc | auto | plain, Lanczos | plain, direct | MpSymm, direct |
|---|---|---|---|---|
| 100 ns | cos 0.999992, amp 1.0000 | cos 0.937137, amp 0.9771 | cos 0.999992, amp 1.0000 | cos 0.999992, amp 2.0000 |
| 10 ns, 1 ns, 0.1 ns | cos 1.000000 | cos 1.000000 | — | — |

At 100 ns, plain direct and MpSymm direct agree to cosine 1 − 2e-16, with an amplitude ratio of
2.000000000165. Plain Lanczos against plain direct gives cosine 0.937502.

**Exp. 2** (256 points, torchspin only). Columns: cosine and amplitude ratio of MpSymm-direct vs
plain-direct; cosine of Lanczos vs direct for each basis; and the automatic result against
plain-direct.

| Case | MpSymm/plain (direct) | Lanczos vs direct: plain / MpSymm | auto vs plain-direct |
|---|---|---|---|
| default LLMK, 100 ns | 1.0000000 / 2.0000 | **0.964772** / 1.000000 | 1.000000 / 1.0000 |
| default LLMK, 10 ns | 1.0000000 / 2.0000 | 1.000000 / 1.000000 | 1.000000 / 1.0000 |
| [10 5 4 4], 100 ns | 1.0000000 / 2.0000 | **0.990881** / 1.000000 | 1.000000 / 0.9999 |
| [12 0 2 2], [12 3 6 2], 100 ns | 1.0000000 / 2.0000 | 1.000000 / 1.000000 | 1.000000 / 1.0000 |
| axial [14 7 0 6] and [14 7 2 0], 100 ns | 1.0000000 / 2.0000 | 1.000000 / 1.000000 | 1.000000 / 0.9998–1.0000 |
| `evenK=False`, `jKmin=-1`, 100 ns | 1.0000000 / 2.0000 | **0.964772** / 1.000000 | 1.000000 / 1.0000 |
| `pImax=[0]`, 100 ns | 1.0000000 / 2.0000 | 1.000000 / 1.000000 | 1.000000 / 1.0000 |
| tilted AFrame [0 0.5 0], 30 ns | 1.0000000 / 2.0000 | 1.000000 / 1.000000 | 1.000000 / 0.9999 |
| no nucleus, g = [2.06 2.03 2.0], 100 ns | 1.0000000 / 2.0000 | 1.000000 / 1.000000 | 1.000000 / 1.0000 |
| ¹⁵N + ¹H, [8 0 2 2], 30 ns | 1.0000000 / 2.0000 | 1.000000 / 1.000000 | 1.000000 / 1.0000 |
| S = 1, D = 100 MHz, [8 0 2 2], 10 ns (outside the rule) | 1.0000000 / 2.0000 | 1.000000 / 1.000000 | 1.000000 / 1.0000 |
| **`highField=True`**, 100 ns | 1.0000000 / **1.0000** | 1.000000 / 1.000000 | 1.000000 / **0.5000** |

**Exp. 3.** ‖L − Lᵀ‖/‖L‖ = 0 for plain and MpSymm bases at LLMK [14 7 2 6], [14 7 0 6], [14 7 2 0],
[10 5 4 4] and [12 0 2 2]. The default Lanczos stops at 40 steps (plain) and 210 (MpSymm) for
[14 7 2 6], and at 60 and 150 for [10 5 4 4].

**Exp. 4** (128 points, Harmonic 0). Plain basis: the default stop is at k = 40, with cosine
0.99938 against direct; k = 100 gives 0.99997; k ≥ 200 gives 1.000000. MpSymm stops at k = 210, and
`highField=True` at k = 80, both with cosine 1.000000.

**Exp. 5a** (LLMK [12 0 2 2], direct): max|I_MpSymm/2 − I_plain| / max|I_plain| =
2.5e-8 / 2.5e-10 (H0 / H1) at 1 ns, 4.7e-8 / 1.8e-10 at 10 ns, and 4.0e-8 / 3.8e-10 at 100 ns.

**Exp. 5b** (¹⁴N, `Potential=[2 0 0 1]`, LLMK [8 0 2 2], 10 ns, direct):

| Grid | MpSymm=True vs plain | auto (plain) vs plain-direct |
|---|---|---|
| `GridSize=1` (no tilt) | cos 1.000000, ratio 2.0000 | cos 1.000000 |
| `GridSize=5` (director tilt) | **cos 0.726262, ratio 5.73** | cos 1.000000 |

**Exp. 6.** Stored cases from `ref_chili_ext.mat`, run through the test module's own harness:

- `fieldsweep_approxlin`, `fieldsweep_explicit`, `fieldsweep_approxinv` and `twomethods_general`:
  cosine 1.000000, amplitude 1.0000. EasySpin ran *plain general* there (`LiouvMethod='general'`)
  while torchspin's automatic rule fired. This is EasySpin-side evidence for the identity, at
  τc = 1 ns.
- `twonuclei_15N_1H`: 1.000000 / 1.0004.
- `simple_tc3.16e-08`: 0.999998 / 0.9996.
- `simple_tc1e-07`: 0.997983 / 1.0664.

**Exp. 7** (`simple_tc1e-07`). Every torchspin route agrees with every other: auto, plain Lanczos,
MpSymm Lanczos at thresholds 1e-6 and 1e-10, and MpSymm direct. Auto against MpSymm-direct/2 gives
cosine 1 − 6e-8 and amplitude 1.0003. All of them are 0.99798 / 1.066 from EasySpin's stored `fast`
output. The residual therefore lies between EasySpin's fast output and the converged general model,
not in torchspin's basis choice. MATLAB check (d) later confirmed the cause: the fast builder's
own Lanczos stop.

**Exp. 8** (S = 1 and ¹⁴N + 2 ¹H at 100 ns, rhombic [10 5 2 4]). These systems are outside the rule,
so the automatic result is plain Lanczos. S = 1: Lanczos vs direct cosine 1.000000, amplitude 1.0007.
The three-nucleus direct solves were stopped unfinished (see the end of the Scope section).

## Open items

- **Two nuclei in MATLAB.** A code comment records fast against general + MpSymm with an amplitude
  ratio of 2.0024, 0.12 % beyond the identity. Unexplained; possibly another early stop.
- **The S = 1 mirror residual** of 2e-8 at D = 500 MHz is unexplained.
- **`Solver='E'`** (eigenvalue solver) depends on the LAPACK build for ill-conditioned cases, so the
  tests do not use it as an exact reference.
- **No converged-basis test**: every test runs at the LLMK chosen for it.

## Implementation claims vs the manuscript

The manuscript (v0.2.2) describes the chili comparison as EasySpin's "specialized fast Liouvillian
construction" against TorchSpin's "general Liouvillian pathway". That is true of the builder, but
incomplete: v0.2.2 already applied the automatic rule to that system, as did every release up to
0.9.37. torchspin's default therefore used the symmetry basis at half amplitude. By the identity
above, that is the converged general-method answer to 1e-8.

The manuscript's NRMSE of 7.23e-5 (τc = 3 ns) is in the regime where the plain Lanczos also
converges (Exp. 1/2 at 1–10 ns), so the numbers are unaffected. Findings 4 and 5 lie outside every
case the manuscript or the test suite reports.
