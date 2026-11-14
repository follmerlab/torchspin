# chili's basis-symmetry and amplitude rules: basis, scope, validation

This page investigates the two "empirical" compatibility rules in torchspin's `chili`. Both are
applied when `ChiliOptions.MpSymm` is left at `None`:

- **Symmetry-selection rule** (`torchspin/chili_sle.py:1038-1046`). The M–pS–pI (Meirovitch) symmetry
  basis is used when all of the following hold:
  - one electron with S = 1/2,
  - at most two nuclei,
  - no orienting potential,
  - a rhombic basis (`LLMK` with Mmax > 0 and Kmax > 0).

  For any other system, the plain general basis is used.
- **Amplitude rule** (`chili_sle.py:1187`). When the symmetry basis was selected automatically, the
  spectrum is divided by 2.

The rules were introduced on 2026-09-02 (CHANGELOG, "Fixed — chili default basis"), and are therefore
also in v0.2.2, the manuscript's release. Their stated basis was numerical: EasySpin's compiled
`fast` builder, its default for these systems, was matched against its general method. The results
were "general + MpSymm at half amplitude" for a rhombic basis and "plain general" for an axial one
([KNOWN_LIMITATIONS § Chili](../KNOWN_LIMITATIONS.md#chili--slow-motion-sle-rewritten-2026-09-01)).

## Findings

1. **Both rules follow from an exact symmetry and the starting vector's normalization. They are not
   empirical fits.** For a system without director tilt, the automatic result equals the converged
   solution of the plain general basis to 1e-8 of the maximum. No EasySpin input is needed for that.
2. **The plain general method is not a different model.** Its ~6 % difference from the `fast` builder
   (cosine 0.937 at τc = 100 ns) comes from the Lanczos continued fraction stopping early. Solved
   directly, the plain basis reproduces EasySpin's default output exactly as well as the automatic
   rule does: cosine 0.999992 and amplitude 1.0000.
3. **The rhombic/axial split has no physical meaning.** The identity holds equally for axial bases.
   The split only marks where, in the stored cases, the plain-basis Lanczos happened to fail.
4. **Defect, being fixed: the amplitude rule is wrong with `highField=True`.** Nothing is removed there,
   so the factor should be 1. torchspin returns half the amplitude of the plain general solution.
   No stored case covers this.
5. **Defect, being fixed, shared with EasySpin's general method: an explicit `MpSymm=True` is wrong
   with director tilt** (an orienting potential on a powder grid): cosine 0.726. The automatic rule never
   selects it there.

These two defects, and the early Lanczos stop in finding 2, are **being fixed (B114–B116, branch
`fix/chili-mpsymm`)**. This page records the evidence for them. The coordinator could not reproduce
the early stop on a nitroxide with LLMK [10 7 6 2]. The reproducing case here uses the default
LLMK [14 7 2 6] at τc = 100 ns (Exp. 1, 3, 4), and [10 5 4 4] (Exp. 2).

## Basis of the rules

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

This is the half amplitude. The pruning also shrinks the basis about 6× (5616 → 969 functions for
the default LLMK and one ¹⁴N).

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

torchspin's port reproduces EasySpin's general method with this behavior, including its stopping
rule.

## Experiments

All runs used torchspin `main` 80c782d (0.9.31) with torch 2.15.0, NumPy 2.5.4 and SciPy 1.18.1, on
CPU:

```
PYTHONPATH=<worktree> OMP_NUM_THREADS=2..4 python expN_*.py <worktree>
```

The scripts are in the session scratch directory (`docs-physics/exp*.py`). Each one is a few lines
of `chili` calls with `ChiliOptions(MpSymm=..., Solver=...)`. MATLAB was not run.

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
not in torchspin's basis choice. Its cause is open; one candidate is the fast builder's own Lanczos
stop.

**Exp. 8** (S = 1 and ¹⁴N + 2 ¹H at 100 ns, rhombic [10 5 2 4]). These systems are outside the rule,
so the automatic result is plain Lanczos. S = 1: Lanczos vs direct cosine 1.000000, amplitude 1.0007.
The three-nucleus direct solves were stopped unfinished (see the end of the Scope section).

## Scope

**Verified against EasySpin (stored cases).** The rule fires in these cases:

- the 4 τc of `test_chili_default_basis.py`;
- the 4 cases of `test_chili_matlab_validation.py`;
- 33 cases of `test_chili_ext_matlab_validation.py`:
  - `simple_*` ×5, `rhombicdiff_*` ×3, `appfield_14N`, `fullA_*` ×2, `fullg_*` ×2, `freqsweep_14N`;
  - `frqdep_*` ×4, `mwphase_*` ×2, `temperature_*` ×3, `magnetictilt_*` ×2;
  - `twocomp_*` ×3, `twonuclei_15N_1H`, `postconv_*` ×2, `largebasis_*` ×2;
- plus the 4 `LiouvMethod='general'` cases of Exp. 6.

Most of these cases use the default LLMK, at τc ≤ 100 ns. All are field or frequency sweeps of
S = 1/2 with 0–2 nuclei. `highField` is false and no potential is present in any of them.

**Verified by the identity (torchspin only, Exp. 2).** The identity was checked on:

- other LLMK, including odd L, Mmax > Kmax and axial bases;
- `evenK=False` and `jKmin=-1`;
- `pImax`;
- tilted tensor frames;
- no nucleus, and two nuclei.

**Outside the scope.**

- **`highField=True`:** the factor should be 1, but the code divides by 2 (finding 4).
- **S > 1/2, more than 2 nuclei, potentials:** the rule does not fire, and the plain general method
  is used, as in EasySpin, whose `fast` builder rejects these systems.
  - Without director tilt the symmetry still holds: S = 1 in Exp. 2, and a potential at
    `GridSize=1` in Exp. 5b.
  - Their default results depend on the plain-basis Lanczos converging, and no test checks that
    (see Exp. 8).
- **Director tilt (potential with `GridSize > 1`):** the symmetry is not exact. An explicit
  `MpSymm=True` is wrong there (finding 5).
- **EasySpin's old-style potential:** EasySpin's `fast` builder also accepts it. torchspin has no
  old-style-potential input, so this case does not arise.
- **Two nuclei in MATLAB:** the code comment records fast against general + MpSymm with an amplitude
  ratio of 2.0024. That is 0.12 % beyond the identity, unexplained.

Exp. 8: for S = 1 (D = 100 MHz, E = 20 MHz, τc = 100 ns, LLMK [10 5 2 4]), the automatic plain Lanczos result equals the direct solve (cosine 1.000000, amplitude 1.0007). The three-nucleus cases (¹⁴N + 2 ¹H) at [10 5 2 4] and [6 0 2 2] did not finish their direct solves within about 30 minutes and were stopped. Out-of-scope convergence remains untested for more than two nuclei.

## Recommendation

**Derivation.** No further derivation is needed for torchspin's side. The identity
$I_{\rm plain}=\tfrac12 I_{\rm MpSymm}$ (no director tilt, pSmin = −1) is exact up to the
negative-frequency term, and is confirmed numerically. What remains empirical is EasySpin's side: that
the `fast` builder computes the converged $I_{\rm plain}$. That needs the MATLAB checks below. It
should not be built into the code as an assumption.

**Proposed code direction (input to the B114–B116 fix; not implemented on this branch).** Replace the empirical condition with the derived one:

- Apply the M–pS–pI pruning whenever there is no director tilt, for any spin system.
- Scale by 1/2 only when pSmin = −1. With `highField=True` the factor is 1.
- Ignore or reject `MpSymm=True` at tilted orientations.

This removes the rhombic/axial distinction, fixes finding 4, and shrinks the basis about 6× for every
untilted system. Separately, make the Lanczos stopping rule robust: require the convergence test on
two consecutive intervals, or flag a near-breakdown |β| and fall back to the direct solver for small
bases. Otherwise `MpSymm=False` and the out-of-scope systems can silently return a truncated
continued fraction.

**Proposed tests (not written).**

1. Identity: auto equals `MpSymm=False, Solver='\\'` (amplitude 1e-6, cosine 1 − 1e-9). Run it over
   default, axial and odd-L bases, 0, 1 and 2 nuclei, and S = 1.
2. `highField=True`: auto amplitude equals plain-direct. This fails on `main`, so add it as a strict
   xfail or together with the fix.
3. Lanczos against direct at 1e-6 for the slow-motion cases, including out-of-scope systems.
4. `MpSymm=True` with `GridSize > 1` and a potential raises, or equals the plain basis.
5. Tighten `simple_tc1e-07` only after the MATLAB check (c) below explains it.

**MATLAB checks for the lead.** Each is one small `matlab -singleCompThread -batch` job. They use the
nitroxide above, and each compares `max(abs(y))` and the cosine:

- **(a)** General Lanczos vs general direct vs fast. Expected: `yd` equals `yf` (cosine ≈ 1, ratio 1),
  and `yg` is off (0.937). This would confirm that EasySpin's general-method difference is its Lanczos
  stop.

  ```matlab
  Sys = struct('g',[2.0089 2.0058 2.0021],'Nucs','14N','A',[16 16 100],'tcorr',1e-7,'lw',[0 0.05]);
  Exp = struct('mwFreq',9.5,'Range',[328 352],'nPoints',2048,'Harmonic',1);
  [~,yf] = chili(Sys,Exp);                                          % default = fast
  [~,yg] = chili(Sys,Exp,struct('LiouvMethod','general'));          % plain, Lanczos
  [~,yd] = chili(Sys,Exp,struct('LiouvMethod','general','Solver','\'));  % plain, direct
  ```

- **(b)** Fast vs general with `highField`. Expected ratio 1. That would confirm finding 4: torchspin
  returns 0.5 of it.

  ```matlab
  [~,f] = chili(Sys,Exp,struct('highField',true));
  [~,g] = chili(Sys,Exp,struct('highField',true,'LiouvMethod','general','Solver','\'));
  ```

- **(c)** For `simple_tc1e-07` (Sys/Exp in `tests/data/generate_chili_ext_refs.m`), compare fast with
  general + `Solver='\'` (and `MpSymm=true`, output/2). This shows whether the 0.998 / 6.6 % residual
  is in EasySpin's fast output.
- **(d)** Optional: ¹⁵N + ¹H with LLMK [8 0 2 2], fast vs general direct, to explain the 2.0024 ratio.

## Implementation claims vs the manuscript

The manuscript (v0.2.2) describes the chili comparison as EasySpin's "specialized fast Liouvillian
construction" against TorchSpin's "general Liouvillian pathway". That is true of the builder, but
incomplete: v0.2.2 already applied the automatic rule to that system. The same holds for 0.9.31.
torchspin's default therefore used the symmetry basis at half amplitude. By the identity above, that
is the converged general-method answer.

The manuscript's NRMSE of 7.23e-5 (τc = 3 ns) is in the regime where the plain Lanczos also
converges (Exp. 1/2 at 1–10 ns), so the numbers are unaffected. Findings 4 and 5 lie outside every
case the manuscript or the test suite reports.
