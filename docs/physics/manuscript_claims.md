# Manuscript-era claims vs the current implementation (developer notes)

This page is for maintainers. It separates the results the manuscript measured on its release from
what the current implementation (0.9.31) does. The manuscript's numbers are historical and are not
edited. A full manuscript-vs-current comparison is planned separately.

**Measured release.** The submitted manuscript (arXiv:2610.18924; source snapshot
`torchspin_submitted_main.tex`, 2026-10-12) says: *"The reference-comparison and performance analyses
used TorchSpin v0.2.2."*

The performance audit
([BENCHMARK_VERIFICATION.md](../../benchmarks/results/workstation_20260904_verified/BENCHMARK_VERIFICATION.md))
records torchspin commit 420cbc9 (v0.2.2, 2026-09-03). That commit is not in this repository: the
pre-0.3.0 history was squashed (CHANGELOG footnote), and the first tag is `v0.3.0`.

`BENCHMARK_REPORT.md` Table 1 and Figures 1–5 come from an earlier run (2026-04-20, v0.1.0 era,
EasySpin 6.0).

The right-hand column describes 0.9.31, with the B85, B86 and chili (B114–B116) rows updated for
0.9.38. The manuscript re-check will re-measure every row.

| Manuscript claim (v0.2.2) | Current status (0.9.31; updated for 0.9.38) |
|---|---|
| garlic Fremy's salt NRMSE < 1e-15 | Still exact: `test_garlic_exact_matlab_validation.py`, positions 1e-6 mT. |
| chili nitroxide (τc 3 ns), NRMSE 7.23e-5. The EasySpin reference is from the *fast* builder; torchspin used "a general Liouvillian pathway". | Numbers stand. The description is incomplete: since 2026-09-02, which includes v0.2.2, torchspin's default for this system is the general method **with** the M–pS–pI symmetry basis at half amplitude (the automatic rule). The rule is exact to 1e-8; 0.9.38 replaced the factor ½ by the exact mirror term (B114–B116). See [chili_empirical_rules.md](../chili_empirical_rules.md). |
| pepper `pepper_rhombiclw`, NRMSE 9.09e-5 | Measured with the half-bin projection convention that EasySpin shares. **B85 (0.9.35)** moved every projected spectrum by Δ/2. The agreement with EasySpin is now a documented deviation, and correctness rests on converged references. |
| salt `salt_invariance_axial1`, cosine 1.00, global amplitude factor 3.65 | The scale difference is the documented convention (EasySpin × Δx/8π²). Since v0.2.2 salt changed its narrow-width convolution rule (0.8.11), edge padding (0.9.6), hyperfine enhancement (0.7.3) and transition selection (0.9.14). The manuscript case has not been re-measured. B85 (0.9.35, centered bins) and B86 (0.9.36, differentiable isotropic path) changed it again. |
| cardamom, NRMSE 0.0201 | Same model. The finite-ensemble residual is expected. Seeded reproducibility was added later (`CardamomPar.seed`). |
| saffron `saffron_2psimple`, NRMSE 0.0101 | Measured **before** the 0.5.0 saffron fixes: the tau phase factor in the 2p/3p/4p ESEEM and HYSCORE routes, and the S = 1/2 manifold assignment. The test had a 10× smaller quadrupole coupling and the wrong grid (20 vs 31). The current suite holds the case at relative L2 ≤ 1e-6. Whether the manuscript's Fig. S2 trace was affected has not been re-measured. |
| spidyan e–¹H, NRMSE 6.39e-5 | This is the residual that 0.6.0 traced to a **test transcription error**: g = 2.00232 instead of gfree, a 3.5e-7 offset accumulating as phase. The manuscript states g = 2.00232. The current agreement is 1e-14..1e-12; no library code changed. |
| curry Gd–Cu μ_eff, NRMSE 1.48e-7 | Unchanged model (`test_curry_bloch_matlab_validation.py`). |
| Performance (Fig. 2, Table 2, Fig. S5): nine workloads, v0.2.2 | Not re-measured. pepper's projection, interpolation and strain paths and the chili default have changed since. README's speedups (chili 7.3×, cardamom 2.0×, Cu(II) 0.34×) are these v0.2.2 numbers. |
| `BENCHMARK_REPORT.md` Table 1 (2026-04-20): pepper amplitude "~15 % (correction factors)"; chili cosine 0.92–0.997; spidyan and cardamom "self-consistent / internal" | Superseded. The empirical pepper factors were removed on 2026-09-01. chili was rewritten on 2026-09-01. spidyan and cardamom are now EasySpin-validated. The report says it is a layered record and defers to the verified campaign. |

Smaller statements in the repository docs that no longer match the tests:

- README says agreement is "cosine ≥ 0.999" for salt and curry. The orientation-selected salt cases
  pass at > 0.99 (measured 0.993–0.998), S = 3/2 at > 0.98, and curry at 0.5–1 % of the maximum.
- `docs/index.md` says "about 2800 tests". The suite has about 5800.
