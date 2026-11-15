# Troubleshooting and reference

Material to consult when a result looks wrong, when you need to know exactly
what is and is not ported from EasySpin, or what changed between versions.

- [Physics and validation](physics_validation.md): what each simulator computes,
  how, and how well it is validated, with a page per simulator.
- [Pitfalls](pitfalls.md): valid settings that quietly degrade a result, and
  how torchspin warns about them.
- [Known limitations](known_limitations.md): what is and is not ported, per
  simulator, with the evidence.
- [Ligand-field tools: technical reference](lft.md) and
  [Telser's spin-Hamiltonian programs vs torchspin](lft_gap_analysis.md).
- [Changelog](changelog.md).

```{toctree}
:hidden:

physics_validation
chili_empirical_rules
pitfalls
known_limitations
lft
lft_gap_analysis
changelog
```
