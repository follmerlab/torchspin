# User guide

Guides that go beyond a single function: what can go wrong, what is and is not
ported, and the ligand-field tools.  For the parameters of a particular
function or class, see the docstrings.

- [Pitfalls](pitfalls.md): valid settings that quietly degrade a result, and
  how torchspin warns about them.
- [Known limitations](../KNOWN_LIMITATIONS.md): what is and is not ported, per
  simulator, with the evidence.  Read this before relying on a result for
  publication.
- [Ligand-field tools](lft.md): Joshua Telser's ligand-field programs as the
  differentiable `torchspin.lft` package.
- [Telser's spin-Hamiltonian programs vs torchspin](lft_gap_analysis.md): how
  DDZHQ, DDPOWH, DJDZE and DDPOWJ inputs map onto torchspin.

```{toctree}
:hidden:

pitfalls
lft
lft_gap_analysis
```
