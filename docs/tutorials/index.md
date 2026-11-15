# Tutorials

Executed Jupyter notebooks from
[`examples/notebooks/`](../../examples/notebooks/).  The site shows them as
committed, outputs included; nothing is re-run during the docs build.  To run
one yourself, open it from a clone of the repository with torchspin installed
(`pip install "torchspin[plot]"`; notebook 09 needs `torchspin[gui]`).

Scripts with the same coverage as EasySpin's example collection live in
[`examples/`](../../examples/) (`solidstate`, `liquids`, `slowmotion`, `endor`,
`fitting`, `magnetometry`).

```{toctree}
:maxdepth: 1

01_torchspin_quickstart
02_gpu_and_autograd
03_spectral_fitting
04_benchmarks
05_fitting_real_world
06_blue_copper_fitting
07_easyspin_parity
08_pepper_playground
09_fit_gui
```
