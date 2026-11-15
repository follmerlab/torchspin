# Gradients and differentiable simulation

torchspin is written in PyTorch, so spectra can carry gradients with respect
to spin Hamiltonian parameters.  Use this for gradient-based fitting,
sensitivity analysis ("which parameter does this feature depend on?"), or a
simulator inside a neural network.

## Gradients through pepper

Give any tensor field of a `SpinSystem` `requires_grad=True`.  pepper then
runs its autograd path ([`pepper_autograd`](../api/generated/torchspin.pepper_autograd.rst),
the same forward calculation) and returns a spectrum on the autograd graph:

```pycon
>>> import torch
>>> from torchspin import SpinSystem, Experiment, pepper
>>> g = torch.tensor([2.0, 2.1, 2.2], dtype=torch.float64, requires_grad=True)
>>> sys = SpinSystem(S=0.5, g=g, lw=1.0)
>>> exp = Experiment(mwFreq=9.5, Range=[300, 350], nPoints=256)
>>> B, spc = pepper(sys, exp)
>>> spc.requires_grad
True
>>> loss = spc.pow(2).sum()
>>> loss.backward()
>>> g.grad.shape
torch.Size([3])
```

## A compact differentiable model

[`differentiable_spectrum`](../api/differentiable_spectrum.rst) takes the
parameters as plain tensors (g principal values, line width, optional
nuclei, hyperfine and zero-field tensors) and returns a powder spectrum, which
is convenient in an optimization loop:

```pycon
>>> from torchspin import differentiable_spectrum
>>> g = torch.tensor([2.0, 2.1, 2.2], dtype=torch.float64, requires_grad=True)
>>> B, spec = differentiable_spectrum(g, lw_mT=1.0, mwFreq_GHz=9.5,
...                                   B_range=(300.0, 350.0), nPoints=256,
...                                   GridSize=10)
>>> (grad_g,) = torch.autograd.grad(spec.pow(2).sum(), g)
>>> grad_g.shape
torch.Size([3])
```

## Fitting by gradient descent

Any PyTorch optimizer works on these tensors; the
[GPU and autograd tutorial](../tutorials/02_gpu_and_autograd.ipynb) fits g
values with Adam.  For least-squares fits with uncertainties, [esfit](fitting.md)
is usually the better tool.  `torchspin.ml` ([API](../api/ml.rst)) adds
parameter specifications, a simulated-spectrum dataset and losses for
training networks.

## GPU

Put the computation on a GPU with `Options(device='cuda')` (pepper, salt,
pepper_autograd) or `device='cuda'` in `differentiable_spectrum`:

```python
from torchspin import Options
B, spc = pepper(sys, exp, Options(device='cuda'))
```

A GPU pays off for batches, gradients and fits, rarely for a single small
simulation.  On a CPU, torchspin uses all cores; `torch.set_num_threads(n)`
limits it.
