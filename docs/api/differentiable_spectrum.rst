differentiable_spectrum
=======================

.. currentmodule:: torchspin

A powder CW EPR spectrum through which gradients flow to every spin
Hamiltonian parameter that is a tensor with ``requires_grad=True``, for
gradient-based fitting, sensitivity analysis and machine learning.

.. autofunction:: differentiable_spectrum

.. seealso::

   :doc:`/tutorials/02_gpu_and_autograd`, and :mod:`torchspin.ml` for training
   and inference on top of the differentiable simulators.
