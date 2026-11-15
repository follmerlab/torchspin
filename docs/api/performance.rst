Performance
===========

.. currentmodule:: torchspin

Optional ``torch.compile`` of the inner kernels and thread/process budgets
for running torchspin on many cores.  See also "Running on a workstation" on
the :doc:`/index` page and the threads section of :doc:`/pitfalls`.

Compilation
-----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   maybe_compile
   set_compile_enabled
   is_compile_available

Threads and processes
---------------------

Thread and process budgets live in the :mod:`torchspin.parallel` submodule.
