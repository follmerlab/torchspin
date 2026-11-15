grid_convergence
================

.. currentmodule:: torchspin

Checks whether a powder spectrum is converged with respect to the orientation
grid by simulating it again on a finer grid and comparing the two.

.. autofunction:: grid_convergence

.. autoclass:: GridConvergence
   :members:
   :exclude-members: __init__, __post_init__

.. seealso::

   ``GridSize`` and ``GridSymmetry`` in :class:`Options`.
