Options
=======

.. currentmodule:: torchspin

:class:`Options` collects computational settings (EasySpin's ``Opt``): the
method (matrix diagonalization or perturbation theory), the orientation grid
(``GridSize``, ``GridSymmetry``), transition selection thresholds, the device
and dtype, and verbosity.  Defaults reproduce EasySpin's defaults unless noted.

.. autoclass:: Options
   :members:
   :exclude-members: __init__, __post_init__

.. seealso::

   :func:`grid_convergence` checks whether ``GridSize`` is fine enough for a
   given spectrum.  :doc:`/pitfalls` lists valid settings that quietly degrade
   a result.
