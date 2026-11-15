SpinSystem
==========

.. currentmodule:: torchspin

A :class:`SpinSystem` holds the spin Hamiltonian parameters of one species
(EasySpin's ``Sys``): electron and nuclear spins, g, zero-field splitting,
hyperfine and quadrupole tensors with their frames, line widths and strains.
Every simulator takes one, or a list of them for multi-component spectra.

Units: energies and couplings in MHz, fields in mT, Euler angles in radians.

.. autoclass:: SpinSystem
   :members:
   :exclude-members: __init__, __post_init__

.. seealso::

   :doc:`/tutorials/01_torchspin_quickstart` builds spin systems step by step.
   Helpers that derive new systems from an existing one are listed under
   :doc:`spin_systems`.
