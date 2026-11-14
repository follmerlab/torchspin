Hamiltonians, levels and transitions
====================================

.. currentmodule:: torchspin

Spin Hamiltonian matrices term by term, spin operators, energy levels,
resonance fields and frequencies, ENDOR frequencies and orientation
selection.  All energies are in MHz and fields in mT.

Hamiltonians
------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   ham
   ham_ez
   ham_zf
   ham_hf
   ham_nz
   ham_nq
   ham_ee
   ham_nn
   ham_cf
   ham_oz
   ham_so
   ham_ezho
   hamsymm

Spin operators
--------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   sop
   commute

Energy levels
-------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   levels
   levelsplot
   LevelsPlotOptions

Resonances
----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   resfreqs_matrix
   resfreqs_perturb
   resfields_eig
   EigOptions
   endorfrq
   endorfrq_perturb
   orisel
   OriselOptions
