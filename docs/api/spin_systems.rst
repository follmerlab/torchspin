Spin systems
============

.. currentmodule:: torchspin

Building, editing and analyzing spin systems: isotopologues and nuclear
bookkeeping, equivalent nuclei, exchange-coupled manifolds, dipolar tensors,
import from ORCA, and starting guesses estimated from a measured spectrum.

The spin system
---------------

Documented on its own page.

.. autosummary::
   :nosignatures:

   SpinSystem

Nuclei and isotopologues
------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   nucspinadd
   nucspinrmv
   nucspinkeep
   isotopologues
   spinvec
   hsdim
   equivsplit
   equivcouple

Couplings and manifolds
-----------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   diptensor
   zfsframes
   spinladder
   SpinManifold

Import and parameter estimates
------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   orca2torchspin
   OrcaData
   estimate_parameters
   field_from_g
   g_from_field
   print_parameter_report
