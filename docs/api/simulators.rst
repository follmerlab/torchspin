Simulators
==========

.. currentmodule:: torchspin

CW EPR, ENDOR and magnetometry simulators besides the core :func:`pepper`
and :func:`garlic`, with their option classes.  Pulse EPR and trajectory-based
simulators have their own pages (:doc:`pulse` and :doc:`trajectories`).

Core simulators
---------------

Documented on their own pages.

.. autosummary::
   :nosignatures:

   pepper
   garlic
   differentiable_spectrum
   grid_convergence

CW EPR
------

.. autosummary::
   :toctree: generated
   :nosignatures:

   chili
   ChiliOptions
   fastmotion
   pepper_autograd
   blochsteady
   BlochOptions

ENDOR
-----

.. autosummary::
   :toctree: generated
   :nosignatures:

   salt

Magnetometry
------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   curry

Batches
-------

Many spin systems or experiments in one call.

.. autosummary::
   :toctree: generated
   :nosignatures:

   batch_pepper
   batch_simulate
