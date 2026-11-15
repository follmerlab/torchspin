Trajectories and molecular dynamics
===================================

.. currentmodule:: torchspin

Trajectory-based CW EPR (:func:`cardamom`), stochastic rotational
trajectories, and tools that turn molecular dynamics trajectories into
cardamom input and orientational potentials.

Trajectory-based simulation
---------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   cardamom
   CardamomPar
   CardamomOptions
   MDInput

Stochastic trajectories
-----------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   stochtraj_diffusion
   DiffusionPar
   stochtraj_jump
   JumpPar

Molecular dynamics
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   mdload
   MDData
   MDInfo
   MDOptions
   mdhmm
   HMMResult
   HMMOptions
   mdtraj2oripot
   oripotentialplot
