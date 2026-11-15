Pulse EPR
=========

.. currentmodule:: torchspin

Pulse EPR simulation (:func:`saffron` for ideal pulses, :func:`spidyan`
for real pulses with explicit time evolution), shaped-pulse waveforms and
hardware models, density-matrix propagation, and DEER analysis kernels.

Simulators
----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   saffron
   PulseExperiment
   SaffronOptions
   find_refocusing_pathways
   spidyan
   SpidyanOptions
   SpidyanInfo
   nucfrq2d

Density matrices and propagation
--------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   sigeq
   evolve
   propint

Pulses and hardware
-------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   pulse
   exciteprofile
   resonator
   resonatorprofile
   transmitter
   rfmixer
   signalprocessing

DEER
----

.. autosummary::
   :toctree: generated
   :nosignatures:

   dipkernel
   dipbackground
