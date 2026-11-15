Experiment
==========

.. currentmodule:: torchspin

An :class:`Experiment` describes the measurement (EasySpin's ``Exp``):
microwave frequency and field range for field-swept spectra, or a fixed field
and frequency range for frequency-swept ones, plus harmonic, modulation
amplitude, temperature, and sample orientation.

Units: microwave frequency in GHz, fields in mT, temperature in K.

.. autoclass:: Experiment
   :members:
   :exclude-members: __init__, __post_init__

.. seealso::

   :class:`Options` for computational settings, :class:`PulseExperiment` for
   pulse experiments in :func:`saffron`.
