Line shapes and spectral processing
===================================

.. currentmodule:: torchspin

Line shape functions, convolution and binning, field modulation, noise,
baseline correction, smoothing and filtering, and time-domain signal
processing.

Line shapes
-----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   gaussian
   lorentzian
   voigtian
   lshape
   convspec
   makespec

Spectrum processing
-------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   deriv
   fieldmod
   addnoise
   basecorr
   rescaledata
   datasmooth
   rcfilt
   hilberttrans
   apowin
   stackplot

Time-domain signal processing
-----------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   fdaxis
   ctafft
   ewrls
   lpsvd
   LPSVDResult
   mlpsvd
   MLPSVDResult
   exponfit
   rapidscan2spc
