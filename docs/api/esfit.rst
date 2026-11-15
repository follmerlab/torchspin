esfit
=====

.. currentmodule:: torchspin

Least-squares fitting of simulated spectra to data with local optimizers
(Nelder-Mead simplex, Levenberg-Marquardt, trust-region, L-BFGS-B, Powell) and
global ones (particle swarm with a simplex polish, genetic algorithm, grid
search, Monte Carlo).  :class:`FitOptions` selects the method, the scaling,
baseline and residual target, and the stopping criteria; :class:`FitResult`
holds the fitted parameters, the fitted spectrum, residuals and, where
available, standard errors and the parameter covariance.

.. autofunction:: esfit

.. autoclass:: FitOptions
   :members:
   :exclude-members: __init__, __post_init__

.. autoclass:: FitResult
   :members:
   :exclude-members: __init__, __post_init__

.. seealso::

   :doc:`/tutorials/03_spectral_fitting`, :doc:`/tutorials/05_fitting_real_world`,
   :doc:`/tutorials/06_blue_copper_fitting` and the interactive panel in
   :doc:`/tutorials/09_fit_gui` (:mod:`torchspin.fitgui`).
