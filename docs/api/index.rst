API reference
=============

.. currentmodule:: torchspin

Everything documented here is importable from the top-level package, e.g.
``from torchspin import SpinSystem, Experiment, Options, pepper``; the
pages use those public names (``torchspin.pepper``), not the internal module
paths.  Units follow EasySpin: energies and couplings in MHz, magnetic fields
in mT, microwave frequencies in GHz, angles in radians.

Core objects
------------

Most work needs only these: describe the spin system and the experiment,
simulate, and fit.

.. autosummary::
   :nosignatures:

   SpinSystem
   Experiment
   Options
   pepper
   garlic
   esfit
   FitOptions
   FitResult
   differentiable_spectrum
   grid_convergence
   GridConvergence

.. toctree::
   :hidden:
   :caption: Core objects

   spinsystem
   experiment
   options
   pepper
   garlic
   esfit
   differentiable_spectrum
   grid_convergence

By topic
--------

- :doc:`simulators`: CW EPR, ENDOR and magnetometry simulators besides
  pepper and garlic
- :doc:`pulse`: saffron, spidyan, pulse shapes and hardware, DEER kernels
- :doc:`trajectories`: cardamom, stochastic trajectories, molecular dynamics
- :doc:`spin_systems`: nuclei, isotopologues, couplings, ORCA import,
  parameter estimates
- :doc:`hamiltonians`: Hamiltonian terms, spin operators, energy levels,
  resonance fields and frequencies
- :doc:`orientations`: Euler angles, quaternions, orientation grids, angular
  momentum algebra
- :doc:`lineshapes`: line shapes, convolution, modulation, baseline and
  time-domain processing
- :doc:`data_units`: spectrometer files, unit conversion, nuclear data
- :doc:`constants`: physical constants and the package version
- :doc:`performance`: ``torch.compile`` switches

.. toctree::
   :hidden:
   :caption: By topic

   simulators
   pulse
   trajectories
   spin_systems
   hamiltonians
   orientations
   lineshapes
   data_units
   constants
   performance

Submodules
----------

Packages that are imported as modules rather than as top-level names:

- :doc:`lft`: ligand-field tools (Telser's programs), with the
  :doc:`/lft` user guide
- :doc:`ml`: parameter specifications, datasets and losses for machine
  learning on simulated spectra
- :doc:`parallel`: thread and process budgets
- :doc:`fitgui`: interactive fitting panel for notebooks

.. toctree::
   :hidden:
   :caption: Submodules

   lft
   ml
   parallel
   fitgui
