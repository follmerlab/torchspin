Rotations, orientations and angular momentum
============================================

.. currentmodule:: torchspin

Euler angles (z-y'-z'' convention, radians), rotation matrices and
quaternions, orientation grids for powder averages, and angular momentum
algebra (Wigner symbols, spherical harmonics, irreducible spherical tensor
operators).

Rotations
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   erot
   eulang
   euler2quat
   quat2euler
   quat2rotmat
   rotmat2quat
   rotaxi2mat
   rotmat2axi
   quatinv
   quatmult
   quatvecmult
   rotateframe
   vec2ang
   ang2vec
   tensor_cart2sph
   tensor_sph2cart

Orientation grids and selection
-------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   sphgrid
   sphrand
   photoselect

Angular momentum
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   wigner3j
   wigner6j
   wignerd
   clebschgordan
   cgmatrix
   spherharm
   plegendre
   isto
   isto2stev
