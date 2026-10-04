"""Physical constants for torchspin.

All values are CODATA 2022 recommended values.
"""

import math as _math

# Bohr magneton (J T^-1)
BMAGN: float = 9.2740100657e-24

# Planck constant (J s)
PLANCK: float = 6.62607015e-34

# Free-electron g-factor (dimensionless, positive convention)
GFREE: float = 2.00231930436092

# Nuclear magneton (J T^-1)
NMAGN: float = 5.0507837393e-27

# Boltzmann constant (J K^-1)
BOLTZMANN: float = 1.380649e-23

# Speed of light in vacuum (cm s^-1)
CLIGHT: float = 2.99792458e10

# Electron volt (J) — exact by SI definition since 2019
EVOLT: float = 1.602176634e-19

# Reduced Planck constant ℏ = h / (2π)  (J s)
HBAR: float = PLANCK / (2 * _math.pi)

# Atomic mass unit  (kg)  — CODATA 2022
AMU: float = 1.66053906660e-27

# Elementary charge  (C)  — exact by SI definition since 2019
ECHARGE: float = 1.602176634e-19

# Electron mass  (kg)  — CODATA 2022
EMASS: float = 9.1093837139e-31

# Proton mass  (kg)  — CODATA 2022
PMASS: float = 1.67262192595e-27

# Neutron mass  (kg)  — CODATA 2022
NMASS: float = 1.67492750056e-27

# Vacuum permittivity ε₀  (F m⁻¹)  — exact: 1/(μ₀ c²)
EPS0: float = 8.8541878188e-12

# Vacuum permeability μ₀  (H m⁻¹)  — CODATA 2022 (no longer exact)
MU0: float = 1.25663706127e-6

# Faraday constant  (C mol⁻¹)  — exact: N_A × e
FARADAY: float = 96485.33212

# Avogadro's number  (mol⁻¹)  — exact by SI definition since 2019
AVOGADRO: float = 6.02214076e23

# Bohr radius  (m)  — CODATA 2022
BOHRRAD: float = 5.29177210544e-11

# Hartree energy  (J)  — CODATA 2022
HARTREE: float = 4.3597447222060e-18

# Rydberg constant × hc  (J)  — CODATA 2022
RYDBERG: float = 2.1798723611030e-18

# Angstrom — molecular-scale length unit  (m)  — exact
ANGSTROM: float = 1e-10

# Electron gyromagnetic ratio  γ_e = g_e × μ_B / ℏ  (rad s⁻¹ T⁻¹)  — positive by convention
GAMMAE: float = GFREE * BMAGN / HBAR

# Nuclear magneton / ℏ  (rad s⁻¹ T⁻¹)  — base for nuclear gyromagnetic ratios
GAMMAN: float = NMAGN / HBAR

# Degree — radians per degree (exact: π/180)
DEGREE: float = _math.pi / 180.0

# Barn — nuclear quadrupole moment unit  (m²)  — exact: 1 b = 10^-28 m²
BARN: float = 1e-28

# Molar gas constant  R = N_A × k_B  (J mol⁻¹ K⁻¹)  — exact
MOLGAS: float = AVOGADRO * BOLTZMANN

# Conversion: MHz/mT = GHz/T
# pre = -BMAGN/PLANCK * g   [Hz/T]  * 1e-9  -> [MHz/mT]
# This is the prefactor used in the Zeeman Hamiltonian (matching EasySpin convention)
