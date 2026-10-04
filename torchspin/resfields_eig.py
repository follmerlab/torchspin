"""
Resonance fields via the eigenfield method (Liouville space).

Solves a generalized eigenvalue problem in Liouville space to find all
resonance fields simultaneously, without root-finding or field sweeping.

Based on:
  Belford, Belford, Burkhalter, J. Magn. Reson. 11, 251-265 (1973)
  MATLAB EasySpin's resfields_eig.m

All energies in MHz, fields in mT.
"""

import numpy as np
import scipy.linalg as la
from typing import Union, Optional, Tuple, List
from dataclasses import dataclass, field

from torchspin.ham import ham
from torchspin.spinsystem import SpinSystem
from torchspin.rotations import erot


@dataclass
class EigOptions:
    """Options for resfields_eig."""
    Threshold: float = 0.0
    Freq2Field: bool = True
    RejectionRatio: float = 1e-8


def resfields_eig(
    sys: SpinSystem,
    mwFreq: float,
    *,
    Range: Optional[Tuple[float, float]] = None,
    mwMode: str = 'perpendicular',
    Orientations: Optional[np.ndarray] = None,
    Opt: Optional[EigOptions] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute resonance fields using the eigenfield method.

    Parameters
    ----------
    sys : SpinSystem
        Spin system specification.
    mwFreq : float
        Microwave frequency in GHz.
    Range : (float, float), optional
        Field range [Bmin, Bmax] in mT. Default: [0, 1e6].
    mwMode : str
        'perpendicular' (default) or 'parallel'.
    Orientations : ndarray, shape (N, 3), optional
        Euler angles [alpha, beta, gamma] for each orientation.
        Default: single orientation along z ([0, 0, 0]).
    Opt : EigOptions, optional
        Algorithm options.

    Returns
    -------
    fields : ndarray or list of ndarray
        Resonance fields in mT. For single orientation, a 1D array.
        For multiple orientations, a list of arrays.
    intensities : ndarray or list of ndarray
        Transition intensities (MHz²/mT²). Only returned if Opt.Threshold > 0
        or if intensities would be useful.
    """
    if Opt is None:
        Opt = EigOptions()

    if Range is None:
        Range = (0.0, 1e6)

    parallel_mode = mwMode.lower() == 'parallel'

    # Convert mwFreq from GHz to MHz
    mwFreq_MHz = mwFreq * 1e3

    # Build Hamiltonian components: H = H0 - B·mu
    H0, mux, muy, muz = ham(sys)

    # Convert to numpy complex128
    H0 = H0.detach().cpu().numpy().astype(complex)
    mux = mux.detach().cpu().numpy().astype(complex)
    muy = muy.detach().cpu().numpy().astype(complex)
    muz = muz.detach().cpu().numpy().astype(complex)

    n = H0.shape[0]
    n2 = n * n

    # Build Liouville space operator A
    # A = I⊗H0 - H0*⊗I + ωmw·I
    I_n = np.eye(n, dtype=complex)
    A = np.kron(I_n, H0) - np.kron(np.conj(H0), I_n) + mwFreq_MHz * np.eye(n2, dtype=complex)

    # Check if A is positive-definite → simple eigenproblem
    evals_A = np.real(la.eigvalsh(A))
    simple_eigenproblem = np.all(evals_A > 0)

    # Vectorized moment operators for intensity calculation
    compute_intensities = True
    mux_vec = mux.T.reshape(-1)
    muy_vec = muy.T.reshape(-1)
    muz_vec = muz.T.reshape(-1)

    # Set up orientations
    if Orientations is None:
        Orientations = np.array([[0.0, 0.0, 0.0]])
    if Orientations.ndim == 1:
        Orientations = Orientations.reshape(1, -1)

    nOrientations = len(Orientations)
    average_over_chi = True  # perpendicular mode averages over chi

    all_fields = []
    all_intensities = []

    for iOri in range(nOrientations):
        angles = Orientations[iOri]
        # Get lab frame axes from Euler rotation
        R = erot(*angles).detach().cpu().numpy().astype(complex)
        xLab = R[0, :]
        yLab = R[1, :]
        zLab = R[2, :]

        # muzL = projection of magnetic moment onto lab z (field direction)
        muzL = zLab[0] * mux + zLab[1] * muy + zLab[2] * muz

        # B = -conj(muzL)⊗I + I⊗muzL  in Liouville space
        B_mat = np.kron(I_n, muzL) - np.kron(np.conj(muzL), I_n)

        if compute_intensities:
            if simple_eigenproblem:
                BB = la.solve(A, B_mat)
                eigenvalues, Vecs = la.eig(BB)
                # Fields are 1/eigenvalue for simple problem
                with np.errstate(divide='ignore', invalid='ignore'):
                    Fields = 1.0 / eigenvalues
                # Sort
                idx = np.argsort(np.real(Fields))
                Fields = Fields[idx]
                Vecs = Vecs[:, idx]
            else:
                eigenvalues, Vecs = la.eig(A, B_mat)
                Fields = eigenvalues
                idx = np.argsort(np.real(Fields))
                Fields = Fields[idx]
                Vecs = Vecs[:, idx]

            # Filter: keep real, positive, finite, in-range eigenfields
            real_part = np.real(Fields)
            imag_part = np.abs(np.imag(Fields))
            valid = (
                (imag_part < Opt.RejectionRatio * np.abs(real_part)) &
                (real_part > 0) &
                np.isfinite(real_part) &
                (real_part > Range[0]) &
                (real_part < Range[1])
            )

            if not np.any(valid):
                all_fields.append(np.array([]))
                all_intensities.append(np.array([]))
                continue

            eigen_fields = np.real(Fields[valid])
            Vecs = Vecs[:, valid]

            # Normalize eigenvectors
            norms = np.sqrt(np.sum(np.abs(Vecs)**2, axis=0))
            Vecs = Vecs / norms[np.newaxis, :]

            # Compute transition rate
            muzL_vec = zLab[0] * mux_vec + zLab[1] * muy_vec + zLab[2] * muz_vec

            if parallel_mode:
                TransitionRate = np.abs(np.sum(muzL_vec[:, np.newaxis] * Vecs, axis=0))**2
            else:
                muxL_vec = xLab[0] * mux_vec + xLab[1] * muy_vec + xLab[2] * muz_vec
                if average_over_chi:
                    muyL_vec = yLab[0] * mux_vec + yLab[1] * muy_vec + yLab[2] * muz_vec
                    TransitionRate = (
                        np.abs(np.sum(muxL_vec[:, np.newaxis] * Vecs, axis=0))**2 +
                        np.abs(np.sum(muyL_vec[:, np.newaxis] * Vecs, axis=0))**2
                    ) / 2
                else:
                    TransitionRate = np.abs(np.sum(muxL_vec[:, np.newaxis] * Vecs, axis=0))**2

            # Polarization factor
            nNuclei = len(sys.Nucs) if sys.Nucs else 0
            I_vals = []
            if nNuclei > 0:
                from torchspin.nucdata import nucspin
                for nuc in sys.Nucs:
                    I_vals.append(nucspin(nuc))
            polarization = 1.0
            for I_val in I_vals:
                polarization /= (2 * I_val + 1)

            # Frequency-to-field conversion factor
            if Opt.Freq2Field:
                Vecs_3d = Vecs.reshape(n, n, -1)
                dBdE = np.zeros(Vecs_3d.shape[2])
                for iVec in range(Vecs_3d.shape[2]):
                    V = Vecs_3d[:, :, iVec]
                    comm = V @ V.conj().T - V.conj().T @ V
                    dBdE[iVec] = 1.0 / max(abs(np.trace(-muzL @ comm)), 1e-30)
            else:
                dBdE = np.ones(len(TransitionRate))

            intensities = polarization * np.real(TransitionRate * dBdE)

            # Apply threshold
            if Opt.Threshold > 0 and len(intensities) > 0:
                max_int = np.max(intensities)
                mask = intensities >= Opt.Threshold * max_int
                eigen_fields = eigen_fields[mask]
                intensities = intensities[mask]

            all_fields.append(eigen_fields)
            all_intensities.append(intensities)

        else:
            # Fields only, no intensities
            if simple_eigenproblem:
                BB = la.solve(A, B_mat)
                eigenvalues = la.eigvals(BB)
                Fields = np.sort(1.0 / eigenvalues)
            else:
                eigenvalues = la.eigvals(A, B_mat)
                Fields = np.sort(eigenvalues)

            real_part = np.real(Fields)
            imag_part = np.abs(np.imag(Fields))
            valid = (
                (imag_part < Opt.RejectionRatio * np.abs(real_part)) &
                (real_part > 0) &
                np.isfinite(real_part) &
                (real_part > Range[0]) &
                (real_part < Range[1])
            )
            all_fields.append(np.real(Fields[valid]))
            all_intensities.append(np.array([]))

    # Single orientation: return arrays directly
    if nOrientations == 1:
        return all_fields[0], all_intensities[0]

    return all_fields, all_intensities
