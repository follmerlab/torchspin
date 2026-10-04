"""
Automatic parameter estimation from EPR spectra.

This module provides tools to automatically extract initial guess parameters
from experimental EPR data, making fitting more user-friendly.
"""

import numpy as np
import torch
from scipy.signal import find_peaks
from typing import Dict, Tuple, Optional

from .constants import BMAGN, PLANCK

# MHz/mT per unit g — resonance condition: g = ν_MHz / (_MHZ_PER_MT * B_mT)
_MHZ_PER_MT: float = BMAGN / PLANCK * 1e-9  # ≈ 13.996 MHz/mT


def estimate_parameters(
    B: np.ndarray,
    spc: np.ndarray,
    mwFreq: float,
    nSpecies: int = 1,
    prominence: float = 0.1
) -> Dict:
    """
    Automatically estimate initial parameters from EPR spectrum.
    
    This function analyzes a derivative EPR spectrum to extract:
    - g-values from peak positions
    - Linewidths from peak widths
    - Suggested field range for simulation
    
    Parameters
    ----------
    B : ndarray
        Magnetic field axis (mT)
    spc : ndarray
        Spectrum intensity (derivative)
    mwFreq : float
        Microwave frequency (GHz)
    nSpecies : int, optional
        Expected number of species (default: 1)
    prominence : float, optional
        Minimum peak prominence as fraction of max signal (default: 0.1)
    
    Returns
    -------
    params : dict
        Dictionary containing:
        - 'g': Estimated g-tensor (sorted high to low)
        - 'lwpp': Estimated linewidth [Gaussian, Lorentzian] in mT
        - 'Range': Suggested field range [min, max] in mT
        - 'CenterSweep': Suggested [center, sweep] in mT
        - 'giso': Isotropic g-value
        - 'peaks': Dictionary with peak analysis details
    
    Examples
    --------
    >>> B, spc, params = eprload('mydata.DTA')
    >>> B = B / 10.0  # Convert Gauss to mT
    >>> est = estimate_parameters(B, spc, params['MWFQ']/1e9)
    >>> print(f"Estimated g: {est['g']}")
    >>> print(f"Estimated linewidth: {est['lwpp']} mT")
    
    Notes
    -----
    - Works best for well-resolved derivative spectra
    - For overlapping lines, may underestimate number of features
    - Linewidth estimates assume Gaussian broadening
    """
    
    # Normalize spectrum
    spc_norm = spc / np.max(np.abs(spc))
    
    # Find derivative peaks (positive and negative)
    pos_peaks, pos_props = find_peaks(spc_norm, prominence=prominence, width=1)
    neg_peaks, neg_props = find_peaks(-spc_norm, prominence=prominence, width=1)
    
    all_peaks = np.concatenate([pos_peaks, neg_peaks])
    all_signs = np.concatenate([np.ones(len(pos_peaks)), -np.ones(len(neg_peaks))])
    all_widths = np.concatenate([pos_props.get('widths', np.zeros(len(pos_peaks))),
                                  neg_props.get('widths', np.zeros(len(neg_peaks)))])
    
    # Sort by field position
    sort_idx = np.argsort(all_peaks)
    all_peaks = all_peaks[sort_idx]
    all_signs = all_signs[sort_idx]
    all_widths = all_widths[sort_idx]
    
    peak_fields = B[all_peaks]
    peak_amps = spc_norm[all_peaks] * all_signs
    
    # Calculate g-values from peak positions using resonance condition:
    #   g = h*ν / (μ_B * B)  →  g = (mwFreq_MHz) / (_MHZ_PER_MT * B_mT)
    g_vals = (mwFreq * 1000) / (peak_fields * _MHZ_PER_MT)
    
    # Remove unphysical g-values (outside reasonable range)
    valid_mask = (g_vals > 1.5) & (g_vals < 3.0)
    g_vals_valid = g_vals[valid_mask]
    peak_fields_valid = peak_fields[valid_mask]
    peak_amps_valid = peak_amps[valid_mask]
    all_widths_valid = all_widths[valid_mask]
    
    if len(g_vals_valid) == 0:
        # No valid peaks found, return default values
        g_center = mwFreq * 1000 / (np.mean(B) * _MHZ_PER_MT)
        return {
            'g': [g_center, g_center, g_center],
            'lwpp': [1.0, 0.0],
            'Range': [B.min(), B.max()],
            'CenterSweep': [np.mean(B), B.max() - B.min()],
            'giso': g_center,
            'peaks': {
                'fields': [],
                'g_values': [],
                'amplitudes': [],
                'widths': []
            }
        }
    
    # Estimate linewidth from peak widths (convert points to mT)
    if len(all_widths_valid) > 0:
        field_step = np.median(np.diff(B))
        peak_widths_mt = all_widths_valid * field_step
        
        # For derivative peak width to Gaussian FWHM:
        # Derivative peak-to-peak width ≈ FWHM * sqrt(3)/2
        # So FWHM ≈ peak_width * 2/sqrt(3) ≈ peak_width * 1.155
        estimated_lw = np.median(peak_widths_mt) * 1.155
    else:
        estimated_lw = 1.0
    
    # Determine g-tensor based on number of expected species
    if nSpecies == 1:
        # Single species - look for rhombic pattern
        if len(g_vals_valid) >= 3:
            # Take the 3 most extreme g-values (highest, lowest, and middle)
            g_sorted = np.sort(g_vals_valid)[::-1]  # Sort descending
            
            # Try to identify principal g-values
            # Look for cluster of peaks (from hyperfine) vs separate features
            g_diff = np.diff(g_sorted)
            
            # Find large gaps (indicating different g-values)
            gap_threshold = np.median(g_diff) * 2 if len(g_diff) > 2 else 0.05
            
            # Extract principal values (one from each cluster)
            g_principal = [g_sorted[0]]
            for i, g in enumerate(g_sorted[1:], 1):
                if abs(g - g_principal[-1]) > gap_threshold:
                    g_principal.append(g)
                if len(g_principal) >= 3:
                    break
            
            # Fill in if we don't have 3 values
            while len(g_principal) < 3:
                g_principal.append(np.mean(g_principal))
            
            g_tensor = g_principal[:3]
        elif len(g_vals_valid) == 2:
            # Axial system
            g_tensor = [g_vals_valid[0], g_vals_valid[1], g_vals_valid[1]]
        else:
            # Single g-value (isotropic)
            g_tensor = [g_vals_valid[0], g_vals_valid[0], g_vals_valid[0]]
    else:
        # Multiple species - just return all g-values found
        g_tensor = list(g_vals_valid[:min(nSpecies * 3, len(g_vals_valid))])
    
    # Ensure g-tensor is sorted high to low
    g_tensor = sorted(g_tensor, reverse=True)[:3]
    
    # Calculate isotropic g
    g_iso = np.mean(g_tensor)
    
    # Determine field range that covers the main features
    # Add some margin around the peaks
    if len(peak_fields_valid) > 0:
        field_min = peak_fields_valid.min()
        field_max = peak_fields_valid.max()
        margin = (field_max - field_min) * 0.2  # 20% margin
        range_min = max(B.min(), field_min - margin)
        range_max = min(B.max(), field_max + margin)
        center = (range_max + range_min) / 2
        sweep = range_max - range_min
    else:
        range_min = B.min()
        range_max = B.max()
        center = np.mean(B)
        sweep = range_max - range_min
    
    return {
        'g': g_tensor,
        'lwpp': [estimated_lw, 0.0],  # Assume Gaussian initially
        'Range': [range_min, range_max],
        'CenterSweep': [center, sweep],
        'giso': g_iso,
        'peaks': {
            'fields': peak_fields_valid.tolist(),
            'g_values': g_vals_valid.tolist(),
            'amplitudes': peak_amps_valid.tolist(),
            'widths': peak_widths_mt.tolist() if len(all_widths_valid) > 0 else []
        }
    }


def field_from_g(g: float, mwFreq: float) -> float:
    """
    Calculate resonance field for given g-value.
    
    Parameters
    ----------
    g : float
        g-value
    mwFreq : float
        Microwave frequency (GHz)
    
    Returns
    -------
    B : float
        Resonance field (mT)
    
    Examples
    --------
    >>> field_from_g(2.0, 9.5)
    339.22
    """
    # B [mT] = h*mwFreq [GHz]*1e9 / (g * BMAGN) * 1e3 = mwFreq_MHz / (g * _MHZ_PER_MT)
    return (mwFreq * 1000) / (g * _MHZ_PER_MT)


def g_from_field(B: float, mwFreq: float) -> float:
    """
    Calculate g-value from resonance field.
    
    Parameters
    ----------
    B : float
        Resonance field (mT)
    mwFreq : float
        Microwave frequency (GHz)
    
    Returns
    -------
    g : float
        g-value
    
    Examples
    --------
    >>> g_from_field(339.22, 9.5)
    2.0
    """
    # g = h*mwFreq_MHz / (BMAGN * B_mT * 1e-3) = mwFreq_MHz / (_MHZ_PER_MT * B_mT)
    return (mwFreq * 1000) / (B * _MHZ_PER_MT)


def print_parameter_report(params: Dict, mwFreq: float) -> None:
    """
    Print a formatted report of estimated parameters.
    
    Parameters
    ----------
    params : dict
        Parameter dictionary from estimate_parameters()
    mwFreq : float
        Microwave frequency (GHz)
    """
    print("="*70)
    print("AUTOMATIC PARAMETER ESTIMATION")
    print("="*70)
    
    print(f"\nMicrowave frequency: {mwFreq:.6f} GHz")
    
    print(f"\nEstimated g-tensor:")
    print(f"  g = [{params['g'][0]:.4f}, {params['g'][1]:.4f}, {params['g'][2]:.4f}]")
    print(f"  g_iso = {params['giso']:.4f}")
    
    # Calculate rhombicity
    g = params['g']
    if len(g) >= 3:
        if g[0] != g[2]:
            rhombicity = (g[1] - g[2]) / (g[0] - g[2])
        else:
            rhombicity = 0.0
        print(f"  Rhombicity = {rhombicity:.4f}")
        
        # Classify symmetry
        if abs(g[0] - g[1]) < 0.01 and abs(g[1] - g[2]) < 0.01:
            symmetry = "Isotropic"
        elif abs(g[1] - g[2]) < 0.01:
            symmetry = "Axial"
        else:
            symmetry = "Rhombic"
        print(f"  Symmetry: {symmetry}")
    
    print(f"\nExpected field positions at {mwFreq:.3f} GHz:")
    for i, gval in enumerate(g[:3], 1):
        B_res = field_from_g(gval, mwFreq)
        print(f"  g{i}={gval:.4f} → {B_res:.2f} mT")
    
    print(f"\nEstimated linewidth:")
    print(f"  lwpp = [{params['lwpp'][0]:.3f}, {params['lwpp'][1]:.3f}] mT")
    print(f"  (Gaussian: {params['lwpp'][0]:.3f} mT, Lorentzian: {params['lwpp'][1]:.3f} mT)")
    
    print(f"\nSuggested field range:")
    print(f"  Range = [{params['Range'][0]:.2f}, {params['Range'][1]:.2f}] mT")
    print(f"  CenterSweep = [{params['CenterSweep'][0]:.2f}, {params['CenterSweep'][1]:.2f}] mT")
    
    peaks = params['peaks']
    if len(peaks['fields']) > 0:
        print(f"\nDetected spectral features:")
        print(f"  Found {len(peaks['fields'])} peaks")
        print(f"  Fields: ", end="")
        for field in peaks['fields'][:5]:
            print(f"{field:.2f} ", end="")
        if len(peaks['fields']) > 5:
            print("...")
        else:
            print()
        print(f"  g-values: ", end="")
        for gval in peaks['g_values'][:5]:
            print(f"{gval:.4f} ", end="")
        if len(peaks['g_values']) > 5:
            print("...")
        else:
            print()
    
    print("="*70)


if __name__ == "__main__":
    # Example usage
    print("Automatic parameter estimation module for torchspin")
    print("Usage:")
    print("  from torchspin import estimate_parameters")
    print("  from torchspin.eprload import eprload")
    print()
    print("  B, spc, params = eprload('data.DTA')")
    print("  B = B / 10.0  # Convert Gauss to mT")
    print("  est = estimate_parameters(B, spc, params['MWFQ']/1e9)")
    print("  print_parameter_report(est, params['MWFQ']/1e9)")
