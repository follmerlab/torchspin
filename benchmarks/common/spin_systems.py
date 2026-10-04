"""Reference test cases used by both MATLAB and Python benchmarks.

Each entry is a self-contained dict that defines:
- spin system parameters (matching EasySpin Sys struct keys)
- experiment parameters (matching EasySpin Exp struct keys)
- options (matching Opt struct)
- expected output keys (for validation)

The MATLAB benchmark scripts import these by mirroring the dicts in MATLAB
struct form. The Python benchmark scripts construct SpinSystem/Experiment
instances directly.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# pepper benchmarks (powder CW EPR, field-swept)
# ---------------------------------------------------------------------------

PEPPER_CASES = {
    'nitroxide_xband': {
        'description': 'Nitroxide radical at X-band (S=1/2 + 14N)',
        'sys': {'S': [0.5], 'g': [[2.009, 2.006, 2.002]],
                'Nucs': '14N', 'A': [[10.0, 10.0, 95.0]],
                'lw': [1.0, 0.0]},
        'exp': {'mwFreq': 9.5, 'Range': [330.0, 350.0],
                'nPoints': 1024, 'Harmonic': 1},
        'opt': {'GridSize': 50, 'GridSymmetry': 'Ci', 'Verbosity': 0},
    },
    'cuII_rhombic': {
        'description': 'Cu(II) rhombic g-tensor at X-band (no nuclei)',
        'sys': {'S': [0.5], 'g': [[2.05, 2.10, 2.30]],
                'lw': [2.0, 0.0]},
        'exp': {'mwFreq': 9.5, 'Range': [280.0, 380.0],
                'nPoints': 1024, 'Harmonic': 1},
        'opt': {'GridSize': 50, 'GridSymmetry': 'Ci', 'Verbosity': 0},
    },
    'organic_radical_gstrain': {
        'description': 'Organic radical with anisotropic g-strain',
        'sys': {'S': [0.5], 'g': [[2.0104, 2.0074, 2.0026]],
                'gStrain': [[0.001, 0.0008, 0.0005]],
                'lw': [0.0, 0.0]},
        'exp': {'mwFreq': 9.5, 'Range': [332.0, 342.0],
                'nPoints': 1024, 'Harmonic': 1},
        'opt': {'GridSize': 31, 'Verbosity': 0},
    },
    'triplet_zfs': {
        'description': 'S=1 triplet with axial ZFS',
        'sys': {'S': [1.0], 'g': [[2.0, 2.0, 2.0]],
                'D': [200.0, 200.0, -400.0],
                'lw': [1.0, 0.0]},
        'exp': {'mwFreq': 9.5, 'Range': [300.0, 380.0],
                'nPoints': 1024, 'Harmonic': 0},
        'opt': {'GridSize': 50, 'Verbosity': 0},
    },
}


# ---------------------------------------------------------------------------
# garlic benchmarks (solution CW EPR, fast motion)
# ---------------------------------------------------------------------------

GARLIC_CASES = {
    'nitroxide_fastmotion': {
        'description': 'Nitroxide in solution, fast motion (anisotropic g and A)',
        'sys': {'S': [0.5], 'g': [[2.009, 2.006, 2.002]],
                'Nucs': '14N', 'A': [[16.0, 16.0, 95.0]],
                'tcorr': 1e-10, 'lw': [0.1, 0.0]},
        'exp': {'mwFreq': 9.5, 'Range': [336.0, 342.0],
                'nPoints': 1024, 'Harmonic': 1},
    },
    'multinuc_pattern': {
        'description': 'Solution radical with 14N + 2x 1H',
        'sys': {'S': [0.5], 'g': [[2.0058, 2.0061, 2.0022]],
                'Nucs': '14N,1H,1H',
                'A': [[16.0, 16.0, 95.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]],
                'lw': [0.2, 0.0]},
        'exp': {'mwFreq': 9.5, 'Range': [330.0, 348.0],
                'nPoints': 1024, 'Harmonic': 1},
    },
}


# ---------------------------------------------------------------------------
# chili benchmarks (slow-motion CW EPR, SLE)
# ---------------------------------------------------------------------------

CHILI_CASES = {
    'nitroxide_tcorr1ns': {
        'description': 'Nitroxide slow-motion at tcorr=1 ns',
        'sys': {'S': [0.5], 'g': [[2.009, 2.006, 2.002]],
                'Nucs': '14N', 'A': [[16.0, 16.0, 95.0]],
                'tcorr': 1e-9, 'lw': [0.0, 0.0]},
        'exp': {'mwFreq': 9.5, 'Range': [332.0, 342.0],
                'nPoints': 256, 'Harmonic': 0},
        'opt': {'LLMK': [14, 7, 2, 6], 'max_iter': 200},
    },
    'nitroxide_tcorr10ns': {
        'description': 'Nitroxide slow-motion at tcorr=10 ns',
        'sys': {'S': [0.5], 'g': [[2.009, 2.006, 2.002]],
                'Nucs': '14N', 'A': [[16.0, 16.0, 95.0]],
                'tcorr': 1e-8, 'lw': [0.0, 0.0]},
        'exp': {'mwFreq': 9.5, 'Range': [332.0, 342.0],
                'nPoints': 256, 'Harmonic': 0},
        'opt': {'LLMK': [14, 7, 2, 6], 'max_iter': 300},
    },
}


# ---------------------------------------------------------------------------
# saffron benchmarks (pulse EPR)
# ---------------------------------------------------------------------------

SAFFRON_CASES = {
    '2pESEEM_1H': {
        'description': '2-pulse ESEEM with 1H',
        'sys': {'S': [0.5], 'g': [[2.0023, 2.0023, 2.0023]],
                'Nucs': '1H', 'A': [[3.0, 3.0, 9.0]]},
        'exp': {'Sequence': '2pESEEM', 'Field': 350.0,
                'dt': 0.010, 'tau': 0.0, 'nPoints': 256},
        'opt': {'GridSize': 30, 'TimeDomain': True},
    },
    '3pESEEM_1H': {
        'description': '3-pulse ESEEM with 1H',
        'sys': {'S': [0.5], 'g': [[2.0023, 2.0023, 2.0023]],
                'Nucs': '1H', 'A': [[3.0, 3.0, 9.0]]},
        'exp': {'Sequence': '3pESEEM', 'Field': 350.0,
                'dt': 0.010, 'tau': 0.1, 'nPoints': 256},
        'opt': {'GridSize': 30, 'TimeDomain': True},
    },
    'HYSCORE_1H': {
        'description': 'HYSCORE with 1H',
        'sys': {'S': [0.5], 'g': [[2.0023, 2.0023, 2.0023]],
                'Nucs': '1H', 'A': [[3.0, 3.0, 9.0]]},
        'exp': {'Sequence': 'HYSCORE', 'Field': 324.9,
                'dt': 0.050, 'nPoints': [128, 128],
                'tau': 0.08, 't1': 0.1, 't2': 0.1},
        'opt': {'GridSize': 20, 'TimeDomain': True},
    },
}


# ---------------------------------------------------------------------------
# esfit benchmarks (parameter fitting)
# ---------------------------------------------------------------------------

ESFIT_CASES = {
    'nitroxide_g_fit': {
        'description': 'Fit 3 g-values of a nitroxide spectrum',
        'sys_true': {'S': [0.5], 'g': [[2.009, 2.006, 2.002]],
                     'lw': [1.0, 0.0]},
        'sys_initial': {'S': [0.5], 'g': [[2.005, 2.005, 2.005]],
                        'lw': [1.0, 0.0]},
        'exp': {'mwFreq': 9.5, 'Range': [330.0, 350.0],
                'nPoints': 1024, 'Harmonic': 1},
        'opt': {'GridSize': 31, 'Verbosity': 0},
        'fit_options': {'method': 'simplex', 'verbosity': 0,
                        'autoscale': 'lsq', 'baseline': -1,
                        'max_iter': 200},
        'p0_keys': ['gx', 'gy', 'gz'],
        'p0_init': [2.005, 2.005, 2.005],
        'p_lb':    [1.99, 1.99, 1.99],
        'p_ub':    [2.05, 2.05, 2.05],
    },
}


def all_cases() -> dict:
    """Return all test cases organized by simulator."""
    return {
        'pepper': PEPPER_CASES,
        'garlic': GARLIC_CASES,
        'chili':  CHILI_CASES,
        'saffron': SAFFRON_CASES,
        'esfit': ESFIT_CASES,
    }
