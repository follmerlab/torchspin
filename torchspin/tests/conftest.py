"""Shared test configuration and fixtures.

Provides CI guard against silent MATLAB reference file skips.
"""

import os
import warnings
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent.parent.parent / "tests" / "data"


def pytest_collection_modifyitems(config, items):
    """Warn if MATLAB reference data directory is missing or empty."""
    if not DATA_DIR.exists():
        warnings.warn(
            f"MATLAB reference data directory not found: {DATA_DIR}\n"
            "MATLAB validation tests will be silently skipped.",
            stacklevel=1,
        )
    else:
        mat_files = list(DATA_DIR.glob("*.mat"))
        if len(mat_files) == 0:
            warnings.warn(
                f"No .mat files found in {DATA_DIR}\n"
                "MATLAB validation tests will be silently skipped.",
                stacklevel=1,
            )

    # In strict CI mode, fail if .mat files are missing
    if os.environ.get("TORCHSPIN_STRICT_REFS"):
        if not DATA_DIR.exists() or len(list(DATA_DIR.glob("*.mat"))) == 0:
            pytest.fail(
                "TORCHSPIN_STRICT_REFS is set but MATLAB reference data is missing. "
                "MATLAB validation tests cannot run."
            )
