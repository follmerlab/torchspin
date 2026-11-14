"""Shared test configuration and fixtures.

Provides CI guard against silent MATLAB reference file skips.
"""

import os
import warnings
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent.parent.parent / "tests" / "data"


def pytest_collection_modifyitems(config, items):
    """Warn if MATLAB reference data directory is missing or empty; mark the B85 moves."""
    _mark_b85_moved(items)
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


def _mark_b85_moved(items):
    """Strict xfail for the EasySpin comparisons moved by the centered projection bins
    (B85, 0.9.35); node ids and measured numbers in ``_b85.MOVED``.  In the
    centered-reference mode (``TORCHSPIN_B85_CENTERED``, used by
    test_b85_easyspin_centered.py) the moved tests run against EasySpin mapped onto
    centered bins and only ``_b85.CENTERED_RESIDUAL`` is marked."""
    from torchspin.tests import _b85
    centered = bool(os.environ.get(_b85.CENTERED_ENV))
    table = _b85.CENTERED_RESIDUAL if centered else _b85.MOVED
    for item in items:
        nid = item.nodeid.split('tests/')[-1]
        if nid in table:
            item.add_marker(_b85.xfail_b85(table[nid]))


if os.environ.get('TORCHSPIN_B85_CENTERED'):
    from torchspin.tests._b85 import install_centered_loadmat
    install_centered_loadmat()
