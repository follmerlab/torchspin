"""Pytest configuration for torchspin_examples smoke tests.

By default only fast tests run (< ~30 s each).
To include slow tests (fitting, matrixperturb, high-spin systems):

    pytest torchspin_examples/tests/ --run-slow

"""
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow examples (fitting, high-spin powder, timing comparisons).",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return  # run everything
    skip_slow = pytest.mark.skip(reason="slow test — run with --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
