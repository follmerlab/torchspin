"""The Examples sections of the public docstrings run as doctests.

Modules are added here once their docstrings have been standardized and their
examples checked against the implementation; extend DOCTEST_MODULES as more of
the public API is covered.
"""
import doctest
import importlib

import pytest

DOCTEST_MODULES = [
    'torchspin.autograd',
    'torchspin.esfit',
    'torchspin.experiment',
    'torchspin.garlic',
    'torchspin.gridcheck',
    'torchspin.lft.bridge',
    'torchspin.lft.ddn',
    'torchspin.lft.ddnfit',
    'torchspin.lft.ddnloop',
    'torchspin.lft.ddnmag',
    'torchspin.lft.ddnpow',
    'torchspin.lft.ddnsus',
    'torchspin.lft.ddpowh',
    'torchspin.lft.ddpowj',
    'torchspin.lft.dipole',
    'torchspin.lft.dlsd5',
    'torchspin.lft.dpn',
    'torchspin.lft.dpnfit',
    'torchspin.lft.dsoxf',
    'torchspin.lft.dtd',
    'torchspin.lft.fit',
    'torchspin.lft.grids',
    'torchspin.lft.ligand_field',
    'torchspin.lft.plot',
    'torchspin.lft.tanabe_sugano',
    'torchspin.lft.telser2torchspin',
    'torchspin.lft.units',
    'torchspin.pepper',
    'torchspin.spinsystem',
]


@pytest.mark.parametrize('name', DOCTEST_MODULES)
def test_docstring_examples(name):
    module = importlib.import_module(name)
    result = doctest.testmod(module, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
                             verbose=False, report=True)
    assert result.attempted > 0, f'{name}: no doctest examples found'
    assert result.failed == 0, f'{name}: {result.failed} of {result.attempted} doctest examples failed'
