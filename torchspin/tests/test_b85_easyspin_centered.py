"""B85 (0.9.35): the EasySpin comparisons moved by the centered projection bins, checked
against EasySpin's spectra mapped onto centered hat bins.

Each comparison in ``_b85.MOVED`` stays as it was and is strict xfail in the normal run
(conftest).  Here the moved tests of one file are re-run in a subprocess with
``TORCHSPIN_B85_CENTERED=1``: their EasySpin reference spectra are mapped onto centered
bins as they are loaded (``_b85.easyspin_centered``), and every one must pass with its
original tolerance, except those in ``_b85.CENTERED_RESIDUAL`` (strict xfail there, with
the measured numbers): tolerances calibrated to EasySpin's box binning at a level no
centered convention meets (stick spectra, 1e-5..1e-9 bounds).
"""
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest

from torchspin.tests import _b85

HERE = Path(__file__).resolve().parent


def _by_file():
    out = defaultdict(list)
    for nid in sorted(_b85.MOVED):
        out[nid.split('::')[0]].append(nid)
    return dict(out)


_FILES = _by_file()


@pytest.mark.slow
@pytest.mark.parametrize('fname', sorted(_FILES))
def test_moved_easyspin_comparisons_vs_centered_easyspin(fname):
    nodes = [str(HERE / nid) for nid in _FILES[fname]]
    env = dict(os.environ, **{_b85.CENTERED_ENV: '1'})
    r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '-rfX', *nodes],
                       cwd=str(HERE.parents[1]), env=env, capture_output=True, text=True)
    tail = '\n'.join(r.stdout.splitlines()[-25:])
    print(tail)
    assert r.returncode == 0, tail
