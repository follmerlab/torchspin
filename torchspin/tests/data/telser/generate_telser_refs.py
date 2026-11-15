"""Freeze the telser-tools outputs used by the torchspin.lft parity tests (refs/<group>.npz).

Run from the torchspin repository root with telser-tools importable, e.g.::

    PYTHONPATH=.:/path/to/telser-tools/src OMP_NUM_THREADS=1 \\
        python torchspin/tests/data/telser/generate_telser_refs.py [group ...]

Every oracle (``torchspin.tests._telser_refs.oracle``) registered by the test modules of the
given groups (``core`` for test_lft_core.py, ...) is evaluated for every case and written, one
compressed npz per group, together with its provenance (telser-tools commit and tree state, date,
Python/NumPy/SciPy/torch versions, thread count, command).  Then record the regeneration in
tests/data/DATA_PROVENANCE.md.  Without arguments every group is regenerated.
"""

from __future__ import annotations

import datetime
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def _telser_provenance() -> dict:
    import telser_tools
    src = Path(telser_tools.__file__).resolve().parent
    out = {"telser_tools_path": str(src)}
    try:
        repo = Path(_git(src, "rev-parse", "--show-toplevel"))
        out["telser_tools_commit"] = _git(repo, "rev-parse", "HEAD")
        out["telser_tools_tree_clean"] = _git(repo, "status", "--porcelain") == ""
    except (subprocess.CalledProcessError, FileNotFoundError):
        out["telser_tools_commit"] = getattr(telser_tools, "__version__", "unknown")
        out["telser_tools_tree_clean"] = None
    return out


def main(argv: list[str]) -> int:
    import numpy as np
    import scipy
    import torch

    from torchspin.tests import _telser_refs as R

    groups = argv or list(R.GROUPS)
    keys = R.all_keys(groups)
    prov = dict(_telser_provenance(),
                date=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
                python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                torch=torch.__version__,              # telser_tools_diff (used by the dpnfit oracles) runs on torch
                omp_num_threads=os.environ.get("OMP_NUM_THREADS"), platform=platform.platform(),
                command="python torchspin/tests/data/telser/generate_telser_refs.py " + " ".join(argv))
    if not prov["telser_tools_tree_clean"]:
        print("warning: the telser-tools tree is not clean (or not a git checkout)", file=sys.stderr)
    for g in groups:
        t0 = time.perf_counter()
        entries = {k: R.compute(fn, c) for k, fn, c in keys[g]}
        p = R.write_group(g, entries, prov)
        print(f"{g:20s} {len(entries):4d} entries  {p.stat().st_size / 1024:8.1f} KiB  "
              f"{time.perf_counter() - t0:6.1f} s")
    total = sum(f.stat().st_size for f in R.REF_DIR.glob("*.npz"))
    print(f"total {total / 1024:.1f} KiB in {R.REF_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
