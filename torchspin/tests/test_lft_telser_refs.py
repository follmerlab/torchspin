"""The frozen telser-tools references (data/telser/refs) against live telser-tools.

The parity tests compare torchspin.lft with telser-tools live when it is importable and with the
frozen copies otherwise (see ``_telser_refs``).  Here, with telser-tools importable, every frozen
value must still equal the live one to round-off (``DRIFT_RTOL`` of each array's largest entry;
bit for bit for digests, integers, strings), so a change in telser-tools is noticed and the files
are regenerated (data/telser/generate_telser_refs.py).  The key sets must match the registered
oracles in both directions; that check needs no telser-tools.
"""

from __future__ import annotations

import pytest

from torchspin.tests import _telser_refs as R

# same-machine regeneration is bit-identical; this allows BLAS/LAPACK thread-count and
# platform round-off in telser-tools' own linear algebra (measured: see DATA_PROVENANCE.md)
DRIFT_RTOL = 1e-10

KEYS = R.all_keys()
CASES = [(g, k, fn, c) for g, ks in sorted(KEYS.items()) for k, fn, c in ks]


@pytest.mark.parametrize("group", sorted(KEYS))
def test_frozen_keys_match_registered_oracles(group):
    stored = set(R.frozen_entries(group))
    registered = {k for k, _, _ in KEYS[group]}
    assert stored == registered, dict(missing=sorted(registered - stored), stale=sorted(stored - registered))


@pytest.mark.parametrize("group,k,fn,case", CASES, ids=[f"{g}:{k}" for g, k, _, _ in CASES])
def test_frozen_reference_matches_live_telser_tools(group, k, fn, case):
    if not R.telser_importable():
        pytest.skip("telser-tools not importable: nothing to compare the frozen reference with")
    live = R._decode(*R._encode(R.compute(fn, case)))
    R.assert_same(live, R.frozen_entries(group)[k], DRIFT_RTOL)
