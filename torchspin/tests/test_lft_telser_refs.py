"""The frozen telser-tools references (data/telser/refs) against live telser-tools.

The parity tests compare torchspin.lft with telser-tools live when it is importable and with the
frozen copies otherwise (see ``_telser_refs``).  Here, with telser-tools importable, every frozen
value must still equal the live one to round-off: ``_telser_refs.DRIFT_RTOL`` (1e-10, relative and
times max(1, the array's largest entry)) for floating-point values, unless the oracle's ``drift``
map gives a measured larger round-off for an ill-conditioned output; exactly for integers,
strings and the structure.  So a change in telser-tools is noticed and the files are regenerated
(data/telser/generate_telser_refs.py).  The key sets must match the registered oracles in both
directions; that check needs no telser-tools.  One test per group, listing every mismatch.
"""

from __future__ import annotations

import pytest

from torchspin.tests import _telser_refs as R

KEYS = R.all_keys()


@pytest.mark.parametrize("group", sorted(KEYS))
def test_frozen_keys_match_registered_oracles(group):
    stored = set(R.frozen_entries(group))
    registered = {k for k, _, _ in KEYS[group]}
    assert stored == registered, dict(missing=sorted(registered - stored), stale=sorted(stored - registered))


@pytest.mark.parametrize("group", sorted(KEYS))
def test_frozen_reference_matches_live_telser_tools(group):
    if not R.telser_importable():
        pytest.skip("telser-tools not importable: nothing to compare the frozen references with")
    frozen = R.frozen_entries(group)
    bad = []
    for k, fn, case in KEYS[group]:
        live = R._decode(*R._encode(R.compute(fn, case)))
        bad += [f"{k}{m}" for m in R.drift_mismatches(live, frozen[k], R.drift_tolerance(fn, k))]
    assert not bad, "frozen reference drifted from live telser-tools (regenerate it?):\n" + "\n".join(bad[:50])
