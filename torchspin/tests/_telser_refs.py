"""Frozen telser-tools outputs for the torchspin.lft parity tests.

torchspin.lft ports Joshua Telser's programs from telser-tools, which CI does not install.  Every
value a parity test takes from telser-tools (results, parsed inputs, constants) comes from an
*oracle*: a function in the test module, registered with :func:`oracle`, that calls telser-tools
and returns plain data (a dict of NumPy arrays and JSON values).  The test asks :func:`ref` for it:

* telser-tools importable (and ``TORCHSPIN_TELSER_REFS`` not ``frozen``): the oracle runs live, so
  local runs still check torchspin against the current telser-tools;
* otherwise: the value stored in ``data/telser/refs/<group>.npz`` by
  ``data/telser/generate_telser_refs.py`` (``TORCHSPIN_TELSER_REFS=frozen`` forces this).

Both paths return the same types (live values go through the same encode/decode as the stored
ones), and the comparisons in the tests are the same.  ``test_lft_telser_refs.py`` checks, when
telser-tools is importable, that every stored value still equals the live one to round-off
(``DRIFT_RTOL``, or the oracle's ``drift`` entry for an output that round-off moves by more) and
that the stored keys match the registered oracles exactly.

Leaves an oracle may return: NumPy arrays (numeric or bool), Python/NumPy scalars, strings,
``None``, and (nested) lists/tuples/dicts of those (tuples come back as lists).  Mostly-zero
arrays can be stored with :func:`sparse` (exact).
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

REF_DIR = Path(__file__).parent / "data" / "telser" / "refs"
ENV = "TORCHSPIN_TELSER_REFS"            # auto (default) | frozen | live

# group -> {oracle name -> (function, case ids)}
ORACLES: dict[str, dict[str, tuple[Callable, tuple]]] = {}
DRIFT_RTOL = 1e-10          # frozen vs live: rtol and rtol * max(1, largest entry) of each float array

# test modules that register oracles (imported by the generator and the drift test); the group
# of an oracle is its module name without the "test_lft_" prefix
MODULES = (
    "test_lft_core", "test_lft_ddn", "test_lft_ddnloop", "test_lft_ddnpow", "test_lft_ddpowh",
    "test_lft_dipole", "test_lft_dlsd5", "test_lft_dpn", "test_lft_dpnfit", "test_lft_dsoxf",
    "test_lft_dtd", "test_lft_tanabe_sugano", "test_lft_telser2torchspin",
)
# not here: test_lft_magnetometry.py compares with telser_tools_diff, which runs on torch, so a frozen copy
# would depend on the torch version (2.2e-10 between torch 2.10 and 2.15 vs rtol 1e-11); it stays live-only
GROUPS = tuple(m.removeprefix("test_lft_") for m in MODULES)


# ------------------------------------------------------------------------------ mode
@lru_cache(maxsize=None)
def telser_importable() -> bool:
    try:
        import telser_tools  # noqa: F401
    except ImportError:
        return False
    return True


def live() -> bool:
    """True when oracles run against telser-tools, False when the frozen references are used."""
    mode = os.environ.get(ENV, "auto").lower()
    if mode not in ("auto", "frozen", "live"):
        raise ValueError(f"{ENV} must be auto, frozen or live, not {mode!r}")
    if mode == "frozen":
        return False
    if mode == "live" and not telser_importable():
        pytest.fail(f"{ENV}=live but telser-tools is not importable")
    return telser_importable()


# Bit-identity of torchspin's copied engines with telser-tools holds within one process (same libm,
# same BLAS). A frozen copy was computed on another platform, whose libm/BLAS may round a
# transcendental or a summation differently by an ulp; the comparison then allows that round-off.
FROZEN_ULPS = 4


def same_bits(got, want, what: str = "", scale_floor: float = 0.0) -> None:
    """Assert ``got == want`` exactly when the oracle ran live; when it is a frozen copy, within
    ``FROZEN_ULPS`` units in the last place of the largest |want| (cross-platform round-off only).
    ``scale_floor``: the scale of the quantities the value was computed from, when larger (a small
    result of O(1) operations carries O(1) round-off)."""
    g = np.asarray(got, dtype=float) if not isinstance(got, np.ndarray) else got
    w = np.asarray(want, dtype=float) if not isinstance(want, np.ndarray) else want
    assert g.shape == w.shape, (what, g.shape, w.shape)
    floats = g.dtype.kind in "fc" and w.dtype.kind in "fc"
    if live() or not floats:
        assert np.array_equal(g, w, equal_nan=floats), what
        return
    nan = np.isnan(w)
    assert np.array_equal(np.isnan(g), nan), what
    scale = max(float(np.abs(w[~nan]).max()) if (~nan).any() else 0.0, scale_floor)
    bound = FROZEN_ULPS * float(np.spacing(max(scale, np.finfo(float).tiny)))
    d = float(np.abs(g[~nan] - w[~nan]).max()) if (~nan).any() else 0.0
    assert d <= bound, f"{what}: frozen copy differs by {d:.3g} (> {FROZEN_ULPS} ulp of {scale:.3g} = {bound:.3g})"


# ------------------------------------------------------------------------------ registry
def oracle(group: str, cases=(None,), drift: dict | None = None):
    """Register ``fn(case) -> plain data`` (or ``fn()`` when ``cases`` is ``(None,)``) under ``group``.

    ``drift`` maps glob patterns (``*`` only; matched against ``key + path``, e.g.
    ``_telser_fit[x]$.stderr``) to the drift tolerance of outputs that round-off moves by more than
    ``DRIFT_RTOL`` between BLAS/LAPACK builds (``None``: not compared); the first match wins."""
    cases = tuple(cases)
    rules = [(re.compile(".*".join(map(re.escape, pat.split("*"))) + r"\Z"), tol) for pat, tol in (drift or {}).items()]

    def deco(fn):
        reg = ORACLES.setdefault(group, {})
        if fn.__name__ in reg and reg[fn.__name__][0] is not fn:
            raise ValueError(f"oracle {group}.{fn.__name__} registered twice")
        reg[fn.__name__] = (fn, cases)
        fn.group = group
        fn.drift_rules = rules
        return fn
    return deco


def drift_tolerance(fn: Callable, k: str) -> Callable[[str], float | None]:
    """path -> drift rtol (None: skip) for the stored entry ``k`` of oracle ``fn``."""
    def tol(path):
        for rx, t in fn.drift_rules:
            if rx.match(k + path):
                return t
        return DRIFT_RTOL
    return tol


def key(fn: Callable, case=None) -> str:
    return fn.__name__ if case is None else f"{fn.__name__}[{case}]"


def compute(fn: Callable, case=None):
    """Run an oracle against telser-tools (no frozen data involved)."""
    return fn() if case is None else fn(case)


def ref(fn: Callable, case=None):
    """The oracle's value: live when telser-tools is importable, else the frozen reference."""
    if live():
        return _decode(*_encode(compute(fn, case)))
    return _decode(*_frozen(fn.group)[key(fn, case)])


# ------------------------------------------------------------------------------ sparse arrays
def sparse(a) -> dict:
    """A mostly-zero array as {shape, index, value} (exact; ``dense`` restores it bit for bit)."""
    a = np.asarray(a)
    idx = np.flatnonzero(a)
    return dict(shape=list(a.shape), index=idx.astype(np.int64), value=a.reshape(-1)[idx])


def dense(d: dict) -> np.ndarray:
    out = np.zeros(int(np.prod(d["shape"])), dtype=d["value"].dtype)
    out[d["index"]] = d["value"]
    return out.reshape(d["shape"])


# ------------------------------------------------------------------------------ encode / decode
def _encode(value) -> tuple[Any, dict[str, np.ndarray]]:
    """Plain data -> (JSON-able tree, {array key: array})."""
    arrays: dict[str, np.ndarray] = {}

    def enc(v, path):
        if isinstance(v, np.ndarray):
            if v.dtype.kind not in "biufc":
                raise TypeError(f"{path}: unsupported array dtype {v.dtype}")
            k = f"a{len(arrays)}"
            arrays[k] = v
            return {"__array__": k}
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, complex):
            return {"__complex__": [v.real, v.imag]}
        if v is None or isinstance(v, (bool, int, float, str)):
            return v
        if isinstance(v, (list, tuple)):
            return [enc(x, f"{path}[{i}]") for i, x in enumerate(v)]
        if isinstance(v, dict):
            if not all(isinstance(k, str) and not k.startswith("__") for k in v):
                raise TypeError(f"{path}: dict keys must be str without a leading '__'")
            return {k: enc(x, f"{path}.{k}") for k, x in v.items()}
        raise TypeError(f"{path}: unsupported oracle value of type {type(v).__name__}")

    return enc(value, "$"), arrays


def _decode(tree, arrays):
    def dec(v):
        if isinstance(v, list):
            return [dec(x) for x in v]
        if isinstance(v, dict):
            if "__array__" in v:
                return np.array(arrays[v["__array__"]])
            if "__complex__" in v:
                return complex(*v["__complex__"])
            return {k: dec(x) for k, x in v.items()}
        return v
    return dec(tree)


# ------------------------------------------------------------------------------ storage
def path_for(group: str) -> Path:
    return REF_DIR / f"{group}.npz"


@lru_cache(maxsize=None)
def _frozen(group: str) -> dict[str, tuple[Any, dict[str, np.ndarray]]]:
    p = path_for(group)
    if not p.exists():
        pytest.fail(f"frozen telser-tools reference {p} is missing; run data/telser/generate_telser_refs.py")
    with np.load(p, allow_pickle=False) as z:
        meta = json.loads(str(z["__json__"]))
        flat = {dt: z[dt] for dt in z.files if dt != "__json__"}
    out = {}
    for k, tree in meta["entries"].items():
        out[k] = (tree, {n: flat[dt][off:off + int(np.prod(shape))].reshape(shape)
                         for n, (dt, off, shape) in meta["arrays"][k].items()})
    return _Missing(out, p)


class _Missing(dict):
    def __init__(self, data, path):
        super().__init__(data)
        self.path = path

    def __missing__(self, k):
        pytest.fail(f"{k} is not in {self.path}; regenerate it with data/telser/generate_telser_refs.py")


@lru_cache(maxsize=None)
def frozen_entries(group: str) -> dict:
    """{key: decoded value} of a stored group."""
    return {k: _decode(*v) for k, v in _frozen(group).items()}


def frozen_provenance(group: str) -> dict:
    with np.load(path_for(group), allow_pickle=False) as z:
        return json.loads(str(z["__json__"]))["provenance"]


def write_group(group: str, entries: dict, provenance: dict) -> Path:
    """Store ``{key: plain data}`` as one compressed npz without pickles: the JSON tree as a 0-d str array
    and the arrays packed into one flat array per dtype (offset and shape in the JSON)."""
    trees, names, parts, sizes = {}, {}, {}, {}
    for k, v in entries.items():
        tree, arrays = _encode(v)
        trees[k], names[k] = tree, {}
        for n in sorted(arrays):
            a = np.asarray(arrays[n])
            dt = a.dtype.str
            names[k][n] = [dt, sizes.get(dt, 0), list(a.shape)]
            parts.setdefault(dt, []).append(a.reshape(-1))
            sizes[dt] = sizes.get(dt, 0) + a.size
    meta = json.dumps({"provenance": provenance, "entries": trees, "arrays": names}, sort_keys=True)
    REF_DIR.mkdir(parents=True, exist_ok=True)
    p = path_for(group)
    np.savez_compressed(p, __json__=np.array(meta), **{dt: np.concatenate(v) for dt, v in parts.items()})
    _frozen.cache_clear()
    frozen_entries.cache_clear()
    return p


def all_keys(groups=GROUPS) -> dict[str, list[tuple[str, Callable, Any]]]:
    """{group: [(key, fn, case), ...]} of every registered oracle case of ``groups``."""
    import importlib
    unknown = sorted(set(groups) - set(GROUPS))
    if unknown:
        raise ValueError(f"unknown groups {unknown}; known: {list(GROUPS)}")
    for g in groups:
        importlib.import_module(f"torchspin.tests.test_lft_{g}")
    return {g: [(key(fn, c), fn, c) for fn, cases in ORACLES.get(g, {}).values() for c in cases] for g in groups}


# ------------------------------------------------------------------------------ telser-tools objects as data
def freeze_obj(obj):
    """A telser-tools input object (dataclass, possibly nested; tuples, lists, dicts, arrays, scalars) as
    plain data that :func:`thaw_obj` turns back into an equivalent object (attribute access and
    ``dataclasses.replace`` work; methods and properties of the original class do not exist)."""
    import dataclasses
    import enum
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {"_dataclass": type(obj).__name__,
                "fields": {f.name: freeze_obj(getattr(obj, f.name)) for f in dataclasses.fields(obj)}}
    if isinstance(obj, enum.Enum):
        raise TypeError(f"enum {obj!r}: store its value explicitly")
    if isinstance(obj, tuple):
        return {"_tuple": [freeze_obj(x) for x in obj]}
    if isinstance(obj, list):
        return [freeze_obj(x) for x in obj]
    if isinstance(obj, dict):
        return {"_dict": [[freeze_obj(k), freeze_obj(v)] for k, v in obj.items()]}
    return obj


@lru_cache(maxsize=None)
def _thawed_class(name: str, fields: tuple):
    import dataclasses
    return dataclasses.make_dataclass(name, list(fields))


def thaw_obj(d):
    if isinstance(d, list):
        return [thaw_obj(x) for x in d]
    if isinstance(d, dict):
        if "_dataclass" in d:
            f = d["fields"]
            return _thawed_class(d["_dataclass"], tuple(f))(**{k: thaw_obj(v) for k, v in f.items()})
        if "_tuple" in d:
            return tuple(thaw_obj(x) for x in d["_tuple"])
        if "_dict" in d:
            return {thaw_obj(k): thaw_obj(v) for k, v in d["_dict"]}
        return {k: thaw_obj(v) for k, v in d.items()}
    return d


# ------------------------------------------------------------------------------ drift comparison
def drift_mismatches(live_value, frozen_value, tol: Callable[[str], float | None], path: str = "$") -> list[str]:
    """Where a live telser-tools output differs from its frozen copy: floating-point arrays beyond
    ``tol(path)`` (relative, and absolute times max(1, largest entry)), floats beyond rel/abs ``tol(path)``,
    anything else (integers, bools, strings, None, shapes, keys) if not equal."""
    a, b = live_value, frozen_value
    t = tol(path)
    if t is None:
        return []
    if isinstance(b, np.ndarray):
        if not isinstance(a, np.ndarray) or a.shape != b.shape or a.dtype != b.dtype:
            return [f"{path}: {getattr(a, 'dtype', type(a))}{getattr(a, 'shape', '')} vs {b.dtype}{b.shape}"]
        if b.dtype.kind in "biu":
            return [] if np.array_equal(a, b) else [f"{path}: integer/bool values differ"]
        fin = np.isfinite(b)
        if not (np.array_equal(np.isfinite(a), fin) and np.array_equal(a[~fin], b[~fin], equal_nan=True)):
            return [f"{path}: non-finite entries differ"]
        if not fin.any():
            return []
        scale = max(np.abs(b[fin]).max(), 1.0)
        d = np.abs(a[fin] - b[fin])
        bad = d > t * scale + t * np.abs(b[fin])
        return [f"{path}: max |live - frozen| = {d.max():.3g} (scale {scale:.3g}, rtol {t:g})"] if bad.any() else []
    if isinstance(b, dict) and b.keys() == {"shape", "index", "value"}:          # sparse(): compare dense
        if not (isinstance(a, dict) and a.keys() == b.keys()):
            return [f"{path}: not a sparse array"]
        return drift_mismatches(dense(a), dense(b), tol, path)
    if isinstance(b, dict):
        if not (isinstance(a, dict) and a.keys() == b.keys()):
            return [f"{path}: keys {sorted(a) if isinstance(a, dict) else type(a).__name__} vs {sorted(b)}"]
        return [m for k in b for m in drift_mismatches(a[k], b[k], tol, f"{path}.{k}")]
    if isinstance(b, list):
        if not (isinstance(a, list) and len(a) == len(b)):
            return [f"{path}: list of {len(a) if isinstance(a, list) else type(a).__name__} vs {len(b)}"]
        return [m for i, (x, y) in enumerate(zip(a, b)) for m in drift_mismatches(x, y, tol, f"{path}[{i}]")]
    if isinstance(b, float) and isinstance(a, float):
        ok = a == pytest.approx(b, rel=t, abs=t, nan_ok=True)
        return [] if ok else [f"{path}: {a!r} vs {b!r} (rtol {t:g})"]
    return [] if (type(a) is type(b) and a == b) else [f"{path}: {a!r} vs {b!r}"]
