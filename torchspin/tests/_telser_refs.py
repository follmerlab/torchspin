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
telser-tools is importable, that every stored value still equals the live one to round-off and
that the stored keys match the registered oracles exactly.

Leaves an oracle may return: NumPy arrays (numeric or bool), Python/NumPy scalars, strings,
``None``, and (nested) lists/tuples/dicts of those (tuples come back as lists).  ``Exact(a)``
stores only a SHA-256 digest of a large array that a test compares bit for bit; in live mode the
array itself is returned, so a failure shows the difference.  Compare with :func:`assert_identical`.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

REF_DIR = Path(__file__).parent / "data" / "telser" / "refs"
ENV = "TORCHSPIN_TELSER_REFS"            # auto (default) | frozen | live

# group -> {oracle name -> (function, case ids)}
ORACLES: dict[str, dict[str, tuple[Callable, tuple]]] = {}

# test modules that register oracles (imported by the generator and the drift test); the group
# of an oracle is its module name without the "test_lft_" prefix
MODULES = (
    "test_lft_core", "test_lft_ddn", "test_lft_ddnloop", "test_lft_ddnpow", "test_lft_ddpowh",
    "test_lft_dipole", "test_lft_dlsd5", "test_lft_dpn", "test_lft_dpnfit", "test_lft_dsoxf",
    "test_lft_dtd", "test_lft_magnetometry", "test_lft_tanabe_sugano", "test_lft_telser2torchspin",
)
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


# ------------------------------------------------------------------------------ registry
def oracle(group: str, cases=(None,)):
    """Register ``fn(case) -> plain data`` (or ``fn()`` when ``cases`` is ``(None,)``) under ``group``."""
    cases = tuple(cases)

    def deco(fn):
        reg = ORACLES.setdefault(group, {})
        if fn.__name__ in reg and reg[fn.__name__][0] is not fn:
            raise ValueError(f"oracle {group}.{fn.__name__} registered twice")
        reg[fn.__name__] = (fn, cases)
        fn.group = group
        return fn
    return deco


def key(fn: Callable, case=None) -> str:
    return fn.__name__ if case is None else f"{fn.__name__}[{case}]"


def compute(fn: Callable, case=None):
    """Run an oracle against telser-tools (no frozen data involved)."""
    return fn() if case is None else fn(case)


def ref(fn: Callable, case=None):
    """The oracle's value: live when telser-tools is importable, else the frozen reference."""
    if live():
        return _decode(*_encode(compute(fn, case)), keep_exact=True)
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


# ------------------------------------------------------------------------------ exact (digest) arrays
@dataclass(frozen=True)
class Exact:
    """Marks an array that is compared bit for bit and stored as a digest only."""
    array: np.ndarray


@dataclass(frozen=True)
class Digest:
    sha256: str
    dtype: str
    shape: tuple


def _canonical(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    if a.dtype.kind in "fc":
        a = a + a.dtype.type(0)                         # -0.0 -> +0.0 (assert_array_equal treats them as equal)
    return a


def digest(a) -> Digest:
    a = _canonical(np.asarray(a))
    h = hashlib.sha256(f"{a.dtype.str}{a.shape}".encode())
    h.update(a.tobytes())
    return Digest(h.hexdigest(), a.dtype.str, tuple(a.shape))


def assert_identical(ours, want) -> None:
    """``np.testing.assert_array_equal`` against an array or (frozen) against its digest."""
    ours = np.asarray(ours)
    if isinstance(want, Digest):
        assert ours.shape == want.shape, (ours.shape, want.shape)
        conv = ours.astype(want.dtype)
        assert np.array_equal(conv, ours), "dtype conversion changed values"
        assert digest(conv) == want, "array differs from the frozen telser-tools array (bitwise)"
    else:
        np.testing.assert_array_equal(ours, want)


# ------------------------------------------------------------------------------ encode / decode
def _encode(value) -> tuple[Any, dict[str, np.ndarray]]:
    """Plain data -> (JSON-able tree, {array key: array})."""
    arrays: dict[str, np.ndarray] = {}

    def enc(v, path):
        if isinstance(v, Exact):
            return {"__exact__": [v.array], "__digest__": digest(v.array).__dict__}
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


def _decode(tree, arrays, keep_exact: bool = False):
    def dec(v):
        if isinstance(v, list):
            return [dec(x) for x in v]
        if isinstance(v, dict):
            if "__array__" in v:
                return np.array(arrays[v["__array__"]])
            if "__complex__" in v:
                return complex(*v["__complex__"])
            if "__digest__" in v:
                if keep_exact and "__exact__" in v:
                    return np.array(v["__exact__"][0])
                d = v["__digest__"]
                return Digest(d["sha256"], d["dtype"], tuple(d["shape"]))
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
        out = {}
        for k, tree in meta["entries"].items():
            names = meta["arrays"][k]
            out[k] = (tree, {n: z[f"{k}/{n}"] for n in names})
    return _Missing(out, p)


class _Missing(dict):
    def __init__(self, data, path):
        super().__init__(data)
        self.path = path

    def __missing__(self, k):
        pytest.fail(f"{k} is not in {self.path}; regenerate it with data/telser/generate_telser_refs.py")


@lru_cache(maxsize=None)
def frozen_entries(group: str) -> dict:
    """{key: decoded value} of a stored group (digests stay digests)."""
    return {k: _decode(*v) for k, v in _frozen(group).items()}


def frozen_provenance(group: str) -> dict:
    with np.load(path_for(group), allow_pickle=False) as z:
        return json.loads(str(z["__json__"]))["provenance"]


def write_group(group: str, entries: dict, provenance: dict) -> Path:
    """Store ``{key: plain data}`` as one compressed npz (no pickles; the JSON tree is a 0-d str array)."""
    trees, names, flat = {}, {}, {}
    for k, v in entries.items():
        tree, arrays = _encode(v)
        trees[k], names[k] = _strip_exact(tree), sorted(arrays)
        for n, a in arrays.items():
            flat[f"{k}/{n}"] = a
    meta = json.dumps({"provenance": provenance, "entries": trees, "arrays": names}, sort_keys=True)
    REF_DIR.mkdir(parents=True, exist_ok=True)
    p = path_for(group)
    np.savez_compressed(p, __json__=np.array(meta), **flat)
    _frozen.cache_clear()
    frozen_entries.cache_clear()
    return p


def _strip_exact(tree):
    if isinstance(tree, list):
        return [_strip_exact(x) for x in tree]
    if isinstance(tree, dict):
        return {k: _strip_exact(v) for k, v in tree.items() if k != "__exact__"}
    return tree


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
def assert_same(live_value, frozen_value, rtol: float, path: str = "$") -> None:
    """Live telser-tools output vs its frozen copy: arrays to ``rtol`` of their scale, the rest exactly
    (floats to ``rtol``), digests bit for bit."""
    a, b = live_value, frozen_value
    if isinstance(b, Digest):
        assert isinstance(a, Digest) and a == b, f"{path}: bitwise digest changed"
        return
    if isinstance(b, np.ndarray):
        assert isinstance(a, np.ndarray), f"{path}: {type(a).__name__} vs array"
        assert a.shape == b.shape and a.dtype == b.dtype, (path, a.shape, b.shape, a.dtype, b.dtype)
        if b.dtype.kind in "biu":
            np.testing.assert_array_equal(a, b, err_msg=path)
        else:
            fin = np.isfinite(b)
            np.testing.assert_array_equal(np.isfinite(a), fin, err_msg=path)
            np.testing.assert_array_equal(a[~fin], b[~fin], err_msg=path)
            scale = np.abs(b[fin]).max() if fin.any() else 0.0
            np.testing.assert_allclose(a[fin], b[fin], rtol=rtol, atol=rtol * scale, err_msg=path)
        return
    if isinstance(b, dict) and b.keys() == {"shape", "index", "value"}:          # sparse(): compare dense
        assert isinstance(a, dict) and a.keys() == b.keys(), path
        assert_same(dense(a), dense(b), rtol, path)
        return
    if isinstance(b, dict):
        assert isinstance(a, dict) and a.keys() == b.keys(), (path, sorted(a) if isinstance(a, dict) else a, sorted(b))
        for k in b:
            assert_same(a[k], b[k], rtol, f"{path}.{k}")
        return
    if isinstance(b, list):
        assert isinstance(a, list) and len(a) == len(b), (path, a, b)
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, rtol, f"{path}[{i}]")
        return
    if isinstance(b, float) and isinstance(a, float):
        assert a == pytest.approx(b, rel=rtol, abs=0.0, nan_ok=True), (path, a, b)
        return
    assert type(a) is type(b) and a == b, (path, a, b)
