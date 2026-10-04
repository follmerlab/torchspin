"""torch.compile integration for torchspin.

Provides a ``maybe_compile`` decorator that conditionally applies
``torch.compile`` to pure-PyTorch numerical kernels.

Compilation is **off by default** to preserve bit-exact numerical
reproducibility.  Enable it by setting the environment variable::

    export TORCHSPIN_COMPILE=1

or by calling ``set_compile_enabled(True)`` at runtime before any
compiled function is first invoked.

Requirements: PyTorch ≥ 2.0.  On older versions, ``maybe_compile``
is a no-op regardless of the flag.
"""
from __future__ import annotations

import functools
import os
from typing import Any, Callable, TypeVar

import torch

F = TypeVar("F", bound=Callable[..., Any])

# ── Runtime flag ──────────────────────────────────────────────────────────────

_compile_enabled: bool | None = None  # None = read from env on first call


def _resolve_flag() -> bool:
    global _compile_enabled
    if _compile_enabled is None:
        _compile_enabled = os.environ.get("TORCHSPIN_COMPILE", "0") == "1"
    return _compile_enabled


def set_compile_enabled(enabled: bool) -> None:
    """Enable or disable ``torch.compile`` for torchspin kernels.

    Must be called *before* the first invocation of any compiled function
    (i.e. at import time or early in a script).
    """
    global _compile_enabled
    _compile_enabled = enabled


def is_compile_available() -> bool:
    """Return True if torch.compile is available.

    pyproject.toml requires PyTorch ≥ 2.0, so this is always True in
    production; kept for defensive use in environments with unusual
    torch versions.
    """
    return hasattr(torch, "compile")


# ── Decorator ─────────────────────────────────────────────────────────────────

def maybe_compile(fn: F | None = None, **compile_kwargs: Any) -> F | Callable[[F], F]:
    """Conditionally apply ``torch.compile`` to a function.

    Usage::

        @maybe_compile
        def my_kernel(x: torch.Tensor) -> torch.Tensor:
            ...

        # With options:
        @maybe_compile(mode="reduce-overhead")
        def my_kernel(x: torch.Tensor) -> torch.Tensor:
            ...

    The decorated function behaves identically to the original when
    compilation is disabled or unavailable.
    """
    def decorator(func: F) -> F:
        # Mutable container so the closure can update the reference
        _compiled: list = [None]

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Lazy: on first call, decide whether to swap in compiled version
            if _compiled[0] is None:
                if _resolve_flag() and is_compile_available():
                    _compiled[0] = torch.compile(func, **compile_kwargs)
                else:
                    _compiled[0] = func
            return _compiled[0](*args, **kwargs)

        # Expose the original for testing / introspection
        wrapper._original = func  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    if fn is not None:
        # Called as @maybe_compile (no parentheses)
        return decorator(fn)
    # Called as @maybe_compile(...) (with keyword args)
    return decorator
