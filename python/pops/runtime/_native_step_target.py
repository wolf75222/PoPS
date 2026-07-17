"""Private protocol separating temporal owners from raw numerical step targets.

The temporal facade owns the controller, restart envelope, and accepted-attempt
accounting.  A raw target only advances the numerical clock.  Keeping that
boundary explicit prevents a controller from calling a facade ``step`` method
that would open a nested temporal transaction.
"""
from __future__ import annotations

import inspect
from typing import Any, Protocol, cast


class _NativeStepTarget(Protocol):
    """Minimal fixed-step numerical target used by every controller."""

    def time(self) -> Any: ...

    def macro_step(self) -> Any: ...

    def step(self, dt: Any) -> Any: ...


class _NativeStepTargetOwner(Protocol):
    """Facade/coordinator explicitly exposing its transaction-free target."""

    def _native_step_target(self) -> Any: ...


def _declares_callable(value: Any, name: str) -> bool:
    """Inspect a declared method without invoking dynamic ``__getattr__`` delegation."""
    try:
        declaration = inspect.getattr_static(value, name)
    except AttributeError:
        return False
    return callable(declaration)


def native_step_target(executor: Any) -> _NativeStepTarget:
    """Return one transaction-free target without guessing private attributes.

    A facade opts into :class:`_NativeStepTargetOwner`; an object that already
    implements the raw target protocol is accepted directly.  There is
    deliberately no fallback to ``executor._s`` or generic delegation.
    """
    if _declares_callable(executor, "_native_step_target"):
        provider = executor._native_step_target
        target = provider()
    else:
        target = executor
    if not all(
        _declares_callable(target, name) for name in ("time", "macro_step", "step")
    ):
        raise TypeError(
            "runtime executor must implement the native step target protocol"
        )
    if target is not executor and _declares_callable(target, "_native_step_target"):
        raise TypeError(
            "native step target owner returned another temporal facade"
        )
    return cast(_NativeStepTarget, target)


__all__ = ["native_step_target"]
