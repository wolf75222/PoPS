"""Shared deep-freeze mechanics for Problem registries."""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


def flatten_freeze_members(*values: Any) -> list[Any]:
    """Flatten declaration containers without interpreting descriptor objects."""
    result = []

    def visit(value: Any) -> None:
        if value is None:
            return
        if callable(getattr(value, "freeze", None)):
            result.append(value)
        elif isinstance(value, Mapping):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                visit(item)
        else:
            result.append(value)

    for value in values:
        visit(value)
    return result


class FreezableRegistry:
    """A registry whose own flag and member freezes commit atomically."""

    _frozen = False

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            if name != "_frozen" or value is not True:
                raise RuntimeError("pops.Problem registry is frozen: cannot change %s" % name)
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_frozen", False):
            raise RuntimeError("pops.Problem registry is frozen: cannot delete %s" % name)
        object.__delattr__(self, name)

    def freeze(self) -> Any:
        from pops.problem._freeze_transaction import freeze_atomically

        if self._frozen:
            return self
        members = tuple(self._freezable_members())

        def commit() -> None:
            self._prepare_freeze()
            for member in members:
                member_freeze = getattr(member, "freeze", None)
                if callable(member_freeze):
                    member_freeze()
            replacements = {
                name: _immutable_copy(value)
                for name, value in vars(self).items()
                if name != "_frozen" and isinstance(
                    value, (Mapping, list, tuple, set, frozenset))
            }
            for name, value in replacements.items():
                object.__setattr__(self, name, value)
            object.__setattr__(self, "_frozen", True)

        freeze_atomically((*members, self), commit)
        return self

    def _freezable_members(self) -> Any:
        return ()

    def _prepare_freeze(self) -> None:
        """Materialize derived registry state inside the atomic freeze transaction."""

    def _guard_frozen(self, what: Any) -> None:
        if self._frozen:
            raise RuntimeError(
                "pops.Problem registry is frozen (ADC-563): cannot %s after pops.compile froze the "
                "Problem; author a fresh Problem and recompile." % what)


def _immutable_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({
            _immutable_copy(key): _immutable_copy(item)
            for key, item in value.items()
        })
    if isinstance(value, (list, tuple)):
        return tuple(_immutable_copy(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_immutable_copy(item) for item in value)
    return value


def inspection_copy(value: Any) -> Any:
    """Return a deeply detached, ordinary-container view of frozen registry data.

    Registry storage deliberately uses mapping proxies, tuples and frozensets after freeze.  Those
    containers must never leak into the public inspection bridge: callers expect a mutable copy and
    ``Problem.to_dict()`` promises JSON-compatible container types.
    """
    if isinstance(value, Mapping):
        return {key: inspection_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [inspection_copy(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [inspection_copy(item) for item in value]
    return value


__all__ = ["FreezableRegistry", "flatten_freeze_members", "inspection_copy"]
