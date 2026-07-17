"""Detached, deeply immutable values at the authoring/compiler boundary.

The public builders intentionally use ordinary Python containers while they are being assembled.
Passing those containers to a compiled artifact would make ``freeze`` cosmetic: a stale list or
dictionary reference could still change what ``bind`` sees.  This module implements the small
protocol used at the boundary:

* mappings become ``MappingProxyType`` values;
* sequences/sets become tuples/frozensets;
* Python record objects are shallow-cloned, then each stored field is detached recursively;
* a retained record must implement ``freeze()`` (or the immutable PoPS value protocol), and its
  freeze hook runs only after every child has been replaced.

The source object is never mutated.  This is deliberately not ``copy.deepcopy``: several valid PoPS
value types (notably native-route strings with structured metadata) are immutable but do not support
the pickle reconstruction protocol used by ``deepcopy``.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import copy
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from types import MappingProxyType
from typing import Any


_ATOMIC = (
    type(None), bool, int, float, complex, str, bytes, range, Decimal, Fraction, Enum,
)


def detached_frozen(value: Any) -> Any:
    """Return a recursively detached, immutable compile-owned value."""
    return _detach(value, {}, set())


def _detach(value: Any, memo: dict[int, Any], active: set[int]) -> Any:
    if isinstance(value, _ATOMIC):
        return value

    # Authenticated identities and core IR type values are immutable by construction.  Keeping
    # them preserves equality/owner semantics while every surrounding mutable container is copied.
    if _intrinsically_immutable(value):
        return value

    marker = id(value)
    if marker in active:
        raise ValueError(
            "cyclic authoring value %s.%s cannot cross the compiled boundary"
            % (type(value).__module__, type(value).__qualname__)
        )
    if marker in memo:
        return memo[marker]

    active.add(marker)
    if isinstance(value, Mapping):
        try:
            result = MappingProxyType({
                _detach(key, memo, active): _detach(item, memo, active)
                for key, item in value.items()
            })
        finally:
            active.remove(marker)
        memo[marker] = result
        return result
    if isinstance(value, (tuple, list)):
        try:
            result = tuple(_detach(item, memo, active) for item in value)
        finally:
            active.remove(marker)
        memo[marker] = result
        return result
    if isinstance(value, (set, frozenset)):
        try:
            result = frozenset(_detach(item, memo, active) for item in value)
        finally:
            active.remove(marker)
        memo[marker] = result
        return result

    try:
        clone = copy(value)
    except Exception as exc:
        raise TypeError(
            "cannot detach %s.%s for AuthoringSnapshot; provide an immutable value object "
            "or a copyable record implementing freeze()"
            % (type(value).__module__, type(value).__qualname__)
        ) from exc
    if clone is value:
        # A foreign object returning itself from __copy__ must explicitly opt into the immutable
        # value protocol; otherwise retaining it would preserve a live authoring reference.
        raise TypeError(
            "%s.%s.__copy__ returned the live authoring object; implement a detached copy or "
            "declare a genuinely immutable value protocol"
            % (type(value).__module__, type(value).__qualname__)
        )
    try:
        # A shallow copy of an already-frozen Descriptor inherits its guard.  The private boundary
        # replaces fields on the copy with detached values, then seals it again.
        if hasattr(clone, "_frozen"):
            object.__setattr__(clone, "_frozen", False)
        for name in _stored_names(type(value), value):
            if name in ("__dict__", "__weakref__", "_frozen") or not hasattr(value, name):
                continue
            object.__setattr__(
                clone, name, _detach(getattr(value, name), memo, active))

        freeze = getattr(clone, "freeze", None)
        if callable(freeze):
            frozen = freeze()
            if frozen is not None and frozen is not clone:
                raise TypeError(
                    "%s.%s.freeze() must seal and return self at the snapshot boundary"
                    % (type(value).__module__, type(value).__qualname__)
                )
            _require_effective_freeze(clone)
        else:
            raise TypeError(
                "%s.%s cannot cross the compiled boundary: retained extension values must "
                "implement freeze() or the immutable PoPS value protocol"
                % (type(value).__module__, type(value).__qualname__)
            )
    finally:
        active.remove(marker)
    memo[marker] = clone
    return clone


def _require_effective_freeze(value: Any) -> None:
    """Verify that ``freeze()`` installed an observable write guard.

    An extension returning ``self`` from a no-op hook is not an immutability protocol.  Probe one
    stored field through the class' normal ``__setattr__``; accepting even an idempotent assignment
    proves the retained record is still mutable.  The probe targets the detached clone only.
    """
    names = [
        name for name in _stored_names(type(value), value)
        if name not in ("__dict__", "__weakref__", "_frozen", "_sealed")
        and hasattr(value, name)
    ]
    name = names[0] if names else "_pops_compiled_freeze_probe"
    current = getattr(value, name, None)
    try:
        setattr(value, name, current)
    except (AttributeError, RuntimeError, TypeError):
        return
    raise TypeError(
        "%s.%s.freeze() did not make retained state immutable (assignment to %r succeeded)"
        % (type(value).__module__, type(value).__qualname__, name)
    )


def _intrinsically_immutable(value: Any) -> bool:
    from pops.model.handles import Handle, OwnerPath

    if isinstance(value, Handle):
        if not value.is_resolved:
            raise ValueError(
                "unresolved authoring Handle %s cannot cross the compiled boundary; resolve it "
                "through Case.resolve first" % value.qualified_id
            )
        return True
    if isinstance(value, OwnerPath):
        if not value.is_canonical:
            raise ValueError(
                "authoring OwnerPath cannot cross the compiled boundary; retain its canonical "
                "identity only"
            )
        return True
    module = type(value).__module__
    if getattr(value, "__pops_ir_immutable__", False) is True:
        return True
    # Route is a structured str subclass; copying it through pickle is unsupported but its fields
    # are constructor-only and it is hashable like the wire token it extends.
    if module == "pops.runtime.routes" and isinstance(value, str):
        return True
    return False


def _stored_names(cls: type, value: Any) -> tuple[str, ...]:
    names = list(getattr(value, "__dict__", {}))
    for owner in cls.__mro__:
        slots = owner.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if name.startswith("__") and not name.endswith("__"):
                name = "_%s%s" % (owner.__name__.lstrip("_"), name)
            if name not in names:
                names.append(name)
    return tuple(names)


__all__ = ["detached_frozen"]
