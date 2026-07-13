"""Small reference-resolution protocol shared by field descriptors."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def resolve_handle(reference: Any, resolver: Any, *, where: str) -> Any:
    """Authenticate one declaration through an assembly resolver."""
    from pops.model import Handle

    if not isinstance(reference, Handle):
        raise TypeError("%s must be a declaration Handle" % where)
    resolve = resolver if callable(resolver) else getattr(resolver, "resolve", None)
    if not callable(resolve):
        raise TypeError(
            "%s reference resolution requires a callable resolver or an object exposing "
            "resolve(handle)" % where
        )
    resolved = resolve(reference)
    if not isinstance(resolved, Handle) or not resolved.is_resolved:
        raise TypeError("%s resolver must return a canonical Handle" % where)
    return resolved


def resolve_value(value: Any, resolver: Any, *, where: str) -> Any:
    """Resolve one protocol value while preserving ordinary container shape."""
    from pops.model import Handle

    if isinstance(value, Handle):
        return resolve_handle(value, resolver, where=where)
    protocol = getattr(value, "resolve_references", None)
    if callable(protocol):
        return protocol(resolver)
    if isinstance(value, Mapping):
        return {
            key: resolve_value(item, resolver, where="%s[%r]" % (where, key))
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(
            resolve_value(item, resolver, where="%s[%d]" % (where, index))
            for index, item in enumerate(value)
        )
    if isinstance(value, list):
        return [
            resolve_value(item, resolver, where="%s[%d]" % (where, index))
            for index, item in enumerate(value)
        ]
    if isinstance(value, set):
        return {resolve_value(item, resolver, where="%s[]" % where) for item in value}
    if isinstance(value, frozenset):
        return frozenset(resolve_value(item, resolver, where="%s[]" % where) for item in value)
    return value


def collect_references(value: Any) -> tuple[Any, ...]:
    """Collect typed references through the same small protocol and ordinary containers."""
    from pops.model import Handle

    references: list[Any] = []

    def add(item: Any) -> None:
        if isinstance(item, Handle):
            if item not in references:
                references.append(item)
            return
        protocol = getattr(item, "declaration_references", None)
        if callable(protocol):
            for reference in protocol():
                if not isinstance(reference, Handle):
                    raise TypeError(
                        "%s.declaration_references() must return only Handle values"
                        % type(item).__name__
                    )
                if reference not in references:
                    references.append(reference)
            return
        if isinstance(item, Mapping):
            for nested in item.values():
                add(nested)
            return
        if isinstance(item, (tuple, list, set, frozenset)):
            for nested in item:
                add(nested)

    add(value)
    return tuple(references)


def canonical_qid(reference: Any, *, where: str) -> str:
    """Project an already authenticated reference for reports/lowering."""
    from pops.model import Handle

    if not isinstance(reference, Handle):
        raise TypeError("%s must be a declaration Handle" % where)
    # canonical_identity() deliberately fails for authoring handles. Reporting must never
    # manufacture authority by calling the internal Handle._resolved() escape hatch.
    reference.canonical_identity()
    return reference.qualified_id


def reference_label(reference: Any, *, where: str) -> str:
    """Transparent report label that never pretends an authoring reference is resolved."""
    from pops.model import Handle

    if not isinstance(reference, Handle):
        raise TypeError("%s must be a declaration Handle" % where)
    if reference.is_resolved:
        return reference.qualified_id
    return "pops.handle.unresolved::%s::%s::%s" % (
        reference.owner_path.presentation(),
        reference.kind,
        reference.local_id,
    )


__all__ = [
    "canonical_qid",
    "collect_references",
    "reference_label",
    "resolve_handle",
    "resolve_value",
]
