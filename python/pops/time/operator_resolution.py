"""Owner-safe resolution of typed model operators inside a time Program.

Public authoring routes accept :class:`pops.model.OperatorHandle` values, never
free operator-name strings.  A handle is meaningful only relative to the exact
registry that declared it: a matching local name is not sufficient.  This
module centralizes that boundary so ``P.call``, ``P.linear_source``, typed RHS
terms, condensed operators, and ``pops.lib.time`` macros cannot drift into
slightly different validation rules.

Private ``_...`` lowering seams may still use registry-local string tokens.
Those strings are resolved by :func:`resolve_registered_operator`; they are not
accepted by :func:`resolve_operator_handle`.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _bound_registry(program: Any, where: str, owner: Any = None) -> tuple[Any, Any]:
    """Return the exact ``(registry, owner)`` selected by semantic provenance."""
    registries = getattr(program, "_operator_registries", None) or {}
    if not registries:
        raise ValueError(
            "%s: no operators are bound; call P.bind_operators(the_declaring_model) first"
            % where)
    if owner is None:
        if len(registries) != 1:
            raise ValueError(
                "%s: operator owner is ambiguous across %d bound model registries; pass an "
                "OperatorHandle or provide block-qualified Program values"
                % (where, len(registries)))
        owner, registry = next(iter(registries.items()))
    else:
        try:
            registry = registries[owner]
        except KeyError:
            known = ", ".join(str(item) for item in registries) or "<none>"
            raise ValueError(
                "%s: no operator registry is bound for owner %s (bound owners: %s)"
                % (where, owner, known)) from None
    registry_owner = getattr(registry, "owner_path", None)
    if registry_owner is None:
        raise ValueError(
            "%s: the bound operator registry has no qualified owner; bind the declaring "
            "Module or physics Model instead of an ownerless registry" % where)
    if owner != registry_owner:
        raise ValueError(
            "%s: the bound source owner %s does not match its registry owner %s"
            % (where, owner, registry_owner))
    return registry, owner


def _owner_from_values(values: Any, where: str) -> Any:
    """Infer exactly one model owner from block-qualified Program arguments."""
    owners = {
        value.block.model_owner_path
        for value in values
        if getattr(value, "block", None) is not None
    }
    if len(owners) > 1:
        raise ValueError(
            "%s: arguments span multiple model owners %s; select a typed cross-model operator "
            "whose protocol explicitly supports that join"
            % (where, sorted(str(owner) for owner in owners)))
    return next(iter(owners)) if owners else None


def _normalize_kinds(expected_kinds: Any) -> frozenset[str] | None:
    if expected_kinds is None:
        return None
    if isinstance(expected_kinds, str):
        return frozenset((expected_kinds,))
    if not isinstance(expected_kinds, Iterable):
        raise TypeError("expected_kinds must be a string, an iterable of strings, or None")
    kinds = frozenset(expected_kinds)
    if not kinds or any(not isinstance(kind, str) or not kind for kind in kinds):
        raise TypeError("expected_kinds must contain one or more non-empty strings")
    return kinds


def resolve_registered_operator(
    program: Any,
    name: Any,
    *,
    where: str,
    expected_kinds: Any = None,
    values: Any = (),
) -> Any:
    """Resolve a registry-local name and enforce its expected operator kind.

    This helper is for private lowering tokens and for typed RHS descriptors
    whose public type carries a declared term category but no owner handle.  It
    deliberately performs no coercion and never accepts an empty/non-string
    name.
    """
    if not isinstance(name, str) or not name:
        raise TypeError("%s: operator name must be a non-empty string" % where)
    registry, _ = _bound_registry(program, where, _owner_from_values(values, where))
    operator = registry.get(name)
    kinds = _normalize_kinds(expected_kinds)
    if kinds is not None and operator.kind not in kinds:
        raise ValueError(
            "%s: registered operator %r has kind %r; expected one of %s"
            % (where, name, operator.kind, sorted(kinds)))
    return operator


def resolve_operator_handle(
    program: Any,
    handle: Any,
    *,
    where: str,
    expected_kinds: Any = None,
    expected_signature: Any = None,
    values: Any = (),
) -> Any:
    """Resolve and validate one public :class:`OperatorHandle` selector.

    Validation is intentionally redundant with the information in the handle:
    the exact owner, kind, and (when carried) structural signature must agree
    with the bound registry.  This prevents a same-named handle from another
    model, or a forged/stale handle, from selecting unrelated physics.
    """
    from pops.model import OperatorHandle

    if not isinstance(handle, OperatorHandle):
        raise TypeError(
            "%s: expected a pops.model.OperatorHandle, got %r" % (where, handle))
    registry, owner = _bound_registry(program, where, handle.owner_path)
    if handle.owner_path != owner:
        raise ValueError(
            "%s: operator handle %r belongs to owner %s, but this Program is bound to %s"
            % (where, handle.name, handle.owner_path, owner))
    argument_owner = _owner_from_values(values, where)
    if argument_owner is not None and argument_owner != owner:
        raise ValueError(
            "%s: operator handle %r belongs to owner %s, but its block-qualified arguments "
            "instantiate model owner %s"
            % (where, handle.name, owner, argument_owner))
    registry_name = registry.target_for_handle(handle.name)
    if handle.registered_operator_name != registry_name:
        raise ValueError(
            "%s: operator handle %r targets %r, but the bound registry authenticates target %r"
            % (where, handle.name, handle.registered_operator_name, registry_name))
    operator = registry.get(registry_name)
    if handle.kind != operator.kind:
        raise ValueError(
            "%s: operator handle %r declares kind %r, but the bound registry declares %r"
            % (where, handle.name, handle.kind, operator.kind))
    kinds = _normalize_kinds(expected_kinds)
    if kinds is not None and operator.kind not in kinds:
        raise ValueError(
            "%s: operator %r has kind %r; expected one of %s"
            % (where, handle.name, operator.kind, sorted(kinds)))
    if handle.signature is not None and handle.signature != operator.signature:
        raise ValueError(
            "%s: operator handle %r carries signature %r, but the bound registry declares %r"
            % (where, handle.name, handle.signature, operator.signature))
    if expected_signature is not None and operator.signature != expected_signature:
        raise ValueError(
            "%s: operator %r has signature %r, expected %r"
            % (where, handle.name, operator.signature, expected_signature))
    return operator


__all__ = ["resolve_operator_handle", "resolve_registered_operator"]
