"""Owner-safe resolution of typed model operators inside a time Program.

Public authoring routes accept :class:`pops.model.OperatorHandle` values, never
free operator-name strings.  A handle is meaningful only relative to the exact
registry that declared it: a matching local name is not sufficient.  This
module centralizes that boundary so operator calls, ``P.linear_source``, typed
RHS terms, matrix-free operators, and ``pops.lib.time`` factories cannot drift
into slightly different validation rules.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _bound_registry(program: Any, where: str, owner: Any = None) -> tuple[Any, Any]:
    """Return the exact ``(registry, owner)`` selected by semantic provenance."""
    registries = getattr(program, "_operator_registries", None) or {}
    if not registries:
        raise ValueError(
            "%s: no operators are bound; declare an authenticated state with "
            "P.state(block[state]) first"
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
    from pops.problem.handles import BlockHandle

    if not isinstance(handle, OperatorHandle):
        raise TypeError(
            "%s: expected a pops.model.OperatorHandle, got %r" % (where, handle))
    declaration = handle.declaration_ref if handle.is_instance else handle
    if declaration is None:
        raise ValueError(
            "%s: instantiated operator handle %r has no declaration provenance"
            % (where, handle.name))
    if not isinstance(declaration, OperatorHandle):
        raise ValueError(
            "%s: instantiated operator handle %r has non-operator declaration provenance"
            % (where, handle.name))
    registry, owner = _bound_registry(program, where, declaration.owner_path)
    if declaration.owner_path != owner:
        raise ValueError(
            "%s: operator handle %r belongs to owner %s, but this Program is bound to %s"
            % (where, handle.name, declaration.owner_path, owner))
    if handle.is_instance:
        block = handle.block_ref
        if not isinstance(block, BlockHandle) or block.model_owner_path != owner:
            raise ValueError(
                "%s: instantiated operator handle %r has inconsistent block/model provenance"
                % (where, handle.name))
        argument_blocks = {
            value.block
            for value in values
            if getattr(value, "block", None) is not None
        }
        if argument_blocks and block not in argument_blocks:
            raise ValueError(
                "%s: operator handle %r belongs to block %r, but none of its arguments comes "
                "from that block (argument blocks: %s)"
                % (where, handle.name, block.local_id,
                   sorted(item.local_id for item in argument_blocks)))
    argument_owners = {
        value.block.model_owner_path
        for value in values
        if getattr(value, "block", None) is not None
    }
    if argument_owners and owner not in argument_owners:
        raise ValueError(
            "%s: operator handle %r belongs to owner %s, but none of its block-qualified "
            "arguments instantiate that owner (argument owners: %s)"
            % (where, handle.name, owner, sorted(str(item) for item in argument_owners)))
    registry_name = registry.target_for_handle(declaration.name)
    if declaration.registered_operator_name != registry_name \
            or handle.registered_operator_name != registry_name:
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
    # A typed operator is the explicit join protocol for cross-model values.  Multiple argument
    # owners are therefore valid only when the registry declaration carries a structural signature;
    # the caller's normal signature checker then authenticates every StateSpace/FieldSpace input in
    # order.  Owner-local operators keep the historical single-owner contract.
    if len(argument_owners) > 1 and operator.signature is None:
        raise ValueError(
            "%s: cross-model operator %r must declare a structural Signature"
            % (where, handle.name))
    return operator


__all__ = ["resolve_operator_handle"]
