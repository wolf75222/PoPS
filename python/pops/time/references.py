"""Typed semantic references used by the time-program IR.

The time language consumes declarations owned by other authoring domains.  This
module is the single boundary that authenticates those declarations and turns
them into canonical data for manifests.  Runtime lowering may project a display
name from a handle, but the authoring graph never replaces the handle with that
name.
"""
from __future__ import annotations

from typing import Any

from pops.model.handles import Handle
from pops.model.ownership import DoubleOwnershipError, OwnerKind, OwnerPath


def bind_program_block(program: Any, block: Any, *, where: str) -> Any:
    """Authenticate ``block`` and bind ``program`` to its exact authoring Case.

    Runtime block indices are local to one Case.  Combining blocks from two Cases would therefore
    let equal local names alias the same ABI slot, and even distinct names would describe no single
    runtime assembly.  The first accepted block fixes the Program's Case authority; every later
    block must carry the same opaque authoring capability, not merely the same canonical case name.
    """
    from pops.problem.handles import BlockHandle

    if not isinstance(where, str) or not where:
        raise TypeError("bind_program_block: where must be a non-empty string")
    if not isinstance(block, BlockHandle):
        raise TypeError("%s: block must be a pops.problem.BlockHandle" % where)
    registry = getattr(block, "_instance_registry", None)
    if registry is None:
        raise ValueError(
            "%s: block %s is detached from its authoritative Case registry"
            % (where, block.qualified_id))
    # Authenticate the complete live block identity before changing Program state.  A fabricated
    # BlockHandle with a copied owner/name must never be able to establish a Case authority.
    registry.canonical_block(block)
    case_owner = OwnerPath.coerce(block.owner_path)
    if case_owner.kind is not OwnerKind.CASE:
        raise ValueError("%s: block owner must be a Case OwnerPath" % where)
    bound = getattr(program, "_case_owner_path", None)
    if bound is None:
        program._case_owner_path = case_owner
        return block
    if bound != case_owner:
        if bound.name == case_owner.name:
            detail = "two distinct Case authoring authorities both named %r" % case_owner.name
        else:
            detail = "Case %r and Case %r" % (bound.name, case_owner.name)
        raise ValueError(
            "%s: one Program cannot combine blocks from %s; build one Program per Case"
            % (where, detail))
    return block


def bind_state_reference(block: Any, declaration: Any) -> tuple[Any, Handle]:
    """Authenticate ``declaration`` in ``block`` and return its qualified instance.

    ``Program.state(block, U)`` deliberately takes the model-local declaration
    ``U``: accepting ``block[U]`` as the second argument would make the block
    argument redundant and would hide accidental double qualification.
    """
    from pops.problem.handles import BlockHandle

    if not isinstance(block, BlockHandle):
        raise TypeError(
            "Program.state: block must be a pops.problem.BlockHandle, not %s"
            % type(block).__name__)
    if not isinstance(declaration, Handle):
        raise TypeError(
            "Program.state: state must be a declared Handle, not %s"
            % type(declaration).__name__)
    if declaration.kind != "state":
        raise TypeError(
            "Program.state: declaration %r has kind %r, expected 'state'"
            % (declaration.local_id, declaration.kind))
    if declaration.is_instance:
        raise DoubleOwnershipError(
            "Program.state: pass the model state declaration as T.state(block, U), "
            "not the already-qualified block[U]")
    qualified = block[declaration]
    if not qualified.is_instance or qualified.block_ref is not block:
        raise ValueError(
            "Program.state: block qualification did not return this block's state instance")
    return block, qualified


def bind_field_reference(program: Any, block: Any, field: Any) -> Any:
    """Authenticate a case-owned field for the exact Case already selected by ``block``."""
    from pops.problem.handles import FieldHandle

    bind_program_block(program, block, where="Program.solve_fields")
    if not isinstance(field, FieldHandle):
        raise TypeError(
            "Program.solve_fields: field must be a case-owned FieldHandle returned by "
            "Problem.add_field(...), not %s" % type(field).__name__)
    if field.owner_path != block.owner_path:
        raise ValueError(
            "Program.solve_fields: field %r and state block %r belong to different Cases"
            % (field.local_id, block.local_id))
    registry = getattr(field, "_field_registry", None)
    if registry is None:
        raise ValueError(
            "Program.solve_fields: field handle %s is detached from its authoritative Case registry"
            % field.qualified_id)
    if registry.owner_path != block.owner_path:
        raise ValueError(
            "Program.solve_fields: field registry and state block belong to different Cases")
    registry.canonicalize(field)
    return field


def block_name(block: Any) -> str:
    """Project the runtime/display block name at an explicit lowering boundary."""
    from pops.problem.handles import BlockHandle

    if not isinstance(block, BlockHandle):
        raise TypeError("expected a BlockHandle, got %r" % (block,))
    return block.local_id


def state_name(state: Any) -> str:
    """Project a state declaration's display name without changing its identity."""
    if not isinstance(state, Handle) or state.kind != "state":
        raise TypeError("expected a qualified state Handle, got %r" % (state,))
    return state.local_id


def field_name(field: Any) -> str:
    """Project a typed named-field selector at a lowering/display boundary."""
    from pops.model import OperatorHandle
    from pops.problem.handles import FieldHandle

    if isinstance(field, FieldHandle):
        return field.local_id
    if isinstance(field, OperatorHandle) and field.kind == "field_operator":
        return field.registered_operator_name
    raise TypeError("expected a FieldHandle or field_operator OperatorHandle, got %r" % field)


def canonical_handle(handle: Any) -> Handle:
    """Return authenticated canonical identity for a handle already accepted by Program.

    Block/state instances are canonicalised by the case registry that issued
    them.  Other handles (notably an operator already authenticated against the
    Program's bound registry) can shed only their process-local authoring
    capability; their complete typed owner path remains intact.
    """
    if not isinstance(handle, Handle):
        raise TypeError("canonical_handle expects a Handle")
    if handle.is_resolved:
        return handle
    if handle.is_instance:
        block = handle.block_ref
        registry = getattr(block, "_instance_registry", None)
        if registry is None:
            raise ValueError(
                "qualified handle %s is detached from its authoritative case registry"
                % handle.qualified_id)
        return registry.canonicalize(handle, block=block)
    from pops.problem.handles import BlockHandle
    if isinstance(handle, BlockHandle):
        registry = getattr(handle, "_instance_registry", None)
        if registry is None:
            raise ValueError(
                "block handle %s is detached from its authoritative case registry"
                % handle.qualified_id)
        return registry.canonical_block(handle)
    from pops.problem.handles import FieldHandle
    if isinstance(handle, FieldHandle):
        registry = getattr(handle, "_field_registry", None)
        if registry is None:
            raise ValueError(
                "field handle %s is detached from its authoritative case registry"
                % handle.qualified_id)
        return registry.canonicalize(handle)
    return handle._resolved()


def handle_data(handle: Any) -> dict[str, Any]:
    """Canonical JSON-ready identity of one Program semantic reference."""
    return canonical_handle(handle).canonical_identity()


__all__ = [
    "bind_field_reference", "bind_program_block", "bind_state_reference", "block_name",
    "canonical_handle", "field_name", "handle_data", "state_name",
]
