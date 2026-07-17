"""Typed semantic references used by the time-program IR.

The time language consumes declarations owned by other authoring domains.  This
module is the single boundary that authenticates those declarations and turns
them into canonical data for manifests.  Runtime lowering may project a display
name from a handle, but the authoring graph never replaces the handle with that
name.
"""
from __future__ import annotations

from typing import Any

from pops.model.handles import Handle, StateHandle
from pops.model.ownership import MissingOwnershipError, OwnerKind, OwnerPath


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


def bind_state_reference(program: Any, declaration: StateHandle) -> tuple[Any, StateHandle]:
    """Authenticate one already-qualified state and bind its exact model registry.

    The sole authoring form is ``Program.state(block[U])``.  The instance handle is the one
    non-redundant capability that identifies the Case, block, model declaration and state.  Its
    issuing registry is also the authoritative route to the model's operator registry, so a user
    never has to repeat ``bind_operators(model.module)`` beside an already-qualified state.
    """
    from pops.problem.handles import BlockHandle

    if not isinstance(declaration, Handle):
        raise TypeError(
            "Program.state: state must be a declared Handle, not %s"
            % type(declaration).__name__)
    if declaration.kind != "state":
        raise TypeError(
            "Program.state: declaration %r has kind %r, expected 'state'"
            % (declaration.local_id, declaration.kind))
    if not declaration.is_instance:
        raise MissingOwnershipError(
            "Program.state: pass the block-qualified state as T.state(block[U]); an unqualified "
            "model state cannot select one Case instance")
    block = declaration.block_ref
    if not isinstance(block, BlockHandle):
        raise MissingOwnershipError(
            "Program.state: qualified state is missing its issuing BlockHandle")
    registry = getattr(block, "_instance_registry", None)
    if registry is None:
        raise MissingOwnershipError(
            "Program.state: state %s is detached from its authoritative Case registry"
            % declaration.qualified_id)
    qualified = registry.qualify(declaration, allow_existing=True)
    if qualified is not declaration:
        raise MissingOwnershipError(
            "Program.state: state %s was not issued by this Case registry"
            % declaration.qualified_id)
    if qualified.block_ref is not block:
        raise ValueError(
            "Program.state: block qualification did not return this block's state instance")
    bind_program_block(program, block, where="Program.state")
    model = registry.spec(block.local_id)["model"]
    operator_source = getattr(model, "module", model)
    program._bind_operators(operator_source)
    return block, qualified


def bind_field_reference(
    program: Any,
    block: Any,
    field: Any,
    *,
    values: Any = (),
) -> tuple[Any, Any, tuple[str, ...], dict[str, bool]]:
    """Authenticate one Case field and return its exact Program contract.

    The returned tuple is ``(handle, FieldSpace, output_components,
    schedule_capabilities)``.  Both the value type consumed by rates and the
    output names come from the registered physical ``FieldOperator``; Program
    authoring never guesses a default field layout from a model or from the
    handle's display name.  Schedule capabilities are the intersection of the
    exact model providers authenticated by the Program, so a composed field is
    cacheable only when every physical contribution declares that property.
    """
    from pops.problem.handles import FieldHandle

    bind_program_block(program, block, where="field operator")
    if not isinstance(field, FieldHandle):
        raise TypeError(
            "field operator: field must be a case-owned FieldHandle returned by "
            "Case.field(...), not %s" % type(field).__name__)
    if field.owner_path != block.owner_path:
        raise ValueError(
            "field operator: field %r and state block %r belong to different Cases"
            % (field.local_id, block.local_id))
    registry = getattr(field, "_field_registry", None)
    if registry is None:
        raise ValueError(
            "field operator: field handle %s is detached from its authoritative Case registry"
            % field.qualified_id)
    if registry.owner_path != block.owner_path:
        raise ValueError(
            "field operator: field registry and state block belong to different Cases")
    registry.canonicalize(field)
    registration = registry.get(field.local_id)
    if registration is None:
        raise ValueError("field operator %r is absent from its Case registry" % field.local_id)
    unknown = registration.operator.unknown
    if getattr(unknown, "block_ref", None) is None:
        block_registry = getattr(block, "_instance_registry", None)
        if block_registry is None:
            raise ValueError("field solve state block is detached from its Case registry")
        unknown = block_registry.qualify(unknown, allow_existing=True)
    source_block = getattr(unknown, "block_ref", None)
    declaration = getattr(unknown, "declaration_ref", None)
    if source_block is None or declaration is None:
        raise TypeError(
            "field operator %r unknown must be a block-qualified FieldSpace"
            % field.local_id)
    source_registry = getattr(source_block, "_instance_registry", None)
    if source_registry is None or source_registry.owner_path != block.owner_path:
        raise ValueError(
            "field operator %r unknown belongs to a different or detached Case"
            % field.local_id)
    spec = source_registry.spec(source_block.local_id)
    model = None if spec is None else spec.get("model")
    field_spaces = getattr(model, "field_spaces", None)
    if not callable(field_spaces):
        field_spaces = getattr(getattr(model, "module", None), "field_spaces", None)
    declared = field_spaces() if callable(field_spaces) else {}
    output_space = declared.get(declaration.local_id)
    from pops.model import FieldSpace
    if not isinstance(output_space, FieldSpace):
        raise TypeError(
            "field operator %r unknown %r has no exact FieldSpace declaration"
            % (field.local_id, declaration.local_id))
    outputs = tuple(output_space.components)
    if not outputs or any(not isinstance(name, str) or not name for name in outputs):
        raise ValueError(
            "field operator %r must resolve non-empty FieldSpace output components"
            % field.local_id)
    provider_spaces = tuple(
        getattr(contribution.provider, "signature", None).output
        if getattr(contribution.provider, "signature", None) is not None else None
        for contribution in registration.operator.providers
    )
    if not provider_spaces or any(not isinstance(item, FieldSpace) for item in provider_spaces):
        raise TypeError(
            "field operator %r providers must declare exact FieldSpace outputs"
            % field.local_id)
    program_space = provider_spaces[0]
    if any(item != program_space for item in provider_spaces[1:]):
        raise ValueError(
            "field operator %r providers disagree on the consumed FieldSpace"
            % field.local_id)
    if tuple(program_space.components) != outputs:
        raise ValueError(
            "field operator %r provider components %r disagree with field outputs %r"
            % (field.local_id, tuple(program_space.components), outputs))
    from pops.time.operator_resolution import resolve_operator_handle

    providers = tuple(
        resolve_operator_handle(
            program,
            contribution.provider,
            where="field operator %r provider" % field.local_id,
            expected_kinds="field_operator",
            values=values,
        )
        for contribution in registration.operator.providers
    )
    capabilities = {
        "cacheable": all(bool(provider.capabilities.get("cacheable"))
                         for provider in providers),
    }
    return field, program_space, outputs, capabilities


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
    "canonical_handle", "handle_data", "state_name",
]
