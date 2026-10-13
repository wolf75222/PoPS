"""Explicit installed-field boundary authority for scalar elliptic stencils."""
from __future__ import annotations

from typing import Any

from pops.model import Handle
from pops.time.references import canonical_handle


def field_boundary_contract(program: Any, boundary: Any, anchor: Any) -> dict[str, Any]:
    if boundary is None:
        return {}
    if not isinstance(boundary, Handle) or boundary.kind != "field":
        raise TypeError("field stencil boundary requires its exact declared field Handle")
    block = anchor.block
    registry = getattr(block, "_instance_registry", None)
    if registry is None:
        raise ValueError("field stencil boundary requires its authoritative Case block")
    registry.canonical_block(block)
    model = registry.spec(block.local_id)["model"]
    module = getattr(model, "module", model)
    declaration = boundary.declaration_ref or boundary
    module.declaration_index().authenticate(declaration)
    if boundary.is_instance and canonical_handle(boundary.block_ref) != canonical_handle(block):
        raise ValueError("field stencil boundary belongs to another block instance")
    from pops.time.field_sampling import declared_field_output_space
    declared_field_output_space(module, declaration)
    return {"field_boundary_owner": boundary,
            "field_boundary_contract": "pops.installed-field-boundary@1"}


def resolved_field_boundary_route(owner: Any, block: Any, field_plans: Any) -> tuple[str, Any]:
    """Select by the exact carrier and block, never by a display name or first field."""
    from collections.abc import Mapping
    if not isinstance(owner, Handle) or owner.kind != "field" or not isinstance(field_plans, Mapping):
        raise ValueError("field stencil has no resolved installed-field boundary authority")
    declaration = canonical_handle(owner.declaration_ref or owner)
    selected = []
    for plan in field_plans.values():
        unknown = plan.operator.unknown
        if canonical_handle(unknown.declaration_ref or unknown) == declaration \
                and canonical_handle(unknown.block_ref) == canonical_handle(block):
            selected.append(plan)
    if len(selected) != 1:
        raise ValueError("field stencil boundary requires one exact installed field plan")
    plan = selected[0]
    return plan.native_options["provider_slot"], plan


__all__ = ["field_boundary_contract", "resolved_field_boundary_route"]
