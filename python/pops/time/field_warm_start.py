"""Explicit seeding of a native field iterate from a consumed scalar solve."""
from __future__ import annotations

from typing import Any

from pops.fields import FieldOutput
from pops.time._program.value_validation import require_top_level
from pops.time.field_context import field_provenance_contains, require_field_read
from pops.time.references import bind_field_reference, canonical_handle


def _consumed_linear_sources(value: Any) -> tuple[Any, ...]:
    result, visited = [], set()

    def visit(node: Any) -> None:
        if node.id in visited:
            return
        visited.add(node.id)
        if node.op == "solve_outcome_component" and len(node.inputs) == 1:
            outcome = node.inputs[0]
            if outcome.op == "solve_outcome" and len(outcome.inputs) == 1 \
                    and outcome.inputs[0].op == "solve_linear":
                result.append(outcome.inputs[0])
                return
        for source in node.inputs:
            visit(source)

    visit(value)
    return tuple(result)


def validate_warm_start(value: Any, *, model_graph: Any = None, field_plans: Any = None) -> Any:
    if value.op != "field_warm_start" or value.vtype != "effect" or len(value.inputs) != 4:
        raise ValueError("field warm start has an invalid effect contract")
    scalar, fields, template, target = value.inputs
    if scalar.vtype != "scalar_field" or template.vtype != "state" or target.vtype != "state" \
            or template.block != target.block or scalar.block != target.block \
            or scalar.clock != target.clock or value.point != target.point \
            or template.space != target.space or template.state_ref != target.state_ref:
        raise ValueError("field warm start requires exact scalar/State owner and target point")
    context = require_field_read(fields, template, "field warm start source")
    if fields.point != template.point or not field_provenance_contains(scalar.field_context, context):
        raise ValueError("field warm start lost its exact original Gauss source context")
    if canonical_handle(value.attrs.get("field")) != canonical_handle(context.field):
        raise ValueError("field warm start targets another field problem")
    solves = _consumed_linear_sources(scalar)
    if not solves or any(solve.prog is not value.prog or solve.block != target.block
                         for solve in solves):
        raise ValueError("field warm start requires a consumed scalar linear solve dependency")
    from pops.time.field_sampling import storage_contract

    if model_graph is None:
        field, _, _, _ = bind_field_reference(value.prog, template.block, context.field,
                                               values=(template,))
        registration = field._field_registry.resolved_registration(field)
        operator = registration.operator
        module = None
    else:
        from pops.codegen.program_emit_field_routes import resolved_field_route
        owner = model_graph.owner_for_block(target.block)
        module = model_graph.source_module_for_owner(owner)
        _slot, plan = resolved_field_route(context.field, field_plans)
        operator = plan.operator
    output = operator.outputs[0] if operator.outputs else None
    if not isinstance(output, FieldOutput) or tuple(scalar.space.components) != (output.name,):
        raise ValueError("field warm start must seed the solved unknown, not a derived output")
    contract = storage_contract(value.prog, template, scalar.attrs.get("scalar_storage"),
                                 scalar.space, source_module=module)
    if any(scalar.attrs.get(key) != item for key, item in contract.items()):
        raise ValueError("field warm start changed its scalar storage authority")
    unknown = scalar.attrs["scalar_storage"].unknowns[0]
    if canonical_handle(unknown.declaration_ref or unknown) != canonical_handle(
            operator.unknown.declaration_ref or operator.unknown) \
            or canonical_handle(operator.unknown.block_ref) != canonical_handle(target.block):
        raise ValueError("field warm start storage is not the exact solved field carrier")
    return context.field


def publish_field_warm_start(program: Any, field: Any, scalar: Any, *, fields: Any,
                             template: Any, state: Any, name: Any = None) -> None:
    values = tuple(require_top_level(program, node, "field warm start")
                   for node in (scalar, fields, template, state))
    node = program._new("effect", "field_warm_start", values, {"field": field}, name,
                        state.block, point=state.point, inherit_state_ref=False)
    validate_warm_start(node)


__all__ = ["publish_field_warm_start", "validate_warm_start"]
