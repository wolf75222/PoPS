"""Resolved scalar field observation and warm-start qualification for lowering."""
from __future__ import annotations

from typing import Any

from pops.fields.problem import FieldStorageBinding
from pops.model import Handle
from pops.time.field_sampling import storage_contract
from pops.time.stencil import StencilAccess


def emitted_sample_contract(value: Any, model_graph: Any, *, field_plans: Any = None,
                             target: str = "system") -> Any:
    """Reauthenticate a registry-free compiled sample against the frozen model/layout plans."""
    from pops.codegen.program_models import ProgramModelGraph
    from pops.problem._field_observations import sampled_field_key
    if type(model_graph) is not ProgramModelGraph or model_graph.scalar_layout_plan is None:
        raise TypeError("compiled FieldSample requires its resolved model graph and LayoutPlan")
    owner = model_graph.owner_for_block(value.block)
    module = model_graph.source_module_for_owner(owner)
    pack = model_graph.resolved_provider_pack_for_block(value.block)
    validate_resolved_scalar_storage(value.prog, model_graph.scalar_layout_plan, target=target)
    sampled_field = value.attrs.get("sample_kind") == "field"
    if len(value.inputs) != (2 if sampled_field else 1):
        raise ValueError("compiled FieldSample changed its exact template arity")
    template = value.inputs[0]
    if value.block != template.block or value.point != template.point or value.state_ref is not None:
        raise ValueError("compiled FieldSample changed its owner, point or output representation")
    expected = storage_contract(value.prog, template, value.attrs["scalar_storage"], value.space,
                                source_module=module)
    if any(value.attrs.get(key) != data for key, data in expected.items()):
        raise ValueError("compiled FieldSample changed its storage authority")
    if sampled_field:
        from pops.time.field_context import require_field_read
        from pops.codegen.program_emit_field_routes import resolved_field_route
        from pops.time.references import canonical_handle

        fields = value.inputs[1]
        if fields.vtype != "fields" or fields.point != template.point \
                or template.vtype != "state":
            raise ValueError("compiled FieldSample changed its exact solve point or State anchor")
        context = require_field_read(fields, template, "compiled field sample")
        if value.field_context != context or value.attrs.get("sample_field") != context.field:
            raise ValueError("compiled FieldSample changed its exact FieldsContext authority")
        source, component = value.attrs.get("source"), value.attrs.get("component")
        key = sampled_field_key(module, source, component, value.space, value.block)
        _slot, plan = resolved_field_route(context.field, field_plans)
        unknown = plan.operator.unknown
        if canonical_handle(unknown.declaration_ref or unknown) != canonical_handle(
                source.declaration_ref or source) \
                or canonical_handle(unknown.block_ref) != canonical_handle(value.block):
            raise ValueError("compiled FieldSample carrier differs from the exact solved field")
        output = plan.native_options["output_route"]
        if key.to_data() not in output["component_keys"] or component not in context.outputs \
                or tuple(context.outputs) != tuple(output["components"]):
            raise ValueError("compiled FieldSample key is not an output of its exact field solve")
        pack.lookup(key)
        return pack.select((key,))
    if value.attrs["sample_kind"] == "state":
        component = value.attrs.get("component")
        if template.vtype != "state" or type(component) is not int \
                or not 0 <= component < len(template.space.components) \
                or template.space.units[component] != value.space.units[0]:
            raise ValueError("compiled FieldSample changed its State component contract")
        declaration = template.state_ref.declaration_ref or template.state_ref
        module.declaration_index().authenticate(declaration)
        return None
    if value.attrs["sample_kind"] != "aux":
        raise ValueError("compiled FieldSample has an unsupported source kind")
    source = value.attrs.get("source")
    if not isinstance(source, Handle) or source.kind != "aux":
        raise ValueError("compiled FieldSample lost its exact auxiliary declaration")
    declaration = source.declaration_ref or source
    module.declaration_index().authenticate(declaration)
    if source.is_instance:
        _, block = source._instance_refs()
        if block._resolved() != value.block._resolved():
            raise ValueError("compiled FieldSample auxiliary belongs to another block instance")
    descriptor = module.aux()[declaration.local_id]
    if descriptor.kind != "cell_scalar" or descriptor.centering != "cell" \
            or descriptor.unit != value.space.units[0] or descriptor.frame != value.space.frame \
            or descriptor.clock != value.space.clock:
        raise ValueError("compiled FieldSample auxiliary physical contract changed")
    from pops.model.provider_pack import ComponentKey
    key = ComponentKey(str(declaration.owner_path.canonical()), "aux",
                       declaration.local_id, declaration.local_id)
    pack.lookup(key)
    return pack.select((key,))


def validate_resolved_scalar_storage(program: Any, layout_plan: Any, *, target: str) -> None:
    from pops.codegen.program_field_plan import _nodes
    for node in _nodes(program):
        storage = node.attrs.get("scalar_storage")
        if storage is None:
            continue
        if target not in ("system", "amr_system"):
            raise NotImplementedError("typed scalar storage requires an exact native System or hierarchy")
        if type(storage) is not FieldStorageBinding:
            raise ValueError("scalar storage binding changed after authoring")
        owner = node.attrs.get("scalar_owner")
        from pops.time.references import canonical_handle
        owner = canonical_handle(owner)
        if layout_plan.layout_for(owner) != storage.layout:
            raise ValueError("scalar storage layout differs from its exact State anchor")
        normalized = layout_plan.normalized(storage.layout)
        if normalized.geometry.frame_id != node.space.frame:
            raise ValueError("scalar storage frame differs from its resolved geometry")
        axes = storage.observation_axes
        if axes is None or axes != tuple(range(normalized.geometry.dimension)):
            raise ValueError("scalar sampling requires an explicit complete support-axis binding")
        if tuple(name for name, _ in node.space.support.coordinates) != normalized.geometry.axis_names:
            raise ValueError("scalar sampling support coordinates differ from the native geometry")
        stencil = node.attrs.get("stencil_access")
        if type(stencil) is StencilAccess and stencil.required_ghost_depth > 0 and (
                normalized.native_spatial_layout is None or
                not all(normalized.native_spatial_layout.periodicity)):
            owner = node.attrs.get("field_boundary_owner")
            if not isinstance(owner, Handle) or owner.kind != "field" or \
                    node.attrs.get("field_boundary_contract") != "pops.installed-field-boundary@1":
                raise NotImplementedError(
                    "nonperiodic typed scalar stencils require conducting field boundary authority "
                    "from one exact installed field")


def validate_resolved_warm_start(value: Any, *, model_graph: Any,
                                 field_plans: Any = None) -> Any:
    """Reauthenticate a compiled warm start against its resolved field route."""
    from pops.time.field_warm_start import warm_start_context, validate_warm_start

    if model_graph is None:
        return validate_warm_start(value)
    context = warm_start_context(value)
    from pops.codegen.program_emit_field_routes import resolved_field_route
    from pops.problem._field_observations import validate_warm_start_storage

    owner = model_graph.owner_for_block(value.inputs[3].block)
    module = model_graph.source_module_for_owner(owner)
    _slot, plan = resolved_field_route(context.field, field_plans)
    validate_warm_start_storage(value, plan.operator, source_module=module)
    return context.field


__all__ = ["emitted_sample_contract", "validate_resolved_scalar_storage",
           "validate_resolved_warm_start"]
