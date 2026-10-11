"""Exact scalar field storage and component observations, independent of conserved Q."""
from __future__ import annotations

from typing import Any

from pops.fields.problem import FieldStorageBinding
from pops.model import FieldSpace, Handle
from pops.time.stencil import StencilAccess
from pops.time.values import ProgramValue
from pops.time._program.value_validation import require_owned


def storage_contract(program: Any, template: Any, storage: Any, space: Any, *,
                     source_module: Any = None) -> dict[str, Any]:
    require_owned(program, template, "scalar field template")
    if template.vtype not in ("state", "scalar_field") or template.block is None:
        raise TypeError("scalar field template requires exact block-qualified State/field storage")
    if type(storage) is not FieldStorageBinding or type(space) is not FieldSpace:
        raise TypeError("scalar storage requires exact FieldStorageBinding and FieldSpace")
    if space.representation != "field":
        raise NotImplementedError(
            "typed scalar storage requires the qualified field representation")
    if space.sampling != "cell":
        raise NotImplementedError(
            "typed scalar storage requires qualified cell sampling; point-value "
            "observations need an explicit qualified transform")
    if len(storage.unknowns) != 1 or len(space.components) != 1 or space.centering != "cell" \
            or space.support is None or space.units[0] is None:
        raise ValueError("scalar storage requires one typed cell-sampled unknown, support and units")
    if template.space is None:
        raise ValueError("scalar field template has no physical space")
    for attribute in ("layout", "centering", "sampling", "frame", "clock", "support"):
        if getattr(space, attribute) != getattr(template.space, attribute):
            raise ValueError("scalar storage template disagrees on " + attribute)
    if source_module is None:
        registry = template.block._instance_registry
        if registry is None:
            raise ValueError("scalar storage admission requires its authoring or resolved source authority")
        registry.canonical_block(template.block)
        model = registry.spec(template.block.local_id)["model"]
        module = getattr(model, "module", model)
    else:
        module = source_module
        if template.block.model_owner_path.canonical() != module.owner_path.canonical():
            raise ValueError("scalar storage template belongs to a foreign resolved model owner")
    unknown = storage.unknowns[0]
    declaration = unknown.declaration_ref or unknown
    module.declaration_index().authenticate(declaration)
    if declaration.kind != "field":
        raise ValueError("scalar storage field is not an exact declared FieldSpace")
    declared = declared_field_output_space(module, declaration)
    projection = None
    if declared != space:
        if space.name != declaration.local_id or space.components[0] not in declared.components:
            raise ValueError("scalar storage must be the exact declared FieldSpace or component projection")
        component = space.components[0]
        if any(getattr(space, attribute) != getattr(declared, attribute) for attribute in (
                "layout", "representation", "centering", "sampling", "frame", "clock", "support", "domain")) \
                or space.units[0] != declared.units[declared.components.index(component)]:
            raise ValueError("scalar storage component projection changed its physical contract")
        # Retain a typed Handle, not an owner string that could outlive reference resolution.
        # The exact ComponentKey is reconstructed only from this authenticated declaration.
        projection = (unknown, component)
    owner = template.state_ref if template.vtype == "state" else template.attrs.get("scalar_owner")
    if not isinstance(owner, Handle) or owner.kind != "state" or not owner.is_instance:
        raise ValueError("scalar storage has no exact qualified State layout anchor")
    return {"scalar_storage": storage, "scalar_owner": owner,
            "scalar_sampling_contract": "pops.scalar-field-storage@1",
            "scalar_storage_component": projection}


def declared_field_output_space(module: Any, source: Any) -> Any:
    """Find a field declaration through its exact registry or Board projection binding."""
    declaration = source.declaration_ref or source
    # Re-enter the frozen declaration authority: detached canonical Handles and live
    # registry-issued Handles carry equal canonical identities but distinct authoring forms.
    declaration = module.declaration_index().authenticate(declaration)
    declared = module.field_spaces().get(declaration.local_id)
    if declared is not None:
        return declared
    from pops.model.provider_pack import _authenticated_field_bindings

    matches = [output for subject, _target, _operator, output, _carrier in
               _authenticated_field_bindings(module,
                   canonical_owner=module.owner_path.canonical(),
                   field_spaces=module.field_spaces())
               if subject == declaration]
    if len(matches) != 1:
        raise ValueError("field has no unique authenticated carrier projection")
    return matches[0]


def _sampled_field_key(module: Any, source: Any, component: Any, space: Any,
                       block: Any) -> Any:
    """Authenticate the scientific storage carrier and its exact physical component."""
    from pops.model.provider_pack import ComponentKey

    if not isinstance(source, Handle) or source.kind != "field" or type(component) is not str:
        raise TypeError("field output sample requires an exact (Field Handle, component) pair")
    declaration = source.declaration_ref or source
    module.declaration_index().authenticate(declaration)
    if source.is_instance and source.block_ref._resolved() != block._resolved():
        raise ValueError("field output sample belongs to another block instance")
    declared = declared_field_output_space(module, declaration)
    if type(declared) is not FieldSpace or component not in declared.components:
        raise ValueError("field output sample component is absent from its declared FieldSpace")
    index = declared.components.index(component)
    if declared.units[index] != space.units[0]:
        raise ValueError("field output sample units differ from the scalar FieldSpace")
    for attribute in ("layout", "centering", "sampling", "frame", "clock", "support"):
        if getattr(declared, attribute) != getattr(space, attribute):
            raise ValueError("field output sample physical contract differs on " + attribute)
    return ComponentKey(str(declaration.owner_path.canonical()), "field",
                        declaration.local_id, component)


def emitted_sample_contract(value: Any, model_graph: Any, *, field_plans: Any = None,
                             target: str = "system") -> Any:
    """Reauthenticate a registry-free compiled sample against the frozen model/layout plans."""
    from pops.codegen.program_models import ProgramModelGraph
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
        key = _sampled_field_key(module, source, component, value.space, value.block)
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
    if source.is_instance and source.block_ref._resolved() != value.block._resolved():
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


def scalar_co_location(left: Any, right: Any, where: str, *, same_space: bool = True) -> None:
    """Physical fields share storage explicitly; shape and display names confer no authority."""
    if left.vtype != "scalar_field" or right.vtype != "scalar_field":
        raise TypeError(where + ": expected scalar fields")
    first, second = left.attrs.get("scalar_storage"), right.attrs.get("scalar_storage")
    if first is None and second is None:
        return
    if type(first) is not FieldStorageBinding or type(second) is not FieldStorageBinding:
        raise ValueError(where + ": both scalar fields need explicit storage authority")
    if first.layout != second.layout or first.observation_axes != second.observation_axes \
            or left.attrs.get("scalar_owner") != right.attrs.get("scalar_owner") \
            or (same_space and left.attrs.get("scalar_storage_component")
                != right.attrs.get("scalar_storage_component")):
        raise ValueError(where + ": scalar fields have different layout/owner bindings")
    if left.point != right.point:
        raise ValueError(where + ": scalar fields have different evaluation points")
    if same_space and left.space != right.space:
        raise ValueError(where + ": scalar field physical spaces differ")
    if not same_space:
        for attribute in ("layout", "centering", "sampling", "frame", "clock", "support"):
            if getattr(left.space, attribute) != getattr(right.space, attribute):
                raise ValueError(where + ": scalar mask co-location differs on " + attribute)


def field_sample(program: Any, source: Any, *, template: Any, storage: Any, space: Any,
                 component: Any = None, name: Any = None) -> Any:
    """Observe one exact native auxiliary, State, or solved-field component.

    A solved-field sample supplies the consumed FieldsContext as ``source`` and
    ``component=(carrier_handle, component_name)``. Its exact solved State and
    temporal point remain explicit dependencies of the materialized sample.
    """
    attrs = storage_contract(program, template, storage, space)
    field_context = None
    if isinstance(source, ProgramValue) and source.vtype == "fields":
        require_owned(program, source, "field sample FieldsContext")
        from pops.time.field_context import require_field_read
        from pops.time.references import bind_field_reference, canonical_handle

        if template.vtype != "state" or source.point != template.point:
            raise ValueError("field output sample requires the exact solved State and point")
        field_context = require_field_read(source, template, "field sample")
        if not isinstance(component, tuple) or len(component) != 2:
            raise TypeError("field output sample component must be (Field Handle, component)")
        carrier, component_name = component
        registry = template.block._instance_registry
        model = registry.spec(template.block.local_id)["model"]
        module = getattr(model, "module", model)
        key = _sampled_field_key(module, carrier, component_name, space, template.block)
        field, _, outputs, _ = bind_field_reference(program, template.block,
                                                   field_context.field, values=(template,))
        registration = field._field_registry.resolved_registration(field)
        unknown = registration.operator.unknown
        if canonical_handle(unknown.declaration_ref or unknown) != canonical_handle(
                carrier.declaration_ref or carrier) \
                or canonical_handle(unknown.block_ref) != canonical_handle(template.block):
            raise ValueError("field output sample carrier differs from the exact solved field")
        if component_name not in outputs or tuple(outputs) != tuple(field_context.outputs):
            raise ValueError("field output sample component is absent from its exact FieldsContext")
        from pops.model.provider_pack import build_provider_pack

        build_provider_pack(module).lookup(key)
        attrs.update(sample_kind="field", source=carrier, component=component_name,
                     sample_field=field_context.field, ncomp=1,
                     stencil_access=StencilAccess.pointwise())
        inputs = (template, source)
    elif isinstance(source, ProgramValue):
        require_owned(program, source, "field sample State")
        if source.vtype != "state" or source.block != template.block or source.point != template.point:
            raise ValueError("field sample State must share the exact template owner and point")
        if type(component) is str:
            component = source.space.components.index(component)
        if type(component) is not int or not 0 <= component < len(source.space.components):
            raise ValueError("field sample requires an exact State component")
        if source.space.units[component] != space.units[0]:
            raise ValueError("field sample State component units differ from its FieldSpace")
        attrs.update(sample_kind="state", component=component, ncomp=1,
                     stencil_access=StencilAccess.pointwise())
        inputs = (source,)
    else:
        if not isinstance(source, Handle) or source.kind != "aux" or component is not None:
            raise TypeError("field sample auxiliary requires one exact scalar Aux Handle")
        registry = template.block._instance_registry
        model = registry.spec(template.block.local_id)["model"]
        module = getattr(model, "module", model)
        declaration = source.declaration_ref or source
        module.declaration_index().authenticate(declaration)
        if source.is_instance and source.block_ref._resolved() != template.block._resolved():
            raise ValueError("field sample auxiliary belongs to a different block instance")
        descriptor = module.aux()[declaration.local_id]
        if descriptor.kind != "cell_scalar" or descriptor.centering != "cell" \
                or descriptor.unit != space.units[0] or descriptor.frame != space.frame \
                or descriptor.clock != space.clock:
            raise ValueError("field sample auxiliary physical contract differs from its FieldSpace")
        from pops.model.provider_pack import build_provider_pack, ComponentKey
        key = ComponentKey(str(declaration.owner_path.canonical()), "aux",
                           declaration.local_id, declaration.local_id)
        pack = build_provider_pack(module)
        pack.lookup(key)  # Missing/unavailable producers are refused before lowering.
        attrs.update(sample_kind="aux", source=source, ncomp=1,
                     stencil_access=StencilAccess.pointwise())
        inputs = (template,)
    return program._new("scalar_field", "field_sample", inputs, attrs, name, template.block,
                        space=space, point=template.point, field_context=field_context,
                        inherit_state_ref=False)


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


__all__ = ["field_sample", "storage_contract", "scalar_co_location", "emitted_sample_contract",
           "validate_resolved_scalar_storage"]
