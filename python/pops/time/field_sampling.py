"""Exact scalar field storage and component observations, independent of conserved Q."""
from __future__ import annotations

from typing import Any

from pops.model import Handle
from pops.time.stencil import StencilAccess
from pops.time.values import ProgramValue
from pops.time.value_support import require_owned_value as require_owned


def storage_contract(program: Any, template: Any, storage: Any, space: Any, *,
                     source_module: Any = None) -> dict[str, Any]:
    require_owned(program, template, "scalar field template")
    from pops.problem._field_observations import scalar_storage_contract

    return scalar_storage_contract(template, storage, space, source_module=source_module)


def declared_field_output_space(module: Any, source: Any) -> Any:
    from pops.problem._field_observations import declared_field_output_space as declared_space

    return declared_space(module, source)


def _sampled_field_key(module: Any, source: Any, component: Any, space: Any,
                       block: Any) -> Any:
    from pops.problem._field_observations import sampled_field_key

    return sampled_field_key(module, source, component, space, block)


def scalar_co_location(left: Any, right: Any, where: str, *, same_space: bool = True) -> None:
    """Physical fields share storage explicitly; shape and names confer no authority."""
    from pops.problem._field_observations import scalar_co_location as validate_co_location

    validate_co_location(left, right, where, same_space=same_space)


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


__all__ = ["field_sample", "storage_contract", "scalar_co_location"]
