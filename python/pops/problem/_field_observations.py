"""Authenticate physical field observations at the Case authoring boundary.

Temporal authoring passes exact handles and values here. Field descriptor types
remain owned by the field domain rather than entering the temporal module DAG.
"""
from __future__ import annotations

from typing import Any

from pops.model import FieldSpace, Handle


def scalar_storage_contract(template: Any, storage: Any, space: Any, *,
                            source_module: Any = None) -> dict[str, Any]:
    from pops.fields.problem import FieldStorageBinding

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


def sampled_field_key(module: Any, source: Any, component: Any, space: Any,
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


def scalar_co_location(left: Any, right: Any, where: str, *, same_space: bool = True) -> None:
    """Physical fields share storage explicitly; shape and display names confer no authority."""
    from pops.fields.problem import FieldStorageBinding

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


def validate_warm_start_storage(value: Any, operator: Any, *,
                                source_module: Any = None) -> None:
    """Authenticate the solved carrier and its exact scalar storage contract."""
    from pops.fields import FieldOutput
    from pops.time.references import canonical_handle

    scalar, _fields, template, target = value.inputs
    output = operator.outputs[0] if operator.outputs else None
    if not isinstance(output, FieldOutput) or tuple(scalar.space.components) != (output.name,):
        raise ValueError("field warm start must seed the solved unknown, not a derived output")
    contract = scalar_storage_contract(template, scalar.attrs.get("scalar_storage"),
                                       scalar.space, source_module=source_module)
    if any(scalar.attrs.get(key) != item for key, item in contract.items()):
        raise ValueError("field warm start changed its scalar storage authority")
    unknown = scalar.attrs["scalar_storage"].unknowns[0]
    if canonical_handle(unknown.declaration_ref or unknown) != canonical_handle(
            operator.unknown.declaration_ref or operator.unknown) \
            or canonical_handle(operator.unknown.block_ref) != canonical_handle(target.block):
        raise ValueError("field warm start storage is not the exact solved field carrier")


__all__ = ["scalar_storage_contract", "declared_field_output_space", "sampled_field_key",
           "scalar_co_location", "validate_warm_start_storage"]
