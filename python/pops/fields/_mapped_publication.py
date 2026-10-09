"""Stage-qualified physical transfers of consumed Field observations."""
from __future__ import annotations
from collections.abc import Mapping
import hashlib

from pops.identity import canonical_bytes
from pops.model import Handle
from pops.time._program.value_validation import require_top_level
from pops.time._authoring import authoring_transaction
from .mapping import ConsumedFieldPort
from ._program_publication import _target_space, _states, _consumer_states


def publish_mapped(solution, physical_map, bindings, *, states=None):
    from pops.mesh import PhysicalSupportMap, LayoutMappingPort, LayoutRepresentation
    from pops.time.field_context import FieldContext
    from pops.time.references import canonical_handle
    if type(physical_map) is not PhysicalSupportMap:
        raise TypeError("mapped Field publication requires an exact PhysicalSupportMap")
    if not isinstance(bindings, Mapping) or not bindings:
        raise TypeError("mapped Field publication requires exact destination to consumed-port bindings")
    if not isinstance(states, Mapping) or not states:
        raise TypeError("mapped Field publication requires exact destination stage States")
    program = solution.packed.prog
    program._guard_mutable("publish mapped consumed Field observations")
    require_top_level(program, solution.packed, "mapped Field publication")
    if program._current_region() != 0:
        raise ValueError("mapped Field publication requires a top-level barrier")
    solve = solution.packed.inputs[0].inputs[0]
    source_states = _states(solve, program)
    if not source_states:
        raise ValueError("mapped Field output has no equation inputs from which to infer storage")
    point = solution.packed.point
    # All equation owners remain explicit provenance. No routing State is selected.
    source_blocks = tuple(sorted(source_states, key=lambda block: block.qualified_id))
    with authoring_transaction(program):
        rows, imports = [], []
        common_space = None
        for destination, port in bindings.items():
            if not isinstance(destination, tuple) or len(destination) != 2:
                raise TypeError("mapped Field destination requires (qualified Field Handle, component)")
            target, component = destination
            if not isinstance(target, Handle) or target.kind != "field" or target.block_ref is None:
                raise TypeError("mapped Field destination must be block-qualified")
            if type(port) is not ConsumedFieldPort or port.field != solution.field:
                raise ValueError("mapped Field port belongs to another physical problem")
            space = _target_space(target)
            if common_space is not None and space != common_space:
                raise ValueError("one mapped publication context requires a common declared input FieldSpace")
            common_space = space
            source_port = LayoutMappingPort(canonical_handle(port.field), LayoutRepresentation.CELL_FIELD_V1, port)
            target_port = LayoutMappingPort(target, LayoutRepresentation.CELL_FIELD_V1, component=component)
            physical_map.validate_ports(source_port, target_port)
            target_states = _consumer_states(program, point, {}, tuple(states), tuple(states.values()))
            if target.block_ref not in target_states:
                raise ValueError("mapped Field destination has no exact stage State")
            if target.block_ref in source_states:
                raise ValueError("mapped Field output must cross distinct source and destination layouts")
            shape = target_states[target.block_ref]
            observed = solution[port.unknown] if port.derivative_axis is None else solution.gradient(
                port.unknown, dimension=physical_map.native_dimension)
            selected = 0 if port.derivative_axis is None else port.derivative_axis
            declaration = {"schema_version": 2, "contract": "mapped-consumed-output@1",
                "physical_map": physical_map.to_data(), "source_field": port.field,
                "target_field": target, "source_port": source_port.to_data(),
                "target_port": target_port.to_data(), "source_point": point, "target_point": point,
                "source_blocks": source_blocks, "field_problem_identity": solution.problem_identity,
                "field_unknown": observed.attrs.get("field_unknown", observed.inputs[0].attrs.get("field_unknown")),
                "ncomp": 1}
            declaration["invocation"] = "pops.program-field-map.v1::" + hashlib.sha256(canonical_bytes({
                "source": port.to_data(), "target": target_port.to_data(),
                "physical_map": physical_map.to_data(), "point": point.to_data()})).hexdigest()
            packed = program._new("scalar_field", "field_map_pack", (observed,),
                {"ncomp": 1, "source_component": selected, "factor": {"numerator": str(port.factor.numerator), "denominator": str(port.factor.denominator)},
                 "port": port.to_data()}, "mapped_field_source", None, point=point,
                space=port.quantity_space(), inherit_state_ref=False)
            program._new("scalar_field", "layout_map_export", (packed,), declaration,
                "mapped_field_export", None, point=point, space=port.quantity_space(), inherit_state_ref=False)
            candidate = program._new("scalar_field", "layout_map_import", (shape,), declaration,
                "mapped_field_candidate", target.block_ref, point=point,
                space=target_port.quantity_space(), inherit_state_ref=False)
            imports.append(candidate)
            rows.append({"target": target, "component": component, "source_component": 0})
        outputs = tuple(dict.fromkeys(row["component"] for row in rows))
        target_states = _consumer_states(program, point, {}, tuple(states), tuple(states.values()))
        target_blocks = {row["target"].block_ref for row in rows}
        if set(target_states) != target_blocks:
            raise ValueError("mapped Field supplemental States must cover exactly the destination blocks")
        context = FieldContext(solution.field, tuple((block, value.id) for block, value in target_states.items()), outputs)
        return program._new("fields", "field_publication", (*imports, *states.values()),
            {"bindings": tuple(rows), "consumer_states": tuple(states),
             "field_problem_identity": solution.problem_identity, "field": solution.field,
             "mapped_output_contract": "mapped-consumed-output@1"},
            "published_mapped_fields", None, point=point, space=common_space,
            field_context=context, inherit_state_ref=False)


def _canonical(value):
    from pops.time._graph.base import strict_data
    from pops.time._program.serialization import _json_ready
    return canonical_bytes(strict_data(_json_ready(value), where="mapped Field authority"))


def validate_pack(value):
    from fractions import Fraction
    from ._program_publication import _source
    from ._program_problem import _identity
    from pops.time._program.serialization import _json_ready
    from ._observation_contract import validate_field_observation
    if value.op != "field_map_pack" or value.vtype != "scalar_field" or len(value.inputs) != 1 or type(value.attrs.get("ncomp")) is not int or value.attrs.get("ncomp") != 1 or value.block is not None or value.state_ref is not None:
        raise ValueError("consumed Field map requires one selected scalar observation")
    source = value.inputs[0]
    width, solve = _source(source)
    port = _json_ready(value.attrs.get("port"))
    selected = value.attrs.get("source_component")
    observed = source if source.op == "field_component" else source.inputs[0]
    if type(selected) is not int or not 0 <= selected < width or value.point != source.point or source.prog is not value.prog:
        raise ValueError("consumed Field map changed its source component or stage")
    if port.get("contract") != "mapped-consumed-output@1" or port.get("field_problem_identity") != observed.attrs["field_problem_identity"] \
            or _canonical(port.get("unknown")) != _canonical(_json_ready(observed.attrs["field_unknown"])):
        raise ValueError("consumed Field map changed its equation or unknown authority")
    expected_axis = None if source.op == "field_component" else selected
    from pops.time._graph.base import _canonical_int, strict_data
    raw_axis = port.get("derivative_axis")
    actual_axis = None if raw_axis is None else _canonical_int(raw_axis)
    if actual_axis != expected_axis:
        raise ValueError("consumed Field map changed its differentiated axis")
    raw_factor = value.attrs.get("factor")
    if not isinstance(raw_factor, Mapping) or set(raw_factor) != {"numerator", "denominator"} or any(type(part) is not str for part in raw_factor.values()):
        raise ValueError("consumed Field map changed its exact scalar factor")
    factor = Fraction(int(raw_factor["numerator"]), int(raw_factor["denominator"]))
    if dict(raw_factor) != {"numerator": str(factor.numerator), "denominator": str(factor.denominator)} or port.get("factor") != dict(raw_factor):
        raise ValueError("consumed Field map changed its exact scalar factor")
    import math
    try:
        if not math.isfinite(float(factor)) or (factor and float(factor) == 0):
            raise ValueError("consumed Field map factor has no finite native value")
    except OverflowError as exc:
        raise ValueError("consumed Field map factor has no finite native value") from exc
    physical = _json_ready(solve.attrs["solve_request"]["physical_problem"])
    if _canonical(port.get("field")) != _canonical(physical["field_handle"]):
        raise ValueError("consumed Field map belongs to another registered Field")
    if source.space is None or value.space is None or value.space.support != source.space.support \
            or value.space.units != (source.space.units[selected],) or value.space.sampling != "cell":
        raise ValueError("consumed Field map has no declared selected physical space")
    if _canonical(strict_data(value.space.to_data(), where="mapped output space")) != _canonical(port.get("space")):
        raise ValueError("consumed Field map changed its immutable port space")
    return selected, factor, solve


def validate_candidate(value):
    if value.op != "layout_map_import" or value.vtype != "scalar_field" \
            or value.attrs.get("contract") != "mapped-consumed-output@1" \
            or len(value.inputs) != 1 or value.inputs[0].vtype != "state" \
            or type(value.attrs.get("ncomp")) is not int or value.attrs.get("ncomp") != 1 or value.state_ref is not None:
        raise ValueError("mapped Field publication lost its private scalar candidate")
    from pops.time._program.serialization import _json_ready
    from pops.identity import Identity
    attrs = value.attrs
    field = Identity.from_token(attrs.get("field_problem_identity"))
    target = attrs.get("target_field")
    if field.domain != "field-problem" or not isinstance(target, Handle) or target.block_ref != value.block \
            or value.inputs[0].block != value.block or value.point != attrs.get("target_point") \
            or value.point != attrs.get("source_point") or value.space is None \
            or value.space.sampling != "cell" or len(value.space.components) != 1:
        raise ValueError("mapped Field candidate changed destination, producer or point authority")
    data = _json_ready(attrs.get("target_port"))
    from pops.time._graph.base import strict_data
    from pops.time.references import canonical_handle
    if _canonical(data.get("subject")) != _canonical(strict_data(canonical_handle(target).canonical_identity(), where="mapped destination")) \
            or data.get("component") != value.space.components[0]:
        raise ValueError("mapped Field candidate changed its selected destination")
    declared = data.get("declared_space")
    component = value.space.components[0]
    if not isinstance(declared, Mapping) or component not in declared.get("components", ()):
        raise ValueError("mapped Field candidate lost its declared destination physical space")
    expected = dict(declared)
    expected["components"] = [component]
    expected["units"] = [declared["units"][declared["components"].index(component)]]
    expected["value_shape"] = [1]
    if _canonical(expected) != _canonical(value.space.to_data()):
        raise ValueError("mapped Field candidate changed its declared destination physical space")
    return value


def validate_mapped_pair(source, target):
    validate_candidate(target)
    if source.vtype != "scalar_field" or source.state_ref is not None or len(source.inputs) != 1 \
            or source.point != target.point or source.point != source.attrs.get("source_point"):
        raise ValueError("mapped Field export changed its stage authority")
    _selected, _factor, solve = validate_pack(source.inputs[0])
    from pops.time._program.serialization import _json_ready
    if _canonical(_json_ready(source.attrs["source_port"]["observation"])) != \
            _canonical(_json_ready(source.inputs[0].attrs["port"])):
        raise ValueError("mapped Field export changed its physical port")
    owners = tuple(sorted(_states(solve, source.prog), key=lambda block: block.qualified_id))
    if source.attrs.get("source_blocks") != owners:
        raise ValueError("mapped Field export lost complete equation owner provenance")
