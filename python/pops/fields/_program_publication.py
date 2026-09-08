"""Explicit publication of consumed solve observations to qualified physics inputs."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pops.model import Handle, Module
from pops.time.field_context import FieldContext
from pops.time._program.value_validation import require_top_level
from pops.fields._observation_contract import validate_field_gradient, validate_field_observation


def _source(value: Any) -> tuple[int, Any]:
    if getattr(value, "op", None) == "field_component":
        return 1, validate_field_observation(value)[2]
    if getattr(value, "op", None) == "field_gradient":
        validate_field_gradient(value)
        return value.attrs["ncomp"], validate_field_observation(value.inputs[0])[2]
    raise ValueError("field publication requires a consumed field observation or gradient")


def _target_space(target: Handle) -> Any:
    registry = getattr(target.block_ref, "_instance_registry", None)
    if registry is None:
        raise ValueError("field publication target requires its authoritative Case registry")
    instances = tuple(block for block in registry.handles().values()
                      if block.model_owner_path.canonical() == target.block_ref.model_owner_path.canonical())
    if len(instances) != 1:
        raise ValueError("consumed field publication cannot share a model-definition provider key across block instances")
    model = registry.spec(target.block_ref.local_id)["model"]
    module = model if isinstance(model, Module) else model.module
    module.declaration_index().authenticate(target.declaration_ref)
    return module.field_spaces()[target.declaration_ref.local_id]


def _states(solve: Any, program: Any) -> dict[Any, Any]:
    from pops.codegen.program_field_plan import _nodes, _reachable

    states = {}
    for node in _reachable(solve, _nodes(program)):
        if node.op in ("field_problem_load", "field_problem_coefficients"):
            for value in node.inputs:
                if value.block in states and states[value.block] is not value:
                    raise ValueError("field publication has ambiguous current state provenance")
                states[value.block] = value
    return states


def publish_field_solution(solution: Any, bindings: Any, *, states: Any = None) -> Any:
    if not isinstance(bindings, Mapping) or not bindings:
        raise TypeError("field publication requires exact field Handle to observation bindings")
    program = solution.packed.prog
    program._guard_mutable("publish consumed field observations")
    rows, inputs = [], []
    expected_solve = solution.packed.inputs[0].inputs[0]
    seen = set()
    field_space = None
    for target, supplied in bindings.items():
        if isinstance(target, tuple) and len(target) == 2:
            target, component = target
        else:
            component = getattr(target, "local_id", None)
        if not isinstance(target, Handle) or target.kind != "field" or target.block_ref is None:
            raise TypeError("field publication destination requires an exact block-qualified field Handle")
        if type(component) is not str or not component:
            raise TypeError("field publication destination component must be an explicit nonempty string")
        target_space = _target_space(target)
        if component not in target_space.components:
            raise ValueError("field publication destination component is not declared by its field space")
        if field_space is not None and target_space != field_space:
            raise ValueError("one field publication context requires a common declared input field space")
        field_space = target_space
        source, selected = supplied if isinstance(supplied, tuple) and len(supplied) == 2 else (supplied, 0)
        require_top_level(program, source, "field publication")
        width, solve = _source(source)
        if solve is not expected_solve or source.point != solution.packed.point:
            raise ValueError("field publication observation belongs to another consumed solve or stage")
        if type(selected) is not int or not 0 <= selected < width:
            raise ValueError("field publication selected source component is outside its exact width")
        key = (target.qualified_id, component)
        if key in seen:
            raise ValueError("field publication cannot publish a destination component twice")
        seen.add(key)
        inputs.append(source)
        rows.append({"target": target, "component": component, "source_component": selected})
    if states is not None and not isinstance(states, Mapping):
        raise TypeError("field publication states must map exact qualified state Handles to Program values")
    supplemental = tuple((states or {}).items())
    target_blocks = {row["target"].block_ref for row in rows}
    if any(getattr(key, "block_ref", None) not in target_blocks for key, _ in supplemental):
        raise ValueError("field publication supplemental state must belong to a destination block")
    consumer_states = _consumer_states(program, solution.packed.point, _states(expected_solve, program),
                                      tuple(key for key, _ in supplemental),
                                      tuple(value for _, value in supplemental))
    if any(row["target"].block_ref not in consumer_states for row in rows):
        raise ValueError("field publication destination block has no exact solve-stage state")
    output_names = tuple(dict.fromkeys(row["component"] for row in rows))
    context = FieldContext(solution.field, tuple((block, value.id) for block, value in consumer_states.items()),
                           output_names)
    return program._new("fields", "field_publication", (*inputs, *(value for _, value in supplemental)),
        {"bindings": tuple(rows), "consumer_states": tuple(key for key, _ in supplemental), "field_problem_identity": solution.problem_identity,
         "field": solution.field}, "published_fields", None,
        field_context=context, space=field_space,
        point=solution.packed.point, inherit_state_ref=False)


def validate_field_publication(value: Any, *, target_space: Any = None) -> tuple[dict[str, Any], ...]:
    if value.op != "field_publication" or value.vtype != "fields" or value.field_context is None:
        raise ValueError("invalid consumed field publication node")
    rows = value.attrs.get("bindings")
    supplemental = value.attrs.get("consumer_states", ())
    if not isinstance(rows, (tuple, list)) or not rows or not isinstance(supplemental, (tuple, list)) \
            or len(rows) + len(supplemental) != len(value.inputs):
        raise ValueError("field publication has inconsistent observation bindings")
    solves = set()
    destinations = set()
    for row, source in zip(rows, value.inputs[:len(rows)], strict=True):
        width, solve = _source(source)
        solves.add(solve.id)
        selected = row.get("source_component")
        target = row.get("target")
        if type(selected) is not int or not 0 <= selected < width \
                or source.point != value.point or source.prog is not value.prog:
            raise ValueError("field publication lost source component or stage authority")
        if not isinstance(target, Handle) or target.kind != "field" or target.block_ref is None:
            raise ValueError("field publication lost its qualified field destination")
        space = (_target_space if target_space is None else target_space)(target)
        component = row.get("component")
        key = (target.qualified_id, component)
        if component not in space.components or space != value.space or key in destinations:
            raise ValueError("field publication lost its exact destination component or field space")
        destinations.add(key)
        observed = source if source.op == "field_component" else source.inputs[0]
        if observed.attrs.get("field_problem_identity") != value.attrs.get("field_problem_identity"):
            raise ValueError("field publication belongs to another physical field problem")
    if len(solves) != 1:
        raise ValueError("field publication must consume one exact joint solve")
    target_blocks = {row["target"].block_ref for row in rows}
    if any(getattr(key, "block_ref", None) not in target_blocks for key in supplemental):
        raise ValueError("field publication supplemental state lost its destination block")
    states = _consumer_states(value.prog, value.point, _states(solve, value.prog),
                              supplemental, value.inputs[len(rows):])
    context = value.field_context
    expected = tuple((block, state.id) for block, state in states.items())
    if context.stage_sources != expected or context.field != value.attrs.get("field") \
            or any(row["target"].block_ref not in states for row in rows):
        raise ValueError("field publication lost its exact solve-stage provenance")
    if tuple(context.outputs) != tuple(dict.fromkeys(row["component"] for row in rows)):
        raise ValueError("field publication context changed its declared outputs")
    return tuple(rows)


def _consumer_states(program: Any, point: Any, states: Any, handles: Any, values: Any) -> dict[Any, Any]:
    from pops.time import StagePoint, TimePoint
    from pops.time.points import point_clock
    from pops.time.references import canonical_handle
    result = dict(states)
    seen = set()
    for handle, value in zip(handles, values, strict=True):
        require_top_level(program, value, "field publication consumer state")
        if not isinstance(handle, Handle) or handle.kind != "state" or handle.block_ref is None \
                or value.vtype != "state" or value.block != handle.block_ref or value.state_ref is None:
            raise ValueError("field publication consumer requires an exact qualified state mapping")
        if canonical_handle(handle).canonical_identity() != canonical_handle(value.state_ref).canonical_identity():
            raise ValueError("field publication consumer state changes its declared identity")
        same_point = value.point == point
        if type(value.point) is TimePoint and type(point) is StagePoint:
            same_point = value.point == point.time
        if value.clock != point_clock(point, "field publication") or not same_point:
            raise ValueError("field publication consumer state belongs to another clock or stage point")
        if handle.block_ref in seen or (handle.block_ref in result and result[handle.block_ref] is not value):
            raise ValueError("field publication consumer state has conflicting or repeated block mappings")
        seen.add(handle.block_ref)
        result[handle.block_ref] = value
    return result


def publication_states(value: Any, *, target_space: Any = None) -> dict[Any, Any]:
    rows = validate_field_publication(value, target_space=target_space)
    return _consumer_states(value.prog, value.point, _states(_source(value.inputs[0])[1], value.prog),
                            value.attrs.get("consumer_states", ()), value.inputs[len(rows):])
