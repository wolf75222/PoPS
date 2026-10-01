"""Closed original accumulation and issued-duration authority (version 1)."""

from __future__ import annotations

from typing import Any

from pops.identity import canonical_bytes
from pops.time._program.serialization import _json_ready


def validate_tau_data(value: Any, *, program: Any = None, point: Any = None) -> dict:
    from pops.fields._program_expression import decode_field_literal
    from pops.time.points import point_clock

    data = _json_ready(value)
    if (
        not isinstance(data, dict)
        or set(data)
        != {"schema_version", "kind", "program_owner", "point", "factor", "window_authority"}
        or type(data["schema_version"]) is not int
        or data["schema_version"] != 1
        or data["kind"] != "issued_frame_duration"
        or data["window_authority"] != "native_issued_cadence_frame@1"
    ):
        raise ValueError("original stage lost its issued duration contract")
    if decode_field_literal(data["factor"]).to_python() <= 0:
        raise ValueError("original stage duration factor must be positive")
    if program is not None and data["program_owner"] != program.owner_path.canonical().to_data():
        raise ValueError("original stage duration belongs to another Program")
    if point is not None:
        if data["point"] != point.to_data():
            raise ValueError("original stage duration changed evaluation point")
        if program is not None and point_clock(point, "original stage") != program.clock:
            raise NotImplementedError(
                "issued field duration requires a declared native clock realization for this clock"
            )
    return data


def stage_projection(problem: Any, program: Any, point: Any, *, authoring: bool = True) -> Any:
    from pops.time.evolved_field_stage import EvolvedOriginalFieldProjection

    rows = [row for row in problem.outputs if type(row) is EvolvedOriginalFieldProjection]
    if not rows:
        return None
    if len(rows) != 1:
        raise ValueError("original stage repeats its accumulation projection")
    projection = rows[0]
    if authoring:
        projection.tau.require_program(program, at=point)
    validate_tau_data(projection.tau.to_data(), program=program, point=point)
    if projection.unknowns != problem.unknowns:
        raise ValueError("original accumulation changed its complete unknown tuple")
    from pops.time.evolved_field_stage import evolved_equations
    from pops.fields._identity import strict_field_data

    if canonical_bytes(
        [strict_field_data(row) for row in evolved_equations(projection)]
    ) != canonical_bytes([strict_field_data(row) for row in problem.equations]):
        raise ValueError("original stage accumulation/rate differs from its original equations")
    return projection


def validate_encoded_tau(expressions: Any, authority: Any) -> None:
    expected = None if authority is None else canonical_bytes(validate_tau_data(authority))

    def visit(node: Any) -> None:
        if not isinstance(node, (tuple, list)) or not node:
            return
        if node[0] == "temporal_tau":
            if (
                len(node) != 2
                or expected is None
                or canonical_bytes(validate_tau_data(node[1])) != expected
            ):
                raise ValueError("original equation uses an undeclared issued duration")
        else:
            for child in node[1:]:
                visit(child)

    for expression in expressions:
        visit(expression)


def validate_additive_capture_reads(captures: Any) -> None:
    """A current State read cannot impersonate a future computed candidate."""
    from pops.time.points import TimePoint

    pending, seen = list(captures), set()
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if value.op == "state" and value.point != TimePoint(value.clock, 0):
            raise ValueError("additive source State read changed its accepted endpoint")
        pending.extend(value.inputs)


def emit_issued_duration(
    authority: Any, program: Any, point: Any, name: str, lines: list, *, operation_id: int
) -> str | None:
    if authority is None:
        return None
    validate_tau_data(authority, program=program, point=point)
    lines += [
        "pops::Real %s = 0;" % name,
        "std::exception_ptr %s_error;" % name,
        'try { %s = ctx.boundary_evaluation_point(%d).dt; if (!std::isfinite(%s) || %s <= 0) throw std::invalid_argument("original stage issued duration is invalid"); }'
        % (name, operation_id, name, name),
        "catch (...) { %s_error = std::current_exception(); }" % name,
        'pops::collectively_rethrow_exception(%s_error, ctx.prepared_execution_lane(), "original stage issued duration");'
        % name,
        "if (pops::all_reduce_max(%s, ctx.prepared_execution_lane()) != -pops::all_reduce_max(-%s, ctx.prepared_execution_lane()))"
        % (name, name),
        '  throw std::invalid_argument("original stage issued durations differ across ranks");',
    ]
    return name


def build_evolved_state(solution: Any, *, target: Any) -> Any:
    from pops.time._authoring import authoring_transaction
    from pops.time._program.value_validation import require_top_level
    from pops.time.handles import StateEndpointHandle
    from pops.time.stencil import StencilAccess

    program = solution.packed.prog
    with authoring_transaction(program):
        program._guard_mutable("publish original field accumulation")
        if type(target) is not StateEndpointHandle:
            raise TypeError("original accumulation requires an exact State endpoint")
        target = program._require_endpoint(target, "original accumulation")
        require_top_level(program, solution.packed, "original accumulation")
        solve = solution.packed.inputs[0].inputs[0]
        if solve.op != "solve_spatial_field":
            raise ValueError("accumulation publication requires an original spatial field solve")
        source = solve.attrs["source_contract"]
        stage = source.get("evolved_stage")
        if stage is None:
            raise ValueError("original field problem declares no evolved accumulation")
        captures = solve.inputs[2 : 2 + solve.attrs["capture_count"]]
        matches = [row for row in captures if row.state_ref == target.state]
        if len(matches) != 1:
            raise ValueError("original accumulation has no exact previous State witness")
        previous = matches[0]
        indices = _projection_indices(stage, previous)
        partitioned = indices != tuple(range(len(stage["previous"])))
        expressions = tuple(source["accumulation"][index] for index in indices)
        partition = {"component_indices": indices} if partitioned else {}
        result = program._new(
            "scalar_field",
            "field_evolved_state",
            (solution.packed, *captures),
            {
                "projection_contract": "pops.evolved-original-field-stage@2" if partitioned else "pops.evolved-original-field-stage@1",
                "ncomp": len(target.space.components),
                "previous_capture_index": next(
                    index for index, row in enumerate(captures) if row is previous
                ),
                "field_problem_identity": solution.problem_identity,
                "expressions": expressions if partitioned else source["accumulation"],
                "stage": stage,
                "stencil_access": StencilAccess.pointwise(),
                **partition,
            },
            "evolved_accumulation",
            target.block,
            space=target.space,
            state_ref=target.state,
            point=target.point,
        )
        validate_evolved_state(result)
        return result


def validate_evolved_state(value: Any) -> Any:
    from pops.model import StateSpace
    from pops.time.references import canonical_handle
    from pops.fields._program_nonlinear_problem import validate_nonlinear_field_request

    if (
        value.op != "field_evolved_state"
        or len(value.inputs) < 2
        or value.attrs.get("projection_contract") not in (
            "pops.evolved-original-field-stage@1", "pops.evolved-original-field-stage@2")
    ):
        raise ValueError("original accumulation publication lost its versioned contract")
    packed = value.inputs[0]
    if packed.op != "solve_outcome_component":
        raise ValueError("original accumulation requires a consumed solve")
    solve = packed.inputs[0].inputs[0]
    validate_nonlinear_field_request(value.prog, solve)
    source = solve.attrs["source_contract"]
    captures = solve.inputs[2 : 2 + solve.attrs["capture_count"]]
    if len(value.inputs) != 1 + len(captures) or any(
        a is not b for a, b in zip(value.inputs[1:], captures, strict=True)
    ):
        raise ValueError("original accumulation changed its exact frozen captures")
    slot = value.attrs.get("previous_capture_index")
    if type(slot) is not int or not 0 <= slot < len(captures):
        raise ValueError("original accumulation lost its previous State slot")
    previous = captures[slot]
    from pops.time.points import TimePoint

    if previous.point != TimePoint(value.prog.clock, 0):
        raise ValueError("original accumulation requires the previous accepted frame endpoint")
    stage = _json_ready(source.get("evolved_stage"))
    indices = _projection_indices(stage, previous)
    partitioned = indices != tuple(range(len(stage["previous"])))
    expected_contract = "pops.evolved-original-field-stage@2" if partitioned else "pops.evolved-original-field-stage@1"
    expressions = tuple(source["accumulation"][index] for index in indices)
    declared_indices = _json_ready(value.attrs.get("component_indices"))
    if value.attrs["projection_contract"] != expected_contract or (
        partitioned and (not isinstance(declared_indices, list)
                         or any(type(index) is not int for index in declared_indices)
                         or declared_indices != list(indices))
    ) or (not partitioned and "component_indices" in value.attrs):
        raise ValueError("original accumulation changed its exact State partition")
    space = value.space
    if (
        not isinstance(space, StateSpace)
        or space.representation != "conservative"
        or space.centering != "cell"
        or space.sampling != "cell_average"
        or space.support is None
    ):
        raise ValueError(
            "original accumulation target requires explicit conservative cell-volume means"
        )
    if (
        previous.space != space
        or previous.state_ref != value.state_ref
        or previous.block != value.block
        or value.point != packed.point
        or value.prog is not packed.prog
        or value.region != packed.region
        or type(value.attrs.get("ncomp")) is not int
        or value.attrs.get("ncomp") != len(space.components)
        or value.attrs.get("field_problem_identity") != source["field_problem_identity"]
        or _json_ready(value.attrs.get("stage")) != stage
        or _json_ready(value.attrs.get("expressions")) != _json_ready(expressions)
    ):
        raise ValueError("original accumulation changed its exact solve, target or declaration")
    from pops._ir.quantity import QuantityRef

    # Quantity identity is authored by the same prior State, in target component order.
    expected = [
        QuantityRef(canonical_handle(previous.state_ref), component, space=space).to_data()
        for component in space.components
    ]
    if [stage["previous"][index] for index in indices] != expected or len(expressions) != len(space.components):
        raise ValueError("original accumulation changed its previous component tuple")
    validate_tau_data(stage["tau"], program=value.prog, point=value.point)
    return solve


def _projection_indices(stage: Any, previous: Any) -> tuple:
    """Select Q only through exact qualified prior State/component identities."""
    from pops._ir.quantity import QuantityRef
    from pops.time.references import canonical_handle

    rows = _json_ready(stage["previous"])
    if len(rows) != len(stage["accumulation"]):
        raise ValueError("original accumulation partition has mismatched declarations")
    result = []
    for component in previous.space.components:
        expected = QuantityRef(canonical_handle(previous.state_ref), component, space=previous.space).to_data()
        matches = [index for index, row in enumerate(rows) if row == expected]
        if len(matches) != 1:
            raise ValueError("original accumulation has no unique exact previous component")
        result.append(matches[0])
    return tuple(result)


def validate_evolved_partitions(program: Any) -> None:
    """Every partition of one solved stage is committed exactly once together."""
    groups = {}
    for value in program._values:
        if value.op == "field_evolved_state" and value.attrs.get("projection_contract") == "pops.evolved-original-field-stage@2":
            solve = validate_evolved_state(value)
            groups.setdefault(solve.id, (solve, []))[1].append(value)
    for solve, values in groups.values():
        seen = []
        for value in values:
            if program._commits.get(value.state_ref) is not value:
                raise ValueError("original accumulation partition must be committed exactly once")
            seen.extend(value.attrs["component_indices"])
        expected = list(range(len(solve.attrs["source_contract"]["evolved_stage"]["previous"])))
        if sorted(seen) != expected:
            raise ValueError("original accumulation partitions do not cover the complete solved stage")


def compile_accumulation(projection: Any, states: Any, unknowns: Any) -> tuple:
    from pops._ir.handle_expr import ValueExpr
    from pops.math import elliptic_terms
    from ._program_expression import encode_field_expression

    result = tuple(
        encode_field_expression(
            sum(
                (term.scale * term.coeff * ValueExpr(term.field) for term in elliptic_terms(row)), 0
            ),
            states,
            unknowns=unknowns,
        )
        for row in projection.accumulation
    )
    validate_encoded_tau(result, None)
    return result
