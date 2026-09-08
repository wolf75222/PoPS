"""Common equation products retain exact captures and explicit native dispositions."""
from dataclasses import replace
from fractions import Fraction

import pytest
from pops.solvers import CG
from pops.time import (Clock, Program, Region, RegionCapture, SolveRequest, SolveRequestError,
                       SolveUnknown, TemporalInterval, TemporalProblemRegion, TimePoint, ValueRef,
                       GraphProgramValue)
from pops.time.method_regions import temporal_value_signature
from pops.time.canonical_data import CanonicalData


def _request(*, offsets=(1,), interval=False, history=False, window=False):
    program = Program("temporal_equations")
    clock = program.clock
    previous = program.scalar_field("previous")
    if history:
        previous = program.value("timed_previous", previous, at=TimePoint(clock, step=-1))
    templates = tuple(program.value("stage_%d" % i, previous,
                      at=TimePoint(clock, offset)) for i, offset in enumerate(offsets))
    span = TemporalInterval(TimePoint(clock), TimePoint(clock, 1))
    unknowns = tuple(SolveUnknown("q%d" % i, value, interval=span if interval else None)
                     for i, value in enumerate(templates))
    refs = {unknown.name: ValueRef(100 + i) for i, unknown in enumerate(unknowns)}
    frozen_ref = ValueRef(200)
    captures = tuple(RegionCapture(refs[unknown.name], unknown.template.clock,
                                   unknown.template.point,
                                   signature=temporal_value_signature(unknown.template))
                     for unknown in unknowns) + (RegionCapture(frozen_ref, previous.clock,
                         previous.point, signature=temporal_value_signature(previous)),)
    equations = {}
    for i, unknown in enumerate(unknowns):
        signature = temporal_value_signature(unknown.template)
        # Each residual depends on the complete declared stage product and frozen
        # previous data. The captures are equation variables, independently seeded.
        node = GraphProgramValue(300 + i, "residual", "scalar_field", "linear_combine",
                                 tuple(refs.values()) + (frozen_ref,), clock,
                                 unknown.template.point, attrs={"attrs": {"coeffs": [[[0, weight]]
                                     for weight in [1] * len(refs) + [-1]]}})
        equations[unknown.name] = Region("equation_%d" % i, captures, (node,), ValueRef(node.node_id),
                                        result_signature=signature)
    problem = TemporalProblemRegion(equations, refs, {"previous": frozen_ref},
                                     windows={unknown.name: span for unknown in unknowns} if window else ())
    request = SolveRequest(problem, unknowns, {"previous": previous},
                            {unknown.name: None for unknown in unknowns})
    return program, request


def test_joint_stage_product_resolves_exact_outputs_and_refuses_native_without_publication():
    program, request = _request(offsets=(Fraction(1, 3), Fraction(2, 3)))
    request = replace(request, outputs=("q1", "q0"))
    resolved = request.problem.resolve(request, program=program)
    assert resolved.contract.to_data()["outputs"] == ["q1", "q0"]
    assert resolved.disposition.to_data()["code"] == "unsupported_unknown_product"
    assert resolved.to_data()["analytical_status"] == "unverified"
    before = program._ir_hash()
    with pytest.raises(SolveRequestError, match="unsupported_unknown_product"):
        program.solve(request, solver=CG(max_iter=4))
    assert program._ir_hash() == before


def test_interval_unknown_changes_identity_without_changing_point_native_identity():
    program, request = _request(interval=True)
    unknown = request.unknowns[0]
    point_unknown = replace(unknown, interval=None)
    assert "interval" not in point_unknown.to_data()
    assert unknown.identity != point_unknown.identity
    resolved = request.problem.resolve(request, program=program)
    assert resolved.disposition.to_data()["code"] == "unsupported_interval_unknown"
    with pytest.raises(SolveRequestError, match="unsupported_interval_unknown"):
        program.solve(request, solver=CG(max_iter=4))


def test_seed_is_reviewable_but_not_part_of_equation_identity():
    program, request = _request()
    first = request.problem.resolve(request, program=program)
    seeded = replace(request, seeds={"q0": request.equation_inputs["previous"]})
    second = seeded.problem.resolve(seeded, program=program)
    assert first.identity == second.identity
    assert first.initialization != second.initialization


@pytest.mark.parametrize("start,end", [(1, 0), (0, 0)])
def test_interval_requires_positive_duration(start, end):
    clock = Clock("macro")
    with pytest.raises(SolveRequestError, match="invalid_interval"):
        TemporalInterval(TimePoint(clock, start), TimePoint(clock, end))


def test_interval_refuses_cross_clock_endpoints_and_outside_template():
    with pytest.raises(SolveRequestError, match="invalid_interval"):
        TemporalInterval(TimePoint(Clock("a")), TimePoint(Clock("b"), 1))
    program, request = _request(offsets=(2,))
    with pytest.raises(SolveRequestError, match="invalid_interval"):
        replace(request.unknowns[0], interval=TemporalInterval(
            TimePoint(program.clock), TimePoint(program.clock, 1)))


@pytest.mark.parametrize("fault", ["owner", "shape", "point", "unbound"])
def test_forged_joint_capture_is_refused(fault):
    program, request = _request(offsets=(Fraction(1, 2), 1))
    region = request.problem.equations["q0"]
    capture = region.captures[0]
    signature = capture.signature.to_data()
    point, ref = capture.point, capture.value
    if fault == "owner":
        signature["block"] = {"qualified_id": "foreign"}
    elif fault == "shape":
        signature["components"] = 17
    elif fault == "point":
        point = TimePoint(program.clock, Fraction(1, 4))
    else:
        ref = ValueRef(900)
    bad_capture = RegionCapture(ref, capture.clock, point, signature=signature)
    # Preserve inner refs for the unbound case so the generic Region itself remains valid.
    captures = region.captures + (bad_capture,) if fault == "unbound" else (bad_capture, *region.captures[1:])
    bad_region = Region(region.name, captures, region.nodes, region.result,
                         result_signature=region.result_signature.to_data())
    equations = {**request.problem.equations, "q0": bad_region}
    bad = replace(request, problem=replace(request.problem, equations=equations))
    with pytest.raises(SolveRequestError, match="unbound_capture|capture_identity_mismatch"):
        bad.problem.resolve(bad, program=program)


def test_missing_duplicate_equations_and_duplicate_capture_ids_fail_closed():
    _, request = _request(offsets=(Fraction(1, 2), 1))
    problem = request.problem
    with pytest.raises(SolveRequestError, match="missing_equation"):
        replace(problem, equations={"q0": problem.equations["q0"]})
    with pytest.raises(SolveRequestError, match="duplicate_binding"):
        replace(problem, equations=(("q0", problem.equations["q0"]),) * 2)
    with pytest.raises(SolveRequestError, match="duplicate_capture"):
        replace(problem, unknown_captures={"q0": ValueRef(100), "q1": ValueRef(100)})


def test_timed_history_and_component_window_are_in_resolved_equation_identity():
    program, request = _request(history=True, window=True)
    data = request.problem.resolve(request, program=program).contract.to_data()
    assert data["windows"][0][1]["end"] == CanonicalData(TimePoint(program.clock, 1).to_data()).to_data()
    equation = data["equations"][0][1]
    assert equation["captures"][-1]["point"]["step"] == CanonicalData({"step": -1}).to_data()["step"]
    bad = replace(request.problem, windows={"q0": TemporalInterval(
        TimePoint(program.clock, 2), TimePoint(program.clock, 3))})
    request = replace(request, problem=bad)
    with pytest.raises(SolveRequestError, match="invalid_window"):
        bad.resolve(request, program=program)


def test_point_native_solver_cannot_silently_discard_interval_unknown():
    from test_general_solve_request import _request as linear_request
    program, request = linear_request()
    unknown = replace(request.unknowns[0], interval=TemporalInterval(
        TimePoint(program.clock), TimePoint(program.clock, 1)))
    with pytest.raises(SolveRequestError, match="unsupported_interval_unknown"):
        program.solve(replace(request, unknowns=(unknown,)), solver=CG(max_iter=4))


def _resolve_advance_region(program, *, interval=False, windows=False):
    """Retain an actual authored expansion as q - advance(U_n)=0 in common regions."""
    graph = program.to_graph()
    endpoint = next(iter(program._commits.values()))
    unknown = SolveUnknown("endpoint", endpoint, interval=TemporalInterval(
        TimePoint(program.clock), TimePoint(program.clock, 1)) if interval else None)
    signature = temporal_value_signature(endpoint)
    unknown_ref = ValueRef(1000000)
    captures = [RegionCapture(unknown_ref, endpoint.clock, endpoint.point, signature=signature)]
    input_refs, inputs, inner = {}, {}, []
    live = {value.id: value for value in program._values}
    for node in graph.nodes:
        if node.kind == "commit":
            continue
        if node.kind == "state_read":
            value = live[node.node_id]
            name = "initial_%d" % node.node_id
            input_refs[name] = ValueRef(node.node_id)
            inputs[name] = value
            captures.append(RegionCapture(input_refs[name], value.clock, value.point,
                                           signature=temporal_value_signature(value)))
        else:
            inner.append(node)
    residual = GraphProgramValue(1000001, "advance_residual", endpoint.vtype,
        "linear_combine", (unknown_ref, ValueRef(endpoint.id)), endpoint.clock,
        endpoint.point, attrs={"attrs": {"coeffs": [[[0, 1]], [[0, -1]]]}})
    region = Region("advance_equation", captures, (*inner, residual), ValueRef(residual.node_id),
                     result_signature=signature, clocks=graph.clocks)
    problem = TemporalProblemRegion({"endpoint": region}, {"endpoint": unknown_ref}, input_refs,
        windows={"endpoint": TemporalInterval(TimePoint(program.clock), TimePoint(program.clock, 1))}
        if windows else ())
    request = SolveRequest(problem, (unknown,), inputs, {"endpoint": None})
    resolved = problem.resolve(request, program=program)
    assert resolved.disposition.to_data()["disposition"] == "unavailable"
    return region, resolved


@pytest.mark.parametrize("form", ["explicit_rk", "dirk", "imex", "splitting", "multistep", "multirate"])
def test_existing_method_expansions_author_and_resolve_as_common_equation_regions(form):
    import pops.lib.time as lt
    from test_time_std_imex_lie_ab import _authoring
    if form == "multirate":
        from test_multirate_history_contract import _interpolated_program
        from pops.time import LinearInterpolation
        program = _interpolated_program(LinearInterpolation())
    else:
        state, explicit, implicit, fields = _authoring("common_" + form)
        if form == "explicit_rk":
            program = lt.SSPRK2(state, rate=explicit, fields=fields)
        elif form == "dirk":
            program = lt.DIRK(state, implicit_operator=implicit, tableau=lt.IMPLICIT_MIDPOINT_TABLEAU)
        elif form == "imex":
            program = lt.IMEX(state, explicit_operator=explicit, implicit_operator=implicit,
                              fields_operator=fields)
        elif form == "multistep":
            program = lt.AdamsBashforth(state, rate=explicit, order=3, fields=fields)
        else:
            def flow(program, value, fraction, *, at):
                return program.value("flow", value + fraction * program.dt * value, at=at)
            program = lt.Strang(state, first=flow, second=flow)
    region, resolved = _resolve_advance_region(program, windows=form == "multirate")
    kinds = {node.kind for node in region.nodes}
    if form == "multirate":
        assert "loop" in kinds and "synchronize" in kinds
        assert resolved.contract.to_data()["windows"]
    elif form == "multistep":
        assert "history" in repr(region.to_data())
    else:
        assert region.nodes


def test_space_time_interval_equation_resolves_without_repeated_rate_evaluations():
    program, request = _request(interval=True)
    region = request.problem.equations["q0"]
    assert [node.op for node in region.nodes] == ["linear_combine"]
    resolved = request.problem.resolve(request, program=program)
    assert resolved.disposition.to_data()["code"] == "unsupported_interval_unknown"
