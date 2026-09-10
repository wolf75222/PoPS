"""General solve requests preserve equation, initialization and result authority."""
from __future__ import annotations

from dataclasses import replace

import pytest

from pops.linalg import LinearOperatorProperties, LinearProblem
from pops.solvers import CG
from pops.time import (
    DerivativeStrategy, FailRun, Program, ResidualSolution,
    SolveRequest, SolveRequestError, SolveUnknown,
)


def _request(*, coefficient=2.0, components=1):
    program = Program("general-request")
    rhs = program.scalar_field("rhs", ncomp=components)
    seed = program.scalar_field("seed", ncomp=components)
    operator = program.matrix_free_operator(
        "operator", domain="scalar" if components == 1 else "vector",
        range_="scalar" if components == 1 else "vector", ncomp=components)
    program.set_apply(operator, lambda _program, _out, value: coefficient * value)
    unknown = SolveUnknown("solution", rhs)
    request = SolveRequest(
        LinearProblem(operator, rhs, nullspace=None,
                      properties=LinearOperatorProperties.symmetric_positive_definite()),
        unknowns=(unknown,), equation_inputs={"operator": operator, "rhs": rhs},
        seeds={"solution": seed},
        problem_metadata={"equation": "2 q = b", "accumulation": None})
    return program, request


def _solve(program, request):
    outcome = program.solve(request, solver=CG(max_iter=4))
    result = outcome.consume(action=FailRun())
    token = next(node for node in reversed(program._values) if node.op == "solve_linear")
    return result, token.attrs["solve_request"]


def test_seed_changes_initialization_but_not_equation_or_problem():
    program, request = _request()
    first, before = _solve(program, request)
    second, after = _solve(program, replace(request, seeds={"solution": None}))
    assert before["equation_inputs"] == after["equation_inputs"]
    assert before["equation_identity"] == after["equation_identity"]
    assert before["problem_identity"] == after["problem_identity"]
    assert before["initialization_identity"] != after["initialization_identity"]
    assert first.problem_identity == second.problem_identity
    assert before["lowering"]["disposition"] == "native"
    assert before["source_mapping"]["rhs"]["name"] == "rhs"
    assert before["derivative"]["route"] == "exact"


def test_frozen_coefficient_changes_problem_identity():
    program, request = _request(coefficient=2.0)
    _, before = _solve(program, request)
    other_program, other = _request(coefficient=3.0)
    _, after = _solve(other_program, other)
    assert before["equation_identity"] != after["equation_identity"]
    assert before["problem_identity"] != after["problem_identity"]


def test_affine_seed_is_encoded_exactly_without_joining_the_equation_inputs():
    program, request = _request()
    _, before = _solve(program, request)
    seed = program.value("scaled_seed", 3.0 * request.problem.rhs)
    _, after = _solve(program, replace(request, seeds={"solution": seed}))
    assert before["problem_identity"] == after["problem_identity"]
    assert before["initialization_identity"] != after["initialization_identity"]


def test_explicit_tuple_projects_one_consumed_authority_by_index_name_and_unknown():
    program, request = _request(components=2)
    result, contract = _solve(program, request)
    assert type(result) is ResidualSolution
    assert result[0] is result["solution"] is result[request.unknowns[0]]
    assert result[0].attrs["unknown_identity"] == request.unknowns[0].identity
    assert result[0].attrs["problem_identity"] == contract["problem_identity"]
    assert len([node for node in program._values if node.op == "solve_outcome"]) == 1
    assert result[0].attrs["ncomp"] == 2
    with pytest.raises(KeyError):
        result[SolveUnknown("other", request.unknowns[0].template)]


def test_graph_separates_declared_unknown_from_seed():
    program, request = _request()
    _solve(program, request)
    graph = program.to_graph()
    solve = next(node for node in graph.nodes if node.kind == "solve")
    assert solve.initial is not None
    assert solve.initial != solve.unknown
    assert next(node for node in graph.nodes if node.node_id == solve.unknown.node_id).kind == "unknown"
    assert solve.to_data()["initial"] == solve.initial.to_data()


@pytest.mark.parametrize(("change", "code"), [
    ({"seeds": {}}, "missing_unknown_binding"),
    ({"seeds": (("solution", None), ("solution", None))}, "duplicate_binding"),
    ({"outputs": ("absent",)}, "invalid_outputs"),
])
def test_missing_duplicate_and_unknown_bindings_are_structured(change, code):
    _, request = _request()
    with pytest.raises(SolveRequestError) as caught:
        replace(request, **change)
    assert caught.value.to_data()["code"] == code


def test_duplicate_unknown_and_unrealized_product_are_not_silently_flattened():
    program, request = _request()
    with pytest.raises(SolveRequestError, match="duplicate_unknown"):
        replace(request, unknowns=request.unknowns * 2)
    other = SolveUnknown("field2", request.unknowns[0].template)
    product = replace(request, unknowns=(*request.unknowns, other),
                      seeds={"solution": None, "field2": None}, outputs=("field2", "solution"))
    assert product.outputs == ("field2", "solution")
    before = program._ir_hash()
    with pytest.raises(SolveRequestError, match="unsupported_unknown_product"):
        program.solve(product, solver=CG(max_iter=4))
    assert program._ir_hash() == before


def test_forged_frozen_input_refuses_before_authoring_publication():
    program, request = _request()
    forged = replace(request, equation_inputs={
        "operator": request.problem.operator, "rhs": request.seeds["solution"]})
    before = program._ir_hash()
    with pytest.raises(SolveRequestError, match="equation_input_mismatch"):
        program.solve(forged, solver=CG(max_iter=4))
    assert program._ir_hash() == before


def test_foreign_seed_has_a_structured_scope_refusal():
    program, request = _request()
    foreign = Program("foreign").scalar_field("seed")
    with pytest.raises(SolveRequestError, match="invalid_binding_scope"):
        program.solve(replace(request, seeds={"solution": foreign}), solver=CG(max_iter=4))


def test_completed_operator_cannot_change_the_bound_equation_before_consume():
    program, request = _request()
    outcome = program.solve(request, solver=CG(max_iter=4))
    operator = program._canonical_value(request.problem.operator)
    program._replace_value(operator, attrs={
        **operator.attrs, "apply_result": 3.0 * operator.attrs["apply_in"]})
    with pytest.raises(SolveRequestError, match="equation_identity_drift"):
        outcome.consume(action=FailRun())
    assert not any(node.op == "solve_outcome" for node in program._values)


@pytest.mark.parametrize("route", ("approximate", "finite_difference", "unavailable"))
def test_unimplemented_derivative_strategy_is_not_substituted(route):
    program, request = _request()
    with pytest.raises(SolveRequestError, match="unsupported_derivative"):
        program.solve(replace(request, derivative=DerivativeStrategy(route)), solver=CG(max_iter=4))


def test_opaque_call_without_derivative_authority_is_not_differentiable():
    with pytest.raises(SolveRequestError, match="unsupported_derivative"):
        DerivativeStrategy("exact").for_native_call(lambda value: value)


def test_request_metadata_is_deeply_frozen_and_not_an_equation_rewrite():
    program, request = _request()
    data = {"physical_equation": ["Q(q+) - U_n - tau D(Q(q+))"],
            "accumulation": {"current": "Q(q+)", "previous": "U_n"}}
    request = replace(request, problem_metadata=data)
    data["physical_equation"].clear()
    assert request.problem_metadata.to_data()["physical_equation"]
    with pytest.raises(SolveRequestError, match="unsupported_residual_interpretation"):
        program.solve(replace(request, residual_interpretation="Q(x)-rhs"), solver=CG(max_iter=4))


def test_legacy_scalar_consume_spelling_and_failure_gate_are_unchanged():
    program, request = _request()
    outcome = program.solve(request.problem, solver=CG(max_iter=4))
    with pytest.raises(TypeError, match="not readable"):
        _ = outcome.token
    with pytest.raises(TypeError, match="requires action"):
        outcome.consume(action=None)
    value = outcome.consume(action=FailRun())
    assert value.op == "solve_outcome_component"
    assert not isinstance(value, ResidualSolution)


def test_general_result_requires_an_explicit_failure_disposition():
    program, request = _request()
    outcome = program.solve(request, solver=CG(max_iter=4))
    with pytest.raises(SolveRequestError, match="unconsumed_result"):
        _ = outcome.token
    with pytest.raises(SolveRequestError, match="missing_failure_disposition"):
        outcome.consume()
    assert not any(node.op == "solve_outcome" for node in program._values)


def test_consumed_result_cannot_escape_its_nested_legal_scope():
    from typed_program_support import typed_state

    program = Program("nested-request")
    state = typed_state(program, "tracer")
    operator = program.matrix_free_operator("A", domain="state", range_="state", ncomp=1)
    program.set_apply(operator, lambda _p, _out, value: 2.0 * value)
    leaked = []

    def body(prog, current):
        problem = LinearProblem(operator, current, nullspace=None,
                                properties=LinearOperatorProperties.symmetric_positive_definite())
        request = SolveRequest(problem, unknowns=(SolveUnknown("q", current),),
                               equation_inputs={"operator": operator, "rhs": current},
                               seeds={"q": None})
        result = prog.solve(request, solver=CG(max_iter=4)).consume(action=FailRun())["q"]
        leaked.append(result)
        return result

    program.range(state, 2, body)
    with pytest.raises(SolveRequestError, match="result_scope"):
        program.norm2(leaked[0])


def test_explicit_cycle_is_structurally_refused_not_treated_as_an_implicit_problem():
    from pops.time._program.solve_request import _equation_value

    program, request = _request()
    rhs = request.problem.rhs
    # Deliberately corrupt an issued immutable node, as an adversarial compiler-input test.
    object.__setattr__(rhs, "inputs", (rhs,))
    with pytest.raises(SolveRequestError, match="explicit_dependency_cycle"):
        _equation_value(program, rhs)


def test_equation_closure_retains_ssa_aliases_without_expanding_shared_subgraphs():
    from pops.time._program.solve_request import _equation_value

    program = Program("shared-closure")
    value = program.scalar_field("seed")
    for index in range(20):
        # Two references to one predecessor must remain two edges to that same
        # version, rather than recursively duplicating its complete subgraph.
        value = program._new("scalar_field", "fixture_pair", (value, value), {},
                             "pair_%d" % index, None)
    data = _equation_value(program, value)
    assert len(data["values"]) == 21
    assert data["values"][0]["inputs"][0] == data["values"][0]["inputs"][1]


def test_problem_input_adapter_uses_existing_solve_entry_only():
    program, request = _request()

    class AdaptedProblem:
        def default_program_solver(self):
            return CG(max_iter=4)

        def bind_program_inputs(self, *, program: Program, values, at):
            assert values == (request.problem.rhs,)
            assert at == request.problem.rhs.point
            return request

    result = program.solve(AdaptedProblem(), values=(request.problem.rhs,),
                           at=request.problem.rhs.point).consume(action=FailRun())
    assert isinstance(result, ResidualSolution)
    with pytest.raises(TypeError, match="bind_program_inputs"):
        program.solve(request.problem, values=(request.problem.rhs,), solver=CG(max_iter=4))
    with pytest.raises(TypeError, match="explicit solver"):
        program.solve(request.problem)


def test_state_result_materialization_follows_consumption_and_preserves_solver_footprint():
    from typed_program_support import typed_state
    from pops.codegen.program_codegen import emit_cpp_program

    program = Program("pointwise-state-request")
    state = typed_state(program, "tracer")
    endpoint = typed_state(program, "tracer", state_name="U").next
    operator = program.matrix_free_operator("A", domain="state", range_="state", ncomp=1)
    program.set_apply(operator, lambda _p, _out, value: 2.0 * value)
    problem = LinearProblem(operator, state, nullspace=None,
                            properties=LinearOperatorProperties.symmetric_positive_definite())
    request = SolveRequest(problem, unknowns=(SolveUnknown("q", state),),
                           equation_inputs={"operator": operator, "rhs": state}, seeds={"q": None})
    result = program.solve(request, solver=CG(max_iter=4)).consume(action=FailRun())["q"]
    program.commit(endpoint, program.value("accepted", result, at=endpoint.point))
    source = emit_cpp_program(program)
    assert "ctx.alloc_scalar_field(1, 0)" in source
    assert "->ghosts() != ctx.state(0).ghosts()" in source
    consume = source.index(".consume(pops::SolveConsumption::kAccept)")
    copy = source.index("pops::PureFieldAlgebra::copy(materialized,")
    assert consume < copy < source.index("ctx.commit_many(")
