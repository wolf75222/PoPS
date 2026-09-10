"""Authenticate a general request against the existing executable solve IR."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from .equation_identity import _equation_value as _equation_value

from pops.identity import make_identity
from pops.time.solve_request import SolveRequestError, SolveUnknown


def _linear_equation(program: Any, token: Any) -> dict[str, Any]:
    from pops.time._program.serialization import _json_ready

    operator, rhs = token.inputs[:2]
    return {
        "kind": "matrix_free_linear",
        "operator": _equation_value(program, operator),
        "rhs": _equation_value(program, rhs),
        "properties": _json_ready(token.attrs["operator_properties"]),
        "nullspace": _json_ready(token.attrs["nullspace_contract"]),
        "gauge": _json_ready(token.attrs["gauge_contract"]),
    }


def _linear_derivative(operator: Any, selected: Any) -> dict[str, Any]:
    if selected.route != "exact":
        raise SolveRequestError(
            "unsupported_derivative",
            "the prepared linear adapter applies the declared linear operator exactly; "
            "it does not implement an approximate or finite-difference derivative strategy")
    # These operations act linearly on the apply input; coefficients and boundaries
    # are frozen captures. The prepared affine session independently checks the
    # numerical zero response before the native solver can publish a candidate.
    exact_ops = {
        "apply_in", "apply_out", "scalar_field", "vector_field", "linear_combine",
        "laplacian", "gradient", "divergence", "apply_laplacian_coeff",
        "fill_boundary", "scalar_field_component", "componentwise_laplacian", "field_problem_apply",
    }
    for node in operator.attrs.get("apply_block") or ():
        if node.op == "field_problem_apply":
            from pops.fields._program_problem import validate_field_apply
            validate_field_apply(node)
        if node.op not in exact_ops:
            raise SolveRequestError(
                "unsupported_derivative",
                "linear request has no exact derivative realization for apply operation %r"
                % node.op)
    return {"route": "exact", "provenance": "prepared_affine_linear_operator",
            "runtime_obligation": "prepared_affine_zero_response"}


def validate_solve_request_node(program: Any, token: Any) -> None:
    """Recheck the equation/initialization seal at validation and native emission."""
    request = token.attrs.get("solve_request")
    if request is None:
        return
    if token.op == "solve_spatial_nonlinear":
        from pops.time._program.spatial_solve import validate_spatial_request

        validate_spatial_request(program, token)
        return
    if token.op != "solve_linear" or not isinstance(request, Mapping):
        raise SolveRequestError("unsupported_lowering", "request has no authenticated native adapter")
    from pops.time._program.serialization import _json_ready

    request = _json_ready(request)
    expected_equation = _linear_equation(program, token)
    equation_identity = make_identity("solve-equation", expected_equation).token
    if request.get("equation_identity") != equation_identity:
        raise SolveRequestError(
            "equation_identity_drift", "the lowered operator, coefficient or RHS changed after binding")
    if request.get("equation_inputs") != expected_equation:
        raise SolveRequestError("equation_identity_drift", "the recorded equation inputs are not exact")
    seed = None if len(token.inputs) == 2 else _equation_value(program, token.inputs[2])
    if request.get("initialization_identity") != make_identity("solve-initialization", seed).token:
        raise SolveRequestError("initialization_identity_drift", "the lowered seed changed after binding")
    if request.get("seed") != seed:
        raise SolveRequestError("initialization_identity_drift", "the recorded seed is not exact")
    if request.get("solver_identity") != token.attrs.get("solver_identity"):
        raise SolveRequestError("solver_identity_drift", "the selected solver changed after binding")
    from pops.time.solve_request import DerivativeStrategy

    actual_derivative = _linear_derivative(
        program._canonical_value(token.inputs[0]), DerivativeStrategy("exact"))
    if dict(request.get("derivative", {})) != actual_derivative:
        raise SolveRequestError("unsupported_derivative", "the selected derivative contract changed")
    unknowns = request.get("unknowns")
    if not isinstance(unknowns, list) or len(unknowns) != 1 or not isinstance(unknowns[0], dict):
        raise SolveRequestError("invalid_unknown", "the native unknown product changed")
    from pops.time.solve_request import residual_name
    unknown_name = residual_name(unknowns[0].get("name"), "SolveUnknown name")
    unknown = SolveUnknown(unknown_name, token)
    if unknown.to_data() != unknowns[0] or request.get("outputs") != [unknown.name]:
        raise SolveRequestError("unknown_type_mismatch", "the result identity/owner/space changed")
    problem_data = {key: request[key] for key in (
        "equation", "physical_problem", "unknowns", "residual_interpretation", "error_interpretation")}
    if problem_data["equation"] != equation_identity or request.get("problem_identity") != \
            make_identity("solve-problem", problem_data).token:
        raise SolveRequestError("problem_identity_drift", "the physical problem mapping changed")
    expected_lowering = {"disposition": "native", "adapter": "prepared_affine_linear_problem"}
    if request.get("lowering") != expected_lowering:
        raise SolveRequestError("unsupported_lowering", "the request claims another native realization")


def build_solve_request(program: Any, request: Any, prepared: Any, *, name: Any) -> Any:
    from pops.linalg import LinearProblem
    from pops.time._program.value_validation import require_owned, validate_input_regions
    from pops.time.solve_outcome import ResidualSolution, SolveOutcome
    from pops.time.method_regions import TemporalProblemRegion, resolve_temporal_problem
    if type(request.problem) is TemporalProblemRegion:
        resolved = resolve_temporal_problem(request, program=program)
        native = resolved.disposition.to_data()
        raise SolveRequestError(native["code"], native["detail"])
    if any(unknown.interval is not None for unknown in request.unknowns):
        raise SolveRequestError("unsupported_interval_unknown", "point-native adapters cannot realize interval unknowns")
    from pops.time.implicit_stage import ImplicitStage

    if type(request.problem) is ImplicitStage:
        from pops.time._program.spatial_solve import build_spatial_request

        return build_spatial_request(program, request, prepared, name=name)

    # Problem families can lower to this adapter, but an arbitrary descriptor never
    # becomes native merely by declaring the same strings as a supported problem.
    if type(request.problem) is not LinearProblem:
        raise SolveRequestError(
            "unsupported_lowering", "this problem has no general-request native realization")
    if len(request.unknowns) != 1:
        raise SolveRequestError(
            "unsupported_unknown_product",
            "the prepared linear adapter supports one typed field (including a packed vector); "
            "a joint unknown product needs an explicit numerical adapter")
    problem = request.problem
    if problem.initial_guess is not None:
        raise SolveRequestError(
            "duplicate_seed_authority", "put initialization in request.seeds, not also in LinearProblem")
    if set(request.equation_inputs) != {"operator", "rhs"}:
        raise SolveRequestError(
            "equation_input_mismatch", "linear equation inputs must name exactly operator and rhs")
    unknown = request.unknowns[0]
    seed = request.seeds[unknown.name]
    bindings = (*request.equation_inputs.values(), unknown.template)
    if seed is not None:
        bindings += (seed,)
    try:
        for value in bindings:
            require_owned(program, value, "SolveRequest binding")
        region = program._current_region()
        validate_input_regions(program, bindings, region, "SolveRequest binding")
    except (TypeError, ValueError) as exc:
        raise SolveRequestError("invalid_binding_scope", str(exc)) from exc
    for key, actual in (("operator", problem.operator), ("rhs", problem.rhs)):
        supplied = request.equation_inputs[key]
        if supplied.prog is not program or supplied.id != actual.id:
            raise SolveRequestError(
                "equation_input_mismatch", "request %s is not the problem's exact SSA input" % key)
    operator = program._canonical_value(problem.operator)
    derivative = _linear_derivative(operator, request.derivative)
    if request.residual_interpretation != "operator(x)-rhs" \
            or request.error_interpretation != "solver_residual_l2":
        raise SolveRequestError(
            "unsupported_residual_interpretation",
            "the linear adapter realizes operator(x)-rhs with the selected solver's L2 residual")
    outcome = prepared.build_program_solve(
        program=program, problem=replace(problem, initial_guess=seed), name=name)
    if not isinstance(outcome, SolveOutcome) or outcome._program is not program:
        raise SolveRequestError("invalid_solve_provider", "provider did not return this Program's outcome")
    token = outcome._token
    actual_unknown = SolveUnknown(unknown.name, token)
    if unknown.to_data() != actual_unknown.to_data():
        raise SolveRequestError(
            "unknown_type_mismatch", "declared unknown does not match the native result's owner/space/point")
    equation = _linear_equation(program, token)
    equation_identity = make_identity("solve-equation", equation).token
    problem_data = {
        "equation": equation_identity,
        "physical_problem": request.problem_metadata.to_data(),
        "unknowns": [item.to_data() for item in request.unknowns],
        "residual_interpretation": request.residual_interpretation,
        "error_interpretation": request.error_interpretation,
    }
    problem_identity = make_identity("solve-problem", problem_data).token
    seed_data = None if seed is None else _equation_value(program, seed)
    contract = {
        "schema": "pops.solve-request.v1",
        **problem_data,
        "problem_identity": problem_identity,
        "equation_identity": equation_identity,
        "initialization_identity": make_identity("solve-initialization", seed_data).token,
        "equation_inputs": equation,
        "seed": seed_data,
        "outputs": list(request.outputs),
        "derivative": derivative,
        "solver_identity": token.attrs["solver_identity"],
        "source_mapping": {key: {"name": value.name, "operation": value.op}
                           for key, value in request.equation_inputs.items()},
        "lowering": {"disposition": "native", "adapter": "prepared_affine_linear_problem"},
        "dependency_boundary": "declared_implicit_unknown",
    }
    token = program._replace_value(token, attrs={**token.attrs, "solve_request": contract})
    outcome._token = token
    original_project = outcome._factory

    def project(consumed: Any) -> Any:
        value = original_project(consumed)
        value = program._replace_value(value, attrs={
            **value.attrs, "unknown_identity": unknown.identity,
            "problem_identity": problem_identity, "result_name": unknown.name})
        return ResidualSolution(
            (value,), names=(unknown.name,), unknowns=(unknown,),
            problem_identity=problem_identity)

    outcome._factory = project
    validate_solve_request_node(program, token)
    return outcome


__all__ = ["build_solve_request", "validate_solve_request_node"]
