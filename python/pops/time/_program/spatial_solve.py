"""The exact request-to-Program adapter for scalar spatial implicit stages."""
from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

from pops.identity import make_identity
from pops.time._graph.base import CanonicalData
from pops.time.solve_request import SolveRequestError, SolveUnknown
from pops.time.values import ProgramValue


@dataclass(frozen=True, slots=True)
class PreparedSpatialNewton:
    controls: CanonicalData
    identity: Any

    def build_program_solve(self, *, program: Any, problem: Any, name: Any = None) -> Any:
        from pops.time.solve_request import SolveRequest

        if type(problem) is not SolveRequest:
            raise SolveRequestError("unsupported_lowering", "spatial Newton requires a SolveRequest")
        return problem.build_program_solve(program=program, prepared_solver=self, name=name)


def prepare_spatial_newton(solver: Any) -> PreparedSpatialNewton:
    from pops.solvers.nonlinear import Newton
    from pops.identity.scalar import scalar_data

    if type(solver) is not Newton:
        raise SolveRequestError("unsupported_solver", "spatial adapter requires the declared Newton")
    controls = {key: value if type(value) is int else scalar_data(value)
                for key, value in solver.options().items()}
    frozen = CanonicalData(controls, where="spatial Newton controls")
    return PreparedSpatialNewton(frozen, make_identity("prepared-spatial-newton", frozen.to_data()))


def spatial_scalar(value: Any) -> int | float:
    if type(value) in (int, float):
        return value
    if isinstance(value, Mapping):
        data = value.get("scalar", value)
        if data.get("kind") == "integer":
            return int(data["value"])
        if data.get("kind") == "binary64":
            return float.fromhex(data["value"])
    raise SolveRequestError("invalid_solver_controls", "spatial controls require exact numeric literals")


def spatial_newton_options(controls: Any) -> dict[str, Any]:
    from pops.solvers import Newton

    return Newton(**{key: spatial_scalar(value) for key, value in controls.items()}).options()


def _equation(program: Any, token: Any) -> dict[str, Any]:
    from pops.time._program.solve_request import _equation_value

    return {"kind": "spatial_accumulation_residual",
            "previous": _equation_value(program, token.inputs[0]),
            "residual": _equation_value(program, token.attrs["residual"]),
            "implicit_unknown": _equation_value(program, token.attrs["iterate"]),
            "mapping": token.attrs["physical_mapping"]}


def _request_data(program: Any, token: Any, unknown: SolveUnknown,
                  physical_problem: Any) -> dict[str, Any]:
    from pops.time._program.solve_request import _equation_value
    from pops.time._program.serialization import _json_ready

    equation = _json_ready(_equation(program, token))
    equation_id = make_identity("solve-equation", equation).token
    seed = None if len(token.inputs) == 1 else _equation_value(program, token.inputs[1])
    problem = {"equation": equation_id, "physical_problem": physical_problem,
               "unknowns": [unknown.to_data()],
               "residual_interpretation": "Q(q)-U_n-tau*R(Q(q))",
               "error_interpretation": "spatial_residual_l2"}
    return {"schema_version": 1, **problem,
            "equation_identity": equation_id, "equation_inputs": equation,
            "problem_identity": make_identity("solve-problem", problem).token,
            "seed": seed, "initialization_identity": make_identity("solve-initialization", seed).token,
            "outputs": [unknown.name], "solver_identity": token.attrs["solver_identity"],
            "derivative": {"route": "finite_difference", "scheme": "central_full_residual",
                           "step": token.attrs["finite_difference_step"]},
            "lowering": {"disposition": "native", "adapter": "prepared_spatial_residual"},
            "source_mapping": {"residual_operations": [node.op for node in token.attrs["residual_block"]]}}


def validate_spatial_request(program: Any, token: Any) -> None:
    from pops.time._program.serialization import _json_ready

    actual = _json_ready(token.attrs["solve_request"])
    unknowns = actual.get("unknowns")
    if not isinstance(unknowns, list) or len(unknowns) != 1:
        raise SolveRequestError("invalid_unknown", "spatial request requires one scalar unknown")
    unknown = SolveUnknown(unknowns[0]["name"], token)
    expected = _request_data(program, token, unknown, actual.get("physical_problem"))
    if actual != _json_ready(expected):
        raise SolveRequestError("equation_identity_drift", "spatial residual, seed or strategy changed")
    controls = _json_ready(token.attrs["newton_controls"])
    spatial_newton_options(controls)
    if token.attrs["solver_identity"] != make_identity("prepared-spatial-newton", controls).token:
        raise SolveRequestError("solver_identity_drift", "spatial Newton controls changed")


def validate_spatial_commit(program: Any, token: Any) -> None:
    """Only the complete solved conservative stage has an admitted exchange quadrature."""
    from pops.time.references import handle_data
    from pops.time._program.serialization import _json_ready
    from pops.time._program.region_validation import _BLOCK_KEYS

    pending = list(program._values)
    while pending:
        value = pending.pop()
        # project() mutates its input even when the returned SSA alias is discarded.
        # Such a correction requires its own accepted inventory/exchange accounting.
        if value.op == "project" and value.block == token.block:
            raise SolveRequestError("unsupported_commit", "spatial stage projection requires a declared correction quadrature")
        for key in _BLOCK_KEYS:
            pending.extend(value.attrs.get(key) or ())

    def unit_wrapper(value: Any) -> Any:
        while value.op == "linear_combine" and len(value.inputs) == 1 \
                and dict(value.attrs["coeffs"][0]) == {0: 1}:
            value = value.inputs[0]
        return value

    commits = [value for value in program._commits.values() if value.block == token.block]
    if len(commits) != 1:
        raise SolveRequestError("unsupported_commit", "spatial solve requires one exact conservative commit")
    endpoint = unit_wrapper(commits[0])
    accumulation = token.attrs["physical_mapping"]["accumulation"]
    if accumulation is not None:
        handle = endpoint.attrs.get("operator_handle")
        if endpoint.op != "local_transform" or len(endpoint.inputs) != 1 or handle is None \
                or _json_ready(handle_data(handle)) != _json_ready(accumulation):
            raise SolveRequestError("unsupported_commit", "spatial solve must commit its exact Q(candidate) mapping")
        endpoint = unit_wrapper(endpoint.inputs[0])
    if endpoint.op != "solve_outcome_component" or endpoint.attrs.get("index") != 0 \
            or len(endpoint.inputs) != 1:
        raise SolveRequestError("unsupported_commit", "weighted or transformed spatial solve commits require a declared quadrature")
    outcome = endpoint.inputs[0]
    if outcome.op != "solve_outcome" or len(outcome.inputs) != 1 or outcome.inputs[0] is not token:
        raise SolveRequestError("unsupported_commit", "spatial solve is not the committed consumed outcome")


def spatial_rate_weight(token: Any, rate: Any) -> Any:
    """Recover tau from the authenticated residual, including exact dt powers."""
    from pops.time.values import _Coeff

    residual = token.attrs["residual"]
    if residual.op != "linear_combine":
        raise SolveRequestError("unsupported_residual_operation", "spatial residual is not affine in its rate")
    coefficients = [coefficient for value, coefficient in zip(
        residual.inputs, residual.attrs["coeffs"], strict=True) if value is rate]
    if len(coefficients) > 1:
        raise SolveRequestError("equation_identity_drift", "spatial residual repeats one rate input")
    return (-_Coeff(coefficients[0] if coefficients else {})).to_polynomial()


def build_spatial_request(program: Any, request: Any, prepared: Any, *, name: Any) -> Any:
    from pops.time._program.value_validation import (
        require_top_level, require_region, require_compatible_spaces,
    )
    from pops.time._program.serialization import _json_ready
    from pops.time.solve_outcome import ResidualSolution, SolveOutcome
    from pops.time.implicit_diffusion import ImplicitDiffusionStage
    from pops.time.references import handle_data
    from pops.identity.scalar import scalar_data

    stage = request.problem
    if type(stage) is not ImplicitDiffusionStage or type(prepared) is not PreparedSpatialNewton:
        raise SolveRequestError("unsupported_lowering", "implicit diffusion requires spatial Newton")
    if len(request.unknowns) != 1:
        raise SolveRequestError("unsupported_unknown_product", "spatial adapter is scalar")
    if request.derivative.route != "finite_difference":
        raise SolveRequestError("unsupported_derivative", "select full-residual finite_difference explicitly")
    if program._recording:
        raise SolveRequestError("unsupported_scope", "spatial solve recording requires a top-level region")
    unknown = request.unknowns[0]
    previous, seed = stage.previous, request.seeds[unknown.name]
    for value in (previous, unknown.template, *(() if seed is None else (seed,))):
        if not isinstance(value, ProgramValue) or value.vtype != "state":
            raise SolveRequestError("invalid_binding", "spatial stage requires typed scalar State values")
        require_top_level(program, value, "spatial solve binding")
        if value.block != previous.block or value.logical_shape.get("n_comp", 1) != 1:
            raise SolveRequestError("unknown_type_mismatch", "spatial stage owner/components differ")
        require_compatible_spaces(previous.space, value.space, "spatial solve binding", typed_pair=True)
    expected_inputs = {"previous": previous, "rate": stage.rate,
                       "accumulation": stage.accumulation, "tau": stage.tau}
    if set(request.equation_inputs) != set(expected_inputs) or any(
            request.equation_inputs[key] is not value for key, value in expected_inputs.items()):
        raise SolveRequestError("equation_input_mismatch", "stage inputs differ from the bound request")
    if request.outputs != (unknown.name,) or request.residual_interpretation != "Q(q)-U_n-tau*R(Q(q))" \
            or request.error_interpretation != "spatial_residual_l2":
        raise SolveRequestError("unsupported_result", "spatial stage result or residual contract differs")
    step = stage.finite_difference_step
    if isinstance(step, bool) or not isinstance(step, (int, float)) or not math.isfinite(step) or step <= 0:
        raise SolveRequestError("invalid_derivative", "finite difference step must be positive and finite")
    sub = []
    program._recording.append(sub)
    try:
        iterate = program._new("state", "state", (), {}, "spatial_iterate", previous.block,
                               space=unknown.template.space, point=unknown.template.point)
        conserved = iterate if stage.accumulation is None else program._call(
            stage.accumulation, iterate, name="spatial_accumulation")
        rate = program._call(stage.rate, conserved, name="spatial_physical_rate")
        residual = program.value("spatial_residual", conserved - previous - stage.tau * rate,
                                 at=iterate.point)
    finally:
        program._recording.pop()
    if not any(node.op == "diffusive_rhs" for node in sub):
        raise SolveRequestError("unsupported_lowering", "stage rate has no selected physical diffusion")
    allowed = {"state", "diffusive_rhs", "local_transform", "linear_combine", "source"}
    if any(node.op not in allowed for node in sub):
        raise SolveRequestError("unsupported_residual_operation", "residual has an unprepared operation")
    region = program._region_for_block(sub)
    require_region(program, residual, region, "spatial residual", vtype="state")
    mapping = {"rate": handle_data(stage.rate),
               "accumulation": None if stage.accumulation is None else handle_data(stage.accumulation),
               "previous_representation": "conserved_U_n=Q(q_n)"}
    token = program._new(
        "state", "solve_spatial_nonlinear", (previous,) if seed is None else (previous, seed),
        {"residual_block": sub, "residual_region": region, "residual": residual,
         "iterate": iterate, "newton_controls": prepared.controls.to_data(),
         "solver_identity": prepared.identity.token, "problem_kind": "spatial_accumulation",
         "finite_difference_step": scalar_data(step), "physical_mapping": mapping},
        name, previous.block, space=unknown.template.space, point=unknown.template.point)
    contract = _request_data(program, token, unknown, _json_ready(request.problem_metadata.to_data()))
    token = program._replace_value(token, attrs={**token.attrs, "solve_request": contract})
    outcome_name = name or token.name

    def project(outcome: Any) -> ResidualSolution:
        value = program._new(
            "state", "solve_outcome_component", (outcome,),
            {"index": 0, "unknown_identity": unknown.identity,
             "problem_identity": token.attrs["solve_request"]["problem_identity"]}, outcome_name + "_value",
            previous.block, space=token.space, point=token.point)
        return ResidualSolution((value,), (unknown.name,), (unknown,),
                                token.attrs["solve_request"]["problem_identity"])

    return SolveOutcome(program, token, project, outcome_name)
