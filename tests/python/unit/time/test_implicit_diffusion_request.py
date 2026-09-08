"""A temporal accumulation residual remains separate from its numerical seed."""
from __future__ import annotations

import pops
import pytest
from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.program_codegen import emit_cpp_program
from pops.time import SolveRequestError
from pops.time.value_metadata import CoeffPolynomial
from tests.python.integration.runtime.test_implicit_diffusion_lifecycle import make_case


def resolve_case(**kwargs):
    case, layout = make_case(16, **kwargs)
    plan = pops.resolve(pops.validate(case), layout=layout)
    return plan


def solve_node(plan):
    return next(node for node in plan.time._values if node.op == "solve_spatial_nonlinear")


def test_public_implicit_stage_resolves_exact_spatial_request():
    plan = resolve_case()
    node = solve_node(plan)
    request = node.attrs["solve_request"]
    assert request["lowering"] == {"disposition": "native", "adapter": "prepared_spatial_residual"}
    assert request["derivative"]["route"] == "finite_difference"
    assert request["residual_interpretation"] == "Q(q)-U_n-tau*R(Q(q))"
    assert "diffusive_rhs" in request["source_mapping"]["residual_operations"]
    assert "nullspace" not in node.attrs
    assert "gauge" not in node.attrs
    assert sum(value.op == "solve_outcome" for value in plan.time._values) == 1


def test_nonlinear_stage_maps_current_coordinate_to_conserved_state_before_rate():
    plan = resolve_case(nonlinear=True, dt=1.0)
    node = solve_node(plan)
    assert node.inputs[0] is not node.inputs[1]
    body = node.attrs["residual_block"]
    rate = next(value for value in body if value.op == "diffusive_rhs")
    accumulation = rate.inputs[0]
    assert accumulation.op == "local_transform"
    assert accumulation.attrs["transform"] == "coordinate_to_energy"
    assert accumulation.inputs[0] is node.attrs["iterate"]
    assert node.attrs["physical_mapping"]["previous_representation"] == "conserved_U_n=Q(q_n)"


def test_spatial_request_detects_changed_residual_coefficients():
    case, _ = make_case(16)
    program = case._time
    node = next(value for value in program._values if value.op == "solve_spatial_nonlinear")
    residual = node.attrs["residual"]
    coefficients = list(residual.attrs["coeffs"])
    coefficients[0] = CoeffPolynomial({0: 7})
    body = list(node.attrs["residual_block"])
    program._recording.append(body)
    try:
        changed = program._replace_value(residual, attrs={**residual.attrs, "coeffs": coefficients})
    finally:
        program._recording.pop()
    changed_token = program._replace_value(node, attrs={**node.attrs, "residual_block": body, "residual": changed})
    from pops.time._program.spatial_solve import validate_spatial_request

    with pytest.raises(SolveRequestError, match="equation_identity_drift"):
        validate_spatial_request(program, changed_token)


def test_spatial_request_emits_prepared_global_newton_and_consumed_copy():
    plan = resolve_case(nonlinear=True, dt=1.0)
    # The plan's model carrier is the authenticated single-block formula view.
    model = plan.blocks[0].model
    emitter, _ = lower_and_validate(model)
    code = emit_cpp_program(plan.time, model=emitter)
    assert "PreparedSpatialResidual<pops::kNativeDimension>" in code
    assert "PreparedDiffusion<pops::kNativeDimension>" in code
    assert ".finite_difference_jvps" in code
    assert "spatial_implicit" in code
    assert "local_nonlinear_solve_report" not in code
    assert "ctx.set_stage_time(1, 1);" in code
    assert code.index(".consume(pops::SolveConsumption::kAccept)") < code.index("stage_accepted_exchanges")


def test_accepted_flux_weight_is_the_actual_residual_coefficient():
    from fractions import Fraction
    from pops.time._program.spatial_solve import spatial_rate_weight

    plan = resolve_case(tau_scale=Fraction(1, 2))
    node = solve_node(plan)
    rate = next(value for value in node.attrs["residual_block"] if value.op == "diffusive_rhs")
    assert dict(spatial_rate_weight(node, rate)) == {1: Fraction(1, 2)}
    code = emit_cpp_program(plan.time, model=lower_and_validate(plan.blocks[0].model)[0])
    assert "implicit-stage:" in code
    assert "stage_accepted_exchanges" in code


def test_spatial_adapter_requires_explicit_supported_derivative():
    with pytest.raises(SolveRequestError, match="unsupported_derivative"):
        resolve_case(derivative_route="exact")


@pytest.mark.parametrize("replacement", ["unused", "history_only", "weighted", "raw_coordinate", "wrong_accumulation"])
def test_spatial_exchange_refuses_unproved_commit_endpoint(replacement):
    case, layout = make_case(16, nonlinear=replacement in ("raw_coordinate", "wrong_accumulation"),
                             commit_mode=replacement)
    with pytest.raises(SolveRequestError, match="unsupported_commit"):
        case._time.validate()
    plan = pops.resolve(pops.validate(case), layout=layout)
    with pytest.raises(SolveRequestError, match="unsupported_commit"):
        emit_cpp_program(plan.time, model=lower_and_validate(plan.blocks[0].model)[0])
