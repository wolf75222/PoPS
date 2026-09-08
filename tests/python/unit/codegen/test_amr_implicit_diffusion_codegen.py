"""The public spatial request has one hierarchy barrier and no accepted reflux basis."""

import pytest
from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.program_codegen import emit_cpp_program
from tests.python.unit.time.test_implicit_diffusion_request import resolve_case


@pytest.mark.parametrize("nonlinear", [False, True])
def test_one_composite_temporal_solve_preserves_current_parent_and_consumed_publication(nonlinear):
    plan = resolve_case(nonlinear=nonlinear)
    source = emit_cpp_program(
        plan.time, model=lower_and_validate(plan.blocks[0].model)[0], target="amr_system"
    )
    assert source.count("ctx.solve_spatial_hierarchy(") == 1
    assert "ctx.advance_synchronized_hierarchy" in source
    assert "with_spatial_parent(0,parent" in source
    assert "require_spatial_previous_reconciled" in source
    assert "frozen_previous" in source
    assert "current_conserved" in source
    assert "attach_diffusive_flux_basis" not in source
    assert "uses_prepared_krylov_fallback" not in source
    assert "stage_accepted_exchanges" in source
    assert "residual_face_weight = -(" in source
    assert "PreparedSpatialResidual<pops::kNativeDimension>" not in source
    if nonlinear:
        assert "auto transform_state_resource_" in source
        assert "coordinate_to_energy" in source


def test_authored_newton_controls_and_derivative_are_retained():
    plan = resolve_case()
    source = emit_cpp_program(
        plan.time, model=lower_and_validate(plan.blocks[0].model)[0], target="amr_system"
    )
    for token in (
        ".max_iterations = 20",
        ".linear_max_iterations = 100",
        ".restart = 30",
        "finite_difference_jvps",
        "full_residual_evaluations",
    ):
        assert token in source
    assert "nullspace" not in source


@pytest.mark.parametrize("kind", ["constant", "variable", "nonlinear_accumulation"])
def test_public_refined_spatial_profile_resolves_its_exact_coordinate_maps(kind):
    import pops
    from tests.python.integration.runtime.test_amr_implicit_diffusion import build

    case, layout = build(16, kind=kind)
    plan = pops.resolve(pops.validate(case), layout=layout)
    assert any(node.op == "solve_spatial_nonlinear" for node in plan.time._values)
