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
    assert "stage_spatial_hierarchy_previous" in source
    assert source.count("ctx.reconcile_spatial_hierarchy_previous(") == 1
    assert "frozen_previous" in source
    assert "current_conserved" in source
    assert "attach_diffusive_flux_basis" not in source
    assert "uses_prepared_krylov_fallback" not in source
    assert "stage_accepted_exchanges" in source
    assert "residual_face_weight = -(" in source
    assert "PreparedSpatialResidual<pops::kNativeDimension>" not in source
    if nonlinear:
        assert "transform_state_resource_" in source
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


@pytest.mark.parametrize("kind", ["constant", "variable", "diagonal", "nonlinear_accumulation"])
def test_public_refined_spatial_profile_resolves_its_exact_coordinate_maps(kind):
    import pops
    from tests.python.integration.runtime.test_amr_implicit_diffusion import build

    case, layout = build(16, kind=kind)
    plan = pops.resolve(pops.validate(case), layout=layout)
    assert any(node.op == "solve_spatial_nonlinear" for node in plan.time._values)


@pytest.mark.parametrize("kind", ["constant", "variable", "diagonal", "nonlinear_accumulation"])
def test_explicit_imex_predictor_is_reconciled_before_one_implicit_solve(kind):
    import pops
    from tests.python.integration.runtime.test_amr_implicit_diffusion import build

    case, layout = build(16, kind=kind, imex=True)
    plan = pops.resolve(pops.validate(case), layout=layout)
    source = emit_cpp_program(
        plan.time, model=lower_and_validate(plan.blocks[0].model)[0], target="amr_system"
    )
    assert source.count("ctx.reconcile_spatial_hierarchy_previous(") == 1
    assert source.index("ctx.reconcile_spatial_hierarchy_previous(") < source.index(
        "ctx.solve_spatial_hierarchy("
    )
    assert "attach_diffusive_flux_basis" not in source
    assert "stage_spatial_hierarchy_previous" in source


@pytest.mark.parametrize("imex", [False, True])
def test_numerical_rejection_profile_preserves_iteration_limit_action(imex):
    import pops
    from pops.time import RejectAttempt
    from tests.python.integration.runtime.test_amr_implicit_diffusion import build

    case, layout = build(
        32,
        kind="nonlinear_accumulation",
        imex=imex,
        newton_iterations=1,
        failure_action=RejectAttempt(statuses=("iteration_limit",)),
    )
    plan = pops.resolve(pops.validate(case), layout=layout)
    source = emit_cpp_program(
        plan.time, model=lower_and_validate(plan.blocks[0].model)[0], target="amr_system"
    )
    assert ".max_iterations = 1," in source
    assert "pops::SolveStatus::kIterationLimit" in source
    assert "pops::SolveAction::kRejectAttempt" in source
    assert source.index("reconcile_spatial_hierarchy_previous") < source.index(
        "solve_spatial_hierarchy"
    )


@pytest.mark.parametrize("imex", [False, True])
def test_nonlinear_coordinate_maps_preserve_bootstrap_and_refresh_phase_contract(imex):
    import pops
    from tests.python.integration.runtime.test_amr_implicit_diffusion import build

    case, layout = build(16, kind="nonlinear_accumulation", imex=imex)
    plan = pops.resolve(pops.validate(case), layout=layout)
    source = emit_cpp_program(
        plan.time, model=lower_and_validate(plan.blocks[0].model)[0], target="amr_system"
    )
    install = source.split('extern "C" void pops_install_program_amr', 1)[1]
    # Bootstrap refreshes the installed level closures as the hierarchy grows.
    # Its phase predicate must recognize the very same Q maps as resolution.
    assert "_refresh_level_programs" in install
    assert "ctx_owner, _refresh_level_programs" in install
    assert "_require_local_transform_level_contract" not in install
    assert "refusing pre-reflux execution" not in install
    assert "temperature_to_energy" in source
    assert source.count("ctx.solve_spatial_hierarchy(") == 1
    assert "ctx.advance_synchronized_hierarchy" in install


@pytest.mark.parametrize("imex", [False, True])
@pytest.mark.parametrize("kind", ["constant", "nonlinear_accumulation"])
def test_spatial_scratch_producers_retain_exact_block_level_and_node_ownership(kind, imex):
    import pops
    from tests.python.integration.runtime.test_amr_implicit_diffusion import build

    case, layout = build(16, kind=kind, imex=imex)
    plan = pops.resolve(pops.validate(case), layout=layout)
    token = next(node for node in plan.time._values if node.op == "solve_spatial_nonlinear")
    source = emit_cpp_program(
        plan.time, model=lower_and_validate(plan.blocks[0].model)[0], target="amr_system"
    )
    assert "spatial_%d_trial = &ctx.scratch_state(%d,1,ctx.state(0));" % (token.id, token.id) in source
    assert "ctx.scratch_state(%d,0,ctx.state(0))" % token.id in source
    assert "ctx.scratch_state_like(" not in source
    transforms = [node for node in (*plan.time._values, *token.attrs["residual_block"])
                  if node.op == "local_transform"]
    if kind == "nonlinear_accumulation":
        assert transforms
    for node in transforms:
        assert "transform_state_resource_%d = &ctx.scratch_state(%d, 0, ctx.state(0));" % (
            node.id, node.id) in source
        assert "transform_status_resource_%d = &ctx.scalar_scratch(%d, 0, ctx.state(0), 1, 0);" % (
            node.id, node.id) in source
    install = source.split('extern "C" void pops_install_program_amr', 1)[1]
    resources = install.split("return _PopsAmrLevelProgram{", 1)[0]
    assert "_trial" not in resources
    assert "transform_state_resource_" not in resources
    assert "transform_status_resource_" not in resources
    assert "pops::PureFieldAlgebra::copy(*spatial_%d_trial,q);" % token.id in source
    assert "frozen_previous" in source
    assert source.count("ctx.solve_spatial_hierarchy(") == 1


@pytest.mark.parametrize("imex", [False, True])
@pytest.mark.parametrize("kind", ["constant", "nonlinear_accumulation"])
def test_hierarchy_solve_releases_level_envelopes_before_its_internal_traversals(kind, imex):
    import pops
    from tests.python.integration.runtime.test_amr_implicit_diffusion import build

    case, layout = build(16, kind=kind, imex=imex)
    plan = pops.resolve(pops.validate(case), layout=layout)
    source = emit_cpp_program(
        plan.time, model=lower_and_validate(plan.blocks[0].model)[0], target="amr_system"
    )
    driver = source.split("auto _advance_hierarchy =", 1)[1].split(
        "ctx.advance_synchronized_hierarchy", 1
    )[0]
    scopes = []
    phases = []
    for line in driver.splitlines():
        code = line.split("//", 1)[0]
        for phase in ("gather", "solve", "publish"):
            if ".%s(hierarchy_dt)" % phase in code:
                phases.append(phase)
                # Reconciliation and residual/JVP callbacks each select all levels.
                # Only gather/publication may retain an outer level checkout.
                assert any("with_program_attempt_level" in scope for scope in scopes) == (
                    phase != "solve"
                )
        for char in code:
            if char == "{":
                scopes.append(code)
            elif char == "}":
                scopes.pop()
    assert phases == ["gather", "solve", "publish"]
