"""The final resolved AMR Program emits only the authenticated AMR install entry."""

from __future__ import annotations

import pytest

import pops.lib.time as libtime
from pops.codegen.program_codegen import emit_cpp_program
from pops.linalg import LinearProblem
from pops.numerics.terms import Flux
from pops.solvers import GMRES
from pops.time import FailRun, FixedDt, Program
from tests.python.integration._final_field_program import (
    compiler_model,
    resolve_periodic_field_program,
    scalar_advection_field_model,
)


def _plan(*, target: str, name: str):
    model = scalar_advection_field_model(name + "-model")
    resolved = resolve_periodic_field_program(
        model,
        lambda state, rate, field: libtime.ForwardEuler(
            state,
            rate=rate,
            fields=field,
            solve_action=FailRun(),
        ),
        name=name,
        block_name="plasma",
        target=target,
        n=16,
    )
    return model, resolved


def test_resolved_amr_program_emits_only_the_amr_install_entry() -> None:
    amr_model, amr = _plan(target="amr_system", name="amr-install")
    amr_source = emit_cpp_program(
        amr.time,
        compiler_model(amr_model),
        target="amr_system",
        field_plans=amr.field_plans,
    )
    system_model, system = _plan(target="system", name="uniform-install")
    system_source = emit_cpp_program(
        system.time,
        compiler_model(system_model),
        target="system",
        field_plans=system.field_plans,
    )

    assert "pops_install_program" in amr_source
    assert (
        'extern "C" bool pops_install_program(\n'
        "    const pops::runtime::program::ProgramHostDescriptor* host,\n"
        "    pops::runtime::program::ProgramCandidateDescriptor* candidate,\n"
        "    pops::runtime::program::ProgramInstallDiagnostic* diagnostic) noexcept"
    ) in amr_source
    assert (
        'extern "C" void pops_install_program(pops::System<pops::kNativeDimension>* sys)'
        not in amr_source
    )
    amr_prepare = amr_source.split("bool program_candidate_prepare", 1)[1].split(
        'extern "C" bool pops_install_program', 1
    )[0]
    amr_inspect = amr_source.split('extern "C" bool pops_install_program', 1)[1]
    assert (
        "make_program_execution_provider<pops::kNativeDimension>(host->preparation)"
        in amr_prepare
    )
    assert "make_program_execution_provider" not in amr_inspect
    assert "ProgramExecutionServices& ctx" not in amr_source
    assert "_make_level_program" in amr_source
    assert "const auto topology = ctx.program_resource_topology();" in amr_source
    assert "ctx.for_each_program_resource_level(" in amr_source
    assert "ctx.set_level(" not in amr_source
    assert "_refresh_level_programs();" in amr_source
    assert "ctx.advance_hierarchy(dt, _advance_level)" in amr_source
    assert "state->hierarchy_resource_refresh = _refresh_level_programs;" in amr_source
    level_advance = amr_source.split("auto _advance_level", 1)[1].split("};", 1)[0]
    assert level_advance.index("_refresh_level_programs();") < level_advance.index(
        "_level_programs->at"
    ), "a transactional regrid must refresh resources before the first level-bundle access"
    assert (
        'extern "C" bool pops_install_program(\n'
        "    const pops::runtime::program::ProgramHostDescriptor* host,\n"
        "    pops::runtime::program::ProgramCandidateDescriptor* candidate,\n"
        "    pops::runtime::program::ProgramInstallDiagnostic* diagnostic) noexcept"
    ) in system_source
    assert 'extern "C" void pops_install_program(pops::System<pops::kNativeDimension>* sys)' not in system_source
    system_prepare = system_source.split("bool program_candidate_prepare", 1)[1].split(
        'extern "C" bool pops_install_program', 1
    )[0]
    system_inspect = system_source.split('extern "C" bool pops_install_program', 1)[1]
    assert (
        "make_program_execution_provider<pops::kNativeDimension>(host->preparation)"
        in system_prepare
    )
    assert "make_program_execution_provider" not in system_inspect
    assert "ctx.install(" not in system_source
    for token in (
        "struct ProgramCandidateState final",
        "std::function<void(double)> step;",
        "host->services.state_store",
        "valid_program_host_descriptor(*host)",
        "ProgramRuntimeKind::uniform",
        "write_program_install_diagnostic(",
        "catch (...)",
        "descriptor.artifact_identity =",
        "descriptor.abi_key =",
        "descriptor.route_manifest =",
        "descriptor.boundary_manifest =",
        "descriptor.persistent_resource_manifest =",
        "descriptor.checkpoint_identity =",
        "descriptor.dt_bound = nullptr;",
        "descriptor.step = &program_candidate_step;",
        "descriptor.destroy = &program_candidate_destroy;",
        "(void)state.release();",
    ):
        assert token in system_source

    for token in (
        "struct ProgramCandidateState final",
        "ProgramRuntimeKind::amr",
        "host->services.state_store",
        "program_candidate_hierarchy_refresh",
        "program_candidate_history_remap",
        "program_candidate_restart_preflight",
        "program_candidate_accepted_snapshot",
        "descriptor.hierarchy_refresh =",
        "descriptor.history_remap_accepted =",
        "descriptor.restart_regrid_preflight =",
        "descriptor.create_accepted_snapshot =",
        "descriptor.destroy = &program_candidate_destroy;",
    ):
        assert token in amr_source
    assert "ctx.install(" not in amr_source
    assert "pops_register_program_provider_routes" not in amr_source
    assert "descriptor.provider_routes = kProgramCandidateProviderRoutes" in amr_source


def test_unknown_program_target_is_rejected_before_emission() -> None:
    model, resolved = _plan(target="system", name="invalid-install-target")
    with pytest.raises(ValueError, match="target"):
        emit_cpp_program(
            resolved.time,
            compiler_model(model),
            target="bogus",
            field_plans=resolved.field_plans,
        )


def test_field_coupled_jacvec_is_materialized_inside_every_amr_level_bundle() -> None:
    """Link the public Program operation to the native L0/L1 numerical oracle.

    ``test_amr_named_field.FieldCoupledRhsJacvecMatchesCenteredDifferenceOnEveryLevel`` proves the
    exact runtime algebra numerically.  This source witness proves that an ordinary resolved public
    Program installs that same field-qualified operation in the per-level AMR bundle rather than in
    a coarse-only side route.
    """

    def factory(state, rate, field):
        del rate
        program = Program("amr-field-coupled-jacvec")
        temporal = program.state(state)
        iterate = program.value("iterate", temporal.n, at=temporal.n.point)
        fields = field(iterate, name="iterate-fields").consume(action=FailRun())
        r0 = program.rhs(
            name="frozen-rhs",
            state=iterate,
            fields=fields,
            terms=(Flux(),),
        )
        operator = program.matrix_free_operator(
            "field-coupled-jacobian", domain="state", range_="state", ncomp=1
        )

        def apply(builder, out, direction):
            return builder.rhs_jacvec(
                out,
                direction,
                iterate=iterate,
                r0=r0,
                c_dt=builder.dt,
                eps=1.0e-7,
                flux=True,
                sources=[],
                field_coupled=True,
            )

        operator = program.set_apply(operator, apply)
        program.solve(
            LinearProblem(operator, temporal.n, at=iterate.point, nullspace=None),
            solver=GMRES(max_iter=4, restart=2, rel_tol=1.0e-8),
            name="correction",
        ).consume(action=FailRun())
        program.commit(
            temporal.next,
            program.value(
                "next",
                temporal.n + program.dt * r0,
                at=temporal.next.point,
            ),
        )
        program.step_strategy(FixedDt(0.01))
        return program

    model = scalar_advection_field_model("amr-field-coupled-jacvec-model")
    resolved = resolve_periodic_field_program(
        model,
        factory,
        name="amr-field-coupled-jacvec",
        block_name="plasma",
        target="amr_system",
        n=16,
    )
    source = emit_cpp_program(
        resolved.time,
        compiler_model(model),
        target="amr_system",
        field_plans=resolved.field_plans,
    )

    materialization = source.split("auto _make_level_program", 1)[1]
    factory_source, refresh_source = materialization.split("auto _refresh_level_programs", 1)
    assert "ctx.evaluate_with_field_state_at(" in factory_source
    assert "ctx.boundary_evaluation_point(" in factory_source
    assert "evaluate_with_field_state_at(*" in factory_source
    level_iteration = refresh_source.index("ctx.for_each_program_resource_level([&](int) {")
    bundle_insert = refresh_source.index(
        "_level_programs->emplace_back(_make_level_program());"
    )
    assert level_iteration < bundle_insert
    assert "ctx.set_level(" not in refresh_source
    assert "ctx.solve_default_field_on_coarse_level(" not in materialization
    assert (
        "ctx.solve_fields_from_state_at("
        not in materialization.split("ctx.evaluate_with_field_state_at(", 1)[1].split("});", 1)[0]
    )
    assert materialization.count("ctx.evaluate_with_field_state_at(") == 1
