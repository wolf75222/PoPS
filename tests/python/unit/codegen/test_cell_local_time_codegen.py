from __future__ import annotations

import pytest

import pops
import pops.codegen.program_codegen as program_codegen
from pops.codegen.program_codegen import emit_cpp_program
from pops.lib import time as libtime
from pops.physics._facade import Model


def _transport_program(factory=libtime.ForwardEuler):
    model = Model("cell_local_transport")
    model.conservative_vars("u")
    rate = model.rate("transport", flux=True, sources=())
    state = next(
        declaration
        for declaration in model.declaration_index().records()
        if declaration.kind == "state"
    )
    block = pops.Case("cell_local_case").block("tracer", model)
    return factory(block[state], rate=rate), model


def test_cell_local_time_contract_is_frozen_rebuilt_and_hashed() -> None:
    global_program, _ = _transport_program()
    local_program, _ = _transport_program()
    local_program.cell_local_time(tick_denominator=100, rung=1)

    assert local_program.cell_local_time_contract().to_data() == {
        "schema_version": 1,
        "tick_denominator": 100,
        "rung": 1,
    }
    assert "cell_local_time" in local_program._serialize(include_provenance=False)
    assert local_program._ir_hash() != global_program._ir_hash()
    rebuilt = local_program._rebuild(lambda _value: True)
    assert rebuilt.cell_local_time_contract() == local_program.cell_local_time_contract()
    local_program.freeze()
    with pytest.raises(RuntimeError, match="frozen"):
        local_program.cell_local_time(tick_denominator=100)


@pytest.mark.parametrize(
    ("tick_denominator", "rung", "message"),
    [
        (0, 0, "positive int"),
        (1.0, 0, "positive int"),
        (10, -1, "in \\[0, 30\\]"),
        (10, 31, "in \\[0, 30\\]"),
    ],
)
def test_cell_local_time_contract_refuses_invalid_integer_clocks(
    tick_denominator, rung, message
) -> None:
    program, _ = _transport_program()
    with pytest.raises(ValueError, match=message):
        program.cell_local_time(tick_denominator=tick_denominator, rung=rung)


def test_amr_codegen_selects_only_the_prepared_cell_local_driver() -> None:
    program, model = _transport_program()
    program.cell_local_time(tick_denominator=100, rung=0)

    source = emit_cpp_program(program, model=model, target="amr_system")

    assert "ctx.configure_primary_clock(" in source
    assert "ctx.prepare_same_level_cell_temporal_execution(" in source
    assert "SameLevelCellTemporalForwardEulerRoute, 1>" in source
    assert "{0, -1," in source
    assert "kProgramCandidateCheckpointShape" in source
    assert "kProgramCandidateResourcePlan" in source
    assert program.clock.qualified_id in source
    assert "ctx_owner->advance_same_level_cell_temporal(dt);" in source
    assert "ctx.advance_hierarchy(dt" not in source
    assert "ctx.advance_synchronized_hierarchy(dt" not in source
    assert "ctx.install(" not in source
    assert "pops_register_program_provider_routes" not in source
    assert "amr_program_context.hpp" not in source
    assert "descriptor.provider_routes = kProgramCandidateProviderRoutes" in source
    for token in (
        'extern "C" bool pops_install_program(',
        "ProgramRuntimeKind::amr",
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
        assert token in source


def test_amr_codegen_emits_one_typed_cell_local_route_per_block() -> None:
    model = Model("cell_local_multiroute")
    model.conservative_vars("u")
    rate = model.rate("transport", flux=True, sources=())
    state = next(
        declaration
        for declaration in model.declaration_index().records()
        if declaration.kind == "state"
    )
    case = pops.Case("cell_local_multiroute_case")
    first = case.block("first", model)
    second = case.block("second", model)
    routes = (
        libtime.RungeKuttaRoute(first[state], rate),
        libtime.RungeKuttaRoute(second[state], rate),
    )
    program = libtime.RungeKutta(
        routes=routes,
        tableau=libtime.FORWARD_EULER_TABLEAU,
    )
    program.cell_local_time(tick_denominator=64, rung=1)

    source = emit_cpp_program(program, model=model, target="amr_system")

    assert "SameLevelCellTemporalForwardEulerRoute, 2>" in source
    assert "{0, -1," in source and "{1, -1," in source
    assert source.count('extern "C" bool pops_install_program(') == 1
    assert "pops_install_program_amr" not in source
    assert "ctx.install(" not in source
    assert "pops_register_program_provider_routes" not in source
    assert "amr_program_context.hpp" not in source


def test_amr_codegen_types_exact_flux_metadata_for_public_facade() -> None:
    program, model = _transport_program()

    source = emit_cpp_program(program, model=model, target="amr_system")

    exact_type = "std::initializer_list<pops::runtime::program::ExactCoefficientTerm>"
    assert source.count('extern "C" bool pops_install_program(') == 1
    assert source.count("ctx.axpy(") == 2
    assert source.count(exact_type + "{{") == 2
    assert ", dt, {{" not in source
    assert "pops_install_program_amr" not in source
    assert "ctx.install(" not in source
    assert "pops_register_program_provider_routes" not in source
    assert "amr_program_context.hpp" not in source


def test_cell_local_codegen_refuses_non_euler_and_nondefault_cadence() -> None:
    multistage, model = _transport_program(libtime.SSPRK2)
    multistage.cell_local_time(tick_denominator=100)
    with pytest.raises(ValueError, match="ForwardEuler"):
        emit_cpp_program(multistage, model=model, target="amr_system")

    strided, model = _transport_program()
    strided.cadence(stride=2)
    strided.cell_local_time(tick_denominator=100)
    with pytest.raises(ValueError, match="default Program cadence"):
        emit_cpp_program(strided, model=model, target="amr_system")


def test_cell_local_codegen_refuses_unqualified_shape_before_body_lowering(monkeypatch) -> None:
    program, model = _transport_program(libtime.SSPRK2)
    program.cell_local_time(tick_denominator=100)

    def body_must_not_run(*_args, **_kwargs):
        pytest.fail("an unqualified cell-local integrator must be refused before body lowering")

    monkeypatch.setattr(program_codegen, "_emit_body", body_must_not_run)
    with pytest.raises(ValueError, match="ForwardEuler"):
        emit_cpp_program(program, model=model, target="amr_system")


def test_cell_local_codegen_refuses_uniform_target() -> None:
    program, model = _transport_program()
    program.cell_local_time(tick_denominator=100)
    with pytest.raises(ValueError, match="target='amr_system'"):
        emit_cpp_program(program, model=model, target="system")
