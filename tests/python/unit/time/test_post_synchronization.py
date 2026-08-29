"""Authoring and lowering of Program.after_synchronization."""

from __future__ import annotations

import pytest

from pops.codegen import program_codegen
from pops.codegen.module_lowering import lower_and_validate
from pops.frames import Cartesian2D
from pops.physics import Model
from pops.time import Program

from typed_program_support import state_refs


def _shift_program(*, relax_in_main: bool = False, with_rhs: bool = False):
    model = Model("post_sync_model", frame=Cartesian2D())
    state = model.state("U", components=("q",))
    transform = model.local_transform("shift", (state[0] + 1.0,), valid_if=state[0] > 0.0)
    program = Program("post_sync_program")
    block, _ = state_refs(program, "fluid", model=model, state=state)
    temporal = program.state(block[state])
    candidate = program.value("candidate", temporal.n, at=temporal.next.point)
    if relax_in_main:
        candidate = program.transform(candidate, transform=transform, name="main_relaxed")
    program.commit(temporal.next, candidate)

    def _relax(program_body):
        if with_rhs:
            program_body._new(
                "rhs", "rhs", (temporal.n,), {"flux": True}, None, temporal.n.block)
            return
        synced = program_body.value("synced", 1.0 * temporal.n, at=temporal.next.point)
        relaxed = program_body.transform(synced, transform=transform, name="relaxed")
        program_body.commit(temporal.next, relaxed)

    if not relax_in_main:
        program.after_synchronization(_relax)
    return model, program


def test_after_synchronization_has_a_closed_semantic_schema() -> None:
    from pops.identity.semantic import program_semantic_data

    model, program = _shift_program()
    payload = program_semantic_data(program)
    assert payload["post_synchronization_commits"]
    del model


def test_after_synchronization_is_top_level_and_unique() -> None:
    model, program = _shift_program()
    assert sum(value.op == "post_synchronization" for value in program._values) == 1
    assert program._post_sync_commits
    with pytest.raises(ValueError, match="at most once"):
        program.after_synchronization(lambda _program: None)
    del model


def test_after_synchronization_refuses_flux_ops() -> None:
    with pytest.raises(ValueError, match="forbids flux"):
        _shift_program(with_rhs=True)


def test_after_synchronization_emits_after_hierarchy_advance() -> None:
    model, program = _shift_program()
    emit_model, source_module = lower_and_validate(model, facade=model)
    assert source_module is model.module
    source = program_codegen.emit_cpp_program(program, model=emit_model, target="amr_system")
    assert "pops_register_program_provider_routes" not in source
    prepare = source.split("bool program_candidate_prepare", 1)[1].split(
        'extern "C" bool pops_install_program', 1
    )[0]
    assert "ctx.advance_hierarchy(dt, _advance_level);" in prepare
    assert ".post_synchronization(dt);" in prepare
    assert prepare.index("ctx.advance_hierarchy") < prepare.index(".post_synchronization(dt);")
    post_sync_lambda = prepare.split("[=](double dt) {", 2)[2]
    assert "ctx.begin_step(dt);" in post_sync_lambda
    assert "ctx.set_stage_time(1, 1);" in post_sync_lambda
    assert post_sync_lambda.index("ctx.begin_step(dt);") < post_sync_lambda.index("ctx.state(")
    assert "ctx.state(" in post_sync_lambda
    assert "refusing pre-reflux execution" not in source


def test_unqualified_transform_still_emits_the_pre_reflux_guard() -> None:
    model, program = _shift_program(relax_in_main=True)
    emit_model, _source_module = lower_and_validate(model, facade=model)
    source = program_codegen.emit_cpp_program(program, model=emit_model, target="amr_system")
    assert "refusing pre-reflux execution" in source


def test_after_synchronization_proves_refined_local_transform() -> None:
    from pops.codegen._resolution import _resolve_amr_program
    from pops.runtime.amr_program_support import AMRProgramSupportContext

    model, program = _shift_program()
    context = AMRProgramSupportContext(
        hierarchy_level_count=2,
        frozen_hierarchy=True,
        shared_block_interfaces=False,
        field_routes_validated=True,
    )
    assert _resolve_amr_program("amr", program, context=context)["status"] == "proven"
    del model


def test_after_synchronization_refuses_cell_local_time() -> None:
    model, program = _shift_program()
    with pytest.raises(ValueError, match="cell_local_time"):
        program.cell_local_time(tick_denominator=2)
    local = Program("cell_local_post_sync")
    local.cell_local_time(tick_denominator=2)
    with pytest.raises(ValueError, match="cell_local_time"):
        local.after_synchronization(lambda _program: None)
    del model
