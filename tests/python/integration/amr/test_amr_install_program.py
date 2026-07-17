"""The final resolved AMR Program emits only the authenticated AMR install entry."""
from __future__ import annotations

import pytest

import pops.lib.time as libtime
from pops.codegen.program_codegen import emit_cpp_program
from pops.time import FailRun
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

    assert "pops_install_program_amr" in amr_source
    assert "make_shared<pops::runtime::program::AmrProgramContext>(sys)" in amr_source
    assert "_make_level_program" in amr_source
    assert "ctx.program_resource_topology_epoch()" in amr_source
    assert "ctx.program_resource_topology_generation()" in amr_source
    assert "_refresh_level_programs();" in amr_source
    assert "ctx.advance_hierarchy(dt, _advance_level)" in amr_source
    level_advance = amr_source.split("auto _advance_level", 1)[1].split("};", 1)[0]
    assert level_advance.index("_refresh_level_programs();") < level_advance.index(
        "_level_programs->at"
    ), "a transactional regrid must refresh resources before the first level-bundle access"
    assert "pops_install_program(" in system_source
    assert "pops_install_program_amr" not in system_source


def test_unknown_program_target_is_rejected_before_emission() -> None:
    model, resolved = _plan(target="system", name="invalid-install-target")
    with pytest.raises(ValueError, match="target"):
        emit_cpp_program(
            resolved.time,
            compiler_model(model),
            target="bogus",
            field_plans=resolved.field_plans,
        )
