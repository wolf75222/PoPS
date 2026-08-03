from __future__ import annotations

import pytest

import pops
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
        tick_denominator, rung, message) -> None:
    program, _ = _transport_program()
    with pytest.raises(ValueError, match=message):
        program.cell_local_time(tick_denominator=tick_denominator, rung=rung)


def test_amr_codegen_selects_only_the_prepared_cell_local_driver() -> None:
    program, model = _transport_program()
    program.cell_local_time(tick_denominator=100, rung=0)

    source = emit_cpp_program(program, model=model, target="amr_system")

    assert "ctx.prepare_same_level_cell_temporal_execution(" in source
    assert program.clock.qualified_id in source
    assert "ctx_owner->advance_same_level_cell_temporal(dt);" in source
    assert "ctx.advance_hierarchy(dt" not in source
    assert "ctx.advance_synchronized_hierarchy(dt" not in source


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


def test_cell_local_codegen_refuses_uniform_target() -> None:
    program, model = _transport_program()
    program.cell_local_time(tick_denominator=100)
    with pytest.raises(ValueError, match="target='amr_system'"):
        emit_cpp_program(program, model=model, target="system")
