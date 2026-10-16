from __future__ import annotations

import pytest

import pops
from pops.codegen.program_codegen import emit_cpp_program
from pops.lib import time as libtime
from pops.physics._facade import Model


def _emit_program(program, model, *, target):
    from pops.codegen.module_lowering import lower_and_validate

    lowered, _ = lower_and_validate(model)
    return emit_cpp_program(program, model=lowered, target=target)


def _compact_euler(routes):
    """Author the exact bounded three-node-per-block local-time language."""
    program = pops.Program("bounded_cell_local_euler")
    states = [program.state(route.state) for route in routes]
    current = [state.n for state in states]
    rates = [route.rate(value) for route, value in zip(routes, current, strict=True)]
    for state, value, rate in zip(states, current, rates, strict=True):
        program.commit(state.next, program.value("accepted", value + program.dt*rate,
                                                at=state.next.point))
    return program


def _transport_program(factory=None):
    model = Model("cell_local_transport")
    (u,) = model.conservative_vars("u")
    model.flux(x=[u], y=[0 * u])
    model.eigenvalues(x=[1 + 0*u], y=[0*u])
    rate = model.rate("transport", flux=True, sources=())
    state = next(
        declaration
        for declaration in model.declaration_index().records()
        if declaration.kind == "state"
    )
    block = pops.Case("cell_local_case").block("tracer", model)
    program = (factory(block[state], rate=rate) if factory is not None else
               _compact_euler((libtime.RungeKuttaRoute(block[state], rate),)))
    return program, model


def _assert_cell_local_source_map(program, source):
    from pops.codegen.program_source_map import emission_scope, source_map

    evidence = source_map(source)
    assert evidence["complete"] and not evidence["unlocated"]
    expected = {value.id: value for value in program._values}
    rows = evidence["operations"]
    assert {row["ssa"] for row in rows} == set(expected)
    assert len(rows) == len(expected)
    assert {row["op"] for row in rows} == {"state", "rhs", "linear_combine"}
    assert len({row["shared_emission_group"] for row in rows}) == 1
    lines = source.splitlines()
    for row in rows:
        value = expected[row["ssa"]]
        assert row["inputs"] == [item.id for item in value.inputs]
        assert row["channel"] == "cell_local_provider_advance"
        assert set(row["shared_ssa"]) == set(expected)
        assert row["cpp_binding"] is None and row["identifier_binding"] is None
        assert row["location_basis"] == "direct_statements"
        assert row["effective_ranges"] == row["ranges"]
        assert [lines[line - 1].strip() for first, last in row["ranges"]
                for line in range(first, last + 1)] == [
                    "ctx_owner->advance_same_level_cell_temporal(dt);"]

    # Keep the provider call and its complete route table intact: without the
    # exact coalesced markers neither the parser nor the emission guard may
    # claim that those SSA operations are accounted for.
    stripped = "\n".join(line for line in lines if not line.lstrip().startswith((
        "// pops-source-map@1 group_begin ", "// pops-source-map@1 group_end ")))
    missing = source_map(stripped)
    assert not missing["complete"]
    assert {row["ssa"] for row in missing["unlocated"]} == set(expected)
    with pytest.raises(ValueError, match="Program source map has unaccounted SSA"):
        emission_scope(lambda _program: stripped)(program)


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

    source = _emit_program(program, model, target="amr_system")

    assert "ctx.configure_primary_clock(" in source
    assert "ctx.prepare_same_level_cell_temporal_execution(" in source
    assert "SameLevelCellTemporalForwardEulerRoute, 1>" in source
    assert "{0, -1," in source
    assert "pops.amr.same-level-transport-euler-stage-flux@2" in source
    assert "pops_program_checkpoint_temporal_cells_per_topology_cell" in source
    assert "return UINT64_C(1);" in source
    assert program.clock.qualified_id in source
    assert "ctx_owner->advance_same_level_cell_temporal(dt);" in source
    assert "ctx.advance_hierarchy(dt" not in source
    assert "ctx.advance_synchronized_hierarchy(dt" not in source
    _assert_cell_local_source_map(program, source)


def test_amr_codegen_emits_one_typed_cell_local_route_per_block() -> None:
    model = Model("cell_local_multiroute")
    (u,) = model.conservative_vars("u")
    model.flux(x=[u], y=[0 * u])
    model.eigenvalues(x=[1 + 0*u], y=[0*u])
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
    program = _compact_euler(routes)
    program.cell_local_time(tick_denominator=64, rung=1)

    source = _emit_program(program, model, target="amr_system")

    assert "SameLevelCellTemporalForwardEulerRoute, 2>" in source
    assert "{0, -1," in source and "{1, -1," in source
    assert "return UINT64_C(2);" in source
    _assert_cell_local_source_map(program, source)


def test_cell_local_codegen_refuses_non_euler_and_nondefault_cadence() -> None:
    multistage, model = _transport_program(libtime.SSPRK2)
    multistage.cell_local_time(tick_denominator=100)
    with pytest.raises(ValueError, match="ForwardEuler"):
        _emit_program(multistage, model, target="amr_system")

    strided, model = _transport_program()
    strided.cadence(stride=2)
    strided.cell_local_time(tick_denominator=100)
    with pytest.raises(ValueError, match="default Program cadence"):
        _emit_program(strided, model, target="amr_system")


def test_cell_local_bounded_provider_refuses_extra_rate_materialization() -> None:
    aliased, model = _transport_program(libtime.ForwardEuler)
    aliased.cell_local_time(tick_denominator=100)
    with pytest.raises(ValueError, match="one transport-only ForwardEuler route"):
        _emit_program(aliased, model, target="amr_system")

    # An unused operation is still outside the exact provider language; source
    # mapping must not silently label it as executed by the coalesced call.
    extra, model = _transport_program()
    extra.norm_inf(next(value for value in extra._values if value.op == "state"))
    extra.cell_local_time(tick_denominator=100)
    with pytest.raises(ValueError, match="one transport-only ForwardEuler route"):
        _emit_program(extra, model, target="amr_system")


def test_cell_local_codegen_refuses_uniform_target() -> None:
    program, model = _transport_program()
    program.cell_local_time(tick_denominator=100)
    with pytest.raises(ValueError, match="target='amr_system'"):
        _emit_program(program, model, target="system")
