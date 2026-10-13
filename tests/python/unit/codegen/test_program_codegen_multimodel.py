"""Whole-Program lowering dispatches every node through its own model owner."""
from __future__ import annotations
import re

from pops.codegen.program_codegen import emit_cpp_program

from pops.codegen._plans import ResolvedBlock
from pops.codegen.program_models import ProgramModelGraph
from pops.physics._facade import Model
from pops.problem import Case
from pops.time import Program


def _model(name, coefficient):
    model = Model(name)
    (u,) = model.conservative_vars("u")
    model.flux(x=[0 * u], y=[0 * u])
    source = model.source_term("decay", [coefficient * u])
    return model, source


def test_two_model_program_emits_each_models_own_source_kernel():
    first, first_source = _model("first_physics", -2)
    second, second_source = _model("second_physics", -7)
    problem = Case(name="coupled")
    first_block = problem.block("first", first)
    second_block = problem.block("second", second)

    program = Program("two-model")._bind_operators(first.module)._bind_operators(second.module)
    first_state = program.state(
        first_block[first.module.state_handle(first.module.state_spaces()["U"])])
    second_state = program.state(
        second_block[second.module.state_handle(second.module.state_spaces()["U"])])
    first_rate = program.source(first_source, state=first_state.n)
    second_rate = program.source(second_source, state=second_state.n)
    program.commit(
        first_state.next,
        program.value(
            "first_next", first_state.n + program.dt * first_rate,
            at=first_state.next.point),
    )
    program.commit(
        second_state.next,
        program.value(
            "second_next", second_state.n + program.dt * second_rate,
            at=second_state.next.point),
    )

    graph = ProgramModelGraph.from_resolved_blocks((
        ResolvedBlock(
            "first", first, None, "production", ("U",), ("test::first::state::U",)),
        ResolvedBlock(
            "second", second, None, "production", ("U",), ("test::second::state::U",)),
    ))
    source = emit_cpp_program(program, model_graph=graph)

    assert '"named_source"' not in source
    assert source.count("ctx.pointwise_active_mask(block_index, kernel_input);") == 2
    for block_index, state, rate in ((0, first_state, first_rate), (1, second_state, second_rate)):
        assert re.search(
            r'pops_shared_source_[0-9a-f]{64}\(ctx, "[^"]+", %d, u%d, r%d\);' %
            (block_index, state.n.id, rate.id), source)
    assert source.count("if (source_has_active_mask_ &&") == 2
    assert source.count("outA(index, 0) = pops::Real(0);") == 2
    assert "first_physics" in source and "second_physics" in source
    assert "pops::Real(-2) * u" in source
    assert "pops::Real(-7) * u" in source
    assert "pops_module_operator_owner" in source
    assert not hasattr(graph, "first_model")
