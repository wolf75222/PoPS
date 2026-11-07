"""Bind inspection must distinguish the Newton candidate from a real State read."""
from __future__ import annotations

from types import SimpleNamespace

import pops
import pytest

from pops.codegen._compiled_artifact import CompiledPlanRecord
from pops.codegen.inspect_compiled import _build_arguments, _required_state_refs
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.time._program.detach import detach_compiled_program
from pops.time.references import handle_data
from tests.python.integration.runtime.test_tensor_transport_composition import composition_case


@pytest.mark.parametrize("implicit", (False, True), ids=("explicit", "implicit"))
def test_composed_tensor_program_emits_and_inspects_only_exact_bind_state(implicit):
    case, layout, _ = composition_case(implicit=implicit)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    rows = []
    for block in resolved.blocks:
        adapter = graph.model_for_block(block.name)
        rows.append(SimpleNamespace(
            block_name=block.name, state_space=block.state_spaces[0],
            n_vars=adapter._m.n_vars, cons_names=tuple(adapter._m.cons_names),
            params={}, provider_components=tuple(adapter._m._provider_components),
            model=adapter))
    compiled = SimpleNamespace(
        plan=CompiledPlanRecord.from_resolved(resolved),
        bind_schema=resolved.bind_schema, target=resolved.target)

    for program in (resolved.time, detach_compiled_program(resolved.time)):
        # The only initial input is the Case-qualified, committed inventory.
        assert _required_state_refs(program) == set(program.commits())
        assert len(_required_state_refs(program)) == 1
        arguments = _build_arguments(compiled, program, tuple(rows))
        assert set(arguments.instances) == {"mixture"}
        assert arguments.instances["mixture"]["required"] is True
        assert arguments.instances["mixture"]["state_identity"] == handle_data(
            next(iter(program.commits())))
        source = emit_cpp_program(program, model=graph)
        assert ("solve_spatial_nonlinear" in source) is implicit


def test_anonymous_state_cannot_imitate_the_owned_spatial_iterate():
    case, _, _ = composition_case(implicit=True)
    program = case._time
    solve = next(value for value in program._values if value.op == "solve_spatial_nonlinear")
    iterate = solve.attrs["iterate"]
    assert iterate.state_ref is None
    assert any(iterate is value for value in solve.attrs["residual_block"])
    assert _required_state_refs(program) == set(program.commits())

    # Same spelling, block, space and point cannot grant an unrelated State
    # read the candidate's private authority.
    program._new("state", "state", (), {}, "spatial_iterate", iterate.block,
                 space=iterate.space, point=iterate.point, inherit_state_ref=False)
    with pytest.raises(ValueError, match="State read has no exact state identity"):
        _required_state_refs(program)
