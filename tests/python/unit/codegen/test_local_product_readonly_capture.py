"""A captured State is a bind input, without becoming an algebraic unknown."""
from types import SimpleNamespace

import pops
import pytest

from pops.codegen._compiled_artifact import CompiledPlanRecord
from pops.codegen.inspect_compiled import _build_arguments
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from tests.python.support.local_product_readonly_case import make_case


def readonly_capture_case(*, reverse=False):
    case, layout = make_case(reverse=reverse)
    return pops.resolve(pops.validate(case), layout=layout)


def _inspection_fixture(resolved):
    graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    rows = []
    for block in resolved.blocks:
        adapter = graph.model_for_block(block.name)
        rows.append(SimpleNamespace(block_name=block.name,
            state_space=block.state_spaces[0], n_vars=adapter._m.n_vars,
            cons_names=tuple(adapter._m.cons_names), params={},
            provider_components=tuple(adapter._m._provider_components), model=adapter))
    compiled = SimpleNamespace(plan=CompiledPlanRecord.from_resolved(resolved),
                               bind_schema=resolved.bind_schema, target=resolved.target)
    return graph, compiled, tuple(rows)


def test_one_unknown_product_already_accepts_frozen_foreign_block_capture():
    resolved = readonly_capture_case()
    graph, _, _ = _inspection_fixture(resolved)
    source = emit_cpp_program(resolved.time, model_graph=graph)
    assert "prepare_local_nonlinear_problem<3>" in source
    assert "prepare_local_nonlinear_problem<6>" not in source
    assert len(resolved.time.commits()) == 1
    token, = [value for value in resolved.time._values
              if value.attrs.get("problem_kind") == "local_residual_product"]
    assert token.attrs["output_count"] == 1
    assert token.attrs["product_widths"] == (3, 3)
    assert token.inputs[0].block != token.inputs[1].block
    assert "local product requires co-located" in source


@pytest.mark.parametrize("reverse", (False, True))
def test_readonly_captured_state_remains_an_exact_required_bind_input(reverse):
    resolved = readonly_capture_case(reverse=reverse)
    _, compiled, rows = _inspection_fixture(resolved)
    arguments = _build_arguments(compiled, resolved.time, rows)
    assert set(arguments.instances) == {"dual", "target"}
    assert arguments.instances["target"]["required"] is True
    assert arguments.instances["target"]["components"] == 3
    assert arguments.instances["target"]["block_identity"] != arguments.instances["dual"]["block_identity"]
    assert arguments.instances["target"]["state_identity"] != arguments.instances["dual"]["state_identity"]
    assert set(arguments.layout_runtime["ghost_depth_by_block"]) == {"dual", "target"}
