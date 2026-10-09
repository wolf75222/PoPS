"""A path-only RuntimeParam must survive the resolved Program owner route."""

import pops
import pytest
from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.program_emit_params import (
    _qualified_param_identity, _runtime_param_refs_in, program_param_entries)
from pops.codegen.program_models import ProgramModelGraph
from tests.python.integration.runtime.test_symbolic_path_param_runtime import _case


def test_resolved_path_speed_parameter_has_one_exact_program_route():
    case, layout, _ = _case(8, 1.e-4)
    plan = pops.resolve(pops.validate(case), layout=layout)
    graph = ProgramModelGraph.from_resolved_blocks(plan.blocks)
    assert program_param_entries(plan.time, graph) == [(0, "path_speed_margin", 0, 1.)]


def test_path_speed_parameter_rejects_sibling_block_of_same_model():
    case, layout, _ = _case(8, 1.e-4)
    first = case.blocks()["transport"]
    model = case._blocks.get(first.local_id)["model"]
    sibling = case.block("transport_copy", model)
    case.numerics(next(iter(case._numerics.values())), block=sibling)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    resolved_first = resolved.blocks[0]
    emitter, _ = lower_and_validate(
        resolved_first.model, state_space=resolved_first.state_spaces[0],
        resolved_operations=resolved_first.resolved_operations,
        numerics=resolved_first.numerics)
    impl = emitter._m
    speed_exprs = impl._path_conservative["kernel"]["parameter_expressions"]
    (ref,) = _runtime_param_refs_in(speed_exprs)
    assert _qualified_param_identity(ref, first, graph_aware=True)
    with pytest.raises(ValueError, match="different block instance"):
        _qualified_param_identity(ref, sibling, graph_aware=True)
