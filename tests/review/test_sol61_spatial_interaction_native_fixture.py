"""Real public source admission; these checks execute no Native module or JIT."""
import ast
from pathlib import Path

import numpy as np
import pops
import pytest

from tests.python.support.spatial_interaction_receipts import build, initial_values, DT, GAMMA
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph


@pytest.mark.parametrize("width,adaptive,failure", ((1, False, None), (3, False, None),
    (3, True, None), (1, False, "budget"), (1, False, "pole"), (1, False, "nonfinite")))
def test_actual_case_resolve_and_emission(width, adaptive, failure):
    case, layout, selected, _ = build(width, adaptive=adaptive, failure=failure)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    program = resolved.time
    ir = program._serialize()
    assert ir["version"] == 18
    interactions = [value for value in program._values if value.op == "spatial_interaction"]
    assert len(interactions) == 2
    assert interactions[0].attrs["contract"] == "pops.spatial-interaction@1"
    assert interactions[1].attrs["contract"] == "pops.spatial-interaction@2"
    assert tuple(interactions[0].attrs["components"]) == selected
    cpp = emit_cpp_program(program, model=ProgramModelGraph.from_resolved_blocks(resolved.blocks),
                           target="amr_system" if adaptive else "system")
    assert "ctx.spatial_interaction_history" in cpp and "ctx.store_history(" in cpp
    assert "ctx.commit_many(" in cpp
    assert any(value.op == "rhs" and tuple(value.attrs["sources"]) == ("default",) for value in program._values)
    assert "1ULL" in cpp if failure == "budget" else "8388608ULL" in cpp
    values = initial_values(width)
    assert values.shape == (width, 6, 8) and np.isfinite(values).all()
    for component in values:
        assert np.ptp(component, axis=1).max() > 0
        assert np.ptp(component, axis=0).max() > 0
    assert DT == .01 and GAMMA == .2


def test_six_native_cases_collect_without_executing_them():
    path = Path(__file__).parents[1]/"python/integration/runtime/test_public_spatial_interaction.py"
    module = ast.parse(path.read_text())
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")]
    assert len(functions) == 2
    assert all(any(isinstance(node, ast.Attribute) and node.attr == "native_loader"
                   for decorator in function.decorator_list for node in ast.walk(decorator)) for function in functions)
    assert "aggregate_cryptographic_binding" in path.read_text()
    assert "SOURCE_SNAPSHOT_ONLY" in path.read_text()
