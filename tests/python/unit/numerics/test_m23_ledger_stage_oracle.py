"""Exercise the native fixture's stage oracle without importing/building PoPS."""
import ast
from pathlib import Path
import re

import pytest


SOURCE = Path(__file__).resolve().parents[2] / "integration/runtime/test_coupled_gradient_exchange_runtime.py"
TREE = ast.parse(SOURCE.read_text())
FUNCTION = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == "_stage_states")
NAMESPACE = {"re": re}
exec(compile(ast.Module(body=[FUNCTION], type_ignores=[]), str(SOURCE), "exec"), NAMESPACE)
STATES = NAMESPACE["_stage_states"]


def context(stage):
    return f"pops.exchange.frame.v1/runtime/StagePoint(name='ssprk2_stage_{stage}', retained_coordinates)/evaluation:{stage+1}"


def test_stage_oracle_is_independent_of_record_insertion_order():
    initial, predictor = object(), object()
    result = STATES((context(1), context(0)), initial, predictor)
    assert result[context(0)] is initial
    assert result[context(1)] is predictor


@pytest.mark.parametrize("contexts", [
    (context(0), context(0) + "different_eval"),
    (context(0), context(2)),
    (context(0), "unqualified_or_unknown"),
    (context(0),),
])
def test_missing_duplicate_or_unknown_stage_is_refused(contexts):
    with pytest.raises(AssertionError):
        STATES(contexts, object(), object())
