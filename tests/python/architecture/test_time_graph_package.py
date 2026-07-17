"""The temporal graph implementation is private, acyclic, and facade-owned."""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TIME = ROOT / "python" / "pops" / "time"
GRAPH = TIME / "_graph"


def _internal_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), str(path))
    result = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module \
                and node.module.startswith("pops.time._graph."):
            result.add(node.module.rsplit(".", 1)[-1])
    return result


def test_temporal_graph_has_one_private_unidirectional_implementation():
    assert not (TIME / "graph.py").exists()
    assert not (TIME / "graph_control.py").exists()
    assert not (TIME / "graph_validation.py").exists()

    allowed = {
        "base": set(),
        "nodes": {"base"},
        "validation": {"base"},
        "control": {"base", "nodes", "validation"},
        "program": {"base", "control", "nodes", "validation"},
    }
    actual = {
        path.stem: _internal_imports(path)
        for path in GRAPH.glob("*.py")
        if path.name != "__init__.py"
    }
    assert actual == allowed


def test_public_graph_identity_is_owned_only_by_pops_time_facade():
    from pops.time import GraphProgramValue, ProgramGraph, ValueRef
    from pops.time._graph.base import ValueRef as InternalValueRef
    from pops.time._graph.nodes import ProgramValue as InternalProgramValue
    from pops.time._graph.program import ProgramGraph as InternalProgramGraph

    assert ProgramGraph is InternalProgramGraph
    assert GraphProgramValue is InternalProgramValue
    assert ValueRef is InternalValueRef
