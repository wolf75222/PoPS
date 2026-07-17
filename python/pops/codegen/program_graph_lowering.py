"""Strict immutable ProgramGraph entry point for native Program lowering."""
from __future__ import annotations

from typing import Any


def emit_program_graph(
    graph: Any, *, lowering_program: Any, model: Any = None,
    model_graph: Any = None, target: str = "system", field_plans: Any = None,
) -> str:
    """Lower exactly ``graph`` through its frozen, graph-equivalent Program adapter."""
    from pops.time import ProgramGraph

    if type(graph) is not ProgramGraph:
        raise TypeError("emit_program_graph requires an exact immutable ProgramGraph")
    if not getattr(lowering_program, "_compiled_detached", False):
        raise TypeError("ProgramGraph lowering adapter must be a detached compiled Program")
    if lowering_program.to_graph().graph_hash != graph.graph_hash:
        raise ValueError("ProgramGraph lowering adapter does not match the compiler input graph")
    from pops.codegen.program_codegen import emit_cpp_program

    source = emit_cpp_program(
        lowering_program, model=model, model_graph=model_graph, target=target,
        field_plans=field_plans,
    )
    if lowering_program.to_graph().graph_hash != graph.graph_hash:
        raise RuntimeError("ProgramGraph lowering mutated or diverged from its compiler input")
    return source


__all__ = ["emit_program_graph"]
