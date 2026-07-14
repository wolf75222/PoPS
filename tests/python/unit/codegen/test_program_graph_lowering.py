"""The immutable ProgramGraph is the exact native-lowering boundary."""
from __future__ import annotations

import pytest

from pops.codegen.program_graph_lowering import emit_program_graph
from pops.time import Program
from pops.time._program.detach import detach_compiled_program


def test_lowering_accepts_only_the_graph_equivalent_detached_adapter(monkeypatch):
    detached = detach_compiled_program(Program("exact"))
    graph = detached.to_graph()
    monkeypatch.setattr(
        "pops.codegen.program_codegen.emit_cpp_program",
        lambda program, **kwargs: "source:%s" % program.name,
    )

    assert emit_program_graph(graph, lowering_program=detached) == "source:exact"

    other = detach_compiled_program(Program("other"))
    with pytest.raises(ValueError, match="does not match"):
        emit_program_graph(other.to_graph(), lowering_program=detached)


def test_lowering_rejects_a_live_authoring_program():
    program = Program("mutable")
    with pytest.raises(TypeError, match="detached compiled Program"):
        emit_program_graph(program.to_graph(), lowering_program=program)
