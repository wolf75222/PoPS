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


def test_detached_graph_reuse_rechecks_identity_and_releases_dead_programs():
    import gc
    import weakref
    from pops.time._program.graph_conversion import _detached_graphs

    program = Program("cached")
    program.int64(3)
    detached = detach_compiled_program(program)
    first = detached.to_graph()
    assert detached.to_graph() is first
    # Even an out-of-contract edit through object.__setattr__ must not turn the
    # conversion cache into permission to bypass the lowering identity check.
    object.__setattr__(detached, "name", "altered")
    second = detached.to_graph()
    assert second.graph_hash != first.graph_hash
    with pytest.raises(ValueError, match="does not match"):
        emit_program_graph(first, lowering_program=detached)
    weak = weakref.ref(detached)
    del detached
    gc.collect()
    assert weak() is None
    assert all(item[1] is not second for item in _detached_graphs.values())


def test_authoring_graph_is_refreshed_after_new_equations():
    program = Program("mutable_equations")
    first = program.to_graph()
    program.int64(3)
    assert program.to_graph().graph_hash != first.graph_hash
