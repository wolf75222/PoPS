"""Public local products leave sparse SSA ids; compile detachment must be lossless."""
import pytest

import pops
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.time._program.detach import detach_compiled_program
from tests.python.support.local_residual_product_case import make_case


@pytest.mark.parametrize("reverse", [False, True])
def test_public_product_detaches_with_sparse_ids_and_exact_generated_program(reverse):
    case, layout, _ = make_case(reverse=reverse)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    source = resolved.time
    original = source._serialize(include_provenance=False)
    ids = tuple(value.id for value in source._values)
    assert max(ids) + 1 > len(ids), "the witness must retain a real authoring-id gap"
    graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    expected_cpp = emit_cpp_program(source, model=graph)

    detached = detach_compiled_program(source)
    assert detached._serialize(include_provenance=False) == original
    assert detached._ir_hash() == source._ir_hash()
    assert emit_cpp_program(detached, model=graph) == expected_cpp
    assert tuple(value.id for value in detached._values) == ids
    assert detached._next_id == source._next_id
    assert not detached._operator_registries
    assert all(value.prog is detached for value in detached._values)
    assert source._serialize(include_provenance=False) == original


def test_noop_rebuild_retains_sparse_ids_but_actual_drop_still_compacts():
    case, layout, _ = make_case()
    source = pops.resolve(pops.validate(case), layout=layout).time
    rebuilt = source._rebuild(lambda _value: True)
    assert rebuilt._serialize(include_provenance=False) == source._serialize(include_provenance=False)
    assert rebuilt._ir_hash() == source._ir_hash()

    # Existing optimizer contract: genuine dead-code removal still compacts ids.
    unused = rebuilt.value("unused", 2 * rebuilt._values[0])
    compacted = rebuilt.eliminate_dead_nodes()
    assert all(value.name != unused.name for value in compacted._values)
    assert tuple(value.id for value in compacted._values) == tuple(range(len(compacted._values)))
    assert len(compacted._commits) == len(source._commits)
