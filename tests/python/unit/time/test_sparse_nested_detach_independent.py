"""A sparse flat Program with nested regions must clone losslessly.

The synthetic id reservation models transient authoring placeholders. The
public LocalResidual product test covers the actual producer of such gaps.
"""
from pops.time import Program
from pops.time._program.detach import detach_compiled_program
from typed_program_support import typed_state


def test_sparse_ids_and_nested_branch_keep_exact_ir_and_next_id():
    program = Program("sparse_nested_review")
    state = typed_state(program, "fluid")
    # Reserve ids as a residual authoring region does before encoding its
    # placeholders as argument roles. No executable node owns these ids.
    program._next_id += 4
    outer = program.norm2(state) > 0
    inner = program.norm2(state) > 1
    selected = program.branch(
        outer,
        lambda builder: builder.branch(
            inner,
            lambda child: child.value("nested_true", 1 * state, at=state.point),
            lambda _child: state,
            name="nested",
        ),
        lambda _builder: state,
        name="outer",
    )
    endpoint = typed_state(program, "fluid", state_name="U").next
    program.commit(endpoint, program.value("final", selected, at=endpoint.point))
    original = program._serialize(include_provenance=False)
    original_ids = tuple(value.id for value in program._values)
    assert max(original_ids) + 1 > len(original_ids)
    assert any(value.op == "branch" for value in program._values)

    for rebuilt in (program._rebuild(lambda _value: True),
                    detach_compiled_program(program)):
        assert rebuilt._serialize(include_provenance=False) == original
        assert rebuilt._ir_hash() == program._ir_hash()
        assert tuple(value.id for value in rebuilt._values) == original_ids
        assert rebuilt._next_id == program._next_id
        assert all(value.prog is rebuilt for value in rebuilt._values)
