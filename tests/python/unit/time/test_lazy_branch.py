from __future__ import annotations

import pytest

from pops.time import Program, TimePoint
from pops.time.graph import Branch, Region
from pops.time.program_region_validation import validate_program_regions
from typed_program_support import typed_state


def _copy(program, value, name):
    return program.value(name, 1 * value, at=value.point)


def test_branch_captures_two_typed_lazy_regions_and_exact_result_signature():
    program = Program("lazy-branch")
    state = typed_state(program, "fluid")
    condition = program.norm2(state) > 0

    selected = program.branch(
        condition,
        lambda T: _copy(T, state, "true_value"),
        lambda T: _copy(T, state, "false_value"),
        name="selected",
    )

    assert selected.vtype == state.vtype
    assert selected.space == state.space
    assert selected.block == state.block
    assert selected.clock == condition.clock
    assert selected.point == condition.point
    assert program._values[-1] is selected
    assert [value.name for value in selected.attrs["true_block"]] == ["true_value"]
    assert [value.name for value in selected.attrs["false_block"]] == ["false_value"]

    branch = next(node for node in program.to_graph().nodes if type(node) is Branch)
    assert type(branch.when_true) is Region
    assert type(branch.when_false) is Region
    assert branch.when_true.nodes[0].name == "true_value"
    assert branch.when_false.nodes[0].name == "false_value"
    assert branch.when_true.result_signature == branch.result_signature
    assert branch.when_false.result_signature == branch.result_signature
    assert branch.result_signature.to_data()["value_type"] == "state"


def test_branch_accepts_identity_capture_and_nested_branch_regions():
    program = Program("nested-branch")
    state = typed_state(program, "fluid")
    outer_condition = program.norm2(state) > 0
    inner_condition = program.norm2(state) > 1

    selected = program.branch(
        outer_condition,
        lambda T: T.branch(
            inner_condition,
            lambda inner: _copy(inner, state, "nested_true"),
            lambda _inner: state,
            name="nested",
        ),
        lambda _T: state,
        name="outer",
    )
    validate_program_regions(program)

    graph = program.to_graph()
    outer = next(node for node in graph.nodes if type(node) is Branch)
    inner = next(node for node in outer.when_true.nodes if type(node) is Branch)
    assert selected.name == "outer"
    assert inner.when_false.nodes == ()
    assert inner.when_false.result.node_id == state.id


@pytest.mark.parametrize(
    ("condition", "when_true", "when_false", "message"),
    [
        (lambda T, U: T.norm2(U), lambda T, U: U, lambda T, U: U,
         "condition must be a scalar Bool"),
        (lambda T, U: T.norm2(U) > 0, lambda T, U: T.norm2(U), lambda T, U: U,
         "same value type"),
    ],
)
def test_branch_rejects_non_bool_condition_and_incompatible_arm_types(
        condition, when_true, when_false, message):
    program = Program("invalid-branch")
    state = typed_state(program, "fluid")
    with pytest.raises(ValueError, match=message):
        program.branch(
            condition(program, state),
            lambda T: when_true(T, state),
            lambda T: when_false(T, state),
        )


def test_branch_rejects_arm_with_different_exact_point():
    program = Program("point-mismatch")
    state = typed_state(program, "fluid")
    condition = program.norm2(state) > 0
    with pytest.raises(ValueError, match="clock and exact point"):
        program.branch(
            condition,
            lambda T: _copy(T, state, "same_point"),
            lambda T: T.value(
                "next_point", state, at=TimePoint(T.clock, step=1)),
        )


def test_branch_codegen_places_each_arm_inside_if_else_only():
    program = Program("branch-codegen")
    state = typed_state(program, "fluid")
    condition = program.norm2(state) > 0
    selected = program.branch(
        condition,
        lambda T: _copy(T, state, "true_only"),
        lambda T: _copy(T, state, "false_only"),
    )
    endpoint = typed_state(program, "fluid", state_name="U").next
    program.commit(endpoint, program.value("next", selected, at=endpoint.point))

    source = program.emit_cpp_program()
    branch_start = source.index("if (")
    else_start = source.index("} else {", branch_start)
    branch_end = source.index("}", else_start + len("} else {"))
    assert "u" in source[branch_start:else_start]
    assert "u" in source[else_start:branch_end]
    assert source.count("} else {") == 1
