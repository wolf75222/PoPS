"""Output transfer capabilities must never hide an integrator's lagged input."""
from __future__ import annotations

import pytest
from types import SimpleNamespace

from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_history_identity import (
    SCALAR_OUTPUT_HISTORY_SPACE,
    history_space_identity,
)
from test_mapped_condensed import _mapped_program


def _stored_potential(*, named=False, depth=1):
    program, model = _mapped_program()
    potential = next(value for value in program._values
                     if value.op == "solve_outcome_component")
    if named:
        potential = program.value("named_potential", 2 * potential - potential)
    program.store_history("disk.potential", potential, depth=depth)
    return program, model, potential


@pytest.mark.parametrize("named", (False, True))
def test_registration_and_checkpoint_share_exact_output_identity(named):
    program, model, _ = _stored_potential(named=named)
    assert history_space_identity(program, "disk.potential") == SCALAR_OUTPUT_HISTORY_SPACE
    source = emit_cpp_program(program, model=model, target="amr_system")
    registrations = [line for line in source.splitlines()
                     if 'ctx.register_history("disk.potential"' in line]
    # Installation and the hierarchy gather callback both register this ring.
    # Every site must match the frozen checkpoint descriptor before the first step.
    assert len(registrations) >= 2
    assert all('"scalar-output-field-v1"' in line for line in registrations)
    checkpoint = source.split("pops_program_checkpoint_history_space_identity", 1)[1]
    assert 'case 0: return "scalar-output-field-v1";' in checkpoint
    assert 'ctx.history("disk.potential"' not in source


@pytest.mark.parametrize("consumer", ("direct", "named_expression", "apply", "nested_branch"))
def test_direct_indirect_and_nested_reads_remain_ordinary_histories(consumer):
    program, model, potential = _stored_potential()
    block = potential.block

    def read(builder):
        value = builder.history("disk.potential", ncomp=1, block=block)
        return builder.value("indirect_lag", 2 * value - value)

    if consumer == "direct":
        program.history("disk.potential", ncomp=1, block=block)
    elif consumer == "named_expression":
        read(program)
    elif consumer == "apply":
        captured = read(program)
        operator = program.matrix_free_operator("lagged_apply")
        operator = program.set_apply(operator, lambda _builder, _out, value: value)
    else:
        condition = program.norm2(potential) > 0
        program.branch(condition,
                       lambda builder: builder.branch(condition, read, read), read)
    assert history_space_identity(program, "disk.potential") == "scalar-field"
    if consumer == "apply":
        # Adversarial frozen apply: ordinary stencil authoring already refuses a
        # lagged capture. Classification must independently follow its affine
        # result even without a flat history root; do not grant a native capability
        # merely because a different validation pass would reject this graph.
        graph = SimpleNamespace(**{
            key: getattr(program, key) for key in (
                "_history_spaces", "_history_blocks", "_history_state_refs",
                "_histories_ncomp", "_histories", "_commits", "_canonical_value",
                "_subblock_value_refs")})
        altered_apply = SimpleNamespace(
            op=operator.op, inputs=operator.inputs,
            attrs={**operator.attrs, "apply_result": 2 * captured})
        graph._values = [altered_apply if value is operator else value
                         for value in program._values
                         if value.op != "history" and value is not captured]
        assert not any(value.op == "history" for value in graph._values)
        assert history_space_identity(graph, "disk.potential") == "scalar-field"
    if consumer in {"direct", "named_expression"}:
        source = emit_cpp_program(program, model=model, target="amr_system")
        registrations = [line for line in source.splitlines()
                         if 'ctx.register_history("disk.potential"' in line]
        assert len(registrations) >= 2
        assert all('"scalar-field"' in line for line in registrations)
        assert 'ctx.history_zero_start("disk.potential"' in source
        assert SCALAR_OUTPUT_HISTORY_SPACE not in source


def test_deeper_output_history_has_no_two_slot_capability():
    program, _, _ = _stored_potential(depth=2)
    assert history_space_identity(program, "disk.potential") == "scalar-field"
