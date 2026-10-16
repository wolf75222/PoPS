"""A resource ceiling is preserved through lowering, never a scientific theorem."""
import pytest

from pops.codegen.program_emit_amr import _flux_expression_budgets
from pops.time import Program
from pops.time._program.detach import detach_compiled_program


@pytest.mark.parametrize("value", ((0, 1), (1, -1), (True, 1), (1,), [1, 1], (1 << 63, 1)))
def test_flux_ceiling_rejects_malformed_limits(value):
    with pytest.raises(ValueError, match="flux_expression_budget"):
        Program("bounded", flux_expression_budget=value)


def test_resource_limits_survive_clone_and_are_part_of_ir_identity():
    program = Program("bounded", flux_expression_budget=(1024, 32),
                      max_exact_coefficient_bytes=512)
    clone = detach_compiled_program(program)
    assert clone.flux_expression_budget == (1024, 32)
    assert clone.max_exact_coefficient_bytes == 512
    assert clone._ir_hash() == program._ir_hash()
    assert program._serialize()["flux_expression_budget"] == (1024, 32)
    assert program._ir_hash() != Program("bounded", flux_expression_budget=(1025, 32),
                                        max_exact_coefficient_bytes=512)._ir_hash()


def test_explicit_budget_applies_to_each_owned_block_without_model_dispatch():
    from tests.python.support.typed_program import program_states, synthetic_module
    program = Program("owned_bound")
    _,states=program_states(program, synthetic_module("owned_bound_model", components=("u",)), ("owner",))
    _=states["owner"].n
    program.flux_expression_budget = (128, 12)
    assert _flux_expression_budgets(program) == ((128, 12),)
    program.flux_expression_budget = (0, 12)
    with pytest.raises(ValueError, match="flux_expression_budget"):
        _flux_expression_budgets(program)


def _large_program(blocks):
    from tests.python.support.typed_program import program_states, synthetic_module
    program=Program("aggregate",flux_expression_budget=((1 << 63)-1,1))
    _,states=program_states(program,synthetic_module("aggregate_model",components=("u",)),blocks)
    for state in states.values():
        _=state.n
    return program


def test_aggregate_budget_rejects_overflow_before_emitting_cpp():
    from pops.codegen.program_emit_amr import _emit_flux_expression_budget
    program=_large_program(("a","b","c"))
    with pytest.raises(ValueError,match="aggregate.*uint64 ABI"):
        _emit_flux_expression_budget(program)


def test_large_legal_budget_is_not_silently_clamped():
    from pops.codegen.program_emit_amr import _emit_flux_expression_budget
    limit=(1 << 63)-1
    program=_large_program(("a","b"))
    assert _flux_expression_budgets(program)==((limit,1),(limit,1))
    source=_emit_flux_expression_budget(program)
    assert f"return UINT64_C({2*limit});" in source
