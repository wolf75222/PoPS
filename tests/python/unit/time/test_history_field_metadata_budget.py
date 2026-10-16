import pytest

from pops.time import Program
from pops.time._program.resource_budgets import validate_history_field_metadata_budget


@pytest.mark.parametrize("budget", [(1, 1), (4096, 8192), ((1 << 63) - 1, (1 << 63) - 1)])
def test_explicit_metadata_budget_is_part_of_program_identity(budget):
    p = Program("history-metadata", history_field_metadata_budget=budget)
    assert p.history_field_metadata_budget == budget
    assert p._serialize(include_provenance=False)["history_field_metadata_budget"] == budget


@pytest.mark.parametrize("budget", [(0, 1), (1, 0), (True, 1024), (1, -2), [1, 2], (1,), (1 << 63, 1)])
def test_invalid_metadata_budget_fails_at_authoring(budget):
    with pytest.raises(ValueError, match="history_field_metadata_budget"):
        Program("invalid", history_field_metadata_budget=budget)


def test_metadata_budget_is_not_a_flux_ceiling():
    p = Program("separate", history_field_metadata_budget=(4000, 8000))
    assert p.flux_expression_budget is None
    validate_history_field_metadata_budget(None)


def test_metadata_capacity_survives_detach_and_is_canonical_identity():
    from pops.identity.semantic import program_semantic_data
    from pops.time._program.detach import detach_compiled_program
    p = Program("history", history_field_metadata_budget=(4096, 32768))
    q = detach_compiled_program(p)
    for data in (p._serialize(), q._serialize(), program_semantic_data(p)):
        assert tuple(data["history_field_metadata_budget"]) == (4096, 32768)
    assert p._ir_hash() == q._ir_hash()
    p.history_field_metadata_budget = (8192, 32768)
    assert p._ir_hash() != q._ir_hash()
    p.history_field_metadata_budget = (True, 32768)
    with pytest.raises(ValueError, match="history_field_metadata_budget"):
        p._serialize()
