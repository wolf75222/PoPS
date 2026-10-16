import pytest
from pops.time import Program
from pops.codegen.program_amr_continuation import checkpoint_exports


@pytest.mark.parametrize("budget", [None, (4096, 16384)])
def test_history_metadata_export_is_independent_of_flux_and_continuation(budget):
    p = Program("history metadata export", history_field_metadata_budget=budget)
    exports = checkpoint_exports(p)
    assert p.flux_expression_budget is None
    for name, count in zip(("storage_bytes", "workspace_bytes"), budget or (0, 0), strict=True):
        assert (f'pops_program_history_field_metadata_{name}() {{ return {count}; }}') in exports
    assert 'pops_program_checkpoint_continuation_root_byte_capacity() { return 0; }' in exports


def test_changed_invalid_budget_cannot_be_emitted_as_checkpoint_authority():
    p = Program("bad history metadata", history_field_metadata_budget=(10, 20))
    p.history_field_metadata_budget = (10, True)
    with pytest.raises(ValueError, match="history_field_metadata_budget"):
        checkpoint_exports(p)
