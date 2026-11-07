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


def _history_program(name, **resources):
    from typed_program_support import typed_state
    p = Program(name, **resources)
    tracked = typed_state(p, "plasma", state_name="U")
    p.keep_history(tracked, depth=1)
    p.commit(tracked.next, p.value("U_next", 1.0 * tracked.n, at=tracked.next.point))
    return p


def _installed_capacity(p):
    from pops.codegen.program_amr_continuation import checkpoint_exports
    from pops.codegen.program_codegen import emit_cpp_program
    source, exports = emit_cpp_program(p), checkpoint_exports(p)
    configured = [line.strip() for line in source.splitlines()
                  if "ctx.configure_history_field_metadata(" in line]
    exported = tuple(
        int(exports.split(f"pops_program_history_field_metadata_{name}() {{ return ", 1)[1]
            .split(";", 1)[0])
        for name in ("storage_bytes", "workspace_bytes"))
    return configured, exported


def test_history_without_authored_budget_installs_the_plan_default():
    from pops.time._program.resource_budgets import (
        DEFAULT_HISTORY_FIELD_METADATA_BUDGET, resolved_history_field_metadata_budget)
    p = _history_program("plan default")
    # The authored Program and its identity keep no infrastructure value (spec 0.3, B3).
    assert p.history_field_metadata_budget is None
    assert p._serialize(include_provenance=False)["history_field_metadata_budget"] is None
    assert resolved_history_field_metadata_budget(p) == DEFAULT_HISTORY_FIELD_METADATA_BUDGET
    configured, exported = _installed_capacity(p)
    assert configured == ["ctx.configure_history_field_metadata(UINT64_C(%d), UINT64_C(%d));"
                          % DEFAULT_HISTORY_FIELD_METADATA_BUDGET]
    assert exported == DEFAULT_HISTORY_FIELD_METADATA_BUDGET


def test_authored_budget_takes_precedence_over_the_plan_default():
    p = _history_program("authored", history_field_metadata_budget=(4096, 32768))
    configured, exported = _installed_capacity(p)
    assert configured == ["ctx.configure_history_field_metadata(UINT64_C(4096), UINT64_C(32768));"]
    assert exported == (4096, 32768)


def test_program_without_history_installs_no_metadata_capacity():
    from typed_program_support import typed_state
    from pops.time._program.resource_budgets import resolved_history_field_metadata_budget
    p = Program("no history")
    tracked = typed_state(p, "plasma", state_name="U")
    p.commit(tracked.next, p.value("U_next", 1.0 * tracked.n, at=tracked.next.point))
    assert resolved_history_field_metadata_budget(p) is None
    assert _installed_capacity(p) == ([], (0, 0))
