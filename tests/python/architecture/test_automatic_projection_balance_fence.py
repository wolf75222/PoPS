"""Projection due markers remain attempt-local on the exact-ranked Program path."""

from pathlib import Path

from tests.python.architecture.test_final_nd_amr_consumers import (
    ROOTS as AMR_CONSUMER_ROOTS,
    _semantic_closure as _amr_semantic_closure,
    _source as _amr_semantic_source,
)


ROOT = Path(__file__).resolve().parents[3]
PROGRAM_STATE = ROOT / "include/pops/runtime/program/program_runtime_state.hpp"
UNIFORM_CONTEXT = ROOT / "include/pops/runtime/program/program_context.hpp"
BALANCE_CODEGEN = ROOT / "python/pops/codegen/program_balance_due.py"
RETIRED_SERVICES = ROOT / "include/pops/runtime/program/program_execution_services.hpp"


def _between(text: str, begin: str, end: str) -> str:
    return text.split(begin, 1)[1].split(end, 1)[0]


def test_generated_due_marker_precedes_operators_and_is_attempt_local() -> None:
    codegen = BALANCE_CODEGEN.read_text(encoding="utf-8")
    emit = _between(
        codegen,
        "def emit_balance_due_guards(",
        "\ndef balance_value_due_expression(",
    )
    assert "automatic_tokens = []" in emit
    assert "if automatic_tokens:" in emit
    assert "ctx.note_automatic_balance_capture_due(%s);" in emit

    state = PROGRAM_STATE.read_text(encoding="utf-8")
    assert "bool automatic_balance_due_ = false;" in state
    capture_due = _between(
        state,
        "[[nodiscard]] bool automatic_balance_capture_due() const noexcept",
        "/// Accumulate one signed, metric-integrated native operator contribution.",
    )
    assert "!balance_replay_active_ && automatic_balance_due_" in capture_due
    assert "automatic_balance_due_ = automatic_balance_due_ || due;" in capture_due


def test_projection_dispatch_lives_on_the_ranked_context_not_a_parallel_service() -> None:
    assert not RETIRED_SERVICES.exists()
    uniform = UNIFORM_CONTEXT.read_text(encoding="utf-8")
    projection = _between(
        uniform,
        "void apply_projection(int program_block, field_type& state_value) const",
        "Real max_wave_speed(",
    )
    assert "const ExecutionLane& lane = prepared_execution_lane();" in projection
    assert "const int runtime_block =" in projection
    assert (
        'resolve_prepared_program_block_(program_block, lane, "Program projection block")'
        in projection
    )
    assert "system_->block_project(runtime_block, state_value);" in projection
    assert "void note_automatic_balance_capture_due(bool due) const" in uniform
    assert "runtime_state().note_automatic_balance_capture_due" in uniform
    assert "template <int Dim>" in uniform

    amr = _amr_semantic_source(_amr_semantic_closure(AMR_CONSUMER_ROOTS["program"]))
    assert "ProgramExecutionServices" not in amr
    projection = _between(
        amr,
        "void apply_projection(int program_block, field_type& detached_candidate) const",
        "Real max_wave_speed(",
    )
    assert "const int runtime_block = sys_block(program_block);" in projection
    assert (
        "const int candidate_owner = projection_candidate_owner_(detached_candidate);" in projection
    )
    assert (
        "project_prepared_amr_block_level_state(runtime_block, active_level_, candidate_owner,"
        in projection
    )
