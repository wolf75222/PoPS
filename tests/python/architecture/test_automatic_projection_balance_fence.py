"""ADC-686: projection balance evidence is due-only, metric, and still private."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROGRAM_STATE = ROOT / "include" / "pops" / "runtime" / "program" / "program_runtime_state.hpp"
EXECUTION_SERVICES = (
    ROOT / "include" / "pops" / "runtime" / "program" / "program_execution_services.hpp"
)
UNIFORM_CONTEXT = ROOT / "include" / "pops" / "runtime" / "program" / "program_context.hpp"
AMR_CONTEXT = ROOT / "include" / "pops" / "runtime" / "program" / "amr_program_context.hpp"
BALANCE_CODEGEN = ROOT / "python" / "pops" / "codegen" / "program_balance_due.py"
UNIFORM_IMPL = ROOT / "src" / "runtime" / "system" / "system_impl.hpp"
AMR_IMPL = ROOT / "src" / "runtime" / "amr" / "amr_system.cpp"


def _between(text: str, begin: str, end: str) -> str:
    return text.split(begin, 1)[1].split(end, 1)[0]


def test_generated_due_marker_precedes_operators_and_is_attempt_local() -> None:
    codegen = BALANCE_CODEGEN.read_text()
    emit = _between(
        codegen,
        "def emit_balance_due_guards(",
        "\ndef balance_value_due_expression(",
    )
    assert "automatic_tokens = []" in emit
    assert "if automatic_tokens:" in emit
    assert "ctx.note_automatic_balance_capture_due(%s);" in emit

    state = PROGRAM_STATE.read_text()
    assert "bool automatic_balance_due_ = false;" in state
    capture_due = _between(
        state,
        "[[nodiscard]] bool automatic_balance_capture_due() const noexcept",
        "/// Accumulate one signed, metric-integrated native operator contribution.",
    )
    assert "!balance_replay_active_ && automatic_balance_due_" in capture_due
    assert "automatic_balance_due_ = automatic_balance_due_ || due;" in capture_due

    attempt_entry = _between(
        state,
        "void begin_step_projection_report()",
        "void note_step_projection(",
    )
    assert "automatic_balance_due_ = false;" in attempt_entry

    uniform = UNIFORM_IMPL.read_text()
    adaptive = AMR_IMPL.read_text()
    for source in (uniform, adaptive):
        assert "automatic_balance_due" in source
        assert "impl.program_.automatic_balance_due_" in source


def test_projection_delta_is_captured_only_when_due_and_stays_qualified() -> None:
    services = EXECUTION_SERVICES.read_text()
    projection = _between(
        services,
        "void apply_projection(int block, MultiFab& state) const",
        "/// Minimum physical cell size used by the native CFL authority.",
    )
    assert "if (!runtime.automatic_balance_capture_due())" in projection
    assert projection.count("program_execution_projection_balance_integrals_") == 2
    assert projection.index("const std::optional<std::vector<Real>> before") < projection.index(
        "program_execution_apply_projection_"
    )
    assert projection.index("program_execution_apply_projection_") < projection.index(
        "const std::optional<std::vector<Real>> after"
    )
    assert "record_automatic_balance_term(" in projection
    assert '"projection"' in projection
    assert "runtime_block, level, component" in projection

    state = PROGRAM_STATE.read_text()
    accepted = _between(
        state,
        "std::map<std::string, Real> accepted_balance_terms(",
        "void begin_balance_due_window(",
    )
    assert "automatic_balance_terms_" not in accepted


def test_uniform_projection_evidence_uses_exact_available_measure() -> None:
    context = UNIFORM_CONTEXT.read_text()
    provider = _between(
        context,
        "std::optional<std::vector<Real>> program_execution_projection_balance_integrals_(",
        "Real program_execution_hmin_() const",
    )
    assert "if (sys_->program_is_polar())" in provider
    assert "return std::nullopt;" in provider
    assert "context.geom.dx() * context.geom.dy()" in provider
    assert "RelativeCellMeasure measure;" in provider
    assert "measure.active_cells = context.domain_mask;" in provider
    assert "measure.inverse_volume_fraction = context.eb_inverse_volume_fraction;" in provider
    assert "pops::reduce_sum(state, component, measure)" in provider


def test_amr_projection_evidence_excludes_covered_cells_and_reduces_once() -> None:
    context = AMR_CONTEXT.read_text()
    provider = _between(
        context,
        "std::optional<std::vector<Real>> program_execution_projection_balance_integrals_(",
        "Real program_execution_hmin_() const",
    )
    assert "active_mask(views, level_, next)" in provider
    assert "CompositeSumKind::Sum" in provider
    assert "local_sum(" in provider
    assert "if (!eng_->level_is_replicated(level_))" in provider
    assert provider.count("all_reduce_sum_inplace(") == 1
    assert "geometry.dx()) * static_cast<double>(geometry.dy())" in provider
    assert "state.n_grow() != live.n_grow()" in provider
    assert "state.local_size() != live.local_size()" in provider
