"""ADC-686: automatic reflux evidence stays exact, sparse, and fail-closed."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROGRAM_STATE = (
    ROOT / "include" / "pops" / "runtime" / "program" / "program_runtime_state.hpp"
)
AMR_CONTEXT = (
    ROOT / "include" / "pops" / "runtime" / "program" / "amr_program_context.hpp"
)
AMR_REFLUX = ROOT / "include" / "pops" / "runtime" / "amr" / "amr_program_reflux.hpp"
AMR_SUBCYCLING = (
    ROOT / "include" / "pops" / "numerics" / "time" / "amr" / "levels"
    / "amr_subcycling.hpp"
)
AMR_PATCH_RANGE = (
    ROOT / "include" / "pops" / "numerics" / "time" / "amr" / "levels"
    / "amr_patch_range.hpp"
)
UNIFORM_IMPL = ROOT / "src" / "runtime" / "system" / "system_impl.hpp"
AMR_IMPL = ROOT / "src" / "runtime" / "amr" / "amr_system.cpp"


def _between(text: str, begin: str, end: str) -> str:
    return text.split(begin, 1)[1].split(end, 1)[0]


def test_automatic_balance_mailbox_is_attempt_local_and_not_a_route_fallback() -> None:
    state = PROGRAM_STATE.read_text()
    assert "struct AutomaticBalanceKey" in state
    assert "std::map<AutomaticBalanceKey, Real> automatic_balance_terms_;" in state
    assert "automatic_balance_terms_.clear();" in state
    assert "record_automatic_balance_term(" in state
    assert "automatic_balance_capture_due()" in state

    accepted = _between(
        state,
        "std::map<std::string, Real> accepted_balance_terms(",
        "void begin_balance_due_window(",
    )
    assert "step_balance_terms_" in accepted
    assert "automatic_balance_terms_" not in accepted

    uniform = UNIFORM_IMPL.read_text()
    adaptive = AMR_IMPL.read_text()
    for source in (uniform, adaptive):
        assert "automatic_balance_terms" in source
        assert "impl.program_.automatic_balance_terms_" in source


def test_reflux_integral_comes_from_the_gathered_sparse_correction() -> None:
    register = AMR_PATCH_RANGE.read_text()
    component_sums = _between(
        register,
        "[[nodiscard]] std::vector<Real> component_sums(",
        "[[nodiscard]] std::size_t lookup_capacity()",
    )
    assert "device_fence();" in component_sums
    assert "cell_measure * buf[offset + component]" in component_sums
    assert "all_reduce" not in component_sums

    transition = AMR_SUBCYCLING.read_text()
    synchronize = _between(
        transition,
        "void synchronize_integrated(",
        "\n private:",
    )
    assert synchronize.index("correction_.gather(communicator);") < synchronize.index(
        "correction_.component_sums(dx * dy)"
    )
    assert synchronize.index("correction_.component_sums(dx * dy)") < synchronize.index(
        "ApplyRefluxRegisterKernel"
    )

    route = AMR_REFLUX.read_text()
    routing = _between(route, "inline void route_reflux_program(", "\n}\n\n}  // namespace detail")
    assert "std::vector<Real>* integrated_state_correction = nullptr" in routing
    assert "integrated_state_correction);" in routing


def test_amr_records_reflux_before_average_down_only_when_balance_is_due() -> None:
    context = AMR_CONTEXT.read_text()
    synchronize = _between(
        context,
        "void synchronize_level_pair_(",
        "void finalize_history_rotation_()",
    )
    assert "automatic_balance_capture_due()" in synchronize
    assert "record_automatic_balance_term(" in synchronize
    assert '"reflux"' in synchronize
    assert "reduce_sum(" not in synchronize
    assert synchronize.index("route_reflux_program(") < synchronize.index(
        "record_automatic_balance_term("
    )
    assert synchronize.index("record_automatic_balance_term(") < synchronize.index(
        "SyncPhase::AverageDown"
    )
