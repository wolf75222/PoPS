"""ADC-686: public Balance routes select qualified native evidence fail-closed."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
LEDGER = ROOT / "python" / "pops" / "_balance_contract.py"
MEASURES = ROOT / "python" / "pops" / "diagnostics" / "measures.py"
CONSUMERS = ROOT / "python" / "pops" / "runtime" / "_runtime_consumers.py"
PROGRAM_STATE = (
    ROOT / "include" / "pops" / "runtime" / "program" / "program_runtime_state.hpp"
)
SYSTEM = ROOT / "src" / "runtime" / "system" / "system_program.cpp"
AMR = ROOT / "src" / "runtime" / "amr" / "amr_system.cpp"
SYSTEM_BINDING = ROOT / "python" / "bindings" / "core" / "init" / "init_system.cpp"
AMR_BINDING = ROOT / "python" / "bindings" / "core" / "init" / "init_amr.cpp"


def _between(text: str, begin: str, end: str) -> str:
    return text.split(begin, 1)[1].split(end, 1)[0]


def test_public_ledger_owns_role_and_exact_automatic_term_selection() -> None:
    ledger = LEDGER.read_text()
    assert "role: Any = None" in ledger
    assert "component: int | None = None" in ledger
    assert "automatic_terms: tuple[str, ...] = ()" in ledger
    assert '{"reflux", "projection"}' in ledger

    measures = MEASURES.read_text()
    balance = _between(measures, "class Balance(_Measure):", "class ConservationCheck")
    assert "role=ledger.role" in balance
    assert '"automatic_terms": list(self.ledger.automatic_terms)' in balance
    assert '"balance_component": self.ledger.component' in balance


def test_runtime_uses_selected_native_entrypoint_only_for_delegated_terms() -> None:
    consumers = CONSUMERS.read_text()
    native = _between(
        consumers,
        "def _native_balance_terms(",
        "def _diagnostic_values(",
    )
    assert '"_selected_accepted_balance_terms"' in native
    assert 'if automatic_terms' in native
    assert "native(route, block, component, list(levels), list(automatic_terms))" in native
    assert "else native(route)" in native

    for binding in (SYSTEM_BINDING, AMR_BINDING):
        assert '"_selected_accepted_balance_terms"' in binding.read_text()


def test_native_selector_requires_complete_owner_level_component_evidence() -> None:
    state = PROGRAM_STATE.read_text()
    selector = _between(
        state,
        "std::map<std::string, Real> selected_accepted_balance_terms(",
        "void begin_balance_due_window(",
    )
    assert "AutomaticBalanceKey key{runtime_block, levels[index], component, term}" in selector
    assert "native producer omitted term" in selector
    assert "both Program and native producer authority" in selector
    assert 'term == "reflux" ? levels.size() - 1 : levels.size()' in selector

    uniform = SYSTEM.read_text()
    assert "const int runtime_block = p_->index(block);" in uniform
    assert "levels != std::vector<int>{0}" in uniform

    adaptive = AMR.read_text()
    assert "const std::size_t runtime_block = p_->block_index_or_throw(block);" in adaptive
    assert "p_->runtime->block_n_vars(runtime_block)" in adaptive
    assert "p_->runtime->nlev()" in adaptive
