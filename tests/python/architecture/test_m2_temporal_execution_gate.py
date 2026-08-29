"""Source-only integrity checks for the executable M2 temporal gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.python.conftest import _requires_process_collection


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "tests/gates/m2_temporal_execution.toml"
RUNNER = ROOT / "scripts/run_m2_gate.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("pops_run_m2_gate", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_m2_manifest_references_only_real_mandatory_proofs():
    data, errors = _load_runner().validate_manifest(MANIFEST)
    assert not errors, "M2 gate matrix is incomplete:\n  " + "\n  ".join(errors)
    assert len(data["check"]) == 65


def test_m2_final_gate_has_no_deferred_requirement():
    data, errors = _load_runner().validate_manifest(MANIFEST)
    assert not errors
    assert data["deferred"] == []
    assert {row["issue"] for row in data["check"]} == {
        "ADC-648",
        "ADC-661",
        "ADC-662",
        "ADC-663",
        "ADC-664",
        "ADC-665",
        "ADC-666",
        "ADC-667",
        "ADC-668",
    }


def test_m2_fallible_solver_evaluations_use_exact_native_proofs():
    data, errors = _load_runner().validate_manifest(MANIFEST)
    assert not errors
    checks = {
        (row["requirement"], row["polarity"]): (
            row["target"],
            row.get("test_regex"),
        )
        for row in data["check"]
        if row["requirement"]
        in {
            "fallible_nonlinear_evaluation",
            "fallible_linear_evaluation",
        }
    }
    assert checks == {
        ("fallible_nonlinear_evaluation", "positive"): (
            "test_newton_robustness",
            r"^NewtonRobustnessTest\.fallible_analytic_jacobian_"
            r"rejects_privately_and_success_keeps_legacy_result$",
        ),
        ("fallible_nonlinear_evaluation", "refusal"): (
            "test_newton_robustness",
            r"^NewtonRobustnessTest\.fallible_source_propagates_"
            r"retry_reject_fail_and_invalid_without_publication$",
        ),
        ("fallible_linear_evaluation", "positive"): (
            "test_coupled_fieldsolve",
            r"^test_coupled_fieldsolve_np2$",
        ),
        ("fallible_linear_evaluation", "refusal"): (
            "test_krylov_collective_contract",
            r"^test_krylov_collective_contract_np2$",
        ),
    }


def test_m2_solve_outcome_exact_once_and_no_early_publication_use_closed_proofs():
    data, errors = _load_runner().validate_manifest(MANIFEST)
    assert not errors
    checks = {
        (row["kind"], row["polarity"], row["target"], row.get("nodeid", row.get("test_regex")))
        for row in data["check"]
        if row["issue"] == "ADC-665" and row["requirement"] == "solve_outcome"
    }
    assert checks == {
        (
            "pytest",
            "positive",
            "solve",
            "tests/python/unit/time/test_solve_outcome_contract.py::"
            "test_reject_attempt_is_explicit_in_the_canonical_program_graph",
        ),
        (
            "pytest",
            "refusal",
            "solve",
            "tests/python/unit/time/test_solve_outcome_contract.py::"
            "test_solve_outcome_refuses_implicit_invalid_and_double_consumption",
        ),
        (
            "pytest",
            "positive",
            "solve",
            "tests/python/unit/time/test_field_solve_outcome.py::"
            "test_consuming_field_solve_publishes_one_exact_field_context",
        ),
        (
            "pytest",
            "refusal",
            "solve",
            "tests/python/unit/time/test_field_solve_outcome.py::"
            "test_field_solve_consume_rejects_untyped_failure_action_atomically",
        ),
        (
            "ctest",
            "positive",
            "test_newton_robustness",
            r"^NewtonRobustnessTest\.prepared_success_publishes_only_on_single_accept$",
        ),
        (
            "ctest",
            "refusal",
            "test_newton_robustness",
            r"^NewtonRobustnessTest\.publication_layout_failure_does_not_consume_the_outcome$",
        ),
        (
            "ctest",
            "positive",
            "test_mpi_field_plan_consensus",
            r"^test_mpi_field_plan_consensus_np2$",
        ),
    }


def test_m2_native_pytest_execution_rejects_every_skip_or_xfail(tmp_path, monkeypatch):
    runner = _load_runner()
    report = tmp_path / "pytest.xml"
    skipped_xml = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<testsuites><testsuite tests="1" skipped="1">'
        '<testcase classname="m2" name="proof"><skipped type="pytest.skip"/></testcase>'
        "</testsuite></testsuites>"
    )
    report.write_text(skipped_xml, encoding="utf-8")
    assert runner._pytest_skip_count(report) == 1

    source = RUNNER.read_text(encoding="utf-8")
    assert 'environment["POPS_REQUIRE_NATIVE_TESTS"] = "1"' in source
    assert '"xfail_strict=true"' in source
    assert "skipped/xfail proof(s); every proof is mandatory" in source

    def successful_pytest_with_a_skip(command, *, cwd, env, check):
        assert cwd == ROOT
        assert env["POPS_REQUIRE_NATIVE_TESTS"] == "1"
        assert check is False
        junit = Path(command[command.index("--junitxml") + 1])
        junit.write_text(skipped_xml, encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", successful_pytest_with_a_skip)
    with pytest.raises(RuntimeError, match="reported 1 skipped/xfail proof"):
        runner._run_pytest(["tests/python/unit/time/test_exact_program_graph.py::proof"])


@pytest.mark.parametrize("selected_count", [0, 2])
def test_m2_ctest_selector_must_resolve_one_exact_test(monkeypatch, tmp_path, selected_count):
    runner = _load_runner()

    def listing(command, **kwargs):
        assert command[:2] == ["ctest", "--test-dir"]
        assert kwargs["check"] is True
        assert kwargs["capture_output"] is True
        return SimpleNamespace(returncode=0, stdout=f"Total Tests: {selected_count}\n")

    monkeypatch.setattr(runner.subprocess, "run", listing)
    with pytest.raises(
        RuntimeError, match=rf"resolved {selected_count} tests, expected exactly 1"
    ):
        runner._run_ctest(
            tmp_path / "build",
            "test_amr_history_ring",
            r"^test_amr_history_ring\\.AcceptedFacadeTransactionCommitsTopologyStateHistoryAndClock$",
        )


def test_m2_pytest_nodeids_are_individually_collectible():
    data, errors = _load_runner().validate_manifest(MANIFEST)
    assert not errors
    process_collected = {
        row["nodeid"].split("::", 1)[0]
        for row in data["check"]
        if row["kind"] == "pytest"
        and _requires_process_collection(ROOT / row["nodeid"].split("::", 1)[0])
    }
    assert process_collected == set()


def test_m2_adc667_history_and_migration_routes_use_exact_proofs():
    data, errors = _load_runner().validate_manifest(MANIFEST)
    assert not errors
    checks = {
        (row["target"], row["polarity"], row["nodeid"])
        for row in data["check"]
        if row["issue"] == "ADC-667" and row["requirement"] == "temporal_restart"
    }
    assert checks == {
        (
            "transaction",
            "positive",
            "tests/python/unit/runtime/test_temporal_restart_state.py"
            "::test_accepted_attempt_advances_cursor_and_round_trips_exact_controller_state",
        ),
        (
            "transaction",
            "refusal",
            "tests/python/unit/runtime/test_temporal_restart_state.py"
            "::test_rejection_preserves_native_cursor_and_makes_checkpoint_ineligible",
        ),
        (
            "schedule",
            "positive",
            "tests/python/unit/time/test_multirate_history_contract.py"
            "::test_history_interpolation_is_an_explicit_cross_clock_provider",
        ),
        (
            "restart",
            "positive",
            "tests/python/unit/runtime/test_temporal_restart_state.py"
            "::test_uniform_child_clock_history_owns_exact_slot_ledger_across_restart",
        ),
        (
            "schedule",
            "refusal",
            "tests/python/unit/time/test_multirate_history_contract.py"
            "::test_cross_clock_extension_without_provider_is_rejected",
        ),
        (
            "restart",
            "positive",
            "tests/python/unit/codegen/test_checkpoint_migration.py"
            "::test_true_frozen_v2_migrates_and_strict_uniform_restart_accepts",
        ),
        (
            "restart",
            "refusal",
            "tests/python/unit/runtime/test_temporal_restart_state.py"
            "::test_frozen_release_v2_fixture_is_refused_offline_and_at_runtime_boundary",
        ),
    }


def test_m2_restart_hierarchy_and_program_only_routes_use_real_exact_proofs():
    data, errors = _load_runner().validate_manifest(MANIFEST)
    assert not errors
    checks = data["check"]
    restart_positive = {
        row.get("nodeid")
        for row in checks
        if row["requirement"] == "restart" and row["polarity"] == "positive"
    }
    assert restart_positive == {
        "tests/python/unit/runtime/test_temporal_restart_state.py"
        "::test_uniform_native_linear_history_uses_bracketing_slots_and_restarts_exactly"
    }
    restart_refusals = {
        row.get("nodeid")
        for row in checks
        if row["requirement"] == "restart" and row["polarity"] == "refusal"
    }
    assert restart_refusals == {
        "tests/python/unit/runtime/test_temporal_restart_state.py"
        "::test_restart_rejects_a_different_installed_nested_clock_schedule"
    }
    hierarchy_ordering = {
        row["polarity"]: (
            row["kind"],
            row["target"],
            row.get("nodeid", row.get("test_regex")),
        )
        for row in checks
        if row["requirement"] == "refined_hierarchy_native_ordering"
    }
    assert hierarchy_ordering == {
        "positive": (
            "pytest",
            "solve",
            "tests/python/unit/codegen/test_composite_tensor_fac_provider.py"
            "::test_header_only_hierarchy_extension_compiles_its_own_generic_provider_identity",
        ),
        "refusal": (
            "ctest",
            "test_amr_history_ring",
            r"^test_amr_history_ring\.FineNonFiniteAfterCoarseSuccessRestoresCompleteAcceptedState$",
        ),
    }
    temporal_routes = {
        row["nodeid"] for row in checks if row["requirement"] == "program_only_temporal_routes"
    }
    assert temporal_routes == {
        "tests/python/architecture/test_program_only_temporal_facades.py"
        "::test_system_temporal_facades_dispatch_only_through_an_installed_program",
        "tests/python/architecture/test_program_only_temporal_facades.py"
        "::test_amr_temporal_facades_use_amr_runtime_only_as_the_spatial_engine",
        "tests/python/architecture/test_program_only_temporal_facades.py"
        "::test_static_system_assembler_is_retired_from_the_final_runtime_surface",
        "tests/python/architecture/test_program_only_temporal_facades.py"
        "::test_nonlinear_amr_semantics_use_the_compiled_program_not_a_blocker",
        "tests/python/architecture/test_program_only_temporal_facades.py"
        "::test_ssprk_semantics_have_only_typed_python_program_authority",
    }


def test_m2_transaction_matrix_retry_and_mpi_proofs_use_closed_selectors():
    data, errors = _load_runner().validate_manifest(MANIFEST)
    assert not errors
    checks = {
        (row["kind"], row["polarity"], row["target"], row.get("nodeid", row.get("test_regex")))
        for row in data["check"]
        if row["issue"] == "ADC-666" and row["requirement"] == "step_transaction"
    }
    assert checks == {
        (
            "ctest",
            "positive",
            "test_system_transaction_authority",
            r"^SystemTransactionAuthority\.RejectRestoresProgramImageAndRetryCommitsOnce$",
        ),
        (
            "ctest",
            "refusal",
            "test_system_transaction_authority",
            r"^SystemTransactionAuthority\.PublicReaderCannotReenterUniformCandidate$",
        ),
        (
            "ctest",
            "positive",
            "test_system_transaction_authority",
            r"^SystemTransactionAuthority\.ExternalAcceptedWindowRemainsComposableWithInternalStep$",
        ),
        (
            "ctest",
            "positive",
            "test_system_transaction_authority",
            r"^SystemTransactionAuthority\.ReusedImageDoesNotAllocateFabStorageAfterWarmup$",
        ),
        (
            "pytest",
            "refusal",
            "transaction",
            "tests/python/unit/runtime/test_step_transaction_runtime.py::"
            "test_fault_injection_matrix_restores_every_available_store_and_reports_exact_phase",
        ),
        (
            "pytest",
            "positive",
            "transaction",
            "tests/python/unit/runtime/test_step_transaction_runtime.py::"
            "test_success_commits_native_clock_cursors_and_attempt_counter_together",
        ),
        (
            "pytest",
            "positive",
            "transaction",
            "tests/python/unit/runtime/test_step_transaction_runtime.py::"
            "test_error_controlled_retry_rolls_back_before_opening_the_next_attempt",
        ),
        (
            "pytest",
            "refusal",
            "transaction",
            "tests/python/unit/runtime/test_step_transaction_runtime.py::"
            "test_error_controlled_retry_exhaustion_restores_the_last_accepted_boundary",
        ),
        (
            "ctest",
            "positive",
            "test_mpi_cell_temporal_program",
            r"^test_mpi_cell_temporal_program_np2$",
        ),
        (
            "ctest",
            "positive",
            "test_mpi_cell_temporal_program_multibox",
            r"^test_mpi_cell_temporal_program_multibox_np2$",
        ),
        (
            "ctest",
            "positive",
            "test_mpi_cell_temporal_program_collective_rollback",
            r"^test_mpi_cell_temporal_program_collective_rollback_np2$",
        ),
        (
            "ctest",
            "refusal",
            "test_mpi_cell_temporal_program_refusal",
            r"^test_mpi_cell_temporal_program_refusal_np2$",
        ),
        (
            "ctest", "positive", "test_program_transaction",
            r"^ProgramTransaction\.BindFreezesTypedOrderBudgetsAndHotCapacity$",
        ),
        (
            "ctest", "refusal", "test_program_transaction",
            r"^ProgramTransaction\.SnapshotFailureDoesNotTouchAcceptedState$",
        ),
        (
            "ctest", "refusal", "test_program_transaction",
            r"^ProgramTransaction\.CandidateAndPrepareFaultsLeaveGenerationUnchanged$",
        ),
        (
            "ctest", "refusal", "test_program_transaction",
            r"^ProgramTransaction\.SolveGuardAndAtomicSealConsensusFaultsRollback$",
        ),
        (
            "ctest", "refusal", "test_program_transaction",
            r"^ProgramTransaction\.HiddenPublishParticipantFailureRestoresAllPublishedState$",
        ),
        (
            "ctest", "refusal", "test_program_transaction",
            r"^ProgramTransaction\.PublishFaultCompensatesEffectsInReverseOrderExactlyOnce$",
        ),
        (
            "ctest", "positive", "test_program_transaction",
            r"^ProgramTransaction\.ReaderSeesOldGenerationAndNewReadersBlockUntilSeal$",
        ),
        (
            "ctest", "refusal", "test_program_transaction",
            r"^ProgramTransaction\.SealIsAtomicAndFinalizeFailureIsFailStopWithoutScientificRollback$",
        ),
        (
            "ctest", "positive", "test_program_transaction",
            r"^ProgramTransaction\.SuccessfulEffectsAndReceiptsAreExactOnce$",
        ),
        (
            "ctest", "positive", "test_program_transaction",
            r"^ProgramTransaction\.FrozenHotPathDoesNotRecordPoPSAllocations$",
        ),
    }
