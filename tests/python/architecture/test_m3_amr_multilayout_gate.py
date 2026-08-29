"""Source-only integrity checks for the executable M3 AMR/multi-layout gate."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "tests/gates/m3_amr_multilayout.toml"
RUNNER = ROOT / "scripts/run_m3_gate.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("pops_run_m3_gate", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_m3_manifest_references_only_real_mandatory_proofs():
    data, errors = _load_runner().validate_manifest(MANIFEST)
    assert not errors, "M3 gate matrix is incomplete:\n  " + "\n  ".join(errors)
    assert len(data["check"]) == 57


def test_m3_gate_pins_exact_ranked_history_publication_and_rollback_proofs():
    data, errors = _load_runner().validate_manifest(MANIFEST)
    assert not errors
    expected = {
        "RetainsAndInterpolatesExactRankedState": "positive",
        "RejectedFacadeAttemptRestoresTopologyStateHistoryAndClock": "positive",
        "RegisteredHistoryRejectsTopologyPublicationBeforeMutation": "refusal",
    }
    for case, polarity in expected.items():
        assert {
            "issue": "ADC-678",
            "requirement": "accepted_state",
            "polarity": polarity,
            "kind": "ctest",
            "target": "test_amr_history_ring",
            "test_regex": "^test_amr_history_ring\\.%s$" % case,
        } in data["check"]

    source = (ROOT / "tests/cpp/integration/amr/test_amr_history_ring.cpp").read_text(
        encoding="utf-8"
    )
    for case in expected:
        assert "TEST(test_amr_history_ring, %s)" % case in source


def test_m3_gate_pins_exact_ranked_partitioned_transfer_proof():
    data, errors = _load_runner().validate_manifest(MANIFEST)
    assert not errors
    assert {
        "issue": "ADC-678",
        "requirement": "accepted_state",
        "polarity": "positive",
        "kind": "ctest",
        "target": "test_mpi_amr_distributed_coarse",
        "test_regex": "^test_mpi_amr_distributed_coarse_rank_parity$",
    } in data["check"]

    source = (ROOT / "tests/cpp/integration/mpi/test_mpi_amr_distributed_coarse.cpp").read_text(
        encoding="utf-8"
    )
    assert "prove_partitioned_transfers<pops::kNativeDimension>" in source
    assert "TransferKind::LinearProlongation" in source
    assert "TransferKind::ConservativeRestriction" in source
    assert "TransferKind::CoarseFineGhostInterpolation" in source
    assert '"exact-ranked-partitioned-transfers\\n"' in source


def test_m3_gate_pins_exact_ranked_temporal_accepted_image_proof():
    data, errors = _load_runner().validate_manifest(MANIFEST)
    assert not errors
    assert {
        "issue": "ADC-678",
        "requirement": "accepted_state",
        "polarity": "positive",
        "kind": "ctest",
        "target": "test_temporal_partition_restart",
        "test_regex": (
            "^test_temporal_partition_restart\\.AcceptedImageIsCanonicalInOneTwoAndThreeDimensions$"
        ),
    } in data["check"]

    source = ROOT / "tests/cpp/integration/amr/test_temporal_partition_restart.cpp"
    assert (
        "TEST(test_temporal_partition_restart, AcceptedImageIsCanonicalInOneTwoAndThreeDimensions)"
    ) in source.read_text(encoding="utf-8")


def test_m3_adc677_clocks_reflux_history_rollback_and_retry_use_closed_proofs():
    data, errors = _load_runner().validate_manifest(MANIFEST)
    assert not errors
    checks = {
        (row["kind"], row["polarity"], row["target"], row.get("nodeid", row.get("test_regex")))
        for row in data["check"]
        if row["issue"] == "ADC-677" and row["requirement"] == "clocks_reflux"
    }
    assert checks == {
        (
            "ctest",
            "positive",
            "test_program_reflux_ledger",
            r"^test_program_reflux_ledger\.CanonicalMetricLedgerAndAcceptedCheckpointAreExactInOneTwoAndThreeDimensions$",
        ),
        (
            "ctest",
            "refusal",
            "test_program_reflux_ledger",
            r"^test_program_reflux_ledger\.InvalidCheckpointAndDuplicateFacesRejectBeforeMutation$",
        ),
        (
            "ctest",
            "positive",
            "test_mpi_amr_program_3d_corner_authority",
            r"^test_mpi_amr_program_3d_corner_authority_np8$",
        ),
        (
            "ctest",
            "positive",
            "test_mpi_amr_program_reflux",
            r"^test_mpi_amr_program_reflux_np2$",
        ),
        (
            "ctest",
            "positive",
            "test_mpi_amr_program_reflux",
            r"^test_mpi_amr_program_reflux_np4$",
        ),
        (
            "ctest",
            "positive",
            "test_nd_flux_ledger",
            r"^test_nd_flux_ledger\.composite_reflux_conserves_accepted_transport_in_1d_2d_3d$",
        ),
        (
            "ctest",
            "positive",
            "test_nd_flux_ledger",
            r"^test_nd_flux_ledger\.exact_stage_weights_are_applied_before_metric_reflux$",
        ),
        (
            "ctest",
            "positive",
            "test_nd_flux_ledger",
            r"^test_nd_flux_ledger\.coarse_window_matches_two_exact_fine_substeps$",
        ),
        (
            "ctest",
            "refusal",
            "test_nd_flux_ledger",
            r"^test_nd_flux_ledger\.rejected_attempt_never_publishes_pending_faces$",
        ),
        (
            "ctest",
            "refusal",
            "test_nd_flux_ledger",
            r"^test_nd_flux_ledger\.failed_commit_preserves_accepted_and_pending_transactions$",
        ),
        (
            "pytest",
            "positive",
            "accepted_state",
            "tests/python/unit/time/test_time_codegen.py::"
            "test_canonical_ssprk_amr_codegen_preserves_exact_distinct_ledger_weights",
        ),
        (
            "pytest",
            "positive",
            "accepted_state",
            "tests/python/integration/amr/test_amr_program_reflux.py::"
            "test_multilevel_ssprk2_conserves_to_roundoff",
        ),
        (
            "pytest",
            "positive",
            "accepted_state",
            "tests/python/integration/amr/test_amr_program_reflux.py::"
            "test_multilevel_midpoint_conserves_and_differs",
        ),
        (
            "pytest",
            "positive",
            "accepted_state",
            "tests/python/integration/amr/test_amr_rational_hierarchy_program.py::"
            "test_public_generated_rational_three_level_two_block_program",
        ),
        (
            "pytest",
            "refusal",
            "accepted_state",
            "tests/python/integration/amr/test_amr_rational_hierarchy_program.py::"
            "test_public_rational_hierarchy_rejects_implicit_fractional_remainder_before_mutation",
        ),
        (
            "pytest",
            "positive",
            "restart_hierarchy_policy",
            "tests/python/integration/io/test_amr_history_reflux_restart.py::"
            "test_strict_restart_preserves_ab2_history_flux_and_reflux_continuation",
        ),
        (
            "pytest",
            "refusal",
            "restart_hierarchy_policy",
            "tests/python/integration/io/test_amr_history_regrid_replay.py::"
            "test_d_corrupted_fingerprint_refused",
        ),
    }


def test_m3_gate_pins_public_rational_three_level_two_block_program_proof():
    data, errors = _load_runner().validate_manifest(MANIFEST)
    assert not errors
    source = (
        ROOT / "tests/python/integration/amr/test_amr_rational_hierarchy_program.py"
    ).read_text(encoding="utf-8")
    expected = {
        "issue": "ADC-677",
        "requirement": "clocks_reflux",
        "polarity": "positive",
        "kind": "pytest",
        "target": "accepted_state",
        "nodeid": (
            "tests/python/integration/amr/test_amr_rational_hierarchy_program.py::"
            "test_public_generated_rational_three_level_two_block_program"
        ),
    }
    assert expected in data["check"]
    refusal = dict(expected)
    refusal["polarity"] = "refusal"
    refusal["nodeid"] = (
        "tests/python/integration/amr/test_amr_rational_hierarchy_program.py::"
        "test_public_rational_hierarchy_rejects_implicit_fractional_remainder_before_mutation"
    )
    assert refusal in data["check"]
    for needle in (
        "pops.validate(case)",
        "pops.resolve(",
        "pops.compile(resolved)",
        "pops.bind(",
        "AMRHierarchy(max_levels=3, ratios=(2, 2))",
        "Fraction(5, 2)",
        "AMRRemainderPolicy.EXPLICIT_FINAL_SUBSTEP",
        "StepAttemptRejected",
        "transaction_stats",
    ):
        assert needle in source


def test_m3_gate_pins_qualified_field_warm_start_restart_and_rollback():
    data, errors = _load_runner().validate_manifest(MANIFEST)
    assert not errors
    nodeid = (
        "tests/python/integration/amr/test_amr_composite_field_carrier.py::"
        "test_fac_overrides_propagate_through_a_refined_final_root_lifecycle"
    )
    assert {
        "issue": "ADC-678",
        "requirement": "accepted_state",
        "polarity": "positive",
        "kind": "pytest",
        "target": "accepted_state",
        "nodeid": nodeid,
    } in data["check"]

    source = (ROOT / "tests/python/integration/amr/test_amr_composite_field_carrier.py").read_text(
        encoding="utf-8"
    )
    assert "accepted_warm_starts = _field_warm_starts(simulation, slot)" in source
    assert "resolved = _resolve(solver, strict_restart=True)" in source
    assert "restarted.restart(checkpoint)" in source
    assert "injected post-field-restore validation failure" in source
    assert "np.testing.assert_array_equal(actual, expected)" in source


def test_m3_gate_pins_transactional_persistent_hysteresis_proofs():
    data, errors = _load_runner().validate_manifest(MANIFEST)
    assert not errors
    checks = data["check"]
    assert {
        "issue": "ADC-678",
        "requirement": "accepted_state",
        "polarity": "positive",
        "kind": "pytest",
        "target": "accepted_state",
        "nodeid": (
            "tests/python/integration/amr/test_amr_regrid_on_restart.py::"
            "test_regrid_on_restart_changes_real_boxes_and_rolls_back_post_regrid_fault"
        ),
    } in checks
    assert {
        "issue": "ADC-678",
        "requirement": "restart_hierarchy_policy",
        "polarity": "positive",
        "kind": "pytest",
        "target": "restart_hierarchy_policy",
        "nodeid": (
            "tests/python/unit/runtime/test_amr_checkpoint_contract.py::"
            "test_regridded_contract_authenticates_transformed_topology_and_level_axes"
        ),
    } in checks
    assert {
        "issue": "ADC-678",
        "requirement": "accepted_state",
        "polarity": "refusal",
        "kind": "pytest",
        "target": "accepted_state",
        "nodeid": (
            "tests/python/unit/amr/test_external_amr_providers.py::"
            "test_external_tagger_requires_exact_candidate_program_capability"
        ),
    } in checks
    assert {
        "issue": "ADC-678",
        "requirement": "accepted_state",
        "polarity": "positive",
        "kind": "ctest",
        "target": "test_amr_seed_no_refine",
        "test_regex": (
            "^test_amr_seed_no_refine\\."
            "OneHierarchySweepAgesHysteresisOnceAndCheckpointRestoreIsTransactional$"
        ),
    } in checks

    provider_source = (ROOT / "tests/python/unit/amr/test_external_amr_providers.py").read_text(
        encoding="utf-8"
    )
    assert "external AMR Tagger persistent_hysteresis is not implemented" in provider_source
    restart_source = (
        ROOT / "tests/python/integration/amr/test_amr_regrid_on_restart.py"
    ).read_text(encoding="utf-8")
    assert "HYSTERESIS_CYCLES = 4" in restart_source
    assert "transformed_cycle == source_cycle + 1" in restart_source
    assert "_assert_same_accepted_image(restarted, rollback_image)" in restart_source
    assert "_runtime_tagging_hysteresis(restarted) == transformed_hysteresis[0]" in restart_source
    native_source = (ROOT / "tests/cpp/integration/amr/test_amr_seed_no_refine.cpp").read_text(
        encoding="utf-8"
    )
    assert "OneHierarchySweepAgesHysteresisOnceAndCheckpointRestoreIsTransactional" in native_source
    assert (
        "all parent levels in one hierarchy sweep must share one hysteresis cycle" in native_source
    )
    assert "restored.restore_checkpoint_accepted_state(accepted)" in native_source
    assert "EXPECT_THROW(restored.restore_checkpoint_accepted_state(invalid)" in native_source


def test_m3_final_gate_has_no_deferred_requirement():
    data, errors = _load_runner().validate_manifest(MANIFEST)
    assert not errors
    assert data["deferred"] == []
    assert {row["issue"] for row in data["check"]} == {
        "ADC-672",
        "ADC-673",
        "ADC-674",
        "ADC-675",
        "ADC-676",
        "ADC-677",
        "ADC-678",
    }


def test_m3_gate_requires_native_positive_and_refusal_proofs_for_every_issue(tmp_path):
    data = MANIFEST.read_text(encoding="utf-8")
    data = data.replace(
        'test_regex = "^test_mpi_system_layout_transfer_np2$"',
        'test_regex = "^test_mpi_system_layout_transfer_np2_removed$"',
        1,
    )
    data = data.replace(
        'polarity = "positive"\nkind = "ctest"\ntarget = "test_mpi_system_layout_transfer"',
        'polarity = "refusal"\nkind = "ctest"\ntarget = "test_mpi_system_layout_transfer"',
        1,
    )
    manifest = tmp_path / "m3.toml"
    manifest.write_text(data, encoding="utf-8")

    _, errors = _load_runner().validate_manifest(manifest)

    assert any("ADC-673 lacks a mandatory native positive proof" in error for error in errors)


def test_m3_gate_rejects_a_missing_or_non_exact_ctest_case_before_build(tmp_path):
    data = MANIFEST.read_text(encoding="utf-8")
    data = data.replace(
        ('test_regex = "^test_amr_history_ring\\\\.RetainsAndInterpolatesExactRankedState$"'),
        'test_regex = "^test_amr_history_ring\\\\.DefinitelyMissingProof$"',
        1,
    )
    manifest = tmp_path / "m3.toml"
    manifest.write_text(data, encoding="utf-8")

    _, errors = _load_runner().validate_manifest(manifest)

    assert any(
        "is not one exact source-registered case for target 'test_amr_history_ring'" in error
        for error in errors
    )

    data = data.replace(
        'test_regex = "^test_amr_history_ring\\\\.DefinitelyMissingProof$"',
        'test_regex = "^test_amr_history_ring\\\\..*$"',
        1,
    )
    manifest.write_text(data, encoding="utf-8")
    _, errors = _load_runner().validate_manifest(manifest)
    assert any(
        "is not one exact source-registered case for target 'test_amr_history_ring'" in error
        for error in errors
    )


def test_m3_gate_does_not_authenticate_commented_or_stringified_gtests():
    source = r"""
TEST(RealFixture, ExecutedProof) {}
// TEST(CommentFixture, NotAProof) {}
/* TEST(BlockCommentFixture, NotAProofEither) {} */
const char* text = "TEST(StringFixture, StillNotAProof)";
const char* raw = R"cpp(TEST(RawStringFixture, StillNotAProof))cpp";
"""

    assert _load_runner()._registered_gtest_cases(source) == {
        "RealFixture.ExecutedProof",
    }


def test_m3_mpi_python_proof_is_exact_and_manifest_owned(monkeypatch):
    runner = _load_runner()
    data, errors = runner.validate_manifest(MANIFEST)
    assert not errors
    checks = data["check"]
    mpi_python = next(check for check in checks if check["kind"] == "mpi_python")
    assert mpi_python == {
        "issue": "ADC-678",
        "requirement": "accepted_state",
        "polarity": "positive",
        "kind": "mpi_python",
        "target": "accepted_state",
        "nodeid": (
            "tests/python/integration/mpi/test_amr_history_mpi.py::"
            "test_amr_history_mpi_in_window_regrid_public_restart_and_distribution_parity"
        ),
        "nproc": 2,
    }
    assert {
        "issue": "ADC-678",
        "requirement": "restart_hierarchy_policy",
        "polarity": "positive",
        "kind": "mpi_python",
        "target": "restart_hierarchy_policy",
        "nodeid": (
            "tests/python/integration/mpi/test_amr_regrid_on_restart_mpi.py::"
            "test_regrid_on_restart_mpi_collective_rollback_and_lineage"
        ),
        "nproc": 2,
    } in checks
    entrypoint = ROOT / "tests/python/integration/mpi/test_amr_regrid_on_restart_mpi.py"
    tree = ast.parse(entrypoint.read_text(encoding="utf-8"), filename=str(entrypoint))
    run_all = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_run_all"
    )
    direct_calls = {
        node.func.id
        for node in ast.walk(run_all)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert {
        "test_regrid_on_restart_mpi_collective_rollback_and_lineage",
        "test_regrid_on_restart_mpi_shared_interface_transaction",
    } <= direct_calls
    assert {
        "issue": "ADC-678",
        "requirement": "restart_hierarchy_policy",
        "polarity": "positive",
        "kind": "mpi_python",
        "target": "restart_hierarchy_policy",
        "nodeid": (
            "tests/python/integration/mpi/test_amr_regrid_on_restart_mpi.py::"
            "test_regrid_on_restart_mpi_shared_interface_transaction"
        ),
        "nproc": 2,
    } in checks
    restart_mpi_source = (
        ROOT / "tests/python/integration/mpi/test_amr_regrid_on_restart_mpi.py"
    ).read_text(encoding="utf-8")
    assert "injected rank-local pre-collective validation failure" in restart_mpi_source
    assert "all(allgather_value(_COMM, caught))" in restart_mpi_source
    assert "_restart_accepted_contract_identity" in restart_mpi_source
    assert 'receipt["history_consensus_identity_before"]' in restart_mpi_source
    assert "both AB2 histories are conservatively rematerialized" in restart_mpi_source
    program_runtime = (ROOT / "include/pops/runtime/program/program_runtime_state.hpp").read_text(
        encoding="utf-8"
    )
    assert "RegridOnRestart requires an authenticated artifact-backed Program" in program_runtime
    assert "artifact lacks its restart preflight/regrid/resync hooks" in program_runtime
    assert "supports shared-interface flux groups only in serial" not in program_runtime
    assert {
        "issue": "ADC-678",
        "requirement": "accepted_state",
        "polarity": "positive",
        "kind": "pytest",
        "target": "accepted_state",
        "nodeid": (
            "tests/python/integration/runtime/test_multi_layout_runtime.py::"
            "test_multi_layout_checkpoint_restart_restores_every_layout_and_mapping_count"
        ),
    } in checks
    assert runner._python_mpi_orchestrators() == {
        "tests/python/integration/mpi/test_amr_rank_change_restart.py"
    }
    assert {
        "issue": "ADC-678",
        "requirement": "accepted_state",
        "polarity": "positive",
        "kind": "pytest",
        "target": "accepted_state",
        "nodeid": (
            "tests/python/integration/mpi/test_amr_rank_change_restart.py::"
            "test_amr_checkpoint_restart_rematerializes_two_ranks_onto_one"
        ),
    } in checks
    assert {
        "issue": "ADC-678",
        "requirement": "accepted_state",
        "polarity": "positive",
        "kind": "ctest",
        "target": "test_mpi_amr_twoblock_parity",
        "test_regex": "^test_mpi_amr_twoblock_parity_np2$",
    } in checks

    monkeypatch.setattr(runner, "_python_mpi_entrypoints", lambda: {})
    _, errors = runner.validate_manifest(MANIFEST)
    assert any(
        "test_amr_history_mpi.py is not a manifest-owned MPI Python entrypoint" in error
        for error in errors
    )

    monkeypatch.setattr(runner, "_python_mpi_orchestrators", lambda: set())
    _, errors = runner.validate_manifest(MANIFEST)
    assert any(
        "test_amr_rank_change_restart.py is not a manifest-owned serial MPI orchestrator" in error
        for error in errors
    )


def test_m3_mpi_python_launch_is_explicit_required_and_check_only_safe(monkeypatch):
    runner = _load_runner()
    relative = "tests/python/integration/mpi/test_amr_history_mpi.py"

    def launcher_or_fail(command):
        if command == "mpiexec":
            return "/opt/mpi/bin/mpiexec"
        raise AssertionError("check-only validation consulted the MPI launcher")

    monkeypatch.setattr(runner.shutil, "which", launcher_or_fail)
    assert runner.main(["--check-only"]) == 0
    _, errors = runner.validate_manifest(MANIFEST)
    assert not errors
    assert runner._mpi_python_command("mpiexec", 2, relative) == [
        "mpiexec",
        "-n",
        "2",
        sys.executable,
        str(ROOT / relative),
    ]
    monkeypatch.setenv("POPS_REQUIRE_MPI_TESTS", "0")
    assert runner._required_mpi_environment()["POPS_REQUIRE_MPI_TESTS"] == "1"
    monkeypatch.setenv("POPS_REQUIRE_NATIVE_TESTS", "0")
    assert runner._required_mpi_environment()["POPS_REQUIRE_NATIVE_TESTS"] == "1"

    monkeypatch.setattr(runner.shutil, "which", lambda _command: None)
    with pytest.raises(RuntimeError, match="required MPI launcher"):
        runner._mpi_python_command("mpiexec", 2, relative)


def test_m3_required_pytest_execution_rejects_every_skip_or_xfail(tmp_path, monkeypatch):
    runner = _load_runner()
    report = tmp_path / "pytest.xml"
    skipped_xml = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<testsuites><testsuite tests="1" skipped="1">'
        '<testcase classname="m3" name="proof"><skipped type="pytest.skip"/></testcase>'
        "</testsuite></testsuites>"
    )
    report.write_text(skipped_xml, encoding="utf-8")
    assert runner._pytest_skip_count(report) == 1

    def successful_pytest_with_a_skip(command, *, cwd, env, check):
        assert cwd == ROOT
        assert env["POPS_REQUIRE_MPI_TESTS"] == "1"
        assert env["POPS_REQUIRE_NATIVE_TESTS"] == "1"
        assert check is False
        assert "xfail_strict=true" in command
        junit = Path(command[command.index("--junitxml") + 1])
        junit.write_text(skipped_xml, encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", successful_pytest_with_a_skip)
    with pytest.raises(RuntimeError, match="reported 1 skipped/xfail proof"):
        runner._run_required_pytest(
            ["tests/python/unit/runtime/test_amr_checkpoint_contract.py::proof"]
        )
