"""Source-only integrity checks for the bounded ADC-757 numerical gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "tests/gates/adc757_prepared_numerics.toml"
RUNNER = ROOT / "scripts/run_adc757_prepared_numerics_gate.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("pops_run_adc757_gate", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_adc757_slice_references_exact_real_mandatory_native_proofs():
    runner = _load_runner()
    data, errors = runner.validate_manifest(MANIFEST)
    assert not errors, "ADC-757 slice matrix is invalid:\n  " + "\n  ".join(errors)
    identities = {
        (
            row.get("kind", "ctest"),
            row.get("target", row.get("path")),
            row.get("test_regex", row.get("test")),
        )
        for row in data["check"]
    }
    assert len(identities) == len(data["check"])
    assert {row["requirement"] for row in data["check"]} == (
        runner.EXPECTED_REQUIREMENTS | runner.EXPECTED_UNAVAILABLE_REQUIREMENTS
    )
    assert data["evidence_from"] == [
        "ADC-682",
        "ADC-711",
        "ADC-733",
        "ADC-737",
        "ADC-749",
        "ADC-750",
        "ADC-751",
        "ADC-752",
        "ADC-753",
        "ADC-754",
        "ADC-755",
        "ADC-756",
    ]
    assert runner.main(["--check-only"]) == 0


def test_adc757_slice_executes_runtime_recovery_publication_proofs():
    runner = _load_runner()
    data, errors = runner.validate_manifest(MANIFEST)
    assert not errors
    claimed = {
        "amr_bootstrap_recovery_publication",
        "amr_history_recovery_publication",
        "physical_boundary_trace_recovery_publication",
        "terminal_source_recovery_publication",
    }
    rows = [row for row in data["check"] if row["requirement"] in claimed]
    assert {row["requirement"] for row in rows} == claimed
    assert {(row["requirement"], row["polarity"]) for row in rows} == {
        (requirement, polarity) for requirement in claimed for polarity in ("positive", "refusal")
    }


def test_adc757_slice_executes_new_prepared_recovery_policy_proofs():
    runner = _load_runner()
    data, errors = runner.validate_manifest(MANIFEST)
    assert not errors
    claimed = {
        "prepared_riemann_recovery_policy",
        "uniform_recovery_warm_start",
    }
    rows = [row for row in data["check"] if row["requirement"] in claimed]
    assert {row["requirement"] for row in rows} == claimed
    assert {(row["requirement"], row["polarity"]) for row in rows} == {
        (requirement, polarity) for requirement in claimed for polarity in ("positive", "refusal")
    }


def test_adc757_slice_executes_qualified_flux_provider_pack_proofs():
    runner = _load_runner()
    data, errors = runner.validate_manifest(MANIFEST)
    assert not errors
    assert [
        row for row in data["check"] if row["requirement"] == "qualified_flux_provider_pack"
    ] == [
        {
            "requirement": "qualified_flux_provider_pack",
            "polarity": "positive",
            "target": "test_flux_interfaces",
            "test_regex": "^FluxProviders\\.ExactDensePackBindsOnlyDeclaredSlots$",
        },
        {
            "requirement": "qualified_flux_provider_pack",
            "polarity": "refusal",
            "kind": "pytest",
            "path": "tests/python/unit/codegen/test_compiler_model_provider.py",
            "test": "test_incomplete_or_false_compiler_provider_is_rejected_before_compile",
        },
    ]


def test_adc757_slice_executes_post_riemann_boundary_flux_proofs():
    runner = _load_runner()
    data, errors = runner.validate_manifest(MANIFEST)
    assert not errors
    assert [row for row in data["check"] if row["requirement"] == "post_riemann_boundary_flux"] == [
        {
            "requirement": "post_riemann_boundary_flux",
            "polarity": "positive",
            "target": "test_prepared_hyperbolic_boundary",
            "test_regex": "^test_prepared_hyperbolic_boundary\\."
            "no_flux_is_enforced_on_the_post_riemann_face_field$",
        },
        {
            "requirement": "post_riemann_boundary_flux",
            "polarity": "refusal",
            "kind": "pytest",
            "path": "tests/python/unit/mesh/test_boundary_topology_ports.py",
            "test": "test_post_riemann_flux_refuses_wrong_component_route_or_output",
        },
    ]


def test_adc757_slice_separates_mpi_executables_from_authenticated_hardware_proofs():
    runner = _load_runner()
    data, errors = runner.validate_manifest(MANIFEST)
    assert not errors
    assert data["deferred"] == list(runner.EXPECTED_DEFERRED)
    assert "mpi_collective_execution" not in data["deferred"]
    assert "gpu_backend_execution" not in data["deferred"]
    assert "accelerator_stream_partitioning" not in data["deferred"]
    assert "performance_baselines_and_end_to_end_benchmarks" not in data["deferred"]
    assert "workspace_reentrancy_and_stream_partitioning" not in data["deferred"]
    assert "remaining_legacy_recovery_and_boundary_authority_deletion" not in data["deferred"]
    assert all("riemann_authority" not in family for family in data["deferred"])
    assert "runtime_consumer_cutover_and_legacy_deletion" not in data["deferred"]
    assert "boundary_geometry_riemann_and_spatial_provider_families" not in data["deferred"]
    assert "amr_regrid_migration_and_restart_coherence" not in data["deferred"]
    assert (
        "remaining_local_time_migration_and_load_balance_runtime_integration"
        not in data["deferred"]
    )
    assert "bounded_cell_local_program_runtime" not in data["deferred"]
    assert "standalone_exact_ranked_cell_temporal_provider" not in data["deferred"]
    assert "remaining_multirank_multibox_amr_local_time_execution" not in data["deferred"]
    assert [row for row in data["check"] if row.get("kind") == "mpi_ctest"] == [
        {
            "requirement": "mpi_collective_execution",
            "polarity": "positive",
            "kind": "mpi_ctest",
            "target": "test_mpi_system_analytic_level_set",
            "test_regex": "^test_mpi_system_analytic_level_set_np2$",
            "nproc": 2,
        },
        {
            "requirement": "mpi_collective_execution",
            "polarity": "refusal",
            "kind": "mpi_ctest",
            "target": "test_mpi_flux_failure_collective",
            "test_regex": "^test_mpi_flux_failure_collective_np2$",
            "nproc": 2,
        },
        {
            "requirement": "amr_rebalance_migration_and_restart_coherence",
            "polarity": "positive",
            "kind": "mpi_ctest",
            "target": "test_mpi_amr_rebalance_migration",
            "test_regex": "^test_mpi_amr_rebalance_migration_np2$",
            "nproc": 2,
        },
        {
            "requirement": "amr_rebalance_migration_and_restart_coherence",
            "polarity": "refusal",
            "kind": "mpi_ctest",
            "target": "test_mpi_amr_rebalance_migration",
            "test_regex": "^test_mpi_amr_rebalance_migration_np4$",
            "nproc": 4,
        },
        {
            "requirement": "bounded_cell_local_program_runtime",
            "polarity": "positive",
            "kind": "mpi_ctest",
            "target": "test_mpi_cell_temporal_program",
            "test_regex": "^test_mpi_cell_temporal_program_np2$",
            "nproc": 2,
        },
        {
            "requirement": "remaining_multirank_multibox_amr_local_time_execution",
            "polarity": "positive",
            "kind": "mpi_ctest",
            "target": "test_mpi_cell_temporal_program_multibox",
            "test_regex": "^test_mpi_cell_temporal_program_multibox_np2$",
            "nproc": 2,
        },
        {
            "requirement": "remaining_multirank_multibox_amr_local_time_execution",
            "polarity": "refusal",
            "kind": "mpi_ctest",
            "target": "test_mpi_cell_temporal_program_collective_rollback",
            "test_regex": "^test_mpi_cell_temporal_program_collective_rollback_np2$",
            "nproc": 2,
        },
    ]
    assert data["hardware_evidence"] == runner.EXPECTED_HARDWARE_EVIDENCE
    hardware_rows = [
        row for row in data["check"] if row["requirement"] in runner.EXPECTED_HARDWARE_REQUIREMENTS
    ]
    assert {(row["requirement"], row["polarity"]) for row in hardware_rows} == {
        (requirement, "refusal") for requirement in runner.EXPECTED_HARDWARE_REQUIREMENTS
    }
    assert all(row["polarity"] != "positive" for row in hardware_rows)
    assert runner.main(["--check-only", "--closure"]) == 3


def test_adc757_slice_includes_exact_public_measured_load_balance_policy_proofs():
    runner = _load_runner()
    data, errors = runner.validate_manifest(MANIFEST)
    assert not errors
    public_rows = [
        row
        for row in data["check"]
        if row.get("kind") == "pytest" and row["requirement"] == "measured_load_balance_decision"
    ]
    assert public_rows == [
        {
            "requirement": "measured_load_balance_decision",
            "polarity": "positive",
            "kind": "pytest",
            "path": "tests/python/unit/amr/test_public_amr_resolution.py",
            "test": "test_measured_knapsack_roundtrips_exact_native_decision_policy",
        },
        {
            "requirement": "measured_load_balance_decision",
            "polarity": "refusal",
            "kind": "pytest",
            "path": "tests/python/unit/amr/test_public_amr_resolution.py",
            "test": "test_measured_knapsack_rejects_invalid_decision_policy",
        },
    ]


def test_adc757_slice_authenticates_standalone_and_installed_cell_temporal_routes():
    runner = _load_runner()
    data, errors = runner.validate_manifest(MANIFEST)
    assert not errors

    migration = [
        row
        for row in data["check"]
        if row["requirement"] == "amr_rebalance_migration_and_restart_coherence"
    ]
    assert [(row["polarity"], row.get("kind", "ctest"), row["target"]) for row in migration] == [
        ("positive", "mpi_ctest", "test_mpi_amr_rebalance_migration"),
        ("refusal", "mpi_ctest", "test_mpi_amr_rebalance_migration"),
        ("positive", "ctest", "test_program_reflux_ledger"),
        ("refusal", "ctest", "test_program_reflux_ledger"),
    ]

    standalone = [
        row
        for row in data["check"]
        if row["requirement"] == "standalone_exact_ranked_cell_temporal_provider"
    ]
    assert [(row["polarity"], row.get("kind", "ctest")) for row in standalone] == [
        ("positive", "ctest"),
        ("refusal", "ctest"),
        ("positive", "pytest"),
        ("refusal", "pytest"),
    ]
    runtime = [
        row for row in data["check"] if row["requirement"] == "bounded_cell_local_program_runtime"
    ]
    assert runtime == [
        {
            "requirement": "bounded_cell_local_program_runtime",
            "polarity": "positive",
            "kind": "pytest",
            "path": "tests/python/unit/codegen/test_cell_local_time_codegen.py",
            "test": "test_amr_cell_local_driver_uses_the_distributed_context_provider",
        },
        {
            "requirement": "bounded_cell_local_program_runtime",
            "polarity": "positive",
            "kind": "mpi_ctest",
            "target": "test_mpi_cell_temporal_program",
            "test_regex": "^test_mpi_cell_temporal_program_np2$",
            "nproc": 2,
        },
        {
            "requirement": "bounded_cell_local_program_runtime",
            "polarity": "refusal",
            "kind": "pytest",
            "path": "tests/python/unit/codegen/test_cell_local_time_codegen.py",
            "test": "test_cell_local_codegen_refuses_uniform_target",
        },
    ]
    multirank = [
        row
        for row in data["check"]
        if row["requirement"] == "remaining_multirank_multibox_amr_local_time_execution"
    ]
    assert [(row["polarity"], row["target"], row["nproc"]) for row in multirank] == [
        ("positive", "test_mpi_cell_temporal_program_multibox", 2),
        ("refusal", "test_mpi_cell_temporal_program_collective_rollback", 2),
    ]
    assert "bounded_cell_local_program_runtime" not in data["deferred"]
    assert "remaining_multirank_multibox_amr_local_time_execution" not in data["deferred"]


def test_adc757_slice_authenticates_the_only_ranked_transport_boundary_authority():
    runner = _load_runner()
    data, errors = runner.validate_manifest(MANIFEST)
    assert not errors
    claimed = {
        "prepared_hyperbolic_boundary_only_transport_authority",
        "polar_runtime_capability_honesty",
    }
    assert [row for row in data["check"] if row["requirement"] in claimed] == [
        {
            "requirement": "prepared_hyperbolic_boundary_only_transport_authority",
            "polarity": "positive",
            "kind": "pytest",
            "path": "tests/python/architecture/test_hyperbolic_boundary_authority_ratchet.py",
            "test": "test_prepared_hyperbolic_boundary_is_the_only_native_transport_authority",
        },
        {
            "requirement": "prepared_hyperbolic_boundary_only_transport_authority",
            "polarity": "refusal",
            "kind": "pytest",
            "path": "tests/python/architecture/test_hyperbolic_boundary_authority_ratchet.py",
            "test": "test_legacy_transport_boundary_authorities_are_deleted",
        },
        {
            "requirement": "polar_runtime_capability_honesty",
            "polarity": "positive",
            "kind": "pytest",
            "path": "tests/python/architecture/test_program_only_temporal_facades.py",
            "test": "test_standalone_polar_elliptic_algorithms_remain_explicit",
        },
        {
            "requirement": "polar_runtime_capability_honesty",
            "polarity": "refusal",
            "kind": "pytest",
            "path": "tests/python/architecture/test_program_only_temporal_facades.py",
            "test": "test_polar_runtime_builder_is_retired_until_an_exact_ranked_metric_provider_exists",
        },
    ]


def test_adc757_slice_authenticates_prepared_batch_as_the_only_recovery_authority():
    runner = _load_runner()
    data, errors = runner.validate_manifest(MANIFEST)
    assert not errors
    assert [
        row
        for row in data["check"]
        if row["requirement"] == "prepared_batch_recovery_only_runtime_authority"
    ] == [
        {
            "requirement": "prepared_batch_recovery_only_runtime_authority",
            "polarity": "positive",
            "kind": "pytest",
            "path": "tests/python/architecture/test_variable_recovery_consumer_cutover.py",
            "test": "test_runtime_materialization_consumes_only_prepared_batch_before_publication",
        },
        {
            "requirement": "prepared_batch_recovery_only_runtime_authority",
            "polarity": "refusal",
            "kind": "pytest",
            "path": "tests/python/architecture/test_variable_recovery_consumer_cutover.py",
            "test": "test_runtime_materialization_has_no_pointwise_compatibility_authority",
        },
        {
            "requirement": "prepared_batch_recovery_only_runtime_authority",
            "polarity": "refusal",
            "target": "test_facade_routing",
            "test_regex": "^FacadeRouting\\."
            "PreparedBlockInstallationRefusesMissingBatchAuthorityWithoutPublication$",
        },
    ]


def test_adc757_slice_executes_host_workspace_reentrancy_without_claiming_streams():
    runner = _load_runner()
    data, errors = runner.validate_manifest(MANIFEST)
    assert not errors
    assert [row for row in data["check"] if row["requirement"] == "host_workspace_reentrancy"] == [
        {
            "requirement": "host_workspace_reentrancy",
            "polarity": "positive",
            "target": "test_krylov_workspace_reentrancy",
            "test_regex": "^test_krylov_workspace_reentrancy\\."
            "distinct_workspaces_run_fresh_operator_and_preconditioner_sessions_concurrently$",
        },
        {
            "requirement": "host_workspace_reentrancy",
            "polarity": "refusal",
            "target": "test_krylov_workspace_reentrancy",
            "test_regex": "^test_krylov_workspace_reentrancy\\."
            "workspace_rebind_reserves_mutation_during_blocking_operator_prepare$",
        },
    ]


def test_adc757_slice_executes_exact_python_ir_and_restart_proofs():
    runner = _load_runner()
    data, errors = runner.validate_manifest(MANIFEST)
    assert not errors
    assert [
        row
        for row in data["check"]
        if row.get("kind") == "pytest"
        and row["requirement"] == "python_ir_generated_abi_and_restart_parity"
    ] == [
        {
            "requirement": "python_ir_generated_abi_and_restart_parity",
            "polarity": "positive",
            "kind": "pytest",
            "path": "tests/python/unit/codegen/test_recovery_admissibility_codegen.py",
            "test": "test_recovery_admissibility_is_emitted_and_hashed",
        },
        {
            "requirement": "python_ir_generated_abi_and_restart_parity",
            "polarity": "refusal",
            "kind": "pytest",
            "path": "tests/python/unit/codegen/test_recovery_admissibility_codegen.py",
            "test": "test_recovery_admissibility_rejects_ambiguous_authoring",
        },
        {
            "requirement": "python_ir_generated_abi_and_restart_parity",
            "polarity": "positive",
            "kind": "pytest",
            "path": "tests/python/unit/runtime/test_amr_checkpoint_contract.py",
            "test": "test_preflight_returns_exact_native_payload_and_counters",
        },
        {
            "requirement": "python_ir_generated_abi_and_restart_parity",
            "polarity": "refusal",
            "kind": "pytest",
            "path": "tests/python/unit/runtime/test_amr_checkpoint_contract.py",
            "test": "test_historical_version_refusal_happens_before_restart_transaction",
        },
    ]
    assert "python_ir_generated_abi_and_restart_parity" not in data["deferred"]


def test_adc757_closure_requires_revision_matched_hardware_evidence(monkeypatch, tmp_path):
    runner = _load_runner()
    monkeypatch.setattr(
        runner,
        "validate_manifest",
        lambda _manifest: (
            {
                "check": [],
                "deferred": [],
                "hardware_evidence": runner.EXPECTED_HARDWARE_EVIDENCE,
            },
            [],
        ),
    )
    assert runner.main(["--check-only", "--closure"]) == 4

    report = tmp_path / "hardware.json"
    report.write_text("{}", encoding="utf-8")
    observed = []
    monkeypatch.setattr(
        runner,
        "_run_hardware_evidence",
        lambda evidence, path, revision: (
            observed.append((evidence, path, revision)) or runner.EXPECTED_HARDWARE_REQUIREMENTS
        ),
    )
    assert (
        runner.main(
            [
                "--check-only",
                "--closure",
                "--hardware-report",
                str(report),
                "--expected-revision",
                "a" * 40,
            ]
        )
        == 0
    )
    assert observed == [(runner.EXPECTED_HARDWARE_EVIDENCE, report, "a" * 40)]


def test_adc757_hardware_evidence_requires_a_full_exact_candidate_revision(tmp_path):
    runner = _load_runner()
    report = tmp_path / "hardware.json"
    report.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="full lowercase 40-hex"):
        runner._run_hardware_evidence(
            runner.EXPECTED_HARDWARE_EVIDENCE,
            report,
            "short-revision",
        )


def test_adc757_manifest_refuses_missing_polarity_and_unknown_target(tmp_path):
    runner = _load_runner()
    source = MANIFEST.read_text(encoding="utf-8")

    missing_refusal = tmp_path / "missing_refusal.toml"
    missing_refusal.write_text(
        source.replace('polarity = "refusal"', 'polarity = "positive"', 1),
        encoding="utf-8",
    )
    _, errors = runner.validate_manifest(missing_refusal)
    assert any("lacks refusal coverage" in error for error in errors)

    unknown_target = tmp_path / "unknown_target.toml"
    unknown_target.write_text(
        source.replace(
            'target = "test_newton_robustness"',
            'target = "test_missing_provider_proof"',
            1,
        ),
        encoding="utf-8",
    )
    _, errors = runner.validate_manifest(unknown_target)
    assert any("unknown CTest target" in error for error in errors)

    wrong_mpi_rank = tmp_path / "wrong_mpi_rank.toml"
    wrong_mpi_rank.write_text(
        source.replace("nproc = 2", "nproc = 4", 1),
        encoding="utf-8",
    )
    _, errors = runner.validate_manifest(wrong_mpi_rank)
    assert any("one exact rank count" in error for error in errors)

    unknown_python_file = tmp_path / "unknown_python_file.toml"
    unknown_python_file.write_text(
        source.replace(
            'path = "tests/python/unit/codegen/test_recovery_admissibility_codegen.py"',
            'path = "tests/python/unit/codegen/test_missing_gate_proof.py"',
            1,
        ),
        encoding="utf-8",
    )
    _, errors = runner.validate_manifest(unknown_python_file)
    assert any("unknown Python test file" in error for error in errors)

    unknown_pytest = tmp_path / "unknown_pytest.toml"
    unknown_pytest.write_text(
        source.replace(
            'test = "test_recovery_admissibility_is_emitted_and_hashed"',
            'test = "test_missing_gate_proof"',
            1,
        ),
        encoding="utf-8",
    )
    _, errors = runner.validate_manifest(unknown_pytest)
    assert any("unknown top-level pytest" in error for error in errors)

    skipped = runner.ast.parse("@pytest.mark.xfail\ndef test_skipped():\n    pass\n").body[0]
    assert runner._pytest_is_skipped(skipped)

    skipped_ctest = tmp_path / "skipped_ctest.toml"
    skipped_ctest.write_text(
        source.replace(
            "^test_krylov_workspace_reentrancy\\\\."
            "distinct_workspaces_run_fresh_operator_and_preconditioner_sessions_concurrently$",
            "^test_krylov_workspace_reentrancy\\\\."
            "rank_local_problem_construction_failure_is_published_before_lane_unwind$",
            1,
        ),
        encoding="utf-8",
    )
    _, errors = runner.validate_manifest(skipped_ctest)
    assert any("selected CTest" in error and "skipped or disabled" in error for error in errors)

    duplicate_hardware = tmp_path / "duplicate_hardware.toml"
    duplicate_hardware.write_text(
        source.replace(
            '  "accelerator_stream_partitioning",\n'
            '  "performance_baselines_and_regression_thresholds",',
            '  "gpu_backend_execution",\n  "performance_baselines_and_regression_thresholds",',
            1,
        ),
        encoding="utf-8",
    )
    _, errors = runner.validate_manifest(duplicate_hardware)
    assert any("hardware_evidence requirements must be unique" in error for error in errors)

    fake_cpu_positive = tmp_path / "fake_cpu_positive.toml"
    fake_cpu_positive.write_text(
        source.replace(
            'requirement = "gpu_backend_execution"\npolarity = "refusal"',
            'requirement = "gpu_backend_execution"\npolarity = "positive"',
            1,
        ),
        encoding="utf-8",
    )
    _, errors = runner.validate_manifest(fake_cpu_positive)
    assert any(
        "hardware positive evidence must come only from hardware_evidence" in error
        for error in errors
    )


def test_adc757_runner_refuses_a_declared_but_unbuilt_proof(monkeypatch, tmp_path):
    runner = _load_runner()

    def empty_ctest_listing(command, **kwargs):
        assert command[:2] == ["ctest", "--test-dir"]
        return SimpleNamespace(returncode=0, stdout="Total Tests: 0\n")

    monkeypatch.setattr(runner.subprocess, "run", empty_ctest_listing)
    with pytest.raises(RuntimeError, match="is not built"):
        runner._run_ctest(
            tmp_path,
            "test_prepared_numerics_gate",
            r"^PreparedNumericsGate\.ConvergedPreparedPathAllocatesNothingAndRollsBack$",
        )


def test_adc757_runner_executes_one_exact_pytest(monkeypatch):
    runner = _load_runner()
    calls = []

    def capture_pytest(command, **kwargs):
        report = Path(command[command.index("--junitxml") + 1])
        report.write_text("<testsuites><testsuite/></testsuites>", encoding="utf-8")
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", capture_pytest)
    runner._run_pytest(
        "tests/python/unit/codegen/test_recovery_admissibility_codegen.py",
        "test_recovery_admissibility_is_emitted_and_hashed",
    )
    [(command, kwargs)] = calls
    assert command[:4] == [runner.sys.executable, "-m", "pytest", "-q"]
    assert command[4:8] == ["--strict-markers", "-o", "xfail_strict=true", "--junitxml"]
    assert command[-1] == (
        "tests/python/unit/codegen/test_recovery_admissibility_codegen.py::"
        "test_recovery_admissibility_is_emitted_and_hashed"
    )
    assert kwargs["cwd"] == runner.ROOT
    assert kwargs["check"] is False
    assert kwargs["env"]["POPS_REQUIRE_MPI_TESTS"] == "1"
    assert kwargs["env"]["POPS_REQUIRE_NATIVE_TESTS"] == "1"


def test_adc757_runner_refuses_a_runtime_skip(monkeypatch):
    runner = _load_runner()

    def skipped_pytest(command, **kwargs):
        report = Path(command[command.index("--junitxml") + 1])
        report.write_text(
            "<testsuites><testsuite><testcase><skipped/></testcase></testsuite></testsuites>",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", skipped_pytest)
    with pytest.raises(RuntimeError, match="skipped/xfail proof"):
        runner._run_pytest(
            "tests/python/unit/codegen/test_recovery_admissibility_codegen.py",
            "test_recovery_admissibility_is_emitted_and_hashed",
        )
