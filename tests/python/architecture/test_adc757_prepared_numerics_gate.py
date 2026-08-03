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
    assert len(data["check"]) == 55
    assert {row["requirement"] for row in data["check"]} == runner.EXPECTED_REQUIREMENTS
    assert data["evidence_from"] == [
        "ADC-682",
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
        (requirement, polarity)
        for requirement in claimed
        for polarity in ("positive", "refusal")
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
            "test_regex": "^test_flux_interfaces\\."
            "generated_provider_requirements_own_native_slot_reads$",
        },
        {
            "requirement": "qualified_flux_provider_pack",
            "polarity": "refusal",
            "kind": "pytest",
            "path": "tests/python/unit/codegen/test_compiler_model_provider.py",
            "test": "test_field_dependent_flux_without_provider_fails_before_native_source",
        },
    ]


def test_adc757_slice_executes_post_riemann_boundary_flux_proofs():
    runner = _load_runner()
    data, errors = runner.validate_manifest(MANIFEST)
    assert not errors
    assert [
        row for row in data["check"]
        if row["requirement"] == "post_riemann_boundary_flux"
    ] == [
        {
            "requirement": "post_riemann_boundary_flux",
            "polarity": "positive",
            "target": "test_amr_native_loader",
            "test_regex": "^test_amr_native_loader\\."
            "PostRiemannBoundaryFluxUsesOutwardOrientationAndPreservesCanonicalFaceStorage$",
        },
        {
            "requirement": "post_riemann_boundary_flux",
            "polarity": "refusal",
            "kind": "pytest",
            "path": "tests/python/unit/mesh/test_boundary_topology_ports.py",
            "test": "test_post_riemann_flux_refuses_wrong_component_route_or_output",
        },
    ]


def test_adc757_slice_claims_only_the_exact_delivered_mpi_collective_proof():
    runner = _load_runner()
    data, errors = runner.validate_manifest(MANIFEST)
    assert not errors
    assert data["deferred"] == list(runner.EXPECTED_DEFERRED)
    assert "mpi_collective_execution" not in data["deferred"]
    assert "gpu_backend_execution" in data["deferred"]
    assert "accelerator_stream_partitioning" in data["deferred"]
    assert "workspace_reentrancy_and_stream_partitioning" not in data["deferred"]
    assert "remaining_legacy_recovery_and_boundary_authority_deletion" in data["deferred"]
    assert all("riemann_authority" not in family for family in data["deferred"])
    assert "runtime_consumer_cutover_and_legacy_deletion" not in data["deferred"]
    assert "boundary_geometry_riemann_and_spatial_provider_families" not in data["deferred"]
    assert [
        row for row in data["check"] if row.get("kind") == "mpi_ctest"
    ] == [
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
    ]
    assert all(
        "gpu" not in row.get("target", row.get("path", "")).lower()
        for row in data["check"]
    )
    assert runner.main(["--check-only", "--closure"]) == 3


def test_adc757_slice_executes_host_workspace_reentrancy_without_claiming_streams():
    runner = _load_runner()
    data, errors = runner.validate_manifest(MANIFEST)
    assert not errors
    assert [
        row for row in data["check"] if row["requirement"] == "host_workspace_reentrancy"
    ] == [
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

    skipped = runner.ast.parse(
        "@pytest.mark.xfail\ndef test_skipped():\n    pass\n"
    ).body[0]
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
