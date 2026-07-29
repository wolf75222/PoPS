"""Source-only integrity checks for the executable M3 AMR/multi-layout gate."""

from __future__ import annotations

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
    assert len(data["check"]) == 39


def test_m3_gate_pins_three_level_subcycled_reflux_proof():
    data, errors = _load_runner().validate_manifest(MANIFEST)
    assert not errors
    assert {
        "issue": "ADC-678",
        "requirement": "accepted_state",
        "polarity": "positive",
        "kind": "ctest",
        "target": "test_amr_history_ring",
        "test_regex": (
            "^test_amr_history_ring\\."
            "ThreeLevelProgramSynchronizesEachRecursiveCatchUp$"
        ),
    } in data["check"]

    source = ROOT / "tests/cpp/integration/amr/test_amr_history_ring.cpp"
    assert (
        "TEST(test_amr_history_ring, "
        "ThreeLevelProgramSynchronizesEachRecursiveCatchUp)"
    ) in source.read_text(encoding="utf-8")


def test_m3_gate_pins_metric_weighted_composite_diagnostic_proof():
    data, errors = _load_runner().validate_manifest(MANIFEST)
    assert not errors
    assert {
        "issue": "ADC-678",
        "requirement": "accepted_state",
        "polarity": "positive",
        "kind": "ctest",
        "target": "test_mpi_amr_distributed_coarse",
        "test_regex": "^test_mpi_amr_distributed_coarse_np2$",
    } in data["check"]

    source = (
        ROOT
        / "tests/cpp/integration/mpi/test_mpi_amr_distributed_coarse.cpp"
    ).read_text(encoding="utf-8")
    assert "runtime::amr::composite_reduce_fields" in source
    assert "std::fabs(integral - 1.25)" in source


def test_m3_gate_pins_fail_closed_persistent_hysteresis_proofs():
    data, errors = _load_runner().validate_manifest(MANIFEST)
    assert not errors
    checks = data["check"]
    assert {
        "issue": "ADC-678",
        "requirement": "accepted_state",
        "polarity": "refusal",
        "kind": "pytest",
        "target": "accepted_state",
        "nodeid": (
            "tests/python/unit/amr/test_public_amr_resolution.py::"
            "test_tagging_resolution_refuses_unimplemented_persistent_hysteresis"
        ),
    } in checks
    assert {
        "issue": "ADC-678",
        "requirement": "accepted_state",
        "polarity": "refusal",
        "kind": "ctest",
        "target": "test_amr_native_loader",
        "test_regex": (
            "^test_amr_native_loader\\."
            "PreparedAmrProvidersExecuteExactTablesAndProvenance$"
        ),
    } in checks

    authoring_source = (
        ROOT / "tests/python/unit/amr/test_public_amr_resolution.py"
    ).read_text(encoding="utf-8")
    assert "test_tagging_resolution_refuses_unimplemented_persistent_hysteresis" in (
        authoring_source
    )
    native_source = (
        ROOT
        / "tests/cpp/integration/native_loader/test_amr_native_loader.cpp"
    ).read_text(encoding="utf-8")
    assert "unsupported_hysteresis.min_cycles = 1" in native_source
    assert "EXPECT_EQ(tag_call_count(), calls_before_hysteresis)" in native_source


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
        "test_amr_rank_change_restart.py is not a manifest-owned serial MPI orchestrator"
        in error
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


def test_m3_required_pytest_execution_rejects_every_skip_or_xfail(
    tmp_path, monkeypatch
):
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
