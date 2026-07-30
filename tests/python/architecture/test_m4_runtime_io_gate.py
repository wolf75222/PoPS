"""Source-only integrity checks for the executable M4 runtime/IO gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "tests/gates/m4_runtime_io.toml"
RUNNER = ROOT / "scripts/run_m4_gate.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("pops_run_m4_gate", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mutated_manifest(tmp_path: Path, old: str, new: str) -> Path:
    source = MANIFEST.read_text(encoding="utf-8")
    assert old in source
    path = tmp_path / "m4.toml"
    path.write_text(source.replace(old, new, 1), encoding="utf-8")
    return path


def test_m4_manifest_is_a_closed_exact_mandatory_matrix():
    data, errors = _load_runner().validate_manifest(MANIFEST)

    assert not errors, "M4 gate matrix is incomplete:\n  " + "\n  ".join(errors)
    assert data["deferred"] == []
    assert len(data["check"]) >= 41
    assert data["issues"] == [
        "ADC-679",
        "ADC-680",
        "ADC-681",
        "ADC-682",
        "ADC-683",
        "ADC-684",
        "ADC-685",
        "ADC-686",
        "ADC-687",
    ]
    assert {row["issue"] for row in data["check"]} == set(data["issues"])


def test_m4_gate_pins_every_external_component_family():
    data, errors = _load_runner().validate_manifest(MANIFEST)
    assert not errors

    executable = {
        (
            row["requirement"],
            row["polarity"],
            row.get("nodeid", row.get("test_regex")),
        )
        for row in data["check"]
    }
    assert {
        (
            "external_flux",
            "positive",
            "tests/python/integration/native_loader/"
            "test_external_component_package.py::"
            "test_source_component_executes_through_generic_native_loader_and_flux_consumer",
        ),
        (
            "external_boundary",
            "positive",
            r"^test_amr_native_loader\."
            r"BoundaryPlanSessionsOwnFreshLaneQualifiedComponentStates$",
        ),
        (
            "external_tagger",
            "positive",
            r"^test_amr_native_loader\."
            r"PreparedAmrProvidersExecuteExactTablesAndProvenance$",
        ),
        (
            "external_transfer",
            "positive",
            "tests/python/integration/runtime/test_multi_layout_runtime.py::"
            "test_two_native_layouts_execute_sliced_programs_and_exact_transfer",
        ),
        (
            "external_solver",
            "positive",
            "tests/python/integration/native_loader/"
            "test_external_field_solver_runtime.py::"
            "test_external_field_pair_executes_and_reports_materialized_topology",
        ),
        (
            "external_writer",
            "positive",
            "tests/python/integration/native_loader/"
            "test_external_component_package.py::"
            "test_qualified_writer_runs_through_uniform_and_amr_runtime_transactions",
        ),
    } <= executable


def test_m4_gate_pins_runtime_instance_multi_layout_and_strict_checkpoint():
    data, errors = _load_runner().validate_manifest(MANIFEST)
    assert not errors
    checks = data["check"]

    assert {
        "issue": "ADC-684",
        "requirement": "runtime_instance",
        "polarity": "positive",
        "kind": "pytest",
        "target": "runtime_instance",
        "nodeid": (
            "tests/python/integration/runtime/test_shared_interface_runtime.py::"
            "test_runtime_instance_executes_one_two_sided_shared_flux"
        ),
    } in checks
    assert {
        "issue": "ADC-684",
        "requirement": "external_transfer",
        "polarity": "positive",
        "kind": "pytest",
        "target": "external_transfer",
        "nodeid": (
            "tests/python/integration/runtime/test_multi_layout_runtime.py::"
            "test_two_native_layouts_execute_sliced_programs_and_exact_transfer"
        ),
    } in checks
    assert {
        "issue": "ADC-685",
        "requirement": "external_writer",
        "polarity": "positive",
        "kind": "pytest",
        "target": "external_writer",
        "nodeid": (
            "tests/python/integration/native_loader/"
            "test_external_component_package.py::"
            "test_qualified_writer_runs_through_uniform_and_amr_runtime_transactions"
        ),
    } in checks
    assert {
        "issue": "ADC-686",
        "requirement": "strict_checkpoint",
        "polarity": "positive",
        "kind": "pytest",
        "target": "strict_checkpoint",
        "nodeid": (
            "tests/python/integration/runtime/test_multi_layout_runtime.py::"
            "test_multi_layout_checkpoint_restart_restores_every_layout_and_mapping_count"
        ),
    } in checks
    assert {
        "issue": "ADC-686",
        "requirement": "strict_checkpoint",
        "polarity": "refusal",
        "kind": "pytest",
        "target": "strict_checkpoint",
        "nodeid": (
            "tests/python/integration/runtime/test_multi_layout_runtime.py::"
            "test_failed_child_restart_rolls_back_already_restored_layouts"
        ),
    } in checks


def test_m4_gate_pins_capability_tamper_and_native_abi_refusals():
    data, errors = _load_runner().validate_manifest(MANIFEST)
    assert not errors

    refusals = {
        row.get("nodeid", row.get("test_regex"))
        for row in data["check"]
        if row["requirement"] == "tamper_capability_abi"
        and row["polarity"] == "refusal"
    }
    assert {
        (
            "tests/python/unit/codegen/test_component_manifest_v2.py::"
            "test_target_capability_refusal_contains_requested_and_supported_evidence"
        ),
        (
            "tests/python/unit/codegen/test_component_packages.py::"
            "test_fixed_binary_cannot_claim_template_genericity"
        ),
        (
            "tests/python/unit/runtime/test_platform_manifest.py::"
            "test_aot_component_rejects_openmpi_mpich_abi_mix_even_with_same_headers_and_standard"
        ),
        r"^test_native_loader_param_overflow\.Runs$",
    } <= refusals


def test_m4_gate_pins_mandatory_native_reopen_and_collective_hdf5_np2():
    data, errors = _load_runner().validate_manifest(MANIFEST)
    assert not errors
    checks = data["check"]

    native_reopen = {
        row["requirement"]: row["nodeid"]
        for row in checks
        if row["requirement"] in {"exact_npz", "exact_hdf5", "exact_paraview"}
        and row["polarity"] == "positive"
    }
    assert native_reopen == {
        "exact_npz": (
            "tests/python/integration/io/m4_native_reopen_proof.py::"
            "test_npz_reopens_with_numpy_without_a_pops_reader"
        ),
        "exact_hdf5": (
            "tests/python/integration/io/m4_native_reopen_proof.py::"
            "test_hdf5_reopens_with_h5py_without_a_pops_reader"
        ),
        "exact_paraview": (
            "tests/python/integration/io/m4_native_reopen_proof.py::"
            "test_paraview_reopens_with_vtk_without_a_pops_reader"
        ),
    }
    source = (
        ROOT / "tests/python/integration/io/m4_native_reopen_proof.py"
    ).read_text(encoding="utf-8")
    assert "pytest.importorskip" not in source
    assert "import h5py" in source
    assert "from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader" in source
    assert {
        "issue": "ADC-686",
        "requirement": "collective_hdf5",
        "polarity": "positive",
        "kind": "ctest",
        "target": "collective_hdf5@test_mpi_hdf5_collective",
        "test_regex": "^test_mpi_hdf5_collective_np2$",
    } in checks


def test_m4_gate_pins_schur_retirement_and_ci_check_only_command():
    data, errors = _load_runner().validate_manifest(MANIFEST)
    assert not errors
    assert {
        "issue": "ADC-687",
        "requirement": "legacy_stepper_retirement",
        "polarity": "positive",
        "kind": "pytest",
        "target": "legacy_stepper_retirement",
        "nodeid": (
            "tests/python/architecture/test_no_schur_header_leak.py::"
            "test_native_source_stage_headers_are_retired"
        ),
    } in data["check"]

    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    job = workflow.split("\n  gate-python-architecture:\n", 1)[1]
    job = job.split("\n  gate-python-build:\n", 1)[0]
    command = "run: python3 scripts/run_m4_gate.py --check-only"
    assert [line.strip() for line in job.splitlines()].count(command) == 1


def test_m4_gate_rejects_fake_nodeid_before_execution(tmp_path):
    manifest = _mutated_manifest(
        tmp_path,
        (
            "tests/python/unit/codegen/test_component_manifest_v2.py::"
            "test_native_parser_normalizer_matches_python_canonical_bytes"
        ),
        (
            "tests/python/unit/codegen/test_component_manifest_v2.py::"
            "test_definitely_missing_m4_proof"
        ),
    )

    _, errors = _load_runner().validate_manifest(manifest)

    assert any("references missing test function" in error for error in errors)


def test_m4_gate_rejects_wildcard_ctest_selector_before_build(tmp_path):
    manifest = _mutated_manifest(
        tmp_path,
        'test_regex = "^test_mpi_hdf5_collective_np2$"',
        'test_regex = "^test_mpi_hdf5_collective_.*$"',
    )

    _, errors = _load_runner().validate_manifest(manifest)

    assert any(
        "is not one exact source-registered case for target "
        "'test_mpi_hdf5_collective'" in error
        for error in errors
    )


def test_m4_gate_rejects_requirement_attributed_to_the_wrong_issue(tmp_path):
    manifest = _mutated_manifest(
        tmp_path,
        (
            'issue = "ADC-679"\n'
            'requirement = "component_manifest"\n'
            'polarity = "positive"'
        ),
        (
            'issue = "ADC-680"\n'
            'requirement = "component_manifest"\n'
            'polarity = "positive"'
        ),
    )

    _, errors = _load_runner().validate_manifest(manifest)

    assert any(
        "requirement 'component_manifest' cannot be attributed to 'ADC-680'" in error
        for error in errors
    )


def test_m4_gate_rejects_importorskip_and_mock_proofs(tmp_path):
    optional_manifest = _mutated_manifest(
        tmp_path,
        (
            "tests/python/integration/io/m4_native_reopen_proof.py::"
            "test_hdf5_reopens_with_h5py_without_a_pops_reader"
        ),
        (
            "tests/python/unit/output/test_exact_writers.py::"
            "test_hdf5_is_reopened_with_native_reader_and_exact_selection"
        ),
    )
    _, optional_errors = _load_runner().validate_manifest(optional_manifest)
    assert any(
        "is not an unconditional real proof" in error
        and "pytest.importorskip" in error
        for error in optional_errors
    )

    mock_manifest = _mutated_manifest(
        tmp_path,
        (
            "tests/python/unit/runtime/test_consumer_transactions.py::"
            "test_graph_and_plan_are_semantic_and_insertion_order_independent"
        ),
        (
            "tests/python/architecture/test_m3_amr_multilayout_gate.py::"
            "test_m3_mpi_python_proof_is_exact_and_manifest_owned"
        ),
    )
    _, mock_errors = _load_runner().validate_manifest(mock_manifest)
    assert any(
        "is not an unconditional real proof" in error
        and "fixture:monkeypatch" in error
        for error in mock_errors
    )


def test_m4_gate_rejects_every_deferred_requirement(tmp_path):
    manifest = _mutated_manifest(
        tmp_path,
        "deferred = []",
        (
            "[[deferred]]\n"
            'issue = "ADC-687"\n'
            'requirement = "legacy_stepper_retirement"\n'
            'reason = "The mandatory Schur retirement proof is deliberately deferred."\n'
            "evidence_paths = "
            '["tests/python/architecture/test_no_schur_header_leak.py"]'
        ),
    )

    data, audit_errors = _load_runner().audit_manifest(manifest)
    assert not audit_errors
    assert len(data["deferred"]) == 1

    _, errors = _load_runner().validate_manifest(manifest)
    assert any("remains deferred" in error for error in errors)


def test_m4_required_pytest_execution_rejects_junit_skips(monkeypatch):
    runner = _load_runner()
    skipped_xml = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<testsuites><testsuite tests="1" skipped="1">'
        '<testcase classname="m4" name="proof">'
        '<skipped type="pytest.skip"/>'
        "</testcase></testsuite></testsuites>"
    )

    def successful_pytest_with_a_skip(command, *, cwd, env, check):
        assert cwd == ROOT
        assert env["POPS_REQUIRE_MPI_TESTS"] == "1"
        assert env["POPS_REQUIRE_NATIVE_TESTS"] == "1"
        assert check is False
        assert "xfail_strict=true" in command
        report = Path(command[command.index("--junitxml") + 1])
        report.write_text(skipped_xml, encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", successful_pytest_with_a_skip)
    with pytest.raises(RuntimeError, match="reported 1 skipped/xfail proof"):
        runner._run_required_pytest(
            [
                "tests/python/integration/io/m4_native_reopen_proof.py::"
                "test_npz_reopens_with_numpy_without_a_pops_reader"
            ]
        )


def test_m4_required_ctest_execution_rejects_junit_skips(tmp_path, monkeypatch):
    runner = _load_runner()
    skipped_xml = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<testsuites><testsuite tests="1" skipped="1">'
        '<testcase classname="m4" name="native-proof">'
        '<skipped type="notrun"/>'
        "</testcase></testsuite></testsuites>"
    )
    calls = 0

    def ctest_with_a_skip(command, **kwargs):
        nonlocal calls
        calls += 1
        assert kwargs["cwd"] == ROOT
        if "-N" in command:
            assert kwargs["check"] is True
            assert kwargs["capture_output"] is True
            return SimpleNamespace(
                returncode=0,
                stdout="Test #1: ComponentInterfaces.Proof\nTotal Tests: 1\n",
            )
        assert kwargs["check"] is False
        report = Path(command[command.index("--output-junit") + 1])
        report.write_text(skipped_xml, encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", ctest_with_a_skip)
    with pytest.raises(RuntimeError, match="reported 1 skipped proof"):
        runner._run_ctest(
            tmp_path / "build",
            "test_component_interfaces",
            r"^ComponentInterfaces\.Proof$",
        )
    assert calls == 2


def test_m4_check_only_never_consults_launcher_or_build(monkeypatch):
    runner = _load_runner()

    def forbidden_call(*_args, **_kwargs):
        raise AssertionError("--check-only attempted to launch an executable")

    monkeypatch.setattr(runner.shutil, "which", forbidden_call)
    monkeypatch.setattr(runner.subprocess, "run", forbidden_call)

    assert runner.main(["--check-only"]) == 0
