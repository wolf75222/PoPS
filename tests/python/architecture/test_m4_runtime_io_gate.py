"""Source-only integrity checks for the executable M4 runtime/IO gate."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import subprocess
import sys
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


def test_m4_manifest_audits_the_exact_open_matrix():
    runner = _load_runner()
    data, errors = runner.audit_manifest(MANIFEST)

    assert not errors, "M4 gate audit is structurally invalid:\n  " + "\n  ".join(errors)
    assert len(data["check"]) == 53
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
    deferred = {
        (row["issue"], row["requirement"], row["polarity"])
        for row in data["deferred"]
    }
    assert deferred == {
        ("ADC-681", "external_boundary", "positive"),
        ("ADC-681", "external_tagger", "positive"),
    }

    _, closure_errors = runner.validate_manifest(MANIFEST)
    assert len(closure_errors) == 2
    assert all("remains deferred" in error for error in closure_errors)


def test_m4_cli_reports_open_and_check_only_fails_closed():
    audit = subprocess.run(
        [sys.executable, str(RUNNER), "--audit-only"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert audit.returncode == 0
    assert "M4 gate source matrix: AUDITED OPEN (53 executable, 2 deferred)" in audit.stdout

    closure = subprocess.run(
        [sys.executable, str(RUNNER), "--check-only"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert closure.returncode == 2
    assert "external_boundary/positive remains deferred" in closure.stdout
    assert "external_tagger/positive remains deferred" in closure.stdout


def test_m4_required_ci_lane_executes_the_complete_installed_gate():
    data, errors = _load_runner().audit_manifest(MANIFEST)
    assert not errors
    nodeid = (
        "tests/python/architecture/test_m4_runtime_io_gate.py::"
        "test_m4_required_ci_lane_executes_the_complete_installed_gate"
    )
    assert [
        row for row in data["check"] if row.get("nodeid") == nodeid
    ] == [{
        "issue": "ADC-687",
        "requirement": "gate_execution",
        "polarity": "positive",
        "kind": "pytest",
        "target": "gate_execution",
        "nodeid": nodeid,
    }]

    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    mpi_job = workflow.split("\n  mpi:\n", 1)[1]
    mpi_job = mpi_job.split("\n  gate-openmp-prewarm:\n", 1)[0]
    assert "if: needs.set-mode.outputs.mpi_required == 'true'" in mpi_job
    assert "python3-vtk9" in mpi_job
    assert "/usr/bin/python3 scripts/run_m4_gate.py --list-ctest-targets" in mpi_job
    assert 'cmake --build --preset ci-mpi --parallel 4 --target "${m4_targets[@]}"' in mpi_job

    complete = mpi_job.split(
        "- name: M4 complete native runtime and scientific I/O gate", 1
    )[1]
    complete = complete.split("- name: ccache stats (MPI)", 1)[0]
    assert "POPS_REQUIRE_MPI_TESTS: \"1\"" in complete
    assert "POPS_REQUIRE_NATIVE_TESTS: \"1\"" in complete
    assert 'OMPI_MCA_io: "^ompio"' in complete
    assert "vtkXMLPUnstructuredGridReader" in complete
    assert "vtkXMLUnstructuredGridReader" in complete
    assert "/usr/bin/python3 scripts/run_m4_gate.py \\" in complete
    assert "--build-dir build-mpi" in complete
    assert "--mpi-exec mpiexec" in complete
    assert "--audit-only" not in complete
    assert "--python-only" not in complete
    assert "continue-on-error" not in complete

    aggregator = workflow.split("\n  gate:\n", 1)[1]
    aggregator = aggregator.split("\n  mpi:\n", 1)[0]
    assert "mpi" in aggregator.split("needs:", 1)[1].splitlines()[0]
    assert '--gate mpi "${{ needs.mpi.result }}"' in aggregator
    assert '"${{ needs.set-mode.outputs.mpi_required }}"' in aggregator

    mpi_filter = workflow.split("\n            mpi:\n", 1)[1]
    mpi_filter = mpi_filter.split("\n            # full", 1)[0]
    for protected_path in (
        "tests/gates/m4_runtime_io.toml",
        "tests/python/architecture/test_m4_runtime_io_gate.py",
        "scripts/run_m4_gate.py",
        ".github/workflows/ci.yml",
    ):
        assert "'%s'" % protected_path in mpi_filter


def test_m4_open_gate_refuses_to_list_targets_as_closed():
    runner = _load_runner()
    data, errors = runner.audit_manifest(MANIFEST)
    assert not errors
    expected = (
        "test_amr_native_loader",
        "test_brick_catalog",
        "test_component_interfaces",
        "test_flux_interfaces",
        "test_mpi_hdf5_collective",
        "test_native_loader_param_overflow",
        "test_platform_manifest",
        "test_program_context_contract",
        "test_program_runtime",
    )
    assert runner._required_ctest_targets(data["check"]) == expected

    listed = subprocess.run(
        [sys.executable, str(RUNNER), "--list-ctest-targets"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert listed.returncode == 2
    assert "remains deferred" in listed.stdout


def test_m4_gate_pins_every_external_component_family():
    data, errors = _load_runner().audit_manifest(MANIFEST)
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

    assert {
        (row["issue"], row["requirement"], row["polarity"])
        for row in data["deferred"]
    } == {
        ("ADC-681", "external_boundary", "positive"),
        ("ADC-681", "external_tagger", "positive"),
    }

    assert (
        "external_solver",
        "positive",
        "tests/python/integration/mpi/test_external_amr_field_solver_mpi.py::"
        "test_external_amr_field_bridge_executes_and_refuses_collectively",
    ) in executable


def test_m4_gate_pins_real_runtime_instance_and_positive_checkpoint_proofs():
    data, errors = _load_runner().audit_manifest(MANIFEST)
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
        "requirement": "runtime_instance",
        "polarity": "positive",
        "kind": "pytest",
        "target": "runtime_instance",
        "nodeid": (
            "tests/python/integration/runtime/test_multi_layout_runtime.py::"
            "test_uniform_amr_and_multi_layout_share_complete_runtime_instance_contract"
        ),
    } in checks
    assert {
        "issue": "ADC-684",
        "requirement": "runtime_instance",
        "polarity": "refusal",
        "kind": "pytest",
        "target": "runtime_instance",
        "nodeid": (
            "tests/python/integration/native_loader/"
            "test_external_field_solver_runtime.py::"
            "test_real_prepared_field_solver_failure_rolls_back_runtime_instance_and_retries"
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
            "tests/python/integration/amr/test_amr_regrid_on_restart.py::"
            "test_authenticated_amr_contract_refusal_rolls_back_native_restart_transaction"
        ),
    } in checks
    selected = {
        row.get("nodeid", row.get("test_regex"))
        for row in checks
    }
    assert (
        "tests/python/integration/runtime/test_multi_layout_runtime.py::"
        "test_mid_step_child_failure_preserves_root_error_and_rolls_back_composite"
    ) not in selected
    assert (
        "tests/python/integration/runtime/test_multi_layout_runtime.py::"
        "test_failed_child_restart_rolls_back_already_restored_layouts"
    ) not in selected


def test_m4_runtime_refusal_uses_a_real_prepared_component_without_step_wrapper():
    data, errors = _load_runner().audit_manifest(MANIFEST)
    assert not errors
    nodeid = (
        "tests/python/integration/native_loader/"
        "test_external_field_solver_runtime.py::"
        "test_real_prepared_field_solver_failure_rolls_back_runtime_instance_and_retries"
    )
    assert [
        row
        for row in data["check"]
        if row.get("nodeid") == nodeid
    ] == [{
        "issue": "ADC-684",
        "requirement": "runtime_instance",
        "polarity": "refusal",
        "kind": "pytest",
        "target": "runtime_instance",
        "nodeid": nodeid,
    }]

    source_path = (
        ROOT
        / "tests/python/integration/native_loader/test_external_field_solver_runtime.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        == "test_real_prepared_field_solver_failure_rolls_back_runtime_instance_and_retries"
    )
    calls = {
        (
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr
            if isinstance(node.func, ast.Attribute)
            else ""
        )
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
    }
    assert {"_component", "compile", "bind", "run"} <= calls
    names = {node.id for node in ast.walk(function) if isinstance(node, ast.Name)}
    assert names.isdisjoint(
        {
            "FailFirstStep",
            "_RankLocalFailureTarget",
            "Mock",
            "MagicMock",
            "SimpleNamespace",
        }
    )
    attributes = {
        node.attr for node in ast.walk(function) if isinstance(node, ast.Attribute)
    }
    assert "_native_step_target" not in attributes
    assert "_engines" not in attributes
    assert not any(isinstance(node, ast.ClassDef) for node in ast.walk(function))


def test_m4_runtime_positive_compiles_all_layout_kinds_without_test_doubles():
    data, errors = _load_runner().audit_manifest(MANIFEST)
    assert not errors
    nodeid = (
        "tests/python/integration/runtime/test_multi_layout_runtime.py::"
        "test_uniform_amr_and_multi_layout_share_complete_runtime_instance_contract"
    )
    assert [
        row for row in data["check"] if row.get("nodeid") == nodeid
    ] == [{
        "issue": "ADC-684",
        "requirement": "runtime_instance",
        "polarity": "positive",
        "kind": "pytest",
        "target": "runtime_instance",
        "nodeid": nodeid,
    }]

    path = ROOT / "tests/python/integration/runtime/test_multi_layout_runtime.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        == "test_uniform_amr_and_multi_layout_share_complete_runtime_instance_contract"
    )
    calls = {
        (
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr
            if isinstance(node.func, ast.Attribute)
            else ""
        )
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
    }
    assert {"compile", "bind", "run", "program_report", "inspect", "integral"} <= calls
    labels = {
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert {"uniform", "amr", "multi-layout"} <= labels
    names = {node.id for node in ast.walk(function) if isinstance(node, ast.Name)}
    assert names.isdisjoint(
        {
            "FailFirstStep",
            "_RankLocalFailureTarget",
            "Mock",
            "MagicMock",
            "SimpleNamespace",
            "monkeypatch",
        }
    )
    attributes = {
        node.attr for node in ast.walk(function) if isinstance(node, ast.Attribute)
    }
    assert "_native_step_target" not in attributes
    assert "_engines" not in attributes
    assert not any(isinstance(node, ast.ClassDef) for node in ast.walk(function))


def test_m4_gate_pins_real_writer_refusal_without_publication_fakes():
    data, errors = _load_runner().audit_manifest(MANIFEST)
    assert not errors
    expected = {
        "issue": "ADC-685",
        "requirement": "consumer_graph",
        "polarity": "refusal",
        "kind": "pytest",
        "target": "consumer_graph",
        "nodeid": (
            "tests/python/integration/native_loader/"
            "test_external_component_package.py::"
            "test_real_writer_collision_compensates_the_complete_consumer_graph_transaction"
        ),
    }
    assert expected in data["check"]

    path = (
        ROOT
        / "tests/python/integration/native_loader/test_external_component_package.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        == "test_real_writer_collision_compensates_the_complete_consumer_graph_transaction"
    )
    calls = {
        (
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr
            if isinstance(node.func, ast.Attribute)
            else ""
        )
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
    }
    assert {
        "_compile_writer",
        "_bind_writer_case",
        "_stage_consumers",
        "accept",
        "_fire_consumers",
    } <= calls
    names = {node.id for node in ast.walk(function) if isinstance(node, ast.Name)}
    assert names.isdisjoint(
        {"_Publisher", "_Prepared", "SimpleNamespace", "Mock", "MagicMock"}
    )


def test_m4_gate_keeps_real_tamper_and_capacity_refusals():
    data, errors = _load_runner().audit_manifest(MANIFEST)
    assert not errors

    refusals = {
        row.get("nodeid", row.get("test_regex"))
        for row in data["check"]
        if row["requirement"] == "tamper_capability_abi"
        and row["polarity"] == "refusal"
    }
    assert {
        (
            "tests/python/unit/codegen/test_component_packages.py::"
            "test_fixed_binary_cannot_claim_template_genericity"
        ),
        r"^test_native_loader_param_overflow\.Runs$",
        r"^test_amr_native_loader\.RefusesComponentBuiltForAnotherNativeAbi$",
        r"^PlatformManifest\.UnknownCapabilityRefusesBeforeKernel$",
        (
            "tests/python/unit/codegen/test_component_packages.py::"
            "test_fixed_binary_bytes_are_authenticated_before_package_use"
        ),
    } <= refusals
    assert (
        "tests/python/unit/codegen/test_component_manifest_v2.py::"
        "test_target_capability_refusal_contains_requested_and_supported_evidence"
    ) not in refusals
    assert (
        "tests/python/unit/runtime/test_platform_manifest.py::"
        "test_aot_component_rejects_openmpi_mpich_abi_mix_even_with_same_headers_and_standard"
    ) not in refusals
    assert {
        (row["issue"], row["requirement"], row["polarity"])
        for row in data["deferred"]
        if row["requirement"] == "tamper_capability_abi"
    } == set()


def test_m4_gate_pins_mandatory_native_reopen_and_collective_hdf5_np2():
    data, errors = _load_runner().audit_manifest(MANIFEST)
    assert not errors
    checks = data["check"]

    native_reopen = {
        row["requirement"]: row["nodeid"]
        for row in checks
        if row["requirement"] in {"exact_npz", "exact_hdf5", "exact_paraview"}
        and row["polarity"] == "positive"
        and row["kind"] == "pytest"
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
    mpi_native = [
        row
        for row in checks
        if row["requirement"] == "exact_paraview"
        and row["kind"] == "mpi_python"
    ]
    assert mpi_native == [{
        "issue": "ADC-686",
        "requirement": "exact_paraview",
        "polarity": "positive",
        "kind": "mpi_python",
        "target": "exact_paraview",
        "nodeid": (
            "tests/python/integration/mpi/test_scientific_output_mpi.py::"
            "_validate_paraview"
        ),
        "nproc": 2,
    }]
    mpi_source = (
        ROOT / "tests/python/integration/mpi/test_scientific_output_mpi.py"
    ).read_text(encoding="utf-8")
    assert "vtkXMLPUnstructuredGridReader" in mpi_source
    assert "vtkXMLUnstructuredGridReader" in mpi_source
    assert "native PVD/PVTU traversal" in mpi_source
    assert "except ImportError" not in mpi_source
    assert {
        "issue": "ADC-686",
        "requirement": "collective_hdf5",
        "polarity": "positive",
        "kind": "ctest",
        "target": "collective_hdf5@test_mpi_hdf5_collective",
        "test_regex": "^test_mpi_hdf5_collective_np2$",
    } in checks


def test_m4_gate_pins_complete_program_only_dispatch_and_fallback_fences():
    data, errors = _load_runner().audit_manifest(MANIFEST)
    assert not errors
    selected = {
        row["nodeid"]
        for row in data["check"]
        if row["requirement"] == "legacy_stepper_retirement"
    }
    assert selected == {
        "tests/python/architecture/test_no_schur_header_leak.py::"
        "test_native_source_stage_headers_are_retired",
        "tests/python/architecture/test_program_only_temporal_facades.py::"
        "test_system_temporal_facades_dispatch_only_through_an_installed_program",
        "tests/python/architecture/test_program_only_temporal_facades.py::"
        "test_amr_temporal_facades_use_amr_runtime_only_as_the_spatial_engine",
        "tests/python/architecture/test_program_only_temporal_facades.py::"
        "test_historical_block_scheduler_is_not_an_installed_temporal_authority",
        "tests/python/architecture/test_program_only_temporal_facades.py::"
        "test_production_has_no_second_amr_time_engine",
        "tests/python/architecture/test_component_interface_dispatch.py::"
        "test_component_trust_boundary_never_classifies_the_scientific_component_type",
        "tests/python/architecture/test_component_interface_dispatch.py::"
        "test_native_registry_has_no_rtti_or_untyped_capability_escape_hatch",
        "tests/python/unit/codegen/test_component_adapters.py::"
        "test_native_interface_is_declared_and_unbound_never_falls_back",
    }

    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    job = workflow.split("\n  gate-python-architecture:\n", 1)[1]
    job = job.split("\n  gate-python-build:\n", 1)[0]
    command = "run: python3 scripts/run_m4_gate.py --check-only"
    assert [line.strip() for line in job.splitlines()].count(command) == 1
    assert "run: python3 scripts/run_m4_gate.py --audit-only" not in job

    documentation = (
        ROOT / "docs/design/m4-conformance-gate.md"
    ).read_text(encoding="utf-8")
    assert "evidence ledger is **REQUIRED BY CI AND CURRENTLY OPEN**" in documentation
    assert "53 executable checks and two" in documentation
    assert "external boundary" in documentation
    assert "external AMR Tagger/Clustering/Reflux" in documentation


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


def test_m4_mpi_entrypoint_accepts_only_the_required_prerequisite_guard(monkeypatch):
    runner = _load_runner()
    data, errors = runner.audit_manifest(MANIFEST)
    assert not errors
    mpi_proof = next(
        row for row in data["check"]
        if row["kind"] == "mpi_python"
    )
    assert mpi_proof["nodeid"] == (
        "tests/python/integration/mpi/test_scientific_output_mpi.py::"
        "_validate_paraview"
    )
    monkeypatch.setenv("POPS_NATIVE_DIM", "2")
    environment = runner._required_environment()
    assert environment["POPS_REQUIRE_MPI_TESTS"] == "1"
    assert environment["POPS_NATIVE_DIM"] == "2"
    assert str(ROOT) in environment["PYTHONPATH"].split(runner.os.pathsep)
    monkeypatch.setattr(runner.shutil, "which", lambda _executable: "/usr/bin/mpiexec")
    command = runner._mpi_python_command("mpiexec", 2, mpi_proof["nodeid"].split("::", 1)[0])
    assert command[:5] == ["mpiexec", "-n", "2", runner.sys.executable, "-c"]
    assert "select_native_dimension(2)" in command[5]
    assert "runpy.run_path(sys.argv[1], run_name='__main__')" in command[5]
    assert command[6] == str(ROOT / mpi_proof["nodeid"].split("::", 1)[0])
    trusted = ast.parse(
        "from tests.python.support.requirements import require_mpi_or_skip\n"
    )
    untrusted = ast.parse("def require_mpi_or_skip(_reason):\n    return None\n")
    assert runner._has_authenticated_mpi_guard(trusted)
    assert not runner._has_authenticated_mpi_guard(untrusted)


def test_m4_execution_refuses_to_infer_a_native_dimension(monkeypatch):
    runner = _load_runner()
    monkeypatch.delenv("POPS_NATIVE_DIM", raising=False)

    with pytest.raises(RuntimeError, match="launcher-provided POPS_NATIVE_DIM=2"):
        runner._required_environment()


def test_m4_gate_keeps_external_component_gaps_explicit():
    runner = _load_runner()
    data, audit_errors = runner.audit_manifest(MANIFEST)
    assert not audit_errors
    assert len(data["deferred"]) == 2

    _, errors = runner.validate_manifest(MANIFEST)
    assert len(errors) == 2
    assert all("remains deferred" in error for error in errors)


def test_m4_required_pytest_execution_rejects_junit_skips(monkeypatch):
    runner = _load_runner()
    monkeypatch.setenv("POPS_NATIVE_DIM", "2")
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


def test_m4_check_only_refuses_open_ledger_without_launcher_or_build(monkeypatch):
    runner = _load_runner()

    def forbidden_call(*_args, **_kwargs):
        raise AssertionError("--check-only attempted to launch an executable")

    monkeypatch.setattr(runner.shutil, "which", forbidden_call)
    monkeypatch.setattr(runner.subprocess, "run", forbidden_call)

    assert runner.main(["--check-only"]) == 2
