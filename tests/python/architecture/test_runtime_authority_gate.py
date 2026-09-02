"""Source-only integrity checks for the closed ADC-700/702/720 gate."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "tests/gates/runtime_authority.toml"
RUNNER = ROOT / "scripts/run_runtime_authority_gate.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("pops_run_runtime_authority_gate", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mutated_manifest(tmp_path: Path, old: str, new: str) -> Path:
    source = MANIFEST.read_text(encoding="utf-8")
    assert old in source
    path = tmp_path / "runtime-authority.toml"
    path.write_text(source.replace(old, new, 1), encoding="utf-8")
    return path


def _write_synthetic_source(root: Path, relative: str, source: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def test_runtime_authority_manifest_is_closed_and_source_owned():
    runner = _load_runner()
    data, errors = runner.audit_manifest(MANIFEST)

    assert errors == []
    assert data["issues"] == ["ADC-700", "ADC-702", "ADC-720"]
    assert data["deferred"] == []
    assert len(data["check"]) == 73
    corner_rows = [
        row
        for row in data["check"]
        if row.get("test_regex") == r"^test_mpi_amr_program_3d_corner_authority_np8$"
    ]
    assert len(corner_rows) == 1
    assert corner_rows[0]["dimensions"] == [3]
    assert sum("dimensions" in row for row in data["check"]) == 4
    assert {
        (
            row["polarity"],
            row["target"],
            row["backend"],
            tuple(row.get("dimensions", ())),
            row["test_regex"],
        )
        for row in data["check"]
        if row["requirement"] == "cell_local_ordinary" and row["kind"] == "ctest"
    } == {
        (
            "positive",
            "cell_local_ordinary@test_mpi_cell_temporal_program",
            "mpi",
            (1, 2, 3),
            r"^test_mpi_cell_temporal_program_np2$",
        ),
        (
            "positive",
            "cell_local_ordinary@test_mpi_cell_temporal_program_multibox",
            "mpi",
            (1, 2, 3),
            r"^test_mpi_cell_temporal_program_multibox_np2$",
        ),
        (
            "positive",
            "cell_local_ordinary@test_mpi_cell_temporal_program_collective_rollback",
            "mpi",
            (1, 2, 3),
            r"^test_mpi_cell_temporal_program_collective_rollback_np2$",
        ),
        (
            "positive",
            "cell_local_ordinary@test_cell_temporal_program_route",
            "serial",
            (),
            r"^test_cell_temporal_program_route\.direct_prepared_subengine_executes_one_exact_algorithm_in_every_dimension$",
        ),
        (
            "refusal",
            "cell_local_ordinary@test_amr_system_contract",
            "serial",
            (),
            r"^test_amr_system_contract\.RefusesCellLocalFluxTablesBeforeResourceMaterialization$",
        ),
        (
            "refusal",
            "cell_local_ordinary@test_mpi_cell_temporal_program_refusal",
            "mpi",
            (),
            r"^test_mpi_cell_temporal_program_refusal_np2$",
        ),
    } == {
        (
            row["polarity"],
            row["target"],
            row["backend"],
            tuple(row.get("dimensions", ())),
            row["test_regex"],
        )
        for row in data["check"]
        if row["requirement"] == "cell_local_ordinary" and row["kind"] == "ctest"
    }
    assert {row["issue"] for row in data["check"]} == set(data["issues"])
    assert {row["backend"] for row in data["check"]} == {"serial", "mpi", "openmp"}
    assert all(row["allocation"] in {"none", "required"} for row in data["check"])
    assert any(row["allocation"] == "required" for row in data["check"])
    allocation_proof_rows = [
        row
        for row in data["check"]
        if all(row.get(key) == value for key, value in runner.ALLOCATION_PROOF_ROW.items())
    ]
    assert len(allocation_proof_rows) == 1
    assert allocation_proof_rows[0]["backend"] == "serial"
    assert {row["requirement"] for row in data["check"]} >= {
        "solve_outcome",
        "transaction_authority",
        "prepared_installation",
        "resource_plan_ceiling",
        "lowering_refusal",
    }
    assert {
        row["nodeid"]
        for row in data["check"]
        if row["requirement"] == "lowering_refusal"
    } == runner.REQUIRED_LOWERING_REFUSAL_NODEIDS
    assert {
        (row["polarity"], row["test_regex"])
        for row in data["check"]
        if row["requirement"] == "transaction_authority"
    } == {
        ("positive", r"^ProgramTransaction\.SuccessfulEffectsAndReceiptsAreExactOnce$"),
        ("refusal", r"^ProgramTransaction\.SolveGuardAndAtomicSealConsensusFaultsRollback$"),
        (
            "refusal",
            r"^ProgramTransaction\.FrozenEffectSlotsRefuseOutOfOrderAndMissingSubmissionsCollectively$",
        ),
        ("refusal", r"^test_mpi_program_transaction_effect_consensus_np2$"),
        (
            "positive",
            r"^ProgramTransaction\.ForeignAcceptedReaderRejectsWriterWithoutBlockingCollectiveProgress$",
        ),
        (
            "positive",
            r"^SystemTransactionAuthority\.ConcurrentAcceptedReaderBlocksDuringCandidate$",
        ),
        (
            "positive",
            r"^AmrTransactionAuthority\.FieldCandidateSavepointRestoresAcceptedImagesWithoutHotAllocation$",
        ),
        (
            "positive",
            r"^AmrTransactionAuthority\.FieldCandidateSavepointAdoptsForwardTopologyWithoutPostSealAllocation$",
        ),
    }
    assert {
        (row["polarity"], row["target"], row["test_regex"])
        for row in data["check"]
        if row["requirement"] == "resource_plan_ceiling"
    } == {
        (
            "positive",
            "resource_plan_ceiling@test_nd_flux_ledger",
            r"^test_nd_flux_ledger\.interface_ledger_resident_footprint_covers_dense_and_hot_carriers$",
        ),
        (
            "refusal",
            "resource_plan_ceiling@test_program_host_descriptor",
            r"^PreparedProgramInstallation\.SealsAnEmptyValuePlanWithHostCapacityExactly$",
        ),
        (
            "positive",
            "resource_plan_ceiling@test_generated_amr_system_block",
            r"^GeneratedAmrSystemBlock\.ForwardSubcyclingStorageCeilingAcceptsConfiguredAPlusB$",
        ),
        (
            "refusal",
            "resource_plan_ceiling@test_generated_amr_system_block",
            r"^GeneratedAmrSystemBlock\.ForwardSubcyclingStorageCeilingRejectsOverflowWithoutClobber$",
        ),
        (
            "positive",
            "resource_plan_ceiling@test_generated_amr_system_block",
            r"^test_generated_amr_system_block_np2$",
        ),
    }
    amr_transaction_hot_path_rows = {
        (row["polarity"], row["target"], row["test_regex"])
        for row in data["check"]
        if row["target"] == "hot_path_allocation@test_amr_transaction_authority"
    }
    # Keep the currently authenticated CTest witnesses mandatory while allowing this source-owned
    # authority suite to grow with additional exact allocation proofs.  The runner still validates
    # every extra row's target, source ownership, and exact selector.
    assert amr_transaction_hot_path_rows >= {
        (
            "positive",
            "hot_path_allocation@test_amr_transaction_authority",
            r"^AmrTransactionAuthority\.AllCancelStaticFluxBasisIsHotAllocationFreeAndRetrySafeAfterResidentWriteFailure$",
        ),
        (
            "refusal",
            "hot_path_allocation@test_amr_transaction_authority",
            r"^AmrTransactionAuthority\.StaticFluxCommitOutsideAdvanceHierarchyRefusesBeforeSnapshotOrMutation$",
        ),
        (
            "positive",
            "hot_path_allocation@test_amr_transaction_authority",
            r"^AmrTransactionAuthority\.RefinedProgramCouplingIsHotAllocationFreeAndRollbackExactWithBoundedDispatch$",
        ),
    }
    assert {
        (row["polarity"], row["test_regex"])
        for row in data["check"]
        if row["requirement"] == "prepared_installation"
    } == {
        ("positive", r"^PreparedProgramInstallation\.RetainsOneImmutableDescriptionAndPreparationWitness$"),
        ("refusal", r"^PreparedProgramInstallation\.RefusesAnOwnerThatWasNotPrepared$"),
    }
    assert any(
        row["requirement"] == "abi_identity"
        and row["polarity"] == "positive"
        and row["target"] == "abi_identity@test_program_abi_symbols"
        for row in data["check"]
    )
    assert {
        (row["polarity"], row["target"], row["test_regex"])
        for row in data["check"]
        if row["requirement"] == "abi_identity"
        and row["kind"] == "ctest"
        and row["target"]
        in {
            "abi_identity@test_program_loader",
            "abi_identity@test_program_execution_services_schur_free",
        }
    } == {
        (
            "refusal",
            "abi_identity@test_program_execution_services_schur_free",
            r"^ProgramInstallationTables\.RejectsMalformedAbiTableShapesAndContents$",
        ),
        (
            "positive",
            "abi_identity@test_program_loader",
            r"^test_program_loader\.Runs$",
        ),
        (
            "positive",
            "abi_identity@test_program_execution_services_schur_free",
            r"^ProgramInstallationTables\.MaterializesEveryDsoViewBeforePreparation$",
        ),
    }
    assert {
        (row["polarity"], row["test_regex"])
        for row in data["check"]
        if row["requirement"] == "solve_outcome"
    } == {
        (
            "positive",
            r"^SolveOutcomeContract\.exhaustive_statuses_are_move_only_exact_once_and_never_publish_early$",
        ),
        (
            "refusal",
            r"^NewtonRobustnessTest\.publication_layout_failure_does_not_consume_the_outcome$",
        ),
        (
            "refusal",
            r"^StepAttemptRejectedHeaderOnly\.MailboxRejectsNonCanonicalRawRecordsBeforeAdoption$",
        ),
    }
    assert {
        (row["polarity"], row["target"], row["test_regex"])
        for row in data["check"]
        if row["requirement"] == "transaction_authority"
        and row["kind"] == "ctest"
        and row["polarity"] == "positive"
    } >= {
        (
            "positive",
            "transaction_authority@test_program_transaction",
            r"^ProgramTransaction\.ForeignAcceptedReaderRejectsWriterWithoutBlockingCollectiveProgress$",
        ),
        (
            "positive",
            "transaction_authority@test_system_transaction_authority",
            r"^SystemTransactionAuthority\.ConcurrentAcceptedReaderBlocksDuringCandidate$",
        ),
    }
    assert {
        (row["issue"], row["target"], row["backend"], row["allocation"], row["test_regex"])
        for row in data["check"]
        if row["target"] == "hot_path_allocation@test_step_change_l2_uniform_allocation"
    } == {
        (
            "ADC-720",
            "hot_path_allocation@test_step_change_l2_uniform_allocation",
            "serial",
            "none",
            r"^StepChangeL2GlobalAllocationProbe\.PreparedUniformSingleAndMultiBlockCallsAllocateNothing$",
        ),
        (
            "ADC-720",
            "hot_path_allocation@test_step_change_l2_uniform_allocation",
            "serial",
            "none",
            r"^StepChangeL2GlobalAllocationProbe\.PreparedTwoLevelAmrSingleAndMultiBlockCallsAllocateNothing$",
        ),
    }
    assert {
        (row["issue"], row["polarity"], row["target"], row["backend"], row["allocation"], row["test_regex"])
        for row in data["check"]
        if row["target"] == "transaction_authority@test_amr_transaction_authority"
    } == {
        (
            "ADC-702",
            "positive",
            "transaction_authority@test_amr_transaction_authority",
            "serial",
            "none",
            r"^AmrTransactionAuthority\.FieldCandidateSavepointRestoresAcceptedImagesWithoutHotAllocation$",
        ),
        (
            "ADC-702",
            "positive",
            "transaction_authority@test_amr_transaction_authority",
            "serial",
            "none",
            r"^AmrTransactionAuthority\.FieldCandidateSavepointAdoptsForwardTopologyWithoutPostSealAllocation$",
        ),
    }
    assert {
        (row["kind"], row["backend"], row["target"], row["nodeid"])
        for row in data["check"]
        if row["kind"] == "mpi_orchestrator"
    } == {
        (
            "mpi_orchestrator",
            "mpi",
            "strict_restart",
            "tests/python/integration/mpi/test_amr_rank_change_restart.py::"
            "test_amr_checkpoint_restart_rematerializes_two_ranks_onto_one",
        )
    }
    assert runner._python_mpi_orchestrators() == {
        "tests/python/integration/mpi/test_amr_rank_change_restart.py"
    }
    assert {
        row["nodeid"]
        for row in data["check"]
        if row["requirement"] == "strict_restart"
        and row["kind"] == "pytest"
    } >= {
        "tests/python/unit/runtime/test_amr_checkpoint_contract.py::test_uniform_and_amr_payload_versions_are_exact_current_integer_scalars",
        "tests/python/unit/runtime/test_amr_checkpoint_contract.py::test_historical_version_refusal_happens_before_restart_transaction",
    }


def test_amr_noop_regrid_seals_cell_temporal_diagnostics_in_hidden_publish():
    source = (ROOT / "src/runtime/amr/amr_system.cpp").read_text(encoding="utf-8")
    branch = source.split("if (!candidate.topology_changed) {", 1)[1].split(
        "retain_topology_authority_noexcept", 1
    )[0]

    publication = "publish_transaction_diagnostics_noexcept();"
    assert publication in branch
    assert branch.index(publication) < branch.index("return true;")


def test_runtime_authority_cli_is_closed_and_lists_the_mpi_targets():
    closed = subprocess.run(
        [sys.executable, str(RUNNER), "--check-only", "--dim", "2"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert closed.returncode == 0
    assert "Runtime authority gate source matrix: CLOSED" in closed.stdout

    targets = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--list-ctest-targets",
            "--backend",
            "mpi",
            "--dim",
            "2",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert targets.returncode == 0
    assert "test_program_transaction" in targets.stdout

    runner = _load_runner()
    data = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    required_targets = runner._required_ctest_targets(data["check"], backend="mpi", dimension=2)
    assert "test_program_execution_services_contract" in required_targets
    assert "test_mpi_cell_temporal_program_refusal" in required_targets


def test_runtime_authority_dimension_qualification_filters_mpi_rows_and_targets():
    runner = _load_runner()
    data = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))

    selected = {
        dimension: runner._selected_checks(
            data["check"], backend="mpi", dimension=dimension
        )
        for dimension in (1, 2, 3)
    }
    assert len(selected[1]) == len(selected[2]) == len(selected[3]) - 1

    corner_target = "test_mpi_amr_program_3d_corner_authority"
    assert corner_target not in runner._required_ctest_targets(
        data["check"], backend="mpi", dimension=1
    )
    assert corner_target in runner._required_ctest_targets(
        data["check"], backend="mpi", dimension=3
    )


def test_runtime_authority_pins_all_cell_temporal_mpi_positive_selectors_in_every_dimension():
    runner = _load_runner()
    data = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = {
        row["target"].split("@", 1)[1]: row
        for row in data["check"]
        if row["kind"] == "ctest"
        and row["backend"] == "mpi"
        and row["requirement"] == "cell_local_ordinary"
    }
    expected_positive = {
        "test_mpi_cell_temporal_program": "^test_mpi_cell_temporal_program_np2$",
        "test_mpi_cell_temporal_program_multibox": "^test_mpi_cell_temporal_program_multibox_np2$",
        "test_mpi_cell_temporal_program_collective_rollback": (
            "^test_mpi_cell_temporal_program_collective_rollback_np2$"
        ),
    }
    assert {
        name: (row["polarity"], row["dimensions"], row["test_regex"])
        for name, row in rows.items()
        if name in expected_positive
    } == {
        name: ("positive", [1, 2, 3], selector)
        for name, selector in expected_positive.items()
    }

    refusal = rows["test_mpi_cell_temporal_program_refusal"]
    assert refusal["polarity"] == "refusal"
    assert refusal["test_regex"] == "^test_mpi_cell_temporal_program_refusal_np2$"
    assert "dimensions" not in refusal
    assert data["deferred"] == []

    suites = runner._cpp_suites()
    for target, selector in expected_positive.items():
        suite = suites[target]
        assert suite["sources"] == [
            "tests/cpp/integration/mpi/test_mpi_cell_temporal_program.cpp"
        ]
        assert suite["labels"] == ["backend", "mpi", "medium"]
        assert suite["mpi_nproc"] == [2]
        assert runner._registered_ctest_cases(target, suite) == {target + "_np2"}

    expected_targets = set(expected_positive)
    for dimension in (1, 2, 3):
        selected = runner._selected_checks(
            data["check"], backend="mpi", dimension=dimension
        )
        selected_rows = {
            row["target"].split("@", 1)[1]: row
            for row in selected
            if row["kind"] == "ctest"
            and row["backend"] == "mpi"
            and row["requirement"] == "cell_local_ordinary"
            and row["polarity"] == "positive"
        }
        assert set(selected_rows) == expected_targets
        assert {
            target: (row["dimensions"], row["test_regex"])
            for target, row in selected_rows.items()
        } == {
            target: ([1, 2, 3], selector)
            for target, selector in expected_positive.items()
        }
        assert expected_targets <= set(
            runner._required_ctest_targets(
                data["check"], backend="mpi", dimension=dimension
            )
        )


def test_runtime_authority_check_only_is_source_only_and_does_not_require_a_dimension(
    monkeypatch,
):
    runner = _load_runner()
    monkeypatch.delenv("POPS_NATIVE_DIM", raising=False)

    assert runner.main(["--check-only"]) == 0


def test_runtime_authority_selected_dimension_accepts_explicit_or_authenticated_environment_value(
    monkeypatch,
):
    runner = _load_runner()
    monkeypatch.delenv("POPS_NATIVE_DIM", raising=False)

    assert runner._selected_native_dimension(2) == 2

    monkeypatch.setenv("POPS_NATIVE_DIM", "3")
    assert runner._selected_native_dimension(None) == 3
    with pytest.raises(ValueError, match="conflicting native dimensions"):
        runner._selected_native_dimension(1)

    monkeypatch.setenv("POPS_NATIVE_DIM", "4")
    with pytest.raises(ValueError, match="POPS_NATIVE_DIM"):
        runner._selected_native_dimension(None)


def test_runtime_authority_composed_ctest_path_filters_rows_and_propagates_dimension(
    monkeypatch, tmp_path
):
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    build_dir = tmp_path / "build"
    manifest = tmp_path / "tests/gates/m3.toml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("", encoding="utf-8")

    calls = []

    class FakeModule:
        @staticmethod
        def validate_manifest(path):
            assert path == manifest
            return {
                "check": [
                    {"kind": "ctest", "target": "all", "test_regex": "all"},
                    {
                        "kind": "ctest",
                        "target": "dim1",
                        "test_regex": "dim1",
                        "dimensions": [1],
                    },
                    {
                        "kind": "ctest",
                        "target": "dim3",
                        "test_regex": "dim3",
                        "dimensions": [3],
                    },
                ]
            }, []

        @staticmethod
        def _run_ctest(build, target, selector):
            calls.append((build, target, selector, os.environ.get("POPS_NATIVE_DIM")))

    monkeypatch.setattr(runner, "_load_composed_runner", lambda *_args: FakeModule)
    environment = {"POPS_NATIVE_DIM": "1", "POPS_TEST_COMPOSED": "yes"}

    runner._run_composed_gate(
        "tests/gates/m3.toml",
        "scripts/run_m3_gate.py",
        build_dir=build_dir,
        mpi_exec="mpiexec",
        environment=environment,
        python_only=False,
        ctest_only=True,
        dimension=1,
    )

    assert calls == [
        (build_dir, "all", "all", "1"),
        (build_dir, "dim1", "dim1", "1"),
    ]
    assert os.environ.get("POPS_TEST_COMPOSED") is None


def test_runtime_authority_composed_m3_receives_explicit_dimension(monkeypatch, tmp_path):
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    manifest = tmp_path / "tests/gates/m3.toml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("", encoding="utf-8")
    calls = []

    class FakeModule:
        @staticmethod
        def validate_manifest(path):
            assert path == manifest
            return {"check": []}, []

        @staticmethod
        def main(argv):
            calls.append((argv, os.environ.get("POPS_NATIVE_DIM")))
            return 0

    monkeypatch.setattr(runner, "_load_composed_runner", lambda *_args: FakeModule)
    runner._run_composed_gate(
        "tests/gates/m3.toml",
        "scripts/run_m3_gate.py",
        build_dir=tmp_path / "build",
        mpi_exec="mpiexec",
        environment={"POPS_NATIVE_DIM": "2"},
        python_only=True,
        ctest_only=False,
        dimension=2,
    )

    assert calls == [
        (
            [
                "--manifest",
                str(manifest),
                "--build-dir",
                str(tmp_path / "build"),
                "--dim",
                "2",
                "--mpi-exec",
                "mpiexec",
                "--python-only",
            ],
            "2",
        )
    ]


@pytest.mark.parametrize(
    ("replacement", "message"),
    (
        ("dimensions = 3", "dimensions must be a list"),
        ("dimensions = []", "dimensions must be non-empty"),
        ("dimensions = [true]", "bool is not accepted"),
        ("dimensions = [0]", "only supported values"),
        ("dimensions = [3, 3]", "dimensions must contain unique values"),
        ("dimensions = [3, 1]", "canonical sorted order"),
    ),
)
def test_runtime_authority_rejects_invalid_dimension_qualifier(
    tmp_path, replacement, message
):
    runner = _load_runner()
    path = _mutated_manifest(tmp_path, "dimensions = [3]", replacement)

    _data, errors = runner.audit_manifest(path)

    assert any(message in error for error in errors)


def test_runtime_authority_rejects_a_deferred_row(tmp_path):
    runner = _load_runner()
    path = _mutated_manifest(tmp_path, "deferred = []", 'deferred = ["ADC-720/cell_local_ordinary"]')
    _data, errors = runner.audit_manifest(path)
    assert any("deferred = []" in error for error in errors)


def test_runtime_authority_rejects_a_fake_nodeid(tmp_path):
    runner = _load_runner()
    path = _mutated_manifest(
        tmp_path,
        "test_removed_public_modules_do_not_exist",
        "test_removed_public_modules_do_not_exist_typo",
    )
    _data, errors = runner.audit_manifest(path)
    assert any("missing test function" in error for error in errors)


def test_runtime_authority_rejects_a_wildcard_ctest_selector(tmp_path):
    runner = _load_runner()
    path = _mutated_manifest(
        tmp_path,
        "^ProgramHostDescriptor\\\\.UniformRequiresTaggedPreparationImageBeforeProviderMaterialization$",
        "^ProgramHostDescriptor\\\\..*$",
    )
    _data, errors = runner.audit_manifest(path)
    assert any("not one exact source-registered case" in error for error in errors)


def test_runtime_authority_rejects_source_gtest_name_for_no_discover_mpi_suite(tmp_path):
    runner = _load_runner()
    path = _mutated_manifest(
        tmp_path,
        "^test_mpi_program_transaction_effect_consensus_np2$",
        r"^test_mpi_program_transaction_effect_consensus\\.RankLocalPrepareFailureRollsBackEveryRank$",
    )
    _data, errors = runner.audit_manifest(path)
    assert any("not one exact source-registered case" in error for error in errors)


def test_runtime_authority_rejects_lowering_selector_drift(tmp_path):
    runner = _load_runner()
    path = _mutated_manifest(
        tmp_path,
        "tests/python/unit/codegen/test_scheduler_codegen.py::test_scratch_skip_refuses_unprepared_stale_state",
        "tests/python/unit/codegen/test_scheduler_codegen.py::test_scratch_zero_sets_the_scratch_to_zero",
    )
    _data, errors = runner.audit_manifest(path)
    assert any("lowering_refusal must pin exactly" in error for error in errors)


@pytest.mark.parametrize(
    ("requirement", "old", "new"),
    (
        (
            "legacy_context_barrier",
            "tests/python/architecture/test_program_only_temporal_facades.py::test_system_temporal_facades_dispatch_only_through_an_installed_program",
            "tests/python/architecture/test_program_only_temporal_facades.py::test_ssprk_semantics_have_only_typed_python_program_authority",
        ),
        (
            "legacy_symbol_barrier",
            "tests/python/architecture/test_no_legacy_runtime_routes.py::test_program_has_one_runtime_branch_spelling",
            "tests/python/architecture/test_program_only_temporal_facades.py::test_ssprk_semantics_have_only_typed_python_program_authority",
        ),
        (
            "legacy_fragment_barrier",
            "tests/python/architecture/test_amr_program_support_parity.py::test_context_include_parser_authenticates_both_delimiters_and_hidden_fragments",
            "tests/python/architecture/test_program_only_temporal_facades.py::test_ssprk_semantics_have_only_typed_python_program_authority",
        ),
        (
            "pending_marker_barrier",
            "tests/python/architecture/test_amr_program_support_parity.py::test_parser_finds_only_explicit_known_deferrals",
            "tests/python/architecture/test_program_only_temporal_facades.py::test_ssprk_semantics_have_only_typed_python_program_authority",
        ),
    ),
)
def test_runtime_authority_rejects_semantic_barrier_nodeid_substitution(
    tmp_path, requirement, old, new
):
    runner = _load_runner()
    path = _mutated_manifest(tmp_path, old, new)
    _data, errors = runner.audit_manifest(path)
    assert any("%s must pin exactly" % requirement in error for error in errors)


def test_runtime_authority_rejects_semantic_barrier_polarity_swap(tmp_path):
    runner = _load_runner()
    path = _mutated_manifest(
        tmp_path,
        'requirement = "legacy_context_barrier"\npolarity = "positive"',
        'requirement = "legacy_context_barrier"\npolarity = "refusal"',
    )
    _data, errors = runner.audit_manifest(path)
    assert any("legacy_context_barrier must pin each polarity" in error for error in errors)


def test_runtime_authority_rejects_a_contract_wildcard_selector(tmp_path):
    runner = _load_runner()
    path = _mutated_manifest(
        tmp_path,
        "^ProgramTransaction\\\\.SuccessfulEffectsAndReceiptsAreExactOnce$",
        "^ProgramTransaction\\\\..*$",
    )
    _data, errors = runner.audit_manifest(path)
    assert any("not one exact source-registered case" in error for error in errors)


def test_runtime_authority_rejects_missing_explicit_contract_polarity(tmp_path):
    runner = _load_runner()
    path = _mutated_manifest(
        tmp_path,
        'requirement = "solve_outcome"\npolarity = "positive"',
        'requirement = "gate_execution"\npolarity = "positive"',
    )
    _data, errors = runner.audit_manifest(path)
    assert any("solve_outcome lacks positive coverage" in error for error in errors)


def test_runtime_authority_rejects_an_optional_or_skipping_pytest(tmp_path):
    runner = _load_runner()
    path = _mutated_manifest(
        tmp_path,
        "tests/python/architecture/test_program_only_temporal_facades.py::test_ssprk_semantics_have_only_typed_python_program_authority",
        "tests/python/unit/runtime/test_schedule_authoring.py::test_always_is_default_recompute",
    )
    _data, errors = runner.audit_manifest(path)
    assert any("importorskip" in error for error in errors)


def test_runtime_authority_source_barriers_reject_retired_contexts(monkeypatch, tmp_path):
    runner = _load_runner()
    authority = tmp_path / "program_abi.hpp"
    authority.write_text("kProgramInstallAbiVersion = 5; ProgramContext; pending:old", encoding="utf-8")
    monkeypatch.setattr(runner, "_authority_sources", lambda: (authority,))
    errors: list[str] = []
    runner._validate_source_barriers(errors)
    assert any("retains retired context names" in error for error in errors)
    assert any("pending marker" in error for error in errors)


@pytest.mark.parametrize(
    ("relative", "source", "needle"),
    (
        (
            "include/pops/runtime/program/detail/legacy.hpp",
            "struct ProgramContext {};\n",
            "retired context names",
        ),
        (
            "include/pops/runtime/program/detail/runtime_context.hpp",
            "struct RuntimeProgramContext {};\n",
            "retired context names",
        ),
        (
            "src/runtime/program/context_adapter.cpp",
            "using runtime_program_context_detail_generated_v2 = int;\n",
            "retired context names",
        ),
        (
            "src/runtime/amr/legacy.cpp",
            "extern void pops_install_program_amr();\n",
            "retired runtime symbols",
        ),
        (
            "src/runtime/amr/legacy_suffix.cpp",
            "extern void pops_install_program_amr_v5_system();\n",
            "retired runtime symbols",
        ),
        (
            "src/runtime/amr/legacy_version_prefix.cpp",
            "extern void pops_install_program_v5_system_amr_v7();\n",
            "retired runtime symbols",
        ),
        (
            "src/runtime/system/legacy_qualified_install.cpp",
            "extern void pops_install_program_system_v5();\n",
            "retired runtime symbols",
        ),
        (
            "include/pops/runtime/program/program_abi.hpp",
            "void seal_program_preparation_host();\n",
            "retired runtime symbols",
        ),
        (
            "include/pops/runtime/program/detail/legacy.inc",
            "// retired implementation fragment\n",
            "forbids AMR runtime fragments",
        ),
        (
            "src/runtime/amr/legacy.inc",
            "// retired implementation fragment outside the program directory\n",
            "forbids AMR runtime fragments",
        ),
        (
            "src/runtime/amr/fallback.cpp",
            "AmrRuntime<2>::step(dt);\n",
            "AmrRuntime::step",
        ),
        (
            "src/runtime/amr/legacy_engine.cpp",
            "AmrEngine<2>::step_level(dt);\n",
            "AmrRuntime::step",
        ),
        (
            "src/runtime/amr/legacy_runtime_alias.cpp",
            "Runtime<2>::advance_macro_step(dt);\n",
            "AmrRuntime::step",
        ),
        (
            "src/runtime/amr/legacy_level_engine_alias.cpp",
            "LevelRuntime::step_level(dt);\n",
            "AmrRuntime::step",
        ),
        (
            "src/runtime/amr/legacy_level_runtime.cpp",
            "amr_level_runtime->advance_level(dt);\n",
            "AmrRuntime::step",
        ),
        (
            "src/runtime/amr/legacy_runtime_member.cpp",
            "amr_runtime_.step_level(dt);\n",
            "AmrRuntime::step",
        ),
        (
            "src/runtime/amr/legacy_subcycling.cpp",
            "subcycling_engine.step(dt);\n",
            "AmrRuntime::step",
        ),
        (
            "src/runtime/amr/legacy_engine.cpp",
            "engine.step(dt);\n",
            "AmrRuntime::step",
        ),
        (
            "src/runtime/amr/legacy_runtime_alias.cpp",
            "using LegacyRuntime = pops::runtime::amr::AmrRuntime<2>;\n"
            "LegacyRuntime::advance_macro_step(dt);\n",
            "AmrRuntime::step",
        ),
        (
            "include/pops/runtime/program/cache_manager.hpp",
            "std::map<int, CacheSlot<2>> slots_;\n",
            "node-id-only scheduler cache",
        ),
        (
            "python/pops/runtime/pending.py",
            'marker = "pending:checkpointed_hierarchy_cache"\n',
            "pending:checkpointed_hierarchy_cache",
        ),
        (
            "include/pops/runtime/program/abi.hpp",
            "constexpr int kProgramInstallAbiVersion = 4;\n",
            "legacy Program install ABI",
        ),
        (
            "python/pops/runtime/abi.py",
            "kProgramInstallAbiVersion = 4\n",
            "legacy Program install ABI",
        ),
        (
            "include/pops/runtime/program/checkpoint.hpp",
            'const char* old = "POPSAST4";\n',
            "legacy checkpoint/magic",
        ),
        (
            "include/pops/runtime/amr/checkpoint_legacy.hpp",
            "constexpr int checkpoint_version = 8;\n",
            "legacy checkpoint/magic",
        ),
        (
            "include/pops/runtime/amr/checkpoint_legacy.hpp",
            "constexpr int amr_checkpoint_version = 11;\n",
            "legacy checkpoint/magic",
        ),
        (
            "include/pops/runtime/amr/checkpoint_legacy.hpp",
            "const char canonical_magic[] = {'P', 'O', 'P', 'S', 'A', 'N', 'D', '5'};\n"
            "const char old_magic[] = {'P', 'O', 'P', 'S', 'A', 'N', 'D', '4'};\n",
            "legacy checkpoint/magic",
        ),
        (
            "src/runtime/amr/parallel_table.cpp",
            "std::unordered_map<int, PreparedProgram> parallel_program_table;\n",
            "parallel runtime authority table",
        ),
        (
            "src/runtime/amr/parallel_dispatch.cpp",
            "void dispatch_parallel_program();\n",
            "parallel runtime authority dispatch",
        ),
        (
            "src/runtime/amr/generic_program_dispatch_table.cpp",
            "ProgramDispatchTable table;\n",
            "secondary Program runtime authority table",
        ),
        (
            "src/runtime/amr/generic_program_table.cpp",
            "int program_table;\n",
            "secondary Program runtime authority table",
        ),
        (
            "src/runtime/amr/generic_program_dispatch.cpp",
            "void program_dispatch();\n",
            "secondary Program runtime authority dispatch",
        ),
        (
            "src/runtime/amr/generic_program_dispatch_variant.cpp",
            "void program_dispatch_v2();\n",
            "secondary Program runtime authority dispatch",
        ),
        (
            "src/runtime/amr/generic_program_dispatch_table_variant.cpp",
            "ProgramDispatchTable_v2 table;\n",
            "secondary Program runtime authority table",
        ),
        (
            "src/runtime/amr/cell_local_installer.cpp",
            "void install_program_cell_local();\n",
            "cell-local installer",
        ),
        (
            "python/pops/runtime/cell_local_installer.py",
            "def install_program_cell_local():\n    pass\n",
            "cell-local installer",
        ),
        (
            "src/runtime/amr/cell_program_installer.cpp",
            "void pops_install_program_cell();\n",
            "cell-local installer",
        ),
        (
            "src/runtime/amr/cell_program_installer_unprefixed.cpp",
            "void install_program_cell();\n",
            "cell-local installer",
        ),
        (
            "src/runtime/amr/cell_temporal_installer.cpp",
            "void pops_install_cell_temporal();\n",
            "cell-local installer",
        ),
        (
            "src/runtime/amr/cell_temporal_installer.cpp",
            "void install_cell_temporal();\n",
            "cell-local installer",
        ),
        (
            "src/runtime/amr/cell_temporal_register.cpp",
            "void register_program_cell_temporal();\n",
            "cell-local installer",
        ),
        (
            "src/runtime/amr/cell_temporal_installer.cpp",
            "void register_cell_temporal_amr();\n",
            "cell-local installer",
        ),
        (
            "src/runtime/amr/cell_temporal_variant.cpp",
            "void install_cell_temporal_cuda_amr_v2();\n",
            "cell-local installer",
        ),
        (
            "src/runtime/amr/cell_program_installer.cpp",
            "void install_program_cell();\n",
            "cell-local installer",
        ),
        (
            "src/runtime/amr/cell_program_variant.cpp",
            "void register_program_cell_backend_v2();\n",
            "cell-local installer",
        ),
        (
            "src/runtime/amr/amr_program_installer.cpp",
            "void install_program_amr_v2();\n",
            "retired Program AMR installer route(s)",
        ),
        (
            "include/pops/runtime/program/installer.hpp",
            "void pops_install_field_boundaries_amr();\n",
            "retired split installers",
        ),
        (
            "include/pops/runtime/program/legacy_split.hpp",
            "void pops_program_route_manifest();\n"
            "void pops_program_dt_bound();\n"
            "void pops_install_field_boundaries();\n",
            "retired runtime symbols",
        ),
        (
            "include/pops/runtime/program/legacy_split_variant.hpp",
            "void pops_program_route_manifest_v7();\n"
            "void pops_program_dt_bound_system();\n"
            "void pops_install_field_boundaries_amr();\n",
            "retired runtime symbols",
        ),
        (
            "include/pops/runtime/program/legacy_short_split_variant.hpp",
            "void pops_program_routes();\n"
            "void pops_program_dt_cuda_amr_v2();\n"
            "void pops_program_boundaries_system();\n"
            "void pops_install_program_boundaries_amr();\n",
            "retired runtime symbols",
        ),
        (
            "include/pops/runtime/program/legacy_singular_split_variant.hpp",
            "void pops_program_route_amr_v2();\n"
            "void pops_program_boundary_cuda_amr_v2();\n"
            "void pops_program_field_boundary_system();\n"
            "void pops_install_field_boundary_backend_v3();\n"
            "void pops_install_program_routes_amr();\n",
            "retired runtime symbols",
        ),
        (
            "include/pops/runtime/program/legacy_generic_program_amr.hpp",
            "void pops_program_flux_amr_v2();\n"
            "void pops_register_program_flux_amr();\n",
            "retired runtime symbols",
        ),
        (
            "include/pops/runtime/program/legacy_split_multi_suffix.hpp",
            "void pops_program_route_manifest_amr_v7();\n"
            "void pops_program_route_manifest_cuda_amr();\n"
            "void pops_program_dt_bound_v2_system();\n"
            "void pops_program_dt_bound_generated_cuda_amr();\n"
            "void pops_install_field_boundaries_v3_amr();\n"
            "void pops_register_program_provider_routes_uniform_v4();\n"
            "void pops_program_route_manifest_cuda_legacy();\n"
            "void pops_program_dt_bound_backend_v2();\n"
            "void pops_install_field_boundaries_legacy_backend();\n",
            "retired runtime symbols",
        ),
        (
            "include/pops/runtime/program/legacy_program_install_variant.hpp",
            "void pops_install_program_cuda_amr_v5();\n",
            "retired runtime symbols",
        ),
        (
            "src/runtime/system/system_program.cpp",
            "void install_program_step();\n",
            "retired public Program installer routes",
        ),
        (
            "include/pops/runtime/program/cache_manager.hpp",
            "void node_ids();\n",
            "node-id-only scheduler cache",
        ),
    ),
)
def test_runtime_authority_recursive_barriers_refuse_explicit_patterns(
    monkeypatch, tmp_path, relative, source, needle
):
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    path = _write_synthetic_source(tmp_path, relative, source)
    errors: list[str] = []
    runner._scan_production_barriers(errors, (path,))
    assert any(needle in error for error in errors)


def test_runtime_authority_source_barriers_ignore_comments_and_docstrings(monkeypatch, tmp_path):
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    cpp = _write_synthetic_source(
        tmp_path,
        "include/pops/runtime/program/detail/comments.hpp",
        "// ProgramContext external_install_amr pending:checkpointed_hierarchy_cache\n"
        "/* AmrRuntime<2>::step() POPSAST4 */\n",
    )
    python = _write_synthetic_source(
        tmp_path,
        "python/pops/runtime/comments.py",
        '"""external_install_amr pending:checkpointed_hierarchy_cache"""\n',
    )
    neutral = _write_synthetic_source(
        tmp_path,
        "include/pops/runtime/program/detail/program_contextual.hpp",
        "",
    )
    errors: list[str] = []
    runner._scan_production_barriers(errors, (cpp, python, neutral))
    assert errors == []


def test_runtime_authority_allows_candidate_owned_generated_step(monkeypatch, tmp_path):
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    path = _write_synthetic_source(
        tmp_path,
        "src/runtime/amr/generated_program_candidate.cpp",
        "void generated_program_candidate(ProgramCandidateState* state, double dt) {\n"
        "  state->step(dt);\n"
        "  context.advance_same_level_cell_temporal(dt);\n"
        "}\n",
    )
    errors: list[str] = []
    runner._scan_production_barriers(errors, (path,))
    assert errors == []


def test_runtime_authority_allows_candidate_owned_generated_level_authority_step(
    monkeypatch, tmp_path
):
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    path = _write_synthetic_source(
        tmp_path,
        "src/runtime/amr/generated_level_program_candidate.cpp",
        "void generated_level_program_candidate(int level, double dt) {\n"
        "  _PopsAmrLevelProgramAuthority::programs.at(level).step(dt);\n"
        "}\n",
    )
    errors: list[str] = []
    runner._scan_production_barriers(errors, (path,))
    assert errors == []


def test_runtime_authority_allows_the_canonical_program_dispatch(monkeypatch, tmp_path):
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    path = _write_synthetic_source(
        tmp_path,
        "src/runtime/amr/canonical_program_dispatch.cpp",
        "void advance(ProgramRuntimeState& program, double dt) {\n"
        "  program.dispatch_cadence_step(time, macro_step, dt, tag);\n"
        "}\n",
    )
    errors: list[str] = []
    runner._scan_production_barriers(errors, (path,))
    assert errors == []


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        (
            "def legacy(amr_engine, dt):\n"
            "    amr_engine.step_level(dt)\n",
            True,
        ),
        (
            "def legacy(engine, dt):\n"
            "    engine.step(dt)\n",
            True,
        ),
        (
            "def legacy(runtime, dt):\n"
            "    runtime.step(dt)\n",
            True,
        ),
        (
            "def legacy(engine, dt):\n"
            "    legacy_runtime = engine\n"
            "    legacy_runtime.advance_level(dt)\n",
            True,
        ),
        (
            "def generated(candidate, dt):\n"
            "    candidate.step(dt)\n",
            False,
        ),
        (
            "def documentation():\n"
            "    return 'amr_engine.step_level(dt)'\n",
            False,
        ),
    ),
)
def test_runtime_authority_scopes_python_amr_step_bypass_to_engine_names(
    monkeypatch, tmp_path, source, expected
):
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    path = _write_synthetic_source(
        tmp_path,
        "python/pops/runtime/amr_step_scope.py",
        source,
    )
    errors: list[str] = []
    runner._scan_production_barriers(errors, (path,))
    assert any("AmrRuntime::step" in error for error in errors) is expected


def test_runtime_authority_allows_only_the_audited_python_program_facade_step(
    monkeypatch, tmp_path
):
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    path = _write_synthetic_source(
        tmp_path,
        "python/pops/runtime/_cadence_install.py",
        "def step_adaptive(engine, cfl):\n"
        '    """Advance through the installed Program."""\n'
        "    dt = float(cfl)\n"
        "    engine.step(dt)\n",
    )
    errors: list[str] = []
    runner._scan_production_barriers(errors, (path,))
    assert errors == []

    path.write_text(
        "def step_adaptive(engine, cfl):\n"
        '    """Advance through the installed Program."""\n'
        "    dt = float(cfl)\n"
        "    engine.step(cfl)\n",
        encoding="utf-8",
    )
    errors = []
    runner._scan_production_barriers(errors, (path,))
    assert any("AmrRuntime::step" in error for error in errors)


def test_runtime_authority_allows_non_authority_near_miss_names(monkeypatch, tmp_path):
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    path = _write_synthetic_source(
        tmp_path,
        "src/runtime/amr/generated_helpers.cpp",
        "struct ProgramDispatchTableOwned {};\n"
        "void program_dispatcher();\n"
        "struct ProgramDispatchTableCandidate {};\n"
        "void program_dispatch_candidate();\n"
        "void install_program_cellular();\n"
        "void pops_programmer_flux_amr();\n"
        "void register_provider_routes_amr();\n",
    )
    neutral_context = _write_synthetic_source(
        tmp_path,
        "include/pops/runtime/program/runtime_program_contextual.hpp",
        "",
    )
    errors: list[str] = []
    runner._scan_production_barriers(errors, (path, neutral_context))
    assert errors == []


def test_runtime_authority_rejects_generic_program_dispatch_authorities(
    monkeypatch, tmp_path
):
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    path = _write_synthetic_source(
        tmp_path,
        "src/runtime/amr/generic_program_dispatch_authorities.cpp",
        "struct ProgramDispatch {};\n"
        "struct ProgramDispatchTable {};\n"
        "void program_dispatch_for_level();\n"
        "int program_table_for_level;\n",
    )
    errors: list[str] = []
    runner._scan_production_barriers(errors, (path,))
    assert any("secondary Program runtime authority table" in error for error in errors)
    assert any("secondary Program runtime authority dispatch" in error for error in errors)


def test_runtime_authority_ignores_native_brick_abi_symbols(monkeypatch, tmp_path):
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    path = _write_synthetic_source(
        tmp_path,
        "src/runtime/amr/native_brick.cpp",
        "\n".join(
            [
                "extern void pops_install_native_amr();",
                "extern void pops_register_provider_routes_amr();",
                "extern void external_install_amr();",
                *("extern void %s();" % symbol for symbol in runner.NATIVE_BRICK_ABI_SYMBOLS),
                "",
            ]
        ),
    )
    errors: list[str] = []
    runner._scan_production_barriers(errors, (path,))
    assert errors == []


def test_runtime_authority_rejects_an_unallowlisted_brick_looking_program_route(
    monkeypatch, tmp_path
):
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    path = _write_synthetic_source(
        tmp_path,
        "src/runtime/amr/brick_looking_program_route.cpp",
        "extern void pops_brick_program_amr_v2();\n",
    )
    errors: list[str] = []
    runner._scan_production_barriers(errors, (path,))
    assert any("retired runtime symbols" in error for error in errors)


def test_runtime_authority_rejects_each_program_amr_abi_symbol(monkeypatch, tmp_path):
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    path = _write_synthetic_source(
        tmp_path,
        "python/pops/codegen/legacy_program_symbols.py",
        "\n".join(
            "SYMBOL = %r" % token
            for token in runner.LEGACY_PROGRAM_AMR_SYMBOL_TOKENS
        )
        + "\n",
    )
    errors: list[str] = []
    runner._scan_production_barriers(errors, (path,))
    symbol_errors = [error for error in errors if "retired runtime symbols" in error]
    assert len(symbol_errors) == 1
    assert all(token in symbol_errors[0] for token in runner.LEGACY_PROGRAM_AMR_SYMBOL_TOKENS)


def test_runtime_authority_rejects_unversioned_and_targeted_program_split_abi_variants(
    monkeypatch, tmp_path
):
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    variants = (
        *runner.LEGACY_PROGRAM_SPLIT_SYMBOL_TOKENS,
        "pops_program_route_manifest_amr",
        "pops_program_dt_bound_v7",
        "pops_install_field_boundaries_system",
        "pops_register_program_provider_routes_amr",
        "pops_program_route_manifest_amr_v7",
        "pops_program_route_manifest_cuda_amr",
        "pops_program_dt_bound_v2_system",
        "pops_program_dt_bound_generated_cuda_amr",
        "pops_install_field_boundaries_v3_amr",
        "pops_register_program_provider_routes_uniform_v4",
        "pops_program_route_manifest_cuda_legacy",
        "pops_program_dt_bound_backend_v2",
        "pops_install_field_boundaries_legacy_backend",
    )
    path = _write_synthetic_source(
        tmp_path,
        "python/pops/codegen/legacy_program_split_variants.py",
        "\n".join("SYMBOL_%d = %r" % (index, token) for index, token in enumerate(variants))
        + "\n",
    )
    errors: list[str] = []
    runner._scan_production_barriers(errors, (path,))
    symbol_errors = [error for error in errors if "retired runtime symbols" in error]
    assert len(symbol_errors) == 1
    assert all(token in symbol_errors[0] for token in variants)

    install_variant = _write_synthetic_source(
        tmp_path,
        "python/pops/codegen/legacy_program_install_variant.py",
        "SYMBOL = 'pops_install_program_cuda_amr_v5'\n",
    )
    errors = []
    runner._scan_production_barriers(errors, (install_variant,))
    assert any(
        "pops_install_program_cuda_amr_v5" in error
        for error in errors
        if "retired runtime symbols" in error
    )


@pytest.mark.parametrize(
    "relative",
    (
        "include/pops/runtime/program/detail/program_context.hpp",
        "include/pops/runtime/program/detail/amr_program_context.hpp",
        "include/pops/runtime/program/detail/runtime_program_context.hpp",
        "include/pops/runtime/program/detail/runtime_program_context_detail.hpp",
        "include/pops/runtime/program/detail/runtime_program_context_v2.hpp",
        "include/pops/runtime/program/detail/runtime_program_context_extra.hpp",
        "include/pops/runtime/program/detail/runtime-program-context-detail.hpp",
        "include/pops/runtime/program/detail/runtime-program-context-v2.hpp",
        "include/pops/runtime/program/detail/AMR-Program-Context.hpp",
        "include/pops/runtime/program/detail/RuntimeProgramContext.hpp",
        "include/pops/runtime/program/detail/RuntimeProgramContextDetail.hpp",
    ),
)
def test_runtime_authority_rejects_retired_context_filenames(monkeypatch, tmp_path, relative):
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    path = _write_synthetic_source(
        tmp_path,
        relative,
        "",
    )
    errors: list[str] = []
    runner._scan_production_barriers(errors, (path,))
    assert any("retired ProgramContext source remains present" in error for error in errors)


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        (
            "def serialize_cache(row):\n"
            "    payload = {'node_id': row['node_id'], 'cache_nodes': [0]}\n"
            "    return payload\n",
            True,
        ),
        (
            "def serialize_cache(row):\n"
            "    return {'cache_node_id': row['node_id']}\n",
            True,
        ),
        (
            "def emit_cache(row):\n"
            "    out = {}\n"
            "    out['cache_nodes'] = row['cache_slots']\n"
            "    return out\n",
            True,
        ),
        (
            "def emit(row):\n"
            "    out = {}\n"
            "    out.setdefault('cache_nodes', row['cache_slots'])\n"
            "    return out\n",
            True,
        ),
        (
            "def emit(row):\n"
            "    out = {}\n"
            "    out.__setitem__('cache_node_id', row['node_id'])\n"
            "    return out\n",
            True,
        ),
        (
            "def emit(row):\n"
            "    temporary = {'cache_nodes': row['cache_slots']}\n"
            "    return temporary\n",
            True,
        ),
        (
            "def emit(row):\n"
            "    temporary = {}\n"
            "    temporary.update({'cache_node_id': row['node_id']})\n"
            "    return temporary\n",
            True,
        ),
        (
            "def serialize_cache_row(row):\n"
            "    return {'node_id': row['node_id']}\n",
            True,
        ),
        (
            "def serialize(row):\n"
            "    return {'node_id': row['node_id']}\n",
            True,
        ),
        (
            "def reject_legacy(row):\n"
            "    if 'cache_nodes' in row or 'cache_node_id' in row:\n"
            "        raise ValueError('legacy cache wire')\n"
            "    return row.get('node_id')\n",
            False,
        ),
        (
            "def reject_legacy(row):\n"
            "    raise ValueError({'cache_nodes': row['cache_nodes']})\n",
            False,
        ),
        (
            "def resolve_control_node(row):\n"
            "    return {'node_id': row['node_id'], 'cache_slots': row['cache_slots']}\n",
            False,
        ),
        (
            "def emit_cache(row):\n"
            "    return dict(node_id=row['node_id'], cache_nodes=row['cache_slots'])\n",
            True,
        ),
        (
            "def serialize_cache_id(row):\n"
            "    return dict(node_id=row['node_id'])\n",
            True,
        ),
        (
            "def resolve_node(row):\n"
            "    return dict(node_id=row['node_id'])\n",
            False,
        ),
        (
            "def build_cache_authority(row):\n"
            "    return {'node_id': row['node_id']}\n",
            True,
        ),
        (
            "def store_cache_authority(row):\n"
            "    cache_plan = {}\n"
            "    cache_plan['node_id'] = row['node_id']\n"
            "    return cache_plan\n",
            True,
        ),
        (
            "def resolve_node(row):\n"
            "    row['node_id'] = row['node_id']\n"
            "    return row\n",
            False,
        ),
        (
            "def inspect_legacy(row):\n"
            "    return row.get('cache_nodes')\n",
            False,
        ),
        (
            "def reject_cache_authority(row):\n"
            "    if 'node_id' in row:\n"
            "        raise ValueError('legacy cache authority')\n"
            "    return row\n",
            False,
        ),
    ),
)
def test_runtime_authority_scopes_python_cache_wire_barrier_to_serialization(
    monkeypatch, tmp_path, source, expected
):
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    path = _write_synthetic_source(
        tmp_path,
        "python/pops/runtime/cache_wire_scope.py",
        source,
    )
    errors: list[str] = []
    runner._scan_production_barriers(errors, (path,))
    cache_errors = [error for error in errors if "node-id-only scheduler cache wire" in error]
    assert bool(cache_errors) is expected


def test_runtime_authority_rejects_direct_facade_provider_construction(monkeypatch, tmp_path):
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    header = _write_synthetic_source(
        tmp_path,
        "include/pops/runtime/program/program_execution_services.hpp",
        "template <int Dim> class ProgramExecutionServices {\n"
        "  ProgramExecutionServices(System<Dim>* system);\n"
        "};\n"
        "template <int Dim> auto make_program_execution_provider(System<Dim>* system);\n"
        "template <int Dim> auto make_program_execution_provider("
        "const ProgramPreparationHostRef& preparation);\n",
    )
    monkeypatch.setattr(runner, "_production_sources", lambda: (header,))
    errors: list[str] = []
    runner._validate_detached_provider_architecture(errors)
    assert any(
        "direct ProgramExecutionServices(System*/AmrSystem*) constructor" in error
        for error in errors
    )
    assert any(
        "direct make_program_execution_provider(System*/AmrSystem*) factory" in error
        for error in errors
    )


def test_runtime_authority_rejects_dead_facade_provider_emission(monkeypatch, tmp_path):
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    source = _write_synthetic_source(
        tmp_path,
        "python/pops/codegen/program_emit_field_boundaries.py",
        'GENERATED = "make_program_execution_provider(sys)"\n',
    )
    monkeypatch.setattr(runner, "_production_sources", lambda: (source,))
    errors: list[str] = []
    runner._validate_detached_provider_architecture(errors)
    assert any("direct make_program_execution_provider facade call" in error for error in errors)


def test_runtime_authority_rejects_removed_transaction_and_plan_aliases(monkeypatch, tmp_path):
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    _write_synthetic_source(
        tmp_path,
        "include/pops/runtime/program/program_transaction.hpp",
        "class ProgramTransactionRegistry final {\n"
        "  using Phase = ProgramTransactionPhase;\n"
        "  void add_participant();\n"
        "  void try_add_participant();\n"
        "  void freeze();\n"
        "  bool frozen();\n"
        "  std::size_t frozen_effect_capacity();\n"
        "  void begin_step();\n"
        "  const int* read();\n"
        "  const int* provisional_read();\n"
        " private:\n"
        "};\n"
        "auto old_phase = ProgramTransactionPhase::Snapshot;\n",
    )
    _write_synthetic_source(
        tmp_path,
        "include/pops/runtime/program/owned_program_installation.hpp",
        "using ResourcePlan = ProgramResourcePlan;\n"
        "using LegacyResourceTuple = std::tuple<int, int>;\n",
    )
    _write_synthetic_source(
        tmp_path,
        "python/pops/codegen/program_persistent_plan.py",
        "ResourcePlan = ProgramResourcePlan\n"
        "PersistentValueTuple = tuple[int, ...]\n"
        "def persistent_slot(program, value):\n"
        "    return int(value.id)\n",
    )

    errors: list[str] = []
    runner._validate_retired_compatibility_aliases(errors)
    assert any("retired transaction compatibility methods" in error for error in errors)
    assert any("non-canonical transaction enum spellings" in error for error in errors)
    assert any("retired transaction type aliases" in error for error in errors)
    assert any("retired registry read aliases" in error for error in errors)
    assert any("retired resource-plan type aliases" in error for error in errors)
    assert any("legacy resource tuple alias" in error for error in errors)
    assert any("retired lowering aliases" in error for error in errors)
    assert any("legacy persistent tuple alias" in error for error in errors)
    assert any("numeric persistent_slot fallback" in error for error in errors)


def test_runtime_authority_requires_explicit_install_transaction_contract(monkeypatch, tmp_path):
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    _write_synthetic_source(
        tmp_path,
        "include/pops/runtime/program/owned_program_installation.hpp",
        "class PreparedProgramInstallation {};\n",
    )
    _write_synthetic_source(
        tmp_path,
        "include/pops/runtime/program/program_transaction.hpp",
        "class ProgramTransaction {};\n"
        "class AcceptedReadLease {};\n"
        "auto acquire_read();\n",
    )
    _write_synthetic_source(
        tmp_path,
        "include/pops/runtime/program/program_runtime_state.hpp",
        "class PreparedArtifactPublication {};\n"
        "void install_prepared_artifact();\n"
        "void publish_prepared_artifact_(int) noexcept;\n",
    )
    _write_synthetic_source(
        tmp_path,
        "include/pops/runtime/program/program_execution_services.hpp",
        "SolveOutcome outcome;\n",
    )
    _write_synthetic_source(
        tmp_path,
        "include/pops/runtime/program/amr_program_checkpoint.hpp",
        "char magic[] = {'P', 'O', 'P', 'S', 'A', 'N', 'D', '5'};\n",
    )
    _write_synthetic_source(
        tmp_path,
        "include/pops/runtime/system/auxiliary_checkpoint.hpp",
        "char magic[] = {'P', 'O', 'P', 'S', 'A', 'U', 'X', '2'};\n",
    )
    _write_synthetic_source(
        tmp_path,
        "include/pops/runtime/amr/persistent_tagging_state.hpp",
        "char magic[] = {'P', 'O', 'P', 'S', 'H', 'Y', 'S', '2'};\n",
    )

    errors: list[str] = []
    runner._validate_required_authority_contract(errors)
    assert any(
        "program_transaction.hpp" in error and "atomic_seal" in error
        for error in errors
    )


def test_runtime_authority_requires_canonical_checkpoint_magics(monkeypatch, tmp_path):
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    for relative in runner.CANONICAL_CHECKPOINT_MAGIC_TOKENS:
        _write_synthetic_source(
            tmp_path,
            relative,
            "char magic[] = {'X', 'X', 'X', 'X'};\n",
        )
    errors: list[str] = []
    runner._validate_required_authority_contract(errors)
    assert all(
        any(relative in error and "canonical checkpoint/magic signature" in error for error in errors)
        for relative in runner.CANONICAL_CHECKPOINT_MAGIC_TOKENS
    )


def test_runtime_authority_requires_current_uniform_and_amr_payload_versions(
    monkeypatch, tmp_path
):
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    for relative in runner.CANONICAL_CHECKPOINT_VERSION_TOKENS:
        _write_synthetic_source(
            tmp_path,
            relative,
            "UNIFORM_CHECKPOINT_PAYLOAD_VERSION = 8\n"
            "AMR_CHECKPOINT_PAYLOAD_VERSION = 11\n",
        )
    errors: list[str] = []
    runner._validate_required_authority_contract(errors)
    assert any("UNIFORM_CHECKPOINT_PAYLOAD_VERSION = 9" in error for error in errors)
    assert any("AMR_CHECKPOINT_PAYLOAD_VERSION = 12" in error for error in errors)


def test_runtime_authority_requires_unconditional_allocation_row():
    runner = _load_runner()
    data = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    checks = [
        row
        for row in data["check"]
        if row.get("test_regex") != runner.ALLOCATION_PROOF_ROW["test_regex"]
    ]
    errors: list[str] = []
    runner._validate_zero_allocation_proof(checks, errors)
    assert any("exactly one allocation proof row" in error for error in errors)


def test_runtime_authority_requires_prepared_allocation_order(monkeypatch, tmp_path):
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    source = _write_synthetic_source(
        tmp_path,
        runner.ALLOCATION_PROOF_SOURCE,
        """TEST(ProgramExecutionServicesContract, GeneratedScratchIsPersistentExactAndNonAliasing) {
  const AllocationEventStats before_reuse = allocation_event_stats();
  install_execution_lane(sim, \"lane\");
  sim.set_program_block_map({0});
  ctx.rhs_scratch(41, 0, state);
  ctx.rhs_scratch(41, 0, state);
  const AllocationEventStats after_reuse = allocation_event_stats();
  EXPECT_EQ(after_reuse.fab_calls, before_reuse.fab_calls);
  EXPECT_EQ(after_reuse.fab_bytes, before_reuse.fab_bytes);
}
""",
    )
    monkeypatch.setattr(
        runner,
        "_cpp_suites",
        lambda: {"allocation": {"sources": [str(source.relative_to(tmp_path))]}},
    )
    errors: list[str] = []
    runner._validate_zero_allocation_proof([dict(runner.ALLOCATION_PROOF_ROW)], errors)
    assert any("bind/prep and warm scratch" in error for error in errors)


def test_runtime_authority_requires_closed_m2_m3_ledgers(monkeypatch, tmp_path):
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    for (
        relative,
        _runner,
        gate,
        _required_issues,
        _expected_count,
        _native_positive_issues,
    ) in runner.COMPOSED_GATE_SPECS:
        _write_synthetic_source(
            tmp_path,
            relative,
            "schema_version = 1\n"
            f'gate = "{gate}"\n'
            "issues = []\n"
            "deferred = []\n"
            "\n"
            "[[check]]\n"
            'issue = "ADC-999"\n'
            'polarity = "positive"\n'
            'kind = "pytest"\n',
        )
    errors: list[str] = []
    runner._validate_composed_gate_closures(errors)
    assert any(
        "m2_temporal_execution.toml" in error and "required issues" in error
        for error in errors
    )
    assert any(
        "m3_amr_multilayout.toml" in error and "required issues" in error
        for error in errors
    )


def test_runtime_authority_rejects_public_amr_roots_and_include_fragments(monkeypatch, tmp_path):
    runner = _load_runner()
    program = tmp_path / "include/pops/runtime/program"
    program.mkdir(parents=True)
    generic = program / "program_execution_services.hpp"
    generic.write_text(
        "template <int Dim> class ProgramExecutionServices {};\n"
        "template <int Dim, class Extra> class ProgramExecutionServices;\n",
        encoding="utf-8",
    )
    (program / "program_execution_services_amr.hpp").write_text(
        "struct AmrProgramExecutionAdapter {};\n", encoding="utf-8"
    )
    (program / "program_execution_services_amr_spatial.inc").write_text(
        "// an implementation fragment\n", encoding="utf-8"
    )
    runtime_state = program / "program_runtime_state.hpp"
    runtime_state.write_text("void dispatch_cadence_step();\n", encoding="utf-8")
    for relative in ("src/runtime/system/system.cpp", "src/runtime/amr/amr_system.cpp"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("void dispatch_cadence_step();\n", encoding="utf-8")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(
        runner,
        "_authority_sources",
        lambda: tuple(sorted((*program.glob("*.hpp"), *program.glob("*.inc")))),
    )

    errors: list[str] = []
    runner._validate_execution_service_architecture(errors)

    assert any("forbidden public AMR execution-services root" in error for error in errors)
    assert any("forbids AMR runtime fragments (.inc)" in error for error in errors)
    assert any("second template parameter" in error for error in errors)
    assert any("AmrProgramExecutionAdapter" in error for error in errors)


def test_runtime_authority_scans_all_production_roots_for_duplicate_abi_and_roots(
    monkeypatch, tmp_path
):
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    generic = _write_synthetic_source(
        tmp_path,
        "include/pops/runtime/program/program_execution_services.hpp",
        "constexpr int kProgramInstallAbiVersion = 5;\n"
        "template <int Dim> class ProgramExecutionServices {};\n",
    )
    runtime_state = _write_synthetic_source(
        tmp_path,
        "include/pops/runtime/program/program_runtime_state.hpp",
        "void dispatch_cadence_step(double) {}\n",
    )
    duplicate_abi = _write_synthetic_source(
        tmp_path,
        "src/runtime/parallel/program_installation.hpp",
        "constexpr int kProgramInstallAbiVersion = 5;\n",
    )
    public_amr_root = _write_synthetic_source(
        tmp_path,
        "src/runtime/amr/program_execution_services_amr.hpp",
        "struct AmrProgramExecutionAdapter {};\n",
    )
    monkeypatch.setattr(runner, "_authority_sources", lambda: (generic, runtime_state))
    monkeypatch.setattr(
        runner,
        "_production_sources",
        lambda: (generic, runtime_state, duplicate_abi, public_amr_root),
    )

    errors: list[str] = []
    runner._validate_source_barriers(errors)

    assert any("exactly one ABI version declaration (v5)" in error for error in errors)
    assert any("forbidden public AMR execution-services root" in error for error in errors)
    assert any("AmrProgramExecutionAdapter" in error for error in errors)


def test_runtime_authority_authenticates_openmp_and_allocator_rows(monkeypatch, tmp_path):
    runner = _load_runner()
    monkeypatch.setenv("POPS_RUNTIME_AUTHORITY_OPENMP", "1")
    monkeypatch.setenv("POPS_RUNTIME_AUTHORITY_ALLOCATION", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.setenv("POPS_KOKKOS_ROOT", str(tmp_path))
    environment = runner._required_environment(
        [{"allocation": "required"}], backend="openmp"
    )
    assert environment["POPS_REQUIRE_NATIVE_TESTS"] == "1"
    assert environment["POPS_REQUIRE_HOT_ALLOCATION_FREE"] == "1"

    monkeypatch.delenv("POPS_RUNTIME_AUTHORITY_OPENMP")
    with pytest.raises(RuntimeError, match="OPENMP=1"):
        runner._required_environment([], backend="openmp")


def test_runtime_authority_refuses_an_unavailable_mpi_launcher(monkeypatch):
    runner = _load_runner()
    monkeypatch.setattr(runner.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="MPI launcher"):
        runner._mpi_python_command(
            "mpiexec",
            2,
            "tests/python/integration/mpi/test_amr_clean_route_program_mpi.py::"
            "test_public_mpi_explicit_and_preset_ssprk2_are_bit_identical",
            dimension=2,
        )


@pytest.mark.parametrize("dimension", (1, 2, 3))
def test_runtime_authority_mpi_bootstrap_selects_each_explicit_dimension(monkeypatch, dimension):
    runner = _load_runner()
    monkeypatch.setattr(runner.shutil, "which", lambda _name: "/usr/bin/mpiexec")

    command = runner._mpi_python_command(
        "mpiexec",
        2,
        "tests/python/integration/mpi/test_amr_clean_route_program_mpi.py::"
        "test_public_mpi_explicit_and_preset_ssprk2_are_bit_identical",
        dimension=dimension,
    )

    bootstrap_index = command.index("-c") + 1
    assert "select_native_dimension(int(sys.argv[1]))" in command[bootstrap_index]
    compile(command[bootstrap_index], "<runtime-authority-mpi-bootstrap>", "exec")
    assert command[bootstrap_index + 1] == str(dimension)
    assert command[-1].endswith(
        "test_amr_clean_route_program_mpi.py::test_public_mpi_explicit_and_preset_ssprk2_are_bit_identical"
    )
    assert "run_name='__runtime_authority_node__'" in command[bootstrap_index]
    assert "_run_all" not in command[bootstrap_index]


def test_runtime_authority_mpi_executor_rejects_file_only_nodeid(monkeypatch):
    runner = _load_runner()
    monkeypatch.setattr(runner.shutil, "which", lambda _name: "/usr/bin/mpiexec")
    with pytest.raises(ValueError, match="exact file::function nodeid"):
        runner._mpi_python_command(
            "mpiexec",
            2,
            "tests/python/integration/mpi/test_amr_clean_route_program_mpi.py",
            dimension=2,
        )


def test_runtime_authority_mpi_bootstrap_executes_only_exact_fixture_nodeid(
    monkeypatch, tmp_path
):
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    fixture = _write_synthetic_source(
        tmp_path,
        "tests/python/fixture_mpi.py",
        "_fails = 0\n"
        "def test_exact():\n"
        "    print('EXACT_FIXTURE')\n"
        "def test_other():\n"
        "    print('OTHER_FIXTURE')\n"
        "    raise RuntimeError('the non-selected function ran')\n"
        "def _run_all():\n"
        "    raise RuntimeError('the file-level runner ran')\n",
    )
    _write_synthetic_source(tmp_path, "pops/__init__.py", "")
    _write_synthetic_source(
        tmp_path,
        "pops/_native_selector.py",
        "def select_native_dimension(dimension):\n"
        "    assert dimension == 2\n",
    )
    monkeypatch.setattr(runner.shutil, "which", lambda _name: "/usr/bin/mpiexec")
    nodeid = "tests/python/fixture_mpi.py::test_exact"
    command = runner._mpi_python_command("mpiexec", 2, nodeid, dimension=2)
    bootstrap = command[command.index("-c") + 1]
    completed = subprocess.run(
        [sys.executable, "-c", bootstrap, "2", str(fixture), nodeid],
        cwd=tmp_path,
        env={"PYTHONPATH": str(tmp_path)},
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "EXACT_FIXTURE" in completed.stdout
    assert "OTHER_FIXTURE" not in completed.stdout


def test_runtime_authority_refuses_ctest_build_dimension_mismatch(tmp_path):
    runner = _load_runner()
    (tmp_path / "CMakeCache.txt").write_text("POPS_NATIVE_DIM:STRING=2\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"POPS_NATIVE_DIM=2, but --dim=3"):
        runner._require_build_native_dimension(tmp_path, 3)


def test_runtime_authority_parser_refuses_an_invalid_dimension():
    runner = _load_runner()

    with pytest.raises(SystemExit) as excinfo:
        runner.main(["--check-only", "--dim", "4"])

    assert excinfo.value.code == 2


def test_runtime_authority_has_no_hard_coded_mpi_native_dimension():
    source = RUNNER.read_text(encoding="utf-8")

    assert "select_native_dimension(2)" not in source


def test_runtime_authority_requires_ci_architecture_mpi_openmp_and_release_wiring():
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    release = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    final_contract = (ROOT / "scripts/final_release_contract.py").read_text(encoding="utf-8")
    assert "for dim in 1 2 3; do" in ci
    assert 'python3 scripts/run_runtime_authority_gate.py --check-only --dim "$dim"' in ci
    assert '--list-ctest-targets --backend mpi --dim "$POPS_NATIVE_DIM"' in ci
    assert "Runtime authority complete ADC-700/702/720 gate" in ci
    assert "--backend mpi" in ci
    assert "--mpi-exec mpiexec" in ci
    assert '--dim "$POPS_NATIVE_DIM"' in ci
    assert "Runtime authority OpenMP/allocation gate" in ci
    assert "POPS_RUNTIME_AUTHORITY_OPENMP" in ci
    assert "POPS_RUNTIME_AUTHORITY_ALLOCATION" in ci
    assert "final-authority-matrix:" in ci
    assert "Execute serial M2 and runtime authority" in ci
    assert "Execute OpenMP M2 and runtime authority" in ci
    assert "Execute MPI M2, M3, and runtime authority" in ci
    assert "full-source-matrix" in release
    assert "Runtime authority source ledger" not in release
    assert 'python scripts/run_runtime_authority_gate.py --check-only --dim "$dim"' not in release
    assert "runtime_authority.toml" in final_contract
    assert "runtime_authority_source_errors" in final_contract
