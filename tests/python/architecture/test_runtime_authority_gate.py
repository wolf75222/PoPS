"""Source-only integrity checks for the closed ADC-700/702/720 gate."""

from __future__ import annotations

import importlib.util
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
    assert len(data["check"]) == 58
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
    required_targets = runner._required_ctest_targets(data["check"], backend="mpi")
    assert "test_program_execution_services_contract" in required_targets
    assert "test_mpi_cell_temporal_program_refusal" in required_targets


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
            "src/runtime/amr/legacy.cpp",
            "extern void pops_install_program_amr();\n",
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
            "src/runtime/amr/fallback.cpp",
            "AmrRuntime<2>::step(dt);\n",
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
            "include/pops/runtime/program/checkpoint.hpp",
            'const char* old = "POPSAST4";\n',
            "legacy checkpoint/magic",
        ),
        (
            "include/pops/runtime/program/installer.hpp",
            "void pops_install_field_boundaries_amr();\n",
            "retired split installers",
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


def test_runtime_authority_ignores_native_brick_abi_symbols(monkeypatch, tmp_path):
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    path = _write_synthetic_source(
        tmp_path,
        "src/runtime/amr/native_brick.cpp",
        "extern void pops_install_native_amr();\n"
        "extern void pops_register_provider_routes_amr();\n"
        "extern void pops_brick_install_amr_v5();\n"
        "extern void external_install_amr();\n",
    )
    errors: list[str] = []
    runner._scan_production_barriers(errors, (path,))
    assert errors == []


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


def test_runtime_authority_rejects_retired_context_filenames(monkeypatch, tmp_path):
    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    path = _write_synthetic_source(
        tmp_path,
        "include/pops/runtime/program/detail/amr_program_context.hpp",
        "",
    )
    errors: list[str] = []
    runner._scan_production_barriers(errors, (path,))
    assert any("retired ProgramContext source remains present" in error for error in errors)


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
            "tests/python/integration/mpi/test_amr_clean_route_program_mpi.py",
            dimension=2,
        )


@pytest.mark.parametrize("dimension", (1, 2, 3))
def test_runtime_authority_mpi_bootstrap_selects_each_explicit_dimension(monkeypatch, dimension):
    runner = _load_runner()
    monkeypatch.setattr(runner.shutil, "which", lambda _name: "/usr/bin/mpiexec")

    command = runner._mpi_python_command(
        "mpiexec",
        2,
        "tests/python/integration/mpi/test_amr_clean_route_program_mpi.py",
        dimension=dimension,
    )

    bootstrap_index = command.index("-c") + 1
    assert "select_native_dimension(int(sys.argv[1]))" in command[bootstrap_index]
    assert command[bootstrap_index + 1] == str(dimension)


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
    assert "full-source-matrix" in release
    assert "Runtime authority source ledger" in release
    assert 'python scripts/run_runtime_authority_gate.py --check-only --dim "$dim"' in release
    assert "runtime_authority.toml" in final_contract
    assert "runtime_authority_source_errors" in final_contract
