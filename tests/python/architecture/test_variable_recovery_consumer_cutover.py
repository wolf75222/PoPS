"""Architecture fence for the bounded ADC-755 runtime-consumer cutover."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
BLOCK_BUILDER = ROOT / "include/pops/runtime/builders/block/block_builder.hpp"
SYSTEM_FIELDS = ROOT / "src/runtime/system/system_fields.cpp"
PROGRAM_SERVICES = ROOT / "include/pops/runtime/program/program_execution_services.hpp"
PROGRAM_CONTEXT = ROOT / "include/pops/runtime/program/program_context.hpp"
AMR_PROGRAM_CONTEXT = ROOT / "include/pops/runtime/program/amr_program_context.hpp"
FLUX_FAILURE = ROOT / "include/pops/numerics/fv/flux_failure.hpp"
FACE_FLUX = ROOT / "include/pops/numerics/spatial/primitives/face_flux.hpp"
RECOVERY = ROOT / "include/pops/numerics/nonlinear/prepared_variable_recovery.hpp"
SPATIAL_RECOVERY_CONSUMERS = (
    FACE_FLUX,
    ROOT / "include/pops/numerics/spatial/operators/cartesian_operator.hpp",
    ROOT / "include/pops/numerics/spatial/operators/masked_operator.hpp",
    ROOT / "include/pops/numerics/spatial/operators/polar_operator.hpp",
    ROOT / "include/pops/numerics/spatial/embedded_boundary/operator.hpp",
)


def _between(source: str, begin: str, end: str) -> str:
    return source.split(begin, 1)[1].split(end, 1)[0]


def test_cell_primitive_conversion_has_one_prepared_fail_closed_authority():
    source = BLOCK_BUILDER.read_text(encoding="utf-8")
    conversion = _between(
        source, "make_cell_convert(const Model& m)", "\n}\n\n}  // namespace pops"
    )

    assert "prepare_model_variable_recovery(m)" in conversion
    assert "recover_prepared_variable(" in conversion
    assert "outcome.publication_permitted()" in conversion
    assert "return recovery_report(outcome)" in conversion
    assert "m.to_primitive" not in conversion


def test_runtime_materialization_consumes_only_prepared_batch_before_publication():
    source = SYSTEM_FIELDS.read_text(encoding="utf-8")
    materialization = _between(
        source,
        "std::vector<double> System::get_primitive_state",
        "\nSolveReport System::solve_fields_in_place_",
    )

    required = materialization.index("if (!s.batch_cons_to_prim)")
    recovery = materialization.index("s.batch_cons_to_prim(cons, prim)", required)
    refusal = materialization.index("if (!batch.publication_permitted())", recovery)
    publication = materialization.index("return prim;", refusal)
    assert required < recovery < refusal < publication
    assert "variable recovery failed" in materialization
    assert "generation-qualified prepared batch recovery consumer" in materialization
    assert "s.cons_to_prim(cell_in.data(), cell_out.data())" not in materialization


def test_runtime_materialization_has_no_pointwise_compatibility_authority():
    source = SYSTEM_FIELDS.read_text(encoding="utf-8")
    materialization = _between(
        source,
        "std::vector<double> System::get_primitive_state",
        "\nSolveReport System::solve_fields_in_place_",
    )

    assert "Compatibility path" not in materialization
    assert "if (s.batch_cons_to_prim)" not in materialization
    assert "std::vector<double> cell_in" not in materialization


def test_type_erased_recovery_report_preserves_actual_method_identity():
    source = RECOVERY.read_text(encoding="utf-8")
    report = _between(source, "struct RecoveryReport {", "\n};\n\nstatic_assert")
    erasure = _between(
        source,
        "POPS_HD inline RecoveryReport recovery_report",
        "\n}\n\ninline constexpr const char* recovery_status_name",
    )

    assert "RecoveryMethodKind selected_method_kind" in report
    assert "RecoveryMethodKind last_method_kind" in report
    assert "outcome.selected_method_kind" in erasure
    assert "outcome.last_method_kind" in erasure


def test_runtime_recovery_failure_names_last_attempted_method():
    source = SYSTEM_FIELDS.read_text(encoding="utf-8")
    materialization = _between(
        source,
        "std::vector<double> System::get_primitive_state",
        "\nSolveReport System::solve_fields_in_place_",
    )

    assert "recovery_method_kind_name(recovery.last_method_kind)" in materialization
    assert "last_method_index=" in materialization
    assert "std::to_string(recovery.last_method)" in materialization


def test_runtime_layer_has_no_independent_direct_primitive_recovery():
    runtime_sources = (
        *ROOT.glob("include/pops/runtime/**/*.hpp"),
        *ROOT.glob("src/runtime/**/*.cpp"),
    )
    bypasses = []
    for path in runtime_sources:
        source = path.read_text(encoding="utf-8")
        if re.search(r"\b(?:m|model)\.to_primitive\s*\(", source):
            bypasses.append(path.relative_to(ROOT).as_posix())
    assert bypasses == []


def test_program_terminal_state_publication_validates_every_candidate_before_first_copy():
    shared = PROGRAM_SERVICES.read_text(encoding="utf-8")
    commit = _between(
        shared,
        "void commit_many(std::initializer_list<std::pair<MultiFab*, const MultiFab*>> commits)",
        "\n  /// Apply every coupled-source operator",
    )
    validation = commit.index("program_execution_validate_commit_candidates_(commits)")
    publication = commit.index("lincomb(*target", validation)
    assert validation < publication

    uniform = PROGRAM_CONTEXT.read_text(encoding="utf-8")
    assert "validate_program_state_publication_candidate(block, *candidate)" in uniform

    amr = AMR_PROGRAM_CONTEXT.read_text(encoding="utf-8")
    assert "require_recoverable_block_candidate_(" in amr
    assert "AMR Program terminal state publication" in amr


def test_face_reconstruction_returns_and_consumes_one_typed_recovery_report():
    reconstruction = FACE_FLUX.read_text(encoding="utf-8")
    assert "struct ReconstructedFaceState" in reconstruction
    assert "prepare_model_variable_recovery(model)" in reconstruction
    assert "recover_prepared_variable(plan" in reconstruction
    assert "recovery_report(outcome)" in reconstruction
    assert "model.to_primitive" not in reconstruction

    failure_channel = FLUX_FAILURE.read_text(encoding="utf-8")
    assert "record_recovery(const RecoveryReport& report" in failure_channel
    assert "recovery_evaluation_status(report.status)" in failure_channel


def test_every_production_spatial_path_consumes_recovery_before_flux_evaluation():
    bypasses = []
    missing_consumers = []
    for path in SPATIAL_RECOVERY_CONSUMERS:
        source = path.read_text(encoding="utf-8")
        if re.search(r"\breconstruct_pp\s*<\s*Model\s*>\s*\(", source):
            bypasses.append(path.relative_to(ROOT).as_posix())
        if "reconstruct_pp_recovered<Model>(" in source and (
            "record_reconstruction_recoveries(" not in source
        ):
            missing_consumers.append(path.relative_to(ROOT).as_posix())
    assert bypasses == []
    assert missing_consumers == []
