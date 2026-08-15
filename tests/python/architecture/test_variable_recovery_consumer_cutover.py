"""Architecture fence for the bounded ADC-755 runtime-consumer cutover."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
GENERATED_UNIFORM_BUILDER = ROOT / "include/pops/runtime/builders/compiled/generated_system_block.hpp"
GENERATED_AMR_BUILDER = ROOT / "include/pops/runtime/builders/compiled/generated_amr_system_block.hpp"
UNIFORM_RECOVERY_CONSUMER = ROOT / "include/pops/runtime/recovery/uniform_recovery_consumer.hpp"
SYSTEM_FIELDS = ROOT / "src/runtime/system/system_fields.cpp"
PROGRAM_CONTEXT = ROOT / "include/pops/runtime/program/program_context.hpp"
AMR_PROGRAM_CONTEXT = ROOT / "include/pops/runtime/program/amr_program_context.hpp"
AMR_SYSTEM = ROOT / "include/pops/runtime/amr_system.hpp"
FLUX_FAILURE = ROOT / "include/pops/numerics/fv/flux_failure.hpp"
FACE_FLUX = ROOT / "include/pops/numerics/spatial/primitives/face_flux.hpp"
ND_RECONSTRUCTION = ROOT / "include/pops/numerics/spatial/nd/reconstruction.hpp"
CARTESIAN_OPERATOR = ROOT / "include/pops/numerics/spatial/operators/cartesian_operator.hpp"
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
    uniform = GENERATED_UNIFORM_BUILDER.read_text(encoding="utf-8")
    amr = GENERATED_AMR_BUILDER.read_text(encoding="utf-8")
    consumer = UNIFORM_RECOVERY_CONSUMER.read_text(encoding="utf-8")

    for source in (uniform, amr):
        assert "prepare_model_variable_recovery(model)" in source
        assert "recover_prepared_variable(recovery_plan, input, initial)" in source
        assert "outcome.publication_permitted()" in source
        assert "return recovery_report(outcome)" in source
        assert "make_uniform_recovery_consumer(model)" in source
    assert "UniformCellRecovery make_uniform_recovery_consumer(const Model& model)" in consumer


def test_runtime_materialization_consumes_only_prepared_batch_before_publication():
    source = SYSTEM_FIELDS.read_text(encoding="utf-8")
    materialization = _between(
        source,
        "std::vector<double> System<Dim>::get_primitive_state",
        "\nSolveReport System<Dim>::solve_fields_in_place_",
    )

    required = materialization.index("if (!block.batch_cons_to_prim)")
    recovery = materialization.index("block.batch_cons_to_prim(conservative, primitive)", required)
    refusal = materialization.index("if (!report.publication_permitted())", recovery)
    publication = materialization.index("return primitive;", refusal)
    assert required < recovery < refusal < publication
    assert "variable recovery failed" in materialization
    assert "UniformRecoveryBatchReport report" in materialization
    assert "s.cons_to_prim(cell_in.data(), cell_out.data())" not in materialization


def test_runtime_materialization_has_no_pointwise_compatibility_authority():
    source = SYSTEM_FIELDS.read_text(encoding="utf-8")
    materialization = _between(
        source,
        "std::vector<double> System<Dim>::get_primitive_state",
        "\nSolveReport System<Dim>::solve_fields_in_place_",
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
        "std::vector<double> System<Dim>::get_primitive_state",
        "\nSolveReport System<Dim>::solve_fields_in_place_",
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
    amr = AMR_PROGRAM_CONTEXT.read_text(encoding="utf-8")
    commit = _between(
        amr,
        "void commit_many(",
        "\n  void apply_coupling_operators(",
    )
    validation = commit.index("validate_prepared_amr_state_publication_candidate(")
    publication = commit.index("*targets[candidate] = std::move(snapshots[candidate])", validation)
    assert validation < publication
    assert "snapshots.emplace_back(*source)" in commit
    assert "AMR Program commit count differs between MPI ranks" in commit
    assert "authenticated_runtime_block_for_state_target_" in commit
    assert "void commit_many(" in amr
    assert "all_reduce_min(state_target) != all_reduce_max(state_target)" in commit

    facade = AMR_SYSTEM.read_text(encoding="utf-8")
    assert "validate_prepared_amr_state_publication_candidate(" in facade


def test_nd_face_reconstruction_consumes_typed_conversion_before_flux_evaluation():
    reconstruction = ND_RECONSTRUCTION.read_text(encoding="utf-8")
    operator = CARTESIAN_OPERATOR.read_text(encoding="utf-8")

    assert "StateConversion<typename Model::State>" in reconstruction
    assert "reconstruct_face_pair<Axis, Variables>" in operator
    left_status = operator.index("traces.left_status != StateConversionStatus::Success")
    right_status = operator.index("traces.right_status != StateConversionStatus::Success")
    evaluation = operator.index("evaluate_numerical_flux_at")
    assert left_status < evaluation
    assert right_status < evaluation
    assert "model.to_primitive" not in reconstruction

    failure_channel = FLUX_FAILURE.read_text(encoding="utf-8")
    assert "record_recovery(const RecoveryReport& report" in failure_channel


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
