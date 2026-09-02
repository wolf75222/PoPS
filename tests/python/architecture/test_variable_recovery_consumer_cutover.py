"""Architecture fence for the bounded ADC-755 runtime-consumer cutover."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
UNIFORM_RECOVERY = ROOT / "include/pops/runtime/recovery/uniform_recovery_consumer.hpp"
GENERATED_UNIFORM = ROOT / "include/pops/runtime/builders/compiled/generated_system_block.hpp"
SYSTEM_FIELDS = ROOT / "src/runtime/system/system_fields.cpp"
PROGRAM_CONTEXT = ROOT / "include/pops/runtime/program/program_execution_services.hpp"
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
    consumer = _between(
        UNIFORM_RECOVERY.read_text(encoding="utf-8"),
        "UniformCellRecovery make_uniform_variable_inversion_consumer",
        "/// Existing explicit-plan entry point",
    )
    generated = GENERATED_UNIFORM.read_text(encoding="utf-8")

    assert "PreparedUniformRecoveryConsumer<N, std::shared_ptr<Authority>>" in consumer
    assert "std::make_shared<Consumer>(std::move(authority))" in consumer
    assert "prepare_model_variable_recovery" not in consumer
    assert "PreparedModelVariableInversionRecovery<Model>>(model)" in generated
    assert "make_uniform_variable_inversion_consumer(recovery)" in generated
    assert "make_uniform_recovery_consumer(model)" not in generated


def test_runtime_materialization_consumes_only_prepared_batch_before_publication():
    source = SYSTEM_FIELDS.read_text(encoding="utf-8")
    materialization = _between(
        source,
        "std::vector<double> System<Dim>::get_primitive_state",
        "\nSolveReport System<Dim>::solve_fields_in_place_",
    )

    required = materialization.index("if (!block.batch_cons_to_prim)")
    recovery = materialization.index(
        "block.batch_cons_to_prim(conservative, primitive)", required
    )
    refusal = materialization.index("if (!report.publication_permitted())", recovery)
    publication = materialization.index("return primitive;", refusal)
    assert required < recovery < refusal < publication
    assert "variable recovery failed" in materialization
    assert "generation-qualified batch recovery provider" in materialization
    assert "block.cons_to_prim" not in materialization


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

    assert "recovery_method_kind_name(report.recovery.last_method_kind)" in materialization
    assert "last_method_index=" in materialization
    assert "std::to_string(report.recovery.last_method)" in materialization
    assert "failed_cell=" in materialization
    assert "std::to_string(report.failed_cell)" in materialization


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
    uniform = PROGRAM_CONTEXT.read_text(encoding="utf-8")
    commit = _between(
        uniform,
        "void commit_many(",
        "\n  void apply_coupling_operators(",
    )
    validation = commit.index("validate_program_state_publication_candidate_(")
    assert commit.count("validate_program_state_publication_candidate_(") == 1
    validation_boundary = commit.index(
        '"ProgramExecutionServices commit mask classification differs between ranks"', validation
    )
    snapshots = commit.index("workspace.commit_snapshots[block]", validation_boundary)
    masked_copy = commit.index("copy_active_valid_cells_(", snapshots)
    fence = commit.index("device_fence();", masked_copy)
    staging_boundary = commit.index(
        "if (all_reduce_max(staging_error ? 1L : 0L, lane) != 0)", fence
    )
    final_copies = [
        match.start()
        for match in re.finditer(
            r"copy_field_storage_\(workspace\.commit_snapshots\["
            r"workspace\.commit_runtime_blocks\[candidate\]\],\s*"
            r"\*workspace\.commit_targets\[candidate\]\)",
            commit,
        )
    ]

    assert validation < validation_boundary < snapshots < masked_copy < fence < staging_boundary
    assert "bind_commit_images" in uniform
    assert "std::vector<field_type> commit_snapshots;" in uniform
    assert final_copies
    assert all(staging_boundary < copy for copy in final_copies)


def test_nd_face_reconstruction_consumes_typed_conversion_before_flux_evaluation():
    reconstruction = ND_RECONSTRUCTION.read_text(encoding="utf-8")
    operator = CARTESIAN_OPERATOR.read_text(encoding="utf-8")

    assert "StateConversion<typename Model::State>" in reconstruction
    assert "model.recover(pops::load_state<Model>(state" in reconstruction
    assert "reconstruct_face_pair<Axis, Variables>" in operator
    left_status = operator.index("traces.left_status != StateConversionStatus::Success")
    right_status = operator.index("traces.right_status != StateConversionStatus::Success")
    evaluation = operator.index("evaluate_numerical_flux_at")
    measure = operator.index("apply_face_measure", evaluation)
    assert left_status < evaluation < measure
    assert right_status < evaluation < measure
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
