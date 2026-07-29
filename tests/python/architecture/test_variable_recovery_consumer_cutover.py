"""Architecture fence for the bounded ADC-755 runtime-consumer cutover."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
BLOCK_BUILDER = ROOT / "include/pops/runtime/builders/block/block_builder.hpp"
SYSTEM_FIELDS = ROOT / "src/runtime/system/system_fields.cpp"
FLUX_FAILURE = ROOT / "include/pops/numerics/fv/flux_failure.hpp"
FACE_FLUX = ROOT / "include/pops/numerics/spatial/primitives/face_flux.hpp"
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


def test_runtime_materialization_consumes_recovery_before_copying_candidate():
    source = SYSTEM_FIELDS.read_text(encoding="utf-8")
    materialization = _between(
        source,
        "std::vector<double> System::get_primitive_state",
        "\nSolveReport System::solve_fields_in_place_",
    )

    recovery = materialization.index("const RecoveryReport recovery")
    refusal = materialization.index("if (!recovery.publication_permitted())")
    publication = materialization.index("prim[static_cast<std::size_t>(c) * nn + k] = cell_out[c]")
    assert recovery < refusal < publication
    assert "variable recovery failed" in materialization


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
