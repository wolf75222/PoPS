"""Pure filesystem contracts for immutable benchmark figure publications."""

from __future__ import annotations

import copy
import itertools
import json
import runpy
import sys
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[3]
VERIFICATION = REPOSITORY / "benchmarks/verification/advection/sine_wave/plot_results.py"
PERFORMANCE = REPOSITORY / "benchmarks/performance/advection_sine/plot_scaling.py"


def test_verification_plot_publication_is_staged_hashed_and_no_clobber(tmp_path: Path) -> None:
    scope = runpy.run_path(str(VERIFICATION))
    data = tmp_path / "input.npz"
    metadata = tmp_path / "input.json"
    data.write_bytes(b"authenticated-npz-placeholder")
    metadata.write_text("{}\n", encoding="utf-8")
    manifest = scope["_input_manifest"](
        [data],
        [metadata],
        [{"result_identity": "a" * 64, "source_fingerprint": "b" * 64}],
        fps=5,
    )
    target = tmp_path / "publication"
    staging = scope["_create_staging_directory"](target)
    (staging / "field.png").write_bytes(b"figure")
    scope["_write_publication_manifest"](staging, manifest)
    published = scope["_publish_staging_directory"](staging, target)

    receipt = json.loads((published / "plot_manifest.json").read_text(encoding="utf-8"))
    assert receipt["publication_identity"] == manifest["publication_identity"]
    assert receipt["inputs"][0]["data_sha256"]
    assert receipt["media"][0]["path"] == "field.png"
    with pytest.raises(FileExistsError):
        scope["_create_staging_directory"](target)


def test_verification_plot_cli_accepts_only_one_complete_before_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scope = runpy.run_path(str(VERIFICATION))
    destination = tmp_path / "publication"
    complete = tmp_path / "COMPLETE.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [str(VERIFICATION), "--complete", str(complete), "--figures", str(destination)],
    )
    arguments = scope["_arguments"]()
    assert arguments.complete == complete
    assert not destination.exists()

    for argv in (
        [str(VERIFICATION), str(tmp_path / "one-case.npz"), "--figures", str(destination)],
        [str(VERIFICATION), "--metadata", str(tmp_path / "one-case.json")],
        [
            str(VERIFICATION),
            "--complete",
            str(complete),
            "--complete",
            str(complete),
            "--figures",
            str(destination),
        ],
    ):
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit):
            scope["_arguments"]()
        assert not destination.exists()

    source = VERIFICATION.read_text(encoding="utf-8")
    assert "def plot(" not in source
    assert "def _publish_plot_set(" not in source


def test_performance_plot_publication_is_staged_hashed_and_no_clobber(tmp_path: Path) -> None:
    import sys

    performance_root = PERFORMANCE.parent
    sys.path.insert(0, str(performance_root))
    try:
        scope = runpy.run_path(str(PERFORMANCE))
    finally:
        sys.path.remove(str(performance_root))
    summary_path = tmp_path / "summary.json"
    summary = {
        "schema": "pops.performance.advection-sine.summary.v2",
        "campaign": "strong-openmp",
        "route": "kokkos_openmp",
        "scaling": "strong",
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    manifest = scope["_publication_manifest"](summary_path, summary)
    target = tmp_path / "publication"
    staging = scope["_create_staging_directory"](target)
    (staging / "scaling_overview.png").write_bytes(b"figure")
    scope["_write_publication_manifest"](staging, manifest)
    published = scope["_publish_staging_directory"](staging, target)

    receipt = json.loads((published / "plot_manifest.json").read_text(encoding="utf-8"))
    assert receipt["input"]["summary_sha256"]
    assert receipt["media"][0]["path"] == "scaling_overview.png"
    with pytest.raises(FileExistsError):
        scope["_create_staging_directory"](target)


def test_plotters_contain_no_deletion_or_replacement_operations() -> None:
    for path in (VERIFICATION, PERFORMANCE):
        source = path.read_text(encoding="utf-8")
        assert ".unlink(" not in source
        assert "os.replace(" not in source
        assert "os.rename(" not in source


def test_verification_complete_consumer_refuses_unsealed_or_partial_input(tmp_path: Path) -> None:
    scope = runpy.run_path(str(VERIFICATION))
    partial = tmp_path / "partial"
    partial.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        scope["_sealed_complete_inputs"](partial / "COMPLETE.json")

    incomplete = tmp_path / "COMPLETE.json"
    incomplete.write_text(json.dumps({"schema_version": "pops.sine-wave.matrix-complete.v1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported schema"):
        scope["_sealed_complete_inputs"](incomplete)

    destination = tmp_path / "publication"
    with pytest.raises(ValueError, match="unsupported schema"):
        scope["_publish_complete_plot_set"](
            complete_path=incomplete,
            figures=destination,
            fps=5,
        )
    assert not destination.exists()


def test_sealed_publication_identity_rehashes_complete_without_a_stale_identity(tmp_path: Path) -> None:
    scope = runpy.run_path(str(VERIFICATION))
    data = tmp_path / "input.npz"
    metadata = tmp_path / "input.json"
    complete_path = tmp_path / "COMPLETE.json"
    data.write_bytes(b"sealed-data")
    metadata.write_text("{}\n", encoding="utf-8")
    complete_path.write_text("{}\n", encoding="utf-8")
    inputs = [{"result_identity": "a" * 64, "source_fingerprint": "b" * 64}]
    provisional = scope["_input_manifest"]([data], [metadata], inputs, fps=5)
    manifest = scope["_sealed_publication_manifest"](
        [data], [metadata], inputs,
        complete_path=complete_path,
        complete={"case_count": 37, "matrix_sha256": "c" * 64},
        fps=5,
    )
    hashed = {key: value for key, value in manifest.items() if key != "publication_identity"}
    assert manifest["publication_identity"] == scope["_canonical_sha256"](hashed)
    assert manifest["publication_identity"] != provisional["publication_identity"]
    assert provisional["publication_identity"] not in json.dumps(hashed, sort_keys=True)


def _mpi_corner_metadata() -> tuple[dict[str, object], dict[str, object]]:
    case = {"mpi_topology": [2, 2, 2], "mpi_ranks": 8}
    coordinates = []
    owners = []
    for rank, coordinate in enumerate(itertools.product(range(2), repeat=3)):
        lower = [16 * value for value in coordinate]
        owners.append({"rank": rank, "local_boxes": [[lower, [value + 16 for value in lower]]]})
        coordinates.append({"rank": rank, "coordinate": list(coordinate)})
    metadata = {
        "coverage": {
            "mpi_topology": {
                "requested_ranks": 8,
                "observed_ranks": 8,
                "expected_spatial_decomposition": [2, 2, 2],
                "ownership_active": True,
                "rank_ownership": owners,
                "rank_coordinates": coordinates,
                "inter_rank_corner_crossing": {
                    "observed": True,
                    "corner_index": [16, 16, 16],
                    "corner_coordinate": [0.5, 0.5, 0.5],
                    "participating_ranks": list(range(8)),
                    "characteristic_start": [0.137, 0.137, 0.137],
                    "velocity": [1.0, 1.0, 1.0],
                    "arrival_time": 0.363,
                },
            }
        }
    }
    return case, metadata


def test_np8_corner_and_mobile_trajectory_receipts_are_strict_and_data_only():
    scope = runpy.run_path(str(VERIFICATION))
    case, mpi_metadata = _mpi_corner_metadata()
    scope["_validate_mpi_topology_receipt"]("d3-mpi-np8-corner", case, mpi_metadata)
    missing_owner = copy.deepcopy(mpi_metadata)
    missing_owner["coverage"]["mpi_topology"]["rank_ownership"][0]["local_boxes"] = []
    with pytest.raises(ValueError, match="ownership witness"):
        scope["_validate_mpi_topology_receipt"]("d3-mpi-np8-corner", case, missing_owner)
    mpi_metadata["coverage"]["mpi_topology"]["inter_rank_corner_crossing"]["participating_ranks"] = [0]
    with pytest.raises(ValueError, match="np8 inter-rank corner"):
        scope["_validate_mpi_topology_receipt"]("d3-mpi-np8-corner", case, mpi_metadata)

    mobile = {
        "dimension": 3,
        "patch_marker": {
            "kind": "prescribed_window",
            "center": [0.25, 0.4, 0.55],
            "velocity": [0.5, 0.25, 0.125],
        },
        "coverage": {
            "witnesses": {
                "prescribed_mobile_regrid": {
                    "observed": True,
                    "expected_trajectory": {
                        "center": [0.25, 0.4, 0.55],
                        "velocity": [0.5, 0.25, 0.125],
                        "periodic_axes": [0, 1, 2],
                        "formula": "(center + velocity * time) mod 1",
                    },
                    "snapshots": [
                        {
                            "index": 0,
                            "time": 0.0,
                            "expected_center": [0.25, 0.4, 0.55],
                            "window_probe_count": 7,
                            "box": {"level": 1, "lower": [1, 1, 1], "upper": [2, 2, 2]},
                            "box_center": [0.375, 0.375, 0.375],
                        },
                        {
                            "index": 1,
                            "time": 0.5,
                            "expected_center": [0.5, 0.525, 0.6125],
                            "window_probe_count": 7,
                            "box": {"level": 1, "lower": [2, 2, 2], "upper": [3, 3, 3]},
                            "box_center": [0.625, 0.625, 0.625],
                        },
                    ],
                }
            }
        },
    }
    times, expected, observed = scope["_mobile_trajectory_rows"](mobile)
    assert times.tolist() == [0.0, 0.5]
    assert expected.shape == observed.shape == (2, 3)
    malformed_box = copy.deepcopy(mobile)
    mobile_snapshots = malformed_box["coverage"]["witnesses"]["prescribed_mobile_regrid"][
        "snapshots"
    ]
    mobile_snapshots[0]["box"]["lower"] = [1, 1]
    with pytest.raises(ValueError, match="snapshot is malformed"):
        scope["_mobile_trajectory_rows"](malformed_box)
    mobile["coverage"]["witnesses"]["prescribed_mobile_regrid"]["snapshots"][1]["expected_center"][0] = 0.6
    with pytest.raises(ValueError, match="sealed formula"):
        scope["_mobile_trajectory_rows"](mobile)


def test_verification_nested_partial_results_are_scoped_ignored() -> None:
    nested = (
        REPOSITORY
        / "benchmarks/verification/advection/sine_wave/benchmarks/verification/advection/sine_wave"
    )
    ignore = REPOSITORY / "benchmarks/verification/advection/sine_wave/.gitignore"
    assert nested.is_dir()
    assert "/benchmarks/" in ignore.read_text(encoding="utf-8")


def test_complete_provenance_entry_rejects_artifact_runtime_and_source_tampering() -> None:
    scope = runpy.run_path(str(VERIFICATION))
    metadata = {
        "provenance": {
            "artifact": {"sha256": "a" * 64},
            "execution": {"runtime": {"backend": "OpenMP"}},
            "source": {
                "repository_sha": "test-revision",
                "repository_dirty": False,
                "tracked_diff_sha256": "0" * 64,
                "files": {"generate_data.py": "b" * 64},
                "build_tree": {
                    "schema_version": "pops.sine-wave.build-source-tree.v1",
                    "roots": [
                        "CMakeLists.txt",
                        "cmake",
                        "include",
                        "src",
                        "python",
                        "pyproject.toml",
                        "schemas",
                        "scripts",
                    ],
                    "files": {"CMakeLists.txt": "c" * 64},
                },
            },
        }
    }
    build_tree = metadata["provenance"]["source"]["build_tree"]
    build_tree["fingerprint"] = scope["_canonical_sha256"](build_tree)
    source_authority = scope["_matrix_source_authority"](metadata)
    entry = {
        "native_artifact": {"sha256": "a" * 64},
        "runtime": {"backend": "OpenMP"},
        "source_authority_fingerprint": source_authority["fingerprint"],
    }
    validate = scope["_validate_complete_provenance_entry"]
    validate(entry, metadata, source_authority, case_id="case")

    entry["native_artifact"] = {"sha256": "c" * 64}
    with pytest.raises(ValueError, match="native artifact/runtime"):
        validate(entry, metadata, source_authority, case_id="case")
    entry["native_artifact"] = metadata["provenance"]["artifact"]
    entry["runtime"] = {"backend": "Cuda"}
    with pytest.raises(ValueError, match="native artifact/runtime"):
        validate(entry, metadata, source_authority, case_id="case")
    entry["runtime"] = metadata["provenance"]["execution"]["runtime"]
    metadata["provenance"]["source"]["tracked_diff_sha256"] = "1" * 64
    with pytest.raises(ValueError, match="source authority"):
        validate(entry, metadata, source_authority, case_id="case")
    metadata["provenance"]["source"]["tracked_diff_sha256"] = "0" * 64
    metadata["provenance"]["source"]["build_tree"]["files"] = {
        "cmake/PopsNativeBuildFingerprint.cmake": "d" * 64
    }
    changed_tree = metadata["provenance"]["source"]["build_tree"]
    changed_tree["fingerprint"] = scope["_canonical_sha256"](
        {key: value for key, value in changed_tree.items() if key != "fingerprint"}
    )
    with pytest.raises(ValueError, match="source authority"):
        validate(entry, metadata, source_authority, case_id="case")


def test_sealed_analysis_separates_final_and_probe_convergence_figures(tmp_path: Path) -> None:
    scope = runpy.run_path(str(VERIFICATION))
    complete_path = tmp_path / "COMPLETE.json"
    complete_path.write_text("{}\n", encoding="utf-8")
    generated = [
        tmp_path / "convergence_dim1_final_time.png",
        tmp_path / "convergence_dim1_probe_time.png",
    ]
    analysis = scope["_write_sealed_analysis"](
        tmp_path,
        complete_path=complete_path,
        complete={
            "matrix": "matrix.v1.json",
            "case_count": 37,
            "matrix_sha256": "a" * 64,
            "convergence": {"dim1": {"orders": {"l1": [1.9, 2.0]}}},
        },
        metadata_by_id={},
        generated=generated,
    )
    text = analysis.read_text(encoding="utf-8")
    assert "![Convergence finale Dim1](convergence_dim1_final_time.png)" in text
    assert "![Convergence finale Dim1](convergence_dim1_probe_time.png)" not in text
    assert "![Convergence probe](convergence_dim1_probe_time.png)" in text


@pytest.mark.parametrize("plotter", (VERIFICATION, PERFORMANCE))
def test_publication_refuses_a_destination_created_at_the_final_rename(
    tmp_path: Path, plotter: Path
) -> None:
    import sys

    sys.path.insert(0, str(plotter.parent))
    try:
        scope = runpy.run_path(str(plotter))
    finally:
        sys.path.remove(str(plotter.parent))
    target = tmp_path / "publication"
    staging = scope["_create_staging_directory"](target)
    namespace = scope["_publish_staging_directory"].__globals__
    original = namespace["_atomic_rename_noreplace"]

    def destination_appears_immediately_before_native_publish(
        source: Path, destination: Path
    ) -> None:
        destination.mkdir()
        original(source, destination)

    namespace["_atomic_rename_noreplace"] = destination_appears_immediately_before_native_publish
    try:
        with pytest.raises(FileExistsError):
            scope["_publish_staging_directory"](staging, target)
    finally:
        namespace["_atomic_rename_noreplace"] = original
    assert target.is_dir()
    assert staging.is_dir()


@pytest.mark.parametrize("plotter", (VERIFICATION, PERFORMANCE))
def test_publication_refuses_existing_and_dangling_target_symlinks(
    tmp_path: Path, plotter: Path
) -> None:
    import sys

    sys.path.insert(0, str(plotter.parent))
    try:
        scope = runpy.run_path(str(plotter))
    finally:
        sys.path.remove(str(plotter.parent))
    exterior = tmp_path / "exterior"
    exterior.mkdir()
    existing_link = tmp_path / "publication-existing-link"
    existing_link.symlink_to(exterior, target_is_directory=True)
    dangling_link = tmp_path / "publication-dangling-link"
    dangling_link.symlink_to(tmp_path / "absent-exterior", target_is_directory=True)

    for target in (existing_link, dangling_link):
        with pytest.raises(FileExistsError):
            scope["_create_staging_directory"](target)
        assert target.is_symlink()
    assert list(exterior.iterdir()) == []
