"""Static contracts for sine-wave build authority and mobile-AMR witnesses."""

from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CASE_DIRECTORY = REPOSITORY_ROOT / "benchmarks" / "verification" / "advection" / "sine_wave"
SUPPORT = CASE_DIRECTORY / "_case_support.py"
MATRIX_DRIVER = CASE_DIRECTORY / "run_matrix.py"


def _scope() -> dict[str, object]:
    return runpy.run_path(str(SUPPORT))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _build_tree_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    (root / "CMakeLists.txt").parent.mkdir(parents=True)
    (root / "CMakeLists.txt").write_text("project(PoPS)\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[build-system]\nrequires = []\n", encoding="utf-8")
    for directory in ("cmake", "include", "src", "python", "schemas", "scripts"):
        leaf = root / directory / "authority.txt"
        leaf.parent.mkdir(parents=True)
        leaf.write_text("authority %s\n" % directory, encoding="utf-8")
    return root


def _metadata(source: dict[str, object], identity: str) -> dict[str, object]:
    return {
        "result_identity": identity,
        "source_fingerprint": _sha256(identity),
        "provenance": {
            "source": source,
            "artifact": {"artifact_identity": "test-artifact-" + identity},
            "execution": {"runtime": {"kokkos_backend": "OpenMP"}},
        },
    }


def _source_authority(build_tree: dict[str, object]) -> dict[str, object]:
    return {
        "repository_sha": "test-revision",
        "repository_dirty": False,
        "tracked_diff_sha256": "0" * 64,
        "files": {"benchmarks/verification/advection/sine_wave/generate_data.py": "1" * 64},
        "build_tree": build_tree,
    }


def test_pyproject_is_a_fail_closed_build_tree_leaf_and_blocks_mixed_complete(tmp_path):
    """A changed packaging/build config cannot be mixed into a complete matrix."""
    support = _scope()
    build_tree = support["_build_source_tree"]
    root = _build_tree_fixture(tmp_path)

    first_tree = build_tree(root)
    (root / "pyproject.toml").write_text(
        "[build-system]\nrequires = [\"setuptools\"]\n", encoding="utf-8"
    )
    second_tree = build_tree(root)

    assert "pyproject.toml" in first_tree["roots"]
    assert first_tree["files"]["pyproject.toml"] != second_tree["files"]["pyproject.toml"]
    assert first_tree["fingerprint"] != second_tree["fingerprint"]

    matrix = runpy.run_path(str(MATRIX_DRIVER))
    output = tmp_path / "matrix"
    output.mkdir()
    results = {}
    for identifier, source in (
        ("before", _source_authority(first_tree)),
        ("after", _source_authority(second_tree)),
    ):
        case_directory = output / identifier
        case_directory.mkdir()
        data_path = case_directory / "result.npz"
        metadata_path = case_directory / "result.json"
        data_path.write_bytes(b"not-a-numerical-run")
        metadata = _metadata(source, identifier)
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        results[identifier] = (data_path, metadata_path, metadata)

    with pytest.raises(RuntimeError, match="source files, or build tree"):
        matrix["_write_complete_manifest"](
            matrix_path=CASE_DIRECTORY / "matrix.v1.json",
            output_root=output,
            cases=({"id": "before"}, {"id": "after"}),
            results=results,
            convergence={},
        )


def _fine_box(center: tuple[float, ...], *, resolution: tuple[int, ...]):
    lower = []
    upper = []
    for coordinate, coarse_cells in zip(center, resolution, strict=True):
        fine_cells = 2 * coarse_cells
        index = int(coordinate * fine_cells)
        lower.append(max(0, index - 4))
        upper.append(min(fine_cells - 1, index + 4))
    return (1, tuple(lower), tuple(upper))


def _mobile_snapshots(
    times: tuple[float, ...],
    *,
    center: tuple[float, ...],
    velocity: tuple[float, ...],
    half_width: tuple[float, ...],
    resolution: tuple[int, ...],
):
    snapshots = []
    for index, time in enumerate(times):
        expected = tuple(
            (initial + speed * time) % 1.0
            for initial, speed in zip(center, velocity, strict=True)
        )
        probes = [expected]
        for axis in range(len(expected)):
            for sign in (-1.0, 1.0):
                point = list(expected)
                point[axis] = (point[axis] + sign * 0.5 * half_width[axis]) % 1.0
                probes.append(tuple(point))
        snapshots.append(
            {
                "patch_rows": tuple(
                    dict.fromkeys(_fine_box(point, resolution=resolution) for point in probes)
                ),
                "regrid_count": index,
                "topology_epoch": index,
            }
        )
    return tuple(snapshots)


def test_mobile_witness_requires_native_boxes_to_follow_the_periodic_prescribed_trajectory():
    witness = _scope()["coverage_witnesses"]
    dimension = 3
    resolution = (32, 32, 32)
    times = (0.0, 0.4, 0.8, 1.2, 1.6)
    center = (0.25, 0.40, 0.55)
    velocity = (0.50, 0.25, 0.125)
    half_width = (0.18, 0.15, 0.13)
    snapshots = _mobile_snapshots(
        times,
        center=center,
        velocity=velocity,
        half_width=half_width,
        resolution=resolution,
    )
    common = {
        "dimension": dimension,
        "velocity": (1.0, 1.0, 1.0),
        "layout": "amr-mobile",
        "resolution": resolution,
        "mode": "diagonal",
        "cycles": 1,
        "timeline_times": times,
        "metrics": {"qualification": {"coarse_fine_interface_seen": True}},
        "witness_reference_point": (0.137, 0.137, 0.137),
        "patch_velocity": velocity,
        "patch_center": center,
        "patch_half_width": half_width,
        "wave_numbers": (1, 2, 3),
        "final_time": times[-1],
        "refinement_ratio": 2,
    }

    accepted = witness(
        obligations=("prescribed_mobile_regrid",), timeline_snapshots=snapshots, **common
    )

    mobile = accepted["prescribed_mobile_regrid"]
    assert mobile["observed"] is True
    assert mobile["expected_trajectory"]["periodic_axes"] == [0, 1, 2]
    assert mobile["snapshots"][-1]["expected_center"][0] == pytest.approx(0.05)

    shifted = _mobile_snapshots(
        times,
        center=tuple((value + 0.25) % 1.0 for value in center),
        velocity=velocity,
        half_width=half_width,
        resolution=resolution,
    )
    rejected = witness(obligations=(), timeline_snapshots=shifted, **common)
    assert rejected["prescribed_mobile_regrid"]["observed"] is False
    assert rejected["prescribed_mobile_regrid"]["missing_snapshot"] == 0

    with pytest.raises(ValueError, match="patch_center and patch_half_width"):
        witness(
            obligations=(),
            timeline_snapshots=snapshots,
            **{
                name: value
                for name, value in common.items()
                if name not in {"patch_center", "patch_half_width"}
            },
        )
