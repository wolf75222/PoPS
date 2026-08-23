"""Static contracts for the authenticated MPI sine-wave verification matrix.

These tests inspect the campaign declaration and its fail-closed receipt
validator only.  They do not execute an advection calculation.
"""
from __future__ import annotations

import copy
import runpy
import sys
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CASE = REPOSITORY_ROOT / "benchmarks" / "verification" / "advection" / "sine_wave"
MATRIX = CASE / "matrix.v1.json"
DRIVER = CASE / "run_matrix.py"


def _scope() -> dict[str, object]:
    return runpy.run_path(str(DRIVER))


def _mpi_metadata() -> dict[str, object]:
    topology = [2, 2, 2]
    owners = []
    coordinates = []
    rank = 0
    for x in range(2):
        for y in range(2):
            for z in range(2):
                lower = [16 * x, 16 * y, 16 * z]
                upper = [value + 16 for value in lower]
                owners.append({"rank": rank, "local_boxes": [[lower, upper]]})
                coordinates.append({"rank": rank, "coordinate": [x, y, z]})
                rank += 1
    return {
        "mpi": True,
        "provenance": {
            "execution": {
                "runtime": {"mpi_compiled": True, "mpi_active": True, "mpi_ranks": 8}
            },
            "source": {"native": {"has_mpi": True}},
        },
        "coverage": {
            "mpi_topology": {
                "requested_ranks": 8,
                "observed_ranks": 8,
                "expected_spatial_decomposition": topology,
                "ownership_active": True,
                "rank_ownership": owners,
                "rank_coordinates": coordinates,
                "inter_rank_corner_crossing": {
                    "observed": True,
                    "corner_index": [16, 16, 16],
                    "corner_coordinate": [0.5, 0.5, 0.5],
                    "participating_ranks": list(range(8)),
                    "velocity": [1.0, 1.0, 1.0],
                    "arrival_time": 0.363,
                },
            }
        },
    }


def test_matrix_adds_a_real_dim3_mpi_phase_and_cartesian_corner_case():
    scope = _scope()
    matrix = scope["_read_matrix"](MATRIX)
    cases = scope["_validate_matrix"](matrix)
    by_id = {case["id"]: case for case in cases}

    assert len(cases) == 37
    assert matrix["build_phases"][-1] == {"id": "dim3-mpi", "dimension": 3, "mpi": True}
    assert {case["mpi_ranks"] for case in cases if case["mpi"]} == {1, 2, 4, 8}
    assert [
        by_id[identifier]["mpi_topology"]
        for identifier in ("d2-mpi-np1", "d2-mpi-np2", "d2-mpi-np4")
    ] == [[1, 1], [1, 2], [2, 2]]
    assert by_id["d3-mpi-np8-corner"]["mpi_topology"] == [2, 2, 2]
    assert by_id["d3-mpi-np8-corner"]["obligations"] == ["block_corner_3d"]
    command = scope["_command"](by_id["d3-mpi-np8-corner"], Path("out"))
    assert command[:3] == [str(Path(sys.executable).absolute().parent / "mpiexec"), "-n", "8"]
    assert command[command.index("--mpi-topology") + 1] == "2,2,2"


def test_mpi_launcher_prefers_an_explicit_romeo_override_and_never_searches_path(monkeypatch):
    scope = _scope()

    monkeypatch.setenv("POPS_MPIEXEC", "/opt/romeo/mpi/bin/mpiexec")
    assert scope["_mpi_launcher"]() == "/opt/romeo/mpi/bin/mpiexec"
    monkeypatch.delenv("POPS_MPIEXEC")
    assert scope["_mpi_launcher"]() == str(Path(sys.executable).absolute().parent / "mpiexec")
    source = DRIVER.read_text(encoding="utf-8")
    assert 'Path(sys.executable).absolute().parent / "mpiexec"' in source
    assert "shutil.which" not in source


def test_mpi_preflight_is_two_rank_native_only_before_distributed_cases(monkeypatch):
    scope = _scope()
    calls = []
    preflight = scope["_preflight_mpi_launcher"]
    runtime_globals = preflight.__globals__

    monkeypatch.setitem(runtime_globals, "_require_mpi_launcher", lambda: None)
    monkeypatch.setitem(runtime_globals, "_mpi_launcher", lambda: "/exact/mpi/mpiexec")
    monkeypatch.setattr(
        scope["subprocess"], "run", lambda command, **kwargs: calls.append((command, kwargs))
    )

    preflight(3, {"OMP_NUM_THREADS": "2"})

    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[:5] == ["/exact/mpi/mpiexec", "-n", "2", sys.executable, "-c"]
    assert command[-1] == "3"
    assert "select_native_dimension" in command[5]
    assert "allgather_bytes" in command[5]
    assert "\npops.run(" not in command[5]
    assert kwargs["check"] is True


def test_mpi_ownership_validator_requires_active_native_receipts_and_a_bijective_topology():
    scope = _scope()
    case = {
        "id": "d3-mpi-np8-corner",
        "dimension": 3,
        "mpi": True,
        "mpi_ranks": 8,
        "mpi_topology": [2, 2, 2],
    }
    metadata = _mpi_metadata()

    scope["_verify_mpi_ownership"](case, metadata)

    inactive = copy.deepcopy(metadata)
    inactive["provenance"]["execution"]["runtime"]["mpi_active"] = False
    with pytest.raises(RuntimeError, match="active compiled MPI provenance"):
        scope["_verify_mpi_ownership"](case, inactive)

    duplicate_rank = copy.deepcopy(metadata)
    duplicate_rank["coverage"]["mpi_topology"]["rank_coordinates"][-1]["rank"] = 0
    with pytest.raises(RuntimeError, match="bijective active rank decomposition"):
        scope["_verify_mpi_ownership"](case, duplicate_rank)

    missing_corner = copy.deepcopy(metadata)
    missing_corner["coverage"]["mpi_topology"]["inter_rank_corner_crossing"] = None
    with pytest.raises(RuntimeError, match="inter-rank ownership-corner crossing"):
        scope["_verify_mpi_ownership"](case, missing_corner)


def test_generator_collects_native_rank_ownership_before_the_single_run_lifecycle():
    source = (CASE / "generate_data.py").read_text(encoding="utf-8")

    assert '"--mpi-topology"' in source
    assert "simulation.local_boxes(\"tracer\")" in source
    assert "allgather_value(context.communicator.handle, local_receipt)" in source
    assert '"expected_spatial_decomposition"' in source
    assert '"inter_rank_corner_crossing"' in source
    assert "patch_center=PATCH_CENTER[: args.dimension]" in source
    assert "patch_half_width=PATCH_HALF_WIDTH[: args.dimension]" in source
    assert source.index("allgather_value(context.communicator.handle, local_receipt)") < source.index(
        "pops.run(simulation"
    )


def test_matrix_driver_seals_common_native_identity_and_complete_build_authority_roots():
    scope = _scope()
    source = DRIVER.read_text(encoding="utf-8")

    assert "pyproject.toml" in scope["BUILD_SOURCE_AUTHORITY_ROOTS"]
    assert "_native_compatibility_identity" in source
    assert '"build_fingerprint"' in source
    assert '"abi_common"' in source
    assert '"native_compatibility"' in source
    assert "native variants disagree on compiler/toolchain/header/Kokkos identity" in source
